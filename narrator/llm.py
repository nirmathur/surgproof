"""The one wrapper every LLM call goes through.

- disk cache keyed by sha256(model, prompt, image hashes, params, schema) in narrator/cache/
- async concurrency semaphore, retries with exponential backoff
- running cost + call counter, hard budget cap (config llm.budget_usd)
- JSON output validated against a pydantic model; one re-ask on invalid JSON, then failed
Usage:
    from narrator import llm
    out = await llm.generate(prompt, model="fast_model", images=[img_bytes], schema=MyModel, tag="l5")
    out is a dict (schema) / str (no schema), or None if the item failed.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from narrator import cfg as C

_CACHE = C.path("cache")
_CACHE.mkdir(parents=True, exist_ok=True)
_LOG = _CACHE / "calls.jsonl"
_lock = threading.Lock()
_client = None
_sems: dict = {}

STATS = {"calls": 0, "cache_hits": 0, "failed": 0, "cost_usd": 0.0, "in_tok": 0, "out_tok": 0}


class BudgetExceeded(RuntimeError):
    pass


def client():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=C.load_env_key())
    return _client


def resolve(model: str) -> str:
    """Accept a role name (fast_model, judge_model, ...) or a concrete model id."""
    return C.cfg()["models"].get(model, model)


def _sem():
    loop = asyncio.get_running_loop()
    if loop not in _sems:
        _sems[loop] = asyncio.Semaphore(C.cfg()["llm"]["concurrency"])
    return _sems[loop]


def _key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _cache_get(k: str):
    f = _CACHE / k[:2] / f"{k}.json"
    if f.is_file():
        return json.loads(f.read_text())
    return None


def _cache_put(k: str, v) -> None:
    d = _CACHE / k[:2]
    d.mkdir(exist_ok=True)
    tmp = d / f"{k}.tmp"
    tmp.write_text(json.dumps(v))
    tmp.replace(d / f"{k}.json")


def _account(model: str, usage, tag: str, latency: float, ok: bool) -> None:
    pin, pout = C.cfg()["llm"]["price"].get(model, [1.0, 5.0])
    it = getattr(usage, "prompt_token_count", 0) or 0
    ot = (getattr(usage, "candidates_token_count", 0) or 0) + (getattr(usage, "thoughts_token_count", 0) or 0)
    cost = it / 1e6 * pin + ot / 1e6 * pout
    with _lock:
        STATS["calls"] += 1
        STATS["in_tok"] += it
        STATS["out_tok"] += ot
        STATS["cost_usd"] += cost
        if not ok:
            STATS["failed"] += 1
        with open(_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "model": model, "tag": tag, "in": it, "out": ot,
                                "cost": round(cost, 6), "latency": round(latency, 2), "ok": ok}) + "\n")


def _check_budget():
    if STATS["cost_usd"] >= C.cfg()["llm"]["budget_usd"]:
        raise BudgetExceeded(f"budget cap {C.cfg()['llm']['budget_usd']} USD reached")


def jpeg(img, max_side: int = 640, crop=None) -> bytes:
    """PIL image or path -> JPEG bytes. crop = (x1, y1, x2, y2) pixels, padded 20px."""
    from PIL import Image
    im = img if isinstance(img, Image.Image) else Image.open(img)
    im = im.convert("RGB")
    if crop is not None:
        x1, y1, x2, y2 = [int(v) for v in crop]
        w, h = im.size
        im = im.crop((max(0, x1 - 20), max(0, y1 - 20), min(w, x2 + 20), min(h, y2 + 20)))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _schema_dict(schema) -> dict | None:
    if schema is None:
        return None
    if isinstance(schema, dict):
        return schema
    return schema.model_json_schema()


def _validate(schema, data):
    if schema is None or isinstance(schema, dict):
        return data
    obj = schema.model_validate(data)
    return obj.model_dump(mode="json")


async def generate(prompt: str, *, model: str = "fast_model", images: Sequence[bytes] = (),
                   schema=None, temperature: float = 0.0, tag: str = "", sample: int = 0,
                   video_url: str | None = None, search: bool = False, system: str | None = None) -> Any:
    """Returns parsed output, or None if the item failed. Never raises on API errors."""
    from google.genai import types
    m = resolve(model)
    img_sig = [hashlib.md5(b).hexdigest() for b in images]
    sname = getattr(schema, "__name__", None) or (json.dumps(schema, sort_keys=True) if schema else None)
    k = _key(m, prompt, img_sig, sname, temperature, sample, video_url, search, system)
    hit = _cache_get(k)
    if hit is not None:
        STATS["cache_hits"] += 1
        return hit.get("out")
    _check_budget()

    parts = [types.Part.from_bytes(data=b, mime_type="image/jpeg") for b in images]
    if video_url:
        parts.append(types.Part(file_data=types.FileData(file_uri=video_url)))
    parts.append(types.Part.from_text(text=prompt))
    conf = dict(temperature=temperature)
    if system:
        conf["system_instruction"] = system
    if schema is not None and not search:  # grounding + json mode don't mix: ask for JSON in text
        conf["response_mime_type"] = "application/json"
        conf["response_json_schema"] = _schema_dict(schema)
    if search:
        conf["tools"] = [types.Tool(google_search=types.GoogleSearch())]

    out, ok, extra = None, False, {}
    async with _sem():
        contents = [types.Content(role="user", parts=parts)]
        for reask in range(2):
            resp = None
            for attempt in range(C.cfg()["llm"]["max_retries"]):
                t0 = time.time()
                try:
                    resp = await client().aio.models.generate_content(
                        model=m, contents=contents, config=types.GenerateContentConfig(**conf))
                    _account(m, resp.usage_metadata, tag, time.time() - t0, True)
                    break
                except Exception as e:  # noqa: BLE001 rate limits, transient
                    _account(m, None, tag, time.time() - t0, False)
                    msg = str(e)[:160]
                    if "API key" in msg or "PERMISSION" in msg:
                        break
                    await asyncio.sleep(min(60, 2 ** attempt * 2))
            if resp is None:
                break
            text = resp.text or ""
            if search:
                try:
                    gm = resp.candidates[0].grounding_metadata
                    extra["sources"] = [{"uri": c.web.uri, "title": c.web.title}
                                        for c in (gm.grounding_chunks or []) if c.web]
                except Exception:  # noqa: BLE001
                    extra["sources"] = []
            if schema is None:
                out, ok = text, True
                break
            try:
                raw = text.strip()
                if raw.startswith("```"):
                    raw = raw.strip("`").split("\n", 1)[1].rsplit("```", 1)[0] if "\n" in raw else raw
                out, ok = _validate(schema, json.loads(raw)), True
                break
            except Exception as e:  # noqa: BLE001 invalid JSON: re-ask once
                contents = contents + [types.Content(role="model", parts=[types.Part.from_text(text=text)]),
                                       types.Content(role="user", parts=[types.Part.from_text(
                                           text=f"That was not valid JSON for the schema ({str(e)[:200]}). "
                                                "Return only the corrected JSON.")])]
    if not ok:
        STATS["failed"] += 1
        with open(_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "model": m, "tag": tag, "failed_item": True}) + "\n")
        return None
    if extra:
        out = {"result": out, **extra} if not isinstance(out, dict) else {**out, "_sources": extra.get("sources")}
    _cache_put(k, {"out": out, "model": m, "tag": tag})
    return out


async def embed(texts: list[str], model: str = "embed_model", batch: int = 0) -> list[list[float] | None]:
    """Cached embeddings, one call per text (gemini-embedding-2 folds a list into ONE embedding)."""
    m = resolve(model)

    async def one(t: str):
        k = _key("embed", m, t)
        hit = _cache_get(k)
        if hit is not None:
            STATS["cache_hits"] += 1
            return hit["out"]
        _check_budget()
        async with _sem():
            for attempt in range(C.cfg()["llm"]["max_retries"]):
                t0 = time.time()
                try:
                    r = await client().aio.models.embed_content(model=m, contents=t)
                    _account(m, None, "embed", time.time() - t0, True)
                    v = list(r.embeddings[0].values)
                    _cache_put(k, {"out": v})
                    return v
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2 ** attempt * 2)
        STATS["failed"] += 1
        return None

    return list(await asyncio.gather(*[one(t) for t in texts]))


def gather(coros, desc: str = ""):
    """Run coroutines concurrently from sync code; failures become None."""
    async def _run():
        async def safe(c):
            try:
                return await c
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                print(f"[{desc}] item failed: {str(e)[:160]}")
                return None
        return await asyncio.gather(*[safe(c) for c in coros])
    return asyncio.run(_run())


def stats_line() -> str:
    s = STATS
    return f"calls={s['calls']} hits={s['cache_hits']} failed={s['failed']} cost=${s['cost_usd']:.3f}"
