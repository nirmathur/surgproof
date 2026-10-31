"""Thin async Gemini wrapper: caching, retries, concurrency limit, and a trace log for the UI."""
import asyncio
import hashlib
import io
import json
import os
import sqlite3
import time

from google import genai
from google.genai import types
from PIL import Image

import config as C


def _ensure_api_key():
    if os.environ.get("GEMINI_API_KEY"):
        return
    env_path = C.ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "GEMINI_API_KEY" and v.strip():
                os.environ["GEMINI_API_KEY"] = v.strip().strip("'").strip('"')
                return


_ensure_api_key()
client = genai.Client() if os.environ.get("GEMINI_API_KEY") else None
_sem = asyncio.Semaphore(C.CONCURRENCY)
_loop_sems = {}
_db = sqlite3.connect(C.CACHE_DB, timeout=30.0, check_same_thread=False)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("create table if not exists cache (k text primary key, v text)")
TRACE = C.OUT_DIR / "trace.jsonl"


def _get_client():
    global client
    if client is None:
        _ensure_api_key()
        client = genai.Client()
    return client


def _get_sem():
    loop = asyncio.get_running_loop()
    sem = _loop_sems.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(C.CONCURRENCY)
        _loop_sems[loop] = sem
    return sem


def trace(agent, msg, **kw):
    """Append a line the Streamlit 'agent reasoning' panel streams."""
    with open(TRACE, "a") as f:
        f.write(json.dumps({"t": time.time(), "agent": agent, "msg": msg, **kw}) + "\n")


def img_part(path_or_img, max_side=768, crop=None):
    im = Image.open(path_or_img).convert("RGB") if not isinstance(path_or_img, Image.Image) else path_or_img
    if crop and len(crop) == 4:  # crop = (ymin, xmin, ymax, xmax) in 0-1000 Gemini convention
        w, h = im.size
        y0, x0, y1, x1 = crop
        if y0 > y1:
            y0, y1 = y1, y0
        if x0 > x1:
            x0, x1 = x1, x0
        pad = 30
        left = max(0, min(w - 1, (x0 - pad) * w // 1000))
        upper = max(0, min(h - 1, (y0 - pad) * h // 1000))
        right = max(left + 1, min(w, (x1 + pad) * w // 1000))
        lower = max(upper + 1, min(h, (y1 + pad) * h // 1000))
        im = im.crop((left, upper, right, lower))
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), buf.getvalue()


def _key(model, parts_sig, prompt, schema_name, extra):
    return hashlib.sha256(json.dumps([model, parts_sig, prompt, schema_name, extra]).encode()).hexdigest()


async def call(prompt, *, model=C.FLASH, images=(), schema=None, tools=None, video_url=None,
               temperature=0.0, tag="", sample=0):
    """images: list of (Part, bytes). sample: change to get independent samples (votes)."""
    sig = [hashlib.md5(b).hexdigest() for _, b in images] + [video_url or ""]
    k = _key(model, sig, prompt, getattr(schema, "__name__", None), [temperature, sample, bool(tools)])
    row = _db.execute("select v from cache where k=?", (k,)).fetchone()
    if row:
        return json.loads(row[0])

    contents = [p for p, _ in images]
    if video_url:
        contents.append(types.Part(file_data=types.FileData(file_uri=video_url)))
    contents.append(prompt)
    cfg = dict(temperature=temperature)
    if schema is not None:
        cfg.update(response_mime_type="application/json", response_schema=schema)
    if tools:
        cfg["tools"] = tools
    async with _get_sem():
        for attempt in range(5):
            try:
                r = await _get_client().aio.models.generate_content(
                    model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
                out = json.loads(r.text) if schema is not None else r.text
                break
            except Exception as e:  # rate limits, transient errors
                wait = 2 ** attempt
                print(f"[{tag}] retry {attempt + 1} in {wait}s: {str(e)[:120]}")
                await asyncio.sleep(wait)
        else:
            raise RuntimeError(f"Gemini failed after retries [{tag}]")
    _db.execute("insert or replace into cache values (?,?)", (k, json.dumps(out)))
    _db.commit()
    return out


SEARCH = [types.Tool(google_search=types.GoogleSearch())]
