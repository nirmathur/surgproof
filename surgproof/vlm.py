"""Vision model client. ask(image_paths, prompt, schema, model) -> parsed, validated JSON.

Providers (SURGPROOF_VLM): claude_cli (default, headless Claude Code, no API key), gemini, mock.
Every accepted response is cached with its raw text, so each label traces back to a model output.
Mock results are tagged "mock:<model>" everywhere, so they can never pass for real ones.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path

SONNET = "claude-sonnet-5-5"
HAIKU = "claude-haiku-5-5"
DEFAULT_CONCURRENCY = 6
MAX_RETRIES = 2  # re-asks after invalid JSON, per spec
TRANSIENT_RETRIES = 3  # re-tries after a CLI failure (rate limit, network, binary being updated)
CALL_TIMEOUT_S = 300
SYSTEM_PROMPT = (
    "You are a careful annotator of laparoscopic surgery video. Use the Read tool to open any image "
    "file named in the request. Your entire final reply must be one JSON object and nothing else."
)
GEMINI_MODELS = {SONNET: "gemini-4-argon", HAIKU: "gemini-3.8-flash"}

_lock = threading.Lock()
_cache_path = Path(os.environ.get("SURGPROOF_CACHE", "jobs/cache.sqlite"))
_limit = int(os.environ.get("SURGPROOF_VLM_CONCURRENCY", DEFAULT_CONCURRENCY))
_sem = threading.BoundedSemaphore(_limit)
_stats = {"calls": 0, "cache_hits": 0, "cost_usd": 0.0}


class VlmError(RuntimeError):
    """The model could not produce a valid answer. Callers must not treat this as 'absent'."""


class VlmAuthError(VlmError):
    """Not logged in or no credentials. Retrying cannot help, so the job should stop."""


def configure(cache_path=None, concurrency=None):
    global _cache_path, _sem, _limit
    with _lock:
        if cache_path is not None and "SURGPROOF_CACHE" not in os.environ:
            _cache_path = Path(cache_path)
        if concurrency and concurrency != _limit:
            _limit = int(concurrency)
            _sem = threading.BoundedSemaphore(_limit)


def provider():
    return os.environ.get("SURGPROOF_VLM", "claude_cli")


def model_tag(model):
    """The model name recorded on every output. Anything but the default provider says so."""
    p = provider()
    return model if p == "claude_cli" else f"{p}:{model}"


def stats():
    with _lock:
        return dict(_stats)


def _bump(key, by=1):
    with _lock:
        _stats[key] += by


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def cache_key(image_paths, prompt, model):
    """sha256(image bytes + prompt + model), so a re-uploaded video reuses its answers."""
    h = hashlib.sha256()
    for p in image_paths:
        h.update(Path(p).read_bytes())
    h.update(prompt.encode())
    h.update(model_tag(model).encode())
    return h.hexdigest()


def _db():
    _cache_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_cache_path, timeout=30.0)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute(
        "create table if not exists vlm_cache (key text primary key, model text, prompt_hash text, prompt text, "
        "parsed text, raw text, cost_usd real, duration_ms integer, created real)"
    )
    return con


def _cache_get(key):
    con = _db()
    try:
        row = con.execute("select parsed from vlm_cache where key=?", (key,)).fetchone()
    finally:
        con.close()
    return json.loads(row[0]) if row else None


def _cache_put(key, model, prompt, parsed, raw, cost, duration_ms):
    con = _db()
    try:
        with con:
            con.execute(
                "insert or replace into vlm_cache values (?,?,?,?,?,?,?,?,?)",
                (key, model_tag(model), prompt_hash(prompt), prompt, json.dumps(parsed), raw, cost, duration_ms, time.time()),
            )
    finally:
        con.close()


def get_raw(key):
    """The stored raw response behind a raw_ref, or None if there is no such record."""
    con = _db()
    try:
        row = con.execute("select model, prompt, raw, parsed, cost_usd, created from vlm_cache where key=?", (key,)).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return dict(zip(("model", "prompt", "raw", "parsed", "cost_usd", "created"), row))


# ---- JSON handling -------------------------------------------------------------------------------

def extract_json(text):
    """Return the first JSON object or array in text, tolerating fences or prose around it."""
    if not isinstance(text, str):
        raise ValueError("model reply is not text")
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return dec.raw_decode(text[i:])[0]
            except ValueError:
                continue
    raise ValueError("no JSON found in model reply")


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _type_ok(value, t):
    if t == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if t == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, _TYPES[t])


def validate(value, schema, path="$"):
    """Small JSON Schema subset: type, enum, required, properties, additionalProperties,
    items, minItems, maxItems, minimum, maximum. Raises ValueError on the first problem."""
    t = schema.get("type")
    if t is not None:
        kinds = t if isinstance(t, list) else [t]
        if not any(_type_ok(value, k) for k in kinds):
            raise ValueError(f"{path}: expected {t}, got {type(value).__name__}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in value:
                raise ValueError(f"{path}: missing {k}")
        if schema.get("additionalProperties") is False:
            extra = set(value) - set(props)
            if extra:
                raise ValueError(f"{path}: unexpected {sorted(extra)}")
        for k, sub in props.items():
            if k in value:
                validate(value[k], sub, f"{path}.{k}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"{path}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ValueError(f"{path}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, v in enumerate(value):
                validate(v, schema["items"], f"{path}[{i}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path}: above maximum")


# ---- providers: each returns (raw_text, cost_usd) ----------------------------------------------

def claude_command(prompt, model):
    """Minimal flags: no user settings, CLAUDE.md, hooks or MCP reach the model. That keeps labels free
    of unrelated context and cuts a call from about $0.12 to about $0.01 (measured)."""
    return [
        os.environ.get("SURGPROOF_CLAUDE_BIN", "claude"), "-p", prompt,
        "--model", model,
        "--output-format", "json",
        "--system-prompt", SYSTEM_PROMPT,
        "--tools", "Read",
        "--allowedTools", "Read",
        "--setting-sources", "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--no-session-persistence",
    ]


def _claude_cli(image_paths, prompt, model):
    # Each call runs in an empty folder holding only its images under neutral names: no project
    # context is discovered from the folder, and no file name hints at the source or a verdict.
    with tempfile.TemporaryDirectory(prefix="surgproof_vlm_") as tmp:
        tmp = os.path.realpath(tmp)
        refs = []
        for i, src in enumerate(image_paths, 1):
            dst = os.path.join(tmp, f"image_{i}{Path(src).suffix.lower()}")
            shutil.copyfile(src, dst)
            refs.append(dst)
        text = prompt
        if refs:
            text = "Use the Read tool to open: " + ", ".join(refs) + "\n\n" + prompt
        try:
            proc = subprocess.run(
                claude_command(text, model), cwd=tmp, capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=CALL_TIMEOUT_S,
            )
        except FileNotFoundError as e:
            raise VlmError(f"claude binary not found: {e}") from e
        except subprocess.TimeoutExpired as e:
            raise VlmError(f"claude timed out after {CALL_TIMEOUT_S}s") from e
    try:
        env = json.loads(proc.stdout)
    except ValueError:
        env = None
    if not isinstance(env, dict):
        raise VlmError(f"claude exited {proc.returncode}, no JSON envelope: {(proc.stderr or proc.stdout)[:300]}")
    if env.get("is_error") or proc.returncode != 0:
        detail = str(env.get("result"))[:300]
        if "not logged in" in detail.lower() or "/login" in detail.lower():
            raise VlmAuthError(detail)
        raise VlmError(f"claude reported an error: {detail}")
    return env.get("result", ""), float(env.get("total_cost_usd") or 0.0)


def _gemini(image_paths, prompt, model):
    from google import genai  # optional, only for SURGPROOF_VLM=gemini
    from google.genai import types

    if not os.environ.get("GEMINI_API_KEY"):
        raise VlmAuthError("SURGPROOF_VLM=gemini needs GEMINI_API_KEY")
    parts = [types.Part.from_bytes(data=Path(p).read_bytes(), mime_type="image/jpeg") for p in image_paths]
    reply = genai.Client().models.generate_content(
        model=GEMINI_MODELS.get(model, model),
        contents=parts + [prompt],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )
    return reply.text, 0.0


def _mock(image_paths, prompt, model):
    """Offline stand-in for tests. Its answers are fixed shapes, not perception; see surgproof/prompts.py."""
    from surgproof.prompts import mock_reply

    return json.dumps(mock_reply(prompt, image_paths, model)), 0.0


PROVIDERS = {"claude_cli": _claude_cli, "gemini": _gemini, "mock": _mock}


def _call_provider(image_paths, prompt, model):
    name = provider()
    if name not in PROVIDERS:
        raise VlmError(f"unknown SURGPROOF_VLM provider: {name}")
    last = None
    for attempt in range(TRANSIENT_RETRIES):
        try:
            return PROVIDERS[name](image_paths, prompt, model)
        except VlmAuthError:
            raise
        except VlmError as e:
            last = e
            if attempt < TRANSIENT_RETRIES - 1:
                time.sleep(2 ** attempt * _backoff_s())
    raise last


def _backoff_s():
    return float(os.environ.get("SURGPROOF_VLM_BACKOFF_S", 5))


def ask(image_paths, prompt, schema, model=SONNET):
    """Ask the model about images. Returns JSON matching schema, or raises VlmError.

    The prompt must not name image paths; they are added per call so that the cache key stays
    stable across jobs. Only validated answers are cached.
    """
    image_paths = [str(p) for p in image_paths]
    key = cache_key(image_paths, prompt, model)
    cached = _cache_get(key)
    if cached is not None:
        _bump("cache_hits")
        return cached
    error = None
    for attempt in range(MAX_RETRIES + 1):
        text = prompt
        if error:
            text = f"{prompt}\n\nYour previous reply was rejected ({error}). Reply with only the required JSON."
        t0 = time.time()
        with _sem:
            raw, cost = _call_provider(image_paths, text, model)
        _bump("calls")
        _bump("cost_usd", cost)
        try:
            parsed = extract_json(raw)
            validate(parsed, schema)
        except ValueError as e:
            error = str(e)[:200]
            continue
        _cache_put(key, model, prompt, parsed, raw, cost, int((time.time() - t0) * 1000))
        return parsed
    raise VlmError(f"invalid JSON after {MAX_RETRIES + 1} attempts: {error}")
