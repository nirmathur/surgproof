"""Config loader. Every narrator module reads parameters from here."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent


@lru_cache(maxsize=1)
def cfg() -> dict:
    return yaml.safe_load((PKG / "config.yaml").read_text())


def path(key: str, **fmt) -> Path:
    if key in ("out", "eval") and os.environ.get("NARRATOR_OUT"):  # separate run (e.g. real inputs)
        base = (ROOT / os.environ["NARRATOR_OUT"]).resolve()
        return base if key == "out" else base / "eval"
    p = cfg()["paths"][key].format(**fmt)
    return (ROOT / p).resolve()


def model(role: str) -> str:
    return cfg()["models"][role]


def split_of(video_id: str) -> str:
    s = cfg()["split"]
    if video_id in s["test"]:
        return "test"
    if video_id in s["tune"]:
        return "tune"
    return "unknown"


def load_env_key() -> str:
    """Return ANTIGRAVITY_API_KEY (env first, then repo .env). Never print it."""
    k = os.environ.get("ANTIGRAVITY_API_KEY")
    if k:
        return k
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ANTIGRAVITY_API_KEY="):
                k = line.split("=", 1)[1].strip().strip("'\"")
                os.environ["ANTIGRAVITY_API_KEY"] = k
                return k
    raise SystemExit("ANTIGRAVITY_API_KEY is unset. Stop: set it in the environment or .env.")


def out_dir(*parts) -> Path:
    p = path("out").joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p
