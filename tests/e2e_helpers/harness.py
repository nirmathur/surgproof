"""Shared e2e plumbing: build small videos, run the real CLI in a subprocess, read job outputs."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAKE_CLAUDE = Path(__file__).resolve().parent / "fake_claude.py"
LABELS = ["gallbladder", "liver", "grasper", "l_hook", "cystic_duct", "blood"]


def synthetic_video(path: Path, seconds: float = 3, size: str = "320x180", fps: int = 10, seed: int = 0) -> Path:
    """Moving test pattern. Not surgery: plumbing only. `seed` shifts the pattern, so videos can differ."""
    path.parent.mkdir(parents=True, exist_ok=True)
    src = f"testsrc2=s={size}:r={fps}:d={seconds}" if seed == 0 else f"mandelbrot=s={size}:r={fps}"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-t", str(seconds),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )
    return path


def base_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SURGPROOF_", "FAKE_CLAUDE_"))}
    env["PYTHONPATH"] = str(ROOT)
    env["SURGPROOF_VLM_BACKOFF_S"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def fake_env(mode: str = "normal", log: Path | None = None) -> dict:
    env = {"SURGPROOF_CLAUDE_BIN": str(FAKE_CLAUDE), "FAKE_CLAUDE_MODE": mode}
    if log:
        env["FAKE_CLAUDE_LOG"] = str(log)
    return env


def run_cli(video: Path, job: Path, env: dict, cwd: Path | None = None, extra_args: list[str] | None = None,
            timeout: int = 600) -> subprocess.CompletedProcess:
    """Runs from a temp cwd unless told otherwise, so nothing can lean on the repo's data/ or raw/."""
    job = Path(job)
    job.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "surgproof.run", str(video), "--out", str(job)] + (extra_args or [])
    return subprocess.run(cmd, cwd=cwd or job.parent, env=base_env(env), capture_output=True, text=True, timeout=timeout)


def run_eval(job: Path, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "surgproof.eval", str(job), *args], cwd=cwd, env=base_env(),
                          capture_output=True, text=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] if Path(path).exists() else []


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)], check=True, capture_output=True, text=True).stdout
    return int(out.strip().strip(","))


def video_fps(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0",
         str(path)], check=True, capture_output=True, text=True).stdout.strip().strip(",")
    n, _, d = out.partition("/")
    return float(n) / float(d or 1)


def cache_rows(cache: Path) -> dict:
    """key -> stored row. Empty if the file does not exist."""
    if not Path(cache).exists():
        return {}
    con = sqlite3.connect(cache)
    try:
        cur = con.execute("select key, model, prompt_hash, parsed, raw from vlm_cache")
        return {k: {"model": m, "prompt_hash": h, "parsed": json.loads(p), "raw": r} for k, m, h, p, r in cur}
    finally:
        con.close()


REVIEWER = "Reviewer One"
PASSWORD = "correct horse battery"


def local_client(app):
    """The server refuses any Host other than this machine's, so a test client has to say it is local."""
    from fastapi.testclient import TestClient

    return TestClient(app, base_url="http://127.0.0.1")


def sign_in(client, name: str = REVIEWER, password: str = PASSWORD) -> None:
    """Create the one local account on first use, as a person would, otherwise log in."""
    if client.get("/api/auth/state").json()["setup_required"]:
        body = {"name": name, "password": password, "setup_code": client.app.state.auth.setup_code()}
        r = client.post("/api/auth/setup", json=body)
    else:
        r = client.post("/api/auth/login", json={"password": password})
    assert r.status_code == 200, r.text


def upload(client, name: str, data: bytes, **fields):
    """A complete upload with the notice accepted. Pass a field as None to leave it out, as an attack would."""
    notice = client.get("/api/notice").json()
    form = {"rights_attested": "yes", "notice_sha256": notice["sha256"], "footage_source": notice["footage_sources"][0], **fields}
    form = {k: v for k, v in form.items() if v is not None}
    return client.post("/api/jobs", files={"file": (name, data, "video/mp4")}, data=form)


HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
HAVE_DATA = (ROOT / "raw" / "CholecSeg8k" / "video12" / "video12_19500").is_dir()
