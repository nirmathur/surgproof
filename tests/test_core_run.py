"""The CLI, run as a real subprocess with the mock provider."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cli(*args, cwd, **env):
    full = {**os.environ, "SURGPROOF_VLM": "mock", "PYTHONPATH": str(ROOT), **env}
    full.pop("SURGPROOF_CACHE", None)
    return subprocess.run([sys.executable, "-m", "surgproof.run", *map(str, args)], cwd=cwd, env=full, capture_output=True, text=True)


def video(path):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x180:rate=25",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path


def test_cli_runs_model_stages_and_says_it_is_mock(tmp_path):
    out = tmp_path / "jobs" / "one"
    res = cli(video(tmp_path / "v.mp4"), "--out", out, "--stop-after", "amber", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "mock:" in res.stderr and "job folder" in res.stdout and "detect" in res.stdout
    assert json.load(open(out / "status.json"))["stage"] == "done"
    assert (tmp_path / "jobs" / "cache.sqlite").exists()
    assert len(json.load(open(out / "meta.json"))["keyframes"]) == 2


def test_cli_kf_fps(tmp_path):
    out = tmp_path / "jobs" / "two"
    res = cli(video(tmp_path / "v.mp4"), "--out", out, "--kf-fps", "2", "--stop-after", "ingest", cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert len(json.load(open(out / "meta.json"))["keyframes"]) == 4


def test_cli_reports_failure_with_exit_code_and_status(tmp_path):
    out = tmp_path / "jobs" / "bad"
    res = cli(tmp_path / "missing.mp4", "--out", out, cwd=tmp_path)
    assert res.returncode == 1 and "FAILED" in res.stderr and "Traceback" not in res.stderr
    assert "no such video" in json.load(open(out / "status.json"))["error"]
