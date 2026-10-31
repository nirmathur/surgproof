"""Tests for tools/record_demo.py (lane: recorder)."""
from __future__ import annotations

import http.server
import re
import socketserver
import subprocess
import threading
from pathlib import Path

import pytest

from tools import record_demo as rd


def test_parse_args_defaults_and_overrides():
    args_default = rd.parse_args([])
    assert args_default.target == "auto"
    assert args_default.web_url == "http://localhost:8600"
    assert args_default.streamlit_url == "http://localhost:8501"
    assert 60.0 <= args_default.duration <= 90.0

    args_web = rd.parse_args(["--target", "web", "--duration", "72.5", "--web-url", "http://127.0.0.1:8600"])
    assert args_web.target == "web"
    assert args_web.duration == 72.5
    assert args_web.web_url == "http://127.0.0.1:8600"

    args_watch = rd.parse_args(["--target", "watch", "--duration", "30.0"])
    assert args_watch.target == "watch"
    assert args_watch.duration == 30.0
    assert args_watch.watch_url == "http://localhost:8600/watch.html"


def test_is_url_up_with_ephemeral_server():
    class QuietHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    with socketserver.TCPServer(("127.0.0.1", 0), QuietHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        assert rd.is_url_up(f"http://127.0.0.1:{port}/", timeout=2.0) is True
        httpd.shutdown()

    assert rd.is_url_up("http://127.0.0.1:59999/", timeout=0.4) is False


def test_transcode_and_probe_video(tmp_path: Path):
    raw_webm = tmp_path / "sample.webm"
    out_mp4 = tmp_path / "sample_out.mp4"
    # Generate a 1.5s 640x360 test pattern webm via ffmpeg lavfi
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360:d=1.5:r=15",
            "-c:v",
            "libvpx",
            "-b:v",
            "200k",
            str(raw_webm),
        ],
        check=True,
        capture_output=True,
    )
    meta = rd.transcode_webm_to_mp4(raw_webm, out_mp4, min_duration=3.0, max_duration=6.0)
    assert out_mp4.is_file()
    assert meta["codec_name"] == "h264"
    assert meta["width"] == 1920
    assert meta["height"] == 1080
    assert 3.0 <= meta["duration"] <= 6.5
    assert meta["size_bytes"] > 1000


def test_record_demo_never_mutates_data_or_fabricates():
    src = Path(rd.__file__).read_text()
    # Must not contain any integrity.py forbidden patterns
    forbidden = [
        r'["\']?box["\']?\s*:\s*\[\s*\d',
        r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']',
        r'["\']\d{1,3}\.\d%["\']',
        r'0\.7\d?\s*\+\s*0\.\d+\s*\*',
        r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}',
    ]
    for pat in forbidden:
        assert not re.search(pat, src), f"Forbidden pattern {pat} found in tools/record_demo.py"

    # Must never write to decisions.csv, results.json, annotations.jsonl, or gate_log.jsonl
    for data_artifact in ("decisions.csv", "results.json", "annotations.jsonl", "gate_log.jsonl", "verified_dataset.jsonl"):
        assert data_artifact not in src


def test_existing_recordings_meet_spec():
    out_dir = rd.OUT_DIR
    for name in ("recording_streamlit.mp4", "recording_web.mp4", "recording_web_v1.mp4"):
        p = out_dir / name
        if p.is_file():
            meta = rd.probe_video(p)
            assert meta["codec_name"] == "h264"
            assert meta["width"] == 1920
            assert meta["height"] == 1080
            assert 60.0 <= meta["duration"] <= 90.0
            assert meta["size_bytes"] > 50_000

    watch_p = out_dir / "recording_watch.mp4"
    if watch_p.is_file():
        meta = rd.probe_video(watch_p)
        assert meta["codec_name"] == "h264"
        assert meta["width"] == 1920
        assert meta["height"] == 1080
        assert 28.0 <= meta["duration"] <= 40.0
        assert meta["size_bytes"] > 30_000
