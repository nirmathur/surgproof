"""The README's install path, for real: clone, `make setup`, `make serve`, then use the running server over HTTP.

Needs about 400 MB and a network, so it only runs when asked:  SURGPROOF_CLEAN_CLONE=1 pytest -s tests/test_qa_clean_clone.py
It clones this checkout's HEAD, so run it from a commit holding all the lanes. The model is the mock: this tests install.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.e2e_helpers.harness import ROOT, synthetic_video

pytestmark = pytest.mark.skipif(os.environ.get("SURGPROOF_CLEAN_CLONE") != "1", reason="set SURGPROOF_CLEAN_CLONE=1: installs for real")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _host_header_status(port, host):
    con = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    con.putrequest("GET", "/api/health", skip_host=True)
    con.putheader("Host", host)
    con.endheaders()
    return con.getresponse().status


@pytest.fixture(scope="module")
def clone(tmp_path_factory):
    d = tmp_path_factory.mktemp("clean") / "surgproof"
    subprocess.run(["git", "clone", "--quiet", "--shared", str(ROOT), str(d)], check=True, timeout=300)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SURGPROOF_", "FAKE_CLAUDE_", "PYTHON"))}
    p = subprocess.run(["make", "setup"], cwd=d, env=env, capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    yield {"dir": d, "env": env, "setup_output": p.stdout + p.stderr}
    shutil.rmtree(d, ignore_errors=True)


def test_setup_leaves_a_private_environment_without_dev_tools(clone):
    py = clone["dir"] / ".venv" / "bin" / "python"
    assert py.exists()
    assert subprocess.run([str(py), "-c", "import fastapi, cv2, numpy"], capture_output=True, timeout=120).returncode == 0
    assert subprocess.run([str(py), "-c", "import pytest"], capture_output=True, timeout=60).returncode != 0, "dev tools in the runtime install"
    status = subprocess.run(["git", "status", "--porcelain"], cwd=clone["dir"], capture_output=True, text=True, timeout=60).stdout
    assert not status.strip(), f"make setup dirtied the tree: {status[:300]}"


def test_serve_works_from_the_clone_end_to_end(clone, tmp_path):
    port = _free_port()
    env = {**clone["env"], "SURGPROOF_VLM": "mock", "SURGPROOF_MIN_FREE_MB": "1", "HOST": "0.0.0.0", "PYTHONUNBUFFERED": "1"}
    log = tmp_path / "serve.log"
    with open(log, "w") as out:
        server = subprocess.Popen(["make", "serve", f"PORT={port}"], cwd=clone["dir"], env=env, stdout=out, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    jar = tmp_path / "cookies"

    def curl(*args):
        p = subprocess.run(["curl", "-s", "-m", "60", "-b", str(jar), "-c", str(jar), *args], capture_output=True, text=True, timeout=90)
        return p.stdout

    def code(path):
        return curl("-o", "/dev/null", "-w", "%{http_code}", base + path)

    try:
        deadline = time.time() + 90
        while time.time() < deadline and "ok" not in curl(base + "/api/health"):
            assert server.poll() is None, log.read_text()[-1500:]
            time.sleep(0.5)
        health = json.loads(curl(base + "/api/health"))
        assert health["tools"]["ffmpeg"] and health["disk"]

        listening = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True, text=True, timeout=30).stdout
        assert f"127.0.0.1:{port}" in listening and "*:" not in listening, f"HOST=0.0.0.0 was obeyed: {listening}"
        assert _host_header_status(port, "evil.example") == 400

        for page in ("/", "/app.js", "/style.css"):
            assert code(page) == "200", page
        assert code("/api/jobs") == "401", "the job list is open without signing in"

        found = re.search(r"setup code[^:]*:\s*(\S+)", log.read_text())
        assert found, f"no setup code printed: {log.read_text()[-500:]}"
        r = curl("-H", "content-type: application/json", "-d", json.dumps({"name": "Clean Clone", "password": "correct horse battery",
                                                                            "setup_code": found.group(1)}), base + "/api/auth/setup")
        assert "Clean Clone" in r, r
        notice = json.loads(curl(base + "/api/notice"))

        video = synthetic_video(tmp_path / "Jane_Doe.mp4", seconds=3)
        r = json.loads(curl("-F", f"file=@{video}", "-F", "rights_attested=yes", "-F", f"notice_sha256={notice['sha256']}",
                            "-F", "footage_source=public", base + "/api/jobs"))
        assert "id" in r, r
        assert "Jane_Doe" not in json.dumps(r)
        jid = r["id"]
        deadline, state = time.time() + 300, None
        while time.time() < deadline:
            job = json.loads(curl(f"{base}/api/jobs/{jid}"))
            state = job["state"]
            if state in ("done", "failed", "stalled"):
                break
            time.sleep(1)
        assert state == "done", job
        items = json.loads(curl(f"{base}/api/jobs/{jid}/final-pass"))["items"]
        assert items
        assert code(f"/api/jobs/{jid}/files/dataset_coco.json") == "403"
    finally:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
