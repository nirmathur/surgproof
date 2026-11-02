"""Regression tests for the web section of the hardening brief. Each reproduces the attack, over real HTTP routes."""
from __future__ import annotations

import ast
import json
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("ffmpeg is required", allow_module_level=True)

from surgproof import server, webauth  # noqa: E402
from tests import web_fixture as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
HANG = "tests.web_fixture:pipeline_hang"


@pytest.fixture(autouse=True)
def roomy_disk(monkeypatch):
    """This machine runs low on disk, and only the disk tests here should depend on that."""
    monkeypatch.setenv("SURGPROOF_MIN_FREE_MB", "1")


@pytest.fixture
def jobs(tmp_path):
    return tmp_path / "jobs"


def open_client(jobs, target="tests.web_fixture:pipeline_ok"):
    return fx.local_client(server.create_app(jobs, pipeline_target=target))


@pytest.fixture
def anon(jobs):
    with open_client(jobs) as client:
        yield client


@pytest.fixture
def client(jobs):
    with open_client(jobs) as c:
        fx.sign_in(c, jobs)
        yield c


@pytest.fixture
def done(jobs):
    pytest.importorskip("surgproof.export")
    return fx.finished_job(jobs)


def wait_for(predicate, seconds=180):
    end = time.time() + seconds
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("timed out")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def send_video(client, video: Path, name="clip.mp4", **fields):
    with video.open("rb") as fh:
        return client.post("/api/jobs", files={"file": (name, fh, "video/mp4")}, data=fields)


def state_of(client, job_id):
    return client.get(f"/api/jobs/{job_id}").json()


def clear_and_release(client):
    for item in client.get("/api/jobs/job-done/final-pass").json()["items"]:
        assert client.post("/api/jobs/job-done/final-pass",
                           json={"action": "approve", "kf": item["kf"], "label": item["label"]}).status_code == 200
    assert client.post("/api/jobs/job-done/final-pass", json={"action": "release"}).status_code == 200


# ---- 1. network exposure ----

@pytest.mark.parametrize("env_host", ["0.0.0.0", "::", "192.168.1.5"])
def test_host_setting_cannot_expose_the_server(monkeypatch, tmp_path, env_host):
    """The attack: HOST=0.0.0.0 make serve."""
    import uvicorn

    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: captured.update(kw))
    monkeypatch.setattr(sys, "argv", ["surgproof.server"])
    monkeypatch.setenv("HOST", env_host)
    monkeypatch.setenv("PORT", "0")
    monkeypatch.setenv("SURGPROOF_JOBS", str(tmp_path / "jobs"))
    try:
        server.main()
    except SystemExit as exc:
        pytest.skip(f"cannot bind a local port here: {exc}")
    assert captured["host"] == "127.0.0.1"


def test_concurrent_setup_creates_one_account(jobs):
    jobs.mkdir()
    auth = webauth.LocalAuth(jobs)
    code = auth.setup_code()
    results = []

    def attempt(name):
        try:
            results.append(auth.setup(name, "long enough pw", code))
        except webauth.AuthError as exc:
            results.append(exc.status)

    import threading

    threads = [threading.Thread(target=attempt, args=(f"N{i}",)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(isinstance(r, str) for r in results) == 1


def test_no_instruction_exposes_the_server():
    for path in (ROOT / "Makefile", ROOT / "surgproof" / "server.py"):
        assert "0.0.0.0" not in path.read_text(encoding="utf-8"), path.name
    assert not re.search(r"environ(\.get)?[\[(]\s*[\"']HOST[\"']", (ROOT / "surgproof" / "server.py").read_text())


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8600", "localhost.evil.example", "192.168.1.5:8600"])
def test_dns_rebinding_host_is_refused(anon, host):
    assert anon.get("/api/health", headers={"Host": host}).status_code == 400
    assert anon.get("/", headers={"Host": host}).status_code == 400


@pytest.mark.parametrize("host", ["localhost", "localhost:8600", "127.0.0.1:8600"])
def test_this_machine_is_accepted(anon, host):
    assert anon.get("/api/health", headers={"Host": host}).status_code == 200


@pytest.mark.parametrize("headers", [
    {"Origin": "https://evil.example"},
    {"Origin": "http://localhost:9999"},
    {"Sec-Fetch-Site": "cross-site"},
    {"Sec-Fetch-Site": "same-site"},
])
def test_other_sites_cannot_post_even_with_a_session(client, headers):
    res = client.post("/api/auth/logout", headers=headers)
    assert res.status_code == 403
    assert client.get("/api/jobs").status_code == 200, "the refused request must not have logged us out"


def test_api_docs_are_not_exposed(anon):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert anon.get(path).status_code == 404, path


# ---- 1. login ----

@pytest.mark.parametrize("method,path", [
    ("get", "/api/jobs"), ("get", "/api/jobs/job-done"), ("get", "/api/jobs/job-done/watch"),
    ("get", "/api/jobs/job-done/overlay?start=0&end=5"), ("get", "/api/jobs/job-done/video"),
    ("get", "/api/jobs/job-done/preview"), ("get", "/api/jobs/job-done/final-pass"),
    ("get", "/api/jobs/job-done/files/frames/kf_00000.jpg"), ("get", "/api/notice"),
    ("post", "/api/jobs"), ("post", "/api/jobs/job-done/retry"), ("post", "/api/jobs/job-done/final-pass"),
    ("delete", "/api/jobs/job-done"),
])
def test_every_job_route_needs_a_session(anon, done, method, path):
    assert getattr(anon, method)(path).status_code == 401
    assert (done / "status.json").exists()


def test_upload_is_refused_before_the_body_is_read_without_a_session(anon, jobs, tmp_path):
    res = send_video(anon, fx.make_video(tmp_path / "clip.mp4"), **fx.attestation())
    assert res.status_code == 401
    assert server.Store(jobs).list_ids() == []
    assert not list(jobs.glob(".upload-*"))


def test_first_visitor_cannot_claim_the_account_without_the_setup_code(anon, jobs):
    body = {"name": "Intruder", "password": "long enough pw", "setup_code": "guess"}
    assert anon.post("/api/auth/setup", json=body).status_code == 403
    assert not (jobs / ".account.json").exists()
    assert anon.get("/api/auth/state").json() == {"setup_required": True, "user": None}


def test_account_files_are_private_and_the_code_is_one_use(anon, jobs):
    code_path = jobs / ".setup_code"
    assert stat.S_IMODE(code_path.stat().st_mode) == 0o600
    fx.sign_in(anon, jobs)
    assert stat.S_IMODE((jobs / ".account.json").stat().st_mode) == 0o600
    assert not code_path.exists()
    assert fx.TEST_PASSWORD not in (jobs / ".account.json").read_text()
    res = anon.post("/api/auth/setup", json={"name": "Second", "password": "another password", "setup_code": "x"})
    assert res.status_code == 409


@pytest.mark.parametrize("name,password,status", [("", "long enough pw", 422), ("Me", "short", 422)])
def test_weak_accounts_are_refused(anon, jobs, name, password, status):
    code = (jobs / ".setup_code").read_text()
    assert anon.post("/api/auth/setup", json={"name": name, "password": password, "setup_code": code}).status_code == status


def test_login_session_cookie_and_logout(anon, jobs):
    fx.sign_in(anon, jobs)
    anon.post("/api/auth/logout")
    assert anon.get("/api/jobs").status_code == 401
    res = anon.post("/api/auth/login", json={"password": "wrong password"})
    assert res.status_code == 401
    res = anon.post("/api/auth/login", json={"password": fx.TEST_PASSWORD})
    assert res.status_code == 200 and res.json()["user"] == fx.TEST_USER
    cookie = res.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert anon.get("/api/jobs").status_code == 200


def test_old_session_cookie_is_dead_after_logout(anon, jobs):
    fx.sign_in(anon, jobs)
    token = anon.cookies[webauth.COOKIE_NAME]
    anon.post("/api/auth/logout")
    anon.cookies.clear()
    anon.cookies.set(webauth.COOKIE_NAME, token)
    assert anon.get("/api/jobs").status_code == 401


def test_password_guessing_is_throttled(anon, jobs):
    fx.sign_in(anon, jobs)
    anon.post("/api/auth/logout")
    codes = [anon.post("/api/auth/login", json={"password": f"guess {i}"}).status_code for i in range(webauth.MAX_FAILURES + 1)]
    assert codes[:webauth.MAX_FAILURES] == [401] * webauth.MAX_FAILURES and codes[-1] == 429
    assert anon.post("/api/auth/login", json={"password": fx.TEST_PASSWORD}).status_code == 429


def test_session_expires(anon, jobs, monkeypatch):
    fx.sign_in(anon, jobs)
    real = time.time
    monkeypatch.setattr(webauth.time, "time", lambda: real() + webauth.SESSION_SECONDS + 1)
    assert anon.get("/api/jobs").status_code == 401


def test_reviewer_identity_cannot_be_typed_in(client, done):
    """The attack: claim someone else's name in the body."""
    res = client.post("/api/jobs/job-done/final-pass",
                      json={"action": "reject", "kf": "kf_00001", "label": "gallbladder", "reviewer": "Dr Somebody Else"})
    assert res.status_code == 200
    row = json.loads((done / "final_pass.jsonl").read_text().splitlines()[-1])
    assert row["reviewer"] == fx.TEST_USER
    for item in client.get("/api/jobs/job-done/final-pass").json()["items"]:
        if not item["verdict"]:
            client.post("/api/jobs/job-done/final-pass", json={"action": "approve", "kf": item["kf"], "label": item["label"]})
    client.post("/api/jobs/job-done/final-pass", json={"action": "release", "reviewer": "Dr Somebody Else"})
    info = json.loads((done / "released_coco.json").read_text())["info"]
    assert info["reviewer"] == fx.TEST_USER
    assert "Somebody" not in (done / "released_coco.json").read_text()
    assert "Somebody" not in (done / "final_pass.jsonl").read_text()


def test_oversized_json_is_refused(anon):
    big = {"password": "x" * (server.MAX_JSON_BYTES + 10)}
    assert anon.post("/api/auth/login", json=big).status_code == 413


# ---- 1. file allowlist ----

@pytest.mark.parametrize("name", ["annotated.mp4", "provenance.jsonl", "dataset_coco.json", "released_coco.json", "final_pass.jsonl"])
def test_unreleased_artifacts_are_gated(client, done, name):
    (done / "final_pass.jsonl").write_text("", encoding="utf-8")
    assert client.get(f"/api/jobs/job-done/files/{name}").status_code == 403
    assert client.get(f"/api/jobs/job-done/files/{name}?download=1").status_code == 403


def test_released_artifacts_are_served_after_release(client, done):
    clear_and_release(client)
    for name in ("annotated.mp4", "released_coco.json", "provenance.jsonl", "final_pass.jsonl"):
        assert client.get(f"/api/jobs/job-done/files/{name}").status_code == 200, name
    for name in server.DOWNLOADABLE:  # the rest appear as the release produces them
        assert client.get(f"/api/jobs/job-done/files/{name}").status_code == (200 if (done / name).exists() else 404), name


@pytest.mark.parametrize("path", [
    "worker.log", "job.json", "status.json", "meta.json", "detect.jsonl", "verify.jsonl", "decisions.jsonl",
    "masks/kf_00000_grasper.png", "tracks/5_grasper.png", "tracks.jsonl", "masks.jsonl", "source.mp4", "upload.mp4",
    "final_pass_queue.json", ".worker.lock", "frames/../job.json", "frames/kf_00000.jpg/", "frames/x.jpg",
    "frames/kf_00000.png", "../.account.json", "../.setup_code", "DATASET_COCO.JSON",
])
def test_everything_off_the_allowlist_is_not_served_even_after_release(client, done, path):
    (done / "worker.log").write_text("internal paths", encoding="utf-8")
    (done / "upload.mp4").write_bytes(b"x")
    clear_and_release(client)
    assert client.get(f"/api/jobs/job-done/files/{path}").status_code == 404, path


def test_release_artifacts_stay_gated_before_release(client, done):
    for name in ("dataset_bundle.zip", "release_manifest.json"):
        (done / name).write_bytes(b"x")
        assert client.get(f"/api/jobs/job-done/files/{name}").status_code == 403


def test_audited_propagated_frame_is_served_like_a_keyframe(client, done):
    shutil.copy(done / "frames" / "kf_00000.jpg", done / "frames" / "prop_000005.jpg")
    assert client.get("/api/jobs/job-done/files/frames/prop_000005.jpg").status_code == 200


def test_frames_are_served_for_the_review_page(client, done):
    res = client.get("/api/jobs/job-done/files/frames/kf_00000.jpg")
    assert res.status_code == 200 and res.content[:2] == b"\xff\xd8"


def test_symlinked_frame_is_not_followed_out_of_the_job(client, done, tmp_path):
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"SECRET")
    target = done / "frames" / "kf_00099.jpg"
    os.symlink(secret, target)
    assert client.get("/api/jobs/job-done/files/frames/kf_00099.jpg").status_code == 404


def test_symlinked_job_folder_is_not_a_job(client, jobs, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "status.json").write_text('{"stage":"detect"}')
    os.symlink(outside, jobs / "job-link")
    assert client.get("/api/jobs/job-link").status_code == 404
    assert "job-link" not in [j["id"] for j in client.get("/api/jobs").json()["jobs"]]
    assert client.delete("/api/jobs/job-link").status_code == 404
    assert outside.exists()


# ---- 2. notice and attestation ----

@pytest.mark.parametrize("fields", [
    {},
    {"rights_attested": "yes"},
    {"rights_attested": "yes", "footage_source": "public"},
    {"rights_attested": "no", "footage_source": "public", "notice_sha256": server.NOTICE_SHA256},
    {"rights_attested": "yes", "footage_source": "public", "notice_sha256": "an old notice"},
    {"rights_attested": "yes", "footage_source": "real_patient", "notice_sha256": server.NOTICE_SHA256},
    {"rights_attested": "yes", "notice_sha256": server.NOTICE_SHA256},
])
def test_upload_is_blocked_without_the_attestation(client, jobs, tmp_path, fields):
    res = send_video(client, fx.make_video(tmp_path / "clip.mp4"), **fields)
    assert res.status_code == 422
    assert server.Store(jobs).list_ids() == []
    assert not list(jobs.glob(".upload-*"))


def test_notice_says_what_it_must(client):
    body = client.get("/api/notice").json()
    assert "third-party" in body["text"] and "public or simulator" in body["text"]
    assert "patient" in body["text"] and "cholecystectomy" in body["text"]
    assert body["sha256"] == server.NOTICE_SHA256


def test_attestation_is_stored_with_the_session_identity(client, jobs, tmp_path):
    res = send_video(client, fx.make_video(tmp_path / "clip.mp4"), **fx.attestation(footage_source="simulator"))
    assert res.status_code == 201
    record = json.loads((jobs / res.json()["id"] / "job.json").read_text())
    stored = record["attestation"]
    assert stored["by"] == fx.TEST_USER and stored["footage_source"] == "simulator"
    assert stored["notice"] == server.UPLOAD_NOTICE and stored["notice_sha256"] == server.NOTICE_SHA256
    assert stored["at"]


def test_upload_filename_is_never_stored_or_listed(client, jobs, tmp_path):
    """A filename like a patient's name must not survive into job records or listings."""
    video = fx.make_video(tmp_path / "clip.mp4")
    res = send_video(client, video, name="SMITH_JOHN_mrn12345.mp4", **fx.attestation())
    assert res.status_code == 201
    job_id = res.json()["id"]
    wait_for(lambda: (jobs / job_id / "worker.log").exists())
    for text in (res.text, client.get("/api/jobs").text, client.get(f"/api/jobs/{job_id}").text,
                 (jobs / job_id / "job.json").read_text(), (jobs / job_id / "status.json").read_text()):
        assert "SMITH" not in text and "mrn12345" not in text
    assert not [p for p in (jobs / job_id).iterdir() if "SMITH" in p.name]
    assert client.get(f"/api/jobs/{job_id}").json()["name"] == f"Job {job_id[-6:]}"


def test_original_name_in_meta_is_not_passed_to_the_browser(client, done):
    meta = json.loads((done / "meta.json").read_text())
    (done / "meta.json").write_text(json.dumps({**meta, "original_name": "SMITH_JOHN.mp4"}))
    assert "SMITH" not in client.get("/api/jobs/job-done").text


def test_original_upload_goes_once_ingest_is_done_but_retry_still_works(jobs, tmp_path, monkeypatch):
    """On the real pipeline with the mock model. Audio and metadata must not survive anywhere in the job."""
    pytest.importorskip("surgproof.pipeline")
    pytest.importorskip("surgproof.export")
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    tagged = tmp_path / "tagged.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=3:size=320x180:rate=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", "-metadata", "title=SMITH_JOHN_MRN12345", str(tagged)], check=True)
    with fx.local_client(server.create_app(jobs)) as real:
        fx.sign_in(real, jobs)
        job_id = send_video(real, tagged, **fx.attestation()).json()["id"]
        job = jobs / job_id
        body = wait_state(real, job_id, "done", "failed", "stalled")
        assert body["state"] == "done", body
        assert not list(job.glob("upload.*")), "the original file, with its audio and metadata, is still there"
        record = json.loads((job / "job.json").read_text())
        assert record["upload"] is None and record["upload_deleted_at"]
        for path in job.rglob("*"):
            if path.is_file():
                assert b"SMITH_JOHN" not in path.read_bytes(), path.name
        streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                                  str(job / "source.mp4")], capture_output=True, text=True).stdout.split()
        assert streams == ["video"]

        # Retry has no upload left to use, so it must run from the stripped copy.
        (job / "annotated.mp4").unlink()
        wait_state(real, job_id, "failed")  # once the worker has exited
        assert real.post(f"/api/jobs/{job_id}/retry").status_code == 200
        body = wait_state(real, job_id, "done", "failed", "stalled")
        assert body["state"] == "done", body
        assert (job / "annotated.mp4").exists()
        assert real.get(f"/api/jobs/{job_id}/video").status_code == 200


def test_upload_is_kept_while_ingest_has_not_finished(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    (job / "upload.mp4").write_bytes(b"original")
    (job / "job.json").write_text(json.dumps({"upload": "upload.mp4"}))
    (job / "status.json").write_text(json.dumps({"stage": "ingest", "error": None}))
    (job / "meta.json").write_text(json.dumps({"source": "source.mp4"}))
    assert server.drop_upload_after_ingest(job) is False, "no source.mp4 yet"
    (job / "source.mp4").write_bytes(b"stripped")
    assert server.drop_upload_after_ingest(job) is False, "ingest is still running"
    assert (job / "upload.mp4").exists()
    (job / "status.json").write_text(json.dumps({"stage": "detect", "error": None}))
    assert server.drop_upload_after_ingest(job) is True
    assert not (job / "upload.mp4").exists()
    assert server.pipeline_input(job).name == "source.mp4"


def test_leftover_upload_is_dropped_at_startup(jobs):
    """A server that died mid-job never got to delete it."""
    job = jobs / "job-x"
    job.mkdir(parents=True)
    (job / "upload.mp4").write_bytes(b"original")
    (job / "source.mp4").write_bytes(b"stripped")
    (job / "job.json").write_text(json.dumps({"upload": "upload.mp4"}))
    (job / "meta.json").write_text(json.dumps({"source": "source.mp4"}))
    (job / "status.json").write_text(json.dumps({"stage": "render", "error": None}))
    with open_client(jobs):
        assert not (job / "upload.mp4").exists()


def test_kf_fps_is_capped(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    for bad in ("6", "0", "-1", "lots"):
        assert send_video(client, video, **fx.attestation(kf_fps=bad)).status_code == 422, bad


# ---- 2. delete and retention ----

def seed_cache(jobs: Path, rows: dict[str, str]) -> Path:
    path = jobs / "cache.sqlite"
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("create table if not exists vlm_cache (key text primary key, model text, prompt_hash text, "
                "prompt text, parsed text, raw text, cost_usd real, duration_ms integer, created real)")
    with con:
        con.executemany("insert into vlm_cache (key, raw) values (?, ?)", list(rows.items()))
    con.close()
    return path


def cache_keys(path: Path) -> set[str]:
    con = sqlite3.connect(path)
    try:
        return {row[0] for row in con.execute("select key from vlm_cache")}
    finally:
        con.close()


def job_with_refs(jobs: Path, job_id: str, refs: list[str], stage: str = "done") -> Path:
    job = jobs / job_id
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"stage": stage, "error": None if stage == "done" else None}))
    fx.write_jsonl(job / "detect.jsonl", [{"kf": "kf_00000", "label": "liver", "raw_ref": refs[0]}])
    if len(refs) > 1:
        fx.write_jsonl(job / "decisions.jsonl", [{"kf": "kf_00000", "label": "liver", "rounds": [{"raw_ref": refs[1]}]}])
    if len(refs) > 2:
        (job / "narration.json").write_text(json.dumps({"segments": [], "raw_ref": refs[2]}))
    return job


def test_delete_removes_the_job_and_its_cache_entries_including_nested_refs(client, jobs):
    job_with_refs(jobs, "job-a", ["a1", "a2", "a3"])
    job_with_refs(jobs, "job-b", ["b1"])
    cache = seed_cache(jobs, {"a1": "RAW-A1-TEXT", "a2": "RAW-A2-TEXT", "a3": "RAW-A3-TEXT", "b1": "RAW-B1-TEXT"})
    res = client.delete("/api/jobs/job-a")
    assert res.status_code == 200 and res.json()["cache_entries_removed"] == 3
    assert not (jobs / "job-a").exists()
    assert client.get("/api/jobs/job-a").status_code == 404
    assert cache_keys(cache) == {"b1"}
    blob = b"".join(p.read_bytes() for p in jobs.glob("cache.sqlite*"))
    assert b"RAW-A1-TEXT" not in blob, "the deleted model response is still readable in the cache file"
    assert b"RAW-B1-TEXT" in blob


def test_delete_keeps_a_response_another_job_still_cites(client, jobs):
    job_with_refs(jobs, "job-a", ["shared", "a2"])
    job_with_refs(jobs, "job-b", ["shared"])
    cache = seed_cache(jobs, {"shared": "x", "a2": "y"})
    assert client.delete("/api/jobs/job-a").status_code == 200
    assert cache_keys(cache) == {"shared"}


def test_delete_clears_uncited_responses_from_aborted_attempts(client, jobs):
    job_with_refs(jobs, "job-a", ["a1"])
    cache = seed_cache(jobs, {"a1": "x", "orphan-from-a-crash": "y"})
    assert client.delete("/api/jobs/job-a").status_code == 200
    assert cache_keys(cache) == set()


def test_delete_leaves_uncited_responses_while_another_job_could_resume(client, jobs):
    job_with_refs(jobs, "job-a", ["a1"])
    job_with_refs(jobs, "job-b", ["b1"], stage="detect")
    cache = seed_cache(jobs, {"a1": "x", "b1": "y", "b2-not-yet-cited": "z"})
    assert client.delete("/api/jobs/job-a").status_code == 200
    assert cache_keys(cache) == {"b1", "b2-not-yet-cited"}


def test_delete_leaves_uncited_rows_in_a_cache_shared_with_other_folders(client, jobs, tmp_path, monkeypatch):
    job_with_refs(jobs, "job-a", ["a1"])
    shared = tmp_path / "shared"
    shared.mkdir()
    cache = seed_cache(shared, {"a1": "x", "someone-elses": "y"})
    monkeypatch.setenv("SURGPROOF_CACHE", str(cache))
    assert client.delete("/api/jobs/job-a").status_code == 200
    assert cache_keys(cache) == {"someone-elses"}


def test_real_pipeline_rows_cover_the_cache(client, jobs, tmp_path, monkeypatch):
    """On the real pipeline with the mock model: after delete, not one response from that job remains."""
    pytest.importorskip("surgproof.pipeline")
    pytest.importorskip("surgproof.export")
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    with fx.local_client(server.create_app(jobs)) as real:
        fx.sign_in(real, jobs)
        job_id = send_video(real, fx.make_video(tmp_path / "clip.mp4"), **fx.attestation()).json()["id"]
        wait_for(lambda: state_of(real, job_id)["state"] in ("done", "failed", "stalled"), 280)
        body = state_of(real, job_id)
        assert body["state"] == "done", body
        cache = jobs / "cache.sqlite"
        assert cache_keys(cache), "the run should have used the cache"
        assert real.delete(f"/api/jobs/{job_id}").status_code == 200
        assert cache_keys(cache) == set()


def test_deleting_a_running_job_kills_its_process_group(jobs, tmp_path):
    with open_client(jobs, HANG) as client:
        fx.sign_in(client, jobs)
        job_id = send_video(client, fx.make_video(tmp_path / "clip.mp4"), **fx.attestation()).json()["id"]
        child_file = jobs / job_id / "child.pid"
        wait_for(child_file.exists)
        child = int(child_file.read_text())
        assert pid_alive(child)
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        wait_for(lambda: not pid_alive(child), 20)
        assert not (jobs / job_id).exists()


def old_job(jobs: Path, job_id: str, days_old: float, refs=("old1",)) -> Path:
    job = job_with_refs(jobs, job_id, list(refs))
    created = datetime.now(timezone.utc) - timedelta(days=days_old)
    (job / "job.json").write_text(json.dumps({"id": job_id, "created": created.isoformat()}))
    return job


def test_retention_deletes_expired_jobs_and_their_cache_at_startup(jobs, monkeypatch):
    monkeypatch.setenv("SURGPROOF_RETENTION_DAYS", "30")
    jobs.mkdir()
    old_job(jobs, "job-old", 31)
    old_job(jobs, "job-new", 29, refs=("new1",))
    cache = seed_cache(jobs, {"old1": "x", "new1": "y"})
    with open_client(jobs) as client:
        fx.sign_in(client, jobs)
        wait_for(lambda: not (jobs / "job-old").exists(), 30)
        assert (jobs / "job-new").exists()
        assert cache_keys(cache) == {"new1"}
        expires = client.get("/api/jobs/job-new").json()["expires"]
        assert expires and datetime.fromisoformat(expires) > datetime.now(timezone.utc)


def test_retention_can_be_turned_off(jobs, monkeypatch):
    monkeypatch.setenv("SURGPROOF_RETENTION_DAYS", "0")
    jobs.mkdir()
    old_job(jobs, "job-old", 400)
    with open_client(jobs) as client:
        fx.sign_in(client, jobs)
        assert client.get("/api/jobs/job-old").json()["expires"] is None
    assert (jobs / "job-old").exists()


def test_retention_never_deletes_a_job_with_a_live_worker(jobs, monkeypatch):
    monkeypatch.setenv("SURGPROOF_RETENTION_DAYS", "30")
    jobs.mkdir()
    job = old_job(jobs, "job-old", 31)
    holder = hold_worker_lock(job)
    try:
        store = server.Store(jobs)
        runner = server.Runner("tests.web_fixture:pipeline_ok", 1)
        assert server.sweep_expired(store, runner) == []
        assert job.exists()
        runner.shutdown()
    finally:
        holder.kill()
        holder.wait()


# ---- 3. disk and upload streaming ----

def low_disk(monkeypatch, free: int):
    real = shutil.disk_usage
    monkeypatch.setattr(server.shutil, "disk_usage",
                        lambda path: type(real(path))(real(path).total, real(path).total - free, free))


def test_upload_is_refused_up_front_when_the_disk_is_short(client, jobs, tmp_path, monkeypatch):
    video = fx.make_video(tmp_path / "clip.mp4")
    low_disk(monkeypatch, 1024)
    res = send_video(client, video, **fx.attestation())
    assert res.status_code == 507
    assert "disk" in res.json()["detail"]
    assert server.Store(jobs).list_ids() == [] and not list(jobs.glob(".upload-*"))


def test_preflight_scales_with_the_upload_size(jobs):
    jobs.mkdir()
    free = shutil.disk_usage(jobs).free
    with pytest.raises(server.HTTPException) as err:
        server.check_disk_for_upload(jobs, free)  # the file alone would fit, the frames and masks would not
    assert err.value.status_code == 507


def test_huge_declared_upload_is_refused_without_reading_it(client, jobs):
    """A lying or honest Content-Length over the limit: answered from the header, before any body is read."""
    def body():
        raise AssertionError("the server read the body")
        yield b""  # pragma: no cover

    res = client.post("/api/jobs", content=body(), headers={
        "Content-Type": "multipart/form-data; boundary=x", "Content-Length": str(server.MAX_BYTES * 2)})
    assert res.status_code == 413
    assert not list(jobs.glob(".upload-*"))


def test_size_limit_stops_a_stream_early_with_no_declared_length(client, jobs, monkeypatch):
    """No Content-Length, so only counting can catch it. It must stop at the limit, not after the whole body."""
    limit = 256 * 1024
    monkeypatch.setattr(server, "MAX_BYTES", limit)
    sent = {"bytes": 0}
    head = (b'--x\r\nContent-Disposition: form-data; name="file"; filename="a.mp4"\r\n'
            b"Content-Type: video/mp4\r\n\r\n")
    total = 8 * 1024**2

    def body():
        yield head
        while sent["bytes"] < total:
            sent["bytes"] += 64 * 1024
            yield b"\0" * (64 * 1024)
        yield b"\r\n--x--\r\n"

    res = client.post("/api/jobs", content=body(), headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert res.status_code == 413
    assert not list(jobs.glob(".upload-*")), "staging must be cleaned"


def test_upload_is_not_spooled_to_a_temp_file_first(client, jobs, tmp_path, monkeypatch):
    """Bytes must land in staging as they arrive, not in a temp file that is copied afterwards."""
    video = fx.make_video(tmp_path / "clip.mp4")
    seen = []
    real = server.UploadStream.write

    def spy(self, chunk):
        real(self, chunk)
        seen.append(self.written)

    monkeypatch.setattr(server.UploadStream, "write", spy)
    assert send_video(client, video, **fx.attestation()).status_code == 201
    assert seen[-1] == video.stat().st_size


def test_disk_filling_during_an_upload_is_a_clean_507(client, jobs, tmp_path, monkeypatch):
    video = fx.make_video(tmp_path / "clip.mp4")
    monkeypatch.setattr(server.UploadStream, "DISK_CHECK_BYTES", 1)
    calls = {"n": 0}
    real = shutil.disk_usage

    def filling(path):
        calls["n"] += 1
        usage = real(path)
        free = usage.free if calls["n"] == 1 else 0  # fine at the preflight, gone once the upload is under way
        return type(usage)(usage.total, usage.total - free, free)

    monkeypatch.setattr(server.shutil, "disk_usage", filling)
    res = send_video(client, video, **fx.attestation())
    assert res.status_code == 507
    assert not list(jobs.glob(".upload-*"))


def test_enospc_while_writing_is_a_clean_507(client, jobs, tmp_path, monkeypatch):
    import errno

    video = fx.make_video(tmp_path / "clip.mp4")

    def full(self, data, start, end):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(server.UploadStream, "_part_data", full)
    # The parser holds a reference to the bound method, so patch before the stream is built: this request does.
    res = send_video(client, video, **fx.attestation())
    assert res.status_code == 507
    assert not list(jobs.glob(".upload-*"))


def test_garbage_multipart_is_refused_and_cleaned(client, jobs):
    res = client.post("/api/jobs", content=b"this is not multipart at all",
                      headers={"Content-Type": "multipart/form-data; boundary=x"})
    assert res.status_code == 422
    assert not list(jobs.glob(".upload-*"))
    assert client.post("/api/jobs", content=b"x", headers={"Content-Type": "application/json"}).status_code == 422


def test_two_files_in_one_upload_are_refused(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    data = video.read_bytes()
    res = client.post("/api/jobs", files=[("file", ("a.mp4", data)), ("video", ("b.mp4", data))], data=fx.attestation())
    assert res.status_code == 422
    assert server.Store(jobs).list_ids() == []


def test_staging_is_swept_at_startup(jobs):
    jobs.mkdir()
    (jobs / ".upload-crashed").mkdir()
    (jobs / ".upload-crashed" / "upload.mp4").write_bytes(b"partial")
    (jobs / ".deleting-half").mkdir()
    with open_client(jobs):
        assert not (jobs / ".upload-crashed").exists() and not (jobs / ".deleting-half").exists()


def test_second_server_on_the_port_refuses_before_touching_the_jobs_folder(monkeypatch, tmp_path):
    import socket

    with socket.socket() as taken:
        try:
            taken.bind(("127.0.0.1", 0))
        except OSError as exc:
            pytest.skip(f"cannot bind a local port here: {exc}")
        taken.listen()
        jobs = tmp_path / "jobs"
        jobs.mkdir()
        (jobs / ".upload-in-flight").mkdir()
        monkeypatch.setattr(sys, "argv", ["surgproof.server"])
        monkeypatch.setenv("PORT", str(taken.getsockname()[1]))
        monkeypatch.setenv("SURGPROOF_JOBS", str(jobs))
        with pytest.raises(SystemExit) as err:
            server.main()
        assert "already running" in str(err.value)
        assert (jobs / ".upload-in-flight").exists()


# ---- 3. workers: nothing dies silently ----

def upload_job(client, tmp_path) -> str:
    return send_video(client, fx.make_video(tmp_path / "clip.mp4"), **fx.attestation()).json()["id"]


def wait_state(client, job_id, *states):
    return wait_for(lambda: (lambda b: b if b["state"] in states else None)(state_of(client, job_id)))


def break_worker_start(monkeypatch):
    """Only the job's worker fails to start: ffprobe and everything else still run."""
    real = subprocess.Popen

    def popen(cmd, *a, **k):
        if "--run-job" in cmd:
            raise OSError("cannot start a process")
        return real(cmd, *a, **k)

    monkeypatch.setattr(server.subprocess, "Popen", popen)


def test_ffprobe_that_cannot_run_is_a_clear_refusal(client, tmp_path, monkeypatch):
    video = fx.make_video(tmp_path / "clip.mp4")

    def broken(*a, **k):
        raise OSError("resource exhausted")

    monkeypatch.setattr(server.subprocess, "run", broken)
    res = send_video(client, video, **fx.attestation())
    assert res.status_code == 422 and "ffprobe could not be run" in res.json()["detail"]


def test_runner_failure_surfaces_instead_of_reading_running(client, jobs, tmp_path, monkeypatch):
    """The attack: anything going wrong in _run itself. Before, the job kept saying queued or running."""
    break_worker_start(monkeypatch)
    job_id = upload_job(client, tmp_path)
    body = wait_state(client, job_id, "failed")
    assert "cannot start a process" in body["error"]


def test_failure_shows_even_when_the_disk_is_too_full_to_record_it(client, jobs, tmp_path, monkeypatch):
    video = fx.make_video(tmp_path / "clip.mp4")
    real_write = server.write_status

    def no_space_after_queueing(job_dir, **fields):
        if "error" in fields and fields["error"]:
            raise OSError(28, "No space left on device")
        return real_write(job_dir, **fields)

    break_worker_start(monkeypatch)
    monkeypatch.setattr(server, "write_status", no_space_after_queueing)
    job_id = send_video(client, video, **fx.attestation()).json()["id"]
    body = wait_state(client, job_id, "failed")
    assert "cannot start a process" in body["error"]


@pytest.mark.parametrize("target,expected", [
    ("tests.web_fixture:pipeline_done_with_nothing", "without finishing"),
    ("tests.web_fixture:pipeline_die_hard", "code -9"),
    ("tests.web_fixture:does_not_exist", "does_not_exist"),
    ("surgproof_no_such_module:run", "surgproof_no_such_module"),
])
def test_worker_endings_all_surface(jobs, tmp_path, target, expected):
    with open_client(jobs, target) as client:
        fx.sign_in(client, jobs)
        job_id = upload_job(client, tmp_path)
        body = wait_state(client, job_id, "failed")
        assert expected in body["error"]


def hold_worker_lock(job: Path) -> subprocess.Popen:
    """A separate process holding the job's worker lock, like an orphan from a previous server."""
    code = ("import fcntl,os,sys,time\n"
            f"f=open({str(job / server.WORKER_LOCK)!r},'a+');fcntl.flock(f,fcntl.LOCK_EX);"
            "f.seek(0);f.truncate();f.write(str(os.getpid()));f.flush();print('ready',flush=True);time.sleep(600)")
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    assert proc.stdout.readline().strip() == "ready"
    return proc


def test_dead_worker_is_detected_at_once_not_after_15_minutes(client, jobs):
    job = jobs / "job-x"
    job.mkdir()
    (job / "status.json").write_text(json.dumps({"stage": "detect", "error": None}))
    (job / server.WORKER_LOCK).write_text("99999")  # the server started it, and nothing holds the lock now
    body = client.get("/api/jobs/job-x").json()
    assert body["state"] == "stalled" and body["state_detail"] == "gone"


def test_live_orphan_is_seen_as_running_and_not_doubled(jobs, tmp_path):
    jobs.mkdir()
    job = jobs / "job-orphan"
    job.mkdir()
    fx.make_video(job / "upload.mp4")
    (job / "job.json").write_text(json.dumps({"id": "job-orphan", "upload": "upload.mp4", "options": {}}))
    (job / "status.json").write_text(json.dumps({"stage": "queued", "error": None}))
    holder = hold_worker_lock(job)
    try:
        with open_client(jobs) as client:
            fx.sign_in(client, jobs)
            time.sleep(0.5)
            assert not (job / "pipeline_called.json").exists(), "the job already has a worker: no second one"
            assert client.get("/api/jobs/job-orphan").json()["state"] == "queued"
    finally:
        holder.kill()
        holder.wait()


def test_only_one_worker_can_run_a_job(jobs, tmp_path):
    jobs.mkdir()
    job = jobs / "job-x"
    job.mkdir()
    holder = hold_worker_lock(job)
    try:
        out = subprocess.run([sys.executable, "-m", "surgproof.server", "--run-job", "tests.web_fixture:pipeline_ok",
                              "x.mp4", str(job), "{}"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert out.returncode != 0 and "another worker" in out.stderr
        assert not (job / "pipeline_called.json").exists()
    finally:
        holder.kill()
        holder.wait()


def test_hung_worker_reads_stalled_and_retry_kills_its_whole_group(jobs, tmp_path, monkeypatch):
    with open_client(jobs, HANG) as client:
        fx.sign_in(client, jobs)
        job_id = upload_job(client, tmp_path)
        job = jobs / job_id
        wait_for((job / "child.pid").exists)
        child = int((job / "child.pid").read_text())
        wait_for(lambda: state_of(client, job_id)["state"] == "running")
        assert client.post(f"/api/jobs/{job_id}/retry").status_code == 409, "a healthy job cannot be retried"

        monkeypatch.setattr(server, "STALL_SECONDS", 0)
        time.sleep(1.1)
        body = state_of(client, job_id)
        assert body["state"] == "stalled" and body["state_detail"] == "hung"
        (job / "child.pid").unlink()
        assert client.post(f"/api/jobs/{job_id}/retry").status_code == 200
        wait_for(lambda: not pid_alive(child), 30)
        monkeypatch.setattr(server, "STALL_SECONDS", 900)
        wait_for((job / "child.pid").exists)  # the new worker came up
        new_child = int((job / "child.pid").read_text())
        assert new_child != child
    wait_for(lambda: not pid_alive(new_child), 30)  # stopping the server takes workers with it


def test_worker_crash_does_not_leave_its_child_running(jobs, tmp_path):
    with open_client(jobs, "tests.web_fixture:pipeline_die_with_child") as client:
        fx.sign_in(client, jobs)
        job_id = upload_job(client, tmp_path)
        wait_state(client, job_id, "failed")
        child = int((jobs / job_id / "child.pid").read_text())
        wait_for(lambda: not pid_alive(child), 20)


def test_stopping_the_server_ends_a_worker_that_has_not_taken_its_lock_yet(jobs, tmp_path):
    """The stop must not depend on the lock file: it hung forever on a worker still starting up."""
    with open_client(jobs, HANG) as client:
        fx.sign_in(client, jobs)
        job_id = upload_job(client, tmp_path)
        wait_for(lambda: bool(client.app.state.runner._procs.get(job_id)), 30)
        proc = client.app.state.runner._procs[job_id]
        started = time.time()
    assert proc.poll() is not None
    assert time.time() - started < 60


def test_deleting_a_job_queued_behind_another_is_immediate(jobs, tmp_path):
    with open_client(jobs, HANG) as client:
        fx.sign_in(client, jobs)
        first = upload_job(client, tmp_path)
        second = upload_job(client, tmp_path)
        wait_for((jobs / first / "child.pid").exists)
        assert state_of(client, second)["state"] == "queued"
        started = time.time()
        assert client.delete(f"/api/jobs/{second}").status_code == 200
        assert time.time() - started < 10
        assert not (jobs / second).exists()
        assert state_of(client, first)["state"] == "running", "the other job is untouched"


def test_retry_is_refused_after_release(client, done):
    """The attack: damage a released job so it reads failed, then retry over the signed-off data."""
    clear_and_release(client)
    before = (done / "released_coco.json").read_bytes()
    (done / "annotated.mp4").unlink()
    assert client.get("/api/jobs/job-done").json()["state"] == "failed"
    res = client.post("/api/jobs/job-done/retry")
    assert res.status_code == 409 and "released" in res.json()["detail"]
    assert (done / "released_coco.json").read_bytes() == before


def test_retry_is_refused_after_a_person_started_reviewing(client, done):
    item = client.get("/api/jobs/job-done/final-pass").json()["items"][0]
    client.post("/api/jobs/job-done/final-pass", json={"action": "approve", "kf": item["kf"], "label": item["label"]})
    (done / "annotated.mp4").unlink()
    res = client.post("/api/jobs/job-done/retry")
    assert res.status_code == 409 and "reviewed" in res.json()["detail"]


# ---- 4. player data and health ----

def test_overlay_says_who_each_annotation_belongs_to(client, done):
    query = {"start": 0, "end": 12}
    frames = client.get("/api/jobs/job-done/overlay", params=query).json()["frames"]
    assert {o["origin"] for o in frames["0"]} == {"ai_keyframe"}
    assert {o["origin"] for o in frames["5"]} == {"propagated"}
    assert not any(o["derived_from_reviewed"] for o in frames["5"])
    unsure = next(o for o in frames["10"] if o["label"] == "gallbladder")
    assert unsure["unsure"] is True

    client.post("/api/jobs/job-done/final-pass", json={"action": "approve", "kf": "kf_00000", "label": "grasper"})
    frames = client.get("/api/jobs/job-done/overlay", params=query).json()["frames"]
    grasper_key = next(o for o in frames["0"] if o["label"] == "grasper")
    other_key = next(o for o in frames["0"] if o["label"] == "gallbladder")
    grasper_tracked = next(o for o in frames["5"] if o["label"] == "grasper")
    assert grasper_key["origin"] == "human"
    assert other_key["origin"] == "ai_keyframe"
    # Tracked from a reviewed keyframe is still not something a person saw.
    assert grasper_tracked["origin"] == "propagated" and grasper_tracked["derived_from_reviewed"] is True


def test_a_person_seeing_a_tracked_frame_marks_only_that_frame(client, done):
    fx.write_jsonl(done / "final_pass.jsonl", [{"kf": "prop_000005", "label": "grasper", "action": "approve", "reviewer": "x"}])
    frames = client.get("/api/jobs/job-done/overlay", params={"start": 5, "end": 7}).json()["frames"]
    assert next(o for o in frames["5"] if o["label"] == "grasper")["origin"] == "human"
    assert next(o for o in frames["6"] if o["label"] == "grasper")["origin"] == "propagated"


def test_approve_on_an_ai_absent_label_needs_no_mask_but_a_silent_present_does(tmp_path):
    job = tmp_path
    (job / "masks").mkdir()
    assert server.can_approve(job, "kf_00000", "blood", False) is True
    assert server.can_approve(job, "kf_00000", "blood", True) is False
    assert server.can_approve(job, "kf_00000", "blood", None) is False


def test_health_reports_disk_and_needs_no_session(anon, monkeypatch):
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    body = anon.get("/api/health").json()
    assert body["disk"]["free_bytes"] > 0 and body["disk"]["total_bytes"] >= body["disk"]["free_bytes"]
    assert body["disk"]["ok"] == (body["disk"]["free_bytes"] >= body["disk"]["min_free_bytes"])
    assert set(body["tools"]) == {"ffmpeg", "ffprobe"}
    assert body["model"]["provider"] == "mock" and "placeholders" in body["model"]["detail"]


def test_health_goes_bad_on_low_disk(anon, monkeypatch):
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    low_disk(monkeypatch, 1024)
    body = anon.get("/api/health").json()
    assert body["ok"] is False and body["disk"]["ok"] is False


def fake_claude(tmp_path: Path, reply: str, code: int = 0) -> Path:
    path = tmp_path / "fake_claude"
    path.write_text(f"#!/bin/sh\ncat <<'END'\n{reply}\nEND\nexit {code}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("reply,ok,word", [
    ('{"loggedIn": true, "email": "someone@example.org", "orgName": "Their Org"}', True, "logged in"),
    ('{"loggedIn": false}', False, "not logged in"),
    ("not json", False, "could not read"),
])
def test_health_reports_the_model_backend_without_leaking_the_account(tmp_path, monkeypatch, reply, ok, word):
    monkeypatch.delenv("SURGPROOF_VLM", raising=False)
    monkeypatch.setenv("SURGPROOF_CLAUDE_BIN", str(fake_claude(tmp_path, reply)))
    body = server.ModelStatus._check()
    assert body["ok"] is ok and word in body["detail"]
    assert "someone@" not in json.dumps(body) and "Their Org" not in json.dumps(body)


def test_health_when_the_model_cli_is_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("SURGPROOF_VLM", raising=False)
    monkeypatch.setenv("SURGPROOF_CLAUDE_BIN", str(tmp_path / "no_such_binary"))
    body = server.ModelStatus._check()
    assert body["ok"] is False and "not installed" in body["detail"]


def test_job_detail_counts_what_a_person_saw_from_the_released_file(client, done):
    clear_and_release(client)
    counts = client.get("/api/jobs/job-done").json()["released_counts"]
    released = json.loads((done / "released_coco.json").read_text())
    assert counts["n_annotations"] == len(released["annotations"])
    seen = [a for a in released["annotations"] if a["attributes"].get("human_reviewed")]
    assert counts["n_human_reviewed"] == len(seen)
    assert counts["n_propagated"] > 0 and counts["n_human_reviewed"] < counts["n_annotations"], "most tracked frames were never seen"


def test_ui_does_not_claim_verification():
    for path in (ROOT / "web" / "app").glob("*"):
        if path.is_file():
            assert not re.search(r"verified|any surgery", path.read_text(encoding="utf-8"), re.I), path.name


# ---- 5. dependencies ----

def locked_names(path: Path) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)==", path.read_text(encoding="utf-8"), re.M)}


def test_lock_files_pin_every_package_with_hashes():
    for name in ("requirements-e2e.lock", "requirements-e2e-dev.lock"):
        text = (ROOT / name).read_text(encoding="utf-8")
        pinned = [b for b in re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text) if re.match(r"^[A-Za-z0-9_.-]+==", b)]
        assert len(pinned) >= 5, name
        for block in pinned:
            assert "--hash=sha256:" in block, block[:60]


def test_dev_dependencies_are_separate():
    runtime = locked_names(ROOT / "requirements-e2e.lock")
    assert not {"pytest", "httpx"} & runtime
    assert {"pytest", "httpx"} <= locked_names(ROOT / "requirements-e2e-dev.lock")


def test_setup_installs_only_the_hash_checked_runtime_lock():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    setup = make[make.index("\nsetup:"):make.index("\nsetup-dev:")]
    assert "--require-hashes -r requirements-e2e.lock" in setup
    assert "requirements-e2e.txt" not in setup and "upgrade pip" not in setup


def test_runtime_lock_covers_what_the_app_imports():
    distribution = {"cv2": "opencv-python-headless", "PIL": "pillow", "python_multipart": "python-multipart"}
    locked = locked_names(ROOT / "requirements-e2e.lock")
    stdlib = set(sys.stdlib_module_names)
    for path in (ROOT / "surgproof").glob("*.py"):
        if path.name == "eval.py":  # the scorer is separate from the app, by design
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or top in ("surgproof", "google"):  # google is the optional Gemini provider
                    continue
                assert distribution.get(top, top).lower() in locked, f"{path.name} imports {top}, which is not locked"
