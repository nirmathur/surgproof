"""surgproof/server.py against a real job layout: upload validation, job states, the release gate, final pass."""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("cv2")
if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    pytest.skip("ffmpeg is required", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402

from surgproof import server  # noqa: E402
from tests import web_fixture as fx  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def jobs(tmp_path):
    return tmp_path / "jobs"


def client_for(jobs, target="tests.web_fixture:pipeline_ok"):
    return TestClient(server.create_app(jobs, pipeline_target=target))


@pytest.fixture
def client(jobs):
    with client_for(jobs) as c:
        yield c


def wait_for(predicate, seconds=180):
    end = time.time() + seconds
    while time.time() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise AssertionError("timed out")


def upload(client, path: Path, name=None):
    with path.open("rb") as fh:
        return client.post("/api/jobs", files={"file": (name or path.name, fh, "video/mp4")})


# ---- upload ----

def test_upload_starts_the_pipeline_with_the_stored_file(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    res = upload(client, video)
    assert res.status_code == 201, res.text
    body = res.json()
    job = jobs / body["id"]
    assert body["filename"] == "clip.mp4"
    assert (job / "upload.mp4").read_bytes() == video.read_bytes()
    called = wait_for(lambda: server.read_json(job / "pipeline_called.json"))
    assert Path(called["video"]).resolve() == (job / "upload.mp4").resolve()
    assert called["options"] == {}


def test_upload_forwards_kf_fps(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    with video.open("rb") as fh:
        res = client.post("/api/jobs", files={"file": ("clip.mp4", fh)}, data={"kf_fps": "2"})
    assert res.status_code == 201
    called = wait_for(lambda: server.read_json(jobs / res.json()["id"] / "pipeline_called.json"))
    assert called["options"] == {"kf_fps": 2.0}


@pytest.mark.parametrize("name", ["clip.avi", "clip", "notes.txt"])
def test_upload_refuses_other_file_types(client, tmp_path, name):
    path = tmp_path / "f"
    path.write_bytes(b"data")
    res = upload(client, path, name)
    assert res.status_code == 422
    assert "mp4" in res.json()["detail"]


def test_upload_refuses_a_non_video_with_a_video_extension(client, jobs, tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"this is not a video" * 100)
    res = upload(client, path)
    assert res.status_code == 422
    assert server.Store(jobs).list_ids() == []
    assert not [p for p in jobs.iterdir() if p.name.startswith(".upload")]


def test_upload_refuses_an_empty_file(client, tmp_path):
    path = tmp_path / "empty.mp4"
    path.write_bytes(b"")
    assert upload(client, path).status_code == 422


def test_upload_refuses_over_five_minutes(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "long.mp4", seconds=301, fps=1, width=32, height=32)
    res = upload(client, video)
    assert res.status_code == 422
    assert "limit is 5 minutes" in res.json()["detail"]
    assert server.Store(jobs).list_ids() == []


def test_uploaded_filename_cannot_escape_the_job_folder(client, jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    res = upload(client, video, "../../evil.mp4")
    assert res.status_code == 201
    assert res.json()["filename"] == "evil.mp4"
    assert not (jobs.parent / "evil.mp4").exists()


def test_cross_origin_post_is_refused(client, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    with video.open("rb") as fh:
        res = client.post("/api/jobs", files={"file": ("clip.mp4", fh)}, headers={"Origin": "https://evil.example"})
    assert res.status_code == 403


# ---- job states: judged from what is on disk ----

def _status_job(jobs, **status):
    job = jobs / "job-x"
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps(status), encoding="utf-8")
    return job


def test_done_status_without_outputs_is_reported_as_failed(client, jobs):
    _status_job(jobs, stage="done", progress=1.0, error=None)
    body = client.get("/api/jobs/job-x").json()
    assert body["state"] == "failed"
    assert "annotated.mp4" in body["error"]


def test_pipeline_exception_shows_as_failed_and_retry_works(jobs, tmp_path):
    video = fx.make_video(tmp_path / "clip.mp4")
    with client_for(jobs, "tests.web_fixture:pipeline_boom") as client:
        job_id = upload(client, video).json()["id"]
        body = wait_for(lambda: (lambda b: b if b["state"] == "failed" else None)(client.get(f"/api/jobs/{job_id}").json()))
        assert "synthetic failure" in body["error"]
        assert client.post(f"/api/jobs/{job_id}/retry").status_code == 200
        wait_for(lambda: client.get(f"/api/jobs/{job_id}").json()["state"] == "failed")


def test_silent_job_is_stalled_by_artifact_age(client, jobs):
    old = "2020-01-01T00:00:00+00:00"
    job = _status_job(jobs, stage="detect", progress=0.2, updated=old, error=None)
    assert client.get("/api/jobs/job-x").json()["state"] == "running"  # folder was written just now
    long_ago = time.time() - server.STALL_SECONDS - 60
    os.utime(job / "status.json", (long_ago, long_ago))
    os.utime(job, (long_ago, long_ago))
    assert client.get("/api/jobs/job-x").json()["state"] == "stalled"


def test_stage_list_marks_order_and_counts_come_from_files(client, jobs):
    job = _status_job(jobs, stage="verify", progress=0.4, error=None)
    (job / "frames").mkdir()
    (job / "frames" / "kf_00000.jpg").write_bytes(b"x")
    fx.write_jsonl(job / "detect.jsonl", [{"kf": "kf_00000", "label": "liver", "present": True}])
    stages = {s["key"]: s for s in client.get("/api/jobs/job-x").json()["stages"]}
    assert [stages[k]["state"] for k in ("ingest", "detect", "verify", "amber")] == ["done", "done", "active", "pending"]
    assert stages["ingest"]["evidence"] == "1 keyframes"
    assert stages["detect"]["evidence"] == "1 keyframes checked"
    assert stages["amber"]["evidence"] is None


def test_preview_shows_the_newest_detected_keyframe(client, jobs):
    job = _status_job(jobs, stage="detect", error=None)
    poly = [[1, 1], [5, 1], [5, 5]]
    fx.write_jsonl(job / "detect.jsonl", [
        {"kf": "kf_00000", "label": "liver", "present": True, "conf": 0.9, "polygon": poly},
        {"kf": "kf_00001", "label": "liver", "present": True, "conf": 0.7, "polygon": poly},
        {"kf": "kf_00001", "label": "blood", "present": False, "conf": 0.7, "polygon": None},
    ])
    body = client.get("/api/jobs/job-x/preview").json()
    assert body["kf"] == "kf_00001"
    assert [o["label"] for o in body["objects"]] == ["liver"]
    assert body["objects"][0]["polygon"] == poly


def test_queued_job_is_picked_up_after_a_restart(jobs, tmp_path):
    job = jobs / "job-q"
    job.mkdir(parents=True)
    fx.make_video(job / "upload.mp4")
    (job / "job.json").write_text(json.dumps({"id": "job-q", "upload": "upload.mp4", "options": {}}), encoding="utf-8")
    (job / "status.json").write_text(json.dumps({"stage": "queued", "error": None}), encoding="utf-8")
    with client_for(jobs):
        wait_for(lambda: (job / "pipeline_called.json").exists())


# ---- paths ----

@pytest.mark.parametrize("path", ["../../secret.txt", "..%2F..%2Fsecret.txt", "frames/../../../secret.txt"])
def test_file_route_cannot_escape_the_job(client, jobs, path):
    _status_job(jobs, stage="detect", error=None)
    (jobs.parent / "secret.txt").write_text("secret", encoding="utf-8")
    res = client.get(f"/api/jobs/job-x/files/{path}")
    assert res.status_code == 404
    assert "secret" not in res.text


def test_unknown_or_hidden_job_ids_are_404(client, jobs):
    jobs.mkdir(exist_ok=True)
    (jobs / ".upload-x").mkdir()
    assert client.get("/api/jobs/nope").status_code == 404
    assert client.get("/api/jobs/.upload-x").status_code == 404


# ---- finished job: watch data, final pass, release (uses the real surgproof.export) ----

@pytest.fixture
def done(jobs):
    pytest.importorskip("surgproof.export")
    return fx.finished_job(jobs)


def items(client):
    return client.get("/api/jobs/job-done/final-pass").json()["items"]


def post(client, **payload):
    return client.post("/api/jobs/job-done/final-pass", json=payload)


def clear_queue(client):
    for item in items(client):
        assert post(client, action="approve", kf=item["kf"], label=item["label"]).status_code == 200


def test_video_supports_range_requests_for_seeking(client, done):
    res = client.get("/api/jobs/job-done/video", headers={"Range": "bytes=0-99"})
    assert res.status_code == 206
    assert len(res.content) == 100
    assert res.headers["content-type"] == "video/mp4"


def test_watch_bundle_comes_from_the_job_files(client, done):
    body = client.get("/api/jobs/job-done/watch").json()
    assert (body["fps"], body["width"], body["height"], body["n_frames"]) == (fx.FPS, fx.W, fx.H, fx.FPS * fx.SECONDS)
    assert [p["phase"] for p in body["phases"]] == ["preparation", "preparation", "clipping_cutting"]
    assert [p["t"] for p in body["phases"]] == [0.0, 1.0, 2.0]
    assert body["narration"][0]["evidence"] == ["kf_00000", "kf_00001"]


def test_watch_refuses_until_finished(client, jobs):
    _status_job(jobs, stage="detect", error=None)
    assert client.get("/api/jobs/job-x/watch").status_code == 409
    assert client.get("/api/jobs/job-x/final-pass").status_code == 409


def test_overlay_returns_the_written_box_and_mask_outline(client, done):
    res = client.get("/api/jobs/job-done/overlay", params={"start": 0, "end": 12}).json()
    keyframe = {o["label"]: o for o in res["frames"]["0"]}
    assert keyframe["grasper"]["box"] == [20, 30, 90, 100]
    assert keyframe["grasper"]["propagated"] is False
    polygon = keyframe["grasper"]["polygons"][0]
    assert min(p[0] for p in polygon) == 20 and max(p[0] for p in polygon) == 89
    tracked = {o["label"]: o for o in res["frames"]["5"]}
    assert tracked["grasper"]["propagated"] is True and tracked["grasper"]["polygons"]
    assert {o["status"] for o in res["frames"]["10"] if o["label"] == "gallbladder"} == {"AMBER"}
    assert "12" not in res["frames"]


def test_overlay_limits_range(client, done):
    assert client.get("/api/jobs/job-done/overlay", params={"start": 0, "end": 601}).status_code == 422
    assert client.get("/api/jobs/job-done/overlay", params={"start": 5, "end": 5}).status_code == 422


def test_mask_outline_scales_to_video_pixels(tmp_path):
    import cv2
    import numpy as np

    mask = np.zeros((50, 100), np.uint8)
    mask[10:20, 10:30] = 255
    path = tmp_path / "m.png"
    cv2.imwrite(str(path), mask)
    xs = [p[0] for p in server.mask_polygons(path, 200, 100)[0]]
    assert min(xs) == 20 and max(xs) == 58


def test_queue_has_every_amber_and_an_audit_sample(client, done):
    state = client.get("/api/jobs/job-done/final-pass").json()
    kinds = [i["kind"] for i in state["items"]]
    amber = [i for i in state["items"] if i["kind"] == "amber"]
    assert [(i["kf"], i["label"]) for i in amber] == [("kf_00001", "gallbladder")]
    assert kinds.count("audit") >= 1
    assert state["total"] == len(kinds) and state["pending"] == len(kinds) and state["locked"] is True
    assert amber[0]["reason"] == "the two passes disagreed"
    assert amber[0]["verify_present"] is False and amber[0]["ai_present"] is True
    assert amber[0]["outline_source"] == "mask" and amber[0]["can_approve"] is True
    assert state["n_audit_pool"] == 5


def test_queue_is_stable_between_requests(client, done):
    assert [i["id"] for i in items(client)] == [i["id"] for i in items(client)]


def test_draft_dataset_and_release_are_locked_until_the_human_finishes(client, done):
    assert client.get("/api/jobs/job-done/files/dataset_coco.json").status_code == 403
    assert client.get("/api/jobs/job-done/files/released_coco.json").status_code == 404
    res = post(client, action="release")
    assert res.status_code == 409
    assert "locked" in res.json()["detail"]
    assert not (done / "released_coco.json").exists()

    queue = items(client)
    for item in queue[:-1]:
        assert post(client, action="approve", kf=item["kf"], label=item["label"]).status_code == 200
    assert post(client, action="release").status_code == 409

    last = queue[-1]
    state = post(client, action="approve", kf=last["kf"], label=last["label"]).json()
    assert state["pending"] == 0 and state["locked"] is False
    state = post(client, action="release", reviewer="R. Reviewer").json()
    assert state["released"] is True
    released = client.get("/api/jobs/job-done/files/released_coco.json").json()
    assert released["info"]["released"] is True and released["info"]["reviewer"] == "R. Reviewer"
    assert client.get("/api/jobs/job-done/files/dataset_coco.json").status_code == 200


def test_no_changes_after_release(client, done):
    clear_queue(client)
    assert post(client, action="release").status_code == 200
    item = items(client)[0]
    assert post(client, action="reject", kf=item["kf"], label=item["label"]).status_code == 409
    assert post(client, action="release").status_code == 409


def test_ambers_are_absent_from_the_draft_and_join_only_when_approved(client, done):
    draft = json.loads((done / "dataset_coco.json").read_text())
    image = next(i for i in draft["images"] if i["frame_index"] == fx.FPS)
    gallbladder = next(c["id"] for c in draft["categories"] if c["name"] == "gallbladder")
    in_draft = [a for a in draft["annotations"] if a["image_id"] == image["id"] and a["category_id"] == gallbladder]
    assert in_draft == []
    clear_queue(client)
    post(client, action="release")
    released = client.get("/api/jobs/job-done/files/released_coco.json").json()
    joined = [a for a in released["annotations"] if a["image_id"] == image["id"] and a["category_id"] == gallbladder]
    assert len(joined) == 1 and joined[0]["attributes"]["human_reviewed"] is True


def test_rejecting_an_amber_keeps_it_out_and_stops_drawing_it(client, done):
    assert post(client, action="reject", kf="kf_00001", label="gallbladder").status_code == 200
    frame = client.get("/api/jobs/job-done/overlay", params={"start": 10, "end": 11}).json()["frames"]["10"]
    assert [o["label"] for o in frame] == ["grasper"]
    item = next(i for i in items(client) if i["kf"] == "kf_00001" and i["label"] == "gallbladder")
    assert item["final_present"] is False and item["verdict"]["action"] == "reject"


def test_human_correction_is_recorded_and_used(client, done):
    polygon = [[10, 10], [60, 10], [60, 60], [10, 60]]
    res = post(client, action="correct", kf="kf_00001", label="gallbladder", polygon=polygon, reviewer="R")
    assert res.status_code == 200
    rows = [json.loads(line) for line in (done / "final_pass.jsonl").read_text().splitlines()]
    assert rows[-1]["polygon"] == polygon and rows[-1]["action"] == "correct" and rows[-1]["reviewer"] == "R"
    drawn = {o["label"]: o for o in client.get("/api/jobs/job-done/overlay", params={"start": 10, "end": 11}).json()["frames"]["10"]}
    assert drawn["gallbladder"]["box"] == [10, 10, 60, 60]
    assert drawn["gallbladder"]["polygons"] == [polygon]
    clear = [i for i in items(client) if not i["verdict"]]
    for item in clear:
        post(client, action="approve", kf=item["kf"], label=item["label"])
    post(client, action="release")
    released = client.get("/api/jobs/job-done/files/released_coco.json").json()
    corrected = [a for a in released["annotations"] if a["attributes"].get("source") == "human_correction"
                 or a["attributes"].get("human_corrected")]
    assert corrected, "the human's polygon must be in the released dataset"


def test_changing_your_mind_keeps_the_earlier_click_on_record(client, done):
    post(client, action="reject", kf="kf_00001", label="gallbladder")
    post(client, action="approve", kf="kf_00001", label="gallbladder")
    rows = (done / "final_pass.jsonl").read_text().splitlines()
    assert len(rows) == 2
    item = next(i for i in items(client) if i["kf"] == "kf_00001" and i["label"] == "gallbladder")
    assert item["verdict"]["action"] == "approve"


def test_audit_disagreements_are_counted(client, done):
    audit = next(i for i in items(client) if i["kind"] == "audit")
    state = post(client, action="reject", kf=audit["kf"], label=audit["label"]).json()
    assert state["audit_reviewed"] == 1 and state["audit_disagreements"] == 1


@pytest.mark.parametrize("payload", [
    {"action": "approve", "kf": "kf_00099", "label": "grasper"},
    {"action": "correct", "kf": "kf_00001", "label": "gallbladder"},
    {"action": "correct", "kf": "kf_00001", "label": "gallbladder", "polygon": [[1, 1], [2, 2]]},
    {"action": "dance", "kf": "kf_00001", "label": "gallbladder"},
])
def test_invalid_clicks_are_refused_and_not_recorded(client, done, payload):
    assert post(client, **payload).status_code == 422
    assert not (done / "final_pass.jsonl").exists()


def test_approving_needs_a_real_mask(client, done):
    (done / "masks" / "kf_00001_gallbladder.png").unlink()
    res = post(client, action="approve", kf="kf_00001", label="gallbladder")
    assert res.status_code == 422
    assert "no AI mask" in res.json()["detail"]
    item = next(i for i in items(client) if i["kf"] == "kf_00001" and i["label"] == "gallbladder")
    assert item["can_approve"] is False


def test_job_detail_reports_outputs_and_final_pass(client, done):
    body = client.get("/api/jobs/job-done").json()
    assert body["state"] == "done"
    assert all(s["state"] == "done" for s in body["stages"])
    assert body["outputs"]["annotated.mp4"]["bytes"] > 0
    assert "released_coco.json" not in body["outputs"]
    assert body["final_pass"]["pending"] == body["final_pass"]["total"] > 0
    assert body["decisions"] == {"accept": 5, "reject": 0, "AMBER": 1}


def test_download_flag_sets_attachment(client, done):
    res = client.get("/api/jobs/job-done/files/provenance.jsonl", params={"download": 1})
    assert res.status_code == 200
    assert res.headers["content-disposition"].startswith("attachment")


# ---- static guards ----

def test_app_pages_are_served(client):
    assert "SurgProof" in client.get("/").text
    for name in ("app.js", "lib.js", "upload.js", "job.js", "watch.js", "finalpass.js", "download.js", "style.css"):
        assert client.get(f"/{name}").status_code == 200, name


def test_no_external_requests_from_the_app():
    for path in (p for p in (ROOT / "web" / "app").iterdir() if p.is_file()):
        assert not re.search(r"https?://", path.read_text(encoding="utf-8")), path.name


def test_makefile_has_the_e2e_targets():
    text = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert re.search(r"^setup:", text, re.M) and re.search(r"^serve:", text, re.M)
    assert "surgproof.server" in text
    assert "PORT ?= 8600" in text
