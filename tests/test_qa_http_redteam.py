"""Red team the server: the release gate, the final-pass lock, file serving, upload checks and the whole web flow."""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from tests.e2e_helpers.harness import HAVE_FFMPEG, ROOT, fake_env, read_json, read_jsonl, run_cli, synthetic_video

needed = ("server", "pipeline", "masks", "propagate", "render", "export")
ready = all((ROOT / "surgproof" / f"{m}.py").exists() for m in needed)
pytestmark = pytest.mark.skipif(not (ready and HAVE_FFMPEG), reason="other lanes not merged, or no ffmpeg")
if ready:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from surgproof import server as srv


@pytest.fixture(scope="module")
def finished_job(tmp_path_factory):
    """Three keyframes: gallbladder, liver, grasper accepted; cystic_duct stays amber on every frame."""
    d = tmp_path_factory.mktemp("job")
    video = synthetic_video(d / "v.mp4", seconds=3)
    job = d / "jobs" / "job1"
    p = run_cli(video, job, fake_env("amber_stuck"))
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    return job


@pytest.fixture
def world(tmp_path, finished_job):
    jobs = tmp_path / "jobs"
    shutil.copytree(finished_job, jobs / "job1")
    (tmp_path / "secret.txt").write_text("SECRET")
    (jobs / "secret2.txt").write_text("SECRET")
    with TestClient(srv.create_app(jobs_dir=jobs)) as c:
        yield {"c": c, "jobs": jobs, "job": jobs / "job1", "fp": "/api/jobs/job1/final-pass", "files": "/api/jobs/job1/files"}


def view(w):
    r = w["c"].get(w["fp"])
    assert r.status_code == 200, r.text
    return r.json()


def click(w, kf, label, action="approve", **kw):
    return w["c"].post(w["fp"], json={"action": action, "kf": kf, "label": label, "reviewer": "qa", **kw})


def release(w):
    return w["c"].post(w["fp"], json={"action": "release", "reviewer": "qa"})


def clear_queue(w, skip=None):
    for it in view(w)["items"]:
        if skip and (it["kf"], it["label"]) == skip:
            continue
        action = "reject" if it["kind"] == "amber" else "approve"
        assert click(w, it["kf"], it["label"], action).status_code == 200


@pytest.mark.parametrize("path", [
    "dataset_coco.json", "./dataset_coco.json", "frames/../dataset_coco.json", "masks/../dataset_coco.json",
    "DATASET_COCO.json", "Dataset_Coco.json", "dataset_coco.JSON", "dataset_coco.json/", "%64ataset_coco.json",
])
def test_draft_dataset_cannot_be_downloaded_before_release(world, path):
    """Includes case variants: the default macOS disk is case-insensitive, and the gate is a string compare."""
    r = world["c"].get(f"{world['files']}/{path}")
    assert r.status_code != 200, f"{path} served the locked dataset"


@pytest.mark.parametrize("path", [
    "../../secret.txt", "..%2f..%2fsecret.txt", "%2e%2e/%2e%2e/secret.txt", "../secret2.txt", "/etc/passwd", "..%2fsecret2.txt",
    "....//....//secret.txt",
])
def test_no_path_traversal(world, path):
    r = world["c"].get(f"{world['files']}/{path}")
    assert r.status_code != 200 or "SECRET" not in r.text, path


def test_symlink_in_job_dir_does_not_escape(world):
    os.symlink(world["jobs"].parent / "secret.txt", world["job"] / "link.txt")
    r = world["c"].get(f"{world['files']}/link.txt")
    assert r.status_code != 200 or "SECRET" not in r.text


@pytest.mark.parametrize("job_id", ["..", "%2e%2e", ".upload-x", "job1%2f..", "nope"])
def test_bad_job_ids_are_refused(world, job_id):
    assert world["c"].get(f"/api/jobs/{job_id}/final-pass").status_code >= 400


def test_release_locked_until_every_item_is_reviewed(world):
    v = view(world)
    assert v["locked"] and v["n_amber"] == 3 and v["n_audit"] >= 1
    r = release(world)
    assert r.status_code == 409
    assert not (world["job"] / "released_coco.json").exists()
    last = v["items"][-1]
    clear_queue(world, skip=(last["kf"], last["label"]))
    assert view(world)["locked"] is True
    assert release(world).status_code == 409
    assert not (world["job"] / "released_coco.json").exists()
    assert world["c"].get(f"{world['files']}/dataset_coco.json").status_code == 403
    assert click(world, last["kf"], last["label"], "reject" if last["kind"] == "amber" else "approve").status_code == 200
    assert view(world)["locked"] is False
    assert release(world).status_code == 200
    assert (world["job"] / "released_coco.json").exists()


def test_clicks_outside_the_queue_are_refused_and_unlock_nothing(world):
    assert click(world, "kf_99999", "liver").status_code == 422
    assert click(world, "kf_00000", "not_a_label").status_code == 422
    assert click(world, None, None).status_code == 422
    queue = {(i["kf"], i["label"]) for i in view(world)["items"]}
    outside = next((k, lb) for k in ("kf_00000", "kf_00001", "kf_00002") for lb in ("blood", "l_hook") if (k, lb) not in queue)
    assert click(world, *outside).status_code == 422
    assert not (world["job"] / "final_pass.jsonl").exists() or not read_jsonl(world["job"] / "final_pass.jsonl")
    assert view(world)["reviewed"] == 0


def test_hand_edited_final_pass_line_for_other_labels_changes_nothing(world):
    """Stray rows in final_pass.jsonl must not add data to the release."""
    clear_queue(world)
    with open(world["job"] / "final_pass.jsonl", "a") as f:
        f.write(json.dumps({"kf": "kf_00000", "label": "blood", "action": "correct", "polygon": [[0, 0], [50, 0], [50, 50]], "reviewer": "x"}) + "\n")
    assert release(world).status_code == 200
    coco = read_json(world["job"] / "released_coco.json")
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert not [a for a in coco["annotations"] if cats[a["category_id"]] == "blood"]


@pytest.mark.parametrize("polygon", [None, [], [[1, 1]], [[1, 1], [2, 2]], [[5, 5], [5, 5], [5, 5]]])
def test_correction_needs_a_real_outline(world, polygon):
    it = view(world)["items"][0]
    r = click(world, it["kf"], it["label"], "correct", polygon=polygon)
    assert r.status_code == 422, r.text


def test_nonsense_action_refused(world):
    it = view(world)["items"][0]
    assert click(world, it["kf"], it["label"], "lgtm").status_code == 422


def test_correction_outside_the_frame_is_not_stored_as_given(world):
    it = view(world)["items"][0]
    r = click(world, it["kf"], it["label"], "correct", polygon=[[-5000, -5000], [90000, 0], [90000, 90000]])
    if r.status_code == 200:
        clear_queue(world)
        assert click(world, it["kf"], it["label"], "correct", polygon=[[-5000, -5000], [90000, 0], [90000, 90000]]).status_code == 200
        assert release(world).status_code == 200
        meta = read_json(world["job"] / "meta.json")
        for a in read_json(world["job"] / "released_coco.json")["annotations"]:
            x, y, w, h = a["bbox"]
            assert x >= 0 and y >= 0 and x + w <= meta["width"] + 1 and y + h <= meta["height"] + 1, a


def test_audit_sample_is_stable_and_cannot_be_rerolled(world):
    a = [(i["kf"], i["label"]) for i in view(world)["items"]]
    b = [(i["kf"], i["label"]) for i in view(world)["items"]]
    assert a == b
    with TestClient(srv.create_app(jobs_dir=world["jobs"])) as c2:
        assert [(i["kf"], i["label"]) for i in c2.get(world["fp"]).json()["items"]] == a


def test_every_amber_is_in_the_queue(world):
    v = view(world)
    ambers = {(r["kf"], r["label"]) for r in read_jsonl(world["job"] / "decisions.jsonl") if r["status"] == "AMBER"}
    assert ambers and ambers <= {(i["kf"], i["label"]) for i in v["items"]}
    assert all(i["kind"] == "audit" for i in v["items"] if (i["kf"], i["label"]) not in ambers)


def test_amber_excluded_until_resolved_then_follows_the_human(world):
    draft = read_json(world["job"] / "dataset_coco.json")
    cats = {c["id"]: c["name"] for c in draft["categories"]}
    assert not [a for a in draft["annotations"] if cats[a["category_id"]] == "cystic_duct"], "amber in the draft dataset"
    items = view(world)["items"]
    meta = read_json(world["job"] / "meta.json")
    corrected = [i for i in items if i["kind"] == "amber"][0]
    poly = [[10, 10], [100, 10], [100, 80], [10, 80]]
    for i in items:
        if i is corrected:
            assert click(world, i["kf"], i["label"], "correct", polygon=poly).status_code == 200
        elif i["kind"] == "amber":
            assert click(world, i["kf"], i["label"], "reject").status_code == 200
        else:
            assert click(world, i["kf"], i["label"], "approve").status_code == 200
    assert release(world).status_code == 200
    rel = read_json(world["job"] / "released_coco.json")
    duct = [a for a in rel["annotations"] if cats[a["category_id"]] == "cystic_duct" and not a["attributes"].get("propagated")]
    assert len(duct) == 1, "exactly the one human-corrected amber"
    frame = next(k["frame_index"] for k in meta["keyframes"] if k["kf"] == corrected["kf"])
    assert {i["id"]: i for i in rel["images"]}[duct[0]["image_id"]]["frame_index"] == frame
    assert "human" in json.dumps(duct[0]["attributes"]).lower(), f"a human outline must say so: {duct[0]['attributes']}"


def test_human_reject_removes_the_label_and_what_flowed_from_it(world):
    items = view(world)["items"]
    audit = [i for i in items if i["kind"] == "audit"][0]
    for i in items:
        target = (i["kf"], i["label"]) == (audit["kf"], audit["label"])
        action = "reject" if target or i["kind"] == "amber" else "approve"
        assert click(world, i["kf"], i["label"], action).status_code == 200
    assert release(world).status_code == 200
    rel = read_json(world["job"] / "released_coco.json")
    cats = {c["id"]: c["name"] for c in rel["categories"]}
    meta = read_json(world["job"] / "meta.json")
    frame = next(k["frame_index"] for k in meta["keyframes"] if k["kf"] == audit["kf"])
    imgs = {i["id"]: i for i in rel["images"]}
    assert not [a for a in rel["annotations"]
                if cats[a["category_id"]] == audit["label"] and imgs[a["image_id"]]["frame_index"] == frame]
    next_kf = frame + int(round(meta["fps"] / meta["kf_fps"]))
    flowed = [a for a in rel["annotations"] if cats[a["category_id"]] == audit["label"]
              and a["attributes"].get("propagated") and frame < imgs[a["image_id"]]["frame_index"] < next_kf
              and a["attributes"].get("keyframe") == audit["kf"]]
    assert not flowed, "propagated annotations built only from the rejected keyframe survived"


def test_no_changes_after_release(world):
    clear_queue(world)
    assert release(world).status_code == 200
    before = (world["job"] / "released_coco.json").read_bytes()
    it = view(world)["items"][0]
    assert click(world, it["kf"], it["label"], "reject").status_code == 409
    assert release(world).status_code == 409
    assert (world["job"] / "released_coco.json").read_bytes() == before


def test_release_is_signed_and_counts_are_true(world):
    clear_queue(world)
    assert release(world).status_code == 200
    info = read_json(world["job"] / "released_coco.json")["info"]
    v = view(world)
    assert info["released"] is True and info["reviewer"] == "qa"
    assert info["final_pass"]["n_amber_resolved"] == v["n_amber"]
    assert info["final_pass"]["n_audit"] == v["n_audit"]
    assert world["c"].get(f"{world['files']}/dataset_coco.json").status_code == 200


def test_anonymous_release_is_not_a_sign_off(world):
    """The release is the one human accountability step. A nameless one is not a sign-off."""
    clear_queue(world)
    r = world["c"].post(world["fp"], json={"action": "release"})
    if r.status_code == 200:
        assert read_json(world["job"] / "released_coco.json")["info"]["reviewer"] not in (None, "", "unnamed reviewer"), "released with no named reviewer"


def test_claimed_done_without_outputs_is_failed(world):
    (world["job"] / "annotated.mp4").unlink()
    j = world["c"].get("/api/jobs/job1").json()
    assert j["state"] == "failed" and "annotated.mp4" in j["error"]


def test_final_pass_refused_on_unfinished_job(world):
    st = read_json(world["job"] / "status.json")
    st["stage"] = "masks"
    (world["job"] / "status.json").write_text(json.dumps(st))
    assert world["c"].get(world["fp"]).status_code >= 400
    assert release(world).status_code >= 400


def test_overlay_only_shows_what_the_tracks_say(world):
    rows = read_jsonl(world["job"] / "tracks.jsonl")
    assert rows
    r = world["c"].get("/api/jobs/job1/overlay?start=0&end=30")
    assert r.status_code == 200, r.text
    frames = r.json()["frames"]
    assert frames, "overlay empty although tracks exist"
    want = {(t["frame_index"], t["label"]) for t in rows if t.get("box") and t["frame_index"] < 30}
    got = {(int(f), o["label"]) for f, objs in frames.items() for o in objs}
    assert got == want


def test_scorer_output_is_not_part_of_the_job_api(world):
    (world["job"] / "eval_report.json").write_text('{"presence_final": {"accuracy": 1}}')
    assert "eval_report" not in json.dumps(world["c"].get("/api/jobs/job1").json())


# ---------- upload ----------

def _post(c, name, data):
    return c.post("/api/jobs", files={"file": (name, data, "application/octet-stream")})


def test_upload_rejects_bad_files(world, tmp_path):
    c = world["c"]
    assert _post(c, "notes.txt", b"hello").status_code == 422
    assert _post(c, "fake.mp4", b"this is not a video").status_code == 422
    assert _post(c, "empty.mp4", b"").status_code == 422
    assert _post(c, "../../evil.mp4", b"x").status_code == 422
    assert not list(world["jobs"].glob(".upload-*")), "failed uploads left folders behind"
    assert not (world["jobs"].parent / "evil.mp4").exists()


def test_upload_refuses_video_over_five_minutes(world, tmp_path):
    import subprocess

    long = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=black:s=64x36:r=1:d=301",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(long)], check=True)
    r = _post(world["c"], "long.mp4", long.read_bytes())
    assert r.status_code == 422 and "limit" in r.text


def test_whole_web_flow_with_mock_model(tmp_path, monkeypatch):
    """Acceptance steps 2 and 3 over HTTP: upload, live progress, then Watch, Final pass, Download."""
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    video = synthetic_video(tmp_path / "v.mp4", seconds=3)
    jobs = tmp_path / "jobs"
    with TestClient(srv.create_app(jobs_dir=jobs)) as c:
        for page in ("/", "/app.js", "/style.css"):
            assert c.get(page).status_code == 200, page
        r = _post(c, "my surgery.mp4", video.read_bytes())
        assert r.status_code == 201, r.text
        jid = r.json()["id"]
        stages, state = set(), None
        deadline = time.time() + 240
        while time.time() < deadline:
            j = c.get(f"/api/jobs/{jid}").json()
            state = j["state"]
            stages.add(j.get("stage"))
            if state in ("done", "failed", "stalled"):
                break
            time.sleep(0.2)
        assert state == "done", j
        assert j["stages"] and all(s["state"] == "done" and s["evidence"] for s in j["stages"]), j["stages"]
        assert len(stages) >= 2, f"progress was not live: {stages}"
        assert c.get(f"/api/jobs/{jid}/video").status_code == 200
        assert c.get(f"/api/jobs/{jid}/watch").status_code == 200
        assert c.get(f"/api/jobs/{jid}/overlay?start=0&end=30").status_code == 200
        for f in ("annotated.mp4", "provenance.jsonl"):
            assert c.get(f"/api/jobs/{jid}/files/{f}?download=true").status_code == 200, f
        assert c.get(f"/api/jobs/{jid}/files/dataset_coco.json").status_code == 403
        fp = f"/api/jobs/{jid}/final-pass"
        for it in c.get(fp).json()["items"]:
            assert c.post(fp, json={"action": "approve", "kf": it["kf"], "label": it["label"], "reviewer": "qa"}).status_code == 200
        assert c.post(fp, json={"action": "release", "reviewer": "qa"}).status_code == 200
        assert c.get(f"/api/jobs/{jid}/files/released_coco.json").status_code == 200
        assert c.get(f"/api/jobs/{jid}/files/dataset_coco.json").status_code == 200
