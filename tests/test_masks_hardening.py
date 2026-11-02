"""Regression tests for the hardening round. Each reproduces an audit attack against the masks lane."""
import hashlib, json, os, subprocess, zipfile

import cv2
import numpy as np
import pytest

from surgproof import export as X, masks as M, propagate as P, render as R

W, H, FPS = 320, 240, 25


def disc(cx=160, cy=120, r=40):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def write_source(job, n):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration={n / FPS}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(job / "source.mp4")], check=True)


def build(tmp_path, kf_rows, tracks=(), n_kf=2, gap=25, video=True, extras=True):
    """kf_rows: (kf_number, label, status, final_present, has_mask)."""
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True, exist_ok=True)
    n_frames = n_kf * gap
    keyframes = [{"kf": f"kf_{i:05d}", "frame_index": i * gap, "t": i * gap / FPS} for i in range(n_kf)]
    for k in keyframes:
        cv2.imwrite(str(job / "frames" / f"{k['kf']}.jpg"), np.full((H, W, 3), 40, np.uint8))
    (job / "meta.json").write_text(json.dumps({"fps": float(FPS), "width": W, "height": H, "n_frames": n_frames, "kf_fps": 1,
                                               "keyframes": keyframes, "source_video": "source.mp4", "vlm_provider": "claude_cli"}))
    if video:
        write_source(job, n_frames)
    det, dec, ver, mrows, trows = [], [], [], [], []
    for i, label, status, present, has_mask in kf_rows:
        kf = f"kf_{i:05d}"
        det.append({"kf": kf, "label": label, "present": bool(present), "conf": 0.8, "polygon": [[1, 1], [50, 1], [50, 50]] if present else None,
                    "model": "claude-sonnet-5-5", "prompt_hash": "ph", "raw_ref": f"r{i}"})
        ver.append({"kf": kf, "label": label, "present": bool(present), "box": [1, 1, 50, 50], "model": "claude-haiku-5-5", "prompt_hash": "vh"})
        dec.append({"kf": kf, "label": label, "status": status, "final_present": present, "rounds": []})
        if has_mask:
            M.write_mask(M.mask_path(job, kf, label), disc())
            mrows.append({"kf": kf, "label": label, "status": status, "seed": "detect_polygon", "fallback": None, "area": int(disc().sum()),
                          "box": M.mask_box(disc()), "path": f"masks/{kf}_{label}.png", "model": "claude-sonnet-5-5", "prompt_hash": "ph"})
            trows.append({"frame_index": i * gap, "label": label, "box": M.mask_box(disc()), "box_raw": M.mask_box(disc()), "propagated": False,
                          "status": status, "source_kfs": [kf], "keyframe": kf, "t": i * gap / FPS, "flow_check": "keyframe", "fb_overlap": None, "shot": 0})
    trows += list(tracks)
    for name, rows in [("detect", det), ("verify", ver), ("decisions", dec), ("masks", mrows), ("tracks", trows)]:
        (job / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for t in tracks:
        if t["propagated"]:
            M.write_mask(M.track_path(job, t["frame_index"], t["label"]), disc())
    if extras:
        (job / "phase.jsonl").write_text("".join(json.dumps({"kf": k["kf"], "phase": "Preparation", "conf": 0.7}) + "\n" for k in keyframes))
        (job / "narration.json").write_text(json.dumps({"segments": [{"t0": 0, "t1": 1, "phase": "Preparation", "text": "A grasper is visible.", "evidence": ["kf_00000"]}]}))
    return job


def track(fi, label, sources, status="accept", check="ok", overlap=0.9, shot=0):
    return {"frame_index": fi, "label": label, "box": [120, 80, 200, 160], "box_raw": [120, 80, 200, 160], "propagated": True, "status": status,
            "source_kfs": sources, "keyframe": None, "t": fi / FPS, "flow_check": check, "fb_overlap": overlap, "shot": shot}


TWO = ["kf_00000", "kf_00001"]
BASE = [(0, "grasper", "accept", True, True), (0, "liver", "AMBER", None, True), (0, "blood", "reject", False, False),
        (1, "grasper", "accept", True, True), (1, "liver", "accept", True, True), (1, "blood", "reject", False, False)]
TRACKS = [track(10, "grasper", TWO), track(11, "grasper", TWO), track(12, "liver", TWO, status="AMBER")]


@pytest.fixture
def job(tmp_path):
    j = build(tmp_path, BASE, TRACKS)
    X.run(j)
    return j


def coco(job, name="dataset_coco.json"):
    return json.loads((job / name).read_text())


def anns(c, label):
    return [a for a in c["annotations"] if a["category_id"] == X.CATEGORY_ID[label]]


def clear(job, overrides=None, default=None):
    """Resolve every open item, like a reviewer agreeing with the AI, except where overridden."""
    overrides = overrides or {}
    rows = []
    for i in X.final_pass_state(job)["remaining"]:
        rows.append({"kf": i["kf"], "label": i["label"], "action": default or "approve", "reviewer": "r", **overrides.get((i["kf"], i["label"]), {})})
    if rows:
        X.record_final_pass(job, rows)


def many(n, label="grasper"):
    return [(i, label, "accept", True, True) for i in range(n)]


# ---- 1. human_reviewed means this exact annotation ----

def test_propagated_annotation_is_never_marked_human_reviewed_because_its_keyframe_was(job):
    clear(job)
    X.release(job, reviewer="r")
    c = coco(job, "released_coco.json")
    q = {(i["frame_index"], i["label"]) for i in X.final_pass_queue(job)}
    frames = {i["id"]: i["frame_index"] for i in c["images"]}
    prop = [a for a in c["annotations"] if a["attributes"]["propagated"]]
    assert prop
    for a in prop:
        label = next(k for k, v in X.CATEGORY_ID.items() if v == a["category_id"])
        assert a["attributes"]["human_reviewed"] is ((frames[a["image_id"]], label) in q)
        assert a["attributes"]["derived_from_reviewed"] is True


def test_derived_from_reviewed_needs_every_source_reviewed(tmp_path):
    job = build(tmp_path, many(40), [track(10, "grasper", TWO)], n_kf=40)
    X.run(job)
    queue = X.final_pass_queue(job)
    reviewed_kfs = {i["kf"] for i in queue}
    clear(job)
    X.release(job, reviewer="r")
    a = next(a for a in coco(job, "released_coco.json")["annotations"] if a["image_id"] == 11)
    assert a["attributes"]["derived_from_reviewed"] is all(k in reviewed_kfs for k in TWO)
    assert a["attributes"]["reviewed_source_keyframes"] == [k for k in TWO if k in reviewed_kfs]


def test_unaudited_accepted_keyframe_is_not_human_reviewed(tmp_path):
    job = build(tmp_path, many(40), n_kf=40)
    X.run(job)
    queued = {i["frame_index"] for i in X.final_pass_queue(job)}
    clear(job)
    X.release(job, reviewer="r")
    for a in coco(job, "released_coco.json")["annotations"]:
        assert a["attributes"]["human_reviewed"] is (a["image_id"] - 1 in queued)


# ---- 2. audit stop rule, and no empty release ----

def fail_audit(job, n):
    bad = [i for i in X.final_pass_queue(job) if i["kind"] == "audit" and i.get("stratum") == "accepted"][:n]
    return {(i["kf"], i["label"]): {"action": "reject"} for i in bad}


def test_failed_audit_blocks_release_and_escalates_to_full_review(tmp_path):
    job = build(tmp_path, many(40), n_kf=40)
    X.run(job)
    n_before = len(X.final_pass_queue(job))
    clear(job, fail_audit(job, 3))
    state = X.final_pass_state(job)
    assert state["escalated"] and state["locked"]
    assert len(X.final_pass_queue(job)) > n_before
    assert sum(i["kind"] == "escalation" for i in X.final_pass_queue(job)) == 40 - sum(
        1 for i in X.final_pass_queue(job) if i["kind"] == "audit" and i["stratum"] == "accepted")
    with pytest.raises(X.ReleaseLocked):
        X.release(job, reviewer="r")
    assert not (job / "released_coco.json").exists()
    clear(job)
    info = X.release(job, reviewer="r")
    assert info["final_pass"]["escalated"] is True and info["final_pass"]["n_audit_failed"] == 3


def test_changing_verdicts_afterwards_cannot_undo_an_escalation(tmp_path):
    job = build(tmp_path, many(40), n_kf=40)
    X.run(job)
    bad = fail_audit(job, 3)
    clear(job, bad)
    X.record_final_pass(job, [{"kf": k[0], "label": k[1], "action": "approve"} for k in bad])
    assert X.final_pass_state(job)["escalated"] and X.final_pass_state(job)["locked"]


def test_hand_written_failures_escalate_at_release_without_passing_through_record(tmp_path):
    job = build(tmp_path, many(40), n_kf=40)
    X.run(job)
    bad = fail_audit(job, 3)
    with open(job / "final_pass.jsonl", "a") as f:
        for i in X.final_pass_queue(job):
            f.write(json.dumps({"kf": i["kf"], "label": i["label"], "action": bad.get((i["kf"], i["label"]), {}).get("action", "approve")}) + "\n")
    with pytest.raises(X.ReleaseLocked):
        X.release(job, reviewer="r")
    assert not (job / "released_coco.json").exists()


def test_audit_within_threshold_releases_and_reports_the_rate(tmp_path):
    rows = many(120)
    job = build(tmp_path, rows, n_kf=120, gap=2)
    X.run(job)
    n_audit = sum(i["kind"] == "audit" for i in X.final_pass_queue(job))
    assert n_audit >= 10, "the threshold needs enough items for one failure to stay under it"
    clear(job, fail_audit(job, 1))
    assert not X.final_pass_state(job)["escalated"]
    info = X.release(job, reviewer="r")
    assert info["final_pass"]["n_audit_failed"] == 1 and info["final_pass"]["audit_failure_rate"] == round(1 / n_audit, 4)


def test_empty_queue_never_releases(tmp_path):
    job = build(tmp_path, [], n_kf=2)
    X.run(job)
    assert X.final_pass_queue(job) == []
    with pytest.raises(X.ReleaseLocked, match="[Nn]othing"):
        X.release(job, reviewer="r")
    assert not (job / "released_coco.json").exists()


def test_correcting_an_ai_absent_item_is_a_failure_but_confirming_absence_is_not(tmp_path):
    rows = [(i, "blood", "reject", False, False) for i in range(40)]
    job = build(tmp_path, rows, n_kf=40)
    X.run(job)
    items = X.final_pass_queue(job)
    assert items and all(i["stratum"] == "ai_absent" for i in items)
    clear(job, default="reject")
    assert not X.final_pass_state(job)["escalated"]
    poly = {"action": "correct", "polygon": [[10, 10], [80, 10], [80, 80]]}
    X.record_final_pass(job, [{"kf": i["kf"], "label": i["label"], "reviewer": "r", **poly} for i in items[:3]])
    assert X.final_pass_state(job)["escalated"]


# ---- 3. audit covers AI rejects and propagated frames, by label ----

def test_audit_covers_every_stratum_for_every_label(tmp_path):
    rows = [(i, lab, "accept", True, True) for i in range(10) for lab in ("grasper", "liver")] + [(i, "blood", "reject", False, False) for i in range(10)]
    tracks = [track(i, lab, ["kf_00000", "kf_00001"]) for i in range(1, 25) for lab in ("grasper", "liver")]
    job = build(tmp_path, rows, tracks, n_kf=10)
    X.run(job)
    seen = {(i["stratum"], i["label"]) for i in X.final_pass_queue(job) if i["kind"] == "audit"}
    assert {("accepted", "grasper"), ("accepted", "liver"), ("ai_absent", "blood"), ("propagated", "grasper"), ("propagated", "liver")} <= seen


def test_propagated_audit_item_shows_the_real_frame_and_mask(job):
    item = next(i for i in X.final_pass_queue(job) if i["stratum"] == "propagated")
    shown = cv2.imread(str(job / item["frame"]))
    cap = cv2.VideoCapture(str(job / "source.mp4"))
    for _ in range(item["frame_index"] + 1):
        ok, frame = cap.read()
    cap.release()
    assert ok and shown.shape == frame.shape and np.abs(shown.astype(int) - frame).mean() < 8
    assert M.read_mask(job / item["mask"]).tolist() == M.read_mask(M.track_path(job, item["frame_index"], item["label"])).tolist()


def test_propagated_audit_failure_drops_its_run_and_counts_it(tmp_path):
    tracks = [track(i, "grasper", TWO) for i in range(1, 25)]
    rows = many(2)
    job = build(tmp_path, rows, tracks)
    X.run(job)
    item = next(i for i in X.final_pass_queue(job) if i["stratum"] == "propagated")
    # Two keyframes plus a propagated item is too small a sample to stay under the threshold, so this escalates; clear that too.
    clear(job, {(item["kf"], item["label"]): {"action": "reject"}})
    clear(job)
    info = X.release(job, reviewer="r")
    c = coco(job, "released_coco.json")
    assert not [a for a in c["annotations"] if a["attributes"]["propagated"]]
    assert info["n_propagated_dropped_after_human_overrule"] == 24 - 1  # all but the frame the human judged directly
    assert "grasper" in next(i for i in c["images"] if i["frame_index"] == item["frame_index"])["attributes"]["excluded_overruled"]


def test_propagated_sample_is_capped_so_review_stays_possible(tmp_path):
    tracks = [track(i, "grasper", TWO) for i in range(1, 1000)]
    job = build(tmp_path, many(2), tracks, video=True, gap=500)
    X.run(job)
    assert sum(i["stratum"] == "propagated" for i in X.final_pass_queue(job)) <= X.MAX_PROPAGATED_AUDIT_PER_LABEL


def test_dataset_never_says_verified(job):
    clear(job)
    X.release(job, reviewer="r")
    for name in ("dataset_coco.json", "released_coco.json"):
        text = (job / name).read_text().lower()
        assert "verified" not in text, name
    first = next(i for i in coco(job)["images"] if i["frame_index"] == 25)
    assert "ai_absent" in first["attributes"] or "human_absent" in first["attributes"]


def test_ai_absent_and_human_absent_stay_distinct(tmp_path):
    rows = [(i, "blood", "reject", False, False) for i in range(40)]
    job = build(tmp_path, rows, n_kf=40)
    X.run(job)
    queued = {i["frame_index"] for i in X.final_pass_queue(job)}
    clear(job, default="reject")
    X.release(job, reviewer="r")
    for img in coco(job, "released_coco.json")["images"]:
        reviewed = img["frame_index"] in queued
        assert ("blood" in img["attributes"]["human_absent"]) is reviewed
        assert ("blood" in img["attributes"]["ai_absent"]) is not reviewed


def amber_accepted_job(tmp_path, n_kf=40):
    """Three false labels the amber loop accepted, feeding a long propagated run, among many both passes accepted."""
    rows = [(i, "liver", "accept", True, True) for i in range(n_kf)] + [(i, "l_hook", "accept", True, True) for i in range(3)]
    tracks = [track(i, "l_hook", ["kf_00000", "kf_00001"]) for i in range(1, 25)] + [track(i, "l_hook", ["kf_00001", "kf_00002"]) for i in range(26, 50)]
    job = build(tmp_path, rows, tracks, n_kf=n_kf)
    decisions = M.read_jsonl(job / "decisions.jsonl")
    for d in decisions:
        d["route"] = "amber_said_present" if d["label"] == "l_hook" else "both_passes_present"
    M.write_jsonl(job / "decisions.jsonl", decisions)
    X.run(job)
    return job


def test_every_amber_accepted_label_is_audited(tmp_path):
    job = amber_accepted_job(tmp_path)
    picked = {(i["kf"], i["label"]) for i in X.final_pass_queue(job) if i["stratum"] == "amber_accepted"}
    assert picked == {(f"kf_{i:05d}", "l_hook") for i in range(3)}


def test_amber_accepted_false_positives_are_caught_escalate_and_take_their_frames_with_them(tmp_path):
    job = amber_accepted_job(tmp_path)
    false = {(f"kf_{i:05d}", "l_hook"): {"action": "reject"} for i in range(3)}
    clear(job, false)
    assert X.final_pass_state(job)["escalated"], "several false labels of one class must trip the stop rule"
    clear(job)
    X.release(job, reviewer="r")
    assert anns(coco(job, "released_coco.json"), "l_hook") == []


def test_amber_accepted_pool_is_capped(tmp_path):
    rows = [(i, "liver", "accept", True, True) for i in range(40)]
    job = build(tmp_path, rows, n_kf=40)
    decisions = M.read_jsonl(job / "decisions.jsonl")
    for d in decisions:
        d["route"] = "amber_said_present"
    M.write_jsonl(job / "decisions.jsonl", decisions)
    X.run(job)
    assert sum(i["stratum"] == "amber_accepted" for i in X.final_pass_queue(job)) == X.MAX_AMBER_ACCEPTED_AUDIT_PER_LABEL


def test_route_reaches_coco_and_provenance(tmp_path):
    job = amber_accepted_job(tmp_path)
    c = coco(job)
    kf = next(a for a in anns(c, "l_hook") if not a["attributes"]["propagated"])
    prop = next(a for a in anns(c, "l_hook") if a["attributes"]["propagated"])
    assert kf["attributes"]["decision_route"] == "amber_said_present"
    assert prop["attributes"]["decision_route"] == ["amber_said_present", "amber_said_present"]
    rows = [r for r in M.read_jsonl(job / "provenance.jsonl") if r["type"] == "keyframe_label" and r["label"] == "liver"]
    assert rows and all(r["route"] == "both_passes_present" for r in rows)


# ---- 4. release re-render, manifest, durable final pass ----

def first_frame(path):
    cap = cv2.VideoCapture(str(path)); ok, f = cap.read(); cap.release(); assert ok
    return f.astype(int)


def test_draft_video_is_marked_and_release_rerenders_it(job):
    R.run(job)
    draft = first_frame(job / "annotated.mp4")
    before = hashlib.sha256((job / "annotated.mp4").read_bytes()).hexdigest()
    clear(job)
    info = X.release(job, reviewer="r")
    assert info["annotated_rerendered"] is True
    assert hashlib.sha256((job / "annotated.mp4").read_bytes()).hexdigest() != before
    assert np.abs(first_frame(job / "annotated.mp4") - draft).mean() > 0.05
    assert R.count_video_frames(job / "annotated.mp4") == 50


def test_draft_footer_and_banner_say_unreviewed_draft():
    assert "draft" in R.status_text(False)[0].lower() and "unreviewed" in R.status_text(False)[0].lower()
    assert "final pass" in R.status_text(True)[0].lower()
    assert "verified" not in " ".join(R.status_text(True) + R.status_text(False)).lower()


def test_released_video_stops_drawing_what_the_human_rejected(tmp_path):
    job = build(tmp_path, BASE, TRACKS)
    X.run(job)
    clear(job, {("kf_00000", "grasper"): {"action": "reject"}, ("kf_00001", "grasper"): {"action": "reject"}})
    clear(job)
    X.release(job, reviewer="r")
    c = coco(job, "released_coco.json")
    rows = R.rows_from_release(c, M.read_jsonl(job / "tracks.jsonl"))
    assert not [r for fr in rows.values() for r in fr if r["label"] == "grasper"]
    assert [r for fr in rows.values() for r in fr if r["label"] == "liver"]


def test_rendered_video_has_no_audio_or_container_metadata(tmp_path):
    job = build(tmp_path, BASE, TRACKS)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={W}x{H}:rate={FPS}:duration=2", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-metadata", "title=patient-name", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(job / "source.mp4")], check=True)
    R.run(job)
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type:format_tags", "-of", "json", str(job / "annotated.mp4")], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    assert [s["codec_type"] for s in d["streams"]] == ["video"]
    assert "patient-name" not in out


def test_failed_render_at_release_releases_nothing(job, monkeypatch):
    clear(job)
    def boom(*a, **k):
        raise RuntimeError("render failed")
    monkeypatch.setattr(R, "run", boom)
    with pytest.raises(RuntimeError):
        X.release(job, reviewer="r")
    for name in ("released_coco.json", "release_manifest.json", "dataset_bundle.zip"):
        assert not (job / name).exists(), name


@pytest.fixture
def released(job):
    clear(job)
    X.release(job, reviewer="r")
    return job


def test_manifest_passes_on_an_untouched_release(released):
    res = X.verify_release(released)
    assert res["ok"] is True and res["problems"] == []
    assert {"released_coco.json", "annotated.mp4", "final_pass.jsonl", "provenance.jsonl", "dataset_bundle.zip"} <= set(res["files"])


@pytest.mark.parametrize("name", ["released_coco.json", "annotated.mp4", "final_pass.jsonl", "provenance.jsonl", "dataset_bundle.zip"])
def test_editing_a_released_file_is_detected(released, name):
    p = released / name
    p.write_bytes(p.read_bytes() + b"\n ")
    res = X.verify_release(released)
    assert res["ok"] is False and any(name in x for x in res["problems"])


def test_rehashing_after_an_edit_without_the_key_is_detected(released):
    p = released / "released_coco.json"
    p.write_bytes(p.read_bytes() + b" ")
    man = json.loads((released / "release_manifest.json").read_text())
    man["files"]["released_coco.json"]["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    man["files"]["released_coco.json"]["bytes"] = p.stat().st_size
    (released / "release_manifest.json").write_text(json.dumps(man))
    res = X.verify_release(released)
    assert res["ok"] is False and any("signature" in x for x in res["problems"])


def test_deleted_manifest_or_file_is_detected(released):
    (released / "provenance.jsonl").unlink()
    assert X.verify_release(released)["ok"] is False
    (released / "release_manifest.json").unlink()
    assert X.verify_release(released)["ok"] is False


def test_signing_key_is_private_and_outside_the_job(released):
    key = X.signing_key_path(released)
    assert released not in key.parents and (key.stat().st_mode & 0o077) == 0
    with zipfile.ZipFile(released / "dataset_bundle.zip") as z:
        assert not [n for n in z.namelist() if "key" in n.lower().rsplit("/", 1)[-1]]


def test_second_release_is_refused_and_changes_nothing(released):
    before = (released / "released_coco.json").read_bytes()
    with pytest.raises(X.ReleaseLocked):
        X.release(released, reviewer="other")
    assert (released / "released_coco.json").read_bytes() == before


def test_export_rerun_does_not_invalidate_an_intact_release(released):
    X.run(released)
    assert X.verify_release(released)["ok"] is True


def test_torn_last_line_is_tolerated_and_repaired(job):
    q = X.final_pass_queue(job)
    X.record_final_pass(job, [{"kf": q[0]["kf"], "label": q[0]["label"], "action": "approve"}])
    with open(job / "final_pass.jsonl", "a") as f:
        f.write('{"kf": "kf_00000", "label": "liv')
    assert list(X.read_final_pass(job)) == [(q[0]["kf"], q[0]["label"])]
    X.record_final_pass(job, [{"kf": q[1]["kf"], "label": q[1]["label"], "action": "approve"}])
    lines = (job / "final_pass.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(l) for l in lines)


def test_corruption_before_the_last_line_is_an_error_not_silently_skipped(job):
    (job / "final_pass.jsonl").write_text('garbage\n{"kf": "kf_00000", "label": "liver", "action": "approve"}\n')
    with pytest.raises(ValueError, match="final_pass.jsonl"):
        X.read_final_pass(job)


def test_final_pass_rows_are_fsynced_before_returning(job, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    q = X.final_pass_queue(job)
    calls.clear()
    X.record_final_pass(job, [{"kf": q[0]["kf"], "label": q[0]["label"], "action": "approve"}])
    assert calls


# ---- 5. propagation cross-check ----

def textured(seed=3):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 60, (H // 8, W // 8, 3)).astype(np.float32)
    return np.clip(cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC) + 90, 0, 255).astype(np.uint8)


def flow_job(tmp_path, n, kf_masks, kf_indices):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    bg = textured()
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                          "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(job / "source.mp4")], stdin=subprocess.PIPE)
    for _ in range(n):
        p.stdin.write(bg.tobytes())
    p.stdin.close(); assert p.wait() == 0
    kfs, rows = [], []
    for j, fi in enumerate(kf_indices):
        kf = f"kf_{j:05d}"
        kfs.append({"kf": kf, "frame_index": fi, "t": fi / FPS})
        cv2.imwrite(str(job / "frames" / f"{kf}.jpg"), bg)
        if fi in kf_masks:
            M.write_mask(M.mask_path(job, kf, "liver"), kf_masks[fi])
            rows.append({"kf": kf, "label": "liver", "status": "accept", "path": f"masks/{kf}_liver.png"})
    (job / "meta.json").write_text(json.dumps({"fps": FPS, "width": W, "height": H, "n_frames": n, "kf_fps": 1, "keyframes": kfs}))
    (job / "masks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return job


def test_keyframes_that_agree_pass_the_flow_check(tmp_path):
    job = flow_job(tmp_path, 26, {0: disc(), 25: disc()}, [0, 25])
    P.run(job)
    rows = [r for r in M.read_jsonl(job / "tracks.jsonl") if r["propagated"]]
    assert rows and all(r["flow_check"] == "ok" and r["status"] == "accept" and r["fb_overlap"] > 0.8 for r in rows)


def test_keyframes_that_disagree_send_the_gap_to_amber(tmp_path):
    job = flow_job(tmp_path, 26, {0: disc(70, 70, 30), 25: disc(250, 170, 30)}, [0, 25])
    P.run(job)
    rows = [r for r in M.read_jsonl(job / "tracks.jsonl") if r["propagated"]]
    assert rows and all(r["flow_check"] == "low_overlap" and r["status"] == "AMBER" and r["fb_overlap"] < P.MIN_FB_OVERLAP for r in rows)


def test_object_at_one_end_only_gets_zero_overlap_and_amber(tmp_path):
    job = flow_job(tmp_path, 26, {0: disc()}, [0, 25])
    P.run(job)
    rows = [r for r in M.read_jsonl(job / "tracks.jsonl") if r["propagated"]]
    assert rows and all(r["fb_overlap"] == 0.0 and r["status"] == "AMBER" for r in rows)


def test_frames_with_no_second_direction_are_never_passed_as_checked(tmp_path):
    job = flow_job(tmp_path, 30, {0: disc(), 15: disc()}, [0, 15])
    P.run(job)
    tail = [r for r in M.read_jsonl(job / "tracks.jsonl") if r["frame_index"] > 15]
    assert tail and all(r["flow_check"] == "none" and r["fb_overlap"] is None and r["status"] == "AMBER" for r in tail)


def test_failed_flow_check_is_excluded_and_cannot_be_cleared_by_approving_the_keyframe(tmp_path):
    tracks = [track(10, "grasper", TWO), track(11, "grasper", TWO, status="AMBER", check="low_overlap", overlap=0.1), track(12, "grasper", TWO, status="AMBER", check="none", overlap=None)]
    job = build(tmp_path, many(2), tracks)
    X.run(job)
    c = coco(job)
    assert c["info"]["n_excluded_failed_flow_check"] == 2
    clear(job)
    X.release(job, reviewer="r")
    c = coco(job, "released_coco.json")
    frames = {a["image_id"] for a in c["annotations"] if a["attributes"]["propagated"]}
    assert frames == {11}


def test_propagated_image_keeps_its_exclusions_so_amber_never_reads_as_negative(tmp_path):
    tracks = [track(12, "liver", TWO, status="AMBER"), track(12, "grasper", TWO), track(13, "liver", TWO, status="AMBER")]
    job = build(tmp_path, BASE, tracks)
    X.run(job)
    images = {i["frame_index"]: i for i in coco(job)["images"]}
    assert images[12]["attributes"]["excluded_amber"] == ["liver"]
    # A frame whose only annotation was excluded still appears, marked, rather than vanishing.
    assert images[13]["attributes"]["excluded_amber"] == ["liver"]
    assert not [a for a in coco(job)["annotations"] if a["image_id"] == 14]
    assert images[12]["attributes"]["exhaustive"] is False and images[12]["attributes"]["ai_absent"] == []


# ---- 6. mask quality and instances ----

def test_every_mask_is_marked_draft_and_no_fallback_is_not_called_quality(job):
    X.run(job)
    for a in coco(job)["annotations"]:
        assert a["attributes"]["mask_quality"] == "draft"
        assert a["attributes"]["source"] in ("ai_polygon", "optical_flow_from_ai_keyframes", "human_correction")
    kf = next(a for a in coco(job)["annotations"] if not a["attributes"]["propagated"])
    assert kf["attributes"]["refinement"] == "grabcut"


def test_masks_jsonl_marks_quality_draft(tmp_path):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    img = np.full((H, W, 3), 30, np.uint8); img[disc()] = (60, 200, 70)
    cv2.imwrite(str(job / "frames" / "kf_00000.jpg"), img)
    (job / "meta.json").write_text(json.dumps({"fps": 25, "width": W, "height": H, "n_frames": 1, "kf_fps": 1, "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0}]}))
    poly = [[120, 80], [200, 80], [200, 160], [120, 160]]
    (job / "detect.jsonl").write_text(json.dumps({"kf": "kf_00000", "label": "liver", "present": True, "conf": 0.9, "polygon": poly, "model": "m", "prompt_hash": "h"}) + "\n")
    (job / "decisions.jsonl").write_text(json.dumps({"kf": "kf_00000", "label": "liver", "status": "accept", "final_present": True, "rounds": []}) + "\n")
    M.run(job)
    assert M.read_jsonl(job / "masks.jsonl")[0]["quality"] == "draft"


def test_instance_ids_follow_continuous_presence(tmp_path):
    tracks = [track(i, "grasper", TWO) for i in (1, 2, 3)] + [track(i, "grasper", TWO) for i in (10, 11)] + [track(12, "grasper", TWO, shot=1)] + [track(2, "liver", TWO)]
    job = build(tmp_path, [(0, "grasper", "accept", True, True), (1, "grasper", "accept", True, True), (0, "liver", "accept", True, True), (1, "liver", "accept", True, True)], tracks)
    X.run(job)
    ids = {}
    cats = {v: k for k, v in X.CATEGORY_ID.items()}
    for a in coco(job)["annotations"]:
        ids[(cats[a["category_id"]], a["image_id"] - 1)] = a["attributes"]["instance_id"]
    assert ids[("grasper", 1)] == ids[("grasper", 2)] == ids[("grasper", 3)] == ids[("grasper", 0)]
    assert ids[("grasper", 10)] == ids[("grasper", 11)] != ids[("grasper", 3)]
    assert ids[("grasper", 12)] != ids[("grasper", 11)], "a scene cut starts a new instance"
    assert ids[("liver", 2)] != ids[("grasper", 2)]


def test_propagate_records_the_shot_for_each_frame(tmp_path):
    bg1, bg2 = textured(3), cv2.bitwise_not(textured(11))
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(job / "source.mp4")], stdin=subprocess.PIPE)
    for i in range(26):
        p.stdin.write((bg1 if i < 13 else bg2).tobytes())
    p.stdin.close(); assert p.wait() == 0
    for j, fi in enumerate((0, 25)):
        cv2.imwrite(str(job / "frames" / f"kf_{j:05d}.jpg"), bg1)
        M.write_mask(M.mask_path(job, f"kf_{j:05d}", "liver"), disc())
    (job / "meta.json").write_text(json.dumps({"fps": FPS, "width": W, "height": H, "n_frames": 26, "kf_fps": 1, "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0}, {"kf": "kf_00001", "frame_index": 25, "t": 1}]}))
    (job / "masks.jsonl").write_text("".join(json.dumps({"kf": f"kf_{j:05d}", "label": "liver", "status": "accept", "path": f"masks/kf_{j:05d}_liver.png"}) + "\n" for j in (0, 1)))
    P.run(job)
    shots = {r["frame_index"]: r["shot"] for r in M.read_jsonl(job / "tracks.jsonl")}
    assert shots[0] == 0 and shots[25] == 1


# ---- 7. dataset bundle ----

@pytest.fixture
def bundle(released):
    with zipfile.ZipFile(released / "dataset_bundle.zip") as z:
        return {"job": released, "names": set(z.namelist()), "read": lambda n: zipfile.ZipFile(released / "dataset_bundle.zip").read(n)}


def test_no_bundle_before_release(job):
    assert not (job / "dataset_bundle.zip").exists()


def test_bundle_holds_everything_the_card_promises(bundle):
    want = {"annotations/released_coco.json", "annotations/provenance.jsonl", "annotations/final_pass.jsonl", "phases.jsonl", "narration.json", "narration.vtt",
            "tips.jsonl", "confidence.json", "versions.json", "prompts/prompts.json", "code/prompts.py", "DATASET_CARD.md", "LICENSES.md",
            "schema/released_coco.schema.json", "tools/extract_frames.py", "source.mp4", "annotated.mp4", "MANIFEST.json"}
    assert want <= bundle["names"], want - bundle["names"]


def test_bundled_dataset_is_byte_for_byte_the_released_one(bundle):
    assert bundle["read"]("annotations/released_coco.json") == (bundle["job"] / "released_coco.json").read_bytes()
    assert bundle["read"]("annotations/provenance.jsonl") == (bundle["job"] / "provenance.jsonl").read_bytes()
    assert bundle["read"]("annotated.mp4") == (bundle["job"] / "annotated.mp4").read_bytes()


def test_every_coco_image_in_the_bundle_resolves(bundle):
    c = json.loads(bundle["read"]("annotations/released_coco.json"))
    assert c["images"]
    n_frames = R.count_video_frames(bundle["job"] / "source.mp4")
    for img in c["images"]:
        a = img["attributes"]
        if a["extracted"]:
            assert img["file_name"] in bundle["names"], img["file_name"]
        else:
            assert a["decode_from"] == {"video": "source.mp4", "frame_index": img["frame_index"]}
            assert 0 <= img["frame_index"] < n_frames


def test_extract_frames_tool_produces_the_named_files(bundle, tmp_path):
    with zipfile.ZipFile(bundle["job"] / "dataset_bundle.zip") as z:
        z.extractall(tmp_path)
    subprocess.run(["python3", "-I", str(tmp_path / "tools/extract_frames.py"), str(tmp_path)], check=True, timeout=120)
    c = json.loads((tmp_path / "annotations/released_coco.json").read_text())
    for img in c["images"]:
        p = tmp_path / img["file_name"]
        assert p.exists(), p
        assert cv2.imread(str(p)).shape[:2] == (H, W)


def test_released_coco_validates_against_its_schema(bundle):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(bundle["read"]("schema/released_coco.schema.json"))
    c = json.loads(bundle["read"]("annotations/released_coco.json"))
    jsonschema.validate(c, schema)
    c["annotations"][0]["attributes"].pop("mask_quality")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(c, schema)


def test_versions_pin_the_models_actually_used_and_the_code(bundle):
    v = json.loads(bundle["read"]("versions.json"))
    assert v["models"]["detect"] == ["claude-sonnet-5-5"] and v["models"]["verify"] == ["claude-haiku-5-5"]
    assert v["vlm_provider"] == "claude_cli"
    assert v["code"]["files"]["surgproof/export.py"] == hashlib.sha256(bundle["read"]("code/export.py")).hexdigest()
    assert "git_commit" in v["code"]


def test_prompt_text_ships_verbatim_and_unmatched_hashes_are_named(bundle):
    from surgproof import prompts as PP
    assert bundle["read"]("code/prompts.py").decode() == open(PP.__file__).read()
    p = json.loads(bundle["read"]("prompts/prompts.json"))
    assert set(p["hashes"]) == {"ph", "vh"}
    assert p["hashes"]["ph"]["text"] is None and p["hashes"]["ph"]["resolved"] is False


def test_prompts_recorded_at_run_time_are_preferred(tmp_path):
    job = build(tmp_path, BASE, TRACKS)
    (job / "prompts.jsonl").write_text(json.dumps({"prompt_hash": "ph", "prompt": "the real prompt"}) + "\n")
    X.run(job)
    clear(job)
    X.release(job, reviewer="r")
    p = json.loads(zipfile.ZipFile(job / "dataset_bundle.zip").read("prompts/prompts.json"))
    assert p["hashes"]["ph"] == {"text": "the real prompt", "resolved": True, "how": "recorded_at_run_time"}


def test_prompt_hash_recomputed_from_the_template_really_matches(tmp_path):
    from surgproof import prompts as PP, vlm
    job = build(tmp_path, BASE, TRACKS)
    real = vlm.prompt_hash(PP.detect_prompt(W, H))
    rows = M.read_jsonl(job / "detect.jsonl")
    for r in rows:
        r["prompt_hash"] = real
    M.write_jsonl(job / "detect.jsonl", rows)
    X.run(job)
    clear(job)
    X.release(job, reviewer="r")
    p = json.loads(zipfile.ZipFile(job / "dataset_bundle.zip").read("prompts/prompts.json"))
    assert p["hashes"][real]["text"] == PP.detect_prompt(W, H) and p["hashes"][real]["how"] == "regenerated_and_hash_checked"


def test_card_counts_come_from_the_data_and_state_the_scope(bundle):
    card = bundle["read"]("DATASET_CARD.md").decode()
    c = json.loads(bundle["read"]("annotations/released_coco.json"))
    assert f"Annotations: {len(c['annotations'])}" in card and f"Images: {len(c['images'])}" in card
    low = card.lower()
    assert "laparoscopic cholecystectomy" in low and "draft" in low and "no real patient video" in low
    assert "verified" not in low and "any surgery" not in low


def test_card_says_when_rights_were_not_attested_and_when_they_were(bundle, tmp_path):
    assert "not attested" in bundle["read"]("LICENSES.md").decode().lower()
    job = build(tmp_path / "attested", BASE, TRACKS)
    (job / "rights.json").write_text(json.dumps({"statement": "public teaching video, licence CC BY"}))
    X.run(job)
    clear(job)
    X.release(job, reviewer="r")
    assert "public teaching video, licence CC BY" in zipfile.ZipFile(job / "dataset_bundle.zip").read("LICENSES.md").decode()


def test_mock_run_is_stated_on_the_card(tmp_path):
    job = build(tmp_path, BASE, TRACKS)
    rows = M.read_jsonl(job / "detect.jsonl")
    for r in rows:
        r["model"] = "mock:claude-sonnet-5-5"
    M.write_jsonl(job / "detect.jsonl", rows)
    X.run(job)
    clear(job)
    X.release(job, reviewer="r")
    assert "mock" in zipfile.ZipFile(job / "dataset_bundle.zip").read("DATASET_CARD.md").decode().lower()


def test_per_label_confidence_is_computed_and_says_it_is_uncalibrated(bundle):
    d = json.loads(bundle["read"]("confidence.json"))
    assert d["calibrated"] is False
    assert d["labels"]["grasper"]["mean_conf"] == 0.8 and d["labels"]["grasper"]["n_keyframes"] == 2
    assert d["labels"]["blood"]["n_ai_absent"] == 2


def test_tip_estimate_finds_the_end_nearest_the_image_centre_and_is_flagged():
    m = np.zeros((H, W), np.uint8)
    cv2.line(m, (0, 0), (150, 110), 1, 9)  # a shaft entering from the corner, ending near the centre
    tip = X.estimate_tip(m.astype(bool))
    assert np.hypot(tip[0] - 150, tip[1] - 110) < 15
    assert X.estimate_tip(np.zeros((H, W), bool)) is None


def test_tips_only_cover_instruments_and_are_flagged_unvalidated(bundle):
    rows = [json.loads(l) for l in bundle["read"]("tips.jsonl").decode().splitlines()]
    assert rows and {r["label"] for r in rows} <= {"grasper", "l_hook"}
    assert all(r["validated"] is False and r["method"] for r in rows)


def test_bundle_manifest_hashes_match_its_contents(bundle):
    man = json.loads(bundle["read"]("MANIFEST.json"))
    assert man["signature"]["alg"] == "HMAC-SHA256"
    for name, info in man["files"].items():
        assert hashlib.sha256(bundle["read"](name)).hexdigest() == info["sha256"], name


def test_bundle_refuses_when_disk_is_too_low_and_releases_nothing(job, monkeypatch):
    clear(job)
    import shutil, collections
    usage = collections.namedtuple("u", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 10, 0))
    with pytest.raises(OSError, match="disk"):
        X.release(job, reviewer="r")
    assert not (job / "released_coco.json").exists()
