"""Stage 9 and the final pass: ambers stay out, release stays locked, and human work is labelled as human."""
import json

import cv2
import numpy as np
import pytest

from surgproof import export as X, masks as M

W, H = 320, 240


def disc(cx=160, cy=120, r=40):
    m = np.zeros((H, W), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def make_job(tmp_path, kf_rows, tracks=(), n_kf=2):
    """kf_rows: (kf_number, label, status, final_present, has_mask)."""
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    keyframes = [{"kf": f"kf_{i:05d}", "frame_index": i * 25, "t": float(i)} for i in range(n_kf)]
    for k in keyframes:
        cv2.imwrite(str(job / "frames" / f"{k['kf']}.jpg"), np.zeros((H, W, 3), np.uint8))
    (job / "meta.json").write_text(json.dumps({"fps": 25.0, "width": W, "height": H, "n_frames": n_kf * 25, "kf_fps": 1, "keyframes": keyframes}))
    det, dec, ver, mrows = [], [], [], []
    for i, label, status, present, has_mask in kf_rows:
        kf = f"kf_{i:05d}"
        det.append({"kf": kf, "label": label, "present": bool(present), "conf": 0.8, "polygon": [[1, 1], [50, 1], [50, 50]] if present else None,
                    "model": "claude-sonnet-5-5", "prompt_hash": f"p{i}", "raw_ref": f"r{i}"})
        ver.append({"kf": kf, "label": label, "present": bool(present), "box": [1, 1, 50, 50], "model": "claude-haiku-5-5", "prompt_hash": "v"})
        dec.append({"kf": kf, "label": label, "status": status, "final_present": present, "rounds": []})
        if has_mask:
            M.write_mask(M.mask_path(job, kf, label), disc())
            mrows.append({"kf": kf, "label": label, "status": status, "seed": "detect_polygon", "fallback": None, "area": int(disc().sum()),
                          "box": M.mask_box(disc()), "path": f"masks/{kf}_{label}.png", "model": "claude-sonnet-5-5", "prompt_hash": f"p{i}"})
    for name, rows in [("detect", det), ("verify", ver), ("decisions", dec), ("masks", mrows), ("tracks", list(tracks))]:
        (job / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for t in tracks:
        if t["propagated"]:
            M.write_mask(M.track_path(job, t["frame_index"], t["label"]), disc())
    return job


def track(fi, label, status, sources):
    return {"frame_index": fi, "label": label, "box": [120, 80, 200, 160], "box_raw": [120, 80, 200, 160], "propagated": True,
            "status": status, "source_kfs": sources, "keyframe": None, "t": fi / 25}


STANDARD = [(0, "grasper", "accept", True, True), (0, "liver", "AMBER", None, True), (0, "blood", "reject", False, False),
            (1, "grasper", "accept", True, True), (1, "liver", "accept", True, True)]
TRACKS = [track(10, "grasper", "accept", ["kf_00000", "kf_00001"]), track(12, "liver", "AMBER", ["kf_00000", "kf_00001"])]


@pytest.fixture
def job(tmp_path):
    return make_job(tmp_path, STANDARD, TRACKS)


def coco(job, name="dataset_coco.json"):
    return json.loads((job / name).read_text())


def anns(c, label):
    cid = X.CATEGORY_ID[label]
    return [a for a in c["annotations"] if a["category_id"] == cid]


def test_coco_structure_is_valid(job):
    X.run(job)
    c = coco(job)
    assert {"info", "images", "categories", "annotations", "licenses"} <= set(c)
    assert sorted(x["name"] for x in c["categories"]) == sorted(M.LABELS)
    ids = {i["id"] for i in c["images"]}
    for a in c["annotations"]:
        assert a["image_id"] in ids and a["iscrowd"] == 0 and a["area"] > 0
        x, y, w, h = a["bbox"]
        assert w > 0 and h > 0 and x + w <= W and y + h <= H
        for poly in a["segmentation"]:
            assert len(poly) >= 6 and len(poly) % 2 == 0
        for key in ("status", "source", "keyframe", "propagated", "model", "prompt_hash"):
            assert key in a["attributes"]
    assert len({a["id"] for a in c["annotations"]}) == len(c["annotations"])


def test_polygons_rasterize_back_to_the_mask():
    m = disc()
    polys = X.mask_to_polygons(m)
    back = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.fillPoly(back, [np.array(p).reshape(-1, 2)], 1)
    assert (back.astype(bool) & m).sum() / (back.astype(bool) | m).sum() > 0.97


def test_multi_part_masks_give_multiple_polygons_and_true_area():
    m = disc(80, 80, 20) | disc(240, 160, 20)
    assert len(X.mask_to_polygons(m)) == 2
    assert X.coco_bbox(M.mask_box(m)) == [60, 60, 201, 121]


def test_ambers_are_excluded_from_the_draft_dataset(job):
    summary = X.run(job)
    c = coco(job)
    assert summary["n_excluded_amber_keyframe"] == 1 and summary["n_excluded_amber_propagated"] == 1
    assert all(a["attributes"]["status"] == "accept" for a in c["annotations"])
    assert len(anns(c, "liver")) == 1  # only kf_00001, and its propagated frame depends on an amber
    first = next(i for i in c["images"] if i["frame_index"] == 0)
    assert first["attributes"]["excluded_amber"] == ["liver"]
    assert first["attributes"]["verified_absent"] == ["blood"]


def test_rejected_labels_never_appear(job):
    X.run(job)
    assert anns(coco(job), "blood") == []


def test_draft_is_marked_unreleased_and_release_file_absent(job):
    X.run(job)
    assert coco(job)["info"]["released"] is False
    assert not (job / "released_coco.json").exists()


def test_propagated_annotation_carries_its_lineage(job):
    X.run(job)
    c = coco(job)
    a = next(a for a in anns(c, "grasper") if a["attributes"]["propagated"])
    assert a["attributes"]["source_keyframes"] == ["kf_00000", "kf_00001"]
    assert a["attributes"]["keyframe"] is None and a["image_id"] == 11
    img = next(i for i in c["images"] if i["id"] == 11)
    assert img["attributes"]["extracted"] is False and img["frame_index"] == 10


def test_every_keyframe_is_an_image_even_when_empty(tmp_path):
    job = make_job(tmp_path, [(0, "grasper", "reject", False, False)], n_kf=3)
    X.run(job)
    assert [i["frame_index"] for i in coco(job)["images"]] == [0, 25, 50]


def test_provenance_traces_each_decision_to_model_and_mask_hash(job):
    X.run(job)
    rows = M.read_jsonl(job / "provenance.jsonl")
    assert rows[0]["type"] == "job"
    kl = {(r["kf"], r["label"]): r for r in rows if r["type"] == "keyframe_label"}
    assert len(kl) == len(STANDARD)
    g = kl[("kf_00000", "grasper")]
    assert g["detect"]["model"] == "claude-sonnet-5-5" and g["detect"]["raw_ref"] == "r0"
    assert g["verify"]["model"] == "claude-haiku-5-5"
    assert len(g["mask"]["sha256"]) == 64
    assert kl[("kf_00000", "blood")]["mask"] is None and kl[("kf_00000", "blood")]["status"] == "reject"
    runs = [r for r in rows if r["type"] == "propagated_run"]
    assert {r["label"] for r in runs} == {"grasper", "liver"}


def test_fallback_mask_is_labelled_unrefined(tmp_path):
    job = make_job(tmp_path, [(0, "grasper", "accept", True, True)])
    rows = M.read_jsonl(job / "masks.jsonl"); rows[0]["fallback"] = "grabcut_leaked"
    M.write_jsonl(job / "masks.jsonl", rows)
    X.run(job)
    assert coco(job)["annotations"][0]["attributes"]["source"] == "ai_polygon_unrefined"


# ---- final pass ----

def many_accepts(n=40):
    return [(i, "grasper" if i % 2 else "liver", "accept", True, True) for i in range(n)]


def test_queue_holds_every_amber_and_a_stable_audit_sample(tmp_path):
    job = make_job(tmp_path, many_accepts() + [(0, "blood", "AMBER", None, True), (1, "blood", "AMBER", None, True)], n_kf=40)
    q = X.final_pass_queue(job)
    assert sum(i["kind"] == "amber" for i in q) == 2
    audit = [i for i in q if i["kind"] == "audit"]
    assert 4 <= len(audit) <= 8
    assert {i["label"] for i in audit} == {"grasper", "liver"}
    # Asking again, even for a smaller sample, cannot re-roll the draw.
    assert X.final_pass_queue(job, audit_frac=0.01) == q


def test_audit_sample_is_not_just_the_first_items(tmp_path):
    job = make_job(tmp_path, many_accepts(), n_kf=40)
    idx = sorted(i["frame_index"] for i in X.final_pass_queue(job))
    assert idx != [25 * k for k in range(len(idx))]


def test_rare_label_still_gets_audited(tmp_path):
    rows = [(i, "liver", "accept", True, True) for i in range(40)] + [(0, "cystic_duct", "accept", True, True)]
    job = make_job(tmp_path, rows, n_kf=40)
    assert "cystic_duct" in {i["label"] for i in X.final_pass_queue(job)}


def test_release_stays_locked_until_every_item_is_done(job):
    X.run(job)
    q = X.final_pass_queue(job)
    assert len(q) >= 2
    with pytest.raises(X.ReleaseLocked):
        X.release(job)
    assert not (job / "released_coco.json").exists()
    X.record_final_pass(job, [{"kf": q[0]["kf"], "label": q[0]["label"], "action": "approve", "reviewer": "h"}])
    st = X.final_pass_state(job)
    assert st["locked"] and st["n_done"] == 1
    with pytest.raises(X.ReleaseLocked):
        X.release(job)
    X.record_final_pass(job, [{"kf": i["kf"], "label": i["label"], "action": "approve"} for i in q[1:]])
    assert not X.final_pass_state(job)["locked"]
    info = X.release(job, reviewer="h")
    assert info["released"] is True and info["human_final_pass"] is True and info["reviewer"] == "h"
    assert (job / "released_coco.json").exists()
    assert X.final_pass_state(job)["released"]


def test_job_with_nothing_to_review_releases_without_human_rows(tmp_path):
    job = make_job(tmp_path, [(0, "blood", "reject", False, False)])
    assert X.final_pass_queue(job) == []
    assert X.release(job)["n_annotations"] == 0


def resolve_all(job, overrides=None, default="approve"):
    overrides = overrides or {}
    rows = []
    for i in X.final_pass_queue(job):
        rows.append({"kf": i["kf"], "label": i["label"], "action": default, **overrides.get((i["kf"], i["label"]), {})})
    X.record_final_pass(job, rows)


def test_approved_amber_enters_release_with_its_propagated_frames(job):
    X.run(job)
    resolve_all(job)
    X.release(job)
    c = coco(job, "released_coco.json")
    liver = anns(c, "liver")
    frames = {a["image_id"] for a in liver}
    assert frames == {1, 26, 13}
    a = next(a for a in liver if a["image_id"] == 1)
    assert a["attributes"]["human_reviewed"] is True and a["attributes"]["source"] == "ai_polygon_grabcut"
    assert c["info"]["n_excluded_amber_keyframe"] == 0


def test_rejected_amber_drops_propagated_frames_and_counts_them(job):
    X.run(job)
    resolve_all(job, {("kf_00000", "liver"): {"action": "reject"}})
    info = X.release(job)
    c = coco(job, "released_coco.json")
    assert {a["image_id"] for a in anns(c, "liver")} == {26}
    assert info["n_propagated_dropped_after_human_overrule"] == 1
    first = next(i for i in c["images"] if i["frame_index"] == 0)
    assert "liver" in first["attributes"]["verified_absent"]


def test_correction_is_marked_human_and_uses_the_humans_polygon(job):
    X.run(job)
    poly = [[10, 10], [60, 10], [60, 60], [10, 60]]
    resolve_all(job, {("kf_00000", "liver"): {"action": "correct", "polygon": poly, "reviewer": "dr"}})
    info = X.release(job)
    c = coco(job, "released_coco.json")
    a = next(a for a in anns(c, "liver") if a["image_id"] == 1)
    assert a["attributes"]["source"] == "human_correction" and a["attributes"]["model"] is None and a["attributes"]["reviewer"] == "dr"
    assert a["bbox"] == [10, 10, 51, 51]
    assert info["n_human_corrections"] == 1
    assert info["n_propagated_dropped_after_human_overrule"] == 1


def test_audit_outcomes_are_reported_from_real_actions(tmp_path):
    job = make_job(tmp_path, many_accepts(), n_kf=40)
    q = X.final_pass_queue(job)
    bad = q[0]
    resolve_all(job, {(bad["kf"], bad["label"]): {"action": "reject"}})
    info = X.release(job)
    fp = info["final_pass"]
    assert fp["n_audit"] == len(q) and fp["n_audit_rejected"] == 1 and fp["n_audit_approved"] == len(q) - 1 and fp["n_audit_pool"] == 40
    c = coco(job, "released_coco.json")
    assert not [a for a in anns(c, bad["label"]) if a["image_id"] == bad["frame_index"] + 1]


def test_release_writes_final_pass_rows_into_provenance(job):
    X.run(job)
    resolve_all(job)
    X.release(job)
    rows = M.read_jsonl(job / "provenance.jsonl")
    assert any(r["type"] == "final_pass" and r["action"] == "approve" for r in rows)


def test_latest_review_wins(job):
    X.run(job)
    resolve_all(job)
    q = X.final_pass_queue(job)
    X.record_final_pass(job, [{"kf": q[0]["kf"], "label": q[0]["label"], "action": "reject"}])
    assert X.read_final_pass(job)[(q[0]["kf"], q[0]["label"])]["action"] == "reject"


@pytest.mark.parametrize("row,msg", [
    ({"kf": "kf_00009", "label": "liver", "action": "approve"}, "not in this job"),
    ({"kf": "kf_00000", "label": "liver", "action": "maybe"}, "unknown"),
    ({"kf": "kf_00000", "label": "liver", "action": "correct"}, "polygon"),
    ({"kf": "kf_00000", "label": "liver", "action": "correct", "polygon": [[1, 1]]}, "polygon"),
])
def test_bad_final_pass_rows_are_refused_and_not_stored(job, row, msg):
    X.run(job)
    with pytest.raises(ValueError, match=msg):
        X.record_final_pass(job, [row])
    assert not (job / "final_pass.jsonl").exists()


def test_a_stray_line_outside_the_queue_cannot_change_the_release(job):
    X.run(job)
    resolve_all(job)
    with open(job / "final_pass.jsonl", "a") as f:
        f.write(json.dumps({"kf": "kf_00001", "label": "cystic_duct", "action": "correct", "polygon": [[1, 1], [9, 1], [9, 9]]}) + "\n")
    X.release(job)
    assert anns(coco(job, "released_coco.json"), "cystic_duct") == []


def test_approving_an_amber_that_has_no_mask_confirms_absence(tmp_path):
    job = make_job(tmp_path, [(0, "blood", "AMBER", None, False), (0, "liver", "AMBER", None, True)])
    X.run(job)
    assert next(i for i in coco(job)["images"] if i["frame_index"] == 0)["attributes"]["excluded_amber"] == ["liver", "blood"]
    resolve_all(job)
    X.release(job)
    c = coco(job, "released_coco.json")
    first = next(i for i in c["images"] if i["frame_index"] == 0)
    assert first["attributes"]["verified_absent"] == ["blood"] and first["attributes"]["excluded_amber"] == []
    assert anns(c, "blood") == [] and len(anns(c, "liver")) == 1
