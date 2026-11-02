"""Stage 5 on synthetic images: known objects, so correctness is checkable without the model or any dataset."""
import json, re
from pathlib import Path

import cv2
import numpy as np
import pytest

from surgproof import masks as M

W, H = 320, 240


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def scene(seed=0):
    """Textured dark-red background, a bright green disc, and a thin grey rod."""
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(0, 6, (H, W, 3)) + np.array([30, 30, 120]), 0, 255).astype(np.uint8)
    disc = np.zeros((H, W), np.uint8)
    cv2.circle(disc, (110, 120), 50, 1, -1)
    rod = np.zeros((H, W), np.uint8)
    cv2.line(rod, (200, 30), (290, 210), 1, 7)
    img[disc > 0] = np.clip(rng.normal(0, 6, (int(disc.sum()), 3)) + np.array([60, 190, 70]), 0, 255)
    img[rod > 0] = np.clip(rng.normal(0, 5, (int(rod.sum()), 3)) + np.array([205, 205, 205]), 0, 255)
    return img, disc.astype(bool), rod.astype(bool)


def contour(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cs, key=cv2.contourArea)[:, 0, :].tolist()


def make_job(tmp_path, detect, decisions, verify=(), img=None):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    img = scene()[0] if img is None else img
    cv2.imwrite(str(job / "frames" / "kf_00000.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 98])
    (job / "meta.json").write_text(json.dumps({
        "fps": 25, "width": W, "height": H, "n_frames": 25, "kf_fps": 1,
        "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0.0}]}))
    for name, rows in [("detect", detect), ("decisions", decisions), ("verify", verify)]:
        (job / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return job


def det(label, polygon, present=True):
    return {"kf": "kf_00000", "label": label, "present": present, "conf": 0.9, "polygon": polygon,
            "model": "m", "prompt_hash": "h", "raw_ref": "r"}


def dec(label, status="accept", final_present=True, rounds=()):
    return {"kf": "kf_00000", "label": label, "status": status, "final_present": final_present, "rounds": list(rounds)}


def test_grabcut_improves_an_offset_oversized_seed():
    img, disc, _ = scene()
    seed = np.zeros((H, W), np.uint8)
    cv2.circle(seed, (120, 125), 62, 1, -1)  # shifted and too big, like a real LLM outline
    out, fallback = M.grabcut_refine(img, seed, "gallbladder")
    assert fallback is None
    assert iou(out, disc) > iou(seed.astype(bool), disc) + 0.1
    # Sure foreground can never be removed, which caps the IoU. The result must come close to that cap.
    dist = cv2.distanceTransform(seed, cv2.DIST_L2, 5)
    sure = dist > min(M.ORGAN_ERODE_FRAC * np.sqrt(seed.sum()), 0.5 * dist.max())
    assert iou(out, disc) > iou(disc | sure, disc) - 0.01


def test_thin_instrument_keeps_its_shape_instead_of_eroding_away():
    img, _, rod = scene()
    seed = cv2.dilate(rod.astype(np.uint8), np.ones((5, 5), np.uint8))
    out, fallback = M.grabcut_refine(img, seed, "grasper")
    assert fallback is None
    assert iou(out, rod) > 0.6


def test_collapse_falls_back_to_seed_and_says_so():
    # A uniform image gives GrabCut nothing, and any fallback must be reported, not hidden.
    img = np.full((H, W, 3), 128, np.uint8)
    seed = np.zeros((H, W), np.uint8)
    cv2.circle(seed, (160, 120), 40, 1, -1)
    out, fallback = M.grabcut_refine(img, seed, "liver")
    assert out.any()
    if fallback:
        assert np.array_equal(out, seed.astype(bool))


def test_run_writes_accept_and_amber_but_not_reject_or_absent(tmp_path):
    img, disc, rod = scene()
    poly = contour(disc)
    job = make_job(tmp_path, [det("gallbladder", poly), det("grasper", contour(rod)), det("liver", poly),
                              det("blood", None, False), det("cystic_duct", poly)],
                   [dec("gallbladder"), dec("grasper", "AMBER", None), dec("liver", "reject", False),
                    dec("blood", "reject", False), dec("cystic_duct", "accept", False)])
    summary = M.run(job)
    names = sorted(p.name for p in (job / "masks").glob("*.png"))
    assert names == ["kf_00000_gallbladder.png", "kf_00000_grasper.png"]
    assert summary["n_masks"] == 2
    rows = {r["label"]: r for r in M.read_jsonl(job / "masks.jsonl")}
    assert rows["grasper"]["status"] == "AMBER"
    m = cv2.imread(str(job / "masks" / "kf_00000_gallbladder.png"), cv2.IMREAD_UNCHANGED)
    assert m.shape == (H, W) and set(np.unique(m)) <= {0, 255}
    x0, y0, x1, y1 = rows["gallbladder"]["box"]
    ys, xs = np.where(m > 0)
    assert (x0, y0, x1, y1) == (xs.min(), ys.min(), xs.max(), ys.max())


def test_detection_without_a_decision_is_amber_never_accepted(tmp_path):
    _, disc, _ = scene()
    job = make_job(tmp_path, [det("gallbladder", contour(disc))], [])
    M.run(job)
    assert M.read_jsonl(job / "masks.jsonl")[0]["status"] == "AMBER"


def test_amber_round_polygon_overrides_detect_polygon(tmp_path):
    _, disc, rod = scene()
    round_row = {"round": 1, "polygon": contour(disc), "model": "round-model", "prompt_hash": "rh"}
    job = make_job(tmp_path, [det("gallbladder", contour(rod))], [dec("gallbladder", rounds=[round_row])])
    M.run(job)
    row = M.read_jsonl(job / "masks.jsonl")[0]
    assert row["seed"] == "amber_round_polygon" and row["model"] == "round-model"
    out = M.read_mask(job / row["path"])
    assert iou(out, disc) > 0.8


def test_present_without_polygon_uses_verifier_box_and_records_it(tmp_path):
    _, disc, _ = scene()
    ver = {"kf": "kf_00000", "label": "gallbladder", "present": True, "box": [60, 70, 160, 170], "model": "hk", "prompt_hash": "vh"}
    job = make_job(tmp_path, [det("gallbladder", None)], [dec("gallbladder")], [ver])
    M.run(job)
    row = M.read_jsonl(job / "masks.jsonl")[0]
    assert row["seed"] == "verify_box" and row["model"] == "hk"
    assert iou(M.read_mask(job / row["path"]), disc) > 0.7


@pytest.mark.parametrize("bad", [None, [], [[1, 2]], [[1, 2], [3, 4]], "nope", [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                 [[float("nan"), 1], [2, 3], [4, 5]], [[5, 5], [5, 5], [5, 5]]])
def test_unusable_polygon_is_skipped_and_reported(tmp_path, bad):
    job = make_job(tmp_path, [det("gallbladder", bad)], [dec("gallbladder")])
    summary = M.run(job)
    assert summary["n_masks"] == 0 and summary["n_skipped"] == 1
    assert not list((job / "masks").glob("*.png"))


def test_out_of_frame_polygon_is_clipped(tmp_path):
    _, disc, _ = scene()
    poly = [[-50, -50], [W + 50, -50], [W + 50, H + 50], [-50, H + 50]]
    job = make_job(tmp_path, [det("liver", poly)], [dec("liver")])
    M.run(job)
    assert M.read_mask(job / "masks" / "kf_00000_liver.png").shape == (H, W)


def test_masks_do_not_overlap_because_front_objects_occlude_back_ones(tmp_path):
    img, disc, rod = scene()
    whole = [[80, 10], [310, 10], [310, 230], [80, 230]]
    job = make_job(tmp_path, [det("liver", whole), det("grasper", contour(rod))], [dec("liver"), dec("grasper")], img=img)
    M.run(job)
    liver = M.read_mask(job / "masks" / "kf_00000_liver.png")
    tool = M.read_mask(job / "masks" / "kf_00000_grasper.png")
    assert (liver & tool).sum() == 0


def test_rerun_removes_stale_masks(tmp_path):
    _, disc, _ = scene()
    job = make_job(tmp_path, [det("gallbladder", contour(disc))], [dec("gallbladder")])
    M.run(job)
    (job / "decisions.jsonl").write_text(json.dumps(dec("gallbladder", "reject", False)) + "\n")
    M.run(job)
    assert not list((job / "masks").glob("*.png"))


def test_status_json_follows_the_contract(tmp_path):
    _, disc, _ = scene()
    job = make_job(tmp_path, [det("gallbladder", contour(disc))], [dec("gallbladder")])
    M.run(job)
    st = json.loads((job / "status.json").read_text())
    assert set(st) == {"stage", "progress", "message", "started", "updated", "error"}
    assert st["stage"] == "masks" and st["progress"] == 1.0


def test_on_progress_replaces_status_writes(tmp_path):
    _, disc, _ = scene()
    job = make_job(tmp_path, [det("gallbladder", contour(disc))], [dec("gallbladder")])
    seen = []
    M.run(job, on_progress=lambda f, m="": seen.append(f))
    assert seen[-1] == 1.0 and not (job / "status.json").exists()


LANE_FILES = ["masks.py", "propagate.py", "render.py", "export.py"]


@pytest.mark.parametrize("name", LANE_FILES)
def test_lane_code_never_touches_ground_truth(name):
    src = (Path(__file__).resolve().parent.parent / "surgproof" / name).read_text()
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for pat in [r"_endo_mask", r"watershed", r"color_mask", r"\bgt_", r"data/index\.csv", r"mask_path\s*=.*raw"]:
        assert not re.search(pat, code), f"{name} matches {pat}"
