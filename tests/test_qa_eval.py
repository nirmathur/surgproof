"""Scorer tests with synthetic ground truth and hand-computed answers. No model, no real dataset."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from surgproof import eval as ev  # noqa: E402

W, H = 64, 48
LIVER = 21      # watershed value for liver
GRASPER = 31
BG = 50
CLIP = "video12_00000"


def textured(seed):
    """Distinct smooth image per frame: survives JPEG like real video, and alignment can tell frames apart."""
    low = np.random.default_rng(seed).integers(0, 255, (6, 8, 3), dtype=np.uint8)
    return np.array(Image.fromarray(low).resize((W, H), Image.BICUBIC))


def gt_mask(frame):
    """Liver: left half, every frame. Grasper: 10x10 block at top right, only on frames below 3."""
    m = np.full((H, W), BG, dtype=np.uint8)
    m[:, : W // 2] = LIVER
    if frame < 3:
        m[0:10, 40:50] = GRASPER
    return m


@pytest.fixture
def world(tmp_path):
    raw = tmp_path / "raw"
    d = raw / "video12" / CLIP
    d.mkdir(parents=True)
    frames = []
    for i in range(6):
        Image.fromarray(textured(i)).save(d / f"frame_{i}_endo.png")
        Image.fromarray(np.dstack([gt_mask(i)] * 3)).save(d / f"frame_{i}_endo_watershed_mask.png")
        frames.append({"frame_index": i, "video": "video12", "clip": CLIP, "frame": i})
    manifest = {"sha256": "x", "fps": 25, "n_frames": 6, "clips": [CLIP], "frames": frames}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))

    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    (job / "masks").mkdir()
    kfs = [{"kf": f"kf_{i:05d}", "frame_index": i, "t": i / 25} for i in (0, 4)]
    for k in kfs:
        Image.fromarray(textured(k["frame_index"])).save(job / "frames" / f"{k['kf']}.jpg", quality=98)
    (job / "meta.json").write_text(json.dumps({"fps": 25, "width": W, "height": H, "n_frames": 6, "kf_fps": 1, "keyframes": kfs}))
    return {"raw": raw, "job": job, "manifest": mp, "tmp": tmp_path}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def decision(kf, label, status, present):
    return {"kf": kf, "label": label, "status": status, "final_present": present, "rounds": []}


def all_decisions(overrides=None):
    """All six labels on both keyframes, correct by default: liver on both, grasper on kf0 only."""
    rows = {}
    for kf, fi in (("kf_00000", 0), ("kf_00004", 4)):
        for lb in ev.LABELS:
            truth = lb == "liver" or (lb == "grasper" and fi < 3)
            rows[(kf, lb)] = decision(kf, lb, "accept" if truth else "reject", truth)
    rows.update(overrides or {})
    return list(rows.values())


def save_mask(job, kf, label, arr):
    Image.fromarray((arr.astype(np.uint8)) * 255).save(job / "masks" / f"{kf}_{label}.png")


def half_liver_mask():
    m = np.zeros((H, W), bool)
    m[:, : W // 4] = True
    return m


def score(world, **kw):
    return ev.score_job(world["job"], world["raw"], world["tmp"], world["manifest"], **kw)


def test_ground_truth_rule_matches_the_rest_of_the_repo():
    """eval.py copies two functions from prepare_data.py so it can run without pandas. They must not drift."""
    pytest.importorskip("pandas")
    import prepare_data as pd_mod

    frames = sorted((ROOT / "raw" / "CholecSeg8k" / "video12" / "video12_19500").glob("*_watershed_mask.png"))[:3]
    if not frames:
        pytest.skip("CholecSeg8k not on this machine")
    for f in frames:
        a, b = ev.mask_to_classes(f), pd_mod.mask_to_classes(f)
        assert (a == b).all()
        pa, pb = ev.presence(a), pd_mod.presence(b)
        assert all(pa[k] == pb[k] for k in pa)


def test_scorer_runs_without_pandas():
    """`make setup` installs requirements-e2e.txt only, so the scorer must not need the research stack."""
    code = "import sys; sys.modules['pandas'] = None; import surgproof.eval; print('ok')"
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and "ok" in p.stdout, p.stderr


def test_iou_primitive():
    a = np.zeros((4, 4), bool); a[:2] = True
    b = np.zeros((4, 4), bool); b[:, :2] = True
    assert ev.iou(a, b, np.ones((4, 4), bool)) == pytest.approx(2 / 6)


def test_iou_ignores_border_pixels():
    a = np.zeros((4, 4), bool); a[0, 0] = a[0, 1] = True
    b = np.zeros((4, 4), bool); b[0, 0] = True
    valid = np.ones((4, 4), bool); valid[0, 1] = False
    assert ev.iou(a, b, valid) == 1.0


def test_box_iou():
    assert ev.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)


def test_perfect_job_scores_perfectly(world):
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    for kf, fi in (("kf_00000", 0), ("kf_00004", 4)):
        save_mask(world["job"], kf, "liver", gt_mask(fi) == LIVER)
    g = np.zeros((H, W), bool); g[0:10, 40:50] = True
    save_mask(world["job"], "kf_00000", "grasper", g)
    r = score(world)
    p = r["presence_final"]
    assert p["accuracy"] == 1.0 and p["n"] == 12 and p["coverage"] == 1.0
    assert r["masks_keyframe"]["iou_with_errors"]["mean"] == 1.0
    assert r["masks_keyframe"]["iou_with_errors"]["n"] == 3
    assert r["masks_keyframe"]["per_label_when_both_present"]["liver"]["n"] == 2


def test_known_iou_value(world):
    """A liver mask covering half the true liver has IoU exactly 0.5 and is not rounded up to anything."""
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    save_mask(world["job"], "kf_00000", "liver", half_liver_mask())
    save_mask(world["job"], "kf_00004", "liver", half_liver_mask())
    g = np.zeros((H, W), bool); g[0:10, 40:50] = True
    save_mask(world["job"], "kf_00000", "grasper", g)
    r = score(world)
    assert r["masks_keyframe"]["per_label_when_both_present"]["liver"]["mean"] == 0.5
    assert r["masks_keyframe"]["per_label_when_both_present"]["grasper"]["mean"] == 1.0


def test_missing_mask_file_scores_zero_and_is_reported(world):
    """A pipeline must not gain IoU by failing to write its worst masks."""
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    save_mask(world["job"], "kf_00000", "liver", gt_mask(0) == LIVER)
    m = score(world)["masks_keyframe"]
    assert m["n_missing_mask_files"] == 2
    assert m["iou_when_both_present"]["n"] == 3
    assert m["iou_when_both_present"]["mean"] == pytest.approx(1 / 3, abs=1e-3)


def test_false_positive_and_negative_count_as_zero_only_in_headline(world):
    over = {("kf_00004", "grasper"): decision("kf_00004", "grasper", "accept", True),   # FP: no grasper on frame 4
            ("kf_00000", "grasper"): decision("kf_00000", "grasper", "reject", False)}  # FN
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions(over))
    save_mask(world["job"], "kf_00000", "liver", gt_mask(0) == LIVER)
    save_mask(world["job"], "kf_00004", "liver", gt_mask(4) == LIVER)
    g = np.zeros((H, W), bool); g[0:10, 40:50] = True
    save_mask(world["job"], "kf_00004", "grasper", g)
    r = score(world)
    m = r["masks_keyframe"]
    assert m["iou_when_both_present"]["mean"] == 1.0 and m["iou_when_both_present"]["n"] == 2
    assert m["iou_with_errors"]["mean"] == 0.5 and m["iou_with_errors"]["n"] == 4
    assert m["n_false_positive_objects"] == 1 and m["n_false_negative_objects"] == 1
    assert r["presence_final"]["accuracy"] == pytest.approx(10 / 12, abs=1e-3)


def test_amber_reduces_coverage_not_accuracy(world):
    over = {("kf_00000", "grasper"): decision("kf_00000", "grasper", "AMBER", None)}
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions(over))
    p = score(world)["presence_final"]
    assert p["n"] == 11 and p["n_amber"] == 1 and p["accuracy"] == 1.0
    assert p["coverage"] == pytest.approx(11 / 12, abs=1e-3)
    assert p["n_missing_rows"] == 0


def test_missing_rows_reduce_coverage(world):
    """Dropping hard labels must be visible: they cost coverage."""
    rows = [r for r in all_decisions() if r["label"] != "grasper"]
    write_jsonl(world["job"] / "decisions.jsonl", rows)
    p = score(world)["presence_final"]
    assert p["n"] == 10 and p["n_missing_rows"] == 2 and p["coverage"] == pytest.approx(10 / 12, abs=1e-3)


def test_majority_baseline_reported(world):
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    p = score(world)["presence_final"]
    assert p["majority_baseline"] == pytest.approx(9 / 12, abs=1e-3)  # 3 present, 9 absent


def test_empty_job_gives_no_number_not_zero_or_one(world):
    r = score(world)
    assert r["presence_final"]["accuracy"] is None and r["presence_final"]["n"] == 0
    assert r["masks_keyframe"]["iou_with_errors"]["mean"] is None


def test_refuses_when_keyframes_are_off_by_one(world):
    """The pipeline claims frame 4, the jpg is really frame 3."""
    Image.fromarray(textured(3)).save(world["job"] / "frames" / "kf_00004.jpg", quality=98)
    Image.fromarray(textured(1)).save(world["job"] / "frames" / "kf_00000.jpg", quality=98)
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    with pytest.raises(ev.Misaligned):
        score(world)
    r = score(world, allow_misaligned=True)
    assert r["alignment"]["ok"] is False and r["alignment"]["n_bad"] == 2


def test_refuses_a_different_video(world):
    for k in ("kf_00000", "kf_00004"):
        Image.fromarray(textured(99)).save(world["job"] / "frames" / f"{k}.jpg", quality=98)
    with pytest.raises(ev.Misaligned):
        score(world)


def test_unregistered_video_has_no_ground_truth(world):
    (world["job"] / "source.mp4").write_bytes(b"not registered")
    with pytest.raises(ev.NoGroundTruth):
        ev.score_job(world["job"], world["raw"], world["tmp"] / "empty_registry")


def test_registered_video_found_by_content_hash_under_any_name(world):
    src = world["job"] / "whatever_the_upload_was_called.mov"
    src.write_bytes(b"some video bytes")
    reg = world["tmp"] / "registry"
    reg.mkdir()
    (reg / f"{ev._sha256(src)}.json").write_text(world["manifest"].read_text())
    assert ev.score_job(world["job"], world["raw"], reg)["n_keyframes"] == 2


def test_source_video_declared_in_meta_outside_job(world):
    src = world["tmp"] / "elsewhere.mp4"
    src.write_bytes(b"cli run on a path")
    meta = json.loads((world["job"] / "meta.json").read_text())
    meta["source_video"] = str(src)
    (world["job"] / "meta.json").write_text(json.dumps(meta))
    reg = world["tmp"] / "registry"
    reg.mkdir()
    (reg / f"{ev._sha256(src)}.json").write_text(world["manifest"].read_text())
    assert ev.score_job(world["job"], world["raw"], reg)["n_keyframes"] == 2


def test_annotated_output_is_not_mistaken_for_source(world):
    (world["job"] / "annotated.mp4").write_bytes(b"x")
    reg = world["tmp"] / "registry"
    reg.mkdir()
    # A registry hit on annotated.mp4's hash must not count; n_frames differs so content matching can't rescue it.
    wrong = json.loads(world["manifest"].read_text())
    wrong["n_frames"] = 999
    (reg / f"{ev._sha256(world['job'] / 'annotated.mp4')}.json").write_text(json.dumps(wrong))
    with pytest.raises(ev.NoGroundTruth):
        ev.score_job(world["job"], world["raw"], reg)


def test_reencoded_source_matched_by_content_only_if_frames_really_match(world):
    (world["job"] / "source.mp4").write_bytes(b"re-encoded, so the hash is unknown")
    reg = world["tmp"] / "registry"
    reg.mkdir()
    (reg / "someotherhash.json").write_text(world["manifest"].read_text())
    assert ev.score_job(world["job"], world["raw"], reg)["n_keyframes"] == 2
    # same frame count, different pictures: not that video, no score
    for k in ("kf_00000", "kf_00004"):
        Image.fromarray(textured(99)).save(world["job"] / "frames" / f"{k}.jpg", quality=98)
    with pytest.raises(ev.NoGroundTruth):
        ev.score_job(world["job"], world["raw"], reg)


def test_non_held_out_score_is_stamped(world):
    m = json.loads(world["manifest"].read_text())
    m["split"] = "calib"
    world["manifest"].write_text(json.dumps(m))
    r = score(world)
    assert r["split"] == "calib" and "not a held-out result" in ev.summary_lines(r)[0]


def test_mock_outputs_are_refused_and_stamped(world):
    write_jsonl(world["job"] / "detect.jsonl", [{"kf": "kf_00000", "label": "liver", "present": True, "conf": 1, "polygon": None, "model": "mock:claude-sonnet-5-5"}])
    with pytest.raises(ev.MockRun):
        score(world)
    r = score(world, allow_mock=True)
    assert r["mock_run"] is True and "MOCK RUN" in ev.summary_lines(r)[0]


def test_tracks_split_keyframe_and_propagated(world):
    job = world["job"]
    write_jsonl(job / "decisions.jsonl", all_decisions())
    save_mask(job, "kf_00000", "liver", gt_mask(0) == LIVER)     # keyframe mask lives in masks/
    (job / "tracks").mkdir()
    half = half_liver_mask()
    Image.fromarray(half.astype(np.uint8) * 255).save(job / "tracks" / "1_liver.png")
    rows = [
        {"frame_index": 0, "label": "liver", "box": [0, 0, W / 2, H], "propagated": False},
        {"frame_index": 1, "label": "liver", "box": [0, 0, W / 4, H], "propagated": True},
        {"frame_index": 2, "label": "liver", "box": [0, 0, W / 2, H], "propagated": True},  # no png
    ]
    write_jsonl(job / "tracks.jsonl", rows)
    t = score(world)["all_frames"]
    assert t["mask_iou_when_both_present"]["keyframe"]["mean"] == 1.0
    assert t["mask_iou_when_both_present"]["propagated"]["mean"] == 0.5
    assert t["n_tracks_without_mask_png"] == 1
    assert t["box_iou_when_both_present"]["propagated"]["n"] == 2
    # untracked frames count as "absent", so frames with no track on a present object are misses
    assert t["presence"]["propagated"]["fn"] > 0


def test_coco_scored_from_its_own_polygons(world):
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    coco = {
        "images": [{"id": 1, "frame_index": 0}],
        "categories": [{"id": 1, "name": "liver"}, {"id": 2, "name": "grasper"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[0, 0, W / 2 - 1, 0, W / 2 - 1, H - 1, 0, H - 1]],
             "attributes": {"status": "accept"}},
            {"id": 2, "image_id": 1, "category_id": 2, "segmentation": [[40, 0, 49, 0, 49, 9, 40, 9]],
             "attributes": {"status": "AMBER"}},
        ],
    }
    (world["job"] / "dataset_coco.json").write_text(json.dumps(coco))
    c = score(world)["dataset_coco"]
    assert c["scored"] and c["n_amber_annotations"] == 1
    assert c["iou_when_both_present"]["n"] == 2
    assert c["iou_when_both_present"]["mean"] > 0.95


def test_coco_without_frame_index_is_unscored_not_guessed(world):
    (world["job"] / "dataset_coco.json").write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}], "annotations": [], "categories": []}))
    c = score(world)["dataset_coco"]
    assert c["scored"] is False


def test_cli_exit_codes_and_report(world):
    write_jsonl(world["job"] / "decisions.jsonl", all_decisions())
    cmd = [sys.executable, "-m", "surgproof.eval", str(world["job"]), "--raw", str(world["raw"]), "--manifest", str(world["manifest"])]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "presence accuracy" in out.stdout and "n=12" in out.stdout
    assert json.loads((world["job"] / "eval_report.json").read_text())["presence_final"]["n"] == 12
    # scorer must not modify pipeline outputs
    assert sorted(p.name for p in world["job"].iterdir()) == sorted(["frames", "masks", "meta.json", "decisions.jsonl", "eval_report.json"])
    Image.fromarray(textured(99)).save(world["job"] / "frames" / "kf_00000.jpg", quality=98)
    Image.fromarray(textured(98)).save(world["job"] / "frames" / "kf_00004.jpg", quality=98)
    assert subprocess.run(cmd, cwd=ROOT, capture_output=True).returncode == 3
    no_gt = [sys.executable, "-m", "surgproof.eval", str(world["job"]), "--raw", str(world["raw"]), "--registry", str(world["tmp"] / "none")]
    assert subprocess.run(no_gt, cwd=ROOT, capture_output=True).returncode == 2


# ---------- video-level aggregate ----------

def _block(tp, fp, fn, tn, label="liver"):
    block = ev.confusion([(True, True)] * tp + [(True, False)] * fp + [(False, True)] * fn + [(False, False)] * tn)
    block.update(per_label={label: dict(block)}, n_expected=block["n"], n_amber=0)
    return block


def _report(clip, job, tp, fp, fn, tn, split="test", label="liver", mock=False, aligned=True):
    b = _block(tp, fp, fn, tn, label)
    return {"job": str(job), "split": split, "clips": [clip], "n_keyframes": 1, "models": ["m"], "mock_run": mock,
            "alignment": {"ok": aligned}, "presence_detect_raw": b, "presence_verify_raw": b, "presence_final": b,
            "masks_keyframe": {"iou_with_errors": {"n": 1, "mean": 0.5}}}


def test_aggregate_weights_videos_not_frames(tmp_path):
    """A huge video must not drown a small one, and two jobs from one video count as one video."""
    reports = [_report("video09_00000", tmp_path, 100, 0, 0, 0), _report("video09_00080", tmp_path, 100, 0, 0, 0),
               _report("video17_00000", tmp_path, 0, 0, 2, 0)]
    r = ev.aggregate(reports)
    assert r["n_videos"] == 2 and r["n_jobs"] == 3
    final = r["presence_final"]
    assert final["video_mean_accuracy"] == 0.5
    assert final["pooled"]["n"] == 202 and final["pooled"]["accuracy"] == round(200 / 202, 4)


def test_video_ci_by_hand():
    assert ev.video_ci([0.5]) is None
    # mean 0.5, sd 0.1, df 2: half width 4.303 * 0.1 / sqrt(3) = 0.24843
    assert ev.video_ci([0.4, 0.5, 0.6]) == [0.2516, 0.7484]
    assert ev.video_ci([0.0, 1.0]) == [0.0, 1.0]  # clipped, never outside 0..1


def test_sensitivity_with_no_positives_is_not_a_number(tmp_path):
    """No blood in the split: sensitivity is unknown, never 0 and never 1."""
    r = ev.aggregate([_report("video09_00000", tmp_path, 0, 0, 0, 5, label="blood")])
    blood = r["presence_final"]["per_label"]["blood"]
    assert blood["sensitivity"]["video_mean"] is None and blood["sensitivity"]["n"] == 0
    assert blood["specificity"]["video_mean"] == 1.0 and blood["specificity"]["ci95_over_videos"] is None


def test_single_positive_video_gets_no_interval(tmp_path):
    reports = [_report("video01_00000", tmp_path, 8, 0, 2, 0, split="train", label="blood"),
               _report("video18_00000", tmp_path, 0, 0, 0, 9, split="train", label="blood")]
    sens = ev.aggregate(reports)["presence_final"]["per_label"]["blood"]["sensitivity"]
    assert sens["n_videos"] == 1 and sens["video_mean"] == 0.8 and sens["ci95_over_videos"] is None


@pytest.mark.parametrize("clip", sorted(ev.SPENT_CLIPS))
def test_aggregate_refuses_spent_clips(tmp_path, clip):
    with pytest.raises(ValueError, match="spent"):
        ev.aggregate([_report(clip, tmp_path, 1, 0, 0, 1)])


def test_aggregate_refuses_mock_misaligned_duplicates_and_mixed_splits(tmp_path):
    with pytest.raises(ev.MockRun):
        ev.aggregate([_report("video09_00000", tmp_path, 1, 0, 0, 1, mock=True)])
    with pytest.raises(ev.Misaligned):
        ev.aggregate([_report("video09_00000", tmp_path, 1, 0, 0, 1, aligned=False)])
    with pytest.raises(ValueError, match="twice"):
        ev.aggregate([_report("video09_00000", tmp_path, 1, 0, 0, 1), _report("video09_00000", tmp_path, 1, 0, 0, 1)])
    with pytest.raises(ValueError, match="pool"):
        ev.aggregate([_report("video09_00000", tmp_path, 1, 0, 0, 1),
                      _report("video01_00000", tmp_path, 1, 0, 0, 1, split="train")])


def test_phase_accuracy_is_never_reported(tmp_path):
    """There is no phase ground truth. Agreement between passes must not be dressed up as accuracy."""
    (tmp_path / "phase.jsonl").write_text(json.dumps({"kf": "k", "phase": "Unknown", "agree": True}) + "\n")
    phase = ev.aggregate([_report("video09_00000", tmp_path, 1, 0, 0, 1)])["phase"]
    assert phase["accuracy"] is None and "no phase labels" in phase["why_not_scored"]
    assert phase["n"] == 1 and phase["agree"] == 1 and phase["unknown"] == 1


def test_coverage_sits_beside_accuracy(tmp_path):
    """A perfect score on the few labels it dared to decide must not read as a perfect score."""
    r = _report("video09_00000", tmp_path, 1, 0, 0, 0)
    r["presence_final"]["n_expected"] = 4
    r["presence_final"]["n_amber"] = 3
    final = ev.aggregate([r])["presence_final"]
    assert final["pooled"]["accuracy"] == 1.0 and final["coverage"] == 0.25 and final["n_amber"] == 3


def test_routes_are_scored_apart_including_amber_suggestions():
    class Truth:
        def present(self, frame, label):
            return label == "liver"

    rows = [{"kf": "a", "label": "liver", "status": "accept", "final_present": True, "route": "both_passes_present"},
            {"kf": "a", "label": "blood", "status": "accept", "final_present": True, "route": "amber_said_present"},
            {"kf": "a", "label": "grasper", "status": "AMBER", "final_present": None, "route": "amber_said_absent", "amber_suggests": "absent"},
            {"kf": "b", "label": "liver", "status": "AMBER", "final_present": None, "route": "amber_said_absent", "amber_suggests": "absent"},
            {"kf": "b", "label": "blood", "status": "AMBER", "final_present": None, "route": "amber_unresolved"}]
    out = ev.score_routes(rows, {"a": 0, "b": 1}, Truth())
    assert out["both_passes_present"]["tp"] == 1
    assert out["amber_said_present"]["fp"] == 1 and out["amber_said_present"]["precision"] == 0.0
    assert out["amber_said_absent"]["tn"] == 1 and out["amber_said_absent"]["fn"] == 1
    assert out["amber_said_absent"]["status_counts"]["AMBER"] == 2
    assert out["amber_unresolved"]["n"] == 0 and out["amber_unresolved"]["status_counts"]["AMBER"] == 1


def test_passes_are_compared_on_the_labels_final_decided():
    """The hard label went amber, so it must leave every pass's score, not only the final one."""
    class Truth:
        def present(self, frame, label):
            return label == "liver"

    dec = [{"kf": "a", "label": "liver", "status": "accept", "final_present": True},
           {"kf": "a", "label": "blood", "status": "AMBER", "final_present": None}]
    detect = [{"kf": "a", "label": "liver", "present": True}, {"kf": "a", "label": "blood", "present": True}]
    verify = [{"kf": "a", "label": "liver", "present": False}, {"kf": "a", "label": "blood", "present": True}]
    out = ev.score_on_same_labels(dec, detect, verify, {"a": 0}, Truth())
    assert out["final"]["n"] == 1 and out["detect"]["n"] == 1 and out["verify"]["n"] == 1
    assert out["final"]["accuracy"] == 1.0 and out["detect"]["accuracy"] == 1.0 and out["verify"]["accuracy"] == 0.0
