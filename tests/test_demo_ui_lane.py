"""UI lane (harald/demo): GT overlay, tracker loop, narration claims, student-vs-zero-shot table."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
import config as C

APP = str(Path(__file__).resolve().parents[1] / "app.py")
REAL_CLIP = app.CHOLEC_ROOT / "video12" / "video12_15750"


def test_watershed_map_ignores_255_and_maps_values():
    m = np.array([[50, 31, 255], [32, 22, 7]], dtype=np.uint8)
    cls = app.watershed_to_class_map(m)
    assert cls.tolist() == [[0, 5, -1], [9, 10, -1]]  # 31 -> grasper(5), 32 -> l_hook(9), 22 -> gallbladder(10)


def test_gt_overlay_tints_only_labelled_pixels(tmp_path):
    img = Image.new("RGB", (4, 2), (0, 0, 0))
    mask = np.zeros((2, 4, 3), dtype=np.uint8)
    mask[..., 0] = [[31, 255, 50, 50], [50, 50, 50, 50]]
    mp = tmp_path / "m.png"
    Image.fromarray(mask).save(mp)
    out = np.array(app.gt_overlay(img, mp))
    assert out[0, 0].sum() > 0          # grasper pixel tinted
    assert out[0, 1].sum() == 0         # 255 border ignored
    assert out[0, 2].sum() == 0         # background untouched


def test_narration_claims_shapes():
    new = {"frame_id": "video12_15750_15758", "claims": [
        {"text": "grasper retracts gallbladder", "status": "verified", "evidence_frames": ["video12_15750_15758"]},
        {"text": "cystic duct clipped", "status": "contradicted"},
        {"text": "bleeding", "status": "weird"},
    ]}
    c = app.narration_claims(new)
    assert [x["status"] for x in c] == ["verified", "contradicted", "unsupported"]
    legacy = {"frame_id": "f", "narration": "The hook dissects.", "unsure_about": ["blood"]}
    c2 = app.narration_claims(legacy)
    assert c2[0]["status"] == "unsupported" and c2[1]["text"].startswith("blood")
    assert app.narration_claims({"frame_id": "f", "narration": "ok", "unsure_about": []})[0]["status"] == "verified"


def test_frame_path_from_id():
    assert app.frame_path_from_id("bad") is None
    if REAL_CLIP.exists():
        assert app.frame_path_from_id("video12_15750_15750") is not None


def _run(tmp_out: Path | None = None) -> AppTest:
    old = C.OUT_DIR
    if tmp_out is not None:
        C.OUT_DIR = tmp_out
    try:
        at = AppTest.from_file(APP, default_timeout=120).run()
    finally:
        C.OUT_DIR = old
    return at


@pytest.mark.skipif(not (C.OUT_DIR / "flags.parquet").exists() or not REAL_CLIP.exists(), reason="needs tracker output + frames")
def test_loop_tab_real_frames_and_gt_toggle():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    tog = [t for t in at.toggle if t.key == "loop_gt"]
    assert tog
    tog[0].set_value(True).run()
    assert not at.exception


def test_student_and_narration_with_fixtures(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    real = C.OUT_DIR
    for name in ("flags.parquet", "eval", "tracker"):
        if (real / name).exists():
            os.symlink(real / name, out / name)
    (out / "results.json").write_text(json.dumps({
        "manual_raw_acc": 0.8, "blank_raw_acc": 0.7,
        "accepted": {"coverage": 0.6, "accuracy": 0.95},
        "tracker_recheck": {"coverage": 0.55, "accuracy": 0.97},
    }))
    (out / "student.json").write_text(json.dumps({
        "mean_acc_by_source": {"single": 0.8, "verified": 0.86, "oracle_upper_bound": 0.9}}))
    (out / "narration_video12.json").write_text(json.dumps([
        {"frame_id": "video12_15750_15750", "claims": [
            {"text": "A grasper holds the gallbladder.", "status": "verified", "evidence_frames": ["video12_15750_15750"]},
            {"text": "The cystic duct is clipped.", "status": "contradicted", "evidence_frames": []}]}]))
    at = _run(out)
    assert not at.exception, [e.value for e in at.exception]
    md = " ".join(m.value for m in at.markdown) + " ".join(s.value for s in at.success)
    assert "A grasper holds the gallbladder." in md
    finals = [s.value for s in at.success if "grasper" in s.value]
    assert finals and "clipped" not in finals[0]  # final narration = verified claims only
    labels = [m.label for m in at.metric]
    assert any("tracker-flagged recheck" in lbl for lbl in labels)


def test_load_label_rows_prefers_decisions_then_labels(tmp_path):
    old = C.OUT_DIR
    C.OUT_DIR = tmp_path
    app.O = tmp_path
    try:
        (tmp_path / "labels_manual.jsonl").write_text(json.dumps(
            {"frame_id": "video12_15750_15750", "video": "video12", "key": "grasper", "present": 1, "sample": 0}) + "\n")
        d, src = app.load_label_rows()
        assert src == "labels_manual.jsonl" and len(d) == 1
        (tmp_path / "decisions.csv").write_text("frame_id,video,key,present,status\nvideo12_15750_15750,video12,grasper,1,accept\n")
        d, src = app.load_label_rows()
        assert src == "decisions.csv" and "status" in d.columns
    finally:
        C.OUT_DIR = old
        app.O = old


def test_narrator_verdict_mapping():
    from types import SimpleNamespace as NS
    assert app.narrator_claim_status(NS(verdict="speak")) == "verified"
    assert app.narrator_claim_status(NS(verdict="drop", L3="fail")) == "contradicted"
    assert app.narrator_claim_status(NS(verdict="drop", L3="abstain")) == "unsupported"
    assert app.narrator_claim_status(NS(verdict="hedge", L1="fail")) == "unsupported"
    assert app._json_list('["a", "b"]') == ["a", "b"] and app._json_list("bad") == []


def test_loop_shows_amber_rounds_and_smoke_warning(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "results.json").write_text(json.dumps({
        "n_test_labels": 90, "manual_raw_acc": 0.9, "accepted": {"coverage": 0.9, "accuracy": 0.98},
        "accepted_5_informative": {"coverage": 0.9, "accuracy": 0.97, "n_test_labels_excl_liver": 75},
        "amber_loop": {"max_rounds": 3, "ambers_by_round_test": [15, 7, 4, 2],
                       "per_round_test": [{"round": i, "accepted_count": 70 + i, "rejected_count": 5} for i in range(4)]}}))
    at = _run(out)
    assert not at.exception, [e.value for e in at.exception]
    vals = [m.value for m in at.metric]
    assert "15 AMBER" in vals and "2 AMBER" in vals
    assert any("Interim smoke run" in w.value for w in at.warning)
