"""Unit tests for label.py (nir-label lane) covering schema contracts and edge cases."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import config as C
import label


REQUIRED_DESIGN_FIELDS = {
    "frame_id",
    "video",
    "video_id",
    "split",
    "key",
    "mode",
    "sample",
    "model",
    "prompt_hash",
    "present",
    "conf",
    "confidence",
    "box",
    "evidence",
    "rationale",
    "pv_match",
    "pv_conf",
    "p_wrong",
    "critic",
}


def test_normalize_key():
    assert label.normalize_key("grasper") == "grasper"
    assert label.normalize_key("Grasper (surgical instrument)") == "grasper"
    assert label.normalize_key("L-hook electrocautery (instrument)") == "l_hook"
    assert label.normalize_key("l-hook") == "l_hook"
    assert label.normalize_key("cystic duct") == "cystic_duct"
    assert label.normalize_key("unknown_tool") is None
    assert label.normalize_key(None) is None


def test_sanitize_box():
    assert label.sanitize_box([100, 200, 400, 500], True) == [100, 200, 400, 500]
    # Inverted coordinates should be sorted and clamped
    assert label.sanitize_box([600, 700, 100, 200], True) == [100, 200, 600, 700]
    assert label.sanitize_box([-50, -10, 1200, 1050], True) == [0, 0, 1000, 1000]
    # Degenerate zero-area boxes should be rejected
    assert label.sanitize_box([0, 0, 0, 0], True) == []
    assert label.sanitize_box([100, 100, 102, 500], True) == []
    # Absent items always have empty box
    assert label.sanitize_box([100, 200, 400, 500], False) == []


def test_clamp_prob():
    assert label.clamp_prob(0.85) == 0.85
    assert label.clamp_prob(85.0) == 0.85
    assert label.clamp_prob(-0.2) == 0.0
    assert label.clamp_prob(float("nan"), default=0.5) == 0.5
    assert label.clamp_prob(None, default=0.25) == 0.25


def test_matches_crop_answer():
    assert label.matches_crop_answer("grasper (surgical instrument)", "grasper") is True
    assert label.matches_crop_answer("Grasper", "grasper") is True
    assert label.matches_crop_answer("L-hook electrocautery (instrument)", "l_hook") is True
    assert label.matches_crop_answer("L-hook", "l_hook") is True
    assert label.matches_crop_answer("none / unclear", "grasper") is False
    assert label.matches_crop_answer("liver", "gallbladder") is False


def test_select_frames_limit():
    rows = []
    for v_idx in range(1, 18):
        v = f"video{v_idx:02d}"
        sp = "test" if v_idx <= 6 else "calib" if v_idx <= 9 else "train"
        for f_idx in range(5):
            rows.append({"frame_id": f"{v}_{f_idx}", "video": v, "split": sp})
    df = pd.DataFrame(rows)
    sub = label.select_frames(df, ["train", "calib", "test"], 40)
    assert len(sub) == 40
    assert sub.video.nunique() == 17
    assert set(sub.split.unique()) == {"train", "calib", "test"}


def test_do_frame_emits_design_contract(monkeypatch):
    async def fake_call(prompt, *, model=None, images=(), schema=None, tools=None, video_url=None,
                        temperature=0.0, tag="", sample=0):
        name = getattr(schema, "__name__", None)
        if name == "FrameLabels":
            return {
                "items": [
                    {"key": "Grasper (surgical instrument)", "present": True, "confidence": 0.92,
                     "box_2d": [100, 150, 400, 450], "evidence": "metallic jaws"},
                    {"key": "liver", "present": True, "confidence": 95.0,
                     "box_2d": [0, 0, 0, 0], "evidence": "reddish parenchyma"},
                ]
            }
        if name == "CropID":
            return {"answer": "grasper (surgical instrument)", "confidence": 0.88}
        if name == "Critique":
            return {
                "rebuttals": [
                    {"key": k, "argument": f"counter {k}", "p_wrong": 0.12}
                    for k in C.LABELS
                ]
            }
        raise AssertionError(f"Unexpected schema {name}")

    monkeypatch.setattr(label.gem, "call", fake_call)
    monkeypatch.setattr(label.gem, "img_part", lambda path, max_side=768, crop=None: ("part", b"bytes"))

    row = SimpleNamespace(
        frame_id="video01_00080_80",
        video="video01",
        split="test",
        path="/tmp/fake.png",
        **{f"gt_{k}": (1 if k in ("grasper", "liver") else 0) for k in C.LABELS},
    )
    recs = asyncio.run(label.do_frame(row, "manual", {"procedure": "chole"}, sample=0, full_checks=True))
    assert len(recs) == len(C.LABELS)
    by_key = {r["key"]: r for r in recs}
    for r in recs:
        assert REQUIRED_DESIGN_FIELDS.issubset(r.keys())
        assert "truth" not in r, "Leakage guard: truth must never be emitted in labels JSONL"
        assert not any(k.startswith("gt_") for k in r), "Leakage guard: gt_* must never be emitted in labels JSONL"
        assert r["video"] == "video01" and r["video_id"] == "video01"
        assert r["conf"] == r["confidence"]
        assert r["evidence"] == r["rationale"]
        assert len(r["prompt_hash"]) == 16

    # Grasper had valid box -> verify_box ran and matched
    assert by_key["grasper"]["present"] == 1
    assert by_key["grasper"]["box"] == [100, 150, 400, 450]
    assert by_key["grasper"]["pv_match"] is True
    assert by_key["grasper"]["pv_conf"] == 0.88
    assert by_key["grasper"]["p_wrong"] == 0.12

    # Liver had degenerate [0,0,0,0] box -> sanitized to [] and pv_match is None
    assert by_key["liver"]["present"] == 1
    assert by_key["liver"]["conf"] == 0.95
    assert by_key["liver"]["box"] == []
    assert by_key["liver"]["pv_match"] is None
    assert by_key["liver"]["pv_conf"] is None


def test_accept_fallback_no_gt_leakage_and_pv_match_types():
    import accept
    import numpy as np

    src = Path(accept.__file__).read_text(encoding="utf-8")
    assert "gt_" not in src, "accept.py must never reference gt_*"
    assert "truth" not in src, "accept.py must never reference truth"

    # Verify pv_match works across bool, numpy.bool_, float64 1.0/0.0/nan, and None
    for pos_val in (True, np.bool_(True), 1.0, 1, "true"):
        r = SimpleNamespace(present=1, conf=0.90, p_wrong=0.10, pv_match=pos_val)
        assert accept.trust(r) == pytest.approx(0.81)
        assert accept.status(accept.trust(r)) == "accept"

    for neg_val in (False, np.bool_(False), 0.0, 0, None, float("nan"), "false"):
        r = SimpleNamespace(present=1, conf=0.90, p_wrong=0.10, pv_match=neg_val)
        assert accept.trust(r) == pytest.approx(0.405)
        assert accept.status(accept.trust(r)) == "AMBER"

    r_rej = SimpleNamespace(present=1, conf=0.50, p_wrong=0.60, pv_match=False)
    assert accept.trust(r_rej) == pytest.approx(0.10)
    assert accept.status(accept.trust(r_rej)) == "reject"


def test_emitted_label_jsonls_zero_gt_and_valid_boxes():
    expected_counts = {
        "labels_manual.jsonl": 6060,
        "labels_blank.jsonl": 1680,
        "labels_endoscapes2023_manual.jsonl": 354,
        "labels_endoscapes2023_blank.jsonl": 354,
        "labels_cholect50_manual.jsonl": 720,
        "labels_cholect50_blank.jsonl": 720,
    }
    for fname, exp_n in expected_counts.items():
        p = C.OUT_DIR / fname
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == exp_n, f"{fname} expected {exp_n} rows, got {len(rows)}"
        for r in rows:
            assert "truth" not in r and not any(k.startswith("gt_") for k in r)
            assert isinstance(r.get("evidence"), str)
            assert "black frame" not in r["evidence"].lower()
            if r["present"] == 1:
                assert len(r["evidence"]) > 0
                box = r.get("box", [])
                assert isinstance(box, list) and len(box) == 4
                y0, x0, y1, x1 = box
                assert 0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000
            else:
                assert r.get("box") == []


