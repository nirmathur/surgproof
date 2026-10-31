"""Tests for app.py (Streamlit UI, placeholder mode, real-artifact wiring, honest claims, prior work)."""
import json
from pathlib import Path

import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
import config as C


def test_parse_box_edge_cases():
    assert app.parse_box(None) is None
    assert app.parse_box("") is None
    assert app.parse_box("[]") is None
    assert app.parse_box("not-json") is None
    assert app.parse_box([100, 200, 100, 400]) is None  # zero height
    # Inverted and out-of-bounds coordinates are safely clamped and ordered
    box = app.parse_box("[800, 900, 100, -50]")
    assert box == [100.0, 0.0, 800.0, 900.0]


def test_draw_handles_missing_image_and_boxes(tmp_path):
    rows = pd.DataFrame([
        {"key": "grasper", "final_present": 1, "status": "accept", "box": "[100, 150, 400, 500]", "final_correct": 1},
        {"key": "blood", "final_present": 1, "status": "escalate", "box": "[700, 600, 300, 200]", "final_correct": 0},
        {"key": "liver", "final_present": 0, "status": "accept", "box": "[]", "final_correct": 1},
    ])
    # Missing file falls back to placeholder canvas without crashing
    im_missing = app.draw(tmp_path / "nonexistent.png", rows, reveal=False, frame_id="video01_00080_80")
    assert isinstance(im_missing, Image.Image)
    assert im_missing.size == (640, 480)

    # Existing image file renders and draws boxes in both normal and reveal modes
    real_img_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 240), color=(40, 50, 60)).save(real_img_path)
    im_real = app.draw(real_img_path, rows, reveal=True, frame_id="video01_00080_80")
    assert isinstance(im_real, Image.Image)
    assert im_real.size == (320, 240)


def test_app_renders_placeholders_and_honest_claims_when_empty(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    coord_dir = tmp_path / "coord_artifacts"
    out_dir.mkdir()
    data_dir.mkdir()
    coord_dir.mkdir()

    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(app, "O", out_dir)
    monkeypatch.setattr(app, "COORD_ARTIFACTS", coord_dir)

    at = AppTest.from_file(str(Path(C.ROOT) / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    # Honest claims and prior work must be on screen
    assert "No human labels used in training" in all_markdown
    assert "upper bounds" in all_markdown.lower()
    assert "SurgVLP" in all_markdown
    assert "SurgSLOT" in all_markdown
    assert "Med-Gemini" in all_markdown

    # Clearly labelled placeholders and PENDING metrics (no fake numbers)
    metric_values = [m.value for m in at.metric]
    assert "PENDING" in metric_values
    warnings_text = "\n".join(w.value for w in at.warning)
    assert "PLACEHOLDER" in warnings_text


def test_app_wires_real_outputs_across_all_tabs(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    coord_dir = tmp_path / "coord_artifacts"
    out_dir.mkdir()
    data_dir.mkdir()
    coord_dir.mkdir()

    # 1. trace.jsonl (including a malformed line to verify resilience)
    (out_dir / "trace.jsonl").write_text(
        '{"t": 1.0, "agent": "argon", "msg": "Wrote cholecystectomy manual"}\n'
        'malformed-json-line\n'
        '{"t": 2.0, "agent": "flash", "msg": "Labelled 40 frames"}\n'
    )

    # 2. manual.json
    manual = {
        "procedure": "laparoscopic cholecystectomy",
        "phases": ["Calot triangle dissection", "Clipping and cutting"],
        "structures": [
            {
                "key": k,
                "name": name,
                "appearance": "distinct endoscopic cue",
                "where_found": "hepatocystic triangle",
                "when_visible": "dissection",
                "confused_with": ["adjacent tissue"],
                "how_to_tell_apart": "colour and tubular continuity",
            }
            for k, (_, name) in C.LABELS.items()
        ],
        "critical_events": ["Critical View of Safety achieved"],
        "general_rules": ["Verify two structures entering gallbladder"],
    }
    (out_dir / "manual.json").write_text(json.dumps(manual))

    # 3. Act 1 atlas dataset
    atlas_dir = out_dir / "atlas" / "laparoscopic-appendectomy"
    atlas_dir.mkdir(parents=True)
    (atlas_dir / "manual.json").write_text(json.dumps(manual))
    (atlas_dir / "dataset.json").write_text(json.dumps({
        "procedure": "laparoscopic appendectomy",
        "source_video": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "created": "2026-10-31 12:30",
        "schema": {"structures": list(C.LABELS.keys()), "phases": ["Mesoappendix division"]},
        "timeline": {
            "phases": [{"name": "Mesoappendix division", "start": "00:10", "end": "02:00", "evidence": "hook cautery", "confidence": 0.91}],
            "events": [{"time": "01:15", "event": "Base ligated", "evidence": "endoloop", "confidence": 0.88}],
            "sightings": [{"time": "00:30", "key": "grasper", "confidence": 0.95}],
        },
        "audit": {"problems": ["Check base stump visibility"], "unverified_items": ["event@01:15"], "overall_confidence": 0.87},
        "human_labels_used": 0,
    }))

    # 4. Act 2 results.json & student.json
    (out_dir / "results.json").write_text(json.dumps({
        "n_test_labels": 360,
        "single_conf": {"curve": [[0.5, 0.90], [1.0, 0.84]], "coverage_at_95": 0.35, "acc_at_60": 0.88, "aurc": 0.08},
        "system_zero_label": {"curve": [[0.5, 0.97], [1.0, 0.87]], "coverage_at_95": 0.68, "acc_at_60": 0.96, "aurc": 0.04},
        "upper_bound_calibrated": {"curve": [[0.5, 0.98], [1.0, 0.88]], "coverage_at_95": 0.72, "acc_at_60": 0.97, "aurc": 0.03},
        "accepted": {"coverage": 0.64, "accuracy": 0.956, "escalated_share": 0.16},
        "blank_raw_acc": 0.77,
        "manual_raw_acc": 0.85,
        "per_label_accepted_acc": {"grasper": 0.96, "liver": 0.98},
    }))
    (out_dir / "student.json").write_text(json.dumps({
        "mean_acc_by_source": {"single": 0.82, "verified": 0.88, "oracle_upper_bound": 0.91},
        "per_label": {
            "grasper": {
                "single": {"n_train": 120, "acc": 0.83, "f1": 0.81, "auroc": 0.90},
                "verified": {"n_train": 80, "acc": 0.89, "f1": 0.88, "auroc": 0.94},
                "oracle_upper_bound": {"n_train": 120, "acc": 0.92, "f1": 0.91, "auroc": 0.96},
            }
        },
    }))

    # 5. Act 3 index.csv, decisions.csv, narration_video12.json
    frame_img = tmp_path / "video12_00080_80.png"
    Image.new("RGB", (320, 240), color=(60, 30, 30)).save(frame_img)
    pd.DataFrame([{"frame_id": "video12_00080_80", "video": "video12", "split": "test", "path": str(frame_img)}]).to_csv(
        data_dir / "index.csv", index=False
    )
    pd.DataFrame([
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "conf": 0.95,
            "box": "[100, 120, 400, 450]",
            "evidence": "grasper jaws visible",
            "truth": 1,
            "pv_match": True,
            "pv_conf": 0.92,
            "p_wrong": 0.05,
            "critic": "clearly grasper",
            "trust": 0.90,
            "status": "accept",
            "final_present": 1,
            "final_by": "flash",
            "final_correct": 1,
        },
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "key": "cystic_duct",
            "present": 1,
            "conf": 0.55,
            "box": "[300, 300, 500, 500]",
            "evidence": "possible duct",
            "truth": 0,
            "pv_match": False,
            "pv_conf": 0.40,
            "p_wrong": 0.65,
            "critic": "not yet skeletonized",
            "trust": 0.19,
            "status": "escalate",
            "final_present": 0,
            "final_by": "argon",
            "final_correct": 1,
        },
    ]).to_csv(out_dir / "decisions.csv", index=False)
    (out_dir / "narration_video12.json").write_text(json.dumps([
        {
            "frame_id": "video12_00080_80",
            "narration": "A grasper retracts tissue while the cystic duct remains unconfirmed.",
            "unsure_about": ["cystic_duct"],
        }
    ]))

    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(app, "O", out_dir)
    monkeypatch.setattr(app, "COORD_ARTIFACTS", coord_dir)

    at = AppTest.from_file(str(Path(C.ROOT) / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"

    metric_vals = [m.value for m in at.metric]
    assert "95.6%" in metric_vals
    assert "64%" in metric_vals
    assert "88.0%" in metric_vals

    all_md = "\n".join(m.value for m in at.markdown)
    assert "A grasper retracts tissue while the cystic duct remains unconfirmed." in all_md
    assert "cystic_duct" in all_md
