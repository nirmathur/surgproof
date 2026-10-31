"""Tests for app.py (SurgProof UI, Human Gate #14, Live Mode #15, placeholders, real-artifact wiring)."""
import asyncio
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
    box = app.parse_box("[800, 900, 100, -50]")
    assert box == [100.0, 0.0, 800.0, 900.0]


def test_join_gt_and_score_jsonl_ignores_unscorable_minus_one(tmp_path):
    idx_df = pd.DataFrame([
        {
            "frame_id": "ext_01",
            "video": "ev1",
            "split": "test",
            "gt_gallbladder": 1,
            "gt_cystic_duct": 0,
            "gt_grasper": -1,
            "gt_l_hook": -1,
            "gt_liver": -1,
            "gt_blood": -1,
        }
    ])
    jsonl_file = tmp_path / "labels_ext.jsonl"
    recs = [
        {"frame_id": "ext_01", "video": "ev1", "split": "test", "key": "gallbladder", "present": 1, "conf": 0.9},
        {"frame_id": "ext_01", "video": "ev1", "split": "test", "key": "cystic_duct", "present": 0, "conf": 0.8},
        {"frame_id": "ext_01", "video": "ev1", "split": "test", "key": "grasper", "present": 1, "conf": 0.9},
    ]
    jsonl_file.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    stats = app.score_jsonl_against_index(jsonl_file, idx_df, split_filter="test")
    assert stats is not None
    assert stats["n_rows"] == 2  # grasper has gt == -1 so it is excluded from scoring
    assert stats["acc_5_nontrivial"] == 1.0


def test_draw_handles_missing_image_and_boxes(tmp_path):
    rows = pd.DataFrame([
        {"key": "grasper", "final_present": 1, "status": "accept", "box": "[100, 150, 400, 500]", "final_correct": 1},
        {"key": "blood", "final_present": 1, "status": "AMBER", "box": "[700, 600, 300, 200]", "final_correct": 0},
        {"key": "liver", "final_present": 0, "status": "accept", "box": "[]", "final_correct": 1},
    ])
    im_missing = app.draw(tmp_path / "nonexistent.png", rows, reveal=False, frame_id="video01_00080_80")
    assert isinstance(im_missing, Image.Image)
    assert im_missing.size == (640, 480)

    real_img_path = tmp_path / "frame.png"
    Image.new("RGB", (320, 240), color=(40, 50, 60)).save(real_img_path)
    im_real = app.draw(real_img_path, rows, reveal=True, frame_id="video01_00080_80")
    assert isinstance(im_real, Image.Image)
    assert im_real.size == (320, 240)


def test_human_gate_locks_export_until_all_amber_cleared(tmp_path):
    log_file = tmp_path / "human_gate_log.jsonl"
    dec_df = pd.DataFrame([
        {"frame_id": "f1", "video": "video12", "key": "grasper", "status": "accept", "final_present": 1, "final_by": "flash"},
        {"frame_id": "f1", "video": "video12", "key": "cystic_duct", "status": "AMBER", "final_present": 1, "final_by": "argon"},
        {"frame_id": "f1", "video": "video12", "key": "blood", "status": "reject", "final_present": 0, "final_by": "flash"},
    ])
    pending, cleared, summary = app.compute_amber_queue(dec_df, app.load_human_gate_log(log_file))
    assert summary["n_accept"] == 1
    assert summary["n_reject"] == 1
    assert summary["n_amber_pending"] == 1
    assert summary["export_unlocked"] is False
    assert len(pending) == 1
    assert len(cleared) == 0

    # Clear the single AMBER item
    app.record_human_gate_decision(
        reviewer="Nir Mathur",
        frame_id="f1",
        video="video12",
        key="cystic_duct",
        agent_status="AMBER",
        agent_present=1,
        human_action="approve_absent",
        human_present=0,
        log_path=log_file,
    )
    gate_log = app.load_human_gate_log(log_file)
    pending2, cleared2, summary2 = app.compute_amber_queue(dec_df, gate_log)
    assert summary2["n_amber_pending"] == 0
    assert summary2["n_amber_cleared"] == 1
    assert summary2["export_unlocked"] is True
    assert len(pending2) == 0
    assert len(cleared2) == 1

    exported = app.build_verified_export(dec_df, gate_log)
    assert exported["human_gate_cleared"] is True
    assert exported["n_verified_labels"] == 2
    by_key = {r["key"]: r for r in exported["labels"]}
    assert by_key["grasper"]["present"] == 1
    assert by_key["cystic_duct"]["present"] == 0
    assert by_key["cystic_duct"]["reviewer"] == "Nir Mathur"


def test_live_mode_pipeline_never_touches_ground_truth(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()
    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(app, "O", out_dir)

    f1 = tmp_path / "v12_1.png"
    f2 = tmp_path / "v12_2.png"
    Image.new("RGB", (160, 120), color=(80, 30, 30)).save(f1)
    Image.new("RGB", (160, 120), color=(80, 40, 40)).save(f2)

    idx_df = pd.DataFrame([
        {
            "frame_id": "video12_00080_08",
            "video": "video12",
            "clip": "video12_00080",
            "path": str(f1),
            "split": "test",
            **{f"gt_{k}": (1 if k == "grasper" else 0) for k in C.LABELS},
        },
        {
            "frame_id": "video12_00080_16",
            "video": "video12",
            "clip": "video12_00080",
            "path": str(f2),
            "split": "test",
            **{f"gt_{k}": (1 if k == "liver" else 0) for k in C.LABELS},
        },
    ])
    idx_df.to_csv(data_dir / "index.csv", index=False)
    (out_dir / "manual.json").write_text(json.dumps({"procedure": "laparoscopic cholecystectomy", "structures": []}))

    async def fake_call(prompt, *, model=None, images=(), schema=None, tools=None, video_url=None, temperature=0.0, tag="", sample=0):
        name = getattr(schema, "__name__", None)
        if name == "FrameLabels":
            return {
                "items": [
                    {
                        "key": k,
                        "present": (k in ("grasper", "cystic_duct")),
                        "confidence": (0.95 if k != "cystic_duct" else 0.60),
                        "box_2d": ([100, 100, 400, 400] if k in ("grasper", "cystic_duct") else []),
                        "evidence": "visual cue",
                    }
                    for k in C.LABELS
                ]
            }
        if name == "CropID":
            return {"answer": C.LABELS["grasper"][1], "confidence": 0.90}
        if name == "Critique":
            return {
                "rebuttals": [
                    {"key": k, "argument": "check", "p_wrong": (0.05 if k != "cystic_duct" else 0.45)}
                    for k in C.LABELS
                ]
            }
        if name == "LiveVerdict":
            return {"present": False, "confidence": 0.65, "reasoning": "still ambiguous -> keep AMBER"}
        return {}

    monkeypatch.setattr(app.gem, "call", fake_call)
    monkeypatch.setattr(app.L.gem, "call", fake_call)
    monkeypatch.setattr(app.gem, "trace", lambda *a, **kw: None)

    live_df, meta = asyncio.run(app.run_live_video_async("video12", n_frames=2, argon_budget=2))
    assert len(live_df) == 12
    assert meta["n_frames"] == 2
    assert meta["under_3_min"] is True
    # Zero GT leakage check: live_df must NOT contain truth, correct, final_correct, or gt_*
    for forbidden in ["truth", "correct", "final_correct", *[f"gt_{k}" for k in C.LABELS]]:
        assert forbidden not in live_df.columns, f"Leakage column {forbidden} found in live verifier output!"

    # Reveal step joins ground truth from index.csv only when requested
    revealed_df = app.join_ground_truth_for_reveal(live_df, idx_df)
    assert "truth" in revealed_df.columns
    assert "final_correct" in revealed_df.columns


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
    assert "SurgProof" in at.title[0].value
    assert "No human labels used in training" in all_markdown
    assert "upper bounds" in all_markdown.lower()
    assert "SurgAtlas" in all_markdown
    assert "SurgVLP" in all_markdown
    assert "SurgSLOT" in all_markdown
    assert "Med-Gemini" in all_markdown

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

    (out_dir / "trace.jsonl").write_text(
        '{"t": 1.0, "agent": "argon", "msg": "Wrote cholecystectomy manual"}\n'
        'malformed-json-line\n'
        '{"t": 2.0, "agent": "flash", "msg": "Labelled 40 frames"}\n'
    )

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

    frame_img = tmp_path / "video12_00080_80.png"
    Image.new("RGB", (320, 240), color=(60, 30, 30)).save(frame_img)
    pd.DataFrame([
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "path": str(frame_img),
            **{f"gt_{k}": (1 if k == "grasper" else 0) for k in C.LABELS},
        }
    ]).to_csv(data_dir / "index.csv", index=False)

    # Leak-free decisions.csv (no truth column inside decisions.csv!)
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
            "pv_match": True,
            "pv_conf": 0.92,
            "p_wrong": 0.05,
            "critic": "clearly grasper",
            "trust": 0.90,
            "status": "accept",
            "final_present": 1,
            "final_by": "flash",
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
            "pv_match": False,
            "pv_conf": 0.40,
            "p_wrong": 0.65,
            "critic": "not yet skeletonized",
            "trust": 0.19,
            "status": "AMBER",
            "final_present": 0,
            "final_by": "argon",
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
