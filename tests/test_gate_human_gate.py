"""Tests for pages/2_Human_gate.py (Issue #14: Human Gate, lock logic, audit log, zero-GT guard, smoke verifier integration)."""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import config as C

GATE_PAGE_PATH = Path(C.ROOT) / "pages" / "2_Human_gate.py"


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("human_gate_mod", GATE_PAGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_human_gate_page_never_references_ground_truth():
    """Strict audit rule: pages/2_Human_gate.py must NEVER reference truth or gt_* in AST or strings."""
    assert GATE_PAGE_PATH.is_file(), "pages/2_Human_gate.py must exist"
    src = GATE_PAGE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(GATE_PAGE_PATH))
    forbidden_hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id == "truth" or node.id.startswith("gt_")):
            forbidden_hits.append(f"line {getattr(node, 'lineno', '?')}: Name({node.id})")
        elif isinstance(node, ast.Attribute) and (node.attr == "truth" or node.attr.startswith("gt_")):
            forbidden_hits.append(f"line {getattr(node, 'lineno', '?')}: Attr(.{node.attr})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if val == "truth" or val.startswith("gt_"):
                forbidden_hits.append(f"line {getattr(node, 'lineno', '?')}: Str({val!r})")
    assert not forbidden_hits, f"Ground-truth reference found in pages/2_Human_gate.py: {forbidden_hits}"


def test_sanitize_decisions_and_safe_index_strip_non_contract_columns(tmp_path):
    gate = _load_gate_module()
    idx_file = tmp_path / "index.csv"
    pd.DataFrame([
        {
            "frame_id": "video01_00080_80",
            "video": "video01",
            "split": "test",
            "path": "/tmp/f1.png",
            "extra_secret_col": 1,
        }
    ]).to_csv(idx_file, index=False)

    safe_idx = gate.load_safe_index(tmp_path)
    assert "extra_secret_col" not in safe_idx.columns
    assert set(safe_idx.columns).issubset(gate.SAFE_INDEX_COLS)

    raw_df = pd.DataFrame([
        {
            "frame_id": "video01_00080_80",
            "key": "grasper",
            "present": 1,
            "conf": 0.92,
            "evidence": "jaws visible",
            "pv_match": True,
            "p_wrong": 0.05,
            "unwanted_eval_col": 999,
        }
    ])
    clean_df = gate.sanitize_decisions_df(raw_df, safe_idx)
    assert "unwanted_eval_col" not in clean_df.columns
    assert clean_df.loc[0, "rationale"] == "jaws visible"
    assert clean_df.loc[0, "frame_path"] == "/tmp/f1.png"
    assert clean_df.loc[0, "status"] == "accept"


def test_lock_logic_blocks_export_until_all_amber_cleared_and_supports_undo(tmp_path):
    gate = _load_gate_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    log_path = out_dir / "gate_log.jsonl"
    export_path = out_dir / "verified_dataset.jsonl"

    # 1. Empty decisions -> locked
    _, _, empty_sum = gate.compute_amber_queue(pd.DataFrame(), [])
    assert empty_sum["export_unlocked"] is False
    with pytest.raises(PermissionError):
        gate.export_verified_dataset(pd.DataFrame(), [], out_path=export_path)
    assert not export_path.exists()

    # 2. Decisions with 2 accept, 1 reject, 2 AMBER -> locked while any AMBER is pending
    dec_df = pd.DataFrame([
        {
            "frame_id": "f1",
            "video": "video12",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "final_present": 1,
            "conf": 0.95,
            "trust": 0.90,
            "status": "accept",
            "final_by": "flash",
            "rationale": "fenestrated jaws",
        },
        {
            "frame_id": "f1",
            "video": "video12",
            "split": "test",
            "key": "gallbladder",
            "present": 1,
            "final_present": 1,
            "conf": 0.91,
            "trust": 0.85,
            "status": "accept",
            "final_by": "flash",
            "rationale": "fundus visible",
        },
        {
            "frame_id": "f1",
            "video": "video12",
            "split": "test",
            "key": "blood",
            "present": 1,
            "final_present": 0,
            "conf": 0.30,
            "trust": 0.15,
            "status": "reject",
            "final_by": "flash",
            "rationale": "reddish reflection",
        },
        {
            "frame_id": "f1",
            "video": "video12",
            "split": "test",
            "key": "cystic_duct",
            "present": 1,
            "final_present": 1,
            "conf": 0.88,
            "trust": 0.44,
            "status": "AMBER",
            "final_by": "argon",
            "rationale": "tubular structure near infundibulum",
        },
        {
            "frame_id": "f2",
            "video": "video12",
            "split": "test",
            "key": "l_hook",
            "present": 1,
            "final_present": 1,
            "conf": 0.82,
            "trust": 0.41,
            "status": "AMBER",
            "final_by": "flash",
            "rationale": "metallic shaft at right margin",
        },
    ])

    pending0, cleared0, sum0 = gate.compute_amber_queue(dec_df, gate.load_gate_log(log_path))
    assert sum0["n_accept"] == 2
    assert sum0["n_reject"] == 1
    assert sum0["n_amber_total"] == 2
    assert sum0["n_amber_pending"] == 2
    assert sum0["export_unlocked"] is False
    assert len(pending0) == 2
    assert len(cleared0) == 0

    with pytest.raises(PermissionError):
        gate.export_verified_dataset(dec_df, gate.load_gate_log(log_path), out_path=export_path)
    assert not export_path.exists()

    # 3. Approve first AMBER item -> still 1 pending -> still locked!
    rec1 = gate.record_gate_decision(
        reviewer="Dr. A. Patel",
        frame_id="f1",
        key="cystic_duct",
        action="approve",
        human_present=1,
        video="video12",
        split="test",
        model_present=1,
        conf=0.88,
        rationale="tubular structure near infundibulum",
        log_path=log_path,
    )
    assert rec1["who"] == "Dr. A. Patel"
    assert rec1["what"] == "approve"
    assert "when" in rec1 and "t" in rec1
    # Check mirror file was also created
    assert (out_dir / "human_gate_log.jsonl").is_file()

    pending1, cleared1, sum1 = gate.compute_amber_queue(dec_df, gate.load_gate_log(log_path))
    assert sum1["n_amber_pending"] == 1
    assert sum1["n_amber_cleared"] == 1
    assert sum1["n_human_approved"] == 1
    assert sum1["export_unlocked"] is False
    with pytest.raises(PermissionError):
        gate.export_verified_dataset(dec_df, gate.load_gate_log(log_path), out_path=export_path)

    # 4. Correct second AMBER item (flip 1 -> 0) -> 0 pending -> UNLOCKED!
    gate.record_gate_decision(
        reviewer="Dr. A. Patel",
        frame_id="f2",
        key="l_hook",
        action="correct",
        human_present=0,
        video="video12",
        split="test",
        model_present=1,
        conf=0.82,
        rationale="metallic shaft at right margin",
        log_path=log_path,
    )

    log_now = gate.load_gate_log(log_path)
    pending2, cleared2, sum2 = gate.compute_amber_queue(dec_df, log_now)
    assert sum2["n_amber_pending"] == 0
    assert sum2["n_amber_cleared"] == 2
    assert sum2["n_human_approved"] == 1
    assert sum2["n_human_corrected"] == 1
    assert sum2["export_unlocked"] is True
    assert pending2.empty
    assert len(cleared2) == 2

    exported = gate.export_verified_dataset(dec_df, log_now, out_path=export_path)
    assert export_path.is_file()
    lines = [json.loads(ln) for ln in export_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 4  # 2 auto-accepted + 2 human-cleared AMBERs (1 rejected excluded)
    assert len(exported) == 4
    by_item = {(r["frame_id"], r["key"]): r for r in lines}
    assert by_item[("f1", "grasper")]["present"] == 1
    assert by_item[("f1", "grasper")]["verification_source"] == "agent_flash"
    assert by_item[("f1", "cystic_duct")]["present"] == 1
    assert by_item[("f1", "cystic_duct")]["final_by"] == "human"
    assert by_item[("f1", "cystic_duct")]["human_action"] == "approve"
    assert by_item[("f2", "l_hook")]["present"] == 0  # Corrected from 1 to 0!
    assert by_item[("f2", "l_hook")]["final_by"] == "human"
    assert by_item[("f2", "l_hook")]["human_action"] == "correct"
    assert ("f1", "blood") not in by_item

    # 5. Undo/reset second AMBER item -> re-locks export!
    gate.record_gate_decision(
        reviewer="Dr. A. Patel",
        frame_id="f2",
        key="l_hook",
        action="reset",
        human_present=None,
        log_path=log_path,
    )
    _, _, sum3 = gate.compute_amber_queue(dec_df, gate.load_gate_log(log_path))
    assert sum3["n_amber_pending"] == 1
    assert sum3["export_unlocked"] is False
    with pytest.raises(PermissionError):
        gate.export_verified_dataset(dec_df, gate.load_gate_log(log_path), out_path= tmp_path / "should_not_write.jsonl")


def test_real_smoke_labels_through_verifier_and_gate_flow(tmp_path):
    """Test against nir-label's real smoke labels run through verifier code without inventing decisions."""
    gate = _load_gate_module()
    smoke_path = Path(C.ROOT) / "coord" / "artifacts" / "labels_smoke.jsonl"
    if not smoke_path.is_file():
        smoke_path = Path(C.OUT_DIR) / "labels_blank.jsonl"
    assert smoke_path.is_file(), "Real smoke or blank labels JSONL must exist on disk"

    dec_df = gate.load_decisions_from_source(smoke_path)
    assert not dec_df.empty
    assert set(dec_df["status"].unique()).issubset({"accept", "AMBER", "reject"})

    log_path = tmp_path / "gate_log.jsonl"
    export_path = tmp_path / "verified_dataset.jsonl"

    pending, cleared, summary = gate.compute_amber_queue(dec_df, [])
    assert summary["n_total"] == len(dec_df)
    assert summary["n_amber_total"] == len(pending)
    assert summary["n_amber_total"] > 0, "Real smoke labels must produce non-zero AMBER items via verifier"
    assert summary["export_unlocked"] is False

    # Clearing all real AMBER items unlocks export
    for idx, r in enumerate(pending.itertuples()):
        m_pres = int(getattr(r, "final_present", getattr(r, "present", 0)))
        action = "approve" if idx % 2 == 0 else "correct"
        h_pres = m_pres if action == "approve" else (1 - m_pres)
        gate.record_gate_decision(
            reviewer="Smoke Test Surgeon",
            frame_id=str(r.frame_id),
            key=str(r.key),
            action=action,
            human_present=h_pres,
            video=str(getattr(r, "video", "")),
            split=str(getattr(r, "split", "test")),
            model_present=m_pres,
            conf=float(getattr(r, "conf", 0.0)),
            rationale=str(getattr(r, "rationale", "")),
            log_path=log_path,
        )

    gate_log = gate.load_gate_log(log_path)
    pending_after, cleared_after, summary_after = gate.compute_amber_queue(dec_df, gate_log)
    assert summary_after["n_amber_pending"] == 0
    assert summary_after["n_amber_cleared"] == summary["n_amber_total"]
    assert summary_after["export_unlocked"] is True
    assert pending_after.empty
    assert len(cleared_after) == summary["n_amber_total"]

    records = gate.export_verified_dataset(dec_df, gate_log, out_path=export_path)
    assert len(records) == summary["n_accept"] + summary["n_amber_total"]
    assert export_path.is_file()


def test_streamlit_human_gate_page_ui_and_fast_clear(tmp_path, monkeypatch):
    """End-to-end Streamlit AppTest for pages/2_Human_gate.py verifying UI lock -> clear -> export."""
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    coord_dir = tmp_path / "coord" / "artifacts"
    out_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    coord_dir.mkdir(parents=True)

    f_img = tmp_path / "frame1.png"
    Image.new("RGB", (320, 240), color=(60, 35, 35)).save(f_img)

    pd.DataFrame([
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "path": str(f_img),
        }
    ]).to_csv(data_dir / "index.csv", index=False)

    pd.DataFrame([
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "final_present": 1,
            "conf": 0.96,
            "trust": 0.91,
            "box": "[100, 120, 420, 480]",
            "evidence": "grasper jaws holding tissue",
            "rationale": "grasper jaws holding tissue",
            "pv_match": True,
            "pv_conf": 0.92,
            "p_wrong": 0.04,
            "critic": "clear grasper",
            "frame_path": str(f_img),
            "status": "accept",
            "final_by": "flash",
        },
        {
            "frame_id": "video12_00080_80",
            "video": "video12",
            "split": "test",
            "key": "cystic_duct",
            "present": 1,
            "final_present": 1,
            "conf": 0.78,
            "trust": 0.45,
            "box": "[260, 300, 520, 580]",
            "evidence": "possible duct structure in hepatocystic triangle",
            "rationale": "possible duct structure in hepatocystic triangle",
            "pv_match": False,
            "pv_conf": 0.42,
            "p_wrong": 0.35,
            "critic": "not skeletonized yet",
            "frame_path": str(f_img),
            "status": "AMBER",
            "final_by": "argon",
        },
    ]).to_csv(out_dir / "decisions.csv", index=False)

    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(C, "ROOT", tmp_path)

    at = AppTest.from_file(str(GATE_PAGE_PATH), default_timeout=15)
    at.run()
    assert not at.exception, f"AppTest raised exception: {at.exception}"

    # 1. Verify AMBER item is displayed with model's label, confidence, and rationale
    all_md = "\n".join(m.value for m in at.markdown)
    assert "cystic_duct" in all_md
    assert "possible duct structure in hepatocystic triangle" in all_md
    assert "ACTION BLOCKED" in all_md

    # 2. Verify Export button is disabled while AMBER item is pending
    export_btn = at.button(key="btn_export_verified_dataset")
    assert export_btn.disabled is True

    # 3. Click the Approve button on the focus card to clear the AMBER item
    approve_btn = at.button(key="focus_approve_video12_00080_80_cystic_duct")
    approve_btn.click().run()
    assert not at.exception, f"AppTest raised exception after approve: {at.exception}"

    # 4. Verify gate_log.jsonl was written and Export button is now ENABLED
    assert (out_dir / "gate_log.jsonl").is_file()
    export_btn_after = at.button(key="btn_export_verified_dataset")
    assert export_btn_after.disabled is False

    # 5. Click Export verified dataset button and confirm out/verified_dataset.jsonl is written
    export_btn_after.click().run()
    assert not at.exception
    assert (out_dir / "verified_dataset.jsonl").is_file()
    exported_lines = [
        json.loads(ln)
        for ln in (out_dir / "verified_dataset.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(exported_lines) == 2


def test_clear_20_real_smoke_ambers_via_ui_and_export_verified_dataset(tmp_path, monkeypatch):
    """Verify clearing >=20 real AMBER items from nir-label smoke labels run through verifier and exporting verified_dataset.jsonl."""
    gate = _load_gate_module()
    smoke_path = Path(C.ROOT) / "coord" / "artifacts" / "labels_smoke.jsonl"
    if not smoke_path.is_file():
        smoke_path = Path(C.OUT_DIR) / "labels_blank.jsonl"
    real_dec_df = gate.load_decisions_from_source(smoke_path)

    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    coord_dir = tmp_path / "coord" / "artifacts"
    out_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)
    coord_dir.mkdir(parents=True)

    # Save the real verifier decisions from smoke labels as out/decisions.csv in temp test dir
    real_dec_df.to_csv(out_dir / "decisions.csv", index=False)

    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(C, "ROOT", tmp_path)

    at = AppTest.from_file(str(GATE_PAGE_PATH), default_timeout=15)
    at.run()
    assert not at.exception

    # Confirm >= 20 AMBERs are pending and export is disabled
    _, _, initial_summary = gate.compute_amber_queue(real_dec_df, [])
    assert initial_summary["n_amber_pending"] >= 20
    assert at.button(key="btn_export_verified_dataset").disabled is True

    # Approve the first item via the Rapid Review Deck button, then fast-forward the rest for 3-min demo speed
    first_row = real_dec_df[real_dec_df["status"] == "AMBER"].iloc[0]
    first_btn_key = f"focus_approve_{first_row['frame_id']}_{first_row['key']}"
    at.button(key=first_btn_key).click().run()
    assert not at.exception
    assert at.button(key="btn_export_verified_dataset").disabled is True

    # Click the Demo Fast-Forward button to clear all remaining AMBER items
    at.button(key="batch_approve_all_remaining").click().run()
    assert not at.exception

    # Export is now unlocked!
    exp_btn = at.button(key="btn_export_verified_dataset")
    assert exp_btn.disabled is False
    exp_btn.click().run()
    assert not at.exception

    verified_jsonl = out_dir / "verified_dataset.jsonl"
    assert verified_jsonl.is_file()
    verified_rows = [json.loads(ln) for ln in verified_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(verified_rows) == initial_summary["n_accept"] + initial_summary["n_amber_total"]
    for vr in verified_rows:
        assert "provenance" in vr
        assert vr["provenance"]["seeding"] == "self_written_manual (out/manual.json)"
        assert "double_check" in vr["provenance"]
        assert "release_gate" in vr["provenance"]


def test_amber_loop_funnel_and_accepted_audit_sample():
    """Verify multi-round Amber Loop funnel calculation and deterministic accepted audit sampling."""
    gate = _load_gate_module()
    dec_df = pd.DataFrame([
        {
            "frame_id": f"f{i}",
            "video": "video12",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "final_present": 1,
            "conf": 0.95,
            "p_wrong": 0.05,
            "pv_match": True,
            "box": "[100, 100, 400, 400]",
            "status": "accept",
            "final_by": "flash",
            "rationale": "jaws visible",
        }
        for i in range(8)
    ] + [
        {
            "frame_id": "f8",
            "video": "video12",
            "split": "test",
            "key": "cystic_duct",
            "present": 1,
            "final_present": 1,
            "conf": 0.90,
            "p_wrong": 0.10,
            "pv_match": False,  # initial trust = 0.90 * 0.90 * 0.5 = 0.405 -> Round 0 AMBER
            "box": "[200, 200, 500, 500]",
            "status": "accept",  # resolved by Argon in Round 1
            "final_by": "argon",
            "rationale": "resolved by Argon loop",
        },
        {
            "frame_id": "f9",
            "video": "video12",
            "split": "test",
            "key": "blood",
            "present": 1,
            "final_present": 1,
            "conf": 0.85,
            "p_wrong": 0.10,
            "pv_match": False,  # initial trust = 0.85 * 0.90 * 0.5 = 0.3825 -> Round 0 AMBER
            "box": "[300, 300, 600, 600]",
            "status": "AMBER",  # still AMBER after loop
            "final_by": "argon",
            "rationale": "still ambiguous",
        },
    ])
    amber_rounds = [
        {"frame_id": "f8", "key": "cystic_duct", "round": 1, "verdict": "accept", "reason": "Confirmed on tighter crop"},
        {"frame_id": "f9", "key": "blood", "round": 1, "verdict": "AMBER", "reason": "Still ambiguous"},
    ]
    funnel = gate.compute_amber_funnel(dec_df, amber_rounds=amber_rounds, n_amber_pending_after_human=0)
    assert len(funnel) == 3
    assert funnel[0]["ambers_remaining"] == 2
    assert funnel[1]["ambers_remaining"] == 1
    assert funnel[1]["resolved_in_step"] == 1
    assert funnel[2]["ambers_remaining"] == 0

    sample_df = gate.sample_accepted_for_audit(dec_df, n_sample=5, seed=0)
    assert len(sample_df) == 5
    assert (sample_df["status"] == "accept").all()


def test_real_decisions_csv_and_amber_rounds_jsonl_funnel_and_design_1d_fields(tmp_path):
    """Verify pages/2_Human_gate.py against real out/decisions.csv and out/amber_rounds.jsonl on disk."""
    gate = _load_gate_module()
    dec_path = Path(C.OUT_DIR) / "decisions.csv"
    if not dec_path.is_file():
        dec_path = Path(C.ROOT) / "coord" / "artifacts" / "decisions.csv"
    assert dec_path.is_file()

    dec_df = gate.load_decisions_from_source(dec_path)
    amber_rounds = gate.load_amber_rounds()
    pending, _, summary = gate.compute_amber_queue(dec_df, [])
    funnel = gate.compute_amber_funnel(dec_df, amber_rounds=amber_rounds, n_amber_pending_after_human=summary["n_amber_pending"])
    assert len(funnel) >= 2
    assert funnel[0]["ambers_remaining"] >= summary["n_amber_total"]

    log_path = tmp_path / "gate_log.jsonl"
    export_path = tmp_path / "verified_dataset.jsonl"
    for r in pending.itertuples():
        m_p = int(getattr(r, "final_present", getattr(r, "present", 0)))
        gate.record_gate_decision(
            reviewer="Dr. Release Reviewer",
            frame_id=str(r.frame_id),
            key=str(r.key),
            action="approve",
            human_present=m_p,
            item_type="amber",
            log_path=log_path,
        )

    gate_log = gate.load_gate_log(log_path)
    assert gate_log[0]["item_type"] == "amber"
    exported = gate.export_verified_dataset(dec_df, gate_log, out_path=export_path, amber_rounds=amber_rounds)
    assert len(exported) == summary["n_accept"] + summary["n_amber_total"]
    for row in exported:
        assert "manual_sources" in row and len(row["manual_sources"]) > 0
        assert "audit_signed_off_by" in row
        assert "released_at" in row
        assert "provenance" in row


def test_human_gate_page_passes_lanes_integrity_patterns():
    """Verify pages/2_Human_gate.py, web/serve.py, and web/gate.js pass all anti-fabrication regexes in lanes/integrity.py."""
    import re
    patterns = [
        (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
    ]
    targets = [
        GATE_PAGE_PATH,
        Path(C.ROOT) / "web" / "serve.py",
        Path(C.ROOT) / "web" / "gate.js",
    ]
    fails = []
    for target in targets:
        assert target.is_file(), f"Missing expected gate file: {target}"
        for i, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "integrity: allow" in s:
                continue
            for pat, what in patterns:
                if re.search(pat, line):
                    fails.append(f"{target.name}:{i}: {what}: {s[:110]}")
    assert not fails, f"Integrity check failed: {fails}"


def test_web_serve_get_and_post_api_gate_lock_and_release_flow(tmp_path):
    """Verify web/serve.py GET/POST /api/gate lock logic, audit sign-off, undo, and verified_dataset.jsonl export."""
    spec = importlib.util.spec_from_file_location("web_serve_mod", Path(C.ROOT) / "web" / "serve.py")
    assert spec is not None and spec.loader is not None
    serve_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve_mod)

    dec_path = Path(C.OUT_DIR) / "decisions.csv"
    if not dec_path.is_file():
        dec_path = Path(C.ROOT) / "coord" / "artifacts" / "decisions.csv"
    assert dec_path.is_file()

    gate_log_path = tmp_path / "gate_log.jsonl"
    human_log_path = tmp_path / "human_gate_log.jsonl"
    verified_path = tmp_path / "verified_dataset.jsonl"

    state0 = serve_mod.build_gate_state_payload(
        scope="test_5",
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
    )
    assert state0["ok"] is True
    assert state0["summary"]["n_pending_amber"] > 0
    assert state0["summary"]["can_release"] is False
    assert len(state0["audit_sample"]) == 20
    for item in state0["pending_ambers"] + state0["audit_sample"]:
        for k in item:
            assert k != "truth" and not k.startswith("gt_") and k != "mask_path"

    # Attempt release while AMBERs remain -> must return 409 CONFLICT
    code_locked, resp_locked = serve_mod.handle_gate_post(
        {"action": "release_dataset", "scope": "test_5", "reviewer": "Dr. Gate Test"},
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_locked) == 409
    assert resp_locked["ok"] is False
    assert not verified_path.exists()

    # Approve first AMBER item, then undo, then batch approve all + sign off audit + release
    first_amber = state0["pending_ambers"][0]
    code_one, resp_one = serve_mod.handle_gate_post(
        {
            "frame_id": first_amber["frame_id"],
            "key": first_amber["key"],
            "human_action": "correct",
            "scope": "test_5",
            "reviewer": "Dr. Gate Test",
        },
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_one) == 200
    assert resp_one["state"]["summary"]["n_cleared_amber"] == 1

    code_undo, resp_undo = serve_mod.handle_gate_post(
        {"action": "undo", "frame_id": first_amber["frame_id"], "key": first_amber["key"], "scope": "test_5"},
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_undo) == 200
    assert resp_undo["state"]["summary"]["n_cleared_amber"] == 0

    code_batch, resp_batch = serve_mod.handle_gate_post(
        {"action": "batch_approve_ambers", "scope": "test_5", "reviewer": "Dr. Gate Test"},
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_batch) == 200
    assert resp_batch["state"]["summary"]["n_pending_amber"] == 0

    code_audit, resp_audit = serve_mod.handle_gate_post(
        {"action": "sign_off_audit_sample", "count": 20, "scope": "test_5", "reviewer": "Dr. Gate Test"},
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_audit) == 200
    assert resp_audit["state"]["summary"]["can_release"] is True

    code_rel, resp_rel = serve_mod.handle_gate_post(
        {"action": "release_dataset", "scope": "test_5", "reviewer": "Dr. Gate Test"},
        decisions_csv_path=dec_path,
        gate_log_path=gate_log_path,
        human_gate_log_path=human_log_path,
        verified_jsonl_path=verified_path,
    )
    assert int(code_rel) == 200
    assert resp_rel["ok"] is True
    assert verified_path.is_file()
    exported_lines = [json.loads(line) for line in verified_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(exported_lines) == state0["summary"]["n_auto_accept"] + state0["summary"]["n_residual_amber"]
    assert all("provenance" in r and "manual_sources" in r for r in exported_lines)


def test_web_serve_live_http_roundtrip():
    """Start web/serve.py HTTP server on an ephemeral port and verify GET /api/health and GET /api/gate."""
    import os
    import threading
    import urllib.request

    spec = importlib.util.spec_from_file_location("web_serve_http_mod", Path(C.ROOT) / "web" / "serve.py")
    assert spec is not None and spec.loader is not None
    serve_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(serve_mod)

    os.environ["SURGPROOF_QUIET"] = "1"
    srv = serve_mod.create_server("127.0.0.1", 0)
    port = srv.server_port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as r1:
            assert r1.status == 200
            h_data = json.loads(r1.read().decode("utf-8"))
            assert h_data["ok"] is True

        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/gate?scope=test_5", timeout=5) as r2:
            assert r2.status == 200
            g_data = json.loads(r2.read().decode("utf-8"))
            assert g_data["ok"] is True
            assert g_data["summary"]["round_0_amber"] >= g_data["summary"]["round_3_amber"]
    finally:
        srv.shutdown()
        srv.server_close()





