"""Comprehensive unit and integration tests for accept.py and verify/ (nir-verifier lane, Issue #6)."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import accept
import config as C
import verify
from eval import audit
from verify.engine import (
    DECISION_OUTPUT_COLS,
    HEADLINE_KEYS,
    VERIFIER_INPUT_COLS,
    _pv_ok,
    _safe_prob,
    apply_argon_verdict,
    argon_revise,
    enrich_results_excl_liver,
    export_verified_dataset_jsonl,
    prepare_verifier_dataframe,
    run_amber_loop,
    status,
    summarize_decisions_zero_label,
    trust,
)


def test_zero_gt_references_in_accept_and_verify():
    """Verify accept.py and all verify/*.py files contain zero AST or substring references to GT."""
    root = Path(accept.__file__).resolve().parent
    files = [root / "accept.py"] + sorted((root / "verify").rglob("*.py"))
    assert len(files) >= 3
    forbidden_substrings = ("gt_", "truth")
    for fpath in files:
        txt = fpath.read_text(encoding="utf-8")
        for sub in forbidden_substrings:
            assert sub not in txt, f"Forbidden substring {sub!r} found in {fpath}"
        tree = ast.parse(txt, filename=str(fpath))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert not any(node.id == s or node.id.startswith(s) for s in forbidden_substrings)


def test_pv_ok_handles_nan_numpy_bool_and_strings():
    for truthy in (True, np.bool_(True), 1, 1.0, np.float64(1.0), "true", "True", "1", "1.0", "yes"):
        assert _pv_ok(truthy) is True, f"Expected True for {truthy!r} ({type(truthy)})"

    for falsy in (
        False,
        np.bool_(False),
        0,
        0.0,
        np.float64(0.0),
        None,
        float("nan"),
        np.nan,
        pd.NA,
        "false",
        "False",
        "0",
        "0.0",
        "nan",
        "None",
        "null",
        "",
        [],
        {},
    ):
        assert _pv_ok(falsy) is False, f"Expected False for {falsy!r} ({type(falsy)})"


def test_safe_prob_handles_nan_and_percentages():
    assert _safe_prob(0.85) == pytest.approx(0.85)
    assert _safe_prob(85.0) == pytest.approx(0.85)
    assert _safe_prob(-0.5) == pytest.approx(0.0)
    assert _safe_prob(float("nan"), default=0.5) == pytest.approx(0.5)
    assert _safe_prob(None, default=0.25) == pytest.approx(0.25)
    assert _safe_prob("invalid", default=0.4) == pytest.approx(0.4)
    assert _safe_prob([1, 2], default=0.3) == pytest.approx(0.3)


def test_trust_and_status_routing():
    # Positive claim with pv_match agreeing -> full score, status='accept'
    for pos_pv in (True, np.bool_(True), 1.0, 1, "true"):
        r = SimpleNamespace(present=1, conf=0.90, p_wrong=0.10, pv_match=pos_pv)
        assert trust(r) == pytest.approx(0.81)
        assert status(trust(r)) == "accept"

    # Positive claim with pv_match disagreeing or NaN -> halved score, status='AMBER'
    for neg_pv in (False, np.bool_(False), 0.0, 0, None, float("nan"), np.nan, "false"):
        r = SimpleNamespace(present=1, conf=0.90, p_wrong=0.10, pv_match=neg_pv)
        assert trust(r) == pytest.approx(0.405)
        assert status(trust(r)) == "AMBER"

    # Absent claim -> pv_match not required, full conf * (1 - p_wrong)
    r_abs = SimpleNamespace(present=0, conf=0.88, p_wrong=0.10, pv_match=float("nan"))
    assert trust(r_abs) == pytest.approx(0.792)
    assert status(trust(r_abs)) == "accept"

    # Low-trust claim -> status='reject'
    r_rej = SimpleNamespace(present=1, conf=0.50, p_wrong=0.60, pv_match=False)
    assert trust(r_rej) == pytest.approx(0.10)
    assert status(trust(r_rej)) == "reject"

    # NaN trust -> status='reject'
    assert status(float("nan")) == "reject"
    assert status(None) == "reject"


def test_argon_revise_prompt_nan_pv_match_guard(monkeypatch):
    captured_prompts = []

    async def fake_call(prompt, *, model=None, images=(), schema=None, tag=""):
        captured_prompts.append(prompt)
        return {"present": False, "confidence": 0.91, "reasoning": "clear hepatocystic triangle"}

    monkeypatch.setattr(accept.gem, "img_part", lambda p: ("part", b"bytes"))
    monkeypatch.setattr(accept.gem, "call", fake_call)

    # Case 1: pv_match is float('nan') (absent label) -> must NOT say 'agreed'
    r_nan = SimpleNamespace(
        frame_id="video01_00080_80",
        key="cystic_duct",
        present=0,
        conf=0.60,
        evidence="not dissected yet",
        pv_match=float("nan"),
        critic="could be hidden",
        p_wrong=0.35,
        frame_path="/tmp/fake.png",
    )
    v1 = asyncio.run(accept.argon_revise(r_nan, {"procedure": "lap chole"}))
    assert "disagreed or was not possible" in captured_prompts[-1]
    assert v1["present"] is False and v1["confidence"] == pytest.approx(0.91)

    # Case 2: pv_match is np.float64(1.0) -> must say 'agreed'
    r_true = SimpleNamespace(
        frame_id="video01_00080_80",
        key="grasper",
        present=1,
        conf=0.75,
        evidence="metallic jaws",
        pv_match=np.float64(1.0),
        critic="minor motion blur",
        p_wrong=0.25,
        frame_path="/tmp/fake.png",
    )
    asyncio.run(accept.argon_revise(r_true, {"procedure": "lap chole"}))
    assert "A blind crop check agreed." in captured_prompts[-1]


def test_apply_argon_verdict_bands():
    d = pd.DataFrame([
        {"frame_id": "f1", "key": "grasper", "present": 1, "pv_match": 1.0, "p_wrong": 0.12, "trust": 0.68, "status": "AMBER", "final_present": 1, "final_by": "flash"},
        {"frame_id": "f2", "key": "l_hook", "present": 1, "pv_match": 1.0, "p_wrong": 0.20, "trust": 0.42, "status": "AMBER", "final_present": 1, "final_by": "flash"},
        {"frame_id": "f3", "key": "blood", "present": 1, "pv_match": 0.0, "p_wrong": 0.40, "trust": 0.38, "status": "AMBER", "final_present": 1, "final_by": "flash"},
        {"frame_id": "f4", "key": "cystic_duct", "present": 0, "pv_match": None, "p_wrong": 0.35, "trust": 0.61, "status": "AMBER", "final_present": 0, "final_by": "flash"},
    ])
    # High-confidence Argon agreement with aligned crop check & low p_wrong -> accept (just above 0.70 floor)
    apply_argon_verdict(d, 0, {"present": True, "confidence": 0.92, "reasoning": "confirmed"})
    assert d.loc[0, "status"] == "accept"
    assert d.loc[0, "final_by"] == "argon"
    assert 0.70 < d.loc[0, "trust"] < 0.75

    # Inter-model disagreement (Flash=1, Argon=0 with moderate conf) -> remains AMBER for Human Gate
    apply_argon_verdict(d, 1, {"present": False, "confidence": 0.55, "reasoning": "obscured by smoke"})
    assert d.loc[1, "status"] == "AMBER"
    assert d.loc[1, "final_present"] == 0
    assert d.loc[1, "final_by"] == "argon"

    # Very low-confidence Argon revision -> reject
    apply_argon_verdict(d, 2, {"present": False, "confidence": 0.20, "reasoning": "hallucinated reflection"})
    assert d.loc[2, "status"] == "reject"
    assert d.loc[2, "final_present"] == 0

    # High-confidence Argon confirmation of absence -> accept
    apply_argon_verdict(d, 3, {"present": False, "confidence": 0.95, "reasoning": "clearly absent"})
    assert d.loc[3, "status"] == "accept"
    assert d.loc[3, "final_present"] == 0
    assert 0.70 < d.loc[3, "trust"] < 0.75


def test_prepare_verifier_dataframe_strips_extra_cols_and_passes_audit():
    raw = pd.DataFrame([
        {
            "frame_id": "video01_00080_80",
            "video_id": "video01",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "confidence": 0.92,
            "rationale": "metallic jaws",
            "box": [100, 100, 400, 400],
            "pv_match": 1.0,
            "pv_conf": 0.88,
            "p_wrong": 0.10,
            "critic": "none",
            "extra_unwanted_col": 123,
        },
        {
            "frame_id": "video01_00080_80",
            "video_id": "video01",
            "split": "test",
            "key": "cystic_duct",
            "present": 1,
            "confidence": 0.88,
            "rationale": "tubular structure",
            "box": [200, 200, 350, 350],
            "pv_match": 0.0,
            "pv_conf": 0.30,
            "p_wrong": 0.10,
            "critic": "triangle not cleared",
        },
    ])
    idx = pd.DataFrame([{"frame_id": "video01_00080_80", "path": "raw/v1.png", "split": "test", "video": "video01"}])
    d = prepare_verifier_dataframe(raw, idx)
    assert list(d.columns) == DECISION_OUTPUT_COLS
    assert "extra_unwanted_col" not in d.columns
    assert d.loc[0, "status"] == "accept"
    assert d.loc[1, "status"] == "AMBER"
    assert d.loc[0, "frame_path"] == "raw/v1.png"

    # Verify independent audit_verifier_leakage passes with 0 issues
    dec_for_audit = d.copy()
    dec_for_audit["truth"] = [1, 0]
    leak_res = audit.audit_verifier_leakage(dec_for_audit)
    assert leak_res["status"] == "PASS"
    assert leak_res["zero_human_label_verified"] is True
    assert leak_res["issues"] == []


def test_enrich_results_excl_liver_exact_math():
    # Build 6 test labels: 5 non-liver accepted (4 correct, 1 wrong -> 0.8000) + 1 liver accepted (1 correct -> 1.0)
    rows = []
    for k in C.LABELS:
        rows.append({
            "frame_id": "video01_00080_80",
            "split": "test",
            "key": k,
            "status": "accept",
            "final_present": 1,
        })
    d = pd.DataFrame(rows)
    # All 6 accepted: 5/6 = 0.8333 accuracy; liver = 1.0
    res = {
        "n_test_labels": 6,
        "accepted": {"coverage": 1.0, "accuracy": round(5 / 6, 4)},
        "per_label_accepted_acc": {
            "grasper": 1.0,
            "l_hook": 1.0,
            "gallbladder": 1.0,
            "cystic_duct": 1.0,
            "blood": 0.0,
            "liver": 1.0,
        },
        "upper_bound_calibrated": {"coverage_at_95": 0.7, "acc_at_60": 0.95, "aurc": 0.04},
    }
    enriched = enrich_results_excl_liver(res, d)
    assert enriched["headline_labels"] == HEADLINE_KEYS
    assert enriched["excluded_uninformative_labels"] == ["liver"]
    assert enriched["accepted_5_informative"]["accuracy"] == pytest.approx(0.8000)
    assert enriched["accepted_5_informative"]["coverage"] == pytest.approx(1.0)
    assert enriched["upper_bound_calibrated"]["is_upper_bound"] is True
    assert enriched["upper_bound_calibrated"]["fitted_on_split"] == "calib"


def test_accept_on_real_blank_labels_in_tmp(tmp_path):
    """Run accept.main (--no-argon) against real out/labels_blank.jsonl into tmp_path without touching shared out/."""
    blank_path = C.OUT_DIR / "labels_blank.jsonl"
    if not blank_path.exists():
        pytest.skip("out/labels_blank.jsonl not present")
    out_csv = tmp_path / "decisions_blank_smoke.csv"
    summary = asyncio.run(
        accept.main(
            use_argon=False,
            argon_budget=0,
            labels_path=blank_path,
            out_path=out_csv,
            skip_eval=True,
        )
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 1680
    assert list(df.columns) == DECISION_OUTPUT_COLS
    assert set(df["status"].unique()).issubset({"accept", "AMBER", "reject"})
    assert summary["n_total"] == 1680
    assert summary["test_5_informative_excl_liver"]["n"] == 1400


def test_three_round_amber_loop_shrinks_ambers_and_logs(tmp_path, monkeypatch):
    """Verify the 3-round Argon Amber Loop shrinks AMBERs round by round (R0 -> R1 -> R2 -> R3) and writes logs."""
    monkeypatch.setattr(accept.gem, "img_part", lambda p, max_side=768, crop=None: ("part", b"bytes"))

    raw = pd.DataFrame([
        # Row 0: clear accept in Round 0
        {
            "frame_id": "v1_1",
            "video": "video01",
            "split": "test",
            "key": "grasper",
            "present": 1,
            "conf": 0.95,
            "pv_match": True,
            "p_wrong": 0.05,
            "box": [100, 100, 400, 400],
            "evidence": "jaws",
            "critic": "ok",
        },
        # Row 1: AMBER in R0 -> resolved to accept in R1 (absent claim confirmed with conf 0.90)
        {
            "frame_id": "v1_1",
            "video": "video01",
            "split": "test",
            "key": "cystic_duct",
            "present": 0,
            "conf": 0.90,
            "pv_match": None,
            "p_wrong": 0.35,
            "box": [],
            "evidence": "not exposed",
            "critic": "maybe hidden",
        },
        # Row 2: AMBER in R0 -> stays AMBER in R1 (conf 0.72), resolved in R2 (conf 0.85)
        {
            "frame_id": "v1_1",
            "video": "video01",
            "split": "test",
            "key": "blood",
            "present": 0,
            "conf": 0.88,
            "pv_match": None,
            "p_wrong": 0.35,
            "box": [],
            "evidence": "clean field",
            "critic": "reddish liver",
        },
        # Row 3: AMBER in R0 -> stays AMBER in R1 & R2, resolved in R3 (3-vote consensus rejects false positive)
        {
            "frame_id": "v1_1",
            "video": "video01",
            "split": "test",
            "key": "l_hook",
            "present": 1,
            "conf": 0.85,
            "pv_match": False,
            "p_wrong": 0.15,
            "box": [200, 200, 500, 500],
            "evidence": "shaft",
            "critic": "suction cannula",
        },
    ])
    idx = pd.DataFrame([{"frame_id": "v1_1", "path": "/tmp/v1_1.png", "split": "test", "video": "video01"}])
    d = prepare_verifier_dataframe(raw, idx)
    assert (d["status_round_0"] == "AMBER").sum() == 3

    async def fake_loop_call(prompt, *, model=None, images=(), schema=None, tag="", temperature=0.0, sample=0):
        if "'cystic_duct'" in prompt:
            return {"present": False, "confidence": 0.95, "reasoning": "R1 confirmed absent"}
        if "'blood'" in prompt:
            if tag == "argon-revise":
                return {"present": False, "confidence": 0.72, "reasoning": "R1 tentative absent"}
            return {"present": False, "confidence": 0.91, "reasoning": "R2 confirmed absent"}
        if "'l_hook'" in prompt:
            return {"present": False, "confidence": 0.72, "reasoning": "suction not hook"}
        return {"present": True, "confidence": 0.90, "reasoning": "ok"}

    logs = asyncio.run(run_amber_loop(d, {"procedure": "lap chole"}, argon_budget=50, max_rounds=3, call_fn=fake_loop_call))
    summary = summarize_decisions_zero_label(d)
    curve = summary["amber_loop"]["ambers_by_round_test"]
    assert curve == [3, 2, 1, 0], f"Expected shrinking AMBER curve [3, 2, 1, 0], got {curve}"
    assert len(logs) == 6  # R1: 3 items, R2: 2 items, R3: 1 item

    ds_path = tmp_path / "verified_dataset.jsonl"
    n_written = export_verified_dataset_jsonl(d, {"sources": [{"url": "https://youtube.com/watch?v=test"}]}, ds_path)
    assert n_written == 3
    assert ds_path.exists()
    ds_rows = [ pd.Series(r) for r in pd.read_json(ds_path, lines=True).to_dict(orient="records") ]
    assert len(ds_rows) == 3
    for r in ds_rows:
        assert "final_present" in r and "final_by" in r
        assert "manual_sources" in r and "audit_signed_off_by" in r and "released_at" in r
    for lrec in logs:
        assert "present" in lrec and "conf" in lrec and "verdict" in lrec and "reason" in lrec


def test_zero_synthetic_score_formulas_or_literal_verdicts_in_verifier_code():
    """Ensure accept.py and verify/*.py have zero hits on lanes/integrity.py display PATTERNS."""
    import re

    root = Path(accept.__file__).resolve().parent
    files = [root / "accept.py"] + sorted((root / "verify").rglob("*.py"))
    patterns = [
        (r'["\']box["\']\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
    ]
    for fpath in files:
        for i, line in enumerate(fpath.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for pat, what in patterns:
                assert not re.search(pat, line), f"{fpath.name}:{i}: {what}: {line.strip()}"


def test_accept_on_smoke_labels_in_tmp(tmp_path):
    """Run accept.main (--no-argon) against coord/artifacts/labels_smoke.jsonl into tmp_path."""
    smoke_path = C.ROOT / "coord" / "artifacts" / "labels_smoke.jsonl"
    if not smoke_path.exists():
        pytest.skip("coord/artifacts/labels_smoke.jsonl not present")
    out_csv = tmp_path / "decisions_smoke.csv"
    summary = asyncio.run(
        accept.main(
            use_argon=False,
            argon_budget=0,
            labels_path=smoke_path,
            out_path=out_csv,
            skip_eval=True,
        )
    )
    assert out_csv.exists()
    df = pd.read_csv(out_csv)
    assert len(df) == 240
    assert list(df.columns) == DECISION_OUTPUT_COLS
    assert set(df["status"].unique()).issubset({"accept", "AMBER", "reject"})
    assert summary["n_total"] == 240


def test_full_decisions_and_results_artifacts_contract():
    """Verify full out/decisions.csv and out/results.json meet DESIGN.md contracts with zero GT leakage."""
    import json

    dec_path = C.OUT_DIR / "decisions.csv"
    res_path = C.OUT_DIR / "results.json"
    if not dec_path.exists() or not res_path.exists():
        pytest.skip("out/decisions.csv or out/results.json not present")

    df = pd.read_csv(dec_path)
    assert len(df) == 6060
    assert list(df.columns) == DECISION_OUTPUT_COLS
    assert not any(c == "truth" or c.startswith("gt_") for c in df.columns)
    assert set(df["status"].unique()).issubset({"accept", "AMBER", "reject"})

    res = json.loads(res_path.read_text())
    assert res["n_test_labels"] == 1680
    assert res["headline_labels"] == HEADLINE_KEYS
    assert res["excluded_uninformative_labels"] == ["liver"]
    assert "accepted_5_informative" in res
    assert res["accepted_5_informative"]["n_test_labels_excl_liver"] == 1400
    assert res["upper_bound_calibrated"]["is_upper_bound"] is True



