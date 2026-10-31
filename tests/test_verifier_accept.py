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
    prepare_verifier_dataframe,
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
    apply_argon_verdict(d, 3, {"present": False, "confidence": 0.90, "reasoning": "clearly absent"})
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
