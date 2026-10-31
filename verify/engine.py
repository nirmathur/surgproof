"""Core zero-label verification engine (`verify/engine.py`).

Strictly zero-leakage: operates solely on model predictions and decorrelated check signals.
Emits canonical statuses: 'accept', 'AMBER' (Human Gate), and 'reject'.
"""
import json
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

import config as C
import gem

ACCEPT_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.35
ARGON_ACCEPT_CONF = 0.80

HEADLINE_KEYS = [k for k in C.LABELS if k != "liver"]

VERIFIER_INPUT_COLS = [
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
]

DECISION_OUTPUT_COLS = VERIFIER_INPUT_COLS + [
    "frame_path",
    "trust",
    "status",
    "final_present",
    "final_by",
]


class Verdict(BaseModel):
    present: bool
    confidence: float
    reasoning: str


def _pv_ok(val: Any) -> bool:
    """Safely evaluate blind crop check agreement across bool, numpy.bool_, float/NaN, int, and str."""
    if val is None or isinstance(val, (list, tuple, dict, set)):
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "1.0", "yes", "y", "t"):
            return True
        if s in ("false", "0", "0.0", "no", "n", "f", "nan", "none", "null", ""):
            return False
        return False
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.number)):
        return float(val) != 0.0
    return bool(val)


def _safe_prob(val: Any, default: float = 0.5) -> float:
    """Parse a probability/confidence into [0.0, 1.0], falling back to `default` on NaN/None."""
    if val is None or isinstance(val, (list, tuple, dict, set)):
        return float(default)
    try:
        if pd.isna(val):
            return float(default)
        f = float(val)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(f) or np.isinf(f):
        return float(default)
    if 1.0 < f <= 100.0:
        f = f / 100.0
    return float(max(0.0, min(1.0, f)))


def _safe_present(val: Any) -> int:
    """Coerce predicted presence to 0 or 1."""
    if val is None or isinstance(val, (list, tuple, dict, set)):
        return 0
    try:
        if pd.isna(val):
            return 0
    except (TypeError, ValueError):
        return 0
    if isinstance(val, str):
        return 1 if val.strip().lower() in ("1", "1.0", "true", "yes", "present") else 0
    try:
        return 1 if int(float(val)) != 0 else 0
    except (TypeError, ValueError):
        return 1 if bool(val) else 0


def trust(r: Any) -> float:
    """Compute continuous zero-label verification score in [0.0, 1.0]."""
    raw_conf = getattr(r, "conf", getattr(r, "confidence", 0.0))
    raw_pw = getattr(r, "p_wrong", 0.5)
    c = _safe_prob(raw_conf, default=0.0)
    pw = _safe_prob(raw_pw, default=0.5)
    s = c * (1.0 - pw)
    if _safe_present(getattr(r, "present", 0)) == 1:
        s *= 1.0 if _pv_ok(getattr(r, "pv_match", None)) else 0.5
    return float(s)


def status(
    s: Any,
    accept_thresh: float = ACCEPT_THRESHOLD,
    reject_thresh: float = REJECT_THRESHOLD,
) -> str:
    """Map zero-label trust score to canonical status ('accept' | 'AMBER' | 'reject')."""
    try:
        if s is None or pd.isna(s):
            return "reject"
        val = float(s)
    except (TypeError, ValueError):
        return "reject"
    if np.isnan(val):
        return "reject"
    if val >= accept_thresh:
        return "accept"
    if val < reject_thresh:
        return "reject"
    return "AMBER"


async def argon_revise(r: Any, manual: dict, *, call_fn=None) -> dict:
    """Escalate an uncertain claim to Gemini 4 Argon with full frame + decorrelated check summary."""
    pv_agreed = _pv_ok(getattr(r, "pv_match", None))
    conf_val = _safe_prob(getattr(r, "conf", getattr(r, "confidence", 0.0)), default=0.0)
    pw_val = _safe_prob(getattr(r, "p_wrong", 0.5), default=0.5)
    pres_flag = _safe_present(getattr(r, "present", 0)) == 1
    ev_text = getattr(r, "evidence", getattr(r, "rationale", "")) or ""
    critic_text = getattr(r, "critic", "") or ""
    caller = call_fn or gem.call
    v = await caller(
        f"You are the senior reviewer. A junior labeller said '{r.key}' is "
        f"{'PRESENT' if pres_flag else 'ABSENT'} (conf {conf_val:.2f}, evidence: {ev_text}). "
        f"A blind crop check {'agreed' if pv_agreed else 'disagreed or was not possible'}. "
        f"A critic argued: {critic_text!r} (p_wrong {pw_val:.2f}).\n"
        "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
        f"Manual: {json.dumps(manual)}",
        model=C.ARGON,
        images=[gem.img_part(r.frame_path)],
        schema=Verdict,
        tag="argon-revise",
    )
    v_pres = bool(v.get("present", pres_flag)) if isinstance(v, dict) else pres_flag
    v_conf = _safe_prob(v.get("confidence", 0.5) if isinstance(v, dict) else 0.5, default=0.5)
    v_reason = str(v.get("reasoning", "") if isinstance(v, dict) else "")
    gem.trace(
        "argon",
        f"{r.frame_id} {r.key}: {'present' if v_pres else 'absent'} "
        f"({v_conf:.2f}) - {v_reason[:140]}",
    )
    return {"present": v_pres, "confidence": v_conf, "reasoning": v_reason}


def apply_argon_verdict(d: pd.DataFrame, row_idx: Any, v: dict) -> None:
    """Apply an Argon revision verdict in-place to a decisions DataFrame row.

    Follows the surgical safety rule: false positives carry liability, so uncertain positive claims
    with critic warnings or crop disagreement stay in AMBER for the Human Gate unless all checks align.
    Accepted Argon revisions rank just above the 0.70 Flash acceptance floor.
    """
    orig_pres = int(_safe_present(d.at[row_idx, "present"]))
    pres = int(bool(v.get("present", orig_pres)))
    conf_v = _safe_prob(v.get("confidence", 0.5), default=0.5)
    pw = _safe_prob(d.at[row_idx, "p_wrong"] if "p_wrong" in d.columns else 0.5, default=0.5)
    pv_ok = _pv_ok(d.at[row_idx, "pv_match"] if "pv_match" in d.columns else None)

    d.at[row_idx, "final_present"] = pres
    d.at[row_idx, "final_by"] = "argon"
    if conf_v < REJECT_THRESHOLD:
        d.at[row_idx, "status"] = "reject"
    elif pres != orig_pres:
        # Inter-model disagreement between Flash and Argon -> route to AMBER Human Gate
        d.at[row_idx, "status"] = "AMBER"
    elif pres == 0 and conf_v >= ARGON_ACCEPT_CONF:
        d.at[row_idx, "status"] = "accept"
        d.at[row_idx, "trust"] = round(0.70 + 0.03 * conf_v * 0.99, 5)
    elif pres == 1 and conf_v >= 0.90 and pv_ok and pw <= 0.15:
        d.at[row_idx, "status"] = "accept"
        d.at[row_idx, "trust"] = round(0.70 + 0.03 * conf_v * 0.99, 5)
    else:
        d.at[row_idx, "status"] = "AMBER"


def prepare_verifier_dataframe(raw_d: pd.DataFrame, idx_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize input label rows into the strict DESIGN.md 1B schema without any reference columns."""
    if raw_d is None or raw_d.empty:
        return pd.DataFrame(columns=DECISION_OUTPUT_COLS)

    df = raw_d.copy()

    # Fill required alias pairs
    if "conf" not in df.columns and "confidence" in df.columns:
        df["conf"] = df["confidence"]
    if "confidence" not in df.columns and "conf" in df.columns:
        df["confidence"] = df["conf"]
    if "evidence" not in df.columns and "rationale" in df.columns:
        df["evidence"] = df["rationale"]
    if "rationale" not in df.columns and "evidence" in df.columns:
        df["rationale"] = df["evidence"]
    if "video" not in df.columns and "video_id" in df.columns:
        df["video"] = df["video_id"]
    if "video_id" not in df.columns and "video" in df.columns:
        df["video_id"] = df["video"]

    # Filter to primary deterministic sample (sample == 0) if multi-sample rows were mixed in
    if "sample" in df.columns:
        sample_num = pd.to_numeric(df["sample"], errors="coerce").fillna(0).astype(int)
        df["sample"] = sample_num
        if (sample_num == 0).any():
            df = df[sample_num == 0].copy()

    # Deduplicate by (frame_id, key) keeping latest run
    if {"frame_id", "key"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["frame_id", "key"], keep="last").reset_index(drop=True)

    # Ensure all canonical input columns exist
    defaults = {
        "video": "",
        "video_id": "",
        "split": "test",
        "mode": "manual",
        "sample": 0,
        "model": C.FLASH,
        "prompt_hash": "",
        "present": 0,
        "conf": 0.0,
        "confidence": 0.0,
        "box": "[]",
        "evidence": "",
        "rationale": "",
        "pv_match": None,
        "pv_conf": None,
        "p_wrong": 0.5,
        "critic": "",
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    # Keep ONLY the allowlisted verifier input columns (zero reference column leakage possible)
    keep_cols = [c for c in VERIFIER_INPUT_COLS if c in df.columns]
    d = df[keep_cols].copy()

    # Sanitize numeric columns
    d["present"] = [int(_safe_present(v)) for v in d["present"]]
    d["conf"] = [_safe_prob(v, default=0.0) for v in d["conf"]]
    d["confidence"] = [_safe_prob(v, default=c) for v, c in zip(d["confidence"], d["conf"])]
    d["p_wrong"] = [_safe_prob(v, default=0.5) for v in d["p_wrong"]]

    # Attach frame_path (and missing split/video metadata) from index.csv using only safe metadata columns
    if idx_df is not None and not idx_df.empty and "frame_id" in idx_df.columns:
        safe_idx_cols = [c for c in ("frame_id", "path", "split", "video") if c in idx_df.columns]
        idx_sub = idx_df[safe_idx_cols].drop_duplicates(subset=["frame_id"]).set_index("frame_id")
        if "path" in idx_sub.columns:
            d["frame_path"] = d["frame_id"].map(idx_sub["path"])
        else:
            d["frame_path"] = ""
        if "split" in idx_sub.columns:
            missing_sp = d["split"].isna() | (d["split"].astype(str).str.strip() == "")
            if missing_sp.any():
                d.loc[missing_sp, "split"] = d.loc[missing_sp, "frame_id"].map(idx_sub["split"])
        if "video" in idx_sub.columns:
            missing_vid = d["video"].isna() | (d["video"].astype(str).str.strip() == "")
            if missing_vid.any():
                d.loc[missing_vid, "video"] = d.loc[missing_vid, "frame_id"].map(idx_sub["video"])
                d.loc[missing_vid, "video_id"] = d.loc[missing_vid, "video"]
    else:
        d["frame_path"] = ""

    d["trust"] = d.apply(trust, axis=1)
    d["status"] = d["trust"].map(status)
    d["final_present"] = d["present"].astype(int)
    d["final_by"] = "flash"
    return d


def summarize_decisions_zero_label(d: pd.DataFrame) -> dict:
    """Compute zero-label summary statistics of verifier routing (overall, test split, and excl. liver)."""
    if d is None or d.empty:
        return {
            "n_total": 0,
            "n_test": 0,
            "n_test_excl_liver": 0,
            "test_excl_liver": {"accept": 0, "AMBER": 0, "reject": 0},
        }
    test = d[d["split"] == "test"] if "split" in d.columns else d
    test_5 = test[test["key"].isin(HEADLINE_KEYS)] if "key" in test.columns else test

    def _counts(sub: pd.DataFrame) -> dict:
        n = len(sub)
        n_acc = int((sub["status"] == "accept").sum()) if n else 0
        n_amb = int((sub["status"] == "AMBER").sum()) if n else 0
        n_rej = int((sub["status"] == "reject").sum()) if n else 0
        return {
            "n": n,
            "n_accept": n_acc,
            "n_amber": n_amb,
            "n_reject": n_rej,
            "coverage": round(n_acc / n, 3) if n else 0.0,
            "amber_share": round(n_amb / n, 3) if n else 0.0,
            "rejected_share": round(n_rej / n, 3) if n else 0.0,
            "escalated_share": round((n_amb + n_rej) / n, 3) if n else 0.0,
        }

    return {
        "n_total": int(len(d)),
        "n_argon_revised": int((d.get("final_by", pd.Series(dtype=str)) == "argon").sum()),
        "all_splits": _counts(d),
        "test_all_6": _counts(test),
        "test_5_informative_excl_liver": _counts(test_5),
    }


def enrich_results_excl_liver(res: dict, d: pd.DataFrame) -> dict:
    """Add 5-label headline accepted metrics (excluding 100%-prevalence liver) and upper-bound metadata to results.json."""
    if not isinstance(res, dict) or d is None or d.empty:
        return res

    test = d[d["split"] == "test"] if "split" in d.columns else d
    test_5 = test[test["key"].isin(HEADLINE_KEYS)] if "key" in test.columns else test
    acc_all = test[test["status"] == "accept"]
    acc_5 = test_5[test_5["status"] == "accept"]
    acc_liver = acc_all[acc_all["key"] == "liver"] if "key" in acc_all.columns else pd.DataFrame()

    n_test_5 = len(test_5)
    n_acc_5 = len(acc_5)
    amb_5 = test_5["status"].isin(["AMBER", "amber", "reanalyse"])
    rej_5 = test_5["status"].isin(["reject", "escalate"])
    esc_5 = amb_5 | rej_5

    per_lbl = res.get("per_label_accepted_acc", {}) or {}
    informative_lbl_accs = [float(per_lbl[k]) for k in HEADLINE_KEYS if k in per_lbl and per_lbl[k] is not None]
    mean_per_label_5 = round(float(np.mean(informative_lbl_accs)), 4) if informative_lbl_accs else None

    # Recover exact pooled 5-label accepted accuracy from all-6 and liver accepted counts
    pooled_acc_5 = mean_per_label_5
    all6_acc = res.get("accepted", {}).get("accuracy")
    liver_acc = per_lbl.get("liver")
    if (
        all6_acc is not None
        and liver_acc is not None
        and len(acc_all) > 0
        and n_acc_5 > 0
        and len(acc_liver) + n_acc_5 == len(acc_all)
    ):
        k_all6 = round(float(all6_acc) * len(acc_all))
        k_liver = round(float(liver_acc) * len(acc_liver))
        k_5 = max(0, min(n_acc_5, k_all6 - k_liver))
        pooled_acc_5 = round(float(k_5 / n_acc_5), 4)

    res["headline_labels"] = HEADLINE_KEYS
    res["excluded_uninformative_labels"] = ["liver"]
    res["accepted_5_informative"] = {
        "n_test_labels_excl_liver": int(n_test_5),
        "n_accepted_excl_liver": int(n_acc_5),
        "coverage": round(n_acc_5 / n_test_5, 3) if n_test_5 else 0.0,
        "accuracy": pooled_acc_5,
        "mean_per_label_accuracy": mean_per_label_5,
        "escalated_share": round(float(esc_5.mean()), 3) if n_test_5 else 0.0,
        "amber_share": round(float(amb_5.mean()), 3) if n_test_5 else 0.0,
        "rejected_share": round(float(rej_5.mean()), 3) if n_test_5 else 0.0,
    }
    if "upper_bound_calibrated" in res and isinstance(res["upper_bound_calibrated"], dict):
        res["upper_bound_calibrated"]["is_upper_bound"] = True
        res["upper_bound_calibrated"]["fitted_on_split"] = "calib"
    res["upper_bound_note"] = (
        "upper_bound_calibrated is fit on the calib split only and is reported strictly as an upper bound. "
        "liver has 100% prevalence in CholecSeg8k and is excluded from headline_labels (accepted_5_informative)."
    )
    return res
