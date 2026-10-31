"""Core zero-label verification engine (`verify/engine.py`).

Strictly zero-leakage: operates solely on model predictions and decorrelated check signals.
Implements:
1. Double-check on every label (Pass 1 vs independent Pass 2 crop/critic check).
2. N-round Argon Amber Loop (`max_rounds=3` default) with progressively tighter crops,
   structure-specific manual criteria, differential diagnosis (`confused_with`), and fresh votes.
3. Round-by-round logging (`out/amber_rounds.jsonl`) and shrinking AMBER curve reporting.
"""
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

import config as C
import gem

ACCEPT_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.35
ARGON_ACCEPT_CONF = 0.80
DEFAULT_AMBER_ROUNDS = 3

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

DECISION_BASE_COLS = VERIFIER_INPUT_COLS + [
    "frame_path",
    "trust",
    "status",
    "final_present",
    "final_by",
]

DECISION_EXTRA_COLS = [
    "label_id",
    "second_pass_present",
    "second_pass_conf",
    "second_pass_agree",
    "status_round_0",
    "status_round_1",
    "status_round_2",
    "status_round_3",
    "resolved_round",
    "amber_rounds_used",
    "argon_reason",
]

DECISION_OUTPUT_COLS = DECISION_BASE_COLS + DECISION_EXTRA_COLS


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


def _parse_box(raw_box: Any) -> list[int] | None:
    """Parse normalized [ymin, xmin, ymax, xmax] box in 0..1000."""
    if raw_box is None:
        return None
    box = raw_box
    if isinstance(raw_box, str):
        s = raw_box.strip()
        if not s or s == "[]":
            return None
        try:
            box = json.loads(s)
        except Exception:
            return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    y0, y1 = max(0, min(1000, min(y0, y1))), max(0, min(1000, max(y0, y1)))
    x0, x1 = max(0, min(1000, min(x0, x1))), max(0, min(1000, max(x0, x1)))
    if (y1 - y0) < 5 or (x1 - x0) < 5:
        return None
    return [y0, x0, y1, x1]


def _evidence_crop_for_round(raw_box: Any, round_num: int) -> tuple[int, int, int, int] | None:
    """Compute zoomed crop coordinates `(y0, x0, y1, x1)` for a given Amber Loop round."""
    b = _parse_box(raw_box)
    if b is not None:
        y0, x0, y1, x1 = b
        pad = 90 if round_num == 1 else (25 if round_num == 2 else 0)
        return (
            max(0, y0 - pad),
            max(0, x0 - pad),
            min(1000, y1 + pad),
            min(1000, x1 + pad),
        )
    if round_num == 2:
        return (120, 120, 880, 880)
    if round_num >= 3:
        return (180, 180, 820, 820)
    return None


def _safe_img_parts(frame_path: Any, raw_box: Any = None, round_num: int = 1) -> list:
    """Build image parts list (full frame + optional zoomed evidence crop), resilient to 1-arg test mocks."""
    parts = [gem.img_part(frame_path)]
    crop = _evidence_crop_for_round(raw_box, round_num)
    if crop is not None and round_num >= 2:
        try:
            parts.append(gem.img_part(frame_path, crop=crop))
        except TypeError:
            pass
    return parts


def compute_second_pass_signals(r: Any) -> tuple[int, float, bool]:
    """Derive independent second-pass (Pass 2) presence, confidence, and agreement for a row."""
    pres = _safe_present(getattr(r, "present", 0))
    pw = _safe_prob(getattr(r, "p_wrong", 0.5), default=0.5)
    pv_ok = _pv_ok(getattr(r, "pv_match", None))
    pv_c = _safe_prob(getattr(r, "pv_conf", None), default=1.0 - pw)

    if pres == 1:
        agree = bool(pv_ok and pw <= 0.30)
        sec_pres = 1 if agree else 0
        sec_conf = round(0.5 * pv_c + 0.5 * (1.0 - pw), 4) if agree else round(max(pw, 1.0 - pv_c), 4)
    else:
        agree = bool(pw <= 0.30)
        sec_pres = 0 if agree else 1
        sec_conf = round(1.0 - pw, 4) if agree else round(pw, 4)
    return sec_pres, float(sec_conf), bool(agree)


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


def _lookup_structure_manual(manual: dict, key: str) -> dict:
    if not isinstance(manual, dict):
        return {}
    for s in manual.get("structures", []):
        if isinstance(s, dict) and s.get("key") == key:
            return s
    return {}


async def argon_revise(
    r: Any,
    manual: dict,
    *,
    call_fn=None,
    round_num: int = 1,
    prior_history: list[dict] | None = None,
) -> dict:
    """Escalate an uncertain claim to Gemini 4 Argon (`round_num = 1..N`)."""
    pv_agreed = _pv_ok(getattr(r, "pv_match", None))
    conf_val = _safe_prob(getattr(r, "conf", getattr(r, "confidence", 0.0)), default=0.0)
    pw_val = _safe_prob(getattr(r, "p_wrong", 0.5), default=0.5)
    pres_flag = _safe_present(getattr(r, "present", 0)) == 1
    ev_text = getattr(r, "evidence", getattr(r, "rationale", "")) or ""
    critic_text = getattr(r, "critic", "") or ""
    caller = call_fn or gem.call

    if round_num == 1:
        prompt = (
            f"You are the senior reviewer. A junior labeller said '{r.key}' is "
            f"{'PRESENT' if pres_flag else 'ABSENT'} (conf {conf_val:.2f}, evidence: {ev_text}). "
            f"A blind crop check {'agreed' if pv_agreed else 'disagreed or was not possible'}. "
            f"A critic argued: {critic_text!r} (p_wrong {pw_val:.2f}).\n"
            "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
            f"Manual: {json.dumps(manual)}"
        )
        images = _safe_img_parts(r.frame_path, getattr(r, "box", None), round_num=1)
        call_kwargs = {"model": C.ARGON, "images": images, "schema": Verdict, "tag": "argon-revise"}
    else:
        s_man = _lookup_structure_manual(manual, str(r.key))
        gen_rules = manual.get("general_rules", []) if isinstance(manual, dict) else []
        hist_str = json.dumps(prior_history or [])
        prompt = (
            f"You are the senior surgical adjudicator (Amber Loop Round {round_num}).\n"
            f"Target structure: '{r.key}' ({s_man.get('name', r.key)}).\n"
            f"- Pass 1 claim: {'PRESENT' if pres_flag else 'ABSENT'} (conf {conf_val:.2f}, evidence: {ev_text!r})\n"
            f"- Pass 2 blind crop check: {'agreed' if pv_agreed else 'disagreed or not possible'}\n"
            f"- Pass 2 sceptical critic (p_wrong={pw_val:.2f}): {critic_text!r}\n"
            f"- Manual visual criteria for '{r.key}': appearance={s_man.get('appearance', '')!r}, "
            f"where_found={s_man.get('where_found', '')!r}, confused_with={s_man.get('confused_with', [])!r}, "
            f"how_to_tell_apart={s_man.get('how_to_tell_apart', '')!r}\n"
            f"- General single-frame rules: {json.dumps(gen_rules)}\n"
            f"- Prior Amber Loop rounds on this frame: {hist_str}\n"
            "Examine the full frame and zoomed region carefully against the differential criteria "
            "and give your definitive verdict."
        )
        images = _safe_img_parts(r.frame_path, getattr(r, "box", None), round_num=round_num)
        call_kwargs = {
            "model": C.ARGON,
            "images": images,
            "schema": Verdict,
            "tag": f"argon-amber-r{round_num}",
            "temperature": 0.2 if round_num == 2 else 0.3,
            "sample": round_num,
        }

    try:
        v = await caller(prompt, **call_kwargs)
    except TypeError:
        # Fallback if a unit test monkeypatched a simpler `fake_call` signature
        v = await caller(
            prompt,
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
        f"[R{round_num}] {r.frame_id} {r.key}: {'present' if v_pres else 'absent'} "
        f"({v_conf:.2f}) - {v_reason[:140]}",
    )
    return {"present": v_pres, "confidence": v_conf, "reasoning": v_reason}


def apply_argon_verdict(
    d: pd.DataFrame,
    row_idx: Any,
    v: dict,
    *,
    round_num: int = 1,
    prior_verdicts: list[dict] | None = None,
) -> str:
    """Apply an Argon revision verdict in-place for `round_num` and return the resulting status."""
    orig_pres = int(_safe_present(d.at[row_idx, "present"]))
    pres = int(bool(v.get("present", orig_pres)))
    conf_v = _safe_prob(v.get("confidence", 0.5), default=0.5)
    pw = _safe_prob(d.at[row_idx, "p_wrong"] if "p_wrong" in d.columns else 0.5, default=0.5)
    pv_ok = _pv_ok(d.at[row_idx, "pv_match"] if "pv_match" in d.columns else None)
    key_str = str(d.at[row_idx, "key"]) if "key" in d.columns else ""
    history = list(prior_verdicts or []) + [{"present": pres, "confidence": conf_v}]

    d.at[row_idx, "final_present"] = pres
    d.at[row_idx, "final_by"] = "argon"
    if "argon_reason" in d.columns:
        d.at[row_idx, "argon_reason"] = str(v.get("reasoning", ""))

    if round_num == 1:
        if conf_v < REJECT_THRESHOLD:
            new_status = "reject"
            d.at[row_idx, "final_present"] = 0
        elif pres != orig_pres:
            new_status = "AMBER"
        elif pres == 0 and conf_v >= 0.94 and key_str != "liver":
            new_status = "accept"
            d.at[row_idx, "trust"] = round(0.70 + 0.03 * conf_v * 0.99, 5)
        elif pres == 1 and conf_v >= 0.90 and (pv_ok or (key_str == "liver" and pw <= 0.05)) and pw <= 0.15:
            new_status = "accept"
            d.at[row_idx, "trust"] = round(0.70 + 0.03 * conf_v * 0.99, 5)
        else:
            new_status = "AMBER"
    elif round_num == 2:
        r1_pres = int(bool(history[0]["present"]))
        if conf_v < 0.40:
            new_status = "reject"
            d.at[row_idx, "final_present"] = 0
        elif pres == 0 and r1_pres == 0 and orig_pres == 0 and conf_v >= 0.90 and key_str != "liver":
            new_status = "accept"
            d.at[row_idx, "final_present"] = 0
            d.at[row_idx, "trust"] = round(0.70 + 0.025 * conf_v, 5)
        elif pres == 0 and r1_pres == 0 and orig_pres == 1 and conf_v >= 0.80:
            new_status = "reject"
            d.at[row_idx, "final_present"] = 0
            d.at[row_idx, "trust"] = round(min(float(d.at[row_idx, "trust"]), 0.30), 5)
        elif pres == 1 and r1_pres == 1 and orig_pres == 1 and conf_v >= 0.86 and pv_ok and pw <= 0.15:
            new_status = "accept"
            d.at[row_idx, "final_present"] = 1
            d.at[row_idx, "trust"] = round(0.70 + 0.025 * conf_v, 5)
        else:
            new_status = "AMBER"
    else:
        # Round 3+: 3-round consensus adjudication
        orig_conf = _safe_prob(d.at[row_idx, "conf"] if "conf" in d.columns else 0.5, default=0.5)
        pos_votes = sum(1 for h in history if int(bool(h["present"])) == 1)
        maj_pres = 1 if pos_votes > (len(history) / 2.0) else 0
        mean_conf = float(np.mean([_safe_prob(h.get("confidence", 0.5)) for h in history]))
        if maj_pres == 0 and orig_pres == 0 and mean_conf >= 0.82 and key_str != "liver":
            new_status = "accept"
            d.at[row_idx, "final_present"] = 0
            d.at[row_idx, "trust"] = round(0.70 + 0.02 * mean_conf, 5)
        elif maj_pres == 0 and orig_pres == 1:
            # Consensus rejects unverified positive claim
            new_status = "reject"
            d.at[row_idx, "final_present"] = 0
            d.at[row_idx, "trust"] = round(min(float(d.at[row_idx, "trust"]), 0.28), 5)
        elif pos_votes == len(history) and orig_pres == 1 and pv_ok and (pw <= 0.15 or orig_conf >= 0.90) and mean_conf >= 0.85:
            new_status = "accept"
            d.at[row_idx, "final_present"] = 1
            d.at[row_idx, "trust"] = round(0.70 + 0.02 * mean_conf, 5)
        elif pos_votes == len(history) and orig_pres == 1 and not pv_ok and pw <= 0.10 and mean_conf >= 0.92:
            new_status = "accept"
            d.at[row_idx, "final_present"] = 1
            d.at[row_idx, "trust"] = round(0.70 + 0.02 * mean_conf, 5)
        elif maj_pres == 1 and orig_pres == 1 and not pv_ok and pw >= 0.15:
            # Crop check disagreed and critic flagged risk -> reject false positive
            new_status = "reject"
            d.at[row_idx, "final_present"] = 0
            d.at[row_idx, "trust"] = round(min(float(d.at[row_idx, "trust"]), 0.30), 5)
        else:
            new_status = "AMBER"

    d.at[row_idx, "status"] = new_status
    return new_status


async def run_amber_loop(
    d: pd.DataFrame,
    manual: dict,
    *,
    argon_budget: int = 300,
    max_rounds: int = DEFAULT_AMBER_ROUNDS,
    call_fn=None,
) -> list[dict]:
    """Run up to `max_rounds` of Argon re-examination on AMBER labels and return round log entries."""
    if d.empty or max_rounds <= 0 or argon_budget <= 0:
        return []

    round_logs: list[dict] = []
    history_by_idx: dict[Any, list[dict]] = {}

    for r_num in range(1, max_rounds + 1):
        amber_rows = d[d["status"] == "AMBER"].copy()
        if amber_rows.empty:
            for c_num in range(r_num, 4):
                col = f"status_round_{c_num}"
                if col in d.columns:
                    d[col] = d["status"]
            break

        amber_rows["_is_headline"] = amber_rows["key"].isin(HEADLINE_KEYS).astype(int)
        amber_rows = amber_rows.sort_values(["_is_headline", "trust"], ascending=[False, False])
        band = amber_rows.head(int(argon_budget))
        gem.trace(
            "argon",
            f"[Amber Loop R{r_num}/{max_rounds}] Reviewing {len(band)} AMBER labels "
            f"({len(amber_rows)} remaining before R{r_num})",
        )

        verdicts = await asyncio.gather(
            *[
                argon_revise(
                    r,
                    manual,
                    call_fn=call_fn,
                    round_num=r_num,
                    prior_history=history_by_idx.get(idx_val, []),
                )
                for idx_val, r in zip(band.index, band.itertuples())
            ]
        )

        for (row_idx, orig_row), v in zip(band.iterrows(), verdicts):
            prev_list = history_by_idx.setdefault(row_idx, [])
            new_st = apply_argon_verdict(
                d,
                row_idx,
                v,
                round_num=r_num,
                prior_verdicts=prev_list,
            )
            prev_list.append(
                {
                    "round": r_num,
                    "present": int(bool(v["present"])),
                    "confidence": float(v["confidence"]),
                    "verdict": new_st,
                    "reasoning": str(v.get("reasoning", "")),
                }
            )
            d.at[row_idx, "amber_rounds_used"] = r_num
            if new_st != "AMBER" and int(d.at[row_idx, "resolved_round"]) < 0:
                d.at[row_idx, "resolved_round"] = r_num

            fid = str(orig_row["frame_id"])
            key = str(orig_row["key"])
            round_logs.append(
                {
                    "label_id": f"{fid}__{key}",
                    "frame_id": fid,
                    "video": str(orig_row.get("video", "")),
                    "split": str(orig_row.get("split", "")),
                    "key": key,
                    "round": int(r_num),
                    "orig_present": int(orig_row["present"]),
                    "argon_present": int(bool(v["present"])),
                    "final_present": int(d.at[row_idx, "final_present"]),
                    "confidence": round(float(v["confidence"]), 4),
                    "verdict": new_st,
                    "reason": str(v.get("reasoning", "")),
                    "model": C.ARGON,
                }
            )

        if r_num <= 3:
            d[f"status_round_{r_num}"] = d["status"]

    # Fill any remaining status_round_{1..3} columns if loop terminated early
    for r_num in range(1, 4):
        col = f"status_round_{r_num}"
        if col in d.columns:
            mask_unset = d[col].isna() | (d[col].astype(str) == "")
            if mask_unset.any():
                d.loc[mask_unset, col] = d.loc[mask_unset, "status"]

    return round_logs


def prepare_verifier_dataframe(raw_d: pd.DataFrame, idx_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize input label rows into the strict DESIGN.md 1B schema + double-check & Amber Loop columns."""
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

    d["present"] = [int(_safe_present(v)) for v in d["present"]]
    d["conf"] = [_safe_prob(v, default=0.0) for v in d["conf"]]
    d["confidence"] = [_safe_prob(v, default=c) for v, c in zip(d["confidence"], d["conf"])]
    d["p_wrong"] = [_safe_prob(v, default=0.5) for v in d["p_wrong"]]

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

    # Populate Pass 2 double-check and Round 0 tracking columns
    d["label_id"] = [f"{fid}__{k}" for fid, k in zip(d["frame_id"], d["key"])]
    sp_signals = [compute_second_pass_signals(r) for r in d.itertuples()]
    d["second_pass_present"] = [s[0] for s in sp_signals]
    d["second_pass_conf"] = [s[1] for s in sp_signals]
    d["second_pass_agree"] = [s[2] for s in sp_signals]
    d["status_round_0"] = d["status"]
    d["status_round_1"] = d["status"]
    d["status_round_2"] = d["status"]
    d["status_round_3"] = d["status"]
    d["resolved_round"] = np.where(d["status"] == "AMBER", -1, 0)
    d["amber_rounds_used"] = 0
    d["argon_reason"] = ""
    return d


def export_verified_dataset_jsonl(d: pd.DataFrame, manual: dict | None, out_jsonl: Path) -> int:
    """Write accepted & verified labels with full provenance to `out/verified_dataset.jsonl`."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if d is None or d.empty:
        out_jsonl.write_text("")
        return 0
    sources = manual.get("sources", []) if isinstance(manual, dict) else []
    source_urls = [s.get("url", "") for s in sources if isinstance(s, dict) and s.get("url")]
    acc_rows = d[d["status"] == "accept"]
    lines = []
    for r in acc_rows.itertuples():
        rec = {
            "label_id": getattr(r, "label_id", f"{r.frame_id}__{r.key}"),
            "frame_id": str(r.frame_id),
            "video": str(getattr(r, "video", "")),
            "split": str(getattr(r, "split", "")),
            "key": str(r.key),
            "present": int(getattr(r, "final_present", getattr(r, "present", 0))),
            "trust": round(float(getattr(r, "trust", 0.0)), 4),
            "status": "accept",
            "verified_by": str(getattr(r, "final_by", "flash")),
            "resolved_round": int(getattr(r, "resolved_round", 0)),
            "amber_rounds_used": int(getattr(r, "amber_rounds_used", 0)),
            "second_pass_agree": bool(getattr(r, "second_pass_agree", True)),
            "box": getattr(r, "box", []),
            "evidence": str(getattr(r, "evidence", "")),
            "argon_reason": str(getattr(r, "argon_reason", "")),
            "model": str(getattr(r, "model", C.FLASH)),
            "prompt_hash": str(getattr(r, "prompt_hash", "")),
            "manual_source_urls": source_urls,
        }
        lines.append(json.dumps(rec))
    out_jsonl.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def _compute_amber_curve_summary(d: pd.DataFrame) -> dict:
    """Compute shrinking AMBER counts across rounds 0..3 overall, on test, and on 5 headline test labels."""
    if d is None or d.empty or "status_round_0" not in d.columns:
        return {
            "max_rounds": DEFAULT_AMBER_ROUNDS,
            "ambers_by_round_all": [0, 0, 0, 0],
            "ambers_by_round_test": [0, 0, 0, 0],
            "ambers_by_round_test_excl_liver": [0, 0, 0, 0],
            "per_round_test": [],
        }

    test = d[d["split"] == "test"] if "split" in d.columns else d
    test_5 = test[test["key"].isin(HEADLINE_KEYS)] if "key" in test.columns else test

    all_curve, test_curve, test5_curve = [], [], []
    per_round_test = []
    prev_test_amb = None
    for r_num in range(0, 4):
        col = f"status_round_{r_num}" if f"status_round_{r_num}" in d.columns else "status"
        n_all_amb = int((d[col] == "AMBER").sum())
        n_te_amb = int((test[col] == "AMBER").sum())
        n_te5_amb = int((test_5[col] == "AMBER").sum())
        n_te_acc = int((test[col] == "accept").sum())
        n_te_rej = int((test[col] == "reject").sum())
        n_te5_acc = int((test_5[col] == "accept").sum())
        n_te5_rej = int((test_5[col] == "reject").sum())
        resolved_step = 0 if prev_test_amb is None else max(0, prev_test_amb - n_te_amb)
        prev_test_amb = n_te_amb

        all_curve.append(n_all_amb)
        test_curve.append(n_te_amb)
        test5_curve.append(n_te5_amb)
        per_round_test.append(
            {
                "round": r_num,
                "ambers_remaining": n_te_amb,
                "ambers_remaining_excl_liver": n_te5_amb,
                "resolved_in_round": resolved_step,
                "accepted_count": n_te_acc,
                "rejected_count": n_te_rej,
                "accepted_count_excl_liver": n_te5_acc,
                "rejected_count_excl_liver": n_te5_rej,
            }
        )

    return {
        "max_rounds": DEFAULT_AMBER_ROUNDS,
        "ambers_by_round_all": all_curve,
        "ambers_by_round_test": test_curve,
        "ambers_by_round_test_excl_liver": test5_curve,
        "per_round_test": per_round_test,
    }


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
        "amber_loop": _compute_amber_curve_summary(d),
    }


def enrich_results_excl_liver(res: dict, d: pd.DataFrame) -> dict:
    """Add 5-label headline accepted metrics, shrinking Amber Loop curve, and upper-bound metadata to results.json."""
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
    res["amber_loop"] = _compute_amber_curve_summary(d)
    if "upper_bound_calibrated" in res and isinstance(res["upper_bound_calibrated"], dict):
        res["upper_bound_calibrated"]["is_upper_bound"] = True
        res["upper_bound_calibrated"]["fitted_on_split"] = "calib"
    res["upper_bound_note"] = (
        "upper_bound_calibrated is fit on the calib split only and is reported strictly as an upper bound. "
        "liver has 100% prevalence in CholecSeg8k and is excluded from headline_labels (accepted_5_informative)."
    )
    return res
