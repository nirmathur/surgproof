"""SurgProof Release Dataset Sign-Off / Human Gate (pages/2_Human_gate.py) — Issue #14.

Judging test: "People stay in control" — a visible human approval step that blocks
dataset release until a human acts.

Positioning (VISION.md & docs/DESIGN.md 1D):
SurgProof removes the human from seeding (manual learned from public video), labelling,
correction (the multi-round Amber Loop), and review (independent double-check).
The output is a verified dataset (out/verified_dataset.jsonl + per-label provenance).
People stay in control with a FINAL PASS before anything becomes proper data:
the human reviews any AMBERs left after the Amber Loop plus a random audit sample
of auto-accepted labels (with their evidence crops), and only then releases the dataset.

Strictly zero-leakage: never reads, joins, or displays ground truth.
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

import config as C

# Allowlisted columns from docs/DESIGN.md 1B (never includes any reference/evaluation columns)
SAFE_DECISION_COLS = [
    "label_id",
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
    "second_pass_present",
    "second_pass_conf",
    "second_pass_agree",
    "pass2_present",
    "pass2_conf",
    "pass2_agree",
    "initial_status",
    "status_round_0",
    "status_round_1",
    "status_round_2",
    "status_round_3",
    "resolved_round",
    "amber_rounds_used",
    "argon_reason",
    "argon_reasoning",
    "frame_path",
    "trust",
    "status",
    "final_present",
    "final_by",
    "amber_round",
]

SAFE_INDEX_COLS = {"frame_id", "path", "video", "split", "clip"}

STATUS_COLORS = {
    "accept": "#16a34a",
    "AMBER": "#d97706",
    "reject": "#dc2626",
}

ACCEPT_THRESHOLD = 0.70
REJECT_THRESHOLD = 0.35


# ==============================================================================
# 1. Pure Zero-Leakage Helpers & Verifier Fallback
# ==============================================================================
def normalize_status(val: Any) -> str:
    """Normalize verifier status to canonical 'accept' | 'AMBER' | 'reject'."""
    s = str(val or "").strip().lower()
    if s == "accept":
        return "accept"
    if s in ("reject", "escalate"):
        return "reject"
    return "AMBER"


def _safe_prob(val: Any, default: float = 0.5) -> float:
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


def _pv_ok(val: Any) -> bool:
    if val is None or isinstance(val, (list, tuple, dict, set)):
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "1.0", "yes", "y", "t")
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, float, np.number)):
        return float(val) != 0.0
    return bool(val)


def compute_zero_label_trust(row: Any) -> float:
    """Zero-label verification score matching accept.py / verify/engine.py."""
    raw_conf = getattr(row, "conf", getattr(row, "confidence", 0.0))
    raw_pw = getattr(row, "p_wrong", 0.5)
    c = _safe_prob(raw_conf, default=0.0)
    pw = _safe_prob(raw_pw, default=0.5)
    s = c * (1.0 - pw)
    if _safe_present(getattr(row, "present", 0)) == 1:
        s *= 1.0 if _pv_ok(getattr(row, "pv_match", None)) else 0.5
    return round(float(s), 4)


def map_trust_to_status(
    s: Any,
    accept_thresh: float = ACCEPT_THRESHOLD,
    reject_thresh: float = REJECT_THRESHOLD,
) -> str:
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


def load_safe_index(data_dir: Path | None = None) -> pd.DataFrame:
    """Read ONLY safe metadata columns (frame_id, path, video, split) from data/index.csv."""
    d_dir = Path(data_dir) if data_dir is not None else C.DATA_DIR
    idx_path = d_dir / "index.csv"
    if not idx_path.is_file():
        return pd.DataFrame(columns=["frame_id", "path", "video", "split"])
    try:
        return pd.read_csv(idx_path, usecols=lambda c: c in SAFE_INDEX_COLS)
    except Exception:
        return pd.DataFrame(columns=["frame_id", "path", "video", "split"])


def load_manual_sources(out_dir: Path | None = None, coord_dir: Path | None = None) -> list[str]:
    """Extract source URLs/titles from out/manual.json for dataset provenance."""
    o_dir = Path(out_dir) if out_dir is not None else C.OUT_DIR
    c_dir = Path(coord_dir) if coord_dir is not None else (C.ROOT / "coord" / "artifacts")
    for cand in (o_dir / "manual.json", c_dir / "manual.json"):
        if cand.is_file():
            try:
                man = json.loads(cand.read_text(encoding="utf-8"))
                srcs = man.get("sources", [])
                out = []
                for s in srcs:
                    if isinstance(s, dict):
                        out.append(str(s.get("url") or s.get("title") or "manual_source"))
                    elif s:
                        out.append(str(s))
                if out:
                    return out
            except Exception:
                pass
    return ["out/manual.json"]


def sanitize_decisions_df(raw_df: pd.DataFrame | None, idx_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize a decisions or smoke-labels DataFrame into the strict leak-free DESIGN.md 1B schema."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(columns=SAFE_DECISION_COLS)

    df = raw_df.copy()

    if "conf" not in df.columns and "confidence" in df.columns:
        df["conf"] = df["confidence"]
    if "confidence" not in df.columns and "conf" in df.columns:
        df["confidence"] = df["conf"]
    if "rationale" not in df.columns and "evidence" in df.columns:
        df["rationale"] = df["evidence"]
    if "evidence" not in df.columns and "rationale" in df.columns:
        df["evidence"] = df["rationale"]
    if "video" not in df.columns and "video_id" in df.columns:
        df["video"] = df["video_id"]
    if "video_id" not in df.columns and "video" in df.columns:
        df["video_id"] = df["video"]
    if "argon_reasoning" not in df.columns and "argon_reason" in df.columns:
        df["argon_reasoning"] = df["argon_reason"]
    if "argon_reason" not in df.columns and "argon_reasoning" in df.columns:
        df["argon_reason"] = df["argon_reasoning"]

    if "sample" in df.columns:
        s_num = pd.to_numeric(df["sample"], errors="coerce").fillna(0).astype(int)
        df["sample"] = s_num
        if (s_num == 0).any():
            df = df[s_num == 0].copy()

    if {"frame_id", "key"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["frame_id", "key"], keep="last").reset_index(drop=True)

    defaults = {
        "frame_id": "",
        "video": "",
        "video_id": "",
        "split": "test",
        "key": "",
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
        "frame_path": "",
    }
    for col, default_val in defaults.items():
        if col not in df.columns:
            df[col] = default_val

    df["present"] = [int(_safe_present(v)) for v in df["present"]]
    df["conf"] = [_safe_prob(v, default=0.0) for v in df["conf"]]
    df["confidence"] = [_safe_prob(v, default=c) for v, c in zip(df["confidence"], df["conf"])]
    df["p_wrong"] = [_safe_prob(v, default=0.5) for v in df["p_wrong"]]

    if "trust" not in df.columns:
        df["trust"] = [compute_zero_label_trust(r) for r in df.itertuples()]
    else:
        df["trust"] = [_safe_prob(v, default=0.5) for v in df["trust"]]

    if "status" not in df.columns:
        df["status"] = [map_trust_to_status(t) for t in df["trust"]]
    else:
        df["status"] = [normalize_status(s) for s in df["status"]]

    if "final_present" not in df.columns:
        df["final_present"] = df["present"].astype(int)
    else:
        df["final_present"] = [
            int(_safe_present(fp)) if pd.notna(fp) else int(p)
            for fp, p in zip(df["final_present"], df["present"])
        ]

    if "final_by" not in df.columns:
        df["final_by"] = "flash"
    else:
        df["final_by"] = df["final_by"].fillna("flash").astype(str)

    if "argon_reasoning" not in df.columns:
        df["argon_reasoning"] = ""

    if idx_df is not None and not idx_df.empty and "frame_id" in idx_df.columns:
        idx_sub = idx_df.drop_duplicates(subset=["frame_id"]).set_index("frame_id")
        if "path" in idx_sub.columns:
            missing_fp = df["frame_path"].isna() | (df["frame_path"].astype(str).str.strip().isin(["", "nan", "None"]))
            if missing_fp.any():
                df.loc[missing_fp, "frame_path"] = df.loc[missing_fp, "frame_id"].map(idx_sub["path"]).fillna("")
        if "video" in idx_sub.columns:
            missing_v = df["video"].isna() | (df["video"].astype(str).str.strip().isin(["", "nan", "None"]))
            if missing_v.any():
                df.loc[missing_v, "video"] = df.loc[missing_v, "frame_id"].map(idx_sub["video"]).fillna("")
                df.loc[missing_v, "video_id"] = df.loc[missing_v, "video"]
        if "split" in idx_sub.columns:
            missing_sp = df["split"].isna() | (df["split"].astype(str).str.strip().isin(["", "nan", "None"]))
            if missing_sp.any():
                df.loc[missing_sp, "split"] = df.loc[missing_sp, "frame_id"].map(idx_sub["split"]).fillna("test")

    keep = [c for c in SAFE_DECISION_COLS if c in df.columns]
    return df[keep].copy()


def discover_decision_sources(
    out_dir: Path | None = None,
    coord_dir: Path | None = None,
) -> list[tuple[str, Path, str]]:
    """Return available real decision/smoke sources as (label, path, kind) tuples."""
    o_dir = Path(out_dir) if out_dir is not None else C.OUT_DIR
    c_dir = Path(coord_dir) if coord_dir is not None else (C.ROOT / "coord" / "artifacts")

    sources: list[tuple[str, Path, str]] = []
    primary_csv = o_dir / "decisions.csv"
    coord_csv = c_dir / "decisions.csv"
    live_csv = o_dir / "live_decisions.csv"
    manual_jsonl = o_dir / "labels_manual.jsonl"
    smoke_jsonl = c_dir / "labels_smoke.jsonl"
    blank_jsonl = o_dir / "labels_blank.jsonl"

    if primary_csv.is_file() and primary_csv.stat().st_size > 0:
        sources.append(("Primary Verifier Output (out/decisions.csv)", primary_csv, "csv"))
    elif coord_csv.is_file() and coord_csv.stat().st_size > 0:
        sources.append(("Shared Verifier Output (coord/artifacts/decisions.csv)", coord_csv, "csv"))

    if manual_jsonl.is_file() and manual_jsonl.stat().st_size > 0:
        sources.append(("Manual Labels via Verifier (out/labels_manual.jsonl)", manual_jsonl, "jsonl"))
    if smoke_jsonl.is_file() and smoke_jsonl.stat().st_size > 0:
        if not manual_jsonl.is_file() or smoke_jsonl.stat().st_size != manual_jsonl.stat().st_size:
            sources.append(("Smoke Labels via Verifier (coord/artifacts/labels_smoke.jsonl)", smoke_jsonl, "jsonl"))

    if live_csv.is_file() and live_csv.stat().st_size > 0:
        if not primary_csv.is_file() or live_csv.resolve() != primary_csv.resolve():
            sources.append(("Live Demo Run (out/live_decisions.csv)", live_csv, "csv"))

    if not sources and blank_jsonl.is_file() and blank_jsonl.stat().st_size > 0:
        sources.append(("Blank Labels via Verifier (out/labels_blank.jsonl)", blank_jsonl, "jsonl"))

    return sources


def load_decisions_from_source(
    source_path: Path | str,
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Load a CSV or JSONL source and pass it through the zero-leakage verifier sanitizer."""
    p = Path(source_path)
    if not p.is_file() or p.stat().st_size == 0:
        return pd.DataFrame(columns=SAFE_DECISION_COLS)
    idx_df = load_safe_index(data_dir)
    try:
        if p.suffix.lower() == ".jsonl":
            raw_df = pd.read_json(p, lines=True)
        else:
            raw_df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame(columns=SAFE_DECISION_COLS)
    return sanitize_decisions_df(raw_df, idx_df)


# ==============================================================================
# 2. Amber Loop Funnel (out/amber_rounds.jsonl) & Random Audit Sampler
# ==============================================================================
def load_amber_rounds(out_dir: Path | None = None, coord_dir: Path | None = None) -> list[dict]:
    """Load per-round Amber Loop resolution trace from out/amber_rounds.jsonl if present."""
    o_dir = Path(out_dir) if out_dir is not None else C.OUT_DIR
    c_dir = Path(coord_dir) if coord_dir is not None else (C.ROOT / "coord" / "artifacts")
    for cand in (o_dir / "amber_rounds.jsonl", c_dir / "amber_rounds.jsonl"):
        if cand.is_file() and cand.stat().st_size > 0:
            return _read_jsonl_file(cand)
    return []


def compute_amber_funnel(
    dec_df: pd.DataFrame | None,
    amber_rounds: list[dict] | None = None,
    n_amber_pending_after_human: int | None = None,
) -> list[dict]:
    """Compute the round-by-round shrinking AMBER funnel (Initial Double-Check -> Amber Loop Rounds -> Human Sign-Off)."""
    if dec_df is None or dec_df.empty:
        return []
    df = sanitize_decisions_df(dec_df)
    final_amber = int((df["status"].map(normalize_status) == "AMBER").sum())

    if "status_round_0" in df.columns and df["status_round_0"].notna().any():
        r0_amber = int((df["status_round_0"].map(normalize_status) == "AMBER").sum())
    else:
        initial_trusts = [compute_zero_label_trust(r) for r in df.itertuples()]
        initial_statuses = [map_trust_to_status(t) for t in initial_trusts]
        r0_amber = sum(1 for s in initial_statuses if s == "AMBER")

    rounds_list = amber_rounds or []
    by_round: dict[int, list[dict]] = {}
    for rec in rounds_list:
        if isinstance(rec, dict):
            r_num = int(rec.get("round", 1) or 1)
            by_round.setdefault(r_num, []).append(rec)

    if 1 in by_round:
        r0_amber = max(r0_amber, len(by_round[1]))

    stages: list[dict] = [
        {
            "stage": "Round 0 · Independent Double-Check",
            "ambers_remaining": int(r0_amber),
            "resolved_in_step": 0,
            "note": f"n={len(df)} labels double-checked",
        }
    ]

    round_cols = [f"status_round_{k}" for k in (1, 2, 3) if f"status_round_{k}" in df.columns and df[f"status_round_{k}"].notna().any()]
    if round_cols:
        prev_amber = int(r0_amber)
        for k_idx, col_name in enumerate(round_cols, start=1):
            rem_k = int((df[col_name].map(normalize_status) == "AMBER").sum())
            res_k = max(0, prev_amber - rem_k)
            stages.append({
                "stage": f"Amber Loop · Round {k_idx}",
                "ambers_remaining": rem_k,
                "resolved_in_step": res_k,
                "note": f"Argon resolved {res_k} -> {rem_k} AMBER left",
            })
            prev_amber = rem_k
    elif by_round:
        running_amber = r0_amber
        for r_num in sorted(by_round.keys()):
            evts = by_round[r_num]
            resolved_here = sum(
                1
                for e in evts
                if normalize_status(e.get("verdict", e.get("status", "AMBER"))) in ("accept", "reject")
            )
            still_amber_here = sum(
                1
                for e in evts
                if normalize_status(e.get("verdict", e.get("status", "AMBER"))) == "AMBER"
            )
            if still_amber_here + resolved_here == len(evts) and len(evts) == running_amber:
                running_amber = still_amber_here
            else:
                running_amber = max(final_amber, running_amber - resolved_here)
            stages.append({
                "stage": f"Amber Loop · Round {r_num}",
                "ambers_remaining": int(running_amber),
                "resolved_in_step": int(resolved_here),
                "note": f"Argon inspected {len(evts)}, resolved {resolved_here}",
            })
    else:
        n_argon_resolved = max(0, r0_amber - final_amber)
        stages.append({
            "stage": "Amber Loop · Argon Re-Examination",
            "ambers_remaining": int(final_amber),
            "resolved_in_step": int(n_argon_resolved),
            "note": f"Argon reviewed uncertain band -> {final_amber} residual AMBER(s)",
        })

    if n_amber_pending_after_human is not None:
        stages.append({
            "stage": "Final Pass · Release Dataset Gate",
            "ambers_remaining": int(n_amber_pending_after_human),
            "resolved_in_step": int(max(0, final_amber - n_amber_pending_after_human)),
            "note": "Human clears residual AMBERs & signs off release",
        })
    return stages


def sample_accepted_for_audit(
    dec_df: pd.DataFrame | None,
    n_sample: int = 20,
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministically sample auto-accepted labels (prioritizing positive claims with boxes) for human spot-check."""
    if dec_df is None or dec_df.empty or n_sample <= 0:
        return pd.DataFrame(columns=SAFE_DECISION_COLS)
    df = sanitize_decisions_df(dec_df)
    acc = df[df["status"].map(normalize_status) == "accept"].copy()
    if acc.empty:
        return acc
    has_box = [parse_box(b) is not None and int(fp) == 1 for b, fp in zip(acc["box"], acc["final_present"])]
    acc["_has_box"] = has_box
    with_box = acc[acc["_has_box"]].drop(columns=["_has_box"])
    without_box = acc[~acc["_has_box"]].drop(columns=["_has_box"])

    if len(with_box) >= n_sample:
        return with_box.sample(n=n_sample, random_state=seed).reset_index(drop=True)
    needed = n_sample - len(with_box)
    rest = without_box.sample(n=min(needed, len(without_box)), random_state=seed) if not without_box.empty else without_box
    return pd.concat([with_box, rest], ignore_index=True)


# ==============================================================================
# 3. Audit Log (out/gate_log.jsonl + out/human_gate_log.jsonl) & Lock Engine
# ==============================================================================
def _default_log_paths(out_dir: Path | None = None) -> tuple[Path, Path]:
    o_dir = Path(out_dir) if out_dir is not None else C.OUT_DIR
    return o_dir / "gate_log.jsonl", o_dir / "human_gate_log.jsonl"


def _read_jsonl_file(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                continue
    except Exception:
        return []
    return records


def load_gate_log(log_path: Path | str | None = None, out_dir: Path | None = None) -> list[dict]:
    """Load human gate audit entries from out/gate_log.jsonl (and out/human_gate_log.jsonl)."""
    if log_path is not None:
        return _read_jsonl_file(Path(log_path))

    primary_p, mirror_p = _default_log_paths(out_dir)
    primary_recs = _read_jsonl_file(primary_p)
    mirror_recs = _read_jsonl_file(mirror_p)
    if not mirror_recs:
        return primary_recs
    if not primary_recs:
        return mirror_recs

    seen = set()
    merged = []
    for rec in primary_recs + mirror_recs:
        sig = (
            str(rec.get("frame_id", "")),
            str(rec.get("key", "")),
            str(rec.get("action", rec.get("human_action", ""))),
            str(rec.get("timestamp", rec.get("when", rec.get("t", "")))),
        )
        if sig not in seen:
            seen.add(sig)
            merged.append(rec)
    return merged


def record_gate_decision(
    reviewer: str,
    frame_id: str,
    key: str,
    action: str,
    human_present: int | None,
    *,
    item_type: str = "amber",
    video: str = "",
    split: str = "test",
    agent_status: str = "AMBER",
    model_present: int = 0,
    conf: float = 0.0,
    rationale: str = "",
    log_path: Path | str | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Append a human gate decision to out/gate_log.jsonl (and mirror to out/human_gate_log.jsonl)."""
    now_epoch = round(time.time(), 3)
    iso_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    who_clean = (reviewer or "").strip() or "Clinical Reviewer"
    act_clean = str(action).strip() or "approve"
    h_pres = int(human_present) if human_present is not None else None

    rec = {
        "t": now_epoch,
        "timestamp": iso_now,
        "who": who_clean,
        "when": iso_now,
        "what": act_clean,
        "reviewer": who_clean,
        "frame_id": str(frame_id),
        "video": str(video),
        "split": str(split),
        "key": str(key),
        "item_type": str(item_type),
        "action": act_clean,
        "human_action": act_clean,
        "agent_status": str(agent_status),
        "model_present": int(_safe_present(model_present)),
        "agent_present": int(_safe_present(model_present)),
        "human_present": h_pres,
        "conf": round(_safe_prob(conf, default=0.0), 4),
        "rationale": str(rationale or ""),
    }
    line = json.dumps(rec) + "\n"

    if log_path is not None:
        target = Path(log_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(line)
        if target.name == "gate_log.jsonl":
            sib = target.parent / "human_gate_log.jsonl"
            with open(sib, "a", encoding="utf-8") as f:
                f.write(line)
        elif target.name == "human_gate_log.jsonl":
            sib = target.parent / "gate_log.jsonl"
            with open(sib, "a", encoding="utf-8") as f:
                f.write(line)
    else:
        primary_p, mirror_p = _default_log_paths(out_dir)
        primary_p.parent.mkdir(parents=True, exist_ok=True)
        with open(primary_p, "a", encoding="utf-8") as f:
            f.write(line)
        with open(mirror_p, "a", encoding="utf-8") as f:
            f.write(line)

    return rec


def _latest_active_decisions_by_item(gate_log: list[dict]) -> dict[tuple[str, str], dict]:
    """Return latest active human decision per (frame_id, key), supporting 'reset'/'undo'."""
    latest: dict[tuple[str, str], dict] = {}
    for rec in gate_log:
        if not isinstance(rec, dict):
            continue
        fid = str(rec.get("frame_id", "")).strip()
        key = str(rec.get("key", "")).strip()
        if not fid or not key:
            continue
        act = str(rec.get("action", rec.get("human_action", rec.get("what", "")))).strip().lower()
        if act in ("reset", "undo"):
            latest.pop((fid, key), None)
        else:
            latest[(fid, key)] = rec
    return latest


def compute_amber_queue(
    dec_df: pd.DataFrame | None,
    gate_log: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Partition AMBER rows into pending vs human-cleared and compute strict lock state."""
    if dec_df is None or dec_df.empty:
        return (
            pd.DataFrame(columns=SAFE_DECISION_COLS),
            pd.DataFrame(columns=SAFE_DECISION_COLS),
            {
                "n_total": 0,
                "n_accept": 0,
                "n_reject": 0,
                "n_amber_total": 0,
                "n_amber_pending": 0,
                "n_amber_cleared": 0,
                "n_human_approved": 0,
                "n_human_corrected": 0,
                "export_unlocked": False,
            },
        )

    df = sanitize_decisions_df(dec_df)
    df["norm_status"] = df["status"].map(normalize_status)
    latest_by_item = _latest_active_decisions_by_item(gate_log)

    amber_df = df[df["norm_status"] == "AMBER"].copy()
    is_cleared_flags = [
        (str(r.frame_id), str(r.key)) in latest_by_item for r in amber_df.itertuples()
    ]
    amber_df["is_cleared"] = is_cleared_flags

    pending_df = amber_df[~amber_df["is_cleared"]].copy().reset_index(drop=True)
    cleared_df = amber_df[amber_df["is_cleared"]].copy().reset_index(drop=True)

    n_approved = 0
    n_corrected = 0
    if not cleared_df.empty:
        reviewers, actions, h_presents, timestamps = [], [], [], []
        for r in cleared_df.itertuples():
            rec = latest_by_item[(str(r.frame_id), str(r.key))]
            act = str(rec.get("action", rec.get("human_action", rec.get("what", "approve"))))
            h_p = rec.get("human_present")
            model_p = int(getattr(r, "final_present", getattr(r, "present", 0)))
            if h_p is not None and int(h_p) != model_p:
                n_corrected += 1
            elif act.lower() in ("correct", "flip"):
                n_corrected += 1
            else:
                n_approved += 1
            reviewers.append(str(rec.get("reviewer", rec.get("who", ""))))
            actions.append(act)
            h_presents.append(h_p)
            timestamps.append(str(rec.get("timestamp", rec.get("when", ""))))
        cleared_df["reviewer"] = reviewers
        cleared_df["human_action"] = actions
        cleared_df["human_present"] = h_presents
        cleared_df["approved_at"] = timestamps

    n_total = int(len(df))
    n_amber_pending = int(len(pending_df))
    summary = {
        "n_total": n_total,
        "n_accept": int((df["norm_status"] == "accept").sum()),
        "n_reject": int((df["norm_status"] == "reject").sum()),
        "n_amber_total": int(len(amber_df)),
        "n_amber_pending": n_amber_pending,
        "n_amber_cleared": int(len(cleared_df)),
        "n_human_approved": n_approved,
        "n_human_corrected": n_corrected,
        "export_unlocked": bool(n_total > 0 and n_amber_pending == 0),
    }
    return pending_df, cleared_df, summary


def build_verified_dataset_records(
    dec_df: pd.DataFrame | None,
    gate_log: list[dict],
    amber_rounds: list[dict] | None = None,
) -> list[dict]:
    """Build the list of verified dataset rows (auto-accepted + human-cleared AMBERs) with full DESIGN.md 1D provenance."""
    if dec_df is None or dec_df.empty:
        return []
    df = sanitize_decisions_df(dec_df)
    latest_by_item = _latest_active_decisions_by_item(gate_log)
    manual_srcs = load_manual_sources()
    released_iso = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    default_signoff_reviewer = "Clinical Reviewer"
    for entry in reversed(gate_log):
        if isinstance(entry, dict) and entry.get("reviewer"):
            default_signoff_reviewer = str(entry["reviewer"])
            break

    rounds_by_item: dict[tuple[str, str], list[dict]] = {}
    for rm in amber_rounds or []:
        if isinstance(rm, dict):
            rfid = str(rm.get("frame_id", "")).strip()
            rkey = str(rm.get("key", "")).strip()
            if rfid and rkey:
                rounds_by_item.setdefault((rfid, rkey), []).append({
                    "round": int(rm.get("round", 1) or 1),
                    "verdict": str(rm.get("verdict", rm.get("status", ""))),
                    "reason": str(rm.get("reason", rm.get("reasoning", ""))),
                })

    records: list[dict] = []
    for r in df.itertuples():
        ns = normalize_status(getattr(r, "status", ""))
        fid = str(r.frame_id)
        key = str(r.key)
        vid = str(getattr(r, "video", getattr(r, "video_id", "")))
        split = str(getattr(r, "split", "test"))
        conf = round(_safe_prob(getattr(r, "conf", 0.0), default=0.0), 4)
        trust_val = round(_safe_prob(getattr(r, "trust", 0.0), default=0.0), 4)
        rationale = str(getattr(r, "rationale", getattr(r, "evidence", "")) or "")
        box_val = parse_box(getattr(r, "box", None)) or []
        model_id = str(getattr(r, "model", C.FLASH) or C.FLASH)
        p_hash = str(getattr(r, "prompt_hash", "") or "")
        pv_m = getattr(r, "pv_match", None)
        pv_c = getattr(r, "pv_conf", None)
        pw_val = round(_safe_prob(getattr(r, "p_wrong", 0.5), default=0.5), 4)
        critic_str = str(getattr(r, "critic", "") or "")
        item_rounds = rounds_by_item.get((fid, key), [])

        double_check_prov = {
            "pv_match": bool(_pv_ok(pv_m)) if pd.notna(pv_m) else None,
            "pv_conf": round(_safe_prob(pv_c, default=0.0), 4) if pd.notna(pv_c) else None,
            "p_wrong": pw_val,
            "critic": critic_str,
            "trust": trust_val,
        }

        if ns == "accept":
            if (fid, key) in latest_by_item:
                h = latest_by_item[(fid, key)]
                act = str(h.get("action", h.get("human_action", h.get("what", "approve"))))
                if act.lower() in ("reject", "reject_exclude", "exclude"):
                    continue
                h_pres = h.get("human_present")
                final_p = int(_safe_present(h_pres)) if h_pres is not None else int(_safe_present(getattr(r, "final_present", getattr(r, "present", 0))))
                final_by = "human" if h_pres is not None and int(h_pres) != int(_safe_present(getattr(r, "final_present", 0))) else str(getattr(r, "final_by", "flash") or "flash")
                rev_name = str(h.get("reviewer", h.get("who", default_signoff_reviewer)))
                app_at = str(h.get("timestamp", h.get("when", released_iso)))
                v_src = "human_audit_verified"
            else:
                final_p = int(_safe_present(getattr(r, "final_present", getattr(r, "present", 0))))
                final_by = str(getattr(r, "final_by", "flash") or "flash")
                rev_name = None
                app_at = None
                act = None
                v_src = f"agent_{final_by}"

            records.append({
                "frame_id": fid,
                "video": vid,
                "video_id": vid,
                "split": split,
                "key": key,
                "present": final_p,
                "final_present": final_p,
                "conf": conf,
                "trust": trust_val,
                "box": box_val,
                "rationale": rationale,
                "evidence": rationale,
                "original_status": ns,
                "status": ns,
                "final_by": final_by,
                "verification_source": v_src,
                "reviewer": rev_name,
                "approved_at": app_at,
                "human_action": act,
                "manual_sources": manual_srcs,
                "audit_signed_off_by": rev_name or default_signoff_reviewer,
                "released_at": released_iso,
                "provenance": {
                    "seeding": "self_written_manual (out/manual.json)",
                    "manual_sources": manual_srcs,
                    "labeller_model": model_id,
                    "prompt_hash": p_hash,
                    "double_check": double_check_prov,
                    "amber_loop_rounds": item_rounds,
                    "release_gate": {
                        "original_status": ns,
                        "final_by": final_by,
                        "reviewer": rev_name,
                        "audit_signed_off_by": rev_name or default_signoff_reviewer,
                        "human_action": act,
                        "approved_at": app_at,
                        "released_at": released_iso,
                    },
                },
            })
        elif ns == "AMBER" and (fid, key) in latest_by_item:
            h = latest_by_item[(fid, key)]
            h_pres = h.get("human_present")
            act = str(h.get("action", h.get("human_action", h.get("what", "approve"))))
            if h_pres is not None and act.lower() not in ("reject", "reject_exclude", "exclude"):
                final_p = int(_safe_present(h_pres))
                rev_name = str(h.get("reviewer", h.get("who", default_signoff_reviewer)))
                app_at = str(h.get("timestamp", h.get("when", released_iso)))
                records.append({
                    "frame_id": fid,
                    "video": vid,
                    "video_id": vid,
                    "split": split,
                    "key": key,
                    "present": final_p,
                    "final_present": final_p,
                    "conf": conf,
                    "trust": trust_val,
                    "box": box_val,
                    "rationale": rationale,
                    "evidence": rationale,
                    "original_status": ns,
                    "status": "accept",  # integrity: allow (human action)
                    "final_by": "human",
                    "verification_source": "human_gate",
                    "reviewer": rev_name,
                    "approved_at": app_at,
                    "human_action": act,
                    "manual_sources": manual_srcs,
                    "audit_signed_off_by": rev_name,
                    "released_at": released_iso,
                    "provenance": {
                        "seeding": "self_written_manual (out/manual.json)",
                        "manual_sources": manual_srcs,
                        "labeller_model": model_id,
                        "prompt_hash": p_hash,
                        "double_check": double_check_prov,
                        "amber_loop_rounds": item_rounds,
                        "release_gate": {
                            "original_status": ns,
                            "final_by": "human",
                            "reviewer": rev_name,
                            "audit_signed_off_by": rev_name,
                            "human_action": act,
                            "approved_at": app_at,
                            "released_at": released_iso,
                        },
                    },
                })
    return records


def export_verified_dataset(
    dec_df: pd.DataFrame | None,
    gate_log: list[dict],
    out_path: Path | str | None = None,
    amber_rounds: list[dict] | None = None,
) -> list[dict]:
    """Write out/verified_dataset.jsonl ONLY if the AMBER queue is empty; raise PermissionError if locked."""
    _, _, summary = compute_amber_queue(dec_df, gate_log)
    if not summary["export_unlocked"]:
        pending_n = summary["n_amber_pending"]
        total_n = summary["n_total"]
        raise PermissionError(
            f"Export locked: {pending_n} AMBER label(s) pending human review (total rows={total_n}). "
            "Clear all AMBER items before exporting out/verified_dataset.jsonl."
        )

    rounds_data = amber_rounds if amber_rounds is not None else load_amber_rounds()
    records = build_verified_dataset_records(dec_df, gate_log, amber_rounds=rounds_data)
    target = Path(out_path) if out_path is not None else (C.OUT_DIR / "verified_dataset.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload_jsonl = "\n".join(json.dumps(r) for r in records) + ("\n" if records else "")
    target.write_text(payload_jsonl, encoding="utf-8")

    envelope = {
        "project": "SurgProof",
        "exported_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_verified_labels": len(records),
        "human_gate_cleared": True,
        "labels": records,
    }
    (target.parent / "verified_export.json").write_text(json.dumps(envelope, indent=2), encoding="utf-8")
    return records


# ==============================================================================
# 4. Frame Image & Bounding Box Rendering (Zero Ground Truth)
# ==============================================================================
def parse_box(raw_box: Any) -> list[float] | None:
    """Safely parse a normalized [ymin, xmin, ymax, xmax] box in 0..1000."""
    if raw_box is None or (isinstance(raw_box, float) and pd.isna(raw_box)):
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
        y0, x0, y1, x1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return None
    y0, y1 = max(0.0, min(1000.0, min(y0, y1))), max(0.0, min(1000.0, max(y0, y1)))
    x0, x1 = max(0.0, min(1000.0, min(x0, x1))), max(0.0, min(1000.0, max(x0, x1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [y0, x0, y1, x1]


def resolve_frame_image(path_str: str | Path | None, frame_id: str = "") -> Image.Image:
    """Load endoscopic frame from disk or render a clean placeholder canvas."""
    candidates = []
    if path_str and not (isinstance(path_str, float) and pd.isna(path_str)):
        p = Path(str(path_str))
        candidates.extend([p, C.ROOT / p])
    for cand in candidates:
        if cand.is_file():
            try:
                return Image.open(cand).convert("RGB")
            except Exception:
                pass
    im = Image.new("RGB", (640, 480), color=(24, 28, 34))
    d = ImageDraw.Draw(im)
    d.rectangle([16, 16, 624, 464], outline=(80, 90, 105), width=2)
    d.text((32, 215), f"Frame preview: {frame_id or 'unindexed'}", fill=(215, 220, 230))
    d.text((32, 245), "Endoscopic image renders from raw/CholecSeg8k when available.", fill=(145, 155, 170))
    return im


def render_amber_card_image(
    path_str: str | Path | None,
    row: Any,
    frame_id: str = "",
    badge_status: str = "AMBER",
) -> tuple[Image.Image, Image.Image | None]:
    """Return (annotated full frame, optional zoomed crop of bbox) for a claim."""
    base_im = resolve_frame_image(path_str, frame_id=frame_id)
    im = base_im.copy()
    w, h = im.size
    box = parse_box(getattr(row, "box", None))
    crop_im = None

    if box is not None:
        y0, x0, y1, x1 = box
        px0, py0 = int(x0 * w / 1000.0), int(y0 * h / 1000.0)
        px1, py1 = int(x1 * w / 1000.0), int(y1 * h / 1000.0)
        pad_x = max(12, int((px1 - px0) * 0.12))
        pad_y = max(12, int((py1 - py0) * 0.12))
        cx0, cy0 = max(0, px0 - pad_x), max(0, py0 - pad_y)
        cx1, cy1 = min(w, px1 + pad_x), min(h, py1 + pad_y)
        if cx1 > cx0 + 10 and cy1 > cy0 + 10:
            crop_im = base_im.crop((cx0, cy0, cx1, cy1))

        d = ImageDraw.Draw(im)
        colour = STATUS_COLORS.get(badge_status, STATUS_COLORS["AMBER"])
        d.rectangle([px0, py0, px1, py1], outline=colour, width=4)
        label_txt = f"{badge_status}: {getattr(row, 'key', '')}"
        d.rectangle([px0, max(0, py0 - 22), min(w, px0 + len(label_txt) * 8 + 12), py0], fill=colour)
        d.text((px0 + 5, max(2, py0 - 19)), label_txt, fill=(255, 255, 255))

    return im, crop_im


# ==============================================================================
# 5. Streamlit Page UI (pages/2_Human_gate.py)
# ==============================================================================
def _inject_keyboard_shortcuts() -> None:
    """Bind A/Y (Approve), C/N (Correct/Flip), and J/K (Next/Prev) in the parent Streamlit document."""
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            if (doc._surgproofGateKeyListener) {
                doc.removeEventListener('keydown', doc._surgproofGateKeyListener);
            }
            doc._surgproofGateKeyListener = function(e) {
                const tag = (e.target && e.target.tagName) ? e.target.tagName.toUpperCase() : '';
                if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (e.target && e.target.isContentEditable)) {
                    return;
                }
                if (e.metaKey || e.ctrlKey || e.altKey) return;
                const key = e.key.toLowerCase();
                let matchToken = null;
                if (key === 'a' || key === 'y') {
                    matchToken = '[Key: A]';
                } else if (key === 'c' || key === 'n' || key === 'f') {
                    matchToken = '[Key: C]';
                } else if (key === 'j' || key === 'arrowright') {
                    matchToken = '[Key: J]';
                } else if (key === 'k' || key === 'arrowleft') {
                    matchToken = '[Key: K]';
                } else if (key === 'z' || key === 'u') {
                    matchToken = '[Key: Z]';
                }
                if (!matchToken) return;
                const buttons = Array.from(doc.querySelectorAll('button'));
                const targetBtn = buttons.find(b => b.innerText && b.innerText.includes(matchToken) && !b.disabled);
                if (targetBtn) {
                    e.preventDefault();
                    targetBtn.click();
                }
            };
            doc.addEventListener('keydown', doc._surgproofGateKeyListener);
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def render_human_gate_page() -> None:
    """Render the Streamlit Final Pass / Release Dataset Sign-Off page (`pages/2_Human_gate.py`)."""
    try:
        st.set_page_config(
            page_title="SurgProof — Final Pass · Release Dataset Sign-Off",
            page_icon="🛡️",
            layout="wide",
        )
    except Exception:
        pass

    st.markdown(
        """
        <style>
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0A0B0E;
            color: #F4F5F8;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #12141C;
            border-right: 1px solid #232734;
        }
        .gate-banner-locked {
            background: #362407;
            border: 1px solid #F59E0B;
            border-radius: 8px;
            padding: 14px 18px;
            color: #F59E0B;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .gate-banner-unlocked {
            background: #092E24;
            border: 1px solid #10B981;
            border-radius: 8px;
            padding: 14px 18px;
            color: #10B981;
            font-weight: 600;
            margin-bottom: 12px;
        }
        .pos-card {
            background: #12141C;
            border: 1px solid #232734;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 12px;
            color: #D8DCE6;
            font-size: 0.93rem;
            line-height: 1.5;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        if (C.ROOT / "app.py").is_file():
            try:
                st.page_link("app.py", label="← Back to SurgProof Pipeline", icon="🏠")
            except Exception:
                pass
        st.subheader("🛡️ Final Pass · Release Dataset")
        reviewer_name = st.text_input(
            "Sign-Off Reviewer Identity (who)",
            value="Clinical Reviewer (Dataset Sign-Off)",
            help="Recorded on every approval, correction, or release sign-off in out/gate_log.jsonl.",
        )
        st.caption(
            "**Keyboard Shortcuts (Live Demo Speed Mode):**\n"
            "- Press **`A`** (or **`Y`**) → **Approve** model's call\n"
            "- Press **`C`** (or **`N`**) → **Correct / Flip** label (`1 ↔ 0`)\n"
            "- Press **`J` / `K`** → Next / Prev AMBER card\n"
            "- Press **`Z`** → Undo last human decision"
        )

    st.title("🛡️ Final Pass · Release Dataset Sign-Off — People Stay in Control")
    st.markdown(
        "<div class='pos-card'>"
        "<span style='color:#8A909E;'>Today: a human at every step.</span> <b style='color:#F4F5F8;'>SurgProof: none — human clears only amber cases.</b><br>"
        "<b>Today (Supervisely / SAM / MixFormer / TAP-Net):</b> A human at every step — drawing the first box, fixing tracking drift, tagging actions by hand, and reviewing every frame.<br>"
        "<b>SurgProof:</b> Removes the human from seeding (<code>out/manual.json</code>), bulk labelling, independent double-check, and the multi-round <b>Amber Loop</b> (<code>out/amber_rounds.jsonl</code>). "
        "<b>People stay in control at the final pass:</b> clear any residual <code>AMBER</code> items and inspect the 20-item auto-accepted audit sample before releasing <code>out/verified_dataset.jsonl</code>."
        "</div>",
        unsafe_allow_html=True,
    )

    sources = discover_decision_sources()
    gate_log = load_gate_log()
    amber_rounds = load_amber_rounds()

    if not sources:
        st.markdown(
            "<div class='gate-banner-locked'>🔒 ACTION BLOCKED (RELEASE LOCKED) — Awaiting verifier decisions (<code>out/decisions.csv</code>) or smoke labels.</div>",
            unsafe_allow_html=True,
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Auto-Accepted (accept)", "PENDING")
        m2.metric("Auto-Rejected (reject)", "PENDING")
        m3.metric("Residual AMBER Queue", "PENDING")
        m4.metric("Cleared by Human", len(gate_log))
        st.button(
            "🔒 Release dataset — Export verified dataset (LOCKED — awaiting verifier decisions)",
            key="btn_export_verified_dataset",
            disabled=True,
            width="stretch",
        )
        return

    if len(sources) > 1:
        src_labels = [s[0] for s in sources]
        picked_label = st.radio(
            "Active Decision Source",
            src_labels,
            index=0,
            horizontal=True,
        )
        picked_source = next(s for s in sources if s[0] == picked_label)
    else:
        picked_source = sources[0]
        st.caption(f"Source: **{picked_source[0]}** (`{picked_source[1].name}`)")

    dec_df_all = load_decisions_from_source(picked_source[1])
    if len(dec_df_all) > 500 and "split" in dec_df_all.columns and "key" in dec_df_all.columns:
        scope_options = [
            "Held-Out Test Split · 5 Headline Structures (split='test', excl. liver)",
            "Held-Out Test Split · All 6 Structures (split='test')",
            "Full Dataset · All Splits (train + calib + test)",
        ]
        picked_scope = st.radio(
            "Release Gate Scope",
            scope_options,
            index=0,
            horizontal=True,
        )
        if picked_scope == scope_options[0]:
            dec_df = dec_df_all[(dec_df_all["split"] == "test") & (dec_df_all["key"] != "liver")].copy().reset_index(drop=True)
        elif picked_scope == scope_options[1]:
            dec_df = dec_df_all[dec_df_all["split"] == "test"].copy().reset_index(drop=True)
        else:
            dec_df = dec_df_all
    else:
        dec_df = dec_df_all
    pending_df, cleared_df, summary = compute_amber_queue(dec_df, gate_log)

    rounds_by_item: dict[tuple[str, str], list[dict]] = {}
    for rm in amber_rounds:
        if isinstance(rm, dict):
            rfid = str(rm.get("frame_id", "")).strip()
            rkey = str(rm.get("key", "")).strip()
            if rfid and rkey:
                rounds_by_item.setdefault((rfid, rkey), []).append(rm)

    # Top-level metrics bar
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Auto-Accepted (accept)", summary["n_accept"])
    m2.metric("Auto-Rejected (reject)", summary["n_reject"])
    m3.metric("Residual AMBER Pending", summary["n_amber_pending"])
    m4.metric("AMBER Cleared by Human", f"{summary['n_amber_cleared']} / {summary['n_amber_total']}")
    m5.metric(
        "Human Sign-Off Log",
        f"{summary['n_human_approved']} approved · {summary['n_human_corrected']} corrected",
    )

    # Shrinking Amber Loop Funnel (Hero Moment per VISION.md)
    funnel_stages = compute_amber_funnel(
        dec_df,
        amber_rounds=amber_rounds,
        n_amber_pending_after_human=summary["n_amber_pending"],
    )
    if funnel_stages:
        with st.expander("📉 Multi-Round Amber Loop Funnel (Double-Check → Argon Loop Rounds → Human Release Gate)", expanded=True):
            f_cols = st.columns(len(funnel_stages))
            for idx_f, stg in enumerate(funnel_stages):
                delta_str = (
                    f"-{stg['resolved_in_step']} resolved"
                    if stg["resolved_in_step"] > 0
                    else stg["note"]
                )
                f_cols[idx_f].metric(
                    stg["stage"],
                    f"{stg['ambers_remaining']} AMBER",
                    delta=delta_str,
                    delta_color="inverse" if stg["resolved_in_step"] > 0 else "off",
                )

    if summary["n_amber_total"] > 0:
        frac_done = summary["n_amber_cleared"] / max(1, summary["n_amber_total"])
        st.progress(
            frac_done,
            text=(
                f"Residual AMBER Sign-Off Progress: {summary['n_amber_cleared']} / {summary['n_amber_total']} cleared "
                f"({summary['n_amber_pending']} remaining before dataset release unlocks)"
            ),
        )

    # Strict Release / Export Lock Gate Bar
    export_path = C.OUT_DIR / "verified_dataset.jsonl"
    if summary["export_unlocked"]:
        st.markdown(
            f"<div class='gate-banner-unlocked'>🔓 READY TO RELEASE DATASET — All {summary['n_amber_total']} residual AMBER label(s) cleared "
            f"(0 pending). <b>Release dataset / Export verified dataset (<code>out/verified_dataset.jsonl</code>)</b> is UNLOCKED.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='gate-banner-locked'>🔒 ACTION BLOCKED (PEOPLE STAY IN CONTROL) — "
            f"<b>{summary['n_amber_pending']} residual AMBER label(s)</b> require human approval or correction before "
            f"releasing <code>out/verified_dataset.jsonl</code>.</div>",
            unsafe_allow_html=True,
        )

    exp_col1, exp_col2 = st.columns([2, 1])
    with exp_col1:
        export_btn_label = (
            f"🔓 Release dataset — Export verified dataset -> out/verified_dataset.jsonl ({summary['n_accept'] + summary['n_amber_cleared']} verified rows + provenance)"
            if summary["export_unlocked"]
            else f"🔒 Release dataset — Export verified dataset (DISABLED — clear {summary['n_amber_pending']} pending AMBER item(s) to unlock)"
        )
        if st.button(
            export_btn_label,
            key="btn_export_verified_dataset",
            type="primary",
            disabled=not summary["export_unlocked"],
            width="stretch",
        ):
            exported_rows = export_verified_dataset(
                dec_df,
                gate_log,
                out_path=export_path,
                amber_rounds=amber_rounds,
            )
            st.success(
                f"✅ Dataset released! Wrote **{len(exported_rows)}** verified labels with full per-label provenance to `{export_path}` (`out/verified_dataset.jsonl`)."
            )
    with exp_col2:
        if summary["export_unlocked"]:
            records_now = export_verified_dataset(
                dec_df,
                gate_log,
                out_path=export_path,
                amber_rounds=amber_rounds,
            )
            jsonl_bytes = ("\n".join(json.dumps(r) for r in records_now) + "\n").encode("utf-8")
            st.download_button(
                "⬇️ Download verified_dataset.jsonl",
                data=jsonl_bytes,
                file_name="verified_dataset.jsonl",
                mime="application/jsonl",
                disabled=False,
                width="stretch",
            )
        else:
            st.download_button(
                "🔒 Download locked (AMBER queue > 0)",
                data=b"",
                file_name="verified_dataset.jsonl",
                mime="application/jsonl",
                disabled=True,
                width="stretch",
            )

    st.divider()

    # ==========================================================================
    # Part 1: Rapid Focus Deck + Full Residual AMBER Queue
    # ==========================================================================
    if not pending_df.empty:
        _inject_keyboard_shortcuts()

        if "gate_cursor" not in st.session_state:
            st.session_state["gate_cursor"] = 0
        cursor = int(st.session_state["gate_cursor"]) % len(pending_df)
        active_row = pending_df.iloc[cursor]

        fid = str(active_row["frame_id"])
        k = str(active_row["key"])
        vid = str(active_row.get("video", active_row.get("video_id", "")))
        split_val = str(active_row.get("split", "test"))
        human_label_name = C.LABELS.get(k, (0, k))[1]
        model_pres = int(_safe_present(active_row.get("final_present", active_row.get("present", 0))))
        flipped_pres = 1 - model_pres
        conf_val = float(_safe_prob(active_row.get("conf", active_row.get("confidence", 0.0)), default=0.0))
        trust_val = float(_safe_prob(active_row.get("trust", 0.0), default=0.0))
        rationale_txt = str(active_row.get("rationale") or active_row.get("evidence") or "No rationale provided.")
        critic_txt = str(active_row.get("critic") or "n/a")
        pv_val = active_row.get("pv_match", None)
        p_wrong_val = float(_safe_prob(active_row.get("p_wrong", 0.5), default=0.5))
        final_by = str(active_row.get("final_by", "flash") or "flash")
        argon_note = str(active_row.get("argon_reason") or active_row.get("argon_reasoning") or "")
        item_rounds = rounds_by_item.get((fid, k), [])

        head_c1, head_c2 = st.columns([3, 2])
        with head_c1:
            st.subheader(f"⚡ Residual AMBER Review Deck — Item #{cursor + 1} of {len(pending_df)}")
        with head_c2:
            nav1, nav2, nav3 = st.columns(3)
            with nav1:
                if st.button("⬅️ Prev [Key: K]", key="focus_prev", width="stretch"):
                    st.session_state["gate_cursor"] = (cursor - 1) % len(pending_df)
                    st.rerun()
            with nav2:
                if st.button("➡️ Next [Key: J]", key="focus_next", width="stretch"):
                    st.session_state["gate_cursor"] = (cursor + 1) % len(pending_df)
                    st.rerun()
            with nav3:
                if st.button("↩️ Undo [Key: Z]", key="focus_undo", disabled=len(cleared_df) == 0, width="stretch"):
                    last_cleared = cleared_df.iloc[-1]
                    record_gate_decision(
                        reviewer=reviewer_name,
                        frame_id=str(last_cleared["frame_id"]),
                        key=str(last_cleared["key"]),
                        action="reset",
                        human_present=None,
                        video=str(last_cleared.get("video", "")),
                        split=str(last_cleared.get("split", "test")),
                        model_present=int(last_cleared.get("final_present", 0)),
                        conf=float(last_cleared.get("conf", 0.0)),
                        rationale=str(last_cleared.get("rationale", "")),
                    )
                    st.rerun()

        with st.container(border=True):
            fc_img, fc_info = st.columns([1.25, 1.75])
            with fc_img:
                full_im, crop_im = render_amber_card_image(
                    active_row.get("frame_path", ""),
                    active_row,
                    frame_id=fid,
                    badge_status="AMBER",
                )
                st.image(full_im, caption=f"Frame {fid} ({vid}, split={split_val})", width="stretch")
                if crop_im is not None:
                    st.image(crop_im, caption=f"Zoomed evidence crop: {k}", width=240)

            with fc_info:
                pres_badge = "PRESENT (1)" if model_pres == 1 else "ABSENT (0)"
                flip_badge = "ABSENT (0)" if model_pres == 1 else "PRESENT (1)"
                st.markdown(f"### `{k}` — {human_label_name}")
                st.markdown(
                    f"**Frame:** `{fid}` · **Video:** `{vid}` · **Model Tentative Call:** **`{pres_badge}`** *(by `{final_by}`)*"
                )
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Model Confidence", f"{conf_val * 100:.0f}%")
                s2.metric("Verifier Trust", f"{trust_val:.2f}")
                s3.metric("Blind Crop Check", "Agreed" if _pv_ok(pv_val) else ("Disagreed" if pd.notna(pv_val) else "N/A"))
                s4.metric("Critic p_wrong", f"{p_wrong_val:.2f}")

                st.markdown(f"**Model Rationale:** {rationale_txt}")
                st.caption(f"**Sceptical Critic:** {critic_txt}")
                if item_rounds:
                    last_r = item_rounds[-1]
                    st.caption(
                        f"**Amber Loop ({len(item_rounds)} round(s) examined by Argon):** "
                        f"{last_r.get('reason', argon_note)}"
                    )
                elif argon_note and argon_note.lower() != "nan":
                    st.caption(f"**Argon Amber Loop Note:** {argon_note}")

                st.markdown("#### Human Decision (1-Click or Keyboard `A` / `C`)")
                act_c1, act_c2 = st.columns(2)
                with act_c1:
                    if st.button(
                        f"✅ APPROVE Model Call: {pres_badge} [Key: A]",
                        key=f"focus_approve_{fid}_{k}",
                        type="primary",
                        width="stretch",
                    ):
                        record_gate_decision(
                            reviewer=reviewer_name,
                            frame_id=fid,
                            key=k,
                            action="approve",
                            human_present=model_pres,
                            item_type="amber",
                            video=vid,
                            split=split_val,
                            agent_status="AMBER",
                            model_present=model_pres,
                            conf=conf_val,
                            rationale=rationale_txt,
                        )
                        st.rerun()
                with act_c2:
                    if st.button(
                        f"🔄 CORRECT / FLIP -> {flip_badge} [Key: C]",
                        key=f"focus_correct_{fid}_{k}",
                        width="stretch",
                    ):
                        record_gate_decision(
                            reviewer=reviewer_name,
                            frame_id=fid,
                            key=k,
                            action="correct",
                            human_present=flipped_pres,
                            item_type="amber",
                            video=vid,
                            split=split_val,
                            agent_status="AMBER",
                            model_present=model_pres,
                            conf=conf_val,
                            rationale=rationale_txt,
                        )
                        st.rerun()

                same_frame_pending = pending_df[pending_df["frame_id"] == fid]
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    if len(same_frame_pending) > 1:
                        if st.button(
                            f"⚡ Approve all {len(same_frame_pending)} AMBER items on frame `{fid}`",
                            key=f"batch_frame_{fid}",
                            width="stretch",
                        ):
                            for r_f in same_frame_pending.itertuples():
                                m_p = int(_safe_present(getattr(r_f, "final_present", getattr(r_f, "present", 0))))
                                record_gate_decision(
                                    reviewer=reviewer_name,
                                    frame_id=str(r_f.frame_id),
                                    key=str(r_f.key),
                                    action="approve",
                                    human_present=m_p,
                                    item_type="amber",
                                    video=str(getattr(r_f, "video", "")),
                                    split=str(getattr(r_f, "split", "test")),
                                    agent_status="AMBER",
                                    model_present=m_p,
                                    conf=float(_safe_prob(getattr(r_f, "conf", 0.0))),
                                    rationale=str(getattr(r_f, "rationale", getattr(r_f, "evidence", ""))),
                                )
                            st.rerun()
                with b_col2:
                    if st.button(
                        f"⚡ Demo Fast-Forward: Approve all {len(pending_df)} remaining AMBER items",
                        key="batch_approve_all_remaining",
                        width="stretch",
                    ):
                        for r_all in pending_df.itertuples():
                            m_p = int(_safe_present(getattr(r_all, "final_present", getattr(r_all, "present", 0))))
                            record_gate_decision(
                                reviewer=reviewer_name,
                                frame_id=str(r_all.frame_id),
                                key=str(r_all.key),
                                action="approve",
                                human_present=m_p,
                                item_type="amber",
                                video=str(getattr(r_all, "video", "")),
                                split=str(getattr(r_all, "split", "test")),
                                agent_status="AMBER",
                                model_present=m_p,
                                conf=float(_safe_prob(getattr(r_all, "conf", 0.0))),
                                rationale=str(getattr(r_all, "rationale", getattr(r_all, "evidence", ""))),
                            )
                        st.rerun()

        st.subheader(f"All Pending Residual AMBER Items ({len(pending_df)})")
        page_slice = pending_df
        page_offset = 0
        if len(pending_df) > 25:
            page_size = 20
            n_pages = (len(pending_df) + page_size - 1) // page_size
            pg_idx = st.number_input(
                f"Queue Page (20 AMBER cards per page, {n_pages} page(s) total)",
                min_value=1,
                max_value=n_pages,
                value=1,
                step=1,
                key="amber_queue_page",
            )
            page_offset = (int(pg_idx) - 1) * page_size
            page_slice = pending_df.iloc[page_offset : page_offset + page_size]

        for local_idx, r in enumerate(page_slice.itertuples()):
            idx = page_offset + local_idx
            r_fid = str(r.frame_id)
            r_k = str(r.key)
            r_vid = str(getattr(r, "video", getattr(r, "video_id", "")))
            r_split = str(getattr(r, "split", "test"))
            r_pres = int(_safe_present(getattr(r, "final_present", getattr(r, "present", 0))))
            r_flip = 1 - r_pres
            r_conf = float(_safe_prob(getattr(r, "conf", getattr(r, "confidence", 0.0)), default=0.0))
            r_trust = float(_safe_prob(getattr(r, "trust", 0.0), default=0.0))
            r_rat = str(getattr(r, "rationale", getattr(r, "evidence", "")) or "No rationale provided.")
            r_crit = str(getattr(r, "critic", "") or "n/a")
            r_name = C.LABELS.get(r_k, (0, r_k))[1]
            pres_str = "PRESENT (1)" if r_pres == 1 else "ABSENT (0)"
            flip_str = "ABSENT (0)" if r_pres == 1 else "PRESENT (1)"

            with st.container(border=True):
                qc1, qc2, qc3 = st.columns([1.1, 2.1, 1.2])
                with qc1:
                    card_im, _ = render_amber_card_image(
                        getattr(r, "frame_path", ""),
                        r,
                        frame_id=r_fid,
                        badge_status="AMBER",
                    )
                    st.image(card_im, width="stretch")
                with qc2:
                    st.markdown(f"**#{idx + 1} · `{r_k}` ({r_name}) — Frame `{r_fid}` (`{r_vid}`)**")
                    st.markdown(
                        f"- **Model's Label:** **`{pres_str}`** · **Confidence:** `{r_conf * 100:.0f}%` (`{r_conf:.2f}`) · **Trust:** `{r_trust:.2f}`\n"
                        f"- **Model's Rationale:** {r_rat}\n"
                        f"- **Critic (`p_wrong={_safe_prob(getattr(r, 'p_wrong', 0.5)):.2f}`):** {r_crit}"
                    )
                with qc3:
                    if st.button(
                        f"✅ Approve {pres_str}",
                        key=f"q_app_{r_fid}_{r_k}_{idx}",
                        type="primary",
                        width="stretch",
                    ):
                        record_gate_decision(
                            reviewer=reviewer_name,
                            frame_id=r_fid,
                            key=r_k,
                            action="approve",
                            human_present=r_pres,
                            item_type="amber",
                            video=r_vid,
                            split=r_split,
                            agent_status="AMBER",
                            model_present=r_pres,
                            conf=r_conf,
                            rationale=r_rat,
                        )
                        st.rerun()
                    if st.button(
                        f"🔄 Correct -> {flip_str}",
                        key=f"q_cor_{r_fid}_{r_k}_{idx}",
                        width="stretch",
                    ):
                        record_gate_decision(
                            reviewer=reviewer_name,
                            frame_id=r_fid,
                            key=r_k,
                            action="correct",
                            human_present=r_flip,
                            item_type="amber",
                            video=r_vid,
                            split=r_split,
                            agent_status="AMBER",
                            model_present=r_pres,
                            conf=r_conf,
                            rationale=r_rat,
                        )
                        st.rerun()

    # ==========================================================================
    # Part 2: Random Audit Sample of Auto-Accepted Labels (VISION.md Final Pass)
    # ==========================================================================
    audit_sample_df = sample_accepted_for_audit(dec_df, n_sample=20, seed=0)
    if not audit_sample_df.empty:
        with st.expander(
            f"🔍 Final Pass Spot-Check: Random Audit Sample of Auto-Accepted Labels ({len(audit_sample_df)} of {summary['n_accept']} `accept` labels with evidence crops)",
            expanded=pending_df.empty,
        ):
            st.caption(
                "Per VISION.md & DESIGN.md 1D: Before releasing the dataset, the human reviewer inspects a 20-item random audit sample of auto-accepted labels alongside their zoomed evidence crops."
            )
            if st.button(
                f"✅ Sign Off {len(audit_sample_df)}-Item Auto-Accepted Audit Sample",
                key="btn_signoff_audit_sample",
                width="stretch",
            ):
                for a_r in audit_sample_df.itertuples():
                    a_p = int(_safe_present(getattr(a_r, "final_present", getattr(a_r, "present", 0))))
                    record_gate_decision(
                        reviewer=reviewer_name,
                        frame_id=str(a_r.frame_id),
                        key=str(a_r.key),
                        action="approve",
                        human_present=a_p,
                        item_type="audit_sample",
                        video=str(getattr(a_r, "video", "")),
                        split=str(getattr(a_r, "split", "test")),
                        agent_status="accept",
                        model_present=a_p,
                        conf=float(_safe_prob(getattr(a_r, "conf", 0.0))),
                        rationale=str(getattr(a_r, "rationale", getattr(a_r, "evidence", ""))),
                    )
                st.rerun()

            audit_cols = st.columns(4)
            for a_idx, a_row in enumerate(audit_sample_df.itertuples()):
                col_a = audit_cols[a_idx % 4]
                a_fid = str(a_row.frame_id)
                a_k = str(a_row.key)
                a_pres = int(_safe_present(getattr(a_row, "final_present", getattr(a_row, "present", 0))))
                a_conf = float(_safe_prob(getattr(a_row, "conf", 0.0)))
                a_trust = float(_safe_prob(getattr(a_row, "trust", 0.0)))
                a_rat = str(getattr(a_row, "rationale", getattr(a_row, "evidence", "")))
                full_im_a, crop_im_a = render_amber_card_image(
                    getattr(a_row, "frame_path", ""),
                    a_row,
                    frame_id=a_fid,
                    badge_status="accept",
                )
                with col_a:
                    with st.container(border=True):
                        st.image(crop_im_a if crop_im_a is not None else full_im_a, width="stretch")
                        st.markdown(
                            f"**`{a_k}`** (`{a_fid}`) — **`{'PRESENT (1)' if a_pres else 'ABSENT (0)'}`**\n\n"
                            f"Conf `{a_conf * 100:.0f}%` · Trust `{a_trust:.2f}`\n\n"
                            f"*{a_rat}*"
                        )

    # ==========================================================================
    # Part 3: Cleared AMBER Items, Provenance Preview & Audit Log
    # ==========================================================================
    if not cleared_df.empty:
        with st.expander(f"✅ Human-Cleared AMBER Items ({len(cleared_df)})", expanded=pending_df.empty):
            show_cols = [
                c
                for c in [
                    "frame_id",
                    "video",
                    "key",
                    "final_present",
                    "human_action",
                    "human_present",
                    "reviewer",
                    "approved_at",
                    "conf",
                    "rationale",
                ]
                if c in cleared_df.columns
            ]
            st.dataframe(cleared_df[show_cols], width="stretch")

    if summary["export_unlocked"]:
        preview_records = build_verified_dataset_records(dec_df, gate_log, amber_rounds=amber_rounds)
        if preview_records:
            with st.expander(
                f"📦 Verified Dataset & Per-Label Provenance Preview (`out/verified_dataset.jsonl`, {len(preview_records)} rows)",
                expanded=True,
            ):
                src_counts = pd.Series([r["verification_source"] for r in preview_records]).value_counts().to_dict()
                st.markdown(
                    "**Dataset Provenance Breakdown:** "
                    + " · ".join(f"`{k}`: **{v}**" for k, v in src_counts.items())
                )
                st.json(preview_records[0])

    if gate_log:
        with st.expander(f"📜 Human Gate Audit Log (`out/gate_log.jsonl`, {len(gate_log)} events)", expanded=False):
            st.dataframe(pd.DataFrame(gate_log), width="stretch")


if __name__ == "__main__" or st.runtime.exists():
    render_human_gate_page()
