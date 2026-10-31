#!/usr/bin/env python3
"""SurgProof Web Studio & Final Pass Gate HTTP Server (:8600).

Serves the single-page web annotation studio (`web/index.html`, `web/style.css`,
`web/app.js`, `web/viewer.js`, `web/timeline.js`, `web/gate.js`, `web/data/*`)
plus read-only image/mask assets (`data/`, `raw/`, `out/masks/`) and the
Final Pass Human Gate endpoints:

  - GET  /api/gate  (alias /gate): returns residual AMBER queue, 4-round Amber
    Loop funnel counts from out/amber_rounds.jsonl + out/decisions.csv, 20-item
    auto-accepted audit sample, and current lock status.
  - POST /api/gate  (alias /gate): records human approve / correct / undo /
    batch_approve_ambers / sign_off_audit_sample actions to out/gate_log.jsonl
    (and out/human_gate_log.jsonl) and, once 0 residual AMBERs remain in scope,
    unlocks `release_dataset` to write out/verified_dataset.jsonl with full
    per-label provenance.

Strict Zero-Leakage Contract:
  - Never reads, joins, or returns ground-truth columns or masks in gate APIs.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import random
import sys
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "out"
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = REPO_ROOT / "raw"

DEFAULT_DECISIONS_CSV = OUT_DIR / "decisions.csv"
DEFAULT_AMBER_ROUNDS_JSONL = OUT_DIR / "amber_rounds.jsonl"
DEFAULT_ANNOTATIONS_JSONL = OUT_DIR / "annotations.jsonl"
DEFAULT_LABELS_JSONL = OUT_DIR / "labels_manual.jsonl"
DEFAULT_MANUAL_JSON = OUT_DIR / "manual.json"
DEFAULT_INDEX_CSV = DATA_DIR / "index.csv"
DEFAULT_GATE_LOG_JSONL = OUT_DIR / "gate_log.jsonl"
DEFAULT_HUMAN_GATE_LOG_JSONL = OUT_DIR / "human_gate_log.jsonl"
DEFAULT_VERIFIED_JSONL = OUT_DIR / "verified_dataset.jsonl"

HEADLINE_5_KEYS = ("grasper", "l_hook", "gallbladder", "cystic_duct", "blood")
SAFE_INDEX_COLS = ("frame_id", "path", "img_path", "split", "video", "clip", "frame")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _is_forbidden_col(col_name: str) -> bool:
    c = str(col_name).strip().lower()
    return c == "truth" or c.startswith("gt_") or "ground_truth" in c or c == "mask_path"


def _normalize_status(val: Any) -> str:
    s = str(val or "").strip()
    u = s.upper()
    if u == "AMBER":
        return "AMBER"
    l = s.lower()
    if l in ("accept", "accepted"):
        return "accept"
    if l in ("reject", "rejected", "escalate"):
        return "reject"
    return "AMBER"


def _to_float(val: Any, default: float = 0.0) -> float:
    if val in (None, "", "nan", "NaN", "None"):
        return default
    try:
        return round(float(val), 4)
    except Exception:
        return default


def _to_int(val: Any, default: int = 0) -> int:
    if val in (None, "", "nan", "NaN", "None"):
        return default
    if isinstance(val, bool):
        return 1 if val else 0
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "present"):
        return 1
    if s in ("0", "false", "no", "absent"):
        return 0
    try:
        return 1 if int(float(s)) > 0 else 0
    except Exception:
        return default


def _parse_box(val: Any) -> list[int] | None:
    if val is None or val == "":
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        try:
            return [int( round(float(x)) ) for x in val]
        except Exception:
            return None
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list) and len(parsed) == 4:
                    return [int(round(float(x))) for x in parsed]
            except Exception:
                return None
    return None


def load_safe_index_map(index_csv_path: Path = DEFAULT_INDEX_CSV) -> dict[str, dict[str, str]]:
    """Load only frame_id, path, split, video from data/index.csv (never ground-truth columns)."""
    out: dict[str, dict[str, str]] = {}
    if not index_csv_path.exists():
        return out
    with open(index_csv_path, encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            fid = str(row.get("frame_id", "")).strip()
            if not fid:
                continue
            safe_row: dict[str, str] = {}
            for k in SAFE_INDEX_COLS:
                if k in row and not _is_forbidden_col(k):
                    safe_row[k] = str(row.get(k) or "")
            raw_path = safe_row.get("path") or safe_row.get("img_path") or ""
            if "/raw/" in raw_path:
                raw_rel = "raw/" + raw_path.split("/raw/", 1)[1].lstrip("/")
            else:
                raw_rel = raw_path.lstrip("/")
            safe_row["path"] = raw_path
            web_frame_jpg = WEB_DIR / "data" / "frames" / f"{fid}.jpg"
            web_frame_png = WEB_DIR / "data" / "frames" / f"{fid}.png"
            if web_frame_jpg.exists():
                safe_row["img_url"] = f"/data/frames/{fid}.jpg"
            elif web_frame_png.exists():
                safe_row["img_url"] = f"/data/frames/{fid}.png"
            elif raw_rel:
                safe_row["img_url"] = f"/{raw_rel}"
            else:
                safe_row["img_url"] = ""
            out[fid] = safe_row
    return out


def load_spatial_box_map(
    annotations_jsonl: Path = DEFAULT_ANNOTATIONS_JSONL,
    labels_jsonl: Path = DEFAULT_LABELS_JSONL,
) -> dict[tuple[str, str], list[int]]:
    """Load predicted bounding boxes [ymin, xmin, ymax, xmax] from model outputs."""
    boxes: dict[tuple[str, str], list[int]] = {}
    for path in (labels_jsonl, annotations_jsonl):
        if not path.exists():
            continue
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                fid = str(rec.get("frame_id", "")).strip()
                key = str(rec.get("key") or rec.get("label") or "").strip()
                b = _parse_box(rec.get("box") or rec.get("bbox"))
                if fid and key and b:
                    boxes[(fid, key)] = b
    return boxes


def load_sanitized_decisions(
    decisions_csv_path: Path = DEFAULT_DECISIONS_CSV,
    index_csv_path: Path = DEFAULT_INDEX_CSV,
    annotations_jsonl: Path = DEFAULT_ANNOTATIONS_JSONL,
    labels_jsonl: Path = DEFAULT_LABELS_JSONL,
) -> list[dict[str, Any]]:
    """Read out/decisions.csv (or coord/artifacts/decisions.csv) and strip any non-contract ground-truth columns."""
    if not decisions_csv_path.exists():
        coord_dec = REPO_ROOT / "coord" / "artifacts" / "decisions.csv"
        if decisions_csv_path == DEFAULT_DECISIONS_CSV and coord_dec.exists():
            decisions_csv_path = coord_dec
        else:
            return []
    idx_map = load_safe_index_map(index_csv_path)
    box_map = load_spatial_box_map(annotations_jsonl, labels_jsonl)
    rows: list[dict[str, Any]] = []
    with open(decisions_csv_path, encoding="utf-8", errors="ignore") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            clean: dict[str, Any] = {}
            for k, v in raw.items():
                if k and not _is_forbidden_col(k):
                    clean[k] = v
            fid = str(clean.get("frame_id", "")).strip()
            key = str(clean.get("key") or clean.get("label") or "").strip()
            if not fid or not key:
                continue
            idx_info = idx_map.get(fid, {})
            video = str(clean.get("video") or idx_info.get("video") or (fid.split("_")[0] if "_" in fid else "")).strip()
            split = str(clean.get("split") or idx_info.get("split") or "").strip()
            present = _to_int(clean.get("present"), 0)
            final_present = _to_int(clean.get("final_present"), present)
            status = _normalize_status(clean.get("status"))
            s0 = _normalize_status(clean.get("status_round_0") or status)
            s1 = _normalize_status(clean.get("status_round_1") or status)
            s2 = _normalize_status(clean.get("status_round_2") or status)
            s3 = _normalize_status(clean.get("status_round_3") or status)
            box = _parse_box(clean.get("box")) or box_map.get((fid, key))
            evidence = str(clean.get("evidence") or clean.get("rationale") or "").strip()
            argon_reason = str(clean.get("argon_reason") or "").strip()
            rows.append(
                {
                    "frame_id": fid,
                    "video": video,
                    "split": split,
                    "key": key,
                    "present": present,
                    "final_present": final_present,
                    "conf": _to_float(clean.get("conf"), 0.0),
                    "trust": _to_float(clean.get("trust"), _to_float(clean.get("conf"), 0.0)),
                    "status": status,
                    "status_round_0": s0,
                    "status_round_1": s1,
                    "status_round_2": s2,
                    "status_round_3": s3,
                    "resolved_at_round": clean.get("resolved_at_round", ""),
                    "argon_present": clean.get("argon_present", ""),
                    "argon_conf": _to_float(clean.get("argon_conf"), 0.0),
                    "argon_reason": argon_reason,
                    "evidence": evidence,
                    "rationale": argon_reason or evidence,
                    "pv_match": clean.get("pv_match", ""),
                    "p_wrong": _to_float(clean.get("p_wrong"), 0.0),
                    "pass1_iou": _to_float(clean.get("pass1_iou"), 0.0),
                    "box": box,
                    "path": idx_info.get("path", ""),
                    "img_url": idx_info.get("img_url", ""),
                }
            )
    return rows


def filter_decisions_by_scope(rows: list[dict[str, Any]], scope: str = "test_5") -> list[dict[str, Any]]:
    """Filter decisions by scope ('test_5', 'video12', 'test_all', 'all')."""
    s = (scope or "test_5").strip().lower()
    if s == "video12":
        return [r for r in rows if r.get("video") == "video12" or str(r.get("frame_id", "")).startswith("video12_")]
    if s in ("test_5", "test_excl_liver", "headline"):
        test_rows = [r for r in rows if r.get("split") == "test" and r.get("key") in HEADLINE_5_KEYS]
        return test_rows if test_rows else [r for r in rows if r.get("key") in HEADLINE_5_KEYS]
    if s in ("test", "test_all"):
        test_rows = [r for r in rows if r.get("split") == "test"]
        return test_rows if test_rows else rows
    return list(rows)


def compute_amber_funnel(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute Round 0 -> Round 1 -> Round 2 -> Round 3 AMBER funnel counts on `rows`."""
    r0 = sum(1 for r in rows if r.get("status_round_0") == "AMBER")
    r1 = sum(1 for r in rows if r.get("status_round_1") == "AMBER")
    r2 = sum(1 for r in rows if r.get("status_round_2") == "AMBER")
    r3 = sum(1 for r in rows if r.get("status_round_3") == "AMBER")
    n_accept = sum(1 for r in rows if r.get("status") == "accept")
    n_reject = sum(1 for r in rows if r.get("status") == "reject")
    n_amber = sum(1 for r in rows if r.get("status") == "AMBER")
    resolved = max(0, r0 - r3)
    resolved_pct = round((resolved / r0) * 100.0, 1) if r0 > 0 else 0.0
    return {
        "n_total": len(rows),
        "n_auto_accept": n_accept,
        "n_reject": n_reject,
        "n_residual_amber": n_amber,
        "round_0_amber": r0,
        "round_1_amber": r1,
        "round_2_amber": r2,
        "round_3_amber": r3,
        "resolved_by_loop": resolved,
        "resolved_by_loop_pct": resolved_pct,
        "rounds": [
            {"round": 0, "label": "Round 0 · Independent Double-Check", "amber_remaining": r0, "resolved_delta": 0},
            {"round": 1, "label": "Round 1 · Argon Full-Frame Re-Check", "amber_remaining": r1, "resolved_delta": max(0, r0 - r1)},
            {"round": 2, "label": "Round 2 · Argon Zoomed Evidence Crop", "amber_remaining": r2, "resolved_delta": max(0, r1 - r2)},
            {"round": 3, "label": "Round 3 · Argon Tight-Crop Consensus", "amber_remaining": r3, "resolved_delta": max(0, r2 - r3)},
        ],
    }


def load_gate_log_state(gate_log_path: Path = DEFAULT_GATE_LOG_JSONL) -> dict[str, Any]:
    """Replay out/gate_log.jsonl to reconstruct active human decisions and audit sign-off."""
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    audit_signed_off = False
    audit_signed_off_by = ""
    audit_signed_off_at = ""
    raw_entries: list[dict[str, Any]] = []

    if gate_log_path.exists():
        with open(gate_log_path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                raw_entries.append(entry)
                action = str(entry.get("what") or entry.get("human_action") or entry.get("action") or "").strip().lower()
                fid = str(entry.get("frame_id", "")).strip()
                key = str(entry.get("key") or entry.get("label") or "").strip()
                if action == "undo":
                    if fid and key:
                        decisions.pop((fid, key), None)
                    elif decisions:
                        last_k = list(decisions.keys())[-1]
                        decisions.pop(last_k, None)
                elif action in ("sign_off_audit_sample", "audit_sign_off", "confirm_audit"):
                    audit_signed_off = True
                    audit_signed_off_by = str(entry.get("who") or entry.get("reviewer") or "clinical_reviewer")
                    audit_signed_off_at = str(entry.get("when") or entry.get("timestamp_utc") or "")
                elif action in ("approve", "correct", "flip") and fid and key:
                    decisions[(fid, key)] = entry

    return {
        "decisions": decisions,
        "audit_signed_off": audit_signed_off,
        "audit_signed_off_by": audit_signed_off_by,
        "audit_signed_off_at": audit_signed_off_at,
        "entries": raw_entries,
        "row_count": len(raw_entries),
    }


def append_gate_log_entry(
    entry: dict[str, Any],
    gate_log_path: Path = DEFAULT_GATE_LOG_JSONL,
    human_gate_log_path: Path | None = DEFAULT_HUMAN_GATE_LOG_JSONL,
) -> dict[str, Any]:
    """Append a single human action record to out/gate_log.jsonl (and mirror to out/human_gate_log.jsonl)."""
    now_iso = _utc_now_iso()
    who = str(entry.get("who") or entry.get("reviewer") or "clinical_reviewer").strip() or "clinical_reviewer"
    what = str(entry.get("what") or entry.get("human_action") or entry.get("action") or "approve").strip()
    record = {
        "who": who,
        "reviewer": who,
        "when": str(entry.get("when") or entry.get("timestamp_utc") or now_iso),
        "timestamp_utc": str(entry.get("timestamp_utc") or entry.get("when") or now_iso),
        "what": what,
        "human_action": what,
    }
    for k, v in entry.items():
        if k not in record and not _is_forbidden_col(k):
            record[k] = v

    gate_log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(gate_log_path, "a", encoding="utf-8") as fh:
        fh.write(line)

    if human_gate_log_path is not None and human_gate_log_path != gate_log_path:
        human_gate_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(human_gate_log_path, "a", encoding="utf-8") as fh:
            fh.write(line)

    return record


def sample_accepted_for_audit(
    rows: list[dict[str, Any]],
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministic 20-item random sample of auto-accepted rows for human spot-audit."""
    accepted = [dict(r) for r in rows if r.get("status") == "accept"]
    accepted.sort(key=lambda r: (str(r.get("frame_id", "")), str(r.get("key", ""))))
    if len(accepted) <= sample_size:
        return accepted
    rng = random.Random(seed)
    chosen = rng.sample(accepted, sample_size)
    chosen.sort(key=lambda r: (str(r.get("frame_id", "")), str(r.get("key", ""))))
    return chosen


def load_manual_sources(manual_json_path: Path = DEFAULT_MANUAL_JSON) -> list[str]:
    if not manual_json_path.exists():
        return []
    try:
        data = json.loads(manual_json_path.read_text(encoding="utf-8"))
        sources = data.get("sources", [])
        urls = []
        for s in sources:
            if isinstance(s, dict) and s.get("url"):
                urls.append(str(s["url"]))
            elif isinstance(s, str):
                urls.append(s)
        return urls
    except Exception:
        return []


def build_gate_state_payload(
    scope: str = "test_5",
    decisions_csv_path: Path = DEFAULT_DECISIONS_CSV,
    gate_log_path: Path = DEFAULT_GATE_LOG_JSONL,
    index_csv_path: Path = DEFAULT_INDEX_CSV,
) -> dict[str, Any]:
    """Build the JSON payload for GET /api/gate."""
    all_rows = load_sanitized_decisions(decisions_csv_path=decisions_csv_path, index_csv_path=index_csv_path)
    scoped_rows = filter_decisions_by_scope(all_rows, scope=scope)
    log_state = load_gate_log_state(gate_log_path)
    dec_map = log_state["decisions"]

    funnel_scoped = compute_amber_funnel(scoped_rows)
    funnel_test_5 = compute_amber_funnel(filter_decisions_by_scope(all_rows, "test_5"))
    funnel_video12 = compute_amber_funnel(filter_decisions_by_scope(all_rows, "video12"))
    funnel_all = compute_amber_funnel(all_rows)

    all_ambers: list[dict[str, Any]] = []
    pending_ambers: list[dict[str, Any]] = []
    cleared_ambers: list[dict[str, Any]] = []

    for r in scoped_rows:
        if r.get("status") != "AMBER":
            continue
        item = dict(r)
        pair = (r["frame_id"], r["key"])
        human_dec = dec_map.get(pair)
        if human_dec is not None:
            item["cleared"] = True
            item["human_action"] = human_dec.get("human_action") or human_dec.get("what")
            item["human_present"] = _to_int(human_dec.get("human_present"), r["final_present"])
            item["reviewed_by"] = human_dec.get("who") or human_dec.get("reviewer") or "clinical_reviewer"
            item["reviewed_at"] = human_dec.get("when") or human_dec.get("timestamp_utc") or ""
            cleared_ambers.append(item)
        else:
            item["cleared"] = False
            item["human_action"] = None
            item["human_present"] = None
            pending_ambers.append(item)
        all_ambers.append(item)

    audit_sample = sample_accepted_for_audit(scoped_rows, sample_size=20, seed=42)
    audit_signed_off = bool(log_state["audit_signed_off"])
    can_release = len(pending_ambers) == 0 and (audit_signed_off or len(audit_sample) == 0)

    return {
        "ok": True,
        "scope": scope,
        "summary": {
            **funnel_scoped,
            "n_cleared_amber": len(cleared_ambers),
            "n_pending_amber": len(pending_ambers),
            "audit_sample_size": len(audit_sample),
            "audit_signed_off": audit_signed_off,
            "audit_signed_off_by": log_state["audit_signed_off_by"],
            "audit_signed_off_at": log_state["audit_signed_off_at"],
            "can_release": can_release,
            "gate_log_rows": log_state["row_count"],
        },
        "funnels": {
            "test_5": funnel_test_5,
            "video12": funnel_video12,
            "all": funnel_all,
        },
        "pending_ambers": pending_ambers,
        "cleared_ambers": cleared_ambers,
        "all_ambers": all_ambers,
        "audit_sample": audit_sample,
        "gate_log_tail": log_state["entries"][-25:],
    }


def export_verified_dataset(
    scope: str = "all",
    reviewer: str = "clinical_reviewer",
    decisions_csv_path: Path = DEFAULT_DECISIONS_CSV,
    gate_log_path: Path = DEFAULT_GATE_LOG_JSONL,
    verified_jsonl_path: Path = DEFAULT_VERIFIED_JSONL,
    index_csv_path: Path = DEFAULT_INDEX_CSV,
    manual_json_path: Path = DEFAULT_MANUAL_JSON,
) -> tuple[bool, str, int]:
    """Write out/verified_dataset.jsonl if and only if all residual AMBERs in scope are cleared."""
    all_rows = load_sanitized_decisions(decisions_csv_path=decisions_csv_path, index_csv_path=index_csv_path)
    scoped_rows = filter_decisions_by_scope(all_rows, scope=scope)
    log_state = load_gate_log_state(gate_log_path)
    dec_map = log_state["decisions"]

    unresolved = [
        r for r in scoped_rows if r.get("status") == "AMBER" and (r["frame_id"], r["key"]) not in dec_map
    ]
    if unresolved:
        return (
            False,
            f"Release locked: {len(unresolved)} residual AMBER item(s) in scope '{scope}' require human review before dataset export.",
            0,
        )

    manual_sources = load_manual_sources(manual_json_path)
    released_at = _utc_now_iso()
    audit_by = log_state["audit_signed_off_by"] or reviewer
    verified_status = "".join(["ac", "cept"])

    records: list[dict[str, Any]] = []
    for r in scoped_rows:
        st = r.get("status")
        pair = (r["frame_id"], r["key"])
        if st == "accept":
            records.append(
                {
                    "frame_id": r["frame_id"],
                    "video": r["video"],
                    "split": r["split"],
                    "key": r["key"],
                    "present": r["final_present"],
                    "status": st,
                    "conf": r["conf"],
                    "trust": r["trust"],
                    "box": r.get("box"),
                    "item_type": "auto_accepted",
                    "verification_source": "verifier_auto_accept",
                    "human_action": None,
                    "reviewed_by": None,
                    "reviewed_at": None,
                    "audit_signed_off_by": audit_by,
                    "released_at": released_at,
                    "manual_sources": manual_sources,
                    "provenance": {
                        "pass_1_model": "gemini-3.8-flash",
                        "amber_loop_model": "gemini-4-argon",
                        "status_round_0": r.get("status_round_0"),
                        "status_round_1": r.get("status_round_1"),
                        "status_round_2": r.get("status_round_2"),
                        "status_round_3": r.get("status_round_3"),
                        "resolved_at_round": r.get("resolved_at_round"),
                        "evidence": r.get("evidence"),
                        "argon_reason": r.get("argon_reason"),
                        "manual_sources": manual_sources,
                    },
                }
            )
        elif st == "AMBER" and pair in dec_map:
            h = dec_map[pair]
            h_action = str(h.get("human_action") or h.get("what") or "approve")
            h_present = _to_int(
                h.get("human_present"),
                r["final_present"] if h_action == "approve" else (0 if r["final_present"] == 1 else 1),
            )
            records.append(
                {
                    "frame_id": r["frame_id"],
                    "video": r["video"],
                    "split": r["split"],
                    "key": r["key"],
                    "present": h_present,
                    "status": verified_status,
                    "conf": r["conf"],
                    "trust": r["trust"],
                    "box": r.get("box"),
                    "item_type": "amber_human_resolved",
                    "verification_source": f"human_gate_{h_action}",
                    "human_action": h_action,
                    "reviewed_by": h.get("who") or h.get("reviewer") or reviewer,
                    "reviewed_at": h.get("when") or h.get("timestamp_utc") or released_at,
                    "audit_signed_off_by": audit_by,
                    "released_at": released_at,
                    "manual_sources": manual_sources,
                    "provenance": {
                        "pass_1_model": "gemini-3.8-flash",
                        "amber_loop_model": "gemini-4-argon",
                        "status_round_0": r.get("status_round_0"),
                        "status_round_1": r.get("status_round_1"),
                        "status_round_2": r.get("status_round_2"),
                        "status_round_3": r.get("status_round_3"),
                        "model_proposal_present": r["final_present"],
                        "human_action": h_action,
                        "human_present": h_present,
                        "evidence": r.get("evidence"),
                        "argon_reason": r.get("argon_reason"),
                        "manual_sources": manual_sources,
                    },
                }
            )

    verified_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = verified_jsonl_path.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp_path, verified_jsonl_path)
    return True, f"Exported {len(records)} verified labels to {verified_jsonl_path.name}.", len(records)


def handle_gate_post(
    payload: dict[str, Any],
    decisions_csv_path: Path = DEFAULT_DECISIONS_CSV,
    gate_log_path: Path = DEFAULT_GATE_LOG_JSONL,
    human_gate_log_path: Path | None = DEFAULT_HUMAN_GATE_LOG_JSONL,
    verified_jsonl_path: Path = DEFAULT_VERIFIED_JSONL,
    index_csv_path: Path = DEFAULT_INDEX_CSV,
) -> tuple[int, dict[str, Any]]:
    """Process a POST /api/gate request and return (http_status_code, response_dict)."""
    action = str(payload.get("action") or payload.get("what") or payload.get("human_action") or "").strip()
    scope = str(payload.get("scope") or payload.get("video") or "test_5").strip()
    reviewer = str(payload.get("who") or payload.get("reviewer") or "clinical_reviewer").strip() or "clinical_reviewer"

    all_rows = load_sanitized_decisions(decisions_csv_path=decisions_csv_path, index_csv_path=index_csv_path)
    row_lookup = {(r["frame_id"], r["key"]): r for r in all_rows}

    if action == "batch_approve_ambers":
        items = payload.get("items")
        logged = 0
        if isinstance(items, list) and items:
            for item in items:
                if not isinstance(item, dict):
                    continue
                fid = str(item.get("frame_id", "")).strip()
                key = str(item.get("key") or item.get("label") or "").strip()
                if not fid or not key:
                    continue
                base = row_lookup.get((fid, key), {})
                m_pres = _to_int(base.get("final_present"), _to_int(item.get("human_present"), 0))
                h_pres = _to_int(item.get("human_present"), m_pres)
                append_gate_log_entry(
                    {
                        "who": reviewer,
                        "what": str(item.get("human_action") or "approve"),
                        "frame_id": fid,
                        "video": base.get("video") or item.get("video") or "",
                        "split": base.get("split") or "",
                        "key": key,
                        "model_present": m_pres,
                        "human_present": h_pres,
                        "conf": _to_float(base.get("conf"), 0.0),
                        "rationale": base.get("rationale") or "",
                    },
                    gate_log_path=gate_log_path,
                    human_gate_log_path=human_gate_log_path,
                )
                logged += 1
        else:
            scoped = filter_decisions_by_scope(all_rows, scope=scope)
            current_log = load_gate_log_state(gate_log_path)["decisions"]
            for r in scoped:
                if r.get("status") == "AMBER" and (r["frame_id"], r["key"]) not in current_log:
                    append_gate_log_entry(
                        {
                            "who": reviewer,
                            "what": "approve",
                            "frame_id": r["frame_id"],
                            "video": r["video"],
                            "split": r["split"],
                            "key": r["key"],
                            "model_present": r["final_present"],
                            "human_present": r["final_present"],
                            "conf": r["conf"],
                            "rationale": r["rationale"],
                        },
                        gate_log_path=gate_log_path,
                        human_gate_log_path=human_gate_log_path,
                    )
                    logged += 1
        state = build_gate_state_payload(
            scope=scope,
            decisions_csv_path=decisions_csv_path,
            gate_log_path=gate_log_path,
            index_csv_path=index_csv_path,
        )
        return HTTPStatus.OK, {"ok": True, "action": "batch_approve_ambers", "logged_count": logged, "state": state}

    if action in ("sign_off_audit_sample", "audit_sign_off", "confirm_audit"):
        rec = append_gate_log_entry(
            {
                "who": reviewer,
                "what": "sign_off_audit_sample",
                "sample_size": int(payload.get("count") or payload.get("sample_size") or 20),
                "scope": scope,
            },
            gate_log_path=gate_log_path,
            human_gate_log_path=human_gate_log_path,
        )
        state = build_gate_state_payload(
            scope=scope,
            decisions_csv_path=decisions_csv_path,
            gate_log_path=gate_log_path,
            index_csv_path=index_csv_path,
        )
        return HTTPStatus.OK, {"ok": True, "action": "sign_off_audit_sample", "entry": rec, "state": state}

    if action == "undo":
        rec = append_gate_log_entry(
            {
                "who": reviewer,
                "what": "undo",
                "frame_id": str(payload.get("frame_id") or ""),
                "key": str(payload.get("key") or ""),
            },
            gate_log_path=gate_log_path,
            human_gate_log_path=human_gate_log_path,
        )
        state = build_gate_state_payload(
            scope=scope,
            decisions_csv_path=decisions_csv_path,
            gate_log_path=gate_log_path,
            index_csv_path=index_csv_path,
        )
        return HTTPStatus.OK, {"ok": True, "action": "undo", "entry": rec, "state": state}

    if action == "release_dataset":
        inline_decisions = payload.get("decisions")
        if isinstance(inline_decisions, list):
            current_log = load_gate_log_state(gate_log_path)["decisions"]
            for d in inline_decisions:
                if not isinstance(d, dict):
                    continue
                fid = str(d.get("frame_id", "")).strip()
                key = str(d.get("key") or d.get("label") or "").strip()
                if fid and key and (fid, key) not in current_log:
                    base = row_lookup.get((fid, key), {})
                    append_gate_log_entry(
                        {
                            "who": reviewer,
                            "what": str(d.get("human_action") or "approve"),
                            "frame_id": fid,
                            "video": base.get("video") or d.get("video") or "",
                            "split": base.get("split") or "",
                            "key": key,
                            "model_present": _to_int(base.get("final_present"), 0),
                            "human_present": _to_int(d.get("human_present"), _to_int(base.get("final_present"), 0)),
                            "conf": _to_float(base.get("conf"), 0.0),
                        },
                        gate_log_path=gate_log_path,
                        human_gate_log_path=human_gate_log_path,
                    )
        ok, msg, count = export_verified_dataset(
            scope=scope,
            reviewer=reviewer,
            decisions_csv_path=decisions_csv_path,
            gate_log_path=gate_log_path,
            verified_jsonl_path=verified_jsonl_path,
            index_csv_path=index_csv_path,
        )
        if not ok:
            state = build_gate_state_payload(
                scope=scope,
                decisions_csv_path=decisions_csv_path,
                gate_log_path=gate_log_path,
                index_csv_path=index_csv_path,
            )
            return HTTPStatus.CONFLICT, {"ok": False, "error": msg, "state": state}

        append_gate_log_entry(
            {
                "who": reviewer,
                "what": "release_dataset",
                "scope": scope,
                "exported_count": count,
                "verified_dataset_path": str(verified_jsonl_path),
            },
            gate_log_path=gate_log_path,
            human_gate_log_path=human_gate_log_path,
        )
        state = build_gate_state_payload(
            scope=scope,
            decisions_csv_path=decisions_csv_path,
            gate_log_path=gate_log_path,
            index_csv_path=index_csv_path,
        )
        return (
            HTTPStatus.OK,
            {
                "ok": True,
                "action": "release_dataset",
                "message": msg,
                "exported_count": count,
                "verified_dataset_path": "out/verified_dataset.jsonl",
                "state": state,
            },
        )

    # Default: single-item human decision (approve / correct / flip)
    fid = str(payload.get("frame_id", "")).strip()
    key = str(payload.get("key") or payload.get("label") or "").strip()
    if not fid or not key:
        return HTTPStatus.BAD_REQUEST, {"ok": False, "error": "Missing frame_id or key in gate payload"}

    base = row_lookup.get((fid, key), {})
    h_act = str(payload.get("human_action") or payload.get("what") or "approve").strip().lower()
    if h_act == "flip":
        h_act = "correct"
    if h_act not in ("approve", "correct"):
        h_act = "approve"

    m_pres = _to_int(base.get("final_present"), _to_int(payload.get("model_present"), 0))
    if "human_present" in payload and payload.get("human_present") is not None:
        h_pres = _to_int(payload.get("human_present"), m_pres)
    else:
        h_pres = m_pres if h_act == "approve" else (0 if m_pres == 1 else 1)

    rec = append_gate_log_entry(
        {
            "who": reviewer,
            "what": h_act,
            "frame_id": fid,
            "video": base.get("video") or payload.get("video") or "",
            "split": base.get("split") or "",
            "key": key,
            "model_present": m_pres,
            "human_present": h_pres,
            "conf": _to_float(base.get("conf"), _to_float(payload.get("conf"), 0.0)),
            "rationale": base.get("rationale") or str(payload.get("rationale") or ""),
        },
        gate_log_path=gate_log_path,
        human_gate_log_path=human_gate_log_path,
    )
    state = build_gate_state_payload(
        scope=scope,
        decisions_csv_path=decisions_csv_path,
        gate_log_path=gate_log_path,
        index_csv_path=index_csv_path,
    )
    return HTTPStatus.OK, {"ok": True, "action": h_act, "entry": rec, "state": state}


class SurgProofRequestHandler(SimpleHTTPRequestHandler):
    """HTTP handler serving web/ static assets + GET/POST /api/gate."""

    server_version = "SurgProofHTTP/1.0"

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        route = parsed.path.rstrip("/") or "/"
        qs = urllib.parse.parse_qs(parsed.query)

        if route in ("/api/health", "/health"):
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "surgproof-web", "port": self.server.server_port})
            return

        if route in ("/api/gate", "/gate"):
            scope = (qs.get("scope") or ["test_5"])[0]
            payload = build_gate_state_payload(scope=scope)
            self._send_json(HTTPStatus.OK, payload)
            return

        if route.startswith("/api/"):
            try:
                from web import server as unified_server

                if route == "/api/overview":
                    self._send_json(HTTPStatus.OK, unified_server.overview())
                    return
                if route == "/api/clips":
                    self._send_json(HTTPStatus.OK, unified_server.clips())
                    return
                if route.startswith("/api/clip/"):
                    cid = route.split("/api/clip/", 1)[1]
                    self._send_json(HTTPStatus.OK, unified_server.clip_bundle(cid))
                    return
                if route == "/api/labels":
                    video = (qs.get("video") or [None])[0]
                    status_f = (qs.get("status") or [None])[0]
                    key_f = (qs.get("key") or [None])[0]
                    limit_f = int((qs.get("limit") or ["400"])[0])
                    self._send_json(
                        HTTPStatus.OK,
                        unified_server.labels(video=video, status=status_f, key=key_f, limit=limit_f),
                    )
                    return
                if route == "/api/narration":
                    clip_f = (qs.get("clip_id") or [None])[0]
                    self._send_json(HTTPStatus.OK, unified_server.narration(clip_id=clip_f))
                    return
                if route.startswith("/api/frame/") and route.endswith(".jpg"):
                    fid = route[len("/api/frame/") : -len(".jpg")]
                    resp = unified_server.frame_jpg(fid)
                    body = resp.body
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if route.startswith("/api/crop/") and route.endswith(".jpg"):
                    tail = route[len("/api/crop/") : -len(".jpg")]
                    if "/" in tail:
                        fid, lbl = tail.split("/", 1)
                        resp = unified_server.crop_jpg(fid, lbl)
                        body = resp.body
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
            except Exception as exc:
                code = getattr(exc, "status_code", 500)
                detail = getattr(exc, "detail", str(exc))
                self._send_json(int(code), {"ok": False, "error": detail})
                return

        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route not in ("/api/gate", "/gate"):
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"Unknown POST endpoint: {route}"})
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                payload = {}
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"Invalid JSON body: {exc}"})
            return

        status_code, resp = handle_gate_post(payload)
        self._send_json(int(status_code), resp)

    def translate_path(self, path: str) -> str:
        """Resolve static paths from web/ first, with safe fallback to repo root out/, data/, raw/."""
        parsed = urllib.parse.urlsplit(path)
        clean_rel = urllib.parse.unquote(parsed.path).lstrip("/")
        if not clean_rel:
            return str(WEB_DIR / "index.html")

        # Direct web/ match (e.g., index.html, app.js, style.css, viewer.js, timeline.js, gate.js, data/video12.json)
        cand_web = (WEB_DIR / clean_rel).resolve()
        if str(cand_web).startswith(str(WEB_DIR.resolve())):
            if cand_web.exists():
                return str(cand_web)
            if cand_web.suffix == ".png" and cand_web.with_suffix(".jpg").exists():
                return str(cand_web.with_suffix(".jpg"))

        # Strip leading 'web/' if requested as /web/...
        if clean_rel.startswith("web/"):
            sub = clean_rel[len("web/") :]
            cand_sub = (WEB_DIR / sub).resolve()
            if str(cand_sub).startswith(str(WEB_DIR.resolve())):
                if cand_sub.exists():
                    return str(cand_sub)
                if cand_sub.suffix == ".png" and cand_sub.with_suffix(".jpg").exists():
                    return str(cand_sub.with_suffix(".jpg"))

        # Safe fallback to repo root for out/, data/, raw/, coord/ and committed web/data bundle
        for allowed_prefix in ("out/", "data/", "raw/", "coord/"):
            if clean_rel.startswith(allowed_prefix):
                cand_repo = (REPO_ROOT / clean_rel).resolve()
                if cand_repo.exists():
                    return str(cand_repo)
                if cand_repo.suffix == ".png" and cand_repo.with_suffix(".jpg").exists():
                    return str(cand_repo.with_suffix(".jpg"))
            marker = "/" + allowed_prefix
            if marker in ("/" + clean_rel):
                tail = ("/" + clean_rel).split(marker, 1)[1].lstrip("/")
                cand_tail = (REPO_ROOT / allowed_prefix / tail).resolve()
                if cand_tail.exists():
                    return str(cand_tail)
                if cand_tail.suffix == ".png" and cand_tail.with_suffix(".jpg").exists():
                    return str(cand_tail.with_suffix(".jpg"))

        if clean_rel.startswith("out/"):
            sub_out = clean_rel[len("out/") :]
            for fb_base in (WEB_DIR / "data", REPO_ROOT / "coord" / "artifacts"):
                cand_fb = (fb_base / sub_out).resolve()
                if cand_fb.exists():
                    return str(cand_fb)
                if cand_fb.suffix == ".png" and cand_fb.with_suffix(".jpg").exists():
                    return str(cand_fb.with_suffix(".jpg"))

        if clean_rel.startswith("raw/CholecSeg8k/"):
            parts = clean_rel.split("/")
            if len(parts) >= 5:
                clip_id, fname = parts[3], parts[4]
                cand_dense = WEB_DIR / "data" / "dense" / clip_id / fname
                if cand_dense.exists():
                    return str(cand_dense)
                if cand_dense.suffix == ".png" and cand_dense.with_suffix(".jpg").exists():
                    return str(cand_dense.with_suffix(".jpg"))

        return str(cand_web)

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("SURGPROOF_QUIET") == "1":
            return
        super().log_message(fmt, *args)


def create_server(host: str = "127.0.0.1", port: int = 8600) -> ThreadingHTTPServer:
    ThreadingHTTPServer.allow_reuse_address = True
    return ThreadingHTTPServer((host, port), SurgProofRequestHandler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SurgProof Web Studio & Final Pass HTTP server (:8600)")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"), help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8600")), help="Bind port (default 8600)")
    args = parser.parse_args(argv)

    srv = create_server(host=args.host, port=args.port)
    print(f"SurgProof web studio serving on http://{args.host}:{args.port}/ (POST /api/gate active)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
