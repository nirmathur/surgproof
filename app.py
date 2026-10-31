"""SurgProof Demo UI.  streamlit run app.py --server.headless true"""
import asyncio
import datetime
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from pydantic import BaseModel

import config as C
import gem
import label as L

st.set_page_config(page_title="SurgProof — No human labels. Only proof.", layout="wide")
O = C.OUT_DIR
COORD_ARTIFACTS = C.ROOT / "coord" / "artifacts"
COL = {
    "accept": "#1f9d55",
    "AMBER": "#d69e2e",
    "amber": "#d69e2e",
    "reanalyse": "#d69e2e",
    "reject": "#e53e3e",
    "escalate": "#e53e3e",
}
NON_TRIVIAL_KEYS = [k for k in C.LABELS if k != "liver"]
CORE_BALANCED_KEYS = ["grasper", "l_hook", "gallbladder"]


class LiveVerdict(BaseModel):
    present: bool
    confidence: float
    reasoning: str


def resolve_artifact(rel_path: str | Path) -> Path:
    """Resolve an artifact from out/ first, falling back to coord/artifacts/."""
    p = O / rel_path
    if p.exists():
        return p
    alt = COORD_ARTIFACTS / Path(rel_path).name
    if alt.exists():
        return alt
    return p


def load_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def normalize_status(val: str) -> str:
    """Normalize verifier status to canonical accept | reject | AMBER per docs/DESIGN.md."""
    s = str(val or "").strip().lower()
    if s == "accept":
        return "accept"
    if s in ("reject", "escalate"):
        return "reject"
    return "AMBER"


def parse_box(raw_box) -> list[float] | None:
    """Safely parse a normalized [ymin, xmin, ymax, xmax] bounding box in 0..1000."""
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
    """Load frame image from disk, or return a clearly labelled placeholder canvas if unavailable."""
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
    im = Image.new("RGB", (640, 480), color=(28, 32, 38))
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 628, 468], outline=(90, 100, 115), width=2)
    title = f"[PLACEHOLDER CANVAS — Frame file unavailable: {frame_id or 'awaiting data/index.csv'}]"
    d.text((28, 220), title, fill=(210, 215, 225))
    d.text((28, 248), "Real endoscopic frame renders here once raw/ & data/index.csv are present.", fill=(150, 160, 175))
    return im


def draw(path, rows: pd.DataFrame, reveal: bool, frame_id: str = "") -> Image.Image:
    """Render frame image with verification-status (or ground-truth reveal) bounding boxes."""
    im = resolve_frame_image(path, frame_id=frame_id)
    w, h = im.size
    d = ImageDraw.Draw(im)
    if rows is None or rows.empty:
        return im
    for r in rows.itertuples():
        box = parse_box(getattr(r, "box", None))
        final_present = bool(getattr(r, "final_present", getattr(r, "present", 0)))
        if final_present and box is not None:
            y0, x0, y1, x1 = box
            status_val = getattr(r, "status", "AMBER")
            colour = COL.get(status_val, "#d69e2e")
            if reveal and hasattr(r, "final_correct") and pd.notna(getattr(r, "final_correct", None)):
                colour = "#1f9d55" if getattr(r, "final_correct", 0) else "#e53e3e"
            d.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], outline=colour, width=4)
            d.text((x0 * w / 1000 + 4, y0 * h / 1000 + 4), str(getattr(r, "key", "")), fill=colour)
    return im


def frame_sort_key(frame_id: str):
    parts = str(frame_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (str(frame_id), 0)


def join_ground_truth_for_reveal(dec_df: pd.DataFrame, idx_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join hidden ground truth from data/index.csv ONLY at evaluation/reveal time (ignoring gt == -1)."""
    df = dec_df.copy()
    if "final_present" not in df.columns and "present" in df.columns:
        df["final_present"] = df["present"]
    if "truth" not in df.columns and idx_df is not None:
        idx_lookup = idx_df.set_index("frame_id") if "frame_id" in idx_df.columns else idx_df
        truths = []
        for r in df.itertuples():
            col = f"gt_{r.key}"
            if r.frame_id in idx_lookup.index and col in idx_lookup.columns:
                val = idx_lookup.loc[r.frame_id, col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                ival = int(val) if pd.notna(val) else -1
                truths.append(ival if ival in (0, 1) else np.nan)
            else:
                truths.append(np.nan)
        df["truth"] = truths
    elif "truth" in df.columns:
        df["truth"] = df["truth"].where(df["truth"].isin([0, 1]), np.nan)
    valid = df["truth"].isin([0, 1])
    if valid.any():
        if "correct" not in df.columns and "present" in df.columns:
            df["correct"] = np.where(valid, (df["present"].astype(float) == df["truth"].astype(float)).astype(float), np.nan)
        if "final_correct" not in df.columns:
            df["final_correct"] = np.where(valid, (df["final_present"].astype(float) == df["truth"].astype(float)).astype(float), np.nan)
    return df


def score_jsonl_against_index(jsonl_path: Path, idx_df: pd.DataFrame | None, split_filter: str = "test") -> dict | None:
    """Compute real accuracy metrics from a leak-free labels_*.jsonl joined with index.csv on held-out test split."""
    if not jsonl_path.exists() or idx_df is None or idx_df.empty:
        return None
    try:
        d = pd.read_json(jsonl_path, lines=True)
        if d.empty:
            return None
        if split_filter and "split" in d.columns:
            d = d[d["split"] == split_filter]
        if d.empty:
            return None
        joined = join_ground_truth_for_reveal(d, idx_df)
        scorable = joined[joined["truth"].isin([0, 1])]
        if scorable.empty:
            return None
        nt = scorable[scorable["key"].isin(NON_TRIVIAL_KEYS)]
        c3 = scorable[scorable["key"].isin(CORE_BALANCED_KEYS)]
        per_key = scorable.groupby("key")["correct"].mean().round(4).to_dict()
        return {
            "n_rows": int(len(scorable)),
            "n_frames": int(scorable["frame_id"].nunique()),
            "acc_core_3": round(float(c3["correct"].mean()), 4) if not c3.empty else None,
            "acc_5_nontrivial": round(float(nt["correct"].mean()), 4) if not nt.empty else None,
            "acc_all_6": round(float(scorable["correct"].mean()), 4),
            "per_key": per_key,
        }
    except Exception:
        return None


# ---------------- Human Gate (#14) Helpers ----------------
def load_human_gate_log(log_path: Path | None = None) -> list[dict]:
    p = log_path or (O / "human_gate_log.jsonl")
    if not p.exists():
        return []
    out = []
    try:
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if ln:
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
    except Exception:
        return []
    return out


def record_human_gate_decision(
    reviewer: str,
    frame_id: str,
    video: str,
    key: str,
    agent_status: str,
    agent_present: int,
    human_action: str,
    human_present: int | None,
    log_path: Path | None = None,
) -> dict:
    p = log_path or (O / "human_gate_log.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "t": round(time.time(), 3),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reviewer": reviewer.strip() or "Anonymous Surgeon",
        "frame_id": str(frame_id),
        "video": str(video),
        "key": str(key),
        "agent_status": str(agent_status),
        "agent_present": int(agent_present),
        "action": str(human_action),
        "human_action": str(human_action),
        "human_present": int(human_present) if human_present is not None else None,
    }
    with open(p, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def compute_amber_queue(dec_df: pd.DataFrame, gate_log: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Partition AMBER rows into pending vs human-cleared and check if export is unlocked."""
    if dec_df is None or dec_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {
            "n_total": 0,
            "n_accept": 0,
            "n_reject": 0,
            "n_amber_total": 0,
            "n_amber_pending": 0,
            "n_amber_cleared": 0,
            "export_unlocked": False,
        }
    df = dec_df.copy()
    df["norm_status"] = df["status"].map(normalize_status)
    latest_by_item = {}
    for rec in gate_log:
        if isinstance(rec, dict) and "frame_id" in rec and "key" in rec:
            latest_by_item[(str(rec["frame_id"]), str(rec["key"]))] = rec

    amber_df = df[df["norm_status"] == "AMBER"].copy()
    amber_df["is_cleared"] = [
        (str(r.frame_id), str(r.key)) in latest_by_item for r in amber_df.itertuples()
    ]
    pending_df = amber_df[~amber_df["is_cleared"]].copy()
    cleared_df = amber_df[amber_df["is_cleared"]].copy()
    if not cleared_df.empty:
        cleared_df["reviewer"] = [
            latest_by_item[(str(r.frame_id), str(r.key))].get("reviewer", "")
            for r in cleared_df.itertuples()
        ]
        cleared_df["human_action"] = [
            latest_by_item[(str(r.frame_id), str(r.key))].get("human_action", "")
            for r in cleared_df.itertuples()
        ]
        cleared_df["human_present"] = [
            latest_by_item[(str(r.frame_id), str(r.key))].get("human_present")
            for r in cleared_df.itertuples()
        ]

    summary = {
        "n_total": int(len(df)),
        "n_accept": int((df["norm_status"] == "accept").sum()),
        "n_reject": int((df["norm_status"] == "reject").sum()),
        "n_amber_total": int(len(amber_df)),
        "n_amber_pending": int(len(pending_df)),
        "n_amber_cleared": int(len(cleared_df)),
        "export_unlocked": bool(len(df) > 0 and len(pending_df) == 0),
    }
    return pending_df, cleared_df, summary


def build_verified_export(dec_df: pd.DataFrame, gate_log: list[dict]) -> dict:
    """Build exportable verified dataset once all AMBER items have been cleared by a human."""
    latest_by_item = {
        (str(r["frame_id"]), str(r["key"])): r
        for r in gate_log
        if isinstance(r, dict) and "frame_id" in r and "key" in r
    }
    records = []
    if dec_df is not None and not dec_df.empty:
        vid_col = "video" if "video" in dec_df.columns else "video_id"
        for r in dec_df.itertuples():
            ns = normalize_status(getattr(r, "status", ""))
            fid = str(r.frame_id)
            key = str(r.key)
            vid = str(getattr(r, vid_col, ""))
            if ns == "accept":
                records.append({
                    "frame_id": fid,
                    "video": vid,
                    "key": key,
                    "present": int(getattr(r, "final_present", getattr(r, "present", 0))),
                    "verification_source": f"agent_{getattr(r, 'final_by', 'flash')}",
                    "reviewer": None,
                })
            elif ns == "AMBER" and (fid, key) in latest_by_item:
                h = latest_by_item[(fid, key)]
                if h.get("human_present") is not None:
                    records.append({
                        "frame_id": fid,
                        "video": vid,
                        "key": key,
                        "present": int(h["human_present"]),
                        "verification_source": "human_gate",
                        "reviewer": h.get("reviewer"),
                        "approved_at": h.get("timestamp"),
                    })
    return {
        "project": "SurgProof",
        "exported_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_verified_labels": len(records),
        "human_gate_cleared": True,
        "labels": records,
    }


# ---------------- Live Mode (#15) Leak-Free Pipeline ----------------
def _live_trust_score(r) -> float:
    s = float(r.conf) * (1.0 - float(r.p_wrong))
    if int(r.present) == 1:
        s *= 1.0 if bool(pd.notna(r.pv_match) and r.pv_match) else 0.5
    return round(float(s), 4)


def _live_initial_status(score: float) -> str:
    if score >= 0.70:
        return "accept"
    if score < 0.35:
        return "reject"
    return "AMBER"


async def _live_do_frame_no_gt(row, manual: dict, sample_seed: int = 0) -> list[dict]:
    """Label and run decorrelated checks on a single frame via label.do_frame (zero GT access)."""
    recs = await L.do_frame(row, "manual", manual, sample_seed, full_checks=True)
    for r in recs:
        r["frame_path"] = str(row.path)
        for forbidden in ("truth", "correct", "final_correct"):
            r.pop(forbidden, None)
    return recs


async def _live_argon_revise(r, manual: dict) -> dict:
    v = await gem.call(
        f"You are the senior surgical reviewer (Gemini 4 Argon). A junior labeller said '{r.key}' is "
        f"{'PRESENT' if r.present else 'ABSENT'} (conf {r.conf:.2f}, evidence: {r.evidence}). "
        f"A blind crop check {'agreed' if r.pv_match else 'disagreed or was not possible'}. "
        f"A sceptical critic argued: {r.critic!r} (p_wrong {r.p_wrong:.2f}).\n"
        "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
        f"Manual: {json.dumps(manual)}",
        model=C.ARGON,
        images=[gem.img_part(r.frame_path)],
        schema=LiveVerdict,
        tag="live-argon-revise",
    )
    gem.trace(
        "argon",
        f"[LIVE] {r.frame_id} {r.key}: {'present' if v['present'] else 'absent'} "
        f"({v['confidence']:.2f}) - {v['reasoning'][:120]}",
    )
    return v


async def run_live_video_async(
    video_id: str,
    n_frames: int = 20,
    argon_budget: int = 12,
    sample_seed: int = 0,
    progress_cb=None,
) -> tuple[pd.DataFrame, dict]:
    """Run live Flash labelling + decorrelated checks + Argon revision on ~20 frames of a test video."""
    t0 = time.perf_counter()
    idx_df = pd.read_csv(C.DATA_DIR / "index.csv")
    v_df = idx_df[idx_df["video"] == video_id].copy()
    # Strip all ground-truth columns immediately so labeller & verifier cannot see GT
    safe_cols = [c for c in ["frame_id", "video", "clip", "path", "split"] if c in v_df.columns]
    v_df = v_df[safe_cols].reset_index(drop=True)
    if len(v_df) > n_frames > 0:
        positions = np.linspace(0, len(v_df) - 1, n_frames, dtype=int)
        v_df = v_df.iloc[positions].reset_index(drop=True)

    manual_path = resolve_artifact("manual.json")
    manual = load_json(manual_path, {})

    gem.trace("flash", f"[LIVE] Labelling & checking {len(v_df)} frames on held-out {video_id}")
    jobs = [_live_do_frame_no_gt(r, manual, sample_seed=sample_seed) for r in v_df.itertuples()]
    recs = []
    total_jobs = len(jobs)
    for idx_done, coro in enumerate(asyncio.as_completed(jobs), 1):
        recs.extend(await coro)
        if progress_cb is not None:
            progress_cb(idx_done, total_jobs, "flash")

    d = pd.DataFrame(recs)
    d["trust"] = d.apply(_live_trust_score, axis=1)
    d["status"] = d["trust"].map(_live_initial_status)
    d["final_present"] = d["present"]
    d["final_by"] = "flash"
    d["argon_reasoning"] = ""

    if argon_budget > 0 and not d.empty:
        amber_band = d[d["status"] == "AMBER"].sort_values("trust", ascending=False).head(argon_budget)
        if not amber_band.empty:
            gem.trace("argon", f"[LIVE] Reviewing {len(amber_band)} AMBER claims on {video_id}")
            verdicts = await asyncio.gather(*[_live_argon_revise(r, manual) for r in amber_band.itertuples()])
            for (row_idx, orig_row), v in zip(amber_band.iterrows(), verdicts):
                d.at[row_idx, "final_present"] = int(bool(v["present"]))
                d.at[row_idx, "final_by"] = "argon"
                d.at[row_idx, "argon_reasoning"] = str(v.get("reasoning", ""))
                conf_v = float(v.get("confidence", 0.0))
                if conf_v >= 0.80:
                    if int(bool(v["present"])) == int(orig_row["present"]):
                        d.at[row_idx, "status"] = "accept"
                        d.at[row_idx, "trust"] = round(0.70 + 0.3 * conf_v * 0.99, 4)
                    else:
                        # Argon confidently overturned the Flash claim -> accept Argon's revised call
                        d.at[row_idx, "status"] = "accept"
                        d.at[row_idx, "trust"] = round(0.70 + 0.25 * conf_v, 4)
                else:
                    # Argon remains uncertain -> keep as AMBER for the Human Gate
                    d.at[row_idx, "status"] = "AMBER"

    elapsed_sec = round(time.perf_counter() - t0, 2)
    meta = {
        "video": video_id,
        "n_frames": int(len(v_df)),
        "n_labels": int(len(d)),
        "elapsed_sec": elapsed_sec,
        "under_3_min": bool(elapsed_sec < 180.0),
        "n_accept": int((d["status"] == "accept").sum()),
        "n_amber": int((d["status"] == "AMBER").sum()),
        "n_reject": int((d["status"] == "reject").sum()),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    O.mkdir(parents=True, exist_ok=True)
    d.to_csv(O / f"live_decisions_{video_id}.csv", index=False)
    d.to_csv(O / "live_decisions.csv", index=False)
    if not (O / "decisions.csv").exists():
        d.to_csv(O / "decisions.csv", index=False)
    (O / f"live_run_{video_id}.json").write_text(json.dumps(meta, indent=2))
    (O / "live_run_latest.json").write_text(json.dumps(meta, indent=2))
    return d, meta


def read_trace_lines(limit: int = 25) -> list[dict]:
    trace_path = resolve_artifact("trace.jsonl")
    if not trace_path.exists():
        return []
    out = []
    try:
        for ln in trace_path.read_text().splitlines()[-limit:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
                if isinstance(rec, dict) and "agent" in rec and "msg" in rec:
                    out.append(rec)
            except Exception:
                continue
    except Exception:
        return []
    return list(reversed(out))


# ---------------- UI lane (harald/demo): real CholecSeg8k frames, GT overlay, tracker loop, narration claims ----------------
CHOLEC_ROOT = C.DATA_DIR / "cholecseg8k" / "CholecSeg8k"
TEST_VIDEOS = ["video09", "video12", "video17", "video28", "video48", "video55"]  # fixed split, see README
GT_IGNORE = 255
GT_COLOURS = {  # one colour per config.LABELS key
    "grasper": (66, 153, 225),
    "l_hook": (159, 122, 234),
    "liver": (160, 82, 45),
    "gallbladder": (72, 187, 120),
    "cystic_duct": (236, 201, 75),
    "blood": (229, 62, 62),
}
CLAIM_COL = {"verified": "#1f9d55", "unsupported": "#d69e2e", "contradicted": "#e53e3e"}


def cholec_frame_path(video_id: str, clip_id: str, frame_idx: int, kind: str = "endo") -> Path:
    """data/cholecseg8k/CholecSeg8k/<video>/<clip>/frame_<N>_<kind>.png (kind: endo | endo_watershed_mask)."""
    return CHOLEC_ROOT / str(video_id) / str(clip_id) / f"frame_{int(frame_idx)}_{kind}.png"


def watershed_to_class_map(mask: np.ndarray) -> np.ndarray:
    """Watershed mask channel 0 -> class-id map via config.WATERSHED_VALUE_TO_CLASS; 255 and unknown -> -1 (ignore)."""
    ch0 = mask[..., 0] if mask.ndim == 3 else mask
    lut = np.full(256, -1, dtype=np.int16)
    for val, cls in C.WATERSHED_VALUE_TO_CLASS.items():
        lut[int(val)] = int(cls)
    lut[GT_IGNORE] = -1
    return lut[ch0]


def gt_overlay(img: Image.Image, mask_path: Path, keys: list[str] | None = None, alpha: float = 0.45) -> Image.Image:
    """Blend ground-truth classes (config.LABELS keys) onto the frame. Evaluation display only (test videos)."""
    if not Path(mask_path).is_file():
        return img
    cls_map = watershed_to_class_map(np.array(Image.open(mask_path)))
    base = np.array(img.convert("RGB")).astype(np.float32)
    if cls_map.shape != base.shape[:2]:
        return img
    for key in keys or list(C.LABELS):
        sel = cls_map == C.LABELS[key][0]
        if sel.any():
            base[sel] = (1 - alpha) * base[sel] + alpha * np.array(GT_COLOURS[key], dtype=np.float32)
    return Image.fromarray(base.clip(0, 255).astype(np.uint8))


def draw_pixel_boxes(img: Image.Image, boxes: pd.DataFrame) -> Image.Image:
    """Draw tracker boxes (pixel x1,y1,x2,y2 on 854x480). Red = flagged for targeted recheck, white = not flagged."""
    im = img.copy()
    d = ImageDraw.Draw(im)
    for r in boxes.itertuples():
        colour = "#e53e3e" if bool(getattr(r, "flagged", False)) else "#f7fafc"
        d.rectangle([r.x1, r.y1, r.x2, r.y2], outline=colour, width=3)
        d.text((r.x1 + 4, r.y1 + 4), str(r.label), fill=colour)
    return im


@st.cache_data(show_spinner=False)
def load_parquet(path_str: str, mtime: float) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path_str)
    except Exception:
        return None


def load_parquet_fresh(p: Path) -> pd.DataFrame | None:
    return load_parquet(str(p), p.stat().st_mtime) if p.exists() else None


def tracker_source_note(source: str) -> str:
    if source == "gemini":
        return "Tracker flags computed on **Gemini** boxes (real labeller output)."
    return (
        f"Tracker flags computed on `{source}` boxes: ground-truth masks **with synthetic injected errors**, "
        "the tracker's development setting. Not a result on Gemini labels yet."
    )


def narration_claims(entry: dict) -> list[dict]:
    """Normalise one narration entry to [{text, status, evidence_frames}].

    Uses `claims` when narrate.py emits it. The legacy {narration, unsure_about} shape is
    mapped to one claim per line with status `verified` only when nothing was unsure;
    unsure items become `unsupported` claims (never shown as fact)."""
    out = []
    for c in entry.get("claims", []) or []:
        if isinstance(c, dict) and c.get("text"):
            status = str(c.get("status", "unsupported")).lower()
            out.append({
                "text": str(c["text"]),
                "status": status if status in CLAIM_COL else "unsupported",
                "evidence_frames": [str(x) for x in c.get("evidence_frames", []) or []],
            })
    if out:
        return out
    if entry.get("narration"):
        out.append({
            "text": str(entry["narration"]),
            "status": "verified" if not entry.get("unsure_about") else "unsupported",
            "evidence_frames": [str(entry.get("frame_id", ""))] if entry.get("frame_id") else [],
        })
    for u in entry.get("unsure_about", []) or []:
        out.append({"text": f"{u} (withheld: not verified)", "status": "unsupported", "evidence_frames": []})
    return out


def frame_path_from_id(frame_id: str) -> Path | None:
    """frame_id `<video>_<clipstart>_<N>` -> CholecSeg8k frame path, if it exists."""
    parts = str(frame_id).split("_")
    if len(parts) == 3 and parts[2].isdigit():
        p = cholec_frame_path(parts[0], f"{parts[0]}_{parts[1]}", int(parts[2]))
        return p if p.is_file() else None
    return None


def render_med_school():
    st.header("Med School: what the agent taught itself")
    man_path = resolve_artifact("manual.json")
    man = load_json(man_path)
    if not man:
        st.info("**[PLACEHOLDER — Awaiting `out/manual.json` from `med_school.py`]**")
        return
    st.caption(f"Source: `{man_path.relative_to(C.ROOT) if man_path.is_relative_to(C.ROOT) else man_path}` (written by `med_school.py`, no human edits)")
    st.markdown(f"**Procedure:** {man.get('procedure', 'n/a')}")
    if man.get("sources"):
        st.markdown("**Teaching sources (narrated video provenance):**")
        for src in man["sources"]:
            if isinstance(src, dict):
                real = " · real intra-op footage" if src.get("is_real_intraop_footage") else ""
                st.markdown(f"- [{src.get('title', src.get('url', 'source'))}]({src.get('url', '')}){real}")
                for ts in src.get("timestamps", []) or []:
                    st.markdown(f"    - {ts}")
            else:
                st.markdown(f"- {src}")
    else:
        st.caption("`sources` (video URL + timestamps) not present in manual.json yet.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Phases learned", len(man.get("phases", [])))
    m2.metric("Structures in visual vocabulary", len(man.get("structures", [])))
    m3.metric("Critical events / cues", len(man.get("critical_events", [])))
    pc, ec = st.columns(2)
    with pc:
        st.subheader("Phases")
        for ph in man.get("phases", []):
            st.markdown(f"- {ph}")
    with ec:
        st.subheader("Critical events (cues)")
        for ev in man.get("critical_events", []):
            st.markdown(f"- {ev}")
        if man.get("general_rules"):
            st.subheader("Labelling rules it set itself")
            for g in man["general_rules"]:
                st.markdown(f"- {g}")
    st.subheader("Visual vocabulary")
    for s in man.get("structures", []):
        with st.expander(f"**{s.get('name', s.get('key', ''))}** (`{s.get('key', '')}`)"):
            st.markdown(f"**Looks like:** {s.get('appearance', '')}")
            st.markdown(f"**Where:** {s.get('where_found', '')}")
            st.markdown(f"**When:** {s.get('when_visible', '')}")
            confused = ", ".join(s.get("confused_with", [])) or "none listed"
            st.markdown(f"**Confused with:** {confused}")
            st.markdown(f"**How to tell apart:** {s.get('how_to_tell_apart', '')}")


def _pct(v):
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "PENDING"


def render_loop():
    st.header("The loop: raw labels → agent self-check → tracker-targeted recheck")
    st.caption(
        "Each round keeps fewer labels but more of them are right. Accuracy is scored on held-out test videos only "
        f"({', '.join(TEST_VIDEOS)}); ground truth never feeds the labeller, the verifier or the tracker's input boxes for Gemini."
    )
    res = load_json(resolve_artifact("results.json")) or {}
    acc = res.get("accepted", {}) if isinstance(res.get("accepted"), dict) else {}
    trk = res.get("tracker_recheck", {}) if isinstance(res.get("tracker_recheck"), dict) else {}
    rounds = [
        {"round": "0 · raw VLM labels (Flash + manual)", "coverage": 1.0 if "manual_raw_acc" in res else None,
         "accuracy_of_kept": res.get("manual_raw_acc"), "source": "results.json: manual_raw_acc"},
        {"round": "1 · agent self-check (accept only)", "coverage": acc.get("coverage"),
         "accuracy_of_kept": acc.get("accuracy"), "source": "results.json: accepted"},
        {"round": "2 · tracker-flagged recheck", "coverage": trk.get("coverage"),
         "accuracy_of_kept": trk.get("accuracy"), "source": "results.json: tracker_recheck"},
    ]
    cols = st.columns(3)
    for c, r in zip(cols, rounds):
        c.metric(r["round"], _pct(r["accuracy_of_kept"]), f"coverage {_pct(r['coverage'])}" if r["coverage"] is not None else None)
    if not res:
        st.warning("**[PLACEHOLDER — Awaiting `out/results.json`]** Round metrics show `PENDING`; no numbers are invented.")
    pts = pd.DataFrame([r for r in rounds if r["coverage"] is not None and r["accuracy_of_kept"] is not None])
    if len(pts) >= 2:
        st.subheader("Accuracy of kept labels vs coverage, per round")
        st.scatter_chart(pts, x="coverage", y="accuracy_of_kept", color="round")
    st.subheader("Baselines (test split)")
    base_rows = [
        {"baseline": "blank (no manual)", "raw_accuracy": res.get("blank_raw_acc"), "coverage_at_95": (res.get("blank_single_conf") or {}).get("coverage_at_95"), "human_labels": "none"},
        {"baseline": "manual (self-written)", "raw_accuracy": res.get("manual_raw_acc"), "coverage_at_95": (res.get("single_conf") or {}).get("coverage_at_95"), "human_labels": "none"},
        {"baseline": "majority vote", "raw_accuracy": None, "coverage_at_95": (res.get("majority_vote") or {}).get("coverage_at_95"), "human_labels": "none"},
        {"baseline": "upper_bound_calibrated [UPPER BOUND]", "raw_accuracy": None, "coverage_at_95": (res.get("upper_bound_calibrated") or {}).get("coverage_at_95"), "human_labels": "calib split GT (upper bound only)"},
    ]
    st.dataframe(pd.DataFrame(base_rows).fillna("PENDING"), width="stretch")

    st.divider()
    st.subheader("Round 2 · Tracking verifier: where to recheck (grasper, L-hook)")
    flags = load_parquet_fresh(O / "flags.parquet")
    summ = load_json(O / "eval" / "summary.json")
    if flags is None or flags.empty:
        st.info("**[PLACEHOLDER — Awaiting `out/flags.parquet` from the tracker lane]**")
        return
    source = str(flags["source"].iloc[0])
    if source == "gemini":
        st.success(tracker_source_note(source))
    else:
        st.warning(tracker_source_note(source))
    if summ:
        st.markdown("**Share of injected errors found within a recheck budget** (test videos, synthetic errors; `out/eval/summary.json`)")
        t1, t2, t3 = st.columns(3)
        for col, b in zip((t1, t2, t3), (5, 10, 20)):
            col.metric(
                f"{b}% budget: tracker-targeted",
                _pct(summ.get(f"found_at_{b}pct_targeted")),
                f"random {_pct(summ.get(f'found_at_{b}pct_random'))} · best single {_pct(summ.get(f'found_at_{b}pct_best'))}",
                delta_color="off",
            )
        st.caption(
            "Suspicion model is fit on tuning videos (ground-truth boxes with synthetic errors) and scored on test videos. "
            "Real Gemini errors (whole missed tracks, consistent mislabelling) will be harder."
        )
    curve_p = O / "eval" / "budget_curve.csv"
    if curve_p.exists():
        bc = pd.read_csv(curve_p).set_index("budget")
        cc, rc = st.columns([2, 1])
        cc.line_chart(bc)
        cc.caption("x = share of frame-labels rechecked, y = share of injected errors found")
        if summ and summ.get("recall_by_type_at_10pct"):
            rc.markdown("**Found at 10% budget, by error type**")
            rc.dataframe(pd.Series(summ["recall_by_type_at_10pct"], name="found").to_frame(), width="stretch")

    st.markdown("**Frame explorer** (real CholecSeg8k frames, held-out test videos)")
    test_flags = flags[flags["video_id"].isin(TEST_VIDEOS)]
    if test_flags.empty:
        st.caption("No test-video rows in flags.parquet.")
        return
    e1, e2, e3 = st.columns([1, 1, 1])
    vid = e1.selectbox("Test video", sorted(test_flags["video_id"].unique()), key="loop_vid")
    vf = test_flags[test_flags["video_id"] == vid]
    clip_rank = vf.groupby("clip_id")["flagged"].sum().sort_values(ascending=False)
    clip = e2.selectbox("Clip (most flags first)", clip_rank.index.tolist(),
                        format_func=lambda c: f"{c} ({int(clip_rank[c])} flagged)", key="loop_clip")
    show_gt = e3.toggle("Overlay ground truth (evaluation only)", key="loop_gt")
    cf = vf[vf["clip_id"] == clip]
    frames_c = sorted(cf["frame_idx"].unique())
    flagged_frames = sorted(cf[cf["flagged"]]["frame_idx"].unique())
    jump = st.checkbox("Step through flagged frames only", value=bool(flagged_frames), key="loop_jump")
    seq = flagged_frames if (jump and flagged_frames) else frames_c
    n = st.select_slider("Frame", options=seq, key=f"loop_f_{clip}_{jump}") if len(seq) > 1 else seq[0]
    img = resolve_frame_image(cholec_frame_path(vid, clip, n), frame_id=f"{clip}_{n}")
    if show_gt:
        img = gt_overlay(img, cholec_frame_path(vid, clip, n, "endo_watershed_mask"))
    det = load_parquet_fresh(O / "tracker" / f"detections_{source}.parquet")
    fr_flags = cf[cf["frame_idx"] == n]
    if det is not None:
        boxes = det[(det["clip_id"] == clip) & (det["frame_idx"] == n)]
        boxes = boxes.merge(fr_flags[["label", "flagged"]], on="label", how="left")
        img = draw_pixel_boxes(img, boxes)
    ic, tc = st.columns([3, 2])
    ic.image(img, caption=f"{clip} frame {n} · red box = flagged for recheck" + (" · GT overlay on" if show_gt else ""), width="stretch")
    if show_gt:
        ic.markdown(" ".join(
            f"<span style='background:rgb{GT_COLOURS[k]};color:white;padding:2px 8px;border-radius:10px;margin:2px'>{k}</span>"
            for k in C.LABELS), unsafe_allow_html=True)
    tc.dataframe(
        fr_flags[["label", "n_det", "suspicion", "flagged", "top_signal"]].set_index("label"),
        width="stretch",
    )
    queue = load_parquet_fresh(O / "tracker" / f"recheck_queue_{source}.parquet")
    if queue is not None and not queue.empty:
        q = queue[(queue["clip_id"] == clip) & (queue["frame_idx"] == n)]
        if not q.empty:
            tc.markdown("**Recheck requested:** " + ", ".join(f"`{r.label}` → `{r.recheck_type}`" for r in q.itertuples()))
        tc.caption(f"Recheck queue: {len(queue)} items total ({int(queue['is_control'].sum())} random controls).")


def claims_html(claims: list[dict]) -> str:
    return "<br>".join(
        f"<span style='background:{CLAIM_COL[c['status']]};color:white;padding:2px 8px;border-radius:10px;"
        f"margin-right:6px;font-size:0.8em'>{c['status']}</span>{c['text']}"
        for c in claims
    )


def render_narration():
    st.subheader("Grounded narration: every claim tagged")
    st.markdown(
        " ".join(
            f"<span style='background:{col};color:white;padding:3px 9px;border-radius:10px;margin-right:6px'>{name}</span>"
            for name, col in CLAIM_COL.items()
        ),
        unsafe_allow_html=True,
    )
    files = sorted(O.glob("narration_*.json")) + sorted((COORD_ARTIFACTS).glob("narration_*.json"))
    if not files:
        st.info("**[PLACEHOLDER — Awaiting `out/narration_<video>.json` from `narrate.py`]**")
        return
    pick = st.selectbox("Narration file", files, format_func=lambda p: p.name, key="narr_file")
    entries = [e for e in (load_json(pick, []) or []) if isinstance(e, dict)]
    rows = []
    for e in entries:
        for c in narration_claims(e):
            rows.append({"frame_id": e.get("frame_id", ""), **c})
    if not rows:
        st.caption("File has no narration entries.")
        return
    cdf = pd.DataFrame(rows)
    k1, k2, k3 = st.columns(3)
    for col, status in zip((k1, k2, k3), CLAIM_COL):
        col.metric(f"{status} claims", int((cdf["status"] == status).sum()))
    if not any("claims" in e for e in entries):
        st.caption("This file uses the legacy `{narration, unsure_about}` shape: a line counts as `verified` only "
                   "if nothing was unsure; `contradicted` needs per-claim tags from `narrate.py`.")
    st.markdown("**Final narration (verified claims only)**")
    final = [r["text"] for r in rows if r["status"] == "verified"]
    st.success(" ".join(final) if final else "No verified claims yet.")
    with st.expander("All claims with evidence frames"):
        for i, r in enumerate(rows[:60]):
            cc1, cc2 = st.columns([3, 2])
            cc1.markdown(claims_html([r]) + f"<br><small>`{r['frame_id']}`</small>", unsafe_allow_html=True)
            ev = [fp for fp in (frame_path_from_id(x) for x in r["evidence_frames"][:3]) if fp]
            if ev:
                cc2.image([str(x) for x in ev], width=160)


# ---------------- Top Landing Header: SurgProof, Honest Claims & Prior Work ----------------
st.title("SurgProof — No human labels. Only proof.")
st.caption(
    "Self-taught surgical video labelling with zero-label verification, an AMBER human gate, "
    "and specialist distillation."
)

hc_col, pw_col = st.columns(2)
with hc_col:
    st.markdown(
        """
**Honest Claims**
- **"No human labels used in training."** Ground truth (`gt_*`) is excluded from labeller and verifier inputs and joined **only** to score held-out test videos (`split == "test"`).
- **Upper bounds explicitly marked:** The `calib` split (`upper_bound_calibrated`) and `oracle_upper_bound` student use human labels and are reported strictly as **upper bounds** for reference only.
- **Act 1 (any surgery) has no ground truth:** Showcase of zero-setup generalization, not accuracy proof. CholecSeg8k is the blind proof.
"""
    )
with pw_col:
    st.markdown(
        """
**Prior Work & Our Step**
- **SurgAtlas**: Large-scale dataset of public surgical videos.
- **SurgVLP** *(CAMMA Strasbourg)*: Learning representations from narrated surgical lectures.
- **SurgSLOT**: Self-evolving surgical data engine relying on **human verification**.
- **Med-Gemini**: Zero-shot Critical View of Safety (CVS) assessment on surgical video.
- **Our Step (`SurgProof`)**: An autonomous Gemini agent **replaces the human verifier**, locks dataset export behind a human AMBER gate, and distills a stronger specialist student.
"""
    )

st.divider()

# ---------------- Sidebar: Pipeline Status, Promo Video & Live Agent Reasoning ----------------
with st.sidebar:
    st.subheader("Pipeline Artifacts")
    artifact_checks = {
        "data/index.csv": (C.DATA_DIR / "index.csv").exists(),
        "data/features.npy": (C.DATA_DIR / "features.npy").exists(),
        "out/manual.json": resolve_artifact("manual.json").exists(),
        "out/labels_manual.jsonl": resolve_artifact("labels_manual.jsonl").exists() or resolve_artifact("labels_smoke.jsonl").exists(),
        "out/decisions.csv": resolve_artifact("decisions.csv").exists(),
        "out/results.json": resolve_artifact("results.json").exists(),
        "out/student.json": resolve_artifact("student.json").exists(),
        "out/live_decisions.csv": resolve_artifact("live_decisions.csv").exists(),
        "out/gate_log.jsonl": resolve_artifact("gate_log.jsonl").exists() or resolve_artifact("human_gate_log.jsonl").exists(),
        "out/flags.parquet (tracker)": (O / "flags.parquet").exists(),
        "out/narration_*.json": any(O.glob("narration_*.json")),
    }
    status_lines = [
        f"- {'🟢 **LIVE**' if ok else '🟡 **PLACEHOLDER**'} `{name}`"
        for name, ok in artifact_checks.items()
    ]
    st.markdown("\n".join(status_lines))
    if st.button("🔄 Refresh outputs"):
        st.rerun()

    st.divider()
    st.subheader("🛡️ Human Gate (#14)")
    gate_page_path = C.ROOT / "pages" / "2_Human_gate.py"
    if gate_page_path.is_file():
        st.page_link("pages/2_Human_gate.py", label="Open Human Gate (AMBER Queue)", icon="🛡️")
    else:
        st.markdown(
            "👉 **[Human Gate (`pages/2_Human_gate.py`)](/Human_gate)** — "
            "Clears `AMBER` queue & unlocks verified dataset export (`harald/gate`)."
        )

    promo_candidates = [O / "promo" / "surgproof_promo.mp4", O / "promo" / "surgatlas_promo.mp4"]
    for promo_path in promo_candidates:
        if promo_path.is_file():
            st.divider()
            st.subheader("Pitch Video (Illustrative + Live Capture)")
            st.video(str(promo_path))
            st.caption("Generated shots are ILLUSTRATIVE; metrics come strictly from real pipeline outputs.")
            break

    st.divider()
    st.subheader("Agent reasoning")
    trace_items = read_trace_lines(limit=25)
    if trace_items:
        for t in trace_items:
            st.markdown(f"**{t['agent']}** {t['msg']}")
    else:
        st.caption("[PLACEHOLDER — No trace yet in out/trace.jsonl.]")

tab_med, tab_loop, tab_live, tab_proof, tab_watch = st.tabs([
    "🎓 1 · Med School",
    "🔁 2 · Label → Verify Loop",
    "⚡ 3 · Live Judge Test (<3 min)",
    "📊 4 · Proof & Student (CholecSeg8k & External Sets)",
    "🎬 5 · Watch, Narration & Any Surgery",
])

with tab_med:
    render_med_school()

with tab_loop:
    render_loop()

# ==============================================================================
# TAB 1: LIVE JUDGE TEST (<3 MIN) — Issue #15
# ==============================================================================
with tab_live:
    st.header("Live Judge Test: Pick a Held-Out Test Video (<3 Minutes)")
    st.info(
        "**Judging test ('live, real input, under 3 minutes'):** `out/manual.json` is cached; a judge picks any held-out test video (1–6); "
        "SurgProof labels ~20 frames with Gemini 3.8 Flash, runs blind crop verification + sceptical critic, escalates the uncertain band to "
        "**Gemini 4 Argon**, routes remaining `AMBER` claims to the **Human Gate (`pages/2_Human_gate.py`)**, and reveals accuracy against hidden ground truth."
    )
    gate_page_live = C.ROOT / "pages" / "2_Human_gate.py"
    if gate_page_live.is_file():
        st.page_link("pages/2_Human_gate.py", label="🛡️ Go to Human Gate (AMBER Approval Queue & Locked Export)", icon="🛡️")
    else:
        st.markdown(
            "🛡️ **Human Gate (`pages/2_Human_gate.py`):** Uncertain (`AMBER`) labels from this live run are routed to the "
            "[**Human Gate page**](/Human_gate) (`pages/2_Human_gate.py`, owned by `harald/gate`), where dataset export stays locked until every `AMBER` item is cleared."
        )

    idx_live_path = C.DATA_DIR / "index.csv"
    if not idx_live_path.exists():
        st.warning(
            "**[PLACEHOLDER — Awaiting `data/index.csv` from `prepare_data.py`]**\n\n"
            "Once `data/index.csv` is available, the 6 held-out test videos appear here for live execution."
        )
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Wall-clock time [PLACEHOLDER]", "PENDING")
        lc2.metric("Auto-accepted (`accept`) [PLACEHOLDER]", "PENDING")
        lc3.metric("Sent to Human Gate (`AMBER`) [PLACEHOLDER]", "PENDING")
        lc4.metric("Auto-rejected (`reject`) [PLACEHOLDER]", "PENDING")
    else:
        idx_all = pd.read_csv(idx_live_path)
        test_vids_live = sorted(idx_all[idx_all["split"] == "test"]["video"].unique()) if "split" in idx_all.columns else sorted(idx_all["video"].unique())
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 2])
        with ctrl1:
            picked_live_vid = st.selectbox(
                "Judge picks held-out test video (1–6)",
                test_vids_live,
                format_func=lambda v: f"{v} (held-out test video #{test_vids_live.index(v) + 1} of {len(test_vids_live)})",
            )
        with ctrl2:
            n_frames_live = st.number_input("Frames to sample", min_value=5, max_value=40, value=20, step=5)
        with ctrl3:
            argon_budget_live = st.number_input("Argon review budget", min_value=0, max_value=40, value=12, step=4)
        with ctrl4:
            st.write("")
            run_live_btn = st.button("▶ Run Live Label + Verify Now", type="primary")

        if run_live_btn:
            prog_bar = st.progress(0.0, text=f"Running live Flash labelling + decorrelated checks on {picked_live_vid}...")
            status_box = st.empty()

            def _update_prog(done, total, stage):
                frac = min(0.95, done / max(1, total))
                prog_bar.progress(frac, text=f"[{stage.upper()}] {done}/{total} frames labelled & checked...")

            live_df, live_meta = asyncio.run(
                run_live_video_async(
                    picked_live_vid,
                    n_frames=int(n_frames_live),
                    argon_budget=int(argon_budget_live),
                    sample_seed=0,
                    progress_cb=_update_prog,
                )
            )
            prog_bar.progress(1.0, text=f"Completed {picked_live_vid} in {live_meta['elapsed_sec']:.2f}s!")
            status_box.success(
                f"Live run finished in **{live_meta['elapsed_sec']:.2f}s** (<180s limit): "
                f"**{live_meta['n_accept']}** accepted, **{live_meta['n_amber']}** AMBER (sent to Human Gate), **{live_meta['n_reject']}** rejected."
            )

        # Display latest or video-specific live run if present
        live_dec_file = O / f"live_decisions_{picked_live_vid}.csv"
        live_meta_file = O / f"live_run_{picked_live_vid}.json"
        if not live_dec_file.exists():
            live_dec_file = resolve_artifact("live_decisions.csv")
            live_meta_file = resolve_artifact("live_run_latest.json")

        if live_dec_file.exists() and live_meta_file.exists():
            l_df = pd.read_csv(live_dec_file)
            l_meta = load_json(live_meta_file, {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Wall-clock time", f"{l_meta.get('elapsed_sec', 0.0):.2f}s", "< 180s target ✓" if l_meta.get("under_3_min", True) else "> 180s")
            m2.metric("Auto-accepted (`accept`)", int((l_df["status"].map(normalize_status) == "accept").sum()))
            m3.metric("AMBER (sent to Human Gate)", int((l_df["status"].map(normalize_status) == "AMBER").sum()))
            m4.metric("Auto-rejected (`reject`)", int((l_df["status"].map(normalize_status) == "reject").sum()))

            reveal_live_gt = st.toggle("🔓 Reveal Hidden Ground Truth on Live Run (`data/index.csv` joined only now)", key="reveal_live_gt")
            if reveal_live_gt:
                eval_df = join_ground_truth_for_reveal(l_df, idx_all)
                eval_nt = eval_df[eval_df["key"].isin(NON_TRIVIAL_KEYS)]
                acc_nt = eval_nt[eval_nt["status"].map(normalize_status) == "accept"]
                acc_all = eval_df[eval_df["status"].map(normalize_status) == "accept"]

                g1, g2, g3, g4 = st.columns(4)
                raw_nt_acc = eval_nt["correct"].mean() if not eval_nt.empty and "correct" in eval_nt else None
                ver_nt_acc = acc_nt["final_correct"].mean() if not acc_nt.empty and "final_correct" in acc_nt else None
                ver_all_acc = acc_all["final_correct"].mean() if not acc_all.empty and "final_correct" in acc_all else None
                cov_nt = len(acc_nt) / len(eval_nt) if len(eval_nt) else 0.0

                g1.metric("Raw Flash Accuracy (5 non-trivial)", f"{raw_nt_acc * 100:.1f}%" if pd.notna(raw_nt_acc) else "n/a")
                g2.metric(
                    "Verified `accept` Accuracy (5 non-trivial)",
                    f"{ver_nt_acc * 100:.1f}%" if pd.notna(ver_nt_acc) else "n/a",
                    f"{(ver_nt_acc - raw_nt_acc) * 100:+.1f} pts vs raw" if (pd.notna(ver_nt_acc) and pd.notna(raw_nt_acc)) else None,
                )
                g3.metric("Verified `accept` Accuracy (all 6 incl. liver)", f"{ver_all_acc * 100:.1f}%" if pd.notna(ver_all_acc) else "n/a")
                g4.metric("Auto-accepted coverage (5 non-trivial)", f"{cov_nt * 100:.0f}%")
                st.dataframe(
                    eval_df[["frame_id", "key", "present", "conf", "pv_match", "p_wrong", "trust", "status", "final_present", "final_by", "truth", "final_correct"]],
                    width="stretch",
                )
            else:
                st.dataframe(
                    l_df[[c for c in ["frame_id", "key", "present", "conf", "pv_match", "p_wrong", "trust", "status", "final_present", "final_by", "evidence", "critic"] if c in l_df.columns]],
                    width="stretch",
                )
        else:
            st.info(
                "**[PLACEHOLDER — Click `▶ Run Live Label + Verify Now` above to execute a live 20-frame run]**\n\n"
                "Wall-clock timing, Argon revisions, AMBER routing, and ground-truth reveal populate immediately after the run."
            )

# ==============================================================================
# TAB 2: PROOF (CHOLECSEG8K & SPECIALIST STUDENT) — Act 2
# ==============================================================================
with tab_proof:
    st.header("Proof: CholecSeg8k, labels hidden from the agent")
    st.info(
        "**Honest claims:**\n"
        "- **No human labels used in training.** Ground truth is used **only** to score held-out test videos (`split == 'test'`).\n"
        "- **Headline metrics report the 5 non-trivial structures** (`grasper`, `l_hook`, `gallbladder`, `cystic_duct`, `blood`), since `liver` is present in 100% of frames.\n"
        "- **`calib` split (`upper_bound_calibrated`)** and **`oracle_upper_bound` student** use human labels and are reported strictly as **upper bounds**."
    )

    idx_file = C.DATA_DIR / "index.csv"
    lbl_file = resolve_artifact("labels_manual.jsonl")
    blank_file = resolve_artifact("labels_blank.jsonl")
    smoke_file = resolve_artifact("labels_smoke.jsonl")
    if idx_file.exists() or lbl_file.exists() or blank_file.exists() or smoke_file.exists():
        prog_cols = st.columns(4)
        if idx_file.exists():
            try:
                idx_df = pd.read_csv(idx_file)
                prog_cols[0].metric("Indexed frames (data/index.csv)", len(idx_df))
                prog_cols[1].metric("Held-out test videos", int(idx_df[idx_df["split"] == "test"]["video"].nunique()))
            except Exception:
                pass
        active_lbl = lbl_file if lbl_file.exists() else smoke_file
        if active_lbl.exists():
            try:
                n_lines = sum(1 for ln in active_lbl.read_text().splitlines() if ln.strip())
                prog_cols[2].metric(f"Manual label rows ({active_lbl.name})", n_lines)
            except Exception:
                pass
        if blank_file.exists():
            try:
                n_blank = sum(1 for ln in blank_file.read_text().splitlines() if ln.strip())
                prog_cols[3].metric("Blank baseline rows (labels_blank.jsonl)", n_blank)
            except Exception:
                pass

    res = load_json(resolve_artifact("results.json"))
    stu = load_json(resolve_artifact("student.json"))
    eval_audit = load_json(resolve_artifact("eval_audit.json"))

    # Show live measured baseline numbers from labels_blank.jsonl / labels_manual.jsonl if already on disk
    if idx_file.exists() and (blank_file.exists() or lbl_file.exists() or smoke_file.exists()):
        idx_df_eval = pd.read_csv(idx_file)
        blank_stats = score_jsonl_against_index(blank_file, idx_df_eval, split_filter="test") if blank_file.exists() else None
        manual_cand = lbl_file if lbl_file.exists() else smoke_file
        manual_stats = score_jsonl_against_index(manual_cand, idx_df_eval, split_filter="test") if manual_cand.exists() else None
        if blank_stats or manual_stats:
            st.subheader("Live Measured Labelling Accuracy on Held-Out Test Split (`data/index.csv`)")
            b_cols = st.columns(4)
            if blank_stats:
                b_cols[0].metric(
                    "Blank Baseline (5 non-trivial labels)",
                    f"{blank_stats['acc_5_nontrivial'] * 100:.1f}%" if blank_stats.get("acc_5_nontrivial") is not None else "n/a",
                    f"{blank_stats['n_frames']} test frames ({blank_stats['n_rows']} labels)",
                )
                b_cols[1].metric("Blank Baseline (all 6 incl. liver)", f"{blank_stats['acc_all_6'] * 100:.1f}%")
            if manual_stats:
                delta_5 = (
                    f"{(manual_stats['acc_5_nontrivial'] - blank_stats['acc_5_nontrivial']) * 100:+.1f} pts vs blank"
                    if (blank_stats and manual_stats.get("acc_5_nontrivial") is not None and blank_stats.get("acc_5_nontrivial") is not None)
                    else f"{manual_stats['n_frames']} test frames"
                )
                b_cols[2].metric(
                    f"Self-Written Manual ({manual_cand.name}, 5 non-trivial)",
                    f"{manual_stats['acc_5_nontrivial'] * 100:.1f}%" if manual_stats.get("acc_5_nontrivial") is not None else "n/a",
                    delta_5,
                )
                b_cols[3].metric(f"Self-Written Manual ({manual_cand.name}, all 6)", f"{manual_stats['acc_all_6'] * 100:.1f}%")
            per_key_rows = []
            for k in C.LABELS:
                row_k = {"label": k, "is_headline_nontrivial": k != "liver"}
                if blank_stats and k in blank_stats["per_key"]:
                    row_k["blank_test_acc"] = blank_stats["per_key"][k]
                if manual_stats and k in manual_stats["per_key"]:
                    row_k["manual_test_acc"] = manual_stats["per_key"][k]
                per_key_rows.append(row_k)
            with st.expander("Per-structure raw test accuracy (`labels_blank.jsonl` / `labels_manual.jsonl`)"):
                st.dataframe(pd.DataFrame(per_key_rows), width="stretch")

    if not res:
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/results.json` from `accept.py`]**\n\n"
            "Metrics below are placeholders (`PENDING`) until `accept.py` finishes scoring on held-out test videos. "
            "No numbers are fabricated."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy on accepted labels [PLACEHOLDER]", "PENDING")
        c2.metric("Labels accepted (coverage) [PLACEHOLDER]", "PENDING")
        c3.metric("Sent to surgeon (AMBER/escalated) [PLACEHOLDER]", "PENDING")
        c4.metric("Self-written manual vs blank [PLACEHOLDER]", "PENDING")

        st.subheader("Selective Prediction Comparison [PLACEHOLDER SCHEMA]")
        placeholder_methods = pd.DataFrame([
            {"method": "blank_single_conf (no manual)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "single_conf (Flash + manual)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "majority_vote (multi-sample)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "system_zero_label (Ours: checks + Argon)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "upper_bound_calibrated [UPPER BOUND]", "uses_human_labels": "Yes (calib split GT)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
        ])
        st.dataframe(placeholder_methods, width="stretch")
    else:
        a = res.get("accepted", {})
        c1, c2, c3, c4 = st.columns(4)
        acc_val = a.get("accuracy")
        cov_val = a.get("coverage")
        esc_val = a.get("escalated_share", a.get("amber_share"))
        c1.metric("Accuracy on accepted labels", f"{acc_val * 100:.1f}%" if isinstance(acc_val, (int, float)) else "n/a")
        c2.metric("Labels accepted (coverage)", f"{cov_val * 100:.0f}%" if isinstance(cov_val, (int, float)) else "n/a")
        c3.metric("Sent to surgeon", f"{esc_val * 100:.0f}%" if isinstance(esc_val, (int, float)) else "n/a")
        if "blank_raw_acc" in res and "manual_raw_acc" in res:
            c4.metric(
                "Self-written manual vs blank",
                f"{res['manual_raw_acc'] * 100:.1f}%",
                f"{(res['manual_raw_acc'] - res['blank_raw_acc']) * 100:+.1f} pts",
            )
        elif "manual_raw_acc" in res:
            c4.metric("Raw accuracy (with self-written manual)", f"{res['manual_raw_acc'] * 100:.1f}%")
        else:
            c4.metric("Held-out test label evaluations", str(res.get("n_test_labels", "n/a")))

        curves = {}
        summary_rows = []
        label_map = {
            "blank_single_conf": "blank_single_conf (no manual)",
            "single_conf": "single_conf (Flash + manual)",
            "majority_vote": "majority_vote",
            "system_zero_label": "system_zero_label (Ours: zero human labels)",
            "upper_bound_calibrated": "upper_bound_calibrated (UPPER BOUND - calib GT)",
        }
        for k, v in res.items():
            if isinstance(v, dict) and "curve" in v:
                display_name = label_map.get(k, k)
                for pt in v["curve"]:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        cov, acc = pt
                        curves.setdefault(cov, {})[display_name] = acc
                summary_rows.append({
                    "method": display_name,
                    "human_labels_used": "Yes (calib split upper bound)" if "upper_bound" in k else "No (0 human labels)",
                    "coverage_at_95": v.get("coverage_at_95"),
                    "acc_at_60": v.get("acc_at_60"),
                    "aurc (lower=better)": v.get("aurc"),
                })
        if curves:
            st.subheader("Accuracy vs share of labels accepted (higher line = better)")
            st.line_chart(pd.DataFrame(curves).T.sort_index().interpolate(method="index"))
            st.caption("Note: `upper_bound_calibrated` fits a logistic regression on the small `calib` split using human labels and is shown strictly as an upper bound.")
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        if res.get("per_label_accepted_acc"):
            st.subheader("Accepted label accuracy by anatomical structure / instrument")
            per_lbl_s = pd.Series(res["per_label_accepted_acc"], name="Accepted Accuracy")
            st.bar_chart(per_lbl_s)

    st.divider()
    st.subheader("Student trained on the system's own labels (test videos, real GT)")
    if not stu:
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/student.json` from `student.py`]**\n\n"
            "DINOv2 linear probe comparison (`single` vs `vote` vs `verified` vs `oracle_upper_bound`) populates here once `student.py` completes."
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Single pseudo-labels [PLACEHOLDER]", "PENDING")
        sc2.metric("Majority vote [PLACEHOLDER]", "PENDING")
        sc3.metric("Verified only (Ours) [PLACEHOLDER]", "PENDING")
        sc4.metric("Oracle GT (UPPER BOUND) [PLACEHOLDER]", "PENDING")
    else:
        mean_acc = stu.get("mean_acc_by_source", {})
        if mean_acc:
            display_series = {
                ("oracle_upper_bound (UPPER BOUND - human GT)" if k == "oracle_upper_bound" else k): v
                for k, v in mean_acc.items()
            }
            sc_cols = st.columns(len(display_series))
            single_base = mean_acc.get("single")
            for idx_c, (k, v) in enumerate(mean_acc.items()):
                lbl = "oracle_upper_bound (UPPER BOUND)" if k == "oracle_upper_bound" else k
                delta = None
                if k == "verified" and isinstance(single_base, (int, float)) and isinstance(v, (int, float)):
                    delta = f"{(v - single_base) * 100:+.1f} pts vs single"
                sc_cols[idx_c].metric(lbl, f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a", delta=delta)
            st.bar_chart(pd.Series(display_series))
        st.caption("oracle_upper_bound uses human labels on the train split and is shown as an upper bound for reference only.")
        zs = (res or {}).get("manual_raw_acc")
        cmp_rows = [{"model": "zero-shot Gemini (raw labels, no student)", "test_acc": zs, "human_labels": "none",
                     "from": "results.json: manual_raw_acc"}]
        names = {"single": "student on raw labels", "vote": "student on vote labels",
                 "verified": "student on verified labels (Ours)", "oracle_upper_bound": "oracle_upper_bound [UPPER BOUND]"}
        for k_src, v_src in mean_acc.items():
            cmp_rows.append({"model": names.get(k_src, k_src), "test_acc": v_src,
                             "human_labels": "train GT (upper bound)" if k_src == "oracle_upper_bound" else "none",
                             "from": f"student.json: mean_acc_by_source.{k_src}"})
        st.markdown("**Specialist vs zero-shot (held-out test videos)**")
        st.dataframe(pd.DataFrame(cmp_rows).fillna("PENDING"), width="stretch")
        st.caption("Numbers are read as written by each lane; check `headline_labels` in student.json, since the pooled "
                   "label set behind `manual_raw_acc` can differ.")
        if stu.get("per_label"):
            rows_stu = []
            for key, by_src in stu["per_label"].items():
                if not isinstance(by_src, dict):
                    continue
                for src_name, metrics in by_src.items():
                    if isinstance(metrics, dict):
                        rows_stu.append({
                            "label": key,
                            "source": "oracle_upper_bound (UPPER BOUND)" if src_name == "oracle_upper_bound" else src_name,
                            "n_train": metrics.get("n_train"),
                            "test_acc": metrics.get("acc"),
                            "test_f1": metrics.get("f1"),
                            "test_auroc": metrics.get("auroc"),
                        })
            if rows_stu:
                with st.expander("Per-structure specialist breakdown (test split)"):
                    st.dataframe(pd.DataFrame(rows_stu), width="stretch")

    if eval_audit:
        st.divider()
        st.subheader("Independent Leakage & Video-Cluster Bootstrap Audit (`out/eval_audit.json`)")
        st.json(eval_audit)
    eval_report_md = resolve_artifact("eval_report.md")
    if eval_report_md.exists():
        with st.expander("View Full Independent Evaluation Audit Report (`out/eval_report.md`)"):
            st.markdown(eval_report_md.read_text())

    # External Held-Out Validation Sets (#16: Endoscapes2023 & CholecT50)
    ext_root = C.DATA_DIR / "external"
    ext_sets = [
        ("endoscapes2023", "Endoscapes2023 (CAMMA Strasbourg — 50 unseen videos, scorable: gallbladder & cystic_duct)"),
        ("cholect50", "CholecT50 (Disjoint non-CholecSeg8k videos — scorable: grasper & l_hook)"),
    ]
    if any((ext_root / slug / "index.csv").exists() for slug, _ in ext_sets):
        st.divider()
        st.subheader("External Zero-Shot Held-Out Datasets (`Endoscapes2023` & `CholecT50`)")
        st.caption(
            "Evaluated with the exact same frozen manual and prompts (zero re-tuning). "
            "Unannotated classes in each external dataset carry `gt_<key> = -1` and are excluded from scoring."
        )
        for slug, title in ext_sets:
            ext_idx_p = ext_root / slug / "index.csv"
            if not ext_idx_p.exists():
                continue
            ext_idx_df = pd.read_csv(ext_idx_p)
            ext_test_df = ext_idx_df[ext_idx_df["split"] == "test"] if "split" in ext_idx_df.columns else ext_idx_df
            ext_blank_p = resolve_artifact(f"labels_{slug}_blank.jsonl")
            ext_man_p = resolve_artifact(f"labels_{slug}_manual.jsonl")
            ext_b_stats = score_jsonl_against_index(ext_blank_p, ext_idx_df, split_filter="test") if ext_blank_p.exists() else None
            ext_m_stats = score_jsonl_against_index(ext_man_p, ext_idx_df, split_filter="test") if ext_man_p.exists() else None

            with st.container(border=True):
                st.markdown(f"**{title}**")
                ec1, ec2, ec3, ec4 = st.columns(4)
                ec1.metric("External test frames", int(len(ext_test_df)))
                ec2.metric("Unseen test videos", int(ext_test_df["video"].nunique()) if "video" in ext_test_df.columns else 0)
                if ext_b_stats:
                    ec3.metric(
                        "Blank baseline (scorable labels)",
                        f"{ext_b_stats['acc_all_6'] * 100:.1f}%",
                        ", ".join(f"{k}={v*100:.1f}%" for k, v in ext_b_stats["per_key"].items()),
                    )
                else:
                    ec3.metric("Blank baseline [PLACEHOLDER]", "PENDING")
                if ext_m_stats:
                    delta_ext = (
                        f"{(ext_m_stats['acc_all_6'] - ext_b_stats['acc_all_6']) * 100:+.1f} pts vs blank"
                        if ext_b_stats else ", ".join(f"{k}={v*100:.1f}%" for k, v in ext_m_stats["per_key"].items())
                    )
                    ec4.metric(
                        "Frozen Manual (scorable labels)",
                        f"{ext_m_stats['acc_all_6'] * 100:.1f}%",
                        delta_ext,
                    )
                else:
                    ec4.metric("Frozen Manual [PLACEHOLDER]", "PENDING")

# ==============================================================================
# TAB 4: WATCH, MANUAL & ANY SURGERY — Act 3 + Manual + Act 1
# ==============================================================================
with tab_watch:
    st.header("Watch: Frame-by-Frame Verification & Grounded Narration")
    st.markdown(
        "**Verification badge legend:** "
        "<span style='background:#1f9d55;color:white;padding:3px 9px;border-radius:10px;margin-right:6px'>accept (verified — safe to narrate)</span>"
        "<span style='background:#d69e2e;color:white;padding:3px 9px;border-radius:10px;margin-right:6px'>AMBER (uncertain — routed to Human Gate)</span>"
        "<span style='background:#e53e3e;color:white;padding:3px 9px;border-radius:10px'>reject (rejected claim)</span>",
        unsafe_allow_html=True,
    )

    dec_path = resolve_artifact("decisions.csv")
    if not dec_path.exists():
        dec_path = resolve_artifact("live_decisions.csv")
    idx_path = C.DATA_DIR / "index.csv"

    if not dec_path.exists():
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/decisions.csv` or `out/live_decisions.csv`]**\n\n"
            "Once `accept.py` or the Live Judge Test finishes, you can scrub or play held-out test videos with verified bounding boxes, "
            "status chips, and ground-truth reveal."
        )
        if idx_path.exists():
            try:
                idx_preview = pd.read_csv(idx_path)
                test_preview = idx_preview[idx_preview["split"] == "test"] if "split" in idx_preview.columns else idx_preview
                if not test_preview.empty:
                    st.info("Previewing unlabelled held-out test frames from `data/index.csv` while awaiting verifier decisions:")
                    vids_p = sorted(test_preview["video"].unique())
                    v_p = st.selectbox("Test video (raw preview)", vids_p, key="raw_preview_vid")
                    frames_p = sorted(test_preview[test_preview["video"] == v_p]["frame_id"].unique(), key=frame_sort_key)
                    i_p = st.slider("Frame (raw preview)", 0, len(frames_p) - 1, 0, key="raw_preview_slider") if len(frames_p) > 1 else 0
                    f_p = frames_p[i_p]
                    row_p = test_preview[test_preview["frame_id"] == f_p].iloc[0]
                    st.image(draw(row_p["path"], pd.DataFrame(), False, frame_id=f_p), width="stretch")
            except Exception:
                pass
        else:
            st.image(draw(None, pd.DataFrame(), False, frame_id="placeholder_preview"), width="stretch")
    else:
        dec_raw = pd.read_csv(dec_path)
        idx_df_w = pd.read_csv(idx_path) if idx_path.exists() else None
        idx_w = idx_df_w.set_index("frame_id") if idx_df_w is not None else None
        vid_col = "video" if "video" in dec_raw.columns else "video_id"

        test_dec = dec_raw[dec_raw["split"] == "test"] if "split" in dec_raw.columns else dec_raw
        if test_dec.empty:
            test_dec = dec_raw
        vids = sorted(test_dec[vid_col].dropna().unique())
        v = st.selectbox("Test video", vids)

        narr_list = load_json(resolve_artifact(f"narration_{v}.json"), [])
        narr = {x["frame_id"]: x for x in narr_list if isinstance(x, dict) and "frame_id" in x}

        frames = sorted(dec_raw[dec_raw[vid_col] == v]["frame_id"].unique(), key=frame_sort_key)
        reveal = st.toggle("Reveal ground truth")
        dec = join_ground_truth_for_reveal(dec_raw, idx_df_w) if reveal else dec_raw

        play = st.button("▶ Play")
        if len(frames) > 1:
            i = st.slider("Frame", 0, len(frames) - 1, 0)
        else:
            i = 0
            if frames:
                st.caption(f"Showing single sampled frame: `{frames[0]}`")
        slot_img, slot_chips, slot_narr = st.empty(), st.empty(), st.empty()

        def lookup_frame_path(f_id: str, rows_f: pd.DataFrame):
            if idx_w is not None and f_id in idx_w.index:
                val = idx_w.loc[f_id, "path"]
                return val.iloc[0] if isinstance(val, pd.Series) else val
            if "frame_path" in rows_f.columns:
                fp = rows_f["frame_path"].dropna()
                if not fp.empty:
                    return fp.iloc[0]
            return None

        def show(f):
            rows = dec[dec["frame_id"] == f]
            f_path = lookup_frame_path(f, rows)
            slot_img.image(draw(f_path, rows, reveal, frame_id=f), width="stretch")
            chips = []
            for r in rows.itertuples():
                status_val = normalize_status(getattr(r, "status", "AMBER"))
                colour = COL.get(status_val, "#d69e2e")
                final_pres = bool(getattr(r, "final_present", getattr(r, "present", 0)))
                label = f"{r.key}: {'yes' if final_pres else 'no'} ({status_val})"
                if reveal and hasattr(r, "final_correct") and pd.notna(getattr(r, "final_correct", None)):
                    label += " ✓" if r.final_correct else " ✗"
                chips.append(
                    f"<span style='background:{colour};color:white;padding:4px 10px;"
                    f"border-radius:12px;margin:3px;display:inline-block'>{label}</span>"
                )
            slot_chips.markdown(" ".join(chips), unsafe_allow_html=True)
            if f in narr:
                slot_narr.markdown(claims_html(narration_claims(narr[f])), unsafe_allow_html=True)
            else:
                acc_present = [r.key for r in rows.itertuples() if normalize_status(getattr(r, "status", "")) == "accept" and getattr(r, "final_present", 0)]
                acc_absent = [r.key for r in rows.itertuples() if normalize_status(getattr(r, "status", "")) == "accept" and not getattr(r, "final_present", 0)]
                unsure_keys = [r.key for r in rows.itertuples() if normalize_status(getattr(r, "status", "")) != "accept"]
                slot_narr.info(
                    f"**[Grounded Status Summary — `narration_{v}.json` optional]**\n\n"
                    f"- **Verified present (`accept`):** {', '.join(acc_present) or 'none'}\n"
                    f"- **Verified absent (`accept`):** {', '.join(acc_absent) or 'none'}\n"
                    f"- **Uncertain / withheld (`AMBER` / `reject`):** {', '.join(unsure_keys) or 'none'}"
                )

        if play:
            for f in frames[i:]:
                show(f)
                time.sleep(1.2)
        elif frames:
            show(frames[i])
            with st.expander("Frame verification audit details"):
                cols_to_show = [
                    c for c in ["key", "present", "conf", "pv_match", "pv_conf", "p_wrong", "trust", "status", "final_present", "final_by", "evidence", "critic"]
                    if c in dec.columns
                ]
                st.dataframe(dec[dec["frame_id"] == frames[i]][cols_to_show], width="stretch")

    st.divider()
    render_narration()

    st.divider()
    st.subheader("Any Surgery, Zero Setup (`out/atlas/*/dataset.json`)")
    st.warning("**Showcase, no ground truth.** Atlas runs on public video of any procedure; nothing here is scored.")
    atlases = sorted((O / "atlas").glob("*/dataset.json")) if (O / "atlas").exists() else []
    if not atlases:
        st.caption("[Optional showcase: run `python atlas.py --procedure ... --video ...` to populate Act 1]")
    else:
        pick = st.selectbox("Procedure", atlases, format_func=lambda p: p.parent.name)
        ds = load_json(pick, {})
        man = load_json(pick.parent / "manual.json", {})
        c1, c2 = st.columns([1, 1])
        with c1:
            if ds.get("source_video"):
                st.video(ds["source_video"])
            st.metric("Human labels used", ds.get("human_labels_used", 0))
            audit_obj = ds.get("audit", {})
            conf_val = audit_obj.get("overall_confidence")
            st.metric("Audit confidence", f"{conf_val:.2f}" if isinstance(conf_val, (int, float)) else "n/a")
        with c2:
            st.subheader("Manual the agent wrote itself")
            for s in man.get("structures", []):
                confused = ", ".join(s.get("confused_with", [])) or "none listed"
                st.markdown(f"**{s.get('name', s.get('key', ''))}**: {s.get('appearance', '')} *(confused with {confused})*")
        tl = ds.get("timeline", {})
        st.dataframe(pd.DataFrame(tl.get("phases", [])), width="stretch")
        st.dataframe(pd.DataFrame(tl.get("events", [])), width="stretch")
        if audit_obj.get("problems"):
            st.warning("Flagged by Argon audit: " + "; ".join(audit_obj["problems"]))
        st.download_button("Download dataset.json", pick.read_text(), file_name="dataset.json")
