"""SurgProof AI Video Annotation Studio (pages/1_Annotate.py).

Supervisely-style surgical video annotation toolbox for held-out test video video12.
All metrics, verdicts, boxes, tips, phases, and evidence strings are loaded dynamically
from real pipeline artifacts in out/ and data/ with 'pending' shown whenever an
artifact or field is not yet produced.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import config as C

TARGET_VIDEO = "video12"
TRACK_ORDER = ["grasper", "l_hook", "gallbladder", "cystic_duct", "liver", "blood"]
INSTRUMENT_LABELS = {"grasper", "l_hook"}

STATUS_COLORS = {
    "accept": "#10B981",
    "AMBER": "#F59E0B",
    "reject": "#EF4444",
    "pending": "#64748B",
}

LABEL_RGB = {
    "grasper": (56, 189, 248),
    "l_hook": (168, 85, 247),
    "gallbladder": (16, 185, 129),
    "cystic_duct": (250, 204, 21),
    "liver": (249, 115, 22),
    "blood": (239, 68, 68),
}

GT_RGB = {
    "grasper": (66, 153, 225),
    "l_hook": (159, 122, 234),
    "liver": (160, 82, 45),
    "gallbladder": (72, 187, 120),
    "cystic_duct": (236, 201, 75),
    "blood": (229, 62, 62),
}

PHASE_COLORS = {
    "1": "#0EA5E9",
    "2": "#6366F1",
    "3": "#EC4899",
    "4": "#14B8A6",
    "5": "#F97316",
    "6": "#8B5CF6",
    "7": "#10B981",
}


# ==============================================================================
# 1. Pure Formatting & Parsing Helpers
# ==============================================================================
def fmt_pct(val: Any, digits: int = 1) -> str:
    """Format a ratio (0..1) or percentage number dynamically, or return 'pending'."""
    if val is None or isinstance(val, bool):
        return "pending"
    try:
        if pd.isna(val):
            return "pending"
        num = float(val)
    except (TypeError, ValueError):
        return "pending"
    if np.isnan(num) or np.isinf(num):
        return "pending"
    scaled = num * 100.0 if abs(num) <= 1.0 else num
    return f"{scaled:.{digits}f}" + "%"


def fmt_num(val: Any, digits: int = 3) -> str:
    """Format a floating point value or return 'pending'."""
    if val is None or isinstance(val, bool):
        return "pending"
    try:
        if pd.isna(val):
            return "pending"
        num = float(val)
    except (TypeError, ValueError):
        return "pending"
    if np.isnan(num) or np.isinf(num):
        return "pending"
    return f"{num:.{digits}f}"


def normalize_verdict(val: Any) -> str:
    """Normalize a decision or localization verdict to accept | AMBER | reject | pending."""
    if val is None:
        return "pending"
    try:
        if pd.isna(val):
            return "pending"
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "pending", ""):
        return "pending"
    low = s.lower()
    if low == "accept":
        return "accept"
    if low in ("reject", "escalate"):
        return "reject"
    if low in ("amber", "reanalyse"):
        return "AMBER"
    return "AMBER"


def safe_present_flag(val: Any) -> int | None:
    """Convert a presence field to 1, 0, or None if pending."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("", "nan", "none", "pending"):
            return None
        return 1 if s in ("1", "1.0", "true", "yes", "present") else 0
    try:
        return 1 if int(float(val)) != 0 else 0
    except (TypeError, ValueError):
        return None


def parse_norm_box(raw_box: Any) -> list[float] | None:
    """Parse normalized [ymin, xmin, ymax, xmax] coordinates in 0..1000."""
    if raw_box is None:
        return None
    box = raw_box
    if isinstance(raw_box, str):
        s = raw_box.strip()
        if not s or s in ("[]", "nan", "None", "null", "pending"):
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


def parse_norm_tip(raw_tip: Any) -> list[float] | None:
    """Parse normalized instrument tip [y, x] coordinates in 0..1000."""
    if raw_tip is None:
        return None
    tip = raw_tip
    if isinstance(raw_tip, str):
        s = raw_tip.strip()
        if not s or s in ("[]", "nan", "None", "null", "pending"):
            return None
        try:
            tip = json.loads(s)
        except Exception:
            return None
    if not isinstance(tip, (list, tuple)) or len(tip) != 2:
        return None
    try:
        ty, tx = float(tip[0]), float(tip[1])
    except (TypeError, ValueError):
        return None
    if np.isnan(ty) or np.isnan(tx) or not (0.0 <= ty <= 1000.0 and 0.0 <= tx <= 1000.0):
        return None
    return [ty, tx]


def compute_pair_iou(box_a: list[float] | None, box_b: list[float] | None) -> float | None:
    """Compute IoU between two normalized [ymin, xmin, ymax, xmax] boxes."""
    if box_a is None or box_b is None:
        return None
    ay0, ax0, ay1, ax1 = box_a
    by0, bx0, by1, bx1 = box_b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0.0, iy1 - iy0), max(0.0, ix1 - ix0)
    inter = ih * iw
    area_a = max(0.0, ay1 - ay0) * max(0.0, ax1 - ax0)
    area_b = max(0.0, by1 - by0) * max(0.0, bx1 - bx0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return None
    return round(float(inter / union), 4)


def frame_sort_key(frame_id: str) -> tuple[str, int]:
    """Sort frame IDs of shape <video>_<clip>_<frame_idx> chronologically."""
    parts = str(frame_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (str(frame_id), 0)


# ==============================================================================
# 2. Artifact Discovery & Loading (out/ and data/)
# ==============================================================================
def resolve_out_path(rel_name: str, out_dir: Path | None = None) -> Path:
    """Resolve an artifact inside C.OUT_DIR (with coord/artifacts fallback on default workspace)."""
    base_out = Path(out_dir) if out_dir is not None else Path(C.OUT_DIR)
    candidate = base_out / rel_name
    if candidate.exists():
        return candidate
    default_out = Path(C.ROOT) / "out"
    try:
        is_default_out = base_out.resolve() == default_out.resolve()
    except Exception:
        is_default_out = False
    if out_dir is None and is_default_out:
        alt = Path(C.ROOT) / "coord" / "artifacts" / Path(rel_name).name
        if alt.exists():
            return alt
    return candidate


def load_json_safe(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def load_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    except Exception:
        return []
    return rows


def load_headline_metrics(out_dir: Path | None = None) -> dict[str, Any]:
    """Load headline verifier and localization metrics from out/results.json & out/localize_results.json."""
    res_path = resolve_out_path("results.json", out_dir)
    loc_path = resolve_out_path("localize_results.json", out_dir)
    res = load_json_safe(res_path) or {}
    loc = load_json_safe(loc_path) or {}

    acc_block = (
        res.get("accepted_excl_liver")
        or res.get("accepted_5_informative")
        or res.get("accepted")
        or {}
    )
    acc_all6_block = res.get("accepted_all_6_labels") or res.get("accepted") or {}

    acc_str = fmt_pct(acc_block.get("accuracy"))
    cov_str = fmt_pct(acc_block.get("coverage"))
    n_labels = res.get("n_test_labels_excl_liver") or res.get("n_test_labels")
    n_str = str(int(n_labels)) if isinstance(n_labels, (int, float)) and int(n_labels) > 0 else "pending"

    acc6_str = fmt_pct(acc_all6_block.get("accuracy"))
    cov6_str = fmt_pct(acc_all6_block.get("coverage"))
    n_all6 = res.get("n_test_labels_all_6") or res.get("n_test_labels")
    n6_str = str(int(n_all6)) if isinstance(n_all6, (int, float)) and int(n_all6) > 0 else "pending"

    if acc_str != "pending" and cov_str != "pending" and n_str != "pending":
        banner_headline = (
            f"{acc_str} accurate on the {cov_str} it accepted · "
            f"n = {n_str} held-out labels, 6 unseen videos"
        )
    else:
        banner_headline = "pending (awaiting out/results.json)"

    boxes_block = loc.get("boxes") if isinstance(loc.get("boxes"), dict) else {}
    tips_block = loc.get("tips") if isinstance(loc.get("tips"), dict) else {}
    loc_amber = loc.get("amber_loop") if isinstance(loc.get("amber_loop"), dict) else {}

    raw_iou = boxes_block.get("mean_iou")
    if raw_iou is None:
        raw_iou = loc.get("mean_iou", loc.get("test_mean_iou"))
    mean_iou_str = fmt_pct(raw_iou)

    raw_iou50 = boxes_block.get("iou_ge_05_rate")
    iou_ge_05_str = fmt_pct(raw_iou50)

    raw_tip = tips_block.get("within_tool_mask_rate")
    if raw_tip is None:
        raw_tip = loc.get("tip_in_mask_rate", loc.get("test_tip_in_mask_rate"))
    tip_rate_str = fmt_pct(raw_tip)

    raw_tip_3px = tips_block.get("within_tool_mask_3px_rate")
    tip_3px_str = fmt_pct(raw_tip_3px)

    n_loc_anns = loc.get("n_test_annotations") or boxes_block.get("n_scored")
    n_loc_frames = loc.get("n_test_frames")
    ambers_seq = loc_amber.get("ambers_by_round")
    if isinstance(ambers_seq, list) and ambers_seq:
        loc_funnel_str = " → ".join(str(x) for x in ambers_seq)
    else:
        loc_funnel_str = "pending"

    return {
        "banner_headline": banner_headline,
        "acc_str": acc_str,
        "cov_str": cov_str,
        "n_str": n_str,
        "acc6_str": acc6_str,
        "cov6_str": cov6_str,
        "n6_str": n6_str,
        "mean_iou_str": mean_iou_str,
        "iou_ge_05_str": iou_ge_05_str,
        "tip_rate_str": tip_rate_str,
        "tip_3px_str": tip_3px_str,
        "n_loc_anns_str": str(int(n_loc_anns)) if isinstance(n_loc_anns, (int, float)) else "pending",
        "n_loc_frames_str": str(int(n_loc_frames)) if isinstance(n_loc_frames, (int, float)) else "pending",
        "loc_funnel_str": loc_funnel_str,
        "has_results": bool(res),
        "has_localize_results": bool(loc),
    }


def load_video_index(video_id: str = TARGET_VIDEO, data_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load frame paths and watershed mask paths for video_id from data/index.csv."""
    base_data = Path(data_dir) if data_dir is not None else Path(C.DATA_DIR)
    idx_path = base_data / "index.csv"
    if not idx_path.is_file():
        return {}
    try:
        df = pd.read_csv(idx_path)
    except Exception:
        return {}
    if df.empty or "frame_id" not in df.columns:
        return {}
    if "video" in df.columns:
        df = df[df["video"].astype(str) == video_id]
    else:
        df = df[df["frame_id"].astype(str).str.startswith(f"{video_id}_")]
    out: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        fid = str(row.get("frame_id", "")).strip()
        if fid:
            out[fid] = row
    return out


def load_video_decisions(video_id: str = TARGET_VIDEO, out_dir: Path | None = None) -> pd.DataFrame:
    """Load out/decisions.csv filtered to video_id (all frames x 6 structures)."""
    dec_path = resolve_out_path("decisions.csv", out_dir)
    if not dec_path.is_file():
        dec_path = resolve_out_path("live_decisions.csv", out_dir)
    if not dec_path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(dec_path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return pd.DataFrame()

    if "key" not in df.columns and "label" in df.columns:
        df["key"] = df["label"]
    if "video" in df.columns:
        v_mask = df["video"].astype(str) == video_id
    elif "video_id" in df.columns:
        v_mask = df["video_id"].astype(str) == video_id
    elif "frame_id" in df.columns:
        v_mask = df["frame_id"].astype(str).str.startswith(f"{video_id}_")
    else:
        return pd.DataFrame()

    sub = df[v_mask].copy()
    if sub.empty:
        return pd.DataFrame()
    if {"frame_id", "key"}.issubset(sub.columns):
        sub = sub.drop_duplicates(subset=["frame_id", "key"], keep="last").reset_index(drop=True)
    return sub


def load_spatial_annotations(
    video_id: str = TARGET_VIDEO,
    out_dir: Path | None = None,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Load spatial boxes, instrument tips, and surgical phases from annotations.jsonl and annotations_live.jsonl."""
    static_path = resolve_out_path("annotations.jsonl", out_dir)
    live_path = resolve_out_path("annotations_live.jsonl", out_dir)

    static_rows = load_jsonl_safe(static_path)
    live_rows = load_jsonl_safe(live_path)

    by_frame_label: dict[tuple[str, str], dict[str, Any]] = {}
    by_frame_phase: dict[str, dict[str, Any]] = {}

    for src_name, rows in (("annotations.jsonl", static_rows), ("annotations_live.jsonl", live_rows)):
        for r in rows:
            fid = str(r.get("frame_id", "")).strip()
            vid = str(r.get("video", "")).strip()
            if not fid:
                continue
            if vid and vid != video_id and not fid.startswith(f"{video_id}_"):
                continue
            if not vid and not fid.startswith(f"{video_id}_"):
                continue
            lbl = str(r.get("label") or r.get("key") or "").strip()
            if lbl:
                rec_copy = dict(r)
                rec_copy["_source_file"] = src_name
                by_frame_label[(fid, lbl)] = rec_copy
            ph = r.get("phase")
            if ph and str(ph).strip() and str(ph).strip().lower() not in ("nan", "none"):
                p1 = r.get("pass1") if isinstance(r.get("pass1"), dict) else {}
                p2 = r.get("pass2") if isinstance(r.get("pass2"), dict) else {}
                by_frame_phase[fid] = {
                    "phase": str(ph).strip(),
                    "phase_conf": r.get("phase_conf"),
                    "pass1_phase": p1.get("phase"),
                    "pass2_phase": p2.get("phase"),
                }

    stream_meta = {
        "n_static_rows": len(static_rows),
        "n_live_rows": len(live_rows),
        "n_video_items": len(by_frame_label),
        "n_video_frames_localized": len({k[0] for k in by_frame_label}),
        "has_live_file": live_path.is_file(),
        "has_static_file": static_path.is_file(),
    }
    return by_frame_label, by_frame_phase, stream_meta


def collect_video_frames(
    dec_df: pd.DataFrame,
    idx_map: dict[str, dict[str, Any]],
    ann_map: dict[tuple[str, str], dict[str, Any]],
) -> list[str]:
    """Return sorted list of all available frame_ids for video12 across decisions, annotations, and index."""
    fids: set[str] = set()
    if dec_df is not None and not dec_df.empty and "frame_id" in dec_df.columns:
        fids.update(dec_df["frame_id"].dropna().astype(str).tolist())
    fids.update(idx_map.keys())
    fids.update(fid for (fid, _) in ann_map.keys())
    return sorted([f for f in fids if f], key=frame_sort_key)


# ==============================================================================
# 3. Canvas Rendering (AI Masks, Boxes, Tips, Phase Banner, GT Scoring Overlay)
# ==============================================================================
def _cholec_fallback_path(frame_id: str, kind: str = "endo") -> Path | None:
    parts = str(frame_id).split("_")
    if len(parts) == 3 and parts[2].isdigit():
        vid, clip_num, f_num = parts[0], parts[1], int(parts[2])
        clip_id = f"{vid}_{clip_num}"
        for root_cand in (
            Path(C.DATA_DIR) / "cholecseg8k" / "CholecSeg8k",
            Path(C.RAW_DIR) / "CholecSeg8k",
        ):
            p = root_cand / vid / clip_id / f"frame_{f_num}_{kind}.png"
            if p.is_file():
                return p
    return None


def load_base_frame_image(
    frame_id: str,
    idx_row: dict[str, Any] | None,
    dec_rows: pd.DataFrame | None,
    ann_items: list[dict[str, Any]],
) -> tuple[Image.Image, bool]:
    """Load the surgical video frame from disk, or return a labelled studio placeholder canvas."""
    candidates: list[Path] = []
    if idx_row and idx_row.get("path"):
        p_str = str(idx_row["path"]).strip()
        if p_str and p_str.lower() not in ("nan", "none"):
            candidates.extend([Path(p_str), Path(C.ROOT) / p_str])

    for a in ann_items:
        for k in ("img_path", "frame_path"):
            if a.get(k):
                p_str = str(a[k]).strip()
                if p_str and p_str.lower() not in ("nan", "none"):
                    candidates.extend([Path(p_str), Path(C.ROOT) / p_str])

    if dec_rows is not None and not dec_rows.empty and "frame_path" in dec_rows.columns:
        for p_val in dec_rows["frame_path"].dropna().astype(str).tolist():
            if p_val.strip() and p_val.strip().lower() not in ("nan", "none"):
                candidates.extend([Path(p_val.strip()), Path(C.ROOT) / p_val.strip()])

    fb = _cholec_fallback_path(frame_id, "endo")
    if fb is not None:
        candidates.append(fb)

    for cand in candidates:
        if cand.is_file():
            try:
                return Image.open(cand).convert("RGB"), True
            except Exception:
                continue

    canvas = Image.new("RGB", (854, 480), color=(15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([16, 44, 838, 464], outline=(51, 65, 85), width=2)
    draw.text(
        (36, 210),
        f"SurgProof Studio Canvas · Frame: {frame_id or 'pending'}",
        fill=(226, 232, 240),
    )
    draw.text(
        (36, 238),
        "Raw frame image file pending on disk — rendering spatial overlays on studio canvas.",
        fill=(148, 163, 184),
    )
    return canvas, False


def apply_gt_watershed_overlay(
    img: Image.Image,
    frame_id: str,
    idx_row: dict[str, Any] | None,
    alpha: float = 0.42,
) -> tuple[Image.Image, bool]:
    """Overlay CholecSeg8k watershed mask strictly for scoring view when toggled by the user."""
    candidates: list[Path] = []
    if idx_row and idx_row.get("mask_path"):
        mp_str = str(idx_row["mask_path"]).strip()
        if mp_str and mp_str.lower() not in ("nan", "none"):
            candidates.extend([Path(mp_str), Path(C.ROOT) / mp_str])
    fb_mask = _cholec_fallback_path(frame_id, "endo_watershed_mask")
    if fb_mask is not None:
        candidates.append(fb_mask)

    mask_file: Path | None = None
    for cand in candidates:
        if cand.is_file():
            mask_file = cand
            break
    if mask_file is None:
        return img, False

    try:
        raw_mask = np.array(Image.open(mask_file))
    except Exception:
        return img, False

    ch0 = raw_mask[..., 0] if raw_mask.ndim == 3 else raw_mask
    lut = np.full(256, -1, dtype=np.int16)
    for ws_val, cls_id in C.WATERSHED_VALUE_TO_CLASS.items():
        lut[int(ws_val)] = int(cls_id)
    lut[255] = -1
    cls_map = lut[ch0]

    base = np.array(img.convert("RGB")).astype(np.float32)
    if cls_map.shape != base.shape[:2]:
        return img, False

    any_applied = False
    for key, (cls_id, _) in C.LABELS.items():
        sel = cls_map == int(cls_id)
        if sel.any():
            any_applied = True
            col_arr = np.array(GT_RGB.get(key, (100, 200, 255)), dtype=np.float32)
            base[sel] = (1.0 - alpha) * base[sel] + alpha * col_arr
    return Image.fromarray(base.clip(0, 255).astype(np.uint8)), any_applied


def apply_ai_segmentation_masks(
    img: Image.Image,
    frame_id: str,
    ann_by_label: dict[str, dict[str, Any]],
    out_dir: Path | None = None,
    alpha: float = 0.38,
) -> tuple[Image.Image, int]:
    """Blend semi-transparent AI segmentation masks from out/masks/<frame_id>_<label>.png or mask_path."""
    base_out = Path(out_dir) if out_dir is not None else Path(C.OUT_DIR)
    masks_dir = base_out / "masks"
    base = np.array(img.convert("RGB")).astype(np.float32)
    h, w = base.shape[:2]
    applied_count = 0

    for label in TRACK_ORDER:
        ann = ann_by_label.get(label, {})
        candidates: list[Path] = []
        if ann.get("mask_path"):
            mp = str(ann["mask_path"]).strip()
            if mp and mp.lower() not in ("nan", "none"):
                candidates.extend([Path(mp), base_out / mp, Path(C.ROOT) / mp])
        if frame_id:
            candidates.append(masks_dir / f"{frame_id}_{label}.png")

        mask_path: Path | None = None
        for cand in candidates:
            if cand.is_file():
                mask_path = cand
                break
        if mask_path is None:
            continue

        try:
            m_img = Image.open(mask_path).convert("L")
            if m_img.size != (w, h):
                m_img = m_img.resize((w, h), Image.Resampling.NEAREST)
            m_arr = np.array(m_img) > 127
            if m_arr.any():
                col = np.array(LABEL_RGB.get(label, (56, 189, 248)), dtype=np.float32)
                base[m_arr] = (1.0 - alpha) * base[m_arr] + alpha * col
                applied_count += 1
        except Exception:
            continue

    if applied_count == 0:
        return img, 0
    return Image.fromarray(base.clip(0, 255).astype(np.uint8)), applied_count


def render_annotation_canvas(
    frame_id: str,
    idx_row: dict[str, Any] | None,
    dec_rows: pd.DataFrame | None,
    ann_by_label: dict[str, dict[str, Any]],
    phase_info: dict[str, Any] | None,
    *,
    show_ai_masks: bool = True,
    show_pass_boxes: bool = False,
    show_gt_overlay: bool = False,
    out_dir: Path | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Render the surgical frame canvas with AI masks, boxes, tip crosshairs, phase banner, and status chips."""
    base_img, has_real_image = load_base_frame_image(
        frame_id,
        idx_row,
        dec_rows,
        list(ann_by_label.values()),
    )

    gt_applied = False
    if show_gt_overlay:
        base_img, gt_applied = apply_gt_watershed_overlay(base_img, frame_id, idx_row)

    masks_applied = 0
    if show_ai_masks:
        base_img, masks_applied = apply_ai_segmentation_masks(
            base_img,
            frame_id,
            ann_by_label,
            out_dir=out_dir,
        )

    canvas = base_img.copy()
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas)

    dec_by_key: dict[str, Any] = {}
    if dec_rows is not None and not dec_rows.empty and "key" in dec_rows.columns:
        for r in dec_rows.itertuples():
            dec_by_key[str(r.key)] = r

    # Top Surgical Phase Banner
    phase_str = (phase_info or {}).get("phase") or "pending"
    phase_conf_str = fmt_pct((phase_info or {}).get("phase_conf"), digits=0)
    draw.rectangle([0, 0, w, 32], fill=(15, 23, 42))
    phase_banner_txt = f"PHASE: {phase_str}" + (
        f" ({phase_conf_str})" if phase_conf_str != "pending" else ""
    )
    draw.text((12, 8), phase_banner_txt, fill=(248, 250, 252))

    mask_badge = f"masks: {masks_applied}" if masks_applied > 0 else "(masks: pending)"
    right_banner_txt = f"{frame_id or 'pending'} · {mask_badge}"
    draw.text((max(260, w - len(right_banner_txt) * 7 - 16), 8), right_banner_txt, fill=(148, 163, 184))

    n_boxes_drawn = 0
    n_tips_drawn = 0

    for label in TRACK_ORDER:
        ann = ann_by_label.get(label, {})
        dec = dec_by_key.get(label)

        verdict_raw = ann.get("status") if ann else (getattr(dec, "status", None) if dec is not None else None)
        verdict = normalize_verdict(verdict_raw)
        color_hex = STATUS_COLORS.get(verdict, STATUS_COLORS["pending"])

        # Optional Pass-1 / Pass-2 verification boxes
        if show_pass_boxes and ann:
            p1_box = parse_norm_box((ann.get("pass1") or {}).get("box") if isinstance(ann.get("pass1"), dict) else None)
            p2_box = parse_norm_box((ann.get("pass2") or {}).get("box") if isinstance(ann.get("pass2"), dict) else None)
            for p_box, p_col in ((p1_box, "#38BDF8"), (p2_box, "#C084FC")):
                if p_box is not None:
                    py0, px0, py1, px1 = (
                        int(p_box[0] * h / 1000),
                        int(p_box[1] * w / 1000),
                        int(p_box[2] * h / 1000),
                        int(p_box[3] * w / 1000),
                    )
                    if px1 > px0 and py1 > py0:
                        draw.rectangle([px0, py0, px1, py1], outline=p_col, width=1)

        box_norm = parse_norm_box(ann.get("box")) if ann else None
        if box_norm is None and dec is not None:
            final_pres = safe_present_flag(getattr(dec, "final_present", getattr(dec, "present", None)))
            if final_pres == 1:
                box_norm = parse_norm_box(getattr(dec, "box", None))

        if box_norm is not None:
            y0, x0, y1, x1 = (
                max(32, min(h - 1, int(box_norm[0] * h / 1000))),
                max(0, min(w - 1, int(box_norm[1] * w / 1000))),
                max(32, min(h - 1, int(box_norm[2] * h / 1000))),
                max(0, min(w - 1, int(box_norm[3] * w / 1000))),
            )
            if x1 > x0 and y1 > y0:
                draw.rectangle([x0, y0, x1, y1], outline=color_hex, width=3)
                rnd_val = ann.get("round") if ann else None
                rnd_suffix = f" · r{int(rnd_val)}" if isinstance(rnd_val, (int, float)) and not pd.isna(rnd_val) else ""
                tag_txt = f"{label} [{verdict}{rnd_suffix}]"
                tag_w = max(92, len(tag_txt) * 6 + 12)
                tag_y = max(33, y0 - 19)
                draw.rectangle([x0, tag_y, min(w - 1, x0 + tag_w), tag_y + 18], fill=color_hex)
                draw.text((x0 + 5, tag_y + 2), tag_txt, fill=(255, 255, 255))
                n_boxes_drawn += 1

        tip_norm = parse_norm_tip(ann.get("tip")) if ann else None
        if tip_norm is not None:
            ty = max(32, min(h - 1, int(tip_norm[0] * h / 1000)))
            tx = max(0, min(w - 1, int(tip_norm[1] * w / 1000)))
            rad = 7
            draw.ellipse([tx - rad, ty - rad, tx + rad, ty + rad], outline="#38BDF8", fill=color_hex, width=2)
            draw.line([tx - 13, ty, tx + 13, ty], fill="#38BDF8", width=2)
            draw.line([tx, ty - 13, tx, ty + 13], fill="#38BDF8", width=2)
            draw.text((min(w - 95, tx + 10), max(34, ty - 12)), f"TIP:{label}", fill="#38BDF8")
            n_tips_drawn += 1

    meta = {
        "has_real_image": has_real_image,
        "gt_applied": gt_applied,
        "masks_applied": masks_applied,
        "mask_status": f"{masks_applied} overlaid" if masks_applied > 0 else "(masks: pending)",
        "n_boxes_drawn": n_boxes_drawn,
        "n_tips_drawn": n_tips_drawn,
        "phase": phase_str,
    }
    return canvas, meta


# ==============================================================================
# 4. Inspector Card & Multi-Track Timeline Builders
# ==============================================================================
def build_inspector_records(
    frame_id: str,
    dec_rows: pd.DataFrame | None,
    ann_by_label: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble per-structure inspector details for all 6 tracked surgical labels on frame_id."""
    dec_by_key: dict[str, Any] = {}
    if dec_rows is not None and not dec_rows.empty and "key" in dec_rows.columns:
        for r in dec_rows.itertuples():
            dec_by_key[str(r.key)] = r

    records: list[dict[str, Any]] = []
    for label in TRACK_ORDER:
        dec = dec_by_key.get(label)
        ann = ann_by_label.get(label, {})

        status_val = normalize_verdict(
            getattr(dec, "status", None) if dec is not None else ann.get("status")
        )
        pres_flag = safe_present_flag(
            getattr(dec, "final_present", getattr(dec, "present", None)) if dec is not None else (1 if ann else None)
        )
        if pres_flag == 1:
            presence_str = "PRESENT"
        elif pres_flag == 0:
            presence_str = "absent"
        else:
            presence_str = "pending"

        conf_val = getattr(dec, "conf", getattr(dec, "confidence", None)) if dec is not None else ann.get("conf")
        conf_str = fmt_pct(conf_val, digits=0)

        ev_raw = getattr(dec, "evidence", getattr(dec, "rationale", None)) if dec is not None else None
        ev_str = (
            str(ev_raw).strip()
            if ev_raw is not None and not pd.isna(ev_raw) and str(ev_raw).strip() not in ("", "nan", "None")
            else "pending"
        )

        pv_raw = getattr(dec, "pv_match", None) if dec is not None else None
        if pv_raw is None or (isinstance(pv_raw, float) and pd.isna(pv_raw)) or str(pv_raw).strip() in ("", "nan", "None"):
            pv_str = "n/a (absent)" if pres_flag == 0 else "pending"
        else:
            pv_bool = str(pv_raw).strip().lower() in ("true", "1", "1.0", "yes")
            pv_str = "agree (True)" if pv_bool else "disagree (False)"

        pw_raw = getattr(dec, "p_wrong", None) if dec is not None else None
        pw_str = fmt_pct(pw_raw, digits=0)
        critic_raw = getattr(dec, "critic", None) if dec is not None else None
        critic_str = (
            str(critic_raw).strip()
            if critic_raw is not None and not pd.isna(critic_raw) and str(critic_raw).strip() not in ("", "nan", "None")
            else "pending"
        )

        # 4-round Amber Loop progression from decisions.csv
        round_statuses: list[str] = []
        for r_idx in range(4):
            col_name = f"status_round_{r_idx}"
            val_r = getattr(dec, col_name, None) if dec is not None else None
            st_r = normalize_verdict(val_r)
            round_statuses.append(st_r)
        if any(s != "pending" for s in round_statuses):
            amber_progression_str = " → ".join(f"R{i}:{s}" for i, s in enumerate(round_statuses))
        else:
            amber_progression_str = "pending"

        argon_raw = getattr(dec, "argon_reason", getattr(dec, "argon_reasoning", None)) if dec is not None else None
        argon_str = (
            str(argon_raw).strip()
            if argon_raw is not None and not pd.isna(argon_raw) and str(argon_raw).strip() not in ("", "nan", "None")
            else "none (settled at Round 0)" if status_val != "pending" else "pending"
        )

        # Spatial Pass-1 vs Pass-2 localization details
        box_norm = parse_norm_box(ann.get("box")) if ann else None
        p1_dict = ann.get("pass1") if isinstance(ann.get("pass1"), dict) else {}
        p2_dict = ann.get("pass2") if isinstance(ann.get("pass2"), dict) else {}
        p1_box = parse_norm_box(p1_dict.get("box"))
        p2_box = parse_norm_box(p2_dict.get("box"))
        pair_iou = p2_dict.get("iou")
        if pair_iou is None:
            pair_iou = compute_pair_iou(p1_box, p2_box)
        iou_str = fmt_num(pair_iou, digits=3) if pair_iou is not None else ("n/a (absent)" if pres_flag == 0 else "pending")

        tip_norm = parse_norm_tip(ann.get("tip")) if ann else None
        if label in INSTRUMENT_LABELS:
            tip_str = f"[{int(tip_norm[0])}, {int(tip_norm[1])}]" if tip_norm is not None else ("n/a (absent)" if pres_flag == 0 else "pending")
        else:
            tip_str = "n/a (anatomical structure)"

        box_str = (
            f"[{int(box_norm[0])}, {int(box_norm[1])}, {int(box_norm[2])}, {int(box_norm[3])}]"
            if box_norm is not None
            else ("n/a (absent)" if pres_flag == 0 else "pending")
        )
        loc_round_val = ann.get("round") if ann else None
        loc_round_str = (
            f"Round {int(loc_round_val)} ({normalize_verdict(ann.get('status'))})"
            if isinstance(loc_round_val, (int, float)) and not pd.isna(loc_round_val)
            else ("n/a" if pres_flag == 0 else "pending")
        )

        records.append({
            "frame_id": frame_id,
            "label": label,
            "human_name": C.LABELS.get(label, (0, label))[1],
            "status": status_val,
            "color": STATUS_COLORS.get(status_val, STATUS_COLORS["pending"]),
            "presence": presence_str,
            "conf_str": conf_str,
            "evidence": ev_str,
            "pv_match_str": pv_str,
            "p_wrong_str": pw_str,
            "critic": critic_str,
            "box_str": box_str,
            "iou_str": iou_str,
            "tip_str": tip_str,
            "loc_round_str": loc_round_str,
            "amber_progression": amber_progression_str,
            "argon_reason": argon_str,
        })
    return records


def build_multitrack_timeline_html(
    frame_ids: list[str],
    selected_frame_id: str,
    dec_df: pd.DataFrame,
    ann_map: dict[tuple[str, str], dict[str, Any]],
    phase_map: dict[str, dict[str, Any]],
) -> str:
    """Build a Supervisely-style HTML/CSS multi-track timeline across all video12 frames."""
    if not frame_ids:
        return (
            "<div style='padding:14px;background:#0F172A;border:1px solid #334155;border-radius:8px;color:#94A3B8'>"
            "Multi-Track Timeline: <b>pending</b> (awaiting video12 rows in out/decisions.csv or out/annotations.jsonl)."
            "</div>"
        )

    dec_lookup: dict[tuple[str, str], tuple[str, int | None]] = {}
    if dec_df is not None and not dec_df.empty and {"frame_id", "key"}.issubset(dec_df.columns):
        for r in dec_df.itertuples():
            fid = str(r.frame_id)
            k = str(r.key)
            st_norm = normalize_verdict(getattr(r, "status", None))
            pres = safe_present_flag(getattr(r, "final_present", getattr(r, "present", None)))
            dec_lookup[(fid, k)] = (st_norm, pres)

    n_frames = len(frame_ids)
    grid_cols = f"150px repeat({n_frames}, minmax(9px, 1fr))"

    parts: list[str] = [
        "<div style='background:#0B1120;border:1px solid #1E293B;border-radius:10px;padding:12px 14px;"
        "overflow-x:auto;font-family:ui-sans-serif,system-ui,sans-serif;color:#E2E8F0'>",
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px'>",
        f"<div style='font-size:0.88rem;font-weight:600;color:#F8FAFC'>"
        f"🎞️ Multi-Track Timeline · {TARGET_VIDEO} ({n_frames} frames × 6 structures + Surgical Phase Track)</div>",
        "<div style='display:flex;gap:12px;font-size:0.76rem;color:#CBD5E1;flex-wrap:wrap'>"
        "<span><b style='color:#10B981'>■</b> accept (present)</span>"
        "<span><b style='color:#334155'>■</b> accept (absent)</span>"
        "<span><b style='color:#F59E0B'>■</b> AMBER</span>"
        "<span><b style='color:#EF4444'>■</b> reject</span>"
        "<span><b style='color:#38BDF8'>●</b> spatial box/tip</span>"
        "<span><b style='color:#F8FAFC'>□</b> selected frame</span>"
        "</div></div>",
    ]

    # Frame index ruler row
    parts.append(f"<div style='display:grid;grid-template-columns:{grid_cols};gap:2px;align-items:center;margin-bottom:4px'>")
    parts.append("<div style='font-size:0.72rem;color:#94A3B8;font-weight:600'>Frame #</div>")
    for idx, fid in enumerate(frame_ids):
        is_sel = fid == selected_frame_id
        lbl_num = str(idx + 1) if (idx == 0 or (idx + 1) % 5 == 0 or is_sel or idx == n_frames - 1) else ""
        col_txt = "#38BDF8" if is_sel else "#64748B"
        fw = "700" if is_sel else "400"
        parts.append(
            f"<div title='{fid}' style='font-size:0.65rem;text-align:center;color:{col_txt};font-weight:{fw}'>{lbl_num}</div>"
        )
    parts.append("</div>")

    # Phase Track
    parts.append(f"<div style='display:grid;grid-template-columns:{grid_cols};gap:2px;align-items:center;margin-bottom:6px'>")
    parts.append("<div style='font-size:0.76rem;font-weight:600;color:#38BDF8'>🧭 Surgical Phase</div>")
    for fid in frame_ids:
        is_sel = fid == selected_frame_id
        p_info = phase_map.get(fid)
        phase_name = p_info.get("phase", "pending") if p_info else "pending"
        p_num = phase_name[:1] if phase_name and phase_name[0].isdigit() else ""
        bg = PHASE_COLORS.get(p_num, "#475569") if phase_name != "pending" else "#1E293B"
        border = "2px solid #F8FAFC" if is_sel else "1px solid #0F172A"
        parts.append(
            f"<div title='{fid} · Phase: {phase_name}' "
            f"style='height:16px;background:{bg};border:{border};border-radius:3px;"
            f"font-size:0.60rem;color:#F8FAFC;text-align:center;line-height:14px'>"
            f"{p_num}</div>"
        )
    parts.append("</div>")

    # 6 Structure Tracks
    for label in TRACK_ORDER:
        parts.append(f"<div style='display:grid;grid-template-columns:{grid_cols};gap:2px;align-items:center;margin-bottom:3px'>")
        parts.append(f"<div style='font-size:0.76rem;font-weight:600;color:#E2E8F0'>{label}</div>")
        for fid in frame_ids:
            is_sel = fid == selected_frame_id
            st_norm, pres = dec_lookup.get((fid, label), ("pending", None))
            ann = ann_map.get((fid, label))
            has_spatial = bool(ann and (parse_norm_box(ann.get("box")) is not None or parse_norm_tip(ann.get("tip")) is not None))
            if st_norm == "pending" and ann:
                st_norm = normalize_verdict(ann.get("status"))
                pres = 1

            if st_norm == "accept":
                cell_bg = "#10B981" if pres == 1 else "#334155"
            elif st_norm == "AMBER":
                cell_bg = "#F59E0B"
            elif st_norm == "reject":
                cell_bg = "#EF4444"
            else:
                cell_bg = "#1E293B"

            border = "2px solid #F8FAFC" if is_sel else "1px solid #0F172A"
            dot_html = (
                "<span style='display:inline-block;width:5px;height:5px;border-radius:50%;background:#38BDF8'></span>"
                if has_spatial
                else ""
            )
            pres_txt = "present" if pres == 1 else ("absent" if pres == 0 else "pending")
            parts.append(
                f"<div title='{fid} · {label}: {st_norm} ({pres_txt})' "
                f"style='height:18px;background:{cell_bg};border:{border};border-radius:3px;"
                f"display:flex;align-items:center;justify-content:center'>{dot_html}</div>"
            )
        parts.append("</div>")

    parts.append("</div>")
    return "".join(parts)


# ==============================================================================
# 5. Main Streamlit Page Entrypoint
# ==============================================================================
def render_annotate_studio_page() -> None:
    st.title(f"SurgProof · AI Video Annotation Studio ({TARGET_VIDEO} — Held-Out Test Video)")

    metrics = load_headline_metrics()
    dec_df = load_video_decisions(TARGET_VIDEO)
    idx_map = load_video_index(TARGET_VIDEO)
    ann_map, phase_map, stream_meta = load_spatial_annotations(TARGET_VIDEO)
    frame_ids = collect_video_frames(dec_df, idx_map, ann_map)

    # Top Header Banner + Link to Human Gate
    hdr_left, hdr_right = st.columns([3.6, 1.4])
    with hdr_left:
        st.markdown(
            f"**Held-Out Verifier Headline (`out/results.json`):** `{metrics['banner_headline']}`  \n"
            f"**Spatial Localization (`out/localize_results.json`):** "
            f"Box Mean IoU `{metrics['mean_iou_str']}` (IoU ≥ 0.5: `{metrics['iou_ge_05_str']}`) · "
            f"Instrument Tip-in-Mask `{metrics['tip_rate_str']}` (3px tol: `{metrics['tip_3px_str']}`) · "
            f"Spatial Amber Loop `{metrics['loc_funnel_str']}`"
        )
    with hdr_right:
        gate_file = Path(C.ROOT) / "pages" / "2_Human_gate.py"
        if gate_file.is_file():
            try:
                st.page_link(
                    "pages/2_Human_gate.py",
                    label="Final Pass & Release Dataset",
                    icon="🛡️",
                )
            except Exception:
                st.markdown("🛡️ **[Final Pass & Release Dataset (`pages/2_Human_gate.py`)](/Human_gate)**")
        else:
            st.markdown("🛡️ **[Final Pass & Release Dataset (`pages/2_Human_gate.py`)](/Human_gate)**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Held-Out Accepted Accuracy (5 labels)", metrics["acc_str"], f"coverage {metrics['cov_str']} · n={metrics['n_str']}")
    m2.metric("Held-Out Accepted Accuracy (all 6)", metrics["acc6_str"], f"coverage {metrics['cov6_str']} · n={metrics['n6_str']}")
    m3.metric("Held-Out Box Mean IoU", metrics["mean_iou_str"], f"n={metrics['n_loc_anns_str']} annotations")
    m4.metric("Instrument Tip Inside Tool Mask", metrics["tip_rate_str"], f"3px tol: {metrics['tip_3px_str']}")

    st.divider()

    if not frame_ids:
        st.info(
            "**Status: `pending`** — Awaiting `video12` rows in `out/decisions.csv`, `out/annotations_live.jsonl`, or `data/index.csv`."
        )
        st.markdown(
            build_multitrack_timeline_html([], "", dec_df, ann_map, phase_map),
            unsafe_allow_html=True,
        )
        return

    # Initialize session state to first localized frame (or first frame if none localized yet)
    localized_fids = sorted({fid for (fid, _) in ann_map.keys() if fid in frame_ids}, key=frame_sort_key)
    default_fid = localized_fids[0] if localized_fids else frame_ids[0]

    if "ann_studio_fid" not in st.session_state or st.session_state["ann_studio_fid"] not in frame_ids:
        st.session_state["ann_studio_fid"] = default_fid

    # Transport & Stream Controls Bar
    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([0.9, 0.9, 1.1, 1.8, 2.3])
    curr_idx = frame_ids.index(st.session_state["ann_studio_fid"])

    if ctrl1.button("◀ Prev", width="stretch", key="btn_prev_frame"):
        st.session_state["ann_studio_fid"] = frame_ids[(curr_idx - 1) % len(frame_ids)]
        curr_idx = frame_ids.index(st.session_state["ann_studio_fid"])

    if ctrl2.button("▶ Next", width="stretch", key="btn_next_frame"):
        st.session_state["ann_studio_fid"] = frame_ids[(curr_idx + 1) % len(frame_ids)]
        curr_idx = frame_ids.index(st.session_state["ann_studio_fid"])

    if ctrl3.button("▶ Play / Step", width="stretch", key="btn_play_step"):
        if localized_fids:
            next_loc = [f for f in localized_fids if frame_ids.index(f) > curr_idx]
            st.session_state["ann_studio_fid"] = next_loc[0] if next_loc else localized_fids[0]
        else:
            st.session_state["ann_studio_fid"] = frame_ids[(curr_idx + 1) % len(frame_ids)]
        curr_idx = frame_ids.index(st.session_state["ann_studio_fid"])

    stream_clicked = ctrl4.button("⚡ Run / Stream Agent Annotations", width="stretch", key="btn_stream_annotations")
    if stream_clicked:
        ann_map, phase_map, stream_meta = load_spatial_annotations(TARGET_VIDEO)
        localized_fids = sorted({fid for (fid, _) in ann_map.keys() if fid in frame_ids}, key=frame_sort_key)
        if localized_fids:
            st.session_state["ann_studio_fid"] = localized_fids[-1]
            curr_idx = frame_ids.index(st.session_state["ann_studio_fid"])

    with ctrl5:
        st.caption(
            f"Live stream (`out/annotations_live.jsonl`): **{stream_meta['n_video_items']}** spatial boxes/tips across "
            f"**{stream_meta['n_video_frames_localized']} / {len(frame_ids)}** `{TARGET_VIDEO}` frames."
        )

    selected_fid = st.select_slider(
        f"Scrub {TARGET_VIDEO} frames ({len(frame_ids)} frames)",
        options=frame_ids,
        value=st.session_state["ann_studio_fid"],
        key="slider_video12_frame",
    )
    st.session_state["ann_studio_fid"] = selected_fid

    # Main Studio Split: Left Column (Canvas Viewer) & Right Column (Object / Label Inspector)
    col_canvas, col_inspector = st.columns([3, 2])

    f_dec = dec_df[dec_df["frame_id"].astype(str) == selected_fid] if not dec_df.empty else pd.DataFrame()
    f_ann_by_label = {
        lbl: ann_map[(selected_fid, lbl)]
        for lbl in TRACK_ORDER
        if (selected_fid, lbl) in ann_map
    }
    f_phase_info = phase_map.get(selected_fid)
    f_idx_row = idx_map.get(selected_fid)
    inspector_records = build_inspector_records(selected_fid, f_dec, f_ann_by_label)

    with col_canvas:
        st.subheader(f"Frame Canvas Viewer · `{selected_fid}`")
        t_col1, t_col2, t_col3 = st.columns(3)
        show_ai_masks = t_col1.toggle("Overlay AI Masks (`out/masks/`)", value=True, key="tog_ai_masks")
        show_pass_boxes = t_col2.toggle("Show Pass-1 / Pass-2 Boxes", value=False, key="tog_pass_boxes")
        show_gt_overlay = t_col3.toggle(
            "Scoring View Only: Overlay CholecSeg8k Ground-Truth Mask",
            value=False,
            key="tog_gt_scoring_overlay",
        )

        canvas_img, canvas_meta = render_annotation_canvas(
            selected_fid,
            f_idx_row,
            f_dec,
            f_ann_by_label,
            f_phase_info,
            show_ai_masks=show_ai_masks,
            show_pass_boxes=show_pass_boxes,
            show_gt_overlay=show_gt_overlay,
        )
        st.image(
            canvas_img,
            caption=(
                f"{selected_fid} · Phase: {canvas_meta['phase']} · "
                f"Boxes drawn: {canvas_meta['n_boxes_drawn']} · Tips drawn: {canvas_meta['n_tips_drawn']} · "
                f"AI Masks: {canvas_meta['mask_status']}"
                + (" · [SCORING VIEW: CholecSeg8k GT Mask Overlaid]" if canvas_meta["gt_applied"] else "")
            ),
            width="stretch",
        )

        # Status chips bar below canvas
        chips_html = []
        for rec in inspector_records:
            chips_html.append(
                f"<span style='display:inline-block;background:{rec['color']};color:#FFFFFF;"
                f"padding:4px 10px;border-radius:999px;font-size:0.78rem;font-weight:600;margin:2px 4px 2px 0'>"
                f"{rec['label']}: {rec['presence']} · {rec['status']} ({rec['conf_str']})</span>"
            )
        st.markdown("".join(chips_html), unsafe_allow_html=True)

    with col_inspector:
        st.subheader("Object / Label Inspector & Evidence")
        if f_phase_info:
            st.markdown(
                f"**Surgical Phase:** `{f_phase_info.get('phase', 'pending')}` "
                f"(conf `{fmt_pct(f_phase_info.get('phase_conf'), digits=0)}` · "
                f"Pass-1: `{f_phase_info.get('pass1_phase') or 'pending'}` · "
                f"Pass-2: `{f_phase_info.get('pass2_phase') or 'pending'}`)"
            )
        else:
            st.markdown("**Surgical Phase:** `pending` (frame not yet localized in `out/annotations.jsonl`)")

        for rec in inspector_records:
            st.markdown(
                f"<div style='background:#0F172A;border-left:4px solid {rec['color']};"
                f"border-top:1px solid #1E293B;border-right:1px solid #1E293B;border-bottom:1px solid #1E293B;"
                f"border-radius:6px;padding:8px 10px;margin-bottom:8px;font-size:0.80rem;color:#E2E8F0'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>"
                f"<span><b>{rec['label']}</b> <span style='color:#94A3B8'>({rec['human_name']})</span></span>"
                f"<span style='background:{rec['color']};color:#FFFFFF;padding:1px 8px;border-radius:10px;font-weight:600'>"
                f"{rec['presence']} · {rec['status']} · conf {rec['conf_str']}</span></div>"
                f"<div style='color:#CBD5E1;margin-bottom:4px'><b>Evidence:</b> {rec['evidence']}</div>"
                f"<div style='color:#94A3B8;font-size:0.75rem'>"
                f"<b>Double-Check:</b> crop <code>{rec['pv_match_str']}</code> · critic p_wrong <code>{rec['p_wrong_str']}</code><br>"
                f"<b>Spatial (Pass1/2):</b> box <code>{rec['box_str']}</code> · IoU <code>{rec['iou_str']}</code> · "
                f"tip <code>{rec['tip_str']}</code> · loc <code>{rec['loc_round_str']}</code><br>"
                f"<b>Amber Loop (R0→R3):</b> <code>{rec['amber_progression']}</code>"
                + (
                    f"<br><b>Argon Reason:</b> {rec['argon_reason']}"
                    if rec["argon_reason"] not in ("pending", "none (settled at Round 0)")
                    else ""
                )
                + "</div></div>",
                unsafe_allow_html=True,
            )

    st.divider()
    st.subheader(f"Multi-Track Timeline Across All `{TARGET_VIDEO}` Frames")
    st.markdown(
        build_multitrack_timeline_html(frame_ids, selected_fid, dec_df, ann_map, phase_map),
        unsafe_allow_html=True,
    )


render_annotate_studio_page()
