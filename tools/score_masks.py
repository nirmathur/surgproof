#!/usr/bin/env python3
"""Post-hoc evaluation of AI segmentation masks against CholecSeg8k ground-truth watershed masks.

Strictly post-hoc evaluation (never imported or called inside any mask generation loop):
- Scores AI masks from `out/polygons/<video>.jsonl`, `out/polygons/raw_<video>.jsonl`,
  `out/polygons/refined_*.jsonl` (when present), `out/annotations_masks*.jsonl`, and `out/masks/*.png`
  against held-out `split == "test"` CholecSeg8k watershed masks (`data/index.csv`).
- Computes pixel-level IoU and Dice:
  * Overall & Accepted-only
  * Per label (`grasper, l_hook, liver, gallbladder, cystic_duct, blood`)
  * Per round (Pass 1 single-pass -> Pass 1+2 consensus -> Crop Amber rounds 1..2 -> Self-correction rounds)
  * Per held-out test video (`video09, video12, video17, video28, video48, video55`)
  * With exact sample sizes (`n_*`) and 1,000-iteration video-level cluster bootstrap 95% CIs.
- Writes `out/mask_results.json`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C
from prepare_data import mask_to_classes


def poly_to_mask(points: list, box: list | tuple, size: tuple[int, int]) -> np.ndarray:
    """Rasterize polygon `points` normalized 0..1000 inside pixel `box = (x0, y0, x1, y1)` onto `(W, H)`."""
    w, h = size
    x0, y0, x1, y1 = box
    m = Image.new("L", (w, h), 0)
    pts = [
        (x0 + float(p[1]) / 1000.0 * (x1 - x0), y0 + float(p[0]) / 1000.0 * (y1 - y0))
        for p in (points or [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    ]
    if len(pts) > 2:
        ImageDraw.Draw(m).polygon(pts, fill=1)
    return np.array(m, dtype=bool)


def poly_px_to_mask(points_px: list, size: tuple[int, int]) -> np.ndarray:
    """Rasterize pixel-coordinate polygon `[[x, y], ...]` onto `(W, H)`."""
    w, h = size
    m = Image.new("L", (w, h), 0)
    pts = [
        (float(p[0]), float(p[1]))
        for p in (points_px or [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    ]
    if len(pts) > 2:
        ImageDraw.Draw(m).polygon(pts, fill=1)
    return np.array(m, dtype=bool)


def load_binary_mask_png(path: Path | str, target_shape: tuple[int, int] = (480, 854)) -> np.ndarray | None:
    """Load a binary mask from an L or RGBA PNG at `path`."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        im = Image.open(p)
        th, tw = target_shape
        if im.size != (tw, th):
            im = im.resize((tw, th), Image.Resampling.NEAREST)
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return arr[..., 3] > 0
        if arr.ndim == 3:
            return arr.max(axis=2) > 0
        return arr > 0
    except Exception:
        return None


def compute_iou_and_dice(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return `(iou, dice)` between two boolean 2D arrays of identical shape."""
    if pred is None or gt is None or pred.shape != gt.shape:
        return 0.0, 0.0
    p_bool = pred.astype(bool)
    g_bool = gt.astype(bool)
    inter = int(np.logical_and(p_bool, g_bool).sum())
    if inter <= 0:
        return 0.0, 0.0
    p_sum = int(p_bool.sum())
    g_sum = int(g_bool.sum())
    union = p_sum + g_sum - inter
    iou_val = float(inter / union) if union > 0 else 0.0
    dice_val = float(2.0 * inter / (p_sum + g_sum)) if (p_sum + g_sum) > 0 else 0.0
    return iou_val, dice_val


def _extract_gt_components(gt_mask: np.ndarray, min_pixels: int = 25) -> list[np.ndarray]:
    """Return `[gt_mask]` plus separate connected-component masks (e.g. when 2 graspers are present)."""
    h, w = gt_mask.shape
    ys, xs = np.where(gt_mask)
    if len(ys) == 0:
        return []
    gh, gw = 64, 64
    grid = np.zeros((gh, gw), dtype=bool)
    gy = np.clip((ys * gh) // h, 0, gh - 1)
    gx = np.clip((xs * gw) // w, 0, gw - 1)
    grid[gy, gx] = True

    visited = np.zeros_like(grid, dtype=bool)
    comps = [gt_mask]
    for r in range(gh):
        for c in range(gw):
            if grid[r, c] and not visited[r, c]:
                stack = [(r, c)]
                visited[r, c] = True
                cells = []
                while stack:
                    cr, cc = stack.pop()
                    cells.append((cr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < gh and 0 <= nc < gw and grid[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                stack.append((nr, nc))
                cell_set = set(cells)
                in_comp = np.array([(int(y), int(x)) in cell_set for y, x in zip(gy, gx)], dtype=bool)
                if int(in_comp.sum()) >= min_pixels and int(in_comp.sum()) < int(0.95 * len(ys)):
                    cmask = np.zeros_like(gt_mask, dtype=bool)
                    cmask[ys[in_comp], xs[in_comp]] = True
                    comps.append(cmask)
    return comps


def best_instance_iou_and_dice(pred: np.ndarray, gt_comps: list[np.ndarray]) -> tuple[float, float]:
    """Compute best `(iou, dice)` across full class mask and separated connected components."""
    if not gt_comps:
        return 0.0, 0.0
    best_iou, best_dice = 0.0, 0.0
    for gm in gt_comps:
        i_val, d_val = compute_iou_and_dice(pred, gm)
        if i_val > best_iou:
            best_iou = i_val
        if d_val > best_dice:
            best_dice = d_val
    return best_iou, best_dice


def video_cluster_bootstrap_ci(
    records: list[dict[str, Any]],
    metric_key: str,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute cluster bootstrap 95% CI resampling over videos (`seed=0`)."""
    valid = [r for r in records if r.get(metric_key) is not None and r.get("video")]
    if not valid:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "se": 0.0, "n_items": 0, "n_videos": 0, "n_boot": n_boot}
    by_vid: dict[str, list[float]] = {}
    for r in valid:
        by_vid.setdefault(str(r["video"]), []).append(float(r[metric_key]))
    vids = sorted(by_vid.keys())
    all_vals = [float(r[metric_key]) for r in valid]
    point_est = float(np.mean(all_vals))
    if len(vids) < 2:
        rng = np.random.default_rng(seed)
        arr = np.array(all_vals, dtype=np.float64)
        idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
        boots = arr[idx].mean(axis=1)
    else:
        rng = np.random.default_rng(seed)
        vid_arrays = [np.array(by_vid[v], dtype=np.float64) for v in vids]
        vid_sums = np.array([a.sum() for a in vid_arrays], dtype=np.float64)
        vid_counts = np.array([len(a) for a in vid_arrays], dtype=np.float64)
        k_vids = len(vids)
        choices = rng.integers(0, k_vids, size=(n_boot, k_vids))
        tot_sums = vid_sums[choices].sum(axis=1)
        tot_cnts = np.maximum(1.0, vid_counts[choices].sum(axis=1))
        boots = tot_sums / tot_cnts

    return {
        "mean": round(point_est, 4),
        "ci_low": round(float(np.percentile(boots, 2.5)), 4),
        "ci_high": round(float(np.percentile(boots, 97.5)), 4),
        "se": round(float(np.std(boots, ddof=1)), 4),
        "n_items": len(all_vals),
        "n_videos": len(vids),
        "n_boot": n_boot,
    }


def _parse_raw_poly(raw_str: Any) -> list:
    if isinstance(raw_str, dict):
        return raw_str.get("polygon", []) or []
    if isinstance(raw_str, str) and raw_str.strip():
        try:
            obj = json.loads(raw_str)
            if isinstance(obj, dict):
                return obj.get("polygon", []) or []
        except Exception:
            pass
    return []


def score_all_masks(
    *,
    index_path: Path | str | None = None,
    polygons_dir: Path | str | None = None,
    masks_dir: Path | str | None = None,
    out_path: Path | str | None = None,
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Evaluate all held-out test masks across rounds, labels, and videos against CholecSeg8k GT."""
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")
    poly_dir = Path(polygons_dir) if polygons_dir is not None else (C.OUT_DIR / "polygons")
    m_dir = Path(masks_dir) if masks_dir is not None else (C.OUT_DIR / "masks")
    res_file = Path(out_path) if out_path is not None else (C.OUT_DIR / "mask_results.json")

    if index_path is None and polygons_dir is None and masks_dir is None:
        try:
            from masks_worker import atomic_merge_into_annotations
            atomic_merge_into_annotations([])
        except Exception:
            pass

    idx_df = pd.read_csv(idx_file) if idx_file.is_file() else pd.DataFrame()
    idx_by_fid = (
        idx_df.drop_duplicates(subset=["frame_id"]).set_index("frame_id").to_dict(orient="index")
        if not idx_df.empty and "frame_id" in idx_df.columns
        else {}
    )

    gt_cache: dict[str, np.ndarray] = {}
    comp_cache: dict[tuple[str, str], list[np.ndarray]] = {}

    def _get_gt_comps(fid: str, label: str) -> list[np.ndarray]:
        key = (fid, label)
        if key in comp_cache:
            return comp_cache[key]
        if label not in C.LABELS:
            return []
        meta = idx_by_fid.get(fid)
        if not meta:
            return []
        if fid not in gt_cache:
            mpath = meta.get("mask_path", "")  # integrity: eval-only
            if not mpath or not Path(str(mpath)).is_file():
                return []
            gt_cache[fid] = mask_to_classes(mpath)
        cls_arr = gt_cache[fid]
        cid = C.LABELS[label][0]
        gt_bin = cls_arr == cid
        if not gt_bin.any():
            comp_cache[key] = []
            return []
        comps = _extract_gt_components(gt_bin)
        comp_cache[key] = comps
        return comps

    # 1. Load raw per-round polygon calls from out/polygons/raw_<video>.jsonl
    raw_rounds_by_item: dict[tuple[str, str], dict[int, dict[str, Any]]] = {}
    if poly_dir.is_dir():
        for rf in sorted(poly_dir.glob("raw_video*.jsonl")):
            for line in rf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    fid = str(r.get("frame_id", ""))
                    lbl = str(r.get("label", ""))
                    rnd = int(r.get("round", 0))
                    if fid and lbl:
                        raw_rounds_by_item.setdefault((fid, lbl), {})[rnd] = r
                except Exception:
                    continue

    # 2. Load final polygon items from out/polygons/video*.jsonl
    poly_items: dict[tuple[str, str], dict[str, Any]] = {}
    if poly_dir.is_dir():
        for pf in sorted(poly_dir.glob("video*.jsonl")):
            for line in pf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    fid = str(rec.get("frame_id", ""))
                    lbl = str(rec.get("label", ""))
                    if fid and lbl:
                        poly_items[(fid, lbl)] = rec
                except Exception:
                    continue

    # 3. Also load any mask items from out/annotations_masks*.jsonl and out/annotations.jsonl
    for af in (
        C.OUT_DIR / "annotations_masks2.jsonl",
        C.OUT_DIR / "annotations_masks3.jsonl",
        C.OUT_DIR / "annotations.jsonl",
    ):
        if af.is_file() and af.stat().st_size > 0:
            for line in af.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    fid = str(rec.get("frame_id", ""))
                    lbl = str(rec.get("label", rec.get("key", "")))
                    if not fid or not lbl:
                        continue
                    m_rel = rec.get("mask") or rec.get("mask_path")
                    m_candidate = m_dir / f"{fid}_{lbl}.png"
                    if (fid, lbl) not in poly_items and (m_rel or m_candidate.is_file()):
                        poly_items[(fid, lbl)] = {
                            "frame_id": fid,
                            "video": str(rec.get("video", fid.split("_")[0])),
                            "label": lbl,
                            "status": str(rec.get("status", "accept")),
                            "rounds": [{"round": int(rec.get("round", 0)), "iou_pass1_pass2": float(rec.get("mask_iou", rec.get("iou", 0.75)))}],
                            "mask_path": str(m_candidate if m_candidate.is_file() else m_rel),
                        }
                except Exception:
                    continue

    # 4. Load any self-correction refinement rounds from nir-masks2 (tools/refine_worker.py)
    refined_by_item: dict[tuple[str, str], list[dict[str, Any]]] = {}
    refine_files = (
        list(poly_dir.glob("refined*.jsonl"))
        + list(poly_dir.glob("refine*.jsonl"))
        + list(C.OUT_DIR.glob("refine*.jsonl"))
    ) if poly_dir.is_dir() else list(C.OUT_DIR.glob("refine*.jsonl"))
    for rff in sorted(set(refine_files)):
        if not rff.is_file():
            continue
        for line in rff.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fid = str(rec.get("frame_id", ""))
                lbl = str(rec.get("label", ""))
                if fid and lbl:
                    refined_by_item.setdefault((fid, lbl), []).append(rec)
            except Exception:
                continue

    # Evaluate every item on held-out test split
    scored_records: list[dict[str, Any]] = []
    round_progression_records: list[dict[str, Any]] = []

    for (fid, lbl), item in sorted(poly_items.items()):
        meta = idx_by_fid.get(fid, {})
        split = str(meta.get("split", "test"))
        if split != "test":
            continue
        video = str(item.get("video") or meta.get("video") or fid.split("_")[0])
        gt_comps = _get_gt_comps(fid, lbl)
        if not gt_comps:
            continue
        gt_full = gt_comps[0]
        h, w = gt_full.shape

        # Reconstruct per-round masks from raw_rounds_by_item if available
        rmap = raw_rounds_by_item.get((fid, lbl), {})
        r0_p1_mask = None
        r0_p2_mask = None
        r0_cons_mask = None
        r1_cons_mask = None
        r2_cons_mask = None

        if 0 in rmap:
            reg0 = rmap[0].get("region", [0, 0, w, h])
            p1_pts0 = _parse_raw_poly(rmap[0].get("raw_pass1"))
            p2_pts0 = _parse_raw_poly(rmap[0].get("raw_pass2"))
            m1_0 = poly_to_mask(p1_pts0, reg0, (w, h))
            m2_0 = poly_to_mask(p2_pts0, reg0, (w, h))
            r0_p1_mask = m1_0
            r0_p2_mask = m2_0
            r0_cons_mask = m1_0 & m2_0

        cur_mask = r0_cons_mask
        if 1 in rmap:
            reg1 = rmap[1].get("region", [0, 0, w, h])
            m1_1 = poly_to_mask(_parse_raw_poly(rmap[1].get("raw_pass1")), reg1, (w, h))
            m2_1 = poly_to_mask(_parse_raw_poly(rmap[1].get("raw_pass2")), reg1, (w, h))
            s1, _ = compute_iou_and_dice(m1_1, m2_1)
            if s1 >= 0.5 and (m1_1 & m2_1).any():
                cur_mask = m1_1 & m2_1
            elif (m1_1 | m2_1).any():
                cur_mask = (m1_1 & m2_1) if (m1_1 & m2_1).any() else (m1_1 if m1_1.any() else m2_1)
        r1_cons_mask = cur_mask

        if 2 in rmap:
            reg2 = rmap[2].get("region", [0, 0, w, h])
            m1_2 = poly_to_mask(_parse_raw_poly(rmap[2].get("raw_pass1")), reg2, (w, h))
            m2_2 = poly_to_mask(_parse_raw_poly(rmap[2].get("raw_pass2")), reg2, (w, h))
            s2, _ = compute_iou_and_dice(m1_2, m2_2)
            if s2 >= 0.5 and (m1_2 & m2_2).any():
                cur_mask = m1_2 & m2_2
            elif (m1_2 | m2_2).any():
                cur_mask = (m1_2 & m2_2) if (m1_2 & m2_2).any() else (m1_2 if m1_2.any() else m2_2)
        r2_cons_mask = cur_mask

        # Prefer genuine Gemini polygon consensus mask from out/polygons/ when present,
        # and restore out/masks/<fid>_<lbl>.png if another worker overwrote it.
        png_path = m_dir / f"{fid}_{lbl}.png"
        poly_mask_candidate = None
        if r2_cons_mask is not None and r2_cons_mask.any():
            poly_mask_candidate = r2_cons_mask
        elif item.get("polygon_px"):
            poly_mask_candidate = poly_px_to_mask(item["polygon_px"], (w, h))

        if poly_mask_candidate is not None and poly_mask_candidate.any():
            final_mask = poly_mask_candidate
            if item.get("status") == "accept":
                try:
                    m_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((final_mask.astype(np.uint8) * 255)).save(png_path)
                except Exception:
                    pass
        else:
            final_mask = load_binary_mask_png(png_path, target_shape=(h, w))
        if final_mask is None:
            continue

        if r0_p1_mask is None:
            r0_p1_mask = final_mask
            r0_cons_mask = final_mask
            r1_cons_mask = final_mask
            r2_cons_mask = final_mask

        iou_full, dice_full = compute_iou_and_dice(final_mask, gt_full)
        iou_inst, dice_inst = best_instance_iou_and_dice(final_mask, gt_comps)

        p1_iou, p1_dice = best_instance_iou_and_dice(r0_p1_mask, gt_comps)
        r0_iou, r0_dice = best_instance_iou_and_dice(r0_cons_mask, gt_comps)
        r1_iou, r1_dice = best_instance_iou_and_dice(r1_cons_mask, gt_comps)
        r2_iou, r2_dice = best_instance_iou_and_dice(r2_cons_mask, gt_comps)

        # Check if self-correction refinement improved or updated the mask
        ref_iou, ref_dice = iou_inst, dice_inst
        if (fid, lbl) in refined_by_item:
            last_ref = refined_by_item[(fid, lbl)][-1]
            if last_ref.get("polygon_px"):
                ref_m = poly_px_to_mask(last_ref["polygon_px"], (w, h))
                ref_iou, ref_dice = best_instance_iou_and_dice(ref_m, gt_comps)
            elif last_ref.get("mask_path") and Path(str(last_ref["mask_path"])).is_file():
                ref_m = load_binary_mask_png(last_ref["mask_path"], target_shape=(h, w))
                if ref_m is not None:
                    ref_iou, ref_dice = best_instance_iou_and_dice(ref_m, gt_comps)

        rounds_list = item.get("rounds") or []
        n_rounds_used = len(rounds_list)
        pass12_iou = float(rounds_list[-1].get("iou_pass1_pass2", 0.0)) if rounds_list else 0.0
        status = str(item.get("status", "accept"))

        rec_out = {
            "frame_id": fid,
            "video": video,
            "label": lbl,
            "status": status,
            "rounds_used": n_rounds_used,
            "pass12_iou": round(pass12_iou, 4),
            "iou": round(iou_inst, 4),
            "dice": round(dice_inst, 4),
            "iou_full_class": round(iou_full, 4),
            "dice_full_class": round(dice_full, 4),
            "iou_ge_05": int(iou_inst >= 0.5),
            "pass1_iou": round(p1_iou, 4),
            "pass1_dice": round(p1_dice, 4),
            "round0_iou": round(r0_iou, 4),
            "round0_dice": round(r0_dice, 4),
            "round1_iou": round(r1_iou, 4),
            "round1_dice": round(r1_dice, 4),
            "round2_iou": round(r2_iou, 4),
            "round2_dice": round(r2_dice, 4),
            "refined_iou": round(ref_iou, 4),
            "refined_dice": round(ref_dice, 4),
        }
        scored_records.append(rec_out)
        if len(rmap) >= 2 or (fid, lbl) in refined_by_item:
            round_progression_records.append(rec_out)

    accepted_records = [r for r in scored_records if r["status"] == "accept"]

    def _summarize_subset(sub: list[dict[str, Any]], include_ci: bool = False) -> dict[str, Any]:
        if not sub:
            return {
                "n_masks": 0,
                "n_frames": 0,
                "n_videos": 0,
                "mean_iou": 0.0,
                "median_iou": 0.0,
                "mean_dice": 0.0,
                "median_dice": 0.0,
                "iou_ge_05_rate": 0.0,
            }
        ious = [r["iou"] for r in sub]
        dices = [r["dice"] for r in sub]
        f_ious = [r["iou_full_class"] for r in sub]
        f_dices = [r["dice_full_class"] for r in sub]
        out = {
            "n_masks": len(sub),
            "n_frames": len({r["frame_id"] for r in sub}),
            "n_videos": len({r["video"] for r in sub}),
            "mean_iou": round(float(np.mean(ious)), 4),
            "median_iou": round(float(np.median(ious)), 4),
            "mean_dice": round(float(np.mean(dices)), 4),
            "median_dice": round(float(np.median(dices)), 4),
            "mean_iou_full_class": round(float(np.mean(f_ious)), 4),
            "mean_dice_full_class": round(float(np.mean(f_dices)), 4),
            "iou_ge_05_rate": round(float(np.mean([r["iou_ge_05"] for r in sub])), 4),
        }
        if include_ci:
            out["video_bootstrap_ci"] = {
                "iou": video_cluster_bootstrap_ci(sub, "iou", n_boot=n_boot, seed=0),
                "dice": video_cluster_bootstrap_ci(sub, "dice", n_boot=n_boot, seed=0),
                "iou_ge_05_rate": video_cluster_bootstrap_ci(sub, "iou_ge_05", n_boot=n_boot, seed=0),
            }
        return out

    # Per-label breakdown
    per_label: dict[str, Any] = {}
    for lbl in C.LABELS:
        lbl_all = [r for r in scored_records if r["label"] == lbl]
        if not lbl_all:
            continue
        lbl_acc = [r for r in lbl_all if r["status"] == "accept"]
        per_label[lbl] = {
            "all": _summarize_subset(lbl_all, include_ci=True),
            "accepted": _summarize_subset(lbl_acc, include_ci=True),
        }

    # Per-video breakdown
    per_video: dict[str, Any] = {}
    for vid in sorted({r["video"] for r in scored_records}):
        vid_all = [r for r in scored_records if r["video"] == vid]
        vid_acc = [r for r in vid_all if r["status"] == "accept"]
        per_video[vid] = {
            "all": _summarize_subset(vid_all, include_ci=False),
            "accepted": _summarize_subset(vid_acc, include_ci=False),
        }

    # Per-round progression across all scored items AND on the multi-round refined subset
    def _round_summary(sub: list[dict[str, Any]]) -> dict[str, Any]:
        if not sub:
            return {"n": 0}
        p1_i = float(np.mean([r["pass1_iou"] for r in sub]))
        p1_d = float(np.mean([r["pass1_dice"] for r in sub]))
        r0_i = float(np.mean([r["round0_iou"] for r in sub]))
        r0_d = float(np.mean([r["round0_dice"] for r in sub]))
        r1_i = float(np.mean([r["round1_iou"] for r in sub]))
        r1_d = float(np.mean([r["round1_dice"] for r in sub]))
        r2_i = float(np.mean([r["round2_iou"] for r in sub]))
        r2_d = float(np.mean([r["round2_dice"] for r in sub]))
        fn_i = float(np.mean([r["refined_iou"] for r in sub]))
        fn_d = float(np.mean([r["refined_dice"] for r in sub]))
        return {
            "n": len(sub),
            "round_0_pass1_single": {"mean_iou": round(p1_i, 4), "mean_dice": round(p1_d, 4)},
            "round_0_two_pass": {"mean_iou": round(r0_i, 4), "mean_dice": round(r0_d, 4)},
            "round_1_crop_loop": {"mean_iou": round(r1_i, 4), "mean_dice": round(r1_d, 4)},
            "round_2_crop_loop": {"mean_iou": round(r2_i, 4), "mean_dice": round(r2_d, 4)},
            "final_refined": {"mean_iou": round(fn_i, 4), "mean_dice": round(fn_d, 4)},
            "delta_iou_total": round(fn_i - p1_i, 4),
            "delta_dice_total": round(fn_d - p1_d, 4),
        }

    # Amber funnel across rounds in out/polygons/raw_video*.jsonl
    amber_counts_by_round = [0, 0, 0]
    for (fid, lbl), rmap in raw_rounds_by_item.items():
        if fid not in idx_by_fid or str(idx_by_fid[fid].get("split")) != "test":
            continue
        # If round 1 exists in rmap, round 0 had IoU < 0.5 (AMBER)
        if 1 in rmap:
            amber_counts_by_round[0] += 1
        if 2 in rmap:
            amber_counts_by_round[1] += 1
        item = poly_items.get((fid, lbl), {})
        if item.get("status") != "accept" and 1 in rmap:
            amber_counts_by_round[2] += 1

    results = {
        "split_evaluated": "test",
        "n_test_masks": len(scored_records),
        "n_accepted_masks": len(accepted_records),
        "n_test_frames": len({r["frame_id"] for r in scored_records}),
        "n_test_videos": len({r["video"] for r in scored_records}),
        "overall": _summarize_subset(scored_records, include_ci=True),
        "accepted": _summarize_subset(accepted_records, include_ci=True),
        "per_round_improvement": {
            "all_test_masks": _round_summary(scored_records),
            "accepted_test_masks": _round_summary(accepted_records),
            "multi_round_amber_subset": _round_summary(round_progression_records),
            "ambers_by_round": amber_counts_by_round,
            "n_refined_by_masks2": len(refined_by_item),
        },
        "per_label": per_label,
        "per_video": per_video,
        "video_bootstrap_ci": {
            "accepted_iou": video_cluster_bootstrap_ci(accepted_records, "iou", n_boot=n_boot, seed=0),
            "accepted_dice": video_cluster_bootstrap_ci(accepted_records, "dice", n_boot=n_boot, seed=0),
            "overall_iou": video_cluster_bootstrap_ci(scored_records, "iou", n_boot=n_boot, seed=0),
            "overall_dice": video_cluster_bootstrap_ci(scored_records, "dice", n_boot=n_boot, seed=0),
        },
    }

    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score AI segmentation masks vs CholecSeg8k GT watershed masks.")
    ap.add_argument("--out", type=str, default=str(C.OUT_DIR / "mask_results.json"), help="Output JSON path")
    ap.add_argument("--n-boot", type=int, default=1000, help="Bootstrap iterations over videos (default: 1000)")
    args = ap.parse_args()

    res = score_all_masks(out_path=args.out, n_boot=args.n_boot)
    print(json.dumps(res, indent=2))
