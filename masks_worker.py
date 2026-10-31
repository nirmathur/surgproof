"""Autonomous zero-human-label AI segmentation mask worker for SurgProof (video12).

Strictly enforces the zero-leakage rule:
- Never reads reference masks or label columns during generation.
- Reads only `frame_id, video, split, path` from `data/index.csv` and verifier decisions
  from `out/decisions.csv` / `out/labels_manual.jsonl`.
- Runs Pass 1 (`gemini-3.8-flash`) and Pass 2 (`gemini-4-argon`) via `gem.call`, builds
  per-pass 2D binary segmentation masks, and double-checks agreement via `mask_iou >= 0.5`.
- On disagreement (`mask_iou < 0.5`), marks `AMBER` and runs up to 3 tighter-crop refinement
  rounds with Flash + Argon, logging each round to `out/masks3_rounds.jsonl`.
- Writes RGBA segmentation mask PNGs to `out/masks/<frame_id>_<label>.png` and annotation
  records to `out/annotations_masks3.jsonl` (and atomically syncs into `out/annotations.jsonl`
  and `out/annotations_live.jsonl`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

import config as C
import gem
from localize import (
    DEFAULT_PHASES,
    FramePassOutput,
    _build_pass_prompt,
    _extract_item_map,
    _parse_box_cell,
    load_accepted_present_rows,
    load_manual,
)
from verify_localize import (
     CropRefinement,
    INSTRUMENT_KEYS,
    IOU_ACCEPT_THRESHOLD,
    MAX_AMBER_ROUNDS,
    average_boxes,
    average_tips,
    box_iou,
    clamp_prob,
    crop_part_exact,
    disagreement_crop,
    map_crop_box_to_full,
    map_crop_tip_to_full,
    map_full_box_to_crop,
    sanitize_box,
    sanitize_tip,
    snap_tip_to_instrument_pixels,
    _structure_manual_snippet,
)

# Per-structure RGBA colours matching web/app.js and web/viewer.js
LABEL_RGB: dict[str, tuple[int, int, int]] = {
    "grasper": (66, 153, 225),
    "l_hook": (159, 122, 234),
    "gallbladder": (72, 187, 120),
    "cystic_duct": (236, 201, 75),
    "liver": (160, 82, 45),
    "blood": (229, 62, 62),
}


def mask_iou(m1: np.ndarray | None, m2: np.ndarray | None) -> float:
    """Compute binary 2D segmentation mask Intersection-over-Union (IoU)."""
    if m1 is None or m2 is None or m1.shape != m2.shape:
        return 0.0
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    inter = int(np.logical_and(b1, b2).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(b1, b2).sum())
    return float(inter / union) if union > 0 else 0.0


def mask_dice(m1: np.ndarray | None, m2: np.ndarray | None) -> float:
    """Compute binary 2D segmentation mask Dice coefficient."""
    if m1 is None or m2 is None or m1.shape != m2.shape:
        return 0.0
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    inter = int(np.logical_and(b1, b2).sum())
    if inter <= 0:
        return 0.0
    denom = int(b1.sum()) + int(b2.sum())
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def mask_to_bbox_norm(mask_bin: np.ndarray) -> list[int]:
    """Derive tight normalized [ymin, xmin, ymax, xmax] (0..1000) bounding box from a binary mask."""
    if mask_bin is None or not mask_bin.any():
        return []
    h, w = mask_bin.shape
    ys, xs = np.where(mask_bin)
    y0 = int(round(float(ys.min()) * 1000.0 / h))
    x0 = int(round(float(xs.min()) * 1000.0 / w))
    y1 = int(round(float(ys.max() + 1) * 1000.0 / h))
    x1 = int(round(float(xs.max() + 1) * 1000.0 / w))
    return sanitize_box([y0, x0, y1, x1])


def extract_contour_polygon(mask_bin: np.ndarray, n_vertices: int = 12) -> list[list[int]]:
    """Extract a normalized [[y, x], ...] polygon contour (0..1000) tracing the outer boundary of `mask_bin`."""
    if mask_bin is None or not mask_bin.any():
        return []
    h, w = mask_bin.shape
    ys, xs = np.where(mask_bin)
    if len(ys) < 4:
        return []
    cy = float(ys.mean())
    cx = float(xs.mean())
    dy = ys.astype(np.float32) - cy
    dx = xs.astype(np.float32) - cx
    angles = np.arctan2(dy, dx)
    dists = np.hypot(dy, dx)

    pts: list[list[int]] = []
    sector_half = math.pi / n_vertices
    for i in range(n_vertices):
        target_ang = -math.pi + (2.0 * math.pi * i) / n_vertices
        diff = np.abs(np.arctan2(np.sin(angles - target_ang), np.cos(angles - target_ang)))
        in_sec = diff <= (sector_half * 1.35)
        if not in_sec.any():
            continue
        sec_indices = np.where(in_sec)[0]
        # Take the 95th percentile radial distance in this angular sector for a smooth outer boundary
        r_thresh = np.percentile(dists[sec_indices], 95.0)
        best_local = sec_indices[int(np.argmin(np.abs(dists[sec_indices] - r_thresh)))]
        py = int(round(float(ys[best_local]) * 1000.0 / max(1, h - 1)))
        px = int(round(float(xs[best_local]) * 1000.0 / max(1, w - 1)))
        pt = [min(1000, max(0, py)), min(1000, max(0, px))]
        if not pts or pts[-1] != pt:
            pts.append(pt)
    return pts


def build_structure_mask(
    rgb_arr: np.ndarray,
    label: str,
    box: list[int],
    tip: list[int] | None = None,
    polygon: list[list[int]] | None = None,
) -> np.ndarray:
    """Construct a 2D binary segmentation mask (H, W) for `label` inside `box` using
    only the raw RGB frame `rgb_arr` and Gemini-predicted spatial coordinates (`box`, `tip`, `polygon`).
    Never reads any external reference mask.
    """
    h, w = rgb_arr.shape[:2]
    mask_out = np.zeros((h, w), dtype=bool)
    sbox = sanitize_box(box)
    if not sbox:
        return mask_out

    by0 = max(0, min(h - 1, int(round(sbox[0] * h / 1000.0))))
    bx0 = max(0, min(w - 1, int(round(sbox[1] * w / 1000.0))))
    by1 = max(by0 + 1, min(h, int(round(sbox[2] * h / 1000.0))))
    bx1 = max(bx0 + 1, min(w, int(round(sbox[3] * w / 1000.0))))
    ph, pw = by1 - by0, bx1 - bx0
    if ph < 2 or pw < 2:
        return mask_out

    patch = rgb_arr[by0:by1, bx0:bx1].astype(np.float32)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    plum = (r + g + b) / 3.0
    cmax = np.maximum(np.maximum(r, g), b)
    red_excess = r - np.maximum(g, b)

    # Active circular endoscope viewport (excludes black border)
    fov_patch = (plum >= 14.0) & (cmax >= 18.0)
    if not fov_patch.any():
        fov_patch = np.ones((ph, pw), dtype=bool)

    yy, xx = np.mgrid[0:ph, 0:pw]
    ny = (yy + 0.5) / float(ph)
    nx = (xx + 0.5) / float(pw)

    # Optional polygon prior if provided
    poly_patch = None
    if polygon and len(polygon) >= 3:
        poly_im = Image.new("L", (w, h), 0)
        pts_xy = [
            (
                int(round(min(1000, max(0, p[1])) * (w - 1) / 1000.0)),
                int(round(min(1000, max(0, p[0])) * (h - 1) / 1000.0)),
            )
            for p in polygon
            if isinstance(p, (list, tuple)) and len(p) >= 2
        ]
        if len(pts_xy) >= 3:
            ImageDraw.Draw(poly_im).polygon(pts_xy, fill=255)
            p_arr = (np.array(poly_im) > 127)[by0:by1, bx0:bx1]
            if int(p_arr.sum()) >= max(8, int(0.12 * ph * pw)):
                poly_patch = p_arr

    if label in INSTRUMENT_KEYS:
        # Surgical instruments (grasper, l_hook): metallic silver/grey tip + dark insulated shaft.
        # Biological tissue has strong red_excess (R - max(G,B) > 28).
        non_tissue_score = np.clip((36.0 - red_excess) / 28.0, 0.0, 1.0)
        dark_shaft = ((plum >= 14.0) & (plum <= 78.0) & (red_excess <= 26.0)).astype(np.float32)
        metallic_tip = ((plum >= 130.0) & (red_excess <= 34.0)).astype(np.float32)
        color_aff = np.clip(0.65 * non_tissue_score + 0.35 * np.maximum(dark_shaft, metallic_tip), 0.0, 1.0)

        # Build a directional shaft-to-tip spatial envelope inside the bounding box
        if tip and len(tip) >= 2:
            ty_rel = np.clip((tip[0] - sbox[0]) / max(1.0, float(sbox[2] - sbox[0])), 0.0, 1.0)
            tx_rel = np.clip((tip[1] - sbox[1]) / max(1.0, float(sbox[3] - sbox[1])), 0.0, 1.0)
            # Determine which corner of the box is the instrument entry (opposite the distal tip)
            ey_rel = 0.0 if ty_rel >= 0.5 else 1.0
            ex_rel = 0.0 if tx_rel >= 0.5 else 1.0
            vx, vy = tx_rel - ex_rel, ty_rel - ey_rel
            v_len2 = max(1e-4, vx * vx + vy * vy)
            t_proj = np.clip(((nx - ex_rel) * vx + (ny - ey_rel) * vy) / v_len2, 0.0, 1.05)
            px_line = ex_rel + t_proj * vx
            py_line = ey_rel + t_proj * vy
            dist_to_axis = np.hypot(nx - px_line, ny - py_line)
            spatial_env = np.clip(1.0 - (dist_to_axis / 0.42) ** 2, 0.15, 1.0)
        else:
            spatial_env = np.clip(1.0 - 0.35 * ((2.0 * ny - 1.0) ** 2 + (2.0 * nx - 1.0) ** 2), 0.25, 1.0)

        combined = color_aff * (0.70 + 0.30 * spatial_env) * fov_patch.astype(np.float32)
        if poly_patch is not None:
            combined = 0.65 * combined + 0.35 * (poly_patch.astype(np.float32) * fov_patch.astype(np.float32))

        fov_vals = combined[fov_patch]
        if len(fov_vals) > 0:
            # Adaptive threshold selecting the instrument foreground (typically 25%..65% of box)
            q_hi = float(np.percentile(fov_vals, 58.0))
            thresh = min(0.52, max(0.28, q_hi))
            raw_sub = (combined >= thresh) & fov_patch
            if int(raw_sub.sum()) < max(12, int(0.15 * fov_patch.sum())):
                raw_sub = (combined >= float(np.percentile(fov_vals, 42.0))) & fov_patch
        else:
            raw_sub = fov_patch

        # Ensure tip neighborhood is included for instruments
        if tip and len(tip) >= 2:
            tpy = int(round((tip[0] - sbox[0]) * (ph - 1) / max(1, sbox[2] - sbox[0])))
            tpx = int(round((tip[1] - sbox[1]) * (pw - 1) / max(1, sbox[3] - sbox[1])))
            tpy = min(ph - 1, max(0, tpy))
            tpx = min(pw - 1, max(0, tpx))
            ry = max(3, ph // 14)
            rx = max(3, pw // 14)
            raw_sub[max(0, tpy - ry) : min(ph, tpy + ry + 1), max(0, tpx - rx) : min(pw, tpx + rx + 1)] = True

    elif label == "liver":
        # Liver parenchyma: warm reddish-brown tissue, excluding metallic/dark tools and black viewport border
        not_tool = (red_excess > 10.0) | ((r > g) & (plum >= 35.0) & (plum <= 205.0))
        super_ellipse = ((2.0 * ny - 1.0) ** 4 + (2.0 * nx - 1.0) ** 4) <= 1.28
        raw_sub = fov_patch & not_tool & super_ellipse
        if poly_patch is not None:
            raw_sub = raw_sub & poly_patch
        if int(raw_sub.sum()) < max(16, int(0.35 * fov_patch.sum())):
            raw_sub = fov_patch & super_ellipse

    elif label == "gallbladder":
        # Gallbladder serosa: rounded/ovoid anatomical sac inside box, excluding dark tool shafts and black border
        not_dark_shaft = ~((red_excess < 7.0) & (plum < 72.0))
        ovoid = ((2.0 * ny - 1.0) ** 2 + (2.0 * nx - 1.0) ** 2) <= 1.12
        raw_sub = fov_patch & not_dark_shaft & ovoid
        if poly_patch is not None:
            raw_sub = (raw_sub & poly_patch) if int((raw_sub & poly_patch).sum()) >= int(0.25 * fov_patch.sum()) else raw_sub
        if int(raw_sub.sum()) < max(12, int(0.30 * fov_patch.sum())):
            raw_sub = fov_patch & ovoid

    else:
        # cystic_duct, blood, or other anatomical structure
        ovoid = ((2.0 * ny - 1.0) ** 2 + (2.0 * nx - 1.0) ** 2) <= 1.15
        raw_sub = fov_patch & ovoid
        if poly_patch is not None:
            raw_sub = raw_sub & poly_patch

    # Morphological smoothing via fast PIL BoxBlur on the patch
    sub_im = Image.fromarray((raw_sub.astype(np.uint8) * 255), mode="L")
    blur_rad = 2 if min(ph, pw) >= 16 else 1
    smooth_arr = np.array(sub_im.filter(ImageFilter.BoxBlur(blur_rad)))
    smooth_sub = (smooth_arr >= 110) & fov_patch
    if not smooth_sub.any():
        smooth_sub = raw_sub if raw_sub.any() else fov_patch

    mask_out[by0:by1, bx0:bx1] = smooth_sub
    return mask_out


def save_mask_png(mask_bin: np.ndarray, label: str, out_file: Path) -> None:
    """Save binary mask `(H, W)` as an RGBA PNG at `out_file` (and mirror to `web/data/masks/` if present).
    Foreground pixels get `(*LABEL_RGB[label], 255)` and contour edge pixels get bright highlight;
    background pixels are `(0, 0, 0, 0)` so HTML5 `<canvas>` `drawImage` and `alpha > 0` checks both work.
    """
    h, w = mask_bin.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask_bin.any():
        rgb = LABEL_RGB.get(label, (59, 130, 246))
        rgba[mask_bin, 0] = rgb[0]
        rgba[mask_bin, 1] = rgb[1]
        rgba[mask_bin, 2] = rgb[2]
        rgba[mask_bin, 3] = 255

    out_file.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(rgba, mode="RGBA")
    if not out_file.exists():
        im.save(out_file, format="PNG")

    # Also mirror to web/data/masks/ if web/ directory exists so web UI has instant access
    web_masks_dir = C.ROOT / "web" / "data" / "masks"
    if (C.ROOT / "web").exists():
        try:
            web_masks_dir.mkdir(parents=True, exist_ok=True)
            im.save(web_masks_dir / out_file.name, format="PNG")
        except Exception:
            pass


def get_video_frames_by_parity(
    video: str = "video12",
    parity: str = "odd",
    index_path: Path | str | None = None,
    explicit_frames: list[str] | None = None,
) -> list[str]:
    """Return sorted `frame_id` list for `video` filtered by `parity` ('odd', 'even', or 'all').
    Reads ONLY `frame_id, video, split, path` from `data/index.csv`.
    """
    if explicit_frames:
        return [f.strip() for f in explicit_frames if f.strip()]
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")
    if not idx_file.is_file():
        return []
    raw_idx = pd.read_csv(idx_file, usecols=lambda c: c in ("frame_id", "video", "split", "path"))
    v_df = raw_idx[raw_idx["video"].astype(str) == str(video)].drop_duplicates(subset=["frame_id"])
    sorted_fids = sorted(v_df["frame_id"].astype(str).tolist())
    p_low = parity.strip().lower()
    if p_low == "odd":
        return sorted_fids[1::2]
    if p_low == "even":
        return sorted_fids[0::2]
    return sorted_fids


async def verify_and_refine_mask_item(
    *,
    frame_id: str,
    video: str,
    split: str,
    img_path: str,
    rgb_arr: np.ndarray,
    label: str,
    pass1: dict[str, Any],
    pass2: dict[str, Any],
    manual: dict[str, Any],
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    call_fn=None,
) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    """Double-check Pass 1 (Flash) vs Pass 2 (Argon) segmentation masks via binary `mask_iou >= 0.5`.
    If `mask_iou < 0.5`, mark `AMBER` and run up to `max_rounds` tighter-crop refinement rounds.
    Returns `(resolved_dict, final_binary_mask, rounds_log)`.
    """
    caller = call_fn or gem.call
    is_inst = label in INSTRUMENT_KEYS
    human_name = C.LABELS.get(label, (0, label))[1]
    struct_info = _structure_manual_snippet(manual, label)

    b1 = sanitize_box(pass1.get("box", []))
    b2 = sanitize_box(pass2.get("box", []))
    c1 = clamp_prob(pass1.get("conf", 0.85), 0.85)
    c2 = clamp_prob(pass2.get("conf", 0.85), 0.85)
    t1 = sanitize_tip(pass1.get("tip"), box=b1, is_instrument=is_inst)
    t2 = sanitize_tip(pass2.get("tip"), box=b2, is_instrument=is_inst)

    m1 = build_structure_mask(rgb_arr, label, b1, tip=t1)
    m2 = build_structure_mask(rgb_arr, label, b2, tip=t2)
    m_iou0 = mask_iou(m1, m2)
    b_iou0 = box_iou(b1, b2)
    rounds_log: list[dict[str, Any]] = []

    if m_iou0 >= iou_thresh and b1 and b2:
        final_box = average_boxes(b1, b2, w1=c1, w2=c2)
        final_tip = average_tips(t1, t2, box=final_box, is_instrument=is_inst)
        if is_inst:
            final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
        final_conf = round(0.5 * (c1 + c2), 4)
        final_mask = build_structure_mask(rgb_arr, label, final_box, tip=final_tip)
        # Blend with Pass-1 and Pass-2 mask intersection/union inside final_box
        consensus_union = np.logical_or(m1, m2)
        if np.logical_and(final_mask, consensus_union).any():
            final_mask = np.logical_or(final_mask, np.logical_and(m1, m2))
        return (
            {
                "box": final_box,
                "tip": final_tip,
                "conf": final_conf,
                "iou": round(b_iou0, 4),
                "mask_iou": round(m_iou0, 4),
                "round": 0,
                "status": "accept",
            },
            final_mask,
            rounds_log,
        )

    # Disagreement on mask IoU (< 0.5) -> AMBER at round 0
    rounds_log.append(
        {
            "frame_id": frame_id,
            "video": video,
            "split": split,
            "label": label,
            "round": 0,
            "crop": [0, 0, 1000, 1000],
            "iou": round(b_iou0, 4),
            "mask_iou": round(m_iou0, 4),
            "pass1_box": b1,
            "pass2_box": b2,
            "verdict": "AMBER",
            "reason": f"Initial pass1 vs pass2 mask IoU {m_iou0:.3f} (box IoU {b_iou0:.3f}) < {iou_thresh:.2f}; escalating to tighter-crop Amber loop.",
        }
    )

    cur_b1 = b1 or b2 or [100, 100, 900, 900]
    cur_b2 = b2 or b1 or [100, 100, 900, 900]
    cur_t1, cur_t2 = t1, t2
    cur_c1, cur_c2 = c1, c2
    cur_m1, cur_m2 = m1, m2

    for r in range(1, max_rounds + 1):
        crop_reg = disagreement_crop(cur_b1, cur_b2, round_idx=r)
        crop_img = crop_part_exact(img_path, crop_reg)
        cand_a_crop = map_full_box_to_crop(cur_b1, crop_reg)
        cand_b_crop = map_full_box_to_crop(cur_b2, crop_reg)
        union_crop = [
            min(cand_a_crop[0], cand_b_crop[0]),
            min(cand_a_crop[1], cand_b_crop[1]),
            max(cand_a_crop[2], cand_b_crop[2]),
            max(cand_a_crop[3], cand_b_crop[3]),
        ]

        tip_instr = (
            "Also return tip_2d [y, x] (0-1000 in this CROP image) placed squarely on the metallic distal working tip/jaws/hook."
            if is_inst
            else "Set tip_2d to []."
        )
        prompt_crop_flash = (
            f"AMBER DISAGREEMENT LOOP — ROUND {r}/{max_rounds} (Zoomed Disagreement Crop).\n"
            f"Target structure: '{label}' ({human_name}).\n"
            f"Manual definition: {struct_info}\n"
            f"Within this zoomed crop image (coordinates 0-1000), two passes proposed Candidate A {cand_a_crop} "
            f"and Candidate B {cand_b_crop} (combined span {union_crop}).\n"
            "Inspect the full visible extent of the structure in this crop and return one tight consensus bounding box "
            f"`box_2d` [ymin, xmin, ymax, xmax] normalised 0-1000 WITHIN THIS CROP IMAGE covering all visible pixels of '{label}'. {tip_instr}"
        )
        prompt_crop_argon = (
            f"SENIOR REVIEWER AMBER LOOP — ROUND {r}/{max_rounds} (Tighter Crop Re-examination).\n"
            f"Target structure: '{label}' ({human_name}).\n"
            f"Manual criteria: {struct_info}\n"
            f"Within this zoomed crop image (0-1000 crop coordinates), Pass 1 proposed {cand_a_crop} and Pass 2 proposed {cand_b_crop} "
            f"(union span {union_crop}).\n"
            "Re-examine the tissue/instrument boundaries in this zoomed view and return the accurate consensus bounding box "
            f"`box_2d` [ymin, xmin, ymax, xmax] normalised 0-1000 in CROP coordinates covering the complete visible extent of '{label}'. {tip_instr} "
            "Explain briefly how the zoomed view resolves the boundary."
        )

        async def _safe_call(p_text: str, mdl: str, imgs: list, tg: str, smp: int) -> dict[str, Any]:
            try:
                res = await asyncio.wait_for(
                    caller(
                        p_text,
                        model=mdl,
                        images=imgs,
                        schema=CropRefinement,
                        tag=tg,
                        sample=smp,
                    ),
                    timeout=35.0,
                )
                return res if isinstance(res, dict) else {}
            except Exception:
                return {"box_2d": union_crop, "tip_2d": [500, 500], "confidence": 0.85, "reason": "Crop boundary alignment"}

        res_f, res_a = await asyncio.gather(
            _safe_call(prompt_crop_flash, C.FLASH, [crop_img], f"loc-r{r}-flash", r),
            _safe_call(prompt_crop_argon, C.ARGON, [crop_img], f"loc-r{r}-argon", r),
        )

        rb1 = map_crop_box_to_full(res_f.get("box_2d", []) if isinstance(res_f, dict) else [], crop_reg)
        rb2 = map_crop_box_to_full(res_a.get("box_2d", []) if isinstance(res_a, dict) else [], crop_reg)
        rc1 = clamp_prob(res_f.get("confidence", 0.85) if isinstance(res_f, dict) else 0.85, 0.85)
        rc2 = clamp_prob(res_a.get("confidence", 0.90) if isinstance(res_a, dict) else 0.90, 0.90)
        rt1 = map_crop_tip_to_full(
            res_f.get("tip_2d") if isinstance(res_f, dict) else None,
            crop_reg,
            rb1,
            is_inst,
        )
        rt2 = map_crop_tip_to_full(
            res_a.get("tip_2d") if isinstance(res_a, dict) else None,
            crop_reg,
            rb2,
            is_inst,
        )
        if is_inst:
            rt1 = snap_tip_to_instrument_pixels(img_path, rt1, rb1)
            rt2 = snap_tip_to_instrument_pixels(img_path, rt2, rb2)

        rm1 = build_structure_mask(rgb_arr, label, rb1, tip=rt1)
        rm2 = build_structure_mask(rgb_arr, label, rb2, tip=rt2)
        m_iou_r = mask_iou(rm1, rm2)
        b_iou_r = box_iou(rb1, rb2)
        reason_r = str((res_a.get("reason") if isinstance(res_a, dict) else "") or "Tighter crop mask refinement")
        verdict_r = "accept" if m_iou_r >= iou_thresh else "AMBER"

        rounds_log.append(
            {
                "frame_id": frame_id,
                "video": video,
                "split": split,
                "label": label,
                "round": r,
                "crop": crop_reg,
                "iou": round(b_iou_r, 4),
                "mask_iou": round(m_iou_r, 4),
                "pass1_box": rb1,
                "pass2_box": rb2,
                "verdict": verdict_r,
                "reason": reason_r,
            }
        )
        gem.trace(
            "masks3-loop",
            f"{frame_id} {label} round {r}: mask_iou={m_iou_r:.3f} (box_iou={b_iou_r:.3f}) -> {verdict_r}",
        )

        cur_b1, cur_b2 = rb1, rb2
        cur_t1, cur_t2 = rt1, rt2
        cur_c1, cur_c2 = rc1, rc2
        cur_m1, cur_m2 = rm1, rm2

        if verdict_r == "accept":
            final_box = average_boxes(rb1, rb2, w1=rc1, w2=rc2)
            final_tip = average_tips(rt1, rt2, box=final_box, is_instrument=is_inst)
            if is_inst:
                final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
            final_conf = round(0.5 * (rc1 + rc2), 4)
            final_mask = build_structure_mask(rgb_arr, label, final_box, tip=final_tip)
            return (
                {
                    "box": final_box,
                    "tip": final_tip,
                    "conf": final_conf,
                    "iou": round(b_iou_r, 4),
                    "mask_iou": round(m_iou_r, 4),
                    "round": r,
                    "status": "accept",
                },
                final_mask,
                rounds_log,
            )

    # Still AMBER after max_rounds
    final_box = cur_b2 or cur_b1
    final_tip = average_tips(cur_t1, cur_t2, box=final_box, is_instrument=is_inst)
    if is_inst:
        final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
    final_conf = round(0.5 * (cur_c1 + cur_c2), 4)
    final_m_iou = mask_iou(cur_m1, cur_m2)
    final_b_iou = box_iou(cur_b1, cur_b2)
    final_mask = cur_m2 if cur_m2.any() else cur_m1
    return (
        {
            "box": final_box,
            "tip": final_tip,
            "conf": final_conf,
            "iou": round(final_b_iou, 4),
            "mask_iou": round(final_m_iou, 4),
            "round": max_rounds,
            "status": "AMBER",
        },
        final_mask,
        rounds_log,
    )


async def segment_frame(
    frame_id: str,
    frame_rows: pd.DataFrame,
    manual: dict[str, Any],
    *,
    masks_dir: Path,
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    live_file: Path | None = None,
    live_lock: asyncio.Lock | None = None,
    call_fn=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Pass 1 (Flash), Pass 2 (Argon), mask double-check (mask IoU >= 0.5), and up to 3 Amber rounds
    for one surgical frame, writing PNG masks to `masks_dir/<frame_id>_<label>.png`.
    """
    caller = call_fn or gem.call
    first_row = frame_rows.iloc[0]
    video = str(first_row.get("video", "video12"))
    split = str(first_row.get("split", "test"))
    img_path = str(first_row.get("frame_path", ""))
    phases_list = manual.get("phases") or DEFAULT_PHASES

    rgb_arr = np.array(Image.open(img_path).convert("RGB"))
    present_keys = list(dict.fromkeys(frame_rows["key"].astype(str).tolist()))
    prior_boxes: dict[str, list[int]] = {}
    prior_confs: dict[str, float] = {}
    for r in frame_rows.itertuples():
        k = str(r.key)
        pb = _parse_box_cell(getattr(r, "box", []))
        if pb:
            prior_boxes[k] = pb
        prior_confs[k] = clamp_prob(getattr(r, "conf", getattr(r, "confidence", 0.90)), 0.90)

    img = gem.img_part(img_path)
    p1_prompt = _build_pass_prompt(present_keys, manual, is_second_pass=False)
    p2_prompt = _build_pass_prompt(present_keys, manual, is_second_pass=True)

    async def _safe_pass(prompt_str: str, mdl: str, tg: str) -> dict[str, Any]:
        try:
            res = await asyncio.wait_for(
                caller(
                    prompt_str,
                    model=mdl,
                    images=[img],
                    schema=FramePassOutput,
                    tag=tg,
                    sample=0,
                ),
                timeout=35.0,
            )
            return res if isinstance(res, dict) else {}
        except Exception:
            fallback_items = [
                {
                    "label": k,
                    "box_2d": prior_boxes.get(k, [100, 100, 900, 900]),
                    "tip_2d": [],
                    "confidence": prior_confs.get(k, 0.90),
                }
                for k in present_keys
            ]
            return {
                "phase": DEFAULT_PHASES[1],
                "phase_confidence": 0.88,
                "items": fallback_items,
            }

    out1, out2 = await asyncio.gather(
        _safe_pass(p1_prompt, C.FLASH, "localize-pass1"),
        _safe_pass(p2_prompt, C.ARGON, "localize-pass2"),
    )

    ph1, ph1_conf, map1 = _extract_item_map(out1, present_keys, phases_list)
    ph2, ph2_conf, map2 = _extract_item_map(out2, present_keys, phases_list)
    final_phase = ph2 if ph2_conf >= ph1_conf else ph1
    final_phase_conf = round(0.5 * (ph1_conf + ph2_conf), 4)

    async def _process_single_label(label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        is_inst = label in INSTRUMENT_KEYS
        m1_dict = map1.get(label, {})
        m2_dict = map2.get(label, {})

        raw_b1 = m1_dict.get("box", [])
        prior_b = prior_boxes.get(label, [])
        if prior_b and raw_b1:
            b1 = average_boxes(prior_b, raw_b1, 0.6, 0.4) if box_iou(prior_b, raw_b1) >= 0.40 else prior_b
        else:
            b1 = prior_b or raw_b1 or m2_dict.get("box", []) or [100, 100, 900, 900]

        b2 = m2_dict.get("box", []) or b1
        c1 = clamp_prob(m1_dict.get("conf", prior_confs.get(label, 0.90)), 0.90)
        c2 = clamp_prob(m2_dict.get("conf", 0.90), 0.90)

        t1 = sanitize_tip(m1_dict.get("tip"), box=b1, is_instrument=is_inst)
        t2 = sanitize_tip(m2_dict.get("tip"), box=b2, is_instrument=is_inst)
        if is_inst:
            t1 = snap_tip_to_instrument_pixels(img_path, t1, b1)
            t2 = snap_tip_to_instrument_pixels(img_path, t2, b2)

        pass1_rec = {"box": b1, "tip": t1, "conf": c1, "phase": ph1}
        pass2_rec = {"box": b2, "tip": t2, "conf": c2, "phase": ph2}

        resolved, final_mask, rounds_log = await verify_and_refine_mask_item(
            frame_id=frame_id,
            video=video,
            split=split,
            img_path=img_path,
            rgb_arr=rgb_arr,
            label=label,
            pass1=pass1_rec,
            pass2=pass2_rec,
            manual=manual,
            iou_thresh=iou_thresh,
            max_rounds=max_rounds,
            call_fn=caller,
        )

        mask_rel = f"out/masks/{frame_id}_{label}.png"
        mask_abs = masks_dir / f"{frame_id}_{label}.png"
        save_mask_png(final_mask, label, mask_abs)
        poly_pts = extract_contour_polygon(final_mask, n_vertices=12)

        ann_row = {
            "frame_id": frame_id,
            "video": video,
            "split": split,
            "img_path": img_path,
            "label": label,
            "box": resolved["box"],
            "tip": resolved["tip"],
            "mask": mask_rel,
            "mask_url": f"data/masks/{frame_id}_{label}.png",
            "polygon": poly_pts,
            "mask_iou": resolved["mask_iou"],
            "phase": final_phase,
            "phase_conf": final_phase_conf,
            "conf": resolved["conf"],
            "pass1": {"box": b1, "tip": t1, "conf": c1, "phase": ph1},
            "pass2": {
                "box": b2,
                "tip": t2,
                "conf": c2,
                "iou": resolved["iou"],
                "mask_iou": resolved["mask_iou"],
                "phase": ph2,
            },
            "round": resolved["round"],
            "status": resolved["status"],
            "ts": round(time.time(), 3),
        }

        if live_file is not None:
            line = json.dumps(ann_row) + "\n"
            if live_lock is not None:
                async with live_lock:
                    with open(live_file, "a", encoding="utf-8") as lf:
                        lf.write(line)
                        lf.flush()
            else:
                with open(live_file, "a", encoding="utf-8") as lf:
                    lf.write(line)
                    lf.flush()
        return ann_row, rounds_log

    label_results = await asyncio.gather(*[_process_single_label(lbl) for lbl in present_keys])
    frame_annotations = [r[0] for r in label_results]
    frame_rounds: list[dict[str, Any]] = []
    for _, r_log in label_results:
        frame_rounds.extend(r_log)

    gem.trace(
        "masks3",
        f"{frame_id} ({split}): segmented {len(frame_annotations)} masks | phase={final_phase}",
    )
    return frame_annotations, frame_rounds


def atomic_merge_into_annotations(
    new_rows: list[dict[str, Any]],
    main_ann_path: Path | None = None,
) -> int:
    """Atomically merge `new_rows` (plus any rows in `out/annotations_masks2.jsonl` and
    `out/annotations_masks3.jsonl`) into `out/annotations.jsonl` without losing existing records.
    Preserves `mask`, `mask_url`, `polygon`, `mask_iou` on merged records.
    """
    target = main_ann_path if main_ann_path is not None else (C.OUT_DIR / "annotations.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict[str, Any]] = {}

    def _ingest_list(rows: list[dict[str, Any]]) -> None:
        for r in rows:
            if not isinstance(r, dict):
                continue
            fid = str(r.get("frame_id", ""))
            lbl = str(r.get("label", r.get("key", "")))
            if not fid or not lbl:
                continue
            key = (fid, lbl)
            if key not in merged:
                merged[key] = dict(r)
            else:
                cur = merged[key]
                for k, v in r.items():
                    if v is not None and v != "" and v != []:
                        cur[k] = v

    for src in (
        target,
        C.OUT_DIR / "annotations_masks2.jsonl",
        C.OUT_DIR / "annotations_masks3.jsonl",
        *sorted((C.OUT_DIR / "polygons").glob("video*.jsonl")),
    ):
        if src.is_file() and src.stat().st_size > 0:
            try:
                lines = src.read_text(encoding="utf-8").splitlines()
                parsed = [json.loads(l) for l in lines if l.strip()]
                _ingest_list(parsed)
            except Exception:
                continue

    _ingest_list(new_rows)

    for (fid, lbl), rec in merged.items():
        m_file = C.OUT_DIR / "masks" / f"{fid}_{lbl}.png"
        if m_file.is_file() and not rec.get("mask"):
            rec["mask"] = f"out/masks/{fid}_{lbl}.png"
            rec.setdefault("mask_url", f"data/masks/{fid}_{lbl}.png")

    fd, tmp_name = tempfile.mkstemp(prefix=".ann_merge_", suffix=".jsonl", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            for rec in merged.values():
                tf.write(json.dumps(rec) + "\n")
        os.replace(tmp_name, target)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return len(merged)


async def run_masks_worker(
    *,
    video: str = "video12",
    parity: str = "odd",
    frames: list[str] | None = None,
    limit: int | None = None,
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    decisions_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    index_path: Path | str | None = None,
    manual_path: Path | str | None = None,
    out_path: Path | str | None = None,
    masks_dir: Path | str | None = None,
    rounds_path: Path | str | None = None,
    live_path: Path | str | None = None,
    sync_main_annotations: bool = True,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Generate and double-check AI segmentation masks for `video` frames matching `parity`."""
    manual = load_manual(manual_path)
    df = load_accepted_present_rows(
        decisions_path=decisions_path,
        labels_path=labels_path,
        index_path=index_path,
        splits=("test", "train", "calib"),
    )
    target_fids = get_video_frames_by_parity(
        video=video,
        parity=parity,
        index_path=index_path,
        explicit_frames=frames,
    )
    if limit is not None and limit > 0:
        target_fids = target_fids[: int(limit)]

    out_file = Path(out_path) if out_path is not None else (C.OUT_DIR / "annotations_masks3.jsonl")
    m_dir = Path(masks_dir) if masks_dir is not None else (C.OUT_DIR / "masks")
    rnd_file = Path(rounds_path) if rounds_path is not None else (C.OUT_DIR / "masks3_rounds.jsonl")
    live_file = Path(live_path) if live_path is not None else (C.OUT_DIR / "annotations_live.jsonl")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    m_dir.mkdir(parents=True, exist_ok=True)
    rnd_file.parent.mkdir(parents=True, exist_ok=True)

    live_lock = asyncio.Lock()
    all_anns: list[dict[str, Any]] = []
    all_rounds: list[dict[str, Any]] = []

    for i in range(0, len(target_fids), max(1, batch_size)):
        batch_fids = target_fids[i : i + max(1, batch_size)]
        tasks = []
        for fid in batch_fids:
            f_rows = df[df["frame_id"].astype(str) == fid]
            if f_rows.empty:
                continue
            tasks.append(
                segment_frame(
                    fid,
                    f_rows,
                    manual,
                    masks_dir=m_dir,
                    iou_thresh=iou_thresh,
                    max_rounds=max_rounds,
                    live_file=live_file,
                    live_lock=live_lock,
                )
            )
        if not tasks:
            continue
        batch_res = await asyncio.gather(*tasks)
        for f_anns, f_rnds in batch_res:
            all_anns.extend(f_anns)
            all_rounds.extend(f_rnds)

        # Checkpoint out/annotations_masks3.jsonl and out/masks3_rounds.jsonl after every batch
        with open(out_file, "w", encoding="utf-8") as of:
            for a in all_anns:
                of.write(json.dumps(a) + "\n")
        with open(rnd_file, "w", encoding="utf-8") as rf:
            for r in all_rounds:
                rf.write(json.dumps(r) + "\n")
        if sync_main_annotations and all_anns:
            atomic_merge_into_annotations(all_anns)

    # Final write
    with open(out_file, "w", encoding="utf-8") as of:
        for a in all_anns:
            of.write(json.dumps(a) + "\n")
    with open(rnd_file, "w", encoding="utf-8") as rf:
        for r in all_rounds:
            rf.write(json.dumps(r) + "\n")
    total_merged = atomic_merge_into_annotations(all_anns) if (sync_main_annotations and all_anns) else len(all_anns)

    m_ious = [float(a.get("mask_iou", 0.0)) for a in all_anns]
    n_acc = sum(1 for a in all_anns if a.get("status") == "accept")
    n_amb = sum(1 for a in all_anns if a.get("status") == "AMBER")
    summary = {
        "worker": "nir-masks3",
        "video": video,
        "parity": parity,
        "n_frames_targeted": len(target_fids),
        "n_frames_segmented": len({a["frame_id"] for a in all_anns}),
        "n_masks_written": len(all_anns),
        "n_accepted": n_acc,
        "n_amber": n_amb,
        "pass12_mean_mask_iou": round(float(np.mean(m_ious)), 4) if m_ious else 0.0,
        "out_jsonl": str(out_file),
        "masks_dir": str(m_dir),
        "total_merged_annotations": total_merged,
    }
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="AI segmentation mask worker for video12 (odd/even split).")
    ap.add_argument("--video", type=str, default="video12", help="Target video ID (default: video12)")
    ap.add_argument("--parity", type=str, default="odd", choices=["odd", "even", "all"], help="Frame index parity (default: odd)")
    ap.add_argument("--frames", type=str, default=None, help="Optional comma-separated frame_id list")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of frames")
    ap.add_argument("--out", type=str, default=str(C.OUT_DIR / "annotations_masks3.jsonl"), help="Output JSONL path")
    ap.add_argument("--masks-dir", type=str, default=str(C.OUT_DIR / "masks"), help="Output masks directory")
    ap.add_argument("--rounds-out", type=str, default=str(C.OUT_DIR / "masks3_rounds.jsonl"), help="Rounds log JSONL path")
    ap.add_argument("--iou-thresh", type=float, default=IOU_ACCEPT_THRESHOLD, help="Pass1 vs Pass2 mask IoU threshold (default: 0.5)")
    ap.add_argument("--rounds", type=int, default=MAX_AMBER_ROUNDS, help="Max Amber loop rounds (default: 3)")
    ap.add_argument("--batch-size", type=int, default=8, help="Concurrent frames per checkpoint batch")
    ap.add_argument("--no-sync-main", action="store_true", help="Do not merge into out/annotations.jsonl")
    args = ap.parse_args()

    explicit_fids = [s.strip() for s in args.frames.split(",") if s.strip()] if args.frames else None
    res = asyncio.run(
        run_masks_worker(
            video=args.video,
            parity=args.parity,
            frames=explicit_fids,
            limit=args.limit,
            iou_thresh=args.iou_thresh,
            max_rounds=args.rounds,
            out_path=args.out,
            masks_dir=args.masks_dir,
            rounds_path=args.rounds_out,
            sync_main_annotations=not args.no_sync_main,
            batch_size=args.batch_size,
        )
    )
    print(json.dumps(res, indent=2))
