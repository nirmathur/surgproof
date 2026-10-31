"""Autonomous zero-human-label AI segmentation mask worker for SurgProof (nir-masks2 / nir-masks3).

Produces per-structure segmentation masks (`out/masks/<frame_id>_<label>.png`) and mask-enriched
localization records (`out/annotations_masks2.jsonl` / `out/annotations_masks3.jsonl`) for `video12`:
- Coordinate split on `video12` (sorted `frame_id` order):
  * `nir-masks2` (`--parity even`): indices 0, 2, 4, ..., 78 (40 frames)
  * `nir-masks3` (`--parity odd`):  indices 1, 3, 5, ..., 79 (40 frames)
- For each verifier-accepted present structure:
  1. Obtains Pass 1 (`gemini-3.8-flash`) and Pass 2 (`gemini-4-argon`) localizations via `localize.py`
     (`gem.call` cached in `cache.sqlite`).
  2. Generates Pass 1 and Pass 2 binary segmentation masks on the RGB frame (`segment_structure_in_box`,
     strictly unsupervised from RGB appearance + predicted box/tip/contour; NEVER reads reference masks).
  3. Double-checks binary mask agreement: `mask_iou(pass1_mask, pass2_mask) >= 0.50` and `box_iou >= 0.50`
     -> `accept` (round 0); else `AMBER` and runs up to 3 tighter-crop refinement rounds.
  4. Writes RGBA overlay PNGs to `out/masks/<frame_id>_<label>.png` (and `web/data/masks/` if present),
     writes `out/annotations_masks2.jsonl`, and atomically merges completed mask annotations into
     `out/annotations.jsonl` and `out/annotations_live.jsonl`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import config as C
import gem
from localize import (
    DEFAULT_PHASES,
    INSTRUMENT_KEYS,
    IOU_ACCEPT_THRESHOLD,
    MAX_AMBER_ROUNDS,
    load_accepted_present_rows,
    load_manual,
    localize_frame,
)
from verify_localize import (
    average_boxes,
    average_tips,
    box_iou,
    clamp_prob,
    evaluate_annotations,
    sanitize_box,
    sanitize_tip,
    snap_tip_to_instrument_pixels,
    verify_and_refine_item,
)

# Signature RGBA colours per surgical structure (matches promo/DESIGN.md & web/viewer.js palette)
LABEL_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "grasper": (59, 130, 246),       # #3B82F6 electric blue
    "l_hook": (6, 182, 212),         # #06B6D4 cyan
    "gallbladder": (16, 185, 129),   # #10B981 emerald green
    "liver": (168, 85, 247),         # #A855F7 violet
    "cystic_duct": (245, 158, 11),   # #F59E0B amber gold
    "blood": (239, 68, 68),          # #EF4444 crimson red
}


def binary_mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute pixel-level Intersection-over-Union (IoU) between two boolean masks."""
    if m1 is None or m2 is None or m1.shape != m2.shape:
        return 0.0
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    inter = int(np.logical_and(b1, b2).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(b1, b2).sum())
    return float(inter / union) if union > 0 else 0.0


def binary_mask_dice(m1: np.ndarray, m2: np.ndarray) -> float:
    """Compute Sorensen-Dice coefficient between two boolean masks."""
    if m1 is None or m2 is None or m1.shape != m2.shape:
        return 0.0
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    inter = int(np.logical_and(b1, b2).sum())
    if inter <= 0:
        return 0.0
    denom = int(b1.sum()) + int(b2.sum())
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def _smooth_binary_mask(mask_roi: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Fast pure-NumPy 3x3 majority filter to smooth jagged edges and fill tiny pinholes."""
    out = mask_roi.astype(bool)
    if out.shape[0] < 3 or out.shape[1] < 3:
        return out
    for _ in range(iterations):
        p = np.pad(out.astype(np.int32), 1, mode="edge")
        neigh_sum = (
            p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:]
            + p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
            + p[2:, 0:-2] + p[2:, 1:-1] + p[2:, 2:]
        )
        out = neigh_sum >= 5
    return out


def _point_to_segment_dist(
    yy: np.ndarray,
    xx: np.ndarray,
    y_a: float,
    x_a: float,
    y_b: float,
    x_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (perpendicular distance, projection parameter t in [0, 1]) from grid (yy, xx) to segment A->B."""
    dy = y_b - y_a
    dx = x_b - x_a
    seg_len_sq = float(dy * dy + dx * dx)
    if seg_len_sq < 1e-6:
        dist = np.sqrt((yy - y_a) ** 2 + (xx - x_a) ** 2)
        return dist, np.zeros_like(dist)
    t = ((yy - y_a) * dy + (xx - x_a) * dx) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)
    proj_y = y_a + t_clamped * dy
    proj_x = x_a + t_clamped * dx
    dist = np.sqrt((yy - proj_y) ** 2 + (xx - proj_x) ** 2)
    return dist, t_clamped


def segment_structure_in_box(
    rgb_arr: np.ndarray,
    box: list[int] | None,
    tip: list[int] | None,
    label: str,
    polygon: list[list[int]] | None = None,
) -> np.ndarray:
    """Unsupervised RGB + spatial contour segmentation inside normalized `box = [ymin, xmin, ymax, xmax]`.

    Never reads reference masks. Uses only:
    - RGB frame pixels (`rgb_arr` of shape `(H, W, 3)`),
    - Model-predicted normalized bounding box `box` (`0..1000`),
    - Model-predicted instrument tip `tip` (`[y, x]` in `0..1000`, if instrument),
    - Optional model-predicted polygon vertices `polygon` (`[[y, x], ...]` in `0..1000`).
    """
    h, w = rgb_arr.shape[:2]
    full_mask = np.zeros((h, w), dtype=bool)
    sbox = sanitize_box(box)
    if not sbox:
        return full_mask

    y0 = max(0, min(h - 1, int(round(sbox[0] * h / 1000.0))))
    x0 = max(0, min(w - 1, int(round(sbox[1] * w / 1000.0))))
    y1 = max(y0 + 1, min(h, int(round(sbox[2] * h / 1000.0))))
    x1 = max(x0 + 1, min(w, int(round(sbox[3] * w / 1000.0))))
    bh, bw = y1 - y0, x1 - x0
    if bh < 2 or bw < 2:
        return full_mask

    patch = rgb_arr[y0:y1, x0:x1].astype(np.float32)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    max_c = np.maximum(r, np.maximum(g, b))
    min_c = np.minimum(r, np.minimum(g, b))
    brightness = (r + g + b) / 3.0
    sat = (max_c - min_c) / np.maximum(max_c, 1.0)
    red_excess = r - np.maximum(g, b)

    # Exclude black endoscopic circular/octagonal viewport border
    fov_patch = (max_c >= 18.0) & (brightness >= 13.0)
    if not fov_patch.any():
        fov_patch = np.ones((bh, bw), dtype=bool)

    # Optional polygon prior if provided
    poly_patch: np.ndarray | None = None
    if polygon and isinstance(polygon, list) and len(polygon) >= 3:
        try:
            pil_poly = Image.new("L", (w, h), 0)
            pts_xy = [
                (
                    int(round(min(1000, max(0, float(pt[1]))) * (w - 1) / 1000.0)),
                    int(round(min(1000, max(0, float(pt[0]))) * (h - 1) / 1000.0)),
                )
                for pt in polygon
                if isinstance(pt, (list, tuple)) and len(pt) >= 2
            ]
            if len(pts_xy) >= 3:
                ImageDraw.Draw(pil_poly).polygon(pts_xy, fill=255)
                poly_arr = (np.array(pil_poly) > 0)[y0:y1, x0:x1]
                if poly_arr.any():
                    poly_patch = poly_arr
        except Exception:
            poly_patch = None

    yy, xx = np.mgrid[0:bh, 0:bw]
    ny = (yy + 0.5) / float(bh)
    nx = (xx + 0.5) / float(bw)

    if label in INSTRUMENT_KEYS:
        # Surgical instruments (grasper, l_hook): metallic silver/grey working jaws/hook + dark matte shaft.
        # In laparoscopy, biological tissue has high red_excess (R - max(G,B)), whereas tools have low red_excess.
        valid_red = red_excess[fov_patch]
        p45 = float(np.percentile(valid_red, 45)) if valid_red.size else 20.0
        p65 = float(np.percentile(valid_red, 65)) if valid_red.size else 30.0
        red_thresh = min(34.0, max(16.0, p50 := 0.5 * (p45 + p65)))

        low_red_mask = (red_excess <= red_thresh) | ((brightness <= 70.0) & (sat <= 0.35)) | (sat <= 0.22)
        inst_appearance = fov_patch & low_red_mask

        # Determine instrument axis from box entry border to distal tip
        st = sanitize_tip(tip, box=sbox, is_instrument=True)
        if st:
            tip_y_local = float(np.clip((st[0] * h / 1000.0) - y0, 0.0, bh - 1.0))
            tip_x_local = float(np.clip((st[1] * w / 1000.0) - x0, 0.0, bw - 1.0))
        else:
            tip_y_local, tip_x_local = float(bh * 0.5), float(bw * 0.5)

        # Candidate entry points along the 4 corners and 4 mid-edges of the bounding box
        candidates = [
            (0.0, 0.0),
            (0.0, float(bw - 1)),
            (float(bh - 1), 0.0),
            (float(bh - 1), float(bw - 1)),
            (0.0, float(bw - 1) * 0.5),
            (float(bh - 1), float(bw - 1) * 0.5),
            (float(bh - 1) * 0.5, 0.0),
            (float(bh - 1) * 0.5, float(bw - 1)),
        ]
        diag = float(np.hypot(bh, bw))
        shaft_radius = max(6.0, 0.19 * min(bh, bw) + 0.05 * diag)

        best_entry = candidates[0]
        best_score = -1e9
        for cy, cx in candidates:
            seg_d = float(np.hypot(cy - tip_y_local, cx - tip_x_local))
            if seg_d < 0.22 * diag:
                continue
            dist_grid, _ = _point_to_segment_dist(yy, xx, cy, cx, tip_y_local, tip_x_local)
            corridor = (dist_grid <= shaft_radius) & fov_patch
            if not corridor.any():
                continue
            # Prefer the entry corner whose corridor has the strongest instrument appearance density
            density = float(inst_appearance[corridor].mean())
            score = density + 0.12 * (seg_d / max(diag, 1.0))
            if score > best_score:
                best_score = score
                best_entry = (cy, cx)

        dist_axis, t_proj = _point_to_segment_dist(
            yy, xx, best_entry[0], best_entry[1], tip_y_local, tip_x_local
        )
        # Tapered corridor: slightly wider near the shaft entry and jaws, tight around the axis
        width_profile = shaft_radius * (1.15 - 0.25 * t_proj)
        corridor_mask = dist_axis <= width_profile
        tip_head_mask = np.sqrt((yy - tip_y_local) ** 2 + (xx - tip_x_local) ** 2) <= (1.25 * shaft_radius)
        spatial_prior = corridor_mask | tip_head_mask
        if poly_patch is not None:
            spatial_prior = spatial_prior | poly_patch

        # Combine appearance and spatial corridor:
        # Inside the tight core of the shaft corridor (`dist_axis <= 0.55 * width_profile`), tolerate specular/blood glare;
        # further out within the box, require strong non-reddish instrument appearance.
        core_shaft = (dist_axis <= 0.58 * width_profile) & (red_excess <= max(red_thresh + 14.0, 38.0))
        outer_inst = (dist_axis <= 1.45 * width_profile) & inst_appearance
        roi_mask = fov_patch & ((spatial_prior & inst_appearance) | core_shaft | outer_inst)

        if int(roi_mask.sum()) < max(12, int(0.06 * bh * bw)):
            roi_mask = fov_patch & (dist_axis <= 0.75 * width_profile)
        roi_mask = _smooth_binary_mask(roi_mask, iterations=2)
    else:
        # Anatomical structures: liver, gallbladder, cystic_duct, blood
        # 1. Exclude obvious foreground metallic/matte-black instrument shafts crossing over the organ
        fg_instrument = (
            fov_patch
            & (red_excess < 7.0)
            & ((brightness < 52.0) | ((sat < 0.16) & (brightness > 65.0)))
        )
        tissue_valid = fov_patch & (~fg_instrument)
        if not tissue_valid.any():
            tissue_valid = fov_patch

        # 2. Sample core organ colour from the central 55% of the box (or polygon interior)
        core_region = (ny >= 0.22) & (ny <= 0.78) & (nx >= 0.22) & (nx <= 0.78) & tissue_valid
        if poly_patch is not None and (poly_patch & tissue_valid).any():
            core_region = poly_patch & tissue_valid
        if not core_region.any():
            core_region = tissue_valid

        r_med = float(np.median(r[core_region]))
        g_med = float(np.median(g[core_region]))
        b_med = float(np.median(b[core_region]))
        re_med = float(np.median(red_excess[core_region]))

        # Colour distance to core organ appearance
        color_dist = np.sqrt(
            0.36 * (r - r_med) ** 2
            + 0.34 * (g - g_med) ** 2
            + 0.30 * (b - b_med) ** 2
            + 0.25 * (red_excess - re_med) ** 2
        )
        valid_dists = color_dist[tissue_valid]
        dist_thr = float(np.percentile(valid_dists, 80)) if valid_dists.size else 42.0
        dist_thr = min(62.0, max(26.0, dist_thr))

        # Smooth super-ellipse spatial envelope inside the bounding box so box corners with adjacent organs are trimmed
        super_ellipse = ((ny - 0.5) / 0.53) ** 4 + ((nx - 0.5) / 0.53) ** 4
        if poly_patch is not None:
            spatial_env = poly_patch | (super_ellipse <= 0.85)
        else:
            spatial_env = super_ellipse <= 1.08

        # Near the center of the box, allow broader colour variation (glare/shadow on the organ);
        # near the box boundary, require closer match to the organ's core colour.
        radial_weight = np.clip(super_ellipse, 0.0, 1.4)
        adaptive_thr = dist_thr * (1.22 - 0.38 * radial_weight)
        roi_mask = tissue_valid & spatial_env & (color_dist <= adaptive_thr)

        if int(roi_mask.sum()) < max(16, int(0.18 * bh * bw)):
            roi_mask = tissue_valid & (super_ellipse <= 0.95)
        roi_mask = _smooth_binary_mask(roi_mask, iterations=2)

    full_mask[y0:y1, x0:x1] = roi_mask
    return full_mask


def extract_mask_polygon(mask_bin: np.ndarray, num_points: int = 16) -> list[list[int]]:
    """Extract a clean clockwise normalized `[y, x]` (0..1000) contour polygon around `mask_bin`."""
    h, w = mask_bin.shape[:2]
    ys, xs = np.where(mask_bin)
    if len(ys) < 3:
        return []
    cy = float(ys.mean())
    cx = float(xs.mean())

    # Boundary pixels of mask_bin: foreground pixels with at least one 4-connected background neighbor
    p = np.pad(mask_bin, 1, mode="constant", constant_values=False)
    interior = p[1:-1, 1:-1] & p[0:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, 0:-2] & p[1:-1, 2:]
    boundary = mask_bin & (~interior)
    by, bx = np.where(boundary)
    if len(by) < 3:
        by, bx = ys, xs

    angles = np.arctan2(by - cy, bx - cx)
    radii = np.hypot(by - cy, bx - cx)
    bins = np.linspace(-np.pi, np.pi, num_points + 1)
    poly: list[list[int]] = []
    for i in range(num_points):
        in_sector = (angles >= bins[i]) & (angles < bins[i + 1])
        if in_sector.any():
            sector_idx = np.where(in_sector)[0]
            # Pick the outer contour point (90th percentile radius in that angular sector)
            r_sec = radii[sector_idx]
            target_r = np.percentile(r_sec, 90)
            best_local = sector_idx[int(np.argmin(np.abs(r_sec - target_r)))]
            py, px = float(by[best_local]), float(bx[best_local])
            ny = int(round(np.clip(py * 1000.0 / max(h - 1, 1), 0, 1000)))
            nx = int(round(np.clip(px * 1000.0 / max(w - 1, 1), 0, 1000)))
            pt = [ny, nx]
            if not poly or poly[-1] != pt:
                poly.append(pt)
    return poly


def save_rgba_mask_png(
    mask_bin: np.ndarray,
    label: str,
    out_png: Path,
    extra_dirs: list[Path] | None = None,
) -> None:
    """Save `mask_bin` `(H, W)` as an RGBA PNG with transparent background `(0, 0, 0, 0)`,
    semi-transparent structure fill (`alpha = 155`), and crisp 2px contour border (`alpha = 245`).
    """
    h, w = mask_bin.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if mask_bin.any():
        r_col, g_col, b_col = LABEL_COLORS_RGB.get(label, (59, 130, 246))
        p = np.pad(mask_bin, 1, mode="constant", constant_values=False)
        interior = p[1:-1, 1:-1] & p[0:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, 0:-2] & p[1:-1, 2:]
        border = mask_bin & (~interior)

        rgba[mask_bin, 0] = r_col
        rgba[mask_bin, 1] = g_col
        rgba[mask_bin, 2] = b_col
        rgba[mask_bin, 3] = 155
        rgba[border, 3] = 245

    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, mode="RGBA").save(out_png, format="PNG", optimize=True)
    for d in extra_dirs or []:
        if d.parent.exists():
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out_png, d / out_png.name)


async def segment_and_verify_frame(
    frame_id: str,
    frame_rows: pd.DataFrame,
    manual: dict[str, Any],
    *,
    existing_frame_anns: dict[str, dict[str, Any]] | None = None,
    masks_dir: Path,
    extra_mask_dirs: list[Path] | None = None,
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    call_fn=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Produce double-checked AI segmentation masks + boxes + tips + phase for all accepted structures in `frame_id`."""
    caller = call_fn or gem.call
    first_row = frame_rows.iloc[0]
    video = str(first_row.get("video", "video12"))
    split = str(first_row.get("split", "test"))
    img_path = str(first_row.get("frame_path", ""))
    present_keys = list(dict.fromkeys(frame_rows["key"].astype(str).tolist()))

    # 1. Reuse existing localized annotations for this frame if all present_keys are already localized,
    #    otherwise run `localize_frame` (Pass 1 Flash + Pass 2 Argon + tighter-crop Amber loop).
    frame_rounds: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    if existing_frame_anns and all(k in existing_frame_anns for k in present_keys):
        by_label = {k: dict(existing_frame_anns[k]) for k in present_keys}
    else:
        loc_anns, loc_rounds = await localize_frame(
            frame_id,
            frame_rows,
            manual,
            iou_thresh=iou_thresh,
            max_rounds=max_rounds,
            live_file=None,
            call_fn=caller,
        )
        frame_rounds.extend(loc_rounds)
        for a in loc_anns:
            by_label[str(a["label"])] = dict(a)

    # Load RGB image once for pixel segmentation
    rgb_arr = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

    out_anns: list[dict[str, Any]] = []
    for label in present_keys:
        ann = by_label.get(label)
        if not ann:
            continue
        is_inst = label in INSTRUMENT_KEYS
        p1 = dict(ann.get("pass1") or {})
        p2 = dict(ann.get("pass2") or {})
        b1 = sanitize_box(p1.get("box") or ann.get("box") or [])
        b2 = sanitize_box(p2.get("box") or ann.get("box") or [])
        t1 = sanitize_tip(p1.get("tip"), box=b1, is_instrument=is_inst)
        t2 = sanitize_tip(p2.get("tip"), box=b2, is_instrument=is_inst)
        c1 = clamp_prob(p1.get("conf", ann.get("conf", 0.90)), 0.90)
        c2 = clamp_prob(p2.get("conf", ann.get("conf", 0.90)), 0.90)

        # Pass 1 & Pass 2 segmentation masks
        m1 = segment_structure_in_box(rgb_arr, b1, t1, label)
        m2 = segment_structure_in_box(rgb_arr, b2, t2, label)
        m_iou0 = binary_mask_iou(m1, m2)
        b_iou0 = box_iou(b1, b2)

        final_box = sanitize_box(ann.get("box") or b1 or b2)
        final_tip = sanitize_tip(ann.get("tip"), box=final_box, is_instrument=is_inst)
        final_conf = clamp_prob(ann.get("conf", 0.5 * (c1 + c2)), 0.90)
        final_round = int(ann.get("round", 0))
        final_status = str(ann.get("status", "accept"))
        final_mask_iou = m_iou0

        # If Pass 1 vs Pass 2 mask IoU < 0.50 and box refinement has not yet run, escalate to tighter-crop loop
        if m_iou0 < iou_thresh and final_round == 0:
            resolved, extra_rounds = await verify_and_refine_item(
                frame_id=frame_id,
                video=video,
                split=split,
                img_path=img_path,
                label=label,
                pass1={"box": b1, "tip": t1, "conf": c1, "phase": p1.get("phase")},
                pass2={"box": b2, "tip": t2, "conf": c2, "iou": round(b_iou0, 4), "phase": p2.get("phase")},
                manual=manual,
                iou_thresh=iou_thresh,
                max_rounds=max_rounds,
                call_fn=caller,
            )
            frame_rounds.extend(extra_rounds)
            final_box = sanitize_box(resolved.get("box") or final_box)
            final_tip = sanitize_tip(resolved.get("tip"), box=final_box, is_instrument=is_inst)
            final_conf = clamp_prob(resolved.get("conf", final_conf), final_conf)
            final_round = int(resolved.get("round", 1))
            final_status = str(resolved.get("status", "AMBER"))

        # Compute refined consensus mask and mask IoU at the resolved box
        m_cons = segment_structure_in_box(rgb_arr, final_box, final_tip, label)
        if final_round > 0:
            # Check final round's pass1_box vs pass2_box masks if rounds ran
            lbl_rounds = [r for r in frame_rounds if r.get("label") == label and int(r.get("round", 0)) > 0]
            if lbl_rounds:
                last_r = lbl_rounds[-1]
                rm1 = segment_structure_in_box(rgb_arr, last_r.get("pass1_box"), final_tip, label)
                rm2 = segment_structure_in_box(rgb_arr, last_r.get("pass2_box"), final_tip, label)
                final_mask_iou = binary_mask_iou(rm1, rm2)
                if final_mask_iou >= iou_thresh and box_iou(last_r.get("pass1_box"), last_r.get("pass2_box")) >= iou_thresh:
                    final_status = "accept"
                else:
                    final_status = "AMBER"
                vote = rm1.astype(np.int32) + rm2.astype(np.int32) + m_cons.astype(np.int32)
                final_mask = vote >= 2
            else:
                final_mask = m_cons
                final_mask_iou = max(m_iou0, binary_mask_iou(m1, m_cons), binary_mask_iou(m2, m_cons))
                if final_mask_iou < iou_thresh:
                    final_status = "AMBER"
        else:
            vote = m1.astype(np.int32) + m2.astype(np.int32) + m_cons.astype(np.int32)
            final_mask = vote >= 2
            if not final_mask.any():
                final_mask = m_cons

        if not final_mask.any():
            final_mask = m_cons

        # Save RGBA PNG to out/masks/<frame_id>_<label>.png
        rel_mask = f"out/masks/{frame_id}_{label}.png"
        png_file = masks_dir / f"{frame_id}_{label}.png"
        save_rgba_mask_png(final_mask, label, png_file, extra_dirs=extra_mask_dirs)

        poly_pts = extract_mask_polygon(final_mask, num_points=16)
        p1["mask_iou"] = round(m_iou0, 4)
        p2["mask_iou"] = round(m_iou0, 4)

        ann_out = {
            "frame_id": frame_id,
            "video": video,
            "split": split,
            "img_path": img_path,
            "label": label,
            "box": final_box,
            "tip": final_tip,
            "mask": rel_mask,
            "mask_url": "/" + rel_mask,
            "mask_iou": round(final_mask_iou, 4),
            "polygon": poly_pts,
            "phase": str(ann.get("phase") or DEFAULT_PHASES[1]),
            "phase_conf": clamp_prob(ann.get("phase_conf", 0.92), 0.92),
            "conf": final_conf,
            "pass1": p1,
            "pass2": p2,
            "round": final_round,
            "status": final_status,
            "ts": round(time.time(), 3),
        }
        out_anns.append(ann_out)

    gem.trace(
        "masks",
        f"{frame_id} ({video}): segmented {len(out_anns)} structures -> out/masks/{frame_id}_*.png",
    )
    return out_anns, frame_rounds


def select_video_frames(
    index_path: Path,
    video: str = "video12",
    parity: str = "even",
    explicit_frames: list[str] | None = None,
) -> list[str]:
    """Return sorted `frame_id` list for `video` filtered by `parity` ('even', 'odd', or 'all')."""
    if explicit_frames:
        return list(dict.fromkeys(explicit_frames))
    if not index_path.is_file():
        return []
    df = pd.read_csv(index_path)
    safe_cols = [c for c in ("frame_id", "video") if c in df.columns]
    sub = df[safe_cols].drop_duplicates(subset=["frame_id"])
    v_fids = sorted(sub[sub["video"].astype(str) == video]["frame_id"].astype(str).tolist())
    if parity == "even":
        return [f for i, f in enumerate(v_fids) if i % 2 == 0]
    if parity == "odd":
        return [f for i, f in enumerate(v_fids) if i % 2 == 1]
    return v_fids


def sync_into_shared_annotations(
    worker_anns: list[dict[str, Any]],
    *,
    shared_ann_file: Path,
    shared_live_file: Path,
    other_worker_files: list[Path] | None = None,
) -> int:
    """Atomically merge `worker_anns` (and any `other_worker_files` such as `annotations_masks3.jsonl`)
    into `out/annotations.jsonl` and append new mask rows to `out/annotations_live.jsonl`.
    Preserves all existing non-video12 annotations and enriches matching `(frame_id, label)` rows with `mask`.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if shared_ann_file.is_file() and shared_ann_file.stat().st_size > 0:
        for line in shared_ann_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fid = str(rec.get("frame_id", ""))
                lbl = str(rec.get("label", ""))
                if fid and lbl:
                    merged[(fid, lbl)] = rec
            except Exception:
                continue

    for wf in other_worker_files or []:
        if wf.is_file() and wf.stat().st_size > 0:
            for line in wf.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    fid = str(rec.get("frame_id", ""))
                    lbl = str(rec.get("label", ""))
                    if fid and lbl:
                        prev = merged.get((fid, lbl), {})
                        merged[(fid, lbl)] = {**prev, **rec}
                except Exception:
                    continue

    for rec in worker_anns:
        fid = str(rec.get("frame_id", ""))
        lbl = str(rec.get("label", ""))
        if fid and lbl:
            prev = merged.get((fid, lbl), {})
            merged[(fid, lbl)] = {**prev, **rec}

    tmp_file = shared_ann_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as tf:
        for rec in merged.values():
            tf.write(json.dumps(rec) + "\n")
    tmp_file.replace(shared_ann_file)

    # Also ensure live_file has the mask-enriched records
    existing_live_with_mask: set[tuple[str, str]] = set()
    if shared_live_file.is_file() and shared_live_file.stat().st_size > 0:
        for line in shared_live_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("mask"):
                    existing_live_with_mask.add((str(r.get("frame_id", "")), str(r.get("label", ""))))
            except Exception:
                continue

    with open(shared_live_file, "a", encoding="utf-8") as lf:
        for rec in worker_anns:
            key = (str(rec.get("frame_id", "")), str(rec.get("label", "")))
            if key not in existing_live_with_mask:
                lf.write(json.dumps(rec) + "\n")
                existing_live_with_mask.add(key)

    return len(merged)


async def run_masks_worker(
    *,
    video: str = "video12",
    parity: str = "even",
    explicit_frames: list[str] | None = None,
    limit: int | None = None,
    out_path: Path | str | None = None,
    masks_dir: Path | str | None = None,
    shared_ann_path: Path | str | None = None,
    shared_live_path: Path | str | None = None,
    rounds_path: Path | str | None = None,
    index_path: Path | str | None = None,
    decisions_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    manual_path: Path | str | None = None,
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    batch_size: int = 8,
    resume: bool = True,
    sync_shared: bool = True,
    call_fn=None,
) -> dict[str, Any]:
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")
    default_out_name = "annotations_masks3.jsonl" if parity == "odd" else "annotations_masks2.jsonl"
    worker_out_file = Path(out_path) if out_path is not None else (C.OUT_DIR / default_out_name)
    m_dir = Path(masks_dir) if masks_dir is not None else (C.OUT_DIR / "masks")
    shared_ann_file = Path(shared_ann_path) if shared_ann_path is not None else (C.OUT_DIR / "annotations.jsonl")
    shared_live_file = Path(shared_live_path) if shared_live_path is not None else (C.OUT_DIR / "annotations_live.jsonl")
    rnd_file = Path(rounds_path) if rounds_path is not None else (C.OUT_DIR / "localize_rounds.jsonl")

    m_dir.mkdir(parents=True, exist_ok=True)
    worker_out_file.parent.mkdir(parents=True, exist_ok=True)
    extra_dirs = [
        C.ROOT / "web" / "data" / "masks",
        Path.home() / "surgatlas-lanes" / "demo" / "web" / "data" / "masks",
        Path.home() / "surgatlas-lanes" / "student" / "web" / "data" / "masks",
    ]

    manual = load_manual(manual_path)
    df = load_accepted_present_rows(
        decisions_path=decisions_path,
        labels_path=labels_path,
        index_path=idx_file,
        splits=("test",),
    )

    target_fids = select_video_frames(idx_file, video=video, parity=parity, explicit_frames=explicit_frames)
    if limit is not None and limit > 0:
        target_fids = target_fids[: int(limit)]

    # Load any existing localized records from shared_ann_file and worker_out_file
    existing_shared_by_frame: dict[str, dict[str, dict[str, Any]]] = {}
    for src_file in (shared_ann_file, worker_out_file):
        if src_file.is_file() and src_file.stat().st_size > 0:
            for line in src_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    fid = str(rec.get("frame_id", ""))
                    lbl = str(rec.get("label", ""))
                    if fid and lbl:
                        existing_shared_by_frame.setdefault(fid, {})[lbl] = rec
                except Exception:
                    continue

    worker_anns_map: dict[tuple[str, str], dict[str, Any]] = {}
    done_fids: set[str] = set()
    if resume and worker_out_file.is_file() and worker_out_file.stat().st_size > 0:
        for line in worker_out_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fid = str(rec.get("frame_id", ""))
                lbl = str(rec.get("label", ""))
                m_rel = str(rec.get("mask", ""))
                if fid and lbl and m_rel and (C.ROOT / m_rel).is_file():
                    worker_anns_map[(fid, lbl)] = rec
            except Exception:
                continue
        for fid in target_fids:
            req_keys = set(df[df["frame_id"].astype(str) == fid]["key"].astype(str).unique())
            if req_keys and all((fid, k) in worker_anns_map for k in req_keys):
                done_fids.add(fid)

    todo_fids = [fid for fid in target_fids if fid not in done_fids]
    new_rounds: list[dict[str, Any]] = []

    for i in range(0, len(todo_fids), max(1, batch_size)):
        batch_fids = todo_fids[i : i + max(1, batch_size)]
        tasks = [
            segment_and_verify_frame(
                fid,
                df[df["frame_id"].astype(str) == fid],
                manual,
                existing_frame_anns=existing_shared_by_frame.get(fid),
                masks_dir=m_dir,
                extra_mask_dirs=extra_dirs,
                iou_thresh=iou_thresh,
                max_rounds=max_rounds,
                call_fn=call_fn,
            )
            for fid in batch_fids
            if not df[df["frame_id"].astype(str) == fid].empty
        ]
        if not tasks:
            continue
        batch_res = await asyncio.gather(*tasks)
        for f_anns, f_rnds in batch_res:
            for a in f_anns:
                worker_anns_map[(str(a["frame_id"]), str(a["label"]))] = a
            new_rounds.extend(f_rnds)

        # Checkpoint worker JSONL
        worker_list = list(worker_anns_map.values())
        with open(worker_out_file, "w", encoding="utf-8") as wf:
            for a in worker_list:
                wf.write(json.dumps(a) + "\n")

        if new_rounds and rnd_file is not None:
            with open(rnd_file, "a", encoding="utf-8") as rf:
                for r in new_rounds:
                    rf.write(json.dumps(r) + "\n")
            new_rounds.clear()

        if sync_shared:
            other_files = [
                C.OUT_DIR / "annotations_masks2.jsonl",
                C.OUT_DIR / "annotations_masks3.jsonl",
            ]
            sync_into_shared_annotations(
                worker_list,
                shared_ann_file=shared_ann_file,
                shared_live_file=shared_live_file,
                other_worker_files=other_files,
            )

    worker_list = list(worker_anns_map.values())
    with open(worker_out_file, "w", encoding="utf-8") as wf:
        for a in worker_list:
            wf.write(json.dumps(a) + "\n")

    total_shared = len(worker_list)
    if sync_shared:
        other_files = [
            C.OUT_DIR / "annotations_masks2.jsonl",
            C.OUT_DIR / "annotations_masks3.jsonl",
        ]
        total_shared = sync_into_shared_annotations(
            worker_list,
            shared_ann_file=shared_ann_file,
            shared_live_file=shared_live_file,
            other_worker_files=other_files,
        )

    mask_ious = [float(a.get("mask_iou", 0.0)) for a in worker_list]
    n_accepted = sum(1 for a in worker_list if a.get("status") == "accept")
    n_amber = sum(1 for a in worker_list if a.get("status") == "AMBER")
    return {
        "video": video,
        "parity": parity,
        "n_target_frames": len(target_fids),
        "n_completed_frames": len({str(a["frame_id"]) for a in worker_list}),
        "n_annotations": len(worker_list),
        "n_accepted": n_accepted,
        "n_amber": n_amber,
        "mean_pass12_mask_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else 0.0,
        "out_file": str(worker_out_file),
        "masks_dir": str(m_dir),
        "total_shared_annotations": total_shared,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate double-checked AI segmentation masks for video12.")
    ap.add_argument("--video", type=str, default="video12", help="Target video ID (default: video12)")
    ap.add_argument("--parity", choices=["even", "odd", "all"], default="even", help="Frame index parity within video")
    ap.add_argument("--frames", type=str, default=None, help="Optional comma-separated explicit frame_ids")
    ap.add_argument("--limit", type=int, default=None, help="Optional frame limit")
    ap.add_argument("--out", type=str, default=None, help="Output JSONL path (default: out/annotations_masks2.jsonl)")
    ap.add_argument("--masks-dir", type=str, default=None, help="Output masks directory (default: out/masks)")
    ap.add_argument("--batch-size", type=int, default=8, help="Concurrent frames per batch")
    ap.add_argument("--no-resume", action="store_true", help="Recompute all target frames from scratch")
    ap.add_argument("--no-sync-shared", action="store_true", help="Do not merge into out/annotations.jsonl")
    args = ap.parse_args()

    explicit = [f.strip() for f in args.frames.split(",") if f.strip()] if args.frames else None
    summary = asyncio.run(
        run_masks_worker(
            video=args.video,
            parity=args.parity,
            explicit_frames=explicit,
            limit=args.limit,
            out_path=args.out,
            masks_dir=args.masks_dir,
            batch_size=args.batch_size,
            resume=not args.no_resume,
            sync_shared=not args.no_sync_shared,
        )
    )
    print(json.dumps(summary, indent=2))
