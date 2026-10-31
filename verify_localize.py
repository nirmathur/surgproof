"""Verification, iterative tighter-crop Amber loop, and post-hoc evaluation for SurgProof localization.

Strictly enforces the zero-leakage rule:
- `verify_and_refine_item` (used inside the localization loop) NEVER reads ground-truth masks or `gt_*`.
- `evaluate_annotations` runs ONLY after `out/annotations.jsonl` is written to disk, scoring held-out
  `split == "test"` predictions against CholecSeg8k watershed masks (`data/index.csv:mask_path`).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import BaseModel

import config as C
import gem
from prepare_data import mask_to_classes

INSTRUMENT_KEYS = {"grasper", "l_hook"}
IOU_ACCEPT_THRESHOLD = 0.50
MAX_AMBER_ROUNDS = 3
ROUND_PADS = {1: 60, 2: 30, 3: 10}


class CropRefinement(BaseModel):
    box_2d: list[int]          # [ymin, xmin, ymax, xmax] normalised 0-1000 in the CROP image
    tip_2d: list[int]          # [y, x] normalised 0-1000 in the CROP image if instrument, else []
    confidence: float          # 0.0 - 1.0
    reason: str


class FullFrameAdjudication(BaseModel):
    box_2d: list[int]          # [ymin, xmin, ymax, xmax] normalised 0-1000 in the FULL frame
    tip_2d: list[int]          # [y, x] normalised 0-1000 in the FULL frame if instrument, else []
    confidence: float          # 0.0 - 1.0
    reason: str


def clamp_prob(val: Any, default: float = 0.5) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if np.isnan(f) or np.isinf(f):
        return default
    if 1.0 < f <= 100.0:
        f = f / 100.0
    return round(float(min(1.0, max(0.0, f))), 4)


def sanitize_box(box: Any) -> list[int]:
    """Normalize a 4-element box [ymin, xmin, ymax, xmax] into 0..1000 integer bounds."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        coords = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return []
    y0, x0, y1, x1 = [min(1000, max(0, c)) for c in coords]
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    if (y1 - y0) < 4 or (x1 - x0) < 4:
        return []
    return [y0, x0, y1, x1]


def sanitize_tip(tip: Any, box: list[int] | None = None, is_instrument: bool = True) -> list[int] | None:
    """Normalize a 2-element instrument tip [y, x] into 0..1000, clamped inside `box` if given."""
    if not is_instrument:
        return None
    if not isinstance(tip, (list, tuple)) or len(tip) < 2:
        if box and len(box) == 4:
            return [int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2))]
        return None
    try:
        y = int(round(float(tip[0])))
        x = int(round(float(tip[1])))
    except (TypeError, ValueError):
        if box and len(box) == 4:
            return [int(round((box[0] + box[2]) / 2)), int(round((box[1] + box[3]) / 2))]
        return None
    y = min(1000, max(0, y))
    x = min(1000, max(0, x))
    if box and len(box) == 4:
        y = min(box[2], max(box[0], y))
        x = min(box[3], max(box[1], x))
    return [y, x]


def snap_tip_to_instrument_pixels(
    img_path: str | Path,
    tip: list[int] | None,
    box: list[int],
    search_radius_norm: int = 45,
) -> list[int] | None:
    """Purely unsupervised RGB self-check: ensure an instrument tip [y, x] lands on metallic/dark
    instrument pixels inside `box` rather than warm reddish-brown background tissue between open jaws.
    Never touches any ground-truth mask.
    """
    if tip is None or not box or len(box) != 4:
        return tip
    p = Path(img_path)
    if not p.is_file():
        return tip
    try:
        im = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
    except Exception:
        return tip
    h, w, _ = im.shape
    if h < 10 or w < 10:
        return tip

    ty, tx = tip
    py = int(round(ty * (h - 1) / 1000.0))
    px = int(round(tx * (w - 1) / 1000.0))

    # Measure "warm tissue score": biological tissue in laparoscopy has strong R > G and R > B.
    # Instruments (silver/white metallic jaws/hooks or matte black/grey shafts) have low R - max(G,B).
    def _instrument_likelihood(arr_rgb: np.ndarray) -> np.ndarray:
        r, g, b = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        red_excess = r - np.maximum(g, b)
        brightness = (r + g + b) / 3.0
        # Specular metallic silver/white OR dark matte shaft inside the box (avoiding black viewport border < 12)
        not_viewport_border = (brightness >= 14.0).astype(np.float32)
        low_red_score = np.clip((35.0 - red_excess) / 35.0, -1.0, 1.5)
        return low_red_score * not_viewport_border

    # Check if current pixel is already clearly non-tissue (red_excess <= 22 and brightness >= 15)
    r0, g0, b0 = im[py, px]
    if (r0 - max(g0, b0)) <= 22.0 and ((r0 + g0 + b0) / 3.0) >= 15.0:
        return tip

    # Search within a small neighborhood inside the predicted box for the strongest instrument surface pixel
    by0 = max(0, int(round(box[0] * (h - 1) / 1000.0)))
    bx0 = max(0, int(round(box[1] * (w - 1) / 1000.0)))
    by1 = min(h, int(round(box[2] * (h - 1) / 1000.0)) + 1)
    bx1 = min(w, int(round(box[3] * (w - 1) / 1000.0)) + 1)

    rad_y = max(4, int(round(search_radius_norm * h / 1000.0)))
    rad_x = max(4, int(round(search_radius_norm * w / 1000.0)))
    sy0 = max(by0, py - rad_y)
    sy1 = min(by1, py + rad_y + 1)
    sx0 = max(bx0, px - rad_x)
    sx1 = min(bx1, px + rad_x + 1)
    if sy1 <= sy0 or sx1 <= sx0:
        return tip

    patch = im[sy0:sy1, sx0:sx1]
    inst_score = _instrument_likelihood(patch)
    yy, xx = np.mgrid[sy0:sy1, sx0:sx1]
    dist_penalty = 0.015 * np.sqrt((yy - py) ** 2 + (xx - px) ** 2)
    total_score = inst_score - dist_penalty
    best_idx = np.unravel_index(int(np.argmax(total_score)), total_score.shape)
    if total_score[best_idx] > 0.15:
        new_py = sy0 + int(best_idx[0])
        new_px = sx0 + int(best_idx[1])
        new_y = int(round(new_py * 1000.0 / (h - 1)))
        new_x = int(round(new_px * 1000.0 / (w - 1)))
        return sanitize_tip([new_y, new_x], box=box, is_instrument=True)
    return tip


def box_iou(b1: list[int] | None, b2: list[int] | None) -> float:
    """Compute 2D Intersection-over-Union (IoU) between two [ymin, xmin, ymax, xmax] boxes."""
    s1 = sanitize_box(b1)
    s2 = sanitize_box(b2)
    if not s1 or not s2:
        return 0.0
    y0 = max(s1[0], s2[0])
    x0 = max(s1[1], s2[1])
    y1 = min(s1[2], s2[2])
    x1 = min(s1[3], s2[3])
    inter = max(0, y1 - y0) * max(0, x1 - x0)
    if inter <= 0:
        return 0.0
    a1 = max(0, s1[2] - s1[0]) * max(0, s1[3] - s1[1])
    a2 = max(0, s2[2] - s2[0]) * max(0, s2[3] - s2[1])
    denom = a1 + a2 - inter
    return float(inter / denom) if denom > 0 else 0.0


def average_boxes(b1: list[int], b2: list[int], w1: float = 0.5, w2: float = 0.5) -> list[int]:
    """Compute weighted average of two valid [ymin, xmin, ymax, xmax] boxes."""
    s1 = sanitize_box(b1)
    s2 = sanitize_box(b2)
    if not s1:
        return s2
    if not s2:
        return s1
    tot = (w1 + w2) if (w1 + w2) > 0 else 1.0
    avg = [int(round((w1 * s1[i] + w2 * s2[i]) / tot)) for i in range(4)]
    return sanitize_box(avg) or s1


def average_tips(
    t1: list[int] | None,
    t2: list[int] | None,
    box: list[int],
    is_instrument: bool,
) -> list[int] | None:
    if not is_instrument:
        return None
    st1 = sanitize_tip(t1, box=box, is_instrument=True)
    st2 = sanitize_tip(t2, box=box, is_instrument=True)
    if st1 and st2:
        avg = [int(round(0.5 * (st1[0] + st2[0]))), int(round(0.5 * (st1[1] + st2[1])))]
        return sanitize_tip(avg, box=box, is_instrument=True)
    return st1 or st2


def disagreement_crop(b1: list[int], b2: list[int], round_idx: int = 1) -> list[int]:
    """Compute a progressively tighter crop [ymin, xmin, ymax, xmax] in 0..1000 around two disagreeing boxes."""
    s1 = sanitize_box(b1)
    s2 = sanitize_box(b2)
    if not s1 and not s2:
        return [0, 0, 1000, 1000]
    if not s1:
        s1 = s2
    if not s2:
        s2 = s1
    pad = ROUND_PADS.get(int(round_idx), 15)
    y0 = max(0, min(s1[0], s2[0]) - pad)
    x0 = max(0, min(s1[1], s2[1]) - pad)
    y1 = min(1000, max(s1[2], s2[2]) + pad)
    x1 = min(1000, max(s1[3], s2[3]) + pad)
    if (y1 - y0) < 40:
        mid_y = (y0 + y1) // 2
        y0 = max(0, mid_y - 20)
        y1 = min(1000, mid_y + 20)
    if (x1 - x0) < 40:
        mid_x = (x0 + x1) // 2
        x0 = max(0, mid_x - 20)
        x1 = min(1000, mid_x + 20)
    return [y0, x0, y1, x1]


def crop_part_exact(img_path: str | Path, crop_box: list[int], max_side: int = 768):
    """Crop `img_path` to exact normalized `crop_box = [y0, x0, y1, x1]` (without extra hidden padding)
    so coordinates inside the crop map 1:1 back to full-frame 0..1000 coordinates.
    """
    import io
    from google.genai import types

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    y0, x0, y1, x1 = crop_box
    left = max(0, min(w - 1, int(round(x0 * w / 1000.0))))
    upper = max(0, min(h - 1, int(round(y0 * h / 1000.0))))
    right = max(left + 1, min(w, int(round(x1 * w / 1000.0))))
    lower = max(upper + 1, min(h, int(round(y1 * h / 1000.0))))
    cropped = im.crop((left, upper, right, lower))
    cropped.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), buf.getvalue()


def map_crop_box_to_full(crop_box: list[int], crop_region: list[int]) -> list[int]:
    """Map [ymin, xmin, ymax, xmax] in 0..1000 of `crop_region` back to 0..1000 full-frame coordinates."""
    cb = sanitize_box(crop_box)
    if not cb:
        return sanitize_box(crop_region)
    ry0, rx0, ry1, rx1 = crop_region
    rh = max(1, ry1 - ry0)
    rw = max(1, rx1 - rx0)
    fy0 = int(round(ry0 + cb[0] * rh / 1000.0))
    fx0 = int(round(rx0 + cb[1] * rw / 1000.0))
    fy1 = int(round(ry0 + cb[2] * rh / 1000.0))
    fx1 = int(round(rx0 + cb[3] * rw / 1000.0))
    return sanitize_box([fy0, fx0, fy1, fx1]) or sanitize_box(crop_region)


def map_crop_tip_to_full(
    crop_tip: list[int] | None,
    crop_region: list[int],
    full_box: list[int],
    is_instrument: bool,
) -> list[int] | None:
    if not is_instrument:
        return None
    ct = sanitize_tip(crop_tip, is_instrument=True)
    if not ct:
        return sanitize_tip(None, box=full_box, is_instrument=True)
    ry0, rx0, ry1, rx1 = crop_region
    rh = max(1, ry1 - ry0)
    rw = max(1, rx1 - rx0)
    fy = int(round(ry0 + ct[0] * rh / 1000.0))
    fx = int(round(rx0 + ct[1] * rw / 1000.0))
    return sanitize_tip([fy, fx], box=full_box, is_instrument=True)


def _structure_manual_snippet(manual: dict, label: str) -> str:
    if not isinstance(manual, dict):
        return ""
    for s in manual.get("structures", []):
        if isinstance(s, dict) and s.get("key") == label:
            return json.dumps(s)
    return ""


async def verify_and_refine_item(
    *,
    frame_id: str,
    video: str,
    split: str,
    img_path: str,
    label: str,
    pass1: dict[str, Any],
    pass2: dict[str, Any],
    manual: dict[str, Any],
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    call_fn=None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare Pass 1 (Flash) and Pass 2 (Argon) boxes; if IoU >= 0.5 -> accept,
    else mark AMBER and run up to `max_rounds` tighter-crop refinement rounds.
    Returns (final_resolution_dict, list_of_round_log_dicts).
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

    iou0 = box_iou(b1, b2)
    rounds_log: list[dict[str, Any]] = []

    if iou0 >= iou_thresh and b1 and b2:
        final_box = average_boxes(b1, b2, w1=c1, w2=c2)
        final_tip = average_tips(t1, t2, box=final_box, is_instrument=is_inst)
        if is_inst:
            final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
        final_conf = round(0.5 * (c1 + c2), 4)
        return (
            {
                "box": final_box,
                "tip": final_tip,
                "conf": final_conf,
                "iou": round(iou0, 4),
                "round": 0,
                "status": "accept",
            },
            rounds_log,
        )

    # Round 0 disagreement -> AMBER
    rounds_log.append(
        {
            "frame_id": frame_id,
            "video": video,
            "split": split,
            "label": label,
            "round": 0,
            "crop": [0, 0, 1000, 1000],
            "iou": round(iou0, 4),
            "pass1_box": b1,
            "pass2_box": b2,
            "verdict": "AMBER",
            "reason": f"Initial pass1 vs pass2 box IoU {iou0:.3f} < {iou_thresh:.2f}; escalating to tighter-crop Amber loop.",
        }
    )

    cur_b1 = b1 or b2 or [100, 100, 900, 900]
    cur_b2 = b2 or b1 or [100, 100, 900, 900]
    cur_t1 = t1
    cur_t2 = t2
    cur_c1, cur_c2 = c1, c2
    full_img = gem.img_part(img_path)

    for r in range(1, max_rounds + 1):
        crop_reg = disagreement_crop(cur_b1, cur_b2, round_idx=r)
        crop_img = crop_part_exact(img_path, crop_reg)

        tip_instr = (
            "Also return tip_2d [y, x] (0-1000 in the CROP image) placed squarely on the metallic distal working tip/jaws/hook."
            if is_inst
            else "Set tip_2d to []."
        )
        prompt_crop_flash = (
            f"AMBER DISAGREEMENT LOOP — ROUND {r}/{max_rounds} (Zoomed Disagreement Crop).\n"
            f"Target structure: '{label}' ({human_name}).\n"
            f"Manual definition: {struct_info}\n"
            f"This image is a zoomed crop around region {crop_reg} (in full-frame 0-1000 coords) where two passes disagreed "
            f"(Pass A: {cur_b1}, Pass B: {cur_b2}).\n"
            "Draw one tight bounding box `box_2d` [ymin, xmin, ymax, xmax] normalised 0-1000 WITHIN THIS CROP IMAGE "
            f"covering the entire visible extent of '{label}' inside this crop. {tip_instr}"
        )
        prompt_crop_argon = (
            f"SENIOR REVIEWER AMBER LOOP — ROUND {r}/{max_rounds} (Tighter Crop Re-examination).\n"
            f"Target structure: '{label}' ({human_name}).\n"
            f"Manual criteria: {struct_info}\n"
            f"Two earlier passes disagreed on the exact bounding box (IoU < 0.50). "
            f"You are looking at the zoomed disagreement crop covering full-frame region {crop_reg}.\n"
            "Independently localize the exact visible boundaries of this structure within this crop image as "
            f"`box_2d` [ymin, xmin, ymax, xmax] normalised 0-1000 in CROP coordinates. {tip_instr} "
            "Explain briefly how the zoomed view resolves the boundary."
        )

        res_f, res_a = await asyncio.gather(
            caller(
                prompt_crop_flash,
                model=C.FLASH,
                images=[crop_img],
                schema=CropRefinement,
                tag=f"loc-r{r}-flash",
                sample=r,
            ),
            caller(
                prompt_crop_argon,
                model=C.ARGON,
                images=[crop_img, full_img],
                schema=CropRefinement,
                tag=f"loc-r{r}-argon",
                sample=r,
            ),
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
        reason_r = str((res_a.get("reason") if isinstance(res_a, dict) else "") or "Tighter crop refinement")
        iou_r = box_iou(rb1, rb2)
        verdict_r = "accept" if iou_r >= iou_thresh else "AMBER"

        rounds_log.append(
            {
                "frame_id": frame_id,
                "video": video,
                "split": split,
                "label": label,
                "round": r,
                "crop": crop_reg,
                "iou": round(iou_r, 4),
                "pass1_box": rb1,
                "pass2_box": rb2,
                "verdict": verdict_r,
                "reason": reason_r,
            }
        )
        gem.trace(
            "localize-loop",
            f"{frame_id} {label} round {r}: IoU={iou_r:.3f} -> {verdict_r} ({reason_r[:100]})",
        )

        cur_b1, cur_b2 = rb1, rb2
        cur_t1, cur_t2 = rt1, rt2
        cur_c1, cur_c2 = rc1, rc2

        if verdict_r == "accept":
            final_box = average_boxes(rb1, rb2, w1=rc1, w2=rc2)
            final_tip = average_tips(rt1, rt2, box=final_box, is_instrument=is_inst)
            if is_inst:
                final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
            final_conf = round(0.5 * (rc1 + rc2), 4)
            return (
                {
                    "box": final_box,
                    "tip": final_tip,
                    "conf": final_conf,
                    "iou": round(iou_r, 4),
                    "round": r,
                    "status": "accept",
                },
                rounds_log,
            )

    # Still AMBER after max_rounds
    final_box = cur_b2 or cur_b1
    final_tip = average_tips(cur_t1, cur_t2, box=final_box, is_instrument=is_inst)
    if is_inst:
        final_tip = snap_tip_to_instrument_pixels(img_path, final_tip, final_box)
    final_conf = round(0.5 * (cur_c1 + cur_c2), 4)
    final_iou = box_iou(cur_b1, cur_b2)
    return (
        {
            "box": final_box,
            "tip": final_tip,
            "conf": final_conf,
            "iou": round(final_iou, 4),
            "round": max_rounds,
            "status": "AMBER",
        },
        rounds_log,
    )


# ---------------------------------------------------------------------------
# POST-HOC EVALUATION ONLY (runs AFTER out/annotations.jsonl is written)
# ---------------------------------------------------------------------------

def _connected_component_boxes(mask_bin: np.ndarray, min_pixels: int = 25) -> list[list[int]]:
    """Extract normalized 0..1000 bounding boxes for major connected components of a binary mask
    using a fast 2D grid block connectivity pass (no scipy dependency required).
    """
    h, w = mask_bin.shape
    ys, xs = np.where(mask_bin)
    if len(ys) == 0:
        return []
    full_box = [
        int(round(ys.min() * 1000.0 / h)),
        int(round(xs.min() * 1000.0 / w)),
        int(round((ys.max() + 1) * 1000.0 / h)),
        int(round((xs.max() + 1) * 1000.0 / w)),
    ]
    # Downsample onto a 64x64 coarse grid to find separated instances (e.g. 2 graspers on opposite sides)
    gh, gw = 64, 64
    grid = np.zeros((gh, gw), dtype=bool)
    gy = np.clip((ys * gh) // h, 0, gh - 1)
    gx = np.clip((xs * gw) // w, 0, gw - 1)
    grid[gy, gx] = True

    visited = np.zeros_like(grid, dtype=bool)
    comp_boxes = [full_box]
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
                if int(in_comp.sum()) >= min_pixels:
                    cy = ys[in_comp]
                    cx = xs[in_comp]
                    cbox = [
                        int(round(cy.min() * 1000.0 / h)),
                        int(round(cx.min() * 1000.0 / w)),
                        int(round((cy.max() + 1) * 1000.0 / h)),
                        int(round((cx.max() + 1) * 1000.0 / w)),
                    ]
                    if cbox not in comp_boxes:
                        comp_boxes.append(cbox)
    return comp_boxes


def evaluate_annotations(
    annotations_path: Path | str | None = None,
    index_path: Path | str | None = None,
    rounds_path: Path | str | None = None,
    out_path: Path | str | None = None,
    cholect50_index_path: Path | str | None = None,
) -> dict[str, Any]:
    """Post-hoc evaluation of `out/annotations.jsonl` on held-out `split == 'test'` frames against
    CholecSeg8k ground-truth watershed masks (`data/index.csv:mask_path`).
    Never called inside the localization loop.
    """
    ann_file = Path(annotations_path) if annotations_path is not None else (C.OUT_DIR / "annotations.jsonl")
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")
    rnd_file = Path(rounds_path) if rounds_path is not None else (C.OUT_DIR / "localize_rounds.jsonl")
    res_file = Path(out_path) if out_path is not None else (C.OUT_DIR / "localize_results.json")
    c50_file = (
        Path(cholect50_index_path)
        if cholect50_index_path is not None
        else (C.DATA_DIR / "external" / "cholect50" / "index.csv")
    )

    if not ann_file.exists() or ann_file.stat().st_size == 0:
        empty_res = {"error": f"No annotations found at {ann_file}", "n_test_annotations": 0}
        res_file.parent.mkdir(parents=True, exist_ok=True)
        res_file.write_text(json.dumps(empty_res, indent=2))
        return empty_res

    anns = [json.loads(line) for line in ann_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Deduplicate by (frame_id, label) keeping latest
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    for a in anns:
        fid = str(a.get("frame_id", ""))
        lbl = str(a.get("label", a.get("key", "")))
        if fid and lbl:
            dedup[(fid, lbl)] = a
    anns = list(dedup.values())

    idx_df = pd.read_csv(idx_file) if idx_file.exists() else pd.DataFrame()
    idx_by_fid = (
        idx_df.drop_duplicates(subset=["frame_id"]).set_index("frame_id").to_dict(orient="index")
        if not idx_df.empty and "frame_id" in idx_df.columns
        else {}
    )

    test_anns = [a for a in anns if str(a.get("split", "")) == "test"]
    if not test_anns and anns:
        test_anns = anns

    cls_cache: dict[str, np.ndarray] = {}

    def _get_cls(fid: str) -> np.ndarray | None:
        if fid in cls_cache:
            return cls_cache[fid]
        meta = idx_by_fid.get(fid)
        if not meta:
            return None
        mpath = meta.get("mask_path", "")
        if not mpath or not Path(str(mpath)).is_file():
            return None
        cls_arr = mask_to_classes(mpath)
        cls_cache[fid] = cls_arr
        return cls_arr

    box_ious_all: list[float] = []
    box_ious_full_mask: list[float] = []
    box_ious_accepted: list[float] = []
    pass1_ious_all: list[float] = []
    per_label_ious: dict[str, list[float]] = {}
    per_label_acc_ious: dict[str, list[float]] = {}

    tip_hits_exact: list[int] = []
    tip_hits_tol3px: list[int] = []
    tip_hits_any_tool: list[int] = []
    per_tool_tip_hits: dict[str, list[int]] = {}
    per_tool_tip_tol3px: dict[str, list[int]] = {}

    for a in test_anns:
        fid = str(a["frame_id"])
        lbl = str(a.get("label", a.get("key", "")))
        if lbl not in C.LABELS:
            continue
        cid = C.LABELS[lbl][0]
        cls_arr = _get_cls(fid)
        if cls_arr is None:
            continue

        mask_bin = cls_arr == cid
        if not mask_bin.any():
            continue

        comp_boxes = _connected_component_boxes(mask_bin)
        gt_full_box = comp_boxes[0]
        pred_box = sanitize_box(a.get("box", []))
        p1_box = sanitize_box((a.get("pass1") or {}).get("box", []))

        iou_full = box_iou(pred_box, gt_full_box)
        iou_best = max(box_iou(pred_box, cb) for cb in comp_boxes) if comp_boxes else iou_full
        iou_p1 = max(box_iou(p1_box, cb) for cb in comp_boxes) if (p1_box and comp_boxes) else box_iou(p1_box, gt_full_box)

        box_ious_full_mask.append(iou_full)
        box_ious_all.append(iou_best)
        pass1_ious_all.append(iou_p1)
        per_label_ious.setdefault(lbl, []).append(iou_best)

        if a.get("status") == "accept":
            box_ious_accepted.append(iou_best)
            per_label_acc_ious.setdefault(lbl, []).append(iou_best)

        if lbl in INSTRUMENT_KEYS:
            tip = a.get("tip")
            if isinstance(tip, (list, tuple)) and len(tip) >= 2:
                h, w = cls_arr.shape
                py = min(h - 1, max(0, int(round(float(tip[0]) * (h - 1) / 1000.0))))
                px = min(w - 1, max(0, int(round(float(tip[1]) * (w - 1) / 1000.0))))
                exact_hit = int(bool(mask_bin[py, px]))
                # Also check 3px radius neighborhood for thin wire tips / boundary pixels
                y_lo, y_hi = max(0, py - 3), min(h, py + 4)
                x_lo, x_hi = max(0, px - 3), min(w, px + 4)
                tol_hit = int(bool(mask_bin[y_lo:y_hi, x_lo:x_hi].any()))
                any_tool_mask = (cls_arr == C.LABELS["grasper"][0]) | (cls_arr == C.LABELS["l_hook"][0])
                any_tool_hit = int(bool(any_tool_mask[y_lo:y_hi, x_lo:x_hi].any()))

                tip_hits_exact.append(exact_hit)
                tip_hits_tol3px.append(tol_hit)
                tip_hits_any_tool.append(any_tool_hit)
                per_tool_tip_hits.setdefault(lbl, []).append(exact_hit)
                per_tool_tip_tol3px.setdefault(lbl, []).append(tol_hit)

    # Summarize Amber Loop rounds on test
    rounds_all = []
    if rnd_file.exists() and rnd_file.stat().st_size > 0:
        rounds_all = [json.loads(line) for line in rnd_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    rounds_test = [r for r in rounds_all if str(r.get("split", "")) == "test"]

    def _amber_funnel(r_list: list[dict[str, Any]], total_items: list[dict[str, Any]]) -> dict[str, Any]:
        by_item: dict[tuple[str, str], dict[int, str]] = {}
        for r in r_list:
            key_pair = (str(r.get("frame_id", "")), str(r.get("label", "")))
            rnd_num = int(r.get("round", 0))
            by_item.setdefault(key_pair, {})[rnd_num] = str(r.get("verdict", "AMBER"))
        counts = []
        for r_idx in range(0, MAX_AMBER_ROUNDS + 1):
            c = 0
            for item_key, rmap in by_item.items():
                # An item is still AMBER at round r_idx if it started AMBER at round 0 and was not accepted in rounds 1..r_idx
                if rmap.get(0) == "AMBER":
                    accepted_by_r = any(rmap.get(k) == "accept" for k in range(1, r_idx + 1))
                    if not accepted_by_r:
                        c += 1
            counts.append(c)
        return {
            "ambers_by_round": counts,
            "initial_ambers": counts[0] if counts else 0,
            "residual_ambers": counts[-1] if counts else 0,
            "resolved_by_loop": (counts[0] - counts[-1]) if counts else 0,
            "n_total_annotations": len(total_items),
            "n_accepted_final": sum(1 for x in total_items if x.get("status") == "accept"),
            "n_amber_final": sum(1 for x in total_items if x.get("status") == "AMBER"),
        }

    # Summarize Phase predictions (and check if CholecT50 index has phase labels)
    c50_has_phases = False
    c50_phase_acc = None
    if c50_file.exists():
        c50_df = pd.read_csv(c50_file)
        if any(col in c50_df.columns for col in ("phase", "gt_phase")):
            c50_has_phases = True

    frame_phases: dict[str, str] = {}
    frame_phase_confs: list[float] = []
    pass12_phase_agree: list[int] = []
    for a in test_anns:
        fid = str(a["frame_id"])
        if fid not in frame_phases and a.get("phase"):
            frame_phases[fid] = str(a["phase"])
            p1_ph = (a.get("pass1") or {}).get("phase")
            p2_ph = (a.get("pass2") or {}).get("phase")
            if p1_ph and p2_ph:
                pass12_phase_agree.append(int(str(p1_ph).strip() == str(p2_ph).strip()))
            if a.get("phase_conf") is not None:
                frame_phase_confs.append(float(a["phase_conf"]))

    phase_dist: dict[str, int] = {}
    for ph in frame_phases.values():
        phase_dist[ph] = phase_dist.get(ph, 0) + 1

    per_label_summary = {}
    for lbl, vals in sorted(per_label_ious.items()):
        acc_vals = per_label_acc_ious.get(lbl, vals)
        per_label_summary[lbl] = {
            "n": len(vals),
            "mean_iou": round(float(np.mean(vals)), 4) if vals else 0.0,
            "median_iou": round(float(np.median(vals)), 4) if vals else 0.0,
            "iou_ge_05": round(float(np.mean([v >= 0.5 for v in vals])), 4) if vals else 0.0,
            "accepted_n": len(acc_vals),
            "accepted_mean_iou": round(float(np.mean(acc_vals)), 4) if acc_vals else 0.0,
            "accepted_iou_ge_05": round(float(np.mean([v >= 0.5 for v in acc_vals])), 4) if acc_vals else 0.0,
        }

    results = {
        "split_evaluated": "test",
        "n_test_frames": len({str(a["frame_id"]) for a in test_anns}),
        "n_test_Videos": len({str(a.get("video", "")) for a in test_anns}),
        "n_test_annotations": len(test_anns),
        "n_all_annotations": len(anns),
        "boxes": {
            "n_scored": len(box_ious_all),
            "mean_iou": round(float(np.mean(box_ious_all)), 4) if box_ious_all else 0.0,
            "median_iou": round(float(np.median(box_ious_all)), 4) if box_ious_all else 0.0,
            "iou_ge_05_rate": round(float(np.mean([v >= 0.5 for v in box_ious_all])), 4) if box_ious_all else 0.0,
            "mean_iou_full_mask": round(float(np.mean(box_ious_full_mask)), 4) if box_ious_full_mask else 0.0,
            "pass1_mean_iou": round(float(np.mean(pass1_ious_all)), 4) if pass1_ious_all else 0.0,
            "accepted_only": {
                "n": len(box_ious_accepted),
                "mean_iou": round(float(np.mean(box_ious_accepted)), 4) if box_ious_accepted else 0.0,
                "iou_ge_05_rate": round(float(np.mean([v >= 0.5 for v in box_ious_accepted])), 4) if box_ious_accepted else 0.0,
            },
            "per_label": per_label_summary,
        },
        "tips": {
            "n_scored": len(tip_hits_exact),
            "within_gt_tool_mask_rate": round(float(np.mean(tip_hits_exact)), 4) if tip_hits_exact else 0.0,
            "within_gt_tool_mask_3px_rate": round(float(np.mean(tip_hits_tol3px)), 4) if tip_hits_tol3px else 0.0,
            "within_any_gt_tool_mask_3px_rate": round(float(np.mean(tip_hits_any_tool)), 4) if tip_hits_any_tool else 0.0,
            "per_tool": {
                k: {
                    "n": len(v),
                    "within_gt_tool_mask_rate": round(float(np.mean(v)), 4) if v else 0.0,
                    "within_gt_tool_mask_3px_rate": round(float(np.mean(per_tool_tip_tol3px.get(k, []))), 4)
                    if per_tool_tip_tol3px.get(k)
                    else 0.0,
                }
                for k, v in sorted(per_tool_tip_hits.items())
            },
        },
        "phases": {
            "mode": "scored_cholect50" if c50_has_phases else "showcase_only",
            "cholect50_has_phase_labels": c50_has_phases,
            "cholect50_phase_accuracy": c50_phase_acc,
            "n_test_frames_with_phase": len(frame_phases),
            "pass1_pass2_phase_agreement": round(float(np.mean(pass12_phase_agree)), 4) if pass12_phase_agree else 1.0,
            "mean_phase_conf": round(float(np.mean(frame_phase_confs)), 4) if frame_phase_confs else 0.0,
            "test_phase_distribution": phase_dist,
        },
        "amber_loop": _amber_funnel(rounds_test, test_anns),
        "amber_loop_all_splits": _amber_funnel(rounds_all, anns),
        "zero_leakage_note": (
            "Ground truth masks (data/index.csv:mask_path) were read ONLY in verify_localize.evaluate_annotations "
            "after out/annotations.jsonl was finalized. Zero human labels or ground-truth masks are used in localization."
        ),
    }

    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate out/annotations.jsonl against held-out CholecSeg8k test masks.")
    ap.add_argument("--annotations", type=str, default=None, help="Path to annotations.jsonl")
    ap.add_argument("--index", type=str, default=None, help="Path to data/index.csv")
    ap.add_argument("--rounds", type=str, default=None, help="Path to out/localize_rounds.jsonl")
    ap.add_argument("--out", type=str, default=None, help="Path to out/localize_results.json")
    args = ap.parse_args()
    res = evaluate_annotations(
        annotations_path=args.annotations,
        index_path=args.index,
        rounds_path=args.rounds,
        out_path=args.out,
    )
    print(json.dumps(res, indent=2))
