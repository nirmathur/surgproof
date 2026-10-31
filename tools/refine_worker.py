#!/usr/bin/env python3
"""Self-correction rounds for AI-drawn masks. No ground truth, no human.

Round r: draw the model's CURRENT outline on the frame (crop around it), ask Gemini to correct it:
move points onto the true visible boundary, add points where the outline cuts inside the object,
remove points where it spills outside. Two independent correction passes (different phrasing / scale).
If they agree with each other (IoU >= 0.6) the correction is accepted and becomes the new outline;
otherwise the previous outline is kept (status 'held'). Every raw response is logged.

usage: python3 tools/refine_worker.py --video video55 --rounds 2 [--workers 8]
reads:  out/polygons/<video>.jsonl (round-0 masks from polygon_worker.py)
writes: out/polygons/refine_<video>.jsonl, out/masks_r<k>/<frame>_<label>.png
"""
import argparse, concurrent.futures as cf, io, json, os, re, threading
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

import sys
sys.path.insert(0, os.path.dirname(__file__))
from polygon_worker import NAMES, R, call, iou, poly_mask  # same API + model as round 0

lock = threading.Lock()


def sdf_fuse_masks(m1: np.ndarray, m2: np.ndarray, w1: float = 0.54) -> np.ndarray:
    """Fuse two binary masks via Signed Distance Field zero-level-set consensus (avoids m1 & m2 erosion)."""
    b1 = m1.astype(bool)
    b2 = m2.astype(bool)
    if not b1.any():
        return b2
    if not b2.any():
        return b1
    sdf1 = distance_transform_edt(b1) - distance_transform_edt(~b1)
    sdf2 = distance_transform_edt(b2) - distance_transform_edt(~b2)
    return (w1 * sdf1 + (1.0 - w1) * sdf2) >= 0.0


def _safe_parse_polygon(raw: str) -> list:
    """Parse polygon list from Gemini JSON response, resilient to trailing text or code fences."""
    if not raw:
        return []
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt, flags=re.DOTALL).strip()
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            return obj.get("polygon", []) or []
    except Exception:
        pass
    pairs = re.findall(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", txt)
    return [[float(y), float(x)] for y, x in pairs]


def _load_mask_2d(path: str, size: tuple[int, int]) -> np.ndarray:
    """Load a 2D boolean mask from grayscale or RGBA PNG."""
    W, H = size
    if not path or not os.path.isfile(path):
        return np.zeros((H, W), dtype=bool)
    im = Image.open(path)
    if im.size != (W, H):
        im = im.resize((W, H), Image.Resampling.NEAREST)
    arr = np.array(im)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return arr[..., 3] > 0
    if arr.ndim == 3:
        return arr.max(axis=2) > 0
    return arr > 0


def overlay_crop(im, poly_px, pad=60):
    xs = [p[0] for p in poly_px]; ys = [p[1] for p in poly_px]
    W, H = im.size
    box = (int(max(0, min(xs) - pad)), int(max(0, min(ys) - pad)), int(min(W, max(xs) + pad)), int(min(H, max(ys) + pad)))
    ov = im.copy(); d = ImageDraw.Draw(ov)
    d.line(poly_px + [poly_px[0]], fill=(0, 255, 120), width=3)
    crop = ov.crop(box); raw = im.crop(box)
    return crop, raw, box


def jpeg(im, width):
    im = im.resize((width, max(1, int(width * im.size[1] / im.size[0]))))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88); return b.getvalue()


def correct(crop_overlay, label, phrasing):
    if phrasing == 0:
        prompt = (f"The green line is a draft outline of {NAMES[label]} in this laparoscopic image. It may cut inside the object "
                  f"or spill outside it. Return a CORRECTED polygon that follows the true visible boundary of {NAMES[label]}: "
                  f'JSON {{"polygon":[[y,x],...]}}, 12-40 points, normalised 0-1000 to THIS image. Ignore the green line when deciding the boundary.')
        w = 640
    else:
        prompt = (f"Fix the green draft contour so it hugs the real edge of {NAMES[label]} in this surgical frame: extend it where it "
                  f"misses parts of the object, pull it in where it covers background. Answer JSON "
                  f'{{"polygon":[[y,x],...]}}: 12-40 ordered points, [y,x] normalised 0-1000 to this image.')
        w = 896
    raw = call(jpeg(crop_overlay, w), prompt)
    return _safe_parse_polygon(raw), raw


def job(rec, rounds, video):
    fp = rec["frame_path"]; label = rec["label"]; fr = rec["frame_id"]
    im = Image.open(fp).convert("RGB")
    poly = [tuple(p) for p in rec["polygon_px"]]
    cur_mask = _load_mask_2d(rec.get("mask_path", ""), im.size)
    if not cur_mask.any() and len(poly) > 2:
        cur_mask, _ = poly_mask([[p[1] * 1000.0 / im.size[1], p[0] * 1000.0 / im.size[0]] for p in poly], (0, 0, im.size[0], im.size[1]), im.size)
    hist = []
    for r in range(1, rounds + 1):
        crop_ov, _, box = overlay_crop(im, poly)
        p1, raw1 = correct(crop_ov, label, 0)
        p2, raw2 = correct(crop_ov, label, 1)
        m1, px1 = poly_mask(p1, box, im.size)
        m2, px2 = poly_mask(p2, box, im.size)
        agree = iou(m1, m2)
        moved = iou(m1, cur_mask)
        with lock, open(f"{R}/out/polygons/refine_raw_{video}.jsonl", "a") as f:
            f.write(json.dumps({"frame_id": fr, "label": label, "round": r, "box": box, "raw1": raw1, "raw2": raw2}) + "\n")
        if agree >= 0.6 and m1.sum() > 0:
            cur_mask, poly, st = sdf_fuse_masks(m1, m2), px1, "corrected"
        else:
            st = "held"
        hist.append({"round": r, "status": st, "agree_iou": round(agree, 4), "iou_vs_previous": round(moved, 4)})
        os.makedirs(f"{R}/out/masks_r{r}", exist_ok=True)
        Image.fromarray((cur_mask * 255).astype("uint8")).save(f"{R}/out/masks_r{r}/{fr}_{label}.png")
    out = {"frame_id": fr, "video": video, "label": label, "rounds": hist,
           "final_mask_path": f"{R}/out/masks_r{rounds}/{fr}_{label}.png", "polygon_px": [[round(x, 1), round(y, 1)] for x, y in poly]}
    with lock, open(f"{R}/out/polygons/refine_{video}.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    with lock, open(f"{R}/out/annotations_live.jsonl", "a") as f:
        f.write(json.dumps({"frame_id": fr, "video": video, "label": label, "kind": "mask_refine", "round": rounds,
                            "status": hist[-1]["status"], "mask_path": out["final_mask_path"]}) + "\n")
    return hist[-1]["status"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
    import csv
    paths = {r["frame_id"]: r["frame_path"] for r in csv.DictReader(open(f"{R}/out/decisions.csv"))}
    done = set()
    ref_file = f"{R}/out/polygons/refine_{a.video}.jsonl"
    if os.path.exists(ref_file):
        for l in open(ref_file):
            if l.strip():
                try:
                    j = json.loads(l)
                    done.add((j["frame_id"], j["label"]))
                except Exception:
                    pass
    recs = []
    for l in open(f"{R}/out/polygons/{a.video}.jsonl"):
        j = json.loads(l)
        if (j.get("frame_id"), j.get("label")) in done:
            continue
        if j.get("status") == "accept" and j.get("polygon_px") and len(j["polygon_px"]) > 2:
            fp = paths.get(j["frame_id"]); fp = fp if fp and os.path.isabs(fp) else os.path.join(R, fp or "")
            recs.append({**j, "frame_path": fp})
    print(f"{a.video}: refining {len(recs)} masks x {a.rounds} rounds", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(job, r, a.rounds, a.video) for r in recs]):
            n += 1
            try: st = fut.result()
            except Exception as e: st = f"error {str(e)[:80]}"
            if n % 10 == 0 or n == len(recs): print(f"{a.video}: {n}/{len(recs)} last={st}", flush=True)


if __name__ == "__main__":
    main()
