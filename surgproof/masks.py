"""Stage 5: turn the AI's polygons into edge-snapped masks.

polygon -> GrabCut seeded by the polygon (eroded = sure foreground, outside dilated = sure background)
-> keep the main component -> smooth contour. Boxes come from mask bounds.

Inputs (jobs/<id>/): meta.json, frames/<kf>.jpg, detect.jsonl, verify.jsonl, decisions.jsonl.
Outputs: masks/<kf>_<label>.png (0/255, keyframe image size), masks.jsonl (one row per mask).

Only AI outputs and video frames are read here. Ground truth never enters this module.
This module also holds the small job-dir helpers shared by propagate, render and export.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

LABELS = ["gallbladder", "liver", "grasper", "l_hook", "cystic_duct", "blood"]
# Front-most first: instruments occlude blood, which occludes anatomy.
FRONT_TO_BACK = ["grasper", "l_hook", "blood", "cystic_duct", "gallbladder", "liver"]
ORGANS = {"gallbladder", "liver"}

ACCEPT, REJECT, AMBER = "accept", "reject", "AMBER"

MAX_SIDE = 960          # GrabCut working resolution cap (longest side, px)
ERODE_FRAC = 0.10       # sure-foreground erosion, as a fraction of sqrt(seed area)
DILATE_FRAC = 0.12      # background band, as a fraction of sqrt(seed area)
MIN_BAND_PX = 6
MIN_SEED_AREA = 16      # px at source resolution; smaller polygons are treated as no seed
MIN_KEEP_RATIO = 0.15   # GrabCut result below this fraction of the seed area has collapsed
MAX_KEEP_RATIO = 4.0    # above this multiple of the seed area it has leaked
ORGAN_FRAGMENT = 0.25   # organs may be split by an occluding tool: keep pieces this big relative to the main one


# ---- job-dir helpers (shared by the masks lane) ----

def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_write(path, data: bytes | str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=1))


def write_jsonl(path, rows):
    atomic_write(path, "".join(json.dumps(r) + "\n" for r in rows))


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def set_status(job, stage, progress, message=""):
    """Update status.json per the contract, preserving `started`."""
    path = Path(job) / "status.json"
    old = {}
    if path.exists():
        try:
            old = read_json(path)
        except (OSError, ValueError):
            pass
    write_json(path, {"stage": stage, "progress": round(float(progress), 4), "message": message,
                      "started": old.get("started") or _now(), "updated": _now(), "error": old.get("error")})


def progress_fn(job, stage, on_progress):
    if on_progress is not None:
        return on_progress
    return lambda frac, msg="": set_status(job, stage, frac, msg)


def frame_size(job, meta):
    """(width, height) of the keyframe images, which every mask matches. Falls back to meta.json.

    The image wins over meta because ffprobe reports unrotated dimensions for rotated phone footage."""
    for k in meta.get("keyframes", []):
        img = cv2.imread(str(Path(job) / "frames" / f"{k['kf']}.jpg"))
        if img is not None:
            return img.shape[1], img.shape[0]
    return int(meta["width"]), int(meta["height"])


def mask_path(job, kf, label):
    return Path(job) / "masks" / f"{kf}_{label}.png"


def track_path(job, frame_index, label):
    return Path(job) / "tracks" / f"{frame_index}_{label}.png"


def write_mask(path, mask):
    ok, buf = cv2.imencode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not ok:
        raise OSError(f"could not encode {path}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf.tobytes())


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"could not read mask {path}")
    return m > 0


def mask_box(mask):
    """[x0, y0, x1, y1] from mask bounds (inclusive pixel coords), or None for an empty mask."""
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def resolve_decisions(job):
    """{(kf, label): decision row}, latest row winning."""
    return {(r["kf"], r["label"]): r for r in read_jsonl(Path(job) / "decisions.jsonl")}


# ---- geometry ----

def clean_polygon(polygon, w, h):
    """Polygon as an int32 Nx2 array clipped to the image, or None if it cannot seed anything."""
    if polygon is None:
        return None
    try:
        p = np.asarray(polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        return None
    p[:, 0] = np.clip(p[:, 0], 0, w - 1)
    p[:, 1] = np.clip(p[:, 1], 0, h - 1)
    return np.round(p).astype(np.int32)


def polygon_mask(polygon, w, h):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [polygon], 1)
    return m


def _disk(radius):
    r = max(1, int(round(radius)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _main_component(mask, label):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    biggest = int(areas.max())
    if label in ORGANS:
        keep = 1 + np.where(areas >= ORGAN_FRAGMENT * biggest)[0]
        return np.isin(lab, keep)
    return lab == 1 + int(areas.argmax())


def _smooth(mask, shape_h, label):
    # Sigma scales with resolution. Instruments get less so their tips survive.
    base = 1.2 if label not in ORGANS else 2.5
    sigma = max(0.8, base * shape_h / 480.0)
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma) > 0.5


def grabcut_refine(img, seed, label, occluders=None):
    """Edge-snap a seed mask. Returns (mask, fallback) where fallback names why the seed was kept, or None.

    `occluders` is a bool mask of objects in front of this one; those pixels become sure background."""
    h, w = seed.shape
    seed = seed.astype(np.uint8)
    seed_area = int(seed.sum())
    if seed_area == 0:
        return np.zeros((h, w), bool), "empty_seed"

    # Work on a crop around the seed at capped resolution: faster, and the colour models stay local.
    root = np.sqrt(seed_area)
    pad = int(max(MIN_BAND_PX, DILATE_FRAC * root)) + 4
    ys, xs = np.where(seed)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    scale = min(1.0, MAX_SIDE / max(x1 - x0, y1 - y0))
    crop_img = img[y0:y1, x0:x1]
    crop_seed = seed[y0:y1, x0:x1]
    crop_occ = occluders[y0:y1, x0:x1].astype(np.uint8) if occluders is not None else None
    if scale < 1.0:
        size = (max(1, int(round((x1 - x0) * scale))), max(1, int(round((y1 - y0) * scale))))
        crop_img = cv2.resize(crop_img, size, interpolation=cv2.INTER_AREA)
        crop_seed = cv2.resize(crop_seed, size, interpolation=cv2.INTER_NEAREST)
        if crop_occ is not None:
            crop_occ = cv2.resize(crop_occ, size, interpolation=cv2.INTER_NEAREST)

    # Sure foreground from the distance transform, capped at half the seed's own thickness,
    # so a thin instrument keeps a core line instead of eroding away entirely.
    dist = cv2.distanceTransform(crop_seed, cv2.DIST_L2, 5)
    thickness = float(dist.max())
    fg = dist > min(ERODE_FRAC * root * scale, 0.5 * thickness)
    band = cv2.dilate(crop_seed, _disk(max(MIN_BAND_PX * scale, DILATE_FRAC * root * scale)))

    gc = np.full(crop_seed.shape, cv2.GC_BGD, np.uint8)
    gc[band > 0] = cv2.GC_PR_BGD
    gc[crop_seed > 0] = cv2.GC_PR_FGD
    gc[fg] = cv2.GC_FGD
    if crop_occ is not None:
        gc[crop_occ > 0] = cv2.GC_BGD
    if not (gc == cv2.GC_FGD).any():
        return seed.astype(bool), "no_sure_foreground"
    # GrabCut needs some background to model; a seed filling the crop has none.
    if not np.isin(gc, (cv2.GC_BGD, cv2.GC_PR_BGD)).any():
        return seed.astype(bool), "no_background"

    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(crop_img, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return seed.astype(bool), "grabcut_error"

    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    if scale < 1.0:
        out = cv2.resize(out.astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0
    full = np.zeros((h, w), bool)
    full[y0:y1, x0:x1] = out
    full = _main_component(full, label)
    full = _smooth(full, h, label)
    if occluders is not None:
        full &= ~occluders.astype(bool)

    area = int(full.sum())
    if area < MIN_KEEP_RATIO * seed_area:
        return seed.astype(bool), "grabcut_collapsed"
    if area > MAX_KEEP_RATIO * seed_area:
        return seed.astype(bool), "grabcut_leaked"
    return full, None


# ---- seeds from the pipeline's records ----

def _latest_round_polygon(decision):
    for r in reversed((decision or {}).get("rounds") or []):
        if isinstance(r, dict) and r.get("polygon"):
            return r["polygon"], r
    return None, None


def plan_seeds(job, meta, w, h):
    """Which (kf, label) get a mask, from what seed. Rejected labels never do.

    Returns {kf: [ {label, status, seed_kind, seed_mask, model, prompt_hash}, ... ]}, plus skipped rows."""
    job = Path(job)
    decisions = resolve_decisions(job)
    detect = {(r["kf"], r["label"]): r for r in read_jsonl(job / "detect.jsonl")}
    verify = {(r["kf"], r["label"]): r for r in read_jsonl(job / "verify.jsonl")}
    plan, skipped = {}, []
    for k in meta["keyframes"]:
        kf = k["kf"]
        for label in LABELS:
            det, dec, ver = detect.get((kf, label)), decisions.get((kf, label)), verify.get((kf, label))
            if det is None and dec is None:
                continue
            # An unadjudicated detection is unverified, which is what AMBER means. Never accepted by default.
            status = dec["status"] if dec else AMBER
            if status == REJECT:
                continue
            if status == ACCEPT and not dec.get("final_present"):
                continue  # accepted as absent
            poly, round_row = _latest_round_polygon(dec)
            model = (round_row or {}).get("model") or (det or {}).get("model")
            prompt_hash = (round_row or {}).get("prompt_hash") or (det or {}).get("prompt_hash")
            kind = "amber_round_polygon" if poly else "detect_polygon"
            poly = poly or (det or {}).get("polygon")
            cleaned = clean_polygon(poly, w, h)
            seed = polygon_mask(cleaned, w, h) if cleaned is not None else None
            if seed is None or int(seed.sum()) < MIN_SEED_AREA:
                # Present by verdict but no usable outline: fall back to the verifier's box if it has one.
                seed, kind = _box_seed(ver, w, h)
                model, prompt_hash = (ver or {}).get("model"), (ver or {}).get("prompt_hash")
            if seed is None:
                skipped.append({"kf": kf, "label": label, "status": status, "reason": "no usable polygon or box"})
                continue
            plan.setdefault(kf, []).append({"label": label, "status": status, "seed_kind": kind, "seed": seed,
                                            "model": model, "prompt_hash": prompt_hash})
    return plan, skipped


def _box_seed(ver, w, h):
    box = (ver or {}).get("box") if (ver or {}).get("present") else None
    if not box or len(box) != 4:
        return None, None
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((max(0.0, x0), min(w - 1.0, x1)))
    y0, y1 = sorted((max(0.0, y0), min(h - 1.0, y1)))
    if (x1 - x0) * (y1 - y0) < MIN_SEED_AREA:
        return None, None
    # A box overshoots its object, so seed from its inner ellipse.
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(1, int((x1 - x0) / 2)), max(1, int((y1 - y0) / 2))),
                0, 0, 360, 1, -1)
    return m, "verify_box"


def _process_keyframe(job, kf, items):
    img = cv2.imread(str(Path(job) / "frames" / f"{kf}.jpg"))
    if img is None:
        raise FileNotFoundError(f"missing keyframe image frames/{kf}.jpg")
    h, w = img.shape[:2]
    order = {lab: i for i, lab in enumerate(FRONT_TO_BACK)}
    occ = np.zeros((h, w), bool)
    rows = []
    for it in sorted(items, key=lambda x: order[x["label"]]):
        label = it["label"]
        mask, fallback = grabcut_refine(img, it["seed"], label, occ if occ.any() else None)
        if not mask.any():
            rows.append({"kf": kf, "label": label, "status": it["status"], "seed": it["seed_kind"],
                         "fallback": fallback, "area": 0, "box": None, "path": None,
                         "model": it["model"], "prompt_hash": it["prompt_hash"]})
            continue
        write_mask(mask_path(job, kf, label), mask)
        rows.append({"kf": kf, "label": label, "status": it["status"], "seed": it["seed_kind"], "fallback": fallback,
                     "seed_area": int(it["seed"].sum()), "area": int(mask.sum()), "box": mask_box(mask),
                     "path": f"masks/{kf}_{label}.png", "model": it["model"], "prompt_hash": it["prompt_hash"]})
        occ |= mask
    return rows


def run(job_dir, on_progress=None, workers=None):
    """Run stage 5. Returns a summary dict; masks that fell back to their raw seed are counted, not hidden."""
    job = Path(job_dir)
    progress = progress_fn(job, "masks", on_progress)
    meta = read_json(job / "meta.json")
    w, h = frame_size(job, meta)
    plan, skipped = plan_seeds(job, meta, w, h)
    out_dir = job / "masks"
    out_dir.mkdir(exist_ok=True)
    for stale in out_dir.glob("*.png"):
        stale.unlink()
    progress(0.0, f"refining masks on {len(plan)} keyframes")

    rows, done = [], 0
    workers = workers or min(8, os.cpu_count() or 4)
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(_process_keyframe, job, kf, items): kf for kf, items in plan.items()}
        for fut in cf.as_completed(futs):
            rows.extend(fut.result())
            done += 1
            progress(done / max(1, len(futs)), f"masks for {done}/{len(futs)} keyframes")
    rows.sort(key=lambda r: (r["kf"], r["label"]))
    write_jsonl(job / "masks.jsonl", rows)

    n_written = sum(1 for r in rows if r["path"])
    summary = {"n_masks": n_written, "n_keyframes": len(plan), "n_skipped": len(skipped), "skipped": skipped,
               "n_seed_fallback": sum(1 for r in rows if r["path"] and r["fallback"])}
    progress(1.0, f"{n_written} masks written, {summary['n_seed_fallback']} kept their raw seed, "
                  f"{len(skipped)} present labels had no usable outline")
    return summary
