"""Step 3: camera motion per consecutive frame pair + per-frame image quality.

Farneback flow on downscaled greyscale, restricted to the visible field and excluding dilated tool boxes
of the given detections source. Stores median dx, dy (full-res px, motion from t-1 to t) and median magnitude.
Image quality (source independent): blur, contrast, brightness, specular fraction inside the visible field.
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from tracker.common import ROOT, Timer, cli, field_mask, load_cfg, load_dets, load_frames, n_workers, skip, work


def _quality(img, field, q):
    s = q["scale"]
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = cv2.resize(field.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    f = cv2.erode(f.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(g, cv2.CV_64F)[f]
    return {"lap_var": float(lap.var()), "contrast": float(g[f].std()), "brightness": float(g[f].mean()),
            "specular": float(((hsv[..., 2] >= q["specular_v"]) & (hsv[..., 1] <= q["specular_s"]))[f].mean())}


def _clip(args):
    rows, boxes_by_frame, field, cfg, do_quality = args
    c = cfg["camera"]
    s = c["scale"]
    H, W = field.shape
    hs, ws = int(round(H * s)), int(round(W * s))
    field_s = cv2.resize(field.astype(np.uint8), (ws, hs), interpolation=cv2.INTER_NEAREST) > 0
    dil = c["tool_dilate_px"]

    def toolmask(fi):
        m = np.zeros((hs, ws), bool)
        for x1, y1, x2, y2 in boxes_by_frame.get(fi, []):
            m[max(int((y1 - dil) * s), 0):int((y2 + dil) * s) + 1, max(int((x1 - dil) * s), 0):int((x2 + dil) * s) + 1] = True
        return m

    fb = c["farneback"]
    out, qual = [], []
    prev, prev_tm = None, None
    for r in rows:
        img = cv2.imread(str(ROOT / r["image_path"]))
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (ws, hs), interpolation=cv2.INTER_AREA)
        tm = toolmask(r["frame_idx"])
        rec = {"video_id": r["video_id"], "clip_id": r["clip_id"], "frame_idx": r["frame_idx"],
               "cam_dx": 0.0, "cam_dy": 0.0, "cam_mag": 0.0, "cam_valid_frac": 0.0}
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, fb["pyr_scale"], fb["levels"], fb["winsize"],
                                                fb["iterations"], fb["poly_n"], fb["poly_sigma"], 0)
            valid = field_s & ~tm & ~prev_tm
            rec["cam_valid_frac"] = float(valid.mean() / max(field_s.mean(), 1e-9))
            if rec["cam_valid_frac"] >= c["min_valid_frac"]:
                fx, fy = flow[..., 0][valid] / s, flow[..., 1][valid] / s
                rec.update(cam_dx=float(np.median(fx)), cam_dy=float(np.median(fy)),
                           cam_mag=float(np.median(np.hypot(fx, fy))))
        out.append(rec)
        if do_quality:
            qual.append({**{k: rec[k] for k in ("video_id", "clip_id", "frame_idx")}, **_quality(img, field, cfg["quality"])})
        prev, prev_tm = g, tm
    return out, qual


def run(cfg, source, force=False):
    out = work(cfg, f"camera_{source}.parquet")
    qout = work(cfg, "quality.parquet")
    if skip(out, force):
        return
    do_q = force or not qout.exists()
    fr = load_frames(cfg)
    dets = load_dets(cfg, source)
    boxes = {k: g[["x1", "y1", "x2", "y2"]].values for k, g in dets.groupby(["clip_id", "frame_idx"])}
    fields = {v: field_mask(cfg, v) for v in fr.video_id.unique()}
    jobs = []
    for clip, g in fr.groupby("clip_id"):
        g = g.sort_values("frame_idx")
        bb = {fi: boxes.get((clip, fi), []) for fi in g.frame_idx}
        jobs.append((g.to_dict("records"), bb, fields[g.video_id.iloc[0]], cfg, do_q))
    with ProcessPoolExecutor(n_workers(cfg)) as ex:
        res = list(ex.map(_clip, jobs))
    cam = pd.DataFrame([r for a, _ in res for r in a])
    cam.to_parquet(out, index=False)
    print(f"camera: median |motion| {cam.cam_mag.median():.2f}px/frame, p95 {cam.cam_mag.quantile(.95):.2f}; "
          f"low-valid pairs {(cam.cam_valid_frac < cfg['camera']['min_valid_frac']).sum()}")
    if do_q:
        q = pd.DataFrame([r for _, b in res for r in b])
        q.to_parquet(qout, index=False)
        print("quality:", q[["lap_var", "contrast", "brightness", "specular"]].median().round(3).to_dict())


if __name__ == "__main__":
    a = cli("Camera motion + image quality").parse_args()
    with Timer("camera"):
        run(load_cfg(), a.source, a.force)
