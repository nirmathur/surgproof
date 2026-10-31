"""Shared helpers: config, paths, caching, robust z-scores, geometry."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
KEYS = ["video_id", "clip_id", "frame_idx", "label"]
DET_COLS = ["video_id", "clip_id", "frame_idx", "label", "instance_id", "x1", "y1", "x2", "y2",
            "cx", "cy", "area", "confidence", "source"]


def load_cfg(path=None):
    with open(path or ROOT / "config.yaml") as f:
        return yaml.safe_load(f)


def p(cfg, key, *parts):
    d = ROOT / cfg["paths"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d.joinpath(*parts) if parts else d


def work(cfg, name):
    return p(cfg, "work", name)


def n_workers(cfg):
    return cfg.get("n_workers") or os.cpu_count() or 1


def cli(desc, source=True):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--force", action="store_true", help="recompute even if cached output exists")
    if source:
        ap.add_argument("--source", default="gt", help="detections source: gt | gt_corrupted | gemini")
    return ap


def skip(path, force):
    if Path(path).exists() and not force:
        print(f"[cache] {Path(path).relative_to(ROOT)} exists, skipping (use --force)")
        return True
    return False


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.name}] {time.time() - self.t:.1f}s")


def robust_z(x):
    """(x - median) / (1.4826 * MAD); falls back to mean abs deviation, then std, for zero-inflated data."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return np.zeros_like(x)
    med = np.median(x[ok])
    dev = np.abs(x[ok] - med)
    scale = 1.4826 * np.median(dev)
    if scale < 1e-9:
        scale = 1.2533 * dev.mean()
    if scale < 1e-9:
        scale = x[ok].std() or 1.0
    return (x - med) / scale


def iou(a, b):
    """IoU between boxes a (N,4) and b (M,4), xyxy. Returns (N,M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


def inter_frac(a, b):
    """Fraction of each box in a covered by each box in b. (N,M)."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    return ix * iy / np.maximum(aa[:, None], 1e-9)


def finish_dets(df, source):
    """Fill derived columns and enforce the detections schema."""
    df = df.copy()
    df["cx"] = (df.x1 + df.x2) / 2
    df["cy"] = (df.y1 + df.y2) / 2
    df["area"] = (df.x2 - df.x1) * (df.y2 - df.y1)
    df["source"] = source
    if "confidence" not in df:
        df["confidence"] = np.nan
    if "instance_id" not in df:
        df["instance_id"] = pd.NA
    df["instance_id"] = df["instance_id"].astype("Int64")
    df["frame_idx"] = df["frame_idx"].astype(int)
    df = df.sort_values(KEYS + ["x1"]).reset_index(drop=True)
    df["det_id"] = np.arange(len(df))
    return df[DET_COLS + ["det_id"] + [c for c in df.columns if c not in DET_COLS + ["det_id"]]]


def load_frames(cfg):
    return pd.read_parquet(work(cfg, "frames.parquet"))


def load_dets(cfg, source):
    return pd.read_parquet(work(cfg, f"detections_{source}.parquet"))


def field_mask(cfg, video):
    import cv2
    return cv2.imread(str(work(cfg, f"field/{video}.png")), cv2.IMREAD_GRAYSCALE) > 0


def edge_distance(cfg, video):
    """Distance (px) from each pixel to the outside of the visible field (image border counts as outside)."""
    import cv2
    m = field_mask(cfg, video).astype(np.uint8)
    m = np.pad(m, 1)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    return d[1:-1, 1:-1]


def box_edge_gap(dist, boxes):
    """Min distance-to-field-edge inside each box. 0 means the box reaches the edge."""
    H, W = dist.shape
    out = np.zeros(len(boxes))
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, float)):
        xa, xb = int(np.clip(np.floor(x1), 0, W - 1)), int(np.clip(np.ceil(x2), 1, W))
        ya, yb = int(np.clip(np.floor(y1), 0, H - 1)), int(np.clip(np.ceil(y2), 1, H))
        if xb <= xa or yb <= ya or x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            out[i] = 0.0
        else:
            out[i] = dist[ya:yb, xa:xb].min()
    return out
