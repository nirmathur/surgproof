"""Build source="gt" detections from watershed masks: connected components per tool class.

This is the only module that reads masks. Everything downstream reads detections_<source>.parquet.
"""
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from tracker.common import ROOT, Timer, cli, finish_dets, load_cfg, load_frames, n_workers, skip, work


def _clip(args):
    rows, cfg = args
    tools = {lab: cfg["classes"][lab] for lab in cfg["tool_labels"]}
    out = []
    for r in rows:
        m = cv2.imread(str(ROOT / r["mask_path"]), cv2.IMREAD_UNCHANGED)
        m = m[..., 0] if m.ndim == 3 else m
        min_area = cfg["detect"]["min_area_frac"] * m.shape[0] * m.shape[1]
        for lab, val in tools.items():
            n, _, st, _ = cv2.connectedComponentsWithStats((m == val).astype(np.uint8), connectivity=cfg["detect"]["connectivity"])
            comps = [(st[i, 4], st[i]) for i in range(1, n) if st[i, 4] >= min_area]
            comps.sort(key=lambda t: -t[0])
            for k, (a, s) in enumerate(comps):
                x, y, w, h = map(float, s[:4])
                out.append({"video_id": r["video_id"], "clip_id": r["clip_id"], "frame_idx": r["frame_idx"],
                            "label": lab, "instance_id": k, "x1": x, "y1": y, "x2": x + w, "y2": y + h,
                            "mask_area": int(a)})
    return out


def build(cfg):
    fr = load_frames(cfg)
    jobs = [(g.to_dict("records"), cfg) for _, g in fr.groupby("clip_id")]
    with ProcessPoolExecutor(n_workers(cfg)) as ex:
        rows = [r for part in ex.map(_clip, jobs) for r in part]
    df = finish_dets(pd.DataFrame(rows), "gt")
    per = df.groupby(["clip_id", "frame_idx", "label"]).size()
    print(f"{len(df)} gt detections; per label: {df.label.value_counts().to_dict()}")
    print("frame-labels with n instances:", per.groupby(level=2).value_counts().sort_index().to_dict())
    n_frames = len(fr)
    for lab in cfg["tool_labels"]:
        print(f"  {lab}: present in {per.xs(lab, level=2).size / n_frames:.1%} of frames")
    return df


if __name__ == "__main__":
    a = cli("Detections from gt masks", source=False).parse_args()
    cfg = load_cfg()
    out = work(cfg, "detections_gt.parquet")
    if not skip(out, a.force):
        with Timer("detect"):
            build(cfg).to_parquet(out, index=False)
