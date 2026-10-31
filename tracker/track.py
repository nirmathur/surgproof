"""Step 4: link detections into tracks per (clip, label).

Positions are camera-compensated: p' = p - C(t), C(t) = cumulative camera shift since clip start. Frame by
frame, active tracks (last seen within max_gap + 1 frames) are matched to detections with Hungarian
assignment on cost = w_dist * dist/diag + w_iou * (1 - IoU), gated by dist/diag < gate_dist_diag or IoU > 0.
"""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from tracker.common import Timer, cli, iou, load_cfg, load_dets, skip, work

BIG = 1e6


def cam_offsets(cam):
    """Cumulative camera shift per (clip_id, frame_idx) -> columns off_x, off_y."""
    cam = cam.sort_values(["clip_id", "frame_idx"]).copy()
    cam["off_x"] = cam.groupby("clip_id").cam_dx.cumsum()
    cam["off_y"] = cam.groupby("clip_id").cam_dy.cumsum()
    return cam[["clip_id", "frame_idx", "off_x", "off_y"]]


def compensate(dets, cam):
    d = dets.merge(cam_offsets(cam), on=["clip_id", "frame_idx"], how="left")
    d[["off_x", "off_y"]] = d[["off_x", "off_y"]].fillna(0.0)
    for c, o in (("x1", "off_x"), ("x2", "off_x"), ("cx", "off_x"), ("y1", "off_y"), ("y2", "off_y"), ("cy", "off_y")):
        d["c" + c] = d[c] - d[o]
    return d


def link(frames, boxes, tcfg):
    """frames: (N,) int, boxes: (N,4) compensated xyxy. Returns (N,) local track ids."""
    order = np.argsort(frames, kind="stable")
    tid = np.full(len(frames), -1)
    tracks = []  # dicts: last_frame, box
    max_gap = tcfg["max_gap"]
    for f in np.unique(frames):
        idx = order[frames[order] == f]
        active = [k for k, t in enumerate(tracks) if f - t["last"] <= max_gap + 1]
        if active and len(idx):
            tb = np.array([tracks[k]["box"] for k in active])
            db = boxes[idx]
            tc = (tb[:, :2] + tb[:, 2:]) / 2
            dc = (db[:, :2] + db[:, 2:]) / 2
            diag = np.hypot(tb[:, 2] - tb[:, 0], tb[:, 3] - tb[:, 1])
            dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / np.maximum(diag[:, None], 1.0)
            ov = iou(tb, db)
            cost = tcfg["w_dist"] * dist + tcfg["w_iou"] * (1 - ov)
            cost[(dist >= tcfg["gate_dist_diag"]) & (ov <= 0)] = BIG
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < BIG:
                    k = active[a]
                    tid[idx[b]] = k
                    tracks[k].update(last=f, box=boxes[idx[b]])
        for j in idx:
            if tid[j] < 0:
                tid[j] = len(tracks)
                tracks.append({"last": f, "box": boxes[j]})
    return tid


def run(cfg, source, force=False):
    out = work(cfg, f"tracks_{source}.parquet")
    if skip(out, force):
        return
    d = compensate(load_dets(cfg, source), pd.read_parquet(work(cfg, f"camera_{source}.parquet")))
    d["track_id"] = -1
    nxt = 0
    for _, g in d.groupby(["clip_id", "label"], sort=True):
        t = link(g.frame_idx.values, g[["cx1", "cy1", "cx2", "cy2"]].values, cfg["track"])
        d.loc[g.index, "track_id"] = t + nxt
        nxt += t.max() + 1
    d.to_parquet(out, index=False)
    L = d.groupby("track_id").size()
    print(f"{len(d)} detections -> {d.track_id.nunique()} tracks; length median {L.median():.0f}, "
          f"single-frame tracks {int((L == 1).sum())}; tracks per clip-label median "
          f"{d.groupby(['clip_id', 'label']).track_id.nunique().median():.0f}")


if __name__ == "__main__":
    a = cli("Track linking").parse_args()
    with Timer("track"):
        run(load_cfg(), a.source, a.force)
