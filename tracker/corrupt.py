"""Step 6: corruption harness. Inject known errors into gt detections (seeded, reproducible).

Outputs detections_gt_corrupted.parquet (source="gt_corrupted") and errors.parquet with one row per affected
(frame, label) row: error_id, error_type, role (primary/secondary), keys, and the box involved.
Each gt detection is touched by at most one error.
"""
import numpy as np
import pandas as pd

from tracker.common import Timer, box_edge_gap, cli, edge_distance, finish_dets, load_cfg, skip, work

SWAP = {"grasper": "l_hook", "l_hook": "grasper"}


def corrupt(tracks, cfg, dists, frame_shape=(480, 854)):
    c = cfg["corrupt"]
    rng = np.random.default_rng(c["seed"])
    H, W = frame_shape
    d = tracks.sort_values(["clip_id", "label", "track_id", "frame_idx"]).reset_index(drop=True).copy()
    d["keep"] = True
    used = np.zeros(len(d), bool)
    errors, eid = [], 0

    def rec(i, etype, role="primary", label=None, box=None):
        r = d.loc[i]
        b = box if box is not None else r[["x1", "y1", "x2", "y2"]].values
        errors.append({"error_id": eid, "error_type": etype, "role": role, "video_id": r.video_id,
                       "clip_id": r.clip_id, "frame_idx": int(r.frame_idx), "label": label or r.label,
                       "orig_det_id": int(r.det_id), "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]})

    n = len(d)
    rates = c["rates"]
    # vanish: per track, cut at a mid-field box and delete the rest of the track
    gap = np.zeros(n)
    for v, g in d.groupby("video_id"):
        gap[g.index] = box_edge_gap(dists[v], g[["x1", "y1", "x2", "y2"]].values)
    margin = cfg["signals"]["edge_margin_px"]
    for t, g in d.groupby("track_id"):
        if rng.random() >= rates["vanish"] or len(g) < 6:
            continue
        idx = g.index.values
        cand = [k for k in range(2, len(idx) - 2) if gap[idx[k]] > margin]
        if not cand:
            continue
        k = cand[rng.integers(len(cand))]
        gone = idx[k + 1:]
        if used[gone].any():
            continue
        for i in gone:
            rec(i, "vanish", "primary" if i == gone[0] else "secondary")
        d.loc[gone, "keep"] = False
        used[gone] = True
        eid += 1

    order = rng.permutation(n)
    for etype in ["drop", "jitter", "label_swap", "size_error"]:
        k = int(round(rates[etype] * n))
        picked = 0
        for i in order:
            if picked >= k:
                break
            if used[i]:
                continue
            r = d.loc[i]
            if etype == "drop":
                L = int(rng.integers(c["drop_len"][0], c["drop_len"][1] + 1))
                tr = d.index[(d.track_id == r.track_id) & (d.frame_idx >= r.frame_idx)][:L]
                if used[tr].any():
                    continue
                for j in tr:
                    rec(j, "drop", "primary" if j == tr[0] else "secondary")
                d.loc[tr, "keep"] = False
                used[tr] = True
            elif etype == "jitter":
                diag = np.hypot(r.x2 - r.x1, r.y2 - r.y1)
                kk = rng.uniform(*c["jitter_k"])
                a = rng.uniform(0, 2 * np.pi)
                dx, dy = kk * diag * np.cos(a), kk * diag * np.sin(a)
                d.loc[i, ["x1", "x2"]] = np.clip([r.x1 + dx, r.x2 + dx], 0, W)
                d.loc[i, ["y1", "y2"]] = np.clip([r.y1 + dy, r.y2 + dy], 0, H)
                if (d.loc[i, "x2"] - d.loc[i, "x1"]) < 4 or (d.loc[i, "y2"] - d.loc[i, "y1"]) < 4:
                    d.loc[i, ["x1", "y1", "x2", "y2"]] = r[["x1", "y1", "x2", "y2"]].values
                    continue
                rec(i, "jitter", box=d.loc[i, ["x1", "y1", "x2", "y2"]].values)
                used[i] = True
            elif etype == "label_swap":
                new = SWAP[r.label]
                d.loc[i, "label"] = new
                rec(i, "label_swap", "primary", label=new)
                rec(i, "label_swap", "secondary", label=r.label)  # the original label row lost a detection
                used[i] = True
            elif etype == "size_error":
                lo, hi = c["size_scale"]
                s = rng.uniform(lo, 0.75) if rng.random() < 0.5 else rng.uniform(1.33, hi)
                cx, cy = (r.x1 + r.x2) / 2, (r.y1 + r.y2) / 2
                hw, hh = (r.x2 - r.x1) * s / 2, (r.y2 - r.y1) * s / 2
                d.loc[i, ["x1", "y1", "x2", "y2"]] = [max(cx - hw, 0), max(cy - hh, 0), min(cx + hw, W), min(cy + hh, H)]
                rec(i, "size_error", box=d.loc[i, ["x1", "y1", "x2", "y2"]].values)
                used[i] = True
            picked += 1
            eid += 1

    # false positives: a box mid-field at a random frame
    frames = d[["video_id", "clip_id", "frame_idx"]].drop_duplicates().reset_index(drop=True)
    fps = []
    k = int(round(rates["false_positive"] * n))
    while len(fps) < k:
        fr = frames.loc[rng.integers(len(frames))]
        dist = dists[fr.video_id]
        side = rng.uniform(*c["fp_size_frac"]) * W
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2)))
        bw, bh = side * np.sqrt(ar), side / np.sqrt(ar)
        ys, xs = np.nonzero(dist > max(bw, bh) / 2 + margin + 5)
        if not len(xs):
            continue
        j = rng.integers(len(xs))
        box = [xs[j] - bw / 2, ys[j] - bh / 2, xs[j] + bw / 2, ys[j] + bh / 2]
        lab = cfg["tool_labels"][int(rng.integers(2))]
        fps.append({"video_id": fr.video_id, "clip_id": fr.clip_id, "frame_idx": int(fr.frame_idx), "label": lab,
                    "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3]})
        errors.append({"error_id": eid, "error_type": "false_positive", "role": "primary", **{kk: fps[-1][kk] for kk in
                       ("video_id", "clip_id", "frame_idx", "label", "x1", "y1", "x2", "y2")}, "orig_det_id": -1})
        eid += 1

    out = d[d.keep].copy()
    out["orig_det_id"] = out.det_id
    out = pd.concat([out, pd.DataFrame(fps).assign(orig_det_id=-1)], ignore_index=True)
    out = out[["video_id", "clip_id", "frame_idx", "label", "instance_id", "x1", "y1", "x2", "y2", "orig_det_id"]]
    out = finish_dets(out, "gt_corrupted")
    err = pd.DataFrame(errors)
    err["frame_idx"] = err.frame_idx.astype(int)
    return out, err


def run(cfg, force=False):
    out = work(cfg, "detections_gt_corrupted.parquet")
    eout = work(cfg, "errors.parquet")
    if skip(out, force) and eout.exists():
        return
    tracks = pd.read_parquet(work(cfg, "tracks_gt.parquet"))
    dists = {v: edge_distance(cfg, v) for v in tracks.video_id.unique()}
    det, err = corrupt(tracks, cfg, dists)
    det.to_parquet(out, index=False)
    err.to_parquet(eout, index=False)
    print(f"{len(det)} corrupted detections; errors by type:",
          err.drop_duplicates("error_id").error_type.value_counts().to_dict(), f"; affected rows {len(err)}")


if __name__ == "__main__":
    a = cli("Corruption harness", source=False).parse_args()
    with Timer("corrupt"):
        run(load_cfg(), a.force)
