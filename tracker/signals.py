"""Step 5: signals per detection, aggregated to one row per (frame, tool label).

Outputs:
  det_signals_<source>.parquet   per detection (with track_id), raw signal values
  signals_<source>.parquet       per (video, clip, frame, label): raw values (max over instances),
                                 z_<signal> robust z-scores per video, and image quality
Motion is in camera-compensated coordinates and normalised by box diagonal. For per-detection jump signals
("in" = vs previous detection in track, "out" = vs next) we take min(in, out) when both exist, so a single bad
frame scores high but its neighbours do not; at track ends only the available side is used.
"""
import numpy as np
import pandas as pd

from tracker.common import KEYS, Timer, box_edge_gap, cli, edge_distance, inter_frac, iou, load_cfg, load_frames, robust_z, skip, work

DET_SIGNALS = ["teleport", "jerk", "size_jump", "aspect_jump", "shaft_to_edge", "swap", "flicker",
               "exit_midframe", "entry_midframe", "short_track", "tool_count"]
MISS_SIGNALS = ["flicker", "exit_midframe", "entry_midframe", "tool_count"]  # can fire on rows without a detection
QUALITY = ["blur", "contrast", "brightness", "specular"]
ALL_SIGNALS = DET_SIGNALS + QUALITY


def _nanmin2(a, b):
    out = np.where(np.isnan(a), b, np.where(np.isnan(b), a, np.minimum(a, b)))
    return out


def track_signals(f, box, cfg):
    """Per-track motion signals. f (n,) frames, box (n,4) compensated. Returns dict of (n,) arrays."""
    n = len(f)
    c = (box[:, :2] + box[:, 2:]) / 2
    w = np.maximum(box[:, 2] - box[:, 0], 1.0)
    h = np.maximum(box[:, 3] - box[:, 1], 1.0)
    diag = np.hypot(w, h)
    nan = np.full(n, np.nan)
    tele_in, size_in, asp_in = nan.copy(), nan.copy(), nan.copy()
    if n >= 2:
        df = np.diff(f).astype(float)
        dd = (diag[1:] + diag[:-1]) / 2
        tele_in[1:] = np.linalg.norm(np.diff(c, axis=0), axis=1) / dd / df
        size_in[1:] = np.abs(np.diff(np.log(w * h)))
        asp_in[1:] = np.abs(np.diff(np.log(w / h)))
    shift = lambda a: np.append(a[1:], np.nan)  # value of the "out" step
    jerk = nan.copy()
    if n >= 3:
        t = (f[1:-1] - f[:-2]) / (f[2:] - f[:-2])
        pred = c[:-2] + (c[2:] - c[:-2]) * t[:, None]
        jerk[1:-1] = np.linalg.norm(c[1:-1] - pred, axis=1) / diag[1:-1]
    return {"teleport": _nanmin2(tele_in, shift(tele_in)), "jerk": jerk,
            "size_jump": _nanmin2(size_in, shift(size_in)), "aspect_jump": _nanmin2(asp_in, shift(asp_in))}


def clip_signals(d, frames, dist, cfg):
    """d: tracked detections for one clip (all labels). frames: sorted frame indices of the clip.
    Returns (det-level DataFrame, list of extra (frame, label, signal, value) for rows without detections)."""
    s = cfg["signals"]
    d = d.sort_values(["track_id", "frame_idx"]).copy()
    for k in DET_SIGNALS:
        d[k] = np.nan if k in ("teleport", "jerk", "size_jump", "aspect_jump") else 0.0
    f0, f1 = frames.min(), frames.max()
    real = d[["x1", "y1", "x2", "y2"]].values
    comp = d[["cx1", "cy1", "cx2", "cy2"]].values
    fr = d.frame_idx.values
    lab = d.label.values
    tr = d.track_id.values
    d["shaft_to_edge"] = box_edge_gap(dist, real)
    extra = []
    pos = {i: j for j, i in enumerate(d.index)}

    for t, g in d.groupby("track_id", sort=False):
        ii = np.array([pos[i] for i in g.index])
        f = fr[ii]
        for k, v in track_signals(f, comp[ii], cfg).items():
            d.iloc[ii, d.columns.get_loc(k)] = v
        L = lab[ii[0]]
        # flicker: internal gaps (tracker only bridges gaps <= max_gap)
        for a in np.where(np.diff(f) > 1)[0]:
            for gf in range(f[a] + 1, f[a + 1]):
                extra.append((gf, L, "flicker", 1.0))
            for j in (ii[a], ii[a + 1]):
                d.iat[j, d.columns.get_loc("flicker")] = max(d.iat[j, d.columns.get_loc("flicker")], 0.5)
        # exit / entry mid-field without an occluding tool box
        for end, ff, nb in (("exit_midframe", f[-1], f[-1] + 1), ("entry_midframe", f[0], f[0] - 1)):
            if not (f0 <= nb <= f1):
                continue  # track touches the clip boundary: normal
            j = ii[-1] if end == "exit_midframe" else ii[0]
            if d.iat[j, d.columns.get_loc("shaft_to_edge")] <= s["edge_margin_px"]:
                continue  # left/entered through the field edge
            others = (tr != t) & np.isin(fr, [ff, nb])
            if others.any() and inter_frac(real[j], real[others]).max() >= s["occlusion_frac"]:
                continue  # plausibly occluded by another tool
            d.iat[j, d.columns.get_loc(end)] = 1.0
            extra.append((nb, L, end, 1.0))
        if len(f) <= s["short_track_len"] and f[0] > f0 and f[-1] < f1:
            d.iloc[ii, d.columns.get_loc("short_track")] = 1.0

    # swap: box sits where the other label's track is (neighbouring frames) more than where its own label is
    k = s["swap_window"]
    swap = np.zeros(len(d))
    for j in range(len(d)):
        near = (np.abs(fr - fr[j]) <= k) & (fr != fr[j])
        other = near & (lab != lab[j])
        if not other.any():
            continue
        own = near & (lab == lab[j]) & (tr == tr[j])
        o = iou(comp[j], comp[other]).max()
        w = iou(comp[j], comp[own]).max() if own.any() else 0.0
        swap[j] = max(0.0, o - w)
    d["swap"] = swap

    # tool count per frame
    cnt = d.groupby("frame_idx").size()
    nh = d[d.label == "l_hook"].groupby("frame_idx").size()
    excess = (cnt - s["max_tools"]).clip(lower=0).add((nh >= 2).astype(float), fill_value=0)
    d["tool_count"] = d.frame_idx.map(excess).fillna(0.0).values
    for ff, v in excess[excess > 0].items():
        for L in cfg["tool_labels"]:
            extra.append((ff, L, "tool_count", float(v)))
    return d, extra


def run(cfg, source, force=False):
    out = work(cfg, f"signals_{source}.parquet")
    dout = work(cfg, f"det_signals_{source}.parquet")
    if skip(out, force):
        return
    frames = load_frames(cfg)
    tracks = pd.read_parquet(work(cfg, f"tracks_{source}.parquet"))
    dets, extras = [], []
    dists = {}
    for clip, fg in frames.groupby("clip_id"):
        v = fg.video_id.iloc[0]
        if v not in dists:
            dists[v] = edge_distance(cfg, v)
        g = tracks[tracks.clip_id == clip]
        if len(g):
            dd, ex = clip_signals(g, fg.frame_idx.values, dists[v], cfg)
            dets.append(dd)
            extras += [(v, clip, *e) for e in ex]
    det = pd.concat(dets).sort_values("det_id")
    det.to_parquet(dout, index=False)

    # one row per (frame, label)
    rows = frames[["video_id", "clip_id", "frame_idx", "split"]].merge(
        pd.DataFrame({"label": cfg["tool_labels"]}), how="cross")
    agg = det.groupby(KEYS).agg(n_det=("det_id", "size"), **{k: (k, "max") for k in DET_SIGNALS}).reset_index()
    rows = rows.merge(agg, on=KEYS, how="left")
    rows["n_det"] = rows.n_det.fillna(0).astype(int)
    if extras:
        ex = pd.DataFrame(extras, columns=KEYS + ["sig", "val"]).groupby(KEYS + ["sig"]).val.max().unstack()
        rows = rows.merge(ex.add_prefix("x_").reset_index(), on=KEYS, how="left")
        for k in MISS_SIGNALS:
            if f"x_{k}" in rows:
                rows[k] = np.fmax(rows[k].values, rows.pop(f"x_{k}").values)
    q = pd.read_parquet(work(cfg, "quality.parquet"))
    q["blur"] = -np.log1p(q.lap_var)
    rows = rows.merge(q[["clip_id", "frame_idx"] + QUALITY], on=["clip_id", "frame_idx"], how="left")
    for k in ["swap", "flicker", "exit_midframe", "entry_midframe", "short_track", "tool_count"]:
        rows[k] = rows[k].fillna(0.0)
    for k in cfg["signals"]["zscore"]:
        rows[f"z_{k}"] = rows.groupby("video_id")[k].transform(lambda x: pd.Series(robust_z(x.values), index=x.index))
    rows.to_parquet(out, index=False)
    present = rows.n_det > 0
    print(f"{len(rows)} frame-label rows ({present.mean():.1%} with a detection)")
    zt, bt = cfg["signals"]["z_thresh"], cfg["signals"]["binary_thresh"]
    fl = {k: float((rows[f"z_{k}"] > zt).mean()) for k in cfg["signals"]["zscore"]}
    fl.update({k: float((rows[k] > bt).mean()) for k in cfg["signals"]["raw"]})
    print("flag rate per signal on this source:", {k: f"{v:.2%}" for k, v in fl.items()})


if __name__ == "__main__":
    a = cli("Signals").parse_args()
    with Timer("signals"):
        run(load_cfg(), a.source, a.force)
