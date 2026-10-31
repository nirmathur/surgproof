"""Step 8: score rows -> flags.parquet, and build the re-check queue.

  python -m tracker.recheck --source gt_corrupted [--budget 0.10]

Outputs (out/tracker/):
  flags_<source>.parquet         one row per (frame, tool label): signals, suspicion, flagged, top_signal
  recheck_queue_<source>.parquet budget B of rows: 80% highest suspicion, 20% seeded uniform controls
Also copies flags_<source>.parquet to out/flags.parquet (the file the partner joins on).
"""
import json
import shutil

import numpy as np
import pandas as pd

from tracker.common import KEYS, ROOT, Timer, cli, load_cfg, load_frames, p, work
from tracker.evaluate import load_model, score


def top_signal(contrib, cfg):
    cols = [c for c in contrib.columns if c != "present" and c.replace("z_", "") not in cfg["signals"]["quality"]]
    c = contrib[cols]
    best = c.idxmax(axis=1).str.replace("z_", "", regex=False)
    best[c.max(axis=1) <= 0] = "none"
    return best


def build_flags(cfg, source):
    rows = pd.read_parquet(work(cfg, f"signals_{source}.parquet"))
    model = load_model(work(cfg, "suspicion_model.json"))
    sus, contrib = score(rows, cfg, model)
    rows["suspicion"] = sus
    rows["flagged"] = sus > model["flag_threshold"]
    rows["top_signal"] = top_signal(contrib, cfg).values
    rows["source"] = source
    sig = cfg["signals"]
    cols = KEYS + ["split", "source", "n_det"] + list(dict.fromkeys(sig["zscore"] + sig["raw"])) + \
        [f"z_{k}" for k in sig["zscore"]] + ["suspicion", "flagged", "top_signal"]
    return rows[cols]


def _box(r):
    return [round(float(r.x1), 1), round(float(r.y1), 1), round(float(r.x2), 1), round(float(r.y2), 1)]


def interpolated_boxes(tr, f, max_gap):
    """For each track of this (clip, label) near frame f: box at f interpolated from nearest detections != f."""
    out = []
    for t, g in tr.groupby("track_id"):
        g = g[(g.frame_idx != f) & ((g.frame_idx - f).abs() <= max_gap + 1)]
        before, after = g[g.frame_idx < f].tail(1), g[g.frame_idx > f].head(1)
        if len(before) and len(after):
            a, b = before.iloc[0], after.iloc[0]
            w = (f - a.frame_idx) / (b.frame_idx - a.frame_idx)
            box = [(1 - w) * x + w * y for x, y in zip(_box(a), _box(b))]
            out.append({"track_id": int(t), "box": [round(v, 1) for v in box], "from_frames": [int(a.frame_idx), int(b.frame_idx)]})
        elif len(before) or len(after):
            a = (before if len(before) else after).iloc[0]
            out.append({"track_id": int(t), "box": _box(a), "from_frames": [int(a.frame_idx)]})
    return out


def payload(r, rtype, tracks, paths, cfg):
    clip, f, lab = r.clip_id, int(r.frame_idx), r.label
    here = tracks[(tracks.clip_id == clip) & (tracks.frame_idx == f) & (tracks.label == lab)]
    pl = {"frame_path": paths.get((clip, f)), "label": lab,
          "boxes": [{"track_id": int(d.track_id), "box": _box(d)} for d in here.itertuples()]}
    if rtype == "verify_at_interpolated_box":
        tr = tracks[(tracks.clip_id == clip) & (tracks.label == lab)]
        pl["interpolated"] = interpolated_boxes(tr, f, cfg["track"]["max_gap"])
    elif rtype == "disambiguate_pair":
        k = cfg["signals"]["swap_window"]
        other = tracks[(tracks.clip_id == clip) & (tracks.label != lab) & ((tracks.frame_idx - f).abs() <= k)]
        if len(here) and len(other):
            from tracker.common import iou
            ov = iou(here[["x1", "y1", "x2", "y2"]].values, other[["x1", "y1", "x2", "y2"]].values).max(0)
            o = other.iloc[int(np.argmax(ov))]
            pl["other"] = {"label": o.label, "frame_idx": int(o.frame_idx), "box": _box(o), "track_id": int(o.track_id),
                           "frame_path": paths.get((clip, int(o.frame_idx)))}
    elif rtype == "frame_strip":
        pl["strip"] = []
        for ff in (f - 1, f, f + 1):
            if (clip, ff) in paths:
                bb = tracks[(tracks.clip_id == clip) & (tracks.frame_idx == ff) & (tracks.label == lab)]
                pl["strip"].append({"frame_idx": ff, "frame_path": paths[(clip, ff)], "boxes": [_box(d) for d in bb.itertuples()]})
    return json.dumps(pl)


def build_queue(cfg, flags, source, budget):
    rc = cfg["recheck"]
    rng = np.random.default_rng(cfg["seed"])
    n = int(round(budget * len(flags)))
    n_ctrl = int(round(rc["control_frac"] * n))
    order = np.lexsort((rng.random(len(flags)), -flags.suspicion.values))
    top = order[:n - n_ctrl]
    rest = np.setdiff1d(np.arange(len(flags)), top)
    ctrl = rng.choice(rest, n_ctrl, replace=False)
    q = pd.concat([flags.iloc[top].assign(is_control=False), flags.iloc[ctrl].assign(is_control=True)])
    q["recheck_type"] = q.top_signal.map(rc["type_map"]).fillna("escalate")
    tracks = pd.read_parquet(work(cfg, f"tracks_{source}.parquet"))
    fr = load_frames(cfg)
    paths = {(c, int(f)): pth for c, f, pth in zip(fr.clip_id, fr.frame_idx, fr.image_path)}
    q["payload"] = [payload(r, t, tracks, paths, cfg) for r, t in zip(q.itertuples(), q.recheck_type)]
    q["priority"] = np.arange(len(q))
    return q[KEYS + ["priority", "suspicion", "top_signal", "recheck_type", "is_control", "payload"]]


def run(cfg, source, budget):
    flags = build_flags(cfg, source)
    fo = work(cfg, f"flags_{source}.parquet")
    flags.to_parquet(fo, index=False)
    shutil.copy(fo, p(cfg, "work").parent / "flags.parquet")
    print(f"flags: {len(flags)} rows, flagged {flags.flagged.mean():.1%}; top_signal of flagged: "
          f"{flags[flags.flagged].top_signal.value_counts().to_dict()}")
    q = build_queue(cfg, flags, source, budget)
    q.to_parquet(work(cfg, f"recheck_queue_{source}.parquet"), index=False)
    print(f"queue: {len(q)} rows (budget {budget:.0%}), controls {int(q.is_control.sum())}; types "
          f"{q.recheck_type.value_counts().to_dict()}")
    ep = work(cfg, "errors.parquet")
    if source == "gt_corrupted" and ep.exists():
        e = pd.read_parquet(ep)
        hit = e.merge(q[KEYS + ["is_control"]], on=KEYS)
        n_err = e.error_id.nunique()
        print(f"  errors covered by queue: targeted {hit[~hit.is_control].error_id.nunique() / n_err:.1%}, "
              f"control {hit[hit.is_control].error_id.nunique() / n_err:.1%} of {n_err}")


if __name__ == "__main__":
    ap = cli("Score rows and build re-check queue")
    ap.add_argument("--budget", type=float, default=None)
    a = ap.parse_args()
    cfg = load_cfg()
    with Timer("recheck"):
        run(cfg, a.source, a.budget if a.budget is not None else cfg["recheck"]["budget"])
