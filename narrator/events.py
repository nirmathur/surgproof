"""Per-clip event timeline from the verified label table + tracker detections.

python -m narrator.events      -> narrator/out/events.jsonl (one event per line)

Event: {event_id "ev:<clip>:<n>", clip_id, video_id, type, label, t_start, t_end, status,
        frames [..], label_keys ["lab:<clip>:<frame>:<label>"], boxes {frame: [[x1,y1,x2,y2]]},
        value, source}
types: tool_enter, tool_exit, tool_moving, tool_still, anatomy_confirmed, anatomy_uncertain,
       anatomy_rejected, blood_appears, blood_grows, smoke_hint
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from narrator import cfg as C
from narrator import data as D

TOOLS = ["grasper", "l_hook"]
ANATOMY = ["liver", "gallbladder", "cystic_duct"]


def tracker_source() -> str:
    for s in ("gemini", "gt_corrupted"):
        if C.path("tracker_detections", source=s).exists():
            return s
    return "none"


def load_tracker():
    src = tracker_source()
    det = pd.read_parquet(C.path("tracker_detections", source=src)) if src != "none" else pd.DataFrame()
    flags = pd.read_parquet(C.path("tracker_flags")) if C.path("tracker_flags").exists() else pd.DataFrame()
    return src, det, flags


def label_key(clip_id, frame_idx, label) -> str:
    return f"lab:{clip_id}:{int(frame_idx)}:{label}"


def _runs(frames: list[int], on: list[bool]):
    """Contiguous runs of True over the sampled frames -> [(first, last)]."""
    runs, start = [], None
    for f, o in zip(frames, on):
        if o and start is None:
            start = f
        if not o and start is not None:
            runs.append((start, prev))
            start = None
        prev = f
    if start is not None:
        runs.append((start, frames[-1]))
    return runs


def clip_events(clip, labels: pd.DataFrame, det: pd.DataFrame, flags: pd.DataFrame,
                lab_source: str, trk_source: str) -> list[dict]:
    ec = C.cfg()["events"]
    frames = D.sampled_frames(clip.first, clip.n_frames)
    L = labels[labels.clip_id == clip.clip_id]
    ev = []

    def add(type_, label, t0, t1, status, fr, value=None, src=lab_source):
        sub = L[(L.label == label) & L.frame_idx.between(t0, t1)]
        boxes = {int(r.frame_idx): json.loads(r.boxes) for r in sub.itertuples() if r.present}
        ev.append(dict(clip_id=clip.clip_id, video_id=clip.video_id, type=type_, label=label,
                       t_start=int(t0), t_end=int(t1), status=status, frames=[int(x) for x in fr],
                       label_keys=[label_key(clip.clip_id, r.frame_idx, label) for r in sub.itertuples()],
                       boxes=boxes, value=value, source=src))

    def present_on(label, statuses=("accepted", "uncertain")):
        s = L[(L.label == label) & L.present & L.status.isin(statuses)]
        fs = set(s.frame_idx)
        return [f in fs for f in frames]

    # tools: enter / exit, moving / still
    for tool in TOOLS:
        on = present_on(tool)
        for a, b in _runs(frames, on):
            n = sum(1 for f in frames if a <= f <= b)
            if n < ec["min_event_frames"]:
                continue
            st = L[(L.label == tool) & L.frame_idx.between(a, b) & L.present]
            status = "uncertain" if (st.status == "uncertain").mean() > 0.5 else "accepted"
            if a > frames[0]:
                add("tool_enter", tool, a, a, status, [a])
            if b < frames[-1]:
                add("tool_exit", tool, b, b, status, [b])
            add("tool_present", tool, a, b, status, [f for f in frames if a <= f <= b])
            if len(det):
                d = det[(det.clip_id == clip.clip_id) & (det.label == tool) & det.frame_idx.between(a, b)]
                if len(d) > 3:
                    c = d.groupby("frame_idx")[["cx", "cy"]].median().sort_index()
                    sp = np.hypot(np.diff(c.cx), np.diff(c.cy)) / np.maximum(1, np.diff(c.index.values))
                    speed = float(np.median(sp)) if len(sp) else 0.0
                    add("tool_moving" if speed > ec["moving_px"] else "tool_still", tool, a, b, status,
                        [f for f in frames if a <= f <= b], value=round(speed, 2), src=f"tracker:{trk_source}")

    # anatomy confirmed / uncertain / rejected
    for lab in ANATOMY:
        sub = L[L.label == lab]
        if sub.empty:
            continue
        acc = sub[sub.present & (sub.status == "accepted")]
        unc = sub[sub.present & (sub.status == "uncertain")]
        rej = sub[sub.status == "rejected"]
        for kind, s in (("anatomy_confirmed", acc), ("anatomy_uncertain", unc), ("anatomy_rejected", rej)):
            if len(s) >= ec["min_event_frames"] or (kind != "anatomy_confirmed" and len(s) >= 1):
                fr = sorted(s.frame_idx.astype(int))
                add(kind, lab, fr[0], fr[-1], kind.split("_")[1], fr)

    # blood appears / grows
    b = L[(L.label == "blood") & L.present].sort_values("frame_idx")
    if len(b) >= ec["min_event_frames"]:
        fr = list(b.frame_idx.astype(int))
        area = [sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in json.loads(bx)) for bx in b.boxes]
        status = "uncertain" if (b.status == "uncertain").mean() > 0.5 else "accepted"
        add("blood_appears", "blood", fr[0], fr[-1], status, fr, value=int(area[0]))
        if area[0] > 0 and max(area) / area[0] >= ec["blood_growth"]:
            add("blood_grows", "blood", fr[0], fr[int(np.argmax(area))], status, fr,
                value=round(max(area) / area[0], 2))

    # smoke hint: hook present + low contrast frames in the tracker quality signals (heuristic)
    if len(flags) and "z_contrast" in flags:
        fq = flags[(flags.clip_id == clip.clip_id) & (flags.label == "l_hook") & (flags.n_det > 0)
                   & (flags.z_contrast < -1.5)]
        fr = sorted(set(fq.frame_idx.astype(int)) & set(frames))
        if len(fr) >= ec["min_event_frames"]:
            add("smoke_hint", "l_hook", fr[0], fr[-1], "heuristic", fr, src=f"tracker:{trk_source}")

    for i, e in enumerate(sorted(ev, key=lambda e: (e["t_start"], e["type"]))):
        e["event_id"] = f"ev:{clip.clip_id}:{i}"
    return sorted(ev, key=lambda e: (e["t_start"], e["type"]))


def build(force: bool = False) -> dict[str, list[dict]]:
    p = C.out_dir() / "events.jsonl"
    if p.exists() and not force:
        out = defaultdict(list)
        for line in p.read_text().splitlines():
            e = json.loads(line)
            out[e["clip_id"]].append(e)
        return dict(out)
    labels = D.load_labels()
    lab_source = str(labels.source.iloc[0]) if len(labels) else "none"
    trk_source, det, flags = load_tracker()
    out = {}
    for clip in D.list_clips().itertuples():
        out[clip.clip_id] = clip_events(clip, labels, det, flags, lab_source, trk_source)
    with open(p, "w") as f:
        for evs in out.values():
            for e in evs:
                f.write(json.dumps(e) + "\n")
    return out


if __name__ == "__main__":
    import collections
    ev = build(force=True)
    n = sum(len(v) for v in ev.values())
    print(f"clips={len(ev)} events={n}", collections.Counter(e["type"] for v in ev.values() for e in v))
