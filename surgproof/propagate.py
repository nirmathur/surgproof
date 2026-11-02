"""Stage 6: carry keyframe masks across every source-video frame with optical flow.

For each gap between keyframes a and b, masks are warped forward from a and backward from b with Farneback flow
and blended by distance to each end. Keyframes always stay the AI's own masks; everything else is flagged
`propagated: true`. Then a 5-frame temporal majority on masks and a 7-frame moving average on boxes.

Inputs: meta.json, masks.jsonl, masks/, the source video.
Outputs: tracks/<frame_index>_<label>.png (propagated frames only, only where the label is present),
tracks.jsonl {frame_index, label, box, box_raw, propagated, status, source_kfs, keyframe, t}.

`box` is the smoothed box for playback; `box_raw` is the exact bound of that frame's mask.
A propagated row is AMBER when any keyframe that fed it is AMBER, so an unreviewed label cannot launder itself
into accepted frames. Flow never crosses a detected scene cut, and `shot` counts the cuts so far.

Every propagated frame is cross-checked: `fb_overlap` is the IoU of the mask carried forward with the mask carried
backward, before blending. `flow_check` is "ok" at or above MIN_FB_OVERLAP, "low_overlap" below it, and "none" when
there is no second direction to compare (past the first or last keyframe, or beyond a cut). Only "ok" can be accepted.
A keyframe's own row has flow_check "keyframe". Overlap measures agreement between two flows, not correctness.

Frames are processed as a stream, so memory depends on the keyframe gap, not the video length.
"""
from __future__ import annotations

import collections
from pathlib import Path

import cv2
import numpy as np

from .masks import (ACCEPT, AMBER, frame_size, mask_box, progress_fn, read_json, read_jsonl, read_mask,
                    track_path, write_jsonl, write_mask)

WORK_SIDE = 480           # flow working resolution (longest side, px)
MAJORITY_WINDOW = 5
BOX_WINDOW = 7
CUT_MEAN_DIFF = 40.0      # mean absolute grey difference (0-255) between frames that marks a scene cut
CUT_HIST_CORR = 0.35      # or histogram correlation below this
MIN_FB_OVERLAP = 0.5      # forward/backward mask IoU below this sends the frame to AMBER
MIN_SPECK_FRAC = 0.05     # drop fragments smaller than this fraction of the label's largest piece
FLOW_ARGS = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=3, poly_n=5, poly_sigma=1.1, flags=0)

VIDEO_STEMS = ("source", "input", "video", "original")
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}


def find_video(job, meta, video=None):
    """The source video: explicit argument, then meta.json, then a conventional name in the job dir."""
    job = Path(job)
    candidates = [video, meta.get("source_video"), meta.get("video"), meta.get("source")]
    for c in candidates:
        if c:
            p = Path(c)
            p = p if p.is_absolute() else job / p
            if p.exists():
                return p
    for stem in VIDEO_STEMS:
        for p in sorted(job.glob(f"{stem}.*")):
            if p.suffix.lower() in VIDEO_EXTS:
                return p
    raise FileNotFoundError(f"no source video for job {job}: set meta.json source_video or pass video=")


def flow(a, b):
    """Dense flow from grey frame a to grey frame b: for each pixel of a, where it lands in b."""
    return cv2.calcOpticalFlowFarneback(a, b, None, **FLOW_ARGS)


def warp(mask, flow_to_source, grid):
    """Pull a soft mask across one step. `flow_to_source` maps each destination pixel to where it came from."""
    gx, gy = grid
    return cv2.remap(mask, (gx + flow_to_source[..., 0]).astype(np.float32),
                     (gy + flow_to_source[..., 1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def is_cut(a, b):
    """True when consecutive grey frames look like a scene change, not motion."""
    if float(np.mean(cv2.absdiff(a, b))) > CUT_MEAN_DIFF:
        return True
    ha = cv2.calcHist([a], [0], None, [32], [0, 256])
    hb = cv2.calcHist([b], [0], None, [32], [0, 256])
    return cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL) < CUT_HIST_CORR


class GreyReader:
    """Sequential reader of working-resolution grey frames."""

    def __init__(self, path, size):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise OSError(f"could not open video {path}")
        self.size = size
        self.pos = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        return cv2.cvtColor(cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)

    def close(self):
        self.cap.release()


def work_size(w, h, side=WORK_SIDE):
    scale = min(1.0, side / max(w, h))
    return max(16, int(round(w * scale))), max(16, int(round(h * scale)))


def _despeck(mask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    return np.isin(lab, 1 + np.where(areas >= MIN_SPECK_FRAC * areas.max())[0])


def _overlap(forward_soft, backward_soft):
    """IoU of the two directions. A direction that carries no mask counts as an empty one, so overlap is 0."""
    if forward_soft is None or backward_soft is None:
        return 0.0
    f, b = forward_soft > 0.5, backward_soft > 0.5
    union = int((f | b).sum())
    return round(float((f & b).sum()) / union, 4) if union else 0.0


def _propagate_gap(greys, a_masks, b_masks):
    """Soft blended masks for one run of consecutive frames.

    greys: grey frames in order. a_masks / b_masks: {label: float32 mask} anchoring greys[0] / greys[-1], or None
    when that end is not a keyframe, in which case that end is itself a frame to fill.
    Returns ({index_in_greys: {label: (soft, contributors, overlap)}}, cuts), with contributors drawn from 'a', 'b',
    overlap the forward/backward IoU (None when only one direction reaches the frame), and cuts one bool per step.
    """
    n = len(greys) - 1
    if n < 1:
        return {}, []
    h, w = greys[0].shape
    grid = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    cuts = [is_cut(greys[i], greys[i + 1]) for i in range(n)]
    forward_valid = [False] * (n + 1)
    backward_valid = [False] * (n + 1)
    forward = [{} for _ in range(n + 1)]
    backward = [{} for _ in range(n + 1)]

    if a_masks is not None:
        forward_valid[0] = True
        cur = dict(a_masks)
        for i in range(n):
            if cuts[i]:
                break
            forward_valid[i + 1] = True
            if cur:
                # Moving forward, each pixel of frame i+1 looks up where it was in frame i.
                back = flow(greys[i + 1], greys[i])
                cur = {lab: warp(m, back, grid) for lab, m in cur.items()}
            forward[i + 1] = cur

    if b_masks is not None:
        backward_valid[n] = True
        cur = dict(b_masks)
        for i in range(n - 1, -1, -1):
            if cuts[i]:
                break
            backward_valid[i] = True
            if cur:
                # Moving backward, each pixel of frame i looks up where it goes in frame i+1.
                fwd = flow(greys[i], greys[i + 1])
                cur = {lab: warp(m, fwd, grid) for lab, m in cur.items()}
            backward[i] = cur

    out = {}
    first = 1 if a_masks is not None else 0
    last = n - 1 if b_masks is not None else n
    for i in range(first, last + 1):
        s = i / n
        wf = (1 - s) if forward_valid[i] else 0.0
        wb = s if backward_valid[i] else 0.0
        # With one end missing or behind a cut, the other end carries the whole frame.
        if a_masks is None or not forward_valid[i]:
            wf, wb = 0.0, (1.0 if backward_valid[i] else 0.0)
        elif b_masks is None or not backward_valid[i]:
            wf, wb = 1.0, 0.0
        frame = {}
        for lab in set(forward[i]) | set(backward[i]):
            f, b = forward[i].get(lab), backward[i].get(lab)
            # A valid end where the label is absent votes 0: an object present at only one end
            # disappears near the midpoint instead of being dragged across the whole gap.
            who = ([] if f is None or wf == 0 else ["a"]) + ([] if b is None or wb == 0 else ["b"])
            if who:
                soft = (wf * f if "a" in who else 0.0) + (wb * b if "b" in who else 0.0)
                both = a_masks is not None and b_masks is not None and forward_valid[i] and backward_valid[i]
                frame[lab] = (soft, who, _overlap(f, b) if both else None)
        out[i] = frame
    return out, cuts


class _Emitter:
    """Applies the temporal majority over a sliding window and writes each finished frame."""

    def __init__(self, job, size, work, fps):
        self.job, self.size, self.work, self.fps = Path(job), size, work, fps
        self.half = MAJORITY_WINDOW // 2
        self.items = []     # (frame_index, kf, shot, {label: (work_mask, status, sources, full_or_None, overlap)})
        self.next = 0       # position in items of the next frame to finish
        self.rows = []

    def push(self, frame_index, kf, labels, shot=0):
        self.items.append((frame_index, kf, shot, labels))
        while self.next + self.half < len(self.items):
            self._finish_next()

    def flush(self):
        while self.next < len(self.items):
            self._finish_next()

    def _finish_next(self):
        pos = self.next
        frame_index, kf, shot, labels = self.items[pos]
        for label, (_, status, sources, full, overlap) in sorted(labels.items()):
            if kf is not None:
                self._write_keyframe(frame_index, kf, label, status, full, shot)
            else:
                self._write_propagated(frame_index, label, status, sources, pos, overlap, shot)
        # A label only the neighbours carry could win a majority here, but this frame would have no source
        # to attribute it to, so it is not synthesised.
        self.next += 1
        while self.next > self.half:
            self.items.pop(0)
            self.next -= 1

    def _t(self, frame_index):
        return round(frame_index / self.fps, 4) if self.fps else None

    def _write_keyframe(self, frame_index, kf, label, status, full, shot):
        box = mask_box(full)
        if box is not None:
            self.rows.append({"frame_index": frame_index, "label": label, "box": box, "box_raw": box,
                              "propagated": False, "status": status, "source_kfs": [kf], "keyframe": kf,
                              "t": self._t(frame_index), "flow_check": "keyframe", "fb_overlap": None, "shot": shot})

    def _write_propagated(self, frame_index, label, status, sources, pos, overlap, shot):
        votes = []
        for j in range(max(0, pos - self.half), min(len(self.items), pos + self.half + 1)):
            entry = self.items[j][3].get(label)
            votes.append(entry[0] if entry is not None else np.zeros(self.work[::-1], bool))
        majority = np.sum(votes, axis=0) * 2 > len(votes)
        if not majority.any():
            return
        mask = cv2.resize(_despeck(majority).astype(np.float32), self.size, interpolation=cv2.INTER_LINEAR) > 0.5
        box = mask_box(mask)
        if box is None:
            return
        write_mask(track_path(self.job, frame_index, label), mask)
        check = "none" if overlap is None else ("ok" if overlap >= MIN_FB_OVERLAP else "low_overlap")
        self.rows.append({"frame_index": frame_index, "label": label, "box": box, "box_raw": box,
                          "propagated": True, "status": status if check == "ok" else AMBER, "source_kfs": sources,
                          "keyframe": None, "t": self._t(frame_index), "flow_check": check, "fb_overlap": overlap,
                          "shot": shot})


def smooth_boxes(rows, window=BOX_WINDOW):
    """Centred moving average of box coords over frames where the label exists. `box_raw` stays exact."""
    half = window // 2
    by_label = collections.defaultdict(dict)
    for r in rows:
        by_label[r["label"]][r["frame_index"]] = r["box_raw"]
    for r in rows:
        boxes = by_label[r["label"]]
        win = [boxes[i] for i in range(r["frame_index"] - half, r["frame_index"] + half + 1) if i in boxes]
        r["box"] = [round(float(np.mean([b[k] for b in win])), 1) for k in range(4)]
    return rows


def run(job_dir, video=None, on_progress=None, work_side=WORK_SIDE):
    """Run stage 6 over the whole source video. Returns a summary with counts."""
    job = Path(job_dir)
    progress = progress_fn(job, "propagate", on_progress)
    meta = read_json(job / "meta.json")
    video_path = find_video(job, meta, video)
    w, h = frame_size(job, meta)
    fps = float(meta.get("fps") or 0) or None
    work = work_size(w, h, work_side)
    n_total = int(meta.get("n_frames") or 0)

    tracks_dir = job / "tracks"
    tracks_dir.mkdir(exist_ok=True)
    for stale in tracks_dir.glob("*.png"):
        stale.unlink()

    keyframes = sorted(meta["keyframes"], key=lambda k: k["frame_index"])
    if not keyframes:
        write_jsonl(job / "tracks.jsonl", [])
        progress(1.0, "no keyframes")
        return {"n_rows": 0, "n_propagated": 0, "n_frames_read": 0, "n_cuts": 0, "n_keyframes_past_end": 0}

    by_kf = collections.defaultdict(dict)
    for r in read_jsonl(job / "masks.jsonl"):
        if r.get("path") and r.get("status") in (ACCEPT, AMBER):
            by_kf[r["kf"]][r["label"]] = r

    def load_anchor(kf):
        full, soft = {}, {}
        for label, r in by_kf.get(kf, {}).items():
            m = read_mask(job / r["path"])
            if m.shape != (h, w):
                m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            full[label] = m
            soft[label] = cv2.resize(m.astype(np.float32), work, interpolation=cv2.INTER_AREA)
        return full, soft

    # Extrapolate past the first and last keyframe by at most one typical gap.
    gaps = [b["frame_index"] - a["frame_index"] for a, b in zip(keyframes, keyframes[1:])]
    tail_limit = int(np.median(gaps)) if gaps else (int(round(fps / float(meta.get("kf_fps") or 1))) if fps else 30)

    reader = GreyReader(video_path, work)
    emitter = _Emitter(job, (w, h), work, fps)
    n_cuts, n_read = 0, 0

    def read_until(frame_index, keep=None):
        """Read frames up to and including frame_index, returning the last `keep` of them (all when None)."""
        nonlocal n_read
        kept = collections.deque(maxlen=keep)
        while reader.pos <= frame_index:
            g = reader.read()
            if g is None:
                break
            n_read += 1
            kept.append(g)
        return list(kept)

    def emit_run(greys, first_index, a, b, a_soft, b_soft):
        nonlocal n_cuts
        frames, cuts = _propagate_gap(greys, a_soft, b_soft)
        for i in sorted(frames):
            out = {}
            for label, (soft, who, overlap) in frames[i].items():
                sources = [(a if c == "a" else b)["kf"] for c in who]
                statuses = [by_kf[kf][label]["status"] for kf in sources]
                out[label] = (soft > 0.5, AMBER if AMBER in statuses else ACCEPT, sources, None, overlap)
            emitter.push(first_index + i, None, out, shot=n_cuts + sum(cuts[:i]))
        n_cuts += sum(cuts)

    def emit_keyframe(k, full, soft):
        labels = {lab: (soft[lab] > 0.5, by_kf[k["kf"]][lab]["status"], [k["kf"]], full[lab], None) for lab in full}
        emitter.push(k["frame_index"], k["kf"], labels, shot=n_cuts)

    def report():
        progress(min(0.99, n_read / max(1, n_total)), f"tracked {n_read}/{n_total} frames")

    # Frames before the first keyframe: backward only, capped at one gap.
    first = keyframes[0]
    full, soft = load_anchor(first["kf"])
    lead = read_until(first["frame_index"], keep=tail_limit + 1)
    if not lead:
        reader.close()
        raise OSError(f"could not decode frame {first['frame_index']} from {video_path}")
    if reader.pos <= first["frame_index"]:
        reader.close()
        raise OSError(f"{video_path} ended at frame {reader.pos}, before the first keyframe {first['kf']}")
    if len(lead) > 1:
        emit_run(lead, first["frame_index"] - (len(lead) - 1), None, first, None, soft)
    emit_keyframe(first, full, soft)
    report()

    prev, prev_soft, prev_grey = first, soft, lead[-1]
    past_end = []
    for k in keyframes[1:]:
        full, soft = load_anchor(k["kf"])
        new = read_until(k["frame_index"])
        if reader.pos <= k["frame_index"]:
            # The container promised more frames than it decodes. The AI's mask still stands, nothing to flow into it.
            past_end.append(k)
            continue
        emit_run([prev_grey] + new, prev["frame_index"], prev, k, prev_soft, soft)
        emit_keyframe(k, full, soft)
        prev, prev_soft, prev_grey = k, soft, new[-1]
        report()

    # Frames after the last decodable keyframe: forward only, capped at one gap.
    tail = [prev_grey]
    while True:
        g = reader.read()
        if g is None:
            break
        n_read += 1
        if len(tail) <= tail_limit:
            tail.append(g)
    reader.close()
    if len(tail) > 1:
        emit_run(tail, prev["frame_index"], prev, None, prev_soft, None)
    for k in past_end:
        full, soft = load_anchor(k["kf"])
        emit_keyframe(k, full, soft)

    emitter.flush()
    rows = smooth_boxes(emitter.rows)
    rows.sort(key=lambda r: (r["frame_index"], r["label"]))
    write_jsonl(job / "tracks.jsonl", rows)
    n_prop = sum(1 for r in rows if r["propagated"])
    progress(1.0, f"{n_prop} propagated masks across {n_read} frames, {n_cuts} scene cuts")
    return {"n_rows": len(rows), "n_propagated": n_prop, "n_frames_read": n_read, "n_cuts": n_cuts,
            "n_keyframes_past_end": len(past_end)}
