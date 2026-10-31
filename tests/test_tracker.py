import copy

import cv2
import numpy as np
import pandas as pd
import pytest

from tracker.common import edge_distance, finish_dets, load_cfg, robust_z
from tracker.corrupt import corrupt
from tracker.camera import _quality
from tracker.signals import clip_signals, track_signals
from tracker.track import link

CFG = load_cfg()
H, W = 200, 300


def border_dist():
    m = np.pad(np.ones((H, W), np.uint8), 1)
    return cv2.distanceTransform(m, cv2.DIST_L2, 3)[1:-1, 1:-1]


def box(cx, cy, w=40, h=20):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def make(dets):
    """dets: list of (frame, label, track_id, box). Camera-compensated == real (no camera motion)."""
    d = pd.DataFrame([{"frame_idx": f, "label": l, "track_id": t, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3]}
                      for f, l, t, b in dets])
    for c in ("x1", "y1", "x2", "y2"):
        d["c" + c] = d[c]
    d["det_id"] = np.arange(len(d))
    return d


def edge_tool(f, cx, label="grasper", t=0):
    """A tool box that reaches the left image edge (a real shaft)."""
    return (f, label, t, [0, 90, cx, 110])


FRAMES = np.arange(0, 20)


# ---------- tracking
def test_link_two_crossing_tools_and_gap():
    frames, boxes, truth = [], [], []
    for f in range(30):
        if f in (10, 11):  # tool A missing for 2 frames (<= max_gap) must keep its id
            pass
        else:
            frames.append(f); boxes.append(box(50 + 3 * f, 50)); truth.append(0)
        frames.append(f); boxes.append(box(250 - 3 * f, 150)); truth.append(1)
    tid = link(np.array(frames), np.array(boxes, float), CFG["track"])
    truth = np.array(truth)
    assert len(set(tid)) == 2
    for k in (0, 1):
        assert len(set(tid[truth == k])) == 1


def test_link_breaks_on_teleport_and_long_gap():
    frames = [0, 1, 2, 3, 10, 11]  # gap of 6 > max_gap
    boxes = [box(50, 50)] * 4 + [box(50, 50)] * 2
    tid = link(np.array(frames), np.array(boxes, float), CFG["track"])
    assert tid[0] == tid[3] and tid[4] != tid[3]
    tid = link(np.array([0, 1, 2]), np.array([box(30, 30), box(30, 30), box(280, 190)], float), CFG["track"])
    assert tid[2] != tid[1]


# ---------- per-track motion signals
def test_teleport_jerk_size_aspect_localise_single_bad_frame():
    f = np.arange(9)
    b = np.array([box(50 + 2 * i, 100) for i in f], float)
    b[4] = box(50 + 2 * 4 + 60, 100)  # jitter
    s = track_signals(f, b, CFG)
    assert np.nanargmax(s["teleport"]) == 4 and s["teleport"][4] > 1 and s["teleport"][3] < 0.2
    assert np.nanargmax(s["jerk"]) == 4
    b = np.array([box(50 + 2 * i, 100) for i in f], float)
    b[4] = box(58, 100, w=100, h=20)  # wider: area and aspect change
    s = track_signals(f, b, CFG)
    assert np.nanargmax(s["size_jump"]) == 4 and s["size_jump"][4] > 0.8 and s["size_jump"][3] < 0.05
    assert np.nanargmax(s["aspect_jump"]) == 4


# ---------- clip-level signals
def test_flicker_marks_gap_and_neighbours():
    d = make([edge_tool(f, 80) for f in FRAMES if f not in (8, 9)])
    det, extra = clip_signals(d, FRAMES, border_dist(), CFG)
    gap_rows = {(e[0], e[2]) for e in extra if e[2] == "flicker"}
    assert gap_rows == {(8, "flicker"), (9, "flicker")}
    assert set(det.frame_idx[det.flicker > 0]) == {7, 10}


def test_exit_midframe_vs_edge_and_occlusion():
    mid = [(f, "grasper", 0, box(150, 100)) for f in range(0, 10)]
    det, extra = clip_signals(make(mid), FRAMES, border_dist(), CFG)
    assert det.exit_midframe.iloc[-1] == 1 and (10, "grasper", "exit_midframe", 1.0) in extra
    assert det.entry_midframe.sum() == 0  # starts at clip start: normal
    # same track ending at the field edge: not flagged
    det, _ = clip_signals(make([edge_tool(f, 80) for f in range(10)]), FRAMES, border_dist(), CFG)
    assert det.exit_midframe.sum() == 0
    # ending under another tool: not flagged
    occ = mid + [(f, "l_hook", 1, box(150, 100, 60, 30)) for f in range(0, 20)]
    det, _ = clip_signals(make(occ), FRAMES, border_dist(), CFG)
    assert det[det.label == "grasper"].exit_midframe.sum() == 0


def test_entry_midframe():
    det, extra = clip_signals(make([(f, "grasper", 0, box(150, 100)) for f in range(5, 20)]), FRAMES, border_dist(), CFG)
    assert det.entry_midframe.iloc[0] == 1 and (4, "grasper", "entry_midframe", 1.0) in extra


def test_shaft_to_edge():
    d = make([edge_tool(0, 80), (0, "l_hook", 1, box(150, 100))])
    det, _ = clip_signals(d, np.array([0]), border_dist(), CFG)
    gap = det.set_index("label").shaft_to_edge
    m = CFG["signals"]["edge_margin_px"]
    assert gap["grasper"] <= 2 < m < 50 < gap["l_hook"]


def test_short_track_false_positive():
    d = make([edge_tool(f, 80) for f in FRAMES] + [(9, "grasper", 1, box(200, 150))])
    det, _ = clip_signals(d, FRAMES, border_dist(), CFG)
    assert det[det.track_id == 1].short_track.iloc[0] == 1 and det[det.track_id == 0].short_track.sum() == 0


def test_swap_detects_label_flip():
    dets = [edge_tool(f, 80, "grasper", 0) for f in FRAMES if f != 10]
    dets += [(10, "l_hook", 1, [0, 90, 80, 110])]  # same box, wrong label
    det, _ = clip_signals(make(dets), FRAMES, border_dist(), CFG)
    assert det[det.label == "l_hook"].swap.iloc[0] > 0.9
    assert det[det.label == "grasper"].swap.max() == 0
    # two tools genuinely crossing: no swap
    cross = [(f, "grasper", 0, box(50 + 10 * f, 100)) for f in FRAMES] + [(f, "l_hook", 1, box(250 - 10 * f, 100)) for f in FRAMES]
    det, _ = clip_signals(make(cross), FRAMES, border_dist(), CFG)
    assert det.swap.max() < 0.5


def test_tool_count():
    dets = [(0, "grasper", k, box(40 + 60 * k, 100)) for k in range(3)] + [(0, "l_hook", 3, box(250, 50)),
                                                                          (0, "l_hook", 4, box(250, 150))]
    det, extra = clip_signals(make(dets), np.array([0]), border_dist(), CFG)
    assert (det.tool_count == 3).all()  # 5 tools: 2 over max 3, plus 2 L-hooks
    det, _ = clip_signals(make([edge_tool(0, 80)]), np.array([0]), border_dist(), CFG)
    assert det.tool_count.sum() == 0


def test_quality_blur():
    rng = np.random.default_rng(0)
    img = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    field = np.ones((H, W), bool)
    sharp = _quality(img, field, CFG["quality"])
    blurred = _quality(cv2.GaussianBlur(img, (15, 15), 5), field, CFG["quality"])
    assert blurred["lap_var"] < sharp["lap_var"] / 10
    spec = img.copy(); spec[:50] = 255
    assert _quality(spec, field, CFG["quality"])["specular"] > 0.2


def test_robust_z():
    x = np.r_[np.random.default_rng(0).normal(0, 1, 500), 50.0]
    z = robust_z(x)
    assert z[-1] > 20 and abs(np.median(z)) < 0.1
    assert np.all(np.isfinite(robust_z(np.r_[np.zeros(100), 1.0])))


# ---------- corruption harness
def synthetic_tracks(n_clips=4):
    rows = []
    for c in range(n_clips):
        for f in range(80):
            rows.append({"video_id": "v0", "clip_id": f"v0_{c}", "frame_idx": f, "label": "grasper",
                         "instance_id": 0, "x1": 0.0, "y1": 200.0, "x2": 300.0 + f, "y2": 260.0})
            rows.append({"video_id": "v0", "clip_id": f"v0_{c}", "frame_idx": f, "label": "l_hook",
                         "instance_id": 0, "x1": 380.0, "y1": 150.0 + f, "x2": 460.0, "y2": 190.0 + f})
    d = finish_dets(pd.DataFrame(rows), "gt")
    d["track_id"] = d.groupby(["clip_id", "label"]).ngroup()
    return d


def test_corruption_reproducible():
    d = synthetic_tracks()
    dist = {"v0": cv2.distanceTransform(np.pad(np.ones((480, 854), np.uint8), 1), cv2.DIST_L2, 3)[1:-1, 1:-1]}
    a_det, a_err = corrupt(d, CFG, dist)
    b_det, b_err = corrupt(d, CFG, dist)
    pd.testing.assert_frame_equal(a_det, b_det)
    pd.testing.assert_frame_equal(a_err, b_err)
    assert (a_det.source == "gt_corrupted").all()
    assert set(a_err.error_type) >= {"drop", "jitter", "label_swap", "false_positive", "size_error"}
    cfg2 = copy.deepcopy(CFG); cfg2["corrupt"]["seed"] = 1
    c_det, _ = corrupt(d, cfg2, dist)
    assert not a_det[["x1", "y1"]].equals(c_det[["x1", "y1"]]) or len(a_det) != len(c_det)
