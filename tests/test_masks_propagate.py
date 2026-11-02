"""Stage 6 on synthetic video: a known moving object, so tracking is checkable without the model or any dataset."""
import json, subprocess

import cv2
import numpy as np
import pytest

from surgproof import masks as M, propagate as P

W, H = 320, 240
FPS = 25


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def background(seed=3):
    # Smooth random texture: Farneback needs gradients, and H.264 keeps them.
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 60, (H // 8, W // 8, 3)).astype(np.float32)
    return np.clip(cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC) + 90, 0, 255).astype(np.uint8)


def truth_at(i, speed=2):
    """Bool mask of the object at frame i."""
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (40 + speed * i, 80), (110 + speed * i, 160), 1, -1)
    return m.astype(bool)


def frame_at(i, bg, speed=2):
    f = bg.copy()
    m = truth_at(i, speed)
    tile = cv2.resize(background(7), (W, H))
    # The object's own texture moves with it.
    shifted = np.roll(tile, speed * i, axis=1)
    f[m] = shifted[m]
    return f


def write_video(path, frames):
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(path)],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def make_job(tmp_path, frames, kf_indices, kf_masks, status="accept", extra_meta=None):
    """kf_masks: {frame_index: {label: bool mask}}"""
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    (job / "masks").mkdir()
    write_video(job / "source.mp4", frames)
    kfs = []
    rows = []
    for n, fi in enumerate(kf_indices):
        kf = f"kf_{n:05d}"
        kfs.append({"kf": kf, "frame_index": fi, "t": fi / FPS})
        cv2.imwrite(str(job / "frames" / f"{kf}.jpg"), frames[fi])
        for label, mask in kf_masks.get(fi, {}).items():
            M.write_mask(M.mask_path(job, kf, label), mask)
            st = status[fi] if isinstance(status, dict) else status
            rows.append({"kf": kf, "label": label, "status": st, "path": f"masks/{kf}_{label}.png", "box": M.mask_box(mask)})
    meta = {"fps": FPS, "width": W, "height": H, "n_frames": len(frames), "kf_fps": 1, "keyframes": kfs}
    meta.update(extra_meta or {})
    (job / "meta.json").write_text(json.dumps(meta))
    (job / "masks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return job


@pytest.fixture(scope="module")
def moving(tmp_path_factory):
    bg = background()
    frames = [frame_at(i, bg) for i in range(60)]
    tmp = tmp_path_factory.mktemp("moving")
    job = make_job(tmp, frames, [0, 25, 50], {i: {"grasper": truth_at(i)} for i in [0, 25, 50]})
    summary = P.run(job)
    return job, summary, M.read_jsonl(job / "tracks.jsonl")


def test_propagated_masks_follow_the_object_in_the_right_direction(moving):
    job, _, rows = moving
    scores, static = [], []
    for r in rows:
        if r["propagated"]:
            m = M.read_mask(M.track_path(job, r["frame_index"], "grasper"))
            scores.append(iou(m, truth_at(r["frame_index"])))
            # Baseline: holding the nearest keyframe's mask with no flow at all.
            nearest = min([0, 25, 50], key=lambda k: abs(k - r["frame_index"]))
            static.append(iou(truth_at(nearest), truth_at(r["frame_index"])))
    assert np.mean(scores) > 0.8
    assert np.mean(scores) > np.mean(static)


def test_warp_direction_matches_true_motion():
    bg = background()
    a, b = cv2.cvtColor(frame_at(10, bg), cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_at(14, bg), cv2.COLOR_BGR2GRAY)
    grid = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    moved = P.warp(truth_at(10).astype(np.float32), P.flow(b, a), grid) > 0.5
    wrong = P.warp(truth_at(10).astype(np.float32), P.flow(a, b), grid) > 0.5
    assert iou(moved, truth_at(14)) > iou(wrong, truth_at(14))


def test_every_frame_covered_and_flags_are_correct(moving):
    job, summary, rows = moving
    assert summary["n_frames_read"] == 60
    assert sorted(r["frame_index"] for r in rows) == list(range(60))
    for r in rows:
        assert r["propagated"] == (r["frame_index"] not in (0, 25, 50))
        assert r["keyframe"] == ({0: "kf_00000", 25: "kf_00001", 50: "kf_00002"}.get(r["frame_index"]))
        assert (M.track_path(job, r["frame_index"], "grasper")).exists() == r["propagated"]


def test_keyframes_stay_the_ais_own_masks(moving):
    job, _, rows = moving
    ai = M.read_mask(M.mask_path(job, "kf_00001", "grasper"))
    assert ai.tolist() == truth_at(25).tolist()
    assert not M.track_path(job, 25, "grasper").exists()
    r = next(r for r in rows if r["frame_index"] == 25)
    assert r["box_raw"] == M.mask_box(ai)


def test_box_raw_is_exact_mask_bound_and_box_is_smoother(moving):
    job, _, rows = moving
    for r in rows:
        if r["propagated"]:
            assert r["box_raw"] == M.mask_box(M.read_mask(M.track_path(job, r["frame_index"], "grasper")))
    rows = sorted(rows, key=lambda r: r["frame_index"])
    jitter = lambda key: np.mean(np.abs(np.diff(np.diff([r[key][0] for r in rows]))))
    assert jitter("box") <= jitter("box_raw")


def test_smooth_boxes_window_skips_absent_frames():
    rows = [{"frame_index": 0, "label": "a", "box_raw": [0, 0, 10, 10]},
            {"frame_index": 1, "label": "a", "box_raw": [2, 2, 12, 12]},
            {"frame_index": 9, "label": "a", "box_raw": [50, 50, 60, 60]}]
    out = P.smooth_boxes(rows)
    assert out[0]["box"] == [1.0, 1.0, 11.0, 11.0]
    assert out[2]["box"] == [50.0, 50.0, 60.0, 60.0]


def test_object_present_at_one_end_only_stops_near_the_midpoint(tmp_path):
    bg = background()
    frames = [frame_at(i, bg, speed=0) for i in range(26)]
    job = make_job(tmp_path, frames, [0, 25], {0: {"grasper": truth_at(0, 0)}})
    P.run(job)
    idx = sorted(r["frame_index"] for r in M.read_jsonl(job / "tracks.jsonl"))
    assert idx[0] == 0 and 9 <= idx[-1] <= 16


def test_amber_keyframe_makes_its_propagated_frames_amber(tmp_path):
    bg = background()
    frames = [frame_at(i, bg) for i in range(26)]
    job = make_job(tmp_path, frames, [0, 25], {i: {"grasper": truth_at(i)} for i in [0, 25]},
                   status={0: "accept", 25: "AMBER"})
    P.run(job)
    rows = M.read_jsonl(job / "tracks.jsonl")
    assert {r["status"] for r in rows if r["propagated"]} == {"AMBER"}
    assert next(r for r in rows if r["frame_index"] == 0)["status"] == "accept"
    assert all(set(r["source_kfs"]) == {"kf_00000", "kf_00001"} for r in rows if r["propagated"])


def test_rejected_or_absent_keyframe_masks_are_not_propagated(tmp_path):
    bg = background()
    frames = [frame_at(i, bg) for i in range(26)]
    job = make_job(tmp_path, frames, [0, 25], {i: {"grasper": truth_at(i)} for i in [0, 25]}, status="reject")
    summary = P.run(job)
    assert summary["n_rows"] == 0 and not list((job / "tracks").glob("*.png"))


def test_flow_does_not_cross_a_scene_cut(tmp_path):
    bg1, bg2 = background(3), cv2.bitwise_not(background(11))
    frames = [frame_at(i, bg1, speed=0) for i in range(13)] + [bg2.copy() for _ in range(13)]
    job = make_job(tmp_path, frames, [0, 25], {0: {"grasper": truth_at(0, 0)}})
    summary = P.run(job)
    assert summary["n_cuts"] == 1
    assert all(r["frame_index"] <= 12 for r in M.read_jsonl(job / "tracks.jsonl"))


def test_frames_before_first_and_after_last_keyframe_are_tracked(tmp_path):
    bg = background()
    frames = [frame_at(i, bg, speed=1) for i in range(40)]
    job = make_job(tmp_path, frames, [5, 30], {i: {"grasper": truth_at(i, 1)} for i in [5, 30]})
    P.run(job)
    idx = sorted(r["frame_index"] for r in M.read_jsonl(job / "tracks.jsonl"))
    assert idx == list(range(40))
    for r in M.read_jsonl(job / "tracks.jsonl"):
        if r["frame_index"] in (0, 39):
            assert iou(M.read_mask(M.track_path(job, r["frame_index"], "grasper")), truth_at(r["frame_index"], 1)) > 0.7


def test_extrapolation_is_capped_at_one_keyframe_gap(tmp_path):
    bg = background()
    frames = [frame_at(0, bg, speed=0) for _ in range(90)]
    job = make_job(tmp_path, frames, [0, 10], {i: {"grasper": truth_at(0, 0)} for i in [0, 10]})
    P.run(job)
    assert max(r["frame_index"] for r in M.read_jsonl(job / "tracks.jsonl")) == 20


def test_keyframe_past_end_of_video_keeps_its_row_and_is_counted(tmp_path):
    bg = background()
    frames = [frame_at(i, bg, speed=0) for i in range(20)]
    job = make_job(tmp_path, frames + [frames[-1]] * 0, [0, 10], {i: {"grasper": truth_at(0, 0)} for i in [0, 10]})
    meta = json.loads((job / "meta.json").read_text())
    cv2.imwrite(str(job / "frames" / "kf_00002.jpg"), frames[0])
    M.write_mask(M.mask_path(job, "kf_00002", "grasper"), truth_at(0, 0))
    meta["keyframes"].append({"kf": "kf_00002", "frame_index": 25, "t": 1.0})
    (job / "meta.json").write_text(json.dumps(meta))
    with open(job / "masks.jsonl", "a") as f:
        f.write(json.dumps({"kf": "kf_00002", "label": "grasper", "status": "accept", "path": "masks/kf_00002_grasper.png"}) + "\n")
    summary = P.run(job)
    assert summary["n_keyframes_past_end"] == 1
    assert any(r["keyframe"] == "kf_00002" for r in M.read_jsonl(job / "tracks.jsonl"))


def test_missing_video_is_a_clear_error(tmp_path):
    bg = background()
    job = make_job(tmp_path, [frame_at(i, bg) for i in range(5)], [0], {0: {"grasper": truth_at(0)}})
    (job / "source.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="source video"):
        P.run(job)


def test_find_video_prefers_argument_then_meta_then_convention(tmp_path):
    (tmp_path / "source.mp4").write_bytes(b"x")
    (tmp_path / "custom.mov").write_bytes(b"x")
    assert P.find_video(tmp_path, {}).name == "source.mp4"
    assert P.find_video(tmp_path, {"source_video": "custom.mov"}).name == "custom.mov"
    assert P.find_video(tmp_path, {"source_video": "custom.mov"}, video=str(tmp_path / "source.mp4")).name == "source.mp4"


def test_high_res_masks_stay_at_keyframe_size(tmp_path):
    # Work resolution is capped, so track PNGs must be scaled back up to match the keyframe masks.
    w, h = 960, 540
    bg = cv2.resize(background(), (w, h))
    frames = [bg.copy() for _ in range(10)]
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "25",
                          "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(job / "source.mp4")], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    m = np.zeros((h, w), bool); m[100:300, 200:500] = True
    cv2.imwrite(str(job / "frames" / "kf_00000.jpg"), frames[0])
    M.write_mask(M.mask_path(job, "kf_00000", "liver"), m)
    (job / "meta.json").write_text(json.dumps({"fps": 25, "width": w, "height": h, "n_frames": 10, "kf_fps": 1,
                                               "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0}]}))
    (job / "masks.jsonl").write_text(json.dumps({"kf": "kf_00000", "label": "liver", "status": "accept", "path": "masks/kf_00000_liver.png"}) + "\n")
    P.run(job)
    out = M.read_mask(M.track_path(job, 5, "liver"))
    assert out.shape == (h, w) and iou(out, m) > 0.95
