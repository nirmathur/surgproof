""""Any laparoscopic video": the formats people actually drop in, through the real CLI with the mock model.
Frame count and coordinates are the claims at stake: a resample or a missed rescale shifts every annotation."""
from __future__ import annotations

import subprocess

import numpy as np
import pytest
from PIL import Image

from tests.e2e_helpers.harness import HAVE_FFMPEG, ROOT, count_frames, read_json, read_jsonl, run_cli

ready = all((ROOT / "surgproof" / f"{m}.py").exists() for m in ("pipeline", "masks", "propagate", "render", "export"))
pytestmark = pytest.mark.skipif(not (ready and HAVE_FFMPEG), reason="other lanes not merged, or no ffmpeg")
MOCK = {"SURGPROOF_VLM": "mock"}


def ff(out, *args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args, str(out)], check=True)
    return out


def src(size, fps, seconds):
    return ["-f", "lavfi", "-i", f"testsrc2=s={size}:r={fps}:d={seconds}"]


def run(video, tmp_path):
    job = tmp_path / "jobs" / "j"
    p = run_cli(video, job, MOCK)
    assert p.returncode == 0, p.stdout[-1200:] + p.stderr[-1200:]
    return job, read_json(job / "meta.json")


def check_consistent(job, meta):
    """Everything downstream agrees with what ingest measured."""
    assert read_json(job / "status.json")["stage"] == "done"
    assert count_frames(job / "annotated.mp4") == meta["n_frames"] == count_frames(job / meta["source"])
    w, h = meta["width"], meta["height"]
    for r in read_jsonl(job / "detect.jsonl"):
        for x, y in r["polygon"] or []:
            assert 0 <= x <= w and 0 <= y <= h, r
    for p in (job / "masks").glob("*.png"):
        assert Image.open(p).size == (w, h), p
    coco = read_json(job / "dataset_coco.json")
    assert coco["annotations"]
    for a in coco["annotations"]:
        x, y, bw, bh = a["bbox"]
        assert x + bw <= w + 1 and y + bh <= h + 1, a
    for k in meta["keyframes"]:
        assert k["frame_index"] < meta["n_frames"]
        assert Image.open(job / "frames" / f"{k['kf']}.jpg").size == (w, h)


def test_mov_with_prores_and_audio(tmp_path):
    v = ff(tmp_path / "a.mov", *src("640x360", 25, 3), "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
           "-c:v", "prores_ks", "-c:a", "pcm_s16le", "-shortest")
    job, meta = run(v, tmp_path)
    assert meta["n_frames"] == 75
    check_consistent(job, meta)


def test_1080p_coordinates_map_back_to_full_frame(tmp_path):
    """The model sees a smaller view. Masks must land where it pointed on the full frame."""
    v = ff(tmp_path / "a.mp4", *src("1920x1080", 25, 2), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    assert (meta["width"], meta["height"]) == (1920, 1080) and meta["view_scale"] > 1
    check_consistent(job, meta)
    grasper = [r for r in read_jsonl(job / "detect.jsonl") if r["label"] == "grasper" and r["polygon"]][0]
    xs = [p[0] for p in grasper["polygon"]]
    # the mock centres the grasper at 44% of the width it is told; scaled back it must still be at 44% of 1920
    assert (min(xs) + max(xs)) / 2 == pytest.approx(0.44 * 1920, abs=30)


def test_60fps_keyframes_stay_one_second_apart(tmp_path):
    v = ff(tmp_path / "a.mp4", *src("640x360", 60, 3), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    assert meta["fps"] == 60 and [k["frame_index"] for k in meta["keyframes"]] == [0, 60, 120]
    check_consistent(job, meta)


def test_fractional_frame_rate(tmp_path):
    v = ff(tmp_path / "a.mp4", "-f", "lavfi", "-i", "testsrc2=s=640x360:r=30000/1001:d=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    assert meta["fps"] == pytest.approx(29.97, abs=0.01)
    check_consistent(job, meta)


def test_variable_frame_rate_keeps_time_and_index_in_step(tmp_path):
    """Screen recordings and phones. Frames are dropped unevenly, so ingest must normalise to constant rate."""
    v = ff(tmp_path / "a.mp4", *src("640x360", 30, 4), "-vf", "select='not(mod(n\\,3))+lt(n\\,30)',setpts=N/30/TB*(1+0.5*gt(N\\,30))",
           "-fps_mode", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    check_consistent(job, meta)
    for k in meta["keyframes"]:
        assert k["t"] == pytest.approx(k["frame_index"] / meta["fps"], abs=0.01)


def test_odd_dimensions(tmp_path):
    v = ff(tmp_path / "a.mp4", *src("853x479", 25, 2), "-c:v", "libx264", "-pix_fmt", "yuv444p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    check_consistent(job, meta)


def test_rotated_phone_video_metadata(tmp_path):
    """A rotation flag means the player shows it sideways-corrected: masks must match what is displayed."""
    base = ff(tmp_path / "b.mp4", *src("640x360", 25, 2), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    v = ff(tmp_path / "a.mp4", "-display_rotation", "90", "-i", str(base), "-c", "copy")
    job, meta = run(v, tmp_path)
    check_consistent(job, meta)


def test_very_short_video(tmp_path):
    v = ff(tmp_path / "a.mp4", *src("640x360", 25, 0.4), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job, meta = run(v, tmp_path)
    assert len(meta["keyframes"]) == 1
    check_consistent(job, meta)


def test_kf_fps_option(tmp_path):
    v = ff(tmp_path / "a.mp4", *src("320x180", 10, 2), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    job = tmp_path / "jobs" / "j"
    assert run_cli(v, job, MOCK, extra_args=["--kf-fps", "2"]).returncode == 0
    assert [k["frame_index"] for k in read_json(job / "meta.json")["keyframes"]] == [0, 5, 10, 15]


def test_corrupt_truncated_file_fails_cleanly(tmp_path):
    good = ff(tmp_path / "a.mp4", *src("320x180", 25, 2), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast")
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(good.read_bytes()[: good.stat().st_size // 3])
    job = tmp_path / "jobs" / "j"
    p = run_cli(bad, job, MOCK)
    assert p.returncode != 0
    assert read_json(job / "status.json")["error"]
    assert not (job / "dataset_coco.json").exists()


def test_video_over_limit_refused_by_cli(tmp_path):
    v = ff(tmp_path / "a.mp4", "-f", "lavfi", "-i", "color=c=black:s=64x36:r=1:d=301", "-c:v", "libx264", "-pix_fmt", "yuv420p")
    job = tmp_path / "jobs" / "j"
    p = run_cli(v, job, MOCK)
    assert p.returncode != 0 and "limit" in (p.stderr + p.stdout)
    assert not list((job / "frames").glob("*.jpg")) if (job / "frames").exists() else True
