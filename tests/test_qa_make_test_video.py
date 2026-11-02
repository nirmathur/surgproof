"""make_test_video.py: refuses non-held-out videos, exact frame map, and registers by content hash."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import make_test_video as mtv  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def fake_raw(tmp_path, video="video12", starts=(0, 80, 400), per_clip=4):
    raw = tmp_path / "raw"
    for s in starts:
        d = raw / video / f"{video}_{s:05d}"
        d.mkdir(parents=True)
        for i in range(per_clip):
            arr = np.full((48, 64, 3), (s + i) % 255, dtype=np.uint8)
            Image.fromarray(arr).save(d / f"frame_{s + i}_endo.png")
            Image.fromarray(arr).save(d / f"frame_{s + i}_endo_watershed_mask.png")
    return raw


def test_longest_run_prefers_contiguous_and_earliest():
    assert mtv.longest_run(["v_00000", "v_00080", "v_00400"]) == ["v_00000", "v_00080"]
    assert mtv.longest_run(["v_00000", "v_00400", "v_00480"]) == ["v_00400", "v_00480"]
    assert mtv.longest_run(["v_00000", "v_00400"]) == ["v_00000"]


@pytest.mark.parametrize("video", ["video01", "video18"])
def test_refuses_training_videos(tmp_path, video):
    raw = fake_raw(tmp_path, video=video)
    with pytest.raises(SystemExit, match="held-out"):
        mtv.make(raw, tmp_path / "o.mp4", tmp_path / "reg", video=video)
    assert not (tmp_path / "o.mp4").exists()


def test_held_out_list_matches_the_repo_split():
    index = ROOT / "data" / "index.csv"
    if not index.exists():
        pytest.skip("no data/index.csv on this machine")
    mtv.check_held_out(["video12"], index)  # raises if the constant disagrees with the index


def test_frame_map_is_consecutive_and_trimmed(tmp_path):
    raw = fake_raw(tmp_path)
    rows = mtv.build_frame_map(raw, ["video12_00000", "video12_00080"], max_frames=6)
    assert [r["frame_index"] for r in rows] == list(range(6))
    assert [r["frame"] for r in rows] == [0, 1, 2, 3, 80, 81]
    assert rows[4]["clip"] == "video12_00080"


@needs_ffmpeg
def test_end_to_end_registers_by_hash(tmp_path):
    raw = fake_raw(tmp_path)
    out = tmp_path / "v" / "t.mp4"
    m = mtv.make(raw, out, tmp_path / "reg", video="video12", seconds=0.24, fps=25)
    assert m["n_frames"] == 6
    assert mtv.count_frames(out) == 6
    assert m["sha256"] == mtv.sha256_file(out)
    saved = json.loads((tmp_path / "reg" / f"{m['sha256']}.json").read_text())
    assert saved["clips"] == ["video12_00000", "video12_00080"]
    # nothing about the ground truth lands next to the video, where a pipeline could pick it up
    assert [p.name for p in out.parent.iterdir()] == ["t.mp4"]


@needs_ffmpeg
def test_real_clip_decodes_to_the_mapped_frames(tmp_path):
    """Real data if present: decoded frame i is closer to its mapped png than to the next one."""
    raw = ROOT / "raw" / "CholecSeg8k"
    if not (raw / "video12" / "video12_19500").is_dir():
        pytest.skip("CholecSeg8k not on this machine")
    import cv2

    out = tmp_path / "t.mp4"
    m = mtv.make(raw, out, tmp_path / "reg", video="video12", seconds=0.4)
    cap = cv2.VideoCapture(str(out))
    for i, r in enumerate(m["frames"][:-1]):
        ok, frame = cap.read()
        assert ok
        this = cv2.imread(str(raw / r["video"] / r["clip"] / f"frame_{r['frame']}_endo.png"))
        nxt = m["frames"][i + 1]
        other = cv2.imread(str(raw / nxt["video"] / nxt["clip"] / f"frame_{nxt['frame']}_endo.png"))
        assert np.abs(frame.astype(int) - this).mean() < np.abs(frame.astype(int) - other).mean()
