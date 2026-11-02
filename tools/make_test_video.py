#!/usr/bin/env python3
"""Assemble CholecSeg8k HELD-OUT clips into an mp4, as if it were a video someone dropped in.

Usage:
  python3 tools/make_test_video.py --video video12 --seconds 6 --out test_videos/video12_6s.mp4
  python3 tools/make_test_video.py --clips video28_00000 video28_00080 --out test_videos/v28.mp4
  python3 tools/make_test_video.py --split calib --video video18 --seconds 6 --out test_videos/calib18.mp4

Tuning a prompt or threshold? Use --split calib. Tuning against a held-out clip spends the held-out set, and
every number on it afterwards is an upper bound. The scorer stamps which split a video came from.
--split train is only for labels the other splits lack; report it apart from held-out numbers.

Clip choice is deterministic and metric-blind: without --clips it takes the longest run of
contiguous clips in the video, earliest first. It never looks at any score.

Writes next to nothing the pipeline can see: the mapping from video frame to dataset frame goes
in a registry keyed by the sha256 of the mp4, at <registry>/<sha256>.json (default
test_videos/.gt_registry/, gitignored with the videos). The scorer, surgproof/eval.py, is the only
reader. CholecSeg8k is CC BY-NC-SA and not ours to redistribute, so the output is never committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Fixed by prepare_data.py (seed 0) and README "The proof". Cross-checked against data/index.csv.
HELD_OUT = ("video09", "video12", "video17", "video28", "video48", "video55")
CLIP_LEN = 80
SOURCE_FPS = 25  # Cholec80 is 25 fps; CholecSeg8k clips are 80 consecutive frames of it.
DEFAULT_REGISTRY = ROOT / "test_videos" / ".gt_registry"
_FRAME = re.compile(r"^frame_(\d+)_endo\.png$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_videos(split: str, index_csv: Path = ROOT / "data" / "index.csv") -> set[str]:
    if not index_csv.exists():
        raise SystemExit(f"--split {split} needs {index_csv} (built by prepare_data.py)")
    import csv

    with open(index_csv) as f:
        return {r["video"] for r in csv.DictReader(f) if r.get("split") == split}


def calib_videos(index_csv: Path = ROOT / "data" / "index.csv") -> set[str]:
    return split_videos("calib", index_csv)


def check_split(videos, split: str, index_csv: Path = ROOT / "data" / "index.csv"):
    """Each video must really belong to the split it is reported under."""
    if split == "test":
        return check_held_out(videos, index_csv)
    if split not in ("calib", "train"):
        raise SystemExit(f"unknown split {split}")
    allowed = split_videos(split, index_csv)
    bad = sorted(set(videos) - allowed)
    if bad:
        raise SystemExit(f"{bad} not in the {split} split {sorted(allowed)}")


def check_held_out(videos, index_csv: Path = ROOT / "data" / "index.csv"):
    """Refuse non-held-out videos. Scoring on train or calib clips would flatter every number."""
    held = set(HELD_OUT)
    if index_csv.exists():
        import csv

        with open(index_csv) as f:
            from_index = {r["video"] for r in csv.DictReader(f) if r.get("split") == "test"}
        if from_index and from_index != held:
            raise SystemExit(f"held-out list disagrees with {index_csv}: {sorted(from_index)} vs {sorted(held)}")
    bad = sorted(set(videos) - held)
    if bad:
        raise SystemExit(f"{bad} not in the held-out split {sorted(held)}. Test videos must be held out.")


def list_clips(raw: Path, video: str) -> list[str]:
    d = raw / video
    if not d.is_dir():
        raise SystemExit(f"no such video directory: {d}")
    return sorted((p.name for p in d.iterdir() if p.is_dir()), key=lambda c: int(c.split("_")[1]))


def longest_run(clips: list[str]) -> list[str]:
    """Longest contiguous run; ties go to the earliest. Contiguous = next start is +80 frames."""
    best: list[str] = []
    cur: list[str] = []
    for c in clips:
        if cur and int(c.split("_")[1]) != int(cur[-1].split("_")[1]) + CLIP_LEN:
            cur = []
        cur.append(c)
        if len(cur) > len(best):
            best = list(cur)
    return best


def clip_frames(raw: Path, clip: str) -> list[int]:
    d = raw / clip.split("_")[0] / clip
    if not d.is_dir():
        raise SystemExit(f"no such clip: {d}")
    nums = sorted(int(m.group(1)) for m in (_FRAME.match(p.name) for p in d.iterdir()) if m)
    if not nums:
        raise SystemExit(f"no frames in {d}")
    return nums


def build_frame_map(raw: Path, clips: list[str], max_frames: int | None) -> list[dict]:
    rows: list[dict] = []
    for clip in clips:
        for n in clip_frames(raw, clip):
            if max_frames is not None and len(rows) >= max_frames:
                return rows
            rows.append({"frame_index": len(rows), "video": clip.split("_")[0], "clip": clip, "frame": n})
    return rows


def encode(raw: Path, frames: list[dict], out: Path, fps: int) -> None:
    """Near-lossless H.264. A concat list of symlinks gives an exact frame count at constant fps."""
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found")
    with tempfile.TemporaryDirectory() as td:
        for r in frames:
            src = raw / r["video"] / r["clip"] / f"frame_{r['frame']}_endo.png"
            if not src.exists():
                raise SystemExit(f"missing frame {src}")
            (Path(td) / f"{r['frame_index']:06d}.png").symlink_to(src.resolve())
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(td) / "%06d.png"),
             "-c:v", "libx264", "-preset", "medium", "-crf", "12", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             "-an", str(out)],
            check=True,
        )


def count_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return int(out.strip().strip(","))


def make(raw: Path, out: Path, registry: Path, clips: list[str] | None = None, video: str | None = None,
         seconds: float | None = None, fps: int = SOURCE_FPS, split: str = "test") -> dict:
    if clips:
        videos = {c.split("_")[0] for c in clips}
    elif video:
        videos = {video}
        clips = longest_run(list_clips(raw, video))
    else:
        raise SystemExit("give --clips or --video")
    check_split(videos, split)
    max_frames = round(seconds * fps) if seconds else None
    frames = build_frame_map(raw, clips, max_frames)
    if max_frames is not None and len(frames) < max_frames:
        print(f"note: only {len(frames)} frames available, wanted {max_frames}", file=sys.stderr)
    encode(raw, frames, out, fps)
    actual = count_frames(out)
    if actual != len(frames):
        raise SystemExit(f"encoded {actual} frames but mapped {len(frames)}; refusing to register a bad map")
    digest = sha256_file(out)
    manifest = {"sha256": digest, "fps": fps, "n_frames": len(frames), "source": "CholecSeg8k", "split": split,
                "clips": sorted({r["clip"] for r in frames}), "frames": frames}
    registry.mkdir(parents=True, exist_ok=True)
    (registry / f"{digest}.json").write_text(json.dumps(manifest))
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=ROOT / "raw" / "CholecSeg8k")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    p.add_argument("--video", help="held-out video; uses its longest contiguous run of clips")
    p.add_argument("--clips", nargs="+", help="explicit clip ids, in playback order")
    p.add_argument("--seconds", type=float, help="trim to this many seconds")
    p.add_argument("--fps", type=int, default=SOURCE_FPS)
    p.add_argument("--split", choices=["test", "calib", "train"], default="test",
                   help="calib is the only split to tune on. train exists only to score rare labels: the held-out and "
                        "calib splits hold no blood and almost no cystic duct, so those numbers can come from nowhere else")
    a = p.parse_args(argv)
    m = make(a.raw, a.out, a.registry, clips=a.clips, video=a.video, seconds=a.seconds, fps=a.fps, split=a.split)
    print(json.dumps({"out": str(a.out), "sha256": m["sha256"], "n_frames": m["n_frames"], "fps": m["fps"],
                      "split": m["split"], "clips": m["clips"]}))


if __name__ == "__main__":
    main()
