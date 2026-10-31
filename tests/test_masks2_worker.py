"""Tests for autonomous zero-human-label AI segmentation mask worker (nir-masks2)."""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import masks_worker as MW


def test_masks_worker_zero_gt_leakage():
    """Ensure masks_worker.py never references ground-truth masks or gt_* columns."""
    src = Path(MW.__file__).read_text(encoding="utf-8")
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert not re.search(r"gt_|\btruth\b|mask_path", line), f"Forbidden GT token in masks_worker.py: {line}"


def test_select_video_frames_even_odd_split(tmp_path: Path):
    idx_file = tmp_path / "index.csv"
    rows = [
        {"frame_id": f"video12_15750_{15750 + 8 * i}", "video": "video12", "split": "test", "path": "x.png"}
        for i in range(10)
    ]
    pd.DataFrame(rows).to_csv(idx_file, index=False)

    even_fids = MW.select_video_frames(idx_file, video="video12", parity="even")
    odd_fids = MW.select_video_frames(idx_file, video="video12", parity="odd")
    all_fids = MW.select_video_frames(idx_file, video="video12", parity="all")

    assert len(all_fids) == 10
    assert len(even_fids) == 5
    assert len(odd_fids) == 5
    assert set(even_fids).isdisjoint(set(odd_fids))
    assert sorted(even_fids + odd_fids) == all_fids
    assert even_fids[0] == "video12_15750_15750"
    assert odd_fids[0] == "video12_15750_15758"


def test_segment_structure_in_box_and_save_png(tmp_path: Path):
    # Create synthetic surgical frame (warm reddish tissue + metallic silver diagonal tool + black border)
    h, w = 240, 427
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    # Warm tissue inside FOV
    rgb[:, 40:387, 0] = 175
    rgb[:, 40:387, 1] = 75
    rgb[:, 40:387, 2] = 65
    # Metallic instrument shaft along diagonal inside [200..600, 200..600]
    yy, xx = np.mgrid[0:h, 0:w]
    tool_band = (np.abs((yy - 48) - (xx - 85)) <= 12) & (yy >= 48) & (yy <= 144) & (xx >= 85) & (xx <= 250)
    rgb[tool_band, 0] = 190
    rgb[tool_band, 1] = 195
    rgb[tool_band, 2] = 200

    m_tool = MW.segment_structure_in_box(rgb, [200, 200, 600, 600], [580, 560], "grasper")
    assert m_tool.shape == (h, w)
    assert m_tool.dtype == bool
    assert int(m_tool.sum()) > 50
    # Black border must be excluded
    assert not m_tool[:, :30].any()

    m_organ = MW.segment_structure_in_box(rgb, [100, 150, 800, 800], None, "liver")
    assert int(m_organ.sum()) > 200
    assert MW.binary_mask_iou(m_organ, m_organ) == 1.0
    assert MW.binary_mask_dice(m_organ, m_organ) == 1.0

    poly = MW.extract_mask_polygon(m_organ, num_points=12)
    assert len(poly) >= 6
    assert all(len(pt) == 2 and 0 <= pt[0] <= 1000 and 0 <= pt[1] <= 1000 for pt in poly)

    out_png = tmp_path / "video12_15750_15750_liver.png"
    MW.save_rgba_mask_png(m_organ, "liver", out_png)
    assert out_png.is_file()
    loaded = np.array(Image.open(out_png))
    assert loaded.shape == (h, w, 4)
    assert int((loaded[..., 3] > 0).sum()) == int(m_organ.sum())


def test_segment_and_verify_frame_amber_escalation(tmp_path: Path):
    img_file = tmp_path / "frame.png"
    rgb = np.full((120, 200, 3), (170, 80, 70), dtype=np.uint8)
    Image.fromarray(rgb).save(img_file)

    df = pd.DataFrame(
        [
            {
                "frame_id": "video12_15750_15750",
                "video": "video12",
                "split": "test",
                "frame_path": str(img_file),
                "key": "gallbladder",
                "status": "accept",
                "final_present": 1,
            }
        ]
    )
    # Disjoint Pass 1 and Pass 2 boxes -> mask_iou == 0.0 -> escalates to tighter-crop loop
    existing = {
        "gallbladder": {
            "frame_id": "video12_15750_15750",
            "video": "video12",
            "split": "test",
            "img_path": str(img_file),
            "label": "gallbladder",
            "box": [100, 100, 300, 300],
            "tip": None,
            "phase": "2. Calot's (Hepatocystic) Triangle Dissection",
            "phase_conf": 0.95,
            "conf": 0.90,
            "pass1": {"box": [50, 50, 250, 250], "tip": None, "conf": 0.90},
            "pass2": {"box": [650, 650, 850, 850], "tip": None, "conf": 0.90, "iou": 0.0},
            "round": 0,
            "status": "AMBER",
        }
    }

    async def fake_caller(prompt, *, model=None, images=(), schema=None, tag="", sample=0, **kw):
        # Agree on crop box in round 1
        return {
            "box_2d": [200, 200, 800, 800],
            "tip_2d": [],
            "confidence": 0.94,
            "reason": "Consensus organ boundary in zoomed crop",
        }

    anns, rounds = asyncio.run(
        MW.segment_and_verify_frame(
            "video12_15750_15750",
            df,
            {"phases": ["2. Calot's (Hepatocystic) Triangle Dissection"], "structures": []},
            existing_frame_anns=existing,
            masks_dir=tmp_path / "masks",
            call_fn=fake_caller,
        )
    )
    assert len(anns) == 1
    rec = anns[0]
    assert rec["status"] == "accept"
    assert rec["round"] == 1
    assert rec["mask_iou"] >= 0.50
    assert (tmp_path / "masks" / "video12_15750_15750_gallbladder.png").is_file()
    assert len(rounds) >= 2
