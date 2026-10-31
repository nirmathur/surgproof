"""Tests for the SurgProof 7-shot product demo pipeline (lane: promo, issue #10)."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re

from PIL import Image

from promo.render import (
    HEIGHT,
    WIDTH,
    generate_contact_sheet,
    load_real_result_number,
    probe_video,
    render_shot_01_raw_frame,
    render_shot_06_result,
    render_shot_07_end,
    stitch_promo_video,
)
from promo.shots import SHOTS, total_storyboard_duration


def test_storyboard_and_design_constraints():
    root = Path(__file__).resolve().parent.parent
    sb_path = root / "promo" / "storyboard.md"
    design_path = root / "promo" / "DESIGN.md"
    assert sb_path.is_file(), "promo/storyboard.md must exist"
    assert design_path.is_file(), "promo/DESIGN.md must exist"

    sb_text = sb_path.read_text(encoding="utf-8")
    assert "SurgProof" in sb_text
    assert "No human labels. Only proof." in sb_text
    # Zero personal names or quote slots in storyboard or shots
    for forbidden in ("Nir Mathur", "Supervisor Quote"):
        assert forbidden not in sb_text

    assert len(SHOTS) == 7
    total_dur = total_storyboard_duration()
    assert 45.0 <= total_dur <= 60.0, f"Expected 45-60s total duration, got {total_dur}"

    for s in SHOTS:
        words = [w for w in re.split(r"\s+", s.caption.strip()) if w]
        assert len(words) <= 6, f"Shot {s.shot_id} caption {s.caption!r} exceeds 6 words ({len(words)})"
        assert s.word_count <= 6


def test_load_real_result_number_honesty(tmp_path: Path):
    # 1. Missing out/results.json -> None (renders "Result: pending")
    assert load_real_result_number(out_dir=tmp_path) is None

    # 2. Populated out/results.json -> exact percentage read
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "accepted_5_informative": {
                    "accuracy": 0.9848,
                    "coverage": 0.88,
                },
                "accepted": {
                    "accuracy": 0.9865,
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_real_result_number(out_dir=tmp_path) == "98.5%"


def test_render_shots_stitch_and_contact_sheet(tmp_path: Path):
    s1 = replace(SHOTS[0], duration_sec=1.5)
    s6 = replace(SHOTS[5], duration_sec=1.5)
    s7 = replace(SHOTS[6], duration_sec=1.5)

    p1 = tmp_path / "shot_01.mp4"
    p6 = tmp_path / "shot_06.mp4"
    p7 = tmp_path / "shot_07.mp4"

    render_shot_01_raw_frame(s1, p1)
    _, hero = render_shot_06_result(s6, p6, out_dir=tmp_path)
    assert hero is None  # tmp_path has no results.json -> pending
    render_shot_07_end(s7, p7)

    for p in (p1, p6, p7):
        meta = probe_video(p)
        assert meta["valid"]
        assert meta["width"] == WIDTH
        assert meta["height"] == HEIGHT
        assert meta["has_audio"]

    stitched = tmp_path / "surgproof_promo.mp4"
    s_meta = stitch_promo_video([p1, p6, p7], out_video=stitched)
    assert s_meta["valid"]
    assert 4.0 <= s_meta["duration_sec"] <= 5.0

    sheet = tmp_path / "contact_sheet.png"
    generate_contact_sheet(video_path=stitched, out_png=sheet)
    assert sheet.is_file()
    im = Image.open(sheet)
    assert im.width > 0 and im.height > 0
