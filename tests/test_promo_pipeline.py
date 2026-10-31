"""Tests for the promo pitch video pipeline (lane: promo, issue #10)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from promo.render import (
    HEIGHT,
    WIDTH,
    build_proof_slot_video,
    compose_illustrative_shot,
    load_real_pipeline_metrics,
    probe_video,
    render_shot_overlay_png,
    stitch_promo_video,
)
from promo.shots import ILLUSTRATIVE_CAPTION, SHOTS, total_storyboard_duration


def test_storyboard_file_and_shot_specs():
    sb_path = Path(__file__).resolve().parent.parent / "promo" / "storyboard.md"
    assert sb_path.is_file(), "promo/storyboard.md must exist"
    text = sb_path.read_text(encoding="utf-8")
    assert "ILLUSTRATIVE" in text
    assert "app.py" in text
    assert "SurgProof" in text
    assert "No human labels. Only proof." in text

    assert 4 <= len(SHOTS) <= 6
    assert any("SurgProof" in s.narration for s in SHOTS)
    total_dur = total_storyboard_duration()
    assert 60.0 <= total_dur <= 90.0, f"Expected 60-90s total duration, got {total_dur}"

    proof_shots = [s for s in SHOTS if s.is_proof_slot]
    illustrative_shots = [s for s in SHOTS if s.illustrative]
    assert len(proof_shots) == 1
    assert len(illustrative_shots) == len(SHOTS) - 1

    for s in illustrative_shots:
        assert "ILLUSTRATIVE" in s.badge_text.upper()
        assert s.BadgeText if hasattr(s, "BadgeText") else s.badge_text == ILLUSTRATIVE_CAPTION
        assert s.prompt.strip(), f"Illustrative shot {s.shot_id} must have a prompt"


def test_load_real_pipeline_metrics_never_fakes_numbers(tmp_path: Path):
    # 1. Empty directory -> all cards pending except strict 0 human training labels rule
    empty_metrics = load_real_pipeline_metrics(out_dir=tmp_path)
    assert not empty_metrics["has_results"]
    assert not empty_metrics["has_student"]
    values = [c["value"] for c in empty_metrics["cards"]]
    assert any("AWAITING" in v for v in values)

    # 2. Populate real-shaped results.json and student.json -> exact numbers reflected
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "accepted": {"accuracy": 0.912, "coverage": 0.64, "escalated_share": 0.18},
                "manual_raw_acc": 0.845,
                "blank_raw_acc": 0.760,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "student.json").write_text(
        json.dumps(
            {
                "mean_acc_by_source": {
                    "verified_pseudo": 0.881,
                    "raw_pseudo": 0.812,
                    "oracle_upper_bound": 0.925,
                }
            }
        ),
        encoding="utf-8",
    )
    pop_metrics = load_real_pipeline_metrics(out_dir=tmp_path)
    assert pop_metrics["has_results"]
    assert pop_metrics["has_student"]
    card_vals = " | ".join(c["value"] for c in pop_metrics["cards"])
    assert "91.2%" in card_vals
    assert "84.5%" in card_vals
    assert "+8.5 pts" in card_vals
    assert "88.1%" in card_vals


def test_overlay_compose_and_stitch(tmp_path: Path):
    shot = SHOTS[0]
    overlay_png = tmp_path / "overlay.png"
    render_shot_overlay_png(shot, overlay_png)
    assert overlay_png.is_file()

    im = Image.open(overlay_png)
    assert im.size == (WIDTH, HEIGHT)
    assert im.mode == "RGBA"
    # Top badge region must have non-zero alpha (burned-in illustrative caption)
    top_crop = im.crop((320, 22, 900, 56))
    alphas = [px[3] for px in top_crop.getdata()]
    assert max(alphas) > 200

    # Create a tiny synthetic 2s raw clip via ffmpeg testsrc, then compose and stitch
    raw_clip = tmp_path / "raw.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=navy:s=640x360:d=2:r=24",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(raw_clip),
        ],
        check=True,
        capture_output=True,
    )

    # Compose a 3-second test shot
    from dataclasses import replace

    short_shot = replace(shot, duration_sec=3.0)
    composed = tmp_path / "shot_01.mp4"
    compose_illustrative_shot(short_shot, raw_clip, overlay_png, composed)
    meta = probe_video(composed)
    assert meta["valid"]
    assert meta["width"] == WIDTH
    assert meta["height"] == HEIGHT
    assert meta["has_audio"]
    assert 2.5 <= meta["duration_sec"] <= 3.5

    # Build proof slot video (2s) without launching Chrome
    proof_shot = replace(SHOTS[4], duration_sec=2.0)
    proof_mp4 = tmp_path / "shot_05.mp4"
    build_proof_slot_video(proof_shot, proof_mp4, out_dir=tmp_path, capture_app=False)
    p_meta = probe_video(proof_mp4)
    assert p_meta["valid"]
    assert 1.6 <= p_meta["duration_sec"] <= 2.5

    # Stitch the two clips
    stitched = tmp_path / "stitched.mp4"
    s_meta = stitch_promo_video([composed, proof_mp4], stitched)
    assert s_meta["valid"]
    assert 4.4 <= s_meta["duration_sec"] <= 5.6
