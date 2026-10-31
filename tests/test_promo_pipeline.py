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
    _load_amber_round_counts,
    _load_target_frame_and_boxes,
    generate_contact_sheet,
    load_real_result_card,
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
    for forbidden in ("Nir Mathur", "Supervisor Quote"):
        assert forbidden not in sb_text

    assert len(SHOTS) == 7
    total_dur = total_storyboard_duration()
    assert 45.0 <= total_dur <= 60.0, f"Expected 45-60s total duration, got {total_dur}"

    for s in SHOTS:
        words = [w for w in re.split(r"\s+", s.caption.strip()) if w]
        assert len(words) <= 6, f"Shot {s.shot_id} caption {s.caption!r} exceeds 6 words ({len(words)})"
        assert s.word_count <= 6


def test_load_real_result_number_and_boxes_honesty(tmp_path: Path):
    # 1. Missing out/results.json -> None (renders "Result: pending")
    assert load_real_result_number(out_dir=tmp_path) is None
    assert load_real_result_card(out_dir=tmp_path) is None

    # 2. Interim smoke results.json (n_test_labels = 90 < 1000) -> still None ("Result: pending")
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "n_test_labels": 90,
                "accepted_5_informative": {
                    "n_test_labels_excl_liver": 75,
                    "accuracy": 0.9853,
                    "coverage": 0.907,
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_real_result_number(out_dir=tmp_path) is None

    # 3. Full test run results.json (n_test_labels = 1680 >= 1000) -> exact headline + subtext
    (tmp_path / "results.json").write_text(
        json.dumps(
            {
                "n_test_labels": 1680,
                "accepted_5_informative": {
                    "n_test_labels_excl_liver": 1400,
                    "accuracy": 0.964,
                    "coverage": 0.88,
                },
            }
        ),
        encoding="utf-8",
    )
    card = load_real_result_card(out_dir=tmp_path)
    assert card is not None
    assert card["headline"] == "96.4% accurate on the 88% it accepted"
    assert card["subtext"] == "n = 1400 held-out labels, 6 unseen videos"

    # 4. Boxes: empty when out/annotations.jsonl is absent; populated only from out/annotations.jsonl
    _, items_no_ann = _load_target_frame_and_boxes(out_dir=tmp_path)
    assert all(it["box"] == [] for it in items_no_ann)

    (tmp_path / "annotations.jsonl").write_text(
        json.dumps({"frame_id": "video12_15750_15758", "label": "grasper", "box": [400, 120, 520, 360]}) + "\n",
        encoding="utf-8",
    )
    _, items_with_ann = _load_target_frame_and_boxes(out_dir=tmp_path)
    by_k = {it["key"]: it["box"] for it in items_with_ann}
    assert by_k["grasper"] == [400, 120, 520, 360]
    assert by_k["l_hook"] == []

    # 5. Amber rounds: empty when out/amber_rounds.jsonl is absent
    assert _load_amber_round_counts(out_dir=tmp_path) == []


def test_render_shots_stitch_and_contact_sheet(tmp_path: Path):
    s1 = replace(SHOTS[0], duration_sec=1.5)
    s6 = replace(SHOTS[5], duration_sec=1.5)
    s7 = replace(SHOTS[6], duration_sec=1.5)

    p1 = tmp_path / "shot_01.mp4"
    p6 = tmp_path / "shot_06.mp4"
    p7 = tmp_path / "shot_07.mp4"

    render_shot_01_raw_frame(s1, p1)
    _, hero = render_shot_06_result(s6, p6, out_dir=tmp_path)
    assert hero is None  # tmp_path has no full results.json -> pending
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
