"""Unit and contract tests for `localize.py` and `verify_localize.py` (`nir-localize`)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from localize import (
    DEFAULT_PHASES,
    localize_frame,
    normalize_phase,
)
from verify_localize import (
    average_boxes,
    box_iou,
    disagreement_crop,
    evaluate_annotations,
    map_crop_box_to_full,
    sanitize_box,
    sanitize_tip,
    verify_and_refine_item,
)


def test_box_iou_and_sanitization():
    assert sanitize_box([100, 200, 500, 600]) == [100, 200, 500, 600]
    assert sanitize_box([500, 600, 100, 200]) == [100, 200, 500, 600]
    assert sanitize_box([100, 100, 101, 101]) == []
    assert abs(box_iou([100, 100, 500, 500], [100, 100, 500, 500]) - 1.0) < 1e-6
    assert box_iou([0, 0, 200, 200], [400, 400, 600, 600]) == 0.0
    avg = average_boxes([100, 200, 500, 600], [120, 220, 520, 620])
    assert avg == [110, 210, 510, 610]


def test_disagreement_crop_and_mapping():
    b1 = [200, 300, 400, 500]
    b2 = [350, 450, 600, 700]
    c1 = disagreement_crop(b1, b2, round_idx=1)
    c2 = disagreement_crop(b1, b2, round_idx=2)
    c3 = disagreement_crop(b1, b2, round_idx=3)
    # Crop gets progressively tighter from round 1 -> 2 -> 3
    area1 = (c1[2] - c1[0]) * (c1[3] - c1[1])
    area2 = (c2[2] - c2[0]) * (c2[3] - c2[1])
    area3 = (c3[2] - c3[0]) * (c3[3] - c3[1])
    assert area1 > area2 > area3
    mapped = map_crop_box_to_full([0, 0, 1000, 1000], c1)
    assert mapped == c1


def test_sanitize_tip_and_phase():
    assert sanitize_tip([350, 450], box=[200, 300, 500, 600], is_instrument=True) == [350, 450]
    assert sanitize_tip([50, 950], box=[200, 300, 500, 600], is_instrument=True) == [200, 600]
    assert sanitize_tip([350, 450], box=[200, 300, 500, 600], is_instrument=False) is None
    ph = normalize_phase("Phase 2: Calot's Triangle Dissection", DEFAULT_PHASES)
    assert ph.startswith("2.")


def test_verify_and_refine_amber_loop_and_post_hoc_eval(tmp_path: Path):
    img_p = tmp_path / "frame_0_endo.png"
    mask_p = tmp_path / "frame_0_endo_watershed_mask.png"

    # Synthetic RGB frame (80x100) + watershed mask with grasper (31 -> class 5) and liver (21 -> class 2)
    rgb = np.full((80, 100, 3), 140, dtype=np.uint8)
    Image.fromarray(rgb).save(img_p)

    mask = np.full((80, 100), 50, dtype=np.uint8)
    # Liver at y: 8..40 (100..500), x: 10..80 (100..800)
    mask[8:40, 10:80] = 21
    # Grasper in foreground at y: 20..60 (250..750), x: 30..70 (300..700)
    mask[20:60, 30:70] = 31
    Image.fromarray(mask).save(mask_p)

    # Mock caller: pass1 and pass2 disagree on liver in round 0, then agree in round 1
    async def mock_caller(prompt, *, model=None, images=(), schema=None, tag="", sample=0, **kw):
        if tag == "localize-pass1":
            return {
                "phase": "2. Calot's (Hepatocystic) Triangle Dissection",
                "phase_confidence": 0.92,
                "items": [
                    {"label": "grasper", "box_2d": [250, 300, 750, 700], "tip_2d": [500, 500], "confidence": 0.94},
                    {"label": "liver", "box_2d": [100, 100, 500, 800], "tip_2d": [], "confidence": 0.91},
                ],
            }
        if tag == "localize-pass2":
            return {
                "phase": "2. Calot's (Hepatocystic) Triangle Dissection",
                "phase_confidence": 0.96,
                "items": [
                    {"label": "grasper", "box_2d": [255, 305, 745, 695], "tip_2d": [495, 505], "confidence": 0.95},
                    # Intentionally small box on liver so Round 0 IoU < 0.5 -> triggers Amber loop
                    {"label": "liver", "box_2d": [100, 100, 220, 300], "tip_2d": [], "confidence": 0.85},
                ],
            }
        # Round 1 tighter crop calls agree on full crop extent
        return {
            "box_2d": [100, 100, 900, 900],
            "tip_2d": [],
            "confidence": 0.93,
            "reason": "Tighter crop confirms full liver lobe extent.",
        }

    frame_df = pd.DataFrame(
        [
            {
                "frame_id": "video09_00832_832",
                "video": "video09",
                "split": "test",
                "key": "grasper",
                "box": "[250, 300, 750, 700]",
                "conf": 0.95,
                "frame_path": str(img_p),
            },
            {
                "frame_id": "video09_00832_832",
                "video": "video09",
                "split": "test",
                "key": "liver",
                "box": "[100, 100, 500, 800]",
                "conf": 0.91,
                "frame_path": str(img_p),
            },
        ]
    )
    live_p = tmp_path / "annotations_live.jsonl"
    anns, rnds = asyncio.run(
        localize_frame(
            "video09_00832_832",
            frame_df,
            {"phases": DEFAULT_PHASES, "structures": []},
            live_file=live_p,
            call_fn=mock_caller,
        )
    )
    assert len(anns) == 2
    by_lbl = {a["label"]: a for a in anns}
    assert by_lbl["grasper"]["status"] == "accept"
    assert by_lbl["grasper"]["round"] == 0
    assert by_lbl["grasper"]["tip"] is not None
    assert by_lbl["liver"]["status"] == "accept"
    assert by_lbl["liver"]["round"] == 1
    assert by_lbl["liver"]["tip"] is None
    assert len(rnds) == 2  # Round 0 AMBER + Round 1 accept

    ann_p = tmp_path / "annotations.jsonl"
    rnd_p = tmp_path / "localize_rounds.jsonl"
    idx_p = tmp_path / "index.csv"
    res_p = tmp_path / "localize_results.json"

    ann_p.write_text("\n".join(json.dumps(a) for a in anns) + "\n")
    rnd_p.write_text("\n".join(json.dumps(r) for r in rnds) + "\n")
    pd.DataFrame(
        [
            {
                "frame_id": "video09_00832_832",
                "video": "video09",
                "split": "test",
                "path": str(img_p),
                "mask_path": str(mask_p),
            }
        ]
    ).to_csv(idx_p, index=False)

    res = evaluate_annotations(
        annotations_path=ann_p,
        index_path=idx_p,
        rounds_path=rnd_p,
        out_path=res_p,
    )
    assert res["n_test_annotations"] == 2
    assert res["boxes"]["mean_iou"] > 0.65
    assert res["tips"]["within_gt_tool_mask_rate"] == 1.0
    assert res["amber_loop"]["ambers_by_round"][0] == 1
    assert res["amber_loop"]["ambers_by_round"][1] == 0
