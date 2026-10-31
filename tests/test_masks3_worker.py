"""Unit and integrity tests for `/Users/nirmathur/surgatlas-lanes/masks3/masks_worker.py`."""
from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

LANE_ROOT = Path(__file__).resolve().parents[1]
if str(LANE_ROOT) not in sys.path:
    sys.path.insert(0, str(LANE_ROOT))

import config as C
import masks_worker as MW
from tools import score_masks as SM


def _executable_lines(py_path: Path) -> list[tuple[int, str]]:
    """Return `(lineno, code_line)` for all executable lines in `py_path`,
    excluding module/class/function docstrings and `#` comments.
    """
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))

    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                expr_node = node.body[0]
                start = getattr(expr_node, "lineno", 0)
                end = getattr(expr_node, "end_lineno", start)
                for ln in range(start, end + 1):
                    docstring_lines.add(ln)

    exec_lines: list[tuple[int, str]] = []
    for idx, raw_line in enumerate(src.splitlines(), start=1):
        if idx in docstring_lines:
            continue
        code_part = raw_line.split("#", 1)[0].strip()
        if not code_part:
            continue
        exec_lines.append((idx, code_part))
    return exec_lines


def test_masks_worker_zero_leakage() -> None:
    """Verify `masks_worker.py` never references `gt_`, `truth`, or `mask_path` in any executable line."""
    worker_path = LANE_ROOT / "masks_worker.py"
    assert worker_path.is_file(), f"Missing {worker_path}"

    forbidden = ("gt_", "truth", "mask_path")
    violations: list[str] = []
    for lineno, code_line in _executable_lines(worker_path):
        low = code_line.lower()
        for token in forbidden:
            if token in low:
                violations.append(f"line {lineno} contains '{token}': {code_line}")

    assert not violations, "Zero-leakage violation(s) found:\n" + "\n".join(violations)


def test_get_video_frames_by_parity() -> None:
    """Verify `video12` parity splitting:
    - `parity='odd'` returns 40 frames starting with index 1 (`video12_15750_15758`)
    - `parity='even'` returns 40 disjoint frames starting with index 0 (`video12_15750_15750`)
    - union = 80 frames and intersection = empty.
    """
    odd_frames = MW.get_video_frames_by_parity(video="video12", parity="odd")
    even_frames = MW.get_video_frames_by_parity(video="video12", parity="even")
    all_frames = MW.get_video_frames_by_parity(video="video12", parity="all")

    assert len(odd_frames) == 40
    assert len(even_frames) == 40
    assert len(all_frames) == 80

    assert odd_frames[0] == "video12_15750_15758"
    assert even_frames[0] == "video12_15750_15750"

    odd_set = set(odd_frames)
    even_set = set(even_frames)
    assert len(odd_set & even_set) == 0
    assert len(odd_set | even_set) == 80
    assert odd_set | even_set == set(all_frames)


def test_mask_iou_and_build_structure_mask(tmp_path: Path, monkeypatch) -> None:
    """Test `mask_iou`, `mask_dice`, `build_structure_mask`, `extract_contour_polygon`, and `save_mask_png`
    on a synthetic RGB surgical frame, verifying RGBA PNG output has transparent background `(0,0,0,0)`
    and non-zero coloured foreground `(R,G,B,255)`.
    """
    # Avoid writing temporary test masks into the shared web/data/masks folder
    monkeypatch.setattr(C, "ROOT", tmp_path)

    m_a = np.zeros((100, 120), dtype=bool)
    m_b = np.zeros((100, 120), dtype=bool)
    m_a[20:60, 30:80] = True
    m_b[20:60, 30:80] = True

    assert MW.mask_iou(m_a, m_b) == 1.0
    assert MW.mask_dice(m_a, m_b) == 1.0
    assert MW.mask_iou(m_a, ~m_a) == 0.0
    assert MW.mask_dice(m_a, ~m_a) == 0.0
    assert MW.mask_iou(None, m_a) == 0.0

    # Synthetic surgical frame: warm reddish tissue background + metallic grey instrument in box
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb[:, :] = (165, 70, 65)  # reddish-brown liver parenchyma
    rgb[50:140, 80:210] = (185, 190, 195)  # metallic surgical instrument region

    for lbl, box, tip in [
        ("grasper", [250, 260, 700, 700], [650, 650]),
        ("liver", [100, 100, 850, 850], None),
        ("gallbladder", [200, 200, 750, 750], None),
        ("cystic_duct", [300, 300, 650, 650], None),
    ]:
        mask = MW.build_structure_mask(rgb, lbl, box, tip=tip)
        assert mask.shape == (200, 300)
        assert mask.dtype == bool
        assert int(mask.sum()) > 50
        # Outside the normalized box must remain False
        assert not mask[:15, :15].any()

        poly = MW.extract_contour_polygon(mask, n_vertices=12)
        assert len(poly) >= 4
        for pt in poly:
            assert len(pt) == 2
            assert 0 <= pt[0] <= 1000
            assert 0 <= pt[1] <= 1000

        out_png = tmp_path / "masks" / f"synth_{lbl}.png"
        MW.save_mask_png(mask, lbl, out_png)
        assert out_png.is_file()

        loaded = np.array(Image.open(out_png))
        assert loaded.shape == (200, 300, 4)
        # Background pixels must be transparent (0, 0, 0, 0)
        bg_pixels = loaded[~mask]
        assert np.all(bg_pixels == 0)
        # Foreground pixels must have expected RGB and alpha == 255
        fg_pixels = loaded[mask]
        exp_r, exp_g, exp_b = MW.LABEL_RGB[lbl]
        assert np.all(fg_pixels[:, 0] == exp_r)
        assert np.all(fg_pixels[:, 1] == exp_g)
        assert np.all(fg_pixels[:, 2] == exp_b)
        assert np.all(fg_pixels[:, 3] == 255)


def test_verify_and_refine_mask_item_accept_and_amber_loop(tmp_path: Path) -> None:
    """Verify `verify_and_refine_mask_item` behavior with a mock `call_fn`:
    1. When Pass 1 and Pass 2 boxes/masks agree (`mask_iou >= 0.5`), returns `status == 'accept'`
       at `round == 0` with empty `rounds_log`.
    2. When Pass 1 and Pass 2 initially disagree (`mask_iou < 0.5`), logs `round == 0` as `AMBER`
       and runs tighter-crop rounds until `mask_iou >= 0.5` resolves to `accept` (or stays `AMBER`
       after `max_rounds`).
    """
    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    rgb[:, :] = (160, 75, 65)
    rgb[40:160, 60:240] = (175, 180, 185)
    synth_img_path = tmp_path / "synth_frame.png"
    Image.fromarray(rgb, mode="RGB").save(synth_img_path)

    manual: dict[str, Any] = {"structures": {"gallbladder": {"definition": "Gallbladder sac"}}}

    # Case 1: Pass 1 and Pass 2 agree immediately (mask_iou >= 0.5)
    calls_made = 0

    async def _unexpected_call(*args, **kwargs):
        nonlocal calls_made
        calls_made += 1
        return {}

    res_acc, mask_acc, log_acc = asyncio.run(
        MW.verify_and_refine_mask_item(
            frame_id="video12_15750_15758",
            video="video12",
            split="test",
            img_path=str(synth_img_path),
            rgb_arr=rgb,
            label="gallbladder",
            pass1={"box": [200, 200, 750, 750], "tip": None, "conf": 0.90},
            pass2={"box": [210, 210, 760, 760], "tip": None, "conf": 0.92},
            manual=manual,
            iou_thresh=0.5,
            max_rounds=3,
            call_fn=_unexpected_call,
        )
    )
    assert res_acc["status"] == "accept"
    assert res_acc["round"] == 0
    assert res_acc["mask_iou"] >= 0.5
    assert log_acc == []
    assert calls_made == 0
    assert mask_acc.shape == (200, 300) and mask_acc.any()

    # Case 2: Pass 1 and Pass 2 initially disagree (mask_iou < 0.5), then agree in Round 1
    async def _converging_call(prompt: str, *, model: str, images: list, schema: Any, tag: str, sample: int):
        return {
            "box_2d": [200, 200, 800, 800],
            "tip_2d": [],
            "confidence": 0.91,
            "reason": "Converged on gallbladder contour in zoomed crop",
        }

    res_conv, mask_conv, log_conv = asyncio.run(
        MW.verify_and_refine_mask_item(
            frame_id="video12_15750_15758",
            video="video12",
            split="test",
            img_path=str(synth_img_path),
            rgb_arr=rgb,
            label="gallbladder",
            pass1={"box": [50, 50, 300, 300], "tip": None, "conf": 0.85},
            pass2={"box": [650, 650, 920, 920], "tip": None, "conf": 0.88},
            manual=manual,
            iou_thresh=0.5,
            max_rounds=3,
            call_fn=_converging_call,
        )
    )
    assert res_conv["status"] == "accept"
    assert res_conv["round"] == 1
    assert res_conv["mask_iou"] >= 0.5
    assert len(log_conv) == 2
    assert log_conv[0]["round"] == 0 and log_conv[0]["verdict"] == "AMBER"
    assert log_conv[1]["round"] == 1 and log_conv[1]["verdict"] == "accept"
    assert mask_conv.shape == (200, 300) and mask_conv.any()

    # Case 3: Pass 1 and Pass 2 keep disagreeing across all max_rounds -> stays AMBER
    async def _diverging_call(prompt: str, *, model: str, images: list, schema: Any, tag: str, sample: int):
        if "flash" in tag:
            return {"box_2d": [50, 50, 250, 250], "tip_2d": [], "confidence": 0.80, "reason": "Upper-left candidate"}
        return {"box_2d": [750, 750, 950, 950], "tip_2d": [], "confidence": 0.82, "reason": "Lower-right candidate"}

    res_amb, mask_amb, log_amb = asyncio.run(
        MW.verify_and_refine_mask_item(
            frame_id="video12_15750_15758",
            video="video12",
            split="test",
            img_path=str(synth_img_path),
            rgb_arr=rgb,
            label="gallbladder",
            pass1={"box": [50, 50, 300, 300], "tip": None, "conf": 0.85},
            pass2={"box": [650, 650, 920, 920], "tip": None, "conf": 0.88},
            manual=manual,
            iou_thresh=0.5,
            max_rounds=2,
            call_fn=_diverging_call,
        )
    )
    assert res_amb["status"] == "AMBER"
    assert res_amb["round"] == 2
    assert len(log_amb) == 3
    assert all(entry["verdict"] == "AMBER" for entry in log_amb)
    assert [entry["round"] for entry in log_amb] == [0, 1, 2]
    assert mask_amb.shape == (200, 300) and mask_amb.any()


def test_score_masks_metrics_and_bootstrap_ci(tmp_path: Path, monkeypatch) -> None:
    """Test `compute_iou_and_dice`, `best_instance_iou_and_dice`, `video_cluster_bootstrap_ci`,
    and `score_all_masks` on synthetic temp directories (and verify `out/mask_results.json` schema
    if present on disk).
    """
    # 1. compute_iou_and_dice
    m1 = np.zeros((40, 60), dtype=bool)
    m2 = np.zeros((40, 60), dtype=bool)
    m1[10:30, 10:30] = True  # 400 px
    m2[10:30, 20:40] = True  # 400 px, overlap [10:30, 20:30] = 200 px, union = 600 px

    iou_eq, dice_eq = SM.compute_iou_and_dice(m1, m1)
    assert iou_eq == 1.0 and dice_eq == 1.0

    iou_ov, dice_ov = SM.compute_iou_and_dice(m1, m2)
    assert abs(iou_ov - (1.0 / 3.0)) < 1e-6
    assert abs(dice_ov - 0.5) < 1e-6

    iou_dis, dice_dis = SM.compute_iou_and_dice(m1, ~m1)
    assert iou_dis == 0.0 and dice_dis == 0.0
    assert SM.compute_iou_and_dice(None, m1) == (0.0, 0.0)

    # 2. best_instance_iou_and_dice (with separated connected components)
    gt_two = np.zeros((80, 100), dtype=bool)
    gt_two[5:25, 5:25] = True    # instance 1
    gt_two[55:75, 70:90] = True  # instance 2
    comps = SM._extract_gt_components(gt_two, min_pixels=10)
    assert len(comps) >= 3

    pred_inst1 = np.zeros((80, 100), dtype=bool)
    pred_inst1[5:25, 5:25] = True
    full_iou, _ = SM.compute_iou_and_dice(pred_inst1, gt_two)
    best_iou, best_dice = SM.best_instance_iou_and_dice(pred_inst1, comps)
    assert full_iou == 0.5
    assert best_iou == 1.0 and best_dice == 1.0
    assert SM.best_instance_iou_and_dice(pred_inst1, []) == (0.0, 0.0)

    # 3. video_cluster_bootstrap_ci
    synth_records = [
        {"video": "video09", "iou": 0.62, "dice": 0.76},
        {"video": "video09", "iou": 0.68, "dice": 0.81},
        {"video": "video12", "iou": 0.54, "dice": 0.70},
        {"video": "video12", "iou": 0.58, "dice": 0.73},
        {"video": "video17", "iou": 0.72, "dice": 0.84},
    ]
    ci = SM.video_cluster_bootstrap_ci(synth_records, "iou", n_boot=200, seed=0)
    assert set(ci.keys()) >= {"mean", "ci_low", "ci_high", "se", "n_items", "n_videos", "n_boot"}
    assert ci["n_items"] == 5
    assert ci["n_videos"] == 3
    assert ci["n_boot"] == 200
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]

    empty_ci = SM.video_cluster_bootstrap_ci([], "iou", n_boot=50, seed=0)
    assert empty_ci["mean"] == 0.0 and empty_ci["n_items"] == 0 and empty_ci["n_videos"] == 0

    # 4. score_all_masks on synthetic temporary directory
    required_keys = {
        "n_test_masks",
        "n_accepted_masks",
        "n_test_videos",
        "overall",
        "accepted",
        "per_round_improvement",
        "per_label",
        "per_video",
        "video_bootstrap_ci",
    }

    real_out_json = LANE_ROOT / "out" / "mask_results.json"
    if real_out_json.is_file():
        real_data = json.loads(real_out_json.read_text(encoding="utf-8"))
        assert required_keys.issubset(real_data.keys())

    tmp_out = tmp_path / "out_synth"
    tmp_poly = tmp_out / "polygons"
    tmp_masks = tmp_out / "masks"
    tmp_poly.mkdir(parents=True, exist_ok=True)
    tmp_masks.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(C, "OUT_DIR", tmp_out)

    # Create dummy GT mask file and monkeypatch SM.mask_to_classes for fast synthetic evaluation
    gt_file = tmp_path / "dummy_gt.png"
    Image.fromarray(np.zeros((60, 80), dtype=np.uint8)).save(gt_file)

    gb_cid = C.LABELS["gallbladder"][0]
    synth_cls = np.zeros((60, 80), dtype=np.uint8)
    synth_cls[12:48, 16:64] = gb_cid
    monkeypatch.setattr(SM, "mask_to_classes", lambda _p: synth_cls)

    idx_csv = tmp_path / "index.csv"
    pd.DataFrame(
        [
            {
                "frame_id": "video12_15750_15750",
                "video": "video12",
                "split": "test",
                "path": str(gt_file),
                "mask_path": str(gt_file),
            },
            {
                "frame_id": "video01_00080_00080",
                "video": "video01",
                "split": "train",
                "path": str(gt_file),
                "mask_path": str(gt_file),
            },
        ]
    ).to_csv(idx_csv, index=False)

    # Save predicted RGBA mask matching the synthetic GT region
    pred_rgba = np.zeros((60, 80, 4), dtype=np.uint8)
    pred_rgba[12:48, 16:64] = (72, 187, 120, 255)
    Image.fromarray(pred_rgba, mode="RGBA").save(tmp_masks / "video12_15750_15750_gallbladder.png")

    (tmp_poly / "video12.jsonl").write_text(
        json.dumps(
            {
                "frame_id": "video12_15750_15750",
                "video": "video12",
                "label": "gallbladder",
                "status": "accept",
                "rounds": [{"round": 0, "iou_pass1_pass2": 0.85}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_res_json = tmp_out / "mask_results.json"
    scored = SM.score_all_masks(
        index_path=idx_csv,
        polygons_dir=tmp_poly,
        masks_dir=tmp_masks,
        out_path=out_res_json,
        n_boot=50,
    )

    assert out_res_json.is_file()
    assert required_keys.issubset(scored.keys())
    assert scored["n_test_masks"] == 1
    assert scored["n_accepted_masks"] == 1
    assert scored["n_test_videos"] == 1
    assert scored["overall"]["mean_iou"] == 1.0
    assert scored["accepted"]["mean_dice"] == 1.0
    assert "gallbladder" in scored["per_label"]
    assert "video12" in scored["per_video"]

