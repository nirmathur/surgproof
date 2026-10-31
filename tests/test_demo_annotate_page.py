"""Tests for pages/1_Annotate.py (Supervisely-style AI Video Annotation Studio for video12)."""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from streamlit.testing.v1 import AppTest

import config as C

ANNOTATE_PAGE_PATH = Path(C.ROOT) / "pages" / "1_Annotate.py"

INTEGRITY_PATTERNS = [
    (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
    (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
    (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
    (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
    (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
]


def _load_annotate_module():
    spec = importlib.util.spec_from_file_location("annotate_studio_mod", ANNOTATE_PAGE_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_annotate_page_passes_integrity_patterns_with_zero_violations():
    """Verify pages/1_Annotate.py has 0 violations against lanes/integrity.py PATTERNS (even in comments)."""
    assert ANNOTATE_PAGE_PATH.is_file(), "pages/1_Annotate.py must exist"
    text = ANNOTATE_PAGE_PATH.read_text(encoding="utf-8")
    assert "integrity: allow" not in text, "pages/1_Annotate.py must never use 'integrity: allow'"

    violations = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for pat, desc in INTEGRITY_PATTERNS:
            if re.search(pat, line):
                violations.append(f"line {line_no} ({desc}): {line.strip()}")
    assert not violations, f"Integrity pattern violations found:\n" + "\n".join(violations)


def test_annotate_page_loads_cleanly_on_real_workspace():
    """Verify AppTest loads pages/1_Annotate.py on the real workspace and renders video12 studio."""
    at = AppTest.from_file(str(ANNOTATE_PAGE_PATH))
    at.run(timeout=20)
    assert not at.exception, f"AppTest raised exception: {at.exception}"

    titles = [t.value for t in at.title]
    assert any("AI Video Annotation Studio" in t and "video12" in t for t in titles)

    # Check buttons exist and can be clicked without error
    button_labels = [b.label for b in at.button]
    assert "◀ Prev" in button_labels
    assert "▶ Next" in button_labels
    assert "▶ Play / Step" in button_labels
    assert "⚡ Run / Stream Agent Annotations" in button_labels

    # Click Next and Stream buttons to verify interactive state transitions
    at.button[1].click().run(timeout=20)
    assert not at.exception
    at.button[3].click().run(timeout=20)
    assert not at.exception

    # Verify GT scoring toggle and AI mask toggle exist
    toggle_labels = [t.label for t in at.toggle]
    assert any("Scoring View Only: Overlay CholecSeg8k Ground-Truth Mask" in lbl for lbl in toggle_labels)
    assert any("Overlay AI Masks" in lbl for lbl in toggle_labels)


def test_annotate_page_loads_cleanly_on_empty_temp_directory(tmp_path, monkeypatch):
    """Verify AppTest loads pages/1_Annotate.py on an empty directory, showing pending without crashing."""
    empty_out = tmp_path / "empty_out"
    empty_data = tmp_path / "empty_data"
    empty_out.mkdir()
    empty_data.mkdir()

    monkeypatch.setattr(C, "OUT_DIR", empty_out)
    monkeypatch.setattr(C, "DATA_DIR", empty_data)

    at = AppTest.from_file(str(ANNOTATE_PAGE_PATH))
    at.run(timeout=20)
    assert not at.exception, f"AppTest raised exception on empty dir: {at.exception}"

    metric_values = [m.value for m in at.metric]
    assert metric_values, "Headline metric widgets should still render"
    assert all(v == "pending" for v in metric_values), f"Expected all 'pending' on empty dir, got {metric_values}"

    all_markdown = "\n".join(m.value for m in at.markdown)
    assert "pending" in all_markdown.lower()


def test_canvas_and_timeline_helpers_with_masks_and_pending(tmp_path):
    """Unit-test mask blending, canvas rendering, inspector cards, and multi-track timeline HTML."""
    mod = _load_annotate_module()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()

    fid = "video12_15750_15758"
    dec_df = pd.DataFrame([
        {
            "frame_id": fid,
            "video": "video12",
            "key": "grasper",
            "present": 1,
            "final_present": 1,
            "conf": 0.96,
            "evidence": "Metallic jaws grasping tissue.",
            "pv_match": True,
            "p_wrong": 0.04,
            "status": "accept",
            "status_round_0": "accept",
            "status_round_1": "accept",
            "status_round_2": "accept",
            "status_round_3": "accept",
            "argon_reason": "",
        }
    ])
    ann_by_label = {
        "grasper": {
            "frame_id": fid,
            "video": "video12",
            "label": "grasper",
            "box": [200, 150, 600, 550],
            "tip": [420, 310],
            "phase": "2. Calot's (Hepatocystic) Triangle Dissection",
            "phase_conf": 0.95,
            "pass1": {"box": [205, 152, 598, 548]},
            "pass2": {"box": [198, 148, 602, 552], "iou": 0.94},
            "round": 0,
            "status": "accept",
        }
    }
    phase_info = {"phase": "2. Calot's (Hepatocystic) Triangle Dissection", "phase_conf": 0.95}

    # 1. Without mask files on disk -> mask_status is '(masks: pending)'
    img1, meta1 = mod.render_annotation_canvas(
        fid,
        None,
        dec_df,
        ann_by_label,
        phase_info,
        show_ai_masks=True,
        show_pass_boxes=True,
        show_gt_overlay=False,
        out_dir=tmp_path,
    )
    assert isinstance(img1, Image.Image)
    assert meta1["masks_applied"] == 0
    assert meta1["mask_status"] == "(masks: pending)"
    assert meta1["n_boxes_drawn"] == 1
    assert meta1["n_tips_drawn"] == 1

    # 2. Write an AI segmentation mask to tmp_path/masks/<fid>_grasper.png -> mask is blended
    mask_arr = np.zeros((480, 854), dtype=np.uint8)
    mask_arr[120:240, 200:360] = 255
    Image.fromarray(mask_arr).save(masks_dir / f"{fid}_grasper.png")

    img2, meta2 = mod.render_annotation_canvas(
        fid,
        None,
        dec_df,
        ann_by_label,
        phase_info,
        show_ai_masks=True,
        out_dir=tmp_path,
    )
    assert isinstance(img2, Image.Image)
    assert meta2["masks_applied"] == 1

    # 3. Inspector records include all 6 structures (with 'pending' for missing structures)
    recs = mod.build_inspector_records(fid, dec_df, ann_by_label)
    assert len(recs) == 6
    by_lbl = {r["label"]: r for r in recs}
    assert by_lbl["grasper"]["status"] == "accept"
    assert by_lbl["grasper"]["presence"] == "PRESENT"
    assert by_lbl["l_hook"]["status"] == "pending"

    # 4. Multi-track timeline HTML renders all 6 tracks + Phase track
    html = mod.build_multitrack_timeline_html(
        [fid],
        fid,
        dec_df,
        {(fid, "grasper"): ann_by_label["grasper"]},
        {fid: phase_info},
    )
    assert "Surgical Phase" in html
    for k in mod.TRACK_ORDER:
        assert k in html
