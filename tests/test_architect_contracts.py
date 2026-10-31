"""Tests for architect-owned contracts (config.py, gem.py, docs/DESIGN.md)."""
from pathlib import Path
from PIL import Image

import config as C
import gem


def test_config_contracts():
    assert set(C.LABELS.keys()) == {
        "grasper",
        "l_hook",
        "liver",
        "gallbladder",
        "cystic_duct",
        "blood",
    }
    for key, (cid, name) in C.LABELS.items():
        assert isinstance(cid, int)
        assert isinstance(name, str) and len(name) > 0
    assert C.FRAME_STRIDE > 0
    assert C.N_TEST_VIDEOS > 0
    assert C.N_CALIB_VIDEOS >= 0


def test_gem_img_part_and_key():
    im = Image.new("RGB", (400, 300), color=(120, 40, 60))
    part, raw_bytes = gem.img_part(im, max_side=256, crop=(100, 100, 800, 800))
    assert len(raw_bytes) > 0
    assert part is not None
    # Degenerate and inverted crops should be clamped safely without raising ValueError
    for bad_crop in [(500, 500, 500, 500), (800, 800, 200, 200), (1200, 1200, 1500, 1500)]:
        p2, b2 = gem.img_part(im, max_side=256, crop=bad_crop)
        assert len(b2) > 0 and p2 is not None
    mode = gem._db.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode.lower() == "wal"
    k1 = gem._key(C.FLASH, ["sig1"], "prompt", "Schema", [0.0, 0, False])
    k2 = gem._key(C.FLASH, ["sig1"], "prompt", "Schema", [0.0, 1, False])
    assert k1 != k2


def test_design_doc_exists_and_covers_all_contracts():
    design_path = Path(C.ROOT) / "docs" / "DESIGN.md"
    assert design_path.is_file(), "docs/DESIGN.md must exist"
    text = design_path.read_text(encoding="utf-8")
    for required in [
        "index.csv",
        "features.npy",
        "manual.json",
        "labels_manual.jsonl",
        "decisions.csv",
        "results.json",
        "student.json",
    ]:
        assert required in text, f"Missing contract section for {required} in docs/DESIGN.md"
