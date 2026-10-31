"""Tests for architect-owned contracts (config.py, gem.py, docs/DESIGN.md, zero-GT verifier guard)."""
import ast
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
        "SurgProof",
        "AMBER",
        "human_gate_log.jsonl",
        "index.csv",
        "features.npy",
        "manual.json",
        "labels_manual.jsonl",
        "decisions.csv",
        "results.json",
        "student.json",
    ]:
        assert required in text, f"Missing contract section for {required} in docs/DESIGN.md"


def _check_file_for_gt_references(py_path: Path):
    src = py_path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_path))
    forbidden_hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.id == "truth" or node.id.startswith("gt_")):
            forbidden_hits.append(f"{py_path.name}:{getattr(node, 'lineno', '?')} Name({node.id})")
        elif isinstance(node, ast.Attribute) and (node.attr == "truth" or node.attr.startswith("gt_")):
            forbidden_hits.append(f"{py_path.name}:{getattr(node, 'lineno', '?')} Attr(.{node.attr})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            val = node.value.strip()
            if val == "truth" or val.startswith("gt_"):
                forbidden_hits.append(f"{py_path.name}:{getattr(node, 'lineno', '?')} Str({val!r})")
    return forbidden_hits


def test_verifier_never_reads_gt_or_truth():
    """Audit rule: accept.py and verify/ must NEVER read or reference `truth` or `gt_*`."""
    root = Path(C.ROOT)
    targets = [root / "accept.py"]
    verify_dir = root / "verify"
    if verify_dir.is_dir():
        targets.extend(sorted(verify_dir.rglob("*.py")))
    all_hits = []
    for t in targets:
        if t.is_file():
            all_hits.extend(_check_file_for_gt_references(t))
    assert not all_hits, (
        f"Ground-truth leakage detected in verifier code! Verifier must never read truth/gt_*: {all_hits}"
    )
