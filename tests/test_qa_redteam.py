"""Red team: try to catch the pipeline, server and UI fabricating. Assumes the other lanes are wrong until shown otherwise.

Static checks read the source. Dynamic checks run the real CLI against a fake `claude` binary that misbehaves on
command: if the model is dead, nothing may appear; if it says nothing is there, nothing may be drawn; if it lies in
narration, the lie may not reach a subtitle. Bugs found here go to docs/e2e_status/qa.md.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.e2e_helpers.harness import (HAVE_FFMPEG, LABELS, ROOT, base_env, cache_rows, fake_env, read_json, read_jsonl,
                                       run_cli, synthetic_video)

SURGPROOF = ROOT / "surgproof"
LICENSED_GT_TOKENS = [
    r"watershed", r"_endo_mask", r"_color_mask", r"\bgt_\w+", r"mask_path", r"index\.csv", r"CholecSeg8k",
    r"gt_registry", r"eval_report", r"\braw/", r"surgatlas", r"prepare_data", r"make_test_video",
]
# Same patterns lanes/integrity.py applies to display code. integrity.py itself does not scan surgproof/ or web/app/.
DISPLAY_PATTERNS = [
    (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
    (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
    (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
    (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
    (r'(accuracy|acc|precision|coverage|iou)\s*[=:]\s*0\.\d{2,}', "hard-coded metric"),
    (r'\b(96\.\d|99\.\d|93\.\d)\s*%', "headline number typed by hand"),
]


def pipeline_sources():
    return [p for p in SURGPROOF.glob("*.py") if p.name != "eval.py"]


def display_sources():
    out = [p for p in (SURGPROOF / "server.py", SURGPROOF / "render.py") if p.exists()]
    app = ROOT / "web" / "app"
    if app.is_dir():
        out += [p for p in app.rglob("*") if p.suffix in {".js", ".html", ".css", ".mjs", ".ts"} and "vendor" not in p.parts]
    return out


# ---------- static: ground truth ----------

def test_eval_is_the_only_ground_truth_reader():
    hits = []
    for p in pipeline_sources():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            for tok in LICENSED_GT_TOKENS:
                if re.search(tok, line):
                    hits.append(f"{p.name}:{i}: {tok}: {line.strip()[:100]}")
    assert not hits, "ground-truth access outside surgproof/eval.py:\n" + "\n".join(hits)


def test_nothing_imports_the_scorer():
    bad = []
    for p in pipeline_sources():
        for node in ast.walk(ast.parse(p.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [f"{node.module or ''}.{a.name}" for a in node.names] + [node.module or ""]
            if any(m in ("surgproof.eval", ".eval", "eval", "surgproof.eval.") or m.endswith("surgproof.eval") for m in mods):
                bad.append(f"{p.name}: {mods}")
        if re.search(r"""import_module\(\s*["']surgproof\.eval|surgproof\.eval["']""", p.read_text()):
            bad.append(f"{p.name}: dynamic import of the scorer")
    assert not bad, bad


def test_pipeline_does_not_import_repo_root_modules_that_touch_truth():
    for p in pipeline_sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.split(".")[0] in {"config", "prepare_data", "accept", "tracker"} for n in names), f"{p.name}: {names}"


def test_web_app_never_mentions_ground_truth():
    app = ROOT / "web" / "app"
    if not app.is_dir():
        pytest.skip("web lane not merged")
    for p in app.rglob("*"):
        if p.suffix in {".js", ".html"} and "vendor" not in p.parts:
            t = p.read_text(errors="ignore")
            for tok in (r"watershed", r"eval_report", r"gt_registry", r"\bgt_\w+"):
                assert not re.search(tok, t), f"{p}: {tok}"


# ---------- static: hand-typed results ----------

def test_no_hand_typed_results_in_display_code():
    hits = []
    for p in display_sources():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.startswith(("#", "//", "*")):
                continue
            for pat, what in DISPLAY_PATTERNS:
                if re.search(pat, line):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {what}: {s[:110]}")
    assert not hits, "\n".join(hits)


def test_integrity_marker_not_used_in_new_code():
    """'integrity: allow' is reserved for human-click lines, and none of this code should need it unreviewed."""
    hits = []
    for p in list(SURGPROOF.glob("*.py")) + display_sources():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if "integrity: allow" in line and "(human action)" not in line:
                hits.append(f"{p.name}:{i}")
    assert not hits, hits


def test_mock_provider_cannot_be_the_silent_default():
    src = (SURGPROOF / "vlm.py").read_text() if (SURGPROOF / "vlm.py").exists() else ""
    if not src:
        pytest.skip("core lane not merged")
    assert re.search(r'environ\.get\(\s*"SURGPROOF_VLM"\s*,\s*"claude_cli"\s*\)', src), "default provider must be claude_cli"
