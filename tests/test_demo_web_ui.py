"""Tests for web/index.html and web/app.js (AI-native Supervisely surgical video annotation studio)."""
from pathlib import Path
import re
import shutil
import subprocess

import config as C

WEB_DIR = Path(C.ROOT) / "web"

INTEGRITY_PATTERNS = [
    (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
    (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
    (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
    (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
    (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
]


def test_web_ui_files_exist_and_have_required_toolbox_elements():
    html_path = WEB_DIR / "index.html"
    js_path = WEB_DIR / "app.js"
    assert html_path.is_file(), "web/index.html must exist"
    assert js_path.is_file(), "web/app.js must exist"

    html = html_path.read_text()
    js = js_path.read_text()

    required_ids = [
        "viewer",
        "objects",
        "timeline",
        "frame-canvas",
        "sel-video",
        "btn-guided-mode",
        "guided-beat-banner",
        "btn-stream-agent",
        "btn-final-pass",
        "chk-ai-masks",
        "chk-ai-polygons",
        "chk-ai-boxes",
        "chk-ai-tips",
        "chk-show-gt",
        "final-pass-drawer",
        "btn-release-dataset",
    ]
    for dom_id in required_ids:
        assert f'id="{dom_id}"' in html, f"Missing #{dom_id} in web/index.html"

    # Must poll live.json every 2 seconds, read video bundles, render polygon_px, and support 7-beat guided mode
    assert "video12" in js
    assert "live.json" in js
    assert "polygon_px" in js
    assert "setGuidedBeat" in js
    assert "pollLiveAnnotations" in js
    assert "2000" in js


def test_web_ui_passes_integrity_patterns_and_node_syntax():
    for fname in ("index.html", "app.js"):
        fpath = WEB_DIR / fname
        for idx, line in enumerate(fpath.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            for pat, desc in INTEGRITY_PATTERNS:
                assert not re.search(pat, line), f"{fname}:{idx} matched forbidden {desc}: {s}"

    node_bin = shutil.which("node")
    if node_bin:
        res = subprocess.run([node_bin, "--check", str(WEB_DIR / "app.js")], capture_output=True, text=True)
        assert res.returncode == 0, f"node --check web/app.js failed: {res.stderr}"
