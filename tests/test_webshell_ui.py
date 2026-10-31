"""Unit and contract tests for nir-webshell (web/index.html, web/style.css, web/app.js)."""
from __future__ import annotations

import pathlib
import re
import subprocess

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"


def test_webshell_files_exist():
    assert (WEB_DIR / "index.html").exists(), "web/index.html must exist"
    assert (WEB_DIR / "style.css").exists(), "web/style.css must exist"
    assert (WEB_DIR / "app.js").exists(), "web/app.js must exist"
    assert (WEB_DIR / "watch.html").exists(), "web/watch.html must exist"
    assert (WEB_DIR / "watch.js").exists(), "web/watch.js must exist"


def test_watch_mode_structure_and_syntax():
    index_html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    assert 'href="watch.html"' in index_html, "Top bar in web/index.html must link to watch.html"

    watch_html = (WEB_DIR / "watch.html").read_text(encoding="utf-8")
    assert "SurgProof · video12 (held-out)" in watch_html
    assert 'id="watch-canvas"' in watch_html
    assert 'id="watch-headline-chip"' in watch_html
    assert 'id="watch-subtitle-text"' in watch_html
    assert 'id="watch-play-btn"' in watch_html
    assert 'id="watch-scrub"' in watch_html
    assert 'src="watch.js"' in watch_html

    watch_js_path = WEB_DIR / "watch.js"
    watch_js = watch_js_path.read_text(encoding="utf-8")
    res = subprocess.run(["node", "--check", str(watch_js_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"node --check failed on web/watch.js: {res.stderr}"
    assert "video12.json" in watch_js
    assert "narration_video12.json" in watch_js


def test_index_html_mount_points_and_structure():
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")

    # Top bar: SurgProof wordmark, video12 badge, headline number with n
    assert "SurgProof" in html
    assert "video12" in html
    assert 'id="headline-acc"' in html
    assert 'id="headline-n"' in html

    # Required component mount points
    assert 'id="viewer"' in html, "Centre canvas mount point #viewer required"
    assert 'id="objects"' in html, "Right panel mount point #objects required"
    assert 'id="timeline"' in html, "Bottom timeline mount point #timeline required"
    assert 'id="final-pass-drawer"' in html, "Final pass drawer required"

    # Script wiring for nir-webcanvas, nir-webtimeline, nir-gate, and app.js
    assert 'src="viewer.js"' in html
    assert 'src="timeline.js"' in html
    assert 'src="app.js"' in html
    assert 'href="style.css"' in html

    # Keyboard hints present
    assert "<kbd>Space</kbd>" in html
    assert "<kbd>F</kbd>" in html


def test_style_css_design_md_tokens():
    css = (WEB_DIR / "style.css").read_text(encoding="utf-8")

    # Exact tokens from promo/DESIGN.md
    for token in (
        "#0A0B0E",  # Canvas background
        "#12141C",  # Elevated card surface
        "#232734",  # Hairline border
        "#F4F5F8",  # Primary text
        "#8A909E",  # Secondary text
        "#3B82F6",  # Single accent blue
        "#10B981",  # Accept green
        "#092E24",  # Accept pill fill
        "#F59E0B",  # Amber unsure
        "#362407",  # Amber pill fill
        "#EF4444",  # Reject red
        "#331014",  # Reject pill fill
    ):
        assert token.lower() in css.lower(), f"Missing promo/DESIGN.md colour token {token} in web/style.css"

    # Typography: Inter / SF Pro, never Arial in font-family declarations
    assert "Inter" in css
    font_lines = [ln for ln in css.splitlines() if "font-family" in ln.lower()]
    assert font_lines, "Expected font-family declarations in web/style.css"
    for ln in font_lines:
        assert "arial" not in ln.lower(), f"Forbidden Arial font fallback in {ln}"


def test_app_js_syntax_and_zero_fabrication():
    app_js_path = WEB_DIR / "app.js"
    js = app_js_path.read_text(encoding="utf-8")

    # Check valid JS syntax via node --check if node is installed
    res = subprocess.run(["node", "--check", str(app_js_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"node --check failed on web/app.js: {res.stderr}"

    # Verify component integration hooks
    assert "SurgProofViewer" in js or "window.Viewer" in js
    assert "SurgProofTimeline" in js or "window.Timeline" in js
    assert "video12.json" in js
    assert "live.json" in js
    assert "/api/gate" in js
    assert "pending" in js

    # Verify lanes/integrity.py display patterns never trigger on web/{index.html,style.css,app.js}
    patterns = [
        (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
    ]
    for fname in ("index.html", "style.css", "app.js", "watch.html", "watch.js"):
        text = (WEB_DIR / fname).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or "integrity: allow" in s:
                continue
            for pat, what in patterns:
                assert not re.search(pat, line), f"{fname}:{lineno}: {what}: {s[:110]}"
