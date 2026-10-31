#!/usr/bin/env python3
"""Automated Playwright Demo Recorder for SurgProof (tools/record_demo.py).

Drives the running SurgProof web studio (http://localhost:8600 -> out/recording_web.mp4)
and the Streamlit fallback (http://localhost:8501 -> out/recording_streamlit.mp4) in real
headless Chromium at 1920x1080 for 60-90 seconds:
  1. Loads the live application on real pipeline artifacts (never alters or invents data).
  2. Plays through the multi-track timeline / frame viewer.
  3. Opens multiple surgical frames with AI segmentation masks, bounding boxes, tip points,
     evidence inspector cards, and post-hoc ground-truth scoring overlay.
  4. Opens the Final Pass / Human Gate panel and displays the residual AMBER queue,
     deterministic 20-item audit sample, and locked dataset release gate.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
DEFAULT_WEB_URL = "http://localhost:8600"
DEFAULT_WATCH_URL = "http://localhost:8600/watch.html"
DEFAULT_STREAMLIT_URL = "http://localhost:8501"
DEFAULT_WEB_OUT = OUT_DIR / "recording_web_v1.mp4"
DEFAULT_WEB_ALIAS_OUT = OUT_DIR / "recording_web.mp4"
DEFAULT_WATCH_OUT = OUT_DIR / "recording_watch.mp4"
DEFAULT_STREAMLIT_OUT = OUT_DIR / "recording_streamlit.mp4"

# Subtle cursor ring + continuous rAF repaint heartbeat so Chromium CDP screencast
# emits smooth frames even while holding on a view for reading.
# Also intercepts POST requests to /api/gate and /gate in-browser so clicking Approve
# during a demo recording updates the UI visually without ever altering server/disk data.
CURSOR_INIT_SCRIPT = """
(() => {
  if (!window.__spOrigFetch && typeof window.fetch === 'function') {
    window.__spOrigFetch = window.fetch.bind(window);
    window.fetch = async function(input, init) {
      const url = typeof input === 'string' ? input : (input && input.url) || '';
      const method = ((init && init.method) || (input && input.method) || 'GET').toUpperCase();
      if (method === 'POST' && url.includes('gate')) {
        try {
          const gateApi = window.FinalPassGate || window.Gate;
          if (gateApi && typeof gateApi.getState === 'function') {
            const st = gateApi.getState();
            const pending = Array.isArray(st.pendingAmbers) ? st.pendingAmbers.slice() : [];
            const cleared = Array.isArray(st.clearedAmbers) ? st.clearedAmbers.slice() : [];
            if (pending.length > 0) {
              const item = pending.shift();
              cleared.push(item);
            }
            const sum = Object.assign({}, st.summary || {});
            sum.n_pending_amber = pending.length;
            sum.n_cleared_amber = (Number(sum.n_cleared_amber) || 0) + 1;
            const nextState = {
              scope: st.scope || 'test_5',
              summary: sum,
              pending_ambers: pending,
              cleared_ambers: cleared,
              audit_sample: st.auditSample || []
            };
            return new Response(JSON.stringify({ ok: true, state: nextState }), {
              status: 200,
              headers: { 'Content-Type': 'application/json' }
            });
          }
        } catch (_e) {}
        return new Response(JSON.stringify({ ok: true }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' }
        });
      }
      return window.__spOrigFetch(input, init);
    };
  }
  const installOverlay = () => {
    if (document.getElementById('sp-rec-overlay')) return;
    const wrap = document.createElement('div');
    wrap.id = 'sp-rec-overlay';
    wrap.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:2147483647;overflow:hidden;';

    const ring = document.createElement('div');
    ring.id = 'sp-rec-cursor';
    ring.style.cssText = [
      'position:fixed',
      'left:960px',
      'top:540px',
      'width:22px',
      'height:22px',
      'margin-left:-11px',
      'margin-top:-11px',
      'border-radius:50%',
      'border:2px solid rgba(59, 130, 246, 0.92)',
      'background:rgba(59, 130, 246, 0.16)',
      'transition:left 0.12s ease-out, top 0.12s ease-out, transform 0.12s ease-out',
      'pointer-events:none'
    ].join(';');

    const dot = document.createElement('div');
    dot.id = 'sp-rec-heartbeat';
    dot.style.cssText = [
      'position:fixed',
      'right:6px',
      'bottom:6px',
      'width:4px',
      'height:4px',
      'border-radius:50%',
      'background:rgba(16, 185, 129, 0.35)',
      'pointer-events:none'
    ].join(';');

    wrap.appendChild(ring);
    wrap.appendChild(dot);
    (document.body || document.documentElement).appendChild(wrap);

    window.addEventListener('mousemove', (e) => {
      ring.style.left = e.clientX + 'px';
      ring.style.top = e.clientY + 'px';
    }, { passive: true });
    window.addEventListener('mousedown', () => {
      ring.style.transform = 'scale(0.72)';
      ring.style.background = 'rgba(16, 185, 129, 0.45)';
    }, { passive: true });
    window.addEventListener('mouseup', () => {
      ring.style.transform = 'scale(1.0)';
      ring.style.background = 'rgba(56, 189, 248, 0.18)';
    }, { passive: true });

    let tick = 0;
    const loop = () => {
      tick = (tick + 1) % 60;
      dot.style.opacity = (0.25 + 0.5 * Math.sin(tick * 0.1047)).toFixed(3);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  };
  if (document.readyState === 'loading') {
    window.addEventListener('DOMContentLoaded', installOverlay);
  } else {
    installOverlay();
  }
  setInterval(installOverlay, 1000);
})();
"""


def is_url_up(url: str, timeout: float = 2.5) -> bool:
    """Return True if HTTP GET to url succeeds with status 200."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(getattr(resp, "status", 0)) == 200
    except Exception:
        return False


def probe_video(video_path: Path) -> dict[str, Any]:
    """Inspect a video file with ffprobe and return resolution, duration, codec, and size."""
    p = Path(video_path)
    if not p.is_file():
        raise FileNotFoundError(f"Video file not found: {p}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration,size",
        "-of",
        "json",
        str(p),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(proc.stdout or "{}")
    stream = (info.get("streams") or [{}])[0]
    fmt = info.get("format") or {}
    return {
        "path": str(p),
        "codec_name": stream.get("codec_name"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": round(float(fmt.get("duration") or 0.0), 2),
        "size_bytes": int(fmt.get("size") or p.stat().st_size),
    }


def transcode_webm_to_mp4(
    webm_path: Path,
    out_mp4: Path,
    min_duration: float = 62.0,
    max_duration: float = 88.0,
) -> dict[str, Any]:
    """Transcode Playwright WebM capture to 1920x1080 30fps H.264 MP4 within [min_duration, max_duration]."""
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    raw_meta = probe_video(webm_path)
    raw_dur = float(raw_meta.get("duration") or 0.0)

    vf_parts = [
        "scale=1920:1080:force_original_aspect_ratio=decrease",
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
    ]
    if 0.0 < raw_dur < min_duration:
        pad_sec = round(min_duration - raw_dur + 1.5, 2)
        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={pad_sec}")
    vf_parts.extend(["fps=30", "format=yuv420p"])

    tmp_mp4 = out_mp4.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(webm_path),
        "-vf",
        ",".join(vf_parts),
        "-t",
        f"{max_duration:.2f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-movflags",
        "+faststart",
        "-an",
        str(tmp_mp4),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.move(str(tmp_mp4), str(out_mp4))
    return probe_video(out_mp4)


def _safe_move_and_click(page: Any, selector: str, pause_ms: int = 900) -> bool:
    """Smoothly move mouse to first visible element matching selector and click it."""
    try:
        loc = page.locator(selector).first
        if loc.count() == 0 or not loc.is_visible():
            return False
        box = loc.bounding_box(timeout=600)
        if box:
            cx = box["x"] + box["width"] / 2.0
            cy = box["y"] + box["height"] / 2.0
            page.mouse.move(cx, cy, steps=12)
            page.wait_for_timeout(150)
            page.mouse.click(cx, cy)
        else:
            loc.click(timeout=700)
        if pause_ms > 0:
            page.wait_for_timeout(pause_ms)
        return True
    except Exception:
        return False


def _safe_hover(page: Any, selector: str, pause_ms: int = 700) -> bool:
    """Smoothly move mouse over first visible element matching selector."""
    try:
        loc = page.locator(selector).first
        if loc.count() == 0 or not loc.is_visible():
            return False
        box = loc.bounding_box(timeout=600)
        if not box:
            return False
        cx = box["x"] + box["width"] / 2.0
        cy = box["y"] + box["height"] / 2.0
        page.mouse.move(cx, cy, steps=10)
        if pause_ms > 0:
            page.wait_for_timeout(pause_ms)
        return True
    except Exception:
        return False


def _smooth_scroll(
    page: Any,
    delta_y: int,
    steps: int = 8,
    step_wait_ms: int = 160,
    container_selector: str | None = None,
) -> None:
    """Scroll smoothly across both window and inner scrollable containers (Streamlit stMain, drawers, etc.)."""
    per_step = int(delta_y / max(1, steps))
    for _ in range(max(1, steps)):
        try:
            page.mouse.wheel(0, per_step)
            page.evaluate(
                """([dy, sel]) => {
                    if (sel) {
                        const custom = document.querySelector(sel);
                        if (custom) { custom.scrollTop += dy; return; }
                    }
                    const candidates = [
                        document.querySelector('#final-pass-mount'),
                        document.querySelector('.drawer-backdrop.open .drawer-body'),
                        document.querySelector('#objects'),
                        document.querySelector('[data-testid="stMain"]'),
                        document.querySelector('[data-testid="stAppViewContainer"]'),
                        document.querySelector('section.main'),
                        document.scrollingElement,
                        document.documentElement
                    ];
                    for (const el of candidates) {
                        if (el && el.scrollHeight > el.clientHeight) {
                            el.scrollTop += dy;
                        }
                    }
                }""",
                [per_step, container_selector],
            )
        except Exception:
            pass
        page.wait_for_timeout(step_wait_ms)


def drive_web_app(page: Any, url: str, target_duration: float = 66.0) -> None:
    """Drive the SurgProof Web Annotation Studio at http://localhost:8600 for ~66s."""
    t0 = time.monotonic()

    # Safety net: intercept any POST request at the Playwright network layer so disk state is never altered
    def _block_mutating_post(route: Any, request: Any) -> None:
        if request.method.upper() == "POST":
            route.fulfill(status=200, content_type="application/json", body='{"ok": true}')
        else:
            route.continue_()

    try:
        page.route("**/api/gate*", _block_mutating_post)
        page.route("**/gate*", _block_mutating_post)
    except Exception:
        pass

    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(2500)

    # 1. Inspect Top Bar: SurgProof branding, video12 badge, held-out proof metric with n, Final Pass
    for sel in (
        ".brand-logo",
        "#video-badge",
        "#headline-acc",
        "#btn-final-pass",
    ):
        _safe_hover(page, sel, pause_ms=550)

    # 2. Press Play on the timeline for ~15s of live playback across surgical frames, then pause
    played = _safe_move_and_click(
        page,
        "#btn-play, button[data-tl-action='play']",
        pause_ms=15000,
    )
    if not played:
        for _ in range(15):
            page.keyboard.press("ArrowRight")
            page.wait_for_timeout(950)
    else:
        _safe_move_and_click(
            page,
            "#btn-play, button[data-tl-action='play']",
            pause_ms=1000,
        )

    # 3. Click 3 frames with masks on the multi-track timeline and inspect AI masks/polygons/boxes/tips
    structure_targets = [
        ("grasper", 8),
        ("gallbladder", 22),
        ("liver", 52),
    ]
    clicked_count = 0
    for label_key, f_idx in structure_targets:
        sel = f".tl-cell[data-label-key='{label_key}'][data-frame-idx='{f_idx}']"
        if _safe_move_and_click(page, sel, pause_ms=1800):
            clicked_count += 1
            _safe_hover(page, "#frame-canvas", pause_ms=900)

    if clicked_count < 3:
        try:
            cells = page.locator(".tl-cell.has-spatial, .tl-track .tl-cell")
            n_cells = cells.count()
            if n_cells > 3:
                for frac in (0.15, 0.45, 0.75):
                    idx = min(n_cells - 1, max(0, int(n_cells * frac)))
                    box = cells.nth(idx).bounding_box(timeout=600)
                    if box:
                        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=10)
                        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                        page.wait_for_timeout(1800)
                        _safe_hover(page, "#frame-canvas", pause_ms=800)
        except Exception:
            pass

    # 4. Toggle mask round r0/r1 if present (or toggle AI Masks / Pass-1/Pass-2 / Scoring GT overlay)
    toggled_round = False
    try:
        sel_round = page.locator("#sel-mask-round, select[data-mask-round]").first
        if sel_round.count() > 0 and sel_round.is_visible():
            _safe_hover(page, "#sel-mask-round, select[data-mask-round]", pause_ms=600)
            sel_round.select_option("1")
            page.wait_for_timeout(2400)
            _safe_hover(page, "#frame-canvas", pause_ms=900)
            sel_round.select_option("0")
            page.wait_for_timeout(1600)
            toggled_round = True
    except Exception:
        pass

    if not toggled_round:
        for r_sel in (
            "[data-mask-round='1'], button:has-text('R1'), button:has-text('r1')",
            "[data-mask-round='0'], button:has-text('R0'), button:has-text('r0')",
        ):
            if _safe_move_and_click(page, r_sel, pause_ms=1800):
                toggled_round = True

    if _safe_move_and_click(page, "label:has(#chk-dual-pass), #chk-dual-pass", pause_ms=2200):
        _safe_hover(page, "#frame-canvas", pause_ms=800)
        _safe_move_and_click(page, "label:has(#chk-dual-pass), #chk-dual-pass", pause_ms=900)

    if _safe_move_and_click(page, "label:has(#chk-show-gt), .gt-toggle, #chk-show-gt", pause_ms=2800):
        _safe_move_and_click(page, "label:has(#chk-show-gt), .gt-toggle, #chk-show-gt", pause_ms=900)

    # 5. Click through right-hand inspector cards (evidence text, 4-round Argon amber loop, manual rule)
    try:
        _safe_hover(page, "#amber-funnel-bar", pause_ms=800)
        cards = page.locator("#objects .label-card")
        n_cards = min(3, cards.count())
        for i in range(n_cards):
            box = cards.nth(i).bounding_box(timeout=500)
            if box:
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=10)
                page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.wait_for_timeout(1100)
    except Exception:
        pass

    # 6. Open Final pass and click Approve on one item (intercepted in-browser so disk data is never altered)
    opened_gate = _safe_move_and_click(
        page,
        "#btn-final-pass, [data-action='final-pass'], button:has-text('Final pass'), button:has-text('Final Pass')",
        pause_ms=2500,
    )
    if opened_gate:
        _safe_hover(page, "#gate-js-scope-select", pause_ms=800)
        # Click Approve on one item
        approved = _safe_move_and_click(
            page,
            "button[data-gate-act='approve-one'], #gate-amber-list button:has-text('Approve')",
            pause_ms=2400,
        )
        if not approved:
            _safe_hover(page, "button[data-gate-act='batch-approve']", pause_ms=1200)
        _smooth_scroll(page, 380, steps=5, step_wait_ms=180, container_selector="#final-pass-mount")
        _safe_hover(page, "#gate-js-sign-off-audit", pause_ms=1000)
        _safe_hover(page, "#gate-js-release-btn, #btn-release-dataset, #gate-lock-msg", pause_ms=1500)
        _smooth_scroll(page, -380, steps=5, step_wait_ms=160, container_selector="#final-pass-mount")
        page.wait_for_timeout(1000)
        _safe_move_and_click(page, "#btn-close-drawer, button:has-text('Close')", pause_ms=1400)

    # 7. Hold and gently scrub timeline until target_duration is reached
    step_idx = 0
    while (time.monotonic() - t0) < target_duration:
        remaining = target_duration - (time.monotonic() - t0)
        if remaining <= 1.5:
            page.wait_for_timeout(int(max(100, remaining * 1000)))
            break
        if step_idx % 2 == 0:
            if not _safe_move_and_click(page, "#btn-next, button[data-tl-action='next']", pause_ms=1500):
                page.keyboard.press("ArrowRight")
                page.wait_for_timeout(1500)
        else:
            _safe_hover(page, "#frame-canvas, canvas", pause_ms=1500)
        step_idx += 1


def _find_watch_asset(filename: str) -> Path | None:
    """Locate watch.html / watch.js / style.css across main web/ or sibling webshell lane."""
    candidates = [
        ROOT / "web" / filename,
        ROOT.parent / "surgatlas" / "web" / filename,
        ROOT.parent / "webshell" / "web" / filename,
        ROOT.parent / "surgatlas-lanes" / "webshell" / "web" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def drive_watch_app(page: Any, url: str, target_duration: float = 30.0) -> None:
    """Drive watch.html (Netflix-style Watch mode with narration subtitles) for ~30s."""
    t0 = time.monotonic()

    # If watch.html is in the webshell worktree and not yet merged into main web/, serve its static shell
    # on the same http://localhost:8600 origin so all /data/ and /out/ requests resolve against :8600.
    if not is_url_up(url):
        def _serve_watch_asset(route: Any, request: Any) -> None:
            req_url = request.url.split("?", 1)[0]
            for fname, ctype in (
                ("watch.html", "text/html; charset=utf-8"),
                ("watch.js", "application/javascript; charset=utf-8"),
                ("style.css", "text/css; charset=utf-8"),
            ):
                if req_url.endswith("/" + fname):
                    asset = _find_watch_asset(fname)
                    if asset is not None:
                        route.fulfill(status=200, content_type=ctype, body=asset.read_bytes())
                        return
            route.continue_()

        page.route("**/*", _serve_watch_asset)

    page.goto(url, wait_until="domcontentloaded", timeout=20000)
    page.wait_for_timeout(2000)

    # watch.js auto-starts playback on load; only click Play if currently paused
    try:
        play_icon = page.locator("#watch-icon-play")
        if play_icon.count() > 0 and play_icon.is_visible():
            _safe_move_and_click(page, "#watch-play-btn", pause_ms=1000)
    except Exception:
        pass

    # Let watch mode play through video12 frames with mask overlays and verified narration subtitles
    step = 0
    while (time.monotonic() - t0) < target_duration:
        remaining = target_duration - (time.monotonic() - t0)
        if remaining <= 1.5:
            page.wait_for_timeout(int(max(100, remaining * 1000)))
            break
        if step % 3 == 0:
            _safe_hover(page, "#watch-subtitles, #watch-stage", pause_ms=2000)
        elif step % 3 == 1:
            _safe_hover(page, "#watch-headline-chip, #watch-header", pause_ms=2000)
        else:
            _safe_hover(page, "#watch-controls, #watch-time", pause_ms=2000)
        step += 1


def drive_streamlit_app(page: Any, url: str, target_duration: float = 68.0) -> None:
    """Drive the SurgProof Streamlit Fallback at http://localhost:8501 for ~68s."""
    t0 = time.monotonic()
    base_url = url.rstrip("/")
    page.goto(base_url, wait_until="domcontentloaded", timeout=25000)
    try:
        page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=12000)
    except Exception:
        pass
    page.wait_for_timeout(3200)

    # Beat 1 (0-18s): Main End-to-End Pipeline View (7 stages, Manual, Labelled vs Double-Checked,
    # 4-Round Argon Amber Loop progression, Specialist Student, Held-Out Proof, Spatial Annotations)
    page.mouse.move(960, 340, steps=12)
    page.wait_for_timeout(2200)
    _smooth_scroll(page, 650, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2200)
    _smooth_scroll(page, 780, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2400)
    _smooth_scroll(page, 780, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2200)

    # Beat 2 (18-46s): AI Video Annotation Studio (pages/1_Annotate.py -> /Annotate)
    # Play through video12 frames, inspect AI masks/boxes/tips, toggle GT mask overlay, show multi-track timeline
    clicked_ann = _safe_move_and_click(
        page,
        "[data-testid='stSidebarNav'] a:has-text('Annotate'), a:has-text('Open AI Video Annotation Studio')",
        pause_ms=3500,
    )
    if not clicked_ann:
        page.goto(f"{base_url}/Annotate", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3500)

    page.mouse.move(820, 420, steps=10)
    page.wait_for_timeout(1600)

    # Step / Play through several localized frames on the timeline
    for _ in range(3):
        if not _safe_move_and_click(page, "button:has-text('▶ Play / Step'), button:has-text('▶ Next')", pause_ms=2200):
            break

    # Click Run / Stream Agent Annotations
    _safe_move_and_click(page, "button:has-text('Run / Stream Agent Annotations')", pause_ms=2500)

    # Scroll down to Frame Canvas Viewer + Object Inspector + Multi-Track Timeline
    _smooth_scroll(page, 460, steps=6, step_wait_ms=200)
    page.wait_for_timeout(2000)

    # Toggle Pass-1 / Pass-2 Boxes and Scoring View CholecSeg8k Ground-Truth Mask
    _safe_move_and_click(page, "label:has-text('Show Pass-1 / Pass-2 Boxes')", pause_ms=2600)
    _safe_move_and_click(
        page,
        "label:has-text('Scoring View Only: Overlay CholecSeg8k Ground-Truth Mask')",
        pause_ms=3200,
    )

    # Scroll down to show full Object / Label Inspector & Multi-Track Timeline across all video12 frames
    _smooth_scroll(page, 760, steps=8, step_wait_ms=220)
    page.wait_for_timeout(3000)
    _smooth_scroll(page, 620, steps=6, step_wait_ms=200)
    page.wait_for_timeout(2600)

    # Beat 3 (46-68s): Open Final Pass / Human Gate (/Human_gate) and show residual AMBER queue + 20-item audit
    clicked_gate = _safe_move_and_click(
        page,
        "[data-testid='stSidebarNav'] a:has-text('Human gate'), a:has-text('Final Pass')",
        pause_ms=3500,
    )
    if not clicked_gate:
        page.goto(f"{base_url}/Human_gate", wait_until="domcontentloaded", timeout=20000)
        page.wait_for_timeout(3500)

    page.mouse.move(960, 360, steps=10)
    page.wait_for_timeout(2200)
    _smooth_scroll(page, 720, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2500)
    _smooth_scroll(page, 820, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2500)
    _smooth_scroll(page, 820, steps=8, step_wait_ms=220)
    page.wait_for_timeout(2200)

    # Gentle scroll hold until target_duration is reached
    direction = -1
    while (time.monotonic() - t0) < target_duration:
        remaining = target_duration - (time.monotonic() - t0)
        if remaining <= 1.8:
            page.wait_for_timeout(int(max(100, remaining * 1000)))
            break
        _smooth_scroll(page, direction * 450, steps=6, step_wait_ms=220)
        page.wait_for_timeout(1200)
        direction *= -1


def record_target(
    target: str,
    url: str,
    out_mp4: Path,
    duration: float = 68.0,
) -> dict[str, Any]:
    """Record 'web', 'watch', or 'streamlit' target into out_mp4 (1920x1080) and return ffprobe metadata."""
    from playwright.sync_api import sync_playwright

    if target == "watch":
        base_origin = url.rsplit("/", 1)[0] if "/" in url[8:] else url
        if not is_url_up(url) and not (is_url_up(base_origin) and _find_watch_asset("watch.html") is not None):
            raise RuntimeError(f"Target server for {target!r} is not reachable at {url}")
    elif not is_url_up(url):
        raise RuntimeError(f"Target server for {target!r} is not reachable at {url}")

    with tempfile.TemporaryDirectory(prefix=f"surgproof_rec_{target}_") as tmp_dir:
        webm_path: Path | None = None
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1920, "height": 1080},
                record_video_dir=tmp_dir,
                record_video_size={"width": 1920, "height": 1080},
                device_scale_factor=1,
            )
            context.add_init_script(CURSOR_INIT_SCRIPT)
            page = context.new_page()
            try:
                if target == "web":
                    drive_web_app(page, url, target_duration=duration)
                elif target == "watch":
                    drive_watch_app(page, url, target_duration=duration)
                elif target == "streamlit":
                    drive_streamlit_app(page, url, target_duration=duration)
                else:
                    raise ValueError(f"Unknown target: {target}")
            finally:
                if page.video:
                    webm_path = Path(page.video.path())
                context.close()
                browser.close()

        if webm_path is None or not webm_path.is_file():
            candidates = list(Path(tmp_dir).glob("*.webm"))
            if not candidates:
                raise RuntimeError(f"Playwright did not produce a .webm recording in {tmp_dir}")
            webm_path = candidates[0]

        if target == "watch":
            min_dur = max(28.0, duration - 2.0)
            max_dur = min(38.0, max(duration + 4.0, 32.0))
        else:
            min_dur = max(60.5, duration - 4.0)
            max_dur = min(89.0, max(duration + 12.0, 75.0))

        meta = transcode_webm_to_mp4(
            webm_path=webm_path,
            out_mp4=Path(out_mp4),
            min_duration=min_dur,
            max_duration=max_dur,
        )
        if target == "web":
            for alias_path in (DEFAULT_WEB_OUT, DEFAULT_WEB_ALIAS_OUT):
                if Path(out_mp4).resolve() != alias_path.resolve():
                    shutil.copy2(str(out_mp4), str(alias_path))
        return meta


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record SurgProof demo videos (1920x1080, 60-90s).")
    parser.add_argument(
        "--target",
        choices=("streamlit", "web", "watch", "auto", "both"),
        default="auto",
        help="Which UI to record: streamlit (:8501), web (:8600), watch (:8600/watch.html), both, or auto.",
    )
    parser.add_argument("--web-url", default=DEFAULT_WEB_URL, help="URL for web app (default http://localhost:8600)")
    parser.add_argument(
        "--watch-url",
        default=DEFAULT_WATCH_URL,
        help="URL for watch mode (default http://localhost:8600/watch.html)",
    )
    parser.add_argument(
        "--streamlit-url",
        default=DEFAULT_STREAMLIT_URL,
        help="URL for Streamlit app (default http://localhost:8501)",
    )
    parser.add_argument("--web-out", type=Path, default=DEFAULT_WEB_OUT, help="Output MP4 path for web app")
    parser.add_argument("--watch-out", type=Path, default=DEFAULT_WATCH_OUT, help="Output MP4 path for watch mode")
    parser.add_argument(
        "--streamlit-out",
        type=Path,
        default=DEFAULT_STREAMLIT_OUT,
        help="Output MP4 path for Streamlit fallback",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=68.0,
        help="Target recording duration in seconds (60-90s, default 68.0).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results: dict[str, Any] = {}

    targets: list[tuple[str, str, Path, float]] = []
    if args.target in ("streamlit", "both"):
        targets.append(("streamlit", args.streamlit_url, args.streamlit_out, args.duration))
    if args.target in ("web", "both"):
        targets.append(("web", args.web_url, args.web_out, args.duration))
    if args.target == "watch":
        watch_dur = 30.0 if args.duration == 68.0 else args.duration
        targets.append(("watch", args.watch_url, args.watch_out, watch_dur))
    if args.target == "auto":
        if is_url_up(args.streamlit_url):
            targets.append(("streamlit", args.streamlit_url, args.streamlit_out, args.duration))
        if is_url_up(args.web_url):
            targets.append(("web", args.web_url, args.web_out, args.duration))
        if is_url_up(args.watch_url):
            targets.append(("watch", args.watch_url, args.watch_out, 30.0))

    if not targets:
        print(
            f"No live targets reachable (checked {args.streamlit_url} and {args.web_url}).",
            file=sys.stderr,
        )
        return 1

    exit_code = 0
    for target_name, url, out_path, dur in targets:
        try:
            print(f"[record_demo] Recording {target_name} from {url} -> {out_path} (target={dur:.1f}s)...")
            meta = record_target(target=target_name, url=url, out_mp4=out_path, duration=dur)
            results[target_name] = meta
            print(f"[record_demo] Completed {target_name}: {json.dumps(meta)}")
        except Exception as exc:
            print(f"[record_demo] ERROR recording {target_name}: {exc}", file=sys.stderr)
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
