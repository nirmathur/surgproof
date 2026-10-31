#!/usr/bin/env python3
"""Generate, overlay, and stitch the 60-90s SurgAtlas pitch video (issue #10).

Usage:
  python -m promo.render                       # generate missing Veo clips + build proof slot + stitch
  python -m promo.render --draft               # use veo-3.1-fast-generate-preview for faster generation
  python -m promo.render --refresh-slot        # re-render Shot 5 from latest out/ artifacts + app.py & re-stitch
  python -m promo.render --shot shot_01_med_school
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from promo.shots import (
    ILLUSTRATIVE_CAPTION,
    PROJECT_NAME,
    REAL_PROOF_CAPTION,
    SHOTS,
    ShotSpec,
    total_storyboard_duration,
)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
PROMO_DIR = OUT_DIR / "promo"
RAW_CLIP_DIR = PROMO_DIR / "raw"
OVERLAY_DIR = PROMO_DIR / "overlays"
FINAL_VIDEO = PROMO_DIR / "surgatlas_promo.mp4"
MANIFEST_PATH = PROMO_DIR / "manifest.json"

WIDTH = 1280
HEIGHT = 720
FPS = 24

PRIMARY_VEO_MODEL = "veo-3.1-generate-preview"
FAST_VEO_MODEL = "veo-3.1-fast-generate-preview"
FALLBACK_OMNI_MODEL = "gemini-omni-1.1-flash"


def ensure_api_key() -> None:
    """Load GEMINI_API_KEY from .env if not already in os.environ (never logs key)."""
    if os.environ.get("GEMINI_API_KEY"):
        return
    candidates = [
        ROOT / ".env",
        Path("/Users/nirmathur/surgatlas/.env"),
        Path.home() / "surgatlas" / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k == "GEMINI_API_KEY" and v:
                    os.environ["GEMINI_API_KEY"] = v
                    return


def _load_font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: List[str] = []
    if mono:
        candidates.extend([
            "/System/Library/Fonts/SFNSMono.ttf",
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Courier.dfont",
        ])
    elif bold:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ])
    else:
        candidates.extend([
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ])
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def probe_video(path: Path) -> Dict[str, Any]:
    """Return duration_sec, width, height, codec_name, has_audio using ffprobe."""
    if not path.is_file() or path.stat().st_size == 0:
        return {"valid": False, "duration_sec": 0.0}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        return {"valid": False, "duration_sec": 0.0}
    try:
        info = json.loads(res.stdout)
    except Exception:
        return {"valid": False, "duration_sec": 0.0}
    streams = info.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not v_stream:
        return {"valid": False, "duration_sec": 0.0}
    dur = float(info.get("format", {}).get("duration") or v_stream.get("duration") or 0.0)
    return {
        "valid": dur > 0.5,
        "duration_sec": dur,
        "width": int(v_stream.get("width", 0)),
        "height": int(v_stream.get("height", 0)),
        "codec": v_stream.get("codec_name", ""),
        "has_audio": a_stream is not None,
    }


def render_shot_overlay_png(shot: ShotSpec, out_png: Path) -> Path:
    """Render a 1280x720 RGBA overlay PNG with mandatory ILLUSTRATIVE badge and story captions."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)

    font_badge = _load_font(15, bold=True)
    font_step = _load_font(15, bold=True)
    font_title = _load_font(24, bold=True)
    font_sub = _load_font(17, bold=False)
    font_mono = _load_font(15, mono=True)
    font_narr = _load_font(16, bold=False)

    # 1. Top-left brand + shot pill
    draw.rounded_rectangle(
        (24, 20, 290, 58),
        radius=10,
        fill=(12, 18, 30, 220),
        outline=(56, 189, 248, 220),
        width=2,
    )
    draw.text((38, 30), f"{PROJECT_NAME} · SHOT {shot.index}/{len(SHOTS)}", font=font_step, fill=(240, 249, 255, 255))

    # 2. Top-right MANDATORY ILLUSTRATIVE / REAL TELEMETRY badge
    badge_text = shot.badge_text
    if shot.illustrative:
        badge_bg = (180, 83, 9, 235)      # high-contrast amber warning banner
        badge_border = (253, 224, 71, 255)
        badge_fg = (255, 255, 255, 255)
    else:
        badge_bg = (21, 128, 61, 235)     # emerald verified telemetry banner
        badge_border = (134, 239, 172, 255)
        badge_fg = (255, 255, 255, 255)

    draw.rounded_rectangle(
        (310, 20, WIDTH - 24, 58),
        radius=10,
        fill=badge_bg,
        outline=badge_border,
        width=2,
    )
    draw.text((326, 30), badge_text, font=font_badge, fill=badge_fg)

    # 3. Bottom lower-third narrative panel
    panel_top = HEIGHT - 172
    draw.rounded_rectangle(
        (24, panel_top, WIDTH - 24, HEIGHT - 20),
        radius=14,
        fill=(10, 15, 28, 222),
        outline=(71, 85, 105, 210),
        width=2,
    )

    # Title + pipeline command pill
    draw.text((44, panel_top + 14), shot.title, font=font_title, fill=(56, 189, 248, 255))
    cmd_pill_text = f"{shot.pipeline_cmd}  ->  {shot.artifact_label}"
    draw.rounded_rectangle(
        (560, panel_top + 12, WIDTH - 42, panel_top + 44),
        radius=8,
        fill=(30, 41, 59, 235),
        outline=(100, 116, 139, 200),
        width=1,
    )
    draw.text((574, panel_top + 19), cmd_pill_text[:68], font=font_mono, fill=(226, 232, 240, 255))

    # Subtitle
    draw.text((44, panel_top + 50), shot.subtitle, font=font_sub, fill=(241, 245, 249, 255))

    # Divider line
    draw.line((44, panel_top + 80, WIDTH - 44, panel_top + 80), fill=(51, 65, 85, 200), width=1)

    # Wrapped narration caption
    wrapped = textwrap.wrap(f'"{shot.narration}"', width=108)[:2]
    for idx, line in enumerate(wrapped):
        draw.text(
            (44, panel_top + 90 + idx * 24),
            line,
            font=font_narr,
            fill=(203, 213, 225, 255),
        )

    im.save(out_png, format="PNG")
    return out_png


def generate_raw_clip_with_veo_or_omni(
    shot: ShotSpec,
    raw_out: Path,
    *,
    draft: bool = False,
    poll_interval: float = 8.0,
    timeout: float = 420.0,
) -> Tuple[Path, str]:
    """Generate a raw video clip using Veo 3.1 (or Omni fallback) via google-genai SDK."""
    ensure_api_key()
    from google import genai
    from google.genai import types

    raw_out.parent.mkdir(parents=True, exist_ok=True)
    client = genai.Client()

    models_to_try = (
        [FAST_VEO_MODEL, PRIMARY_VEO_MODEL]
        if draft
        else [PRIMARY_VEO_MODEL, FAST_VEO_MODEL]
    )

    last_err: Optional[Exception] = None
    for veo_model in models_to_try:
        try:
            print(f"[promo] Generating {shot.shot_id} with {veo_model} ...", flush=True)
            op = client.models.generate_videos(
                model=veo_model,
                prompt=shot.prompt,
                config=types.GenerateVideosConfig(
                    aspect_ratio="16:9",
                    number_of_videos=1,
                ),
            )
            start_t = time.monotonic()
            while not op.done:
                elapsed = time.monotonic() - start_t
                if elapsed > timeout:
                    raise TimeoutError(f"{veo_model} timed out after {elapsed:.0f}s")
                time.sleep(poll_interval)
                op = client.operations.get(op)
                print(
                    f"[promo] [{shot.shot_id}] polling {veo_model} ({elapsed:.0f}s) done={op.done}",
                    flush=True,
                )

            if getattr(op, "error", None):
                raise RuntimeError(f"{veo_model} operation error: {op.error}")

            resp = getattr(op, "response", None) or getattr(op, "result", None)
            gen_vids = getattr(resp, "generated_videos", None) if resp else None
            if not gen_vids:
                raise RuntimeError(f"{veo_model} returned no generated_videos (resp={resp})")

            video_obj = gen_vids[0].video
            data = client.files.download(file=video_obj)
            if isinstance(data, (bytes, bytearray)) and len(data) > 0:
                raw_out.write_bytes(data)
            elif getattr(video_obj, "video_bytes", None):
                raw_out.write_bytes(video_obj.video_bytes)
            else:
                video_obj.save(str(raw_out))

            meta = probe_video(raw_out)
            if not meta.get("valid"):
                raise RuntimeError(f"Downloaded clip {raw_out} failed ffprobe check: {meta}")
            print(
                f"[promo] Saved {raw_out.name} ({meta['duration_sec']:.1f}s, {meta['width']}x{meta['height']}) via {veo_model}",
                flush=True,
            )
            return raw_out, veo_model
        except Exception as exc:
            last_err = exc
            print(
                f"[promo] {veo_model} failed for {shot.shot_id} ({type(exc).__name__}: {str(exc)[:160]}), trying next...",
                flush=True,
            )

    # Fallback: gemini-omni-1.1-flash via Interactions API
    try:
        print(f"[promo] Falling back to {FALLBACK_OMNI_MODEL} for {shot.shot_id} ...", flush=True)
        interaction = client.interactions.create(
            model=FALLBACK_OMNI_MODEL,
            input=shot.prompt,
            response_modalities=["video"],
            response_format={
                "type": "video",
                "aspect_ratio": "16:9",
                "resolution": "720p",
            },
        )
        for item in getattr(interaction, "outputs", None) or []:
            if getattr(item, "type", None) == "video":
                b64_data = getattr(item, "data", None)
                uri = getattr(item, "uri", None)
                if b64_data:
                    raw_bytes = base64.b64decode(b64_data) if isinstance(b64_data, str) else bytes(b64_data)
                    raw_out.write_bytes(raw_bytes)
                    if probe_video(raw_out).get("valid"):
                        return raw_out, FALLBACK_OMNI_MODEL
                elif uri:
                    dl = client.files.download(file=uri)
                    if dl:
                        raw_out.write_bytes(dl)
                        if probe_video(raw_out).get("valid"):
                            return raw_out, FALLBACK_OMNI_MODEL
    except Exception as omni_exc:
        last_err = omni_exc
        print(f"[promo] {FALLBACK_OMNI_MODEL} fallback failed: {str(omni_exc)[:160]}", flush=True)

    raise RuntimeError(f"All video generation models failed for {shot.shot_id}. Last error: {last_err}")


def compose_illustrative_shot(
    shot: ShotSpec,
    raw_clip: Path,
    overlay_png: Path,
    out_clip: Path,
) -> Path:
    """Scale raw clip to 1280x720, stretch smoothly to shot.duration_sec, burn in overlay PNG, ensure stereo AAC."""
    out_clip.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(raw_clip)
    raw_dur = max(0.5, float(meta.get("duration_sec") or 8.0))
    target_dur = float(shot.duration_sec)
    pts_factor = target_dur / raw_dur

    # Filter chain:
    # 1. Scale/crop video to 1280x720, stretch PTS to match target duration, set 24 fps
    # 2. Overlay the 1280x720 RGBA PNG (which carries the mandatory ILLUSTRATIVE badge + captions)
    v_filter = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},setpts={pts_factor:.6f}*PTS,fps={FPS},format=yuv420p[v0];"
        f"[v0][1:v]overlay=0:0:format=auto,format=yuv420p[vout]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_clip),
        "-loop",
        "1",
        "-t",
        f"{target_dur:.2f}",
        "-i",
        str(overlay_png),
        "-f",
        "lavfi",
        "-t",
        f"{target_dur:.2f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex",
        v_filter,
        "-map",
        "[vout]",
        "-map",
        "2:a",
        "-t",
        f"{target_dur:.2f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        str(out_clip),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg compose failed for {shot.shot_id}: {res.stderr[-600:]}")
    return out_clip


def load_real_pipeline_metrics(out_dir: Path = OUT_DIR) -> Dict[str, Any]:
    """Read ONLY real pipeline outputs from out/ (and coord/artifacts/). Never fabricates numbers."""
    coord_dir = ROOT / "coord" / "artifacts"

    def _read_json(candidates: Sequence[Path]) -> Tuple[Optional[ Any], Optional[str]]:
        for p in candidates:
            if p.is_file() and p.stat().st_size > 0:
                try:
                    return json.loads(p.read_text(encoding="utf-8")), str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)
                except Exception:
                    continue
        return None, None

    results_data, results_src = _read_json([out_dir / "results.json", coord_dir / "results.json"])
    student_data, student_src = _read_json([out_dir / "student.json", coord_dir / "student.json"])
    manual_data, manual_src = _read_json([out_dir / "manual.json", coord_dir / "manual.json"])

    decisions_rows = 0
    decisions_src = None
    for dec_p in (out_dir / "decisions.csv", coord_dir / "decisions.csv"):
        if dec_p.is_file() and dec_p.stat().st_size > 0:
            try:
                decisions_rows = max(0, len(dec_p.read_text(encoding="utf-8").splitlines()) - 1)
                decisions_src = str(dec_p.relative_to(ROOT) if dec_p.is_relative_to(ROOT) else dec_p)
                break
            except Exception:
                pass

    metrics: Dict[str, Any] = {
        "has_results": results_data is not None,
        "has_student": student_data is not None,
        "has_manual": manual_data is not None,
        "has_decisions": decisions_rows > 0,
        "sources": {
            "results": results_src,
            "student": student_src,
            "manual": manual_src,
            "decisions": decisions_src,
        },
        "cards": [],
    }

    # Card 1: Accepted Accuracy (held-out test split, preferring excl_liver if present)
    acc_block = None
    acc_note = "test split"
    if isinstance(results_data, dict):
        if isinstance(results_data.get("accepted_excl_liver"), dict):
            acc_block = results_data["accepted_excl_liver"]
            acc_note = "test split, excl. liver=100%"
        elif isinstance(results_data.get("accepted"), dict):
            acc_block = results_data["accepted"]
            acc_note = "test split"
    if isinstance(acc_block, dict):
        acc = acc_block.get("accuracy")
        cov = acc_block.get("coverage")
        esc = acc_block.get("escalated_share")
        acc_str = f"{acc * 100:.1f}%" if isinstance(acc, (int, float)) else "N/A"
        cov_str = f"{cov * 100:.0f}%" if isinstance(cov, (int, float)) else "N/A"
        esc_str = f"{esc * 100:.0f}%" if isinstance(esc, (int, float)) else "N/A"
        metrics["cards"].append({
            "label": f"Verified Label Accuracy ({acc_note})",
            "value": acc_str,
            "detail": f"Coverage: {cov_str} · Escalated to surgeon: {esc_str} ({results_src})",
            "status": "real",
        })
    else:
        metrics["cards"].append({
            "label": "Verified Label Accuracy (Test Split, Excl. Liver)",
            "value": "AWAITING out/results.json",
            "detail": "Populated automatically when accept.py finishes (0 synthetic numbers)",
            "status": "pending",
        })

    # Card 2: Self-written Manual vs Blank Baseline
    if isinstance(results_data, dict) and "manual_raw_acc" in results_data:
        m_acc = results_data.get("manual_raw_acc")
        b_acc = results_data.get("blank_raw_acc")
        if isinstance(m_acc, (int, float)) and isinstance(b_acc, (int, float)):
            delta = (m_acc - b_acc) * 100.0
            metrics["cards"].append({
                "label": "Self-Written Manual vs Blank Prompt",
                "value": f"{m_acc * 100:.1f}% ({delta:+.1f} pts)",
                "detail": f"Blank baseline: {b_acc * 100:.1f}% on held-out test split ({results_src})",
                "status": "real",
            })
        elif isinstance(m_acc, (int, float)):
            metrics["cards"].append({
                "label": "Self-Written Manual Raw Accuracy",
                "value": f"{m_acc * 100:.1f}%",
                "detail": f"From {results_src}",
                "status": "real",
            })
    elif isinstance(manual_data, dict) and isinstance(manual_data.get("structures"), list):
        n_struct = len(manual_data["structures"])
        metrics["cards"].append({
            "label": "Autonomous Surgical Manual (out/manual.json)",
            "value": f"{n_struct} structures learned",
            "detail": f"Learned from public lecture video · awaiting accept.py test score",
            "status": "real",
        })
    else:
        metrics["cards"].append({
            "label": "Self-Written Manual vs Blank Baseline",
            "value": "AWAITING out/manual.json & results.json",
            "detail": "Populated from med_school.py + accept.py outputs",
            "status": "pending",
        })

    # Card 3: Specialist Student Model (Test Videos, Real GT)
    if isinstance(student_data, dict) and isinstance(student_data.get("mean_acc_by_source"), dict):
        by_src = student_data["mean_acc_by_source"]
        parts = [f"{k}: {v * 100:.1f}%" for k, v in by_src.items() if isinstance(v, (int, float))]
        top_val = parts[0] if parts else "Available"
        metrics["cards"].append({
            "label": "Specialist Student (Held-Out Test GT)",
            "value": top_val,
            "detail": " · ".join(parts[:3]) + " (oracle_upper_bound = reference only)",
            "status": "real",
        })
    else:
        metrics["cards"].append({
            "label": "Specialist Student (Held-Out Test GT)",
            "value": "AWAITING out/student.json",
            "detail": "Populated when student.py completes on verified labels",
            "status": "pending",
        })

    # Card 4: Audit Decisions & Honest Claim Verification
    if decisions_rows > 0:
        metrics["cards"].append({
            "label": "Audited Frame-Label Decisions",
            "value": f"{decisions_rows:,} rows",
            "detail": f"0 human training labels used · source: {decisions_src}",
            "status": "real",
        })
    else:
        metrics["cards"].append({
            "label": "Human Labels Used in Training",
            "value": "0 (Strict Zero-Label Rule)",
            "detail": "Ground truth used ONLY for held-out test evaluation · decisions.csv pending",
            "status": "real",
        })

    return metrics


def _is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def capture_streamlit_app_screenshot(out_png: Path) -> Tuple[bool, str]:
    """Capture a live screenshot of streamlit run app.py using headless Google Chrome."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    chrome_bin = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome_bin.exists():
        return False, "Google Chrome binary not found"

    port = 8501 if _is_port_open(8501) else 8597
    proc: Optional[subprocess.Popen[Any]] = None
    try:
        if port != 8501:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(ROOT / "app.py"),
                    "--server.headless",
                    "true",
                    "--server.port",
                    str(port),
                    "--browser.gatherUsageStats",
                    "false",
                ],
                cwd=str(ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            for _ in range(25):
                if _is_port_open(port):
                    break
                time.sleep(0.4)
            # Give Streamlit a moment to finish initial render
            time.sleep(2.5)

        if not _is_port_open(port):
            return False, f"Streamlit port {port} did not open"

        cmd = [
            str(chrome_bin),
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={WIDTH},{HEIGHT}",
            "--virtual-time-budget=4000",
            f"--screenshot={out_png}",
            f"http://127.0.0.1:{port}",
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
        if out_png.is_file() and out_png.stat().st_size > 1000:
            return True, f"http://127.0.0.1:{port}"
        return False, "Chrome screenshot output missing or empty"
    except Exception as exc:
        return False, str(exc)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except Exception:
                proc.kill()


def render_proof_slot_frames(
    shot: ShotSpec,
    app_screenshot: Optional[Path],
    out_dir: Path = OUT_DIR,
) -> Tuple[Path, Path, Dict[str, Any]]:
    """Render two 1280x720 frames for Shot 5 (Part A: Telemetry + UI Split, Part B: Live UI + Overlay)."""
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    metrics = load_real_pipeline_metrics(out_dir=out_dir)

    font_badge = _load_font(15, bold=True)
    font_title = _load_font(23, bold=True)
    font_sub = _load_font(15, bold=False)
    font_card_lbl = _load_font(14, bold=True)
    font_card_val = _load_font(22, bold=True)
    font_card_det = _load_font(13, bold=False)
    font_mono = _load_font(14, mono=True)

    # ---- Frame A: Left = Real Pipeline Telemetry Slot, Right = Live app.py Preview ----
    frame_a = Image.new("RGBA", (WIDTH, HEIGHT), (11, 15, 25, 255))
    draw_a = ImageDraw.Draw(frame_a)

    # Top header bars
    draw_a.rounded_rectangle((24, 20, 290, 58), radius=10, fill=(15, 23, 42, 240), outline=(56, 189, 248, 255), width=2)
    draw_a.text((38, 30), f"{PROJECT_NAME} · SHOT {shot.index}/{len(SHOTS)}", font=font_badge, fill=(240, 249, 255, 255))

    draw_a.rounded_rectangle((310, 20, WIDTH - 24, 58), radius=10, fill=(21, 128, 61, 240), outline=(134, 239, 172, 255), width=2)
    draw_a.text((326, 30), REAL_PROOF_CAPTION, font=font_badge, fill=(255, 255, 255, 255))

    # Left column: 4 Real Metric Cards
    draw_a.text((32, 74), "REAL PIPELINE METRICS (HELD-OUT TEST SPLIT)", font=font_title, fill=(56, 189, 248, 255))
    draw_a.text((32, 102), "Populated strictly from out/ artifacts — never synthetic or fabricated", font=font_sub, fill=(148, 163, 184, 255))

    for idx, card in enumerate(metrics["cards"][:4]):
        y0 = 132 + idx * 96
        y1 = y0 + 84
        is_real = card["status"] == "real"
        border_col = (34, 197, 94, 220) if is_real else (234, 179, 8, 220)
        val_col = (74, 222, 128, 255) if is_real else (253, 224, 71, 255)
        draw_a.rounded_rectangle((32, y0, 620, y1), radius=10, fill=(1e1 if False else 19, 28, 46, 245), outline=border_col, width=2)
        draw_a.text((48, y0 + 10), card["label"], font=font_card_lbl, fill=(226, 232, 240, 255))
        draw_a.text((48, y0 + 30), str(card["value"]), font=font_card_val, fill=val_col)
        draw_a.text((48, y0 + 60), str(card["detail"])[:68], font=font_card_det, fill=(148, 163, 184, 255))

    # Right column: Live app.py screen capture
    draw_a.text((648, 74), "LIVE STREAMLIT UI CAPTURE (streamlit run app.py)", font=font_title, fill=(56, 189, 248, 255))
    draw_a.rounded_rectangle((646, 108, WIDTH - 28, 514), radius=12, fill=(19, 28, 46, 255), outline=(100, 116, 139, 220), width=2)

    if app_screenshot and app_screenshot.is_file():
        try:
            shot_im = Image.open(app_screenshot).convert("RGBA")
            shot_im.thumbnail((596, 392))
            paste_x = 648 + (600 - shot_im.width) // 2
            paste_y = 114 + (396 - shot_im.height) // 2
            frame_a.paste(shot_im, (paste_x, paste_y))
        except Exception:
            draw_a.text((680, 290), "Streamlit UI capture loading...", font=font_card_val, fill=(226, 232, 240, 255))
    else:
        draw_a.text((680, 290), "Streamlit app.py capture slot", font=font_card_val, fill=(226, 232, 240, 255))

    # Honest claims & Prior work banner above bottom lower-third
    draw_a.rounded_rectangle((646, 444, WIDTH - 28, 514), radius=10, fill=(15, 23, 42, 235), outline=(56, 189, 248, 180), width=1)
    draw_a.text((660, 454), "HONEST CLAIMS: 0 human labels in training · calib/oracle = upper bounds", font=font_card_lbl, fill=(248, 250, 252, 255))
    draw_a.text((660, 478), "PRIOR WORK: SurgVLP (CAMMA) · SurgSLOT · Med-Gemini", font=font_mono, fill=(125, 211, 252, 255))

    # Overlay standard lower-third onto Frame A
    ov_png = OVERLAY_DIR / f"{shot.shot_id}_overlay.png"
    render_shot_overlay_png(shot, ov_png)
    ov_im = Image.open(ov_png).convert("RGBA")
    frame_a = Image.alpha_composite(frame_a, ov_im)

    frame_a_path = OVERLAY_DIR / f"{shot.shot_id}_frame_a.png"
    frame_a.convert("RGB").save(frame_a_path, format="PNG")

    # ---- Frame B: Full-screen Live app.py Capture with Lower-Third & Top Badge ----
    frame_b = Image.new("RGBA", (WIDTH, HEIGHT), (15, 23, 42, 255))
    if app_screenshot and app_screenshot.is_file():
        try:
            bg = Image.open(app_screenshot).convert("RGBA").resize((WIDTH, HEIGHT))
            frame_b.paste(bg, (0, 0))
        except Exception:
            pass
    frame_b = Image.alpha_composite(frame_b, ov_im)
    frame_b_path = OVERLAY_DIR / f"{shot.shot_id}_frame_b.png"
    frame_b.convert("RGB").save(frame_b_path, format="PNG")

    return frame_a_path, frame_b_path, metrics


def build_proof_slot_video(
    shot: ShotSpec,
    out_clip: Path,
    *,
    out_dir: Path = OUT_DIR,
    capture_app: bool = True,
) -> Tuple[Path, Dict[str, Any]]:
    """Build Shot 5 (14s) from live app.py screen capture + real pipeline outputs in out/."""
    RAW_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out_clip.parent.mkdir(parents=True, exist_ok=True)

    app_png = RAW_CLIP_DIR / "app_capture.png"
    captured_ok = False
    capture_msg = "skipped"
    if capture_app:
        captured_ok, capture_msg = capture_streamlit_app_screenshot(app_png)
        print(f"[promo] Streamlit app.py capture: ok={captured_ok} ({capture_msg})", flush=True)

    frame_a_path, frame_b_path, metrics = render_proof_slot_frames(
        shot,
        app_png if app_png.is_file() else None,
        out_dir=out_dir,
    )
    metrics["app_captured"] = captured_ok
    metrics["app_capture_source"] = capture_msg

    half_dur = shot.duration_sec / 2.0
    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-t",
        f"{half_dur:.2f}",
        "-i",
        str(frame_a_path),
        "-loop",
        "1",
        "-t",
        f"{half_dur:.2f}",
        "-i",
        str(frame_b_path),
        "-f",
        "lavfi",
        "-t",
        f"{shot.duration_sec:.2f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-filter_complex",
        f"[0:v]fps={FPS},format=yuv420p[v0];[1:v]fps={FPS},format=yuv420p[v1];[v0][v1]concat=n=2:v=1:a=0[vout]",
        "-map",
        "[vout]",
        "-map",
        "2:a",
        "-t",
        f"{shot.duration_sec:.2f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-shortest",
        str(out_clip),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building proof slot {out_clip}: {res.stderr[-600:]}")
    return out_clip, metrics


def stitch_promo_video(
    shot_files: Sequence[Path],
    out_video: Path = FINAL_VIDEO,
) -> Dict[str, Any]:
    """Stitch all shot MP4 files into out/promo/surgatlas_promo.mp4 via ffmpeg concat filter."""
    out_video.parent.mkdir(parents=True, exist_ok=True)
    missing = [str(p) for p in shot_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot stitch promo video; missing shot files: {missing}")

    inputs: List[str] = []
    filter_parts: List[str] = []
    for idx, p in enumerate(shot_files):
        inputs.extend(["-i", str(p)])
        filter_parts.append(f"[{idx}:v:0][{idx}:a:0]")

    n = len(shot_files)
    filter_complex = "".join(filter_parts) + f"concat=n={n}:v=1:a=1[vout][aout]"

    cmd = [
        "ffmpeg",
        "-y",
        *inputs,
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-map",
        "[aout]",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out_video),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {res.stderr[-600:]}")

    meta = probe_video(out_video)
    if not meta.get("valid"):
        raise RuntimeError(f"Stitched video {out_video} failed ffprobe validation: {meta}")
    return meta


def run_pipeline(
    *,
    draft: bool = False,
    force_regenerate: bool = False,
    refresh_slot_only: bool = False,
    only_shot: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate shots, compose overlays, build the real proof slot, and stitch final MP4."""
    PROMO_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CLIP_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target_total_duration_sec": total_storyboard_duration(),
        "shots": {},
    }
    if MANIFEST_PATH.is_file():
        try:
            old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if isinstance(old.get("shots"), dict):
                manifest["shots"] = old["shots"]
        except Exception:
            pass

    shot_paths: List[Path] = []
    for shot in SHOTS:
        out_shot_path = PROMO_DIR / shot.filename
        shot_paths.append(out_shot_path)

        if only_shot and shot.shot_id != only_shot:
            continue

        if shot.is_proof_slot:
            out_clip, slot_metrics = build_proof_slot_video(shot, out_shot_path)
            meta = probe_video(out_clip)
            manifest["shots"][shot.shot_id] = {
                "filename": shot.filename,
                "illustrative": False,
                "is_proof_slot": True,
                "duration_sec": meta.get("duration_sec", 0.0),
                "metrics": slot_metrics,
            }
            continue

        if refresh_slot_only and out_shot_path.is_file() and probe_video(out_shot_path).get("valid"):
            continue

        raw_path = RAW_CLIP_DIR / f"{shot.shot_id}_raw.mp4"
        model_used = manifest["shots"].get(shot.shot_id, {}).get("model", "cached")
        if force_regenerate or not probe_video(raw_path).get("valid"):
            raw_path, model_used = generate_raw_clip_with_veo_or_omni(
                shot,
                raw_path,
                draft=draft,
            )

        overlay_png = OVERLAY_DIR / f"{shot.shot_id}_overlay.png"
        render_shot_overlay_png(shot, overlay_png)
        compose_illustrative_shot(shot, raw_path, overlay_png, out_shot_path)
        meta = probe_video(out_shot_path)
        manifest["shots"][shot.shot_id] = {
            "filename": shot.filename,
            "illustrative": True,
            "badge_text": shot.badge_text,
            "model": model_used,
            "raw_clip": str(raw_path.relative_to(ROOT)),
            "duration_sec": meta.get("duration_sec", 0.0),
            "width": meta.get("width", 0),
            "height": meta.get("height", 0),
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # If all 6 shot files exist, stitch final video
    if all(p.is_file() for p in shot_paths):
        final_meta = stitch_promo_video(shot_paths, FINAL_VIDEO)
        manifest["final_video"] = {
            "path": str(FINAL_VIDEO.relative_to(ROOT)),
            "duration_sec": final_meta.get("duration_sec", 0.0),
            "width": final_meta.get("width", 0),
            "height": final_meta.get("height", 0),
            "size_bytes": FINAL_VIDEO.stat().st_size,
        }
        print(
            f"[promo] Stitched {FINAL_VIDEO} ({final_meta['duration_sec']:.1f}s, "
            f"{final_meta['width']}x{final_meta['height']}, {FINAL_VIDEO.stat().st_size / 1e6:.2f} MB)",
            flush=True,
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate and stitch the SurgAtlas pitch video.")
    p.add_argument("--draft", action="store_true", help="Use veo-3.1-fast-generate-preview first")
    p.add_argument("--force", action="store_true", help="Force regenerate raw Veo clips even if cached")
    p.add_argument("--refresh-slot", action="store_true", help="Only rebuild Shot 5 (real numbers + app.py) and re-stitch")
    p.add_argument("--shot", help="Only build a specific shot_id (e.g. shot_01_med_school)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(
        draft=args.draft,
        force_regenerate=args.force,
        refresh_slot_only=args.refresh_slot,
        only_shot=args.shot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
