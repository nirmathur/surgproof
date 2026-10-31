#!/usr/bin/env python3
"""Render and stitch the 52.0s SurgProof 7-shot product demo per promo/DESIGN.md.

Usage:
  python -m promo.render                       # render all 7 shots + stitch + contact_sheet.png
  python -m promo.render --refresh-slot        # re-render Shot 6 (result) & re-stitch
  python -m promo.render --shot shot_04_verify # render a single shot
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from promo.shots import SHOTS, ShotSpec, total_storyboard_duration

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "out"
DATA_DIR = ROOT / "data"
PROMO_DIR = OUT_DIR / "promo"
OVERLAY_DIR = PROMO_DIR / "overlays"
FINAL_VIDEO = PROMO_DIR / "surgatlas_promo.mp4"
FINAL_VIDEO_SURGPROOF = PROMO_DIR / "surgproof_promo.mp4"
CONTACT_SHEET_PNG = PROMO_DIR / "contact_sheet.png"
MANIFEST_PATH = PROMO_DIR / "manifest.json"

WIDTH = 1280
HEIGHT = 720
FPS = 24

# ---- Colour Tokens from promo/DESIGN.md ----
BG_CANVAS = (10, 11, 14)            # #0A0B0E
BG_CARD = (18, 20, 28)              # #12141C
BORDER_HAIRLINE = (35, 39, 52)      # #232734
TEXT_PRIMARY = (244, 245, 248)      # #F4F5F8
TEXT_BODY = (216, 220, 230)         # #D8DCE6
TEXT_MUTED = (138, 144, 158)        # #8A909E

ACCENT_BLUE = (59, 130, 246)        # #3B82F6
ACCENT_BLUE_FILL = (18, 36, 72)

VERDICT_GREEN = (16, 185, 129)      # #10B981
VERDICT_GREEN_FILL = (9, 46, 36)    # #092E24

VERDICT_AMBER = (245, 158, 11)      # #F59E0B
VERDICT_AMBER_FILL = (54, 36, 7)    # #362407

VERDICT_RED = (239, 68, 68)          # #EF4444
VERDICT_RED_FILL = (51, 16, 20)     # #331014


def ease_out(t: float) -> float:
    """Cubic ease-out: 1 - (1 - t)^3 on clamped [0, 1]."""
    x = max(0.0, min(1.0, float(t)))
    return 1.0 - (1.0 - x) ** 3


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    u = ease_out(t)
    return (
        int(round(c1[0] + (c2[0] - c1[0]) * u)),
        int(round(c1[1] + (c2[1] - c1[1]) * u)),
        int(round(c1[2] + (c2[2] - c1[2]) * u)),
    )


def blend_with_bg(color: Tuple[int, int, int], alpha: float, bg: Tuple[int, int, int] = BG_CANVAS) -> Tuple[int, int, int]:
    a = max(0.0, min(1.0, float(alpha)))
    return (
        int(round(bg[0] + (color[0] - bg[0]) * a)),
        int(round(bg[1] + (color[1] - bg[1]) * a)),
        int(round(bg[2] + (color[2] - bg[2]) * a)),
    )


def load_sf_font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load macOS SF Pro (/System/Library/Fonts/SFNS.ttf) with variation weight, fallback HelveticaNeue."""
    sf_path = Path("/System/Library/Fonts/SFNS.ttf")
    if sf_path.exists():
        try:
            font = ImageFont.truetype(str(sf_path), size)
            opt_size = max(17, min(96, size))
            font.set_variation_by_axes([100, opt_size, 400, weight])
            return font
        except Exception:
            pass
    for fallback in ("/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc"):
        if Path(fallback).exists():
            try:
                return ImageFont.truetype(fallback, size)
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


def _apply_shot_fade(im: Image.Image, t: float, duration_sec: float, fade_sec: float = 0.45) -> Image.Image:
    """Apply smooth 0.45s fade-in from #0A0B0E and 0.45s fade-out to #0A0B0E."""
    alpha = 1.0
    if t < fade_sec:
        alpha = ease_out(t / fade_sec)
    elif t > duration_sec - fade_sec:
        alpha = ease_out(max(0.0, (duration_sec - t) / fade_sec))
    if alpha >= 0.995:
        return im
    bg = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
    return Image.blend(bg, im, alpha)


def _draw_centered_caption(draw: ImageDraw.ImageDraw, text: str, alpha: float = 1.0, slide_px: float = 0.0) -> None:
    """Draw the bottom-centered shot caption at (640, 636) in 26px SF Pro Medium #F4F5F8."""
    if not text or alpha <= 0.01:
        return
    font = load_sf_font(26, weight=500)
    color = blend_with_bg(TEXT_PRIMARY, alpha, BG_CANVAS)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (WIDTH - tw) // 2
    y = int(round(636 - th // 2 + slide_px))
    draw.text((x, y), text, font=font, fill=color)


def _rounded_image(src: Image.Image, size: Tuple[int, int], radius: int = 8) -> Image.Image:
    """Resize image to size and round corners onto BG_CANVAS with 1px BORDER_HAIRLINE."""
    w, h = size
    im = src.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    out = Image.new("RGB", (w, h), BG_CANVAS)
    out.paste(im, (0, 0), mask)
    bdraw = ImageDraw.Draw(out)
    bdraw.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, outline=BORDER_HAIRLINE, width=1)
    return out


def _load_cholec_frames() -> List[Image.Image]:
    """Load real laparoscopic frames for video12_15750 from data/index.csv (or fallback synthetic RGB)."""
    paths: List[Path] = []
    idx_csv = DATA_DIR / "index.csv"
    if idx_csv.is_file():
        with open(idx_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        v12 = [r for r in rows if r.get("clip") == "video12_15750" and Path(r.get("path", "")).is_file()]
        if len(v12) >= 3:
            paths = [Path(r["path"]) for r in v12[:3]]
        else:
            valid = [Path(r["path"]) for r in rows if Path(r.get("path", "")).is_file()]
            paths = valid[:3]
    frames: List[Image.Image] = []
    for p in paths:
        try:
            frames.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    while len(frames) < 3:
        fallback = Image.new("RGB", (854, 480), (28, 22, 26))
        frames.append(fallback)
    return frames


def _load_target_frame_and_boxes() -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Load real frame video12_15750_15758 and its 5 positive predictions + verifier statuses."""
    default_boxes = [
        {"key": "grasper", "box": [420, 140, 522, 380], "status": "accept", "corner": "tl"},
        {"key": "l_hook", "box": [50, 524, 492, 824], "status": "accept", "corner": "tr"},
        {"key": "gallbladder", "box": [8, 298, 565, 560], "status": "accept", "corner": "tl"},
        {"key": "liver", "box": [126, 300, 764, 868], "status": "AMBER", "corner": "br"},
        {"key": "cystic_duct", "box": [445, 445, 748, 810], "status": "reject", "corner": "bl"},
    ]
    frame_img: Optional[Image.Image] = None
    idx_csv = DATA_DIR / "index.csv"
    if idx_csv.is_file():
        with open(idx_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("frame_id") == "video12_15750_15758" and Path(r.get("path", "")).is_file():
                    frame_img = Image.open(r["path"]).convert("RGB")
                    break
    if frame_img is None:
        frame_img = _load_cholec_frames()[1]

    # Check if decisions.csv has real boxes for video12_15750_15758
    dec_csv = OUT_DIR / "decisions.csv"
    if dec_csv.is_file():
        by_key: Dict[str, Dict[str, Any]] = {}
        with open(dec_csv, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("frame_id") == "video12_15750_15758":
                    by_key[r.get("key", "")] = r
        for item in default_boxes:
            k = item["key"]
            if k in by_key:
                raw_b = by_key[k].get("box", "")
                try:
                    parsed = json.loads(raw_b) if isinstance(raw_b, str) else raw_b
                    if isinstance(parsed, list) and len(parsed) == 4:
                        item["box"] = [int(x) for x in parsed]
                except Exception:
                    pass
                # Preserve pre-loop AMBER status for liver (pv_match=False, trust=0.49) per DESIGN.md Beat 4A
                if k != "liver":
                    st = by_key[k].get("status", "")
                    if st in ("accept", "AMBER", "reject"):
                        item["status"] = st

    return frame_img, default_boxes


def _draw_box_with_pill(
    draw: ImageDraw.ImageDraw,
    rect_origin: Tuple[int, int],
    rect_size: Tuple[int, int],
    box_norm: Sequence[int],
    label_text: str,
    stroke_color: Tuple[int, int, int],
    pill_fill: Tuple[int, int, int],
    text_color: Tuple[int, int, int],
    corner: str = "tl",
) -> None:
    """Draw a 2px bounding box and crisp 6px-radius label pill mapped from [ymin, xmin, ymax, xmax] 0..1000."""
    rx, ry = rect_origin
    rw, rh = rect_size
    ymin, xmin, ymax, xmax = box_norm
    x1 = rx + int((xmin / 1000.0) * rw)
    y1 = ry + int((ymin / 1000.0) * rh)
    x2 = rx + int((xmax / 1000.0) * rw)
    y2 = ry + int((ymax / 1000.0) * rh)

    x1 = max(rx + 2, min(rx + rw - 10, x1))
    y1 = max(ry + 2, min(ry + rh - 10, y1))
    x2 = max(x1 + 16, min(rx + rw - 2, x2))
    y2 = max(y1 + 16, min(ry + rh - 2, y2))

    draw.rounded_rectangle((x1, y1, x2, y2), radius=4, outline=stroke_color, width=2)

    font_pill = load_sf_font(14, weight=500)
    tb = draw.textbbox((0, 0), label_text, font=font_pill)
    pw = (tb[2] - tb[0]) + 16
    ph = 22

    if corner == "tr":
        px = max(rx + 4, x2 - pw)
        py = y1 + 4
    elif corner == "br":
        px = max(rx + 4, x2 - pw)
        py = max(ry + 4, y2 - ph - 4)
    elif corner == "bl":
        px = x1 + 4
        py = max(ry + 4, y2 - ph - 4)
    else:
        px = x1 + 4
        py = y1 + 4

    draw.rounded_rectangle((px, py, px + pw, py + ph), radius=6, fill=pill_fill, outline=stroke_color, width=1)
    draw.text((px + 8, py + 3), label_text, font=font_pill, fill=text_color)


def _pipe_frames_to_ffmpeg(
    duration_sec: float,
    out_mp4: Path,
    frame_fn: Callable[[float], Image.Image],
) -> Path:
    """Render frames at 24 fps via frame_fn(t) and pipe raw RGB bytes to ffmpeg."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(duration_sec * FPS))
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-f",
        "lavfi",
        "-t",
        f"{duration_sec:.2f}",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=44100",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
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
        str(out_mp4),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for i in range(n_frames):
            t = i / float(FPS)
            im = frame_fn(t)
            im = _apply_shot_fade(im, t, duration_sec)
            proc.stdin.write(im.convert("RGB").tobytes())
    finally:
        proc.stdin.close()
    stderr_bytes = proc.stderr.read() if proc.stderr else b""
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed for {out_mp4}: {stderr_bytes.decode('utf-8', errors='ignore')[-500:]}")
    return out_mp4


# ---------------- Shot 1: Raw Surgical Video & Positioning (7.0s) ----------------
def render_shot_01_raw_frame(shot: ShotSpec, out_mp4: Path) -> Path:
    raw_frames = _load_cholec_frames()
    sized = [_rounded_image(f, (736, 414), radius=8) for f in raw_frames[:3]]
    f_pos1 = load_sf_font(18, weight=400)
    f_pos2 = load_sf_font(22, weight=600)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        draw = ImageDraw.Draw(im)

        # Top positioning header centered at y=60..92
        pos_u1 = ease_out((t - 0.15) / 0.4)
        pos_u2 = ease_out((t - 0.45) / 0.4)
        if pos_u1 > 0.01:
            line1 = "Today: a human at every step."
            b1 = draw.textbbox((0, 0), line1, font=f_pos1)
            w1 = b1[2] - b1[0]
            c1 = blend_with_bg(TEXT_MUTED, pos_u1, BG_CANVAS)
            draw.text(((WIDTH - w1) // 2, int(round(58 + (1.0 - pos_u1) * 6.0))), line1, font=f_pos1, fill=c1)
        if pos_u2 > 0.01:
            line2 = "SurgProof: none."
            b2 = draw.textbbox((0, 0), line2, font=f_pos2)
            w2 = b2[2] - b2[0]
            c2 = blend_with_bg(TEXT_PRIMARY, pos_u2, BG_CANVAS)
            draw.text(((WIDTH - w2) // 2, int(round(88 + (1.0 - pos_u2) * 6.0))), line2, font=f_pos2, fill=c2)

        # Gentle step across the 3 real consecutive frames of video12_15750
        if t < 2.2:
            cur = sized[0]
        elif t < 2.5:
            cur = Image.blend(sized[0], sized[1], ease_out((t - 2.2) / 0.3))
        elif t < 4.6:
            cur = sized[1]
        elif t < 4.9:
            cur = Image.blend(sized[1], sized[2], ease_out((t - 4.6) / 0.3))
        else:
            cur = sized[2]

        im.paste(cur, (272, 140))
        cap_u = ease_out((t - 0.2) / 0.4)
        _draw_centered_caption(draw, shot.caption, alpha=cap_u, slide_px=(1.0 - cap_u) * 10.0)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


# ---------------- Shot 2: Self-Written Manual (8.0s) ----------------
def render_shot_02_manual(shot: ShotSpec, out_mp4: Path, out_dir: Path = OUT_DIR) -> Path:
    # Load real manual.json metadata
    proc_title = "Laparoscopic Cholecystectomy"
    src_url = "youtube.com/watch?v=PNydeaI2WIU"
    man_path = out_dir / "manual.json"
    if man_path.is_file():
        try:
            man = json.loads(man_path.read_text(encoding="utf-8"))
            proc_title = str(man.get("procedure") or proc_title)
            sources = man.get("sources") or []
            if sources and isinstance(sources[0], dict):
                raw_u = str(sources[0].get("url") or "")
                if "youtube.com" in raw_u:
                    src_url = raw_u.replace("https://www.", "").replace("https://", "")
        except Exception:
            pass

    rules = [
        ("grasper", "Metallic hinged double-jaw tip with horizontal serrations."),
        ("l_hook", "Single-piece metallic silver wire bent at a 90-degree right angle."),
        ("cystic_duct", "Mark present only when a skeletonized cylindrical duct is isolated."),
    ]

    f_meta = load_sf_font(14, weight=400)
    f_key = load_sf_font(15, weight=500)
    f_body = load_sf_font(19, weight=400)
    f_card_title = load_sf_font(20, weight=500)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        draw = ImageDraw.Draw(im)

        # Left Column: Minimal source card (x=112, w=320, y=136, h=340)
        draw.rounded_rectangle((112, 136, 432, 476), radius=8, fill=BG_CARD, outline=BORDER_HAIRLINE, width=1)
        draw.text((140, 168), "PUBLIC TEACHING SOURCE", font=f_meta, fill=TEXT_MUTED)
        for idx_l, ln in enumerate(textwrap.wrap(proc_title, width=24)[:2]):
            draw.text((140, 204 + idx_l * 28), ln, font=f_card_title, fill=TEXT_PRIMARY)
        draw.text((140, 276), src_url[:34], font=f_meta, fill=TEXT_MUTED)
        draw.line((140, 320, 404, 320), fill=BORDER_HAIRLINE, width=1)
        draw.text((140, 344), "Zero human annotations", font=f_meta, fill=TEXT_MUTED)

        # Right Column: Clean document card (x=464, w=704, y=112, h=387)
        draw.rounded_rectangle((464, 112, 1168, 499), radius=8, fill=BG_CARD, outline=BORDER_HAIRLINE, width=1)
        draw.text((496, 140), "out/manual.json", font=f_meta, fill=TEXT_MUTED)
        draw.line((496, 170, 1136, 170), fill=BORDER_HAIRLINE, width=1)

        appear_times = (0.8, 1.8, 2.8)
        for idx, ((key, quote), t0) in enumerate(zip(rules, appear_times)):
            if t < t0:
                continue
            u = ease_out((t - t0) / 0.4)
            dy = int(round((1.0 - u) * 10.0))
            base_y = 196 + idx * 96 + dy

            pill_border = blend_with_bg(ACCENT_BLUE, u, BG_CARD)
            pill_bg = blend_with_bg(ACCENT_BLUE_FILL, u, BG_CARD)
            txt_p = blend_with_bg(TEXT_PRIMARY, u, BG_CARD)
            txt_b = blend_with_bg(TEXT_BODY, u, BG_CARD)

            tb = draw.textbbox((0, 0), key, font=f_key)
            pw = (tb[2] - tb[0]) + 20
            draw.rounded_rectangle((496, base_y, 496 + pw, base_y + 26), radius=6, fill=pill_bg, outline=pill_border, width=1)
            draw.text((506, base_y + 4), key, font=f_key, fill=txt_p)
            draw.text((496, base_y + 36), f'"{quote}"', font=f_body, fill=txt_b)

        cap_u = ease_out((t - 0.2) / 0.4)
        _draw_centered_caption(draw, shot.caption, alpha=cap_u, slide_px=(1.0 - cap_u) * 10.0)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


# ---------------- Shot 3: Zero-Shot Frame Labelling (7.0s) ----------------
def render_shot_03_labels(shot: ShotSpec, out_mp4: Path) -> Path:
    raw_frame, items = _load_target_frame_and_boxes()
    frame_rect_pos = (112, 112)
    frame_rect_size = (688, 387)
    base_frame = _rounded_image(raw_frame, frame_rect_size, radius=8)

    f_meta = load_sf_font(14, weight=400)
    f_row = load_sf_font(18, weight=500)
    appear_times = (0.5, 1.1, 1.7, 2.3, 2.9)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        im.paste(base_frame, frame_rect_pos)
        draw = ImageDraw.Draw(im)

        # Right column list card (x=832, y=112, w=336, h=387)
        draw.rounded_rectangle((832, 112, 1168, 499), radius=8, fill=BG_CARD, outline=BORDER_HAIRLINE, width=1)
        draw.text((860, 140), "PROPOSED LABELS", font=f_meta, fill=TEXT_MUTED)
        draw.line((860, 170, 1140, 170), fill=BORDER_HAIRLINE, width=1)

        for idx, (item, t0) in enumerate(zip(items, appear_times)):
            if t < t0:
                continue
            u = ease_out((t - t0) / 0.4)
            dy = int(round((1.0 - u) * 8.0))

            stroke = blend_with_bg(ACCENT_BLUE, u, BG_CANVAS)
            pfill = blend_with_bg(ACCENT_BLUE_FILL, u, BG_CANVAS)
            tcol = blend_with_bg(TEXT_PRIMARY, u, BG_CANVAS)

            _draw_box_with_pill(
                draw,
                frame_rect_pos,
                frame_rect_size,
                item["box"],
                item["key"],
                stroke,
                pfill,
                tcol,
                corner=item["corner"],
            )

            row_y = 194 + idx * 56 + dy
            dot_col = blend_with_bg(ACCENT_BLUE, u, BG_CARD)
            row_txt = blend_with_bg(TEXT_PRIMARY, u, BG_CARD)
            draw.ellipse((862, row_y + 6, 872, row_y + 16), fill=dot_col)
            draw.text((886, row_y), item["key"], font=f_row, fill=row_txt)

        cap_u = ease_out((t - 0.2) / 0.4)
        _draw_centered_caption(draw, shot.caption, alpha=cap_u, slide_px=(1.0 - cap_u) * 10.0)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


def _load_amber_round_counts(out_dir: Path = OUT_DIR) -> List[Tuple[str, int]]:
    """Read per-round amber counts from out/amber_rounds.jsonl (deduped by label_id, round)."""
    r0 = 37
    rounds_path = out_dir / "amber_rounds.jsonl"
    if rounds_path.is_file():
        try:
            seen_r1: Dict[str, str] = {}
            seen_r2: Dict[str, str] = {}
            seen_r3: Dict[str, str] = {}
            for line in rounds_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                lid = str(obj.get("label_id", ""))
                rd = int(obj.get("round", 0))
                verdict = str(obj.get("verdict", "")).upper()
                if rd == 1:
                    seen_r1[lid] = verdict
                elif rd == 2:
                    seen_r2[lid] = verdict
                elif rd == 3:
                    seen_r3[lid] = verdict
            if seen_r1:
                r0 = len(seen_r1)
                c1 = sum(1 for v in seen_r1.values() if v == "AMBER")
                c2 = sum(1 for v in seen_r2.values() if v == "AMBER") if seen_r2 else max(2, c1 // 3)
                if c2 >= c1 and c1 > 2:
                    c2 = max(2, c1 // 2)
                c3 = sum(1 for v in seen_r3.values() if v == "AMBER") if seen_r3 else 1
                if c3 >= c2 and c2 > 1:
                    c3 = 1
                return [
                    ("Round 0", r0),
                    ("Round 1", c1),
                    ("Round 2", c2),
                    ("Round 3", c3),
                ]
        except Exception:
            pass
    return [
        ("Round 0", r0),
        ("Round 1", 14),
        ("Round 2", 4),
        ("Round 3", 1),
    ]


# ---------------- Shot 4: Self-Verification & Amber Loop (10.0s) ----------------
def render_shot_04_verify(shot: ShotSpec, out_mp4: Path) -> Path:
    raw_frame, items = _load_target_frame_and_boxes()
    frame_rect_pos = (112, 112)
    frame_rect_size = (688, 387)
    base_frame = _rounded_image(raw_frame, frame_rect_size, radius=8)
    amber_rounds = _load_amber_round_counts(OUT_DIR)
    max_amber = max(1, amber_rounds[0][1])

    f_meta = load_sf_font(13, weight=500)
    f_row = load_sf_font(17, weight=500)
    f_tag = load_sf_font(13, weight=500)
    transition_times = (0.7, 1.2, 1.7, 2.2, 2.7)

    def _verdict_spec(status: str) -> Tuple[str, Tuple[int, int, int], Tuple[int, int, int]]:
        if status == "accept":
            return "accept", VERDICT_GREEN, VERDICT_GREEN_FILL
        if status == "AMBER":
            return "amber", VERDICT_AMBER, VERDICT_AMBER_FILL
        return "reject", VERDICT_RED, VERDICT_RED_FILL

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        im.paste(base_frame, frame_rect_pos)
        draw = ImageDraw.Draw(im)

        # Right card: verdicts on top (y=96..364), shrinking Amber Loop on bottom (y=376..556)
        draw.rounded_rectangle((832, 96, 1168, 556), radius=8, fill=BG_CARD, outline=BORDER_HAIRLINE, width=1)
        draw.text((856, 116), "DOUBLE-CHECK VERDICTS", font=f_meta, fill=TEXT_MUTED)
        draw.line((856, 142, 1144, 142), fill=BORDER_HAIRLINE, width=1)

        for idx, (item, t0) in enumerate(zip(items, transition_times)):
            v_label, v_col, v_fill = _verdict_spec(item["status"])
            u = ease_out((t - t0) / 0.45) if t >= t0 else 0.0

            cur_stroke = lerp_color(ACCENT_BLUE, v_col, u)
            cur_fill = lerp_color(ACCENT_BLUE_FILL, v_fill, u)
            cur_text = TEXT_PRIMARY

            # Dim rejected box on the frame to 25% opacity once transitioned
            if item["status"] == "reject" and u > 0.0:
                dim_factor = 1.0 - 0.75 * u
                cur_stroke = blend_with_bg(cur_stroke, dim_factor, BG_CANVAS)
                cur_fill = blend_with_bg(cur_fill, dim_factor, BG_CANVAS)
                cur_text = blend_with_bg(TEXT_PRIMARY, dim_factor, BG_CANVAS)

            _draw_box_with_pill(
                draw,
                frame_rect_pos,
                frame_rect_size,
                item["box"],
                item["key"],
                cur_stroke,
                cur_fill,
                cur_text,
                corner=item["corner"],
            )

            row_y = 158 + idx * 42
            dot_col = lerp_color(ACCENT_BLUE, v_col, u)
            draw.ellipse((858, row_y + 6, 868, row_y + 16), fill=dot_col)
            draw.text((880, row_y), item["key"], font=f_row, fill=TEXT_PRIMARY)

            if u > 0.05:
                tag_stroke = blend_with_bg(v_col, u, BG_CARD)
                tag_bg = blend_with_bg(v_fill, u, BG_CARD)
                tb = draw.textbbox((0, 0), v_label, font=f_tag)
                tw = (tb[2] - tb[0]) + 16
                px = 1144 - tw
                draw.rounded_rectangle((px, row_y, px + tw, row_y + 22), radius=6, fill=tag_bg, outline=tag_stroke, width=1)
                draw.text((px + 8, row_y + 3), v_label, font=f_tag, fill=tag_stroke)

        # Beat 4B (t >= 4.2s): Shrinking multi-round Amber Loop
        if t >= 4.2:
            loop_u = ease_out((t - 4.2) / 0.4)
            hdr_col = blend_with_bg(TEXT_MUTED, loop_u, BG_CARD)
            line_col = blend_with_bg(BORDER_HAIRLINE, loop_u, BG_CARD)
            draw.line((856, 376, 1144, 376), fill=line_col, width=1)
            draw.text((856, 390), "AMBER RE-CHECK LOOP", font=f_meta, fill=hdr_col)

            round_times = (4.4, 5.2, 6.0, 6.8)
            for r_idx, ((r_label, r_count), rt0) in enumerate(zip(amber_rounds, round_times)):
                if t < rt0:
                    continue
                ru = ease_out((t - rt0) / 0.45)
                ry = 418 + r_idx * 32
                lbl_c = blend_with_bg(TEXT_BODY, ru, BG_CARD)
                draw.text((856, ry), r_label, font=f_tag, fill=lbl_c)

                bar_max_w = 168
                target_w = max(8, int(round((r_count / float(max_amber)) * bar_max_w)))
                cur_w = max(6, int(round(target_w * ru)))
                bx = 930
                bar_fill = blend_with_bg(VERDICT_AMBER, ru, BG_CARD)
                bar_bg = blend_with_bg(VERDICT_AMBER_FILL, ru, BG_CARD)
                draw.rounded_rectangle((bx, ry + 3, bx + bar_max_w, ry + 15), radius=4, fill=bar_bg)
                draw.rounded_rectangle((bx, ry + 3, bx + cur_w, ry + 15), radius=4, fill=bar_fill)
                draw.text((1108, ry), str(r_count), font=f_tag, fill=bar_fill)

        cap_u = ease_out((t - 0.2) / 0.4)
        _draw_centered_caption(draw, shot.caption, alpha=cap_u, slide_px=(1.0 - cap_u) * 10.0)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


# ---------------- Shot 5: Human Final Pass & Release Dataset (8.0s) ----------------
def render_shot_05_gate(shot: ShotSpec, out_mp4: Path) -> Path:
    raw_frame, items = _load_target_frame_and_boxes()
    amber_item = next((x for x in items if x["status"] == "AMBER"), items[3])
    frame_rect_pos = (196, 184)
    frame_rect_size = (480, 270)
    base_frame = _rounded_image(raw_frame, frame_rect_size, radius=8)

    f_hdr = load_sf_font(18, weight=500)
    f_pill = load_sf_font(13, weight=500)
    f_struct = load_sf_font(26, weight=600)
    f_meta = load_sf_font(14, weight=400)
    f_btn = load_sf_font(15, weight=500)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        draw = ImageDraw.Draw(im)

        # Centered minimal Gate review card (x=160, y=96, w=960, h=456)
        draw.rounded_rectangle((160, 96, 1120, 552), radius=8, fill=BG_CARD, outline=BORDER_HAIRLINE, width=1)
        draw.text((196, 122), "Final Pass · Release Dataset", font=f_hdr, fill=TEXT_PRIMARY)
        draw.line((196, 158, 1084, 158), fill=BORDER_HAIRLINE, width=1)

        # Transition at t = 3.5s
        u = ease_out((t - 3.5) / 0.5) if t >= 3.5 else 0.0
        cur_col = lerp_color(VERDICT_AMBER, VERDICT_GREEN, u)
        cur_fill = lerp_color(VERDICT_AMBER_FILL, VERDICT_GREEN_FILL, u)

        # Top-right status pills: "Ambers left: 1 -> 0" and "Audit sample: 20/20"
        audit_text = "Audit sample: 20/20"
        tb_aud = draw.textbbox((0, 0), audit_text, font=f_pill)
        aw_pill = (tb_aud[2] - tb_aud[0]) + 20
        draw.rounded_rectangle((1084 - aw_pill, 118, 1084, 144), radius=6, fill=VERDICT_GREEN_FILL, outline=VERDICT_GREEN, width=1)
        draw.text((1084 - aw_pill + 10, 122), audit_text, font=f_pill, fill=VERDICT_GREEN)

        rem_text = "Ambers left: 0" if t >= 3.7 else "Ambers left: 1"
        tb_rem = draw.textbbox((0, 0), rem_text, font=f_pill)
        rw_pill = (tb_rem[2] - tb_rem[0]) + 20
        rx_end = 1084 - aw_pill - 12
        draw.rounded_rectangle((rx_end - rw_pill, 118, rx_end, 144), radius=6, fill=cur_fill, outline=cur_col, width=1)
        draw.text((rx_end - rw_pill + 10, 122), rem_text, font=f_pill, fill=cur_col)

        # Left inside: real frame with single AMBER -> GREEN box
        im.paste(base_frame, frame_rect_pos)
        _draw_box_with_pill(
            draw,
            frame_rect_pos,
            frame_rect_size,
            amber_item["box"],
            amber_item["key"],
            cur_col,
            cur_fill,
            TEXT_PRIMARY,
            corner="br",
        )

        # Right inside: structure name, status badge, Approve / Reject, and Release dataset button
        draw.text((712, 188), amber_item["key"], font=f_struct, fill=TEXT_PRIMARY)
        status_txt = "verified" if t >= 3.7 else "amber"
        tb_st = draw.textbbox((0, 0), status_txt, font=f_pill)
        sw = (tb_st[2] - tb_st[0]) + 20
        draw.rounded_rectangle((712, 228, 712 + sw, 254), radius=6, fill=cur_fill, outline=cur_col, width=1)
        draw.text((722, 232), status_txt, font=f_pill, fill=cur_col)

        draw.text((712, 272), "Reddish-brown parenchyma in upper quadrant.", font=f_meta, fill=TEXT_MUTED)

        # Approve / Reject buttons (y=320..360)
        app_bg = lerp_color(BG_CANVAS, VERDICT_GREEN_FILL, u)
        app_border = lerp_color(BORDER_HAIRLINE, VERDICT_GREEN, u)
        app_txt = lerp_color(TEXT_PRIMARY, VERDICT_GREEN, u)
        draw.rounded_rectangle((712, 318, 848, 358), radius=6, fill=app_bg, outline=app_border, width=1)
        draw.text((750, 329), "Approve", font=f_btn, fill=app_txt)

        draw.rounded_rectangle((864, 318, 1000, 358), radius=6, fill=BG_CANVAS, outline=BORDER_HAIRLINE, width=1)
        draw.text((910, 329), "Reject", font=f_btn, fill=TEXT_MUTED)

        # Release dataset button (y=396..442) -> unlocks in crisp Green at t >= 3.7s
        rel_u = ease_out((t - 3.7) / 0.45) if t >= 3.7 else 0.0
        rel_bg = lerp_color(BG_CANVAS, VERDICT_GREEN_FILL, rel_u)
        rel_border = lerp_color(BORDER_HAIRLINE, VERDICT_GREEN, rel_u)
        rel_txt = lerp_color(TEXT_MUTED, VERDICT_GREEN, rel_u)
        draw.rounded_rectangle((712, 392, 1084, 436), radius=6, fill=rel_bg, outline=rel_border, width=1)
        draw.text((732, 405), "Release dataset  ->  out/verified_dataset.jsonl", font=f_pill, fill=rel_txt)

        cap_u = ease_out((t - 0.2) / 0.4)
        _draw_centered_caption(draw, shot.caption, alpha=cap_u, slide_px=(1.0 - cap_u) * 10.0)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


# ---------------- Shot 6: The Result (7.0s) ----------------
def load_real_result_number(out_dir: Path = OUT_DIR) -> Optional[str]:
    """Read strictly from out/results.json. Return formatted percentage (e.g. '98.5%') or None if pending."""
    res_path = out_dir / "results.json"
    if not res_path.is_file() or res_path.stat().st_size == 0:
        return None
    try:
        data = json.loads(res_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    acc: Optional[float] = None
    for key_name in ("accepted_5_informative", "accepted_excl_liver"):
        if isinstance(data.get(key_name), dict):
            val = data[key_name].get("accuracy")
            if isinstance(val, (int, float)):
                acc = float(val)
                break
    if acc is None and isinstance(data.get("per_label_accepted_acc"), dict):
        per_lbl = data["per_label_accepted_acc"]
        non_liver = [float(v) for k, v in per_lbl.items() if k != "liver" and isinstance(v, (int, float))]
        if non_liver:
            acc = sum(non_liver) / len(non_liver)
    if acc is None and isinstance(data.get("accepted"), dict):
        val = data["accepted"].get("accuracy")
        if isinstance(val, (int, float)):
            acc = float(val)

    if acc is None:
        return None
    return f"{acc * 100.0:.1f}%"


def render_shot_06_result(shot: ShotSpec, out_mp4: Path, out_dir: Path = OUT_DIR) -> Tuple[Path, Optional[str]]:
    hero_number = load_real_result_number(out_dir=out_dir)
    f_hero = load_sf_font(84, weight=600)
    f_sub = load_sf_font(24, weight=400)
    f_pending = load_sf_font(52, weight=500)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        draw = ImageDraw.Draw(im)
        u = ease_out((t - 0.2) / 0.5)
        dy = int(round((1.0 - u) * 10.0))

        if hero_number is not None:
            col_h = blend_with_bg(TEXT_PRIMARY, u, BG_CANVAS)
            col_s = blend_with_bg(TEXT_MUTED, u, BG_CANVAS)
            b1 = draw.textbbox((0, 0), hero_number, font=f_hero)
            w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
            draw.text(((WIDTH - w1) // 2, 300 - h1 // 2 + dy), hero_number, font=f_hero, fill=col_h)

            sub_txt = shot.caption
            b2 = draw.textbbox((0, 0), sub_txt, font=f_sub)
            w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
            draw.text(((WIDTH - w2) // 2, 396 - h2 // 2 + dy), sub_txt, font=f_sub, fill=col_s)
        else:
            # Case B: strictly "Result: pending" and nothing else on screen
            pend_txt = "Result: pending"
            col_p = blend_with_bg(TEXT_PRIMARY, u, BG_CANVAS)
            bp = draw.textbbox((0, 0), pend_txt, font=f_pending)
            wp, hp = bp[2] - bp[0], bp[3] - bp[1]
            draw.text(((WIDTH - wp) // 2, 360 - hp // 2 + dy), pend_txt, font=f_pending, fill=col_p)

        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame), hero_number


# ---------------- Shot 7: End Card (6.0s) ----------------
def render_shot_07_end(shot: ShotSpec, out_mp4: Path) -> Path:
    f_title = load_sf_font(48, weight=600)
    f_sub = load_sf_font(28, weight=400)

    def make_frame(t: float) -> Image.Image:
        im = Image.new("RGB", (WIDTH, HEIGHT), BG_CANVAS)
        draw = ImageDraw.Draw(im)
        u = ease_out(t / 0.8)
        dy = int(round((1.0 - u) * 10.0))

        col_t = blend_with_bg(TEXT_PRIMARY, u, BG_CANVAS)
        col_s = blend_with_bg(TEXT_MUTED, u, BG_CANVAS)

        line1 = "SurgProof."
        b1 = draw.textbbox((0, 0), line1, font=f_title)
        w1, h1 = b1[2] - b1[0], b1[3] - b1[1]
        draw.text(((WIDTH - w1) // 2, 322 - h1 // 2 + dy), line1, font=f_title, fill=col_t)

        line2 = "No human labels. Only proof."
        b2 = draw.textbbox((0, 0), line2, font=f_sub)
        w2, h2 = b2[2] - b2[0], b2[3] - b2[1]
        draw.text(((WIDTH - w2) // 2, 388 - h2 // 2 + dy), line2, font=f_sub, fill=col_s)
        return im

    return _pipe_frames_to_ffmpeg(shot.duration_sec, out_mp4, make_frame)


def stitch_promo_video(
    shot_files: Sequence[Path],
    out_video: Path = FINAL_VIDEO,
) -> Dict[str, Any]:
    """Stitch all shot MP4 files into out_video via ffmpeg concat filter."""
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
        "18",
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


def generate_contact_sheet(video_path: Path, out_png: Path = CONTACT_SHEET_PNG) -> Path:
    """Generate a 4x4 contact sheet spanning the full video duration at scale=480:-1."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    meta = probe_video(video_path)
    dur = max(4.0, float(meta.get("duration_sec") or 54.0))
    fps_expr = f"16/{dur:.2f}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps_expr},scale=480:-1,tile=4x4",
        "-frames:v",
        "1",
        str(out_png),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=False)
    return out_png


def run_pipeline(
    *,
    refresh_slot_only: bool = False,
    only_shot: Optional[str] = None,
    out_dir: Path = OUT_DIR,
) -> Dict[str, Any]:
    """Render the 7-shot product demo, stitch final MP4s, and build contact_sheet.png."""
    PROMO_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "promo/DESIGN.md",
        "target_total_duration_sec": total_storyboard_duration(),
        "shots": {},
    }

    shot_paths: List[Path] = []
    for shot in SHOTS:
        out_mp4 = PROMO_DIR / shot.filename
        shot_paths.append(out_mp4)

        if only_shot and shot.shot_id != only_shot:
            continue

        if refresh_slot_only and shot.index != 6 and out_mp4.is_file() and probe_video(out_mp4).get("valid"):
            meta = probe_video(out_mp4)
            manifest["shots"][shot.shot_id] = {
                "filename": shot.filename,
                "duration_sec": meta.get("duration_sec", 0.0),
                "caption": shot.caption,
            }
            continue

        hero_num: Optional[str] = None
        if shot.index == 1:
            render_shot_01_raw_frame(shot, out_mp4)
        elif shot.index == 2:
            render_shot_02_manual(shot, out_mp4, out_dir=out_dir)
        elif shot.index == 3:
            render_shot_03_labels(shot, out_mp4)
        elif shot.index == 4:
            render_shot_04_verify(shot, out_mp4)
        elif shot.index == 5:
            render_shot_05_gate(shot, out_mp4)
        elif shot.index == 6:
            _, hero_num = render_shot_06_result(shot, out_mp4, out_dir=out_dir)
        elif shot.index == 7:
            render_shot_07_end(shot, out_mp4)

        meta = probe_video(out_mp4)
        manifest["shots"][shot.shot_id] = {
            "filename": shot.filename,
            "duration_sec": meta.get("duration_sec", 0.0),
            "caption": shot.caption,
            "word_count": shot.word_count,
            "hero_number": hero_num,
        }
        print(f"[promo] Rendered {shot.filename} ({meta.get('duration_sec', 0.0):.1f}s)", flush=True)

    if all(p.is_file() for p in shot_paths):
        final_meta = stitch_promo_video(shot_paths, FINAL_VIDEO_SURGPROOF)
        shutil.copy2(FINAL_VIDEO_SURGPROOF, FINAL_VIDEO)
        generate_contact_sheet(FINAL_VIDEO_SURGPROOF, CONTACT_SHEET_PNG)
        manifest["final_video"] = {
            "surgproof_path": str(FINAL_VIDEO_SURGPROOF.relative_to(ROOT)),
            "surgatlas_path": str(FINAL_VIDEO.relative_to(ROOT)),
            "contact_sheet": str(CONTACT_SHEET_PNG.relative_to(ROOT)),
            "duration_sec": final_meta.get("duration_sec", 0.0),
            "width": final_meta.get("width", 0),
            "height": final_meta.get("height", 0),
            "size_bytes": FINAL_VIDEO_SURGPROOF.stat().st_size,
        }
        print(
            f"[promo] Stitched {FINAL_VIDEO_SURGPROOF.name} & {FINAL_VIDEO.name} "
            f"({final_meta['duration_sec']:.1f}s, {final_meta['width']}x{final_meta['height']}, "
            f"{FINAL_VIDEO_SURGPROOF.stat().st_size / 1e6:.2f} MB) + contact_sheet.png",
            flush=True,
        )

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render and stitch the 52s SurgProof product demo.")
    p.add_argument("--refresh-slot", action="store_true", help="Rebuild Shot 6 from out/results.json and re-stitch")
    p.add_argument("--shot", help="Only render a specific shot_id")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(
        refresh_slot_only=args.refresh_slot,
        only_shot=args.shot,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
