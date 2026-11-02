"""Stage 8: render annotated.mp4 at source fps, one output frame per decoded source frame.

House style (tools/render_dense.py CLEAN mode): phase header, clean boxes with label chips, optional translucent
masks, small footer. Every box, mask, phase and subtitle drawn here comes from the job's files; nothing is typed in.
Unreviewed (AMBER) labels are drawn in amber and marked "unsure" so they cannot read as accepted, and the footer
says whether each frame is an AI keyframe or propagated.

Before release the video carries a banner saying it is an unreviewed AI draft. export.release re-renders it from the
released dataset, so it stops showing what the human rejected and marks what the human corrected or reviewed.
Audio and container metadata are never copied from the source.

Inputs: meta.json, tracks.jsonl, masks/, tracks/, phase.jsonl, narration.json, the source video, released_coco.json.
Outputs: annotated.mp4, narration.vtt (the same cues, for the Watch page).
"""
from __future__ import annotations

import bisect
import collections
import fractions
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .export import clear_stale_final_pass
from .masks import AMBER, FRONT_TO_BACK, atomic_write, mask_path, progress_fn, read_json, read_jsonl, track_path
from .propagate import find_video

COLOUR = {"gallbladder": (64, 200, 140), "liver": (214, 120, 92), "grasper": (90, 170, 255), "l_hook": (250, 190, 70),
          "cystic_duct": (190, 130, 240), "blood": (235, 70, 80)}   # RGB
AMBER_RGB = (245, 158, 11)
NAME = {"gallbladder": "Gallbladder", "liver": "Liver", "grasper": "Grasper", "l_hook": "L-hook",
        "cystic_duct": "Cystic duct", "blood": "Blood"}
CANVAS_BG = (14, 11, 10)        # BGR of #0A0B0E
BAND = (20, 16, 14)
TEXT = (244, 245, 248)
MUTED = (138, 144, 158)
FONTS = ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"]
MASK_ALPHA_FIXED = 72      # out of 256, about 28%
FFMPEG_TIMEOUT = 300    # seconds for ffprobe, and for ffmpeg to finish once it has all the frames
SUBTITLE_MAX_LINE_CHARS = 72


def _font(size):
    for f in FONTS:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


class Sprites:
    """Text rendered once with PIL and cached as premultiplied BGR plus alpha, so frames stay fast."""

    def __init__(self):
        self.cache = {}
        self.fonts = {}

    def text(self, s, size, colour=TEXT):
        key = (s, size, colour)
        if key not in self.cache:
            font = self.fonts.setdefault(size, _font(size))
            x0, y0, x1, y1 = font.getbbox(s)
            w, h = max(1, int(x1 - x0) + 2), max(1, int(y1 - y0) + 2)
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((1 - x0, 1 - y0), s, font=font, fill=255)
            a = np.asarray(im, dtype=np.float32) / 255.0
            self.cache[key] = (a, np.array(colour[::-1], dtype=np.float32))
        return self.cache[key]


def blit(canvas, sprite, x, y):
    """Alpha-blend a text sprite onto the canvas, clipped to its bounds."""
    a, col = sprite
    h, w = a.shape
    x, y = int(x), int(y)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)
    if x1 <= x0 or y1 <= y0:
        return
    al = a[y0 - y:y1 - y, x0 - x:x1 - x, None]
    roi = canvas[y0:y1, x0:x1]
    roi[:] = (roi * (1 - al) + col * al).astype(np.uint8)


def prettify_phase(phase):
    """CalotTriangleDissection -> Calot triangle dissection."""
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(phase)).replace("_", " ").split()
    return " ".join([words[0]] + [w.lower() for w in words[1:]]) if words else ""


def status_text(released):
    """(banner, footer note). The only claims the video makes about its own review state."""
    if released:
        return "AFTER HUMAN FINAL PASS", "AI annotation after human final pass (sample audit, masks are draft)"
    return "AI DRAFT · UNREVIEWED", "AI draft, unreviewed, pending human final pass"


def fmt_time(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


# ---- subtitles ----

def split_cue_text(text):
    """Sentence-sized pieces, each of which wraps to at most two subtitle lines, so no text is ever cut."""
    pieces = []
    for sentence in re.split(r"(?<=[.!?])\s+", str(text).strip()):
        lines = wrap_line(sentence)
        pieces.extend(" ".join(lines[i:i + 2]) for i in range(0, len(lines), 2))
    return pieces


def wrap_line(text, limit=SUBTITLE_MAX_LINE_CHARS):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > limit:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def build_cues(narration):
    """[(t0, t1, text)] with each segment's time shared across its pieces by length."""
    cues = []
    for seg in (narration or {}).get("segments", []):
        t0, t1 = float(seg["t0"]), float(seg["t1"])
        pieces = split_cue_text(seg.get("text", ""))
        total = sum(len(p) for p in pieces)
        if not pieces or t1 <= t0:
            continue
        t = t0
        for p in pieces:
            dur = (t1 - t0) * len(p) / total
            cues.append((round(t, 3), round(t + dur, 3), p))
            t += dur
    return cues


def _vtt_time(t):
    h, rem = divmod(float(t), 3600)
    m, s = divmod(rem, 60)
    return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"


def write_vtt(path, cues):
    body = "WEBVTT\n\n" + "".join(f"{_vtt_time(a)} --> {_vtt_time(b)}\n{txt}\n\n" for a, b, txt in cues)
    atomic_write(path, body)


# ---- layout ----

def canvas_size(src_w, src_h):
    return (1920, 1080) if src_h >= 900 or src_w >= 1600 else (1280, 720)


class Layout:
    def __init__(self, src_w, src_h):
        self.W, self.H = canvas_size(src_w, src_h)
        self.band = int(self.H * 0.065)
        avail_h = self.H - 2 * self.band
        scale = min(self.W / src_w, avail_h / src_h)
        self.fw, self.fh = int(src_w * scale) // 2 * 2, int(src_h * scale) // 2 * 2
        self.ox, self.oy = (self.W - self.fw) // 2, self.band + (avail_h - self.fh) // 2
        self.sx, self.sy = self.fw / src_w, self.fh / src_h
        unit = self.H / 720
        self.size_header = int(22 * unit)
        self.size_chip = int(16 * unit)
        self.size_foot = int(15 * unit)
        self.size_sub = int(22 * unit)
        self.line = max(2, int(2 * unit))
        self.background = np.full((self.H, self.W, 3), CANVAS_BG, np.uint8)


def draw_mask(view, mask, colour, line):
    """Translucent fill and an anti-aliased outline. `mask` is already at view size."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bgr = np.array(colour[::-1], dtype=np.uint16)
    # Fixed-point blend on the mask's own bounding region: this runs for every label on every frame.
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    roi, sel = view[y:y + h, x:x + w], mask[y:y + h, x:x + w]
    keep, add = 256 - MASK_ALPHA_FIXED, bgr * MASK_ALPHA_FIXED
    roi[sel] = ((roi[sel].astype(np.uint16) * keep + add) >> 8).astype(np.uint8)
    cv2.drawContours(view, contours, -1, tuple(int(c) for c in bgr), line, cv2.LINE_AA)


def draw_chip(canvas, sprites, text, x, y, colour, size, placed):
    sprite = sprites.text(text, size)
    th, tw = sprite[0].shape
    pad_x, pad_y = int(size * 0.55), int(size * 0.28)
    w, h = tw + 2 * pad_x, th + 2 * pad_y
    x = int(min(max(0, x), canvas.shape[1] - w))
    y = int(max(0, y))
    # Step down past any chip already placed here, so overlapping objects stay readable.
    for _ in range(8):
        clash = next((p for p in placed if x < p[0] + p[2] and p[0] < x + w and y < p[1] + p[3] and p[1] < y + h), None)
        if clash is None:
            break
        y = clash[1] + clash[3] + 2
    placed.append((x, y, w, h))
    bgr = tuple(int(c) for c in colour[::-1])
    cv2.rectangle(canvas, (x, y), (x + w, y + h), BAND, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), bgr, 1, cv2.LINE_AA)
    blit(canvas, sprite, x + pad_x, y + pad_y)


def draw_subtitle(canvas, sprites, text, lay):
    lines = wrap_line(text)
    sprs = [sprites.text(line, lay.size_sub) for line in lines]
    lh = int(lay.size_sub * 1.35)
    w = max(s[0].shape[1] for s in sprs) + 2 * lay.size_sub
    h = lh * len(sprs) + lay.size_sub // 2
    x = (lay.W - w) // 2
    y = lay.oy + lay.fh - h - lay.size_sub
    roi = canvas[y:y + h, x:x + w]
    roi[:] = (roi * 0.28).astype(np.uint8)
    for i, s in enumerate(sprs):
        blit(canvas, s, (lay.W - s[0].shape[1]) // 2, y + lay.size_sub // 4 + i * lh)


# ---- frame drawing ----

def rows_from_release(coco, tracks):
    """{frame_index: [row]} holding only what the released dataset contains.

    AI annotations keep their track row, for the smoothed box and the mask file. A human correction carries its polygon."""
    by_key = {(t["frame_index"], t["label"]): t for t in tracks}
    names = {c["id"]: c["name"] for c in coco["categories"]}
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    out = collections.defaultdict(list)
    for a in coco["annotations"]:
        at, fi, label = a["attributes"], frames[a["image_id"]], names[a["category_id"]]
        human = at.get("source") == "human_correction"
        track = by_key.get((fi, label))
        x, y, w, h = a["bbox"]
        row = {"frame_index": fi, "label": label, "status": at["status"], "propagated": at.get("propagated", False),
               "keyframe": at.get("keyframe"), "box": track["box"] if track and not human else [x, y, x + w - 1, y + h - 1]}
        row["review"] = "human" if human else ("reviewed" if at.get("human_reviewed") else None)
        if human or track is None:
            row["segmentation"] = a["segmentation"]
        out[fi].append(row)
    return out


class Annotations:
    """Per-frame lookup over tracks.jsonl, or over the released dataset once there is one, and phases."""

    def __init__(self, job, meta, released=None):
        self.job = Path(job)
        tracks = read_jsonl(self.job / "tracks.jsonl")
        if released is not None:
            self.by_frame = rows_from_release(released, tracks)
        else:
            self.by_frame = collections.defaultdict(list)
            for r in tracks:
                self.by_frame[r["frame_index"]].append(r)
        kf_index = {k["kf"]: k["frame_index"] for k in meta["keyframes"]}
        phases = []
        for r in read_jsonl(self.job / "phase.jsonl"):
            if r.get("phase") and r["kf"] in kf_index:
                phases.append((kf_index[r["kf"]], r["phase"], r.get("conf")))
        phases.sort(key=lambda p: p[0])
        self.phases = phases
        self.phase_idx = [p[0] for p in phases]

    def phase_at(self, frame_index):
        """The nearest keyframe's phase: phases are only ever measured on keyframes."""
        if not self.phases:
            return None
        i = bisect.bisect_left(self.phase_idx, frame_index)
        near = [j for j in (i - 1, i) if 0 <= j < len(self.phases)]
        return self.phases[min(near, key=lambda j: abs(self.phase_idx[j] - frame_index))]

    def mask_file(self, row):
        if row["propagated"]:
            return track_path(self.job, row["frame_index"], row["label"])
        return mask_path(self.job, row["keyframe"], row["label"])


def _view_mask(row, ann, lay):
    """The row's mask at view size, or None. A human correction is rasterised from its own polygon."""
    if row.get("segmentation"):
        m = np.zeros((lay.fh, lay.fw), np.uint8)
        for poly in row["segmentation"]:
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2) * (lay.sx, lay.sy)
            cv2.fillPoly(m, [pts.astype(np.int32)], 1)
        return m > 0
    m = cv2.imread(str(ann.mask_file(row)), cv2.IMREAD_GRAYSCALE)
    return None if m is None else cv2.resize(m, (lay.fw, lay.fh), interpolation=cv2.INTER_LINEAR) > 127


def draw_frame(frame, rows, phase, subtitle, frame_index, t, ann, lay, sprites, show_masks, footer_note, banner=None):
    canvas = lay.background.copy()
    view = cv2.resize(frame, (lay.fw, lay.fh), interpolation=cv2.INTER_AREA if lay.sx < 1 else cv2.INTER_LINEAR)
    order = {lab: i for i, lab in enumerate(FRONT_TO_BACK)}
    # Back to front, so instruments draw over anatomy.
    rows = sorted(rows, key=lambda r: -order.get(r["label"], 99))

    def colour_of(r):
        return AMBER_RGB if r["status"] == AMBER else COLOUR.get(r["label"], TEXT)

    if show_masks:
        for r in rows:
            m = _view_mask(r, ann, lay)
            if m is not None and m.any():
                draw_mask(view, m, colour_of(r), lay.line)
    canvas[lay.oy:lay.oy + lay.fh, lay.ox:lay.ox + lay.fw] = view

    placed = []
    for r in rows:
        bgr = tuple(int(c) for c in colour_of(r)[::-1])
        x0, y0, x1, y1 = r["box"]
        p0 = (int(round(lay.ox + x0 * lay.sx)), int(round(lay.oy + y0 * lay.sy)))
        p1 = (int(round(lay.ox + x1 * lay.sx)), int(round(lay.oy + y1 * lay.sy)))
        cv2.rectangle(canvas, p0, p1, bgr, lay.line, cv2.LINE_AA)
        suffix = " · unsure" if r["status"] == AMBER else {"human": " · human", "reviewed": " · reviewed"}.get(r.get("review"), "")
        chip = NAME.get(r["label"], r["label"]) + suffix
        chip_h = int(lay.size_chip * 1.9)
        y = p0[1] - chip_h - 2 if p0[1] - chip_h - 2 >= lay.oy else p0[1] + 2
        draw_chip(canvas, sprites, chip, p0[0], y, colour_of(r), lay.size_chip, placed)

    pad = int(lay.W * 0.02)
    if phase is not None:
        txt = prettify_phase(phase[1]) + (f"  ({phase[2]:.2f})" if isinstance(phase[2], (int, float)) else "")
        hdr = sprites.text("Surgical phase", lay.size_foot, MUTED)
        blit(canvas, hdr, pad, (lay.band - hdr[0].shape[0]) // 2)
        s = sprites.text(txt, lay.size_header)
        blit(canvas, s, pad + hdr[0].shape[1] + pad // 2, (lay.band - s[0].shape[0]) // 2)

    if banner:
        s = sprites.text(banner, lay.size_foot)
        bw, bh = s[0].shape[1] + lay.size_foot, s[0].shape[0] + lay.size_foot // 2
        bx, by = lay.W - pad - bw, (lay.band - bh) // 2
        col = tuple(int(c) for c in AMBER_RGB[::-1]) if "DRAFT" in banner else tuple(int(c) for c in MUTED[::-1])
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), col, 1, cv2.LINE_AA)
        blit(canvas, s, bx + lay.size_foot // 2, by + lay.size_foot // 4)

    if subtitle:
        draw_subtitle(canvas, sprites, subtitle, lay)

    foot_y = lay.H - lay.band + (lay.band - int(lay.size_foot * 1.2)) // 2
    brand = sprites.text("SurgProof", lay.size_foot)
    blit(canvas, brand, pad, foot_y)
    blit(canvas, sprites.text(footer_note, lay.size_foot, MUTED), pad + brand[0].shape[1] + pad // 2, foot_y)
    if rows:
        kind = "AI keyframe" if any(not r["propagated"] for r in rows) else "propagated from AI keyframes"
    else:
        kind = "no annotation"
    right = sprites.text(f"{kind} · {fmt_time(t)}", lay.size_foot, MUTED)
    blit(canvas, right, lay.W - pad - right[0].shape[1], foot_y)
    return canvas


def _fps_arg(fps):
    return str(fractions.Fraction(float(fps)).limit_denominator(1001))


def count_video_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True, timeout=FFMPEG_TIMEOUT).stdout
    return int(out.strip().strip(",").split(",")[0])


def run(job_dir, video=None, show_masks=True, subtitles=True, on_progress=None, crf=20, released=None, output=None):
    """Run stage 8. Returns {"path", "n_frames", "n_cues"}. Raises if the output does not match the source.

    `released` is a released COCO dict to draw instead of the draft tracks; export.release passes it with `output`, a
    temporary path, so a failed render leaves the job as it was. Without it an existing released_coco.json is used."""
    job = Path(job_dir)
    progress = progress_fn(job, "render", on_progress)
    clear_stale_final_pass(job)     # render runs before export, and the video reports the review state
    if released is None and (job / "released_coco.json").exists():
        released = read_json(job / "released_coco.json")
    meta = read_json(job / "meta.json")
    video_path = find_video(job, meta, video)
    fps = float(meta["fps"])
    n_total = int(meta.get("n_frames") or 0)

    narration = read_json(job / "narration.json") if (job / "narration.json").exists() else None
    cues = build_cues(narration)
    write_vtt(job / "narration.vtt", cues)
    cue_starts = [c[0] for c in cues]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"could not open video {video_path}")
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise OSError(f"could not decode any frame from {video_path}")

    src_h, src_w = frame.shape[:2]
    lay = Layout(src_w, src_h)
    ann = Annotations(job, meta, released)
    sprites = Sprites()
    banner, footer_note = status_text(released is not None)

    final_out = Path(output) if output else job / "annotated.mp4"
    tmp_out = job / ".annotated.tmp.mp4"
    with tempfile.TemporaryFile() as ffmpeg_err:
        proc = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{lay.W}x{lay.H}",
             "-r", _fps_arg(fps), "-protocol_whitelist", "file,pipe", "-i", "pipe:0", "-map", "0:v:0", "-an", "-map_metadata", "-1",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp_out)],
            stdin=subprocess.PIPE, stderr=ffmpeg_err)
        n = 0
        try:
            while ok:
                t = n / fps
                subtitle = None
                if subtitles and cues:
                    i = bisect.bisect_right(cue_starts, t) - 1
                    if i >= 0 and t < cues[i][1]:
                        subtitle = cues[i][2]
                out = draw_frame(frame, ann.by_frame.get(n, []), ann.phase_at(n), subtitle, n, t, ann, lay, sprites,
                                 show_masks, footer_note, banner)
                proc.stdin.write(out.tobytes())
                n += 1
                if n % 50 == 0:
                    progress(min(0.99, n / max(1, n_total)), f"rendered {n}/{n_total} frames")
                ok, frame = cap.read()
            proc.stdin.close()
            code = proc.wait(timeout=FFMPEG_TIMEOUT)
        except BrokenPipeError:
            code = proc.wait(timeout=FFMPEG_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            code = proc.wait()
        finally:
            cap.release()
        if code != 0:
            ffmpeg_err.seek(0)
            detail = ffmpeg_err.read().decode(errors="replace")[-500:]
            tmp_out.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed ({code}): {detail}")

    # Verify the artifact rather than trusting the exit code.
    written = count_video_frames(tmp_out)
    if written != n:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"annotated video has {written} frames but {n} were rendered")
    tmp_out.replace(final_out)
    progress(1.0, f"{final_out.name}: {n} frames at {fps:g} fps, {len(cues)} subtitle cues")
    return {"path": str(final_out), "n_frames": n, "n_cues": len(cues)}
