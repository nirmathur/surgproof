"""The SurgProof pipeline: video in, verified annotations out. run() is the only entry point.

Stages 1-4 and 7 live here. Stages 5, 6, 8 and 9 are surgproof.masks, propagate, render and export.
Each stage writes to the job folder and updates status.json. A model failure is never recorded as
"absent": it becomes AMBER, and the job stops if failures dominate.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import subprocess
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from datetime import datetime, timezone
from pathlib import Path

import cv2

from surgproof import prompts as P
from surgproof import vlm

STAGES = ["ingest", "detect", "verify", "amber", "masks", "propagate", "narrate", "render", "export"]
WEIGHTS = {"ingest": 3, "detect": 35, "verify": 15, "amber": 10, "masks": 8, "propagate": 10, "narrate": 7, "render": 10, "export": 2}
EXTERNAL_STAGES = {"masks": "surgproof.masks", "propagate": "surgproof.propagate", "render": "surgproof.render", "export": "surgproof.export"}

MODEL_VIEW_MAX_SIDE = 1024  # the model sees at most this; its coordinates are scaled back to full frame
MAX_MINUTES = 5.0
ACCEPT_CONF = 0.6  # both passes must be at least this sure to settle a label without the amber loop
MIN_BOX_IOU = 0.1  # both passes say present but point at different places: that is a dispute
OUT_OF_FRAME_TOLERANCE = 0.1  # fraction of the frame a coordinate may overshoot before it is rejected
AMBER_ROUNDS = 2
RESOLVE_CONF = 0.7  # the amber loop settles a label only at this confidence
AMBER_PAD = {1: 0.35, 2: 0.15}
AMBER_MIN_SIDE = 768  # crops are magnified to at least this long side
MAX_FAILED_FRACTION = 0.5
PHASE_WINDOW = 5
PHASE_MIN_RUN = 3  # keyframes: a shorter run of one phase is flicker from judging single frames
NARRATE_MAX_SECONDS = 45.0
PHASE_UNSURE_CONF = 0.6

MENTIONS = {
    "gallbladder": r"\bgall\s?bladders?\b",
    "liver": r"\blivers?\b",
    "grasper": r"\bgraspers?\b",
    "l_hook": r"\bl[- ]?hooks?\b|\bhooks?\b",
    "cystic_duct": r"\bcystic ducts?\b|\bducts?\b",
    "blood": r"\bblood\b|\bbleed\w*|\bhaemorrhag\w*|\bhemorrhag\w*",
}
# Narration grammar. After labels become @tokens, a clause must match one whole shape below or the sentence is
# dropped. Word lists and negation tests kept failing here: "No L-hook, blood is visible" uses only allowed words
# and contains a negation, yet claims blood. Shapes also fix which list of verified facts a clause is checked against.
_ITEM = r"(?:(?:the|a|an)\s+)?@(?:gallbladder|liver|grasper|l_hook|cystic_duct|blood)"
_LIST = rf"{_ITEM}(?:(?:\s*,\s*{_ITEM})*\s*,?\s+(?:and|or)\s+{_ITEM})?"  # a list of two or more needs its and/or
_BE = r"(?:is|are)"
_SEEN = r"(?:visible|seen|present|shown|in\s+view)"
_PLACE = r"(?:\s+(?:in|throughout|across|within)\s+(?:(?:this|these|all|each|every|the)\s+)*(?:segments?|frames?|keyframes?|view))?"
_NEAR = rf"(?:\s+near\s+{_ITEM})?"
_PHASE = r"(?:(?:the|a)\s+)?@phase(?:\s+phase)?"
CLAUSE_SHAPES = [(kind, re.compile(pattern)) for kind, pattern in [
    ("visible", rf"{_LIST}\s+(?:{_BE}\s+(?:also\s+|still\s+)?|remains?\s+|appears?\s+)?{_SEEN}{_NEAR}{_PLACE}"),
    ("visible", rf"there\s+{_BE}\s+{_LIST}{_PLACE}"),
    ("not_visible", rf"no\s+{_LIST}\s+{_BE}\s+{_SEEN}{_PLACE}"),
    ("not_visible", rf"{_LIST}\s+{_BE}\s+not\s+{_SEEN}{_PLACE}"),
    ("not_visible", rf"there\s+{_BE}\s+no\s+{_LIST}{_PLACE}"),
    ("not_visible", rf"neither\s+{_ITEM}\s+nor\s+{_ITEM}\s+{_BE}\s+{_SEEN}{_PLACE}"),
    ("unsure", rf"{_LIST}\s+(?:could\s+not|cannot)\s+be\s+confirmed{_PLACE}"),
    ("unsure", rf"{_LIST}\s+{_BE}\s+(?:uncertain|unsure|unclear|unconfirmed|not\s+confirmed){_PLACE}"),
    ("unsure", rf"{_LIST}\s+may\s+be\s+{_SEEN}{_PLACE}"),
    ("phase_uncertain", r"the\s+phase\s+is\s+(?:uncertain|unsure|unclear|unconfirmed)"),
    ("phase_named", rf"(?:this\s+segment|the\s+phase|it)\s+is\s+(?:labell?ed\s+(?:as\s+)?)?{_PHASE}"),
    ("phase_named", rf"{_PHASE}\s+(?:continues|is\s+ongoing|is\s+underway)"),
    ("checks_disagree", r"(?:the\s+)?phase\s+checks\s+(?:disagree|disagreed)"),
    ("low_confidence", r"(?:the\s+)?(?:phase\s+)?confidence\s+is\s+low"),
]]
UNIVERSAL_PLACE = re.compile(r"\b(?:all|every|each|throughout|across|these|segments?|frames|keyframes)\b")
CLAUSE_BREAK = re.compile(r"[;.]|,?\s*\b(?:but|while|whereas|although|though|yet)\b", re.I)


class PipelineError(RuntimeError):
    pass


# ---- job folder helpers ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path, obj):
    tmp = f"{path}.tmp.{threading.get_ident()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def _write_jsonl(path, rows):
    tmp = f"{path}.tmp.{threading.get_ident()}"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# Everything a pipeline run or a human review leaves in a job folder. Named explicitly, so that a mistaken
# --out can never empty a folder of the user's own files. upload.* and job.json belong to the web lane and stay.
OUTPUT_FILES = (
    "status.json", "meta.json", "detect.jsonl", "phase.jsonl", "verify.jsonl", "decisions.jsonl", "masks.jsonl",
    "tracks.jsonl", "narration.json", "narration.vtt", "annotated.mp4", "dataset_coco.json", "provenance.jsonl",
    "final_pass.jsonl", "final_pass_queue.json", "released_coco.json", "source.mp4", "partial_source.mp4",
)
OUTPUT_DIRS = ("frames", "frames_view", "amber", "masks", "tracks")


def clear_outputs(job, keep=None):
    """A new run must not inherit masks, a dataset or a human sign-off from an earlier one."""
    keep = Path(keep).resolve() if keep else None
    for name in OUTPUT_FILES:
        path = job / name
        if path.is_file() and path.resolve() != keep:
            path.unlink()
    for name in OUTPUT_DIRS:
        if (job / name).is_dir():
            shutil.rmtree(job / name)


class Status:
    """status.json. progress is overall (0-1). stage_progress is within the current stage."""

    def __init__(self, job_dir, on_status=None, meter=None):
        self.path = Path(job_dir) / "status.json"
        self.meter = meter if meter is not None else {"cost_usd": 0.0}
        self.on_status = on_status
        self.lock = threading.Lock()
        self.started = _now()
        self.state = {name: "pending" for name in STAGES}
        self.data = {"stage": "ingest", "progress": 0.0, "message": "starting", "started": self.started,
                     "updated": self.started, "error": None, "stage_index": 0, "n_stages": len(STAGES),
                     "stage_progress": 0.0, "stages": self._stages(), "cost_usd": 0.0}
        self._flush()

    def _stages(self):
        return [{"name": name, "state": self.state[name]} for name in STAGES]

    def _flush(self):
        self.data["updated"] = _now()
        self.data["cost_usd"] = round(self.meter["cost_usd"], 4)
        _write_json(self.path, self.data)
        if self.on_status:
            self.on_status(dict(self.data))

    def begin(self, stage, message=""):
        with self.lock:
            self.state[stage] = "running"
            self.data.update(stage=stage, stage_index=STAGES.index(stage), stage_progress=0.0,
                             message=message or stage, stages=self._stages(), progress=self._overall(stage, 0.0))
            self._flush()

    def update(self, stage, fraction, message):
        with self.lock:
            fraction = max(0.0, min(1.0, float(fraction)))
            self.data.update(stage_progress=fraction, message=message, progress=self._overall(stage, fraction))
            self._flush()

    def end(self, stage, message=""):
        with self.lock:
            self.state[stage] = "done"
            self.data.update(stage_progress=1.0, stages=self._stages(), progress=self._overall(stage, 1.0))
            if message:
                self.data["message"] = message
            self._flush()

    def _overall(self, stage, fraction):
        done = sum(WEIGHTS[s] for s in STAGES[:STAGES.index(stage)])
        return round((done + WEIGHTS[stage] * fraction) / sum(WEIGHTS.values()), 4)

    def fail(self, stage, error):
        with self.lock:
            self.state[stage] = "failed"
            self.data.update(error=error, message=f"failed in {stage}", stages=self._stages())
            self._flush()

    def finish(self, message):
        with self.lock:
            self.data.update(stage="done", progress=1.0, stage_progress=1.0, message=message,
                             stage_index=len(STAGES), stages=self._stages())
            self._flush()


# ---- geometry -------------------------------------------------------------------------------------

def _inside(x, y, width, height):
    """Near the frame is fine, models overshoot edges. Far outside means a wrong coordinate space, and
    clamping that to the frame edge would invent a huge mask."""
    margin_x, margin_y = width * OUT_OF_FRAME_TOLERANCE, height * OUT_OF_FRAME_TOLERANCE
    return -margin_x <= x <= width + margin_x and -margin_y <= y <= height + margin_y


def clean_polygon(polygon, scale, width, height):
    """Scale model coordinates to full-frame pixels and clamp. None if it is not a real outline."""
    if not polygon:
        return None
    pts = []
    for x, y in polygon:
        if not _inside(x * scale, y * scale, width, height):
            return None
        p = [round(min(max(x * scale, 0), width - 1), 1), round(min(max(y * scale, 0), height - 1), 1)]
        if not pts or pts[-1] != p:
            pts.append(p)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts.pop()
    if len(pts) < 3:
        return None
    area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
    return pts if area >= 4 else None


def clean_box(box, scale, width, height):
    if not box:
        return None
    x0, y0, x1, y1 = box
    if not (_inside(x0 * scale, y0 * scale, width, height) and _inside(x1 * scale, y1 * scale, width, height)):
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    out = [round(min(max(x0 * scale, 0), width - 1), 1), round(min(max(y0 * scale, 0), height - 1), 1),
           round(min(max(x1 * scale, 0), width), 1), round(min(max(y1 * scale, 0), height), 1)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def polygon_box(polygon):
    xs, ys = zip(*polygon)
    return [min(xs), min(ys), max(xs), max(ys)]


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def crop_region(boxes, pad, width, height):
    """The disputed region: the union of the boxes plus padding. The whole frame if nothing was located."""
    boxes = [b for b in boxes if b]
    if not boxes:
        return [0, 0, width, height]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    px, py = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0, x1, y1 = x0 - px, y0 - py, x1 + px, y1 + py
    min_w, min_h = width * 0.25, height * 0.25  # thin instruments still need surroundings to be judged
    if x1 - x0 < min_w:
        cx = (x0 + x1) / 2
        x0, x1 = cx - min_w / 2, cx + min_w / 2
    if y1 - y0 < min_h:
        cy = (y0 + y1) / 2
        y0, y1 = cy - min_h / 2, cy + min_h / 2
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(width, int(round(x1))), min(height, int(round(y1)))
    return [x0, y0, x1, y1]


# ---- stage 1: ingest ------------------------------------------------------------------------------

def _probe(video):
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=codec_name,pix_fmt,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration,format_name",
             "-of", "json", str(video)],
            capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"could not read this video: {e.stderr.strip()[:300]}") from e
    info = json.loads(out)
    if not info.get("streams"):
        raise PipelineError("this file has no video stream")
    return info["streams"][0], info.get("format", {})


def _rate(text):
    num, _, den = str(text).partition("/")
    try:
        return float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def _make_playable(video, dest, stream, fmt, fps):
    """Write source.mp4: H.264, constant frame rate, so frame index and time stay in step and any browser
    plays it. Streams are copied when the file already qualifies."""
    variable = abs(_rate(stream.get("r_frame_rate")) - fps) > 0.01 * max(fps, 1)
    h264 = stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p"
    base = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?"]
    partial = dest.with_name(f"partial_{dest.name}")  # so the input can safely be dest itself
    tail = ["-movflags", "+faststart", str(partial)]
    if h264 and not variable:
        copy = subprocess.run(base + ["-c", "copy"] + tail, capture_output=True, text=True)
        if copy.returncode == 0:
            os.replace(partial, dest)
            return
    encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
              "-vf", f"fps={fps},pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:a", "aac"]
    try:
        subprocess.run(base + encode + tail, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise PipelineError(f"could not convert this video: {e.stderr.strip()[:300]}") from e
    os.replace(partial, dest)


def ingest(video, job, status, kf_fps=1.0, max_minutes=MAX_MINUTES):
    video = Path(video)
    if not video.is_file():
        raise PipelineError(f"no such video: {video}")
    if kf_fps <= 0:
        raise PipelineError("kf_fps must be above 0")
    stream, fmt = _probe(video)
    fps = _rate(stream.get("avg_frame_rate"))
    if fps <= 0:
        raise PipelineError("could not determine the frame rate of this video")
    duration = float(fmt.get("duration") or 0) or int(stream.get("nb_read_packets") or 0) / fps
    if max_minutes and duration > max_minutes * 60:
        raise PipelineError(f"video is {duration / 60:.1f} minutes long, the limit is {max_minutes:g}")

    status.update("ingest", 0.05, "preparing a playable copy")
    source = job / "source.mp4"
    _make_playable(video, source, stream, fmt, fps)

    frames_dir, view_dir = job / "frames", job / "frames_view"
    frames_dir.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise PipelineError("could not decode this video")
    expected = max(int(stream.get("nb_read_packets") or 0), 1)
    step = fps / kf_fps
    keyframes, index, width, height, scale = [], 0, 0, 0, 1.0
    next_target = 0
    while cap.grab():
        if index == next_target:
            ok, frame = cap.retrieve()
            if not ok:
                raise PipelineError(f"could not decode frame {index}")
            height, width = frame.shape[:2]
            name = f"kf_{len(keyframes):05d}"
            cv2.imwrite(str(frames_dir / f"{name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            long_side = max(width, height)
            if long_side > MODEL_VIEW_MAX_SIDE:
                view_dir.mkdir(exist_ok=True)
                f = MODEL_VIEW_MAX_SIDE / long_side
                view = cv2.resize(frame, (round(width * f), round(height * f)), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(view_dir / f"{name}.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 92])
                scale = long_side / MODEL_VIEW_MAX_SIDE
            keyframes.append({"kf": name, "frame_index": index, "t": round(index / fps, 3)})
            next_target = round(len(keyframes) * step)
            if next_target <= index:
                next_target = index + 1
            status.update("ingest", 0.1 + 0.9 * min(index / expected, 1.0), f"extracted {len(keyframes)} keyframes")
        index += 1
    cap.release()
    if not keyframes:
        raise PipelineError("no frames could be decoded from this video")
    meta = {
        "fps": fps, "width": width, "height": height, "n_frames": index, "kf_fps": kf_fps,
        "keyframes": keyframes, "duration": round(index / fps, 3), "source": source.name, "source_video": source.name,
        "original_name": video.name, "view_scale": scale, "vlm_provider": vlm.provider(),
    }
    _write_json(job / "meta.json", meta)
    return meta


def _view_path(job, kf):
    view = job / "frames_view" / f"{kf}.jpg"
    return view if view.exists() else job / "frames" / f"{kf}.jpg"


# ---- shared helper: run one model call per item, streaming finished rows --------------------------

def _fan_out(items, worker, stage, status, label):
    """Call worker(item) concurrently. Collects (item, result, error). Stops the job on auth failure."""
    results = []
    if not items:
        return results
    done = 0
    with ThreadPoolExecutor(max_workers=vlm.concurrency()) as pool:
        futures = {pool.submit(copy_context().run, worker, item): item for item in items}
        try:
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    results.append((item, fut.result(), None))
                except vlm.VlmAuthError:
                    raise
                except vlm.VlmError as e:
                    results.append((item, None, str(e)))
                done += 1
                status.update(stage, done / len(items), f"{label} {done}/{len(items)}")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    return results


def _guard_failures(results, stage):
    failed = [e for _, _, e in results if e]
    if results and len(failed) / len(results) > MAX_FAILED_FRACTION:
        raise PipelineError(f"{stage}: the model failed on {len(failed)} of {len(results)} calls, stopping. First error: {failed[0]}")


# ---- stage 2: detect ------------------------------------------------------------------------------

def _view_size(meta):
    return round(meta["width"] / meta["view_scale"]), round(meta["height"] / meta["view_scale"])


def detect(job, meta, status):
    w, h = _view_size(meta)
    prompt, model = P.detect_prompt(w, h), vlm.SONNET
    tag, ph = vlm.model_tag(model), vlm.prompt_hash(prompt)
    path, phase_path = job / "detect.jsonl", job / "phase.jsonl"
    _write_jsonl(path, [])
    _write_jsonl(phase_path, [])
    lock = threading.Lock()
    rows, phases = [], []

    def worker(kf):
        image = _view_path(job, kf["kf"])
        reply = vlm.ask([image], prompt, P.DETECT_SCHEMA, model)
        raw_ref = vlm.cache_key([image], prompt, model)
        new = []
        for label in P.LABELS:
            o = reply["objects"][label]
            poly = clean_polygon(o["polygon"], meta["view_scale"], meta["width"], meta["height"]) if o["present"] else None
            new.append({"kf": kf["kf"], "label": label, "present": o["present"], "conf": o["conf"], "polygon": poly,
                        "model": tag, "prompt_hash": ph, "raw_ref": raw_ref})
        phase = {"kf": kf["kf"], "phase": reply["phase"]["name"], "conf": reply["phase"]["conf"]}
        with lock:  # streamed so the job page can show labels arriving
            rows.extend(new)
            phases.append(phase)
            with open(path, "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in new)
        return True

    results = _fan_out(meta["keyframes"], worker, "detect", status, "keyframe")
    _guard_failures(results, "detect")
    for kf, _, error in results:
        if error:
            rows.extend({"kf": kf["kf"], "label": label, "present": None, "conf": 0.0, "polygon": None, "model": tag,
                         "prompt_hash": ph, "raw_ref": None, "error": error} for label in P.LABELS)
            phases.append({"kf": kf["kf"], "phase": P.UNKNOWN_PHASE, "conf": 0.0, "error": error})
    order = {k["kf"]: i for i, k in enumerate(meta["keyframes"])}
    rows.sort(key=lambda r: (order[r["kf"]], P.LABELS.index(r["label"])))
    phases.sort(key=lambda r: order[r["kf"]])
    _write_jsonl(path, rows)
    _write_jsonl(phase_path, phases)
    return rows, phases


# ---- stage 3: verify ------------------------------------------------------------------------------

def verify(job, meta, phases, status):
    w, h = _view_size(meta)
    prompt, model = P.verify_prompt(w, h), vlm.HAIKU
    tag, ph = vlm.model_tag(model), vlm.prompt_hash(prompt)
    lock = threading.Lock()
    rows, verify_phase = [], {}

    def worker(kf):
        image = _view_path(job, kf["kf"])
        reply = vlm.ask([image], prompt, P.VERIFY_SCHEMA, model)
        raw_ref = vlm.cache_key([image], prompt, model)
        new = []
        for label in P.LABELS:
            o = reply["objects"][label]
            box = clean_box(o["box"], meta["view_scale"], meta["width"], meta["height"]) if o["present"] else None
            new.append({"kf": kf["kf"], "label": label, "present": o["present"], "conf": o["conf"], "box": box,
                        "model": tag, "prompt_hash": ph, "raw_ref": raw_ref})
        with lock:
            rows.extend(new)
            verify_phase[kf["kf"]] = reply["phase"]
        return True

    results = _fan_out(meta["keyframes"], worker, "verify", status, "keyframe")
    _guard_failures(results, "verify")
    for kf, _, error in results:
        if error:
            rows.extend({"kf": kf["kf"], "label": label, "present": None, "conf": 0.0, "box": None, "model": tag,
                         "prompt_hash": ph, "raw_ref": None, "error": error} for label in P.LABELS)
    order = {k["kf"]: i for i, k in enumerate(meta["keyframes"])}
    rows.sort(key=lambda r: (order[r["kf"]], P.LABELS.index(r["label"])))
    _write_jsonl(job / "verify.jsonl", rows)
    for row in phases:  # the second look at the phase, kept beside the first
        second = verify_phase.get(row["kf"])
        row["verify_phase"] = second["name"] if second else None
        row["verify_conf"] = second["conf"] if second else None
    stabilise_phases(phases)
    _write_jsonl(job / "phase.jsonl", phases)
    return rows


def stabilise_phases(rows, window=PHASE_WINDOW, min_run=PHASE_MIN_RUN):
    """Single-frame phase calls flicker. Set each row's phase to a confidence-weighted vote over nearby
    keyframes, then absorb very short runs, so the video header, Watch page and subtitles all show the same
    phase. The model's own answer stays in raw_phase and raw_conf. Works in place."""
    raw = [(r["phase"], r["conf"]) for r in rows]
    names = []
    for i, (name, _) in enumerate(raw):
        votes = Counter()
        for other, conf in raw[max(0, i - window // 2):i + window // 2 + 1]:
            votes[other] += max(conf, 0.01)
        best = max(votes.values())
        names.append(name if votes[name] == best else next(k for k, v in votes.items() if v == best))
    i = 0
    while i < len(names):
        j = i
        while j < len(names) and names[j] == names[i]:
            j += 1
        if j - i < min_run and (i > 0 or j < len(names)):
            names[i:j] = [names[i - 1] if i > 0 else names[j]] * (j - i)
            if i == 0:
                continue  # the first run took on its neighbour's name, so measure the run again
        i = j
    for i, (row, name) in enumerate(zip(rows, names)):
        lo, hi = max(0, i - window // 2), i + window // 2 + 1
        support = [conf for other, conf in raw[lo:hi] if other == name]
        row.update(raw_phase=raw[i][0], raw_conf=raw[i][1], phase=name,
                   conf=round(raw[i][1] if raw[i][0] == name else (sum(support) / len(support) if support else 0.0), 3))
        row["agree"] = row.get("verify_phase") == name
    return rows


# ---- first decision, before the amber loop --------------------------------------------------------

def first_decision(d, v):
    """Compare the two independent passes for one keyframe and label. Returns (status, reason)."""
    if d["present"] is None or v["present"] is None:
        return "AMBER", "a model call failed"
    if d["present"] != v["present"]:
        return "AMBER", "the two passes disagree on presence"
    if d["conf"] < ACCEPT_CONF or v["conf"] < ACCEPT_CONF:
        return "AMBER", "low confidence"
    if not d["present"]:
        return "reject", "both passes say absent"
    if not d["polygon"]:
        return "AMBER", "present but no usable outline"
    if v["box"] and box_iou(polygon_box(d["polygon"]), v["box"]) < MIN_BOX_IOU:
        return "AMBER", "the two passes point at different places"
    return "accept", "both passes agree"


# ---- stage 4: amber loop --------------------------------------------------------------------------

def _crop_for_model(job, kf, label, round_no, region):
    frame = cv2.imread(str(job / "frames" / f"{kf}.jpg"))
    x0, y0, x1, y1 = region
    crop = frame[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    mag = max(1.0, AMBER_MIN_SIDE / max(h, w))
    if mag > 1.0:
        crop = cv2.resize(crop, (round(w * mag), round(h * mag)), interpolation=cv2.INTER_CUBIC)
    folder = job / "amber"
    folder.mkdir(exist_ok=True)
    path = folder / f"{kf}_{label}_r{round_no}.jpg"
    cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path, crop.shape[1], crop.shape[0], mag


def _resolve(job, meta, row, d, v):
    """Up to AMBER_ROUNDS blind looks at a tighter crop. Returns the finished decision row."""
    boxes = [polygon_box(d["polygon"]) if d.get("polygon") else None, v.get("box")]
    model = vlm.SONNET
    for round_no in range(1, AMBER_ROUNDS + 1):
        region = crop_region(boxes, AMBER_PAD[round_no], meta["width"], meta["height"])
        crop_path, cw, ch, mag = _crop_for_model(job, row["kf"], row["label"], round_no, region)
        prompt = P.amber_prompt(row["label"], cw, ch, round_no)
        reply = vlm.ask([crop_path], prompt, P.AMBER_SCHEMA, model)
        poly = None
        if reply["present"] and reply["polygon"]:
            inner = clean_polygon(reply["polygon"], 1 / mag, region[2] - region[0], region[3] - region[1])
            if inner:
                poly = [[round(x + region[0], 1), round(y + region[1], 1)] for x, y in inner]
        row["rounds"].append({
            "round": round_no, "crop": region, "present": reply["present"], "conf": reply["conf"],
            "polygon": poly, "reason": reply["reason"], "model": vlm.model_tag(model),
            "prompt_hash": vlm.prompt_hash(prompt), "raw_ref": vlm.cache_key([crop_path], prompt, model),
        })
        if reply["conf"] >= RESOLVE_CONF:
            if not reply["present"]:
                row.update(status="reject", final_present=False, polygon=None, reason=f"amber round {round_no}: absent")
                return row
            # The latest outline any round drew, else the first pass's outline.
            outline = next((r["polygon"] for r in reversed(row["rounds"]) if r["polygon"]), d.get("polygon"))
            if outline:
                row.update(status="accept", final_present=True, polygon=outline, reason=f"amber round {round_no}: present")
                return row
        if poly:  # look closer at where this round found it
            boxes = [polygon_box(poly)]
    return row  # still AMBER: goes to the human


def decide(job, meta, detect_rows, verify_rows, status):
    d_idx = {(r["kf"], r["label"]): r for r in detect_rows}
    v_idx = {(r["kf"], r["label"]): r for r in verify_rows}
    rows = []
    for kf in meta["keyframes"]:
        for label in P.LABELS:
            key = (kf["kf"], label)
            state, reason = first_decision(d_idx[key], v_idx[key])
            row = {"kf": kf["kf"], "label": label, "status": state, "final_present": None, "rounds": [], "reason": reason, "polygon": None}
            if state == "accept":
                row.update(final_present=True, polygon=d_idx[key]["polygon"])
            elif state == "reject":
                row["final_present"] = False
            rows.append(row)
    ambers = [r for r in rows if r["status"] == "AMBER"]
    status.update("amber", 0.0, f"{len(ambers)} ambers after the second pass")

    def worker(row):
        return _resolve(job, meta, row, d_idx[(row["kf"], row["label"])], v_idx[(row["kf"], row["label"])])

    results = _fan_out(ambers, worker, "amber", status, "amber")
    _guard_failures(results, "amber")
    for row, _, error in results:
        if error:  # stays AMBER for the human: a failed call settles nothing
            row["reason"] = f"{row['reason']}; amber loop call failed: {error}"
    _write_jsonl(job / "decisions.jsonl", rows)
    return rows


# ---- stage 7: narrate -----------------------------------------------------------------------------

def phase_segments(meta, phases):
    """Runs of one phase, split so no subtitle covers more than NARRATE_MAX_SECONDS."""
    segments = []
    for i, kf in enumerate(meta["keyframes"]):
        name = phases[i]["phase"]
        if segments and segments[-1]["phase"] == name and kf["t"] - segments[-1]["t0"] < NARRATE_MAX_SECONDS:
            segments[-1]["kfs"].append(i)
        else:
            segments.append({"phase": name, "t0": kf["t"], "kfs": [i]})
    for s, nxt in zip(segments, segments[1:] + [None]):
        s["t1"] = nxt["t0"] if nxt else meta["duration"]
    return segments


def segment_facts(segment, meta, phases, decisions):
    by = {}
    for r in decisions:
        by.setdefault(r["kf"], []).append(r)
    keyframes = []
    for i in segment["kfs"]:
        kf = meta["keyframes"][i]
        rows = by.get(kf["kf"], [])
        keyframes.append({
            "kf": kf["kf"], "t": kf["t"],
            "visible": [r["label"] for r in rows if r["status"] == "accept"],
            "not_visible": [r["label"] for r in rows if r["status"] == "reject"],
            "unsure": [r["label"] for r in rows if r["status"] == "AMBER"],
        })
    chosen = [phases[i] for i in segment["kfs"]]
    conf = sum(p["conf"] for p in chosen) / len(chosen)
    agree = all(p.get("agree") for p in chosen)
    return {"phase": segment["phase"], "t0": segment["t0"], "t1": segment["t1"],
            "phase_conf": round(conf, 2), "phase_checks_agree": agree, "keyframes": keyframes}


def _phase_pattern(phase):
    words = re.findall(r"[A-Z][a-z]+", phase or "")
    if not words or phase == P.UNKNOWN_PHASE:
        return None
    return r"[\s-]*(?:and[\s-]*)?".join(w + r"(?:'s)?" for w in words)


def sentence_supported(sentence, facts):
    """True only if a sentence states presence, absence or doubt in a known shape, and each clause
    holds on every keyframe it cites. Anything else is dropped, and the fallback text covers the gap."""
    text, evidence = sentence["text"].strip(), sentence["evidence"]
    frames = {k["kf"]: k for k in facts["keyframes"]}
    if not text or not evidence or any(e not in frames for e in evidence):
        return False
    pattern = _phase_pattern(facts.get("phase"))
    if pattern:  # first, because a phase name can contain a label
        text = re.sub(pattern, " @phase ", text, flags=re.I)
    for label, label_pattern in MENTIONS.items():
        text = re.sub(label_pattern, f"@{label}", text, flags=re.I)
    clauses = [" ".join(c.split()).lower() for c in CLAUSE_BREAK.split(text)]
    clauses = [c for c in clauses if c]
    if not clauses:
        return False
    for clause in clauses:
        kind = next((k for k, shape in CLAUSE_SHAPES if shape.fullmatch(clause)), None)
        if kind is None:
            return False
        if kind == "phase_uncertain" and not phase_is_uncertain(facts):
            return False
        if kind == "phase_named" and phase_is_uncertain(facts):
            return False
        if kind == "checks_disagree" and facts["phase_checks_agree"]:
            return False
        if kind == "low_confidence" and facts["phase_conf"] >= PHASE_UNSURE_CONF:
            return False
        if kind in ("visible", "not_visible", "unsure"):
            # A claim about the whole segment has to cite the whole segment.
            if UNIVERSAL_PLACE.search(clause) and set(evidence) != set(frames):
                return False
            for label in re.findall(r"@(\w+)", clause):
                if any(label not in frames[e][kind] for e in evidence):
                    return False
    return True


def phase_is_uncertain(facts):
    return facts["phase"] == P.UNKNOWN_PHASE or not facts["phase_checks_agree"] or facts["phase_conf"] < PHASE_UNSURE_CONF


def fallback_text(facts):
    """Used when no model sentence passes the check. Built directly from the decisions."""
    def seen(key):
        return sorted({label for k in facts["keyframes"] for label in k[key]})
    parts = []
    visible, unsure, absent = seen("visible"), seen("unsure"), seen("not_visible")
    if visible:
        parts.append("Verified in this segment: " + ", ".join(visible).replace("_", " ") + ".")
    if unsure:
        parts.append("Could not be confirmed: " + ", ".join(unsure).replace("_", " ") + ".")
    if not parts and absent:
        parts.append("Checked and not seen: " + ", ".join(absent).replace("_", " ") + ".")
    return " ".join(parts) or "No checks completed for this segment."


def _segment_text(kept, facts):
    text = " ".join(s["text"].strip() for s in kept) if kept else fallback_text(facts)
    if phase_is_uncertain(facts) and not re.search(r"\bphase\b[^.;]*\b(uncertain|unsure|unclear|unconfirmed)\b", text, re.I):
        text += " The phase is uncertain."
    return text


def narrate(job, meta, phases, decisions, status):
    segments = phase_segments(meta, phases)
    model = vlm.SONNET

    def worker(seg):
        facts = segment_facts(seg, meta, phases, decisions)
        prompt = P.narrate_prompt(facts)
        reply = vlm.ask([], prompt, P.NARRATE_SCHEMA, model)
        kept = [s for s in reply["sentences"] if sentence_supported(s, facts)]
        return facts, kept, len(reply["sentences"]) - len(kept), prompt

    # No failure guard here: a segment whose call failed gets the decisions-based text, and its error is recorded.
    results = _fan_out(segments, worker, "narrate", status, "segment")
    out = []
    for seg, result, error in results:
        facts = result[0] if result else segment_facts(seg, meta, phases, decisions)
        kept = result[1] if result else []
        evidence = sorted({e for s in kept for e in s["evidence"]})
        generated = bool(kept)
        if not kept:
            evidence = [k["kf"] for k in facts["keyframes"] if k["visible"] or k["unsure"]]
        out.append({
            "t0": seg["t0"], "t1": seg["t1"], "phase": seg["phase"],
            "text": _segment_text(kept, facts),
            "evidence": evidence, "sentences": kept,
            "source": vlm.model_tag(model) if generated else "decisions",
            "dropped_sentences": result[2] if result else 0,
            "prompt_hash": vlm.prompt_hash(result[3]) if result else None,
            "raw_ref": vlm.cache_key([], result[3], model) if result else None,
            "error": error,
        })
    out.sort(key=lambda s: s["t0"])
    _write_json(job / "narration.json", {"segments": out})
    return out


# ---- stages owned by the masks lane ---------------------------------------------------------------

def _run_external(stage, job, status):
    import importlib

    try:
        module = importlib.import_module(EXTERNAL_STAGES[stage])
    except ModuleNotFoundError as e:
        if e.name == EXTERNAL_STAGES[stage]:
            raise PipelineError(f"{EXTERNAL_STAGES[stage]} is not installed in this checkout") from e
        raise
    report = lambda fraction, message="": status.update(stage, fraction, message or stage)
    params = inspect.signature(module.run).parameters
    keyword = next((k for k in ("on_progress", "progress") if k in params), None)
    module.run(job, **({keyword: report} if keyword else {}))


# ---- orchestration --------------------------------------------------------------------------------

def run(video, out, kf_fps=1.0, max_minutes=MAX_MINUTES, stop_after=None, concurrency=None, on_status=None):
    """Run every stage on one video. Returns the final status dict. Raises on failure, after
    recording the error in status.json."""
    job = Path(out)
    job.mkdir(parents=True, exist_ok=True)
    (job / ".gitignore").write_text("*\n")  # frames of licensed video must never reach git
    clear_outputs(job, keep=video)
    vlm.configure(cache_path=job.parent / "cache.sqlite", concurrency=concurrency)
    status = Status(job, on_status, vlm.start_meter())
    current = "ingest"

    def step(stage):
        nonlocal current
        current = stage
        status.begin(stage)

    try:
        step("ingest")
        meta = ingest(video, job, status, kf_fps=kf_fps, max_minutes=max_minutes)
        status.end("ingest", f"{len(meta['keyframes'])} keyframes from {meta['n_frames']} frames")
        if stop_after != "ingest":
            step("detect")
            detect_rows, phases = detect(job, meta, status)
            status.end("detect")
        if stop_after not in ("ingest", "detect"):
            step("verify")
            verify_rows = verify(job, meta, phases, status)
            status.end("verify")
            step("amber")
            decisions = decide(job, meta, detect_rows, verify_rows, status)
            counts = Counter(r["status"] for r in decisions)
            status.end("amber", f"{counts['accept']} accept, {counts['reject']} reject, {counts['AMBER']} amber")
        if stop_after in ("ingest", "detect", "amber"):
            status.finish(f"stopped after {stop_after}")
            return status.data
        for stage in ("masks", "propagate"):
            step(stage)
            _run_external(stage, job, status)
            status.end(stage)
        step("narrate")
        narrate(job, meta, phases, decisions, status)
        status.end("narrate")
        for stage in ("render", "export"):
            step(stage)
            _run_external(stage, job, status)
            status.end(stage)
        counts = Counter(r["status"] for r in decisions)
        status.finish(f"done: {counts['accept']} accept, {counts['reject']} reject, {counts['AMBER']} amber for the final pass")
        return status.data
    except Exception as e:
        status.fail(current, f"{e}" if isinstance(e, (PipelineError, vlm.VlmError)) else f"{e}\n{traceback.format_exc(limit=5)}")
        raise
