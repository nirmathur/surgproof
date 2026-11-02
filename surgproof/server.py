"""SurgProof web server: upload a video, watch the pipeline run, review it, release it.

Run with `make serve` or `python -m surgproof.server` (http://localhost:8600).

This module only reads pipeline artifacts from jobs/<id>/ (see docs/E2E_SPEC.md) and records human
clicks. It holds no boxes, verdicts or scores of its own, and it never reads ground truth. Everything
the UI shows is computed from those files on each request.
"""
from __future__ import annotations

import concurrent.futures as cf
import contextlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
APP_DIR = REPO / "web" / "app"
PIPELINE_TARGET = "surgproof.pipeline:run"

MAX_SECONDS = 300
MAX_BYTES = 8 * 1024**3
ALLOWED_SUFFIXES = {".mp4", ".mov"}
STALL_SECONDS = 15 * 60
AUDIT_FRACTION = 0.10
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")

STAGES = [
    ("ingest", "Ingest"),
    ("detect", "Detect"),
    ("verify", "Verify"),
    ("amber", "Amber loop"),
    ("masks", "Masks"),
    ("propagate", "Propagate"),
    ("narrate", "Narrate"),
    ("render", "Render"),
    ("export", "Export"),
]
STAGE_KEYS = [key for key, _ in STAGES]
TERMINAL_STAGES = {"done"}
# A job is only "done" if these exist. A status line saying so is not enough.
REQUIRED_OUTPUTS = ["decisions.jsonl", "annotated.mp4", "dataset_coco.json", "provenance.jsonl"]
# The draft dataset is not exported over HTTP until a human has released it (VISION.md, FINAL PASS).
RELEASE_GATED_FILES = {"dataset_coco.json"}
DOWNLOADABLE = ["annotated.mp4", "released_coco.json", "provenance.jsonl", "final_pass.jsonl"]

HELD_STATUS = "AMBER"
ACTION_APPROVE, ACTION_REJECT, ACTION_CORRECT = "approve", "reject", "correct"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, default: Any = None) -> Any:
    """Tolerant reader: the pipeline may be mid-write in another process."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue  # last line may be half-written while the pipeline is running
    except OSError:
        pass
    return rows


def write_json_atomic(path: Path, payload: Any) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, path)


def write_status(job_dir: Path, **fields: Any) -> None:
    status = read_json(job_dir / "status.json", {}) or {}
    status.update(fields)
    status["updated"] = now_iso()
    write_json_atomic(job_dir / "status.json", status)


def normalise_stage(stage: Any) -> str:
    return str(stage or "").strip().lower()


def _parse_iso(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- video probing


def probe_video(path: Path) -> dict:
    """ffprobe the file. Raises ValueError with a message fit to show the user."""
    cmd = ["ffprobe", "-v", "error", "-protocol_whitelist", "file", "-show_format", "-show_streams",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise ValueError("ffprobe is not installed on this machine") from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError("ffprobe timed out on this file") from exc
    if out.returncode != 0:
        raise ValueError("this file could not be read as a video")
    data = json.loads(out.stdout or "{}")
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("this file has no video stream")
    try:
        duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    except ValueError:
        duration = 0.0
    if duration <= 0:
        raise ValueError("could not read the duration of this video")
    return {
        "duration": duration,
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
        "pix_fmt": video.get("pix_fmt"),
        "r_frame_rate": video.get("r_frame_rate"),
        "avg_frame_rate": video.get("avg_frame_rate"),
    }


# ---------------------------------------------------------------- background runner


def _run_pipeline(target: str, video: str, out_dir: str, options: dict) -> None:
    """Runs in the job's child process. Importing here keeps the server up if the pipeline fails to import."""
    module_name, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_name), func_name)
    func(video, out_dir, **options)


class Runner:
    """Up to max_jobs pipelines at once, each in its own process, so a native crash cannot take the server down."""

    def __init__(self, target: str, max_jobs: int):
        self.target = target
        self.active: set[str] = set()
        self._procs: dict[str, subprocess.Popen] = {}
        self._lock = threading.Lock()
        self._stopping = False
        self._pool = cf.ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="job")

    def submit(self, job_id: str, video: Path, job_dir: Path, options: dict) -> None:
        with self._lock:
            if job_id in self.active:
                raise RuntimeError(f"job {job_id} is already active")
            self.active.add(job_id)
        future = self._pool.submit(self._run, job_id, video, job_dir, options)
        future.add_done_callback(lambda _: self._release(job_id))

    def _release(self, job_id: str) -> None:
        with self._lock:
            self.active.discard(job_id)
            self._procs.pop(job_id, None)

    def _run(self, job_id: str, video: Path, job_dir: Path, options: dict) -> None:
        cmd = [sys.executable, "-m", "surgproof.server", "--run-job", self.target, str(video), str(job_dir),
               json.dumps(options)]
        log_path = job_dir / "worker.log"
        with log_path.open("ab") as log:
            with self._lock:
                if self._stopping:
                    return
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
                self._procs[job_id] = proc
            code = proc.wait()
        if code == 0:
            return
        # The pipeline records its own errors. This covers everything it cannot: import errors, native crashes.
        status = read_json(job_dir / "status.json", {}) or {}
        if not status.get("error"):
            if self._stopping:
                reason = "the server stopped while this job was running"
            else:
                reason = f"the worker exited with code {code}: {_log_tail(log_path)}"
            write_status(job_dir, error=reason)

    def shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            procs = list(self._procs.values())
        for proc in procs:
            with contextlib.suppress(OSError):
                proc.terminate()
        self._pool.shutdown(wait=True, cancel_futures=True)


def _log_tail(path: Path, n_chars: int = 600) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-n_chars:].strip()
    except OSError:
        return ""


# ---------------------------------------------------------------- job store


class FileCache:
    """Memoise an expensive parse of a file by its (mtime, size)."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], tuple[tuple[int, int], Any]] = {}
        self._lock = threading.Lock()

    def get(self, kind: str, path: Path, build: Callable[[], Any], default: Any = None) -> Any:
        try:
            st = path.stat()
        except OSError:
            return default
        stamp = (st.st_mtime_ns, st.st_size)
        key = (kind, str(path))
        with self._lock:
            hit = self._store.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        value = build()
        with self._lock:
            self._store[key] = (stamp, value)
        return value


class Store:
    def __init__(self, jobs_dir: Path):
        self.jobs_dir = Path(jobs_dir)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        ignore = self.jobs_dir / ".gitignore"
        if not ignore.exists():
            ignore.write_text("*\n", encoding="utf-8")
        self.cache = FileCache()
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def lock(self, job_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(job_id, threading.Lock())

    def job_dir(self, job_id: str) -> Path:
        if not JOB_ID_RE.match(job_id) or job_id.startswith("."):
            raise HTTPException(404, "no such job")
        path = self.jobs_dir / job_id
        if not path.is_dir():
            raise HTTPException(404, "no such job")
        return path

    def list_ids(self) -> list[str]:
        ids = []
        for entry in os.scandir(self.jobs_dir):
            if entry.is_dir() and JOB_ID_RE.match(entry.name) and not entry.name.startswith("."):
                if (Path(entry.path) / "status.json").exists() or (Path(entry.path) / "job.json").exists():
                    ids.append(entry.name)
        return ids

    # ------------------------------------------------------------ cached file views

    def rows(self, job_dir: Path, name: str) -> list[dict]:
        path = job_dir / name
        return self.cache.get("rows", path, lambda: read_jsonl(path), [])

    def meta(self, job_dir: Path) -> dict:
        return read_json(job_dir / "meta.json", {}) or {}

    def indexed(self, job_dir: Path, name: str) -> dict[tuple[str, str], dict]:
        """{(kf, label): row}, latest row winning."""
        path = job_dir / name
        return self.cache.get("indexed", path, lambda: {(r.get("kf"), r.get("label")): r for r in read_jsonl(path)}, {})

    def decisions(self, job_dir: Path) -> dict[tuple[str, str], dict]:
        return self.indexed(job_dir, "decisions.jsonl")


def upload_path(job_dir: Path) -> Optional[Path]:
    """The file as uploaded: what a retry feeds back to the pipeline."""
    record = read_json(job_dir / "job.json", {}) or {}
    if record.get("upload") and (job_dir / record["upload"]).is_file():
        return job_dir / record["upload"]
    return None


def playable_video(job_dir: Path) -> Optional[Path]:
    """meta.json `source` names the browser-playable copy written by ingest."""
    name = (read_json(job_dir / "meta.json", {}) or {}).get("source")
    if not name:
        return None
    resolved_job = job_dir.resolve()
    path = (resolved_job / str(name)).resolve()
    return path if path.is_relative_to(resolved_job) and path.is_file() else None


def is_release_gated(job_dir: Path, target: Path) -> bool:
    """Compare by file identity: on a case-insensitive disk DATASET_COCO.json is the same file."""
    for name in RELEASE_GATED_FILES:
        gated = job_dir / name
        if gated.exists() and os.path.samefile(gated, target):
            return True
    return False


def last_activity(job_dir: Path, status: dict) -> Optional[datetime]:
    """Newest write in the job folder: liveness comes from artifact freshness, not from a process handle."""
    newest = 0.0
    with contextlib.suppress(OSError):
        for entry in os.scandir(job_dir):
            with contextlib.suppress(OSError):
                newest = max(newest, entry.stat().st_mtime)
    candidates = [datetime.fromtimestamp(newest, timezone.utc)] if newest else []
    updated = _parse_iso(status.get("updated"))
    if updated:
        candidates.append(updated)
    return max(candidates) if candidates else None


def job_state(job_dir: Path, status: dict, active: set[str]) -> dict:
    """One of queued | running | stalled | failed | done, each justified by what is on disk."""
    stage = normalise_stage(status.get("stage"))
    missing = [name for name in REQUIRED_OUTPUTS if not (job_dir / name).exists()]
    if status.get("error"):
        return {"state": "failed", "error": str(status["error"])}
    finished = stage in TERMINAL_STAGES
    if finished and not missing:
        return {"state": "done", "error": None}
    if job_dir.name in active:
        running = stage not in ("", "queued")
        return {"state": "running" if running else "queued", "error": None}
    if finished:
        return {"state": "failed", "error": "pipeline reported finished, but missing: " + ", ".join(missing)}
    if stage in ("", "queued"):
        return {"state": "queued", "error": None}
    activity = last_activity(job_dir, status)
    if activity and (datetime.now(timezone.utc) - activity).total_seconds() > STALL_SECONDS:
        return {"state": "stalled", "error": None}
    return {"state": "running", "error": None}


def stage_evidence(store: Store, job_dir: Path) -> dict[str, str]:
    """Per-stage counts read from the artifacts, so progress cannot be claimed without output."""
    frames = job_dir / "frames"
    n_frames = len(list(frames.glob("kf_*.jpg"))) if frames.is_dir() else 0
    evidence: dict[str, str] = {}
    if n_frames:
        evidence["ingest"] = f"{n_frames} keyframes"
    for stage, name in (("detect", "detect.jsonl"), ("verify", "verify.jsonl")):
        kfs = {r.get("kf") for r in store.rows(job_dir, name)}
        if kfs:
            evidence[stage] = f"{len(kfs)} keyframes checked"
    tally = decision_tally(store, job_dir)
    if any(tally.values()):
        evidence["amber"] = f"{tally['accept']} accept, {tally['reject']} reject, {tally[HELD_STATUS]} amber"
    masks = job_dir / "masks"
    n_masks = len(list(masks.glob("*.png"))) if masks.is_dir() else 0
    if n_masks:
        evidence["masks"] = f"{n_masks} masks"
    tracks = store.rows(job_dir, "tracks.jsonl")
    if tracks:
        evidence["propagate"] = f"{len({r.get('frame_index') for r in tracks})} frames tracked"
    narration = read_json(job_dir / "narration.json", {}) or {}
    if narration.get("segments"):
        evidence["narrate"] = f"{len(narration['segments'])} segments"
    annotated = job_dir / "annotated.mp4"
    if annotated.exists():
        evidence["render"] = f"{annotated.stat().st_size / 1e6:.1f} MB"
    if (job_dir / "dataset_coco.json").exists():
        coco = store.cache.get("coco", job_dir / "dataset_coco.json",
                               lambda: read_json(job_dir / "dataset_coco.json", {}) or {}, {})
        evidence["export"] = f"{len(coco.get('annotations', []))} annotations"
    return evidence


def decision_tally(store: Store, job_dir: Path) -> dict[str, int]:
    tally = {"accept": 0, "reject": 0, HELD_STATUS: 0}
    for row in store.decisions(job_dir).values():
        if row.get("status") in tally:
            tally[row["status"]] += 1
    return tally


def stage_list(status: dict, state: str, evidence: dict[str, str]) -> list[dict]:
    current = normalise_stage(status.get("stage"))
    index = STAGE_KEYS.index(current) if current in STAGE_KEYS else None
    out = []
    for i, (key, label) in enumerate(STAGES):
        if state == "done":
            stage_state = "done"
        elif index is None:
            stage_state = "pending"
        elif i < index:
            stage_state = "done"
        elif i == index:
            stage_state = "active" if state in ("running", "queued") else state
        else:
            stage_state = "pending"
        out.append({"key": key, "label": label, "state": stage_state, "evidence": evidence.get(key)})
    return out


def job_summary(store: Store, job_dir: Path, active: set[str]) -> dict:
    status = read_json(job_dir / "status.json", {}) or {}
    record = read_json(job_dir / "job.json", {}) or {}
    verdict = job_state(job_dir, status, active)
    activity = last_activity(job_dir, status)
    return {
        "id": job_dir.name,
        "filename": record.get("filename") or job_dir.name,
        "created": record.get("created") or status.get("started"),
        "duration": (record.get("probe") or {}).get("duration"),
        "state": verdict["state"],
        "error": verdict["error"],
        "stage": status.get("stage"),
        "progress": status.get("progress"),
        "stage_progress": status.get("stage_progress"),
        "cost_usd": status.get("cost_usd"),
        "message": status.get("message"),
        "started": status.get("started"),
        "updated": status.get("updated"),
        "last_activity": activity.isoformat(timespec="seconds") if activity else None,
        "released": (job_dir / "released_coco.json").exists(),
    }


# ---------------------------------------------------------------- mask outlines


def mask_polygons(path: Path, width: Optional[int], height: Optional[int]) -> list[list[list[float]]]:
    """Outer contours of a mask PNG, scaled to video pixels. Nonzero means inside the mask."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return []
    if img.ndim == 3:
        img = img[:, :, 3] if img.shape[2] == 4 else img.max(axis=2)
    binary = np.uint8(img > 0) * 255
    mask_h, mask_w = binary.shape[:2]
    scale_x = (width or mask_w) / mask_w
    scale_y = (height or mask_h) / mask_h
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        simplified = cv2.approxPolyDP(contour, 1.0, True).reshape(-1, 2)
        if len(simplified) >= 3:
            polygons.append([[round(float(x) * scale_x, 1), round(float(y) * scale_y, 1)] for x, y in simplified])
    return polygons


def keyframe_maps(meta: dict) -> tuple[dict[str, int], dict[int, str], list[int]]:
    frame_of, kf_of = {}, {}
    for row in meta.get("keyframes") or []:
        if row.get("kf") is None or row.get("frame_index") is None:
            continue
        frame_of[row["kf"]] = int(row["frame_index"])
        kf_of[int(row["frame_index"])] = row["kf"]
    return frame_of, kf_of, sorted(kf_of)


# ---------------------------------------------------------------- final pass


def _export():
    from surgproof import export  # masks lane: queue, validation and the released dataset live there

    return export


def proposal_for(store: Store, job_dir: Path, kf: str, label: str, decision: dict) -> dict:
    """What the AI proposes for this (kf, label), with the evidence behind it."""
    detect = store.indexed(job_dir, "detect.jsonl").get((kf, label), {})
    verify = store.indexed(job_dir, "verify.jsonl").get((kf, label), {})
    present = decision.get("final_present")
    if present is None:
        answers = [r.get("present") for r in decision.get("rounds") or [] if r.get("present") is not None]
        present = answers[-1] if answers else detect.get("present")
    return {"present": bool(present) if present is not None else None, "detect": detect, "verify": verify}


def outline_for(job_dir: Path, meta: dict, kf: str, label: str, decision: dict, proposal: dict) -> tuple[list, str]:
    """The outline a reviewer judges: the snapped mask, else the latest polygon the model gave for it."""
    mask = job_dir / "masks" / f"{kf}_{label}.png"
    if mask.exists():
        polygons = mask_polygons(mask, meta.get("width"), meta.get("height"))
        if polygons:
            return polygons, "mask"
    round_polygons = [r.get("polygon") for r in decision.get("rounds") or [] if r.get("polygon")]
    for polygon, source in ((decision.get("polygon"), "decision"),
                            (round_polygons[-1] if round_polygons else None, "amber loop"),
                            ((proposal.get("detect") or {}).get("polygon"), "detect")):
        if polygon and len(polygon) >= 3:
            return [polygon], source
    return [], "none"


def latest_verdicts(job_dir: Path) -> dict[tuple[str, str], dict]:
    """final_pass.jsonl is append-only: the newest row for a label wins, earlier clicks stay on record."""
    verdicts: dict[tuple[str, str], dict] = {}
    for row in read_jsonl(job_dir / "final_pass.jsonl"):
        if row.get("kf") is not None:
            verdicts[(row["kf"], row.get("label"))] = row
    return verdicts


def human_final_present(verdict: dict) -> bool:
    return verdict.get("action") != ACTION_REJECT


def has_mask(job_dir: Path, kf: str, label: str) -> bool:
    """surgproof.export can only keep a keyframe annotation if the mask file exists."""
    return (job_dir / "masks" / f"{kf}_{label}.png").exists()


def final_pass_view(store: Store, job_dir: Path, include_items: bool = True) -> dict:
    """The review queue and its progress. The queue and the lock themselves are decided by surgproof.export."""
    export = _export()
    queue = export.final_pass_queue(job_dir, audit_frac=AUDIT_FRACTION)
    lock = export.final_pass_state(job_dir)
    decisions = store.decisions(job_dir)
    verdicts = latest_verdicts(job_dir)
    meta = store.meta(job_dir)

    audit_verdicts = [verdicts[(e["kf"], e["label"])] for e in queue
                      if e["kind"] == "audit" and (e["kf"], e["label"]) in verdicts]
    view = {
        "total": lock["n_required"],
        "reviewed": lock["n_done"],
        "pending": lock["n_required"] - lock["n_done"],
        "locked": lock["locked"],
        "n_amber": lock["n_amber"],
        "n_audit": lock["n_audit"],
        "n_audit_pool": (read_json(job_dir / "final_pass_queue.json", {}) or {}).get("n_accepted_pool"),
        "audit_reviewed": len(audit_verdicts),
        "audit_disagreements": sum(1 for v in audit_verdicts if v.get("action") != ACTION_APPROVE),
        "released": lock["released"],
        "width": meta.get("width"),
        "height": meta.get("height"),
    }
    if not include_items:
        return view
    view["items"] = []
    for entry in queue:
        kf, label = entry["kf"], entry["label"]
        decision = decisions.get((kf, label), {})
        proposal = proposal_for(store, job_dir, kf, label, decision)
        verdict = verdicts.get((kf, label))
        outline, outline_source = outline_for(job_dir, meta, kf, label, decision, proposal)
        view["items"].append({
            "id": entry["id"],
            "kf": kf,
            "label": label,
            "kind": entry["kind"],
            "frame_url": f"/api/jobs/{job_dir.name}/files/{entry['frame']}",
            "frame_index": entry.get("frame_index"),
            "ai_status": entry.get("status"),
            "ai_present": proposal["present"],
            "detect_conf": entry.get("conf"),
            "verify_present": proposal["verify"].get("present"),
            "verify_box": proposal["verify"].get("box"),
            "reason": decision.get("reason"),
            "rounds": [{k: v for k, v in r.items() if k != "polygon"} for r in entry.get("rounds") or []],
            "outline": outline,
            "outline_source": outline_source,
            "can_approve": has_mask(job_dir, kf, label),
            "verdict": verdict,
            "final_present": human_final_present(verdict) if verdict else None,
        })
    return view


def check_correction(polygon: list[list[float]], meta: dict) -> None:
    """A human outline must enclose something, and lie on the frame the human was looking at."""
    if len(polygon) < 3 or any(len(point) != 2 for point in polygon):
        raise HTTPException(422, "a correction needs a polygon of at least 3 points")
    width, height = meta.get("width"), meta.get("height")
    if width and height and any(not (0 <= x <= width and 0 <= y <= height) for x, y in polygon):
        raise HTTPException(422, "the polygon falls outside the frame")
    doubled = sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(polygon, polygon[1:] + polygon[:1]))
    if doubled == 0:
        raise HTTPException(422, "the polygon encloses no area")


class FinalPassBody(BaseModel):
    """One human click, or the release. `action` is approve | reject | correct | release."""

    action: str
    kf: Optional[str] = None
    label: Optional[str] = None
    polygon: Optional[list[list[float]]] = None
    reviewer: Optional[str] = None
    note: Optional[str] = None


# ---------------------------------------------------------------- app


def create_app(jobs_dir: Optional[Path] = None, pipeline_target: str = PIPELINE_TARGET,
               max_jobs: Optional[int] = None) -> FastAPI:
    store = Store(Path(jobs_dir or os.environ.get("SURGPROOF_JOBS") or REPO / "jobs"))
    runner = Runner(pipeline_target, max_jobs or int(os.environ.get("SURGPROOF_MAX_JOBS", "1")))
    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Jobs that were queued when the server last stopped have no worker: give them one.
        for job_id in store.list_ids():
            job_dir = store.jobs_dir / job_id
            status = read_json(job_dir / "status.json", {}) or {}
            video = upload_path(job_dir)
            if normalise_stage(status.get("stage")) == "queued" and not status.get("error") and video:
                options = (read_json(job_dir / "job.json", {}) or {}).get("options") or {}
                runner.submit(job_id, video, job_dir, options)
        yield
        runner.shutdown()

    app = FastAPI(title="SurgProof", lifespan=lifespan)
    app.state.store = store
    app.state.runner = runner

    @app.middleware("http")
    async def same_origin_writes(request: Request, call_next):
        # This server has no login, so stop other sites in the browser from posting to it.
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin.split("://", 1)[-1] != request.headers.get("host"):
                return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
        return await call_next(request)

    # ------------------------------------------------------------ jobs

    @app.get("/api/health")
    def health() -> dict:
        return {
            "ok": True,
            "ffmpeg": bool(shutil.which("ffmpeg")),
            "ffprobe": bool(shutil.which("ffprobe")),
            "claude": bool(shutil.which("claude")),
            "max_seconds": MAX_SECONDS,
        }

    @app.post("/api/jobs", status_code=201)
    def create_job(file: Optional[UploadFile] = File(None), video: Optional[UploadFile] = File(None),
                   kf_fps: Optional[float] = Form(None)) -> dict:
        upload = file or video
        if upload is None or not upload.filename:
            raise HTTPException(422, "send the video as a multipart field named 'file'")
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(422, f"only mp4 and mov are supported, got '{suffix or upload.filename}'")
        if kf_fps is not None and not 0 < kf_fps <= 30:
            raise HTTPException(422, "kf_fps must be between 0 and 30")

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        staging = store.jobs_dir / f".upload-{job_id}"
        staging.mkdir()
        try:
            dest = staging / f"upload{suffix}"
            written = 0
            with dest.open("wb") as out:
                while chunk := upload.file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_BYTES:
                        raise HTTPException(413, f"file is larger than {MAX_BYTES // 1024**3} GB")
                    out.write(chunk)
            if written == 0:
                raise HTTPException(422, "the uploaded file is empty")
            try:
                probe = probe_video(dest)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if probe["duration"] > MAX_SECONDS + 0.5:
                raise HTTPException(422, f"video is {probe['duration'] / 60:.1f} minutes, the limit is "
                                         f"{MAX_SECONDS // 60} minutes")
            options = {"kf_fps": kf_fps} if kf_fps is not None else {}
            record = {
                "id": job_id,
                "filename": Path(upload.filename).name,
                "created": now_iso(),
                "upload": dest.name,
                "bytes": written,
                "probe": probe,
                "options": options,
            }
            write_json_atomic(staging / "job.json", record)
            write_status(staging, stage="queued", progress=0.0, message="Waiting for a worker",
                         started=None, error=None)
            job_dir = store.jobs_dir / job_id
            os.replace(staging, job_dir)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        runner.submit(job_id, job_dir / dest.name, job_dir, options)
        return job_summary(store, job_dir, runner.active)

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        jobs = [job_summary(store, store.jobs_dir / job_id, runner.active) for job_id in store.list_ids()]
        jobs.sort(key=lambda j: j.get("created") or "", reverse=True)
        return {"jobs": jobs}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        summary = job_summary(store, job_dir, runner.active)
        status = read_json(job_dir / "status.json", {}) or {}
        evidence = stage_evidence(store, job_dir)
        summary["stages"] = stage_list(status, summary["state"], evidence)
        summary["decisions"] = decision_tally(store, job_dir)
        summary["meta"] = store.meta(job_dir)
        if summary["state"] == "done":
            with store.lock(job_id):
                view = final_pass_view(store, job_dir, include_items=False)
            summary["final_pass"] = {k: view[k] for k in ("total", "reviewed", "pending", "released")}
            summary["outputs"] = {name: {"bytes": (job_dir / name).stat().st_size}
                                  for name in DOWNLOADABLE if (job_dir / name).exists()}
        return summary

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        summary = job_summary(store, job_dir, runner.active)
        if summary["state"] not in ("failed", "stalled"):
            raise HTTPException(409, f"job is {summary['state']}, only failed or stalled jobs can be retried")
        video = upload_path(job_dir)
        if video is None:
            raise HTTPException(409, "the uploaded video for this job is not available to retry")
        options = (read_json(job_dir / "job.json", {}) or {}).get("options") or {}
        write_status(job_dir, stage="queued", progress=0.0, message="Waiting for a worker", error=None)
        runner.submit(job_id, video, job_dir, options)
        return job_summary(store, job_dir, runner.active)

    @app.get("/api/jobs/{job_id}/preview")
    def job_preview(job_id: str) -> dict:
        """The newest keyframe the detector finished, with its real output, for the live progress view."""
        job_dir = store.job_dir(job_id)
        rows = store.rows(job_dir, "detect.jsonl")
        meta = store.meta(job_dir)
        if not rows:
            frames = sorted((job_dir / "frames").glob("kf_*.jpg")) if (job_dir / "frames").is_dir() else []
            if not frames:
                return {"kf": None}
            return {"kf": frames[-1].stem, "frame_url": f"/api/jobs/{job_id}/files/frames/{frames[-1].name}",
                    "objects": [], "width": meta.get("width"), "height": meta.get("height")}
        kf = rows[-1].get("kf")
        decisions = store.decisions(job_dir)
        objects = [{
            "label": row.get("label"),
            "conf": row.get("conf"),
            "polygon": row.get("polygon"),
            "status": (decisions.get((kf, row.get("label"))) or {}).get("status"),
        } for row in rows if row.get("kf") == kf and row.get("present") and row.get("polygon")]
        return {"kf": kf, "frame_url": f"/api/jobs/{job_id}/files/frames/{kf}.jpg", "objects": objects,
                "width": meta.get("width"), "height": meta.get("height")}

    # ------------------------------------------------------------ files and video

    @app.get("/api/jobs/{job_id}/files/{path:path}")
    def job_file(job_id: str, path: str, download: bool = False) -> Response:
        job_dir = store.job_dir(job_id).resolve()
        target = (job_dir / path).resolve()
        if not target.is_relative_to(job_dir) or not target.is_file():
            raise HTTPException(404, "no such file")
        if is_release_gated(job_dir, target) and not (job_dir / "released_coco.json").exists():
            raise HTTPException(403, "the dataset is locked until a human completes the final pass and releases it")
        name = f"{job_id}_{target.name}" if download else None
        return FileResponse(target, filename=name, content_disposition_type="attachment" if download else "inline")

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> Response:
        """The H.264, constant-frame-rate copy ingest makes, so frame index and time stay in step."""
        video = playable_video(store.job_dir(job_id))
        if video is None:
            raise HTTPException(404, "no playable video for this job yet")
        return FileResponse(video, media_type="video/mp4")

    # ------------------------------------------------------------ watch

    def require_done(job_dir: Path) -> None:
        summary = job_summary(store, job_dir, runner.active)
        if summary["state"] != "done":
            raise HTTPException(409, f"job is {summary['state']}, not finished")

    @app.get("/api/jobs/{job_id}/watch")
    def watch_bundle(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        require_done(job_dir)
        meta = store.meta(job_dir)
        frame_of, _, _ = keyframe_maps(meta)
        times = {row["kf"]: row.get("t") for row in meta.get("keyframes") or [] if row.get("kf")}
        phases = [{"kf": row.get("kf"), "t": times.get(row.get("kf")), "phase": row.get("phase"),
                   "conf": row.get("conf")} for row in store.rows(job_dir, "phase.jsonl")]
        phases = sorted((p for p in phases if p["t"] is not None), key=lambda p: p["t"])
        narration = read_json(job_dir / "narration.json", {}) or {}
        return {
            "fps": meta.get("fps"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "n_frames": meta.get("n_frames"),
            "keyframes": sorted(frame_of.values()),
            "phases": phases,
            "narration": narration.get("segments") or [],
        }

    @app.get("/api/jobs/{job_id}/overlay")
    def overlay(job_id: str, start: int = 0, end: int = 0) -> dict:
        """Boxes and mask outlines for frames in [start, end). Only what the pipeline wrote."""
        job_dir = store.job_dir(job_id)
        if end <= start or end - start > 600:
            raise HTTPException(422, "ask for between 1 and 600 frames")
        meta = store.meta(job_dir)
        width, height = meta.get("width"), meta.get("height")
        by_frame = store.cache.get("tracks_by_frame", job_dir / "tracks.jsonl",
                                   lambda: _group_tracks(read_jsonl(job_dir / "tracks.jsonl")), {})
        verdicts = latest_verdicts(job_dir)
        frames: dict[str, list[dict]] = {}
        for frame in range(start, end):
            out = []
            for row in by_frame.get(frame, ()):
                label = row.get("label")
                keyframe = row.get("keyframe")
                sources = row.get("source_kfs") or ([keyframe] if keyframe else [])
                rulings = [verdicts[(kf, label)] for kf in sources if (kf, label) in verdicts]
                overruled = [r for r in rulings if r.get("action") in (ACTION_REJECT, ACTION_CORRECT)]
                box, polygons = row.get("box"), None
                if overruled:
                    # Same rule as the released dataset: an overruled mask, and what flowed from it, is gone.
                    drawn = next((r for r in overruled if r.get("kf") == keyframe and r.get("polygon")), None)
                    if drawn is None:
                        continue
                    polygons = [drawn["polygon"]]
                    xs, ys = zip(*drawn["polygon"])
                    box = [min(xs), min(ys), max(xs), max(ys)]
                if polygons is None:
                    png = (job_dir / "masks" / f"{keyframe}_{label}.png" if keyframe
                           else job_dir / "tracks" / f"{frame}_{label}.png")
                    polygons = mask_polygons(png, width, height) if png.exists() else []
                out.append({
                    "label": label,
                    "box": box,
                    "propagated": bool(row.get("propagated")),
                    "status": row.get("status"),
                    "human": bool(rulings),
                    "polygons": polygons,
                })
            if out:
                frames[str(frame)] = out
        return {"start": start, "end": end, "frames": frames}

    # ------------------------------------------------------------ final pass

    @app.get("/api/jobs/{job_id}/final-pass")
    def get_final_pass(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        require_done(job_dir)
        with store.lock(job_id):
            return final_pass_view(store, job_dir)

    @app.post("/api/jobs/{job_id}/final-pass")
    def post_final_pass(job_id: str, body: FinalPassBody) -> dict:
        job_dir = store.job_dir(job_id)
        require_done(job_dir)
        export = _export()
        reviewer = (body.reviewer or "").strip()[:80]
        with store.lock(job_id):
            if (job_dir / "released_coco.json").exists():
                raise HTTPException(409, "this dataset has already been released")
            if body.action == "release":
                if not reviewer:
                    raise HTTPException(422, "a release is a sign-off: give the reviewer's name")
                try:
                    export.release(job_dir, reviewer=reviewer)
                except export.ReleaseLocked as exc:
                    raise HTTPException(409, f"release is locked: {exc}") from exc
            elif body.action in (ACTION_APPROVE, ACTION_REJECT, ACTION_CORRECT):
                if body.action == ACTION_APPROVE and not has_mask(job_dir, body.kf or "", body.label or ""):
                    raise HTTPException(422, "there is no AI mask to approve here: mark it not present, "
                                             "or draw a correction")
                row = {"kf": body.kf, "label": body.label, "action": body.action,
                       "reviewer": reviewer or None, "ts": now_iso()}
                if body.polygon is not None:
                    check_correction(body.polygon, store.meta(job_dir))
                    row["polygon"] = body.polygon
                if body.note:
                    row["note"] = body.note
                try:
                    export.record_final_pass(job_dir, [row])
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            else:
                raise HTTPException(422, f"unknown action '{body.action}'")
            return final_pass_view(store, job_dir, include_items=False)

    app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="app")
    return app


def _group_tracks(rows: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        if row.get("frame_index") is not None:
            grouped.setdefault(int(row["frame_index"]), []).append(row)
    return grouped


def main() -> None:
    if len(sys.argv) == 6 and sys.argv[1] == "--run-job":
        _run_pipeline(sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5]))
        return
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8600"))
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    main()
