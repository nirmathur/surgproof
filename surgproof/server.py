"""SurgProof web server: upload a video, watch the pipeline run, review it, release it.

Run with `make serve` or `python -m surgproof.server` (http://localhost:8600). A single-user research tool for
laparoscopic cholecystectomy, on public or simulator footage only. It binds to this machine alone.

This module only reads pipeline artifacts from jobs/<id>/ (see docs/E2E_SPEC.md) and records human
clicks. It holds no boxes, verdicts or scores of its own, and it never reads ground truth. Everything
the UI shows is computed from those files on each request.
"""
from __future__ import annotations

import asyncio
import concurrent.futures as cf
import contextlib
import errno
import fcntl
import hashlib
import http.cookies
import importlib
import json
import logging
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.datastructures import Headers
from starlette.requests import ClientDisconnect
from starlette.middleware.trustedhost import TrustedHostMiddleware

from surgproof.webauth import COOKIE_NAME, SESSION_SECONDS, AuthError, LocalAuth

log = logging.getLogger("surgproof.server")

REPO = Path(__file__).resolve().parent.parent
APP_DIR = REPO / "web" / "app"
PIPELINE_TARGET = "surgproof.pipeline:run"

BIND_HOST = "127.0.0.1"  # fixed: there is no setting that exposes this server to the network
ALLOWED_HOSTS = ("localhost", "127.0.0.1")
MAX_SECONDS = 300
MAX_BYTES = 4 * 1024**3
MAX_JSON_BYTES = 1024**2
MAX_FIELD_BYTES = 4096
MAX_KF_FPS = 5.0
ALLOWED_SUFFIXES = {".mp4", ".mov"}
FOOTAGE_SOURCES = ("public", "simulator")
SPACE_FACTOR = 3  # frames, masks and tracks can take several times the size of the upload
STALL_SECONDS = 15 * 60
AUDIT_FRACTION = 0.10
RETENTION_SWEEP_SECONDS = 3600
STOP_GRACE_SECONDS = 5
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
PUBLIC_API = {"/api/health", "/api/auth/state", "/api/auth/setup", "/api/auth/login", "/api/auth/logout"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
WORKER_LOCK = ".worker.lock"
RELEASE_STATUS = "release_status.json"
RELEASE_WAIT_POLL_SECONDS = 0.2
UPLOAD_CHECK_SECONDS = 0.5

UPLOAD_NOTICE = (
    "Frames from this video are sent to a third-party AI model provider to be labelled. "
    "This build is for public or simulator laparoscopic cholecystectomy footage only. "
    "Do not upload real patient video. "
    "By continuing you confirm that you have the right to use this footage and to send it to that provider."
)
NOTICE_SHA256 = hashlib.sha256(UPLOAD_NOTICE.encode("utf-8")).hexdigest()

STAGES = [
    ("ingest", "Ingest"),
    ("detect", "Detect"),
    ("verify", "Second pass"),
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
# Served over HTTP only by exact name. Nothing else in a job folder is reachable: not masks, logs, raw model rows.
FRAME_FILE_RE = re.compile(r"^frames/(kf_|prop_)\d+\.jpg$")  # prop_: a propagated frame queued for audit
# Nothing is exported until a person has released it (VISION.md, FINAL PASS). Watch shows the AI draft in the app.
DOWNLOADABLE = ["annotated.mp4", "released_coco.json", "dataset_bundle.zip", "provenance.jsonl", "final_pass.jsonl",
                "release_manifest.json"]
RELEASE_GATED_FILES = {"dataset_coco.json", *DOWNLOADABLE}
RAW_REF_FILES = ("detect.jsonl", "verify.jsonl", "decisions.jsonl", "phase.jsonl", "narration.json", "provenance.jsonl")

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
    except OSError as exc:
        raise ValueError(f"ffprobe could not be run: {exc}") from exc
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
    """Runs in the job's child process. The kernel drops the lock when this process dies, however it dies."""
    lock_file = _take_worker_lock(Path(out_dir))  # held for as long as this name is in scope
    module_name, _, func_name = target.partition(":")
    func = getattr(importlib.import_module(module_name), func_name)
    func(video, out_dir, **options)
    lock_file.close()


def _take_worker_lock(job_dir: Path):
    """One worker per job, pipeline or release. The kernel drops the lock when the process dies, however it dies."""
    lock_path = job_dir / WORKER_LOCK
    lock_file = lock_path.open("a+", encoding="utf-8")
    for attempt in range(30):  # the server's liveness probe holds this lock for an instant
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except OSError:
            if attempt == 29:
                raise SystemExit(f"another worker already holds {lock_path}") from None
            time.sleep(0.1)
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


def _write_release_status(job_dir: Path, **fields: Any) -> None:
    status = read_json(job_dir / RELEASE_STATUS, {}) or {}
    status.update(fields)
    status["updated"] = now_iso()
    write_json_atomic(job_dir / RELEASE_STATUS, status)


def _run_release(job_dir_text: str, reviewer: str) -> None:
    """Runs in the release's child process: rendering takes minutes, and must not hold up the server or the job lock."""
    job_dir = Path(job_dir_text)
    lock_file = _take_worker_lock(job_dir)
    from surgproof import export

    def progress(fraction: Any = None, message: Any = "") -> None:
        fields: dict = {"message": str(message or "Rendering the reviewed video")}
        if isinstance(fraction, (int, float)):
            fields["progress"] = max(0.0, min(1.0, float(fraction)))
        _write_release_status(job_dir, **fields)

    try:
        export.release(job_dir, reviewer=reviewer, on_progress=progress)
    except export.ReleaseLocked as exc:
        _write_release_status(job_dir, state="failed", locked=True, error=str(exc), finished=now_iso())
        lock_file.close()
        return
    except BaseException as exc:
        _write_release_status(job_dir, state="failed", locked=False, error=f"{type(exc).__name__}: {exc}", finished=now_iso())
        raise
    _write_release_status(job_dir, state="done", progress=1.0, message="Released", finished=now_iso())
    lock_file.close()


_probe_guard = threading.Lock()  # two probes at once would each read the other's lock as a live worker


def worker_pid(job_dir: Path) -> Optional[int]:
    """The pid of a live worker for this job, or None. Being able to take the lock proves there is none."""
    path = job_dir / WORKER_LOCK
    with _probe_guard:
        try:
            with path.open("r", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    text = fh.read().strip()
                    return int(text) if text.isdigit() else -1  # held, but its pid is not written yet
                fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            pass
    return None


def kill_worker_group(job_dir: Path) -> bool:
    """SIGTERM, then SIGKILL, to the worker's whole process group: it takes claude and ffmpeg children with it."""
    pid = worker_pid(job_dir)
    if pid is None or pid <= 0:
        return pid is None
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGTERM)
    deadline = time.time() + STOP_GRACE_SECONDS
    while time.time() < deadline and worker_pid(job_dir) is not None:
        time.sleep(0.1)
    if worker_pid(job_dir) is not None:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(pid, signal.SIGKILL)
        deadline = time.time() + STOP_GRACE_SECONDS
        while time.time() < deadline and worker_pid(job_dir) is not None:
            time.sleep(0.1)
    # The group is signalled even after the leader exits: a child that outlived it must go too.
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, signal.SIGKILL)
    return worker_pid(job_dir) is None


class Runner:
    """Up to max_jobs pipelines at once, each in its own process group, so a crash cannot take the server down."""

    def __init__(self, target: str, max_jobs: int):
        self.target = target
        self.active: set[str] = set()
        self.failures: dict[str, str] = {}  # for when the disk is too full to record a failure in status.json
        self._lock = threading.Lock()
        self._stopping = False
        self._stopped_jobs: set[str] = set()
        self._dirs: dict[str, Path] = {}
        self._procs: dict[str, subprocess.Popen] = {}
        self._futures: dict[str, cf.Future] = {}
        self._threads: list[threading.Thread] = []
        self._pool = cf.ThreadPoolExecutor(max_workers=max_jobs, thread_name_prefix="job")

    def submit(self, job_id: str, video: Path, job_dir: Path, options: dict) -> None:
        with self._lock:
            if job_id in self.active:
                raise RuntimeError(f"job {job_id} is already active")
            self.active.add(job_id)
            self.failures.pop(job_id, None)
            self._stopped_jobs.discard(job_id)
            self._dirs[job_id] = job_dir
        future = self._pool.submit(self._run, job_id, video, job_dir, options)
        with self._lock:
            self._futures[job_id] = future
        future.add_done_callback(lambda _: self._release(job_id))

    def _release(self, job_id: str) -> None:
        with self._lock:
            self.active.discard(job_id)
            self._dirs.pop(job_id, None)
            self._procs.pop(job_id, None)
            self._futures.pop(job_id, None)

    def _fail(self, job_id: str, job_dir: Path, reason: str) -> None:
        log.error("job %s failed: %s", job_id, reason)
        try:
            write_status(job_dir, error=reason)
        except OSError as exc:
            log.error("job %s: could not record the failure: %s", job_id, exc)
            with self._lock:
                self.failures[job_id] = reason

    def _run(self, job_id: str, video: Path, job_dir: Path, options: dict) -> None:
        # Everything is caught here: a job that dies must never be left reading as "running".
        try:
            self._run_worker(job_id, video, job_dir, options)
        except BaseException as exc:  # noqa: BLE001
            log.exception("job %s: the runner failed", job_id)
            self._fail(job_id, job_dir, f"the job runner failed: {type(exc).__name__}: {exc}")

    def _run_worker(self, job_id: str, video: Path, job_dir: Path, options: dict) -> None:
        cmd = [sys.executable, "-m", "surgproof.server", "--run-job", self.target, str(video), str(job_dir),
               json.dumps(options)]
        log_path = job_dir / "worker.log"
        with log_path.open("ab") as out:
            with self._lock:  # with the check, so a stop either sees this process or prevents it
                if self._stopping or job_id in self._stopped_jobs:
                    return
                proc = subprocess.Popen(cmd, cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                self._procs[job_id] = proc
            while (code := proc.poll()) is None:
                self._drop_upload(job_id, job_dir)
                time.sleep(UPLOAD_CHECK_SECONDS)
        # However the worker ended, a claude or ffmpeg it started must not run on, spending money, with no owner.
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        self._drop_upload(job_id, job_dir)
        status = read_json(job_dir / "status.json", {}) or {}
        finished = normalise_stage(status.get("stage")) in TERMINAL_STAGES
        with self._lock:
            stopped = job_id in self._stopped_jobs
        if code == 0 and finished:
            return
        if status.get("error"):
            return  # the pipeline recorded its own failure
        if stopped:
            return
        if self._stopping:
            self._fail(job_id, job_dir, "the server stopped while this job was running")
        elif code == 0:
            self._fail(job_id, job_dir, "the worker exited without finishing the job")
        else:
            self._fail(job_id, job_dir, f"the worker exited with code {code}: {_log_tail(log_path)}")

    @staticmethod
    def _drop_upload(job_id: str, job_dir: Path) -> None:
        try:
            drop_upload_after_ingest(job_dir)
        except OSError as exc:
            log.error("job %s: could not delete the original upload: %s", job_id, exc)

    def stop(self, job_id: str, job_dir: Path, timeout: float = 20.0) -> None:
        """Kill this job's worker and wait until the runner has let go of it."""
        with self._lock:
            self._stopped_jobs.add(job_id)
            future = self._futures.get(job_id)
        if future is not None and future.cancel():  # still waiting for a free slot: there is nothing to kill
            self._release(job_id)
            return
        self._signal_group(job_id, signal.SIGTERM)
        grace = time.time() + STOP_GRACE_SECONDS
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if job_id not in self.active:
                    return
            if time.time() >= grace:
                self._signal_group(job_id, signal.SIGKILL)
            time.sleep(0.1)
        raise RuntimeError(f"could not stop the worker for job {job_id}")

    def start_release(self, job_id: str, job_dir: Path, reviewer: str) -> None:
        """Not through the job pool: a person who finished reviewing should not queue behind someone else's upload."""
        with self._lock:
            if job_id in self.active or self._stopping:
                raise RuntimeError("this job already has a worker")
            self.active.add(job_id)
            self._dirs[job_id] = job_dir
        _write_release_status(job_dir, state="running", progress=0.0, message="Starting the release",
                              reviewer=reviewer, started=now_iso(), error=None, locked=False)
        thread = threading.Thread(target=self._release_thread, args=(job_id, job_dir, reviewer), name=f"release-{job_id}")
        with self._lock:
            self._threads.append(thread)
        thread.start()

    def _release_thread(self, job_id: str, job_dir: Path, reviewer: str) -> None:
        try:
            log_path = job_dir / "release.log"
            with log_path.open("ab") as out:
                with self._lock:
                    if self._stopping:
                        raise RuntimeError("the server is stopping")
                    proc = subprocess.Popen([sys.executable, "-m", "surgproof.server", "--run-release", str(job_dir), reviewer],
                                            cwd=REPO, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                    self._procs[job_id] = proc
                code = proc.wait()
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)  # a render it started must not outlive it
            status = read_json(job_dir / RELEASE_STATUS, {}) or {}
            if status.get("state") == "running":
                reason = ("the server stopped during the release" if self._stopping
                          else f"the release worker exited with code {code}: {_log_tail(log_path)}")
                _write_release_status(job_dir, state="failed", locked=False, error=reason, finished=now_iso())
        except BaseException as exc:  # noqa: BLE001
            log.exception("job %s: the release failed", job_id)
            with contextlib.suppress(OSError):
                _write_release_status(job_dir, state="failed", locked=False, error=f"{type(exc).__name__}: {exc}", finished=now_iso())
        finally:
            self._release(job_id)

    def _signal_group(self, job_id: str, sig: int) -> None:
        """By the process we started, not the lock file: this works even before the worker has taken its lock."""
        with self._lock:
            proc = self._procs.get(job_id)
        if proc is not None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, sig)

    def shutdown(self) -> None:
        with self._lock:
            self._stopping = True
            jobs = list(self._procs)
        for job_id in jobs:
            self._signal_group(job_id, signal.SIGTERM)

        def finish_killing() -> None:
            time.sleep(STOP_GRACE_SECONDS)
            for job_id in jobs:
                self._signal_group(job_id, signal.SIGKILL)

        threading.Thread(target=finish_killing, daemon=True).start()
        self._pool.shutdown(wait=True, cancel_futures=True)
        for thread in self._threads:
            thread.join(timeout=STOP_GRACE_SECONDS * 3)


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
        if path.is_symlink() or not path.is_dir():
            raise HTTPException(404, "no such job")
        return path

    def list_ids(self) -> list[str]:
        ids = []
        for entry in os.scandir(self.jobs_dir):
            if entry.is_dir(follow_symlinks=False) and JOB_ID_RE.match(entry.name) and not entry.name.startswith("."):
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
    """The file as uploaded, with whatever audio and metadata it came with. Gone once ingest has finished."""
    record = read_json(job_dir / "job.json", {}) or {}
    if record.get("upload") and (job_dir / record["upload"]).is_file():
        return job_dir / record["upload"]
    return None


def pipeline_input(job_dir: Path) -> Optional[Path]:
    """What a retry feeds the pipeline: the upload, or after ingest the stripped copy (core accepts it as its own input)."""
    return upload_path(job_dir) or playable_video(job_dir)


def drop_upload_after_ingest(job_dir: Path) -> bool:
    """Delete the original file as soon as ingest has written the stripped source.mp4. Nothing later reads it."""
    upload = upload_path(job_dir)
    if upload is None or playable_video(job_dir) is None:
        return False
    status = read_json(job_dir / "status.json", {}) or {}
    stage = normalise_stage(status.get("stage"))
    if status.get("error") and not (job_dir / "frames").is_dir():
        return False
    if stage not in TERMINAL_STAGES and stage not in STAGE_KEYS[1:]:
        return False
    upload.unlink(missing_ok=True)
    record = read_json(job_dir / "job.json", {}) or {}
    record["upload"] = None
    record["upload_deleted_at"] = now_iso()
    write_json_atomic(job_dir / "job.json", record)
    return True


def playable_video(job_dir: Path) -> Optional[Path]:
    """meta.json `source` names the browser-playable copy written by ingest."""
    name = (read_json(job_dir / "meta.json", {}) or {}).get("source")
    if not name:
        return None
    resolved_job = job_dir.resolve()
    path = (resolved_job / str(name)).resolve()
    return path if path.is_relative_to(resolved_job) and path.is_file() else None


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


def release_job(job_dir: Path, runner: Runner) -> Optional[dict]:
    """The release's progress. What is on disk says, unless nothing is running it any more."""
    status = read_json(job_dir / RELEASE_STATUS)
    if not status:
        return None
    if status.get("state") == "running":
        if (job_dir / "released_coco.json").exists():
            status.update(state="done", progress=1.0)  # committed, and the worker has not written its last line yet
        elif job_dir.name not in runner.active and worker_pid(job_dir) is None:
            status.update(state="failed", locked=False, error="the release was interrupted before it finished: nothing was released")
    return status


def release_running(job_dir: Path, runner: Runner) -> bool:
    return (release_job(job_dir, runner) or {}).get("state") == "running"


def job_state(job_dir: Path, status: dict, runner: Runner) -> dict:
    """One of queued | running | stalled | failed | done, each justified by what is on disk."""
    stage = normalise_stage(status.get("stage"))
    missing = [name for name in REQUIRED_OUTPUTS if not (job_dir / name).exists()]
    remembered = runner.failures.get(job_dir.name)
    if status.get("error") or remembered:
        return {"state": "failed", "error": str(status.get("error") or remembered)}
    finished = stage in TERMINAL_STAGES
    if finished and not missing:
        return {"state": "done", "error": None}
    in_runner = job_dir.name in runner.active
    if in_runner or worker_pid(job_dir) is not None:
        if stage in ("", "queued"):
            return {"state": "queued", "error": None}
        activity = last_activity(job_dir, status)
        if activity and (datetime.now(timezone.utc) - activity).total_seconds() > STALL_SECONDS:
            return {"state": "stalled", "error": None, "detail": "hung"}
        return {"state": "running", "error": None}
    if finished:
        return {"state": "failed", "error": "pipeline reported finished, but missing: " + ", ".join(missing)}
    if stage in ("", "queued") or (job_dir / WORKER_LOCK).exists():
        # This server started it (or was meant to) and nothing holds it now: no need to wait for it to go stale.
        return {"state": "stalled", "error": None, "detail": "gone"}
    # Started outside this server, by the CLI: only the age of its output can tell.
    activity = last_activity(job_dir, status)
    if activity and (datetime.now(timezone.utc) - activity).total_seconds() > STALL_SECONDS:
        return {"state": "stalled", "error": None, "detail": "gone"}
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


def retention_days() -> float:
    return float(os.environ.get("SURGPROOF_RETENTION_DAYS", "30"))


def job_created(job_dir: Path) -> Optional[datetime]:
    record = read_json(job_dir / "job.json", {}) or {}
    created = _parse_iso(record.get("created")) if record.get("created") else None
    if created:
        return created
    with contextlib.suppress(OSError):
        return datetime.fromtimestamp((job_dir / "status.json").stat().st_mtime, timezone.utc)
    return None


def display_name(job_id: str) -> str:
    """A name that says nothing about the source: upload filenames are not stored or listed anywhere."""
    return f"Job {job_id[-6:]}"


def job_summary(store: Store, job_dir: Path, runner: Runner) -> dict:
    status = read_json(job_dir / "status.json", {}) or {}
    record = read_json(job_dir / "job.json", {}) or {}
    verdict = job_state(job_dir, status, runner)
    activity = last_activity(job_dir, status)
    created = job_created(job_dir)
    days = retention_days()
    return {
        "id": job_dir.name,
        "name": display_name(job_dir.name),
        "created": record.get("created") or status.get("started"),
        "expires": (created + timedelta(days=days)).isoformat(timespec="seconds") if created and days > 0 else None,
        "duration": (record.get("probe") or {}).get("duration"),
        "state": verdict["state"],
        "state_detail": verdict.get("detail"),
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


# ---------------------------------------------------------------- disk, cache, deletion


def disk_report(folder: Path) -> dict:
    usage = shutil.disk_usage(folder)
    minimum = min_free_bytes()
    return {"free_bytes": usage.free, "total_bytes": usage.total, "min_free_bytes": minimum, "ok": usage.free >= minimum}


def min_free_bytes() -> int:
    return int(float(os.environ.get("SURGPROOF_MIN_FREE_MB", "500")) * 1024**2)


def check_disk_for_upload(folder: Path, size: Optional[int]) -> None:
    free = shutil.disk_usage(folder).free
    needed = min_free_bytes() + (size or 0) * SPACE_FACTOR
    if free < needed:
        raise HTTPException(507, f"not enough disk space: {free / 1e6:.0f} MB free, this upload needs "
                                 f"about {needed / 1e6:.0f} MB. Delete old jobs, or free some space.")


def cache_path(store: Store) -> Path:
    override = os.environ.get("SURGPROOF_CACHE")
    return Path(override) if override else store.jobs_dir / "cache.sqlite"


def job_raw_refs(job_dir: Path) -> set[str]:
    """Every model response this job's rows point at, wherever in a row the reference sits."""
    refs: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "raw_ref" and isinstance(inner, str):
                    refs.add(inner)
                else:
                    walk(inner)
        elif isinstance(value, list):
            for inner in value:
                walk(inner)

    for name in RAW_REF_FILES:
        path = job_dir / name
        if name.endswith(".jsonl"):
            for row in read_jsonl(path):
                walk(row)
        else:
            walk(read_json(path))
    return refs


def purge_cache(store: Store, job_dir: Path) -> int:
    """Remove this job's model responses, keeping any another job still cites.

    When every other job has finished, responses nobody cites go too. They can only come from this job's
    aborted attempts, and their rows do not say so. While another job could still resume, they stay.
    """
    path = cache_path(store)
    if not path.exists():
        return 0
    shared = bool(os.environ.get("SURGPROOF_CACHE"))  # may hold other folders' jobs, which this cannot see
    others = [store.jobs_dir / name for name in store.list_ids() if name != job_dir.name]
    keep: set[str] = set()
    all_finished = True
    for other in others:
        keep |= job_raw_refs(other)
        status = read_json(other / "status.json", {}) or {}
        if normalise_stage(status.get("stage")) not in TERMINAL_STAGES:
            all_finished = False
    mine = job_raw_refs(job_dir) - keep
    con = sqlite3.connect(path, timeout=30.0)
    try:
        con.execute("PRAGMA secure_delete=ON")  # overwrite the deleted text rather than just unlinking the pages
        has_table = con.execute("select 1 from sqlite_master where type='table' and name='vlm_cache'").fetchone()
        if not has_table:
            return 0
        if all_finished and not shared:
            stale = [row[0] for row in con.execute("select key from vlm_cache") if row[0] not in keep]
        else:
            stale = sorted(mine)
        with con:
            con.executemany("delete from vlm_cache where key = ?", [(key,) for key in stale])
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return len(stale)
    finally:
        con.close()


def delete_job_files(store: Store, job_dir: Path) -> int:
    """Cache first, so a failure part-way leaves the job listed and the delete can be repeated."""
    removed = purge_cache(store, job_dir)
    doomed = store.jobs_dir / f".deleting-{job_dir.name}-{uuid.uuid4().hex[:6]}"
    os.replace(job_dir, doomed)
    shutil.rmtree(doomed, ignore_errors=True)
    return removed


def sweep_leftovers(jobs_dir: Path) -> int:
    """Staging folders and half-deleted jobs from a previous run. Called once, before any request is served."""
    count = 0
    for pattern in (".upload-*", ".deleting-*"):
        for path in jobs_dir.glob(pattern):
            if not path.is_symlink():
                shutil.rmtree(path, ignore_errors=True)
                count += 1
    return count


def sweep_expired(store: Store, runner: Runner) -> list[str]:
    days = retention_days()
    if days <= 0:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    deleted = []
    for job_id in store.list_ids():
        job_dir = store.jobs_dir / job_id
        created = job_created(job_dir)
        if created is None or created >= cutoff:
            continue
        with store.lock(job_id):
            if job_id in runner.active or worker_pid(job_dir) is not None:
                continue
            delete_job_files(store, job_dir)
        log.info("job %s passed the %g-day retention and was deleted", job_id, days)
        deleted.append(job_id)
    return deleted


# ---------------------------------------------------------------- model backend status


class ModelStatus:
    """Whether the model backend looks usable, without spending a model call. Cached: this runs subprocesses."""

    TTL_SECONDS = 60

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[dict] = None
        self._at = 0.0

    def get(self) -> dict:
        with self._lock:
            if self._value is not None and time.time() - self._at < self.TTL_SECONDS:
                return self._value
        value = self._check()
        with self._lock:
            self._value, self._at = value, time.time()
        return value

    @staticmethod
    def _check() -> dict:
        provider = os.environ.get("SURGPROOF_VLM", "claude_cli")
        if provider == "mock":
            return {"provider": provider, "ok": True, "detail": "mock model: output is placeholders, not AI labels"}
        if provider == "gemini":
            has_key = bool(os.environ.get("GEMINI_API_KEY"))
            return {"provider": provider, "ok": has_key, "detail": "API key present, not tested" if has_key else "GEMINI_API_KEY is not set"}
        binary = os.environ.get("SURGPROOF_CLAUDE_BIN", "claude")
        if not shutil.which(binary):
            return {"provider": provider, "ok": False, "detail": f"{binary} is not installed"}
        try:
            out = subprocess.run([binary, "auth", "status"], capture_output=True, text=True, timeout=15)
            logged_in = bool(json.loads(out.stdout).get("loggedIn"))  # only this field: the rest is account detail
        except (OSError, ValueError, AttributeError, subprocess.SubprocessError):
            return {"provider": provider, "ok": False, "detail": "could not read the login status of the claude CLI"}
        if not logged_in:
            return {"provider": provider, "ok": False, "detail": "the claude CLI is not logged in"}
        return {"provider": provider, "ok": True, "detail": "claude CLI logged in. Usage limits are not checked."}


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
        if isinstance(row, dict) and row.get("kf") is not None:
            verdicts[(row["kf"], row.get("label"))] = row
    return verdicts


def human_final_present(verdict: dict) -> bool:
    return verdict.get("action") != ACTION_REJECT


def has_mask(job_dir: Path, kf: str, label: str) -> bool:
    return (job_dir / "masks" / f"{kf}_{label}.png").exists()


def can_approve(job_dir: Path, kf: str, label: str, ai_present: Optional[bool]) -> bool:
    """Approve agrees with the AI, so there must be something to agree with: its mask, or its answer of absent.

    In surgproof.export an approve with no mask means absent. An AI that said present but drew nothing must not
    turn into that by a click on Approve.
    """
    return has_mask(job_dir, kf, label) or ai_present is False


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
    pools = read_json(job_dir / "final_pass_queue.json", {}) or {}
    audit_by_stratum: dict[str, int] = {}
    for entry in queue:
        if entry["kind"] == "audit":
            key = entry.get("stratum") or "accepted"
            audit_by_stratum[key] = audit_by_stratum.get(key, 0) + 1
    view = {
        "total": lock["n_required"],
        "reviewed": lock["n_done"],
        "pending": lock["n_required"] - lock["n_done"],
        "locked": lock["locked"],
        "n_amber": lock["n_amber"],
        "n_audit": lock["n_audit"],
        "n_audit_pool": pools.get("n_accepted_pool"),
        "audit_by_stratum": audit_by_stratum,
        "audit_reviewed": len(audit_verdicts),
        "audit_disagreements": lock.get("n_audit_failed", sum(1 for v in audit_verdicts if v.get("action") != ACTION_APPROVE)),
        "n_escalation": lock.get("n_escalation", 0),
        "escalated": lock.get("escalated", False),
        "escalation_reason": lock.get("escalation_reason"),
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
        if entry.get("stratum") == "propagated":
            proposal["present"] = True  # a propagated frame is only ever queued where the label was carried
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
            "can_approve": can_approve(job_dir, kf, label, proposal["present"]),
            "stratum": entry.get("stratum"),
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
    note: Optional[str] = None
    wait: bool = False  # release only: answer when it has finished, for scripts. The page polls instead.


# ---------------------------------------------------------------- app


class UploadRejected(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class UploadStream:
    """Parses a multipart body as it arrives and writes the video straight to staging.

    Nothing is spooled first, so the size limit stops an oversized upload at the limit, not after it has landed.
    """

    DISK_CHECK_BYTES = 64 * 1024**2

    def __init__(self, boundary: bytes, staging: Path, jobs_dir: Path):
        self.staging = staging
        self.jobs_dir = jobs_dir
        self.fields: dict[str, str] = {}
        self.dest: Optional[Path] = None
        self.written = 0
        self._out = None
        self._next_disk_check = self.DISK_CHECK_BYTES
        self._reset_part()
        self.parser = MultipartParser(boundary, {
            "on_part_begin": self._reset_part,
            "on_header_field": self._header_field,
            "on_header_value": self._header_value,
            "on_header_end": self._header_end,
            "on_headers_finished": self._headers_finished,
            "on_part_data": self._part_data,
            "on_part_end": self._part_end,
        })

    def _reset_part(self) -> None:
        self._field_name, self._header_name, self._header_buf = None, b"", b""
        self._disposition = b""
        self._buffer = b""
        self._in_file = False

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._header_name += data[start:end]

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._header_buf += data[start:end]

    def _header_end(self) -> None:
        if self._header_name.lower() == b"content-disposition":
            self._disposition = self._header_buf
        self._header_name, self._header_buf = b"", b""

    def _headers_finished(self) -> None:
        _, params = parse_options_header(self._disposition)
        name = params.get(b"name", b"").decode("utf-8", "replace")
        filename = params.get(b"filename")
        if name in ("file", "video") and filename is not None:
            if self.dest is not None:
                raise UploadRejected(422, "send one video at a time")
            suffix = Path(filename.decode("utf-8", "replace")).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                raise UploadRejected(422, f"only mp4 and mov are supported, got '{suffix or 'no extension'}'")
            self.dest = self.staging / f"upload{suffix}"
            self._out = self.dest.open("wb")
            self._in_file = True
        else:
            self._field_name = name

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        if self._in_file:
            self.written += end - start
            if self.written > MAX_BYTES:
                raise UploadRejected(413, f"the file is larger than the {MAX_BYTES // 1024**3} GB limit")
            if self.written >= self._next_disk_check:
                self._next_disk_check += self.DISK_CHECK_BYTES
                if shutil.disk_usage(self.jobs_dir).free < min_free_bytes():
                    raise UploadRejected(507, "the disk filled up during this upload")
            self._out.write(data[start:end])
        else:
            self._buffer += data[start:end]
            if len(self._buffer) > MAX_FIELD_BYTES:
                raise UploadRejected(413, "a form field is too large")

    def _part_end(self) -> None:
        if self._in_file:
            self._out.close()
            self._out = None
        elif self._field_name:
            self.fields[self._field_name] = self._buffer.decode("utf-8", "replace")

    def write(self, chunk: bytes) -> None:
        self.parser.write(chunk)

    def close(self) -> None:
        if self._out is not None:
            self._out.close()
            self._out = None


class Guard:
    """Refuses anything that did not come from a signed-in page on this machine, before a route sees it."""

    def __init__(self, app: Any, auth: LocalAuth):
        self.app = app
        self.auth = auth

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        method, path = scope["method"], scope["path"]
        if method not in SAFE_METHODS:
            origin = headers.get("origin")
            if origin and origin.split("://", 1)[-1] != headers.get("host"):
                await JSONResponse({"detail": "cross-origin request refused"}, status_code=403)(scope, receive, send)
                return
            if headers.get("sec-fetch-site") in ("cross-site", "same-site"):
                await JSONResponse({"detail": "cross-site request refused"}, status_code=403)(scope, receive, send)
                return
        cookies: http.cookies.SimpleCookie = http.cookies.SimpleCookie()
        with contextlib.suppress(http.cookies.CookieError):
            cookies.load(headers.get("cookie", ""))
        token = cookies[COOKIE_NAME].value if COOKIE_NAME in cookies else None
        user = self.auth.user_for(token)
        state = scope.setdefault("state", {})
        state["user"], state["token"] = user, token
        if path.startswith("/api/") and path not in PUBLIC_API and user is None:
            await JSONResponse({"detail": "sign in first"}, status_code=401)(scope, receive, send)
            return
        is_upload = method == "POST" and path == "/api/jobs"
        if method not in SAFE_METHODS and not is_upload:
            limited = await self._limit_body(headers, scope, receive, send)
            if limited is None:
                return
            receive = limited
        await self.app(scope, receive, send)

    @staticmethod
    async def _limit_body(headers: Headers, scope: dict, receive: Callable, send: Callable) -> Optional[Callable]:
        length = headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_JSON_BYTES:
            await JSONResponse({"detail": "request body too large"}, status_code=413)(scope, receive, send)
            return None
        seen = 0

        async def counted() -> dict:
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > MAX_JSON_BYTES:  # no honest Content-Length: drop it rather than buffer it
                return {"type": "http.disconnect"}
            return message

        return counted


class AuthBody(BaseModel):
    password: str
    name: str = ""
    setup_code: str = ""


def _session_response(payload: dict, token: str) -> JSONResponse:
    response = JSONResponse(payload)
    response.set_cookie(COOKIE_NAME, token, max_age=SESSION_SECONDS, httponly=True, samesite="strict", path="/")
    return response


def released_counts(store: Store, job_dir: Path) -> Optional[dict]:
    """How much of the released dataset a person actually looked at, counted from the file itself."""
    path = job_dir / "released_coco.json"

    def build() -> dict:
        coco = read_json(path, {}) or {}
        attributes = [a.get("attributes") or {} for a in coco.get("annotations", [])]
        return {
            "n_annotations": len(attributes),
            # surgproof.export sets this only for the exact annotation a person saw, tracked frames included.
            "n_human_reviewed": sum(1 for a in attributes if a.get("human_reviewed")),
            "n_propagated": sum(1 for a in attributes if a.get("propagated")),
            # False: there was no source video, so annotated.mp4 is still the unreviewed draft.
            "annotated_rerendered": (coco.get("info") or {}).get("annotated_rerendered"),
        }

    return store.cache.get("released_counts", path, build, None)


def create_app(jobs_dir: Optional[Path] = None, pipeline_target: str = PIPELINE_TARGET,
               max_jobs: Optional[int] = None) -> FastAPI:
    store = Store(Path(jobs_dir or os.environ.get("SURGPROOF_JOBS") or REPO / "jobs"))
    auth = LocalAuth(store.jobs_dir)
    runner = Runner(pipeline_target, max_jobs or int(os.environ.get("SURGPROOF_MAX_JOBS", "1")))
    outline_pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="outline")
    model_status = ModelStatus()
    auth.setup_code()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        sweep_leftovers(store.jobs_dir)
        # Jobs that were queued when the server last stopped have no worker: give them one.
        for job_id in store.list_ids():
            job_dir = store.jobs_dir / job_id
            status = read_json(job_dir / "status.json", {}) or {}
            video = pipeline_input(job_dir)
            with contextlib.suppress(OSError):
                drop_upload_after_ingest(job_dir)  # a server that died mid-job left it behind
            if (normalise_stage(status.get("stage")) == "queued" and not status.get("error") and video
                    and worker_pid(job_dir) is None):
                options = (read_json(job_dir / "job.json", {}) or {}).get("options") or {}
                runner.submit(job_id, video, job_dir, options)

        stop_sweeping = threading.Event()

        def retention_loop() -> None:
            while True:
                try:
                    sweep_expired(store, runner)
                except Exception:  # noqa: BLE001
                    log.exception("the retention sweep failed")
                if stop_sweeping.wait(RETENTION_SWEEP_SECONDS):
                    return

        threading.Thread(target=retention_loop, daemon=True, name="retention").start()
        yield
        stop_sweeping.set()
        await asyncio.to_thread(runner.shutdown)
        outline_pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="SurgProof", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store
    app.state.runner = runner
    app.state.auth = auth
    app.add_middleware(Guard, auth=auth)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(ALLOWED_HOSTS))

    @app.exception_handler(json.JSONDecodeError)
    async def corrupt_job_file(_request: Request, exc: json.JSONDecodeError) -> JSONResponse:
        # A damaged JSON file in a job folder. Say so, rather than a bare 500 that reads as a server bug.
        log.error("refused a request: %s", exc)
        return JSONResponse({"detail": f"a file in this job is damaged or invalid, so this was refused: {exc}"}, status_code=500)

    # ------------------------------------------------------------ session

    @app.get("/api/auth/state")
    def auth_state(request: Request) -> dict:
        return {"setup_required": auth.account() is None, "user": request.state.user}

    @app.post("/api/auth/setup")
    def auth_setup(body: AuthBody) -> Response:
        try:
            token = auth.setup(body.name, body.password, body.setup_code)
        except AuthError as exc:
            return JSONResponse({"detail": exc.message}, status_code=exc.status)
        return _session_response({"user": auth.user_for(token)}, token)

    @app.post("/api/auth/login")
    def auth_login(body: AuthBody) -> Response:
        try:
            token = auth.login(body.password)
        except AuthError as exc:
            return JSONResponse({"detail": exc.message}, status_code=exc.status)
        return _session_response({"user": auth.user_for(token)}, token)

    @app.post("/api/auth/logout")
    def auth_logout(request: Request) -> Response:
        auth.logout(request.state.token)
        response = JSONResponse({"user": None})
        response.delete_cookie(COOKIE_NAME, path="/")
        return response

    # ------------------------------------------------------------ jobs

    @app.get("/api/health")
    def health() -> dict:
        disk = disk_report(store.jobs_dir)
        tools = {name: bool(shutil.which(name)) for name in ("ffmpeg", "ffprobe")}
        model = model_status.get()
        return {
            "ok": disk["ok"] and all(tools.values()) and model["ok"],
            "disk": disk,
            "tools": tools,
            "model": model,
            "max_seconds": MAX_SECONDS,
            "max_bytes": MAX_BYTES,
        }

    @app.get("/api/notice")
    def notice() -> dict:
        return {"text": UPLOAD_NOTICE, "sha256": NOTICE_SHA256, "footage_sources": list(FOOTAGE_SOURCES)}

    def finish_upload(sink: UploadStream, job_id: str, staging: Path, user: str) -> dict:
        fields = sink.fields
        if fields.get("notice_sha256") != NOTICE_SHA256 or fields.get("rights_attested") not in ("yes", "true", "on"):
            raise UploadRejected(422, "confirm the upload notice first: rights_attested and the notice_sha256 "
                                      "from /api/notice are required")
        if fields.get("footage_source") not in FOOTAGE_SOURCES:
            raise UploadRejected(422, "this build takes public or simulator footage only: "
                                      f"footage_source must be one of {', '.join(FOOTAGE_SOURCES)}")
        options: dict = {}
        if fields.get("kf_fps"):
            try:
                options["kf_fps"] = float(fields["kf_fps"])
            except ValueError:
                raise UploadRejected(422, "kf_fps must be a number") from None
            if not 0 < options["kf_fps"] <= MAX_KF_FPS:
                raise UploadRejected(422, f"kf_fps must be above 0 and at most {MAX_KF_FPS:g}")
        if sink.dest is None:
            raise UploadRejected(422, "send the video as a multipart field named 'file'")
        if sink.written == 0:
            raise UploadRejected(422, "the uploaded file is empty")
        try:
            probe = probe_video(sink.dest)
        except ValueError as exc:
            raise UploadRejected(422, str(exc)) from exc
        if probe["duration"] > MAX_SECONDS + 0.5:
            raise UploadRejected(422, f"video is {probe['duration'] / 60:.1f} minutes, the limit is "
                                      f"{MAX_SECONDS // 60} minutes")
        record = {
            "id": job_id,
            "created": now_iso(),
            "upload": sink.dest.name,
            "bytes": sink.written,
            "probe": probe,
            "options": options,
            "attestation": {
                "by": user,
                "at": now_iso(),
                "footage_source": fields["footage_source"],
                "notice": UPLOAD_NOTICE,
                "notice_sha256": NOTICE_SHA256,
            },
        }
        write_json_atomic(staging / "job.json", record)
        write_status(staging, stage="queued", progress=0.0, message="Waiting for a worker", started=None, error=None)
        job_dir = store.jobs_dir / job_id
        os.replace(staging, job_dir)
        runner.submit(job_id, job_dir / sink.dest.name, job_dir, options)
        return job_summary(store, job_dir, runner)

    @app.post("/api/jobs", status_code=201)
    async def create_job(request: Request) -> dict:
        content_type, params = parse_options_header(request.headers.get("content-type", ""))
        boundary = params.get(b"boundary")
        if content_type != b"multipart/form-data" or not boundary:
            raise HTTPException(422, "send the video as a multipart field named 'file'")
        length = request.headers.get("content-length")
        size = int(length) if length and length.isdigit() else None
        if size is not None and size > MAX_BYTES + MAX_JSON_BYTES:
            raise HTTPException(413, f"the file is larger than the {MAX_BYTES // 1024**3} GB limit")
        await run_in_threadpool(check_disk_for_upload, store.jobs_dir, size)

        job_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        staging = store.jobs_dir / f".upload-{job_id}"
        staging.mkdir()
        sink = UploadStream(boundary, staging, store.jobs_dir)
        try:
            async for chunk in request.stream():
                if chunk:
                    await run_in_threadpool(sink.write, chunk)
            sink.close()
            return await run_in_threadpool(finish_upload, sink, job_id, staging, request.state.user)
        except UploadRejected as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except FormParserError as exc:
            raise HTTPException(422, "this upload could not be read as a multipart form") from exc
        except ClientDisconnect as exc:
            raise HTTPException(400, "the upload was interrupted") from exc
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise HTTPException(507, "the disk filled up during this upload") from exc
            raise
        finally:
            sink.close()
            if staging.exists():
                await run_in_threadpool(shutil.rmtree, staging, True)

    @app.get("/api/jobs")
    def list_jobs() -> dict:
        jobs = [job_summary(store, store.jobs_dir / job_id, runner) for job_id in store.list_ids()]
        jobs.sort(key=lambda j: j.get("created") or "", reverse=True)
        return {"jobs": jobs}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        summary = job_summary(store, job_dir, runner)
        status = read_json(job_dir / "status.json", {}) or {}
        evidence = stage_evidence(store, job_dir)
        summary["stages"] = stage_list(status, summary["state"], evidence)
        summary["decisions"] = decision_tally(store, job_dir)
        summary["meta"] = {k: v for k, v in store.meta(job_dir).items() if k != "original_name"}
        if summary["state"] == "done":
            with store.lock(job_id):
                view = final_pass_view(store, job_dir, include_items=False)
            view["release_job"] = release_job(job_dir, runner)
            summary["final_pass"] = {k: view[k] for k in (
                "total", "reviewed", "pending", "released", "n_amber", "n_audit", "n_audit_pool", "n_escalation",
                "audit_by_stratum", "audit_disagreements", "release_job")}
            summary["outputs"] = {name: {"bytes": (job_dir / name).stat().st_size}
                                  for name in DOWNLOADABLE if (job_dir / name).exists()}
            summary["released_counts"] = released_counts(store, job_dir) if view["released"] else None
        return summary

    @app.get("/api/jobs/{job_id}/release-check")
    def release_check(job_id: str) -> dict:
        """Re-hash the released files against the signed manifest, so an edit after release shows."""
        job_dir = store.job_dir(job_id)
        if not (job_dir / "released_coco.json").exists():
            raise HTTPException(409, "this job has not been released")
        result = _export().verify_release(job_dir)
        return {"ok": bool(result.get("ok")), "problems": result.get("problems") or []}

    @app.post("/api/jobs/{job_id}/retry")
    def retry_job(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        with store.lock(job_id):
            summary = job_summary(store, job_dir, runner)
            if release_running(job_dir, runner):
                raise HTTPException(409, "this job is being released")
            if summary["released"]:
                raise HTTPException(409, "this job has been released: a rerun would overwrite what was signed off")
            if read_jsonl(job_dir / "final_pass.jsonl"):
                raise HTTPException(409, "a person has already reviewed this job: a rerun would discard that review")
            if summary["state"] not in ("failed", "stalled"):
                raise HTTPException(409, f"job is {summary['state']}, only failed or stalled jobs can be retried")
            video = pipeline_input(job_dir)
            if video is None:
                raise HTTPException(409, "the video for this job is not available to retry")
            try:
                if job_id in runner.active:
                    runner.stop(job_id, job_dir)
                elif not kill_worker_group(job_dir):
                    raise RuntimeError("the old worker would not stop")
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            options = (read_json(job_dir / "job.json", {}) or {}).get("options") or {}
            write_status(job_dir, stage="queued", progress=0.0, message="Waiting for a worker", error=None)
            runner.submit(job_id, video, job_dir, options)
            return job_summary(store, job_dir, runner)

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        with store.lock(job_id):
            if release_running(job_dir, runner):
                raise HTTPException(409, "this job is being released: delete it when that has finished")
            try:
                if job_id in runner.active:
                    runner.stop(job_id, job_dir)
                elif not kill_worker_group(job_dir):
                    raise RuntimeError("the worker would not stop")
            except RuntimeError as exc:
                raise HTTPException(409, str(exc)) from exc
            removed = delete_job_files(store, job_dir)
        return {"deleted": job_id, "cache_entries_removed": removed}

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
        """An allowlist of exact names, so a new file in a job folder is private until someone decides otherwise."""
        job_dir = store.job_dir(job_id).resolve()
        gated = path in RELEASE_GATED_FILES
        if not gated and not FRAME_FILE_RE.match(path):
            raise HTTPException(404, "no such file")
        if gated and not (job_dir / "released_coco.json").is_file():
            raise HTTPException(403, "locked until a person completes the final pass and releases this dataset")
        target = job_dir / path
        if target.is_symlink() or not target.is_file() or not target.resolve().is_relative_to(job_dir):
            raise HTTPException(404, "no such file")
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
        summary = job_summary(store, job_dir, runner)
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
        """Boxes and mask outlines for frames in [start, end). Only what the pipeline wrote, and whose it is."""
        job_dir = store.job_dir(job_id)
        if end <= start or end - start > 600:
            raise HTTPException(422, "ask for between 1 and 600 frames")
        meta = store.meta(job_dir)
        width, height = meta.get("width"), meta.get("height")
        by_frame = store.cache.get("tracks_by_frame", job_dir / "tracks.jsonl",
                                   lambda: _group_tracks(read_jsonl(job_dir / "tracks.jsonl")), {})
        verdicts = latest_verdicts(job_dir)
        prop_prefix = "prop_"  # how surgproof.export names a propagated frame queued for audit
        frames: dict[str, list[dict]] = {}
        pending: list[tuple[dict, Path]] = []
        for frame in range(start, end):
            out = []
            for row in by_frame.get(frame, ()):
                label = row.get("label")
                keyframe = row.get("keyframe")
                sources = row.get("source_kfs") or ([keyframe] if keyframe else [])
                audited = verdicts.get((f"{prop_prefix}{frame:06d}", label))  # this exact tracked frame, audited
                rulings = [verdicts[(kf, label)] for kf in sources if (kf, label) in verdicts]
                if audited:
                    rulings.append(audited)
                overruled = [r for r in rulings if r.get("action") in (ACTION_REJECT, ACTION_CORRECT)]
                box, polygons = row.get("box"), None
                if overruled:
                    # Same rule as the released dataset: an overruled mask, and what flowed from it, is gone.
                    exact = (keyframe, f"{prop_prefix}{frame:06d}")
                    drawn = next((r for r in overruled if r.get("kf") in exact and r.get("polygon")), None)
                    if drawn is None:
                        continue
                    polygons = [drawn["polygon"]]
                    xs, ys = zip(*drawn["polygon"])
                    box = [min(xs), min(ys), max(xs), max(ys)]
                # A person saw this exact keyframe label, or the AI drew it, or optical flow carried it here.
                seen_by_human = audited is not None or (bool(keyframe) and (keyframe, label) in verdicts)
                origin = "human" if seen_by_human else ("ai_keyframe" if keyframe else "propagated")
                # "human" means this exact annotation was seen, on a tracked frame as much as on a keyframe.
                entry = {
                    "label": label,
                    "box": box,
                    "propagated": bool(row.get("propagated")),
                    "status": row.get("status"),
                    "origin": origin,
                    "derived_from_reviewed": origin == "propagated" and bool(rulings),
                    "unsure": row.get("status") == HELD_STATUS and not rulings,
                    "polygons": polygons or [],
                }
                if polygons is None:
                    png = (job_dir / "masks" / f"{keyframe}_{label}.png" if keyframe
                           else job_dir / "tracks" / f"{frame}_{label}.png")
                    pending.append((entry, png))
                out.append(entry)
            if out:
                frames[str(frame)] = out

        def outline(item: tuple[dict, Path]) -> None:
            entry, png = item
            if png.exists():
                entry["polygons"] = mask_polygons(png, width, height)

        # OpenCV releases the GIL, and at 1080p decoding one after another cannot keep up with playback.
        list(outline_pool.map(outline, pending))
        return {"start": start, "end": end, "frames": frames}

    # ------------------------------------------------------------ final pass

    @app.get("/api/jobs/{job_id}/final-pass")
    def get_final_pass(job_id: str) -> dict:
        job_dir = store.job_dir(job_id)
        require_done(job_dir)
        with store.lock(job_id):
            view = final_pass_view(store, job_dir)
        view["release_job"] = release_job(job_dir, runner)
        return view

    def begin_release(job_id: str, job_dir: Path, reviewer: str) -> None:
        """Everything that can be refused at once is refused here. The slow part then runs on its own."""
        export = _export()
        state = export.final_pass_state(job_dir)
        if state.get("empty"):
            raise HTTPException(409, "release is locked: nothing to review, and an empty final pass never releases")
        if state["locked"]:
            remaining = len(state.get("remaining") or [])
            raise HTTPException(409, f"release is locked: {remaining} of {state['n_required']} final-pass items are unresolved")
        playable = playable_video(job_dir)
        check_disk_for_upload(store.jobs_dir, playable.stat().st_size if playable else None)
        try:
            runner.start_release(job_id, job_dir, reviewer)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/jobs/{job_id}/final-pass")
    def post_final_pass(job_id: str, body: FinalPassBody, request: Request, response: Response) -> dict:
        job_dir = store.job_dir(job_id)
        require_done(job_dir)
        export = _export()
        reviewer = request.state.user  # from the session: any name in the request body is ignored
        with store.lock(job_id):
            if (job_dir / "released_coco.json").exists():
                raise HTTPException(409, "this dataset has already been released")
            if release_running(job_dir, runner):
                raise HTTPException(409, "this job is being released, so nothing can change")
            if body.action == "release":
                begin_release(job_id, job_dir, reviewer)
                if not body.wait:
                    response.status_code = 202
                    view = final_pass_view(store, job_dir, include_items=False)
                    view["release_job"] = release_job(job_dir, runner)
                    return view
            elif body.action in (ACTION_APPROVE, ACTION_REJECT, ACTION_CORRECT):
                kf, label = body.kf or "", body.label or ""
                decision = store.decisions(job_dir).get((kf, label), {})
                if (body.action == ACTION_APPROVE and not has_mask(job_dir, kf, label)
                        and not can_approve(job_dir, kf, label, proposal_for(store, job_dir, kf, label, decision)["present"])):
                    raise HTTPException(422, "there is no AI mask to approve here: mark it not present, "
                                             "or draw a correction")
                row = {"kf": body.kf, "label": body.label, "action": body.action,
                       "reviewer": reviewer, "ts": now_iso()}
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
            if body.action != "release":
                view = final_pass_view(store, job_dir, include_items=False)
                view["release_job"] = release_job(job_dir, runner)
                return view
        # wait=true: outside the job lock, so progress and everything else stay readable meanwhile.
        while release_running(job_dir, runner) or job_id in runner.active:  # the worker is gone, not just finished
            time.sleep(RELEASE_WAIT_POLL_SECONDS)
        outcome = release_job(job_dir, runner) or {}
        if outcome.get("state") != "done":
            status = 409 if outcome.get("locked") else 500
            raise HTTPException(status, f"release failed: {outcome.get('error')}")
        with store.lock(job_id):
            view = final_pass_view(store, job_dir, include_items=False)
        view["release_job"] = outcome
        return view

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
    if len(sys.argv) == 4 and sys.argv[1] == "--run-release":
        _run_release(sys.argv[2], sys.argv[3])
        return
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    port = int(os.environ.get("PORT", "8600"))
    # Before anything touches the jobs folder: a second server must not sweep the first one's uploads.
    with socket.socket() as probe:
        try:
            probe.bind((BIND_HOST, port))
        except OSError as exc:
            raise SystemExit(f"cannot use {BIND_HOST}:{port}: {exc}. Is SurgProof already running?") from None
    app = create_app()
    code = app.state.auth.setup_code()
    if code:
        print(f"\nFirst run. Open http://localhost:{port} and create your account with this setup code: {code}\n",
              flush=True)
    uvicorn.run(app, host=BIND_HOST, port=port, server_header=False)


if __name__ == "__main__":
    main()
