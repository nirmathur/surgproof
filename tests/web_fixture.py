"""Builds a synthetic job folder in the stage I/O contract, for the web lane's tests only.

Everything here is test data, drawn over ffmpeg's test pattern. Nothing from this file reaches the
product: the server reads whatever is on disk in the job folder, and tests point it at a temp folder.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, SECONDS = 320, 180, 10, 3
LABELS = ("grasper", "gallbladder")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def make_video(path: Path, seconds: float = SECONDS, fps: int = FPS, width: int = W, height: int = H) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc=duration={seconds}:size={width}x{height}:rate={fps}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast", "-movflags", "+faststart", str(path)],
        check=True, capture_output=True)
    return path


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


def make_contract_files(job: Path) -> None:
    """Stage outputs through narration, plus annotated.mp4. export.run builds the dataset from these."""
    make_video(job / "source.mp4")
    keyframes = [{"kf": f"kf_{i:05d}", "frame_index": i * FPS, "t": float(i)} for i in range(SECONDS)]
    meta = {"fps": FPS, "width": W, "height": H, "n_frames": FPS * SECONDS, "kf_fps": 1.0, "keyframes": keyframes,
            "duration": float(SECONDS), "source": "source.mp4", "original_name": "clip.mp4"}
    (job / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    frames = job / "frames"
    masks = job / "masks"
    tracks = job / "tracks"
    for folder in (frames, masks, tracks):
        folder.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(job / "source.mp4"))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % FPS == 0:
            cv2.imwrite(str(frames / f"kf_{index // FPS:05d}.jpg"), frame)
        index += 1
    cap.release()

    detect, verify, decisions, phases, track_rows = [], [], [], [], []
    for i, kf in enumerate(keyframes):
        name = kf["kf"]
        phases.append({"kf": name, "phase": "preparation" if i < 2 else "clipping_cutting", "conf": 0.9})
        for j, label in enumerate(LABELS):
            box = [20 + 10 * i + 120 * j, 30, 90 + 10 * i + 120 * j, 100]
            polygon = [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]]
            # kf_00001 / gallbladder is the one disputed label in this synthetic job.
            status = "AMBER" if (i == 1 and label == "gallbladder") else "accept"
            detect.append({"kf": name, "label": label, "present": True, "conf": 0.8, "polygon": polygon,
                           "model": "test:detect", "prompt_hash": "t1", "raw_ref": "t"})
            verify.append({"kf": name, "label": label, "present": status != "AMBER", "box": box,
                           "model": "test:verify", "prompt_hash": "t2"})
            decisions.append({"kf": name, "label": label, "status": status,
                              "final_present": True if status == "accept" else None, "rounds": [],
                              "reason": "the two passes disagreed" if status == "AMBER" else "both agree",
                              "polygon": polygon if status == "accept" else None})
            cv2.imwrite(str(masks / f"{name}_{label}.png"), _mask(box[0], box[1], box[2], box[3]))
    for frame_index in range(FPS * SECONDS):
        kf_pos = frame_index // FPS
        is_key = frame_index % FPS == 0
        for j, label in enumerate(LABELS):
            box = [20 + 10 * kf_pos + 120 * j, 30, 90 + 10 * kf_pos + 120 * j, 100]
            status = "AMBER" if (kf_pos == 1 and label == "gallbladder") else "accept"
            name = keyframes[kf_pos]["kf"]
            if not is_key:
                cv2.imwrite(str(tracks / f"{frame_index}_{label}.png"), _mask(box[0], box[1], box[2], box[3]))
            track_rows.append({"frame_index": frame_index, "label": label, "box": box, "box_raw": box,
                               "propagated": not is_key, "status": status, "source_kfs": [name],
                               "fb_overlap": None if is_key else 0.95, "flow_check": "keyframe" if is_key else "ok",
                               "keyframe": name if is_key else None, "t": frame_index / FPS})
    write_jsonl(job / "detect.jsonl", detect)
    write_jsonl(job / "verify.jsonl", verify)
    write_jsonl(job / "decisions.jsonl", decisions)
    write_jsonl(job / "phase.jsonl", phases)
    write_jsonl(job / "tracks.jsonl", track_rows)
    (job / "narration.json").write_text(json.dumps({"segments": [
        {"t0": 0.0, "t1": 2.0, "phase": "preparation", "text": "The grasper is visible. The gallbladder is held.",
         "evidence": ["kf_00000", "kf_00001"]},
        {"t0": 2.0, "t1": 3.0, "phase": "clipping_cutting", "text": "Unsure about the duct here.",
         "evidence": ["kf_00002"]},
    ]}), encoding="utf-8")
    shutil.copy(job / "source.mp4", job / "annotated.mp4")


def write_done(job: Path) -> None:
    (job / "status.json").write_text(json.dumps({
        "stage": "done", "progress": 1.0, "stage_progress": 1.0, "message": "done", "started": "2026-11-02T18:00:00+00:00",
        "updated": "2026-11-02T18:01:00+00:00", "error": None, "cost_usd": 0.0}), encoding="utf-8")


def finished_job(jobs_dir: Path, job_id: str = "job-done") -> Path:
    """A complete job on disk, with the dataset built by the real surgproof.export."""
    from surgproof import export

    job = jobs_dir / job_id
    job.mkdir(parents=True)
    make_contract_files(job)
    write_jsonl(job / "masks.jsonl", [
        {"kf": row["kf"], "label": row["label"], "status": row["status"], "path": f"masks/{row['kf']}_{row['label']}.png",
         "seed": "polygon", "fallback": False, "box": None}
        for row in map(json.loads, (job / "decisions.jsonl").read_text().splitlines())])
    export.run(job)
    write_done(job)
    return job


# ---- a real client: same sign-in and attestation route a browser uses ----

TEST_USER = "Test Reviewer"
TEST_PASSWORD = "test-password-1"


def local_client(app):
    """base_url matters: the server refuses any Host other than this machine's."""
    from fastapi.testclient import TestClient

    return TestClient(app, base_url="http://localhost")


def sign_in(client, jobs: Path, name: str = TEST_USER, password: str = TEST_PASSWORD) -> None:
    """Create the account with the setup code on first use, as a person would, otherwise log in."""
    if client.get("/api/auth/state").json()["setup_required"]:
        code = (Path(jobs) / ".setup_code").read_text(encoding="utf-8")
        res = client.post("/api/auth/setup", json={"name": name, "password": password, "setup_code": code})
    else:
        res = client.post("/api/auth/login", json={"password": password})
    assert res.status_code == 200, res.text


def attestation(**overrides) -> dict:
    from surgproof.server import NOTICE_SHA256

    return {"rights_attested": "yes", "footage_source": "public", "notice_sha256": NOTICE_SHA256, **overrides}


# ---- stand-ins for surgproof.pipeline.run, imported by name in the server's worker process ----


def pipeline_ok(video: str, out: str, **options) -> None:
    """Pretend pipeline that proves the worker got the right arguments and writes a done status."""
    job = Path(out)
    (job / "pipeline_called.json").write_text(json.dumps({"video": video, "options": options}), encoding="utf-8")
    assert Path(video).is_file()
    time.sleep(0.2)
    (job / "status.json").write_text(json.dumps({"stage": "render", "progress": 0.9, "error": None}), encoding="utf-8")


def pipeline_boom(video: str, out: str, **options) -> None:
    raise RuntimeError("synthetic failure in the pipeline")


def pipeline_done_with_nothing(video: str, out: str, **options) -> None:
    """Exits cleanly having written no terminal status: must never read as running."""


def pipeline_hang(video: str, out: str, **options) -> None:
    """A stuck job with a child process, like a pipeline waiting on claude or ffmpeg."""
    child = subprocess.Popen(["sleep", "600"])
    (Path(out) / "child.pid").write_text(str(child.pid), encoding="utf-8")
    (Path(out) / "status.json").write_text(json.dumps({"stage": "detect", "progress": 0.1, "error": None}), encoding="utf-8")
    time.sleep(600)


def pipeline_die_with_child(video: str, out: str, **options) -> None:
    """Dies leaving a child running, like a crash while waiting on claude."""
    import os
    import signal

    child = subprocess.Popen(["sleep", "600"])
    (Path(out) / "child.pid").write_text(str(child.pid), encoding="utf-8")
    os.kill(os.getpid(), signal.SIGKILL)


def pipeline_die_hard(video: str, out: str, **options) -> None:
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)
