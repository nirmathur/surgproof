"""All four stages on a real CholecSeg8k clip, seeded by the AI's own stored masks (out/dense_smooth), no model calls.

Only *_endo.png frames and AI outputs are read. No ground-truth file is opened. Quality here means agreement with
the AI's own per-frame masks at frames that keyframes did not supply; scoring against ground truth is surgproof/eval.py's job.
Skips on a clean clone, where neither the dataset nor out/ exists.
"""
import glob, json, os, subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from surgproof import export as X, masks as M, propagate as P, render as R

ROOT = Path(__file__).resolve().parent.parent
CLIP = "video12_15750"
FRAMES = sorted(glob.glob(str(ROOT / f"raw/CholecSeg8k/video12/{CLIP}/*_endo.png")))
AI_DIR = ROOT / "out/dense_smooth" / CLIP
pytestmark = pytest.mark.skipif(not FRAMES or not AI_DIR.exists(), reason="needs local CholecSeg8k frames and out/dense_smooth")
LABELS = ["gallbladder", "liver", "grasper", "l_hook"]
KF_STEP = 25


def ai_mask(i, label):
    name = os.path.basename(FRAMES[i]).replace("_endo.png", "")
    p = AI_DIR / f"{name}_{label}.png"
    return cv2.imread(str(p), 0) > 0 if p.exists() else None


def iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


@pytest.fixture(scope="module")
def job(tmp_path_factory):
    job = tmp_path_factory.mktemp("real") / "job"
    (job / "frames").mkdir(parents=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "25", "-pattern_type", "glob", "-i",
                    str(ROOT / f"raw/CholecSeg8k/video12/{CLIP}/*_endo.png"), "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                    str(job / "source.mp4")], check=True)
    h, w = cv2.imread(FRAMES[0]).shape[:2]
    kfs, det, dec, ver = [], [], [], []
    for n, i in enumerate(range(0, len(FRAMES), KF_STEP)):
        kf = f"kf_{n:05d}"
        kfs.append({"kf": kf, "frame_index": i, "t": i / 25})
        cv2.imwrite(str(job / "frames" / f"{kf}.jpg"), cv2.imread(FRAMES[i]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        for label in LABELS:
            m = ai_mask(i, label)
            present = m is not None and m.any()
            poly = None
            if present:
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                poly = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 3, True)[:, 0, :].tolist()
            det.append({"kf": kf, "label": label, "present": bool(present), "conf": 0.9, "polygon": poly, "model": "stored-ai-mask", "prompt_hash": "dense_smooth", "raw_ref": str(AI_DIR)})
            dec.append({"kf": kf, "label": label, "status": "accept" if present else "reject", "final_present": bool(present), "rounds": []})
    (job / "meta.json").write_text(json.dumps({"fps": 25, "width": w, "height": h, "n_frames": len(FRAMES), "kf_fps": 1, "keyframes": kfs}))
    for name, rows in [("detect", det), ("decisions", dec), ("verify", ver)]:
        (job / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (job / "phase.jsonl").write_text("".join(json.dumps({"kf": k["kf"], "phase": "GallbladderDissection", "conf": 0.8}) + "\n" for k in kfs))
    M.run(job)
    P.run(job)
    return job


def test_masks_stay_close_to_their_seed_on_real_frames(job):
    rows = [r for r in M.read_jsonl(job / "masks.jsonl") if r["path"]]
    assert rows
    fallback = sum(1 for r in rows if r["fallback"])
    assert fallback <= len(rows) // 2, f"{fallback}/{len(rows)} fell back to raw seed"
    for r in rows:
        i = next(k["frame_index"] for k in json.loads((job / "meta.json").read_text())["keyframes"] if k["kf"] == r["kf"])
        assert iou(M.read_mask(job / r["path"]), ai_mask(i, r["label"])) > 0.3


def test_propagation_agrees_with_the_ais_own_per_frame_masks(job):
    scores, baseline = [], []
    for r in M.read_jsonl(job / "tracks.jsonl"):
        if r["propagated"]:
            ai = ai_mask(r["frame_index"], r["label"])
            if ai is not None and ai.any():
                scores.append(iou(M.read_mask(M.track_path(job, r["frame_index"], r["label"])), ai))
    assert len(scores) > 100
    assert np.mean(scores) > 0.5, f"mean agreement {np.mean(scores):.3f} over n={len(scores)}"


def test_all_frames_have_tracks_and_render_matches_the_clip(job):
    assert {r["frame_index"] for r in M.read_jsonl(job / "tracks.jsonl")} == set(range(len(FRAMES)))
    res = R.run(job)
    assert res["n_frames"] == len(FRAMES) == R.count_video_frames(job / "annotated.mp4")


def test_export_of_the_real_clip_has_keyframe_and_propagated_annotations(job):
    X.run(job)
    c = json.loads((job / "dataset_coco.json").read_text())
    kinds = [a["attributes"]["propagated"] for a in c["annotations"]]
    assert any(kinds) and not all(kinds)
    assert len({a["image_id"] for a in c["annotations"]}) == len(FRAMES)
    X.final_pass_queue(job)
    assert X.final_pass_state(job)["locked"]
