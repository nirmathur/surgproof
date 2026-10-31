"""End-to-end mock pipeline runner and pytest suite for Surgical Atlas.

Usage:
  python mock_test.py                          # run full pipeline end-to-end in isolated out/mock_run
  python mock_test.py all                      # same as above
  python mock_test.py med                      # run single step (uses out/mock_run unless --in-place)
  python mock_test.py label manual train,calib,test 0
  python mock_test.py accept
  python mock_test.py student
  python mock_test.py narrate video01
  python mock_test.py atlas
  pytest mock_test.py                          # run as pytest test
"""
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, ".")
os.environ.setdefault("GEMINI_API_KEY", "mock-key-for-testing")

import config as C
import gem

# Reverse map: class_id -> watershed pixel value
CLASS_TO_WATERSHED = {c: v for v, c in C.WATERSHED_VALUE_TO_CLASS.items()}
LABEL_KEYS = list(C.LABELS.keys())

# Lookup populated by prepare_mock_workspace so fake() can simulate realistic labeller/verifier correlation
_FRAME_GT = {}


def _det_float(seed_str, lo=0.0, hi=1.0):
    h = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return lo + (hi - lo) * ((h % 10000) / 10000.0)


async def fake(prompt, *, model=None, images=(), schema=None, tools=None, video_url=None,
               temperature=0, tag="", sample=0):
    n = getattr(schema, "__name__", None)
    sig = hashlib.md5((prompt + str(sample) + str(tag) + str(len(images))).encode()).hexdigest()[:10]
    img_hash = hashlib.md5(images[0][1]).hexdigest()[:12] if images else sig

    if n == "FrameLabels":
        gt_row = _FRAME_GT.get(img_hash)
        items = []
        for idx_k, k in enumerate(LABEL_KEYS):
            if gt_row is not None:
                truth = int(gt_row[k])
                # Deterministic ~70% accuracy on single pass; errors get low conf so verifier filters them out
                roll = _det_float(f"{img_hash}:{k}:{sample}:err")
                is_correct = roll < (0.72 if sample == 0 else 0.76)
                present = bool(truth if is_correct else 1 - truth)
                conf = 0.92 if is_correct else 0.52
            else:
                present = _det_float(f"{sig}:{k}:{sample}:p") < 0.5
                conf = round(_det_float(f"{sig}:{k}:{sample}:c", 0.55, 0.95), 2)
            items.append({
                "key": k,
                "present": present,
                "confidence": conf,
                "box_2d": [100, 100, 400, 400] if present else [],
                "evidence": f"visual cue for {k}",
            })
        return {"items": items}

    if n == "CropID":
        # Extract which structure name is correct when high-conf, or pick deterministically
        names = [v[1] for v in C.LABELS.values()]
        # In verify_box, prompt asks about a cropped region; match the claimed key when crop came from valid detection
        idx_choice = int(hashlib.md5(img_hash.encode()).hexdigest()[:4], 16) % len(names)
        # Check if we recorded the active crop match hint
        matched_name = _FRAME_GT.get(f"crop:{img_hash}", names[idx_choice])
        return {"answer": matched_name, "confidence": 0.9}

    if n == "Critique":
        # Parse claims from prompt if present to correlate p_wrong with correctness
        rebuttals = []
        gt_row = _FRAME_GT.get(img_hash)
        claims_by_key = {}
        if "Claims:" in prompt:
            try:
                claims_list = json.loads(prompt.split("Claims:", 1)[1].strip())
                claims_by_key = {c["key"]: (c["claim"] == "present") for c in claims_list}
            except Exception:
                pass
        for k in LABEL_KEYS:
            if gt_row is not None and k in claims_by_key:
                is_correct = int(claims_by_key[k]) == int(gt_row[k])
                p_wrong = 0.06 if is_correct else 0.68
            else:
                p_wrong = round(_det_float(f"{img_hash}:{k}:pw", 0.05, 0.45), 2)
            rebuttals.append({"key": k, "argument": "checked anatomical landmarks", "p_wrong": p_wrong})
        return {"rebuttals": rebuttals}

    if n == "Verdict":
        gt_row = _FRAME_GT.get(img_hash)
        target_key = None
        for k in LABEL_KEYS:
            if f"'{k}'" in prompt:
                target_key = k
                break
        if gt_row is not None and target_key is not None:
            # Argon reviser resolves a fraction of uncertain cases accurately
            truth = bool(gt_row[target_key])
            resolve = _det_float(f"{img_hash}:{target_key}:argon") < 0.65
            return {
                "present": truth if resolve else (not truth),
                "confidence": 0.88 if resolve else 0.45,
                "reasoning": "verified against hepatocystic landmarks",
            }
        return {"present": _det_float(f"{sig}:v") < 0.5, "confidence": 0.85, "reasoning": "looked closely"}

    if n == "Manual":
        return {
            "procedure": "laparoscopic cholecystectomy",
            "phases": ["preparation", "calot_triangle_dissection", "clipping_and_cutting", "gallbladder_dissection"],
            "structures": [
                {
                    "key": k,
                    "name": C.LABELS[k][1],
                    "appearance": f"characteristic endoscopic appearance of {k}",
                    "where_found": "hepatocystic surgical field",
                    "when_visible": "dissection phase",
                    "confused_with": ["adjacent tissue"],
                    "how_to_tell_apart": f"distinctive shape and colour of {k}",
                }
                for k in LABEL_KEYS
            ],
            "critical_events": ["critical_view_of_safety"],
            "general_rules": ["verify hepatocystic triangle before clipping"],
        }

    if n == "VideoLesson":
        return {"rules": ["clear hepatocystic triangle"], "moments": ["00:10 critical view of safety"]}

    if n == "Line":
        return {"narration": "Gallbladder retracted superiorly to expose hepatocystic triangle.", "unsure_about": []}

    if n == "Timeline":
        return {
            "phases": [{"name": "dissection", "start": "00:00", "end": "01:00", "evidence": "trocar entry", "confidence": 0.9}],
            "events": [{"time": "00:30", "event": "critical_view_of_safety", "evidence": "two structures visible", "confidence": 0.85}],
            "sightings": [{"time": "00:10", "key": "liver", "confidence": 0.92}],
        }

    if n == "Audit":
        return {"problems": [], "unverified_items": [], "overall_confidence": 0.88}

    return "research notes on surgical anatomy and safety landmarks"


def prepare_mock_workspace(base_dir: Path, n_videos: int = 12, frames_per_video: int = 3):
    """Build synthetic raw frames + watershed masks, run prepare_data.build, and create features.npy."""
    import prepare_data

    raw_dir = base_dir / "raw"
    data_dir = base_dir / "data"
    out_dir = base_dir / "out"
    for d in (raw_dir, data_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    bg_val = CLASS_TO_WATERSHED[0]
    rng = np.random.default_rng(C.SEED)

    # Create synthetic videos with deterministic class presence so every split has pos & neg for every label
    for v_idx in range(1, n_videos + 1):
        v_name = f"video{v_idx:02d}"
        clip_dir = raw_dir / "CholecSeg8k" / v_name / f"{v_name}_00080"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for f_idx in range(frames_per_video):
            frame_num = 80 + f_idx * C.FRAME_STRIDE
            endo_path = clip_dir / f"frame_{frame_num}_endo.png"
            mask_path = clip_dir / f"frame_{frame_num}_endo_watershed_mask.png"

            mask = np.full((60, 60), bg_val, dtype=np.uint8)
            img_arr = np.full((60, 60, 3), 20 + v_idx * 7 + f_idx * 19, dtype=np.uint8)

            gt_this_frame = {}
            for k_idx, (key, (cid, _)) in enumerate(C.LABELS.items()):
                # Ensure balanced positive/negative across frames in every video
                is_pos = ((v_idx + f_idx + k_idx) % 2 == 0)
                gt_this_frame[key] = int(is_pos)
                if is_pos:
                    r0, r1 = k_idx * 8, (k_idx + 1) * 8
                    mask[r0:r1, 5:35] = CLASS_TO_WATERSHED[cid]
                    img_arr[r0:r1, 5:35, k_idx % 3] = 200

            Image.fromarray(img_arr).save(endo_path)
            Image.fromarray(mask).save(mask_path)

            # Register image bytes hash for gem.img_part(endo_path) so fake() knows ground truth
            _, full_bytes = gem.img_part(endo_path)
            full_hash = hashlib.md5(full_bytes).hexdigest()[:12]
            _FRAME_GT[full_hash] = gt_this_frame

            # Also register crop bytes hash -> most likely structure name
            _, crop_bytes = gem.img_part(endo_path, crop=[100, 100, 400, 400])
            crop_hash = hashlib.md5(crop_bytes).hexdigest()[:12]
            pos_names = [C.LABELS[k][1] for k in LABEL_KEYS if gt_this_frame[k]]
            if pos_names:
                _FRAME_GT[f"crop:{crop_hash}"] = pos_names[0]

    # Run prepare_data.build to generate data/index.csv
    old_stride = C.FRAME_STRIDE
    try:
        C.FRAME_STRIDE = 1  # we only created the sampled frames
        prepare_data.build(raw_dir / "CholecSeg8k")
    finally:
        C.FRAME_STRIDE = old_stride

    idx = pd.read_csv(data_dir / "index.csv")
    # Build synthetic DINOv2-shaped (N, 768) features correlated with hidden ground truth
    dim = 768
    X = rng.normal(0.0, 0.35, size=(len(idx), dim)).astype(np.float32)
    for k_idx, key in enumerate(LABEL_KEYS):
        y = idx[f"gt_{key}"].values.astype(np.float32) * 2.0 - 1.0  # -1 or +1
        sl = slice(k_idx * 32, (k_idx + 1) * 32)
        X[:, sl] += y[:, None] * 1.25
    np.save(data_dir / "features.npy", X)
    return idx


@contextmanager
def mock_env(work_dir: Path, in_place: bool = False):
    """Redirect config paths, SQLite cache, and gem.call so mock runs never touch live artifacts."""
    import accept
    import atlas
    import label
    import med_school
    import narrate

    orig = (
        C.RAW_DIR,
        C.DATA_DIR,
        C.OUT_DIR,
        C.CACHE_DB,
        gem.TRACE,
        gem._db,
        gem.call,
        med_school.gem.call,
        label.gem.call,
        accept.gem.call,
        narrate.gem.call,
        atlas.gem.call,
    )
    mem_db = sqlite3.connect(":memory:", check_same_thread=False)
    mem_db.execute("create table if not exists cache (k text primary key, v text)")
    try:
        if not in_place:
            C.RAW_DIR = work_dir / "raw"
            C.DATA_DIR = work_dir / "data"
            C.OUT_DIR = work_dir / "out"
            C.CACHE_DB = work_dir / "cache.sqlite"
            C.DATA_DIR.mkdir(parents=True, exist_ok=True)
            C.OUT_DIR.mkdir(parents=True, exist_ok=True)
            gem.TRACE = C.OUT_DIR / "trace.jsonl"
        gem._db = mem_db
        gem.call = fake
        med_school.gem.call = fake
        label.gem.call = fake
        accept.gem.call = fake
        narrate.gem.call = fake
        atlas.gem.call = fake

        # Make verify_box return True when single-pass prediction matches ground truth
        orig_verify_box = label.verify_box

        async def mock_verify_box(path, key, box):
            _, full_bytes = gem.img_part(path)
            full_hash = hashlib.md5(full_bytes).hexdigest()[:12]
            gt_row = _FRAME_GT.get(full_hash)
            if gt_row is not None:
                is_correct = (_det_float(f"{full_hash}:{key}:0:err") < 0.72)
                return bool(is_correct), (0.91 if is_correct else 0.42)
            return await orig_verify_box(path, key, box)

        label.verify_box = mock_verify_box
        try:
            yield
        finally:
            label.verify_box = orig_verify_box
    finally:
        (
            C.RAW_DIR,
            C.DATA_DIR,
            C.OUT_DIR,
            C.CACHE_DB,
            gem.TRACE,
            gem._db,
            gem.call,
            med_school.gem.call,
            label.gem.call,
            accept.gem.call,
            narrate.gem.call,
            atlas.gem.call,
        ) = orig
        mem_db.close()


def run_step(step: str, extra_args=()):
    import accept
    import atlas
    import label
    import med_school
    import narrate
    import student

    if not (C.DATA_DIR / "index.csv").exists() or not (C.DATA_DIR / "features.npy").exists():
        prepare_mock_workspace(C.OUT_DIR.parent)

    if step == "med":
        old_argv = sys.argv[:]
        try:
            sys.argv = ["med_school.py"]
            med_school.main()
        finally:
            sys.argv = old_argv
    elif step == "label":
        if not (C.OUT_DIR / "manual.json").exists():
            run_step("med")
        mode = extra_args[0] if len(extra_args) > 0 else "manual"
        splits = extra_args[1].split(",") if len(extra_args) > 1 else ["train", "calib", "test"]
        votes = int(extra_args[2]) if len(extra_args) > 2 else 0
        limit = int(extra_args[3]) if len(extra_args) > 3 else 0
        asyncio.run(label.run(mode, splits, votes, limit))
    elif step == "accept":
        if not (C.OUT_DIR / "labels_manual.jsonl").exists():
            run_step("label", ("manual", "train,calib,test", "0"))
        asyncio.run(accept.main(True, 300))
    elif step == "student":
        if not (C.OUT_DIR / "decisions.csv").exists():
            run_step("accept")
        student.main()
    elif step == "narrate":
        if not (C.OUT_DIR / "decisions.csv").exists():
            run_step("accept")
        dec = pd.read_csv(C.OUT_DIR / "decisions.csv")
        default_vid = sorted(dec[dec.split == "test"].video.unique())[0]
        vid = extra_args[0] if len(extra_args) > 0 else default_vid
        asyncio.run(narrate.main(vid))
    elif step == "atlas":
        asyncio.run(atlas.main("laparoscopic appendectomy", "https://youtube.com/watch?v=x"))
    elif step == "all":
        run_step("med")
        run_step("label", ("manual", "train,calib,test", "0"))
        run_step("label", ("blank", "test", "0"))
        run_step("label", ("manual", "train,test", "2"))
        run_step("accept")
        run_step("student")
        dec = pd.read_csv(C.OUT_DIR / "decisions.csv")
        test_vid = sorted(dec[dec.split == "test"].video.unique())[0]
        run_step("narrate", (test_vid,))
        run_step("atlas")
        _verify_outputs(C.DATA_DIR, C.OUT_DIR, test_vid)
    else:
        raise ValueError(f"Unknown step: {step}")


def _verify_outputs(data_dir: Path, out_dir: Path, test_vid: str):
    required = [
        data_dir / "index.csv",
        data_dir / "features.npy",
        out_dir / "manual.json",
        out_dir / "research_notes.md",
        out_dir / "labels_manual.jsonl",
        out_dir / "labels_blank.jsonl",
        out_dir / "labels_manual_v2.jsonl",
        out_dir / "decisions.csv",
        out_dir / "results.json",
        out_dir / "student.json",
        out_dir / "student_probs.csv",
        out_dir / f"narration_{test_vid}.json",
        out_dir / "atlas" / "laparoscopic-appendectomy" / "dataset.json",
        out_dir / "atlas" / "laparoscopic-appendectomy" / "DATASET_CARD.md",
    ]
    for p in required:
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty output: {p}"

    stu = json.loads((out_dir / "student.json").read_text())
    assert "mean_acc_by_source" in stu and "per_label" in stu
    for src in ("single", "vote", "verified", "oracle_upper_bound"):
        assert src in stu["mean_acc_by_source"], f"Missing student source: {src}"
    assert stu["mean_acc_by_source"]["verified"] >= stu["mean_acc_by_source"]["single"], (
        f"Expected verified ({stu['mean_acc_by_source']['verified']}) >= single ({stu['mean_acc_by_source']['single']})"
    )
    print(f"\n[mock_test] ALL {len(required)} ARTIFACTS VERIFIED E2E in {out_dir}")


def test_mock_pipeline_end_to_end(tmp_path):
    """Pytest entry point: runs the entire pipeline end-to-end in tmp_path."""
    with mock_env(tmp_path, in_place=False):
        prepare_mock_workspace(tmp_path)
        run_step("all")


def main():
    import tempfile

    random.seed(0)
    args = [a for a in sys.argv[1:] if a != "--in-place"]
    in_place = ("--in-place" in sys.argv[1:]) or (os.environ.get("MOCK_IN_PLACE") == "1")
    step = args[0] if args else "all"
    extra = args[1:]
    env_dir = os.environ.get("MOCK_WORK_DIR")
    if env_dir or in_place:
        work_dir = Path(env_dir) if env_dir else C.OUT_DIR.parent
        with mock_env(work_dir, in_place=in_place):
            run_step(step, extra)
    else:
        with tempfile.TemporaryDirectory(prefix="surgatlas_mock_") as td:
            with mock_env(Path(td), in_place=False):
                run_step(step, extra)


if __name__ == "__main__":
    main()
