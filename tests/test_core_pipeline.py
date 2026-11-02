"""pipeline.py: stages 1-4, 7 and orchestration. No real model calls: vlm.ask is replaced or the mock provider is used."""
import json
import re
import subprocess
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from surgproof import pipeline as PL
from surgproof import prompts as P
from surgproof import vlm

ROOT = Path(__file__).resolve().parent.parent


def make_video(path, seconds=3, fps=25, size="320x180", codec="libx264", extra=()):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={fps}",
                    "-c:v", codec, *(["-pix_fmt", "yuv420p"] if codec == "libx264" else []), *extra, str(path)], check=True)
    return path


class Quiet:
    def update(self, *a):
        pass


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "cache.sqlite")
    return tmp_path


def job_with(tmp_path, width=320, height=180, kfs=2, scale=1.0):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    keyframes = []
    for i in range(kfs):
        kf = f"kf_{i:05d}"
        cv2.imwrite(str(job / "frames" / f"{kf}.jpg"), np.random.RandomState(i).randint(0, 255, (height, width, 3), dtype=np.uint8))
        keyframes.append({"kf": kf, "frame_index": i * 25, "t": float(i)})
    return job, {"fps": 25.0, "width": width, "height": height, "n_frames": kfs * 25, "kf_fps": 1.0, "keyframes": keyframes,
                 "duration": float(kfs), "view_scale": scale}


def det(kf="kf_00000", label="grasper", present=True, conf=0.9, polygon=((10, 10), (60, 10), (60, 60), (10, 60))):
    return {"kf": kf, "label": label, "present": present, "conf": conf, "polygon": [list(p) for p in polygon] if polygon else None}


def ver(kf="kf_00000", label="grasper", present=True, conf=0.9, box=(12, 12, 58, 58)):
    return {"kf": kf, "label": label, "present": present, "conf": conf, "box": list(box) if box else None}


# ---- ingest ---------------------------------------------------------------------------------------

def test_ingest_frame_index_is_the_exact_frame(tmp_path):
    video = make_video(tmp_path / "v.mp4")
    job = tmp_path / "j"
    job.mkdir()
    meta = PL.ingest(video, job, Quiet())
    assert meta["n_frames"] == 75 and meta["fps"] == 25 and (meta["width"], meta["height"]) == (320, 180)
    assert [k["frame_index"] for k in meta["keyframes"]] == [0, 25, 50]
    assert [k["t"] for k in meta["keyframes"]] == [0.0, 1.0, 2.0]
    assert json.load(open(job / "meta.json"))["keyframes"] == meta["keyframes"]
    cap = cv2.VideoCapture(str(job / meta["source"]))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    assert len(frames) == 75
    for k in meta["keyframes"]:
        saved = cv2.imread(str(job / "frames" / f"{k['kf']}.jpg")).astype(float)
        errors = [np.abs(saved - frames[i]).mean() for i in (k["frame_index"], max(k["frame_index"] - 1, 0), k["frame_index"] + 1)]
        assert errors[0] < 3 and errors[0] <= min(errors)


def test_ingest_non_integer_frame_rate_and_higher_kf_rate(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=2, fps="30000/1001")
    job = tmp_path / "j"
    job.mkdir()
    meta = PL.ingest(video, job, Quiet(), kf_fps=2.0)
    assert meta["fps"] == pytest.approx(29.97, abs=0.01)
    idx = [k["frame_index"] for k in meta["keyframes"]]
    assert idx == sorted(set(idx)) and len(idx) == 4 and idx[1] == 15
    assert meta["keyframes"][1]["t"] == pytest.approx(15 / meta["fps"], abs=0.001)


def test_ingest_scales_model_view_for_large_video(tmp_path):
    video = make_video(tmp_path / "v.mp4", seconds=1, size="1920x1080")
    job = tmp_path / "j"
    job.mkdir()
    meta = PL.ingest(video, job, Quiet())
    assert meta["view_scale"] == pytest.approx(1920 / 1024)
    assert cv2.imread(str(job / "frames/kf_00000.jpg")).shape[:2] == (1080, 1920)
    assert cv2.imread(str(job / "frames_view/kf_00000.jpg")).shape[:2] == (576, 1024)


def test_ingest_makes_any_codec_browser_playable(tmp_path):
    video = make_video(tmp_path / "v.mov", seconds=1, codec="mpeg4")
    job = tmp_path / "j"
    job.mkdir()
    meta = PL.ingest(video, job, Quiet())
    codec = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                            "-of", "csv=p=0", str(job / meta["source"])], capture_output=True, text=True).stdout.strip()
    assert codec == "h264,yuv420p" and meta["n_frames"] == 25


def test_ingest_refuses_too_long_missing_or_not_a_video(tmp_path):
    job = tmp_path / "j"
    job.mkdir()
    with pytest.raises(PL.PipelineError, match="limit"):
        PL.ingest(make_video(tmp_path / "v.mp4", seconds=2), job, Quiet(), max_minutes=0.01)
    with pytest.raises(PL.PipelineError, match="no such"):
        PL.ingest(tmp_path / "missing.mp4", job, Quiet())
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video")
    with pytest.raises(PL.PipelineError):
        PL.ingest(bad, job, Quiet())


# ---- geometry -------------------------------------------------------------------------------------

def test_clean_polygon_scales_clamps_and_rejects_degenerate():
    assert PL.clean_polygon([[10, 10], [20, 10], [20, 20]], 2.0, 100, 100) == [[20, 20], [40, 20], [40, 40]]
    assert PL.clean_polygon([[-5, -5], [500, -5], [500, 500]], 1.0, 100, 100) == [[0, 0], [99, 0], [99, 99]]
    assert PL.clean_polygon(None, 1.0, 100, 100) is None
    assert PL.clean_polygon([[1, 1], [2, 2]], 1.0, 100, 100) is None
    assert PL.clean_polygon([[1, 1], [2, 2], [3, 3]], 1.0, 100, 100) is None  # zero area


def test_clean_box_orders_and_rejects_empty():
    assert PL.clean_box([30, 40, 10, 20], 1.0, 100, 100) == [10, 20, 30, 40]
    assert PL.clean_box([5, 5, 5, 9], 1.0, 100, 100) is None


def test_box_iou():
    assert PL.box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert PL.box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0
    assert PL.box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_crop_region_whole_frame_when_unlocated_and_minimum_size():
    assert PL.crop_region([None, None], 0.2, 800, 400) == [0, 0, 800, 400]
    x0, y0, x1, y1 = PL.crop_region([[100, 100, 104, 104]], 0.2, 800, 400)
    assert x1 - x0 >= 199 and y1 - y0 >= 99 and x0 <= 100 and x1 >= 104
    assert PL.crop_region([[790, 390, 800, 400]], 0.2, 800, 400)[2:] == [800, 400]


# ---- the first decision ---------------------------------------------------------------------------

@pytest.mark.parametrize("d, v, expected", [
    (det(), ver(), "accept"),
    (det(present=False, polygon=None), ver(present=False, box=None), "reject"),
    (det(), ver(present=False, box=None), "AMBER"),
    (det(present=False, polygon=None), ver(), "AMBER"),
    (det(conf=0.3), ver(), "AMBER"),
    (det(), ver(conf=0.3), "AMBER"),
    (det(present=False, conf=0.3, polygon=None), ver(present=False, box=None), "AMBER"),
    (det(polygon=None), ver(), "AMBER"),
    (det(), ver(box=(200, 200, 260, 260)), "AMBER"),
    ({**det(), "present": None}, ver(), "AMBER"),
    (det(), {**ver(), "present": None}, "AMBER"),
])
def test_first_decision(d, v, expected):
    assert PL.first_decision(d, v)[0] == expected


# ---- amber loop -----------------------------------------------------------------------------------

def amber_case(tmp_path, monkeypatch, replies):
    job, meta = job_with(tmp_path, kfs=1)
    seen = []

    def fake_ask(images, prompt, schema, model=None):
        seen.append({"images": images, "prompt": prompt, "model": model, "shape": cv2.imread(str(images[0])).shape})
        reply = replies[len(seen) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    d = det(present=False, polygon=None)
    rows = [{**det(label=k, present=False, polygon=None)} for k in P.LABELS if k != "grasper"] + [d]
    vs = [{**ver(label=k, present=False, box=None)} for k in P.LABELS if k != "grasper"] + [ver()]
    out = PL.decide(job, meta, rows, vs, types.SimpleNamespace(update=lambda *a: None))
    row = next(r for r in out if r["label"] == "grasper")
    assert json.loads((job / "decisions.jsonl").read_text().splitlines()[2])["label"] == "grasper"
    return row, seen


def test_amber_accept_converts_crop_coordinates_to_frame(tmp_path, monkeypatch):
    reply = {"present": True, "conf": 0.9, "polygon": [[0, 0], [100, 0], [100, 100]], "reason": "jaws visible"}
    row, seen = amber_case(tmp_path, monkeypatch, [reply])
    assert row["status"] == "accept" and row["final_present"] is True and len(row["rounds"]) == 1
    x0, y0, x1, y1 = row["rounds"][0]["crop"]
    mag = seen[0]["shape"][1] / (x1 - x0)
    assert mag > 1  # magnified
    assert row["polygon"][0] == [x0, y0]
    assert row["polygon"][1][0] == pytest.approx(x0 + 100 / mag, abs=0.2)
    assert row["polygon"] == row["rounds"][0]["polygon"]
    assert seen[0]["model"] == vlm.SONNET
    assert row["rounds"][0]["raw_ref"] and row["rounds"][0]["prompt_hash"]


def test_amber_reject_on_confident_absent(tmp_path, monkeypatch):
    row, _ = amber_case(tmp_path, monkeypatch, [{"present": False, "conf": 0.9, "polygon": None, "reason": "fat only"}])
    assert row["status"] == "reject" and row["final_present"] is False and row["polygon"] is None


def test_amber_stays_amber_after_two_low_confidence_rounds(tmp_path, monkeypatch):
    low = {"present": True, "conf": 0.4, "polygon": [[0, 0], [100, 0], [100, 100]], "reason": "maybe"}
    row, seen = amber_case(tmp_path, monkeypatch, [low, low])
    assert row["status"] == "AMBER" and row["final_present"] is None and len(row["rounds"]) == 2
    assert len(seen) == 2
    assert "criterion" in seen[1]["prompt"] and "criterion" not in seen[0]["prompt"]


def test_amber_present_without_any_outline_is_not_accepted(tmp_path, monkeypatch):
    bare = {"present": True, "conf": 0.9, "polygon": None, "reason": "seen"}
    row, _ = amber_case(tmp_path, monkeypatch, [bare, bare])
    assert row["status"] == "AMBER" and row["polygon"] is None


def test_amber_call_failure_stays_amber_and_never_rejects(tmp_path, monkeypatch):
    # One of many ambers fails: that label stays AMBER while the job continues.
    job, meta = job_with(tmp_path, kfs=1)
    calls = []

    def fake_ask(images, prompt, schema, model=None):
        calls.append(prompt)
        if "grasper" in prompt:
            raise vlm.VlmError("model unavailable")
        return {"present": False, "conf": 0.9, "polygon": None, "reason": "no"}

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    rows = [det(label=k, present=False, polygon=None) for k in P.LABELS]
    vs = [ver(label=k) for k in P.LABELS]
    out = PL.decide(job, meta, rows, vs, types.SimpleNamespace(update=lambda *a: None))
    row = next(r for r in out if r["label"] == "grasper")
    assert row["status"] == "AMBER" and row["final_present"] is None and "failed" in row["reason"]


def test_amber_prompts_never_reveal_earlier_verdicts():
    for label in P.LABELS:
        for n in (1, 2):
            text = P.amber_prompt(label, 100, 100, n).lower()
            assert not re.search(r"disagree|earlier|previous|first pass|second pass|verifier|detector", text)
    assert not re.search(r"polygon|detector|previous|first pass", P.verify_prompt(100, 100).lower())


# ---- detect and verify ----------------------------------------------------------------------------

def test_detect_scales_polygons_to_full_frame_and_records_provenance(tmp_path, monkeypatch):
    job, meta = job_with(tmp_path, width=640, height=360, kfs=1, scale=2.0)

    def fake_ask(images, prompt, schema, model=None):
        return {"phase": {"name": "Preparation", "conf": 0.7}, "objects": {
            k: {"present": k == "liver", "conf": 0.9, "polygon": [[10, 10], [50, 10], [50, 50]] if k == "liver" else None} for k in P.LABELS}}

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    rows, phases = PL.detect(job, meta, types.SimpleNamespace(update=lambda *a: None))
    liver = next(r for r in rows if r["label"] == "liver")
    assert liver["polygon"][1] == [100, 20] and liver["model"] == vlm.SONNET and len(liver["raw_ref"]) == 64 and liver["prompt_hash"]
    assert len(rows) == 6 and phases == [{"kf": "kf_00000", "phase": "Preparation", "conf": 0.7}]
    disk = [json.loads(line) for line in open(job / "detect.jsonl")]
    assert [r["label"] for r in disk] == P.LABELS


def test_detect_failure_is_recorded_as_unknown_never_absent(tmp_path, monkeypatch):
    job, meta = job_with(tmp_path, kfs=3)
    calls = []

    def fake_ask(images, prompt, schema, model=None):
        calls.append(1)
        if len(calls) == 1:
            raise vlm.VlmError("model unavailable")
        return {"phase": {"name": "Preparation", "conf": 0.7}, "objects": {k: {"present": False, "conf": 0.9, "polygon": None} for k in P.LABELS}}

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    rows, phases = PL.detect(job, meta, types.SimpleNamespace(update=lambda *a: None))
    failed = [r for r in rows if r["present"] is None]
    assert len(failed) == 6 and all(r["error"] and r["raw_ref"] is None for r in failed)
    assert sum(1 for p in phases if p["phase"] == P.UNKNOWN_PHASE) == 1


def test_detect_stops_when_most_calls_fail(tmp_path, monkeypatch):
    job, meta = job_with(tmp_path, kfs=3)

    def fake_ask(*a, **k):
        raise vlm.VlmError("over limit")

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    with pytest.raises(PL.PipelineError, match="failed on 3 of 3"):
        PL.detect(job, meta, types.SimpleNamespace(update=lambda *a: None))


def test_verify_sees_only_the_image_and_records_phase_agreement(tmp_path, monkeypatch):
    job, meta = job_with(tmp_path, kfs=1)
    seen = {}

    def fake_ask(images, prompt, schema, model=None):
        seen.update(images=images, prompt=prompt, model=model)
        return {"phase": {"name": "Preparation", "conf": 0.5}, "objects": {k: {"present": False, "conf": 0.9, "box": None} for k in P.LABELS}}

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    phases = [{"kf": "kf_00000", "phase": "Preparation", "conf": 0.8}]
    PL.verify(job, meta, phases, types.SimpleNamespace(update=lambda *a: None))
    assert seen["model"] == vlm.HAIKU and len(seen["images"]) == 1
    assert phases[0]["agree"] is True and phases[0]["verify_phase"] == "Preparation"
    assert json.loads(open(job / "phase.jsonl").readline())["verify_conf"] == 0.5


# ---- narration ------------------------------------------------------------------------------------

FACTS = {"phase": "GallbladderDissection", "keyframes": [
    {"kf": "kf_00001", "t": 1.0, "visible": ["liver", "grasper"], "not_visible": ["blood", "l_hook"], "unsure": ["cystic_duct"]},
    {"kf": "kf_00002", "t": 2.0, "visible": ["liver"], "not_visible": ["blood", "grasper"], "unsure": []},
]}


@pytest.mark.parametrize("text, evidence, ok", [
    ("The liver and a grasper are visible.", ["kf_00001"], True),
    ("A grasper is visible.", ["kf_00001", "kf_00002"], False),  # not visible on the second citation
    ("A grasper is visible.", ["kf_00009"], False),  # cites a keyframe outside this segment
    ("The liver is visible.", [], False),  # no evidence
    ("An L-hook is working.", ["kf_00001"], False),  # verified absent
    ("No L-hook is in view.", ["kf_00001"], True),
    ("There is bleeding.", ["kf_00001"], False),
    ("There is no bleeding.", ["kf_00001"], True),
    ("The cystic duct is visible.", ["kf_00001"], False),  # still amber, stated as fact
    ("The cystic duct could not be confirmed.", ["kf_00001"], True),
    ("The gallbladder is visible.", ["kf_00001"], False),  # never decided here
    ("A clip is applied to the liver.", ["kf_00001"], False),  # no such label
    ("The cystic artery is exposed near the liver.", ["kf_00001"], False),
    ("Gallbladder dissection continues, with the liver in view.", ["kf_00001"], True),  # the phase's own name
])
def test_sentence_supported(text, evidence, ok):
    assert PL.sentence_supported({"text": text, "evidence": evidence}, FACTS) is ok


def narration_setup(tmp_path):
    job, meta = job_with(tmp_path, kfs=2)
    phases = [{"kf": k["kf"], "phase": "Preparation", "conf": 0.9, "agree": True} for k in meta["keyframes"]]
    decisions = [
        {"kf": "kf_00000", "label": "liver", "status": "accept"},
        {"kf": "kf_00000", "label": "blood", "status": "AMBER"},
        {"kf": "kf_00001", "label": "liver", "status": "reject"},
    ]
    return job, meta, phases, decisions


def test_narration_facts_are_only_decisions_and_bad_sentences_are_dropped(tmp_path, monkeypatch):
    job, meta, phases, decisions = narration_setup(tmp_path)
    seen = {}

    def fake_ask(images, prompt, schema, model=None):
        seen.update(images=images, prompt=prompt)
        return {"sentences": [{"text": "The liver is in view.", "evidence": ["kf_00000"]},
                              {"text": "Blood is pooling.", "evidence": ["kf_00000"]}]}

    monkeypatch.setattr(PL.vlm, "ask", fake_ask)
    segments = PL.narrate(job, meta, phases, decisions, types.SimpleNamespace(update=lambda *a: None))
    assert seen["images"] == []
    assert segments[0]["text"] == "The liver is in view." and segments[0]["evidence"] == ["kf_00000"]
    assert segments[0]["dropped_sentences"] == 1 and segments[0]["source"].endswith(vlm.SONNET)
    assert segments[0]["t0"] == 0.0 and segments[0]["t1"] == meta["duration"]
    assert json.load(open(job / "narration.json"))["segments"][0]["text"] == "The liver is in view."


def test_narration_falls_back_to_decisions_when_nothing_survives(tmp_path, monkeypatch):
    job, meta, phases, decisions = narration_setup(tmp_path)
    monkeypatch.setattr(PL.vlm, "ask", lambda *a, **k: {"sentences": [{"text": "A stapler fires.", "evidence": ["kf_00000"]}]})
    seg = PL.narrate(job, meta, phases, decisions, types.SimpleNamespace(update=lambda *a: None))[0]
    assert seg["source"] == "decisions" and "stapler" not in seg["text"]
    assert "Verified in this segment: liver." in seg["text"] and "Could not be confirmed: blood." in seg["text"]


def test_smooth_phases_removes_flicker_and_keeps_real_changes():
    names = ["A", "A", "B", "A", "A", "C", "C", "C", "C", "C"]
    smoothed = PL.smooth_phases([{"phase": n} for n in names])
    assert smoothed[2] == "A" and smoothed[-1] == "C" and smoothed[0] == "A"


def test_phase_segments_cover_the_video_and_split_long_runs():
    n = 100
    meta = {"keyframes": [{"kf": f"kf_{i:05d}", "t": float(i)} for i in range(n)], "duration": 100.0}
    phases = [{"phase": "Preparation" if i < 10 else "ClippingCutting"} for i in range(n)]
    segments = PL.phase_segments(meta, phases)
    assert segments[0]["t0"] == 0.0 and segments[-1]["t1"] == 100.0
    assert all(a["t1"] == b["t0"] for a, b in zip(segments, segments[1:]))
    assert all(s["t1"] - s["t0"] <= PL.NARRATE_MAX_SECONDS for s in segments)
    assert {s["phase"] for s in segments} == {"Preparation", "ClippingCutting"}


# ---- orchestration --------------------------------------------------------------------------------

@pytest.fixture
def stub_stages(monkeypatch):
    called = []

    def with_on_progress(stage):
        def stage_run(job, video=None, on_progress=None):
            on_progress(0.5, f"{stage} half")
            called.append(stage)
        return stage_run

    def with_progress(stage):
        def stage_run(job, progress=None):
            progress(0.5, f"{stage} half")
            called.append(stage)
        return stage_run

    def with_neither(stage):
        def stage_run(job):
            called.append(stage)
        return stage_run

    for stage, make in [("masks", with_on_progress), ("propagate", with_progress), ("render", with_on_progress), ("export", with_neither)]:
        monkeypatch.setitem(sys.modules, PL.EXTERNAL_STAGES[stage], types.SimpleNamespace(run=make(stage)))
    return called


def test_full_run_with_mock_provider(isolated, stub_stages):
    video = make_video(isolated / "v.mp4")
    job = isolated / "jobs" / "abc"
    updates = []
    final = PL.run(video, job, on_status=lambda s: updates.append(s))
    assert stub_stages == ["masks", "propagate", "render", "export"]
    assert final["stage"] == "done" and final["progress"] == 1.0 and final["error"] is None
    disk = json.load(open(job / "status.json"))
    assert disk["stage"] == "done" and all(s["state"] == "done" for s in disk["stages"]) and disk["started"] and disk["updated"]
    progress = [u["progress"] for u in updates]
    assert progress == sorted(progress)
    assert {u["stage"] for u in updates} >= set(PL.STAGES)
    assert (job / ".gitignore").read_text() == "*\n"
    for name in ("meta.json", "detect.jsonl", "phase.jsonl", "verify.jsonl", "decisions.jsonl", "narration.json", "source.mp4"):
        assert (job / name).exists(), name
    detect_rows = PL.read_jsonl(job / "detect.jsonl")
    decisions = PL.read_jsonl(job / "decisions.jsonl")
    assert len(detect_rows) == len(decisions) == 3 * len(P.LABELS)
    assert all(r["model"].startswith("mock:") for r in detect_rows + PL.read_jsonl(job / "verify.jsonl"))
    assert json.load(open(job / "meta.json"))["vlm_provider"] == "mock"
    duct = [r for r in decisions if r["label"] == "cystic_duct"]
    assert all(r["status"] == "reject" and len(r["rounds"]) == 1 for r in duct)  # went through the amber loop
    assert all(r["status"] == "accept" and r["polygon"] for r in decisions if r["label"] == "liver")
    assert all(r["rounds"][0]["model"].startswith("mock:") for r in duct)
    narration = json.load(open(job / "narration.json"))["segments"]
    assert narration and narration[0]["phase"] == P.PHASES[1]


def test_rerun_uses_the_cache(isolated, stub_stages, monkeypatch):
    video = make_video(isolated / "v.mp4", seconds=1)
    PL.run(video, isolated / "jobs" / "a")
    before = vlm.stats()["calls"]
    PL.run(video, isolated / "jobs" / "b")
    assert vlm.stats()["calls"] == before


def test_auth_failure_is_recorded_and_raised(isolated, stub_stages, monkeypatch):
    video = make_video(isolated / "v.mp4", seconds=1)

    def not_logged_in(*a, **k):
        raise vlm.VlmAuthError("Not logged in")

    monkeypatch.setattr(PL.vlm, "ask", not_logged_in)
    with pytest.raises(vlm.VlmAuthError):
        PL.run(video, isolated / "jobs" / "x")
    status = json.load(open(isolated / "jobs" / "x" / "status.json"))
    assert status["error"] == "Not logged in" and status["stage"] == "detect"
    assert next(s for s in status["stages"] if s["name"] == "detect")["state"] == "failed"
    assert not stub_stages


def test_missing_lane_module_fails_clearly(isolated, monkeypatch):
    monkeypatch.setitem(sys.modules, "surgproof.masks", None)
    video = make_video(isolated / "v.mp4", seconds=1)
    with pytest.raises((PL.PipelineError, ImportError)):
        PL.run(video, isolated / "jobs" / "y")
    assert json.load(open(isolated / "jobs" / "y" / "status.json"))["error"]


def test_stop_after_stops_before_masks(isolated, stub_stages):
    video = make_video(isolated / "v.mp4", seconds=1)
    final = PL.run(video, isolated / "jobs" / "z", stop_after="amber")
    assert final["stage"] == "done" and "stopped" in final["message"]
    assert not stub_stages and not (isolated / "jobs" / "z" / "narration.json").exists()


def test_core_code_never_touches_ground_truth():
    pattern = re.compile(r"gt_|ground.?truth|\btruth\b|mask_path|watershed|CholecSeg|tools/make_test_video|surgproof\.eval", re.I)
    for name in ("vlm.py", "pipeline.py", "prompts.py", "run.py"):
        hits = [line for line in (ROOT / "surgproof" / name).read_text().splitlines() if pattern.search(line)]
        assert not hits, f"{name}: {hits}"
