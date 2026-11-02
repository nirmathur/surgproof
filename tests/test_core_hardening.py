"""Regression tests for the core items in the 2 Nov audit: scope gate, amber limits, metadata, ingest safety, cost, narration."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from surgproof import pipeline as PL
from surgproof import prompts as P
from surgproof import run as RUN
from surgproof import vlm

ROOT = Path(__file__).resolve().parent.parent


def make_video(path, seconds=3, fps=25, size="320x180", extra=(), audio=False):
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size={size}:rate={fps}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=duration={seconds}", "-c:a", "aac"]
    subprocess.run(cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", *extra, str(path)], check=True)
    return path


def probe(path, entries):
    return json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", entries, "-of", "json", str(path)],
                                     capture_output=True, text=True, check=True).stdout)


class Quiet:
    def update(self, *a):
        pass


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    monkeypatch.delenv("SURGPROOF_MAX_COST_USD", raising=False)
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "cache.sqlite")
    return tmp_path


@pytest.fixture
def stub_stages(monkeypatch):
    called = []

    def make(stage):
        def stage_run(job):
            called.append(stage)
        return stage_run

    for stage, module in PL.EXTERNAL_STAGES.items():
        monkeypatch.setitem(sys.modules, module, types.SimpleNamespace(run=make(stage)))
    return called


def detect_reply(present=("liver",), scene=None, phase="CalotTriangleDissection"):
    reply = {"phase": {"name": phase, "conf": 0.9}, "objects": {
        k: {"present": k in present, "conf": 0.9, "polygon": [[10, 10], [60, 10], [60, 60], [10, 60]] if k in present else None}
        for k in P.LABELS}}
    if scene is not None:
        reply["scene"] = scene
    return reply


def verify_reply(present=("liver",)):
    return {"phase": {"name": "CalotTriangleDissection", "conf": 0.9}, "objects": {
        k: {"present": k in present, "conf": 0.9, "box": [10, 10, 60, 60] if k in present else None} for k in P.LABELS}}


def scripted(monkeypatch, detect=None, verify=None, amber=None, narrate=None):
    """Replace vlm.ask. Records calls by kind."""
    calls = {"detect": [], "verify": [], "amber": [], "narrate": []}

    def fake(images, prompt, schema, model=None):
        kind = schema.get("title")
        calls[kind].append(prompt)
        source = {"detect": detect, "verify": verify, "amber": amber, "narrate": narrate}[kind]
        reply = source() if callable(source) else source
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(PL.vlm, "ask", fake)
    return calls


LAPAROSCOPIC = {"laparoscopic": True, "procedure": "cholecystectomy", "other_procedure": None}


# ---- 1. input gate --------------------------------------------------------------------------------

def scene_rows(n, **fields):
    return [{"kf": f"kf_{i:05d}", **LAPAROSCOPIC, **fields} for i in range(n)]


def phase_rows(n, name="CalotTriangleDissection"):
    return [{"kf": f"kf_{i:05d}", "phase": name, "conf": 0.9} for i in range(n)]


def test_scope_accepts_cholecystectomy_and_unclear_frames():
    assert PL.scope_problem(scene_rows(5), phase_rows(5)) is None
    assert PL.scope_problem(scene_rows(5, procedure="unclear"), phase_rows(5)) is None


@pytest.mark.parametrize("scenes, phases, words", [
    (scene_rows(5, laparoscopic=False), phase_rows(5), "not laparoscopic"),
    (scene_rows(5, procedure="other", other_procedure="appendectomy"), phase_rows(5), "appendectomy"),
    (scene_rows(5), phase_rows(5, P.UNKNOWN_PHASE), "phase"),
])
def test_scope_names_the_problem_and_the_supported_procedure(scenes, phases, words):
    message = PL.scope_problem(scenes, phases)
    assert words in message and "laparoscopic cholecystectomy" in message


def test_scope_ignores_frames_that_failed_and_a_minority_of_bad_frames():
    scenes = scene_rows(2, laparoscopic=False) + scene_rows(3)
    assert PL.scope_problem(scenes, phase_rows(5)) is None
    assert PL.scope_problem([], []) is None


def test_non_surgical_video_stops_after_a_small_sample(isolated, stub_stages, monkeypatch):
    calls = scripted(monkeypatch, detect=lambda: detect_reply(present=(), scene={"laparoscopic": False, "procedure": "unclear", "other_procedure": None}, phase=P.UNKNOWN_PHASE))
    job = isolated / "jobs" / "cat"
    with pytest.raises(PL.OutOfScope):
        PL.run(make_video(isolated / "v.mp4", seconds=12), job)
    status = json.load(open(job / "status.json"))
    assert status["result"] == "out_of_scope" and "laparoscopic cholecystectomy" in status["error"]
    assert len(calls["detect"]) == PL.GATE_SAMPLE < 13 and not calls["verify"] and not stub_stages


def test_other_procedure_is_an_out_of_scope_result(isolated, stub_stages, monkeypatch):
    other = {"laparoscopic": True, "procedure": "other", "other_procedure": "hernia repair"}
    scripted(monkeypatch, detect=lambda: detect_reply(scene=other))
    with pytest.raises(PL.OutOfScope):
        PL.run(make_video(isolated / "v.mp4", seconds=2), isolated / "jobs" / "h")
    status = json.load(open(isolated / "jobs" / "h" / "status.json"))
    assert status["result"] == "out_of_scope" and "hernia repair" in status["error"]


def test_nothing_accepted_fails_the_job_clearly(isolated, stub_stages, monkeypatch):
    scripted(monkeypatch, detect=lambda: detect_reply(present=(), scene=LAPAROSCOPIC), verify=lambda: verify_reply(present=()))
    job = isolated / "jobs" / "none"
    with pytest.raises(PL.OutOfScope):
        PL.run(make_video(isolated / "v.mp4", seconds=2), job)
    status = json.load(open(job / "status.json"))
    assert status["result"] == "nothing_accepted" and "laparoscopic cholecystectomy" in status["error"]
    assert not stub_stages
    assert (job / "decisions.jsonl").exists()  # kept, so the outcome can be inspected


# ---- 2. amber limits and routes -------------------------------------------------------------------

def one_label_decision(tmp_path, monkeypatch, d, v, amber):
    job = tmp_path / "job"
    (job / "frames").mkdir(parents=True)
    import cv2
    import numpy as np
    cv2.imwrite(str(job / "frames/kf_00000.jpg"), np.random.RandomState(0).randint(0, 255, (180, 320, 3), dtype=np.uint8))
    meta = {"width": 320, "height": 180, "keyframes": [{"kf": "kf_00000", "frame_index": 0, "t": 0.0}]}
    base = lambda k: {"kf": "kf_00000", "label": k, "present": False, "conf": 0.9, "polygon": None, "box": None}
    detect = [base(k) for k in P.LABELS if k != "grasper"] + ([{**base("grasper"), **d}] if d else [])
    verify = [base(k) for k in P.LABELS if k != "grasper"] + ([{**base("grasper"), **v}] if v else [])
    calls = scripted(monkeypatch, amber=amber)
    rows = PL.decide(job, meta, detect, verify, types.SimpleNamespace(update=lambda *a: None))
    return next(r for r in rows if r["label"] == "grasper"), calls, rows


POLY = [[10, 10], [60, 10], [60, 60], [10, 60]]


def test_amber_cannot_settle_a_reject(tmp_path, monkeypatch):
    row, calls, _ = one_label_decision(tmp_path, monkeypatch, {"present": True, "polygon": POLY}, {"present": False},
                                       {"present": False, "conf": 0.99, "polygon": None, "reason": "no instrument"})
    assert row["status"] == "AMBER" and row["final_present"] is None
    assert row["route"] == "amber_said_absent" and len(calls["amber"]) == 1


def test_amber_can_still_settle_present(tmp_path, monkeypatch):
    row, _, _ = one_label_decision(tmp_path, monkeypatch, {"present": True, "polygon": POLY}, {"present": False},
                                   {"present": True, "conf": 0.9, "polygon": [[0, 0], [100, 0], [100, 100]], "reason": "shaft"})
    assert row["status"] == "accept" and row["route"] == "amber_said_present"


def test_unresolved_amber_has_its_route(tmp_path, monkeypatch):
    row, calls, _ = one_label_decision(tmp_path, monkeypatch, {"present": True, "polygon": POLY}, {"present": False},
                                       {"present": True, "conf": 0.3, "polygon": None, "reason": "unsure"})
    assert row["status"] == "AMBER" and row["route"] == "amber_unresolved" and len(calls["amber"]) == 2


@pytest.mark.parametrize("d, v", [
    ({"present": True, "polygon": POLY}, {"present": None, "error": "failed"}),  # second pass failed
    ({"present": None, "error": "failed"}, {"present": True, "box": [1, 1, 50, 50]}),  # first pass failed
    ({"present": True, "polygon": POLY}, None),  # second pass row missing
    (None, {"present": True, "box": [1, 1, 50, 50]}),  # first pass row missing
])
def test_failed_or_missing_pass_stays_amber_without_a_loop(tmp_path, monkeypatch, d, v):
    row, calls, _ = one_label_decision(tmp_path, monkeypatch, d, v, {"present": True, "conf": 0.99, "polygon": POLY, "reason": "x"})
    assert row["status"] == "AMBER" and row["final_present"] is None and row["route"] == "pass_failed"
    assert not calls["amber"]


def test_every_decision_carries_a_route_and_passes_agreeing_keep_theirs(tmp_path, monkeypatch):
    _, _, rows = one_label_decision(tmp_path, monkeypatch, {"present": True, "polygon": POLY}, {"present": True, "box": [10, 10, 60, 60]}, None)
    routes = {r["label"]: r["route"] for r in rows}
    assert routes["grasper"] == "both_passes_present" and routes["liver"] == "both_passes_absent"
    assert all(json.loads(line)["route"] for line in open(tmp_path / "job" / "decisions.jsonl"))


# ---- 3. metadata and names ------------------------------------------------------------------------

SECRET = "PATIENT_JANE_DOE_MRN123"


@pytest.mark.parametrize("extra", [[], ["-vf", "setpts=N/25/TB"]])
def test_source_has_no_audio_or_metadata(tmp_path, extra):
    video = make_video(tmp_path / f"{SECRET}.mp4", seconds=1, audio=True,
                       extra=["-metadata", f"title={SECRET}", "-metadata", f"comment={SECRET}", "-metadata", "location=+51.5-0.1/",
                              "-metadata:s:v", f"handler_name={SECRET}", *extra])
    job = tmp_path / "j"
    job.mkdir()
    PL.ingest(video, job, Quiet())
    info = probe(job / "source.mp4", "stream=codec_type:format_tags:stream_tags")
    assert [s["codec_type"] for s in info["streams"]] == ["video"]
    tags = {**info["format"].get("tags", {}), **info["streams"][0].get("tags", {})}
    assert not {"title", "comment", "location", "creation_time"} & set(tags)
    assert SECRET.encode() not in (job / "source.mp4").read_bytes()


def test_transcoded_source_also_loses_audio_and_metadata(tmp_path):
    video = tmp_path / "v.mov"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=1:size=320x180:rate=25", "-f", "lavfi", "-i", "sine=duration=1",
                    "-c:v", "mpeg4", "-c:a", "aac", "-metadata", f"title={SECRET}", str(video)], check=True)
    job = tmp_path / "j"
    job.mkdir()
    PL.ingest(video, job, Quiet())
    assert [s["codec_type"] for s in probe(job / "source.mp4", "stream=codec_type")["streams"]] == ["video"]
    assert SECRET.encode() not in (job / "source.mp4").read_bytes()


def test_upload_name_is_stored_nowhere_and_display_names_are_random(isolated, stub_stages):
    video = make_video(isolated / f"{SECRET}.mp4", seconds=1)
    PL.run(video, isolated / "jobs" / "a")
    PL.run(video, isolated / "jobs" / "b")
    a, b = (json.load(open(isolated / "jobs" / n / "meta.json")) for n in "ab")
    assert a["display_name"] != b["display_name"] and "original_name" not in a
    for name in ("meta.json", "status.json", "decisions.jsonl", "narration.json", "detect.jsonl"):
        assert SECRET not in (isolated / "jobs" / "a" / name).read_text(), name


@pytest.mark.parametrize("make_bad", ["missing", "garbage"])
def test_failure_messages_do_not_echo_the_file_name(isolated, stub_stages, make_bad):
    video = isolated / f"{SECRET}.mp4"
    if make_bad == "garbage":
        video.write_bytes(b"not a video")
    with pytest.raises(PL.PipelineError):
        PL.run(video, isolated / "jobs" / "x")
    assert SECRET not in (isolated / "jobs" / "x" / "status.json").read_text()


def test_cli_default_job_folder_does_not_use_the_file_name(isolated, stub_stages, monkeypatch):
    monkeypatch.chdir(isolated)
    video = make_video(isolated / f"{SECRET}.mp4", seconds=1)
    assert RUN.main([str(video), "--stop-after", "ingest"]) == 0
    assert [p for p in (isolated / "jobs").iterdir() if p.is_dir()] and not [p for p in (isolated / "jobs").iterdir() if SECRET in p.name]


# ---- 4. safe, frame-accurate ingest ---------------------------------------------------------------

@pytest.mark.parametrize("pts, uniform", [
    ([i * 0.04 for i in range(50)], True),
    ([round(i / 29.97, 3) for i in range(100)], True),  # millisecond rounding of 29.97 fps
    ([0.0, 0.04, 0.08, 0.5, 0.54, 0.58], False),  # a gap
    ([i * 0.04 + (0.015 if i % 2 else 0) for i in range(50)], False),  # same average rate, alternating jitter
    ([0.0], True),
    ([], True),
])
def test_pts_uniform(pts, uniform):
    assert PL.pts_uniform(pts) is uniform


def retimed_video(path, tmp_path, seconds, expression):
    """A real mp4 whose packet timestamps are changed by expression, in milliseconds."""
    plain = tmp_path / "plain.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=320x180:rate=25",
                    "-c:v", "libx264", "-bf", "0", "-pix_fmt", "yuv420p", "-video_track_timescale", "1000", str(plain)], check=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(plain), "-c", "copy", "-bsf:v", f"setts=ts='{expression}'",
                    "-video_track_timescale", "1000", str(path)], check=True)
    return path


def test_jittered_video_is_re_encoded_to_uniform_timestamps(tmp_path):
    video = retimed_video(tmp_path / "v.mp4", tmp_path, 2, "TS+15*mod(N,2)")
    assert not PL.pts_uniform(PL._read_pts(video))  # the attack is real, or this test proves nothing
    job = tmp_path / "j"
    job.mkdir()
    PL.ingest(video, job, Quiet())
    assert PL.pts_uniform(PL._read_pts(job / "source.mp4"))


def test_one_dropped_frame_is_caught_though_the_frame_rates_still_match(tmp_path):
    """After a gap, frame index and time disagree for the rest of the video. The rates differ by well under 1%."""
    video = retimed_video(tmp_path / "v.mp4", tmp_path, 8, "TS+40*gte(N,100)")
    stream, _ = PL._probe(video)
    r, avg = PL._rate(stream["r_frame_rate"]), PL._rate(stream["avg_frame_rate"])
    assert abs(r - avg) <= 0.01 * max(avg, 1)  # a frame-rate comparison cannot see this
    assert not PL.pts_uniform(PL._read_pts(video))
    job = tmp_path / "j"
    job.mkdir()
    meta = PL.ingest(video, job, Quiet())
    assert PL.pts_uniform(PL._read_pts(job / "source.mp4"))
    frames = PL._read_pts(job / "source.mp4")
    assert frames[150] == pytest.approx(150 / meta["fps"], abs=1 / meta["fps"])  # index and time agree after the gap


def test_every_external_call_has_a_timeout_and_a_protocol_whitelist(tmp_path, monkeypatch):
    real, seen = subprocess.run, []

    def spy(cmd, *a, **k):
        seen.append((cmd, k))
        return real(cmd, *a, **k)

    video = make_video(tmp_path / "v.mp4", seconds=1)
    monkeypatch.setattr(PL.subprocess, "run", spy)
    job = tmp_path / "j"
    job.mkdir()
    PL.ingest(video, job, Quiet())
    assert len(seen) >= 3  # probe, timestamps, ffmpeg
    for cmd, kwargs in seen:
        assert kwargs.get("timeout"), cmd
        assert cmd[cmd.index("-protocol_whitelist") + 1] == "file,pipe", cmd
        assert cmd.index("-protocol_whitelist") < len(cmd) - 1


def test_a_hung_ffmpeg_fails_the_job(tmp_path, monkeypatch):
    def hang(cmd, *a, **k):
        raise subprocess.TimeoutExpired(cmd, 1)

    video = make_video(tmp_path / "v.mp4", seconds=1)
    monkeypatch.setattr(PL.subprocess, "run", hang)
    job = tmp_path / "j"
    job.mkdir()
    with pytest.raises(PL.PipelineError, match="timed out"):
        PL.ingest(video, job, Quiet())


def test_playlist_disguised_as_mp4_is_refused(tmp_path):
    secret = tmp_path / "secret.ts"
    make_video(secret, seconds=1)
    bad = tmp_path / "evil.mp4"
    bad.write_text(f"#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:0\n#EXTINF:1.0,\n{secret}\n#EXT-X-ENDLIST\n")
    job = tmp_path / "j"
    job.mkdir()
    with pytest.raises(PL.PipelineError):
        PL.ingest(bad, job, Quiet())
    assert not (job / "source.mp4").exists()


@pytest.mark.parametrize("kwargs, words", [
    ({"size": "4000x200"}, "resolution"),
    ({"fps": 240}, "frame rate"),
])
def test_size_and_frame_rate_caps(tmp_path, kwargs, words):
    video = make_video(tmp_path / "v.mp4", seconds=1, **kwargs)
    job = tmp_path / "j"
    job.mkdir()
    with pytest.raises(PL.PipelineError, match=words):
        PL.ingest(video, job, Quiet())


# ---- 5. cost, keyframe rate, plan limits ----------------------------------------------------------

def test_kf_fps_cap(tmp_path):
    job = tmp_path / "j"
    job.mkdir()
    with pytest.raises(PL.PipelineError, match="kf_fps"):
        PL.ingest(make_video(tmp_path / "v.mp4", seconds=1), job, Quiet(), kf_fps=PL.MAX_KF_FPS + 1)


def test_estimated_cost_over_the_cap_stops_before_any_model_call(isolated, stub_stages, monkeypatch):
    calls = scripted(monkeypatch, detect=lambda: detect_reply(scene=LAPAROSCOPIC))
    monkeypatch.setenv("SURGPROOF_MAX_COST_USD", "0.01")
    job = isolated / "jobs" / "c"
    with pytest.raises(PL.PipelineError, match="cost"):
        PL.run(make_video(isolated / "v.mp4", seconds=3), job)
    assert not calls["detect"] and "cost" in json.load(open(job / "status.json"))["error"]


def counting_provider(monkeypatch, cost):
    from surgproof.prompts import mock_reply
    seen = []

    def provider(paths, prompt, model):
        seen.append(prompt)
        return json.dumps(mock_reply(prompt, paths, model)), cost

    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", provider)
    return seen


def test_spend_cap_stops_the_job_and_a_rerun_resumes_from_cache(isolated, stub_stages, monkeypatch):
    seen = counting_provider(monkeypatch, cost=0.06)
    monkeypatch.setenv("SURGPROOF_MAX_COST_USD", "0.2")
    video = make_video(isolated / "v.mp4", seconds=4)
    job = isolated / "jobs" / "s"
    with pytest.raises(PL.PipelineError, match="cost"):
        PL.run(video, job, concurrency=1)
    first = len(seen)
    assert 0 < first < 8
    assert json.load(open(job / "status.json"))["cost_usd"] <= 0.2 + 0.06
    monkeypatch.setenv("SURGPROOF_MAX_COST_USD", "50")
    final = PL.run(video, job, concurrency=1)
    assert final["stage"] == "done"
    detects = [p for p in seen if p.startswith("This is one frame")]
    n_keyframes = len(json.load(open(job / "meta.json"))["keyframes"])
    assert len(detects) == n_keyframes  # each asked once across both runs: nothing already paid for was repeated


@pytest.mark.parametrize("text", [
    "Claude usage limit reached. Your limit will reset at 3pm",
    "You've hit your session limit · resets 7pm",
    "5-hour limit reached",
    "API Error: 429 rate_limit_error",
    "HTTP 429",
    "status: 429",
    "error code: 429",
])
def test_plan_limit_messages_are_recognised(text):
    assert vlm.is_usage_limit(text)


@pytest.mark.parametrize("text", [
    "Not logged in", "boom", "Image too large",
    "Parse error at line 429 of the response",  # a bare 429 is not a rate limit
    "cannot read /tmp/frames/kf_00429/429.jpg",
    "output token limit reached",  # a real failure: pausing would only hide it for hours
    "file size limit reached",
    "context limit reached",
])
def test_other_errors_are_not_plan_limits(text):
    assert not vlm.is_usage_limit(text)


def test_plan_limit_pauses_then_resumes_without_failing(isolated, stub_stages, monkeypatch):
    from surgproof.prompts import mock_reply
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setenv("SURGPROOF_VLM_PAUSE_POLL_S", "0")
    state = {"n": 0}

    def provider(paths, prompt, model):
        state["n"] += 1
        if state["n"] <= 2:
            raise vlm.VlmUsageLimit("You've hit your limit")
        return json.dumps(mock_reply(prompt, paths, model)), 0.0

    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", provider)
    updates = []
    final = PL.run(make_video(isolated / "v.mp4", seconds=2), isolated / "jobs" / "p", on_status=lambda s: updates.append(dict(s)), concurrency=1)
    assert final["stage"] == "done" and final["error"] is None and final["paused"] is False
    assert any(u.get("paused") and "limit" in u["message"] for u in updates)


def test_plan_limit_that_outlasts_the_wait_leaves_a_resumable_job(isolated, stub_stages, monkeypatch):
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setenv("SURGPROOF_VLM_PAUSE_POLL_S", "0")
    monkeypatch.setenv("SURGPROOF_VLM_MAX_PAUSE_S", "0")

    def provider(paths, prompt, model):
        raise vlm.VlmUsageLimit("You've hit your limit")

    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", provider)
    job = isolated / "jobs" / "q"
    with pytest.raises(vlm.VlmUsageLimit):
        PL.run(make_video(isolated / "v.mp4", seconds=2), job)
    status = json.load(open(job / "status.json"))
    assert status["paused"] is True and status["result"] == "paused" and "resume" in status["error"]


def test_cli_classifies_a_plan_limit_from_the_binary(tmp_path, monkeypatch):
    binary = tmp_path / "fake_claude"
    binary.write_text(f"#!{sys.executable}\nimport json\nprint(json.dumps({{'is_error': True, 'result': 'You have hit your usage limit', 'total_cost_usd': 0}}))\n")
    binary.chmod(0o755)
    monkeypatch.setenv("SURGPROOF_CLAUDE_BIN", str(binary))
    with pytest.raises(vlm.VlmUsageLimit):
        vlm._claude_cli([], "q", "m")


# ---- 6. narration wording -------------------------------------------------------------------------

def narration_case(tmp_path, monkeypatch, sentences):
    job = tmp_path / "job"
    job.mkdir()
    meta = {"keyframes": [{"kf": "kf_00000", "t": 0.0}], "duration": 1.0}
    phases = [{"kf": "kf_00000", "phase": "Preparation", "conf": 0.9, "agree": True}]
    decisions = [{"kf": "kf_00000", "label": "liver", "status": "accept"}, {"kf": "kf_00000", "label": "blood", "status": "AMBER"}]
    scripted(monkeypatch, narrate=sentences)
    PL.narrate(job, meta, phases, decisions, types.SimpleNamespace(update=lambda *a: None))
    return json.load(open(job / "narration.json"))


def test_every_narration_says_ai_draft(tmp_path, monkeypatch):
    out = narration_case(tmp_path, monkeypatch, {"sentences": [{"text": "The liver is visible.", "evidence": ["kf_00000"]}]})
    assert out["draft"] is True and out["notice"].startswith("AI draft")
    assert out["segments"][0]["text"] == "AI draft. The liver is visible."


def test_fallback_narration_says_ai_draft_and_not_verified(tmp_path, monkeypatch):
    out = narration_case(tmp_path, monkeypatch, {"sentences": [{"text": "A stapler fires.", "evidence": ["kf_00000"]}]})
    text = out["segments"][0]["text"]
    assert text.startswith("AI draft.") and "verified" not in text.lower() and "liver" in text and "blood" in text


def test_no_verified_wording_reaches_the_user_or_the_model():
    assert "verified" not in P.narrate_prompt({"phase": "Preparation"}).lower()
    assert "verified" not in RUN.__doc__.lower()
    source = (ROOT / "surgproof" / "run.py").read_text().lower()
    assert "verified" not in source


def test_sampling_first_still_asks_every_keyframe_exactly_once(isolated, stub_stages, monkeypatch):
    seen = counting_provider(monkeypatch, cost=0.0)
    job = isolated / "jobs" / "all"
    PL.run(make_video(isolated / "v.mp4", seconds=12), job, stop_after="detect")
    keyframes = [k["kf"] for k in json.load(open(job / "meta.json"))["keyframes"]]
    rows = PL.read_jsonl(job / "detect.jsonl")
    assert len(keyframes) > PL.GATE_SAMPLE and sum(p.startswith("This is one frame") for p in seen) == len(keyframes)
    assert [r["kf"] for r in rows if r["label"] == "liver"] == keyframes  # in order, none missing, none repeated
    assert len(rows) == len(keyframes) * len(P.LABELS)
    assert [r["kf"] for r in PL.read_jsonl(job / "phase.jsonl")] == keyframes
