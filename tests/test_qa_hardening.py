"""Regression tests for the 2 Nov audit: each one replays the attack through the real CLI or HTTP route.

Run them against the code before the hardening round and they fail. A test here passing is evidence, not proof:
the orchestrator re-runs the audit's attacks itself.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import pytest

from tests.e2e_helpers.harness import (HAVE_FFMPEG, REVIEWER, ROOT, base_env, cache_rows, fake_env, local_client, read_json, read_jsonl,
                                        run_cli, sign_in, synthetic_video, upload)

needed = ("server", "pipeline", "masks", "propagate", "render", "export")
ready = all((ROOT / "surgproof" / f"{m}.py").exists() for m in needed)
pytestmark = pytest.mark.skipif(not (ready and HAVE_FFMPEG), reason="other lanes not merged, or no ffmpeg")
if ready:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from surgproof import server as srv

PATIENT = "Jane_Doe_MRN5551234"
SECRET = "SECRETTAGVALUE"


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def all_text_under(folder):
    """Every text-like file in a folder, as (name, content)."""
    for p in Path(folder).rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".vtt", ".txt", ".md", ".csv", ""):
            yield p, p.read_text(errors="replace")


def tagged_video(path):
    """Audio, a title, a comment, a location: everything a phone or a hospital system might leave in a file."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=3", "-f", "lavfi", "-i",
         "sine=frequency=440:duration=3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         "-metadata", f"title={SECRET}", "-metadata", f"comment={SECRET}", "-metadata", f"artist={SECRET}",
         "-metadata", "location=+51.5074-000.1278/", "-movflags", "use_metadata_tags", str(path)], check=True)
    return path


@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    d = tmp_path_factory.mktemp("job")
    job = d / "jobs" / "job1"
    p = run_cli(synthetic_video(d / "v.mp4", seconds=4), job, fake_env("amber_stuck"))
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    return job


def client(jobs, **kw):
    return local_client(srv.create_app(jobs_dir=jobs, **kw))


@pytest.fixture(autouse=True)
def room_on_disk(monkeypatch):
    """The upload preflight is its own test. Elsewhere a nearly full machine must not fail unrelated attacks."""
    monkeypatch.setenv("SURGPROOF_MIN_FREE_MB", "1")


@pytest.fixture
def world(tmp_path, finished):
    jobs = tmp_path / "jobs"
    shutil.copytree(finished, jobs / "job1")
    (jobs / "secret2.txt").write_text("SECRET")
    with client(jobs) as c:
        sign_in(c)
        yield {"c": c, "jobs": jobs, "job": jobs / "job1", "fp": "/api/jobs/job1/final-pass", "files": "/api/jobs/job1/files"}


def items(w):
    r = w["c"].get(w["fp"])
    assert r.status_code == 200, r.text
    return r.json()["items"]


def act(w, it, action="approve", **kw):
    body = {"action": action, "kf": it.get("kf"), "label": it.get("label"), **kw}
    if it.get("frame_index") is not None:
        body["frame_index"] = it["frame_index"]
    return w["c"].post(w["fp"], json=body)


def clear_all(w):
    for it in items(w):
        r = act(w, it)
        assert r.status_code == 200, r.text


def release(w):
    return w["c"].post(w["fp"], json={"action": "release"})


# ---------- core 1: input gate ----------

@pytest.mark.parametrize("mode", ["not_surgery", "phase_unknown", "all_absent"])
def test_core1_footage_that_is_not_the_procedure_fails_and_names_it(tmp_path, mode):
    """A cat video, or any video where nothing can be labelled, must not come out as a dataset of negatives."""
    job = tmp_path / "job"
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, fake_env(mode))
    assert p.returncode != 0, p.stdout[-800:]
    st = read_json(job / "status.json")
    assert st["error"] and "cholecystectomy" in st["error"].lower(), st
    assert not (job / "dataset_coco.json").exists()


def test_core1_other_procedure_is_an_out_of_scope_result(tmp_path):
    job = tmp_path / "job"
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, fake_env("other_procedure"))
    assert p.returncode != 0
    st = read_json(job / "status.json")
    assert st.get("result") == "out_of_scope", st
    assert "cholecystectomy" in st["error"].lower()
    assert not (job / "dataset_coco.json").exists()


# ---------- core 2: amber may not settle a reject, and a dead second pass stays amber ----------

def test_core2_amber_loop_cannot_settle_a_reject(tmp_path):
    job = tmp_path / "job"
    run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, fake_env("amber_says_absent"))
    went_through_loop = [r for r in read_jsonl(job / "decisions.jsonl") if r["rounds"]]
    assert went_through_loop, "the mock's disagreement never reached the amber loop"
    bad = [(r["kf"], r["label"]) for r in went_through_loop if r["status"] != "AMBER"]
    assert not bad, f"amber said absent and the label left amber: {bad}"


def test_core2_dead_second_pass_fails_the_job_and_settles_nothing(tmp_path):
    job = tmp_path / "job"
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, fake_env("second_pass_dead"))
    assert p.returncode != 0, "one pass alone produced a dataset"
    assert not (job / "dataset_coco.json").exists()
    assert not [r for r in read_jsonl(job / "decisions.jsonl") if r["status"] != "AMBER"]


def test_core2_keyframe_missing_its_second_pass_stays_amber_even_if_the_loop_is_sure(tmp_path):
    """One keyframe loses its second pass, and the amber loop would happily say present. That is still one opinion."""
    job = tmp_path / "job"
    env = {**fake_env("second_pass_partly_dead"), "FAKE_CLAUDE_COUNTER": str(tmp_path / "count"), "SURGPROOF_VLM_CONCURRENCY": "1"}
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, env)
    assert p.returncode == 0, p.stdout[-500:] + p.stderr[-800:]
    rows = read_jsonl(job / "decisions.jsonl")
    lost = [r for r in rows if r.get("route") == "pass_failed"]
    assert lost, sorted({r.get("route") for r in rows})
    assert {r["status"] for r in lost} == {"AMBER"}, lost[0]
    assert all(r["final_present"] is None for r in lost)
    assert any(r["status"] == "accept" for r in rows), "the other keyframes should still decide"


def test_core2_every_label_exports_how_it_was_decided(finished):
    rows = read_jsonl(finished / "decisions.jsonl")
    assert rows and all(r.get("route") for r in rows)
    coco = read_json(finished / "dataset_coco.json")
    keyframe_anns = [a for a in coco["annotations"] if not a["attributes"].get("propagated")]
    assert keyframe_anns
    assert all(a["attributes"].get("decision_route") for a in keyframe_anns), keyframe_anns[0]["attributes"]
    flowed = [a for a in coco["annotations"] if a["attributes"].get("propagated")]
    assert flowed and all(a["attributes"].get("decision_route") for a in flowed)
    provenance = [r for r in read_jsonl(finished / "provenance.jsonl") if r.get("type") == "keyframe_label"]
    assert provenance, "provenance has no per-label rows"
    assert all(r.get("route") for r in provenance), [r for r in provenance if not r.get("route")][:1]


# ---------- core 3: ingest keeps no identity ----------

def test_core3_metadata_and_audio_do_not_survive(tmp_path):
    video = tagged_video(tmp_path / f"{PATIENT}.mp4")
    assert probe(video)["format"].get("tags", {}).get("title") == SECRET  # the attack is really in the file
    job = tmp_path / "job"
    p = run_cli(video, job, fake_env("amber_stuck"))
    assert p.returncode == 0, p.stderr[-800:]
    for name in ("source.mp4", "annotated.mp4"):
        info = probe(job / name)
        assert all(s["codec_type"] != "audio" for s in info["streams"]), f"{name} kept audio"
        assert SECRET.encode() not in (job / name).read_bytes(), f"{name} kept metadata"
        assert b"51.5074" not in (job / name).read_bytes(), f"{name} kept the location"
    for frame in (job / "frames").glob("*.jpg"):
        assert SECRET.encode() not in frame.read_bytes()


def test_core3_upload_filename_is_stored_nowhere(tmp_path):
    job = tmp_path / "job"
    p = run_cli(tagged_video(tmp_path / f"{PATIENT}.mp4"), job, fake_env("amber_stuck"))
    assert p.returncode == 0
    for path, text in all_text_under(job):
        assert PATIENT not in text and SECRET not in text, path
    assert not [x for x in job.rglob("*") if PATIENT in x.name]


def test_core3_cli_default_job_folder_does_not_carry_the_name(tmp_path):
    video = tagged_video(tmp_path / f"{PATIENT}.mp4")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    import sys

    p = subprocess.run([sys.executable, "-m", "surgproof.run", str(video)], cwd=cwd, env=base_env(fake_env()),
                       capture_output=True, text=True, timeout=300)
    assert PATIENT not in p.stdout + p.stderr, (p.stdout + p.stderr)[-400:]
    assert not [x for x in cwd.rglob("*") if PATIENT in x.name]


def test_core3_web_upload_name_is_not_kept_or_listed(tmp_path, monkeypatch):
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    video = tagged_video(tmp_path / "v.mp4")
    jobs = tmp_path / "jobs"
    with client(jobs) as c:
        sign_in(c)
        r = upload(c, f"{PATIENT}.mp4", video.read_bytes())
        assert r.status_code == 201, r.text
        assert PATIENT not in r.text
        jid = r.json()["id"]
        deadline = time.time() + 180
        while time.time() < deadline and c.get(f"/api/jobs/{jid}").json()["state"] not in ("done", "failed", "stalled"):
            time.sleep(0.3)
        assert PATIENT not in c.get("/api/jobs").text and PATIENT not in c.get(f"/api/jobs/{jid}").text
    for path, text in all_text_under(jobs):
        assert PATIENT not in text, path
    assert not [x for x in jobs.rglob("*") if PATIENT in x.name]


def test_core3_failure_message_does_not_echo_the_name(tmp_path):
    bad = tmp_path / f"{PATIENT}.mp4"
    bad.write_bytes(b"not a video")
    job = tmp_path / "job"
    p = run_cli(bad, job, fake_env())
    assert p.returncode != 0
    assert PATIENT not in p.stdout + p.stderr
    for path, text in all_text_under(job):
        assert PATIENT not in text, path


# ---------- core 4: timestamps, timeouts, protocols, caps ----------

def _pts(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0",
                          str(path)], check=True, capture_output=True, text=True).stdout
    return [float(x.strip(",")) for x in out.split() if x.strip(",")]


def test_core4_jittered_timestamps_are_normalised(tmp_path):
    """Average frame rate looks perfectly normal, so a frame-rate ratio test passes it. The timestamps do not."""
    video = tmp_path / "jitter.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=s=320x180:r=10:d=4", "-vf",
         "setpts='(N/10+0.04*mod(N,2))/TB'", "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)],
        check=True)
    before = _pts(video)
    gaps = [b - a for a, b in zip(before, before[1:])]
    assert max(gaps) - min(gaps) > 0.05, "the test video is not actually jittered"
    job = tmp_path / "job"
    p = run_cli(video, job, fake_env("amber_stuck"))
    assert p.returncode == 0, p.stderr[-800:]
    pts = _pts(job / "source.mp4")
    gaps = [b - a for a, b in zip(pts, pts[1:])]
    assert max(gaps) - min(gaps) < 0.005, f"source.mp4 timestamps are still uneven: {min(gaps)} to {max(gaps)}"


def test_core4_every_media_subprocess_has_a_timeout():
    """A hung ffmpeg on a crafted file must not hold the job forever."""
    missing = []
    for name in ("pipeline", "render", "export", "masks", "propagate", "server", "vlm"):
        path = ROOT / "surgproof" / f"{name}.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("run", "check_output", "check_call", "call")
                    and getattr(node.func.value, "id", "") == "subprocess"
                    and not any(k.arg == "timeout" or k.arg is None for k in node.keywords)):
                missing.append(f"{name}.py:{node.lineno}")
    assert not missing, missing


@pytest.mark.parametrize("build", ["hls", "ffconcat"])
def test_core4_playlist_disguised_as_mp4_cannot_read_other_files(tmp_path, build):
    outside = synthetic_video(tmp_path / "outside" / "private.mp4", seconds=2)
    video = tmp_path / "evil.mp4"
    if build == "hls":
        video.write_text(f"#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-TARGETDURATION:3\n#EXTINF:2.0,\nfile://{outside}\n#EXT-X-ENDLIST\n")
    else:
        video.write_text(f"ffconcat version 1.0\nfile '{outside}'\n")
    job = tmp_path / "job"
    p = run_cli(video, job, fake_env())
    assert p.returncode != 0, "a text playlist was accepted as a video"
    assert not list(job.glob("frames/*.jpg"))


def test_core4_playlist_upload_is_refused_by_the_server(world, tmp_path):
    outside = synthetic_video(tmp_path / "outside" / "private.mp4", seconds=2)
    data = f"#EXTM3U\n#EXTINF:2.0,\nfile://{outside}\n#EXT-X-ENDLIST\n".encode()
    assert upload(world["c"], "evil.mp4", data).status_code == 422


@pytest.mark.parametrize("src", ["color=c=black:s=6000x4000:r=5:d=0.4", "color=c=black:s=64x36:r=1000:d=0.1"])
def test_core4_absurd_size_or_frame_rate_is_refused(tmp_path, src):
    video = tmp_path / "big.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", src, "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(video)], check=True)
    job = tmp_path / "job"
    log = tmp_path / "calls.log"
    p = run_cli(video, job, fake_env(log=log))
    assert p.returncode != 0
    assert "limit" in (p.stdout + p.stderr).lower()
    assert not log.exists(), "the model was called on a video that should have been refused"


# ---------- core 5: cost and plan limits ----------

def test_core5_absurd_keyframe_rate_is_refused_before_any_call(tmp_path):
    log = tmp_path / "calls.log"
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=2), tmp_path / "job", fake_env(log=log), extra_args=["--kf-fps", "1000"])
    assert p.returncode != 0 and not log.exists()


def test_core5_cost_cap_stops_spending_and_a_rerun_pays_for_nothing_twice(tmp_path):
    log = tmp_path / "calls.log"
    video = synthetic_video(tmp_path / "v.mp4", seconds=4)
    job = tmp_path / "job"
    env = {**fake_env("amber_stuck", log=log), "FAKE_CLAUDE_COST": "0.5", "SURGPROOF_MAX_COST_USD": "1.2",
           "SURGPROOF_CACHE": str(tmp_path / "cache.sqlite"), "SURGPROOF_VLM_CONCURRENCY": "1"}
    p = run_cli(video, job, env)
    assert p.returncode != 0, "spent past the cap"
    st = read_json(job / "status.json")
    assert st.get("result") == "cost_cap", st
    assert st["cost_usd"] <= 1.2 + 0.5
    first = len(log.read_text().splitlines())
    p = run_cli(video, job, {**env, "SURGPROOF_MAX_COST_USD": "1000"})
    assert p.returncode == 0, p.stderr[-800:]
    prompts = [json.loads(x)["argv"] for x in log.read_text().splitlines()]
    bodies = [a[a.index("-p") + 1] for a in prompts]
    assert len(bodies) == len(set(bodies)), "a call already paid for was made again"
    assert first < len(bodies)


def test_core5_plan_limit_pauses_and_carries_on(tmp_path):
    job = tmp_path / "job"
    env = {**fake_env("plan_limit_once"), "FAKE_CLAUDE_COUNTER": str(tmp_path / "count"),
           "SURGPROOF_VLM_PAUSE_POLL_S": "0.1", "SURGPROOF_VLM_CONCURRENCY": "1"}
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=3), job, env)
    assert p.returncode == 0, p.stdout[-400:] + p.stderr[-800:]
    assert read_json(job / "status.json")["stage"] == "done"


def test_core5_plan_limit_that_never_lifts_leaves_a_paused_job_not_a_failed_one(tmp_path):
    job = tmp_path / "job"
    env = {**fake_env("plan_limit"), "SURGPROOF_VLM_PAUSE_POLL_S": "0.05", "SURGPROOF_VLM_MAX_PAUSE_S": "0.2"}
    video = synthetic_video(tmp_path / "v.mp4", seconds=3)
    p = run_cli(video, job, env)
    st = read_json(job / "status.json")
    assert st.get("paused") is True and st.get("result") == "paused", st
    assert "usage limit" in (st.get("error") or st.get("message") or "").lower()


@pytest.mark.parametrize("wording", [
    "Claude usage limit reached. Your limit will reset at 7pm",
    "Claude AI usage limit reached|1793000000",
    "You've hit your limit",
    "You've hit your usage limit. Resets 7pm",
    "5-hour limit reached",
    "Weekly limit reached, resets Monday",
    "rate limit exceeded",
    # Strings found in the installed Claude Code 2.1.316 binary. Which of them reach a headless call is not known.
    "Usage limit reached",
    "You've reached your Fable limit",
    "You've hit your limit \u00b7 resets 7pm",
    "Your 5-hour limit resets at 7pm",
    "until your limit resets at 7pm",
])
def test_core5_real_limit_wordings_are_recognised(wording):
    """The pause rests on recognising the wording of a limit, so each known form has to match."""
    from surgproof import vlm

    assert vlm.is_usage_limit(wording), wording


@pytest.mark.parametrize("wording", [
    "Parse error at line 429 of the response",
    "cannot read /tmp/frames/kf_00429/429.jpg",
    "connection refused",
    "Not logged in. Please run /login",
    "output token limit reached",
    # From the installed binary too. The first two name the usage limit only to say it is not the cause.
    "Server is temporarily limiting requests (not your usage limit)",
    "Claude is temporarily limiting requests to the usage page (not your usage limit). Try again in a moment.",
    "Context limit reached",
    "Budget limit reached ($5.00 of $5)",
    "Concurrent subagent limit reached.",
])
def test_core5_ordinary_errors_are_not_mistaken_for_a_limit(wording):
    """A real failure read as a plan limit would sit paused for hours instead of failing."""
    from surgproof import vlm

    assert not vlm.is_usage_limit(wording), wording


# ---------- core 6: narration is labelled for what it is ----------

@pytest.mark.parametrize("mode", ["amber_stuck", "narrate_lies"])
def test_core6_subtitles_say_ai_draft(tmp_path, mode):
    job = tmp_path / "job"
    run_cli(synthetic_video(tmp_path / "v.mp4", seconds=3), job, fake_env(mode))
    vtt = (job / "narration.vtt").read_text()
    assert "AI draft" in vtt, vtt[:300]
    assert read_json(job / "narration.json")["segments"]
    assert all("verified" not in s["text"].lower() for s in read_json(job / "narration.json")["segments"])


def test_no_output_or_screen_says_verified(finished):
    """The word promises more than one AI pass and a sampled human check can deliver."""
    for path, text in all_text_under(finished):
        if path.name in ("decisions.jsonl", "narration.json", "narration.vtt", "status.json", "dataset_coco.json"):
            assert not re.search(r"\bverified\b|verified_absent", text, re.I), path
    shown = list((ROOT / "web" / "app").glob("*.*"))
    assert shown
    for path in shown:
        assert not re.search(r"\bverified\b", path.read_text(errors="replace"), re.I), path


# ---------- masks 1: human_reviewed means a human saw that exact annotation ----------

def _coco_after_release(w):
    clear_all(w)
    r = release(w)
    assert r.status_code == 200, r.text
    return read_json(w["job"] / "released_coco.json")


def test_masks1_propagated_annotation_is_not_human_reviewed(world):
    """Approving a keyframe says nothing about the frames flow drew from it. Only an audited frame was seen."""
    seen = {i["kf"] for i in items(world) if str(i["kf"]).startswith("prop_")}
    coco = _coco_after_release(world)
    flowed = [a for a in coco["annotations"] if a["attributes"].get("propagated")]
    assert flowed, "the job has no propagated annotations to test"
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    audited = {(int(k.split("_")[1]), i["label"]) for i in items(world) for k in [i["kf"]] if k in seen}
    claimed = {(frames[a["image_id"]], cats[a["category_id"]]) for a in flowed if a["attributes"]["human_reviewed"]}
    assert claimed <= audited, f"{len(claimed - audited)} propagated annotations claim a person saw them"
    assert len(claimed) < len(flowed), "every propagated frame claims a person saw it"
    assert all("derived_from_reviewed" in a["attributes"] for a in flowed)
    assert any(a["attributes"]["derived_from_reviewed"] for a in flowed if not a["attributes"]["human_reviewed"])


def test_masks1_keyframe_nobody_opened_is_not_human_reviewed(world):
    seen = {(i["kf"], i["label"]) for i in items(world)}
    coco = _coco_after_release(world)
    meta = read_json(world["job"] / "meta.json")
    kf_at = {k["frame_index"]: k["kf"] for k in meta["keyframes"]}
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    checked = 0
    for a in coco["annotations"]:
        if a["attributes"].get("propagated"):
            continue
        key = (kf_at[frames[a["image_id"]]], cats[a["category_id"]])
        checked += 1
        assert bool(a["attributes"]["human_reviewed"]) == (key in seen), key
    assert checked


# ---------- masks 2: the audit can stop a release ----------

def test_masks2_failed_audit_blocks_release_and_escalates(world):
    before = len(items(world))
    audit = [i for i in items(world) if i.get("kind") != "amber"]
    assert audit
    for it in items(world):
        act(world, it, "reject") if it.get("kind") != "amber" else act(world, it)
    state = world["c"].get(world["fp"]).json()
    r = release(world)
    assert r.status_code >= 400, "a failed audit was released"
    assert not (world["job"] / "released_coco.json").exists()
    assert state.get("escalated") or len(items(world)) > before, "failure did not escalate to full review"


def test_masks2_empty_queue_never_releases(world, monkeypatch):
    from surgproof import export

    monkeypatch.setattr(export, "final_pass_queue", lambda *a, **k: [])
    q = world["job"] / "final_pass_queue.json"
    if q.exists():
        q.write_text("[]")
    with pytest.raises(Exception):
        export.release(world["job"], reviewer="qa")
    assert not (world["job"] / "released_coco.json").exists()


# ---------- masks 3: the audit looks where errors hide ----------

def test_masks3_audit_includes_rejects_and_propagated_frames_for_each_label(world):
    queue = items(world)
    kinds = {i.get("stratum") or i.get("kind") for i in queue}
    joined = " ".join(str(k) for k in kinds).lower()
    assert "propagat" in joined, kinds
    assert "reject" in joined or "absent" in joined, kinds
    labels = {i["label"] for i in queue if i.get("kind") != "amber"}
    decided = {r["label"] for r in read_jsonl(world["job"] / "decisions.jsonl") if r["status"] != "AMBER"}
    assert decided <= labels, f"labels with no audit sample: {decided - labels}"


def test_masks3_dataset_says_ai_absent(finished):
    text = (finished / "dataset_coco.json").read_text()
    assert "verified_absent" not in text and "ai_absent" in text


# ---------- masks 4: what was released is what you have ----------

def test_masks4_release_rerenders_the_video_and_the_draft_is_marked(world):
    from surgproof import render

    banner, note = render.status_text(False)
    assert "UNREVIEWED" in banner.upper() and "DRAFT" in banner.upper(), "the pre-release banner must say unreviewed draft"
    assert "UNREVIEWED" not in render.status_text(True)[0].upper()
    before = (world["job"] / "annotated.mp4").read_bytes()
    _coco_after_release(world)
    assert (world["job"] / "annotated.mp4").read_bytes() != before, "annotated.mp4 was not re-rendered on release"


def test_masks4_editing_a_released_file_is_detected(world):
    from surgproof import export

    _coco_after_release(world)
    assert export.verify_release(world["job"])["ok"] is True
    coco = read_json(world["job"] / "released_coco.json")
    coco["annotations"] = coco["annotations"][:-1]
    (world["job"] / "released_coco.json").write_text(json.dumps(coco))
    assert export.verify_release(world["job"])["ok"] is False


def test_masks4_torn_last_line_does_not_break_the_final_pass(world):
    it = items(world)[0]
    assert act(world, it).status_code == 200
    with open(world["job"] / "final_pass.jsonl", "a") as f:
        f.write('{"kf": "kf_00000", "label": "li')
    r = world["c"].get(world["fp"])
    assert r.status_code == 200, r.text
    assert act(world, items(world)[0]).status_code == 200


# ---------- masks 5: propagation has to agree with itself ----------

def test_masks5_object_that_teleports_goes_amber_between_keyframes(tmp_path):
    job = tmp_path / "job"
    env = {**fake_env("polygon_jumps"), "FAKE_CLAUDE_COUNTER": str(tmp_path / "count"), "SURGPROOF_VLM_CONCURRENCY": "1"}
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=4), job, env)
    assert p.returncode == 0, p.stdout[-400:] + p.stderr[-800:]
    rows = [r for r in read_jsonl(job / "tracks.jsonl") if r.get("propagated")]
    assert rows
    assert any(r.get("fb_overlap") is not None for r in rows), "no forward/backward overlap recorded per frame"
    low = [r for r in rows if r.get("flow_check") == "low_overlap"]
    assert low, "objects jumped across the frame and flow still agreed"
    assert {r["status"] for r in low} == {"AMBER"}, "low overlap did not go amber"
    # And the dataset does not carry them.
    coco = read_json(job / "dataset_coco.json")
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    exported = {(frames[a["image_id"]], cats[a["category_id"]]) for a in coco["annotations"]}
    assert not exported & {(r["frame_index"], r["label"]) for r in low}


def test_masks5_excluded_amber_never_reads_as_a_negative(finished):
    """A training script reads a missing annotation as absent, unless the image says that label was left out."""
    coco = read_json(finished / "dataset_coco.json")
    amber = {r["label"] for r in read_jsonl(finished / "decisions.jsonl") if r["status"] == "AMBER"}
    assert amber
    keyframes = {k["frame_index"] for k in read_json(finished / "meta.json")["keyframes"]}
    flowed = [i for i in coco["images"] if i["frame_index"] not in keyframes]
    assert flowed
    held = [i for i in flowed if set(i["attributes"]["excluded_amber"]) & amber]
    assert held, "amber labels are missing from propagated images with nothing saying so"
    assert all(i["attributes"]["exhaustive"] is False for i in flowed)


# ---------- masks 6: masks are drafts ----------

def test_masks6_every_mask_is_a_draft_with_an_instance_id(finished):
    coco = read_json(finished / "dataset_coco.json")
    assert coco["annotations"]
    for a in coco["annotations"]:
        at = a["attributes"]
        assert at.get("mask_quality") == "draft", at
        assert isinstance(at.get("instance_id"), int), at
    assert coco["info"]["mask_quality"] == "draft"
    for row in read_jsonl(finished / "masks.jsonl"):
        assert row.get("quality") in ("draft", None) or "fallback" not in str(row.get("quality"))


# ---------- masks 7: the bundle stands on its own ----------

def test_masks7_bundle_resolves_and_explains_itself(world):
    import zipfile

    assert world["c"].get(f"{world['files']}/dataset_bundle.zip").status_code in (403, 404), "bundle served before release"
    _coco_after_release(world)
    bundle = world["job"] / "dataset_bundle.zip"
    assert bundle.exists()
    assert world["c"].get(f"{world['files']}/dataset_bundle.zip").status_code == 200

    with zipfile.ZipFile(bundle) as z:
        names = set(z.namelist())
        coco = json.loads(z.read(next(n for n in names if n.endswith("released_coco.json"))))
        missing = [i["file_name"] for i in coco["images"] if i["file_name"] not in names and "source.mp4" not in names]
        assert not missing, missing[:5]
        for want in ("card", "schema", "licen", "phase", "narration", "tips", "prompts", "versions"):
            assert any(want in n.lower() for n in names), f"bundle has nothing for {want}: {sorted(names)[:30]}"
        prompts = z.read(next(n for n in names if "prompts" in n.lower())).decode()
        assert "This is one frame" in prompts, "full prompt text missing"
        assert re.search(r"claude-[a-z]+-\d", json.dumps(coco["info"]) + prompts) or "mock" in prompts


# ---------- web 1: local, signed in, allowlisted ----------

def test_web1_serve_binds_loopback_even_if_told_otherwise(tmp_path):
    assert "0.0.0.0" not in (ROOT / "Makefile").read_text()
    assert "0.0.0.0" not in (ROOT / "README.md").read_text()
    assert srv.BIND_HOST == "127.0.0.1"
    assert "HOST" not in re.findall(r"environ(?:\.get)?\W+(\w+)", (ROOT / "surgproof" / "server.py").read_text())


@pytest.mark.parametrize("path", ["/api/jobs", "/api/jobs/job1", "/api/jobs/job1/final-pass", "/api/jobs/job1/files/frames/kf_00000.jpg",
                                  "/api/jobs/job1/video", "/api/jobs/job1/overlay?start=0&end=5"])
def test_web1_nothing_is_served_without_signing_in(tmp_path, finished, path):
    jobs = tmp_path / "jobs"
    shutil.copytree(finished, jobs / "job1")
    with client(jobs) as c:
        assert c.get(path).status_code == 401, path


def test_web1_reviewer_comes_from_the_session_not_the_body(world):
    it = items(world)[0]
    assert act(world, it, reviewer="Dr Forged").status_code == 200
    rows = read_jsonl(world["job"] / "final_pass.jsonl")
    assert rows[-1]["reviewer"] == REVIEWER, rows[-1]
    assert "Dr Forged" not in (world["job"] / "final_pass.jsonl").read_text()


def test_web1_forged_host_header_is_refused(world):
    """DNS rebinding: a web page on evil.example reaches 127.0.0.1 with its own Host."""
    assert world["c"].get("/api/health", headers={"host": "evil.example"}).status_code == 400


def test_web1_session_cookie_is_not_readable_from_script(tmp_path):
    with client(tmp_path / "jobs") as c:
        code = c.app.state.auth.setup_code()
        r = c.post("/api/auth/setup", json={"name": "a", "password": "correct horse battery", "setup_code": code})
        cookie = r.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite" in cookie


def test_web1_wrong_password_and_wrong_setup_code_fail(tmp_path):
    with client(tmp_path / "jobs") as c:
        assert c.post("/api/auth/setup", json={"name": "a", "password": "correct horse battery", "setup_code": "guess"}).status_code >= 400
        sign_in(c)
        c.post("/api/auth/logout")
        assert c.get("/api/jobs").status_code == 401
        assert c.post("/api/auth/login", json={"password": "wrong"}).status_code == 401


@pytest.mark.parametrize("name", ["decisions.jsonl", "detect.jsonl", "meta.json", "job.json", "status.json", "phase.jsonl",
                                  "masks.jsonl", "tracks.jsonl", "eval_report.json", "../secret2.txt", "..%2fsecret2.txt",
                                  "frames/../../secret2.txt", "upload.mp4", "source.mp4", "amber/x.jpg"])
def test_web1_file_route_is_an_allowlist(world, name):
    if name == "eval_report.json":
        (world["job"] / name).write_text("{}")
    assert world["c"].get(f"{world['files']}/{name}").status_code in (403, 404), name


@pytest.mark.parametrize("name", ["dataset_coco.json", "DATASET_COCO.json", "released_coco.json", "final_pass.jsonl",
                                  "./dataset_coco.json", "frames/../dataset_coco.json"])
def test_web1_unreleased_artifacts_are_gated_however_you_spell_them(world, name):
    assert world["c"].get(f"{world['files']}/{name}").status_code in (403, 404), name


# ---------- web 2: notice, attestation, delete, retention ----------

@pytest.mark.parametrize("fields", [{"rights_attested": None}, {"notice_sha256": "0" * 64}, {"footage_source": "hospital"},
                                    {"footage_source": None}])
def test_web2_upload_without_the_notice_and_attestation_is_refused(world, tmp_path, fields):
    data = synthetic_video(tmp_path / "v.mp4", seconds=2).read_bytes()
    r = upload(world["c"], "v.mp4", data, **fields)
    assert r.status_code == 422, r.text
    assert not list(world["jobs"].glob(".upload-*"))


def test_web2_attestation_is_stored_with_who_signed(tmp_path, monkeypatch):
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    jobs = tmp_path / "jobs"
    with client(jobs) as c:
        sign_in(c)
        r = upload(c, "v.mp4", synthetic_video(tmp_path / "v.mp4", seconds=2).read_bytes())
        assert r.status_code == 201, r.text
        rec = read_json(jobs / r.json()["id"] / "job.json")
    assert rec["attestation"]["by"] == REVIEWER and rec["attestation"]["footage_source"]


def test_web2_delete_removes_the_job_and_its_cache_entries(tmp_path, monkeypatch):
    cache = tmp_path / "cache.sqlite"
    job = tmp_path / "jobs" / "job1"
    p = run_cli(synthetic_video(tmp_path / "v.mp4", seconds=3), job, {**fake_env("amber_stuck"), "SURGPROOF_CACHE": str(cache)})
    assert p.returncode == 0
    assert cache_rows(cache)
    monkeypatch.setenv("SURGPROOF_CACHE", str(cache))
    with client(tmp_path / "jobs") as c:
        sign_in(c)
        r = c.delete("/api/jobs/job1")
        assert r.status_code == 200, r.text
        assert c.get("/api/jobs/job1").status_code == 404
    assert not job.exists()
    assert not cache_rows(cache), "the model's answers about a deleted video stayed in the cache"


def test_web2_old_jobs_are_swept_at_startup(tmp_path, finished, monkeypatch):
    jobs = tmp_path / "jobs"
    shutil.copytree(finished, jobs / "old")
    shutil.copytree(finished, jobs / "new")
    (jobs / "old" / "job.json").write_text(json.dumps({"id": "old", "created": "2020-01-01T00:00:00+00:00"}))
    (jobs / "new" / "job.json").write_text(json.dumps({"id": "new", "created": srv.now_iso()}))
    monkeypatch.setenv("SURGPROOF_RETENTION_DAYS", "30")
    with client(jobs):
        pass
    assert not (jobs / "old").exists() and (jobs / "new").exists()


# ---------- web 3: disk, uploads, failures, reruns ----------

def test_web3_full_disk_is_refused_before_the_body_is_read(world, tmp_path, monkeypatch):
    monkeypatch.setenv("SURGPROOF_MIN_FREE_MB", str(10 ** 12))
    r = upload(world["c"], "v.mp4", synthetic_video(tmp_path / "v.mp4", seconds=1).read_bytes())
    assert r.status_code == 507, r.text
    assert not list(world["jobs"].glob(".upload-*"))


def test_web3_oversized_upload_is_refused_from_its_headers(world):
    r = world["c"].post("/api/jobs", content=b"x", headers={"content-type": "multipart/form-data; boundary=b",
                                                            "content-length": str(10 ** 13)})
    assert r.status_code == 413, r.text
    assert not list(world["jobs"].glob(".upload-*"))


def test_web3_stream_over_the_limit_stops_early(world, monkeypatch):
    monkeypatch.setattr(srv, "MAX_BYTES", 1024)
    r = upload(world["c"], "v.mp4", b"\x00" * 200_000)
    assert r.status_code == 413, r.text
    assert not list(world["jobs"].glob(".upload-*"))


def test_web3_leftover_staging_is_swept_at_startup(tmp_path):
    jobs = tmp_path / "jobs"
    (jobs / ".upload-crashed").mkdir(parents=True)
    (jobs / ".upload-crashed" / "upload.mp4").write_bytes(b"x" * 100)
    with client(jobs):
        pass
    assert not (jobs / ".upload-crashed").exists()


def test_web3_crash_in_the_worker_is_surfaced_not_silently_running(tmp_path, monkeypatch):
    video = synthetic_video(tmp_path / "v.mp4", seconds=2)
    jobs = tmp_path / "jobs"
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    with client(jobs, pipeline_target="surgproof.does_not_exist:run") as c:
        sign_in(c)
        r = upload(c, "v.mp4", video.read_bytes())
        assert r.status_code == 201, r.text
        jid = r.json()["id"]
        deadline, j = time.time() + 60, {}
        while time.time() < deadline:
            j = c.get(f"/api/jobs/{jid}").json()
            if j["state"] in ("failed", "stalled"):
                break
            time.sleep(0.2)
        assert j["state"] == "failed" and j["error"], j


def test_web3_dead_worker_is_reported_as_stalled(world):
    st = read_json(world["job"] / "status.json")
    st.update(stage="masks", error=None, progress=0.5)
    (world["job"] / "status.json").write_text(json.dumps(st))
    (world["job"] / srv.WORKER_LOCK).write_text("0")  # this server started it, and nothing holds the lock now
    assert world["c"].get("/api/jobs/job1").json()["state"] == "stalled"


def test_web3_retry_after_release_is_refused(world):
    _coco_after_release(world)
    before = (world["job"] / "released_coco.json").read_bytes()
    assert world["c"].post("/api/jobs/job1/retry").status_code == 409
    assert (world["job"] / "released_coco.json").read_bytes() == before


def test_web3_retry_after_a_review_has_started_is_refused(world):
    assert act(world, items(world)[0]).status_code == 200
    st = read_json(world["job"] / "status.json")
    st.update(error="x", stage="masks")
    (world["job"] / "status.json").write_text(json.dumps(st))
    assert world["c"].post("/api/jobs/job1/retry").status_code == 409


def test_web3_server_kills_the_whole_process_group():
    text = (ROOT / "surgproof" / "server.py").read_text()
    assert "start_new_session=True" in text and "killpg" in text


# ---------- web 4: the screens say what happened ----------

def test_web4_health_reports_disk_and_model(world):
    h = world["c"].get("/api/health").json()
    assert h["disk"]["free_bytes"] > 0 or h["disk"].get("free_mb", 0) > 0
    assert isinstance(h["model"]["ok"], bool) and h["model"]["provider"]


def test_web4_player_draws_human_ai_and_propagated_differently(world):
    objs = [o for f in world["c"].get("/api/jobs/job1/overlay?start=0&end=40").json()["frames"].values() for o in f]
    assert objs
    assert {o.get("origin") for o in objs} >= {"ai_keyframe", "propagated"}, {o.get("origin") for o in objs}
    js = (ROOT / "web" / "app" / "watch.js").read_text() + (ROOT / "web" / "app" / "lib.js").read_text()
    assert all(k in js for k in ("human", "ai_keyframe", "propagated"))


def test_web4_watch_shows_a_failure_when_overlays_cannot_load():
    js = (ROOT / "web" / "app" / "watch.js").read_text()
    assert re.search(r"catch|\.ok\b", js) and re.search(r"could not|cannot|failed|unavailable", js, re.I)


def test_web4_download_page_wording_matches_the_review():
    js = (ROOT / "web" / "app" / "download.js").read_text()
    assert re.search(r"draft", js, re.I)
    assert not re.search(r"\bverified\b|fully reviewed|all frames reviewed", js, re.I)


# ---------- web 5: dependencies are pinned ----------

def test_web5_runtime_lock_pins_everything_and_holds_no_dev_tools():
    lock = (ROOT / "requirements-e2e.lock").read_text()
    rows = [x.strip() for x in lock.splitlines() if x.strip() and not x.lstrip().startswith(("#", "--"))]
    pinned = [x for x in rows if not x.startswith("sha256")]
    assert pinned and all(re.match(r"^[\w.\-\[\]]+==[\w.\-+]+", x) for x in pinned), [x for x in pinned if "==" not in x][:5]
    assert not any(x.lower().startswith(("pytest", "httpx==")) for x in pinned), "dev tools in the runtime lock"
    assert "requirements-e2e.lock" in (ROOT / "Makefile").read_text()
    assert re.search(r"(?m)^pytest", (ROOT / "requirements-e2e-dev.txt").read_text())
