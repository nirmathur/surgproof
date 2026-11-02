"""Acceptance: the real CLI on a 6-second video with a mocked model, checked against the stage I/O contract
in docs/E2E_SPEC.md. Then the same run for real, when SURGPROOF_E2E_REAL=1.

The mock proves plumbing only. No number from it means anything, and the scorer refuses to report one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.e2e_helpers.harness import (HAVE_DATA, HAVE_FFMPEG, LABELS, ROOT, cache_rows, count_frames, read_json,
                                       read_jsonl, run_cli, run_eval, synthetic_video, video_fps)

pytestmark = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg not installed")
pytest.importorskip("surgproof.pipeline", reason="core lane not merged in this checkout")
for _mod in ("masks", "propagate", "render", "export"):
    pytest.importorskip(f"surgproof.{_mod}", reason="masks lane not merged in this checkout")

SECONDS = 6
STATUS_KEYS = {"stage", "progress", "message", "started", "updated", "error"}


@pytest.fixture(scope="module")
def test_video(tmp_path_factory):
    """A held-out CholecSeg8k clip when this machine has the data, a synthetic pattern otherwise."""
    d = tmp_path_factory.mktemp("video")
    if HAVE_DATA:
        from tools import make_test_video as mtv

        out = d / "video12_6s.mp4"
        registry = d / "registry"
        mtv.make(ROOT / "raw" / "CholecSeg8k", out, registry, video="video12", seconds=SECONDS)
        return {"path": out, "registry": registry, "real_frames": True}
    return {"path": synthetic_video(d / "synthetic.mp4", seconds=SECONDS, size="854x480", fps=25), "registry": None, "real_frames": False}


@pytest.fixture(scope="module")
def mock_job(tmp_path_factory, test_video):
    parent = tmp_path_factory.mktemp("jobs")
    job = parent / "mock"
    repo_cache = ROOT / "jobs" / "cache.sqlite"
    before = repo_cache.stat().st_mtime if repo_cache.exists() else None
    proc = run_cli(test_video["path"], job, {"SURGPROOF_VLM": "mock"})
    assert proc.returncode == 0, f"CLI failed\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
    after = repo_cache.stat().st_mtime if repo_cache.exists() else None
    return {"job": job, "cache": parent / "cache.sqlite", "stdout": proc.stdout, "repo_cache_untouched": before == after}


def _meta(job):
    return read_json(job["job"] / "meta.json")


def test_status_finished_with_contract_fields(mock_job):
    s = read_json(mock_job["job"] / "status.json")
    assert STATUS_KEYS <= set(s)
    assert s["error"] is None and s["progress"] == 1.0 and s["stage"] == "done"


def test_cli_reports_every_stage(mock_job):
    for stage in ("ingest", "detect", "verify", "amber", "masks", "propagate", "narrate", "render", "export"):
        assert f" {stage}" in mock_job["stdout"], f"no progress line for {stage}"


def test_meta_matches_the_video(mock_job, test_video):
    m = _meta(mock_job)
    assert m["n_frames"] == count_frames(test_video["path"]) == SECONDS * 25
    assert m["fps"] == 25 and m["width"] == 854 and m["height"] == 480 and m["kf_fps"] == 1
    assert [k["frame_index"] for k in m["keyframes"]] == [i * 25 for i in range(SECONDS)]
    for k in m["keyframes"]:
        assert (mock_job["job"] / "frames" / f"{k['kf']}.jpg").exists()
        assert k["t"] == pytest.approx(k["frame_index"] / 25)


def test_one_row_per_keyframe_and_label(mock_job):
    want = {(k["kf"], lb) for k in _meta(mock_job)["keyframes"] for lb in LABELS}
    for name in ("detect", "verify", "decisions"):
        rows = read_jsonl(mock_job["job"] / f"{name}.jsonl")
        got = [(r["kf"], r["label"]) for r in rows]
        assert len(got) == len(set(got)) == len(want), name
        assert set(got) == want, name


def test_row_fields(mock_job):
    j = mock_job["job"]
    for r in read_jsonl(j / "detect.jsonl"):
        assert {"kf", "label", "present", "conf", "polygon", "model", "prompt_hash", "raw_ref"} <= set(r)
        assert (r["polygon"] is None) or (r["present"] and len(r["polygon"]) >= 3)
    for r in read_jsonl(j / "verify.jsonl"):
        assert {"kf", "label", "present", "box", "model", "prompt_hash"} <= set(r)
        assert r["box"] is None or len(r["box"]) == 4
    for r in read_jsonl(j / "decisions.jsonl"):
        assert {"kf", "label", "status", "final_present", "rounds"} <= set(r)
        assert r["status"] in ("accept", "reject", "AMBER")
    phases = read_jsonl(j / "phase.jsonl")
    assert len(phases) == len(_meta(mock_job)["keyframes"])
    assert all({"kf", "phase", "conf"} <= set(p) for p in phases)


def test_mock_cannot_pass_for_real(mock_job):
    """Every model field on every artifact says mock. Nothing is laundered into a real-looking row."""
    j = mock_job["job"]
    for name in ("detect", "verify"):
        for r in read_jsonl(j / f"{name}.jsonl"):
            assert r["model"].startswith("mock:"), r
    coco = read_json(j / "dataset_coco.json")
    assert coco["annotations"], "mock run produced no annotations, so this test proves nothing"
    for a in coco["annotations"]:
        assert a["attributes"]["model"].startswith("mock:"), a


def test_repo_cache_not_polluted_by_tests(mock_job):
    assert mock_job["repo_cache_untouched"]
    assert all(v["model"].startswith("mock:") for v in cache_rows(mock_job["cache"]).values())


def test_masks_only_for_accepted_present(mock_job):
    j = mock_job["job"]
    meta = _meta(mock_job)
    decisions = read_jsonl(j / "decisions.jsonl")
    want = {f"{r['kf']}_{r['label']}.png" for r in decisions if r["status"] == "accept" and r["final_present"]}
    amber = {f"{r['kf']}_{r['label']}.png" for r in decisions if r["status"] == "AMBER"}
    got = {p.name for p in (j / "masks").glob("*.png")}
    assert want, "no masks at all: nothing tested"
    assert want <= got, f"accepted labels with no mask: {want - got}"
    # A mask beyond the accepted ones may only be a proposal for a reviewer: never a rejected label, always marked.
    # Or a propagated frame drawn out because the audit queued that exact frame for a person to look at.
    queued = {f"{i['kf']}_{i['label']}.png" for i in read_json(j / "final_pass_queue.json")["items"]}
    drawn_for_audit = {name for name in got - want if name.startswith("prop_")}
    assert drawn_for_audit <= queued, f"propagated masks nobody queued: {drawn_for_audit - queued}"
    proposals = got - want - drawn_for_audit
    assert proposals <= amber, f"masks for labels that are neither accepted nor amber: {proposals - amber}"
    marked = {f"{r['kf']}_{r['label']}.png": r["status"] for r in read_jsonl(j / "masks.jsonl")}
    assert all(marked.get(name) == "AMBER" for name in proposals), {n: marked.get(n) for n in proposals}
    # And a proposal stays out of the dataset.
    coco = read_json(j / "dataset_coco.json")
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    frames = {i["id"]: i.get("frame_index") for i in coco["images"]}
    amber_at = {(k["frame_index"], r["label"]) for r in decisions if r["status"] == "AMBER"
                for k in meta["keyframes"] if k["kf"] == r["kf"]}
    leaked = [a["id"] for a in coco["annotations"] if (frames.get(a["image_id"]), cats[a["category_id"]]) in amber_at]
    assert not leaked, f"amber proposals reached the dataset: {leaked[:5]}"
    for p in (j / "masks").glob("*.png"):
        m = np.array(Image.open(p))
        assert m.shape == (meta["height"], meta["width"])
        assert m.any(), f"{p.name} is empty"


def test_tracks_cover_the_video_and_flag_propagation(mock_job):
    j = mock_job["job"]
    meta = _meta(mock_job)
    rows = read_jsonl(j / "tracks.jsonl")
    assert rows
    kf_frames = {k["frame_index"] for k in meta["keyframes"]}
    for r in rows:
        assert {"frame_index", "label", "box", "propagated"} <= set(r)
        assert 0 <= r["frame_index"] < meta["n_frames"]
        assert r["propagated"] == (r["frame_index"] not in kf_frames), r
    assert {r["frame_index"] for r in rows} - kf_frames, "no propagated frames"
    for p in (j / "tracks").glob("*.png"):
        fi = int(p.name.split("_")[0])
        assert fi not in kf_frames, f"{p.name}: keyframes must be the AI's own masks, not propagated"


def test_annotated_video_at_source_frame_rate(mock_job, test_video):
    a = mock_job["job"] / "annotated.mp4"
    assert count_frames(a) == count_frames(test_video["path"])
    assert video_fps(a) == pytest.approx(video_fps(test_video["path"]), rel=0.001)


def test_annotated_video_is_not_just_the_source(mock_job):
    """The render may upscale to its house size, so compare at the source's size."""
    import cv2

    meta = _meta(mock_job)
    j = mock_job["job"]
    a, s = cv2.VideoCapture(str(j / "annotated.mp4")), cv2.VideoCapture(str(j / meta.get("source", "source.mp4")))
    ok1, f1 = a.read()
    ok2, f2 = s.read()
    assert ok1 and ok2
    f1 = cv2.resize(f1, (f2.shape[1], f2.shape[0]))
    assert np.abs(f1.astype(int) - f2).mean() > 0.5, "overlay drew nothing"


def test_narration_cites_real_keyframes(mock_job):
    n = read_json(mock_job["job"] / "narration.json")
    kfs = {k["kf"] for k in _meta(mock_job)["keyframes"]}
    assert n["segments"]
    for s in n["segments"]:
        assert {"t0", "t1", "phase", "text", "evidence"} <= set(s)
        assert s["evidence"], s
        assert set(s["evidence"]) <= kfs


def test_coco_is_valid_and_excludes_ambers(mock_job):
    j = mock_job["job"]
    meta = _meta(mock_job)
    coco = read_json(j / "dataset_coco.json")
    assert {"images", "annotations", "categories"} <= set(coco)
    images = {i["id"]: i for i in coco["images"]}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    assert set(cats.values()) <= set(LABELS)
    assert len({a["id"] for a in coco["annotations"]}) == len(coco["annotations"])
    for a in coco["annotations"]:
        assert a["image_id"] in images and a["category_id"] in cats
        x, y, w, h = a["bbox"]
        assert x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= meta["width"] + 1 and y + h <= meta["height"] + 1
        assert a["segmentation"] and all(len(p) >= 6 for p in a["segmentation"])
        at = a["attributes"]
        assert {"status", "source", "model", "prompt_hash"} <= set(at)
        assert at["status"] != "AMBER"
    keyframe_anns = [a for a in coco["annotations"] if not a["attributes"].get("propagated")]
    accepted = [r for r in read_jsonl(j / "decisions.jsonl") if r["status"] == "accept" and r["final_present"]]
    assert len(keyframe_anns) == len(accepted)


def test_no_release_before_final_pass(mock_job):
    assert not (mock_job["job"] / "released_coco.json").exists()


def test_every_label_traces_to_a_recorded_response(mock_job):
    """raw_ref resolves in the cache, and the row says what the stored reply said."""
    j = mock_job["job"]
    cache = cache_rows(mock_job["cache"])
    assert cache
    for r in read_jsonl(j / "detect.jsonl"):
        stored = cache[r["raw_ref"]]
        assert stored["raw"], "no raw response kept"
        assert stored["parsed"]["objects"][r["label"]]["present"] == r["present"]
        assert stored["parsed"]["objects"][r["label"]]["conf"] == r["conf"]
    keys = {r["raw_ref"] for r in read_jsonl(j / "detect.jsonl")}
    assert len(keys) == len(_meta(mock_job)["keyframes"]), "keyframes share a response"


def test_provenance_covers_every_exported_annotation(mock_job):
    """Each keyframe annotation has an accepted provenance row whose mask hash matches the file on disk; each
    propagated annotation sits inside a recorded run from accepted keyframes."""
    import hashlib

    j = mock_job["job"]
    prov = read_jsonl(j / "provenance.jsonl")
    coco = read_json(j / "dataset_coco.json")
    assert coco["annotations"]
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    kf_rows = {(r["kf"], r["label"]): r for r in prov if r["type"] == "keyframe_label"}
    runs = [r for r in prov if r["type"] == "propagated_run"]
    cache = cache_rows(mock_job["cache"])
    for a in coco["annotations"]:
        at, label, fi = a["attributes"], cats[a["category_id"]], frames[a["image_id"]]
        if at.get("propagated"):
            assert any(r["label"] == label and r["source_kfs"] == at["source_keyframes"] and r["first_frame"] <= fi <= r["last_frame"]
                       for r in runs), a["id"]
            for kf in at["source_keyframes"]:
                assert kf_rows[(kf, label)]["status"] == "accept", f"propagated from a keyframe that was not accepted: {kf} {label}"
        else:
            row = kf_rows[(at["keyframe"], label)]
            assert row["status"] == "accept" and row["final_present"] is True
            assert row["detect"]["raw_ref"] in cache
            assert hashlib.sha256((j / row["mask"]["path"]).read_bytes()).hexdigest() == row["mask"]["sha256"]


def test_scorer_refuses_a_mock_run(mock_job, test_video):
    args = ["--registry", str(test_video["registry"])] if test_video["registry"] else []
    p = run_eval(mock_job["job"], *args)
    assert p.returncode == 4, p.stderr
    assert not (mock_job["job"] / "eval_report.json").exists()


@pytest.mark.skipif(not HAVE_DATA, reason="CholecSeg8k not on this machine")
def test_scorer_finds_truth_for_remuxed_source_and_frames_align(mock_job, test_video):
    """The pipeline's own keyframes are the frames the map says they are. Plumbing, not a score."""
    p = run_eval(mock_job["job"], "--registry", str(test_video["registry"]), "--allow-mock")
    assert p.returncode == 0, p.stderr
    r = read_json(mock_job["job"] / "eval_report.json")
    assert r["mock_run"] is True
    assert r["alignment"]["ok"] and r["alignment"]["n_bad"] == 0
    final = r["presence_final"]
    # Every keyframe and label is accounted for: resolved, or amber and counted against coverage. None lost.
    assert final["n"] + final["n_amber"] == SECONDS * 6, final
    assert final["n_missing_rows"] == 0 and final["n_undecided"] == 0, final
    assert final["coverage"] == round(final["n"] / (SECONDS * 6), 4)
    assert r["presence_detect_raw"]["n"] == SECONDS * 6, "detect lost a row"
    (mock_job["job"] / "eval_report.json").unlink()


@pytest.mark.skipif(os.environ.get("SURGPROOF_E2E_REAL") != "1", reason="set SURGPROOF_E2E_REAL=1: real model calls, costs money")
@pytest.mark.skipif(not HAVE_DATA, reason="CholecSeg8k not on this machine")
def test_real_run_end_to_end(test_video):
    """The real thing: headless Claude on a held-out clip, scored afterwards. Job kept in jobs/e2e_real for inspection."""
    job = ROOT / "jobs" / "e2e_real"
    proc = run_cli(test_video["path"], job, {}, cwd=ROOT, timeout=3000)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    p = run_eval(job, "--registry", str(test_video["registry"]))
    assert p.returncode == 0, p.stderr
    r = read_json(job / "eval_report.json")
    assert r["mock_run"] is False
    assert r["alignment"]["ok"]
    assert set(r["models"]) == {"claude-sonnet-5-5", "claude-haiku-5-5"}
    assert r["presence_final"]["n"] > 0
    print(p.stdout)
