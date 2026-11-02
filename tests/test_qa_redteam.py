"""Red team: try to catch the pipeline, server and UI fabricating. Assumes the other lanes are wrong until shown otherwise.

Static checks read the source. Dynamic checks run the real CLI against a fake `claude` binary that misbehaves on
command: if the model is dead, nothing may appear; if it says nothing is there, nothing may be drawn; if it lies in
narration, the lie may not reach a subtitle. Bugs found here go to docs/e2e_status/qa.md.
"""
from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.e2e_helpers.harness import (HAVE_FFMPEG, LABELS, ROOT, base_env, cache_rows, fake_env, read_json, read_jsonl,
                                       run_cli, synthetic_video)

SURGPROOF = ROOT / "surgproof"
LICENSED_GT_TOKENS = [
    r"watershed", r"_endo_mask", r"_color_mask", r"\bgt_\w+", r"index\.csv", r"CholecSeg8k",
    r"gt_registry", r"eval_report", r"\braw/", r"surgatlas", r"prepare_data", r"make_test_video",
]
# Same patterns lanes/integrity.py applies to display code. integrity.py itself does not scan surgproof/ or web/app/.
DISPLAY_PATTERNS = [
    (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
    (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
    (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
    (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
    (r'(accuracy|acc|precision|coverage|iou)\s*[=:]\s*0\.\d{2,}', "hard-coded metric"),
    (r'\b(96\.\d|99\.\d|93\.\d)\s*%', "headline number typed by hand"),
]


def pipeline_sources():
    return [p for p in SURGPROOF.glob("*.py") if p.name != "eval.py"]


def display_sources():
    out = [p for p in (SURGPROOF / "server.py", SURGPROOF / "render.py") if p.exists()]
    app = ROOT / "web" / "app"
    if app.is_dir():
        out += [p for p in app.rglob("*") if p.suffix in {".js", ".html", ".css", ".mjs", ".ts"} and "vendor" not in p.parts]
    return out


# ---------- static: ground truth ----------

def test_eval_is_the_only_ground_truth_reader():
    hits = []
    # The dataset's licence notice has to name the footage it covers. It is the only mention allowed, and only verbatim.
    licence_notice = "grants no rights to the footage. Footage from CholecSeg8k or Cholec80 is CC BY-NC-SA 4.0"
    for p in pipeline_sources():
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("#") or (p.name == "export.py" and licence_notice in line):
                continue
            for tok in LICENSED_GT_TOKENS:
                if re.search(tok, line):
                    hits.append(f"{p.name}:{i}: {tok}: {line.strip()[:100]}")
    assert not hits, "ground-truth access outside surgproof/eval.py:\n" + "\n".join(hits)


def test_nothing_imports_the_scorer():
    bad = []
    for p in pipeline_sources():
        for node in ast.walk(ast.parse(p.read_text())):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [f"{node.module or ''}.{a.name}" for a in node.names] + [node.module or ""]
            if any(m in ("surgproof.eval", ".eval", "eval", "surgproof.eval.") or m.endswith("surgproof.eval") for m in mods):
                bad.append(f"{p.name}: {mods}")
        if re.search(r"""import_module\(\s*["']surgproof\.eval|surgproof\.eval["']""", p.read_text()):
            bad.append(f"{p.name}: dynamic import of the scorer")
    assert not bad, bad


def test_pipeline_does_not_import_repo_root_modules_that_touch_truth():
    for p in pipeline_sources():
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any(n.split(".")[0] in {"config", "prepare_data", "accept", "tracker"} for n in names), f"{p.name}: {names}"


def test_web_app_never_mentions_ground_truth():
    app = ROOT / "web" / "app"
    if not app.is_dir():
        pytest.skip("web lane not merged")
    for p in app.rglob("*"):
        if p.suffix in {".js", ".html"} and "vendor" not in p.parts:
            t = p.read_text(errors="ignore")
            for tok in (r"watershed", r"eval_report", r"gt_registry", r"\bgt_\w+"):
                assert not re.search(tok, t), f"{p}: {tok}"


# ---------- static: hand-typed results ----------

def test_no_hand_typed_results_in_display_code():
    hits = []
    for p in display_sources():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            s = line.strip()
            if s.startswith(("#", "//", "*")):
                continue
            for pat, what in DISPLAY_PATTERNS:
                if re.search(pat, line):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {what}: {s[:110]}")
    assert not hits, "\n".join(hits)


def test_integrity_marker_not_used_in_new_code():
    """'integrity: allow' is reserved for human-click lines, and none of this code should need it unreviewed."""
    hits = []
    for p in list(SURGPROOF.glob("*.py")) + display_sources():
        for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
            if "integrity: allow" in line and "(human action)" not in line:
                hits.append(f"{p.name}:{i}")
    assert not hits, hits


def test_mock_provider_cannot_be_the_silent_default():
    src = (SURGPROOF / "vlm.py").read_text() if (SURGPROOF / "vlm.py").exists() else ""
    if not src:
        pytest.skip("core lane not merged")
    assert re.search(r'environ\.get\(\s*"SURGPROOF_VLM"\s*,\s*"claude_cli"\s*\)', src), "default provider must be claude_cli"


# ---------- dynamic: misbehaving model, real CLI, real claude_cli code path ----------

pipeline_ready = all((SURGPROOF / f"{m}.py").exists() for m in ("pipeline", "run", "vlm", "prompts", "masks", "propagate", "render", "export"))
dynamic = pytest.mark.skipif(not (pipeline_ready and HAVE_FFMPEG), reason="core and masks lanes not merged, or no ffmpeg")
CANARY_NAME = "SECRET_CANARY_video12_19500.mp4"


class Scenarios:
    """Each (mode, seconds) runs once per session, in its own folder with its own model cache."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.video = synthetic_video(tmp / "v" / CANARY_NAME, seconds=3)
        self.done = {}

    def run(self, mode, name=None):
        key = name or mode
        if key not in self.done:
            parent = self.tmp / key
            job = parent / "job"
            log = parent / "calls.jsonl"
            parent.mkdir(parents=True)
            proc = run_cli(self.video, job, fake_env(mode, log))
            self.done[key] = {"proc": proc, "job": job, "cache": parent / "cache.sqlite", "calls": read_jsonl(log) if log.exists() else [], "log": log}
        return self.done[key]


@pytest.fixture(scope="session")
def scen(tmp_path_factory):
    return Scenarios(tmp_path_factory.mktemp("redteam"))


def decisions(s):
    return read_jsonl(s["job"] / "decisions.jsonl")


def coco_anns(s):
    return read_json(s["job"] / "dataset_coco.json")["annotations"]


def assert_failed_with_no_dataset(s):
    """Nothing accepted fails the job by name. It keeps its decisions for inspection and ships no dataset."""
    assert s["proc"].returncode != 0, "a job with nothing accepted claimed success"
    assert read_json(s["job"] / "status.json").get("result") == "nothing_accepted"
    assert not (s["job"] / "dataset_coco.json").exists() or coco_anns(s) == []
    assert not (s["job"] / "released_coco.json").exists()


def prompt_of(call):
    a = call["argv"]
    return a[a.index("-p") + 1]


def model_of(call):
    a = call["argv"]
    return a[a.index("--model") + 1]


def body_of(call):
    return re.sub(r"^Use the Read tool to open: .*?\n\n", "", prompt_of(call), flags=re.S)


@dynamic
@pytest.mark.parametrize("mode", ["dead", "garbage", "wrong_schema"])
def test_broken_model_produces_nothing(scen, mode):
    """A model that returned no valid answer must not leave one label, mask, video frame or cached row behind."""
    s = scen.run(mode)
    assert s["proc"].returncode != 0, s["proc"].stdout[-800:]
    st = read_json(s["job"] / "status.json")
    assert st["error"], "failure not recorded in status.json"
    assert st["stage"] != "done"
    assert not any(r.get("status") == "accept" for r in decisions(s))
    for f in ("annotated.mp4", "dataset_coco.json", "provenance.jsonl", "released_coco.json"):
        assert not (s["job"] / f).exists(), f
    assert not list((s["job"] / "masks").glob("*.png")) if (s["job"] / "masks").exists() else True
    assert cache_rows(s["cache"]) == {}, "an invalid answer was cached"


@dynamic
def test_logged_out_fails_fast_with_a_useful_message(scen):
    s = scen.run("logged_out")
    assert s["proc"].returncode != 0
    assert re.search(r"log(ged)?\s?in", s["proc"].stderr + read_json(s["job"] / "status.json")["error"], re.I)
    assert len(s["calls"]) <= 6, "kept calling after an auth failure"


@dynamic
def test_nothing_seen_means_nothing_drawn(scen):
    """Proves the output comes from the model: when it says absent, no default or fallback annotation appears."""
    s = scen.run("all_absent")
    assert_failed_with_no_dataset(s)
    rows = decisions(s)
    assert rows and all(r["status"] == "reject" and r["final_present"] is False for r in rows)
    assert not list((s["job"] / "masks").glob("*.png"))
    assert not [r for r in read_jsonl(s["job"] / "tracks.jsonl") if r.get("box")]


@dynamic
def test_unsure_model_stays_amber_and_nothing_is_exported(scen):
    s = scen.run("unsure")
    assert_failed_with_no_dataset(s)
    rows = decisions(s)
    assert rows and all(r["status"] == "AMBER" for r in rows)
    assert all(r["final_present"] is None for r in rows), "an amber row claims an answer"
    assert all(len(r["rounds"]) == 2 for r in rows), "amber loop did not run its rounds"
    # Masks for ambers may exist so the reviewer can see the proposal, but they must be marked as such.
    mask_rows = read_jsonl(s["job"] / "masks.jsonl")
    assert all(r["status"] == "AMBER" for r in mask_rows), [r for r in mask_rows if r["status"] != "AMBER"][:2]


@dynamic
@pytest.mark.parametrize("mode,forbidden", [
    ("narrate_lies", "clipped"),
    ("narrate_fake_evidence", "kf_99999"),
    ("narrate_no_evidence", "The grasper is visible."),
])
def test_narration_lies_do_not_reach_subtitles(scen, mode, forbidden):
    s = scen.run(mode)
    assert s["proc"].returncode == 0, s["proc"].stderr[-800:]
    kfs = {k["kf"] for k in read_json(s["job"] / "meta.json")["keyframes"]}
    n = read_json(s["job"] / "narration.json")
    assert n["segments"]
    for seg in n["segments"]:
        assert forbidden not in seg["text"], seg
        assert forbidden not in json.dumps(seg["evidence"])
        assert set(seg["evidence"]) <= kfs
    vtt = s["job"] / "narration.vtt"
    if vtt.exists():
        assert forbidden not in vtt.read_text()


@dynamic
def test_degenerate_polygon_is_never_accepted(scen):
    s = scen.run("polygon_degenerate")
    rows = decisions(s)
    assert rows, "the job kept no decisions to inspect"
    for r in rows:
        if r["status"] == "accept":
            assert r.get("polygon") and len(r["polygon"]) >= 3, r
    if not any(r["status"] == "accept" for r in rows):
        assert_failed_with_no_dataset(s)


@dynamic
def test_absurd_coordinates_do_not_become_a_full_frame_mask(scen):
    """Polygon wildly outside the image (wrong coordinate space). Clamping it to the frame edge invents a mask."""
    s = scen.run("polygon_out_of_bounds")
    rows = decisions(s)
    assert rows, "the job kept no decisions to inspect"
    accepted = [r for r in rows if r["status"] == "accept"]
    assert not accepted, f"accepted {len(accepted)} labels from out-of-image polygons, e.g. {accepted[0]['polygon']}"
    assert_failed_with_no_dataset(s)


def _centroid_x(path):
    ys, xs = np.where(np.array(Image.open(path)) > 0)
    return float(xs.mean())


@dynamic
def test_masks_follow_what_the_model_said(scen):
    a, b = scen.run("agree_all"), scen.run("shifted_right")
    assert a["proc"].returncode == 0 and b["proc"].returncode == 0
    p = "masks/kf_00000_grasper.png"
    assert (a["job"] / p).exists() and (b["job"] / p).exists()
    assert _centroid_x(b["job"] / p) > _centroid_x(a["job"] / p) + 30


@dynamic
def test_second_pass_is_independent_and_uses_the_other_model(scen):
    s = scen.run("agree_all")
    n_kf = len(read_json(s["job"] / "meta.json")["keyframes"])
    detect = [c for c in s["calls"] if body_of(c).startswith("This is one frame")]
    verify = [c for c in s["calls"] if body_of(c).startswith("You are auditing")]
    assert len(detect) == len(verify) == n_kf, "not every keyframe was checked twice"
    assert {model_of(c) for c in detect} == {"claude-sonnet-5-5"}
    assert {model_of(c) for c in verify} == {"claude-haiku-5-5"}
    # one prompt for every frame means the verifier is told nothing about what detection found there
    assert len({body_of(c) for c in verify}) == 1
    assert body_of(verify[0]) != body_of(detect[0])


@dynamic
def test_amber_prompts_do_not_reveal_earlier_answers(scen):
    s = scen.run("normal")  # the mock's verify pass disputes cystic_duct, which reaches the amber loop
    ambers = [c for c in s["calls"] if "magnified crop" in body_of(c)[:80]]
    assert ambers, "amber loop never ran"
    for c in ambers:
        assert not re.search(r"\b(detect\w*|verif\w*|disagree\w*|first pass|second pass|previous)\b", body_of(c), re.I), body_of(c)[:400]


@dynamic
def test_each_call_is_sandboxed_from_the_project(scen):
    """The model sees neutral file names in an empty folder, no project files, no video name."""
    s = scen.run("agree_all")
    assert s["calls"]
    for c in s["calls"]:
        assert set(c["listdir"]) <= {"image_1.jpg", "image_1.png"}, c["listdir"]
        assert not str(c["cwd"]).startswith(str(ROOT))
        text = prompt_of(c)
        assert "CANARY" not in text and "video12" not in text, "source file name reached the model"
        assert str(s["job"]) not in text, "job path reached the model"
        assert "--setting-sources" in c["argv"], "user settings and CLAUDE.md reach the model"


@dynamic
def test_read_tool_is_scoped_to_the_call_folder(scen):
    """--allowedTools Read with no path lets the model open any file, including ground truth masks and the scorer."""
    s = scen.run("agree_all")
    a = s["calls"][0]["argv"]
    allowed = a[a.index("--allowedTools") + 1]
    assert allowed != "Read" and allowed.startswith("Read("), f"model may read anything on disk: --allowedTools {allowed}"


@dynamic
def test_row_values_match_the_recorded_response(scen):
    s = scen.run("agree_all")
    cache = cache_rows(s["cache"])
    rows = read_jsonl(s["job"] / "detect.jsonl")
    assert rows
    for r in rows:
        o = cache[r["raw_ref"]]["parsed"]["objects"][r["label"]]
        assert (r["present"], r["conf"]) == (o["present"], o["conf"])
    for r in read_jsonl(s["job"] / "provenance.jsonl"):
        ref = r.get("raw_ref")
        if ref:
            assert ref in cache, f"provenance points at a response that does not exist: {r}"


@dynamic
def test_clean_room_run_reads_no_ground_truth(tmp_path):
    """Run with only surgproof/ minus the scorer on the path, from an empty folder, with every Python file open
    audited. Passing means the pipeline neither needs nor opens ground truth, repo data or the scorer."""
    pkg = tmp_path / "pkg" / "surgproof"
    shutil.copytree(SURGPROOF, pkg, ignore=shutil.ignore_patterns("eval.py", "__pycache__"))
    audit = tmp_path / "audit.log"
    runner = tmp_path / "audited_run.py"
    runner.write_text(
        "import os, runpy, sys\n"
        f"log = open({str(audit)!r}, 'a')\n"
        "def hook(ev, args):\n"
        "    if ev == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):\n"
        "        log.write(os.fspath(args[0]) if not isinstance(args[0], bytes) else args[0].decode('utf8', 'replace')); log.write('\\n'); log.flush()\n"
        "    elif ev == 'subprocess.Popen':\n"
        "        log.write('EXEC ' + str(args[1])[:400] + '\\n'); log.flush()\n"
        "sys.addaudithook(hook)\n"
        "sys.argv = ['surgproof.run'] + sys.argv[1:]\n"
        "runpy.run_module('surgproof.run', run_name='__main__', alter_sys=True)\n"
    )
    video = synthetic_video(tmp_path / "v" / CANARY_NAME, seconds=2)
    work = tmp_path / "elsewhere"
    work.mkdir()
    import subprocess
    import sys

    env = base_env({**fake_env("agree_all"), "PYTHONPATH": str(tmp_path / "pkg")})
    p = subprocess.run([sys.executable, str(runner), str(video), "--out", str(work / "job")], cwd=work, env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]
    opened = audit.read_text()
    assert (work / "job" / "annotated.mp4").exists()
    for bad in (r"_watershed_mask", r"_color_mask", r"_endo_mask", r"index\.csv", r"raw/CholecSeg8k", r"gt_registry", r"surgproof/eval"):
        m = re.search(rf".*{bad}.*", opened)
        assert not m, f"pipeline opened {m.group(0)}"
    assert str(tmp_path / "pkg" / "surgproof" / "pipeline.py") in opened, "did not actually run the clean-room copy"
    assert str(ROOT / "surgproof" / "pipeline.py") not in opened


@dynamic
def test_rerun_into_the_same_folder_leaves_no_stale_results(tmp_path):
    """A second video in a used job folder must not inherit the first one's masks, dataset or human sign-off."""
    parent = tmp_path / "jobs"
    parent.mkdir()
    job = parent / "job"
    first = synthetic_video(tmp_path / "a.mp4", seconds=3)
    assert run_cli(first, job, fake_env("agree_all")).returncode == 0
    (job / "final_pass.jsonl").write_text('{"kf": "kf_00000", "label": "liver", "action": "approve"}\n')
    (job / "released_coco.json").write_text("{}")
    second = synthetic_video(tmp_path / "b.mp4", seconds=1, size="160x90")
    # The second video has nothing to accept, so it fails. Its failure must still not leave the first one's results.
    assert run_cli(second, job, fake_env("all_absent"), extra_args=None).returncode != 0
    assert read_json(job / "status.json").get("result") == "nothing_accepted"
    assert not (job / "dataset_coco.json").exists() or read_json(job / "dataset_coco.json")["annotations"] == []
    assert not (job / "released_coco.json").exists(), "a human release from the previous video survived"
    assert not (job / "final_pass.jsonl").exists(), "human verdicts from the previous video survived"
    assert not list((job / "masks").glob("*.png")), "masks from the previous video survived"
    assert not list((job / "tracks").glob("*.png")), "tracks from the previous video survived"


# ---------- narration: the check itself ----------

FACTS = {"phase": "CalotTriangleDissection", "t0": 0, "t1": 5, "phase_conf": 0.9, "phase_checks_agree": True, "keyframes": [
    {"kf": "kf_00000", "t": 0, "visible": ["gallbladder", "liver", "grasper"], "not_visible": ["blood", "l_hook"], "unsure": ["cystic_duct"]},
]}


@pytest.mark.skipif(not (SURGPROOF / "pipeline.py").exists(), reason="core lane not merged")
@pytest.mark.parametrize("text", [
    "A grasper is visible near the liver.",
    "No blood is visible.",
    "The L-hook is not in view.",
    "The cystic duct could not be confirmed.",
])
def test_narration_check_keeps_supported_sentences(text):
    from surgproof.pipeline import sentence_supported

    assert sentence_supported({"text": text, "evidence": ["kf_00000"]}, FACTS)


@pytest.mark.skipif(not (SURGPROOF / "pipeline.py").exists(), reason="core lane not merged")
@pytest.mark.parametrize("text", [
    "Blood is visible.",                                            # verified absent
    "The L-hook is dissecting the gallbladder.",                    # verified absent
    "The cystic duct is visible.",                                  # amber, stated as fact
    "The cystic artery has been clipped.",                          # no label for it
    "There is no doubt that blood is pouring from the liver.",      # negation word elsewhere in the sentence
    "The grasper is not far from active bleeding.",                 # same trick
    "The gallbladder has been perforated.",                         # a finding nobody verified
    "The grasper has torn the liver.",                              # an event nobody verified
    "Haemostasis has been achieved and the patient is stable.",     # no label named at all
    "The surgeon is working safely.",                               # no label named at all
    "The gallbladder is not absent, and blood is not absent.",      # double negative hides a claim about blood
    "No grasper is visible, but blood is visible.",                 # second clause lies, first clause is about something else
    "Blood is visible, not the L-hook.",                            # negation in one part must not excuse the other
    "The liver is not visible.",                                    # verified present, stated absent
    "The cystic duct is not visible.",                              # amber, stated as verified absent
    "Blood could not be confirmed.",                                # verified absent, reported as doubt
    "The grasper is visible in every frame of the operation.",      # beyond the cited evidence
    "Blood is seen, no L-hook.",                                    # a verified-absent label stated as seen, negation trailing
    "Blood and no L-hook are visible.",                             # negation attached to one of two labels
    "No L-hook, blood is visible.",                                 # negation leading, claim after the comma
    "There is not an L-hook, there is blood.",                      # two clauses, one comma
    "Blood is visible; no L-hook is visible.",                      # semicolon
    "Blood is visible (no L-hook).",                                # brackets
])
def test_narration_check_rejects_unverified_claims(text):
    from surgproof.pipeline import sentence_supported

    assert not sentence_supported({"text": text, "evidence": ["kf_00000"]}, FACTS), text


TWO_KF = {"phase": "CalotTriangleDissection", "t0": 0, "t1": 5, "phase_conf": 0.9, "phase_checks_agree": True, "keyframes": [
    {"kf": "kf_00000", "t": 0, "visible": ["grasper", "liver"], "not_visible": ["blood"], "unsure": []},
    {"kf": "kf_00001", "t": 1, "visible": ["liver", "blood"], "not_visible": ["grasper"], "unsure": []},
]}


@pytest.mark.skipif(not (SURGPROOF / "pipeline.py").exists(), reason="core lane not merged")
@pytest.mark.parametrize("text", [
    "The grasper is visible in all frames.",
    "The grasper is visible throughout this segment.",
    "The grasper is visible in every frame.",
    "Blood is not visible throughout this segment.",
    "There is no blood in all frames.",
])
def test_narration_cannot_generalise_past_the_keyframes_it_cites(text):
    """Each is true on kf_00000 and false on kf_00001. Citing only the good keyframe must not make it true."""
    from surgproof.pipeline import sentence_supported

    assert not sentence_supported({"text": text, "evidence": ["kf_00000"]}, TWO_KF), text


@pytest.mark.skipif(not (SURGPROOF / "pipeline.py").exists(), reason="core lane not merged")
def test_narration_may_generalise_when_it_cites_every_keyframe():
    from surgproof.pipeline import sentence_supported

    assert sentence_supported({"text": "The liver is visible throughout this segment.", "evidence": ["kf_00000", "kf_00001"]}, TWO_KF)


@pytest.mark.skipif(not (SURGPROOF / "pipeline.py").exists(), reason="core lane not merged")
@pytest.mark.parametrize("text,agree,conf", [
    ("The phase checks disagree.", True, 0.9),                          # they agreed
    ("Confidence is low.", True, 0.9),                                  # it was high
    ("The phase is uncertain.", True, 0.9),                             # it was not
    ("This segment is the Calot Triangle Dissection phase.", False, 0.9),   # checks disagree, stated flatly
    ("It is Calot Triangle Dissection.", True, 0.3),                    # low confidence, stated flatly
    ("Calot Triangle Dissection continues.", False, 0.3),
])
def test_statements_about_the_phase_are_checked_against_the_facts(text, agree, conf):
    from surgproof.pipeline import sentence_supported

    facts = {**TWO_KF, "phase_checks_agree": agree, "phase_conf": conf}
    assert not sentence_supported({"text": text, "evidence": ["kf_00000"]}, facts), text
