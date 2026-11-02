"""README "Run it" must stay true: every command, file and make target it names has to exist."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
RUN_IT = README[README.index("## Run it"):README.index("## Where it goes next")]


def test_no_em_dashes_in_run_it():
    assert "—" not in RUN_IT


def test_default_path_needs_no_api_key():
    head = RUN_IT[:RUN_IT.index("<details>")]
    assert "GEMINI" not in head and "API_KEY" not in head


@pytest.mark.parametrize("path", sorted(set(re.findall(r"(?:tools|tests|web)/[\w./-]+\.(?:py|txt)", RUN_IT))))
def test_named_files_exist(path):
    assert (ROOT / path).exists(), path


@pytest.mark.parametrize("target", sorted(set(re.findall(r"^make ([\w-]+)", RUN_IT, re.M))))
def test_make_targets_exist(target):
    if target in ("serve", "setup-full") and "serve:" not in (ROOT / "Makefile").read_text():
        pytest.skip("web lane Makefile not merged in this checkout")
    assert re.search(rf"^{re.escape(target)}:", (ROOT / "Makefile").read_text(), re.M), target


@pytest.mark.parametrize("module", ["surgproof.eval", "surgproof.run"])
def test_cli_modules_answer_help(module):
    if not (ROOT / (module.replace(".", "/") + ".py")).exists():
        pytest.skip(f"{module} not merged in this checkout")
    p = subprocess.run([sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_test_video_tool_answers_help():
    p = subprocess.run([sys.executable, "tools/make_test_video.py", "--help"], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0 and "--seconds" in p.stdout


def test_proof_numbers_are_attributed_to_the_pipeline_that_made_them():
    assert "earlier Gemini 4 Argon pipeline" in RUN_IT


# ---------- scope and claims, for the whole README ----------

@pytest.mark.parametrize("pattern,why", [
    (r"96\.[0-9]", "the headline from the earlier pipeline"),
    (r"93\.2", "the headline from the earlier pipeline"),
    (r"\bverified\b", "promises more than one AI pass and a sampled check"),
    (r"(?i)critical view|\bCVS\b", "not something this build assesses"),
    (r"(?i)operative.record", "not an operative record"),
    (r"(?i)\blive\b", "no live claims"),
    (r"(?i)any (surgery|procedure|surgical)", "one procedure only"),
    (r"0\.0\.0\.0", "localhost only"),
    (r"\u2014", "no em dashes"),
    (r"(?i)launch film|integrity%20checker|takes the person out", "marketing from the earlier pipeline"),
])
def test_readme_makes_no_retired_claim(pattern, why):
    hits = re.findall(pattern, README)
    assert not hits, f"{why}: {hits[:3]}"


@pytest.mark.parametrize("phrase", [
    "No real patient video", "public or simulator footage only", "Laparoscopic cholecystectomy only",
    "127.0.0.1", "not a medical device", "third-party model",
])
def test_readme_states_the_scope(phrase):
    assert phrase.lower() in README[:README.index("## How it works")].lower() or phrase in README, phrase


def _results():
    return json.loads((ROOT / "docs" / "e2e_status" / "qa_results.json").read_text())["groups"]


@pytest.mark.parametrize("group,row", [("heldout_never_scored_3_videos", "Videos never scored before"),
                                       ("heldout_all_unspent_6_videos", "All unused clips")])
def test_results_table_matches_the_results_file(group, row):
    """No number typed by hand: each cell has to be what the scorer wrote."""
    g = _results()[group]
    line = next(x for x in README.splitlines() if x.startswith(f"| {row}"))
    final = g["presence_final"]
    want = [f"{final['video_mean_accuracy']:.3f}", f"{final['ci95_over_videos'][0]:.3f} to {final['ci95_over_videos'][1]:.3f}",
            f"{final['coverage']:.3f}", f"| {final['pooled']['n']} |", f"| {final['n_amber']} |",
            f"{g['same_labels']['detect']['pooled']['accuracy']:.3f}"]
    for cell in want:
        assert cell in line, (cell, line)


def test_results_say_the_uncomfortable_parts():
    g = _results()
    train = g["rare_labels_train_video01"]["presence_detect_raw"]["per_label"]["cystic_duct"]["pooled"]
    assert f"{train['tp']} of {train['tp'] + train['fn']} found by detect" in README
    routes = g["heldout_all_unspent_6_videos"]["by_route"]
    assert f"{routes['amber_said_present']['tp']} of {routes['amber_said_present']['n']}" in README
    assert "Phase accuracy is not measured" in README
    iou = g["heldout_all_unspent_6_videos"]["iou_with_errors"]
    assert f"{iou['video_mean']:.3f} over 6 videos" in README
    same = g["heldout_all_unspent_6_videos"]["same_labels"]
    if same["final"]["pooled"]["accuracy"] <= same["detect"]["pooled"]["accuracy"]:
        assert "did not beat one pass" in README


def test_scored_results_never_use_a_spent_clip():
    from surgproof import eval as ev

    for name, g in _results().items():
        assert not set(g["clips"]) & ev.SPENT_CLIPS, name
    assert _results()["heldout_never_scored_3_videos"]["splits"] == ["test"]
    assert _results()["heldout_all_unspent_6_videos"]["splits"] == ["test"]
    assert all("mock" not in m for g in _results().values() for m in g["models"])


def test_the_hardest_excluded_clip_is_disclosed():
    """Leaving spent clips out is right for a clean test, and it flatters the table, so the README has to say so."""
    x = json.loads((ROOT / "docs" / "e2e_status" / "qa_results.json").read_text())["excluded_spent_clip"]
    assert f"{x['coverage'] * 100:.1f}% of labels" in README and f"{x['iou_with_errors']:.3f}" in README
