"""README "Run it" must stay true: every command, file and make target it names has to exist."""
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
