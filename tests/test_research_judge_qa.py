"""Tests for docs/JUDGE_QA.md ensuring all 12 judge questions, 2-3 sentence answers,
and cited numbers match committed out/*.json and docs/research/* artifacts."""
import json
import os
import re

ROOT = os.path.expanduser("~/surgatlas")
QA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs",
    "JUDGE_QA.md",
)


def test_judge_qa_exists_and_has_12_questions():
    assert os.path.exists(QA_PATH), f"Missing {QA_PATH}"
    text = open(QA_PATH, encoding="utf-8").read()
    assert "## Top 3 Risks for Judges" in text
    headings = re.findall(r"^### (\d+)\.\s+(.+)$", text, flags=re.M)
    assert len(headings) == 12, f"Expected 12 questions, found {len(headings)}"
    assert [int(n) for n, _ in headings] == list(range(1, 13))


def test_judge_qa_answers_are_2_to_3_sentences_and_cite_sources():
    text = open(QA_PATH, encoding="utf-8").read()
    blocks = re.split(r"^### \d+\..+$", text, flags=re.M)[1:]
    assert len(blocks) == 12
    for idx, block in enumerate(blocks, 1):
        block = block.split("\n---")[0].strip()
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        assert lines[0].startswith("**Q:"), f"Q{idx} missing **Q: line"
        answer = " ".join(lines[1:])
        # Count sentences in the answer paragraph (splitting on '. ' at sentence boundaries)
        # Normalise common abbreviations like 'et al.', 'vs.', 'e.g.' first
        cleaned = re.sub(r"\b(et al|vs|e\.g|i\.e|Sci|Data|Commun)\.", r"\1", answer)
        sentences = [s.strip() for s in re.split(r"\.\s+(?=[A-Z*`])", cleaned) if s.strip()]
        assert 2 <= len(sentences) <= 3, (
            f"Q{idx} expected 2-3 sentences, got {len(sentences)}: {sentences}"
        )
        # Must cite at least one out/ or narrator/ artifact and one docs/research/ source
        assert re.search(r"out/|lanes/integrity\.py", answer), f"Q{idx} missing out/ artifact citation"
        assert "docs/research/" in answer, f"Q{idx} missing docs/research/ citation"
        assert re.search(r"\bn\s*=\s*[\d,]+", answer), f"Q{idx} missing sample size n=..."


def test_judge_qa_numbers_match_json_artifacts():
    text = open(QA_PATH, encoding="utf-8").read()

    results = json.load(open(os.path.join(ROOT, "out", "results.json")))
    audit = json.load(open(os.path.join(ROOT, "out", "eval_audit.json")))
    loc = json.load(open(os.path.join(ROOT, "out", "localize_results.json")))
    masks = json.load(open(os.path.join(ROOT, "out", "mask_results.json")))
    student = json.load(open(os.path.join(ROOT, "out", "student.json")))
    vote = json.load(open(os.path.join(ROOT, "out", "vote_experiment.json")))

    # Verify exact key numbers cited in JUDGE_QA.md match JSON files
    assert f"{results['accepted_excl_liver']['accuracy']*100:.2f}%" in text  # 96.55%
    assert str(results["n_test_labels_excl_liver"]) in text or "1,400" in text
    assert f"{audit['verifier_test_evaluation']['core_3_balanced_labels']['accepted_acc']['point']*100:.2f}%" in text  # 94.72%
    assert f"{loc['boxes']['mean_iou']:.4f}" in text  # 0.7614
    assert f"{masks['per_round_improvement']['accepted_test_masks']['final_refined']['mean_iou']:.4f}" in text  # 0.6459
    assert f"{masks['per_round_improvement']['accepted_test_masks']['final_refined']['mean_dice']:.4f}" in text  # 0.7595
    assert f"{student['mean_acc_by_source']['verified']*100:.2f}%" in text  # 90.14%
    assert f"{vote['summary']['vote5']['mean_iou']:.4f}" in text  # 0.7130
