"""Unit tests for visual verifier layers L5, L6, L7, and L5_CLIP.

All LLM calls are monkeypatched to ensure no network calls occur during testing.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from narrator.check.base import Ctx
from narrator.check.l5_blind import (
    build_question,
    find_frames,
    map_answer,
    run as run_l5,
    verb_to_ing,
)
from narrator.check.l6_perturb import run as run_l6
from narrator.check.l7_knowledge import (
    normalize_text,
    run as run_l7,
    verify_quote,
)
from narrator.check.l5_clip import run as run_l5_clip


# ============================================================================
# 1. Question Building & Verb Conjugation
# ============================================================================

def test_verb_to_ing():
    assert verb_to_ing("grasp") == "grasping"
    assert verb_to_ing("retract") == "retracting"
    assert verb_to_ing("dissect") == "dissecting"
    assert verb_to_ing("coagulate") == "coagulating"
    assert verb_to_ing("clip") == "clipping"
    assert verb_to_ing("cut") == "cutting"
    assert verb_to_ing("holding") == "holding"
    assert verb_to_ing("") == "operating on"


def test_build_question_presence_absence():
    c_pres = {
        "type": "presence",
        "modality": "observed",
        "entities": {"target": "gallbladder"},
        "text": "The surgeon clearly sees the large distended gallbladder.",
    }
    q_pres, ent = build_question(c_pres)
    assert q_pres == "Is there a gallbladder visible in this image?"
    assert ent == "gallbladder"
    # Hard rule: narration text must NEVER be in question
    assert "clearly sees" not in q_pres
    assert "large distended" not in q_pres

    c_abs = {
        "type": "absence",
        "modality": "observed",
        "entities": {"target": "cystic_duct"},
        "text": "No cystic duct is visible at this stage.",
    }
    q_abs, ent = build_question(c_abs)
    assert q_abs == "Is there a cystic duct visible in this image?"
    assert ent == "cystic_duct"


def test_build_question_action_finding():
    c_action = {
        "type": "action",
        "modality": "observed",
        "entities": {"instrument": "grasper", "verb": "retract", "target": "gallbladder"},
        "text": "A grasper is firmly retracting the gallbladder neck.",
    }
    q_act, ent = build_question(c_action)
    assert q_act == "Is a grasper retracting the gallbladder in this image?"
    assert ent == "grasper"
    assert "firmly" not in q_act

    c_finding = {
        "type": "finding",
        "modality": "observed",
        "entities": {"finding": "blood"},
        "text": "There appears to be minor blood oozing from the liver bed.",
    }
    q_find, ent = build_question(c_finding)
    assert q_find == "Is there blood visible in this image?"
    assert ent == "blood"


def test_build_question_invalid_or_nonvisual():
    c_empty = {"type": "presence", "modality": "observed", "entities": {}}
    q, ent = build_question(c_empty)
    assert q is None
    assert ent is None


def test_find_frames(tmp_path):
    with patch("narrator.data.frame_path") as mock_path:
        def fake_fp(clip_id, f):
            p = tmp_path / f"frame_{f}.png"
            if f in (80, 88):
                p.touch()
            return p
        mock_path.side_effect = fake_fp
        frames = find_frames("clip_1", 80, 90)
        assert 80 in frames
        assert 88 in frames
        assert len(frames) == 2


# ============================================================================
# 2. Answer Mapping & Absence Inversion
# ============================================================================

def test_map_answer_asserted_presence():
    # Yes -> pass
    res_yes = map_answer("YES ", 0.95, polarity="asserted", is_absence=False)
    assert res_yes["result"] == "pass"
    assert not res_yes["hedge"]

    # No -> fail
    res_no = map_answer("no", 0.90, polarity="asserted", is_absence=False)
    assert res_no["result"] == "fail"

    # Unsure -> pass with hedge
    res_unsure = map_answer("unsure", 0.50, polarity="asserted", is_absence=False)
    assert res_unsure["result"] == "pass"
    assert res_unsure["hedge"] is True


def test_map_answer_absence_and_negated_inversion():
    # Inverted: No -> pass (absence confirmed)
    res_no = map_answer("no", 0.95, polarity="asserted", is_absence=True)
    assert res_no["result"] == "pass"
    assert not res_no["hedge"]

    # Inverted: Yes -> fail (entity observed when claimed absent)
    res_yes = map_answer("yes", 0.90, polarity="asserted", is_absence=True)
    assert res_yes["result"] == "fail"

    # Negated polarity without type=absence
    res_neg_no = map_answer("no", 0.95, polarity="negated", is_absence=False)
    assert res_neg_no["result"] == "pass"
    res_neg_yes = map_answer("yes", 0.95, polarity="negated", is_absence=False)
    assert res_neg_yes["result"] == "fail"


def test_map_answer_hedged_polarity():
    res_hedged = map_answer("yes", 0.85, polarity="hedged", is_absence=False)
    assert res_hedged["result"] == "pass"
    assert res_hedged["hedge"] is True


# ============================================================================
# 3. Full L5 Run with Monkeypatched LLM
# ============================================================================

def test_l5_run_modalities_and_crops(tmp_path):
    clip_dir = tmp_path / "CholecSeg8k" / "video01" / "video01_00080"
    clip_dir.mkdir(parents=True, exist_ok=True)
    fake_frame = clip_dir / "frame_80_endo.png"
    fake_frame.write_bytes(b"dummy_png_bytes")

    claims = [
        # 0. Observed presence (pass)
        {
            "claim_id": "c1",
            "clip_id": "video01_00080",
            "t_start": 80,
            "t_end": 80,
            "type": "presence",
            "modality": "observed",
            "polarity": "asserted",
            "entities": {"target": "gallbladder"},
        },
        # 1. Non-visual modality (abstain)
        {
            "claim_id": "c2",
            "clip_id": "video01_00080",
            "t_start": 80,
            "t_end": 80,
            "type": "knowledge",
            "modality": "knowledge",
            "polarity": "asserted",
            "entities": {},
        },
        # 2. Non-visual type (abstain)
        {
            "claim_id": "c3",
            "clip_id": "video01_00080",
            "t_start": 80,
            "t_end": 80,
            "type": "phase",
            "modality": "observed",
            "polarity": "asserted",
            "entities": {"phase": "preparation"},
        },
    ]

    labels_df = pd.DataFrame([
        {
            "video_id": "video01",
            "clip_id": "video01_00080",
            "frame_idx": 80,
            "label": "gallbladder",
            "present": True,
            "status": "accepted",
            "boxes": json.dumps([[50, 50, 200, 200]]),
        }
    ])
    ctx = Ctx(labels=labels_df)

    async def fake_generate(prompt, **kwargs):
        if "gallbladder" in prompt:
            return {"answer": "yes", "confidence": 0.95}
        return {"answer": "no", "confidence": 0.90}

    with patch("narrator.data.frame_path", return_value=fake_frame), \
         patch("narrator.llm.jpeg", return_value=b"fake_jpeg"), \
         patch("narrator.llm.generate", side_effect=fake_generate):
        results = run_l5(claims, ctx)

    assert len(results) == 3
    assert results[0]["result"] == "pass"
    assert results[1]["result"] == "abstain"
    assert "non-visual modality" in results[1]["reason"]
    assert results[2]["result"] == "abstain"
    assert "non-visual claim type" in results[2]["reason"]
    # Check that ctx.extra['L5'] was populated
    assert ctx.extra.get("L5") == results


# ============================================================================
# 4. L6 Perturbation & Deterministic Sampling
# ============================================================================

def test_l6_deterministic_sampling():
    claims = [
        {
            "claim_id": f"c_{i}",
            "clip_id": f"video01_{i:05d}",
            "t_start": 100,
            "t_end": 100,
            "type": "presence",
            "modality": "observed",
            "entities": {"target": "liver"},
        }
        for i in range(20)
    ]
    # Simulate L5 passed all of them
    ctx = Ctx(extra={"L5": [{"result": "pass"} for _ in range(20)]})

    # Monkeypatch config to sample 5 items with seed 42
    with patch("narrator.cfg.cfg", return_value={"check": {"l6_sample": 5}, "seed": 42}), \
         patch("narrator.llm.jpeg", return_value=b"fake_jpeg"), \
         patch("narrator.data.frame_path", return_value=Path("/tmp/fake.png")), \
         patch.object(Path, "is_file", return_value=True), \
         patch("narrator.check.l6_perturb.get_shuffled_frame_path", return_value=Path("/tmp/shuf.png")):

        async def fake_gen(prompt, **kwargs):
            return {"answer": "no", "confidence": 0.9}

        with patch("narrator.llm.generate", side_effect=fake_gen):
            res1 = run_l6(claims, ctx)
            res2 = run_l6(claims, ctx)

        # Sampling must be 100% deterministic between identical runs
        passed_indices_1 = [i for i, r in enumerate(res1) if r["result"] != "abstain"]
        passed_indices_2 = [i for i, r in enumerate(res2) if r["result"] != "abstain"]
        assert passed_indices_1 == passed_indices_2
        assert len(passed_indices_1) == 5


def test_l6_prior_shortcuts_detection():
    claim = {
        "claim_id": "c_prior",
        "clip_id": "video01_00080",
        "t_start": 80,
        "t_end": 80,
        "type": "presence",
        "modality": "observed",
        "entities": {"target": "liver"},
    }
    ctx = Ctx(extra={"L5": [{"result": "pass"}]})

    # Case A: Black frame hallucinated "yes" -> FAIL
    async def fake_gen_priors(prompt, tag="", **kwargs):
        if tag == "l6_black":
            return {"answer": "yes", "confidence": 0.9}
        return {"answer": "no", "confidence": 0.9}

    with patch("narrator.cfg.cfg", return_value={"check": {"l6_sample": 1}, "seed": 0}), \
         patch("narrator.llm.jpeg", return_value=b"fake_jpeg"), \
         patch("narrator.data.frame_path", return_value=Path("/tmp/fake.png")), \
         patch.object(Path, "is_file", return_value=True), \
         patch("narrator.check.l6_perturb.get_shuffled_frame_path", return_value=Path("/tmp/shuf.png")), \
         patch("narrator.llm.generate", side_effect=fake_gen_priors):
        results = run_l6([claim], ctx)

    assert results[0]["result"] == "fail"
    assert "answer from priors, not pixels" in results[0]["reason"]

    # Case B: Both perturbations answered "no" -> PASS
    async def fake_gen_clean(prompt, **kwargs):
        return {"answer": "no", "confidence": 0.95}

    with patch("narrator.cfg.cfg", return_value={"check": {"l6_sample": 1}, "seed": 0}), \
         patch("narrator.llm.jpeg", return_value=b"fake_jpeg"), \
         patch("narrator.data.frame_path", return_value=Path("/tmp/fake.png")), \
         patch.object(Path, "is_file", return_value=True), \
         patch("narrator.check.l6_perturb.get_shuffled_frame_path", return_value=Path("/tmp/shuf.png")), \
         patch("narrator.llm.generate", side_effect=fake_gen_clean):
        results_clean = run_l6([claim], ctx)

    assert results_clean[0]["result"] == "pass"


# ============================================================================
# 5. L7 Knowledge Support & Quote Substring Verification
# ============================================================================

def test_l7_quote_substring_verification():
    text = "The critical view of safety requires clearing Calot's triangle and separating the lower gallbladder."
    # Exact substring
    assert verify_quote("critical view of safety", text) is True
    # Case difference
    assert verify_quote("CRITICAL VIEW OF SAFETY", text) is True
    # Whitespace difference
    assert verify_quote("critical  view\n  of   safety", text) is True
    # Normalised substring with quotes/punctuation
    assert verify_quote("clearing Calot's triangle", text) is True
    # Hallucinated word -> must be False
    assert verify_quote("critical view of complete safety", text) is False
    # Empty -> False
    assert verify_quote("", text) is False
    assert verify_quote("anything", "") is False


def test_l7_rules_and_verbatim_quote_enforcement():
    class MockKB:
        def __init__(self):
            self.rules = {"anatomy.r4u_line_safety": {"id": "anatomy.r4u_line_safety"}}

        def retrieve(self, query, k=4):
            return [{
                "passage_id": "pass_1",
                "text": "Dissection staying above the R4U line avoids major vasculobiliary injury.",
            }]

    ctx = Ctx(kb=MockKB())

    # Case A: Unknown rule id in kb_refs -> FAIL immediately
    c_bad_rule = {
        "claim_id": "c_bad_rule",
        "modality": "knowledge",
        "statement": "Stay above R4U line.",
        "kb_refs": ["unknown_rule_xyz"],
    }
    res_bad = run_l7([c_bad_rule], ctx)
    assert res_bad[0]["result"] == "fail"
    assert "not found in kb.rules" in res_bad[0]["reason"]

    # Case B: Supported with verbatim quote -> PASS
    c_valid = {
        "claim_id": "c_valid",
        "modality": "knowledge",
        "statement": "Dissection above the R4U line avoids major injury.",
        "kb_refs": ["anatomy.r4u_line_safety"],
    }

    async def fake_judge_supported(prompt, **kwargs):
        return {
            "verdict": "supported",
            "quote": "Dissection staying above the R4U line avoids major vasculobiliary injury.",
            "passage_id": "pass_1",
        }

    with patch("narrator.llm.generate", side_effect=fake_judge_supported):
        res_supp = run_l7([c_valid], ctx)

    assert res_supp[0]["result"] == "pass"
    assert "supported by pass_1" in res_supp[0]["reason"]

    # Case C: Model said supported but hallucinated quote -> treated as not_found -> FAIL
    async def fake_judge_hallucinated_quote(prompt, **kwargs):
        return {
            "verdict": "supported",
            "quote": "The surgeon MUST always stay above R4U according to 2024 guidelines.",
            "passage_id": "pass_1",
        }

    with patch("narrator.llm.generate", side_effect=fake_judge_hallucinated_quote):
        res_hallu = run_l7([c_valid], ctx)

    assert res_hallu[0]["result"] == "fail"
    assert "unsupported by KB" in res_hallu[0]["reason"]

    # Case D: Contradicted with verbatim quote -> FAIL
    async def fake_judge_contradicted(prompt, **kwargs):
        return {
            "verdict": "contradicted",
            "quote": "Dissection staying above the R4U line avoids major vasculobiliary injury.",
            "passage_id": "pass_1",
        }

    with patch("narrator.llm.generate", side_effect=fake_judge_contradicted):
        res_contra = run_l7([c_valid], ctx)

    assert res_contra[0]["result"] == "fail"
    assert "contradicted by pass_1" in res_contra[0]["reason"]

    # Case E: Empty retrieval -> abstain
    class EmptyKB:
        def __init__(self):
            self.rules = {}
        def retrieve(self, query, k=4):
            return []

    ctx_empty = Ctx(kb=EmptyKB())
    c_empty_ret = {"modality": "knowledge", "statement": "Some query"}
    res_empty = run_l7([c_empty_ret], ctx_empty)
    assert res_empty[0]["result"] == "abstain"
    assert "no passages retrieved" in res_empty[0]["reason"]


# ============================================================================
# 6. L5 CLIP Layer (Graceful Abstain)
# ============================================================================

def test_l5_clip_graceful_abstain():
    claim = {"type": "presence", "modality": "observed"}
    ctx = Ctx()
    results = run_l5_clip([claim], ctx)
    assert len(results) == 1
    assert results[0]["result"] == "abstain"
