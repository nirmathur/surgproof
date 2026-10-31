"""Tests for decision node review and scoring against reference policies."""
from __future__ import annotations

from pathlib import Path
import yaml
import pytest

from narrator.review import review, load_decisions



def _negated_missing(claims):
    """Explicit 'criterion not met' claims for every CVS criterion not confirmed (silence is not failure)."""
    have = {c["entities"].get("cvs_criterion") for c in claims
            if c.get("type") == "cvs_criterion" and c.get("verdict") in ("speak", "hedge")}
    out = []
    for i, k in enumerate(sorted({"cvs_triangle_cleared", "cvs_lower_gb_off_plate", "cvs_two_structures"} - have)):
        out.append({"claim_id": f"neg{i}", "type": "cvs_criterion", "entities": {"cvs_criterion": k},
                    "verdict": "speak", "polarity": "negated", "t_start": 100, "t_end": 140})
    return out

def test_decisions_yaml_structure():
    """Verify decisions.yaml has all 8 required nodes and valid schema."""
    kb = load_decisions()
    assert "nodes" in kb, "decisions.yaml missing 'nodes' section"
    assert "rules" in kb, "decisions.yaml missing 'rules' section"

    nodes = kb["nodes"]
    assert len(nodes) == 8, f"Expected 8 decision nodes, got {len(nodes)}"

    expected_ids = [
        "difficulty_assessment",
        "exposure_retraction",
        "dissection_plane_r4u",
        "cvs_pause",
        "cvs_not_reached_action",
        "bailout_choice",
        "clip_and_divide",
        "gb_bed_haemostasis",
    ]
    actual_ids = [n["id"] for n in nodes]
    assert actual_ids == expected_ids

    for n in nodes:
        assert "id" in n
        assert "order" in n
        assert "guideline_action" in n
        assert "alternatives" in n and isinstance(n["alternatives"], list) and len(n["alternatives"]) >= 1
        assert "evidence_grade" in n
        assert "source_id" in n
        assert "quote" in n
        assert "quote_verified" in n and isinstance(n["quote_verified"], bool)
        assert "seed" in n
        assert "detect" in n
        assert "from" in n["detect"] and isinstance(n["detect"]["from"], list)
        assert "rule" in n["detect"]

        # Check quote length is strictly < 15 words
        words = n["quote"].strip().split()
        assert len(words) < 15, f"Quote for {n['id']} exceeds 14 words: '{n['quote']}' ({len(words)} words)"

    rules = kb["rules"]
    assert len(rules) == 8, f"Expected 8 rules, got {len(rules)}"
    for r in rules:
        assert r["id"].startswith("decision.")
        assert "statement" in r
        assert "source_id" in r
        assert "quote" in r
        assert len(r["quote"].strip().split()) < 15
        assert isinstance(r["quote_verified"], bool)
        if r["quote_verified"]:  # set only by kb/strict.py after a verbatim match in fetched text
            assert r.get("quote_status") == "verified"
        assert "evidence_grade" in r
        assert isinstance(r["seed"], bool)


def test_node_scoring_ok():
    """When all 3 CVS criteria are confirmed and clip is applied, cvs_pause is 'ok'."""
    claims = [
        {
            "claim_id": "c1",
            "type": "grade",
            "entities": {"grade": "parkland_2"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 0,
            "t_end": 40,
        },
        {
            "claim_id": "c2",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_triangle_cleared"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 100,
            "t_end": 140,
        },
        {
            "claim_id": "c3",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_lower_gb_off_plate"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 140,
            "t_end": 180,
        },
        {
            "claim_id": "c4",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_two_structures"},
            "verdict": "hedge",
            "polarity": "asserted",
            "t_start": 180,
            "t_end": 220,
        },
        {
            "claim_id": "c5",
            "type": "action",
            "entities": {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 250,
            "t_end": 290,
        },
    ]

    res = review(claims, video="test_ok")
    assert res["scored"] is False
    assert res["label"] == "deviation from reference policy"
    assert res["difficulty"]["grade"] == "parkland_2"

    node_map = {n["node_id"]: n for n in res["nodes"]}
    assert node_map["difficulty_assessment"]["verdict"] == "ok"
    assert node_map["cvs_pause"]["verdict"] == "ok"
    assert node_map["clip_and_divide"]["verdict"] == "ok"
    assert node_map["cvs_not_reached_action"]["verdict"] == "ok"


def test_node_scoring_inaccuracy():
    """When only 2 of 3 CVS criteria are confirmed and clip is applied, cvs_pause is 'inaccuracy'."""
    claims = [
        {
            "claim_id": "c1",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_triangle_cleared"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 100,
            "t_end": 140,
        },
        {
            "claim_id": "c2",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_two_structures"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 140,
            "t_end": 180,
        },
        # 3rd criterion missing or dropped
        {
            "claim_id": "c3",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_lower_gb_off_plate"},
            "verdict": "drop",  # dropped claims do NOT count
            "polarity": "asserted",
            "t_start": 180,
            "t_end": 200,
        },
        {
            "claim_id": "c4",
            "type": "action",
            "entities": {"verb": "clip"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 210,
            "t_end": 250,
        },
    ]

    claims = claims + _negated_missing(claims)
    res = review(claims, video="test_inaccuracy")
    node_map = {n["node_id"]: n for n in res["nodes"]}
    assert node_map["cvs_pause"]["verdict"] == "inaccuracy"
    assert node_map["clip_and_divide"]["verdict"] == "inaccuracy"


def test_node_scoring_inaccuracy_high_difficulty():
    """When Parkland >= 4, inaccuracy notes high difficulty."""
    claims = [
        {
            "claim_id": "c0",
            "type": "grade",
            "entities": {"grade": "parkland_4"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 0,
            "t_end": 20,
        },
        {
            "claim_id": "c1",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_triangle_cleared"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 100,
            "t_end": 140,
        },
        {
            "claim_id": "c2",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_two_structures"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 140,
            "t_end": 180,
        },
        {
            "claim_id": "c3",
            "type": "action",
            "entities": {"verb": "clip"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 210,
            "t_end": 250,
        },
    ]

    claims = claims + _negated_missing(claims)
    res = review(claims, video="test_high_diff")
    assert res["difficulty"]["grade"] == "parkland_4"
    node_map = {n["node_id"]: n for n in res["nodes"]}
    cvs_node = node_map["cvs_pause"]
    assert cvs_node["verdict"] == "inaccuracy"
    assert "high difficulty" in cvs_node.get("observed", "").lower() or cvs_node.get("note") == "high difficulty"


def test_node_scoring_blunder():
    """When <= 1 criterion confirmed prior to clipping, verdict is 'blunder'."""
    claims = [
        {
            "claim_id": "c1",
            "type": "cvs_criterion",
            "entities": {"cvs_criterion": "cvs_triangle_cleared"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 100,
            "t_end": 140,
        },
        {
            "claim_id": "c2",
            "type": "action",
            "entities": {"verb": "clip"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 150,
            "t_end": 190,
        },
    ]

    claims = claims + _negated_missing(claims)
    res = review(claims, video="test_blunder")
    node_map = {n["node_id"]: n for n in res["nodes"]}
    assert node_map["cvs_pause"]["verdict"] == "blunder"
    assert node_map["clip_and_divide"]["verdict"] == "blunder"


def test_node_scoring_missed_alternative():
    """When CVS is not reached by time limit without imaging/bailout, verdict is 'missed_alternative'."""
    claims = [
        {
            "claim_id": "c1",
            "type": "action",
            "entities": {"instrument": "hook", "verb": "dissect", "target": "cystic_pedicle"},
            "verdict": "speak",
            "polarity": "asserted",
            "t_start": 0,
            "t_end": 1500,  # exceeds default 1200s time limit
        },
    ]

    res = review(claims, video="test_missed", time_limit_s=1200.0)
    node_map = {n["node_id"]: n for n in res["nodes"]}
    assert node_map["cvs_not_reached_action"]["verdict"] == "missed_alternative"


def test_node_scoring_unknown():
    """When no claims exist, verdicts are 'unknown'."""
    res = review([], video="test_empty")
    assert res["difficulty"]["grade"] == "unknown"
    for n in res["nodes"]:
        assert n["verdict"] == "unknown"
        assert n["evidence_grade"] != ""
        assert n["source_quote"] != ""


def test_unassessed_cvs_is_unknown_not_blunder():
    """One criterion confirmed, the other two never mentioned: no safety flag on silence."""
    claims = [{"claim_id": "c1", "type": "cvs_criterion", "entities": {"cvs_criterion": "cvs_triangle_cleared"},
               "verdict": "speak", "polarity": "asserted", "t_start": 100, "t_end": 140},
              {"claim_id": "c2", "type": "action", "entities": {"verb": "clip"}, "verdict": "speak",
               "polarity": "asserted", "t_start": 150, "t_end": 190}]
    node_map = {n["node_id"]: n for n in review(claims, video="t")["nodes"]}
    assert node_map["cvs_pause"]["verdict"] == "unknown"
    assert node_map["clip_and_divide"]["verdict"] == "unknown"
