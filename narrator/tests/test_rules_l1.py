"""Unit tests for L1 schema and vocabulary claim checker."""
from __future__ import annotations

import pytest

from narrator.check.base import Ctx
from narrator.check.l1_schema import run, NAME


def test_l1_name():
    assert NAME == "L1"


def test_l1_valid_claims_pass():
    ctx = Ctx()
    claims = [
        {
            "claim_id": "c1",
            "clip_id": "video01_0001",
            "t_start": 10,
            "t_end": 20,
            "type": "presence",
            "modality": "observed",
            "polarity": "asserted",
            "entities": {"instrument": "grasper"},
        },
        {
            "claim_id": "c2",
            "clip_id": "video01_0001",
            "t_start": 10,
            "t_end": 20,
            "type": "action",
            "modality": "observed",
            "polarity": "asserted",
            "entities": {"instrument": "hook", "verb": "coagulate", "target": "gallbladder"},
        },
        {
            "claim_id": "c3",
            "clip_id": "video01_0001",
            "t_start": 0,
            "t_end": 100,
            "type": "phase",
            "modality": "inference",
            "polarity": "asserted",
            "entities": {"phase": "calot_triangle_dissection"},
        },
        {
            "claim_id": "c4",
            "clip_id": "video01_0001",
            "t_start": 50,
            "t_end": 60,
            "type": "cvs_criterion",
            "modality": "observed",
            "polarity": "asserted",
            "entities": {"cvs_criterion": "cvs_triangle_cleared"},
        },
        {
            "claim_id": "c5",
            "clip_id": "video01_0001",
            "t_start": 0,
            "t_end": 50,
            "type": "grade",
            "modality": "inference",
            "polarity": "asserted",
            "entities": {"grade": "parkland_2"},
        },
        {
            "claim_id": "c6",
            "clip_id": "video01_0001",
            "t_start": 0,
            "t_end": 50,
            "type": "condition",
            "modality": "inference",
            "polarity": "asserted",
            "entities": {"condition": "acute_cholecystitis"},
        },
        {
            "claim_id": "c7",
            "clip_id": "video01_0001",
            "t_start": 40,
            "t_end": 45,
            "type": "decision_point",
            "modality": "inference",
            "polarity": "asserted",
            "entities": {"decision": "cvs_pause"},
        },
        {
            "claim_id": "c8",
            "clip_id": "video01_0001",
            "t_start": 0,
            "t_end": 0,
            "type": "knowledge",
            "modality": "knowledge",
            "polarity": "asserted",
            "statement": "Anatomical rule",
            "kb_refs": ["rule_1"],
        },
    ]

    res_list = run(claims, ctx)
    assert len(res_list) == len(claims)
    for r in res_list:
        assert r["result"] == "pass", f"Expected pass, got {r}"


def test_l1_fail_invalid_entity_enum():
    ctx = Ctx()
    claim = {
        "claim_id": "bad_enum",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "presence",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "laser_scalpel"},
    }
    res = run([claim], ctx)[0]
    assert res["result"] == "fail"
    assert "laser_scalpel" in res["reason"]


def test_l1_fail_type_entity_inconsistency():
    ctx = Ctx()
    # Action without verb
    c_action = {
        "claim_id": "bad_action",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "action",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "hook"},
    }
    assert run([c_action], ctx)[0]["result"] == "fail"

    # Presence with no entity
    c_pres = {
        "claim_id": "bad_pres",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "presence",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {},
    }
    assert run([c_pres], ctx)[0]["result"] == "fail"

    # Knowledge without statement or kb_refs
    c_know = {
        "claim_id": "bad_know",
        "clip_id": "video01_0001",
        "t_start": 0,
        "t_end": 0,
        "type": "knowledge",
        "modality": "knowledge",
        "polarity": "asserted",
        "statement": "",
        "kb_refs": [],
    }
    assert run([c_know], ctx)[0]["result"] == "fail"


def test_l1_fail_modality_mismatch():
    ctx = Ctx()
    # Presence requires observed, not knowledge
    c_pres = {
        "claim_id": "bad_mod",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "presence",
        "modality": "knowledge",
        "polarity": "asserted",
        "entities": {"instrument": "grasper"},
    }
    res = run([c_pres], ctx)[0]
    assert res["result"] == "fail"
    assert "observed" in res["reason"]


def test_l1_fail_t_start_greater_than_t_end():
    ctx = Ctx()
    c = {
        "claim_id": "bad_time",
        "clip_id": "video01_0001",
        "t_start": 50,
        "t_end": 20,
        "type": "presence",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "grasper"},
    }
    res = run([c], ctx)[0]
    assert res["result"] == "fail"
    assert "t_start" in res["reason"]
