"""Unit tests for L2 domain rules checker layer."""
from __future__ import annotations

import pytest

from narrator.check.base import Ctx
from narrator.check.l2_rules import run, NAME


def test_l2_name():
    assert NAME == "L2"


def test_l2_valid_instrument_verb_and_target():
    ctx = Ctx()
    # Pass: grasper + grasp + gallbladder
    good_action = {
        "claim_id": "act1",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "action",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "grasper", "verb": "grasp", "target": "gallbladder"},
    }
    r = run([good_action], ctx)[0]
    assert r["result"] == "pass"

    # Fail: clipper cannot coagulate
    bad_iv = {
        "claim_id": "act2",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "action",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "clipper", "verb": "coagulate", "target": "gallbladder"},
    }
    r = run([bad_iv], ctx)[0]
    assert r["result"] == "fail"
    assert "cannot perform" in r["reason"]

    # Fail: cannot clip liver
    bad_vt = {
        "claim_id": "act3",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "action",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "clipper", "verb": "clip", "target": "liver"},
    }
    r = run([bad_vt], ctx)[0]
    assert r["result"] == "fail"
    assert "cannot target" in r["reason"]


def test_l2_phase_transition_rules():
    ctx = Ctx()
    # Valid forward sequence
    p1 = {
        "claim_id": "p1",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 30,
        "type": "phase",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"phase": "preparation"},
    }
    p2 = {
        "claim_id": "p2",
        "clip_id": "video01_0001",
        "t_start": 40,
        "t_end": 60,
        "type": "phase",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"phase": "calot_triangle_dissection"},
    }
    res = run([p1, p2], ctx)
    assert res[0]["result"] == "pass"
    assert res[1]["result"] == "pass"

    # Invalid sequence: jumping back from gallbladder_packaging to calot_triangle_dissection
    p_pack = {
        "claim_id": "p_pack",
        "clip_id": "video01_0002",
        "t_start": 100,
        "t_end": 120,
        "type": "phase",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"phase": "gallbladder_packaging"},
    }
    p_jump_back = {
        "claim_id": "p_jump",
        "clip_id": "video01_0002",
        "t_start": 130,
        "t_end": 150,
        "type": "phase",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"phase": "calot_triangle_dissection"},
    }
    res = run([p_pack, p_jump_back], ctx)
    assert res[0]["result"] == "pass"
    assert res[1]["result"] == "fail"
    assert "Invalid phase transition" in res[1]["reason"]


def test_l2_tool_fits_phase():
    # Pass: clipper in clipping_cutting phase
    ctx_known = Ctx(clip_phase={"video01_0001": "clipping_cutting"})
    claim_clipper = {
        "claim_id": "c_clip",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "presence",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "clipper"},
    }
    res = run([claim_clipper], ctx_known)[0]
    assert res["result"] == "pass"

    # Fail: clipper in calot_triangle_dissection
    ctx_calot = Ctx(clip_phase={"video01_0001": "calot_triangle_dissection"})
    res = run([claim_clipper], ctx_calot)[0]
    assert res["result"] == "fail"
    assert "does not fit phase" in res["reason"]

    # Abstain: phase unknown and no other rules apply to simple instrument presence
    ctx_unknown = Ctx(clip_phase={})
    res = run([claim_clipper], ctx_unknown)[0]
    assert res["result"] == "abstain"


def test_l2_cvs_achieved_after_clip_action_fails():
    ctx = Ctx()
    # Action claim: clip
    clip_action = {
        "claim_id": "act_clip",
        "clip_id": "video01_0001",
        "t_start": 50,
        "t_end": 60,
        "type": "action",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"},
    }
    # Later CVS achieved claim
    cvs_claim = {
        "claim_id": "cvs_achieved",
        "clip_id": "video01_0001",
        "t_start": 80,
        "t_end": 90,
        "type": "cvs_criterion",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"cvs_criterion": "cvs_triangle_cleared"},
    }
    res = run([clip_action, cvs_claim], ctx)
    assert res[1]["result"] == "fail"
    assert "after earlier" in res[1]["reason"]

    # Prior CVS achieved claim passes
    cvs_prior = {
        "claim_id": "cvs_prior",
        "clip_id": "video01_0001",
        "t_start": 30,
        "t_end": 40,
        "type": "cvs_criterion",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"cvs_criterion": "cvs_triangle_cleared"},
    }
    res2 = run([cvs_prior, clip_action], ctx)
    assert res2[0]["result"] == "pass"


def test_l2_grade_claims():
    ctx = Ctx()
    finding_claim = {
        "claim_id": "f_adh",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "finding",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"finding": "adhesions_minor"},
    }
    # Parkland 2 requires adhesions_minor
    parkland_2 = {
        "claim_id": "p2",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 30,
        "type": "grade",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"grade": "parkland_2"},
    }
    # Parkland 4 requires adhesions_dense
    parkland_4 = {
        "claim_id": "p4",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 30,
        "type": "grade",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"grade": "parkland_4"},
    }
    # Tokyo grade should fail immediately
    tokyo_claim = {
        "claim_id": "tokyo",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 30,
        "type": "grade",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"grade": "tokyo_2"},
    }

    res = run([finding_claim, parkland_2, parkland_4, tokyo_claim], ctx)
    assert res[1]["result"] == "pass"
    assert res[2]["result"] == "fail"
    assert "requires findings" in res[2]["reason"]
    assert res[3]["result"] == "fail"
    assert "Tokyo severity grades" in res[3]["reason"]


def test_l2_condition_claims():
    ctx = Ctx()
    # 1. Non-inferable conditions fail immediately
    choledocho = {
        "claim_id": "c_cho",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "condition",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"condition": "choledocholithiasis"},
    }
    assert run([choledocho], ctx)[0]["result"] == "fail"

    # 2. Visible condition acute_cholecystitis passes with supporting finding (hyperaemia)
    finding_hyp = {
        "claim_id": "f_hyp",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "finding",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"finding": "hyperaemia"},
    }
    acute = {
        "claim_id": "c_acute",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "condition",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"condition": "acute_cholecystitis"},
    }
    res = run([finding_hyp, acute], ctx)
    assert res[1]["result"] == "pass"

    # 3. Acute cholecystitis fails if no supporting findings
    assert run([acute], ctx)[0]["result"] == "fail"

    # 4. Chronic cholecystitis fails if against finding (distension) is present
    finding_adh = {
        "claim_id": "f_adh_dense",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "finding",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"finding": "adhesions_dense"},
    }
    finding_dist = {
        "claim_id": "f_dist",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "finding",
        "modality": "observed",
        "polarity": "asserted",
        "entities": {"finding": "distension"},
    }
    chronic = {
        "claim_id": "c_chronic",
        "clip_id": "video01_0001",
        "t_start": 10,
        "t_end": 20,
        "type": "condition",
        "modality": "inference",
        "polarity": "asserted",
        "entities": {"condition": "chronic_cholecystitis"},
    }
    # Note: chronic_cholecystitis in conditions.yaml has visibility: needs_non_video_data,
    # so it fails on visibility
    r_chronic = run([finding_adh, finding_dist, chronic], ctx)[2]
    assert r_chronic["result"] == "fail"
