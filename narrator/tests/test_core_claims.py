"""Claim schema validation, evidence join keys, L4 evidence layer, verdict logic, rendering."""
import json

import pandas as pd
import pytest

from narrator.check import verdict
from narrator.check import l4_evidence as L4
from narrator.check.base import Ctx
from narrator.events import label_key
from narrator.render import sentence
from narrator.schema import Claim, normalise_entity, scored_class


def claim(**kw):
    base = dict(claim_id="g-c-0", clip_id="video12_15750", t_start=15750, t_end=15758, type="presence",
                modality="observed", entities={"instrument": "grasper"}, polarity="asserted")
    base.update(kw)
    return Claim.model_validate(base).model_dump(mode="json")


def labels():
    rows = [("video12", "video12_15750", 15750, "grasper", True, "accepted", "[[1,1,50,50]]"),
            ("video12", "video12_15750", 15758, "grasper", True, "accepted", "[[1,1,50,50]]"),
            ("video12", "video12_15750", 15750, "liver", True, "uncertain", "[[0,0,400,400]]"),
            ("video12", "video12_15750", 15750, "cystic_duct", False, "rejected", "[]")]
    return pd.DataFrame(rows, columns=["video_id", "clip_id", "frame_idx", "label", "present", "status", "boxes"])


def ctx():
    return Ctx(labels=labels(), flags=pd.DataFrame(), events={})


def test_schema_rejects_bad_type():
    with pytest.raises(Exception):
        claim(type="opinion")


def test_schema_roundtrip():
    c = claim()
    assert c["entities"]["instrument"] == "grasper" and c["mode"] == "grounded"


def test_synonyms_normalise():
    assert normalise_entity("instrument", "L-hook") == "hook"
    assert normalise_entity("target", "Calot's triangle") == "hepatocystic_triangle"
    assert scored_class({"entities": {"instrument": "hook"}}) == "l_hook"
    assert scored_class({"entities": {"target": "cystic_artery"}}) is None


def test_label_key_matches_join_columns():
    k = label_key("video12_15750", 15758, "grasper")
    _, clip, fr, lab = k.split(":")
    row = labels().query("clip_id == @clip and frame_idx == @fr and label == @lab".replace("@fr", fr))
    assert len(row) == 1


def test_l4_presence_pass_hedge_fail():
    r = L4.run([claim(), claim(entities={"target": "liver"}), claim(entities={"target": "cystic_duct"})], ctx())
    assert [x["result"] for x in r] == ["pass", "pass", "fail"]
    assert r[1]["hedge"] is True


def test_l4_absence_and_bad_citation():
    r = L4.run([claim(type="absence", entities={"target": "cystic_duct"}),
                claim(type="absence", entities={"instrument": "grasper"}),
                claim(entities={"instrument": "grasper"}, evidence_refs=["lab:video12_15750:15750:liver"]),
                claim(entities={"target": "cystic_duct"}, evidence_refs=["lab:video12_15750:15750:cystic_duct"])],
               ctx())
    assert [x["result"] for x in r] == ["pass", "fail", "pass", "fail"]


def test_l4_abstains_on_unscorable():
    r = L4.run([claim(entities={"target": "cystic_artery"}), claim(type="phase", modality="inference",
                                                                  entities={"phase": "clipping_cutting"})], ctx())
    assert [x["result"] for x in r] == ["abstain", "abstain"]


def test_verdict_logic():
    ok = {"result": "pass", "reason": "", "hedge": False}
    hd = {"result": "pass", "reason": "", "hedge": True}
    fl = {"result": "fail", "reason": "x", "hedge": False}
    ab = {"result": "abstain", "reason": "", "hedge": False}
    assert verdict(claim(), {"L1": ok, "L4": ok})[0] == "speak"
    assert verdict(claim(), {"L1": ok, "L4": hd})[0] == "hedge"
    assert verdict(claim(), {"L1": ok, "L4": fl}) == ("drop", ["L4"])
    assert verdict(claim(), {"L1": ok, "L4": ab}) == ("drop", ["unsupported"])


def test_render_hedge_wording():
    c = claim(type="cvs_criterion", modality="inference", entities={"cvs_criterion": "cvs_two_structures"})
    c["verdict"] = "hedge"
    assert "Holding off on CVS" in sentence(c)
