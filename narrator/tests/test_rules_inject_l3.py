"""Injection harness reproducibility + L3 temporal layer."""
import pandas as pd

from narrator.check import l3_temporal as L3
from narrator.check.base import Ctx
from narrator.eval import inject as INJ


def gold():
    return [{"claim_id": f"g{i}", "clip_id": "video12_15750", "t_start": 15750 + 8 * i, "t_end": 15758 + 8 * i,
             "type": "presence", "modality": "observed", "entities": {"instrument": "grasper" if i % 2 else None,
                                                                     "target": None if i % 2 else "liver"},
             "polarity": "asserted", "evidence_refs": ["ev:video12_15750:0"], "kb_refs": [], "confidence": 0.9,
             "text": "", "mode": "grounded", "source": "gold", "statement": ""} for i in range(6)]


def test_inject_reproducible_and_complete():
    kw = dict(seed=3, per_type=2, absent_classes={"video12_15750": ["cystic_duct"]},
              clip_bounds={"video12_15750": (15750, 15829)})
    a, b = INJ.inject(gold(), **kw), INJ.inject(gold(), **kw)
    assert a == b
    assert set(INJ.INJECTION_TYPES) <= {c["injected_error"] for c in a}
    assert all(c["source"] == "injected" and "-inj-" in c["claim_id"] for c in a)


def test_inject_does_not_mutate_input():
    g = gold()
    INJ.inject(g, seed=0, per_type=1)
    assert g == gold()


def test_l3_contradiction_and_single_frame():
    base = gold()[:1][0]
    contra = dict(base, claim_id="x", type="absence")
    single = dict(base, claim_id="y", t_start=15790, t_end=15790, evidence_refs=["lab:video12_15750:15790:liver"])
    ctx = Ctx(labels=pd.DataFrame(), extra={"clip_bounds": {"video12_15750": (15750, 15829)}})
    r = L3.run([base, contra, single], ctx)
    assert r[0]["result"] == "pass"
    assert r[1]["result"] == "fail"
    assert r[2]["result"] == "fail"


def test_l3_out_of_clip_bounds():
    c = dict(gold()[0], t_start=99999, t_end=99999, evidence_refs=["ev:video12_15750:0"])
    ctx = Ctx(labels=pd.DataFrame(), extra={"clip_bounds": {"video12_15750": (15750, 15829)}})
    assert L3.run([c], ctx)[0]["result"] == "fail"
