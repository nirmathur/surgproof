"""Interface every checker layer implements.

Each layer module (check/l1_schema.py ... check/l7_knowledge.py, check/proportionality.py) exposes:

    NAME = "L2"
    def run(claims: list[dict], ctx: Ctx) -> list[dict]

returning one result per claim, aligned with `claims`:
    {"result": "pass" | "fail" | "abstain", "reason": str, "hedge": bool}
- "abstain" = this layer cannot judge this claim (wrong type, no evidence, not scorable).
- "hedge": True on a pass means "supported but uncertain": the verdict becomes hedge.
A layer must never raise for one bad claim: catch, and return abstain with the error as reason.

A claim dict is narrator.schema.Claim.model_dump(): keys claim_id, clip_id, t_start, t_end, type,
modality, entities{instrument, verb, target, finding, condition, grade, phase, cvs_criterion, decision},
polarity, evidence_refs, kb_refs, confidence, text, mode, source, statement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class Ctx:
    labels: pd.DataFrame | None = None       # labeller table (real or mock), see data.build_mock_labels
    flags: pd.DataFrame | None = None        # tracker flags.parquet
    events: dict = field(default_factory=dict)   # clip_id -> list[event dict] (events.py)
    kb: Any = None                           # narrator.kb loader object (rules + retrieve)
    clip_phase: dict = field(default_factory=dict)  # clip_id -> phase hint, if known
    extra: dict = field(default_factory=dict)


def res(result: str, reason: str = "", hedge: bool = False) -> dict:
    return {"result": result, "reason": reason, "hedge": hedge}


def safe_map(fn, claims, ctx) -> list[dict]:
    out = []
    for c in claims:
        try:
            out.append(fn(c, ctx))
        except Exception as e:  # noqa: BLE001
            out.append(res("abstain", f"error: {str(e)[:120]}"))
    return out
