"""Claim schema (the atomic unit of narration) and the vocabulary it is checked against."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

from narrator import cfg as C

CLAIM_TYPES = ["presence", "absence", "action", "phase", "cvs_criterion", "finding", "condition",
               "grade", "decision_point", "knowledge"]
MODALITIES = ["observed", "knowledge", "inference"]
POLARITIES = ["asserted", "negated", "hedged"]
VERDICTS = ["speak", "hedge", "drop"]
LAYER_RESULTS = ["pass", "fail", "abstain"]


@lru_cache(maxsize=1)
def vocab() -> dict:
    return yaml.safe_load((C.path("kb") / "vocab.yaml").read_text())


def enum(field: str) -> list[str]:
    v = vocab()
    key = {"instrument": "instruments", "verb": "verbs", "target": "targets", "finding": "findings",
           "condition": "conditions", "grade": "grades", "phase": "phases",
           "cvs_criterion": "cvs_criteria", "decision": "decisions"}[field]
    x = v[key]
    return list(x.keys()) if isinstance(x, dict) else list(x)


ENTITY_FIELDS = ["instrument", "verb", "target", "finding", "condition", "grade", "phase",
                 "cvs_criterion", "decision"]


class Entities(BaseModel):
    instrument: Optional[str] = None
    verb: Optional[str] = None
    target: Optional[str] = None
    finding: Optional[str] = None
    condition: Optional[str] = None
    grade: Optional[str] = None
    phase: Optional[str] = None
    cvs_criterion: Optional[str] = None
    decision: Optional[str] = None


class Claim(BaseModel):
    claim_id: str
    clip_id: str
    t_start: int = Field(description="first absolute frame index the claim covers")
    t_end: int = Field(description="last absolute frame index the claim covers")
    type: Literal["presence", "absence", "action", "phase", "cvs_criterion", "finding", "condition",
                  "grade", "decision_point", "knowledge"]
    modality: Literal["observed", "knowledge", "inference"]
    entities: Entities = Entities()
    polarity: Literal["asserted", "negated", "hedged"] = "asserted"
    evidence_refs: list[str] = []
    kb_refs: list[str] = []
    confidence: float = 0.5
    text: str = Field("", description="the original sentence/phrase this claim came from")
    mode: Literal["free", "grounded"] = "grounded"
    source: str = "gemini"         # gemini | mock | injected
    statement: str = ""            # knowledge claims: the asserted fact in words


class ClaimList(BaseModel):
    """What Gemini returns: claims without ids/clip/mode (filled in by code)."""
    class Item(BaseModel):
        t_start: int
        t_end: int
        type: str
        modality: str
        entities: Entities = Field(description="REQUIRED: the enum values this claim is about (instrument/target/finding/phase/...)")
        polarity: str = "asserted"
        evidence_refs: list[str] = []
        kb_refs: list[str] = []
        confidence: float = 0.5
        text: str = ""
        statement: str = ""
    claims: list[Item]


class LayerResult(BaseModel):
    layer: str          # L1..L7, P (proportionality)
    result: Literal["pass", "fail", "abstain"]
    reason: str = ""
    hedge: bool = False  # pass, but evidence uncertain -> hedge


def scored_class(claim: dict) -> Optional[str]:
    """Map a claim's entity to a CholecSeg8k scored class, or None if not scorable."""
    v = vocab()
    e = claim.get("entities") or {}
    f = e.get("finding")
    if f and not ((v["findings"].get(f) or {}).get("scored_class")):
        return None  # the claim is about a finding on a structure (e.g. "no perforation of the GB"), not the structure
    for field, table in (("instrument", "instruments"), ("target", "targets"), ("finding", "findings")):
        val = e.get(field)
        if val and isinstance(v[table], dict) and val in v[table]:
            sc = (v[table][val] or {}).get("scored_class")
            if sc:
                return sc
    return None


def normalise_entity(field: str, value: Optional[str]) -> Optional[str]:
    """Map synonyms onto canonical enum values; return None-string sentinel if unknown."""
    if value is None or value == "":
        return None
    v = vocab()
    key = {"instrument": "instruments", "verb": "verbs", "target": "targets", "finding": "findings"}.get(field)
    s = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key and isinstance(v[key], dict):
        for canon, meta in v[key].items():
            syns = [canon] + list((meta or {}).get("synonyms", []))
            if s in [x.lower().replace("-", "_").replace(" ", "_") for x in syns]:
                return canon
        return str(value)
    return s
