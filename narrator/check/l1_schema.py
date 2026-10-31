"""L1 schema and vocabulary claim checker layer."""
from __future__ import annotations

from typing import Any

from narrator.check.base import Ctx, res, safe_map
from narrator import schema

NAME = "L1"


def _check_claim(c: Any, ctx: Ctx) -> dict:
    if not isinstance(c, dict):
        return res("fail", "Claim must be a dict")

    # Required identifiers
    claim_id = c.get("claim_id")
    if not claim_id or not isinstance(claim_id, str):
        return res("fail", "Missing or non-string 'claim_id'")

    clip_id = c.get("clip_id")
    if not clip_id or not isinstance(clip_id, str):
        return res("fail", "Missing or non-string 'clip_id'")

    # Type check
    claim_type = c.get("type")
    if claim_type not in schema.CLAIM_TYPES:
        return res("fail", f"Invalid claim type '{claim_type}'; allowed: {schema.CLAIM_TYPES}")

    # Polarity check
    polarity = c.get("polarity")
    if polarity not in schema.POLARITIES:
        return res("fail", f"Invalid polarity '{polarity}'; allowed: {schema.POLARITIES}")

    # Modality check
    modality = c.get("modality")
    if modality not in schema.MODALITIES:
        return res("fail", f"Invalid modality '{modality}'; allowed: {schema.MODALITIES}")

    # Modality valid for type
    if claim_type in ("presence", "absence", "action", "finding"):
        if modality != "observed":
            return res("fail", f"Claim type '{claim_type}' requires modality 'observed', got '{modality}'")
    elif claim_type == "knowledge":
        if modality != "knowledge":
            return res("fail", f"Claim type 'knowledge' requires modality 'knowledge', got '{modality}'")

    # Temporal range check
    t_start = c.get("t_start")
    t_end = c.get("t_end")
    if t_start is None or t_end is None or not isinstance(t_start, int) or not isinstance(t_end, int):
        return res("fail", "t_start and t_end must be non-null integers")
    if t_start > t_end:
        return res("fail", f"t_start ({t_start}) must be <= t_end ({t_end})")

    # Entities extraction
    raw_entities = c.get("entities")
    if hasattr(raw_entities, "model_dump"):
        entities = raw_entities.model_dump()
    elif isinstance(raw_entities, dict):
        entities = raw_entities
    else:
        entities = {}

    # Check non-null entity values against vocab enum
    for field in schema.ENTITY_FIELDS:
        val = entities.get(field)
        if val is not None and val != "":
            allowed = schema.enum(field)
            if val not in allowed:
                return res("fail", f"Entity field '{field}' has invalid value '{val}' (not in enum)")

    # Type <-> entity consistency
    if claim_type in ("presence", "absence"):
        if not (entities.get("instrument") or entities.get("target") or entities.get("finding")):
            return res("fail", f"Type '{claim_type}' requires at least one of instrument, target, or finding")

    elif claim_type == "action":
        if not (entities.get("instrument") and entities.get("verb")):
            return res("fail", "Type 'action' requires both instrument and verb entities")

    elif claim_type == "phase":
        if not entities.get("phase"):
            return res("fail", "Type 'phase' requires phase entity")

    elif claim_type == "cvs_criterion":
        if not entities.get("cvs_criterion"):
            return res("fail", "Type 'cvs_criterion' requires cvs_criterion entity")

    elif claim_type == "grade":
        if not entities.get("grade"):
            return res("fail", "Type 'grade' requires grade entity")

    elif claim_type == "condition":
        if not entities.get("condition"):
            return res("fail", "Type 'condition' requires condition entity")

    elif claim_type == "decision_point":
        if not entities.get("decision"):
            return res("fail", "Type 'decision_point' requires decision entity")

    elif claim_type == "knowledge":
        statement = c.get("statement")
        kb_refs = c.get("kb_refs")
        has_stmt = isinstance(statement, str) and statement.strip() != ""
        has_kb = isinstance(kb_refs, list) and len(kb_refs) > 0
        if not (has_stmt or has_kb):
            return res("fail", "Type 'knowledge' requires non-empty statement or kb_refs")

    return res("pass", "Schema and vocabulary valid")


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L1 schema and vocabulary validation on claims."""
    return safe_map(_check_claim, claims, ctx)
