"""Proportionality checker layer (NAME = 'P').

A condition/grade claim must be the most conservative one the confirmed findings support:
e.g. parkland_4 claimed but only parkland_2 required_findings present -> fail with reason naming
the conservative alternative.
"""
from __future__ import annotations

from typing import Any

from narrator.check.base import Ctx, res
from narrator.check.l2_rules import get_grading_kb, get_conditions_kb

NAME = "P"

PARKLAND_SCALE = ["parkland_1", "parkland_2", "parkland_3", "parkland_4", "parkland_5"]
NASSAR_SCALE = ["nassar_1", "nassar_2", "nassar_3", "nassar_4", "nassar_5"]


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run proportionality validation on claims."""
    try:
        grades_kb, _ = get_grading_kb()
        conditions_kb, _ = get_conditions_kb()
    except Exception as e:
        return [res("abstain", f"error loading KB in proportionality: {str(e)[:120]}")] * len(claims)

    passed_ids = ctx.extra.get("passed_ids") if ctx.extra else None

    # Pre-index confirmed finding claims per clip
    clip_findings: dict[str, set[str]] = {}
    for c in claims:
        if not isinstance(c, dict):
            continue
        clip_id = c.get("clip_id", "")
        ctype = c.get("type")
        cmod = c.get("modality")
        cpol = c.get("polarity", "asserted")
        if ctype == "finding" and cmod == "observed" and cpol == "asserted":
            if passed_ids is None or c.get("claim_id") in passed_ids:
                raw_entities = c.get("entities") or {}
                if hasattr(raw_entities, "model_dump"):
                    raw_entities = raw_entities.model_dump()
                f = raw_entities.get("finding")
                if f:
                    clip_findings.setdefault(clip_id, set()).add(f)

    results = []
    for c in claims:
        try:
            if not isinstance(c, dict):
                results.append(res("fail", "Claim must be a dict"))
                continue

            ctype = c.get("type")
            clip_id = c.get("clip_id", "")
            raw_entities = c.get("entities")
            if hasattr(raw_entities, "model_dump"):
                entities = raw_entities.model_dump()
            elif isinstance(raw_entities, dict):
                entities = raw_entities
            else:
                entities = {}

            grade = entities.get("grade")
            condition = entities.get("condition")

            # 1. Grade claims
            if ctype == "grade" or grade:
                if not grade:
                    results.append(res("fail", "Grade claim missing grade entity"))
                    continue

                if "tokyo" in grade.lower():
                    results.append(res("fail", f"Tokyo severity grades ('{grade}') are not inferable from video"))
                    continue

                if grade.startswith("parkland_"):
                    scale = PARKLAND_SCALE
                elif grade.startswith("nassar_"):
                    scale = NASSAR_SCALE
                else:
                    results.append(res("abstain", f"Proportionality check only covers Parkland and Nassar scales, got '{grade}'"))
                    continue

                if grade not in scale:
                    results.append(res("fail", f"Unknown grade '{grade}' in scale"))
                    continue

                confirmed = clip_findings.get(clip_id, set())

                # Find highest grade in scale whose required findings are all satisfied
                supported_grades = []
                for g in scale:
                    g_info = grades_kb.get(g, {})
                    required = g_info.get("required_findings", [])
                    if all(rf in confirmed for rf in required):
                        supported_grades.append(g)

                max_supported = supported_grades[-1] if supported_grades else scale[0]
                claimed_idx = scale.index(grade)
                supported_idx = scale.index(max_supported)

                if claimed_idx > supported_idx:
                    results.append(
                        res(
                            "fail",
                            f"Claimed grade '{grade}' is escalated; confirmed findings {list(confirmed)} only support conservative alternative '{max_supported}'",
                        )
                    )
                else:
                    results.append(
                        res(
                            "pass",
                            f"Grade '{grade}' is proportional to confirmed findings (supported up to '{max_supported}')",
                        )
                    )
                continue

            # 2. Condition claims
            if ctype == "condition" or condition:
                if not condition:
                    results.append(res("fail", "Condition claim missing condition entity"))
                    continue

                cond_info = conditions_kb.get(condition, {})
                vis = cond_info.get("visibility", "visible")
                if vis in ("not_inferable", "needs_non_video_data"):
                    results.append(res("fail", f"Condition '{condition}' has visibility '{vis}' and cannot be inferred from video alone"))
                    continue

                confirmed = clip_findings.get(clip_id, set())
                supporting = set(cond_info.get("supporting_findings", []))
                against = set(cond_info.get("against_findings", []))

                has_supp = bool(supporting.intersection(confirmed))
                has_against = bool(against.intersection(confirmed))

                if not has_supp or has_against:
                    # Determine conservative alternative
                    conservative_alt = None
                    if condition in ("gangrenous_cholecystitis", "empyema"):
                        # If acute findings present, conservative alternative is acute_cholecystitis
                        acute_findings = set(conditions_kb.get("acute_cholecystitis", {}).get("supporting_findings", []))
                        if acute_findings.intersection(confirmed):
                            conservative_alt = "acute_cholecystitis"
                        else:
                            conservative_alt = "chronic_cholecystitis"
                    elif condition == "acute_cholecystitis":
                        conservative_alt = "chronic_cholecystitis"

                    if conservative_alt:
                        results.append(
                            res(
                                "fail",
                                f"Claimed condition '{condition}' not supported by confirmed findings; conservative alternative is '{conservative_alt}'",
                            )
                        )
                    else:
                        results.append(
                            res(
                                "fail",
                                f"Claimed condition '{condition}' is not supported by confirmed findings",
                            )
                        )
                else:
                    results.append(
                        res(
                            "pass",
                            f"Condition '{condition}' is proportional to confirmed findings {list(supporting.intersection(confirmed))}",
                        )
                    )
                continue

            # Non-applicable
            results.append(res("abstain", "proportionality applies only to grade and condition claims"))

        except Exception as e:
            results.append(res("abstain", f"error in proportionality: {str(e)[:120]}"))

    return results
