"""L2 domain rules claim checker layer."""
from __future__ import annotations

from typing import Any
import yaml

from narrator.check.base import Ctx, res
from narrator import cfg as C
from narrator import schema

NAME = "L2"

# Built-in fallback defaults in case KB files are missing or incomplete
DEFAULT_PHASE_TOOLS = {
    "preparation": ["grasper", "hook"],
    "calot_triangle_dissection": ["grasper", "hook", "bipolar", "irrigator"],
    "clipping_cutting": ["grasper", "clipper", "scissors"],
    "gallbladder_dissection": ["grasper", "hook", "bipolar"],
    "gallbladder_packaging": ["grasper", "specimen_bag_tool"],
    "cleaning_coagulation": ["grasper", "irrigator", "hook", "bipolar"],
    "gallbladder_retraction": ["grasper", "specimen_bag_tool"],
}

DEFAULT_TRANSITIONS = [
    ["preparation", "preparation"],
    ["preparation", "calot_triangle_dissection"],
    ["calot_triangle_dissection", "calot_triangle_dissection"],
    ["calot_triangle_dissection", "clipping_cutting"],
    ["clipping_cutting", "clipping_cutting"],
    ["clipping_cutting", "gallbladder_dissection"],
    ["gallbladder_dissection", "gallbladder_dissection"],
    ["gallbladder_dissection", "gallbladder_packaging"],
    ["gallbladder_dissection", "cleaning_coagulation"],
    ["gallbladder_dissection", "gallbladder_retraction"],
    ["gallbladder_packaging", "gallbladder_packaging"],
    ["gallbladder_packaging", "cleaning_coagulation"],
    ["cleaning_coagulation", "cleaning_coagulation"],
    ["cleaning_coagulation", "gallbladder_packaging"],
    ["cleaning_coagulation", "gallbladder_retraction"],
    ["gallbladder_packaging", "gallbladder_retraction"],
    ["gallbladder_retraction", "gallbladder_retraction"],
]

DEFAULT_GRADES = {
    "parkland_1": {"required_findings": []},
    "parkland_2": {"required_findings": ["adhesions_minor"]},
    "parkland_3": {"required_findings": ["hyperaemia"]},
    "parkland_4": {"required_findings": ["adhesions_dense"]},
    "parkland_5": {"required_findings": ["necrosis"]},
    "nassar_1": {"required_findings": []},
    "nassar_2": {"required_findings": ["adhesions_minor"]},
    "nassar_3": {"required_findings": ["adhesions_dense"]},
    "nassar_4": {"required_findings": ["distension"]},
    "nassar_5": {"required_findings": ["necrosis"]},
}

DEFAULT_CONDITIONS = {
    "acute_cholecystitis": {
        "visibility": "visible",
        "supporting_findings": ["hyperaemia", "wall_thickening", "distension", "pericholecystic_fluid"],
        "against_findings": [],
    },
    "chronic_cholecystitis": {
        "visibility": "needs_non_video_data",
        "supporting_findings": ["adhesions_dense", "wall_thickening"],
        "against_findings": ["distension"],
    },
    "gangrenous_cholecystitis": {
        "visibility": "visible",
        "supporting_findings": ["necrosis", "perforation", "bile_spillage"],
        "against_findings": [],
    },
    "mirizzi_syndrome": {
        "visibility": "needs_non_video_data",
        "supporting_findings": ["distension", "adhesions_dense"],
        "against_findings": [],
    },
    "empyema": {
        "visibility": "visible",
        "supporting_findings": ["distension", "wall_thickening", "pericholecystic_fluid"],
        "against_findings": [],
    },
    "gallbladder_cancer": {
        "visibility": "not_inferable",
        "supporting_findings": ["wall_thickening"],
        "against_findings": [],
    },
    "biliary_colic": {
        "visibility": "not_inferable",
        "supporting_findings": [],
        "against_findings": ["necrosis", "perforation"],
    },
    "choledocholithiasis": {
        "visibility": "not_inferable",
        "supporting_findings": [],
        "against_findings": [],
    },
}


def _read_kb_yaml(filename: str) -> dict | None:
    path = C.path("kb") / filename
    if path.is_file():
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            return None
    return None


def get_phases_kb() -> tuple[dict[str, list[str]], set[tuple[str, str]], bool]:
    data = _read_kb_yaml("phases.yaml")
    if not data or "phases" not in data:
        transitions_set = {(a, b) for a, b in DEFAULT_TRANSITIONS}
        return DEFAULT_PHASE_TOOLS, transitions_set, True

    phase_tools = {}
    for p in data.get("phases", []):
        if isinstance(p, dict) and "name" in p:
            phase_tools[p["name"]] = p.get("tools", [])

    transitions_set = set()
    for item in data.get("transitions", []):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            transitions_set.add((item[0], item[1]))
    # Always allow remaining in same phase
    for name in phase_tools:
        transitions_set.add((name, name))
    return phase_tools, transitions_set, False


def get_grading_kb() -> tuple[dict[str, dict], bool]:
    data = _read_kb_yaml("grading.yaml")
    if not data or "grades" not in data:
        return DEFAULT_GRADES, True
    grades = dict(DEFAULT_GRADES)
    grades.update(data.get("grades", {}))
    return grades, False


def get_conditions_kb() -> tuple[dict[str, dict], bool]:
    data = _read_kb_yaml("conditions.yaml")
    if not data or "conditions" not in data:
        return DEFAULT_CONDITIONS, True
    conds = dict(DEFAULT_CONDITIONS)
    conds.update(data.get("conditions", {}))
    return conds, False


def _video_of(c: dict) -> str:
    return c.get("video_id") or c.get("clip_id", "").split("_")[0]


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L2 domain rules check on claims."""
    try:
        v = schema.vocab()
        valid_instrument_verbs = v.get("valid_instrument_verbs", {})
        invalid_verb_targets = v.get("invalid_verb_targets", {})
        phase_tools, allowed_transitions, phases_fallback = get_phases_kb()
        grades_kb, grading_fallback = get_grading_kb()
        conditions_kb, conditions_fallback = get_conditions_kb()
    except Exception as e:
        return [res("abstain", f"error loading KB in L2: {str(e)[:120]}")] * len(claims)

    passed_ids = ctx.extra.get("passed_ids") if ctx.extra else None

    # Pre-index finding claims per clip
    clip_findings: dict[str, set[str]] = {}
    for c in claims:
        if not isinstance(c, dict):
            continue
        clip_id = c.get("clip_id", "")
        ctype = c.get("type")
        cmod = c.get("modality")
        cpol = c.get("polarity")
        if ctype == "finding" and cmod == "observed" and cpol == "asserted":
            if passed_ids is None or c.get("claim_id") in passed_ids:
                entities = c.get("entities") or {}
                if hasattr(entities, "model_dump"):
                    entities = entities.model_dump()
                f = entities.get("finding")
                if f:
                    clip_findings.setdefault(clip_id, set()).add(f)

    # Pre-index clip/cut actions per video
    # List of (t_start, claim_index, verb, claim)
    video_clip_actions: dict[str, list[dict]] = {}
    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        cpol = c.get("polarity", "asserted")
        entities = c.get("entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()
        verb = entities.get("verb")
        if ctype == "action" and cpol == "asserted" and verb in ("clip", "cut"):
            vid = _video_of(c)
            entry = dict(c)
            entry["_orig_idx"] = idx
            video_clip_actions.setdefault(vid, []).append(entry)

    # Pre-index phase claims per video (sorted by t_start then original index)
    video_phase_claims: dict[str, list[dict]] = {}
    for idx, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        ctype = c.get("type")
        entities = c.get("entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()
        phase = entities.get("phase")
        if ctype == "phase" and phase:
            vid = _video_of(c)
            entry = dict(c)
            entry["_orig_idx"] = idx
            video_phase_claims.setdefault(vid, []).append(entry)

    for vid in video_phase_claims:
        video_phase_claims[vid].sort(key=lambda x: (x.get("t_start", 0), x.get("_orig_idx", 0)))

    results = []
    for idx, c in enumerate(claims):
        try:
            if not isinstance(c, dict):
                results.append(res("fail", "Claim must be a dict"))
                continue

            ctype = c.get("type")
            cpol = c.get("polarity", "asserted")
            clip_id = c.get("clip_id", "")
            vid = _video_of(c)
            raw_entities = c.get("entities")
            if hasattr(raw_entities, "model_dump"):
                entities = raw_entities.model_dump()
            elif isinstance(raw_entities, dict):
                entities = raw_entities
            else:
                entities = {}

            instrument = entities.get("instrument")
            verb = entities.get("verb")
            target = entities.get("target")
            phase = entities.get("phase")
            grade = entities.get("grade")
            condition = entities.get("condition")
            cvs_criterion = entities.get("cvs_criterion")
            decision = entities.get("decision")

            rules_applied = 0
            rule_pass_reasons = []

            # 1. Valid instrument-verb and verb-target
            if instrument and verb:
                rules_applied += 1
                allowed_verbs = valid_instrument_verbs.get(instrument)
                if allowed_verbs is not None and verb not in allowed_verbs:
                    results.append(res("fail", f"Invalid instrument-verb: instrument '{instrument}' cannot perform '{verb}'"))
                    continue
                rule_pass_reasons.append(f"instrument '{instrument}' valid with '{verb}'")

            if verb and target:
                rules_applied += 1
                prohibited_targets = invalid_verb_targets.get(verb)
                if prohibited_targets is not None and target in prohibited_targets:
                    results.append(res("fail", f"Invalid verb-target: verb '{verb}' cannot target '{target}'"))
                    continue
                rule_pass_reasons.append(f"verb '{verb}' valid for target '{target}'")

            # 2. Phase claims: transition from previous phase claim must be allowed
            if ctype == "phase" or phase:
                rules_applied += 1
                if not phase:
                    results.append(res("fail", "Phase claim missing phase entity"))
                    continue

                # Find preceding phase claim in the same video
                phases_seq = video_phase_claims.get(vid, [])
                prev_phase = None
                for pc in phases_seq:
                    if pc.get("claim_id") == c.get("claim_id") and pc.get("_orig_idx") == idx:
                        break
                    # Previous in chronological / sequence order
                    if pc.get("t_start", 0) < c.get("t_start", 0) or (
                        pc.get("t_start", 0) == c.get("t_start", 0) and pc.get("_orig_idx", 0) < idx
                    ):
                        prev_phase = (pc.get("entities") or {}).get("phase")

                if prev_phase is not None:
                    if (prev_phase, phase) not in allowed_transitions and prev_phase != phase:
                        results.append(res("fail", f"Invalid phase transition: '{prev_phase}' -> '{phase}' is not allowed"))
                        continue
                    rule_pass_reasons.append(f"valid transition '{prev_phase}' -> '{phase}'")
                else:
                    rule_pass_reasons.append(f"first phase claim '{phase}'")

                results.append(res("pass", "; ".join(rule_pass_reasons)))
                continue

            # 3. Tool fits the phase
            if instrument:
                # Find phase from ctx.clip_phase or latest phase claim
                current_phase = ctx.clip_phase.get(clip_id) if ctx.clip_phase else None
                if not current_phase:
                    phases_seq = video_phase_claims.get(vid, [])
                    for pc in phases_seq:
                        if pc.get("t_start", 0) <= c.get("t_start", 0):
                            current_phase = (pc.get("entities") or {}).get("phase")

                if current_phase:
                    rules_applied += 1
                    allowed_tools = phase_tools.get(current_phase, [])
                    if instrument not in allowed_tools:
                        fb_note = " (using fallback phases)" if phases_fallback else ""
                        results.append(res("fail", f"Tool '{instrument}' does not fit phase '{current_phase}' (expected {allowed_tools}){fb_note}"))
                        continue
                    rule_pass_reasons.append(f"tool '{instrument}' fits phase '{current_phase}'")

            # 4. "CVS achieved" after any clip/cut action claim earlier in the same video -> fail
            is_cvs_achieved = False
            if ctype == "cvs_criterion" and cpol == "asserted":
                is_cvs_achieved = True
            elif ctype == "decision_point" and decision == "cvs_pause" and cpol == "asserted":
                is_cvs_achieved = True

            if is_cvs_achieved:
                rules_applied += 1
                clip_actions = video_clip_actions.get(vid, [])
                has_earlier_clip = False
                earlier_info = ""
                for act in clip_actions:
                    act_t = act.get("t_start", 0)
                    c_t = c.get("t_start", 0)
                    if act_t < c_t or (act_t == c_t and act.get("_orig_idx", 0) < idx):
                        has_earlier_clip = True
                        earlier_verb = (act.get("entities") or {}).get("verb", "clip")
                        earlier_info = f"action '{earlier_verb}' at frame {act_t}"
                        break

                if has_earlier_clip:
                    results.append(res("fail", f"CVS achieved claimed at frame {c.get('t_start')} after earlier {earlier_info} in video"))
                    continue
                rule_pass_reasons.append("CVS achieved claim is temporally prior to clipping/cutting")
                results.append(res("pass", "; ".join(rule_pass_reasons)))
                continue

            # 5 & 6. Grade claims
            if ctype == "grade" or grade:
                rules_applied += 1
                if not grade:
                    results.append(res("fail", "Grade claim missing grade entity"))
                    continue

                if "tokyo" in grade.lower():
                    results.append(res("fail", f"Tokyo severity grades ('{grade}') are not inferable from video"))
                    continue

                grade_info = grades_kb.get(grade)
                if grade_info is None:
                    results.append(res("fail", f"Unknown grade '{grade}'"))
                    continue

                required = grade_info.get("required_findings", [])
                confirmed = clip_findings.get(clip_id, set())
                missing = [rf for rf in required if rf not in confirmed]
                if missing:
                    fb_note = " (using fallback grading)" if grading_fallback else ""
                    results.append(res("fail", f"Grade '{grade}' requires findings {missing} not asserted by observed findings in clip{fb_note}"))
                    continue

                rule_pass_reasons.append(f"grade '{grade}' supported by findings {required}")
                results.append(res("pass", "; ".join(rule_pass_reasons)))
                continue

            # 7 & 8. Condition claims
            if ctype == "condition" or condition:
                rules_applied += 1
                if not condition:
                    results.append(res("fail", "Condition claim missing condition entity"))
                    continue

                cond_info = conditions_kb.get(condition)
                if cond_info is None:
                    results.append(res("fail", f"Unknown condition '{condition}'"))
                    continue

                vis = cond_info.get("visibility", "visible")
                if vis in ("not_inferable", "needs_non_video_data"):
                    fb_note = " (using fallback conditions)" if conditions_fallback else ""
                    results.append(res("fail", f"Condition '{condition}' has visibility '{vis}' and cannot be inferred from video alone{fb_note}"))
                    continue

                supporting = set(cond_info.get("supporting_findings", []))
                against = set(cond_info.get("against_findings", []))
                confirmed = clip_findings.get(clip_id, set())

                present_against = against.intersection(confirmed)
                if present_against:
                    results.append(res("fail", f"Condition '{condition}' contradicted by against finding(s): {list(present_against)}"))
                    continue

                present_supp = supporting.intersection(confirmed)
                if not present_supp:
                    fb_note = " (using fallback conditions)" if conditions_fallback else ""
                    results.append(res("fail", f"Condition '{condition}' requires >=1 supporting finding from {list(supporting)}, none found{fb_note}"))
                    continue

                rule_pass_reasons.append(f"condition '{condition}' supported by visible findings {list(present_supp)}")
                results.append(res("pass", "; ".join(rule_pass_reasons)))
                continue

            if rules_applied > 0:
                results.append(res("pass", "; ".join(rule_pass_reasons)))
            else:
                results.append(res("abstain", "no L2 rules apply to this claim"))

        except Exception as e:
            results.append(res("abstain", f"error in L2: {str(e)[:120]}"))

    return results
