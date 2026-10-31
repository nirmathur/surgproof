"""Error injection harness for claim validation and evaluation.

Takes correct claims and returns corrupted copies, each with an injected error:
- claim['injected_error'] = <error_type>
- claim['claim_id'] with suffix '-inj-<error_type>-<n>'
- claim['source'] = 'injected'
"""
from __future__ import annotations

import copy
import random
from typing import Any

import pandas as pd

from narrator import cfg as C
from narrator import schema

INJECTION_TYPES = [
    "entity_swap",
    "absent_structure",
    "impossible_combination",
    "phase_order_violation",
    "cvs_after_clip",
    "grade_escalation",
    "not_inferable_condition",
    "fabricated_knowledge",
    "shifted_timestamp",
]

DEFAULT_FALLBACK_CLAIM = {
    "claim_id": "base-claim-0",
    "clip_id": "video01_0001",
    "t_start": 100,
    "t_end": 140,
    "type": "presence",
    "modality": "observed",
    "polarity": "asserted",
    "entities": {"instrument": "grasper", "target": "gallbladder"},
    "source": "gold",
    "confidence": 1.0,
    "text": "Grasper and gallbladder are present",
    "mode": "grounded",
}


def make_gold_claims(gt_df: pd.DataFrame, clips_df: pd.DataFrame | None = None) -> list[dict]:
    """Build CORRECT presence, absence, phase, and knowledge claims from GT data."""
    scored_classes = C.cfg().get("scored_classes", ["grasper", "l_hook", "liver", "gallbladder", "cystic_duct", "blood"])
    claims: list[dict] = []

    if gt_df is None or gt_df.empty:
        return [dict(DEFAULT_FALLBACK_CLAIM)]

    clips = gt_df["clip_id"].unique()
    for clip_id in sorted(clips):
        clip_rows = gt_df[gt_df["clip_id"] == clip_id]
        first_frame = int(clip_rows["frame_idx"].min()) if not clip_rows.empty else 0
        last_frame = int(clip_rows["frame_idx"].max()) if not clip_rows.empty else 80

        for sc in scored_classes:
            rows_sc = clip_rows[clip_rows["label"] == sc]
            is_present = bool(rows_sc["present"].any()) if not rows_sc.empty else False

            entities: dict[str, str] = {}
            if sc in ("grasper", "hook"):
                entities["instrument"] = sc
            elif sc == "l_hook":
                entities["instrument"] = "hook"
            elif sc == "blood":
                entities["finding"] = "blood"
            else:
                entities["target"] = sc

            if is_present:
                pres_frames = rows_sc[rows_sc["present"]]["frame_idx"].tolist()
                t_start = int(min(pres_frames))
                t_end = int(max(pres_frames))
                claims.append(
                    {
                        "claim_id": f"gold-pres-{clip_id}-{sc}",
                        "clip_id": clip_id,
                        "t_start": t_start,
                        "t_end": t_end,
                        "type": "presence",
                        "modality": "observed",
                        "polarity": "asserted",
                        "entities": entities,
                        "source": "gold",
                        "text": f"{sc} is present",
                        "confidence": 1.0,
                        "mode": "grounded",
                    }
                )
            else:
                claims.append(
                    {
                        "claim_id": f"gold-abs-{clip_id}-{sc}",
                        "clip_id": clip_id,
                        "t_start": first_frame,
                        "t_end": last_frame,
                        "type": "absence",
                        "modality": "observed",
                        "polarity": "asserted",
                        "entities": entities,
                        "source": "gold",
                        "text": f"{sc} is absent",
                        "confidence": 1.0,
                        "mode": "grounded",
                    }
                )

        # Correct phase claim
        claims.append(
            {
                "claim_id": f"gold-phase-{clip_id}",
                "clip_id": clip_id,
                "t_start": first_frame,
                "t_end": last_frame,
                "type": "phase",
                "modality": "inference",
                "polarity": "asserted",
                "entities": {"phase": "calot_triangle_dissection"},
                "source": "gold",
                "text": "Surgical phase is calot_triangle_dissection",
                "confidence": 1.0,
                "mode": "grounded",
            }
        )

    # Correct knowledge claim citing real rule in KB
    first_clip = claims[0]["clip_id"] if claims else "video01_0001"
    claims.append(
        {
            "claim_id": "gold-know-cvs1",
            "clip_id": first_clip,
            "t_start": 0,
            "t_end": 0,
            "type": "knowledge",
            "modality": "knowledge",
            "polarity": "asserted",
            "statement": "The hepatocystic triangle must be completely cleared of fat and fibrous tissue",
            "kb_refs": ["cvs.c1_triangle_cleared"],
            "entities": {},
            "source": "gold",
            "text": "Anatomical guideline: hepatocystic triangle dissection",
            "confidence": 1.0,
            "mode": "grounded",
        }
    )

    return claims


def inject(
    claims: list[dict],
    seed: int = 0,
    per_type: int | None = None,
    absent_classes: dict[str, list[str]] | None = None,
    clip_bounds: dict[str, tuple[int, int]] | None = None,
) -> list[dict]:
    """Generate corrupted claims from correct claims across 9 error types."""
    rng = random.Random(seed)
    count = per_type if per_type is not None else 1
    absent_classes = absent_classes or {}
    clip_bounds = clip_bounds or {}

    pool = claims if claims else [dict(DEFAULT_FALLBACK_CLAIM)]
    corrupted: list[dict] = []

    # 1. entity_swap (grasper<->hook, liver<->gallbladder, etc.)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        raw_e = base.get("entities") or {}
        entities = raw_e.model_dump() if hasattr(raw_e, "model_dump") else dict(raw_e)

        inst = entities.get("instrument")
        tgt = entities.get("target")
        if inst == "grasper":
            entities["instrument"] = "hook"
        elif inst == "hook":
            entities["instrument"] = "grasper"
        elif tgt == "gallbladder":
            entities["target"] = "liver"
        elif tgt == "liver":
            entities["target"] = "gallbladder"
        elif inst:
            entities["instrument"] = "hook" if inst != "hook" else "grasper"
        elif tgt:
            entities["target"] = "liver" if tgt != "liver" else "gallbladder"
        else:
            entities["instrument"] = "hook"

        base["entities"] = entities
        base["claim_id"] = f"{orig_id}-inj-entity_swap-{n}"
        base["source"] = "injected"
        base["injected_error"] = "entity_swap"
        base["text"] = f"Injected swap: {entities}"
        corrupted.append(base)

    # 2. absent_structure (presence claim for a scored class absent in GT)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        clip_id = base.get("clip_id", "video01_0001")

        candidates = absent_classes.get(clip_id, ["blood", "cystic_duct", "hook", "liver"])
        cls_choice = candidates[n % len(candidates)]

        entities = {}
        if cls_choice in ("grasper", "hook"):
            entities["instrument"] = cls_choice
        elif cls_choice == "blood":
            entities["finding"] = "blood"
        else:
            entities["target"] = cls_choice

        base["type"] = "presence"
        base["modality"] = "observed"
        base["polarity"] = "asserted"
        base["entities"] = entities
        base["claim_id"] = f"{orig_id}-inj-absent_structure-{n}"
        base["source"] = "injected"
        base["injected_error"] = "absent_structure"
        base["text"] = f"Structure {cls_choice} is present"
        corrupted.append(base)

    # 3. impossible_combination (e.g. clipper+coagulate+liver, scissors grasp, clip liver)
    combos = [
        {"instrument": "clipper", "verb": "coagulate", "target": "liver"},
        {"instrument": "scissors", "verb": "grasp", "target": "gallbladder"},
        {"instrument": "clipper", "verb": "clip", "target": "liver"},
        {"instrument": "irrigator", "verb": "cut", "target": "gut"},
    ]
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        combo = combos[n % len(combos)]

        base["type"] = "action"
        base["modality"] = "observed"
        base["polarity"] = "asserted"
        base["entities"] = dict(combo)
        base["claim_id"] = f"{orig_id}-inj-impossible_combination-{n}"
        base["source"] = "injected"
        base["injected_error"] = "impossible_combination"
        base["text"] = f"{combo['instrument']} {combo['verb']}s {combo.get('target', '')}"
        corrupted.append(base)

    # 4. phase_order_violation (phase claim jumping back e.g. gallbladder_packaging -> calot_triangle_dissection)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        start_t = base.get("t_start", 100)

        # Prior phase claim in same clip/video: gallbladder_packaging
        prior = copy.deepcopy(base)
        prior["type"] = "phase"
        prior["modality"] = "inference"
        prior["polarity"] = "asserted"
        prior["entities"] = {"phase": "gallbladder_packaging"}
        prior["t_start"] = max(0, start_t - 40)
        prior["t_end"] = max(10, start_t - 20)
        prior["claim_id"] = f"{orig_id}-inj-phase_order_violation-prior-{n}"
        prior["source"] = "injected"
        prior["injected_error"] = "phase_order_violation"
        prior["text"] = "Phase is gallbladder_packaging"
        corrupted.append(prior)

        # Jumping back to calot_triangle_dissection
        jump = copy.deepcopy(base)
        jump["type"] = "phase"
        jump["modality"] = "inference"
        jump["polarity"] = "asserted"
        jump["entities"] = {"phase": "calot_triangle_dissection"}
        jump["t_start"] = max(20, start_t)
        jump["t_end"] = max(30, start_t + 20)
        jump["claim_id"] = f"{orig_id}-inj-phase_order_violation-{n}"
        jump["source"] = "injected"
        jump["injected_error"] = "phase_order_violation"
        jump["text"] = "Phase is calot_triangle_dissection"
        corrupted.append(jump)

    # 5. cvs_after_clip (clip action claim + later cvs_criterion asserted claim)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        start_t = base.get("t_start", 100)

        # Prior clip action
        clip_act = copy.deepcopy(base)
        clip_act["type"] = "action"
        clip_act["modality"] = "observed"
        clip_act["polarity"] = "asserted"
        clip_act["entities"] = {"instrument": "clipper", "verb": "clip", "target": "cystic_duct"}
        clip_act["t_start"] = max(0, start_t - 40)
        clip_act["t_end"] = max(10, start_t - 20)
        clip_act["claim_id"] = f"{orig_id}-inj-cvs_after_clip-prior-{n}"
        clip_act["source"] = "injected"
        clip_act["injected_error"] = "cvs_after_clip"
        clip_act["text"] = "Clipper applied to cystic duct"
        corrupted.append(clip_act)

        # Later CVS achieved claim
        cvs_c = copy.deepcopy(base)
        cvs_c["type"] = "cvs_criterion"
        cvs_c["modality"] = "observed"
        cvs_c["polarity"] = "asserted"
        cvs_c["entities"] = {"cvs_criterion": "cvs_triangle_cleared"}
        cvs_c["t_start"] = max(20, start_t)
        cvs_c["t_end"] = max(30, start_t + 20)
        cvs_c["claim_id"] = f"{orig_id}-inj-cvs_after_clip-{n}"
        cvs_c["source"] = "injected"
        cvs_c["injected_error"] = "cvs_after_clip"
        cvs_c["text"] = "Critical view of safety achieved"
        corrupted.append(cvs_c)

    # 6. grade_escalation (grade claim higher than findings support)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")

        base["type"] = "grade"
        base["modality"] = "inference"
        base["polarity"] = "asserted"
        base["entities"] = {"grade": "parkland_4"}
        base["claim_id"] = f"{orig_id}-inj-grade_escalation-{n}"
        base["source"] = "injected"
        base["injected_error"] = "grade_escalation"
        base["text"] = "Parkland grade 4 difficulty"
        corrupted.append(base)

    # 7. not_inferable_condition (condition with visibility not_inferable/needs_non_video_data)
    cond_choices = ["choledocholithiasis", "gallbladder_cancer", "biliary_colic"]
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        cond = cond_choices[n % len(cond_choices)]

        base["type"] = "condition"
        base["modality"] = "inference"
        base["polarity"] = "asserted"
        base["entities"] = {"condition": cond}
        base["claim_id"] = f"{orig_id}-inj-not_inferable_condition-{n}"
        base["source"] = "injected"
        base["injected_error"] = "not_inferable_condition"
        base["text"] = f"Patient has {cond}"
        corrupted.append(base)

    # 8. fabricated_knowledge (statement with fake rule id)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")

        base["type"] = "knowledge"
        base["modality"] = "knowledge"
        base["polarity"] = "asserted"
        base["statement"] = "The cystic duct is always longer than 5 cm"
        base["kb_refs"] = ["fake_rule_99999", "non_existent_kb_ref"]
        base["entities"] = {}
        base["claim_id"] = f"{orig_id}-inj-fabricated_knowledge-{n}"
        base["source"] = "injected"
        base["injected_error"] = "fabricated_knowledge"
        base["text"] = "Knowledge: The cystic duct is always longer than 5 cm"
        corrupted.append(base)

    # 9. shifted_timestamp (shifted outside clip bounds)
    for n in range(count):
        base = copy.deepcopy(pool[n % len(pool)])
        orig_id = base.get("claim_id", f"claim_{n}")
        clip_id = base.get("clip_id", "video01_0001")

        if clip_id in clip_bounds:
            first, last = clip_bounds[clip_id]
            base["t_start"] = last + 50
            base["t_end"] = last + 100
        else:
            base["t_start"] = 99999
            base["t_end"] = 100050

        base["claim_id"] = f"{orig_id}-inj-shifted_timestamp-{n}"
        base["source"] = "injected"
        base["injected_error"] = "shifted_timestamp"
        corrupted.append(base)

    return corrupted
