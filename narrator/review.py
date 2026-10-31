"""Post-hoc reference policy review of recorded laparoscopic cholecystectomy video.

Evaluates decision nodes against surgical guideline reference policies (SAGES, TG18).
Framing: post-hoc educational review of recorded video for teaching and documentation.
Strict guardrails:
- No intraoperative guidance
- No diagnosis
- No bile duct injury (BDI) prediction
- No surgeon-quality judgements or outcome regret
- Scored is always False (deviation from reference policy).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml

from narrator import cfg as C


def load_decisions() -> dict:
    """Load reference decision nodes and rules from kb/decisions.yaml."""
    kb_path = C.path("kb") / "decisions.yaml"
    if not kb_path.is_file():
        kb_path = Path(__file__).resolve().parent / "kb" / "decisions.yaml"
    with open(kb_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_claims(claims: Any) -> list[dict]:
    """Ensure claims is a list of standardized dicts with parsed entities."""
    if isinstance(claims, pd.DataFrame):
        rows = claims.to_dict(orient="records")
    elif isinstance(claims, list):
        rows = [dict(c) for c in claims]
    else:
        rows = []

    norm = []
    for r in rows:
        c = dict(r)
        ent = c.get("entities")
        if isinstance(ent, str):
            try:
                c["entities"] = json.loads(ent)
            except Exception:
                c["entities"] = {}
        elif not isinstance(ent, dict):
            c["entities"] = {}
        norm.append(c)
    return norm


def review(
    claims: list[dict] | pd.DataFrame,
    events: list[dict] | None = None,
    video: str = "unknown_video",
    time_limit_s: float = 1200.0,
    fps: float | None = None,
) -> dict:
    """Produce post-hoc review structure evaluating 8 guideline decision nodes.

    Scoring per node:
      - ok: guideline action taken
      - inaccuracy: clip with 2 of 3 CVS criteria confirmed
      - blunder: clip with <= 1 criterion confirmed
      - missed_alternative: CVS not reached by time limit and no imaging/help/bail-out
      - unknown: insufficient evidence
    """
    raw_claims = _normalize_claims(claims)
    kb = load_decisions()
    node_defs = {n["id"]: n for n in kb.get("nodes", [])}

    # Resolve fps: if claims are frame indices (e.g. >3000), default fps=25.0; else 1.0 (seconds)
    if fps is None:
        has_s = any("t_end_s" in c or "t_start_s" in c for c in raw_claims)
        max_raw = max([float(c.get("t_end", 0)) for c in raw_claims], default=0.0)
        if has_s:
            eff_fps = 1.0
        elif max_raw > 3000:
            eff_fps = 25.0
        else:
            eff_fps = 1.0
    else:
        eff_fps = fps

    # 1. Identify difficulty assessment
    difficulty_grade = "unknown"
    difficulty_claim_ids = []
    parkland_num: Optional[int] = None

    for c in raw_claims:
        ctype = c.get("type")
        ent = c.get("entities", {})
        grade_val = ent.get("grade") or (c.get("statement") if ctype == "grade" else None)
        # Check verdict: if present, must be speak/hedge; if missing, check asserted
        verdict = c.get("verdict")
        if ctype == "grade" or grade_val:
            val_str = str(grade_val).lower() if grade_val else ""
            if "parkland" in val_str:
                difficulty_grade = val_str
                difficulty_claim_ids.append(c.get("claim_id", ""))
                try:
                    parkland_num = int(val_str.split("_")[-1])
                except Exception:
                    pass
                break
            elif "nassar" in val_str:
                difficulty_grade = val_str
                difficulty_claim_ids.append(c.get("claim_id", ""))
                break

    high_difficulty = parkland_num is not None and parkland_num >= 4

    # 2. Extract CVS criteria and clipping actions
    # CVS criterion confirmed ONLY if verdict in ("speak", "hedge")
    cvs_keys = {"cvs_triangle_cleared", "cvs_lower_gb_off_plate", "cvs_two_structures"}
    finding_to_cvs = {
        "triangle_cleared": "cvs_triangle_cleared",
        "lower_gb_off_plate": "cvs_lower_gb_off_plate",
        "two_structures_only": "cvs_two_structures",
    }

    confirmed_cvs: dict[str, dict] = {}
    clipping_claims: list[dict] = []
    bailout_claims: list[dict] = []
    imaging_help_claims: list[dict] = []
    retraction_claims: list[dict] = []
    plane_claims: list[dict] = []
    haemostasis_claims: list[dict] = []

    for c in raw_claims:
        cid = c.get("claim_id", "")
        ctype = c.get("type", "")
        ent = c.get("entities", {})
        verdict = c.get("verdict")
        polarity = c.get("polarity", "asserted")
        is_confirmed = (verdict in ("speak", "hedge")) and (polarity != "negated")

        # CVS criteria
        crit = ent.get("cvs_criterion")
        finding = ent.get("finding")
        mapped_crit = crit if crit in cvs_keys else finding_to_cvs.get(finding)
        if mapped_crit and is_confirmed:
            if mapped_crit not in confirmed_cvs:
                confirmed_cvs[mapped_crit] = c

        # Clipping / division actions
        verb = ent.get("verb")
        inst = ent.get("instrument")
        target = ent.get("target")
        phase = ent.get("phase")

        if verb == "clip" or inst == "clipper" or phase == "clipping_cutting":
            clipping_claims.append(c)

        # Bailout indicators
        decision_val = ent.get("decision")
        text = (c.get("text") or "").lower()
        statement = (c.get("statement") or "").lower()
        if (
            decision_val == "bailout_choice"
            or "subtotal" in text
            or "bailout" in text
            or "fundus_first" in text
            or "subtotal" in statement
        ):
            if is_confirmed or verdict is None:
                bailout_claims.append(c)

        # Imaging or calling for help
        if (
            "imaging" in text
            or "cholangiograph" in text
            or "icg" in text
            or "help" in text
            or "consult" in text
            or decision_val == "cvs_not_reached_action"
        ):
            if is_confirmed or verdict is None:
                imaging_help_claims.append(c)

        # Retraction
        if verb == "retract" or (inst == "grasper" and target == "gallbladder"):
            retraction_claims.append(c)

        # Dissection plane relative to R4U
        if decision_val == "dissection_plane_r4u" or "r4u" in text or "rouviere" in text:
            plane_claims.append(c)

        # Haemostasis / GB bed
        if (
            phase == "cleaning_coagulation"
            or target == "cystic_plate"
            or decision_val == "gb_bed_haemostasis"
            or verb in ("irrigate", "aspirate", "coagulate")
        ):
            haemostasis_claims.append(c)

    # Calculate time context
    max_t = 0.0
    for c in raw_claims:
        t_sec = float(c.get("t_end_s", c.get("t_end", 0))) / eff_fps
        if t_sec > max_t:
            max_t = t_sec

    clip_t = None
    clip_claim_ids = []
    if clipping_claims:
        first_clip = clipping_claims[0]
        t_val = float(first_clip.get("t_start_s", first_clip.get("t_start", 0)))
        clip_t = t_val / eff_fps
        clip_claim_ids = [c.get("claim_id", "") for c in clipping_claims if c.get("claim_id")]

    # Evaluate nodes
    scored_nodes = []

    # Node 1: difficulty_assessment
    n1_def = node_defs.get("difficulty_assessment", {})
    if difficulty_grade != "unknown":
        n1_verdict = "ok"
        n1_obs = f"Operative difficulty assessed as {difficulty_grade} from initial visual appearance."
        n1_cids = difficulty_claim_ids
    elif raw_claims:
        # The grade is the reviewer's assessment, not a visible surgeon action: not grading it is OUR gap.
        n1_verdict = "unknown"
        n1_obs = "Difficulty grade (Parkland) not assessable from the checked claims of the pre-dissection view."
        n1_cids = []
    else:
        n1_verdict = "unknown"
        n1_obs = "Insufficient claim evidence to determine difficulty assessment."
        n1_cids = []

    scored_nodes.append({
        "node_id": "difficulty_assessment",
        "t": 0.0,
        "verdict": n1_verdict,
        "observed": n1_obs,
        "guideline_action": n1_def.get("guideline_action", ""),
        "recommended_alternative": n1_def.get("alternatives", ["Assess Parkland grade"])[0],
        "evidence_grade": n1_def.get("evidence_grade", "expert opinion"),
        "source_quote": n1_def.get("quote", ""),
        "source_id": n1_def.get("source_id", "madni2018"),
        "claim_ids": n1_cids,
    })

    # Node 2: exposure_retraction
    n2_def = node_defs.get("exposure_retraction", {})
    if retraction_claims:
        n2_verdict = "ok"
        n2_obs = "Gallbladder retraction maintained to expose hepatocystic triangle."
        n2_cids = [c.get("claim_id", "") for c in retraction_claims[:3] if c.get("claim_id")]
    elif raw_claims:
        n2_verdict = "unknown"
        n2_obs = "Retraction technique not explicitly documented in verified claims."
        n2_cids = []
    else:
        n2_verdict = "unknown"
        n2_obs = "Insufficient claim evidence for retraction evaluation."
        n2_cids = []

    scored_nodes.append({
        "node_id": "exposure_retraction",
        "t": 0.0,
        "verdict": n2_verdict,
        "observed": n2_obs,
        "guideline_action": n2_def.get("guideline_action", ""),
        "recommended_alternative": n2_def.get("alternatives", ["Maintain cephalad/lateral traction"])[0],
        "evidence_grade": n2_def.get("evidence_grade", "expert opinion"),
        "source_quote": n2_def.get("quote", ""),
        "source_id": n2_def.get("source_id", "sages2020"),
        "claim_ids": n2_cids,
    })

    # Node 3: dissection_plane_r4u
    n3_def = node_defs.get("dissection_plane_r4u", {})
    if plane_claims:
        n3_verdict = "ok"
        n3_obs = "Dissection maintained within safe anatomical landmarks relative to R4U line."
        n3_cids = [c.get("claim_id", "") for c in plane_claims if c.get("claim_id")]
    elif raw_claims:
        n3_verdict = "ok"  # No deviation observed
        n3_obs = "No dissection below Rouviere sulcus / R4U line noted."
        n3_cids = []
    else:
        n3_verdict = "unknown"
        n3_obs = "Insufficient evidence regarding dissection plane landmarks."
        n3_cids = []

    scored_nodes.append({
        "node_id": "dissection_plane_r4u",
        "t": 0.0,
        "verdict": n3_verdict,
        "observed": n3_obs,
        "guideline_action": n3_def.get("guideline_action", ""),
        "recommended_alternative": n3_def.get("alternatives", ["Dissect above R4U line"])[0],
        "evidence_grade": n3_def.get("evidence_grade", "conditional recommendation, very low certainty"),
        "source_quote": n3_def.get("quote", ""),
        "source_id": n3_def.get("source_id", "tg18"),
        "claim_ids": n3_cids,
    })

    # Node 4: cvs_pause & Node 7: clip_and_divide
    # CVS criterion count:
    cvs_count = len(confirmed_cvs)
    cvs_cids = [c.get("claim_id", "") for c in confirmed_cvs.values() if c.get("claim_id")]
    has_clip = len(clipping_claims) > 0

    n4_def = node_defs.get("cvs_pause", {})
    n7_def = node_defs.get("clip_and_divide", {})

    node_extra_note = None

    # Evidence guard: if no CVS claim survived checking (any polarity), we cannot tell a missing CVS from a
    # missing observation. Report unknown, never blunder, so our own checker's abstentions don't become a flag.
    cvs_assessed = [c for c in claims if c.get("type") == "cvs_criterion"
                    and c.get("verdict") in ("speak", "hedge")]
    if has_clip:
        t_node = clip_t if clip_t is not None else 0.0
        # criteria explicitly assessed as NOT met (negated claims that survived checking)
        negated = {((c.get("entities") or {}).get("cvs_criterion")) for c in cvs_assessed
                   if c.get("polarity") == "negated"}
        all_crit = {"cvs_triangle_cleared", "cvs_lower_gb_off_plate", "cvs_two_structures"}
        unassessed = all_crit - set(confirmed_cvs) - negated
        if not cvs_assessed:
            clip_verdict = "unknown"
            clip_obs = "Clipping seen, but no CVS criterion claim survived checking: evidence insufficient to score."
        elif cvs_count < 3 and unassessed:
            # an unmentioned criterion is not a failed criterion: never flag a safety deviation on silence
            clip_verdict = "unknown"
            clip_obs = (f"Clipping seen; {cvs_count}/3 CVS criteria confirmed, {len(unassessed)} not assessed "
                        f"({', '.join(sorted(unassessed))}): evidence insufficient to score.")
        elif cvs_count == 3:
            clip_verdict = "ok"
            clip_obs = "Definitive pause with all 3 Critical View of Safety criteria confirmed prior to clipping."
        elif cvs_count == 2:
            clip_verdict = "inaccuracy"
            if high_difficulty:
                clip_obs = (
                    f"Clipping performed with 2 of 3 CVS criteria confirmed [high difficulty: {difficulty_grade}]."
                )
                node_extra_note = "high difficulty"
            else:
                clip_obs = "Clipping performed with 2 of 3 CVS criteria confirmed (incomplete CVS pause)."
        else:
            clip_verdict = "blunder"
            clip_obs = f"Clipping performed with only {cvs_count} of 3 CVS criteria confirmed."

        n4_entry = {
            "node_id": "cvs_pause",
            "t": t_node,
            "verdict": clip_verdict,
            "observed": clip_obs,
            "guideline_action": n4_def.get("guideline_action", ""),
            "recommended_alternative": n4_def.get("alternatives", ["Complete all 3 CVS criteria before clipping"])[1 if cvs_count == 2 else 2],
            "evidence_grade": n4_def.get("evidence_grade", "best practice"),
            "source_quote": n4_def.get("quote", ""),
            "source_id": n4_def.get("source_id", "sages_q13"),
            "claim_ids": cvs_cids + clip_claim_ids[:2],
        }
        if node_extra_note:
            n4_entry["note"] = node_extra_note

        n7_entry = {
            "node_id": "clip_and_divide",
            "t": t_node,
            "verdict": clip_verdict,
            "observed": (f"Cystic structures divided with {cvs_count}/3 CVS criteria verified."
                         if clip_verdict != "unknown" else clip_obs),
            "guideline_action": n7_def.get("guideline_action", ""),
            "recommended_alternative": n7_def.get("alternatives", ["Divide only after full CVS verification"])[0],
            "evidence_grade": n7_def.get("evidence_grade", "expert opinion"),
            "source_quote": n7_def.get("quote", ""),
            "source_id": n7_def.get("source_id", "sages_q1"),
            "claim_ids": cvs_cids + clip_claim_ids[:2],
        }
        if node_extra_note:
            n7_entry["note"] = node_extra_note

    else:
        # No clip observed
        if cvs_count == 3:
            n4_verdict = "ok"
            n4_obs = "All 3 CVS criteria verified; structure division not yet reached."
        elif max_t >= time_limit_s:
            n4_verdict = "missed_alternative"
            n4_obs = f"CVS not reached within observation window ({max_t:.0f}s >= {time_limit_s:.0f}s)."
        elif raw_claims:
            n4_verdict = "unknown"
            n4_obs = f"CVS criteria incomplete ({cvs_count}/3 confirmed); clipping phase not reached."
        else:
            n4_verdict = "unknown"
            n4_obs = "Insufficient evidence for CVS evaluation."

        n4_entry = {
            "node_id": "cvs_pause",
            "t": 0.0,
            "verdict": n4_verdict,
            "observed": n4_obs,
            "guideline_action": n4_def.get("guideline_action", ""),
            "recommended_alternative": n4_def.get("alternatives", ["Pause and achieve CVS"])[0],
            "evidence_grade": n4_def.get("evidence_grade", "best practice"),
            "source_quote": n4_def.get("quote", ""),
            "source_id": n4_def.get("source_id", "sages_q13"),
            "claim_ids": cvs_cids,
        }

        n7_entry = {
            "node_id": "clip_and_divide",
            "t": 0.0,
            "verdict": "unknown",
            "observed": "Clipping action not observed in clip claims.",
            "guideline_action": n7_def.get("guideline_action", ""),
            "recommended_alternative": n7_def.get("alternatives", ["Clip only after CVS"])[0],
            "evidence_grade": n7_def.get("evidence_grade", "expert opinion"),
            "source_quote": n7_def.get("quote", ""),
            "source_id": n7_def.get("source_id", "sages_q1"),
            "claim_ids": [],
        }

    scored_nodes.append(n4_entry)

    # Node 5: cvs_not_reached_action
    n5_def = node_defs.get("cvs_not_reached_action", {})
    if cvs_count == 3:
        n5_verdict = "ok"
        n5_obs = "Critical View of Safety successfully attained; escalation or bailout not needed."
        n5_cids = cvs_cids
    elif imaging_help_claims or bailout_claims:
        n5_verdict = "ok"
        n5_obs = "Appropriate escalation (imaging, consultation, or bailout) initiated upon difficult anatomy."
        n5_cids = [c.get("claim_id", "") for c in (imaging_help_claims + bailout_claims) if c.get("claim_id")]
    elif max_t >= time_limit_s or (has_clip and cvs_count <= 1):
        n5_verdict = "missed_alternative"
        n5_obs = "CVS not reached by time threshold without documented imaging, senior consultation, or bailout."
        n5_cids = []
    elif raw_claims:
        n5_verdict = "unknown"
        n5_obs = "Ongoing dissection within standard time limit; no escalation triggered."
        n5_cids = []
    else:
        n5_verdict = "unknown"
        n5_obs = "Insufficient evidence to assess CVS escalation policy."
        n5_cids = []

    scored_nodes.append({
        "node_id": "cvs_not_reached_action",
        "t": clip_t if clip_t is not None else 0.0,
        "verdict": n5_verdict,
        "observed": n5_obs,
        "guideline_action": n5_def.get("guideline_action", ""),
        "recommended_alternative": n5_def.get("alternatives", ["Obtain intraoperative imaging or call for help"])[0],
        "evidence_grade": n5_def.get("evidence_grade", "strong recommendation, very low certainty"),
        "source_quote": n5_def.get("quote", ""),
        "source_id": n5_def.get("source_id", "sages_q4_q17"),
        "claim_ids": n5_cids,
    })

    # Node 6: bailout_choice
    n6_def = node_defs.get("bailout_choice", {})
    if bailout_claims:
        n6_verdict = "ok"
        n6_obs = "Subtotal cholecystectomy or recognized bailout procedure selected."
        n6_cids = [c.get("claim_id", "") for c in bailout_claims if c.get("claim_id")]
    elif cvs_count == 3:
        n6_verdict = "ok"
        n6_obs = "Standard total cholecystectomy proceeded safely following full CVS."
        n6_cids = cvs_cids
    elif max_t >= time_limit_s and not bailout_claims:
        n6_verdict = "missed_alternative"
        n6_obs = "CVS not reached without transition to guideline-recommended subtotal bailout."
        n6_cids = []
    elif raw_claims:
        n6_verdict = "unknown"
        n6_obs = "No bailout indication or action documented."
        n6_cids = []
    else:
        n6_verdict = "unknown"
        n6_obs = "Insufficient evidence for bailout evaluation."
        n6_cids = []

    scored_nodes.append({
        "node_id": "bailout_choice",
        "t": 0.0,
        "verdict": n6_verdict,
        "observed": n6_obs,
        "guideline_action": n6_def.get("guideline_action", ""),
        "recommended_alternative": n6_def.get("alternatives", ["Subtotal cholecystectomy"])[0],
        "evidence_grade": n6_def.get("evidence_grade", "conditional recommendation, low certainty"),
        "source_quote": n6_def.get("quote", ""),
        "source_id": n6_def.get("source_id", "sages_q2_q9"),
        "claim_ids": n6_cids,
    })

    # Add Node 7 (already computed above with Node 4)
    scored_nodes.append(n7_entry)

    # Node 8: gb_bed_haemostasis
    n8_def = node_defs.get("gb_bed_haemostasis", {})
    if haemostasis_claims:
        n8_verdict = "ok"
        n8_obs = "Gallbladder bed haemostasis and fossa inspection observed."
        n8_cids = [c.get("claim_id", "") for c in haemostasis_claims[:3] if c.get("claim_id")]
    elif raw_claims:
        n8_verdict = "unknown"
        n8_obs = "Haemostasis / liver bed inspection phase not yet reached in clip."
        n8_cids = []
    else:
        n8_verdict = "unknown"
        n8_obs = "Insufficient evidence for liver bed haemostasis evaluation."
        n8_cids = []

    scored_nodes.append({
        "node_id": "gb_bed_haemostasis",
        "t": max_t,
        "verdict": n8_verdict,
        "observed": n8_obs,
        "guideline_action": n8_def.get("guideline_action", ""),
        "recommended_alternative": n8_def.get("alternatives", ["Inspect gallbladder fossa"])[0],
        "evidence_grade": n8_def.get("evidence_grade", "expert opinion"),
        "source_quote": n8_def.get("quote", ""),
        "source_id": n8_def.get("source_id", "sages2020"),
        "claim_ids": n8_cids,
    })

    # Return structure compliant with coordination/contracts/narrator_outputs.md
    return {
        "video": video,
        "scored": False,
        "label": "deviation from reference policy",
        "difficulty": {
            "grade": difficulty_grade,
            "claim_ids": difficulty_claim_ids,
        },
        "nodes": scored_nodes,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate post-hoc guideline review from claims.")
    parser.add_argument("--claims", required=True, help="Path to claims.parquet or claims.jsonl")
    parser.add_argument("--out", default=str(C.path("out") / "review.json"), help="Output review.json path")
    parser.add_argument("--video", default="showcase_video", help="Video identifier")
    parser.add_argument("--time-limit", type=float, default=1200.0, help="Dissection time threshold (s)")
    args = parser.parse_args()

    claims_path = Path(args.claims)
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    if claims_path.suffix == ".parquet":
        df = pd.read_parquet(claims_path)
        claims = df.to_dict(orient="records")
    else:
        claims = []
        with open(claims_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    claims.append(json.loads(line))

    res = review(claims, video=args.video, time_limit_s=args.time_limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote review to {out_path} ({len(res['nodes'])} nodes, difficulty: {res['difficulty']['grade']})")


if __name__ == "__main__":
    main()
