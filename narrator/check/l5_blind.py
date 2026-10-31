"""L5 blind visual check layer.

Evaluates observed presence, absence, action, and finding claims directly
against surgical video frames using fast_model. Never passes narration text.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from narrator import data
from narrator import llm
from narrator.check.base import Ctx, res
from narrator.schema import scored_class

NAME = "L5"

SYSTEM_INSTRUCTION = (
    "You are an expert surgical visual verifier evaluating recorded laparoscopic cholecystectomy "
    "video frames for educational review.\n"
    "Key anatomical and instrument definitions:\n"
    "- Grasper: laparoscopic forceps or grasping instrument used to hold or retract tissue.\n"
    "- L-hook / hook: hook-shaped electrocautery instrument with an L-bend at the tip used for dissection/coagulation.\n"
    "- Liver: large reddish-brown organ / liver parenchyma.\n"
    "- Gallbladder: pear-shaped hollow organ situated beneath the liver bed.\n"
    "- Cystic duct: tubular duct connecting gallbladder to common bile duct in Calot's triangle.\n"
    "- Blood: active pooling, oozing blood, or dark blood clot.\n"
    "- Calot's / hepatocystic triangle: triangle bounded by cystic duct, common hepatic duct, and liver edge."
)

PROMPT_TEMPLATE = (
    "{system}\n\n"
    "Question: {question}\n\n"
    "Inspect the surgical frame(s) carefully. "
    "If a second image is provided, it is a focused crop of the suspected region.\n"
    "Answer 'yes' if visible evidence clearly confirms the question, "
    "'no' if clearly not present or contradicted, or 'unsure' if ambiguous.\n"
    "Return JSON matching the schema."
)


class VisualCheckAnswer(BaseModel):
    answer: Literal["yes", "no", "unsure"]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


def verb_to_ing(verb: str) -> str:
    """Convert verb to present participle (-ing) form."""
    v = verb.strip().lower()
    if not v:
        return "operating on"
    if v.endswith("ing"):
        return v
    if v.endswith("e") and not v.endswith("ee"):
        return v[:-1] + "ing"
    if len(v) >= 3 and v[-1] in "ptbmnd" and v[-2] in "aeiou" and v[-3] not in "aeiou":
        return v + v[-1] + "ing"
    return v + "ing"


def build_question(claim: dict) -> tuple[Optional[str], Optional[str]]:
    """Build a yes/no question from the claim ONLY, never passing narration text.

    Returns (question, entity_name) or (None, None).
    """
    ctype = claim.get("type")
    entities = claim.get("entities") or {}
    if hasattr(entities, "model_dump"):
        entities = entities.model_dump()

    target = entities.get("target")
    instrument = entities.get("instrument")
    finding = entities.get("finding")
    verb = entities.get("verb")
    sc = scored_class(claim)

    if ctype in ("presence", "absence"):
        entity = target or instrument or finding or sc
        if not entity:
            return None, None
        display = str(entity).replace("_", " ")
        return f"Is there a {display} visible in this image?", str(entity)

    elif ctype == "action":
        inst = (instrument or "surgical instrument").replace("_", " ")
        v_ing = verb_to_ing(verb or "manipulat")
        tgt = (target or "tissue").replace("_", " ")
        return f"Is a {inst} {v_ing} the {tgt} in this image?", str(instrument or target or "instrument")

    elif ctype == "finding":
        f = (finding or target or sc or "finding").replace("_", " ")
        return f"Is there {f} visible in this image?", str(finding or target or sc or "finding")

    return None, None


def find_frames(clip_id: str, t_start: int, t_end: int) -> list[int]:
    """Find up to 2 frames in [t_start, t_end] that exist on disk."""
    if t_start > t_end:
        t_start, t_end = t_end, t_start

    candidates: list[int] = []
    mid = (t_start + t_end) // 2
    candidates.append(mid)
    if t_start != mid:
        candidates.append(t_start)
    if t_end != mid and t_end != t_start:
        candidates.append(t_end)

    existing: list[int] = []
    for f in candidates:
        p = data.frame_path(clip_id, f)
        if p.is_file() and f not in existing:
            existing.append(f)
            if len(existing) >= 2:
                return existing

    for f in range(t_start, t_end + 1):
        if f in existing:
            continue
        p = data.frame_path(clip_id, f)
        if p.is_file():
            existing.append(f)
            if len(existing) >= 2:
                return existing

    return existing


def get_box_crop(
    ctx: Ctx, clip_id: str, frame_idx: int, entity: Optional[str], sc: Optional[str], frame_path: Path
) -> Optional[bytes]:
    """Return cropped JPEG bytes if a bounding box exists for the entity in ctx.labels."""
    if ctx.labels is None or ctx.labels.empty:
        return None
    try:
        labels_to_match = set()
        if entity:
            labels_to_match.add(str(entity).lower().replace(" ", "_"))
        if sc:
            labels_to_match.add(str(sc).lower())

        sub = ctx.labels[(ctx.labels["clip_id"] == clip_id) & (ctx.labels["frame_idx"] == frame_idx)]
        if sub.empty:
            return None
        sub = sub[sub["label"].astype(str).str.lower().isin(labels_to_match)]
        if sub.empty:
            return None

        for _, row in sub.iterrows():
            raw_boxes = row.get("boxes")
            if not raw_boxes or raw_boxes == "[]":
                continue
            boxes = json.loads(raw_boxes) if isinstance(raw_boxes, str) else raw_boxes
            if boxes and isinstance(boxes, list):
                best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                return llm.jpeg(frame_path, crop=best_box)
    except Exception:
        return None
    return None


def map_answer(ans: str, conf: float, polarity: str, is_absence: bool) -> dict:
    """Map LLM answer and confidence to a pass/fail/hedge result dict."""
    ans = ans.lower().strip()
    is_negated = is_absence or (polarity == "negated")
    is_hedged = (polarity == "hedged")

    if ans == "unsure":
        return res("pass", f"visual check uncertain (confidence {conf:.2f})", hedge=True)

    if not is_negated:
        if ans == "yes":
            return res("pass", f"visual confirmation (confidence {conf:.2f})", hedge=is_hedged)
        else:
            return res("fail", f"entity not observed in frame (confidence {conf:.2f})")
    else:
        if ans == "no":
            return res("pass", f"visual confirmation of absence (confidence {conf:.2f})", hedge=is_hedged)
        else:
            return res("fail", f"entity observed when absence claimed (confidence {conf:.2f})")


async def evaluate_claim(claim: dict, ctx: Ctx) -> dict:
    """Async evaluation of a single visual claim."""
    modality = claim.get("modality")
    ctype = claim.get("type")
    if modality != "observed":
        return res("abstain", f"non-visual modality '{modality}'")
    if ctype not in ("presence", "absence", "action", "finding"):
        return res("abstain", f"non-visual claim type '{ctype}'")

    question, entity = build_question(claim)
    if not question:
        return res("abstain", "unable to build question from visual entities")

    clip_id = claim.get("clip_id", "")
    t_start = int(claim.get("t_start", 0))
    t_end = int(claim.get("t_end", t_start))

    frame_indices = find_frames(clip_id, t_start, t_end)
    if not frame_indices:
        return res("abstain", f"no frame on disk for {clip_id} in [{t_start}, {t_end}]")

    primary_f = frame_indices[0]
    primary_path = data.frame_path(clip_id, primary_f)
    if not primary_path.is_file():
        return res("abstain", f"frame file missing: {primary_path}")

    sc = scored_class(claim)
    crop_bytes = get_box_crop(ctx, clip_id, primary_f, entity, sc, primary_path)

    images: list[bytes] = []
    try:
        images.append(llm.jpeg(primary_path))
        if crop_bytes is not None:
            images.append(crop_bytes)
        elif len(frame_indices) > 1:
            second_path = data.frame_path(clip_id, frame_indices[1])
            if second_path.is_file():
                images.append(llm.jpeg(second_path))
    except Exception as e:
        return res("abstain", f"image loading failed: {str(e)[:80]}")

    prompt = PROMPT_TEMPLATE.format(system=SYSTEM_INSTRUCTION, question=question)

    out = await llm.generate(
        prompt=prompt,
        model="fast_model",
        images=images,
        schema=VisualCheckAnswer,
        tag="l5",
    )
    if out is None:
        return res("abstain", "llm call failed")

    ans = str(out.get("answer", "unsure"))
    conf = float(out.get("confidence", 1.0))
    polarity = claim.get("polarity", "asserted")
    is_abs = (ctype == "absence")
    return map_answer(ans, conf, polarity, is_abs)


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L5 blind visual check over claims in batch."""
    if not claims:
        return []

    coros = [evaluate_claim(c, ctx) for c in claims]
    results = llm.gather(coros, desc="L5")

    clean_results: list[dict] = []
    for r in results:
        if r is None or not isinstance(r, dict):
            clean_results.append(res("abstain", "claim execution error"))
        else:
            clean_results.append(r)

    ctx.extra["L5"] = clean_results
    return clean_results
