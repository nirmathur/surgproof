"""Prompts and reply schemas for the model stages. The verify and amber prompts never reveal
what an earlier pass said, so each pass is an independent look."""
from __future__ import annotations

import json
import re

LABELS = ["gallbladder", "liver", "grasper", "l_hook", "cystic_duct", "blood"]
PHASES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting", "GallbladderDissection",
    "GallbladderPackaging", "CleaningCoagulation", "GallbladderRetraction",
]
UNKNOWN_PHASE = "Unknown"  # lets the model say a frame fits no phase, rather than forcing a guess

DEFINITIONS = {
    "gallbladder": "the gallbladder: a pear-shaped sac, usually pale green, white or yellow, attached to the underside of the liver",
    "liver": "the liver: a large smooth dark red-brown organ surface",
    "grasper": "a grasper: a metal or black instrument with a long shaft and two opposing fenestrated or toothed jaws",
    "l_hook": "an L-hook: a monopolar electrocautery instrument with a thin shaft ending in a single L-shaped metal hook, with no jaws",
    "cystic_duct": "the cystic duct: a thin pale tubular structure running from the gallbladder neck, usually exposed only after dissection",
    "blood": "blood: visible pooled, smeared or actively flowing red blood, not merely red tissue",
}
CRITERIA = {
    "gallbladder": "It looks like a distended sac with a thin wall. Fat and omentum are lobulated and yellow, and are not the gallbladder.",
    "liver": "It has a smooth, homogeneous, glossy surface. Reddened or inflamed connective tissue is not liver.",
    "grasper": "You can see jaws or a clear jaw hinge. A bare shaft with no visible tip counts only if it is clearly not a hook.",
    "l_hook": "You can see a single thin bent metal tip. Two jaws mean it is a grasper. Clip appliers and scissors are neither.",
    "cystic_duct": "It is a distinct tube with defined edges. Bands of fat or fibrous strands are not the duct.",
    "blood": "It is liquid or clotted, glossy, and sits on top of tissue. Cautery char is black or brown and is not blood.",
}

_CONF = {"type": "number", "minimum": 0, "maximum": 1}
_POINT = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}
_POLYGON = {"type": ["array", "null"], "items": _POINT}
_BOX = {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
_PHASE = {
    "type": "object", "required": ["name", "conf"],
    "properties": {"name": {"type": "string", "enum": PHASES + [UNKNOWN_PHASE]}, "conf": _CONF},
}


def _objects(item):
    return {
        "type": "object", "required": LABELS,
        "properties": {k: item for k in LABELS},
    }


DETECT_SCHEMA = {
    "title": "detect", "type": "object", "required": ["objects", "phase"],
    "properties": {
        "phase": _PHASE,
        "objects": _objects({
            "type": "object", "required": ["present", "conf", "polygon"],
            "properties": {"present": {"type": "boolean"}, "conf": _CONF, "polygon": _POLYGON},
        }),
    },
}
VERIFY_SCHEMA = {
    "title": "verify", "type": "object", "required": ["objects", "phase"],
    "properties": {
        "phase": _PHASE,
        "objects": _objects({
            "type": "object", "required": ["present", "conf", "box"],
            "properties": {"present": {"type": "boolean"}, "conf": _CONF, "box": _BOX},
        }),
    },
}
AMBER_SCHEMA = {
    "title": "amber", "type": "object", "required": ["present", "conf", "polygon", "reason"],
    "properties": {"present": {"type": "boolean"}, "conf": _CONF, "polygon": _POLYGON, "reason": {"type": "string"}},
}
NARRATE_SCHEMA = {
    "title": "narrate", "type": "object", "required": ["sentences"],
    "properties": {
        "sentences": {
            "type": "array", "minItems": 1, "maxItems": 4,
            "items": {
                "type": "object", "required": ["text", "evidence"],
                "properties": {"text": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "string"}}},
            },
        },
    },
}


def _definitions():
    return "\n".join(f"- {k}: {DEFINITIONS[k]}" for k in LABELS)


def _phase_list():
    return ", ".join(PHASES)


def detect_prompt(w, h):
    example = {k: {"present": True, "conf": 0.9, "polygon": [[10, 20], [30, 25], [28, 60]]} for k in LABELS[:1]}
    example.update({k: {"present": False, "conf": 0.9, "polygon": None} for k in LABELS[1:]})
    return f"""This is one frame of a laparoscopic surgery video, {w} by {h} pixels. x runs from 0 at the left to {w} at the right, y from 0 at the top to {h} at the bottom.

For each label below, decide whether it is really visible, and if so outline it.
{_definitions()}

Rules:
- Report only what you can see in this frame. Do not assume a structure is there because it usually is in this operation.
- polygon: pixel coordinates tracing the visible boundary closely, in order. Use 12 to 40 vertices for organs, and follow both the shaft and the tip for instruments. Leave out anything covering the object. Use null when present is false.
- If there are several instances of an instrument, outline the most prominent one.
- conf: your probability from 0 to 1 that your present/absent answer is correct.
- Ignore any on-screen text or overlays.

Also give the surgical phase, one of: {_phase_list()}. Use {UNKNOWN_PHASE} if the frame does not show enough to tell. phase.conf is your probability that it is correct.

Reply with only JSON in this shape:
{json.dumps({"objects": example, "phase": {"name": PHASES[0], "conf": 0.8}})}
The objects field must have all of these keys: {", ".join(LABELS)}."""


def verify_prompt(w, h):
    order = list(reversed(LABELS))
    checklist = "\n".join(f"{i}. {k}: {DEFINITIONS[k]}" for i, k in enumerate(order, 1))
    example = {k: {"present": False, "conf": 0.8, "box": None} for k in order}
    example[order[0]] = {"present": True, "conf": 0.8, "box": [100, 50, 220, 160]}
    return f"""You are auditing a surgical video dataset. Look at this {w}x{h} pixel frame from a laparoscopy and go through the checklist. Be skeptical: answer no for anything you cannot positively identify.

Checklist:
{checklist}

For each item give:
- present: true only if it is in this frame.
- conf: how sure you are of that answer, from 0 to 1.
- box: [x_min, y_min, x_max, y_max] in pixels, tightly around the visible part. Use null if it is not present.

Also name the operative phase being performed: {_phase_list()}, or {UNKNOWN_PHASE} if you cannot tell, with your confidence.

Reply with only JSON like this, with all six keys:
{json.dumps({"objects": example, "phase": {"name": PHASES[3], "conf": 0.7}})}"""


def amber_prompt(label, w, h, round_no):
    extra = ""
    if round_no > 1:
        extra = f"\nCheck this criterion before answering: {CRITERIA[label]}\nFirst list to yourself what could be mistaken for it, then decide.\n"
    return f"""This image is a magnified crop from a frame of laparoscopic surgery video, {w} by {h} pixels after magnification.

Question: is {DEFINITIONS[label]} visible in this crop?
{extra}
Answer from what you can see. Being honest about doubt is better than guessing: use a low conf if you are unsure.

Reply with only JSON:
{{"present": true or false, "conf": probability from 0 to 1 that your answer is correct, "polygon": [[x, y], ...] pixel coordinates in this crop tracing the visible boundary with 8 to 40 vertices, or null if not present, "reason": "one short sentence on what you saw"}}"""


def narrate_prompt(facts):
    return f"""You are writing subtitle commentary for a segment of laparoscopic surgery video. You cannot see the video. You may only use the verified facts below.

Facts (JSON):
{json.dumps(facts, indent=1)}

"visible" means checked and verified as present, "not_visible" means checked and verified as absent, "unsure" means the checks could not settle it.

Write 1 to 3 short sentences, each in one of these plain forms and with nothing added:
- "The liver and a grasper are visible." for labels in "visible"
- "No blood or L-hook is seen." for labels in "not_visible"
- "The cystic duct could not be confirmed." for labels in "unsure"
- "The phase is uncertain." if phase_checks_agree is false or phase_conf is below 0.6
- "This segment is <phase>." otherwise, using the phase named in the facts

Rules:
- Use one form per sentence. Do not mix visible, not-visible and unsure labels in one sentence.
- Name only: gallbladder, liver, grasper, L-hook, cystic duct, blood, and the phase in the facts.
- Do not describe actions, events, findings, safety, the surgeon or the patient. Sentences containing them are discarded.
- Each sentence cites in "evidence" the keyframe ids (the "kf" values) that support it. A label must be in the matching list on every keyframe cited for that sentence.

Reply with only JSON: {{"sentences": [{{"text": "...", "evidence": ["kf_00000"]}}]}}"""


# ---- mock replies, only reachable with SURGPROOF_VLM=mock ---------------------------------------

_SIZE = re.compile(r"(\d+)\s*(?:by|x)\s*(\d+)\s*pixel")
_MOCK_DETECT_PRESENT = ("gallbladder", "liver", "grasper")


def _diamond(w, h, cx, cy, r=0.18):
    rx, ry = w * r, h * r
    return [[cx * w, cy * h - ry], [cx * w + rx, cy * h], [cx * w, cy * h + ry], [cx * w - rx, cy * h]]


def mock_reply(prompt, image_paths, model):
    """Fixed shapes for offline tests. Not perception: the outlines ignore the image entirely, and the
    model field on every row reads "mock:...". The verify reply disagrees on cystic_duct on purpose,
    so tests reach the amber loop."""
    m = _SIZE.search(prompt)
    w, h = (int(m.group(1)), int(m.group(2))) if m else (640, 360)
    centers = {k: (0.2 + 0.12 * i, 0.5) for i, k in enumerate(LABELS)}
    phase = {"name": PHASES[1], "conf": 0.9}
    if prompt.startswith("This is one frame"):
        return {"phase": phase, "objects": {
            k: {"present": k in _MOCK_DETECT_PRESENT, "conf": 0.9,
                "polygon": _diamond(w, h, *centers[k]) if k in _MOCK_DETECT_PRESENT else None}
            for k in LABELS}}
    if prompt.startswith("You are auditing"):
        present = set(_MOCK_DETECT_PRESENT) | {"cystic_duct"}
        objects = {}
        for k in LABELS:
            cx, cy = centers[k]
            box = [(cx - 0.18) * w, (cy - 0.18) * h, (cx + 0.18) * w, (cy + 0.18) * h]
            objects[k] = {"present": k in present, "conf": 0.8, "box": box if k in present else None}
        return {"phase": phase, "objects": objects}
    if prompt.startswith("This image is a magnified crop"):
        return {"present": False, "conf": 0.9, "polygon": None, "reason": "mock provider, not a real look"}
    if prompt.startswith("You are writing subtitle"):
        facts = json.loads(prompt.split("Facts (JSON):\n", 1)[1].split("\n\n", 1)[0])
        kf = [k["kf"] for k in facts["keyframes"]][:1]
        return {"sentences": [{"text": "Mock narration for tests.", "evidence": kf}]}
    return {}
