"""Step 3: 'medical school'. Argon researches a procedure with Google Search, optionally watches
narrated YouTube videos, and writes its own labelling manual (machine-readable JSON).

  python med_school.py                                   # cholecystectomy manual for our 6 labels
  python med_school.py --video https://youtube.com/...   # add narrated video(s) as teaching material
"""
import argparse
import asyncio
import json

from pydantic import BaseModel

import config as C
import gem


DEFAULT_LAP_CHOLE_VIDEO = "https://www.youtube.com/watch?v=PNydeaI2WIU"


class Cue(BaseModel):
    key: str
    name: str
    appearance: str          # colour, texture, shape on laparoscopic video
    where_found: str         # typical location in the frame / relative to other anatomy
    when_visible: str        # which phases it tends to appear in
    confused_with: list[str]
    how_to_tell_apart: str


class Manual(BaseModel):
    procedure: str
    phases: list[str]
    structures: list[Cue]
    critical_events: list[str]
    general_rules: list[str]


class VideoLesson(BaseModel):
    rules: list[str]                 # visual rules learned from the narration
    moments: list[str]               # "mm:ss - what the surgeon said / showed"


def validate_manual(manual: dict, keys: list[str] | None = None) -> dict:
    """Validate Manual schema and verify all requested label keys have substantive visual criteria."""
    parsed = Manual.model_validate(manual)
    by_key = {s.key: s for s in parsed.structures}
    if keys:
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise ValueError(f"Manual missing required structure keys: {missing}")
        # Preserve canonical key order for requested keys
        ordered = [by_key[k] for k in keys] + [s for s in parsed.structures if s.key not in set(keys)]
        parsed.structures = ordered
    for s in parsed.structures:
        if not s.appearance.strip() or not s.where_found.strip() or not s.how_to_tell_apart.strip():
            raise ValueError(f"Structure '{s.key}' has empty visual criteria fields")
        if not s.confused_with:
            raise ValueError(f"Structure '{s.key}' must list at least one confuser in confused_with")
    if not parsed.phases or not parsed.critical_events or not parsed.general_rules:
        raise ValueError("Manual must include non-empty phases, critical_events, and general_rules")
    return parsed.model_dump()


async def research(procedure, keys):
    want = (f"Focus on these structures/instruments: {', '.join(keys)}." if keys else
            "Decide yourself which anatomical structures, instruments and events matter most.")
    gem.trace("argon", f"Researching '{procedure}' with Google Search")
    notes = await gem.call(
        f"You are preparing to label laparoscopic video of a {procedure}. Use Google Search to find "
        f"authoritative sources (surgical society guidance, anatomy references, papers). {want}\n"
        "Write dense notes on: the phases in order; for each structure/instrument what it LOOKS like "
        "on endoscopic video (colour, texture, shape, typical position), when it is visible, what it is "
        "commonly confused with and how to tell apart; critical safety events (e.g. critical view of "
        "safety). Cite sources inline.",
        model=C.ARGON, tools=gem.SEARCH, tag="research")
    return notes


async def watch(procedure, url, keys=None):
    target_hint = (f" Pay special attention to how these target classes look on camera: {', '.join(keys)}."
                   if keys else "")
    gem.trace("flash", f"Watching narrated video {url}")
    return await gem.call(
        f"This is a narrated {procedure}.{target_hint} Act as a student learning to recognise anatomy "
        "and instruments on single endoscopic frames. Extract concrete VISUAL recognition rules the "
        "surgeon teaches or demonstrates (colour, texture, specular highlights, entry direction, when "
        "a structure is exposed vs hidden by fat/peritoneum, and how to distinguish lookalikes) and "
        "key timestamped visual moments.",
        video_url=url, schema=VideoLesson, tag="watch")


async def build_manual(procedure, keys=None, video_urls=()):
    notes, *lessons = await asyncio.gather(research(procedure, keys),
                                           *[watch(procedure, u, keys=keys) for u in video_urls])
    lesson_txt = "\n".join(json.dumps(l) for l in lessons)
    key_rule = (f"Use2 EXACTLY these keys for structures in this exact order: {json.dumps(keys)}." if keys else
                "Choose short snake_case keys for 6-12 structures/instruments you think matter most.")
    key_rule = key_rule.replace("Use2 ", "Use ")
    gem.trace("argon", "Writing labelling manual from research notes and video lessons")
    manual = await gem.call(
        f"Turn the research and video lessons below into a machine-readable visual labelling manual for "
        f"{procedure} video. {key_rule}\n"
        "Be concrete and visual: another vision model will use this manual to decide present/absent (0/1) "
        "and draw tight 2D bounding boxes on SINGLE static frames.\n"
        "Requirements:\n"
        "1. For each structure in `structures`, give detailed single-frame visual criteria in `appearance` "
        "(colour, texture, lighting/specular reflections, shape/geometry), `where_found` (quadrant, port "
        "entry angle, anatomical neighbours), `when_visible` (phases and anatomical exposure preconditions), "
        "`confused_with` (specific visual lookalikes on endoscopic frames), and `how_to_tell_apart` "
        "(strict positive vs negative visual rules so the labeller avoids false positives and false negatives).\n"
        "2. In `general_rules`, write concrete SINGLE-FRAME LABELLING DECISION RULES for the frame labeller "
        "(e.g., checking frame periphery/edges for partially visible instrument shafts/jaws or liver border; "
        "distinguishing `grasper` hinged/fenestrated jaws from `l_hook` 90-degree wire hook tip; marking "
        "`cystic_duct` present ONLY when a distinct dissected tubular duct stub is visibly exposed below the "
        "gallbladder infundibulum and NOT when buried in undissected Calot's fat; marking `blood` present "
        "ONLY when distinct free liquid red blood or dark pooled blood clot is visible and NOT normal "
        "reddish-brown liver parenchyma, pink serosa, or dry cautery char).\n\n"
        f"RESEARCH NOTES:\n{notes}\n\nVIDEO LESSONS:\n{lesson_txt}",
        model=C.ARGON, schema=Manual, tag="manual")
    manual = validate_manual(manual, keys)
    return manual, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", default="laparoscopic cholecystectomy")
    ap.add_argument("--video", action="append", default=[])
    a = ap.parse_args()
    is_chole = "cholecystectomy" in a.procedure.lower()
    keys = list(C.LABELS) if is_chole else None
    videos = a.video or ([DEFAULT_LAP_CHOLE_VIDEO] if is_chole else [])
    manual, notes = asyncio.run(build_manual(a.procedure, keys, videos))
    (C.OUT_DIR / "manual.json").write_text(json.dumps(manual, indent=2))
    (C.OUT_DIR / "research_notes.md").write_text(notes)
    coord_dir = C.ROOT / "coord" / "artifacts"
    if is_chole and coord_dir.is_dir():
        (coord_dir / "manual.json").write_text(json.dumps(manual, indent=2))
    print(json.dumps(manual, indent=2)[:3000])
    print("\nsaved out/manual.json and out/research_notes.md")


if __name__ == "__main__":
    main()
