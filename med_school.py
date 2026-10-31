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


async def watch(procedure, url):
    gem.trace("flash", f"Watching narrated video {url}")
    return await gem.call(
        f"This is a narrated {procedure}. Act as a student learning to recognise anatomy and instruments "
        "on video. Extract concrete VISUAL rules the surgeon teaches (what a structure looks like, when "
        "it appears) and key timestamped moments.",
        video_url=url, schema=VideoLesson, tag="watch")


async def build_manual(procedure, keys=None, video_urls=()):
    notes, *lessons = await asyncio.gather(research(procedure, keys),
                                           *[watch(procedure, u) for u in video_urls])
    lesson_txt = "\n".join(json.dumps(l) for l in lessons)
    key_rule = (f"Use exactly these keys for structures: {json.dumps(keys)}." if keys else
                "Choose short snake_case keys for 6-12 structures/instruments you think matter most.")
    gem.trace("argon", "Writing labelling manual from research notes and video lessons")
    manual = await gem.call(
        f"Turn the research below into a labelling manual for {procedure} video. {key_rule}\n"
        "Be concrete and visual: another model will use this manual to label single frames.\n\n"
        f"RESEARCH NOTES:\n{notes}\n\nVIDEO LESSONS:\n{lesson_txt}",
        model=C.ARGON, schema=Manual, tag="manual")
    return manual, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", default="laparoscopic cholecystectomy")
    ap.add_argument("--video", action="append", default=[])
    a = ap.parse_args()
    keys = list(C.LABELS) if "cholecystectomy" in a.procedure else None
    manual, notes = asyncio.run(build_manual(a.procedure, keys, a.video))
    (C.OUT_DIR / "manual.json").write_text(json.dumps(manual, indent=2))
    (C.OUT_DIR / "research_notes.md").write_text(notes)
    print(json.dumps(manual, indent=2)[:3000])
    print("\nsaved out/manual.json and out/research_notes.md")


if __name__ == "__main__":
    main()
