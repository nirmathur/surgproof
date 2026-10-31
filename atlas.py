"""Act 1: any surgery, zero setup. Procedure name + a YouTube video -> machine-readable dataset.
No ground truth here: this is the showcase; Act 2 (CholecSeg8k) is the proof.

  python atlas.py --procedure "laparoscopic appendectomy" --video https://www.youtube.com/watch?v=...
"""
import argparse
import asyncio
import json
import re
import time

from pydantic import BaseModel

import config as C
import gem
from med_school import build_manual


class Phase(BaseModel):
    name: str
    start: str      # mm:ss
    end: str
    evidence: str
    confidence: float


class Event(BaseModel):
    time: str
    event: str
    evidence: str
    confidence: float


class Sighting(BaseModel):
    time: str
    key: str
    confidence: float


class Timeline(BaseModel):
    phases: list[Phase]
    events: list[Event]
    sightings: list[Sighting]


class Audit(BaseModel):
    problems: list[str]            # impossible orderings, missing prerequisites, contradictions
    unverified_items: list[str]    # e.g. "event@12:30" or "phase:Dissection"
    overall_confidence: float


async def main(procedure, video):
    slug = re.sub(r"[^a-z0-9]+", "-", procedure.lower()).strip("-")
    out = C.OUT_DIR / "atlas" / slug
    out.mkdir(parents=True, exist_ok=True)

    manual, notes = await build_manual(procedure, None, [video])
    gem.trace("flash", f"Structuring video into phases, events and sightings for {procedure}")
    tl = await gem.call(
        f"Using this labelling manual you wrote, structure the {procedure} video into a timeline: "
        "phases (mm:ss start/end), key events (especially safety-critical ones), and sightings of each "
        "structure key. Give confidence honestly.\nManual: " + json.dumps(manual),
        video_url=video, schema=Timeline, tag="timeline")
    gem.trace("argon", "Auditing the timeline for impossible or unsupported claims")
    audit = await gem.call(
        f"Audit this machine-generated timeline of a {procedure} against surgical logic and the manual. "
        "Flag impossible phase orderings, events before their prerequisites, and low-evidence claims.\n"
        f"Manual: {json.dumps(manual)}\nTimeline: {json.dumps(tl)}",
        model=C.ARGON, schema=Audit, tag="audit")

    dataset = {"procedure": procedure, "source_video": video, "created": time.strftime("%Y-%m-%d %H:%M"),
               "schema": {"structures": [s["key"] for s in manual["structures"]], "phases": manual["phases"]},
               "timeline": tl, "audit": audit, "human_labels_used": 0}
    (out / "dataset.json").write_text(json.dumps(dataset, indent=2))
    (out / "manual.json").write_text(json.dumps(manual, indent=2))
    (out / "research_notes.md").write_text(notes)
    card = (f"# Dataset card: {procedure}\n\nGenerated autonomously from public sources and one public video. "
            f"Human labels used: 0.\n\n- Source: {video}\n- Phases: {len(tl['phases'])}\n"
            f"- Events: {len(tl['events'])}\n- Structure sightings: {len(tl['sightings'])}\n"
            f"- Audit confidence: {audit['overall_confidence']:.2f}\n- Flagged problems: {len(audit['problems'])}\n\n"
            "Not validated against expert ground truth. See the CholecSeg8k evaluation for measured accuracy.\n")
    (out / "DATASET_CARD.md").write_text(card)
    print(card)
    print(f"saved to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedure", required=True)
    ap.add_argument("--video", required=True)
    a = ap.parse_args()
    asyncio.run(main(a.procedure, a.video))
