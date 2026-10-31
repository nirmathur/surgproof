"""Step 3: 'medical school'. Argon researches a procedure with Google Search, optionally watches
narrated YouTube videos, and writes its own labelling manual (machine-readable JSON).

  python med_school.py                                   # cholecystectomy manual for our 6 labels
  python med_school.py --video https://youtube.com/...   # add narrated video(s) as teaching material
"""
import argparse
import asyncio
import json

from pydantic import BaseModel, Field

import config as C
import gem


DEFAULT_LAP_CHOLE_VIDEO = "https://www.youtube.com/watch?v=PNydeaI2WIU"


class Source(BaseModel):
    url: str
    title: str
    source_type: str = "video"
    is_real_intraop_footage: bool = False
    real_intraop_footage_note: str = ""
    timestamps: list[str] = Field(default_factory=list)
    structure_timestamps: dict[str, list[str]] = Field(default_factory=dict)


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
    sources: list[Source] = Field(default_factory=list)
    phases: list[str]
    structures: list[Cue]
    critical_events: list[str]
    general_rules: list[str]


class _ManualResponseSchema(BaseModel):
    """Gemini Developer API response_schema (avoids dict additionalProperties in Source)."""
    procedure: str
    phases: list[str]
    structures: list[Cue]
    critical_events: list[str]
    general_rules: list[str]


_ManualResponseSchema.__name__ = "Manual"


class VideoLesson(BaseModel):
    rules: list[str]                 # visual rules learned from the narration
    moments: list[str]               # "mm:ss - what the surgeon said / showed"


DEFAULT_LAP_CHOLE_SOURCES: list[dict] = [
    {
        "url": "https://www.youtube.com/watch?v=PNydeaI2WIU",
        "title": "Ellie Scrubs In: Laparoscopic Cholecystectomy",
        "source_type": "video",
        "is_real_intraop_footage": False,
        "real_intraop_footage_note": (
            "No continuous dynamic intra-operative video footage. This video is a narrated surgical "
            "clerkship lecture presented over a single static intra-operative laparoscopic still "
            "photograph (displayed 00:00-04:25) showing the liver, gallbladder retracted by two "
            "laparoscopic graspers, and Calot's triangle region; dynamic phase transitions and "
            "L-hook electrocautery visual criteria come from the voiceover narration timestamps "
            "and the text sources below."
        ),
        "timestamps": [
            "00:00-04:25 - Static intra-operative laparoscopic photograph showing liver, gallbladder under cephalad/lateral retraction by two graspers, and adjacent hepatocystic fat",
            "00:23 - Step 1: Patient positioning and operating room setup",
            "00:51 - Step 2: Peritoneal access, insufflation, and four-port placement",
            "01:37-01:59 - Step 3: Diagnostic laparoscopy and fundic/infundibular grasper retraction exposing Calot's triangle",
            "02:05-02:40 - Step 4: Electrocautery dissection of Calot's triangle, Critical View of Safety isolation of cystic duct and cystic artery, clipping and division",
            "02:47-03:08 - Step 5: Separation of gallbladder from the hepatic bed using hook cautery (02:56), electrocautery hemostasis (03:00), and umbilical bag retrieval",
            "03:20-03:27 - Step 6: Final inspection of liver bed and clipped ductal/arterial stumps for bleeding or bile leak, and port closure",
        ],
        "structure_timestamps": {
            "grasper": [
                "00:00-04:25 (visual: two metallic laparoscopic graspers visible in the static intra-operative photograph holding the gallbladder fundus and infundibulum)",
                "01:48-01:54 (narration: retracting gallbladder fundus superiorly and laterally using a grasper via lateral port)",
                "01:54-01:59 (narration: retracting infundibulum medially and inferiorly with right-hand grasper to open Calot's triangle)",
            ],
            "l_hook": [
                "02:06-02:11 (narration: electrocautery dissection of peritoneum and fibrofatty tissue in Calot's triangle; not visually depicted in the static frame)",
                "02:54-03:00 (narration: dissecting gallbladder off the liver bed using hook cautery at 02:56; detailed single-frame hook geometry from text sources)",
            ],
            "liver": [
                "00:00-04:25 (visual: dark reddish-brown hepatic parenchyma with glistening Glisson's capsule specular reflections in the static intra-operative photograph)",
                "02:52-02:56 (narration: dissecting gallbladder from the hepatic/liver bed)",
                "03:20-03:25 (narration: inspecting the liver bed for bleeding or bile leak)",
            ],
            "gallbladder": [
                "00:00-04:25 (visual: distended pear-shaped gallbladder serosa held under traction beneath the liver edge in the static intra-operative photograph)",
                "00:06-00:09, 00:48-00:51 (narration: overview and initial exposure moving bowel away from gallbladder)",
                "01:48-01:59 (narration: cephalad fundic and inferolateral infundibular retraction)",
                "02:22-02:26, 02:47-03:16 (narration: two structures entering gallbladder in CVS, retrograde removal from liver bed, and umbilical retrieval)",
            ],
            "cystic_duct": [
                "00:00-04:25 (visual: infundibular neck region leading into Calot's triangle in the static intra-operative photograph)",
                "02:08-02:27 (narration: skeletonization and isolation of the cystic duct and cystic artery in Calot's triangle to achieve the Critical View of Safety)",
                "02:36-02:40 (narration: clipping and dividing the isolated cystic duct)",
            ],
            "blood": [
                "00:00-04:25 (visual: minor red vascular/peritoneal surface staining along the hepatic bed and pericholecystic tissue in the static photograph)",
                "03:00-03:04 (narration: controlling liver bed bleeding with electrocautery or hemostatic agents)",
                "03:20-03:25 (narration: final inspection of the liver bed and clip sites for active bleeding)",
            ],
        },
    },
    {
        "url": "https://arxiv.org/abs/2012.12453",
        "title": "CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80 (Hong et al., 2020)",
        "source_type": "text",
        "is_real_intraop_footage": True,
        "real_intraop_footage_note": "Text & benchmark reference defining pixel-level visual class boundaries on real intra-operative Cholec80 laparoscopic cholecystectomy frames.",
        "timestamps": [],
        "structure_timestamps": {
            "grasper": ["Section 2 / Table 1: Laparoscopic grasper shaft and hinged jaw class definition"],
            "l_hook": ["Section 2 / Table 1: L-hook electrocautery instrument class definition"],
            "liver": ["Section 2 / Table 1: Liver parenchyma and Glisson's capsule class definition"],
            "gallbladder": ["Section 2 / Table 1: Gallbladder sac and infundibulum class definition (excluding cystic duct)"],
            "cystic_duct": ["Section 2 / Table 1: Isolated tubular cystic duct class definition"],
            "blood": ["Section 2 / Table 1: Free extravasated blood / clot class definition"],
        },
    },
    {
        "url": "https://arxiv.org/abs/2307.10926",
        "title": "Endoscapes: A Dataset for Critical View of Safety Assessment and Hepatocystic Anatomy Segmentation in Laparoscopic Cholecystectomy (Murali et al., MICCAI 2023)",
        "source_type": "text",
        "is_real_intraop_footage": True,
        "real_intraop_footage_note": "Text & benchmark reference for hepatocystic anatomy and Critical View of Safety criteria on real laparoscopic video.",
        "timestamps": [],
        "structure_timestamps": {
            "liver": ["Hepatocystic window background liver (Segment V) and cystic plate exposure criteria"],
            "gallbladder": ["Lower one-third gallbladder detachment from cystic plate"],
            "cystic_duct": ["Skeletonized tubular cystic duct vs cystic artery isolation in Calot's triangle"],
        },
    },
    {
        "url": "https://www.sages.org/safe-cholecystectomy-program/",
        "title": "SAGES Safe Cholecystectomy Program — Strategies for Minimizing Bile Duct Injuries",
        "source_type": "text",
        "is_real_intraop_footage": False,
        "real_intraop_footage_note": "Clinical surgical society guideline text defining the 3-part Critical View of Safety (CVS) checklist and R4U line / Rouviere's sulcus landmarks.",
        "timestamps": [],
        "structure_timestamps": {
            "cystic_duct": ["CVS Criterion 1 & 3: Hepatocystic triangle fat clearance and two-structure-only duct/artery confirmation"],
            "gallbladder": ["CVS Criterion 2: Lower third dissected off the fibrous cystic plate"],
            "liver": ["Rouviere's sulcus and Segment IV/V R4U safety line landmarks"],
        },
    },
    {
        "url": "https://pubmed.ncbi.nlm.nih.gov/7818839/",
        "title": "An Analysis of the Problem of Biliary Injury During Laparoscopic Cholecystectomy — Critical View of Safety (Strasberg et al., J Am Coll Surg 1995; Strasberg & Brunt, Ann Surg 2017)",
        "source_type": "text",
        "is_real_intraop_footage": False,
        "real_intraop_footage_note": "Peer-reviewed surgical literature establishing the Critical View of Safety and the infundibular ('tunnel') error trap.",
        "timestamps": [],
        "structure_timestamps": {
            "cystic_duct": ["Differentiation of cystic duct (2-3 mm, beaded valves of Heister) from common bile duct (CBD) and cystic artery"],
        },
    },
    {
        "url": "https://ieeexplore.ieee.org/document/7519080",
        "title": "EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos / Cholec80 (Twinanda et al., IEEE TMI 2017)",
        "source_type": "text",
        "is_real_intraop_footage": True,
        "real_intraop_footage_note": "Benchmark paper defining the 7 ordered surgical phases and tool presence annotations on 80 real laparoscopic cholecystectomy videos.",
        "timestamps": [],
        "structure_timestamps": {
            "grasper": ["Phases 1-7: Retraction and tissue handling across all surgical phases"],
            "l_hook": ["Phases 2, 4, 6: Electrocautery dissection of Calot's triangle and liver bed"],
        },
    },
    {
        "url": "https://pubmed.ncbi.nlm.nih.gov/29090866/",
        "title": "Tokyo Guidelines 2018 (TG18): Safe Steps in Laparoscopic Cholecystectomy for Acute Cholecystitis (Wakabayashi et al., J Hepatobiliary Pancreat Sci 2018)",
        "source_type": "text",
        "is_real_intraop_footage": False,
        "real_intraop_footage_note": "International consensus guideline text on acute cholecystitis visual appearance, subserosal cystic plate dissection, and hemorrhage/thermal injury avoidance.",
        "timestamps": [],
        "structure_timestamps": {
            "gallbladder": ["Edematous, thick-walled, or beefy-red/violaceous serosal appearance in acute/chronic inflammation"],
            "blood": ["Middle hepatic vein tributary oozing when breaching the fibrous cystic plate and thermal/clipping hazards"],
        },
    },
]


def build_sources_list(procedure: str, keys: list[str] | None, video_urls, lessons: list[dict]) -> list[dict]:
    """Build structured provenance entries for video and text sources used to write the manual."""
    is_chole = "cholecystectomy" in procedure.lower()
    urls = list(video_urls or [])
    if is_chole and (not urls or urls == [DEFAULT_LAP_CHOLE_VIDEO]):
        return [Source.model_validate(s).model_dump() for s in DEFAULT_LAP_CHOLE_SOURCES]

    out: list[dict] = []
    for idx, u in enumerate(urls):
        if u == DEFAULT_LAP_CHOLE_VIDEO:
            out.append(Source.model_validate(DEFAULT_LAP_CHOLE_SOURCES[0]).model_dump())
            continue
        lesson = lessons[idx] if idx < len(lessons) and isinstance(lessons[idx], dict) else {}
        moments = [str(m) for m in lesson.get("moments", []) if m]
        struct_ts: dict[str, list[str]] = {}
        for k in (keys or []):
            k_terms = [k.lower(), k.lower().replace("_", " ")]
            if k in C.LABELS:
                k_terms.append(C.LABELS[k][1].lower().split(" (")[0])
            matched = [m for m in moments if any(t in m.lower() for t in k_terms)]
            if matched:
                struct_ts[k] = matched
        out.append(
            Source(
                url=u,
                title=f"Narrated {procedure} teaching video",
                source_type="video",
                is_real_intraop_footage=True,
                real_intraop_footage_note="Narrated surgical teaching video processed by Gemini video analysis.",
                timestamps=moments,
                structure_timestamps=struct_ts,
            ).model_dump()
        )
    if is_chole:
        for s in DEFAULT_LAP_CHOLE_SOURCES[1:]:
            out.append(Source.model_validate(s).model_dump())
    return out


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
        model=C.ARGON, schema=_ManualResponseSchema, tag="manual")
    if isinstance(manual, dict) and not manual.get("sources"):
        manual = {
            "procedure": manual.get("procedure", procedure),
            "sources": build_sources_list(procedure, [s.get("key", "") for s in manual.get("structures", []) if isinstance(s, dict)] or keys, video_urls, lessons),
            **{k: v for k, v in manual.items() if k not in ("procedure", "sources")},
        }
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
    if is_chole and C.OUT_DIR.resolve() == (C.ROOT / "out").resolve() and coord_dir.is_dir():
        (coord_dir / "manual.json").write_text(json.dumps(manual, indent=2))
    print(json.dumps(manual, indent=2)[:3000])
    print("\nsaved out/manual.json and out/research_notes.md")


if __name__ == "__main__":
    main()
