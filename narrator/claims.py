"""Atomic claim generation, two modes on the same clips with the same model.

python -m narrator.claims --mode free|grounded|both [--split test] [--max-clips N]
-> narrator/out/claims_<mode>.jsonl  (+ free narration text in narrator/out/free_text.jsonl)

free:     Gemini sees sampled frames and narrates like an expert commentator (prose). A separate
          extraction call breaks the prose into claims (same schema). Baseline.
grounded: Gemini sees the frames + event timeline + verified label table + relevant KB rules and must
          cite evidence for every claim.
"""
from __future__ import annotations

import argparse
import json
import time

import yaml

from narrator import cfg as C
from narrator import data as D
from narrator import events as E
from narrator import llm
from narrator.schema import Claim, ClaimList, enum, normalise_entity, vocab

NARRATOR_RULES = """Rules:
- Only atomic claims, one fact each. No free prose.
- Name a condition only through observed findings plus a KB rule id.
- Never use Tokyo grading. No "should have", no judgement of surgeon quality, no prediction of bile duct injury.
- Do not claim anything about structures the evidence cannot support (there is no cystic artery or clipper label in the evidence).
- If the evidence for something is uncertain, use polarity "hedged".
- EVERY claim must fill `entities` with enum values (e.g. presence of liver -> entities.target="liver"; L-hook -> entities.instrument="hook"; blood -> entities.finding="blood"). A claim with empty entities is invalid.
- An observation on a single frame needs an event id (ev:...) as evidence; otherwise give the frame span it holds over.
- t_start/t_end are absolute frame indices from the frame labels given."""


def _enums_text() -> str:
    v = vocab()
    return "\n".join(f"- {f}: {', '.join(enum(f))}" for f in
                     ["instrument", "verb", "target", "finding", "phase", "cvs_criterion", "grade", "condition", "decision"]) \
        + f"\nClaim types: presence, absence, action, phase, cvs_criterion, finding, condition, grade, decision_point, knowledge." \
          f"\nModality: observed (seen in frames), knowledge (general fact), inference (derived)." \
          f"\nPolarity: asserted, negated, hedged."


def clip_frames(clip) -> list[int]:
    n = C.cfg()["claims"]["frames_per_clip"]
    step = max(1, clip.n_frames // n)
    return [clip.first + i * step for i in range(n)]


def frame_images(clip, frames) -> list[bytes]:
    ms = C.cfg()["claims"]["max_side"]
    return [llm.jpeg(D.frame_path(clip.clip_id, f), max_side=ms) for f in frames]


def kb_rules_text(max_rules: int = 40) -> tuple[str, set]:
    """Compact list of KB rules (id: statement) for the grounded prompt."""
    rules = []
    for p in sorted(C.path("kb").glob("*.yaml")):
        try:
            y = yaml.safe_load(p.read_text()) or {}
        except Exception:  # noqa: BLE001
            continue
        for r in (y.get("rules") or []):
            if isinstance(r, dict) and r.get("id") and r.get("statement"):
                tag = "" if r.get("quote_verified") else " [unverified: hedge]"
                rules.append(f"{r['id']}: {r['statement']}{tag}")
    rules = rules[:max_rules]
    return "\n".join(rules), {r.split(":")[0] for r in rules}


def events_text(evs: list[dict]) -> str:
    lines = []
    for e in evs:
        v = f" value={e['value']}" if e.get("value") is not None else ""
        lines.append(f"{e['event_id']} {e['type']} {e['label']} frames {e['t_start']}-{e['t_end']} status={e['status']}{v}")
    return "\n".join(lines) or "(no events)"


def labels_text(labels, clip_id, frames) -> str:
    sub = labels[(labels.clip_id == clip_id)]
    lines = []
    for f in sorted(sub.frame_idx.unique()):
        s = sub[sub.frame_idx == f]
        items = [f"{r.label}({r.status})" for r in s.itertuples() if r.present]
        lines.append(f"frame {f}: " + (", ".join(items) or "none") + f"  [keys lab:{clip_id}:{f}:<label>]")
    return "\n".join(lines)


def _postprocess(items: list[dict], clip, mode: str, text_by_default: str = "") -> list[dict]:
    out = []
    for i, it in enumerate(items):
        ents = {k: normalise_entity(k, v) for k, v in (it.get("entities") or {}).items() if v not in (None, "")}
        d = dict(it, entities=ents, claim_id=f"{mode}-{clip.clip_id}-{i}", clip_id=clip.clip_id, mode=mode,
                 source="gemini")
        d["t_start"] = int(d.get("t_start") or clip.first)
        d["t_end"] = int(d.get("t_end") or d["t_start"])
        try:
            out.append(Claim.model_validate(d).model_dump(mode="json"))
        except Exception:  # noqa: BLE001  invalid enums for type/modality: keep for L1 to fail
            d.setdefault("polarity", "asserted")
            out.append(d)
    return out


async def free_clip(clip) -> dict:
    frames = clip_frames(clip)
    imgs = frame_images(clip, frames)
    t0 = time.time()
    prose = await llm.generate(
        "You are an expert laparoscopic surgeon commentating a recorded laparoscopic cholecystectomy for "
        f"surgical trainees. These are {len(frames)} frames in order, at absolute frame indices {frames} "
        "(25 fps). Narrate what is happening as an expert commentator would: instruments, what they are doing, "
        "anatomy visible, the phase, findings, CVS status, difficulty. 5-8 sentences. Mention frame indices.",
        model="fast_model", images=imgs, tag="free_narrate")
    if not prose:
        return {"clip_id": clip.clip_id, "claims": [], "failed": True}
    ext = await llm.generate(
        NARRATOR_RULES.split("Rules:")[0] + "Break this surgical narration into atomic claims. EVERY claim must fill `entities` with enum values (liver -> target=liver, L-hook -> instrument=hook, blood -> finding=blood). One fact per claim, every fact in the narration must "
        "become a claim (do not add or drop facts). Use only these enum values for entities (closest match; "
        "leave a field null if not applicable):\n" + _enums_text() +
        f"\nClip frames span {clip.first}..{clip.last}. evidence_refs: [] (the narration cites none).\n\n"
        f"NARRATION:\n{prose}", model="fast_model", schema=ClaimList, tag="free_extract")
    items = (ext or {}).get("claims", [])
    return {"clip_id": clip.clip_id, "text": prose, "claims": _postprocess(items, clip, "free"),
            "latency": time.time() - t0, "failed": ext is None}


async def grounded_clip(clip, labels, evs, kb_text) -> dict:
    frames = clip_frames(clip)
    imgs = frame_images(clip, frames)
    t0 = time.time()
    prompt = (
        "You narrate a recorded laparoscopic cholecystectomy for surgical teaching (post-hoc review, not guidance). "
        f"Images: {len(frames)} frames at absolute frame indices {frames}. Produce ATOMIC CLAIMS only.\n"
        + NARRATOR_RULES +
        "\n- Every observed claim MUST cite evidence_refs: event ids (ev:...) or label keys (lab:<clip>:<frame>:<label>) "
        "from the lists below. No evidence -> do not make the claim.\n"
        "- Labels marked (uncertain) -> polarity hedged. Labels not listed are not verified: do not assert them.\n"
        "- Knowledge/inference claims MUST cite kb_refs from the KB rules below; rules marked [unverified] -> hedged.\n"
        "- Also state absence of key structures when the label table verifies they are absent across the clip "
        "(type absence, polarity asserted).\n\n"
        "Allowed enums:\n" + _enums_text() +
        f"\n\nEVENT TIMELINE (clip {clip.clip_id}, frames {clip.first}..{clip.last}):\n{events_text(evs)}"
        f"\n\nVERIFIED LABEL TABLE (sampled frames):\n{labels_text(labels, clip.clip_id, frames)}"
        f"\n\nKB RULES:\n{kb_text}")
    out = await llm.generate(prompt, model="fast_model", images=imgs, schema=ClaimList, tag="grounded")
    items = (out or {}).get("claims", [])
    return {"clip_id": clip.clip_id, "claims": _postprocess(items, clip, "grounded"),
            "latency": time.time() - t0, "failed": out is None}


def select_clips(split: str | None, max_clips: int | None):
    clips = D.list_clips()
    if split:
        clips = clips[clips.split.isin(split.split(","))]
    mc = max_clips or C.cfg()["claims"]["max_clips"]
    if mc:
        clips = clips.groupby("video_id").head(max(1, mc // max(1, clips.video_id.nunique()))).head(mc)
    return list(clips.itertuples())


def run(mode: str, split: str | None = "test", max_clips: int | None = None) -> list[dict]:
    clips = select_clips(split, max_clips)
    evs = E.build()
    labels = D.load_labels()
    kb_text, _ = kb_rules_text()
    if mode == "free":
        res = llm.gather([free_clip(c) for c in clips], desc="free")
    else:
        res = llm.gather([grounded_clip(c, labels, evs.get(c.clip_id, []), kb_text) for c in clips], desc="grounded")
    res = [r for r in res if r]
    with open(C.out_dir() / f"claims_{mode}.jsonl", "w") as f:
        for r in res:
            for c in r["claims"]:
                f.write(json.dumps(c) + "\n")
    with open(C.out_dir() / f"gen_{mode}.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps({k: v for k, v in r.items() if k != "claims"} | {"n_claims": len(r["claims"])}) + "\n")
    n = sum(len(r["claims"]) for r in res)
    print(f"[{mode}] clips={len(res)} claims={n} failed={sum(r.get('failed', False) for r in res)} {llm.stats_line()}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="both", choices=["free", "grounded", "both"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-clips", type=int, default=None)
    a = ap.parse_args()
    for m in (["free", "grounded"] if a.mode == "both" else [a.mode]):
        run(m, a.split, a.max_clips)
