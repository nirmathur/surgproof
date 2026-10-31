"""Step 4: label frames with Flash, then verify with decorrelated checks.

Checks per frame:
  1. labeller   - full frame, presence + confidence + box for each label
  2. point-and-verify - crop each claimed box, ask a FRESH call what it is (blind to the claim)
  3. critic     - argue each claim is wrong, return p_wrong

  python label.py --mode manual            # main run (all splits), uses out/manual.json
  python label.py --mode blank --splits test   # baseline: no manual
  python label.py --mode manual --votes 2 --splits train,test   # extra samples for majority vote
  python label.py --limit 40               # quick smoke test
"""
import argparse
import asyncio
import json

import pandas as pd
from pydantic import BaseModel

import config as C
import gem


class Item(BaseModel):
    key: str
    present: bool
    confidence: float          # 0-1 that `present` is correct
    box_2d: list[int]          # [ymin, xmin, ymax, xmax] 0-1000, empty if absent
    evidence: str


class FrameLabels(BaseModel):
    items: list[Item]


class CropID(BaseModel):
    answer: str
    confidence: float


class Rebuttal(BaseModel):
    key: str
    argument: str
    p_wrong: float


class Critique(BaseModel):
    rebuttals: list[Rebuttal]


NAMES = {k: v[1] for k, v in C.LABELS.items()}


def label_prompt(manual):
    base = ("You are labelling a single frame from laparoscopic cholecystectomy video.\n"
            f"For EACH of these keys decide if it is visible: {json.dumps(NAMES)}.\n"
            "Return one item per key. confidence = probability your present/absent call is correct. "
            "If present, give a tight box_2d [ymin,xmin,ymax,xmax] normalised 0-1000; if absent, []. "
            "Evidence: one short phrase on what you saw.")
    if manual:
        base += "\n\nUse this labelling manual you wrote earlier:\n" + json.dumps(manual)
    return base


async def verify_box(path, key, box):
    part = gem.img_part(path, crop=box)
    options = list(NAMES.values()) + ["none / unclear"]
    r = await gem.call(
        "This is a crop from a laparoscopic surgery frame. What is the main structure or instrument "
        f"in it? Answer with exactly one of: {json.dumps(options)}.",
        images=[part], schema=CropID, tag="verify")
    return r["answer"].strip().lower() == NAMES[key].lower(), r["confidence"]


async def critique(path, items):
    claims = [{"key": i["key"], "claim": "present" if i["present"] else "absent"} for i in items]
    r = await gem.call(
        "You are a sceptical surgical reviewer. For each claim about this frame, make the strongest "
        "case that it is WRONG, then give p_wrong (0-1) honestly; low if the claim is clearly right.\n"
        f"Claims: {json.dumps(claims)}",
        images=[gem.img_part(path)], schema=Critique, tag="critic")
    return {x["key"]: x for x in r["rebuttals"]}


async def do_frame(row, mode, manual, sample, full_checks):
    temp = 0.0 if sample == 0 else 0.7
    lab = await gem.call(label_prompt(manual if mode == "manual" else None),
                         images=[gem.img_part(row.path)], schema=FrameLabels,
                         temperature=temp, sample=sample, tag=f"label-{mode}")
    items = {i["key"]: i for i in lab["items"] if i["key"] in NAMES}
    out = []
    crit = await critique(row.path, list(items.values())) if full_checks else {}
    for key in NAMES:
        it = items.get(key, {"present": False, "confidence": 0.0, "box_2d": [], "evidence": "missing"})
        rec = {"frame_id": row.frame_id, "video": row.video, "split": row.split, "key": key,
               "mode": mode, "sample": sample, "present": int(it["present"]),
               "conf": float(it["confidence"]), "box": it["box_2d"], "evidence": it["evidence"],
               "truth": int(getattr(row, f"gt_{key}"))}
        if full_checks:
            if it["present"] and len(it["box_2d"]) == 4:
                rec["pv_match"], rec["pv_conf"] = await verify_box(row.path, key, it["box_2d"])
            else:
                rec["pv_match"], rec["pv_conf"] = (None, None)
            rec["p_wrong"] = float(crit.get(key, {}).get("p_wrong", 0.5))
            rec["critic"] = crit.get(key, {}).get("argument", "")
        out.append(rec)
    return out


async def run(mode, splits, votes, limit):
    df = pd.read_csv(C.DATA_DIR / "index.csv")
    df = df[df.split.isin(splits)]
    if limit:
        df = df.groupby("video").head(max(1, limit // df.video.nunique()))
    manual = json.loads((C.OUT_DIR / "manual.json").read_text()) if mode == "manual" else None
    samples = range(1, votes + 1) if votes else [0]
    jobs = [do_frame(r, mode, manual, s, full_checks=(s == 0 and mode == "manual"))
            for r in df.itertuples() for s in samples]
    gem.trace("flash", f"Labelling {len(df)} frames, mode={mode}, samples={list(samples)}")
    recs = []
    for i, coro in enumerate(asyncio.as_completed(jobs), 1):
        recs += await coro
        if i % 20 == 0:
            print(f"{i}/{len(jobs)} frames done")
    tag = f"{mode}_v{votes}" if votes else mode
    path = C.OUT_DIR / f"labels_{tag}.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    d = pd.DataFrame(recs)
    acc = (d.present == d.truth).mean()
    print(f"saved {path}. raw accuracy vs hidden GT: {acc:.3f}")
    print(d.assign(ok=d.present == d.truth).groupby("key").ok.mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manual", "blank"], default="manual")
    ap.add_argument("--splits", default="train,calib,test")
    ap.add_argument("--votes", type=int, default=0, help="extra sampled runs for majority vote")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(run(a.mode, a.splits.split(","), a.votes, a.limit))
