"""Step 5: decide which labels to trust, send the uncertain band to Argon, and write decisions.csv.

Zero-label rule (no human labels used): trust score from the checks, thresholds fixed in advance.
Statuses emitted: 'accept' (verified), 'AMBER' (abstain -> Human Gate), 'reject' (rejected).
Post-decision evaluation runs separately via eval.score_decisions after decisions.csv is saved.

  python accept.py            # writes out/decisions.csv and out/results.json
  python accept.py --no-argon # skip the Argon reviser (faster)
"""
import argparse
import asyncio
import json

import pandas as pd
from pydantic import BaseModel

import config as C
import gem
from eval.score_decisions import evaluate_decisions

VERIFIER_INPUT_COLS = [
    "frame_id",
    "video",
    "video_id",
    "split",
    "key",
    "mode",
    "sample",
    "model",
    "prompt_hash",
    "present",
    "conf",
    "confidence",
    "box",
    "evidence",
    "rationale",
    "pv_match",
    "pv_conf",
    "p_wrong",
    "critic",
]


class Verdict(BaseModel):
    present: bool
    confidence: float
    reasoning: str


def _pv_ok(val) -> bool:
    return bool(pd.notna(val) and val)


def trust(r):
    s = float(r.conf) * (1.0 - float(r.p_wrong))
    if int(r.present) == 1:
        s *= 1.0 if _pv_ok(getattr(r, "pv_match", None)) else 0.5
    return s


def status(s):
    return "accept" if s >= 0.70 else "reject" if s < 0.35 else "AMBER"


async def argon_revise(r, manual):
    pv_agreed = _pv_ok(getattr(r, "pv_match", None))
    v = await gem.call(
        f"You are the senior reviewer. A junior labeller said '{r.key}' is "
        f"{'PRESENT' if r.present else 'ABSENT'} (conf {r.conf:.2f}, evidence: {r.evidence}). "
        f"A blind crop check {'agreed' if pv_agreed else 'disagreed or was not possible'}. "
        f"A critic argued: {r.critic!r} (p_wrong {r.p_wrong:.2f}).\n"
        "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
        f"Manual: {json.dumps(manual)}",
        model=C.ARGON, images=[gem.img_part(r.frame_path)], schema=Verdict, tag="argon-revise")
    gem.trace("argon", f"{r.frame_id} {r.key}: {'present' if v['present'] else 'absent'} "
              f"({v['confidence']:.2f}) - {v['reasoning'][:140]}")
    return v


async def main(use_argon, argon_budget):
    idx = pd.read_csv(C.DATA_DIR / "index.csv")[["frame_id", "path"]].set_index("frame_id")
    raw_d = pd.read_json(C.OUT_DIR / "labels_manual.jsonl", lines=True)
    if "conf" not in raw_d.columns and "confidence" in raw_d.columns:
        raw_d["conf"] = raw_d["confidence"]
    if "evidence" not in raw_d.columns and "rationale" in raw_d.columns:
        raw_d["evidence"] = raw_d["rationale"]
    if "video" not in raw_d.columns and "video_id" in raw_d.columns:
        raw_d["video"] = raw_d["video_id"]
    keep_cols = [c for c in VERIFIER_INPUT_COLS if c in raw_d.columns]
    d = raw_d[keep_cols].copy()
    d["frame_path"] = d.frame_id.map(idx.path)
    d["trust"] = d.apply(trust, axis=1)
    d["status"] = d.trust.map(status)
    d["final_present"] = d.present
    d["final_by"] = "flash"

    if use_argon:
        manual = json.loads((C.OUT_DIR / "manual.json").read_text())
        band = d[d.status != "accept"].sort_values("trust", ascending=False).head(argon_budget)
        gem.trace("argon", f"Reviewing {len(band)} uncertain labels "
                  f"({(d.status != 'accept').sum()} flagged in total)")
        verdicts = await asyncio.gather(*[argon_revise(r, manual) for r in band.itertuples()])
        for (i, _), v in zip(band.iterrows(), verdicts):
            d.at[i, "final_present"] = int(v["present"])
            d.at[i, "final_by"] = "argon"
            if v["confidence"] >= 0.8:
                d.at[i, "status"] = "accept"
                d.at[i, "trust"] = 0.70 + 0.3 * v["confidence"] * 0.99  # ranks just above flash-accepts floor
            else:
                d.at[i, "status"] = "AMBER"
    d.to_csv(C.OUT_DIR / "decisions.csv", index=False)
    evaluate_decisions(C.DATA_DIR, C.OUT_DIR)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-argon", action="store_true")
    ap.add_argument("--argon-budget", type=int, default=300)
    a = ap.parse_args()
    asyncio.run(main(not a.no_argon, a.argon_budget))
