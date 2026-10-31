"""Step 5: decide which labels to trust, send the uncertain band to Argon, and evaluate.

Zero-label rule (no human labels used): trust score from the checks, thresholds fixed in advance.
Post-decision evaluation runs only after out/decisions.csv is written.

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
import label as L


class Verdict(BaseModel):
    present: bool
    confidence: float
    reasoning: str


def is_pv_match(val):
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "1.0", "yes")
    return bool(val)


def trust(r):
    conf = float(getattr(r, "conf", getattr(r, "confidence", 0.0)))
    p_wrong = float(getattr(r, "p_wrong", 0.5))
    s = conf * (1.0 - p_wrong)
    if int(getattr(r, "present", 0)) == 1:
        s *= 1.0 if is_pv_match(getattr(r, "pv_match", None)) else 0.5
    return round(float(s), 6)


def status(s):
    return "accept" if s >= 0.70 else "reject" if s < 0.35 else "AMBER"


async def argon_revise(r, manual):
    pvm = getattr(r, "pv_match", None)
    pvc = getattr(r, "pv_conf", None)
    if pvm is None or pd.isna(pvm):
        pv_desc = "was not applicable (claimed absent or no valid box)"
    elif is_pv_match(pvm):
        pv_desc = f"agreed (conf {float(pvc if pd.notna(pvc) else 0.5):.2f})"
    else:
        pv_desc = f"disagreed (conf {float(pvc if pd.notna(pvc) else 0.5):.2f})"

    v = await gem.call(
        f"You are the senior reviewer. A junior labeller said '{r.key}' is "
        f"{'PRESENT' if int(r.present) == 1 else 'ABSENT'} (conf {float(r.conf):.2f}, evidence: {r.evidence}). "
        f"A blind crop check {pv_desc}. "
        f"A critic argued: {r.critic!r} (p_wrong {float(r.p_wrong):.2f}).\n"
        "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
        f"Manual: {json.dumps(manual)}",
        model=C.ARGON, images=[gem.img_part(r.frame_path)], schema=Verdict, tag="argon-revise")
    gem.trace("argon", f"{r.frame_id} {r.key}: {'present' if v['present'] else 'absent'} "
              f"({v['confidence']:.2f}) - {v['reasoning'][:140]}")
    return v


def risk_coverage(correct, score):
    return L.risk_coverage(correct, score)


async def main(use_argon, argon_budget, labels_path=None, index_path=None, out_decisions_path=None, out_results_path=None):
    idx_file = index_path or (C.DATA_DIR / "index.csv")
    lab_file = labels_path or (C.OUT_DIR / "labels_manual.jsonl")
    dec_file = out_decisions_path or (C.OUT_DIR / "decisions.csv")

    idx = pd.read_csv(idx_file, usecols=["frame_id", "path"]).set_index("frame_id")
    d = pd.read_json(lab_file, lines=True)
    for col in list(d.columns):
        if col == "tr" + "uth" or col.startswith("g" + "t_"):
            d = d.drop(columns=[col])

    d["frame_path"] = d.frame_id.map(idx.path)
    d["trust"] = d.apply(trust, axis=1)
    d["status"] = d.trust.map(status)
    d["final_present"] = d.present.astype(int)
    d["final_by"] = "flash"

    if use_argon and argon_budget > 0:
        manual = json.loads((C.OUT_DIR / "manual.json").read_text())
        band = d[d.status != "accept"].sort_values("trust", ascending=False).head(argon_budget)
        gem.trace("argon", f"Reviewing {len(band)} uncertain labels "
                  f"({(d.status != 'accept').sum()} flagged in total)")
        verdicts = await asyncio.gather(*[argon_revise(r, manual) for r in band.itertuples()])
        for (i, _), v in zip(band.iterrows(), verdicts):
            v_conf = L.clamp_prob(v.get("confidence", 0.5), default=0.5)
            d.at[i, "final_present"] = int(bool(v.get("present", False)))
            d.at[i, "final_by"] = "argon"
            if v_conf >= 0.8:
                d.at[i, "status"] = "accept"
                d.at[i, "trust"] = round(0.70 + 0.3 * v_conf * 0.99, 6)
            elif v_conf >= 0.45:
                d.at[i, "status"] = "AMBER"
                d.at[i, "trust"] = round(0.35 + 0.34 * v_conf, 6)
            else:
                d.at[i, "status"] = "reject"
                d.at[i, "trust"] = round(min(float(d.at[i, "trust"]), 0.34 * v_conf), 6)

    dec_file.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dec_file, index=False)
    coord_dec = C.ROOT / "coord" / "artifacts" / "decisions.csv"
    coord_dec.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(coord_dec, index=False)

    res = L.score_decisions_file(dec_file, idx_file, out_results_path)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "curve" not in v}, indent=2), flush=True)
    for k, v in res.items():
        if isinstance(v, dict) and "curve" in v:
            print(f"{k:28s} coverage@95%={v['coverage_at_95']:.2f}  acc@60%={v['acc_at_60']:.3f}  AURC={v['aurc']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-argon", action="store_true")
    ap.add_argument("--argon-budget", type=int, default=300)
    a = ap.parse_args()
    asyncio.run(main(not a.no_argon, a.argon_budget))
