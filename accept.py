"""Step 5: decide which labels to trust, send the uncertain band to Argon, and evaluate.

Zero-label rule (no human labels used): trust score from the checks, thresholds fixed in advance.
Upper-bound rule: logistic regression fitted on the small 'calib' split (clearly labelled as such).

  python accept.py            # writes out/decisions.csv and out/results.json
  python accept.py --no-argon # skip the Argon reviser (faster)
"""
import argparse
import asyncio
import json

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

import config as C
import gem


class Verdict(BaseModel):
    present: bool
    confidence: float
    reasoning: str


def trust(r):
    s = r.conf * (1 - r.p_wrong)
    if r.present:
        s *= 1.0 if r.pv_match is True else 0.5
    return s


def status(s):
    return "accept" if s >= 0.70 else "escalate" if s < 0.35 else "reanalyse"


async def argon_revise(r, manual):
    v = await gem.call(
        f"You are the senior reviewer. A junior labeller said '{r.key}' is "
        f"{'PRESENT' if r.present else 'ABSENT'} (conf {r.conf:.2f}, evidence: {r.evidence}). "
        f"A blind crop check {'agreed' if r.pv_match else 'disagreed or was not possible'}. "
        f"A critic argued: {r.critic!r} (p_wrong {r.p_wrong:.2f}).\n"
        "Look at the frame yourself and give the final call. Be willing to say you are unsure.\n"
        f"Manual: {json.dumps(manual)}",
        model=C.ARGON, images=[gem.img_part(r.frame_path)], schema=Verdict, tag="argon-revise")
    gem.trace("argon", f"{r.frame_id} {r.key}: {'present' if v['present'] else 'absent'} "
              f"({v['confidence']:.2f}) - {v['reasoning'][:140]}")
    return v


def risk_coverage(correct, score):
    order = np.argsort(-np.asarray(score))
    c = np.asarray(correct)[order]
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    pts = [(round(float(x), 3), round(float(y), 4)) for x, y in zip(cov[::max(1, len(c) // 50)], acc[::max(1, len(c) // 50)])]
    at95 = float(cov[acc >= 0.95].max()) if (acc >= 0.95).any() else 0.0
    return {"curve": pts, "coverage_at_95": round(at95, 3),
            "acc_at_60": round(float(acc[int(0.6 * len(c)) - 1]), 4), "aurc": round(float((1 - acc).mean()), 4)}


async def main(use_argon, argon_budget):
    idx = pd.read_csv(C.DATA_DIR / "index.csv").set_index("frame_id")
    d = pd.read_json(C.OUT_DIR / "labels_manual.jsonl", lines=True)
    d["frame_path"] = d.frame_id.map(idx.path)
    d["correct"] = (d.present == d.truth).astype(int)
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
                d.at[i, "status"] = "escalate"
    d["final_correct"] = (d.final_present == d.truth).astype(int)
    d.to_csv(C.OUT_DIR / "decisions.csv", index=False)

    test = d[d.split == "test"]
    res = {"n_test_labels": len(test)}
    res["single_conf"] = risk_coverage(test.correct, test.conf)
    res["system_zero_label"] = risk_coverage(test.final_correct, test.trust)
    acc_t = test[test.status == "accept"]
    res["accepted"] = {"coverage": round(len(acc_t) / len(test), 3),
                       "accuracy": round(acc_t.final_correct.mean(), 4) if len(acc_t) else None,
                       "escalated_share": round((test.status == "escalate").mean(), 3)}

    # baselines if available
    try:
        b = pd.read_json(C.OUT_DIR / "labels_blank.jsonl", lines=True)
        b = b[b.split == "test"]
        res["blank_single_conf"] = risk_coverage((b.present == b["truth"]).astype(int), b.conf)
        res["blank_raw_acc"] = round((b.present == b["truth"]).mean(), 4)
        res["manual_raw_acc"] = round(test.correct.mean(), 4)
    except (FileNotFoundError, ValueError):
        pass
    try:
        v = pd.read_json(C.OUT_DIR / "labels_manual_v2.jsonl", lines=True)
        allv = pd.concat([d[["frame_id", "key", "present", "conf", "truth", "split"]], v[["frame_id", "key", "present", "conf", "truth", "split"]]])
        g = allv[allv.split == "test"].groupby(["frame_id", "key"]).agg(p=("present", "mean"), c=("conf", "mean"), truth=("truth", "first"))
        g["vote"] = (g.p >= 0.5).astype(int)
        g["agree"] = np.maximum(g.p, 1 - g.p) * g.c
        res["majority_vote"] = risk_coverage((g.vote == g["truth"]).astype(int), g.agree)
    except (FileNotFoundError, ValueError):
        pass

    # upper bound: fitted on small labelled calib split
    cal = d[d.split == "calib"]
    if len(cal) > 50 and cal.correct.nunique() == 2:
        feats = lambda x: np.c_[x.conf, x.p_wrong, x.present, x.pv_match.fillna(0.5).astype(float), x.pv_conf.fillna(0.5)]
        lr = LogisticRegression().fit(feats(cal), cal.correct)
        res["upper_bound_calibrated"] = risk_coverage(test.correct, lr.predict_proba(feats(test))[:, 1])

    res["per_label_accepted_acc"] = acc_t.groupby("key").final_correct.mean().round(3).to_dict()
    (C.OUT_DIR / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "curve" not in v}, indent=2))
    for k, v in res.items():
        if isinstance(v, dict) and "curve" in v:
            print(f"{k:28s} coverage@95%={v['coverage_at_95']:.2f}  acc@60%={v['acc_at_60']:.3f}  AURC={v['aurc']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-argon", action="store_true")
    ap.add_argument("--argon-budget", type=int, default=300)
    a = ap.parse_args()
    asyncio.run(main(not a.no_argon, a.argon_budget))
