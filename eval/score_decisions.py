"""Post-decision evaluation step: joins decisions.csv with data/index.csv ground truth ONLY after decisions exist.

Verifier (`accept.py` / `verify/`) never touches ground truth (`truth` / `gt_*`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config as C


def risk_coverage(correct, score):
    order = np.argsort(-np.asarray(score))
    c = np.asarray(correct)[order]
    if len(c) == 0:
        return {"curve": [], "coverage_at_95": 0.0, "acc_at_60": 0.0, "aurc": 0.0}
    cov = np.arange(1, len(c) + 1) / len(c)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    step = max(1, len(c) // 50)
    pts = [(round(float(x), 3), round(float(y), 4)) for x, y in zip(cov[::step], acc[::step])]
    at95 = float(cov[acc >= 0.95].max()) if (acc >= 0.95).any() else 0.0
    idx60 = max(0, int(0.6 * len(c)) - 1)
    return {
        "curve": pts,
        "coverage_at_95": round(at95, 3),
        "acc_at_60": round(float(acc[idx60]), 4),
        "aurc": round(float((1 - acc).mean()), 4),
    }


def _attach_gt(df: pd.DataFrame, idx: pd.DataFrame) -> pd.DataFrame:
    """Attach ground truth from data/index.csv (`gt_<key>`) by (frame_id, key)."""
    out = df.copy()
    idx_by_fid = idx.set_index("frame_id") if "frame_id" in idx.columns else idx
    gt_vals = []
    for r in out.itertuples():
        col = f"gt_{r.key}"
        if r.frame_id in idx_by_fid.index and col in idx_by_fid.columns:
            gt_vals.append(int(idx_by_fid.loc[r.frame_id, col]))
        elif "truth" in out.columns and pd.notna(getattr(r, "truth", None)):
            gt_vals.append(int(getattr(r, "truth")))
        else:
            gt_vals.append(0)
    out["truth"] = gt_vals
    out["correct"] = (out["present"] == out["truth"]).astype(int)
    if "final_present" in out.columns:
        out["final_correct"] = (out["final_present"] == out["truth"]).astype(int)
    return out


def evaluate_decisions(data_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir or C.DATA_DIR)
    out_dir = Path(out_dir or C.OUT_DIR)
    idx = pd.read_csv(data_dir / "index.csv")
    d = pd.read_csv(out_dir / "decisions.csv")
    d = _attach_gt(d, idx)

    test = d[d.split == "test"]
    res = {"n_test_labels": len(test)}
    res["single_conf"] = risk_coverage(test.correct, test.conf)
    res["system_zero_label"] = risk_coverage(test.final_correct, test.trust)
    acc_t = test[test.status == "accept"]
    escalated_mask = test.status.isin(["reject", "escalate", "AMBER", "reanalyse"])
    amber_mask = test.status.isin(["AMBER", "amber", "reanalyse"])
    reject_mask = test.status.isin(["reject", "escalate"])
    res["accepted"] = {
        "coverage": round(len(acc_t) / len(test), 3) if len(test) else 0.0,
        "accuracy": round(float(acc_t.final_correct.mean()), 4) if len(acc_t) else None,
        "escalated_share": round(float(escalated_mask.mean()), 3) if len(test) else 0.0,
        "amber_share": round(float(amber_mask.mean()), 3) if len(test) else 0.0,
        "rejected_share": round(float(reject_mask.mean()), 3) if len(test) else 0.0,
    }

    # Baselines if available
    try:
        b = pd.read_json(out_dir / "labels_blank.jsonl", lines=True)
        b = _attach_gt(b[b.split == "test"], idx)
        res["blank_single_conf"] = risk_coverage(b.correct, b.conf)
        res["blank_raw_acc"] = round(float(b.correct.mean()), 4)
        res["manual_raw_acc"] = round(float(test.correct.mean()), 4)
    except (FileNotFoundError, ValueError):
        pass

    try:
        v = pd.read_json(out_dir / "labels_manual_v2.jsonl", lines=True)
        v = _attach_gt(v, idx)
        allv = pd.concat(
            [
                d[["frame_id", "key", "present", "conf", "truth", "split"]],
                v[["frame_id", "key", "present", "conf", "truth", "split"]],
            ]
        )
        g = (
            allv[allv.split == "test"]
            .groupby(["frame_id", "key"])
            .agg(p=("present", "mean"), c=("conf", "mean"), truth=("truth", "first"))
        )
        g["vote"] = (g.p >= 0.5).astype(int)
        g["agree"] = np.maximum(g.p, 1 - g.p) * g.c
        res["majority_vote"] = risk_coverage((g.vote == g["truth"]).astype(int), g.agree)
    except (FileNotFoundError, ValueError):
        pass

    # Upper bound: fitted on small labelled calib split (reported as upper bound only)
    cal = d[d.split == "calib"]
    if len(cal) > 50 and cal.correct.nunique() == 2 and len(test) > 0:
        feats = lambda x: np.c_[
            x.conf,
            x.p_wrong,
            x.present,
            pd.to_numeric(x.pv_match, errors="coerce").fillna(0.5).astype(float),
            pd.to_numeric(x.pv_conf, errors="coerce").fillna(0.5).astype(float),
        ]
        lr = LogisticRegression().fit(feats(cal), cal.correct)
        res["upper_bound_calibrated"] = risk_coverage(test.correct, lr.predict_proba(feats(test))[:, 1])

    res["per_label_accepted_acc"] = (
        acc_t.groupby("key").final_correct.mean().round(3).to_dict() if len(acc_t) else {}
    )
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "curve" not in v}, indent=2))
    for k, v in res.items():
        if isinstance(v, dict) and "curve" in v:
            print(f"{k:28s} coverage@95%={v['coverage_at_95']:.2f}  acc@60%={v['acc_at_60']:.3f}  AURC={v['aurc']:.4f}")
    return res
