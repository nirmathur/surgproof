"""Post-decision evaluation step: joins decisions.csv with data/index.csv ground truth ONLY after decisions exist.

Verifier (`accept.py` / `verify/`) never touches ground truth (`truth` / `gt_*`).
Headline metrics exclude `liver` (100% prevalence across all 1,010 frames), reporting `liver` per-label.
Also scores the Amber resolution loop per round on TEST (`amber_loop_by_round`).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import config as C

UNINFORMATIVE_LABELS = {"liver"}
HEADLINE_LABELS = [k for k in C.LABELS if k not in UNINFORMATIVE_LABELS]


def _pv_ok(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
        return False
    if isinstance(val, str):
        return val.strip().lower() in ("true", "1", "1.0", "yes")
    return bool(val)


def _initial_trust(r) -> float:
    conf = float(getattr(r, "conf", getattr(r, "confidence", 0.5)))
    p_wrong = float(getattr(r, "p_wrong", 0.0) if pd.notna(getattr(r, "p_wrong", 0.0)) else 0.0)
    s = conf * (1.0 - p_wrong)
    if int(getattr(r, "present", 0)) == 1:
        s *= 1.0 if _pv_ok(getattr(r, "pv_match", None)) else 0.5
    return s


def _initial_status(s: float) -> str:
    return "accept" if s >= 0.70 else "reject" if s < 0.35 else "AMBER"


def score_amber_rounds(test_df: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Score the Amber loop per round on TEST (ambers remaining & precision of resolved items).

    Reads `out/amber_rounds.jsonl` if present; otherwise reconstructs Round 0 (initial double-check)
    and Round 1 (Argon revision) from `decisions.csv`. Never runs inside the verifier loop.
    """
    if len(test_df) == 0:
        return []

    df = test_df.copy()
    if "init_trust" not in df.columns:
        df["init_trust"] = [_initial_trust(r) for r in df.itertuples()]
    if "status_round_0" in df.columns:
        df["init_status"] = df["status_round_0"].astype(str)
    elif "init_status" not in df.columns:
        df["init_status"] = df["init_trust"].map(_initial_status)

    rounds_out: list[dict] = []
    r0_acc = df[df["init_status"] == "accept"]
    r0_rej = df[df["init_status"].isin(["reject", "escalate"])]
    r0_amb = df[df["init_status"].isin(["AMBER", "amber", "reanalyse"])]
    rounds_out.append({
        "round": 0,
        "stage": "initial_double_check",
        "n_test_labels": int(len(df)),
        "ambers_remaining": int(len(r0_amb)),
        "accepted_count": int(len(r0_acc)),
        "rejected_count": int(len(r0_rej)),
        "resolved_in_round": 0,
        "resolved_in_round_precision": None,
        "cumulative_accepted_precision": round(float(r0_acc["correct"].mean()), 4) if len(r0_acc) else None,
    })

    # If status_round_1..N columns exist directly on decisions.csv, score directly from them
    round_cols = sorted(
        [c for c in df.columns if c.startswith("status_round_") and c.split("_")[-1].isdigit() and int(c.split("_")[-1]) >= 1],
        key=lambda c: int(c.split("_")[-1]),
    )
    if round_cols:
        fc_col = "final_correct" if "final_correct" in df.columns else "correct"
        prev_col = "init_status"
        for col in round_cols:
            r_int = int(col.split("_")[-1])
            st_cur = df[col].astype(str)
            st_prev = df[prev_col].astype(str)
            r_acc = df[st_cur == "accept"]
            r_rej = df[st_cur.isin(["reject", "escalate"])]
            r_amb = df[st_cur.isin(["AMBER", "amber", "reanalyse"])]
            was_amb = st_prev.isin(["AMBER", "amber", "reanalyse"])
            now_res = st_cur.isin(["accept", "reject", "escalate"])
            res_sub = df[was_amb & now_res]
            res_correct = []
            for row in res_sub.itertuples():
                v_st = str(getattr(row, col))
                if v_st == "accept":
                    res_correct.append(int(getattr(row, fc_col)))
                else:
                    res_correct.append(int(int(row.present) != int(row.truth) or int(getattr(row, fc_col)) == 1))
            rounds_out.append({
                "round": r_int,
                "stage": f"amber_loop_round_{r_int}",
                "n_test_labels": int(len(df)),
                "ambers_remaining": int(len(r_amb)),
                "accepted_count": int(len(r_acc)),
                "rejected_count": int(len(r_rej)),
                "resolved_in_round": int(len(res_correct)),
                "resolved_in_round_precision": round(float(np.mean(res_correct)), 4) if res_correct else None,
                "cumulative_accepted_precision": round(float(r_acc[fc_col].mean()), 4) if len(r_acc) else None,
            })
            prev_col = col
        return rounds_out

    amber_rounds_path = out_dir / "amber_rounds.jsonl"
    if amber_rounds_path.exists() and amber_rounds_path.stat().st_size > 0:
        try:
            ar_df = pd.read_json(amber_rounds_path, lines=True)
            if len(ar_df) > 0 and "round" in ar_df.columns:
                # Track state across rounds for test split items
                state = {}
                for r in df.itertuples():
                    lid = f"{r.frame_id}__{r.key}"
                    state[lid] = {
                        "status": r.init_status,
                        "present": int(r.present),
                        "truth": int(r.truth),
                    }
                for r_num in sorted(ar_df["round"].dropna().unique()):
                    r_int = int(r_num)
                    sub = ar_df[ar_df["round"] == r_int]
                    resolved_correct = []
                    for row in sub.itertuples():
                        lid = getattr(row, "label_id", None)
                        if not lid and hasattr(row, "frame_id") and hasattr(row, "key"):
                            lid = f"{row.frame_id}__{row.key}"
                        if lid not in state:
                            continue
                        verdict = str(getattr(row, "verdict", getattr(row, "status", "AMBER")))
                        if verdict in ("accept", "reject"):
                            pred_pres = int(
                                getattr(
                                    row,
                                    "final_present",
                                    getattr(row, "argon_present", getattr(row, "present", state[lid]["present"])),
                                )
                            )
                            truth_val = state[lid]["truth"]
                            is_corr = (pred_pres == truth_val) if verdict == "accept" else (state[lid]["present"] != truth_val or pred_pres == truth_val)
                            resolved_correct.append(int(is_corr))
                            state[lid]["status"] = verdict
                            state[lid]["present"] = pred_pres
                    amb_rem = sum(1 for v in state.values() if v["status"] in ("AMBER", "amber", "reanalyse"))
                    acc_items = [v for v in state.values() if v["status"] == "accept"]
                    rej_items = [v for v in state.values() if v["status"] in ("reject", "escalate")]
                    cum_acc_prec = (
                        round(float(np.mean([int(v["present"] == v["truth"]) for v in acc_items])), 4)
                        if acc_items
                        else None
                    )
                    rounds_out.append({
                        "round": r_int,
                        "stage": f"amber_loop_round_{r_int}",
                        "n_test_labels": int(len(df)),
                        "ambers_remaining": int(amb_rem),
                        "accepted_count": int(len(acc_items)),
                        "rejected_count": int(len(rej_items)),
                        "resolved_in_round": int(len(resolved_correct)),
                        "resolved_in_round_precision": round(float(np.mean(resolved_correct)), 4) if resolved_correct else None,
                        "cumulative_accepted_precision": cum_acc_prec,
                    })
                return rounds_out
        except Exception:
            pass

    # Fallback: Round 1 from final status in decisions.csv (when Argon reviser ran on AMBER band)
    r1_acc = df[df["status"] == "accept"]
    r1_rej = df[df["status"].isin(["reject", "escalate"])]
    r1_amb = df[df["status"].isin(["AMBER", "amber", "reanalyse"])]
    resolved_from_r0 = r0_amb[~r0_amb["status"].isin(["AMBER", "amber", "reanalyse"])]
    if len(resolved_from_r0) > 0 or (len(r1_amb) != len(r0_amb)):
        fc_col = "final_correct" if "final_correct" in resolved_from_r0.columns else "correct"
        rounds_out.append({
            "round": 1,
            "stage": "argon_amber_resolution",
            "n_test_labels": int(len(df)),
            "ambers_remaining": int(len(r1_amb)),
            "accepted_count": int(len(r1_acc)),
            "rejected_count": int(len(r1_rej)),
            "resolved_in_round": int(len(resolved_from_r0)),
            "resolved_in_round_precision": round(float(resolved_from_r0[fc_col].mean()), 4) if len(resolved_from_r0) else None,
            "cumulative_accepted_precision": round(float(r1_acc[fc_col].mean()), 4) if len(r1_acc) else None,
        })
    return rounds_out


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


def _accepted_summary(sub_df: pd.DataFrame) -> dict:
    acc_t = sub_df[sub_df.status == "accept"]
    escalated_mask = sub_df.status.isin(["reject", "escalate", "AMBER", "amber", "reanalyse"])
    amber_mask = sub_df.status.isin(["AMBER", "amber", "reanalyse"])
    reject_mask = sub_df.status.isin(["reject", "escalate"])
    return {
        "coverage": round(len(acc_t) / len(sub_df), 3) if len(sub_df) else 0.0,
        "accuracy": round(float(acc_t.final_correct.mean()), 4) if len(acc_t) else None,
        "escalated_share": round(float(escalated_mask.mean()), 3) if len(sub_df) else 0.0,
        "amber_share": round(float(amber_mask.mean()), 3) if len(sub_df) else 0.0,
        "rejected_share": round(float(reject_mask.mean()), 3) if len(sub_df) else 0.0,
    }


def evaluate_decisions(data_dir: Path | None = None, out_dir: Path | None = None) -> dict:
    data_dir = Path(data_dir or C.DATA_DIR)
    out_dir = Path(out_dir or C.OUT_DIR)
    idx = pd.read_csv(data_dir / "index.csv")
    d = pd.read_csv(out_dir / "decisions.csv")
    d = _attach_gt(d, idx)

    test_all = d[d.split == "test"]
    test_no_liver = test_all[~test_all["key"].isin(UNINFORMATIVE_LABELS)]
    # Headline uses the 5 informative labels (excluding liver=100% prevalence);
    # falls back to test_all only in tiny synthetic unit tests that have no non-liver rows.
    test = test_no_liver if len(test_no_liver) > 0 else test_all

    res = {
        "headline_labels": HEADLINE_LABELS,
        "excluded_uninformative_labels": sorted(UNINFORMATIVE_LABELS),
        "n_test_labels": len(test_all),
        "n_test_labels_excl_liver": len(test),
        "n_test_labels_all_6": len(test_all),
    }
    res["single_conf"] = risk_coverage(test.correct, test.conf)
    res["system_zero_label"] = risk_coverage(test.final_correct, test.trust)
    res["accepted"] = _accepted_summary(test_all)
    res["accepted_excl_liver"] = _accepted_summary(test)
    res["accepted_all_6_labels"] = res["accepted"]
    res["amber_loop_by_round"] = score_amber_rounds(test, out_dir)
    res["amber_loop_by_round_all_6"] = score_amber_rounds(test_all, out_dir)
    res["amber_loop_by_round_all_splits"] = score_amber_rounds(d, out_dir)

    if res["amber_loop_by_round"] and res["amber_loop_by_round_all_6"]:
        r_hl = res["amber_loop_by_round"]
        r_all6 = res["amber_loop_by_round_all_6"]
        r_all_sp = res["amber_loop_by_round_all_splits"]
        res["amber_loop"] = {
            "max_rounds": max((x["round"] for x in r_hl), default=0),
            "ambers_by_round_all": [x["ambers_remaining"] for x in r_all_sp],
            "ambers_by_round_test": [x["ambers_remaining"] for x in r_all6],
            "ambers_by_round_test_excl_liver": [x["ambers_remaining"] for x in r_hl],
            "per_round_test": [
                {
                    "round": x6["round"],
                    "ambers_remaining": x6["ambers_remaining"],
                    "ambers_remaining_excl_liver": x5["ambers_remaining"],
                    "resolved_in_round": x6["resolved_in_round"],
                    "resolved_in_round_excl_liver": x5["resolved_in_round"],
                    "resolved_in_round_precision": x6["resolved_in_round_precision"],
                    "resolved_in_round_precision_excl_liver": x5["resolved_in_round_precision"],
                    "accepted_count": x6["accepted_count"],
                    "rejected_count": x6["rejected_count"],
                    "accepted_count_excl_liver": x5["accepted_count"],
                    "rejected_count_excl_liver": x5["rejected_count"],
                    "cumulative_accepted_precision": x6["cumulative_accepted_precision"],
                    "cumulative_accepted_precision_excl_liver": x5["cumulative_accepted_precision"],
                }
                for x6, x5 in zip(r_all6, r_hl)
            ],
        }

    # Baselines if available (headline excludes liver)
    blank_p = out_dir / "labels_blank.jsonl"
    if blank_p.exists() and blank_p.stat().st_size > 0:
        try:
            b = pd.read_json(blank_p, lines=True)
            b_test_all = _attach_gt(b[b.split == "test"], idx)
            b_test_hl = b_test_all[~b_test_all["key"].isin(UNINFORMATIVE_LABELS)]
            b_test = b_test_hl if len(b_test_hl) > 0 else b_test_all
            # Align blank comparison to the exact same test frame_ids present in decisions.csv when smoke-testing
            test_fids = set(test["frame_id"].unique())
            b_aligned = b_test[b_test["frame_id"].isin(test_fids)] if len(test_fids) > 0 else b_test
            if len(b_aligned) == 0:
                b_aligned = b_test
            res["blank_single_conf"] = risk_coverage(b_test.correct, b_test.conf)
            res["blank_raw_acc"] = round(float(b_aligned.correct.mean()), 4)
            res["blank_raw_acc_full_test"] = round(float(b_test.correct.mean()), 4)
            res["manual_raw_acc"] = round(float(test.correct.mean()), 4)
        except (FileNotFoundError, ValueError):
            pass

    v2_p = out_dir / "labels_manual_v2.jsonl"
    if v2_p.exists() and v2_p.stat().st_size > 0:
        try:
            v = pd.read_json(v2_p, lines=True)
            v = _attach_gt(v, idx)
            allv = pd.concat(
                [
                    d[["frame_id", "key", "present", "conf", "truth", "split"]],
                    v[["frame_id", "key", "present", "conf", "truth", "split"]],
                ]
            )
            allv_test = allv[(allv.split == "test") & (~allv["key"].isin(UNINFORMATIVE_LABELS))]
            if len(allv_test) == 0:
                allv_test = allv[allv.split == "test"]
            g = (
                allv_test
                .groupby(["frame_id", "key"])
                .agg(p=("present", "mean"), c=("conf", "mean"), truth=("truth", "first"))
            )
            g["vote"] = (g.p >= 0.5).astype(int)
            g["agree"] = np.maximum(g.p, 1 - g.p) * g.c
            res["majority_vote"] = risk_coverage((g.vote == g["truth"]).astype(int), g.agree)
        except (FileNotFoundError, ValueError):
            pass

    # Upper bound: fitted on small labelled calib split (reported as upper bound only)
    cal_all = d[d.split == "calib"]
    cal_hl = cal_all[~cal_all["key"].isin(UNINFORMATIVE_LABELS)]
    cal = cal_hl if len(cal_hl) > 20 else cal_all
    if len(cal) > 20 and cal.correct.nunique() == 2 and len(test) > 0:
        feats = lambda x: np.c_[
            x.conf,
            x.p_wrong,
            x.present,
            pd.to_numeric(x.pv_match, errors="coerce").fillna(0.5).astype(float),
            pd.to_numeric(x.pv_conf, errors="coerce").fillna(0.5).astype(float),
        ]
        lr = LogisticRegression().fit(feats(cal), cal.correct)
        ub = risk_coverage(test.correct, lr.predict_proba(feats(test))[:, 1])
        ub["is_upper_bound"] = True
        ub["note"] = "UPPER BOUND ONLY: logistic regression calibrated using human ground-truth labels on calib split"
        res["upper_bound_calibrated"] = ub

    acc_t_all = test_all[test_all.status == "accept"]
    res["per_label_accepted_acc"] = (
        acc_t_all.groupby("key").final_correct.mean().round(3).to_dict() if len(acc_t_all) else {}
    )
    acc_t_hl = test[test.status == "accept"]
    per_lbl = res["per_label_accepted_acc"]
    hl_lbl_accs = [float(per_lbl[k]) for k in HEADLINE_LABELS if k in per_lbl and per_lbl[k] is not None]
    res["accepted_5_informative"] = {
        "n_test_labels_excl_liver": int(len(test)),
        "n_accepted_excl_liver": int(len(acc_t_hl)),
        "coverage": res["accepted_excl_liver"]["coverage"],
        "accuracy": res["accepted_excl_liver"]["accuracy"],
        "mean_per_label_accuracy": round(float(np.mean(hl_lbl_accs)), 4) if hl_lbl_accs else None,
        "escalated_share": res["accepted_excl_liver"]["escalated_share"],
        "amber_share": res["accepted_excl_liver"]["amber_share"],
        "rejected_share": res["accepted_excl_liver"]["rejected_share"],
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or "curve" not in v}, indent=2))
    for k, v in res.items():
        if isinstance(v, dict) and "curve" in v:
            print(f"{k:28s} coverage@95%={v['coverage_at_95']:.2f}  acc@60%={v['acc_at_60']:.3f}  AURC={v['aurc']:.4f}")
    return res

