"""Step 6: the system teaches itself a specialist. Same student, three label sources:
  single   - every Gemini label on train videos (more data, noisier)
  vote     - majority of 3 samples (needs labels_manual_v2.jsonl)
  verified - only labels the system accepted (fewer, cleaner)
  oracle   - human ground truth (UPPER BOUND ONLY, not part of the claim)
Evaluated on held-out test videos against hidden ground truth.
Note: 'liver' is present in 100% of CholecSeg8k frames (zero information), so it is reported in
per_label['liver'] but excluded from headline means (mean_acc_by_source, mean_f1_by_source, mean_auroc_by_source).

  python student.py            # writes out/student.json and out/student_probs.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

import config as C

UPPER_BOUND_SOURCES = {"oracle_upper_bound"}
UNINFORMATIVE_LABELS = {"liver"}
HEADLINE_LABELS = [k for k in C.LABELS if k not in UNINFORMATIVE_LABELS]
CORE_BALANCED_LABELS = ["grasper", "l_hook", "gallbladder"]
SOURCE_ORDER = ["single", "vote", "verified", "oracle_upper_bound"]
SOURCE_META = {
    "single": {
        "is_upper_bound": False,
        "uses_human_labels": False,
        "description": "Every single-pass Gemini label on train videos (unfiltered baseline)",
    },
    "vote": {
        "is_upper_bound": False,
        "uses_human_labels": False,
        "description": "Majority vote across multiple Gemini samples on train videos (multi-sample baseline)",
    },
    "verified": {
        "is_upper_bound": False,
        "uses_human_labels": False,
        "description": "Only train labels accepted by the zero-human-label verification pipeline",
    },
    "oracle_upper_bound": {
        "is_upper_bound": True,
        "uses_human_labels": True,
        "description": "UPPER BOUND ONLY: trained on human ground-truth labels on train videos (not part of claim)",
    },
}


def _l2_normalize(X):
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-8)


def fit_eval(X, y, Xte, yte, *, is_upper_bound=False, informative=True, return_model=False, sample_weight=None):
    y = np.asarray(y, dtype=int)
    yte = np.asarray(yte, dtype=int)
    valid_tr = np.isin(y, [0, 1])
    X, y = X[valid_tr], y[valid_tr]
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)[valid_tr]
    if len(y) == 0:
        return (None, None, None) if return_model else (None, None)
    uniq = np.unique(y)
    m = None
    if len(uniq) < 2:
        # Constant-class fallback so every source is evaluated across the same label set
        const_val = float(uniq[0])
        p = np.full(len(yte), const_val, dtype=float)
        single_class = True
    else:
        Xn = _l2_normalize(X)
        Xten = _l2_normalize(Xte)
        m = LogisticRegression(max_iter=2000, class_weight=None, C=5.0, random_state=C.SEED).fit(
            Xn, y, sample_weight=sample_weight
        )
        p = m.predict_proba(Xten)[:, 1]
        single_class = False

    valid_te = np.isin(yte, [0, 1])
    if not np.any(valid_te):
        res_unscorable = {
            "n_train": int(len(y)),
            "n_pos_train": int(y.sum()),
            "n_test_scorable": 0,
            "acc": None,
            "f1": None,
            "is_upper_bound": bool(is_upper_bound),
            "informative": False,
            "excluded_from_headline": True,
            "unscorable_on_test": True,
        }
        return (res_unscorable, p, m) if return_model else (res_unscorable, p)

    yte_v = yte[valid_te]
    p_v = p[valid_te]
    preds = (p_v >= 0.5).astype(int)
    out = {
        "n_train": int(len(y)),
        "n_pos_train": int(y.sum()),
        "n_test_scorable": int(len(yte_v)),
        "n_pos_test_gt": int(yte_v.sum()),
        "pos_rate_train": round(float(y.mean()), 4),
        "pos_rate_test_gt": round(float(yte_v.mean()), 4),
        "acc": round(float((preds == yte_v).mean()), 4),
        "balanced_acc": round(float(balanced_accuracy_score(yte_v, preds)), 4) if len(np.unique(yte_v)) == 2 else None,
        "f1": round(float(f1_score(yte_v, preds, zero_division=0)), 4),
        "pred_pos_rate_test": round(float(preds.mean()), 4),
        "is_upper_bound": bool(is_upper_bound),
        "informative": bool(informative),
        "excluded_from_headline": not bool(informative),
    }
    if single_class:
        out["single_class_fallback"] = True
    if len(np.unique(yte_v)) == 2:
        out["auroc"] = round(float(roc_auc_score(yte_v, p_v)), 4)
    return (out, p, m) if return_model else (out, p)


def _eval_trained_models_on_external(models_by_key_source: dict, const_by_key_source: dict, data_dir):
    ext_root = data_dir / "external"
    if not ext_root.exists():
        return {}
    ext_results = {}
    for ext_idx_p in sorted(ext_root.glob("*/index.csv")):
        ds_name = ext_idx_p.parent.name
        feat_p = ext_idx_p.parent / "features.npy"
        if not feat_p.exists():
            continue
        ext_idx = pd.read_csv(ext_idx_p)
        ext_X = np.load(feat_p)
        if len(ext_X) != len(ext_idx):
            continue
        te_mask = (ext_idx.split == "test").values if "split" in ext_idx.columns else np.ones(len(ext_idx), dtype=bool)
        ext_te = ext_idx[te_mask]
        X_ext_te = _l2_normalize(ext_X[te_mask])
        if len(ext_te) == 0:
            continue

        allowed_ext_keys = {"endoscapes2023": {"gallbladder", "cystic_duct"}, "cholect50": {"grasper", "l_hook"}}.get(
            ds_name, set(C.LABELS)
        )
        per_label_ext = {}
        for key in C.LABELS:
            if key not in allowed_ext_keys:
                continue
            col = f"gt_{key}"
            if col not in ext_te.columns:
                continue
            yte = pd.to_numeric(ext_te[col], errors="coerce").values
            valid = np.isin(yte, [0, 1])
            if not np.any(valid):
                continue
            yte_v = yte[valid].astype(int)
            X_v = X_ext_te[valid]
            per_label_ext[key] = {}
            for name, m in models_by_key_source.get(key, {}).items():
                if m is not None:
                    p_v = m.predict_proba(X_v)[:, 1]
                else:
                    cval = const_by_key_source.get(key, {}).get(name, 0.0)
                    p_v = np.full(len(yte_v), cval, dtype=float)
                preds = (p_v >= 0.5).astype(int)
                entry = {
                    "n_test_scorable": int(len(yte_v)),
                    "n_pos_test_gt": int(yte_v.sum()),
                    "acc": round(float((preds == yte_v).mean()), 4),
                    "f1": round(float(f1_score(yte_v, preds, zero_division=0)), 4),
                    "is_upper_bound": bool(name in UPPER_BOUND_SOURCES),
                }
                if len(np.unique(yte_v)) == 2:
                    entry["auroc"] = round(float(roc_auc_score(yte_v, p_v)), 4)
                per_label_ext[key][name] = entry

        mean_acc_ext, mean_auroc_ext = {}, {}
        for name in SOURCE_ORDER:
            vals = [per_label_ext[k][name]["acc"] for k in per_label_ext if name in per_label_ext[k]]
            aucs = [per_label_ext[k][name]["auroc"] for k in per_label_ext if name in per_label_ext[k] and "auroc" in per_label_ext[k][name]]
            if vals:
                mean_acc_ext[name] = round(float(np.mean(vals)), 4)
            if aucs:
                mean_auroc_ext[name] = round(float(np.mean(aucs)), 4)
        ext_results[ds_name] = {
            "n_test_frames": int(len(ext_te)),
            "scorable_labels": sorted(per_label_ext.keys()),
            "mean_acc_by_source": mean_acc_ext,
            "mean_auroc_by_source": mean_auroc_ext,
            "per_label": per_label_ext,
        }
    return ext_results


def _load_vote_source(tr):
    vote_files = sorted(C.OUT_DIR.glob("labels_manual_v*.jsonl"))
    target = C.OUT_DIR / "labels_manual_v2.jsonl"
    if target.exists():
        vpaths = [target]
    elif vote_files:
        vpaths = vote_files
    else:
        return None
    try:
        v_dfs = [pd.read_json(p, lines=True) for p in vpaths]
        v = pd.concat(v_dfs, ignore_index=True)
        if "split" in v.columns:
            v = v[v.split == "train"]
        allv = pd.concat([
            tr[["frame_id", "key", "present"]],
            v[["frame_id", "key", "present"]],
        ], ignore_index=True)
        return allv.groupby(["frame_id", "key"], as_index=False).present.mean().assign(
            present=lambda d: (d.present >= 0.5).astype(int)
        )
    except (FileNotFoundError, ValueError, KeyError):
        return None


def main():
    idx = pd.read_csv(C.DATA_DIR / "index.csv")
    X = np.load(C.DATA_DIR / "features.npy")
    if len(X) != len(idx):
        raise ValueError(f"features.npy length ({len(X)}) != index.csv length ({len(idx)})")
    row_of = {f: i for i, f in enumerate(idx.frame_id)}
    split_of = dict(zip(idx.frame_id, idx.split))

    dec = pd.read_csv(C.OUT_DIR / "decisions.csv")
    if "split" not in dec.columns:
        dec["split"] = dec.frame_id.map(split_of)
    if "final_present" not in dec.columns:
        dec["final_present"] = dec["present"]
    dec = dec.drop_duplicates(subset=["frame_id", "key"], keep="last")

    # If labels_manual.jsonl in the same OUT_DIR has additional train rows not yet written to decisions.csv,
    # backfill their zero-label double-checked status while keeping decisions.csv authoritative on all shared rows.
    manual_p = C.OUT_DIR / "labels_manual.jsonl"
    if manual_p.exists() and manual_p.stat().st_size > 0:
        try:
            raw_m = pd.read_json(manual_p, lines=True)
            if len(raw_m) > len(dec) and {"frame_id", "key", "present"}.issubset(raw_m.columns):
                from eval.score_decisions import _initial_status, _initial_trust

                if "split" not in raw_m.columns:
                    raw_m["split"] = raw_m.frame_id.map(split_of)
                if "conf" not in raw_m.columns and "confidence" in raw_m.columns:
                    raw_m["conf"] = raw_m["confidence"]
                raw_m["trust"] = [_initial_trust(r) for r in raw_m.itertuples()]
                raw_m["status"] = raw_m["trust"].map(_initial_status)
                raw_m["final_present"] = raw_m["present"].astype(int)
                combined = pd.concat([raw_m, dec], ignore_index=True)
                dec = combined.drop_duplicates(subset=["frame_id", "key"], keep="last")
        except Exception:
            pass

    # Enforce strict split separation: student trains ONLY on 'train' split, evaluates ONLY on 'test' split
    tr = dec[dec.split == "train"].copy()
    te_idx = idx[idx.split == "test"].copy()
    Xte = X[te_idx.index.values]

    vote = _load_vote_source(tr)
    if vote is None:
        print("no vote labels found, skipping vote source")

    results, probs = {}, {"frame_id": te_idx.frame_id.values}
    models_by_key_source, const_by_key_source = {}, {}
    for key in C.LABELS:
        yte = te_idx[f"gt_{key}"].values
        is_informative = key not in UNINFORMATIVE_LABELS
        t = tr[tr.key == key]
        acc_mask = t.status.astype(str).str.lower().isin(["accept", "accepted"])
        sources = {
            "single": t[["frame_id", "present"]],
            "verified": t[acc_mask][["frame_id", "final_present"]].rename(columns={"final_present": "present"}),
            "oracle_upper_bound": idx[idx.split == "train"][["frame_id", f"gt_{key}"]].rename(columns={f"gt_{key}": "present"}),
        }
        if vote is not None:
            sources["vote"] = vote[vote.key == key][["frame_id", "present"]]
        results[key] = {}
        models_by_key_source[key] = {}
        const_by_key_source[key] = {}
        for name in [s for s in SOURCE_ORDER if s in sources]:
            s = sources[name]
            s = s[s.frame_id.isin(row_of)]
            rows = [row_of[f] for f in s.frame_id]
            y_vals = s.present.values
            r, p, m = fit_eval(
                X[rows],
                y_vals,
                Xte,
                yte,
                is_upper_bound=(name in UPPER_BOUND_SOURCES),
                informative=is_informative,
                return_model=True,
            )
            results[key][name] = r
            if r is not None:
                models_by_key_source[key][name] = m
                const_by_key_source[key][name] = float(s.present.values[0]) if len(s) else 0.0
            if p is not None:
                probs[f"{key}__{name}"] = p

    # Headline means exclude uninformative 100%-prevalence labels (liver) per audit rule
    summary, summary_f1, summary_auroc, summary_n_train = {}, {}, {}, {}
    summary_core3_acc, summary_core3_f1, summary_core3_auroc = {}, {}, {}
    summary_all6_non_headline = {}
    for name in SOURCE_ORDER:
        accs = [results[k][name]["acc"] for k in HEADLINE_LABELS if results.get(k, {}).get(name) and results[k][name].get("acc") is not None]
        f1s = [results[k][name]["f1"] for k in HEADLINE_LABELS if results.get(k, {}).get(name) and results[k][name].get("f1") is not None]
        aucs = [results[k][name]["auroc"] for k in HEADLINE_LABELS if results.get(k, {}).get(name) and results[k][name].get("auroc") is not None]
        ntr = [results[k][name]["n_train"] for k in HEADLINE_LABELS if results.get(k, {}).get(name) and results[k][name].get("n_train") is not None]
        c3_accs = [results[k][name]["acc"] for k in CORE_BALANCED_LABELS if results.get(k, {}).get(name) and results[k][name].get("acc") is not None]
        c3_f1s = [results[k][name]["f1"] for k in CORE_BALANCED_LABELS if results.get(k, {}).get(name) and results[k][name].get("f1") is not None]
        c3_aucs = [results[k][name]["auroc"] for k in CORE_BALANCED_LABELS if results.get(k, {}).get(name) and results[k][name].get("auroc") is not None]
        accs_all6 = [results[k][name]["acc"] for k in C.LABELS if results.get(k, {}).get(name) and results[k][name].get("acc") is not None]
        if accs:
            summary[name] = round(float(np.mean(accs)), 4)
        if f1s:
            summary_f1[name] = round(float(np.mean(f1s)), 4)
        if aucs:
            summary_auroc[name] = round(float(np.mean(aucs)), 4)
        if ntr:
            summary_n_train[name] = round(float(np.mean(ntr)), 1)
        if c3_accs:
            summary_core3_acc[name] = round(float(np.mean(c3_accs)), 4)
        if c3_f1s:
            summary_core3_f1[name] = round(float(np.mean(c3_f1s)), 4)
        if c3_aucs:
            summary_core3_auroc[name] = round(float(np.mean(c3_aucs)), 4)
        if accs_all6:
            summary_all6_non_headline[name] = round(float(np.mean(accs_all6)), 4)

    # Video-level cluster bootstrap 95% CIs (2,000 resamples across unseen test videos)
    bootstrap_ci_by_source = {}
    paired_bootstrap_vs_single = {}
    if "video" in te_idx.columns and len(te_idx) > 0:
        vids = list(pd.unique(te_idx.video))
        n_v = len(vids)
        vid_masks = [te_idx.video.values == v for v in vids]
        vid_counts = np.array([int(m.sum()) for m in vid_masks], dtype=float)
        rng = np.random.default_rng(C.SEED)
        draws = rng.integers(0, n_v, size=(2000, n_v))
        denom = vid_counts[draws].sum(axis=1)  # shape (2000,)

        boot_headline_draws = {}
        for name in [s for s in SOURCE_ORDER if s in summary]:
            per_k_draws = []
            for k in HEADLINE_LABELS:
                col = f"{k}__{name}"
                if col not in probs or not results.get(k, {}).get(name):
                    continue
                pred_k = (np.asarray(probs[col]) >= 0.5).astype(int)
                yte_k = te_idx[f"gt_{k}"].values.astype(int)
                corr_k = (pred_k == yte_k).astype(float)
                v_corr = np.array([float(corr_k[m].sum()) for m in vid_masks])
                k_draws = v_corr[draws].sum(axis=1) / np.maximum(denom, 1.0)
                per_k_draws.append(k_draws)
                results[k][name]["acc_ci_95"] = [
                    round(float(np.percentile(k_draws, 2.5)), 4),
                    round(float(np.percentile(k_draws, 97.5)), 4),
                ]
            if per_k_draws:
                mean_draws = np.mean(np.stack(per_k_draws, axis=0), axis=0)
                boot_headline_draws[name] = mean_draws
                bootstrap_ci_by_source[name] = {
                    "mean_acc": summary[name],
                    "ci_95": [
                        round(float(np.percentile(mean_draws, 2.5)), 4),
                        round(float(np.percentile(mean_draws, 97.5)), 4),
                    ],
                    "n_bootstrap": 2000,
                    "cluster_unit": "video",
                    "n_clusters": int(n_v),
                }
        if "verified" in boot_headline_draws and "single" in boot_headline_draws:
            diff_draws = boot_headline_draws["verified"] - boot_headline_draws["single"]
            paired_bootstrap_vs_single = {
                "delta_mean_acc": round(float(summary["verified"] - summary["single"]), 4),
                "ci_95": [
                    round(float(np.percentile(diff_draws, 2.5)), 4),
                    round(float(np.percentile(diff_draws, 97.5)), 4),
                ],
                "p_verified_gt_single": round(float((diff_draws > 0).mean()), 4),
                "n_bootstrap": 2000,
                "cluster_unit": "video",
            }

    out = {
        "mean_acc_by_source": summary,
        "mean_f1_by_source": summary_f1,
        "mean_auroc_by_source": summary_auroc,
        "mean_n_train_by_source": summary_n_train,
        "bootstrap_ci_by_source": bootstrap_ci_by_source,
        "video_bootstrap_ci": bootstrap_ci_by_source,
        "paired_bootstrap_verified_vs_single": paired_bootstrap_vs_single,
        "core_balanced_3_labels": CORE_BALANCED_LABELS,
        "core_balanced_3_mean_acc_by_source": summary_core3_acc,
        "core_balanced_3_mean_f1_by_source": summary_core3_f1,
        "core_balanced_3_mean_auroc_by_source": summary_core3_auroc,
        "headline_labels": HEADLINE_LABELS,
        "excluded_uninformative_labels": sorted(UNINFORMATIVE_LABELS),
        "excluded_label_note": (
            "liver is present in 100% of CholecSeg8k frames (carries zero discriminative information). "
            "It is reported in per_label['liver'] and excluded from headline mean_acc_by_source. "
            "On the test split, cystic_duct (1/280 pos) and blood (0/280 pos) test false-positive suppression "
            "(specificity), while grasper, l_hook, and gallbladder have balanced positive/negative test support."
        ),
        "non_headline_all6_mean_acc_by_source": summary_all6_non_headline,
        "evaluation_split": "test",
        "n_test_frames": int(len(te_idx)),
        "n_test_videos": int(te_idx.video.nunique()) if "video" in te_idx.columns else None,
        "upper_bounds": sorted(UPPER_BOUND_SOURCES),
        "upper_bound_note": (
            "oracle_upper_bound is trained on human ground-truth labels on train videos and is "
            "reported as an UPPER BOUND ONLY (not part of the zero-human-label claim)."
        ),
        "source_metadata": {k: v for k, v in SOURCE_META.items() if k in summary},
        "per_label": results,
        "external_test_evaluation": _eval_trained_models_on_external(models_by_key_source, const_by_key_source, C.DATA_DIR),
    }
    (C.OUT_DIR / "student.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(probs).to_csv(C.OUT_DIR / "student_probs.csv", index=False)

    print("=== Student Specialist Evaluation on Held-Out TEST Split (5 Informative Labels; liver excluded from headline) ===")
    for name in [s for s in SOURCE_ORDER if s in summary]:
        ub_tag = " [UPPER BOUND - human GT]" if name in UPPER_BOUND_SOURCES else " [zero human labels]"
        acc_v = summary.get(name, float("nan"))
        f1_v = summary_f1.get(name, float("nan"))
        auc_v = summary_auroc.get(name, float("nan"))
        ntr_v = summary_n_train.get(name, float("nan"))
        print(f"  {name:20s}{ub_tag:28s} | mean_acc={acc_v:.4f}  mean_f1={f1_v:.4f}  mean_auroc={auc_v:.4f}  mean_n_train={ntr_v:.1f}")
    print("\n--- Per-Label Test Accuracy (including uninformative liver reported separately) ---")
    for k in C.LABELS:
        tag = " [EXCLUDED FROM HEADLINE: 100% prevalence]" if k in UNINFORMATIVE_LABELS else ""
        row_str = "  ".join(
            f"{s}={results[k][s]['acc']:.4f} (n={results[k][s]['n_train']})"
            for s in SOURCE_ORDER if results.get(k, {}).get(s)
        )
        print(f"  {k:14s}: {row_str}{tag}")
    print(json.dumps(summary, indent=2))
    print("headline: does 'verified' beat 'single' (and 'vote') with fewer labels? (oracle_upper_bound = UPPER BOUND ONLY)")
    return out


if __name__ == "__main__":
    main()
