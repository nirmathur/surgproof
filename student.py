"""Step 6: the system teaches itself a specialist. Same student, three label sources:
  single   - every Gemini label on train videos (more data, noisier)
  vote     - majority of 3 samples (needs labels_manual_v2.jsonl)
  verified - only labels the system accepted (fewer, cleaner)
  oracle   - human ground truth (UPPER BOUND ONLY, not part of the claim)
Evaluated on held-out test videos against hidden ground truth.

  python student.py            # writes out/student.json and out/student_probs.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

import config as C


def fit_eval(X, y, Xte, yte):
    if len(np.unique(y)) < 2:
        return None, None
    m = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5).fit(X, y)
    p = m.predict_proba(Xte)[:, 1]
    out = {"n_train": int(len(y)), "acc": round(float(((p >= 0.5) == yte).mean()), 4),
           "f1": round(float(f1_score(yte, p >= 0.5, zero_division=0)), 4)}
    if len(np.unique(yte)) == 2:
        out["auroc"] = round(float(roc_auc_score(yte, p)), 4)
    return out, p


def main():
    idx = pd.read_csv(C.DATA_DIR / "index.csv")
    X = np.load(C.DATA_DIR / "features.npy")
    row_of = {f: i for i, f in enumerate(idx.frame_id)}
    dec = pd.read_csv(C.OUT_DIR / "decisions.csv")
    tr = dec[dec.split == "train"]
    te_idx = idx[idx.split == "test"]
    Xte = X[te_idx.index]

    vote = None
    try:
        v = pd.read_json(C.OUT_DIR / "labels_manual_v2.jsonl", lines=True)
        allv = pd.concat([tr[["frame_id", "key", "present"]], v[v.split == "train"][["frame_id", "key", "present"]]])
        vote = allv.groupby(["frame_id", "key"]).present.mean().ge(0.5).astype(int).reset_index()
    except (FileNotFoundError, ValueError):
        print("no vote labels found, skipping vote source")

    results, probs = {}, {"frame_id": te_idx.frame_id.values}
    for key in C.LABELS:
        yte = te_idx[f"gt_{key}"].values
        t = tr[tr.key == key]
        sources = {
            "single": t[["frame_id", "present"]],
            "verified": t[t.status == "accept"][["frame_id", "final_present"]].rename(columns={"final_present": "present"}),
            "oracle_upper_bound": idx[idx.split == "train"][["frame_id", f"gt_{key}"]].rename(columns={f"gt_{key}": "present"}),
        }
        if vote is not None:
            sources["vote"] = vote[vote.key == key][["frame_id", "present"]]
        results[key] = {}
        for name, s in sources.items():
            rows = [row_of[f] for f in s.frame_id]
            r, p = fit_eval(X[rows], s.present.values, Xte, yte)
            results[key][name] = r
            if p is not None:
                probs[f"{key}__{name}"] = p
    summary = {}
    for name in ["single", "vote", "verified", "oracle_upper_bound"]:
        vals = [results[k][name]["acc"] for k in C.LABELS if results[k].get(name)]
        if vals:
            summary[name] = round(float(np.mean(vals)), 4)
    out = {"mean_acc_by_source": summary, "per_label": results}
    (C.OUT_DIR / "student.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(probs).to_csv(C.OUT_DIR / "student_probs.csv", index=False)
    print(json.dumps(summary, indent=2))
    print("headline: does 'verified' beat 'single' (and 'vote') with fewer labels?")


if __name__ == "__main__":
    main()
