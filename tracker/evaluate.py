"""Step 7: evaluate signals against injected errors; fit the suspicion model.

Unit of decision: one (frame, tool label) row. An error counts as found if any of its rows is flagged / in budget.
- signal_table.csv: per signal precision (rows) and recall by error type (errors), on test videos
- logistic regression over all signals fit on tuning videos -> out/tracker/suspicion_model.json
- budget_curve.png/csv: share of injected errors found vs re-check budget on test videos
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from tracker.common import KEYS, Timer, cli, load_cfg, p, work

Z_CLIP = 20.0


def feature_names(cfg):
    s = cfg["signals"]
    return [f"z_{k}" for k in s["zscore"]] + list(s["raw"]) + ["present"]


def features(rows, cfg):
    X = pd.DataFrame(index=rows.index)
    for k in cfg["signals"]["zscore"]:
        X[f"z_{k}"] = rows[f"z_{k}"].fillna(0.0).clip(-Z_CLIP, Z_CLIP)
    for k in cfg["signals"]["raw"]:
        X[k] = rows[k].fillna(0.0)
    X["present"] = (rows.n_det > 0).astype(float)
    return X[feature_names(cfg)]


def save_model(path, lr, mu, sd, names):
    json.dump({"features": names, "mean": mu.tolist(), "scale": sd.tolist(),
               "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}, open(path, "w"), indent=1)


def load_model(path):
    return json.load(open(path))


def score(rows, cfg, model):
    """Suspicion = P(error) from the saved logistic model, plus per-feature contributions (coef * standardised x)."""
    X = features(rows, cfg)[model["features"]].values
    Z = (X - np.array(model["mean"])) / np.array(model["scale"])
    contrib = Z * np.array(model["coef"])
    logit = contrib.sum(1) + model["intercept"]
    return 1 / (1 + np.exp(-logit)), pd.DataFrame(contrib, columns=model["features"], index=rows.index)


def flagged(rows, cfg):
    s = cfg["signals"]
    F = {}
    for k in s["zscore"]:
        F[k] = rows[f"z_{k}"].fillna(0) > s["z_thresh"]
    for k in s["raw"]:
        F[k] = rows[k].fillna(0) > s["binary_thresh"]
    return pd.DataFrame(F)


def label_rows(rows, err):
    e = err.drop_duplicates(KEYS + ["error_id"])
    rows = rows.reset_index(drop=True)
    m = rows[KEYS].reset_index().merge(e[KEYS + ["error_id", "error_type"]], on=KEYS)
    y = np.zeros(len(rows), bool)
    y[m["index"].values] = True
    return rows, y, m  # m: row index <-> error_id mapping


def found_curve(order_idx, m, n_rows, budgets, err_ids):
    """Share of errors (err_ids) with at least one row among the first b*n_rows of order_idx."""
    rank = np.empty(n_rows, int)
    rank[order_idx] = np.arange(n_rows)
    first = m.assign(r=rank[m["index"].values]).groupby("error_id").r.min().reindex(err_ids).fillna(np.inf).values
    return np.array([(first < b * n_rows).mean() for b in budgets])


def order_by(values, rng):
    v = np.nan_to_num(np.asarray(values, float), nan=-np.inf)
    return np.lexsort((rng.random(len(v)), -v))  # descending, random tie-break


def run(cfg):
    out = p(cfg, "eval")
    rows = pd.read_parquet(work(cfg, "signals_gt_corrupted.parquet"))
    err = pd.read_parquet(work(cfg, "errors.parquet"))
    rows, y, m = label_rows(rows, err)
    etype = err.drop_duplicates("error_id").set_index("error_id").error_type
    tune = (rows.split == "tune").values
    test = ~tune
    print(f"rows: tune {tune.sum()}, test {test.sum()}; error rows {y.mean():.1%}; errors tune/test "
          f"{m[tune[m['index']]].error_id.nunique()}/{m[test[m['index']]].error_id.nunique()}")

    # --- per-signal table on test rows
    F = flagged(rows, cfg)
    mt = m[test[m["index"].values]]
    recs = []
    for k in F.columns:
        f = F[k].values
        hit = mt[f[mt["index"].values]].error_id.unique()
        r = {"signal": k, "flag_rate": f[test].mean(), "precision": y[test & f].mean() if (test & f).any() else np.nan}
        for t in sorted(etype.unique()):
            ids = mt.error_id[mt.error_type == t].unique()
            r[f"recall_{t}"] = np.isin(ids, hit).mean() if len(ids) else np.nan
        r["recall_all"] = np.isin(mt.error_id.unique(), hit).mean()
        recs.append(r)
    tab = pd.DataFrame(recs).sort_values("recall_all", ascending=False)
    tab.to_csv(out / "signal_table.csv", index=False, float_format="%.3f")
    print(tab.round(3).to_string(index=False))

    # --- logistic regression on tuning rows
    X = features(rows, cfg)
    mu, sd = X[tune].mean().values, X[tune].std().replace(0, 1).values
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", random_state=cfg["seed"])
    lr.fit((X[tune].values - mu) / sd, y[tune])
    mpath = work(cfg, "suspicion_model.json")
    save_model(mpath, lr, mu, sd, list(X.columns))
    sus, _ = score(rows, cfg, load_model(mpath))
    model = load_model(mpath)
    model["flag_threshold"] = float(np.quantile(sus[tune], cfg["recheck"]["flag_quantile"]))
    json.dump(model, open(mpath, "w"), indent=1)
    auc_tune, auc_test = roc_auc_score(y[tune], sus[tune]), roc_auc_score(y[test], sus[test])
    coefs = pd.Series(lr.coef_[0], index=X.columns).sort_values(ascending=False)
    coefs.to_csv(out / "suspicion_coefs.csv", header=["coef"])
    print(f"suspicion AUC tune {auc_tune:.3f} test {auc_test:.3f}; top coefs {coefs.head(6).round(2).to_dict()}")

    # --- budget curves on test
    budgets = np.linspace(0, cfg["eval"]["budgets_max"], cfg["eval"]["n_budget_points"])
    rng = np.random.default_rng(cfg["seed"])
    ti = np.where(test)[0]
    remap = -np.ones(len(rows), int)
    remap[ti] = np.arange(len(ti))
    mT = mt.assign(index=remap[mt["index"].values])
    ids = mT.error_id.unique()
    curves = {"targeted (suspicion)": found_curve(order_by(sus[test], rng), mT, len(ti), budgets, ids)}
    curves["random"] = np.mean([found_curve(rng.permutation(len(ti)), mT, len(ti), budgets, ids)
                                for _ in range(cfg["eval"]["random_repeats"])], 0)
    # best single signal: chosen on tuning rows by area under its curve
    ui = np.where(tune)[0]
    remap_u = -np.ones(len(rows), int)
    remap_u[ui] = np.arange(len(ui))
    mU = m[tune[m["index"].values]]
    mU = mU.assign(index=remap_u[mU["index"].values])
    ids_u = mU.error_id.unique()
    cand = [c for c in X.columns if c != "present" and c.replace("z_", "") not in cfg["signals"]["quality"]]
    area = {c: found_curve(order_by(X[c].values[tune], rng), mU, len(ui), budgets, ids_u).mean() for c in cand}
    best = max(area, key=area.get)
    curves[f"best single signal ({best})"] = found_curve(order_by(X[best].values[test], rng), mT, len(ti), budgets, ids)
    cdf = pd.DataFrame(curves, index=pd.Index(budgets, name="budget"))
    cdf.to_csv(out / "budget_curve.csv", float_format="%.4f")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    styles = {"targeted (suspicion)": dict(color="#1f5fbf", lw=2.6), "random": dict(color="#888888", lw=1.6, ls="--")}
    for name, v in curves.items():
        ax.plot(budgets * 100, v * 100, label=name, **styles.get(name, dict(color="#d9822b", lw=1.8)))
    for b in (0.05, 0.10, 0.20):
        i = int(np.argmin(np.abs(budgets - b)))
        ax.annotate(f"{curves['targeted (suspicion)'][i]:.0%}", (b * 100, curves["targeted (suspicion)"][i] * 100),
                    textcoords="offset points", xytext=(-6, 7), fontsize=9, color="#1f5fbf")
    ax.set(xlabel="Re-check budget (% of frame-labels in test videos)", ylabel="Injected errors found (%)",
           xlim=(0, budgets[-1] * 100), ylim=(0, 100),
           title=f"Targeted re-checks find errors faster than random (test videos, n={len(ids)} errors)")
    ax.title.set_fontsize(10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "budget_curve.png", dpi=150)

    # recall by error type at 10% budget for the targeted ranking
    k = int(0.10 * len(ti))
    top = set(order_by(sus[test], rng)[:k])
    by = mT.assign(hit=mT["index"].isin(top)).groupby(["error_type", "error_id"]).hit.any().groupby("error_type").mean()
    by.to_csv(out / "recall_by_type_at_10pct.csv", header=["recall"])
    summary = {"auc_tune": auc_tune, "auc_test": auc_test, "best_single_signal": best,
               **{f"found_at_{int(b*100)}pct_{n.split(' ')[0]}": float(cdf.loc[cdf.index[np.argmin(np.abs(budgets - b))], n])
                  for b in (0.05, 0.10, 0.20) for n in curves},
               "recall_by_type_at_10pct": by.round(3).to_dict()}
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    cli("Evaluate signals on corrupted data", source=False).parse_args()
    with Timer("evaluate"):
        run(load_cfg())
