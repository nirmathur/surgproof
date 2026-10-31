"""Eval items 2 + 3: injected wrong claims and error correlation between layers. Test videos only.

Clean base: correct presence/absence/phase/knowledge claims built from GT (inject.make_gold_claims) on test clips,
plus Gemini grounded claims that are GT-correct. Each injected error is checked together with its context claim
(e.g. the clip action before a "CVS achieved" claim); context claims ("-prior-") are not scored.
Outputs: out/eval/injection.csv, injection.png, layer_correlation.csv
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from narrator import cfg as C  # noqa: E402
from narrator import data as D  # noqa: E402
from narrator.check import apply  # noqa: E402
from narrator.check.__main__ import build_ctx  # noqa: E402
from narrator.eval import inject as INJ  # noqa: E402

EV = C.path("eval")
N_CLEAN = 240
PER_TYPE = 24


def mcc(tp, fp, fn, tn) -> float:
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return float((tp * tn - fp * fn) / d) if d else 0.0


def clean_base(seed=0) -> list[dict]:
    gt = D.build_gt()
    gt = gt[gt.split == "test"]
    clips = D.list_clips()
    gold = INJ.make_gold_claims(gt, clips[clips.split == "test"])
    # keep only claims whose correctness is established: GT presence/absence, and knowledge claims citing a
    # quote-verified KB rule. make_gold_claims' phase claims are NOT verifiable (CholecSeg8k has no phase labels).
    import json as _j
    qv = _j.loads((C.out_dir() / "kb_quotes.json").read_text()) if (C.out_dir() / "kb_quotes.json").exists() else {}
    gold = [c for c in gold if c.get("type") in ("presence", "absence")
            or (c.get("type") == "knowledge" and all(qv.get(k, {}).get("quote_verified") for k in c.get("kb_refs") or ["?"]))]
    rng = random.Random(seed)
    rng.shuffle(gold)
    gold = gold[:N_CLEAN]
    for i, c in enumerate(gold):
        c.setdefault("mode", "grounded")
        c["claim_id"] = f"gold-{i}-{c.get('claim_id', '')}"
        c["source"] = "gold"
    return gold


def absent_map() -> dict:
    gt = D.build_gt()
    gt = gt[(gt.split == "test") & gt.label.isin(C.cfg()["scored_classes"])]
    pres = gt.groupby(["clip_id", "label"]).present.any().reset_index()
    out = defaultdict(list)
    for r in pres[~pres.present].itertuples():
        out[r.clip_id].append(r.label)
    return dict(out)


def unit_of(cid: str) -> str:
    return cid.replace("-prior", "")


def run_eval(seed=0):
    ctx = build_ctx()
    clean = clean_base(seed)
    inj = INJ.inject(clean, seed=seed, per_type=PER_TYPE, absent_classes=absent_map(),
                     clip_bounds=ctx.extra["clip_bounds"])
    checked_clean = apply(clean, ctx)
    units = defaultdict(list)
    for c in inj:
        units[unit_of(c["claim_id"])].append(c)
    # check every unit with its own context; rule layers are cheap, LLM layers are cached per question
    flat = []
    from narrator.check import verdict
    # batch LLM layers once over all scored injected claims (no context needed for L5/L6/L7)
    scored = [next(c for c in u if "-prior-" not in c["claim_id"]) for u in units.values()]
    llm_res = apply(scored, ctx, only=["L5", "L6", "L7", "L5c"])
    llm_by_id = {r["claim_id"]: r for r in llm_res}
    for key, u in units.items():
        r = apply(u, ctx, only=["L1", "L2", "L3", "L4", "P"])
        for row in r:
            if "-prior-" in row["claim_id"]:
                continue
            lr = llm_by_id.get(row["claim_id"], {})
            for L in ("L5", "L6", "L7", "L5c"):
                if L in lr:
                    row[L] = lr[L]
                    row["reasons"][L] = lr["reasons"].get(L, "")
            # recompute verdict with all layers
            layer_res = {L: {"result": row[L], "hedge": L in (row.get("hedges") or []) or L in (lr.get("hedges") or [])}
                         for L in ("L1", "L2", "L3", "L4", "P", "L5", "L6", "L7", "L5c") if L in row}
            row["verdict"], row["decided_by"] = verdict(row, layer_res)
            flat.append(row)
    return checked_clean, flat


def metrics(clean: list[dict], injected: list[dict]) -> pd.DataFrame:
    layers = [L for L in ("L1", "L2", "L3", "L4", "P", "L5", "L6", "L7", "L5c") if any(L in c for c in clean + injected)]
    rows = []
    cdf = pd.DataFrame(clean)
    idf = pd.DataFrame(injected)
    for et, g in idf.groupby("injected_error"):
        for L in layers + ["full_stack"]:
            pred_inj = (g.verdict == "drop") if L == "full_stack" else (g.get(L) == "fail")
            pred_cln = (cdf.verdict == "drop") if L == "full_stack" else (cdf.get(L) == "fail")
            tp, fn = int(pred_inj.sum()), int((~pred_inj).sum())
            fp, tn = int(pred_cln.sum()), int((~pred_cln).sum())
            rows.append({"error_type": et, "layer": L, "n_injected": len(g), "n_clean": len(cdf),
                         "recall": tp / max(1, tp + fn),
                         # per-type precision with all clean false alarms pooled: lower bound, see RESULTS.md
                         "precision": tp / max(1, tp + fp) if tp + fp else np.nan,
                         "mcc": mcc(tp, fp, fn, tn), "tp": tp, "fp": fp,
                         "abstain_rate": float((g.get(L) == "abstain").mean()) if L != "full_stack" else 0.0})
    # overall
    for L in layers + ["full_stack"]:
        pred_inj = (idf.verdict == "drop") if L == "full_stack" else (idf.get(L) == "fail")
        pred_cln = (cdf.verdict == "drop") if L == "full_stack" else (cdf.get(L) == "fail")
        tp, fn, fp, tn = int(pred_inj.sum()), int((~pred_inj).sum()), int(pred_cln.sum()), int((~pred_cln).sum())
        rows.append({"error_type": "ALL", "layer": L, "n_injected": len(idf), "n_clean": len(cdf),
                     "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp) if tp + fp else np.nan,
                     "mcc": mcc(tp, fp, fn, tn), "tp": tp, "fp": fp,
                     "abstain_rate": float((idf.get(L) == "abstain").mean()) if L != "full_stack" else 0.0})
    t = pd.DataFrame(rows)
    fpr = {L: float(((cdf.verdict == "drop") if L == "full_stack" else (cdf.get(L) == "fail")).mean())
           for L in layers + ["full_stack"]}
    t["clean_false_drop_rate"] = t.layer.map(fpr)
    return t


def correlation(rows: list[dict], is_error, source: str) -> pd.DataFrame:
    """For each layer pair, on erroneous claims where both layers gave a verdict (not abstain):
    observed P(both miss) vs P(A miss) * P(B miss) under independence. Miss = layer did not fail an error."""
    df = pd.DataFrame(rows)
    df = df[df.apply(is_error, axis=1)]
    layers = [L for L in ("L1", "L2", "L3", "L4", "P", "L5", "L6", "L7", "L5c") if L in df]
    out = []
    for a, b in combinations(layers, 2):
        s = df[df[a].isin(["pass", "fail"]) & df[b].isin(["pass", "fail"])]
        if len(s) < 5:
            continue
        ma, mb = s[a] == "pass", s[b] == "pass"
        both, ind = float((ma & mb).mean()), float(ma.mean() * mb.mean())
        phi = float(np.corrcoef(ma, mb)[0, 1]) if ma.std() > 0 and mb.std() > 0 else np.nan
        out.append({"source": source, "layer_a": a, "layer_b": b, "n": len(s), "miss_a": float(ma.mean()),
                    "miss_b": float(mb.mean()), "both_miss": both, "independent_both_miss": ind,
                    "ratio": both / ind if ind else np.nan, "phi": phi})
    return pd.DataFrame(out)


def chart(t: pd.DataFrame, mock: bool):
    piv = t[t.layer != "full_stack"].pivot(index="error_type", columns="layer", values="recall")
    full = t[t.layer == "full_stack"].set_index("error_type").recall
    piv["full_stack"] = full
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v > 0.6 else "#1f2328")
    ax.set_title("Recall on injected wrong claims, per layer and full stack (test videos)", loc="left", fontsize=11)
    note = "Synthetic errors on correct claims. L4 uses a MOCK labeller table." if mock else "Synthetic errors on correct claims."
    fig.text(0.01, 0.01, note, fontsize=8, color="#6e7781")
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(EV / "injection.png", dpi=150); plt.close(fig)


def drop_accidentally_true(injected: list[dict]) -> tuple[list[dict], int]:
    """An injected claim that GT says is TRUE (e.g. grasper->hook swap while a hook is in view) is not an error."""
    from narrator.eval import score
    df = pd.DataFrame(injected)
    if df.empty:
        return injected, 0
    tmp = df[["claim_id", "clip_id", "t_start", "t_end", "type", "modality", "entities", "polarity"]].copy()
    tmp["entities"] = tmp.entities.apply(lambda e: json.dumps(e) if not isinstance(e, str) else e)
    ann = score.annotate(tmp)
    true_ids = set(ann[ann.gt_correct == True].claim_id)  # noqa: E712
    return [c for c in injected if c["claim_id"] not in true_ids], len(true_ids)


def main(seed=0):
    clean, injected = run_eval(seed)
    injected, n_true = drop_accidentally_true(injected)
    print(f"[injection] excluded {n_true} injected claims that GT says are actually true")
    t = metrics(clean, injected)
    mock = str(D.load_labels().source.iloc[0]) == "mock"
    t["l4_inputs"] = "mock" if mock else "real"
    t.to_csv(EV / "injection.csv", index=False)
    chart(t, mock)
    corr = [correlation(injected, lambda r: True, "injected_errors")]
    scored_p = C.out_dir() / "claims_scored.parquet"
    if scored_p.exists():
        s = pd.read_parquet(scored_p)
        s = s[s.gt_correct.notna()]
        corr.append(correlation(s.to_dict("records"), lambda r: r["gt_correct"] is False or r["gt_correct"] == 0,
                                "real_gemini_claims_wrong_vs_gt"))
    corr = pd.concat([c for c in corr if len(c)], ignore_index=True) if any(len(c) for c in corr) else pd.DataFrame()
    corr.to_csv(EV / "layer_correlation.csv", index=False)
    pd.DataFrame(injected).assign(**{k: lambda d, k=k: d[k].apply(json.dumps) for k in ("entities", "reasons", "decided_by")}) \
        .drop(columns=[c for c in ("evidence_refs", "kb_refs", "hedges") if c in pd.DataFrame(injected)]) \
        .to_parquet(C.out_dir() / "injected_checked.parquet", index=False)
    print(t[t.error_type == "ALL"].to_string())
    print(t[t.layer == "full_stack"][["error_type", "recall", "precision", "mcc"]].to_string())
    if len(corr):
        print(corr.to_string())


if __name__ == "__main__":
    main()
