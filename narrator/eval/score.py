"""Eval item 1: free vs grounded against CholecSeg8k masks (test videos only).

Precision and hallucination count ASSERTED claims (speak, or unhedged in raw mode); hedged claims are scored by
hedge_correctness (hedged item really uncertain in the label table, or absent in GT).
Scored claim types: presence / absence of the scored classes (grasper, l_hook, liver, gallbladder,
cystic_duct, blood). GT: class covers > presence.min_frac of pixels in any sampled GT frame inside the
claim window (nearest sampled frame if the window holds none).

Conditions: free_raw (everything the baseline said), free_checked (baseline after our checker, speak+hedge),
grounded_raw, grounded_checked (what the narrator actually says).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from narrator import cfg as C
from narrator import data as D
from narrator.schema import scored_class


def gt_lookup() -> dict:
    gt = D.build_gt()
    gt = gt[gt.label.isin(C.cfg()["scored_classes"])]
    return {key: g for key, g in gt.groupby(["clip_id", "label"])}


def gt_present(lk: dict, clip_id: str, cls: str, t0: int, t1: int):
    g = lk.get((clip_id, cls))
    if g is None:
        return None
    w = g[g.frame_idx.between(t0, t1)]
    if w.empty:
        w = g.iloc[[int((g.frame_idx - (t0 + t1) / 2).abs().argmin())]]
    return bool(w.present.any())


def annotate(df: pd.DataFrame) -> pd.DataFrame:
    """Add scored_class and gt_correct (None if not scorable)."""
    lk = gt_lookup()
    sc, ok = [], []
    for c in df.to_dict("records"):
        e = json.loads(c["entities"]) if isinstance(c["entities"], str) else c["entities"]
        cls = scored_class({"entities": e})
        if cls is None or c["type"] not in ("presence", "absence", "finding", "action") or c.get("modality") != "observed":
            sc.append(None); ok.append(None); continue
        p = gt_present(lk, c["clip_id"], cls, int(c["t_start"]), int(c["t_end"]))
        if p is None:
            sc.append(None); ok.append(None); continue
        neg = c["type"] == "absence" or c.get("polarity") == "negated"
        sc.append(cls)
        ok.append((not p) if neg else p)
    df = df.copy()
    df["scored_class"], df["gt_correct"] = sc, ok
    return df


def condition_metrics(d: pd.DataFrame, clips: list[str], lk: dict, labels: pd.DataFrame) -> dict:
    s = d[d.gt_correct.notna()]
    asserted = s[s.verdict_eff != "hedge"]          # what the narrator states as fact
    pos_all = s[~((s.type == "absence") | (s.polarity == "negated"))]
    pos = asserted[~((asserted.type == "absence") | (asserted.polarity == "negated"))]
    halluc = pos[pos.gt_correct == False]  # noqa: E712  hedged claims are scored by hedge_correctness instead
    n_clips = max(1, len(clips))
    # coverage: share of (clip, present scored class) pairs mentioned by any positive claim
    present_pairs = {(c, k[1]) for k, g in lk.items() for c in [k[0]] if c in clips and g.present.any()}
    mentioned = {(r.clip_id, r.scored_class) for r in pos_all.itertuples()}
    hed = s[s.verdict_eff == "hedge"]
    hedge_ok = []
    for r in hed.itertuples():
        lab = labels[(labels.clip_id == r.clip_id) & (labels.label == r.scored_class)
                     & labels.frame_idx.between(r.t_start, r.t_end)]
        uncertain = bool((lab.status == "uncertain").any()) if len(lab) else False
        hedge_ok.append(uncertain or not r.gt_correct)
    return {
        "n_claims": int(len(d)), "n_scored": int(len(s)), "n_scored_asserted": int(len(asserted)),
        "claim_precision": float(asserted.gt_correct.mean()) if len(asserted) else np.nan,
        "hallucinated_structures_per_clip": len({(r.clip_id, r.scored_class) for r in halluc.itertuples()}) / n_clips,
        "hallucinated_claims_per_clip": len(halluc) / n_clips,
        "hallucination_rate": float((pos.gt_correct == False).mean()) if len(pos) else np.nan,  # noqa: E712
        "coverage": len(mentioned & present_pairs) / max(1, len(present_pairs)),
        "n_hedged_scored": int(len(hed)),
        "hedge_correctness": float(np.mean(hedge_ok)) if hedge_ok else np.nan,
    }


LAYERS = ["L1", "L2", "L3", "L4", "P", "L5", "L6", "L7", "L5c"]


def verdict_without(df: pd.DataFrame, drop_layers: set) -> pd.Series:
    """Recompute the verdict from stored layer results with some layers removed (ablation)."""
    from narrator.check import verdict
    out = []
    for c in df.to_dict("records"):
        hedges = set(json.loads(c["hedges"])) if isinstance(c.get("hedges"), str) else set(c.get("hedges") or [])
        lr = {L: {"result": c[L], "hedge": L in hedges} for L in LAYERS
              if L in c and isinstance(c[L], str) and c[L] and L not in drop_layers}
        out.append(verdict(c, lr)[0])
    return pd.Series(out, index=df.index)


def free_vs_grounded(df: pd.DataFrame) -> pd.DataFrame:
    df = annotate(df[df.split == "test"])
    labels = D.load_labels()
    lk = gt_lookup()
    rows = []
    for mode in ("free", "grounded"):
        d = df[df["mode"] == mode]
        clips = sorted(d.clip_id.unique())
        raw = d.assign(verdict_eff=np.where(d.polarity == "hedged", "hedge", "speak"))
        chk = d[d.verdict.isin(["speak", "hedge"])].assign(verdict_eff=lambda x: x.verdict)
        v_ind = verdict_without(d, {"L4"})  # no label-table evidence: rules + blind visual + perturbation + KB
        ind = d[v_ind.isin(["speak", "hedge"])].assign(verdict_eff=v_ind[v_ind.isin(["speak", "hedge"])])
        for cond, dd in ((f"{mode}_raw", raw), (f"{mode}_checked_noL4", ind), (f"{mode}_checked", chk)):
            m = condition_metrics(dd, clips, lk, labels)
            rows.append({"condition": cond, "n_clips": len(clips), **m})
    return pd.DataFrame(rows), df
