"""python -m narrator.eval  -> narrator/out/eval/*.csv, *.png (test videos only)

1. free vs grounded vs masks (score.py)
2. injected wrong claims, recall/precision/MCC per error type per layer + full stack (injection.py)
3. error correlation between layers (injection.py)
4. cost and latency per clip
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from narrator import cfg as C  # noqa: E402
from narrator import data as D  # noqa: E402
from narrator.eval import score  # noqa: E402

EV = C.path("eval")
EV.mkdir(parents=True, exist_ok=True)
INK, MUTED, GRID = "#1f2328", "#6e7781", "#e6e8eb"
COL = {"free_raw": "#c4553a", "free_checked_noL4": "#d98a76", "free_checked": "#ecbcb0",
       "grounded_raw": "#9db4d6", "grounded_checked_noL4": "#5d82b8", "grounded_checked": "#2f5d9e"}


def label_source() -> str:
    lab = D.load_labels()
    return str(lab.source.iloc[0]) if len(lab) else "none"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)


def chart_free_vs_grounded(t: pd.DataFrame, mock: bool):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, title) in zip(axes, [("hallucination_rate", "Hallucination rate (lower is better)"),
                                       ("hallucinated_structures_per_clip", "Hallucinated structures per clip"),
                                       ("claim_precision", "Claim precision (higher is better)")]):
        vals = t.set_index("condition")[col]
        bars = ax.bar(range(len(vals)), vals.values, color=[COL.get(c, MUTED) for c in vals.index], width=0.65)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9, color=INK)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([c.replace("_", "\n") for c in vals.index], fontsize=8)
        ax.set_title(title, fontsize=10, color=INK, loc="left"); _style(ax)
    n = int(t.n_clips.max())
    sub = f"Test videos (6), n={n} clips per mode. Scored vs CholecSeg8k masks."
    if mock:
        sub += (" '*_checked' uses L4 on a MOCK label table (GT + noise): circular, NOT a real result."
                " '*_checked_noL4' uses only rules + blind visual (L5) + perturbation (L6) + KB (L7): independent of GT.")
    fig.suptitle("Free VLM narration vs grounded, checked narration", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.90, sub, fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(EV / "hallucination.png", dpi=150); plt.close(fig)


def cost_latency() -> pd.DataFrame:
    log = C.path("cache") / "calls.jsonl"
    calls = pd.DataFrame([json.loads(l) for l in log.read_text().splitlines() if l.strip()]) if log.exists() else pd.DataFrame()
    rows = []
    for mode, tags in (("free", ["free_narrate", "free_extract"]), ("grounded", ["grounded"])):
        gen = C.out_dir() / f"gen_{mode}.jsonl"
        g = pd.DataFrame([json.loads(l) for l in gen.read_text().splitlines()]) if gen.exists() else pd.DataFrame()
        c = calls[calls.tag.isin(tags) & calls.get("ok", True).fillna(False).astype(bool)] if len(calls) else calls
        n = max(1, len(g))
        rows.append({"mode": mode, "n_clips": len(g), "calls_per_clip": len(c) / n if len(c) else None,
                     "cost_usd_per_clip": float(c.cost.sum()) / n if len(c) else None,
                     "tokens_in_per_clip": float(c["in"].sum()) / n if len(c) else None,
                     "latency_s_per_clip": float(g.latency.mean()) if "latency" in g and len(g) else None,
                     "note": "generation only; checker cost reported separately"})
    if len(calls):
        for tag in ("l5", "l6", "l7"):
            c = calls[calls.tag.astype(str).str.lower().str.startswith(tag)]
            if len(c):
                rows.append({"mode": f"checker_{tag}", "n_clips": None, "calls_per_clip": None,
                             "cost_usd_per_clip": None, "tokens_in_per_clip": None, "latency_s_per_clip": None,
                             "note": f"{len(c)} calls, ${c.cost.sum():.3f} total, mean latency {c.latency.mean():.1f}s"})
    return pd.DataFrame(rows)


def main():
    claims = pd.read_parquet(C.out_dir() / "claims.parquet")
    mock = label_source() == "mock"
    t, ann = score.free_vs_grounded(claims)
    t["label_source"] = label_source()
    t.to_csv(EV / "free_vs_grounded.csv", index=False)
    ann.to_parquet(C.out_dir() / "claims_scored.parquet", index=False)
    chart_free_vs_grounded(t, mock)
    print(t.to_string())
    cl = cost_latency(); cl.to_csv(EV / "cost_latency.csv", index=False); print(cl.to_string())
    try:
        from narrator.eval import injection
        injection.main()
    except Exception as e:  # noqa: BLE001
        print(f"[eval] injection eval unavailable: {e}")


if __name__ == "__main__":
    main()
