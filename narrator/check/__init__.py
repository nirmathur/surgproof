"""Claim checker: separate layers, each pass/fail/abstain, then a verdict per claim.

Verdict:
- drop:  any layer failed (decided_by = failing layers)
- hedge: no failure, and (a layer passed with hedge, or polarity hedged, or no evidence layer could
         support it but it is a rule-checked structural claim: phase / cvs_criterion / decision_point)
- speak: no failure and at least one evidence layer (L4, L5, L7, L5c) passed without hedge
- otherwise drop as "unsupported" (nothing could back it)
"""
from __future__ import annotations

import importlib

from narrator.check.base import Ctx

LAYER_MODULES = [("L1", "l1_schema"), ("L2", "l2_rules"), ("L3", "l3_temporal"), ("L4", "l4_evidence"),
                 ("P", "proportionality"), ("L5", "l5_blind"), ("L6", "l6_perturb"), ("L7", "l7_knowledge"),
                 ("L5c", "l5_clip")]
EVIDENCE_LAYERS = ["L4", "L5", "L7"]
# Report-only layers: run and stored, never decide a verdict. L5c (zero-shot OpenCLIP) measured at chance
# (52% on 200 GT test cases, DECISIONS.md), so it only feeds the error-correlation table.
REPORT_ONLY = {"L5c"}
STRUCTURAL = {"phase", "cvs_criterion", "decision_point"}


def available_layers(only: list[str] | None = None) -> list[tuple[str, object]]:
    out = []
    for name, mod in LAYER_MODULES:
        if only and name not in only:
            continue
        try:
            out.append((name, importlib.import_module(f"narrator.check.{mod}")))
        except Exception as e:  # noqa: BLE001  layer not built yet / broken: skip, report
            print(f"[check] layer {name} unavailable: {str(e)[:100]}")
    return out


def run_layers(claims: list[dict], ctx: Ctx, only: list[str] | None = None) -> dict[str, list[dict]]:
    results = {}
    for name, mod in available_layers(only):
        try:
            r = mod.run(claims, ctx)
            assert len(r) == len(claims), f"{name} returned {len(r)} results for {len(claims)} claims"
        except Exception as e:  # noqa: BLE001
            print(f"[check] layer {name} crashed: {str(e)[:160]}")
            r = [{"result": "abstain", "reason": f"layer crashed: {str(e)[:80]}", "hedge": False}] * len(claims)
        results[name] = r
        ctx.extra[name] = r
    return results


def verdict(claim: dict, layer_res: dict[str, dict]) -> tuple[str, list[str]]:
    layer_res = {n: r for n, r in layer_res.items() if n not in REPORT_ONLY}
    fails = [n for n, r in layer_res.items() if r and r["result"] == "fail"]
    if fails:
        return "drop", fails
    hedges = [n for n, r in layer_res.items() if r and r["result"] == "pass" and r.get("hedge")]
    support = [n for n in EVIDENCE_LAYERS if layer_res.get(n) and layer_res[n]["result"] == "pass"
               and not layer_res[n].get("hedge")]
    if hedges or claim.get("polarity") == "hedged":
        if support or hedges:
            return "hedge", hedges or ["polarity"]
    if support:
        return "speak", support
    if claim.get("type") in STRUCTURAL:
        return "hedge", ["no_evidence_layer"]
    return "drop", ["unsupported"]


def apply(claims: list[dict], ctx: Ctx, only: list[str] | None = None) -> list[dict]:
    results = run_layers(claims, ctx, only)
    out = []
    for i, c in enumerate(claims):
        lr = {n: results[n][i] for n in results}
        v, by = verdict(c, lr)
        row = dict(c)
        for n in results:
            row[n] = lr[n]["result"]
        row["reasons"] = {n: lr[n].get("reason", "") for n in results}
        row["hedges"] = [n for n in results if lr[n].get("hedge")]
        row["verdict"], row["decided_by"] = v, by
        out.append(row)
    return out
