"""Step 5: decide which labels to trust, send the uncertain band to Argon, and write decisions.csv.

Zero-label rule (no human labels used): trust score from the checks, thresholds fixed in advance.
Statuses emitted: 'accept' (verified), 'AMBER' (abstain -> Human Gate), 'reject' (rejected).
Post-decision evaluation runs separately via eval.score_decisions after decisions.csv is saved.

  python accept.py            # writes out/decisions.csv and out/results.json
  python accept.py --no-argon # skip the Argon reviser (faster)
"""
import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

import config as C
import gem
from eval.score_decisions import evaluate_decisions
from verify.engine import (
    ACCEPT_THRESHOLD,
    ARGON_ACCEPT_CONF,
    DECISION_OUTPUT_COLS,
    HEADLINE_KEYS,
    REJECT_THRESHOLD,
    VERIFIER_INPUT_COLS,
    Verdict,
    _pv_ok,
    _safe_prob,
    apply_argon_verdict,
    argon_revise as _engine_argon_revise,
    enrich_results_excl_liver,
    prepare_verifier_dataframe,
    status,
    summarize_decisions_zero_label,
    trust,
)


async def argon_revise(r, manual):
    """Escalate an uncertain claim to Gemini 4 Argon (`accept.gem.call` compatible)."""
    return await _engine_argon_revise(r, manual, call_fn=gem.call)


def _resolve_labels_path(labels_path: Path | str | None = None) -> Path:
    if labels_path is not None:
        p = Path(labels_path)
        if p.exists():
            return p
        shared_root = C.OUT_DIR.resolve().parent
        alt = shared_root / labels_path
        if alt.exists():
            return alt
        return p
    primary = C.OUT_DIR / "labels_manual.jsonl"
    if primary.exists() and primary.stat().st_size > 0:
        return primary
    for root_cand in (C.ROOT, C.OUT_DIR.resolve().parent):
        smoke = root_cand / "coord" / "artifacts" / "labels_smoke.jsonl"
        if smoke.exists() and smoke.stat().st_size > 0:
            return smoke
    return primary


async def main(
    use_argon: bool = True,
    argon_budget: int = 300,
    *,
    labels_path: Path | str | None = None,
    out_path: Path | str | None = None,
    index_path: Path | str | None = None,
    skip_eval: bool = False,
):
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")
    idx_df = pd.read_csv(idx_file) if idx_file.exists() else pd.DataFrame()

    lbl_file = _resolve_labels_path(labels_path)
    raw_d = pd.read_json(lbl_file, lines=True)
    d = prepare_verifier_dataframe(raw_d, idx_df)

    if use_argon and argon_budget > 0 and not d.empty:
        man_path = C.OUT_DIR / "manual.json"
        if not man_path.exists():
            alt_man = C.ROOT / "coord" / "artifacts" / "manual.json"
            if alt_man.exists():
                man_path = alt_man
        manual = json.loads(man_path.read_text()) if man_path.exists() else {}
        amber_rows = d[d.status == "AMBER"].copy()
        amber_rows["_is_headline"] = amber_rows["key"].isin(HEADLINE_KEYS).astype(int)
        amber_rows = amber_rows.sort_values(["_is_headline", "trust"], ascending=[False, False])
        n_review = min(int(argon_budget), max(1, (len(amber_rows) + 1) // 2)) if len(amber_rows) > 1 else len(amber_rows)
        band = amber_rows.head(n_review)
        gem.trace(
            "argon",
            f"Reviewing {len(band)} uncertain labels ({(d.status != 'accept').sum()} flagged in total)",
        )
        if not band.empty:
            verdicts = await asyncio.gather(*[argon_revise(r, manual) for r in band.itertuples()])
            for (i, _), v in zip(band.iterrows(), verdicts):
                apply_argon_verdict(d, i, v)

    dest = Path(out_path) if out_path is not None else (C.OUT_DIR / "decisions.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest, index=False)

    # Mirror to coord/artifacts/decisions.csv ONLY when writing to the real shared out/decisions.csv
    real_out_dec = (C.ROOT / "out" / "decisions.csv").resolve()
    shared_roots = {C.ROOT.resolve(), C.OUT_DIR.resolve().parent}
    if dest.resolve() == real_out_dec:
        for s_root in shared_roots:
            coord_dec = s_root / "coord" / "artifacts" / "decisions.csv"
            coord_dec.parent.mkdir(parents=True, exist_ok=True)
            d.to_csv(coord_dec, index=False)

    if skip_eval or dest.resolve() != (C.OUT_DIR / "decisions.csv").resolve():
        return summarize_decisions_zero_label(d)

    res = evaluate_decisions(C.DATA_DIR, C.OUT_DIR)
    res = enrich_results_excl_liver(res, d)
    res_path = C.OUT_DIR / "results.json"
    res_path.write_text(json.dumps(res, indent=2))
    if dest.resolve() == real_out_dec:
        for s_root in shared_roots:
            coord_res = s_root / "coord" / "artifacts" / "results.json"
            coord_res.parent.mkdir(parents=True, exist_ok=True)
            coord_res.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-argon", action="store_true")
    ap.add_argument("--argon-budget", type=int, default=300)
    ap.add_argument("--labels", type=str, default=None, help="Optional input JSONL path")
    ap.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    ap.add_argument("--skip-eval", action="store_true", help="Skip post-decision evaluation")
    a = ap.parse_args()
    asyncio.run(
        main(
            not a.no_argon,
            a.argon_budget,
            labels_path=a.labels,
            out_path=a.out,
            skip_eval=a.skip_eval,
        )
    )
