"""Step 5: double-check every label, run the N-round Argon Amber Loop, and write decisions.csv.

Zero-label rule (no human labels used):
1. Every label gets an independent second pass (blind crop + sceptical critic).
   Both agree -> 'accept' or 'reject'; disagree or uncertain -> 'AMBER'.
2. Amber Loop: Argon re-examines each AMBER for up to N rounds (N=3 default: zoomed crop +
   manual criteria -> tighter crop + differential diagnosis -> multi-vote consensus adjudication),
   logging every round to out/amber_rounds.jsonl and shrinking AMBERs before the Human Gate.
3. Post-decision evaluation runs separately via eval.score_decisions after decisions.csv is saved.

  python accept.py                  # 3-round Amber Loop -> out/decisions.csv, out/amber_rounds.jsonl, out/results.json
  python accept.py --amber-rounds 3 # explicit N rounds
  python accept.py --no-argon       # skip the Argon Amber Loop (faster)
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
    DECISION_EXTRA_COLS,
    DECISION_OUTPUT_COLS,
    DEFAULT_AMBER_ROUNDS,
    HEADLINE_KEYS,
    REJECT_THRESHOLD,
    VERIFIER_INPUT_COLS,
    Verdict,
    _pv_ok,
    _safe_prob,
    apply_argon_verdict,
    argon_revise as _engine_argon_revise,
    compute_second_pass_signals,
    enrich_results_excl_liver,
    export_verified_dataset_jsonl,
    prepare_verifier_dataframe,
    run_amber_loop,
    status,
    summarize_decisions_zero_label,
    trust,
)


async def argon_revise(r, manual, *, round_num: int = 1, prior_history: list[dict] | None = None):
    """Escalate an uncertain claim to Gemini 4 Argon (`accept.gem.call` compatible)."""
    return await _engine_argon_revise(
        r,
        manual,
        call_fn=gem.call,
        round_num=round_num,
        prior_history=prior_history,
    )


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
    amber_rounds: int = DEFAULT_AMBER_ROUNDS,
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

    man_path = C.OUT_DIR / "manual.json"
    if not man_path.exists():
        for root_cand in (C.ROOT, C.OUT_DIR.resolve().parent):
            alt_man = root_cand / "coord" / "artifacts" / "manual.json"
            if alt_man.exists():
                man_path = alt_man
                break
    manual = json.loads(man_path.read_text()) if man_path.exists() else {}

    round_logs: list[dict] = []
    if use_argon and argon_budget > 0 and amber_rounds > 0 and not d.empty:
        round_logs = await run_amber_loop(
            d,
            manual,
            argon_budget=argon_budget,
            max_rounds=amber_rounds,
            call_fn=gem.call,
        )

    dest = Path(out_path) if out_path is not None else (C.OUT_DIR / "decisions.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(dest, index=False)

    # Write Amber Loop log (`out/amber_rounds.jsonl`) and verified dataset (`out/verified_dataset.jsonl`)
    out_dir_for_aux = dest.parent
    suffix = dest.stem.removeprefix("decisions") if dest.stem.startswith("decisions_") else ""
    rounds_path = out_dir_for_aux / f"amber_rounds{suffix}.jsonl"
    rounds_path.write_text(
        "\n".join(json.dumps(rec) for rec in round_logs) + ("\n" if round_logs else "")
    )
    verified_ds_path = out_dir_for_aux / f"verified_dataset{suffix}.jsonl"
    export_verified_dataset_jsonl(d, manual, verified_ds_path)

    # Mirror to shared coord/artifacts/ ONLY when writing to the real shared out/decisions.csv
    real_out_dec = (C.ROOT / "out" / "decisions.csv").resolve()
    shared_root = C.OUT_DIR.resolve().parent
    if dest.resolve() == real_out_dec:
        coord_dir = shared_root / "coord" / "artifacts"
        coord_dir.mkdir(parents=True, exist_ok=True)
        d.to_csv(coord_dir / "decisions.csv", index=False)
        (coord_dir / "amber_rounds.jsonl").write_text(rounds_path.read_text())
        (coord_dir / "verified_dataset.jsonl").write_text(verified_ds_path.read_text())

    if skip_eval or dest.resolve() != (C.OUT_DIR / "decisions.csv").resolve():
        return summarize_decisions_zero_label(d)

    res = evaluate_decisions(C.DATA_DIR, C.OUT_DIR)
    res = enrich_results_excl_liver(res, d)
    res_path = C.OUT_DIR / "results.json"
    res_path.write_text(json.dumps(res, indent=2))
    if dest.resolve() == real_out_dec:
        coord_res = shared_root / "coord" / "artifacts" / "results.json"
        coord_res.parent.mkdir(parents=True, exist_ok=True)
        coord_res.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-argon", action="store_true")
    ap.add_argument("--argon-budget", type=int, default=300)
    ap.add_argument("--amber-rounds", type=int, default=DEFAULT_AMBER_ROUNDS)
    ap.add_argument("--labels", type=str, default=None, help="Optional input JSONL path")
    ap.add_argument("--out", type=str, default=None, help="Optional output CSV path")
    ap.add_argument("--index", type=str, default=None, help="Optional index CSV path")
    ap.add_argument("--skip-eval", action="store_true", help="Skip post-decision evaluation")
    a = ap.parse_args()
    asyncio.run(
        main(
            not a.no_argon,
            a.argon_budget,
            amber_rounds=a.amber_rounds,
            labels_path=a.labels,
            out_path=a.out,
            index_path=a.index,
            skip_eval=a.skip_eval,
        )
    )
