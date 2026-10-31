"""Zero-label verification package for SurgProof (`verify/`).

Implements:
1. Independent second-pass double-check on every label (Pass 1 vs Pass 2 -> accept / reject / AMBER).
2. Multi-round Argon Amber Loop (N=3 rounds default: zoomed crop + manual criteria -> tighter crop +
   differential diagnosis -> multi-vote consensus adjudication), logged to `out/amber_rounds.jsonl`.
3. Dataset export (`out/verified_dataset.jsonl`) with full provenance per label.
"""
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
    argon_revise,
    compute_second_pass_signals,
    enrich_results_excl_liver,
    export_verified_dataset_jsonl,
    prepare_verifier_dataframe,
    run_amber_loop,
    status,
    summarize_decisions_zero_label,
    trust,
)

__all__ = [
    "ACCEPT_THRESHOLD",
    "ARGON_ACCEPT_CONF",
    "DECISION_EXTRA_COLS",
    "DECISION_OUTPUT_COLS",
    "DEFAULT_AMBER_ROUNDS",
    "HEADLINE_KEYS",
    "REJECT_THRESHOLD",
    "VERIFIER_INPUT_COLS",
    "Verdict",
    "_pv_ok",
    "_safe_prob",
    "apply_argon_verdict",
    "argon_revise",
    "compute_second_pass_signals",
    "enrich_results_excl_liver",
    "export_verified_dataset_jsonl",
    "prepare_verifier_dataframe",
    "run_amber_loop",
    "status",
    "summarize_decisions_zero_label",
    "trust",
]
