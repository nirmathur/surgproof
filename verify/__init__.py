"""Zero-label verification package for SurgProof (`verify/`).

Replaces human verification using decorrelated visual checks (blind crop check + sceptical critic)
and Gemini 4 Argon escalation for uncertain claims, routing unresolved items to the AMBER Human Gate.
"""
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
    argon_revise,
    enrich_results_excl_liver,
    prepare_verifier_dataframe,
    status,
    summarize_decisions_zero_label,
    trust,
)

__all__ = [
    "ACCEPT_THRESHOLD",
    "ARGON_ACCEPT_CONF",
    "DECISION_OUTPUT_COLS",
    "HEADLINE_KEYS",
    "REJECT_THRESHOLD",
    "VERIFIER_INPUT_COLS",
    "Verdict",
    "_pv_ok",
    "_safe_prob",
    "apply_argon_verdict",
    "argon_revise",
    "enrich_results_excl_liver",
    "prepare_verifier_dataframe",
    "status",
    "summarize_decisions_zero_label",
    "trust",
]
