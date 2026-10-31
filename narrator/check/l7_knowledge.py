"""L7 knowledge support check layer.

Verifies knowledge and inference claims (and any claim with kb_refs) against
retrieved passages from the knowledge base using judge_model.
Verifies quotes are verbatim substrings before accepting support.
"""
from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel

from narrator import cfg as C
from narrator import llm
from narrator.check.base import Ctx, res

NAME = "L7"


class KBJudgeOutput(BaseModel):
    verdict: Literal["supported", "contradicted", "not_found"]
    quote: str = ""
    passage_id: str = ""


def normalize_text(s: str) -> str:
    """Normalise whitespace and case for robust substring comparison."""
    if not s:
        return ""
    return re.sub(r"\s+", " ", s.lower()).strip()


def verify_quote(quote: str, text: str) -> bool:
    """Check if quote is a normalised verbatim substring of text."""
    if not quote or not text:
        return False
    norm_q = normalize_text(quote)
    norm_t = normalize_text(text)
    if not norm_q:
        return False
    return norm_q in norm_t


def get_or_load_kb(ctx: Ctx) -> Any:
    """Return existing ctx.kb or attempt to load narrator.kb."""
    if ctx.kb is not None:
        return ctx.kb
    try:
        import narrator.kb as n_kb
        if hasattr(n_kb, "load"):
            ctx.kb = n_kb.load()
        else:
            ctx.kb = n_kb
        return ctx.kb
    except Exception:
        return None


async def evaluate_claim(claim: dict, ctx: Ctx) -> dict:
    """Async evaluation of knowledge / inference claim against KB."""
    modality = claim.get("modality")
    kb_refs = claim.get("kb_refs") or []

    # 1. Target claim check
    if modality not in ("knowledge", "inference") and not kb_refs:
        return res("abstain", f"modality '{modality}' without kb_refs is non-knowledge")

    # 2. Knowledge base availability
    kb = get_or_load_kb(ctx)
    if kb is None:
        return res("abstain", "narrator.kb unavailable")

    # 3. Check kb_refs against ctx.kb.rules
    if kb_refs and hasattr(kb, "rules") and kb.rules is not None:
        for ref in kb_refs:
            if ref not in kb.rules:
                return res("fail", f"kb_ref '{ref}' not found in kb.rules")

    # 4. Passage retrieval
    query = claim.get("statement") or claim.get("text") or ""
    if not query.strip():
        return res("abstain", "empty query statement/text for KB check")

    top_k = C.cfg().get("check", {}).get("l7_top_k", 4)
    try:
        passages = kb.retrieve(query, k=top_k)
    except Exception as e:
        return res("abstain", f"kb retrieval failed: {str(e)[:80]}")

    if not passages:
        return res("abstain", "no passages retrieved from KB")

    # 5. Build passages prompt
    passages_text_parts: list[str] = []
    passage_map: dict[str, str] = {}
    for p in passages:
        pid = str(p.get("passage_id") or p.get("id") or "")
        ptext = str(p.get("text") or p.get("content") or "")
        passage_map[pid] = ptext
        passages_text_parts.append(f"[Passage ID: {pid}]\n{ptext}")

    passages_block = "\n\n".join(passages_text_parts)

    prompt = (
        "You are a rigorous surgical knowledge judge. You are evaluating whether a surgical claim "
        "is supported, contradicted, or unsupported by verified knowledge base passages.\n\n"
        f"Claim to evaluate: {query}\n\n"
        f"Retrieved Passages:\n{passages_block}\n\n"
        "Instructions:\n"
        "- If any passage directly supports the claim, set verdict='supported', quote the exact verbatim "
        "substring from that passage, and provide its passage_id.\n"
        "- If any passage directly contradicts the claim, set verdict='contradicted', quote the exact "
        "verbatim substring showing the contradiction, and provide its passage_id.\n"
        "- If the passages neither directly support nor contradict the claim, set verdict='not_found', "
        "quote='', and passage_id=''.\n"
        "- CRITICAL: The quote MUST be an exact verbatim substring from the passage text.\n"
        "Return JSON conforming to the schema."
    )

    out = await llm.generate(
        prompt=prompt,
        model="judge_model",
        schema=KBJudgeOutput,
        tag="l7",
    )
    if out is None:
        return res("abstain", "judge_model call failed")

    verdict = str(out.get("verdict", "not_found")).lower().strip()
    quote = str(out.get("quote", "")).strip()
    pid = str(out.get("passage_id", "")).strip()

    # 6. Verify quote verbatim in passage
    if verdict in ("supported", "contradicted"):
        ptext = passage_map.get(pid)
        is_valid = verify_quote(quote, ptext) if ptext else False
        if not is_valid:
            # Check if quote matches any other passage
            for alt_pid, alt_text in passage_map.items():
                if verify_quote(quote, alt_text):
                    pid = alt_pid
                    is_valid = True
                    break
        if not is_valid:
            verdict = "not_found"

    # 7. Map verdict
    if verdict == "supported":
        return res("pass", f"supported by {pid}: '{quote}'")
    elif verdict == "contradicted":
        return res("fail", f"contradicted by {pid}: '{quote}'")
    else:
        return res("fail", "unsupported by KB")


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L7 knowledge support check over claims in batch."""
    if not claims:
        return []

    coros = [evaluate_claim(c, ctx) for c in claims]
    results = llm.gather(coros, desc="L7")

    clean_results: list[dict] = []
    for r in results:
        if r is None or not isinstance(r, dict):
            clean_results.append(res("abstain", "l7 execution error"))
        else:
            clean_results.append(r)

    ctx.extra["L7"] = clean_results
    return clean_results
