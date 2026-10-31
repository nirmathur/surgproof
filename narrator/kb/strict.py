"""Strict quote verification. Runs after `python -m narrator.kb build`.

python -m narrator.kb.strict [--judge]

A quote counts as verified ONLY if it is a (normalised) substring of source text fetched directly over HTTP:
narrator/kb/raw_http/<source_id>.txt, or narrator/kb/raw/<source_id>.txt whose meta says retrieved_via=http.
Text retrieved via Gemini search grounding is Gemini-written, so a match there does not count. It is recorded as
quote_status=grounding_only and stays quote_verified: false.
If the quote is found in a different HTTP source than the rule cites, the rule's source_id is corrected.
--judge: for unverified rules, ask judge_model for a verbatim quote from HTTP-only passages (code re-checks it).
Writes quote_verified / quote_status back into kb/*.yaml, plus out/kb_report.json and out/kb_quotes.json.
"""
from __future__ import annotations

import argparse
import json
import re

import yaml

from narrator import cfg as C
from narrator import llm
from narrator.kb import verify_quote

KB = C.path("kb")
RULE_FILES = ["phases.yaml", "cvs.yaml", "grading.yaml", "conditions.yaml", "anatomy.yaml", "findings.yaml",
              "implications.yaml", "decisions.yaml"]


def http_texts() -> dict[str, str]:
    out = {}
    for p in (KB / "raw").glob("*.txt"):
        meta = KB / "raw" / f"{p.stem}.meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("retrieved_via") == "http":
            out[p.stem] = p.read_text(errors="ignore")
    for p in (KB / "raw_http").glob("*.txt"):  # direct downloads (PDF/HTML -> text) win
        out[p.stem] = p.read_text(errors="ignore")
    return out


def grounding_texts() -> dict[str, str]:
    out = {}
    for p in (KB / "raw").glob("*.txt"):
        meta = KB / "raw" / f"{p.stem}.meta.json"
        if meta.exists() and json.loads(meta.read_text()).get("retrieved_via") != "http":
            out[p.stem] = p.read_text(errors="ignore")
    return out


def classify(rule: dict, http: dict, grd: dict) -> tuple[str, str | None]:
    """-> (status, source_id_where_found). status: verified | grounding_only | not_found | no_quote"""
    q = (rule.get("quote") or "").strip()
    if not q:
        return "no_quote", None
    sid = rule.get("source_id")
    if sid in http and verify_quote(q, http[sid]):
        return "verified", sid
    for s, t in http.items():
        if verify_quote(q, t):
            return "verified", s
    for s, t in grd.items():
        if verify_quote(q, t):
            return "grounding_only", s
    return "not_found", None


def _sentences(text: str):
    return [s.strip() for s in re.split(r"(?<=[.;:])\s+", re.sub(r"\s+", " ", text)) if 20 < len(s) < 400]


async def judge_quote(rule: dict, http: dict) -> tuple[str, str] | None:
    """Give the judge the HTTP sentences sharing the most words with the statement; code re-verifies."""
    words = set(re.findall(r"[a-z]{4,}", (rule.get("statement") or "").lower()))
    cands = []
    for s, t in http.items():
        for sent in _sentences(t):
            ov = len(words & set(re.findall(r"[a-z]{4,}", sent.lower())))
            if ov >= 2:
                cands.append((ov, s, sent))
    cands = sorted(cands, reverse=True)[:12]
    if not cands:
        return None
    ctx = "\n".join(f"[{i}] ({s}) {sent}" for i, (_, s, sent) in enumerate(cands))
    out = await llm.generate(
        f"Rule: {rule.get('statement')}\n\nSource sentences:\n{ctx}\n\nIf one sentence directly supports the rule, "
        "return JSON {\"index\": i, \"quote\": \"<verbatim span from that sentence, under 15 words>\"}. "
        "Otherwise return {\"index\": -1, \"quote\": \"\"}. The quote must be copied exactly.",
        model="judge_model", schema={"type": "object", "properties": {"index": {"type": "integer"},
                                                                    "quote": {"type": "string"}},
                                     "required": ["index", "quote"]}, tag="kb_strict_judge")
    if not out or out.get("index", -1) < 0 or out["index"] >= len(cands):
        return None
    _, s, sent = cands[out["index"]]
    q = out.get("quote", "")
    if q and len(q.split()) < 15 and verify_quote(q, sent) and verify_quote(q, http[s]):
        return s, q
    return None


def main(use_judge: bool = False):
    http, grd = http_texts(), grounding_texts()
    report = {"rules_total": 0, "quote_verified": 0, "grounding_only": 0, "not_found": 0, "seed_only": 0,
              "judge_verified": 0, "per_file": {}, "http_sources": sorted(http), "grounding_sources": sorted(grd),
              "rule": "verified = quote found verbatim in directly fetched source text"}
    quotes = {}
    sources = (yaml.safe_load((KB / "sources.yaml").read_text()) or {})
    sources = sources.get("sources", sources)
    pending = []
    docs = {}
    for fname in RULE_FILES:
        p = KB / fname
        if not p.exists():
            continue
        y = yaml.safe_load(p.read_text()) or {}
        docs[fname] = y
        for r in y.get("rules") or []:
            st, where = classify(r, http, grd)
            if st == "verified" and where != r.get("source_id"):
                r["source_id_original"] = r.get("source_id")
                r["source_id"] = where
            if st != "verified" and use_judge:
                pending.append(r)
            r["quote_status"] = st
            r["quote_verified"] = st == "verified"
    if pending:
        res = llm.gather([judge_quote(r, http) for r in pending], desc="kb_strict")
        for r, out in zip(pending, res):
            if out:
                s, q = out
                r["quote_original"], r["quote"] = r.get("quote"), q
                if s != r.get("source_id"):
                    r["source_id_original"], r["source_id"] = r.get("source_id"), s
                r["quote_status"], r["quote_verified"] = "verified", True
                r["seed"] = False
                report["judge_verified"] += 1
    for fname, y in docs.items():
        f = {"total": 0, "verified": 0}
        for r in y.get("rules") or []:
            report["rules_total"] += 1; f["total"] += 1
            st = r["quote_status"]
            if st == "verified":
                report["quote_verified"] += 1; f["verified"] += 1
            elif st == "grounding_only":
                report["grounding_only"] += 1
            else:
                report["not_found"] += 1
            if r.get("seed") and st != "verified":
                report["seed_only"] += 1
            src = sources.get(r.get("source_id"), {}) if isinstance(sources, dict) else {}
            quotes[r["id"]] = {"statement": r.get("statement"), "quote": r.get("quote"), "source_id": r.get("source_id"),
                               "url": (src or {}).get("url"), "quote_verified": r["quote_verified"],
                               "quote_status": st, "evidence_grade": r.get("evidence_grade"), "file": fname}
        report["per_file"][fname] = f
        (KB / fname).write_text(yaml.safe_dump(y, sort_keys=False, allow_unicode=True, width=120))
    (C.out_dir() / "kb_report.json").write_text(json.dumps(report, indent=1))
    (C.out_dir() / "kb_quotes.json").write_text(json.dumps(quotes, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "per_file"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    main(ap.parse_args().judge)
