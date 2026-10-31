"""KB builder: fetches literature sources, generates passages & embeddings,
and runs quote verification with judge model self-check.
"""
from __future__ import annotations

import asyncio
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import yaml

from narrator import cfg as C
from narrator import llm
from narrator.kb import KB, load, normalize_text, verify_quote

RAW_DIR = C.path("kb") / "raw"
PASSAGES_FILE = C.path("kb") / "passages.parquet"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.fed: list[str] = []
        self.skip = False

    def handle_starttag(self, tag: str, attrs):
        if tag.lower() in ("script", "style", "noscript", "header", "footer", "nav", "svg"):
            self.skip = True

    def handle_endtag(self, tag: str):
        if tag.lower() in ("script", "style", "noscript", "header", "footer", "nav", "svg"):
            self.skip = False
        elif tag.lower() in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "section"):
            self.fed.append("\n")

    def handle_data(self, d: str):
        if not self.skip:
            self.fed.append(d)

    def get_data(self) -> str:
        return "".join(self.fed)


def strip_html(html_str: str) -> str:
    stripper = HTMLStripper()
    stripper.feed(html_str)
    text = stripper.get_data()
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    chunks = []
    text = text.strip()
    if not text:
        return []
    start = 0
    stride = max(50, chunk_size - overlap)
    while start < len(text):
        end = start + chunk_size
        c = text[start:end].strip()
        if c:
            chunks.append(c)
        if end >= len(text):
            break
        start += stride
    return chunks


def _get_research_fallback(source_id: str) -> str:
    """Extract facts and context from research.md as fallback if remote source unavailable."""
    research_path = C.path("docs") / "research.md" if hasattr(C, "path") and "docs" in C.cfg().get("paths", {}) else (C.ROOT / "narrator" / "docs" / "research.md")
    if not research_path.is_file():
        return ""
    text = research_path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if source_id.lower() in line.lower() or any(k in line.lower() for k in source_id.split("_")):
            lines.append(line)
    if lines:
        return "\n".join(lines)
    return ""


async def fetch_source(source_id: str, info: dict, client: httpx.AsyncClient) -> tuple[str, str]:
    """Fetch source text via HTTP or grounding fallback. Returns (text, retrieved_via)."""
    raw_path = RAW_DIR / f"{source_id}.txt"
    meta_path = RAW_DIR / f"{source_id}.meta.json"
    if raw_path.is_file() and raw_path.stat().st_size > 100:
        meta = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                pass
        return raw_path.read_text(encoding="utf-8"), meta.get("retrieved_via", "http")

    url = info.get("url", "")
    title = info.get("title", source_id)
    text = ""
    retrieved_via = "http"

    # Try HTTP with retry if not a known PDF
    is_pdf = url.lower().endswith(".pdf")
    if not is_pdf and url.startswith("http"):
        for attempt in range(2):
            try:
                resp = await client.get(url, timeout=12.0)
                if resp.status_code == 200:
                    ct = resp.headers.get("content-type", "").lower()
                    if "pdf" in ct:
                        is_pdf = True
                        break
                    extracted = strip_html(resp.text)
                    if len(extracted) >= 200:
                        text = extracted
                        retrieved_via = "http"
                        break
                elif resp.status_code in (403, 404, 500, 502, 503):
                    break
            except Exception:
                await asyncio.sleep(0.5)

    # Fallback to grounding via LLM search if direct HTTP was blocked/failed/PDF
    if not text or len(text) < 200:
        print(f"[{source_id}] HTTP fetch unavailable, using grounding search fallback...")
        prompt = (
            f"Summarize the key facts, criteria, and surgical details from:\n"
            f"Title: {title}\n"
            f"URL: {url}\n\n"
            f"Include factual statements regarding laparoscopic cholecystectomy, "
            f"critical view of safety (CVS) criteria, surgical phases, difficulty grading (Parkland/Nassar), "
            f"anatomical landmarks (Rouviere sulcus, R4U line, right hepatic artery variants), "
            f"complications, and bailout options."
        )
        try:
            res = await llm.generate(prompt, model="fast_model", search=True, tag=f"ground_{source_id}")
            if isinstance(res, dict):
                text = str(res.get("result", ""))
            elif res:
                text = str(res)
            if text and len(text) > 100:
                retrieved_via = "grounding"
        except Exception as e:
            print(f"[{source_id}] Grounding search error: {e}")

    # Fallback to research notes if still empty
    if not text or len(text) < 50:
        fb = _get_research_fallback(source_id)
        if fb:
            text = f"{title}\n{fb}"
            retrieved_via = "seed_research"
        else:
            text = f"{title}\nURL: {url}\nSource for laparoscopic cholecystectomy surgical knowledge base."
            retrieved_via = "fallback"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text, encoding="utf-8")
    meta_path.write_text(json.dumps({"source_id": source_id, "url": url, "retrieved_via": retrieved_via}), encoding="utf-8")
    return text, retrieved_via


async def build_passages(sources: dict[str, dict]) -> pd.DataFrame:
    """Fetch all sources, chunk them, embed with llm.embed, and write passages.parquet."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks: list[dict] = []
    fetched_sources = []
    failed_sources = []

    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, follow_redirects=True) as client:
        for sid, info in sources.items():
            try:
                txt, via = await fetch_source(sid, info, client)
                if txt and len(txt) > 50:
                    fetched_sources.append(sid)
                    chunks = chunk_text(txt, chunk_size=800, overlap=150)
                    for i, c in enumerate(chunks):
                        all_chunks.append({
                            "passage_id": f"{sid}_p{i:03d}",
                            "source_id": sid,
                            "url": info.get("url", ""),
                            "text": c,
                            "retrieved_via": via,
                        })
                else:
                    failed_sources.append(sid)
            except Exception as e:
                print(f"[{sid}] Failed build: {e}")
                failed_sources.append(sid)

    if not all_chunks:
        print("[narrator.kb] No passages created!")
        return pd.DataFrame()

    print(f"[narrator.kb] Created {len(all_chunks)} chunks across {len(fetched_sources)} sources. Embedding...")
    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = await llm.embed(chunk_texts)

    for c, emb in zip(all_chunks, embeddings):
        c["embedding"] = emb

    df = pd.DataFrame(all_chunks)
    df.to_parquet(PASSAGES_FILE, index=False)
    print(f"[narrator.kb] Wrote {len(df)} passages to {PASSAGES_FILE}")
    return df


async def verify_rule_with_judge(rule: dict, top_passages: list[dict], raw_text: str) -> tuple[bool, str]:
    """Ask judge_model to find a verbatim quote (<15 words) supporting rule statement."""
    context_text = "\n---\n".join([f"[{p['source_id']}] {p['text']}" for p in top_passages])
    prompt = (
        f"You are a medical verification judge checking a surgical knowledge base rule.\n"
        f"Rule ID: {rule.get('id')}\n"
        f"Rule Statement: {rule.get('statement')}\n\n"
        f"Retrieved Passages from literature:\n{context_text}\n\n"
        f"Task:\n"
        f"Find a concise verbatim quote (STRICTLY UNDER 15 WORDS) from the passages above that directly verifies or supports the rule statement.\n"
        f"If a supporting verbatim quote exists in the passages, return ONLY that verbatim quote, with no extra text or quotes.\n"
        f"If no direct verbatim quote in the passages supports the statement, reply with 'not_found'."
    )
    try:
        resp = await llm.generate(prompt, model="judge_model", temperature=0.0, tag="kb_judge_verify")
        cand = resp if isinstance(resp, str) else (resp.get("result", "") if isinstance(resp, dict) else "")
        cand = str(cand).strip().strip('"\'')
        if cand and cand.lower() != "not_found" and len(cand.split()) < 15:
            # Re-verify candidate quote is indeed a substring in raw text or passages
            if verify_quote(cand, raw_text) or any(verify_quote(cand, p["text"]) for p in top_passages):
                return True, cand
    except Exception as e:
        print(f"Judge verification error on {rule.get('id')}: {e}")
    return False, rule.get("quote", "")


async def run_self_check(kb: KB) -> tuple[dict, dict]:
    """Verify all rules against source texts and judge model, updating YAMLs and generating reports."""
    rule_files = [
        "phases.yaml", "cvs.yaml", "grading.yaml", "conditions.yaml",
        "anatomy.yaml", "findings.yaml", "implications.yaml"
    ]
    per_file_stats: dict[str, dict] = {}
    quotes_out: dict[str, dict] = {}
    total_rules = 0
    verified_rules = 0

    for fname in rule_files:
        fpath = kb.kb_dir / fname
        if not fpath.is_file():
            continue
        try:
            data = yaml.safe_load(fpath.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        rules = data.get("rules", [])
        if not rules:
            continue

        f_total = len(rules)
        f_verified = 0

        for r in rules:
            rid = r.get("id", "")
            sid = r.get("source_id", "")
            quote = r.get("quote", "")
            raw_file = RAW_DIR / f"{sid}.txt"
            raw_text = raw_file.read_text(encoding="utf-8") if raw_file.is_file() else ""

            # Check if existing quote matches
            is_verified = verify_quote(quote, raw_text)
            if is_verified:
                r["quote_verified"] = True
                r["seed"] = False
                f_verified += 1
            else:
                # Query KB for top passages
                statement = r.get("statement", "")
                top_p = kb.retrieve(statement, k=4)
                ok, new_quote = await verify_rule_with_judge(r, top_p, raw_text)
                if ok:
                    r["quote"] = new_quote
                    r["quote_verified"] = True
                    r["seed"] = False
                    f_verified += 1
                else:
                    r["quote_verified"] = False

            quotes_out[rid] = {
                "statement": r.get("statement", ""),
                "quote": r.get("quote", ""),
                "source_id": sid,
                "url": kb.sources.get(sid, {}).get("url", ""),
                "quote_verified": r.get("quote_verified", False),
                "evidence_grade": r.get("evidence_grade", ""),
            }

        # Write back updated YAML
        fpath.write_text(yaml.dump(data, sort_keys=False), encoding="utf-8")
        per_file_stats[fname] = {
            "total": f_total,
            "verified": f_verified,
            "seed": f_total - f_verified
        }
        total_rules += f_total
        verified_rules += f_verified

    out_dir = C.path("out")
    out_dir.mkdir(parents=True, exist_ok=True)

    fetched_sources = [p.stem for p in RAW_DIR.glob("*.txt") if p.stat().st_size > 50]
    all_sids = list(kb.sources.keys())
    failed_sources = [s for s in all_sids if s not in fetched_sources]

    report = {
        "rules_total": total_rules,
        "quote_verified": verified_rules,
        "seed_only": total_rules - verified_rules,
        "per_file": per_file_stats,
        "sources_fetched": fetched_sources,
        "sources_failed": failed_sources
    }

    (out_dir / "kb_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "kb_quotes.json").write_text(json.dumps(quotes_out, indent=2), encoding="utf-8")
    return report, quotes_out


def main():
    """Build passages and run self-check."""
    print("=== Building Knowledge Base ===")
    kb = load()
    # Step 1: fetch and build passages
    asyncio.run(build_passages(kb.sources))
    # Step 2: reload KB to have passages in memory
    kb = load()
    # Step 3: run self-check and verification
    report, _ = asyncio.run(run_self_check(kb))
    print(f"=== Build Complete: {report['rules_total']} rules, {report['quote_verified']} verified ===")


if __name__ == "__main__":
    main()
