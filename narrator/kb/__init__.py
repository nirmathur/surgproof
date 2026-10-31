"""Narrator Knowledge Base (KB) loader and retrieval API."""
from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi

from narrator import cfg as C


def normalize_text(s: str) -> str:
    """Normalise case, smart quotes, dashes, and whitespace."""
    if not s:
        return ""
    s = s.lower()
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("`", "'")
    s = s.replace("—", "-").replace("–", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def verify_quote(quote: str, text: str) -> bool:
    """Return True if quote is a normalised substring of text."""
    if not quote or not text:
        return False
    norm_q = normalize_text(quote)
    norm_t = normalize_text(text)
    if not norm_q:
        return False
    if norm_q in norm_t:
        return True
    clean_q = re.sub(r"[^\w\s]", "", norm_q)
    clean_t = re.sub(r"[^\w\s]", "", norm_t)
    clean_q = re.sub(r"\s+", " ", clean_q).strip()
    clean_t = re.sub(r"\s+", " ", clean_t).strip()
    if clean_q and clean_q in clean_t:
        return True
    return False


def _tokenize(text: str) -> list[str]:
    clean = re.sub(r"[^\w\s]", " ", normalize_text(text))
    return [t for t in clean.split() if t]


class KB:
    """Knowledge base holding surgical rules, vocabularies, and passage retrieval."""

    def __init__(self, kb_dir: Optional[Path] = None):
        self.kb_dir = kb_dir or C.path("kb")
        self._yaml_cache: dict[str, dict] = {}
        self.rules: dict[str, dict] = {}
        self.sources: dict[str, dict] = {}
        self.passages: Optional[pd.DataFrame] = None
        self._bm25: Optional[BM25Okapi] = None
        self._embeddings: Optional[np.ndarray] = None
        self._load_yaml_and_rules()
        self._load_passages()

    def _load_yaml_and_rules(self) -> None:
        """Load all YAML files and index rules."""
        for yf in sorted(self.kb_dir.glob("*.yaml")):
            stem = yf.stem
            name = yf.name
            try:
                data = yaml.safe_load(yf.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            self._yaml_cache[stem] = data
            self._yaml_cache[name] = data
            if stem == "sources":
                self.sources = data

            # Extract top-level rules list if present
            if isinstance(data, dict) and "rules" in data and isinstance(data["rules"], list):
                for r in data["rules"]:
                    if isinstance(r, dict) and "id" in r:
                        r_copy = dict(r)
                        r_copy["file"] = yf.name
                        self.rules[r["id"]] = r_copy

    def _load_passages(self) -> None:
        """Load passages.parquet if present and prepare search indices."""
        pq = self.kb_dir / "passages.parquet"
        if not pq.is_file():
            return
        try:
            df = pd.read_parquet(pq)
            if df.empty:
                return
            self.passages = df.reset_index(drop=True)
            corpus = [_tokenize(t) for t in self.passages["text"].fillna("")]
            self._bm25 = BM25Okapi(corpus)

            # Check for embeddings
            if "embedding" in self.passages.columns:
                first_emb = self.passages["embedding"].iloc[0]
                if first_emb is not None and len(first_emb) > 0:
                    embs = np.array(self.passages["embedding"].tolist(), dtype=np.float32)
                    self._embeddings = embs
        except Exception as e:
            print(f"[narrator.kb] Error loading passages: {e}")

    def yaml(self, name: str) -> dict:
        """Return parsed dictionary for given YAML name (e.g. 'phases' or 'phases.yaml')."""
        if name in self._yaml_cache:
            return self._yaml_cache[name]
        path = self.kb_dir / (name if name.endswith(".yaml") else f"{name}.yaml")
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self._yaml_cache[name] = data
            return data
        return {}

    def retrieve(self, query: str, k: int = 4) -> list[dict[str, Any]]:
        """Retrieve top-k passages matching query via hybrid dense + BM25 reciprocal rank fusion."""
        if self.passages is None or self.passages.empty:
            return []

        n_docs = len(self.passages)
        k = min(k, n_docs)
        q_tokens = _tokenize(query)

        # BM25 scores
        if self._bm25 is not None and q_tokens:
            bm25_scores = self._bm25.get_scores(q_tokens)
            bm25_ranks = np.argsort(-bm25_scores)
        else:
            bm25_scores = np.zeros(n_docs)
            bm25_ranks = np.arange(n_docs)

        # Dense embedding scores
        q_emb = None
        if self._embeddings is not None and len(self._embeddings) == n_docs:
            try:
                from narrator import llm

                async def _get_emb():
                    return await llm.embed([query])

                try:
                    asyncio.get_running_loop()
                    in_loop = True
                except RuntimeError:
                    in_loop = False
                if in_loop:  # called from async code (L7): embed on a worker thread with its own loop
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(1) as ex:
                        res = ex.submit(lambda: asyncio.run(_get_emb())).result()
                else:
                    res = asyncio.run(_get_emb())
                if res and res[0] is not None:
                    q_emb = np.array(res[0], dtype=np.float32)
            except Exception:
                q_emb = None

        if q_emb is not None:
            # Cosine similarity
            doc_norms = np.linalg.norm(self._embeddings, axis=1)
            q_norm = np.linalg.norm(q_emb)
            denom = np.where((doc_norms * q_norm) == 0, 1e-9, doc_norms * q_norm)
            cos_sim = np.dot(self._embeddings, q_emb) / denom
            dense_ranks = np.argsort(-cos_sim)

            # Reciprocal rank fusion (RRF with constant 60)
            rrf_scores = np.zeros(n_docs, dtype=np.float32)
            for rank, idx in enumerate(dense_ranks):
                rrf_scores[idx] += 1.0 / (60.0 + rank + 1)
            for rank, idx in enumerate(bm25_ranks):
                rrf_scores[idx] += 1.0 / (60.0 + rank + 1)

            final_order = np.argsort(-rrf_scores)[:k]
            scores = rrf_scores[final_order]
        else:
            final_order = bm25_ranks[:k]
            scores = bm25_scores[final_order]

        results = []
        for rank_idx, score_val in zip(final_order, scores):
            row = self.passages.iloc[rank_idx]
            results.append({
                "passage_id": str(row.get("passage_id", "")),
                "source_id": str(row.get("source_id", "")),
                "url": str(row.get("url", "")),
                "text": str(row.get("text", "")),
                "score": float(score_val),
            })
        return results


def load(kb_dir: Optional[Path] = None) -> KB:
    """Load and return the KB instance."""
    return KB(kb_dir=kb_dir)
