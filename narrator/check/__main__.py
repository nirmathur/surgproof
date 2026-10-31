"""python -m narrator.check [--layers L1,L2,L4] [--modes free,grounded]
Reads narrator/out/claims_<mode>.jsonl, runs every available layer, writes narrator/out/claims.parquet."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from narrator import cfg as C
from narrator import data as D
from narrator import events as E
from narrator import llm
from narrator.check import apply
from narrator.check.base import Ctx


def build_ctx() -> Ctx:
    labels = D.load_labels()
    _, _, flags = E.load_tracker()
    clips = D.list_clips()
    try:
        from narrator import kb as KB
        kb = KB.load()
        if C.cfg()["check"].get("l7_http_only", True) and getattr(kb, "passages", None) is not None:
            raw_retrieve = kb.retrieve
            via = dict(zip(kb.passages.passage_id, kb.passages.retrieved_via))

            def http_retrieve(query, k=4):  # grounding passages are Gemini-written: not independent evidence
                return [p for p in raw_retrieve(query, k=k * 6) if via.get(p["passage_id"]) == "http"][:k]
            kb.retrieve = http_retrieve
    except Exception as e:  # noqa: BLE001
        print(f"[check] KB unavailable: {str(e)[:100]}")
        kb = None
    ctx = Ctx(labels=labels, flags=flags, events=E.build(), kb=kb)
    ctx.extra["clip_bounds"] = {r.clip_id: (int(r.first), int(r.last)) for r in clips.itertuples()}
    return ctx


def load_claims(modes) -> list[dict]:
    out = []
    for m in modes:
        p = C.out_dir() / f"claims_{m}.jsonl"
        if p.exists():
            out += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return out


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("entities", "evidence_refs", "kb_refs", "reasons", "decided_by", "hedges"):
        if col in df:
            df[col] = df[col].apply(lambda v: json.dumps(v, default=str))
    df["video_id"] = df.clip_id.str.split("_").str[0]
    df["split"] = df.video_id.map(C.split_of)
    return df


def main(layers=None, modes=("free", "grounded"), out_name="claims.parquet"):
    claims = load_claims(modes)
    ctx = build_ctx()
    rows = []
    for m in modes:  # check each mode separately so L3/L2 sequences don't mix modes
        sub = [c for c in claims if c.get("mode") == m]
        if sub:
            rows += apply(sub, ctx, layers)
    df = to_frame(rows)
    df.to_parquet(C.out_dir() / out_name, index=False)
    print(df.groupby(["mode", "verdict"]).size().unstack(fill_value=0))
    print(llm.stats_line())
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default=None)
    ap.add_argument("--modes", default="free,grounded")
    a = ap.parse_args()
    main(a.layers.split(",") if a.layers else None, a.modes.split(","))
