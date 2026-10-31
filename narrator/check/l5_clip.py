"""L5c: non-Gemini zero-shot presence check (OpenCLIP ViT-B-32, laion2b). Independent model family.

Runs narrator/clip_score.py in the isolated narrator/.venv-clip (open_clip needs a newer torch than the shared
venv; we do not change shared packages). Abstains if that venv is missing.
For observed presence/absence of a scored class: rank of the class among 7 prompts on the claim's middle frame.
presence: rank <= 3 -> pass, rank >= 6 -> fail, else abstain. absence/negated: inverted. No labels used.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile

from narrator import cfg as C
from narrator import data as D
from narrator.check.base import Ctx, res
from narrator.schema import scored_class

NAME = "L5c"
PY = C.PKG / ".venv-clip" / "bin" / "python"
CACHE = C.path("cache") / "clip_scores.json"


def _frame(c) -> int:
    clip = D.list_clips_cached().get(c["clip_id"])
    mid = (int(c["t_start"]) + int(c["t_end"])) // 2
    if clip:
        mid = min(max(mid, clip[0]), clip[1])
    return mid


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    if not PY.exists():
        return [res("abstain", "CLIP venv missing") for _ in claims]
    todo, keys = [], []
    for c in claims:
        cls = scored_class(c)
        if c.get("modality") != "observed" or c.get("type") not in ("presence", "absence") or cls is None:
            keys.append(None)
            continue
        p = D.frame_path(c["clip_id"], _frame(c))
        if not p.exists():
            keys.append(None)
            continue
        k = hashlib.md5(f"{p}|{cls}".encode()).hexdigest()
        keys.append((k, cls))
        todo.append({"key": k, "path": str(p), "cls": cls})
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    need = [t for t in todo if t["key"] not in cache]
    if need:
        with tempfile.TemporaryDirectory() as d:
            i, o = f"{d}/in.json", f"{d}/out.json"
            json.dump(need, open(i, "w"))
            r = subprocess.run([str(PY), str(C.PKG / "clip_score.py"), i, o], capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                return [res("abstain", f"CLIP failed: {r.stderr[-120:]}") for _ in claims]
            cache.update(json.load(open(o)))
            CACHE.write_text(json.dumps(cache))
    out = []
    for c, kc in zip(claims, keys):
        if kc is None:
            out.append(res("abstain", "not an observed presence/absence claim on a scored class"))
            continue
        rank = cache[kc[0]]["rank"]
        neg = c["type"] == "absence" or c.get("polarity") == "negated"
        seen = True if rank <= 3 else (False if rank >= 6 else None)
        if seen is None:
            out.append(res("abstain", f"CLIP rank {rank}/7: undecided"))
        elif seen != neg:
            out.append(res("pass", f"CLIP rank {rank}/7"))
        else:
            out.append(res("fail", f"CLIP rank {rank}/7 contradicts the claim"))
    return out
