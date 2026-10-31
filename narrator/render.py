"""Render checked claims into narration sentences. Deterministic templates: rendering adds no facts.

python -m narrator.render -> narrator/out/narration.jsonl
grounded: speak + hedge claims only, grouped per time window, each sentence keeps its claim_ids.
free:     the baseline's own sentences, all kept, with verdicts aligned so the UI can mark drops red.
"""
from __future__ import annotations

import json
from collections import defaultdict

import pandas as pd

from narrator import cfg as C

NICE = {"l_hook": "L-hook", "hook": "L-hook", "cystic_duct": "cystic duct", "gallbladder": "gallbladder",
        "grasper": "grasper", "liver": "liver", "blood": "blood", "cystic_plate": "cystic plate",
        "hepatocystic_triangle": "hepatocystic triangle", "cvs_triangle_cleared": "hepatocystic triangle cleared",
        "cvs_lower_gb_off_plate": "lower gallbladder off the cystic plate",
        "cvs_two_structures": "only two structures entering the gallbladder"}
VERB_ING = {"grasp": "grasping", "retract": "retracting", "dissect": "dissecting", "coagulate": "coagulating",
            "clip": "clipping", "cut": "cutting", "aspirate": "aspirating", "irrigate": "irrigating", "pack": "bagging"}


def nice(x):
    return NICE.get(x, str(x).replace("_", " ")) if x else ""


def _ents(c):
    e = c["entities"]
    return json.loads(e) if isinstance(e, str) else (e or {})


def subject(c) -> str:
    e = _ents(c)
    return nice(e.get("instrument") or e.get("target") or e.get("finding") or e.get("phase")
                or e.get("cvs_criterion") or e.get("grade") or e.get("condition") or e.get("decision"))


def sentence(c) -> str:
    e, t, hedged = _ents(c), c["type"], c["verdict"] == "hedge"
    s = subject(c)
    if t == "presence":
        return f"Possible {s}; not confirmed." if hedged else f"{s.capitalize()} in view."
    if t == "absence":
        return f"{s.capitalize()} not confirmed in view." if hedged else f"No {s} in view."
    if t == "action":
        v = VERB_ING.get(e.get("verb"), nice(e.get("verb")))
        tgt = f" the {nice(e.get('target'))}" if e.get("target") and e.get("target") != "null_target" else ""
        base = f"{nice(e.get('instrument')).capitalize()} {v}{tgt}".strip()
        return f"Looks like the {base[0].lower() + base[1:]}; holding that loosely." if hedged else f"{base}."
    if t == "phase":
        return f"Probably {s} phase; the timeline alone can't confirm it." if hedged else f"Phase: {s}."
    if t == "cvs_criterion":
        if hedged:
            return f"{s.capitalize()}: not yet confirmed. Holding off on CVS."
        return f"CVS criterion met: {s}."
    if t == "finding":
        return f"Possible {s}; not confirmed." if hedged else f"{s.capitalize()} seen."
    if t == "decision_point":
        return f"Decision point: {s} (not confirmed)." if hedged else f"Decision point: {s}."
    txt = c.get("statement") or c.get("text") or s
    return f"Reference (hedged): {txt}" if hedged else f"Reference: {txt}"


def render(df: pd.DataFrame) -> list[dict]:
    win = C.cfg()["claims"]["window_frames"]
    lines = []
    g = df[(df["mode"] == "grounded") & df.verdict.isin(["speak", "hedge"])]
    for clip_id, sub in g.groupby("clip_id"):
        first = int(sub.t_start.min())
        buckets = defaultdict(list)
        for c in sub.to_dict("records"):
            buckets[(int(c["t_start"]) - first) // win].append(c)
        for b, cs in sorted(buckets.items()):
            seen = {}
            for c in sorted(cs, key=lambda c: (c["verdict"] != "speak", c["t_start"])):
                s = sentence(c)
                seen.setdefault(s, []).append(c)
            for s, group in seen.items():
                lines.append({"clip_id": clip_id, "t_start": min(c["t_start"] for c in group),
                              "t_end": max(c["t_end"] for c in group), "text": s,
                              "claim_ids": [c["claim_id"] for c in group],
                              "verdicts": [c["verdict"] for c in group], "mode": "grounded",
                              "source": group[0].get("source", "gemini")})
    f = df[df["mode"] == "free"]
    for clip_id, sub in f.groupby("clip_id"):
        by_text = defaultdict(list)
        for c in sub.to_dict("records"):
            by_text[c.get("text") or sentence(c)].append(c)
        for s, group in by_text.items():
            lines.append({"clip_id": clip_id, "t_start": min(c["t_start"] for c in group),
                          "t_end": max(c["t_end"] for c in group), "text": s,
                          "claim_ids": [c["claim_id"] for c in group],
                          "verdicts": [c["verdict"] for c in group], "mode": "free", "source": "gemini"})
    return lines


def main():
    df = pd.read_parquet(C.out_dir() / "claims.parquet")
    lines = render(df)
    with open(C.out_dir() / "narration.jsonl", "w") as fh:
        for l in lines:
            fh.write(json.dumps(l) + "\n")
    print(f"narration lines: {len(lines)} (grounded {sum(l['mode'] == 'grounded' for l in lines)})")


if __name__ == "__main__":
    main()
