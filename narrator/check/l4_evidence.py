"""L4 evidence: observed claims must match the verified label table and the tracker.

- presence (asserted/hedged) of a scorable class: accepted label in the window -> pass; only uncertain -> pass+hedge;
  absent/rejected -> fail.
- absence (or negated presence): accepted label present -> fail; only uncertain -> pass+hedge; absent -> pass.
- cited label keys (lab:...) that do not show the claimed class -> fail.
- tools: median tracker suspicion over the window above check.uncertain_suspicion -> hedge (P1.1).
- action: the instrument must be present; a "still" tracker event over the window with an active verb -> hedge.
Non-observed or non-scorable claims -> abstain. Labels table: ctx.labels (real or mock).
"""
from __future__ import annotations

from narrator import cfg as C
from narrator.check.base import Ctx, res, safe_map
from narrator.schema import scored_class

NAME = "L4"
ACTIVE_VERBS = {"grasp", "retract", "dissect", "coagulate", "clip", "cut", "aspirate", "irrigate", "pack"}


def _window(labels, clip_id, label, t0, t1):
    sub = labels[(labels.clip_id == clip_id) & (labels.label == label)]
    w = sub[sub.frame_idx.between(t0, t1)]
    if w.empty:  # claim window falls between sampled frames: nearest sampled frame of the clip
        allf = labels[labels.clip_id == clip_id].frame_idx
        if allf.empty:
            return w, False
        near = int(allf.iloc[(allf - (t0 + t1) / 2).abs().argmin()])
        w = sub[sub.frame_idx == near]
    return w, True


def _suspicion(ctx: Ctx, clip_id, label, t0, t1) -> float:
    f = ctx.flags
    if f is None or f.empty or label not in ("grasper", "l_hook"):
        return 0.0
    s = f[(f.clip_id == clip_id) & (f.label == label) & f.frame_idx.between(t0, t1) & (f.n_det > 0)]
    return float(s.suspicion.median()) if len(s) else 0.0


def _check(c: dict, ctx: Ctx) -> dict:
    if c.get("modality") != "observed" or c.get("type") not in ("presence", "absence", "action", "finding"):
        return res("abstain", "not an observed visual claim")
    cls = scored_class(c)
    if cls is None:
        return res("abstain", "entity not in the verified label set")
    L = ctx.labels
    t0, t1 = int(c["t_start"]), int(c["t_end"])
    w, has_clip = _window(L, c["clip_id"], cls, t0, t1)
    if not has_clip:
        return res("abstain", "no label rows for this clip")
    acc = w[w.present & (w.status == "accepted")]
    unc = w[w.present & (w.status == "uncertain")]
    negated = c["type"] == "absence" or c.get("polarity") == "negated"

    # cited label keys must show the class
    for ref in c.get("evidence_refs") or []:
        if ref.startswith("lab:"):
            try:
                _, clip, fr, lab = ref.split(":")
            except ValueError:
                return res("fail", f"malformed evidence ref {ref}")
            if not negated and lab == cls:
                r = L[(L.clip_id == clip) & (L.frame_idx == int(fr)) & (L.label == lab) & L.present]
                if r.empty:
                    return res("fail", f"cited {ref} but the label table has no {lab} there")

    susp = _suspicion(ctx, c["clip_id"], cls, t0, t1)
    thr = C.cfg()["check"]["uncertain_suspicion"]
    if negated:
        if len(acc):
            return res("fail", f"{cls} accepted at frames {sorted(acc.frame_idx)[:3]}")
        if len(unc):
            return res("pass", f"{cls} only uncertain in window", hedge=True)
        return res("pass", f"{cls} absent in label table")
    if len(acc):
        if susp > thr:
            return res("pass", f"{cls} accepted but tracker suspicion {susp:.2f}", hedge=True)
        if c["type"] == "action" and (c.get("entities") or {}).get("verb") in ACTIVE_VERBS:
            still = [e for e in ctx.events.get(c["clip_id"], []) if e["type"] == "tool_still"
                     and e["label"] == cls and e["t_start"] <= t1 and e["t_end"] >= t0]
            if still:
                return res("pass", f"{cls} present but tracker says still ({still[0]['event_id']})", hedge=True)
        return res("pass", f"{cls} accepted in window")
    if len(unc):
        return res("pass", f"{cls} only uncertain in window", hedge=True)
    return res("fail", f"{cls} not in the verified label table for frames {t0}-{t1}")


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    return safe_map(_check, claims, ctx)
