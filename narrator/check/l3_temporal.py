"""L3 temporal claim checker layer."""
from __future__ import annotations

from typing import Any

from narrator.check.base import Ctx, res

NAME = "L3"


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L3 temporal consistency check on claims."""
    clip_bounds = ctx.extra.get("clip_bounds", {}) if ctx.extra else {}

    results = []
    for idx, c in enumerate(claims):
        try:
            if not isinstance(c, dict):
                results.append(res("fail", "Claim must be a dict"))
                continue

            clip_id = c.get("clip_id", "")
            t_start = c.get("t_start")
            t_end = c.get("t_end")
            ctype = c.get("type")
            cmod = c.get("modality")
            cpol = c.get("polarity", "asserted")
            ev_refs = c.get("evidence_refs") or []

            # 1. Clip bounds check
            if clip_id in clip_bounds:
                first, last = clip_bounds[clip_id]
                if t_start is not None and t_end is not None:
                    if t_start < first or t_end > last:
                        results.append(
                            res(
                                "fail",
                                f"Claim timestamps [{t_start}, {t_end}] outside clip bounds [{first}, {last}]",
                            )
                        )
                        continue

            # 2. Single-frame observed presence/action check
            if ctype in ("presence", "action") and cmod == "observed":
                if t_start is not None and t_end is not None and t_start == t_end:
                    has_ev = any(isinstance(r, str) and r.startswith("ev:") for r in ev_refs)
                    if not has_ev:
                        results.append(
                            res(
                                "fail",
                                f"Single-frame observed {ctype} claim at frame {t_start} requires an event id (ev:...)",
                            )
                        )
                        continue

            # 3. Contradiction within clip
            raw_entities = c.get("entities")
            if hasattr(raw_entities, "model_dump"):
                entities = raw_entities.model_dump()
            elif isinstance(raw_entities, dict):
                entities = raw_entities
            else:
                entities = {}

            c_ev_set = set(ev_refs)
            contradiction_found = False
            contradiction_reason = ""

            for prev_idx in range(idx):
                prev_c = claims[prev_idx]
                if not isinstance(prev_c, dict):
                    continue
                if prev_c.get("clip_id") != clip_id:
                    continue

                prev_start = prev_c.get("t_start")
                prev_end = prev_c.get("t_end")
                if t_start is None or t_end is None or prev_start is None or prev_end is None:
                    continue

                # Check time overlap
                if max(t_start, prev_start) > min(t_end, prev_end):
                    continue

                # Check same entity
                prev_entities = prev_c.get("entities")
                if hasattr(prev_entities, "model_dump"):
                    prev_entities = prev_entities.model_dump()
                elif not isinstance(prev_entities, dict):
                    prev_entities = {}

                shared_entity = None
                for ef in ("instrument", "target", "finding"):
                    v1 = entities.get(ef)
                    v2 = prev_entities.get(ef)
                    if v1 and v2 and v1 == v2:
                        shared_entity = (ef, v1)
                        break

                if not shared_entity:
                    continue

                # Check opposite polarity or presence vs absence
                prev_type = prev_c.get("type")
                prev_pol = prev_c.get("polarity", "asserted")

                is_contra = False
                if (ctype == "presence" and prev_type == "absence") or (ctype == "absence" and prev_type == "presence"):
                    is_contra = True
                elif ctype == prev_type and (
                    (cpol == "asserted" and prev_pol == "negated") or (cpol == "negated" and prev_pol == "asserted")
                ):
                    is_contra = True

                if is_contra:
                    prev_ev_set = set(prev_c.get("evidence_refs") or [])
                    # No new evidence: c_ev_set is a subset of prev_ev_set
                    if c_ev_set <= prev_ev_set:
                        contradiction_found = True
                        prev_id = prev_c.get("claim_id", f"claim_{prev_idx}")
                        contradiction_reason = (
                            f"Contradicts earlier claim '{prev_id}' regarding {shared_entity[0]}='{shared_entity[1]}' "
                            f"over overlapping frames [{max(t_start, prev_start)}, {min(t_end, prev_end)}] without new evidence"
                        )
                        break

            if contradiction_found:
                results.append(res("fail", contradiction_reason))
                continue

            results.append(res("pass", "Temporal assertions valid"))

        except Exception as e:
            results.append(res("abstain", f"error in L3: {str(e)[:120]}"))

    return results
