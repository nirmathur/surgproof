"""L6 perturbation layer.

Tests robustness against hallucination / prior bias. Re-asks the L5 question
on a seeded sample of passed claims with:
  (a) all-black image
  (b) shuffled frame from a different video
If the answer stays "yes" on the blacked-out frame -> fail ("answer from priors, not pixels"). The shuffled answer is
reported only: another lap chole frame usually contains the same structures.
Unsampled claims abstain.
"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image

from narrator import cfg as C
from narrator import data
from narrator import llm
from narrator.check.base import Ctx, res
from narrator.check.l5_blind import (
    PROMPT_TEMPLATE,
    SYSTEM_INSTRUCTION,
    VisualCheckAnswer,
    build_question,
    find_frames,
)

NAME = "L6"


@lru_cache(maxsize=1)
def _get_clips_df():
    try:
        return data.list_clips()
    except Exception:
        return None


def get_shuffled_frame_path(clip_id: str) -> Optional[Path]:
    """Deterministically find a frame from a different surgical video."""
    df = _get_clips_df()
    if df is None or df.empty:
        return None

    current_vid = clip_id.split("_")[0]
    other_vids = sorted(v for v in df["video_id"].unique() if v != current_vid)
    if not other_vids:
        return None

    # Deterministic choice based on clip_id
    chosen_vid = other_vids[hash(clip_id) % len(other_vids)]
    sub = df[df["video_id"] == chosen_vid]
    if sub.empty:
        return None

    first_row = sub.iloc[0]
    p = data.frame_path(first_row["clip_id"], int(first_row["first"]))
    if p.is_file():
        return p

    # Fallback to scanning frames in the clip
    for f in range(int(first_row["first"]), int(first_row["last"]) + 1):
        cand = data.frame_path(first_row["clip_id"], f)
        if cand.is_file():
            return cand

    return None


def make_black_jpeg(base_path: Optional[Path]) -> bytes:
    """Create an all-black image matching base dimensions, or default 854x480."""
    size = (854, 480)
    if base_path and base_path.is_file():
        try:
            with Image.open(base_path) as im:
                size = im.size
        except Exception:
            pass
    black_im = Image.new("RGB", size, (0, 0, 0))
    return llm.jpeg(black_im)


async def evaluate_perturbation(claim: dict, ctx: Ctx) -> dict:
    """Run blacked and shuffled frame perturbations for a single claim."""
    question, _ = build_question(claim)
    if not question:
        return res("abstain", "unable to build question for perturbation")

    clip_id = claim.get("clip_id", "")
    t_start = int(claim.get("t_start", 0))
    t_end = int(claim.get("t_end", t_start))

    frame_indices = find_frames(clip_id, t_start, t_end)
    primary_path = data.frame_path(clip_id, frame_indices[0]) if frame_indices else None

    # (a) Blacked out frame
    black_bytes = make_black_jpeg(primary_path)

    # (b) Shuffled frame from a different video
    shuf_path = get_shuffled_frame_path(clip_id)
    if shuf_path is None or not shuf_path.is_file():
        return res("abstain", "unable to locate shuffled frame from different video")

    try:
        shuf_bytes = llm.jpeg(shuf_path)
    except Exception as e:
        return res("abstain", f"failed loading shuffled frame: {str(e)[:80]}")

    prompt = PROMPT_TEMPLATE.format(system=SYSTEM_INSTRUCTION, question=question)

    # Re-ask on blacked frame
    resp_black = await llm.generate(
        prompt=prompt,
        model="fast_model",
        images=[black_bytes],
        schema=VisualCheckAnswer,
        tag="l6_black",
    )

    # Re-ask on shuffled frame
    resp_shuf = await llm.generate(
        prompt=prompt,
        model="fast_model",
        images=[shuf_bytes],
        schema=VisualCheckAnswer,
        tag="l6_shuf",
    )

    if resp_black is None and resp_shuf is None:
        return res("abstain", "perturbation llm calls failed")

    ans_black = str((resp_black or {}).get("answer", "")).lower().strip()
    ans_shuf = str((resp_shuf or {}).get("answer", "")).lower().strip()

    # Decisive test: "yes" on an all-black frame means the answer came from priors.
    # A shuffled frame from another lap chole video usually DOES contain liver / gallbladder / tools, so "yes"
    # there is not evidence of a prior: it is reported in the reason only (DECISIONS.md).
    if ans_black == "yes":
        return res("fail", f"answer from priors, not pixels (yes on blacked-out frame; shuffled={ans_shuf or 'n/a'})")
    if ans_black not in ("no", "unsure"):
        return res("abstain", "no answer on blacked-out frame")
    return res("pass", f"no on blacked-out frame (shuffled={ans_shuf or 'n/a'})")


def run(claims: list[dict], ctx: Ctx) -> list[dict]:
    """Run L6 perturbation checks on a seeded sample of passing L5 claims."""
    if not claims:
        return []

    # 1. Determine candidate claims
    l5_results = ctx.extra.get("L5")
    if l5_results is not None and len(l5_results) == len(claims):
        candidate_indices = [
            i for i, r in enumerate(l5_results)
            if r.get("result") == "pass" and claims[i].get("modality") == "observed"
        ]
    else:
        candidate_indices = [
            i for i, c in enumerate(claims)
            if c.get("modality") == "observed" and c.get("type") in ("presence", "action", "finding")
        ]

    # 2. Seeded deterministic sampling
    cfg_data = C.cfg()
    sample_size = cfg_data.get("check", {}).get("l6_sample", 60)
    seed = cfg_data.get("seed", 0)

    if sample_size is not None and len(candidate_indices) > sample_size:
        rng = random.Random(seed)
        sampled_set = set(rng.sample(sorted(candidate_indices), sample_size))
    else:
        sampled_set = set(candidate_indices)

    # 3. Build async coroutines for sampled claims
    async_items: list[tuple[int, Any]] = []
    for idx in sorted(sampled_set):
        async_items.append((idx, evaluate_perturbation(claims[idx], ctx)))

    if async_items:
        coros = [item[1] for item in async_items]
        results = llm.gather(coros, desc="L6")
    else:
        results = []

    # 4. Construct aligned output
    res_map = {}
    for (idx, _), r in zip(async_items, results):
        if r is None or not isinstance(r, dict):
            res_map[idx] = res("abstain", "perturbation execution error")
        else:
            res_map[idx] = r

    aligned: list[dict] = []
    for i in range(len(claims)):
        if i in res_map:
            aligned.append(res_map[i])
        else:
            aligned.append(res("abstain", "not in perturbation sample"))

    ctx.extra["L6"] = aligned
    return aligned
