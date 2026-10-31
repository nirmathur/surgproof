"""Canonical 7-shot specification for the 52.0-second SurgProof product demo (promo/DESIGN.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PROJECT_NAME = "SurgProof"
PROJECT_TAGLINE = "No human labels. Only proof."


@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    index: int
    filename: str
    duration_sec: float
    caption: str
    max_words: int = 6

    @property
    def word_count(self) -> int:
        return len(self.caption.split())


SHOTS: Sequence[ShotSpec] = (
    ShotSpec(
        shot_id="shot_01_raw_frame",
        index=1,
        filename="shot_01_raw_frame.mp4",
        duration_sec=7.0,
        caption="Surgical video. No labels.",
    ),
    ShotSpec(
        shot_id="shot_02_manual",
        index=2,
        filename="shot_02_manual.mp4",
        duration_sec=8.0,
        caption="Writes its own visual manual.",
    ),
    ShotSpec(
        shot_id="shot_03_labels",
        index=3,
        filename="shot_03_labels.mp4",
        duration_sec=7.0,
        caption="Labels a real frame.",
    ),
    ShotSpec(
        shot_id="shot_04_verify",
        index=4,
        filename="shot_04_verify.mp4",
        duration_sec=10.0,
        caption="Checks itself. Green, amber, red.",
    ),
    ShotSpec(
        shot_id="shot_05_gate",
        index=5,
        filename="shot_05_gate.mp4",
        duration_sec=8.0,
        caption="Human clears only amber cases.",
    ),
    ShotSpec(
        shot_id="shot_06_result",
        index=6,
        filename="shot_06_result.mp4",
        duration_sec=7.0,
        caption="Held-out test accuracy, zero labels.",
    ),
    ShotSpec(
        shot_id="shot_07_end",
        index=7,
        filename="shot_07_end.mp4",
        duration_sec=7.0,
        caption="SurgProof. No human labels. Only proof.",
    ),
)


def total_storyboard_duration() -> float:
    return sum(s.duration_sec for s in SHOTS)
