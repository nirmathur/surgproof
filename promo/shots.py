"""Canonical 6-shot specification for the 75-second {{NAME}} pitch video."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


PROJECT_NAME = os.environ.get("PROJECT_NAME", "{{NAME}}")

ILLUSTRATIVE_CAPTION = (
    "ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)"
)
REAL_PROOF_CAPTION = (
    "REAL PIPELINE TELEMETRY & LIVE APP.PY CAPTURE — NO HUMAN LABELS IN TRAINING"
)


@dataclass(frozen=True)
class ShotSpec:
    shot_id: str
    index: int
    filename: str
    title: str
    subtitle: str
    pipeline_cmd: str
    artifact_label: str
    duration_sec: float
    illustrative: bool
    prompt: str
    narration: str
    is_proof_slot: bool = False

    @property
    def badge_text(self) -> str:
        return ILLUSTRATIVE_CAPTION if self.illustrative else REAL_PROOF_CAPTION


SHOTS: Sequence[ShotSpec] = (
    ShotSpec(
        shot_id="shot_01_med_school",
        index=1,
        filename="shot_01_med_school.mp4",
        title="ACT 1 · AUTONOMOUS MEDICAL SCHOOL",
        subtitle="Gemini 4 Argon studies public narrated surgery video & writes its own visual manual",
        pipeline_cmd="python med_school.py --video <URL>",
        artifact_label="out/manual.json · 0 human labels",
        duration_sec=12.0,
        illustrative=True,
        prompt=(
            "Cinematic wide shot inside a dark-mode surgical AI research lab. "
            "A sleek glass workstation monitor displays an abstract surgical lecture "
            "waveform transforming into glowing structured knowledge nodes and "
            "anatomical wireframe schematics. Warm rim lighting, shallow depth of field, "
            "clean modern interface aesthetic, no text or charts, no people."
        ),
        narration=(
            "Surgical AI usually requires thousands of hours of surgeon annotation. "
            "{{NAME}} begins at Autonomous Medical School: Gemini 4 Argon watches public "
            "surgical lectures and writes its own structured visual manual with zero human labels."
        ),
    ),
    ShotSpec(
        shot_id="shot_02_label",
        index=2,
        filename="shot_02_label.mp4",
        title="ACT 2 · ZERO-SHOT BULK FRAME LABELLING",
        subtitle="Gemini 3.8 Flash labels instruments & anatomy across surgical frames using its manual",
        pipeline_cmd="python label.py",
        artifact_label="out/labels_manual.jsonl · 6 target classes",
        duration_sec=12.0,
        illustrative=True,
        prompt=(
            "Cinematic close-up of a high-resolution medical imaging monitor scanning "
            "a sequence of synthetic laparoscopic tissue phantom frames. Glowing cyan "
            "spatial bounding boxes and coordinate reticles lock onto surgical instruments "
            "and anatomical regions in smooth motion. Dark laboratory background, volumetric light, "
            "no readable text or fake charts."
        ),
        narration=(
            "Guided by the manual it authored itself, Gemini 3.8 Flash scans laparoscopic "
            "video frames in parallel, proposing bounding boxes and confidence scores for "
            "grasper, L-hook, gallbladder, cystic duct, blood, and liver."
        ),
    ),
    ShotSpec(
        shot_id="shot_03_verify",
        index=3,
        filename="shot_03_verify.mp4",
        title="ACT 3 · AUTONOMOUS SELF-VERIFICATION LOOP",
        subtitle="Agent checks its own labels: Accept (green) · Re-analyse crop (amber) · Escalate (red)",
        pipeline_cmd="python accept.py",
        artifact_label="out/decisions.csv · replaces human verifier",
        duration_sec=12.0,
        illustrative=True,
        prompt=(
            "Futuristic AI verification workstation screen showing a high-precision optical "
            "zoom lens inspecting a highlighted crop of a synthetic surgical simulation frame. "
            "Amber inspection rings pulse and lock into emerald-green verified indicators, "
            "while one uncertain frame routes into a crimson escalation lane. Smooth camera dolly, "
            "photorealistic lighting, no text gibberish."
        ),
        narration=(
            "Where prior systems like SurgSLOT needed human verifiers, {{NAME}} checks itself: "
            "cropping uncertain regions for a second opinion, accepting verified labels, "
            "and escalating ambiguous cases to a surgeon."
        ),
    ),
    ShotSpec(
        shot_id="shot_04_student",
        index=4,
        filename="shot_04_student.mp4",
        title="ACT 4 · SPECIALIST STUDENT DISTILLATION",
        subtitle="Only self-verified labels train the specialist classifier on visual embeddings",
        pipeline_cmd="python student.py",
        artifact_label="out/student.json · 0 human training labels",
        duration_sec=12.0,
        illustrative=True,
        prompt=(
            "Cinematic macro visualization of a neural distillation pipeline: streams of "
            "glowing data particles pass through a crystalline verification filter that blocks "
            "noisy red sparks and transmits pure emerald-green verified vectors into a compact "
            "glowing specialist processor core. Dark sci-fi documentary cinematography."
        ),
        narration=(
            "Only labels that survive self-verification train a lightweight specialist classifier "
            "on visual embeddings. Zero human labels touch training; ground truth only evaluates "
            "held-out test videos."
        ),
    ),
    ShotSpec(
        shot_id="shot_05_proof_slot",
        index=5,
        filename="shot_05_proof_slot.mp4",
        title="ACT 5 · REAL PIPELINE PROOF & STREAMLIT DEMO",
        subtitle="Held-out CholecSeg8k test evaluation + live app.py screen capture (no synthetic numbers)",
        pipeline_cmd="streamlit run app.py",
        artifact_label="out/results.json · out/student.json · app.py",
        duration_sec=14.0,
        illustrative=False,
        is_proof_slot=True,
        prompt="",
        narration=(
            "On held-out CholecSeg8k test videos, {{NAME}} reports real pipeline accuracy, "
            "coverage, and specialist student gains inside the live Streamlit interface, "
            "with calibration and oracle runs clearly marked as upper bounds."
        ),
    ),
    ShotSpec(
        shot_id="shot_06_narrate",
        index=6,
        filename="shot_06_narrate.mp4",
        title="ACT 6 · GROUNDED SURGICAL NARRATION",
        subtitle="Narrates the operation using only verified anatomy & audited timeline events",
        pipeline_cmd="python narrate.py --video <test_video>",
        artifact_label="out/narration_*.json · SurgVLP · SurgSLOT · Med-Gemini",
        duration_sec=13.0,
        illustrative=True,
        prompt=(
            "Sleek surgical education theatre display showing a synchronized endoscopic "
            "training replay timeline with glowing phase markers and clean audio waveform "
            "bars beneath the video frame. Modern dark clinical viewing room with soft "
            "ambient blue lighting, slow camera push-in, cinematic finish, no readable text."
        ),
        narration=(
            "Finally, {{NAME}} narrates the operation like Netflix for surgery, speaking "
            "only from verified labels it can stand behind. Built on SurgVLP, SurgSLOT, "
            "and Med-Gemini."
        ),
    ),
)


def total_storyboard_duration() -> float:
    return sum(s.duration_sec for s in SHOTS)
