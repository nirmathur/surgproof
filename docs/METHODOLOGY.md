# SurgProof methodology

How SurgProof turns surgical video into a verified dataset and verified narration, stage by stage. Each step names
the file that implements it, the models it calls, what it reads and what it writes. Numbers are in `PITCH.md` and
`narrator/out/eval/RESULTS.md`. This document describes procedure, not results.

**Core rule.** No human label is used to label, check or train. Ground truth (GT) is read only to score held-out
test videos after all decisions are written. Anything that uses GT for anything else is marked as an upper bound or
a development setting.

---

## 0. Overview

```
public teaching video ──► 1. manual (Argon)
                                │
CholecSeg8k frames ──► 2. prepare ──► 3. label (Flash) ──► 4. double-check + amber loop (Flash/Argon) ──► decisions
                                                                                                          │
                     5. boxes, tips, phase (Flash + Argon) ◄── accepted "present" labels ◄────────────────┤
                     6. masks: polygons, 2 passes ──► 7. mask self-correction                             │
                                                                                                          │
                     8. motion verifier (no LLM) ──► suspicion score, re-check queue                      │
                     9. narration claims ──► 8-layer claim checker ──► speak / hedge / drop               │
                                                                                                          ▼
                     10. human gate: ambers + 20-item audit ──► out/verified_dataset.jsonl ──► 11. student
                     12. scoring on held-out videos (GT joined here only)    13. integrity checker
```

| Model | Role |
|---|---|
| `gemini-3.8-flash` | bulk labelling, crop checks, critic, polygon drafts, narration claims, blind visual check (L5) |
| `gemini-4-argon` | manual writing, amber loop adjudication, pass-2 localisation, knowledge judge (L7) |
| none (OpenCV, SciPy) | motion verifier, rule layers L1–L4 and P of the claim checker |
| OpenCLIP ViT-B-32 | report-only, non-Gemini presence check (L5c) |

Every Gemini response is cached (`cache.sqlite`, `narrator/cache/`) so runs are reproducible and reruns are free.

---

## 1. Data and splits — `prepare_data.py`

- **Source:** CholecSeg8k, 101 clips of 80 consecutive frames at 25 fps (about 3.2 s each) from 17 laparoscopic
  cholecystectomy videos. 8,080 frames with pixel masks. CC BY-NC-SA 4.0, kept outside git.
- **Labels in scope** (`config.py: LABELS`): grasper, L-hook, liver, gallbladder, cystic duct, blood.
- **GT presence** (scoring only): a class is present if it covers ≥ 0.2% of the frame (≥ 0.1% for blood and cystic duct).
  The watershed mask value convention is auto-detected (`--inspect` to confirm by eye).
- **Split by video, seed 0:** 6 test videos (final scoring only), 3 calibration videos (upper-bound experiments only),
  the rest train. Test videos: video09, 12, 17, 28, 48, 55. No video appears in two splits.
- **Frame sampling:** every 8th frame of each clip (`FRAME_STRIDE = 8`), 10 frames per clip, about 1,010 frames.
  Each frame is sent to Gemini as a single still image. The model never sees video or neighbouring frames.
- **Dense mode (demo clips only):** `tools/dense_worker.py` labels all 80 frames of a chosen clip (see §6).
- **External sets** (generalisation, labelled after the manual was frozen): CholecT50 (40 videos, grasper and L-hook),
  Endoscapes2023 (10 videos, gallbladder and cystic duct). Scored only on the classes each set annotates.

## 2. The manual — `med_school.py`

Argon researches the procedure with Google Search and narrated public teaching videos, then writes its own
machine-readable labelling manual (`out/manual.json`): for each structure, what it looks like, where it is found and
what it is confused with. Every entry cites a source URL. The manual is frozen before test and external labelling.

## 3. Labelling — `label.py`

Per frame, Flash returns for each label: present/absent, confidence, a box (0–1000 coordinates) and a text evidence
string. Three decorrelated signals are recorded per label:

1. **Labeller:** full frame, with the manual in the prompt (`--mode manual`) or without it (`--mode blank`, baseline).
2. **Point-and-verify:** the claimed box is cropped and a fresh call, blind to the claim, says what it shows.
3. **Critic:** a separate call argues the claim is wrong and returns `p_wrong`.

Optional `--votes N` adds sampled runs (temperature 0.7) for majority-vote experiments.

## 4. Double-check and amber loop — `accept.py`, `verify/engine.py`

- **Trust score:** `confidence × (1 − p_wrong)`, halved if the blind crop check disagrees.
- **Status:** trust ≥ 0.70 → accept; < 0.35 → reject; in between, or the passes disagree → **AMBER**.
- **Amber loop (up to 3 rounds, Argon):** round 1 a zoomed crop plus the manual's criteria; round 2 a tighter crop
  plus differential diagnosis against the manual's `confused_with` list; round 3 multi-vote consensus. An amber
  resolves when Argon's verdict reaches confidence ≥ 0.80. Every round is logged to `out/amber_rounds.jsonl`.
- **Output:** `out/decisions.csv` (one row per frame × label, with status and evidence).

## 5. Localisation: boxes, tips, phase — `localize.py`

For labels accepted as present: pass 1 (Flash) and pass 2 (Argon, independent) each predict a box, tip points for
grasper and L-hook, and the surgical phase. If the two boxes have IoU ≥ 0.5 the box is accepted. Otherwise it goes
AMBER and is re-run on tighter crops for up to 3 rounds (`out/localize_rounds.jsonl`). Output: `out/annotations.jsonl`.

## 6. Segmentation masks — `tools/polygon_worker.py`, `tools/dense_worker.py`

- **Standard:** each accepted object is traced as a polygon twice (different phrasing and image scale). IoU ≥ 0.5
  → accepted, mask = intersection of the two. Otherwise AMBER and re-drawn on a tighter crop for up to 2 rounds.
- **Dense demo clips:** every frame, 4 structures, 3 drafts each; a pixel is kept if ≥ 2 drafts include it, an
  object is absent if ≥ 2 drafts are empty. About 960 calls per clip. Scored afterwards by `tools/score_dense.py`.
- **Vote experiment** (`tools/vote_experiment.py`): compares 1, 2, 3 and 5 drafts on a sample, scored against GT.

## 7. Mask self-correction — `tools/refine_worker.py`

The model's current outline is drawn on the frame and Gemini is asked to correct it (move, add or remove points).
Two independent correction passes; if they agree (IoU ≥ 0.6) the correction replaces the outline, otherwise the
previous outline is kept (`held`). Run for 2 rounds.

## 8. Motion verifier — `tracker/` (no LLM)

An independent check that shares no model with the labeller. Pipeline (`python -m tracker.<step> --source <src>`):

1. `index` — frames in consecutive order per clip.
2. `detect` — detections per frame. **Currently built from GT masks** (`source=gt`), the only module that reads masks.
3. `camera` — camera motion per frame pair (Farneback optical flow, tool regions excluded) and image quality
   (blur, contrast, brightness, specular highlights).
4. `track` — detections linked into tracks with camera motion removed (Hungarian matching on distance and IoU).
5. `signals` — 15 physical-plausibility signals: teleport, one-frame flicker, abrupt size change, identity swap,
   appearing mid-field (instruments enter through ports, so a real shaft reaches the edge of the view), etc.
6. `recheck` — signals combined into one suspicion score; re-check queue = 80% highest suspicion + 20% random
   controls, at a budget (default 10% of frame-labels). Output: `out/flags.parquet`.

**Evaluation:** `corrupt` injects 252 synthetic errors of 5 kinds (swap, false positive, size, drop, shift) into the
GT detections (`source=gt_corrupted`), and recall is measured per re-check budget against random re-checks and the
best single signal.

**Status:** the `--source gemini` input path exists but has not been run. It needs consecutive frames, so it can only
use Gemini output from dense clips (§6), not the stride-8 labels.

## 9. Narration and claim checker — `narrator/`

Post-hoc review of recorded video for teaching and documentation. Not intraoperative guidance, not diagnosis.

1. `events` — labels and tracker flags become a timeline (tool enters/exits, moving/still, anatomy
   confirmed/uncertain/rejected, blood).
2. `claims` — Flash writes **atomic claims** with timestamps, in two modes: *free* (frames only) and *grounded*
   (frames plus the label table).
3. `check` — every claim passes 8 layers, each returning pass / fail / abstain with a reason:

| Layer | Check | Model |
|---|---|---|
| L1 schema | entities in the vocabulary, consistent types | none |
| L2 rules | instrument-verb-target, phase order, tool fits phase, no "CVS achieved" after clipping, no non-inferable conditions; 15 rules quote-verified against sources | none |
| L3 temporal | no unexplained contradictions, no single-frame observations without an event | none |
| L4 evidence | claim matches the verified label table; amber label or high tracker suspicion → hedge | none |
| L5 blind visual | fresh call sees only the frame/crop and a yes/no question, never the narration | Flash |
| L6 perturbation | same question on a blacked-out frame; an unchanged "yes" means the answer came from priors | Flash |
| L7 knowledge | judge checks against retrieved KB passages and must quote a span; code verifies the span exists | Argon |
| P proportionality | a grade or condition must be the most conservative one the findings support | none |

4. **Verdict:** any fail → drop. Otherwise any hedge or uncertain evidence → hedge. An evidence layer passed → speak.
   Nothing backed it → drop.
5. `render` — surviving claims become narration text and TTS (`out/narration.jsonl`).
6. `eval` — free vs grounded scored against masks; 164 injected wrong claims of 9 types for per-layer recall.

**Status:** the main run used a **mock** label table (GT masks plus seeded noise) and tracker flags from corrupted
GT. Numbers that depend on L4 or on the grounded prompt are not real results. The real-label run is a 6-clip smoke test.

## 10. Human gate — `web/serve.py`, `app.py`

Before release a person: (a) clears every remaining AMBER with its evidence crops, and (b) signs off a deterministic
20-item random audit sample of auto-accepted labels. Actions are logged to `out/gate_log.jsonl`. The export
`out/verified_dataset.jsonl` stays locked until ambers left = 0 and the audit is signed off.

## 11. Student — `student.py`

A linear probe on DINOv2 features, trained on train videos only, from four label sources: *single* (every Gemini
label), *vote* (majority of 3 samples), *verified* (only accepted labels) and *oracle* (GT, upper bound only, not part
of the claim). Evaluated on every frame of the test videos. Liver (present in 100% of frames) is excluded from headline means.

## 12. Scoring — `eval/score_decisions.py`, `verify_localize.py`, `tools/score_masks.py`, `narrator/eval`

- GT is joined only here, only for test (or external) videos, after decisions are written.
- Confidence intervals: 2,000 bootstrap resamples of **whole videos**, paired deltas against the baseline.
- Every result is reported next to a baseline: single-pass labels, random re-checks or the best single signal.
- Liver is excluded from headline numbers. Calibration and oracle results are labelled upper bounds.

## 13. Integrity — `lanes/integrity.py`

Fails the build if display code contains hand-typed boxes, verdicts or metrics; if GT leaks into labeller output or
the verifier/localiser reads it; if the manual lacks sources or results lack sample sizes; or if a narration sentence
cites decision rows that don't exist. An independent spot audit re-labelled 24 random accepted test labels.

---

## Known gaps and open items

- **Tracker on Gemini labels:** not run (see §8). All motion-verifier numbers are on injected errors in GT boxes.
- **Narrator on real labels:** smoke test only (n = 6 clips). Mock-dependent numbers are plumbing, not results.
- **Same-family checks:** L5, L7 and the amber loop are Gemini checking Gemini. The one non-Gemini visual check
  (OpenCLIP) was at chance. The motion verifier is the only check with no model in common with the labeller.
- **Small n:** 6 test videos. Intervals are wide.
- **Pretraining overlap:** Cholec80 frames may be in Gemini's training data; external sets are reported separately.
- **To reconcile before submission:**
  - `PITCH.md` (Step table, "Label" row) says Gemini labels "every frame of whole consecutive clips". The main
    pipeline labels every 8th frame; only dense demo clips are every frame.
  - `README.md` reports 96.6% on 93.2% accepted over 1,400 labels; `PITCH.md` reports 98.5% over 75 decisions.
    These appear to be different evaluation sets and should be labelled as such.
