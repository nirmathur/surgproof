# SurgProof: surgical training data, labelled and checked by machines, released by a person

*Laparoscopic cholecystectomy · results as of 31 Oct 2026*

Surgical AI is held back by expert annotation: surgeons cannot spend hours drawing boxes on video frames.
SurgProof labels surgical video with Gemini and checks every label against independent evidence.
Labels the checks disagree on are re-examined in a loop, and a person is sent only the few that stay in doubt.
A specialist model is then trained only on labels that survived.

| | |
|---|---|
| **98.5%** | accuracy of the labels SurgProof accepts. Single-pass Gemini labels: 90.7%. Held-out test videos, 95% CI 95.8–100%. |
| **8 → 1** | uncertain labels left for a person after 3 loop rounds. All 7 labels the loop resolved were correct. Out of 75 test labels. |
| **88%** | of injected tool-label errors found by re-checking 10% of labels. Random re-checks of the same 10% find 16%. Motion verifier, no LLM. |
| **0** | human labels used to label, check or train. Ground truth is used only to score held-out videos at the end. |

---

## Value creation: a person's time goes to the labels machines disagree on

Current surgical annotation tools speed people up, but a person still seeds, corrects and reviews every clip.
SurgProof replaces each of those steps with an agent. People keep one job: a final pass before anything becomes a dataset.

| Step | Typical annotation tool | SurgProof |
|---|---|---|
| Seed | A person draws the first box or clicks the first points | The agent writes its own procedure manual from public teaching video: 7 sources, 57 timestamped lessons |
| Label | A person tags frames with model assistance | Gemini labels every frame of whole consecutive clips |
| Correct | A person fixes tracking errors and restarts the tracker | An amber loop re-examines disputed labels with tighter crops, the manual's criteria and fresh votes |
| Review | Reviewer and domain-expert roles check the result | An independent second pass, a motion verifier without an LLM, and a claim checker for narration |
| Release | Expert sign-off | A person clears the remaining ambers and a random audit sample of accepted labels. Export stays locked until that is done |

On the six held-out test videos, the person's queue after the loop was **1 of 75 labels**, plus the audit sample.
What they receive is a dataset with provenance: each accepted label carries the evidence and the checks that passed it.

The same machinery makes generated narration safer. Narration checked by SurgProof's claim checker named no structure
that wasn't in the frame, and it dropped 44 of 60 safety-critical claims (critical view of safety, grades,
clip-and-cut decisions) that no independent evidence supported.

---

## Use cases: one verified pipeline, four places it creates value

Every use case below draws on the same output: surgical video turned into labels, tracks and statements that each
carry their evidence. The status says how far we have taken it in this project.

### 1. Operative documentation for surgeons
**Status: prototype**

Surgeons write operative notes from memory after long procedures. SurgProof can draft the note from the video: what
was done, in which order, and with which instruments. Each statement links to the frames that support it, and the
surgeon edits and signs.

- **Built:** grounded narration and a per-clip surgery report. The claim checker drops or hedges statements nothing
  backs: of 60 safety-critical claims, it dropped 44 and hedged 12.
- **Needs:** full-length operations, the hospital's note templates, and evaluation by surgeons.

### 2. Training data for surgical foundation models and robotics
**Status: shown here**

Surgical foundation models and surgical robots need large, labelled video. SurgProof produces it without an
annotation team: presence labels, boxes and instrument tracks with camera motion removed. Each label records which
model, prompt, sources and checks produced it.

- **Built:** a verified dataset export with provenance per label, 98.5% accurate on held-out video. Instrument
  trajectories come from the motion verifier.
- **Needs:** more procedures and label types (phases, actions, tool–tissue contact), licensed source video, and 3D
  or kinematic data for robotics.

### 3. Knowledge repositories for surgical education
**Status: prototype**

Residents learn from scattered lectures, papers and videos. The agent condenses public teaching material into a
structured manual and rule base: phases, anatomy, safety criteria and common decisions, each traced to a source and a
video timestamp. It can be paired with labelled example frames.

- **Built:** a self-written manual from 7 sources with 57 timestamped lessons, and a 39-rule knowledge base. A
  pipeline also builds an atlas for other procedures from a single public video, presented as a showcase, not as proof.
- **Needs:** expert review. Only 15 of the 39 rules are quote-verified against their sources so far.

### 4. Live assistance during surgery
**Status: concept**

During an operation, a model could name instruments and structures in view and warn before a critical step, for
example when the critical view of safety has not been shown before clipping. The model that runs live would be the
small specialist trained on verified labels, not Gemini.

- **Built:** the specialist model and the safety rules it would check against. Everything in this project runs offline.
- **Needs:** real-time speed (Gemini narration takes about 30 s per clip, and specialist speed is not yet measured),
  prospective clinical validation and regulatory approval. It would only ever support the surgeon, never act alone.

---

## Methodology: six steps from public video to a verified dataset

Shown on laparoscopic cholecystectomy, the one procedure with public pixel-level ground truth (CholecSeg8k, 17 videos).
Six videos are held out for testing. No frame from a test video is used for tuning.

1. **Learn the procedure.** A Gemini Argon agent studies public teaching sources and writes a manual: anatomy,
   instruments, phases and safety criteria, each with its source and timestamp.
2. **Label whole clips.** Gemini Flash labels consecutive frames with the manual in hand: which instruments and
   structures are present (grasper, L-hook, gallbladder, cystic duct, blood) and where they are.
3. **Check every label independently.** Three kinds of check, built so they fail differently from the labeller:
   - **Second pass.** A different prompt, a zoomed crop and a stronger model look again. Both passes agree: the label
     is accepted (green) or rejected (red). They disagree, or confidence is low: the label turns amber.
   - **Motion and physics verifier.** Tracks each instrument across frames, removes camera motion and asks what
     physically can't happen: a tool that teleports, flickers out for a frame, changes size abruptly, swaps identity,
     or appears mid-field. Instruments enter through ports, so a real shaft reaches the edge of the scope view.
     15 signals feed one suspicion score and a ranked re-check queue. No LLM; about 1 minute for 8,080 frames on a laptop.
   - **Claim checker.** Every sentence of generated narration passes 7 layers: schema, rules quoted from the surgical
     literature, temporal order, the verified labels, a blind visual check, a blacked-out-frame test, and a knowledge
     check. Claims without support are dropped or hedged. 15 rules are quote-verified against fetched sources.
4. **Loop on disagreement.** Argon re-examines each amber for up to three rounds with a tighter crop, the manual's
   criteria and fresh votes. Each round resolves some ambers to accept or reject.
5. **Final human pass.** A person reviews what is still amber plus a random audit sample of auto-accepted labels,
   with their evidence crops. Nothing is exported before sign-off.
6. **Train a specialist on verified labels only.** A small student model (a linear probe on DINOv2 features) learns
   only from accepted labels and is scored on every frame of the held-out videos.

---

## Evidence: what the checks buy, on videos the system never saw

Every number below comes from held-out videos. Confidence intervals resample whole videos, not frames, because frames
from one video are correlated.

| Held-out set | Single pass | Accepted by SurgProof | Gain (95% CI) | Labels kept |
|---|---:|---:|---:|---:|
| CholecSeg8k test (6 videos, 5 labels, 75 decisions) | 90.7% | 98.5% | +7.9 pts (4.5 to 11.1) | 90.7% |
| CholecT50, external (40 videos, grasper and L-hook) | 89.6% | 91.2% | +1.6 pts (0.2 to 3.4) | 95.0% |
| Endoscapes2023, external (10 videos, gallbladder and cystic duct) | 67.6% | 70.5% | +2.9 pts (0.6 to 7.8) | 93.9% |

The manual was frozen before the external sets were labelled. Accuracy is over the labels each set can score.

### The amber loop shrinks the human queue on every set
Labels still amber after the initial double-check and after each loop round:

| Set | Initial | Round 1 | Round 2 | Round 3 |
|---|---:|---:|---:|---:|
| CholecSeg8k test | 8 | 6 | 3 | 1 |
| CholecT50 | 29 | 28 | 13 | 3 |
| Endoscapes2023 | 28 | 28 | 6 | 2 |

### Targeted re-checks find tool-label errors far faster than random ones
Motion verifier on held-out videos: share of 252 injected errors found, by re-check budget (% of frame-labels).
Chart: `out/eval/budget_curve.png`.

| Re-check budget | Ranked by suspicion score | Best single signal (size jump) | Random |
|---:|---:|---:|---:|
| 5% | 58.7% | 38.9% | 8.8% |
| 10% | 87.7% | 46.8% | 16.1% |
| 20% | 94.8% | 55.2% | 28.8% |
| 30% | 96.4% | 64.7% | 41.0% |

At a 10% budget the ranked queue finds every injected label swap, 96% of false positives, 90% of size errors, 89% of
dropped detections and 81% of shifted boxes. Tracks that end early in mid-field are the hardest (46%). Combining all
signals roughly doubles what the best single signal finds.

### Specialist trained on verified labels
Trained on 554 verified labels instead of all 630, the student scores **90.3%** on held-out frames against **88.9%**.
The gain is +1.4 points (95% CI −0.1 to +2.4) and positive in 97% of video resamples. A student trained on human
ground-truth labels scores 88.1%.

### Narration that only says what it can back
Over 28 test clips, grounded narration named **0.43** structures per clip that weren't there. After the claim checker
it named **0**, while still covering 91% of scorable content. On 164 deliberately wrong claims, the checker caught **98.8%**.

---

## How we kept it honest

- **Split by video.** Tuning and test sets share no video. All modules use the same split.
- **No ground truth in the loop.** Label files carry no ground-truth fields; an audit checks every file. Ground truth
  is joined only at scoring time.
- **Video-level confidence intervals.** 2,000 bootstrap resamples of whole videos, and paired deltas.
- **Uninformative labels removed.** Liver is in 100% of frames and is left out of the headline numbers.
- **Upper bounds labelled.** Anything that used human labels is reported as an upper bound only.
- **Baselines alongside.** Every result is compared with single-pass labels, random re-checks or the best single signal.

## What is not proven yet

- **Small samples.** 75 test decisions across 6 videos. Intervals are wide.
- **The manual did not help raw labelling here.** Without it, Gemini scored 92.5% on the test labels; with it, 92.3%.
- **The student gain is small.** Its interval touches zero.
- **Injected errors are easier than real ones.** Motion-verifier and claim-checker recall come from synthetic errors.
  The motion verifier has not run on Gemini's own boxes yet.
- **Gemini checks Gemini.** Some misses between Gemini-based checks are correlated. The one non-Gemini visual model we
  tried (zero-shot CLIP) was at chance. The motion verifier is the one check with no model in common with the labeller.
- **Cystic duct remains hard.** On Endoscapes2023, raw accuracy for the cystic duct is 38%.
- **Use cases beyond the dataset are unevaluated.** Documentation, education and live assistance have not been tested
  with surgeons or patients.

---

*Data: CholecSeg8k (CC BY-NC-SA 4.0), CholecT50, Endoscapes2023. Sources in this repository:
`coord/artifacts/eval_report.md`, `out/eval/`, `narrator/out/eval/RESULTS.md`. Narration cost with full checking:
about $0.10 per clip. Web version: `web/story/index.html`.*
