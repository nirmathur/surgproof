# SurgProof

**No human labels. Only proof.**

**Demo video:** https://drive.google.com/open?id=1UfRDw7hE2OxEk_wMoK98qnTvxrzuAZT5  (also in `docs/media/surgproof_submission.mp4`)

## The problem
Surgical AI is starved of expert labels. *"The major bottleneck for data annotation in surgical applications is access to expert knowledge."* (Maier-Hein et al., surgical data science consensus, 2022; quote verified verbatim against the saved source in `docs/research/sources/`). Today's best annotation tools (Supervisely, CVAT, Encord) still need a human to seed every object, fix every tracking error and review every frame.

## What SurgProof does
An agent built on Gemini 4 Argon removes the human from surgical dataset creation:

1. **Learns the procedure** from a public teaching video and writes its own visual manual (`out/manual.json`, with source URLs).
2. **Labels real frames** (presence of instruments and anatomy), then **checks itself**: every label gets an independent second pass. Agreement means accept or reject; disagreement means AMBER, and an Argon **amber loop** re-examines it on tighter crops.
3. **Draws the annotations itself**: boxes, instrument-tip points, the surgical phase of each frame, and **segmentation masks** traced as polygons. Each is drawn twice, and kept only when the two passes agree.
4. **Corrects its own masks**: the model sees its outline over the frame and fixes it. A correction is kept only when two independent correction passes agree.
5. **A human does one final pass**, covering the remaining ambers plus a random audit sample, before the dataset is released (`out/verified_dataset.jsonl`).

Ground truth is never read by the agent. It is used only afterwards, to score held-out videos.

## Results (held-out test videos the system never tuned on)
All numbers below were recomputed by an independent acceptance check from the raw files (`out/decisions.csv`, the mask PNGs, and ground truth), not taken from the pipeline's own summaries.

| What | Result | n |
|---|---|---|
| Presence labels, after self-check (excluding liver, present in 100% of frames) | **96.6% accurate on the 93.2% it accepted** (video-level 95% CI 93.6% to 99.4%); raw single pass 92.3% | 1,400 labels, 6 unseen CholecSeg8k videos |
| Boxes vs ground-truth masks | mean IoU 0.76 | 258 accepted boxes |
| AI-drawn masks, first draft (2 passes, intersection) | mean IoU 0.55 | 585 masks, 6 videos |
| ...after one self-correction round | mean IoU 0.60 | 578 masks |
| ...3-draft majority vote (production rule) | **mean IoU 0.65** | 320+ masks (run finishing at submission) |

**We measured our own pipeline and changed it.** Keeping only the overlap of two drafts shrinks masks, because the model's outlines tend to under-cover the object. Comparing ways of combining drafts on the **calib** split (the only split we tune on; 60 objects): intersection 0.56, single draft 0.66, 3-vote majority 0.69, 5-vote majority 0.71, union of 5 drafts 0.75. The same ranking held on a 48-object test sample (`out/vote_experiment*.json`). Production uses the 3-vote majority. Union-of-5 is the calib-selected next step.

Numbers are read from `out/results.json`, `out/localize_results.json`, `out/mask_results*.json` and `out/vote_experiment_calib.json`. External sets (Endoscapes2023: gallbladder and cystic duct; CholecT50: grasper and hook) are scored only on the labels they actually annotate.

## Integrity
`python3 lanes/integrity.py` fails the build if display code contains hand-typed boxes, verdicts or metrics; if ground truth leaks into the labeller or verifier; if a label or decision cannot be traced to a recorded Gemini response; or if a narration sentence lacks accepted evidence. It reported 0 FAIL at submission. Independent spot audit: the orchestrator re-labelled 24 random accepted test labels with its own prompt and agreed on 23 of 24; the system was correct on all 24 against ground truth.

Known limits: 6 test videos is a small n (see the confidence intervals). Gemini may have seen public Cholec80 frames during pretraining, which is why external datasets are reported separately. Round 2 of self-correction added nothing over round 1.

## Run it
```bash
cp .env.example .env                 # set GEMINI_API_KEY in .env (or export it)
make setup && make run               # or: ./run.sh (serves unified app on http://localhost:8600)
```
Manual steps:
```bash
pip install -r requirements.txt -r web/requirements.txt
python3 web/export.py                # builds web/data from out/
python3 web/serve.py                 # annotation toolbox on http://localhost:8600
```
Pipeline: `prepare_data.py` → `med_school.py` → `label.py` → `accept.py` (verifier and amber loop) → `localize.py` → `tools/polygon_worker.py` (AI masks) → `tools/refine_worker.py` (self-correction) → `student.py` and `eval/`.

Demo video: `out/promo/` (built only from real pipeline outputs and real frames; AI-generated imagery is tagged as such).

## Prior work
SurgAtlas (large dataset of public surgical videos), SurgVLP, SurgSLOT, Med-Gemini. Our step: the agent replaces the human annotator and verifier, end to end, with every claim traceable to a recorded model response.

## Next
Any procedure from public teaching video. Critical View of Safety assessment. Mask propagation across frames. Export to COCO, Supervisely and CVAT formats.

Data: CholecSeg8k, Endoscapes2023 and CholecT50 (CC BY-NC-SA 4.0). Not redistributed here.
