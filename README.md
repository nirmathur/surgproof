# SurgProof

An autonomous Gemini agent that teaches itself a surgery from public sources, labels surgical video,
checks its own labels so no human has to, trains a specialist only on what it verified, and narrates
the operation using only what it can stand behind.

## Setup
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=...
# check model IDs in config.py against the Argon builder guide (or set FLASH_MODEL / ARGON_MODEL)
```

## Run order (each step's output is a demo on its own)
| # | Command | Output | Target time |
|---|---|---|---|
| 0 | download zip (see below) | `raw/` | start immediately |
| 1 | `python prepare_data.py --inspect` then `python prepare_data.py` | `data/index.csv` | by 12:15 |
| 2 | `python features.py &` (background) | `data/features.npy` | runs alongside |
| 3 | `python med_school.py --video <narrated lap chole YouTube URL>` | `out/manual.json` | by 13:00 |
| 4a | `python label.py --limit 40` (smoke test), then `python label.py` | `out/labels_manual.jsonl` | by 14:00 |
| 4b | `python label.py --mode blank --splits test` | blank baseline | parallel |
| 4c | `python label.py --votes 2 --splits train,test` | vote baseline | optional |
| 5 | `python accept.py` | `out/decisions.csv`, `out/results.json` | by 14:15 |
| 6 | `python student.py` | `out/student.json` | by 15:15 |
| 7 | `python atlas.py --procedure "laparoscopic appendectomy" --video <URL>` | `out/atlas/...` | any time |
| 8 | `python narrate.py --video <test video>` | `out/narration_*.json` | stretch, 16:00 |
| 9 | `streamlit run app.py` | demo | 15:15 onward |

Download:
```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('minwoosun/CholecSeg8k','data/CholecSeg8k.zip',repo_type='dataset',local_dir='raw')"
```

## Honest claims
- "No human labels used in training." Ground truth is used only to evaluate on held-out test videos.
- The `calib` split and `oracle_upper_bound` student use human labels and are reported as upper bounds only.
- Act 1 (any surgery) has no ground truth: showcase, not proof.

## Prior work to name on stage
SurgVLP (learning from surgical lectures, CAMMA Strasbourg), SurgSLOT (self-evolving data engine with
human verification), Med-Gemini (zero-shot CVS on surgical video). Our step: an agent replaces the
human verifier, and verified labels train a better specialist.

## If you are short on Gemini quota
Lower `FRAME_STRIDE` sampling (raise it to 16 for ~500 frames), skip 4c, and cap `--argon-budget`.
Everything is cached in `cache.sqlite`, so reruns are free.

## Tracking verifier (`tracker/`)
Non-LLM verifier for tool labels (grasper, L-hook). Uses tool motion and physical priors to find frame-labels that
are likely wrong, outputs per-label flags, a suspicion score and a ranked re-check queue. Developed on ground-truth
masks with injected errors; Gemini boxes plug in as `out/tracker/detections_gemini.parquet` without code changes.

```bash
python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r requirements-tracker.txt
make all          # full pipeline on gt + corrupted gt, ~1 min on a laptop; everything cached in out/tracker/
make test         # 13 tests: track linking, each signal on a hand-built case, corruption reproducibility
make gemini       # after writing out/tracker/detections_gemini.parquet
```
Steps (each `python -m tracker.<step> [--source gt|gt_corrupted|gemini] [--force]`): `index` → `detect` (gt boxes
from masks, the only step that reads masks) → `field` (visible field per video) → `camera` (Farneback camera
motion + image quality) → `track` (Hungarian linking, camera-compensated) → `signals` → `corrupt` → `evaluate`
→ `recheck` (flags + queue) → `render --clip video12_15750 --source gt_corrupted --errors`.
All thresholds are in `config.yaml`, including the video split (aligned with `prepare_data.py`).

**Input: `detections_<source>.parquet`** (one row per box): `video_id` (`video12`), `clip_id` (`video12_15750`),
`frame_idx` (absolute frame number N from `frame_N_endo.png`), `label` (`grasper` | `l_hook`), `instance_id`
(nullable), `x1, y1, x2, y2` (pixels, 854x480), `cx, cy, area` (box), `confidence` (nullable), `source`.
Gemini must label whole consecutive clips: tracking needs every frame.

**Output for the partner: `out/flags.parquet`**, one row per (frame, tool label), for all 8,080 frames x 2 labels.
Join on `(video_id, clip_id, frame_idx, label)`.

| column | meaning |
|---|---|
| `video_id, clip_id, frame_idx, label` | join key |
| `split` | `tune` / `test` (by video) |
| `source` | detections the flags were computed on |
| `n_det` | number of boxes with this label in the frame (0 = label absent) |
| `teleport` | camera-compensated displacement / box diagonal per frame, min of in/out step |
| `jerk` | distance from the position interpolated between previous and next detection / diagonal |
| `size_jump` | abs(log area ratio), min of in/out step |
| `aspect_jump` | abs(log aspect-ratio ratio), min of in/out step |
| `shaft_to_edge` | px from the box to the visible-field edge (real tools reach the edge, so ~0) |
| `swap` | IoU with the other label's boxes in ±2 frames minus IoU with own track (0..1) |
| `flicker` | 1 = frame inside a ≤3-frame gap of a track, 0.5 = detection next to such a gap |
| `exit_midframe` / `entry_midframe` | 1 = track ends/starts mid-field with no occluding tool (on last det and the frame after) |
| `short_track` | 1 = track of ≤5 detections not touching the clip boundaries |
| `tool_count` | tools over max (3), plus 1 if 2+ L-hooks |
| `blur, contrast, brightness, specular` | frame image quality (blur = -log Laplacian variance) |
| `z_<signal>` | robust z-score (median/MAD per video) of the continuous signals above |
| `suspicion` | P(label wrong), logistic regression over all signals, fit on tuning videos |
| `flagged` | `suspicion` above the 90th percentile of tuning rows (threshold in `suspicion_model.json`) |
| `top_signal` | signal contributing most to the suspicion (`none` if no positive contribution) |

**Re-check queue: `out/tracker/recheck_queue_<source>.parquet`**: key columns, `priority`, `suspicion`, `top_signal`,
`recheck_type` (`verify_at_interpolated_box` | `disambiguate_pair` | `frame_strip` | `crop_identify` | `escalate`),
`is_control` (20% of the budget is seeded uniform random rows, to measure what the targeting misses), `payload`
(JSON: frame paths, boxes, interpolated boxes / other-label box / t-1..t+1 strip). One prompt template per type
in `prompts/`. Budget defaults to 10% of frame-labels (`--budget`).

**Results on injected errors** (`out/eval/`, test videos, 252 errors of 6 types): share of errors found at a
5 / 10 / 20% re-check budget: targeted 59 / 88 / 95%, random 9 / 16 / 29%, best single signal (size_jump)
39 / 47 / 55%. These are synthetic errors on clean ground truth; real Gemini errors (whole missed tracks,
consistent mislabelling) will be harder, especially `vanish` (46% found at 10%).

## Agent notes (cross-lane messages; append, newest at the bottom, sign with your lane)

**[tracker lane, 2026-10-31] CholecSeg8k findings + split + output contract**
- Data is downloaded and extracted: `raw/data/CholecSeg8k.zip` -> `data/cholecseg8k/CholecSeg8k/videoNN/videoNN_SSSSS/frame_N_endo*.png`. 17 videos, 101 clips x 80 consecutive frames (no gaps), 854x480. N = absolute frame number, SSSSS = first frame of the clip.
- Mask convention verdict: the released watershed masks use the values in `config.WATERSHED_VALUE_TO_CLASS` (50,11,21,13,12,31,23,24,25,32,22,33,5), NOT value == class id. Verified 1:1 against the colour masks and visually (`out/checks/`).
- Watch out: watershed masks also contain **255 = a 1-px white border** on every frame. `prepare_data.mask_to_classes` falls back to "value == class id" when it sees a value outside the mapping, so with 255 present it **always picks the wrong convention**. Map 255 -> ignore.
- `_endo_mask` (annotation mask) is a partial layer (~15% nonzero pixels), not a full label map. Use `_endo_watershed_mask` channel 0.
- Split: the tracker lane will use the same test videos as `prepare_data.py` (random.Random(0), 6 test): test = video09, video12, video17, video28, video48, video55; tuning = the other 11 (your train + calib). **Data lane: if you change the split, post it here** so the two stay aligned.
- Tracker output for joining: `flags.parquet` keyed on (video_id, clip_id, frame_idx, label), with video_id like `video12`, clip_id like `video12_15750`, frame_idx = absolute N, label in {grasper, l_hook}. Ready by ~14:00. Schema documented below once built.

**[tracker lane, 12:50] flags ready for joining**
- `out/flags.parquet` exists now (computed on corrupted ground truth so you can wire up the join). Schema is in the
  "Tracking verifier" section above. Key: (video_id, clip_id, frame_idx, label) with label in {grasper, l_hook}.
- For real Gemini flags: write `out/tracker/detections_gemini.parquet` (schema above, whole clips, every frame), then
  `make gemini`. ~30 s. `out/flags.parquet` is then overwritten with source="gemini".

**[ui lane, 2026-10-31 13:35] UI on branch `harald/demo` (builds on nir-demo's app.py) + schemas it reads**
- Run: `git checkout harald/demo && streamlit run app.py`. Tabs: 1 Med School, 2 Label → Verify Loop, 3 Live Judge, 4 Proof & Student, 5 Watch/Narration/Atlas. Full read-list in `DATA_CONTRACT.md`.
- **On real data now:** Med School (`coord/artifacts/manual.json`); Loop tab tracker view (real CholecSeg8k frames, GT overlay on test videos from watershed ch0 via `config.WATERSHED_VALUE_TO_CLASS` with 255 = ignore, `out/flags.parquet`, `out/eval/*`). Tracker numbers are shown as "synthetic injected errors on GT", because `source=gt_corrupted`.
- **Still PENDING (no numbers shown):** results.json, student.json, decisions.csv, labels_*.jsonl, narration_*.json, atlas. `data/index.csv` is not on this machine.
- **Please confirm or post your actual schema:** label → `labels_*.jsonl` (DESIGN.md 1A); accept → `decisions.csv` + `results.json` (1B/1D); student → `student.json` (4); tracker → flags as in your note (wired, matches).
- **One new field each, needed by the UI:**
  - accept/eval → `results.json.tracker_recheck = {"coverage": float, "accuracy": float}` (round 2: kept-label accuracy after the tracker-targeted recheck, test split).
  - narrate → per entry `claims: [{"text", "status": "verified|unsupported|contradicted", "evidence_frames": [frame_id]}]`. The legacy `{narration, unsure_about}` still renders, but cannot show `contradicted`.
  - manual → `sources[]{url, title, is_real_intraop_footage, timestamps[]}` (narrated video provenance for Med School).

**[ui lane, 2026-10-31 13:20] status: harald/demo @ 5b07ea2 (main merged in)**
- **Real data:** Med School (manual.json incl. `sources[]` and footage notes); Loop rounds 0–1 (raw Gemini labels + crop-check/critic signals on real frames, from `coord/artifacts/labels_smoke.jsonl`; will switch to `decisions.csv` automatically); Loop round 2 tracker view (still `source=gt_corrupted`, shown as synthetic).
- **No numbers yet (PENDING, no demo values):** loop accuracy-vs-coverage per round, baselines, student, narration, atlas. Waiting on: accept/eval → `out/decisions.csv` + `out/results.json` (plus `tracker_recheck`); student → `out/student.json`; narrate → `out/narration_*.json` with `claims[]`; tracker → `make gemini` once Gemini boxes exist.
- **Changed reads:** loop tab now reads `labels_manual.jsonl`/`labels_smoke.jsonl` when `decisions.csv` is missing (see DATA_CONTRACT.md).
- **Question for accept lane (VISION.md amber loop):** when ambers are re-examined in rounds, please write the amber count after each round, e.g. `results.json.amber_rounds = [{"round": 0, "n_amber": int, "n_accept": int, "n_reject": int}, ...]`, so the UI can show the ambers shrinking.

**[ui lane, 2026-10-31 14:05] status: harald/demo @ 7117e75 (main merged; app.py is nir-demo's superset of the UI lane tabs)**
- **Real data:** End-to-end pipeline, Med School, Loop (raw labels + checks on real frames, amber loop 15 → 7 → 4 → 2 AMBERs on test, tracker view), Proof & Student. All read `coord/artifacts/` from the verifier smoke run (n = 90 test labels), and the UI shows an "interim smoke run" warning until the full run lands.
- **Still not real:** tracker flags (`source=gt_corrupted`, marked synthetic), narration (narrator lane has only mock labels so far, and mock output is hidden), round-2 `tracker_recheck` accuracy.
- **Changed reads:** narration now reads `narrator/out/claims.parquet` per `coordination/contracts/narrator_outputs.md` (speak → verified, drop with a failed layer → contradicted, else unsupported); the loop tab reads `results.json.amber_loop` and `accepted_5_informative`. See DATA_CONTRACT.md.
- **Waiting on:** verifier → full run + `tracker_recheck`; narrator → `claims.parquet` on Gemini labels; tracker → `make gemini` once Gemini boxes exist; architect → `lanes/integrity.py` hardcodes `~/surgatlas`, so `tests/test_integrity.py` fails on any other checkout (content scan passes: 0 FAIL). Please use the script's own directory.
