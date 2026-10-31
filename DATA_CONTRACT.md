# DATA_CONTRACT.md: what `app.py` reads (UI lane, harald/demo)

Canonical producer schemas live in `docs/DESIGN.md` (architect). This file lists what the UI actually
reads, with the fallback when a file is missing. The real files win: if a producer's output differs,
the UI loader changes, not the producer. Every lookup goes to `out/<name>` first, then `coord/artifacts/<name>`.

Status at 2026-10-31 13:20 on Harald's machine (after merging main): **real** = file exists and the UI renders it.

| File | Producer | Tab | Status | Fallback when missing |
|---|---|---|---|---|
| `out/manual.json` | med_school.py | 1 Med School | real (`coord/artifacts/manual.json`, incl. `sources[]`) | PLACEHOLDER note |
| `data/cholecseg8k/CholecSeg8k/<video>/<clip>/frame_<N>_endo.png` | dataset | 2 Loop, 5 Narration | real | grey placeholder canvas |
| `.../frame_<N>_endo_watershed_mask.png` | dataset | 2 Loop (GT overlay toggle, test videos only) | real | overlay skipped |
| `out/flags.parquet` | tracker | 2 Loop | real, but `source=gt_corrupted` (synthetic errors) | PLACEHOLDER note |
| `out/tracker/detections_<source>.parquet` | tracker | 2 Loop (boxes) | real | no boxes drawn |
| `out/tracker/recheck_queue_<source>.parquet` | tracker | 2 Loop | real | hidden |
| `out/eval/summary.json`, `out/eval/budget_curve.csv` | tracker | 2 Loop | real (synthetic errors) | hidden |
| `out/results.json` | accept.py / eval/ | 2 Loop, 4 Proof | missing | `PENDING` metrics, no numbers |
| `data/index.csv` | prepare_data.py | 3 Live, 4 Proof, 5 Watch | missing here (on Nir's machine) | PLACEHOLDER |
| `out/labels_manual.jsonl` → else `coord/artifacts/labels_smoke.jsonl` | label.py | 2 Loop (rounds 0–1 viewer), 4 Proof | smoke set real (40 frames x 6 keys, matches DESIGN.md 1A) | PLACEHOLDER |
| `out/labels_blank.jsonl` | label.py | 4 Proof | missing | hidden |
| `out/decisions.csv` | accept.py | 2 Loop (preferred over labels), 5 Watch, Human Gate | missing | labels without status (grey boxes) |
| `out/student.json` | student.py | 4 Proof | missing | `PENDING` metrics |
| `out/narration_<video>.json` | narrate.py | 5 Watch | missing | PLACEHOLDER note |
| `out/atlas/*/dataset.json` | atlas.py | 5 Watch ("showcase, no ground truth") | missing | caption |

## Fields the UI uses

**manual.json**: `procedure`, `phases[]`, `critical_events[]`, `general_rules[]`, `structures[]{key, name, appearance,
where_found, when_visible, confused_with[], how_to_tell_apart}`, and optionally `sources[]{url, title, is_real_intraop_footage,
timestamps[], source_type, real_intraop_footage_note}`. The note is shown under each source (e.g. a lecture over a still photo is not intra-op video).

**GT overlay**: watershed mask channel 0 → class id via `config.WATERSHED_VALUE_TO_CLASS`; `255` (1-px border) and unknown
values → ignore. Classes drawn are the `config.LABELS` keys. Shown only for test videos (video09, 12, 17, 28, 48, 55), as an
evaluation view.

**flags.parquet** (key `video_id, clip_id, frame_idx, label`): `split`, `source`, `n_det`, `suspicion`, `flagged`, `top_signal`.
`source != "gemini"` makes the UI show a "synthetic errors, not a Gemini result" warning.
**detections_<source>.parquet**: `clip_id, frame_idx, label, x1, y1, x2, y2` (pixels, 854x480).
**recheck_queue_<source>.parquet**: `clip_id, frame_idx, label, recheck_type, is_control`.
**eval/summary.json**: `found_at_{5,10,20}pct_{targeted,random,best}`, `recall_by_type_at_10pct{}`.
**eval/budget_curve.csv**: `budget` + one column per method.

**results.json** (DESIGN.md 1D). Loop rounds read:
- round 0 raw: `manual_raw_acc` (coverage 1.0)
- round 1 self-check: `accepted.{coverage, accuracy}`
- round 2 tracker recheck: **`tracker_recheck.{coverage, accuracy}`**. Not in DESIGN.md yet; requested from accept/eval.
Baselines: `blank_raw_acc`, `manual_raw_acc`, `{blank_single_conf, single_conf, majority_vote, upper_bound_calibrated}.coverage_at_95`.
Proof tab: `accepted`, every `*.curve` `[[coverage, accuracy], ...]`, `coverage_at_95`, `acc_at_60`, `aurc`, `per_label_accepted_acc`.
`upper_bound_calibrated` is always labelled UPPER BOUND.

**student.json** (DESIGN.md 4): `mean_acc_by_source{single, vote, verified, oracle_upper_bound}`, `per_label{key: {source:
{n_train, acc, f1, auroc}}}`. Compared with zero-shot = `results.json.manual_raw_acc`. `oracle_upper_bound` always labelled UPPER BOUND.

**narration_<video>.json**: a list of entries. Preferred shape (requested from narrate lane):
```json
{"frame_id": "video12_15750_15758", "claims": [
  {"text": "A grasper retracts the gallbladder.", "status": "verified", "evidence_frames": ["video12_15750_15758"]},
  {"text": "The cystic duct is clipped.", "status": "contradicted", "evidence_frames": ["video12_15750_15766"]}]}
```
`status` ∈ `verified | unsupported | contradicted` (anything else → `unsupported`). The final narration shown = verified claims only.
Legacy shape `{frame_id, narration, unsure_about[]}` (current narrate.py) still works: a line is `verified` only if
`unsure_about` is empty, and each unsure item becomes an `unsupported` claim.

**decisions.csv, labels_*.jsonl, data/index.csv, human_gate_log.jsonl**: exactly as `docs/DESIGN.md` 1A–1C and 2.
`status` normalised: `accept`; `reject`/`escalate` → reject; anything else → AMBER.

**Loop rounds 0–1 viewer**: rows from `decisions.csv` if present, else `labels_manual.jsonl` (`sample == 0`), else
`labels_smoke.jsonl`. Uses `frame_id, video, key, present, conf, box, evidence, pv_match, p_wrong, critic`, plus `status,
final_present, final_by, frame_path` when decisions exist. Frame image from `frame_path`, else derived from `frame_id`.
No accuracy is computed here (GT scoring comes from `results.json`).

## Update 2026-10-31 14:05 (main merged; app.py = nir-demo's superset of the UI lane's tabs)
- **results.json** now real (smoke: `n_test_labels = 90`). The loop tab also reads `n_test_labels`,
  `accepted_5_informative.{accuracy, coverage, n_test_labels_excl_liver}` and
  `amber_loop.{max_rounds, ambers_by_round_test[], per_round_test[]{accepted_count, rejected_count}}`.
  Below 1000 scored test labels, the UI shows an "interim smoke run" warning.
- **decisions.csv, student.json, amber_rounds.jsonl, verified_dataset.jsonl** are real in `coord/artifacts/` (DESIGN.md 1B/1D/4).
- **Narration**: reads `narrator/out/claims.parquet` first (contract `coordination/contracts/narrator_outputs.md`):
  `claim_id, clip_id, video_id, mode, t_start, t_end, text, verdict, L1..L7, P, source`, plus `narrator/out/evidence/index.json`.
  Mapping: `speak` → verified; `drop` with any layer `fail` → contradicted; everything else → unsupported. Final narration = verified only.
  Rows with `source = mock` are hidden. `gt_correct` is never shown. Falls back to `out/narration_<video>.json`.
