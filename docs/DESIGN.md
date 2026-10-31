# SurgAtlas File Contracts (`docs/DESIGN.md`)

Canonical schemas between pipeline steps and across machines (`nir-*` and `harald-*`). Do not rename columns or keys without posting `-> <lane>` on Issue #1 and requesting an architect update.

## Shared Label Keys (`config.LABELS`)
`grasper` (5), `l_hook` (9), `liver` (2), `gallbladder` (10), `cystic_duct` (8), `blood` (7).

---

## 1. TOP PRIORITY: Labeller -> Verifier Contract (`label.py` -> `accept.py` / `verify/`)

### 1A. `out/labels_*.jsonl` & `coord/artifacts/labels_smoke.jsonl` (`nir-label` -> `harald-verifier`, `nir-student`)
Files:
- `out/labels_manual.jsonl`: main run (`mode="manual"`, `sample=0`, all splits `train,calib,test`)
- `coord/artifacts/labels_smoke.jsonl`: 40-frame smoke-test subset committed for `harald-verifier`
- `out/labels_blank.jsonl`: baseline without manual (`mode="blank"`, `split="test"`)
- `out/labels_manual_v2.jsonl`: extra stochastic samples (`votes=2`, `sample in {1, 2}`)

**Row format**: One flat JSON object per `(frame_id, key, sample)` line (6 lines per frame per sample, one for each key in `config.LABELS`):
```json
{
  "frame_id": "video01_00080_80",
  "video": "video01",
  "video_id": "video01",
  "split": "train",
  "key": "grasper",
  "mode": "manual",
  "sample": 0,
  "model": "gemini-3.8-flash",
  "prompt_hash": "a1b2c3d4e5f60123",
  "present": 1,
  "conf": 0.92,
  "confidence": 0.92,
  "box": [120, 300, 450, 620],
  "evidence": "metallic jaws grasping gallbladder infundibulum",
  "rationale": "metallic jaws grasping gallbladder infundibulum",
  "pv_match": true,
  "pv_conf": 0.88,
  "p_wrong": 0.10,
  "critic": "could be clip applier shaft, but fenestrated jaws confirm grasper",
  "truth": 1
}
```
- **Required fields (`nir-label` emits, `harald-verifier` reads)**:
  - `frame_id` (`str`): unique frame identifier `<clip>_<frame_num>` (e.g. `video01_00080_80`), joins with `data/index.csv`.
  - `video` & `video_id` (`str`): video folder name (e.g. `"video01"`). Emit both (or verifier accepts `video_id or video`) so downstream scripts (`accept.py`, `student.py`, `narrate.py`, `app.py`) never break.
  - `split` (`str`): `"train"` | `"calib"` | `"test"`.
  - `key` (`str`): one of the 6 `config.LABELS` keys.
  - `present` (`int`): `1` if predicted present, `0` if absent.
  - `conf` & `confidence` (`float` in `[0.0, 1.0]`): labeller confidence that `present` is correct. Emit `conf` (and alias `confidence`) for backward compatibility.
  - `evidence` & `rationale` (`str`): 1-phrase visual rationale from the labeller. Emit `evidence` (and alias `rationale`).
  - `box` (`list[int]`): `[ymin, xmin, ymax, xmax]` normalized `0..1000` when `present == 1`, or `[]` when `present == 0`.
  - `mode` (`str`): `"manual"` | `"blank"`.
  - `sample` (`int`): `0` for deterministic primary run, `1..N` for vote samples.
  - `model` (`str`): Gemini model ID used (e.g. `"gemini-3.8-flash"`).
  - `prompt_hash` (`str`): 16-char SHA-256 hex prefix of the labelling prompt (tracks manual/prompt version).
  - **Decorrelated check fields** (present when `full_checks=True`, else `null` / `0.5` / `""`):
    - `pv_match` (`bool | null`): blind crop verification agreement (`true`/`false` when `present == 1` and `len(box) == 4`, else `null`).
    - `pv_conf` (`float | null`): blind crop confidence in `[0.0, 1.0]`.
    - `p_wrong` (`float` in `[0.0, 1.0]`): sceptical critic probability that the claim is wrong.
    - `critic` (`str`): critic counter-argument.
  - `truth` (`int`: `0 | 1`): ground-truth label from `index.csv` (`gt_<key>`). **STRICT RULE**: Verifier logic (`trust`, `status`, `final_present`) must NEVER read `truth` on `train` or `test`. `truth` is only used to compute evaluation metrics on `test` (and optional upper-bound logistic regression calibration on `calib`).

---

### 1B. `out/decisions.csv` & `coord/artifacts/decisions.csv` (`harald-verifier` -> `nir-student`, `harald-demo`, `harald-narrate`)
One row per `(frame_id, key)` (for `mode == "manual", sample == 0`):
- **Inherited columns**:
  - `frame_id`, `video`, `split`, `key`, `mode`, `sample`, `present`, `conf`, `box`, `evidence`, `truth`, `pv_match`, `pv_conf`, `p_wrong`, `critic` (plus `video_id`, `model`, `prompt_hash` if present)
- **Added by `accept.py` (`harald-verifier`)**:
  - `frame_path` (`str`): path to frame image from `data/index.csv.path` (optional if running without raw frames)
  - `correct` (`int`: `0 | 1`): `int(present == truth)` (for evaluation only)
  - `trust` (`float` in `[0.0, 1.0]`): continuous verification score used for risk-coverage ranking
  - `status` (`str`): `"accept"` | `"reanalyse"` | `"escalate"`
  - `final_present` (`int`: `0 | 1`): final predicted presence after verifier / Argon revision
  - `final_by` (`str`): `"flash"` | `"argon"`
  - `final_correct` (`int`: `0 | 1`): `int(final_present == truth)` (for evaluation only)

---

### 1C. `out/results.json` & `coord/artifacts/results.json` (`harald-verifier` -> `harald-demo`, `nir-student` eval)
Evaluated strictly on `split == "test"` (`upper_bound_calibrated` fits on `split == "calib"` only):
```json
{
  "n_test_labels": 360,
  "single_conf": {"curve": [[0.02, 1.0], "..."], "coverage_at_95": 0.42, "acc_at_60": 0.89, "aurc": 0.081},
  "system_zero_label": {"curve": [[0.02, 1.0], "..."], "coverage_at_95": 0.65, "acc_at_60": 0.96, "aurc": 0.042},
  "accepted": {
    "coverage": 0.62,
    "accuracy": 0.954,
    "escalated_share": 0.18
  },
  "blank_single_conf": {"curve": ["..."], "coverage_at_95": 0.25, "acc_at_60": 0.81, "aurc": 0.12},
  "blank_raw_acc": 0.78,
  "manual_raw_acc": 0.85,
  "majority_vote": {"curve": ["..."], "coverage_at_95": 0.50, "acc_at_60": 0.91, "aurc": 0.065},
  "upper_bound_calibrated": {"curve": ["..."], "coverage_at_95": 0.70, "acc_at_60": 0.97, "aurc": 0.035},
  "per_label_accepted_acc": {"grasper": 0.96, "l_hook": 0.95, "liver": 0.98, "gallbladder": 0.94, "cystic_duct": 0.91, "blood": 0.93}
}
```

---

## 2. `data/index.csv` & `data/features.npy` (`prepare_data.py` -> `features.py`, `label.py`, `accept.py`, `student.py`, `app.py`)
One row per sampled frame (`FRAME_STRIDE = 8`, split by `video` into `train`, `calib`, `test`):
- `frame_id` (`str`, `<clip>_<frame_num>`, e.g. `video01_00080_80`)
- `video` (`str`, e.g. `video01`)
- `clip` (`str`, e.g. `video01_00080`)
- `path` (`str`, relative or local path to `*_endo.png`)
- `mask_path` (`str`, path to `*_endo_watershed_mask.png`)
- `split` (`str`: `"train"` | `"calib"` | `"test"`, partitioned at the `video` level with `SEED = 0`)
- `gt_<key>` (`int`: `0 | 1`) and `frac_<key>` (`float` in `[0.0, 1.0]`) for each `<key>` in `config.LABELS`
- **`data/features.npy`**: `float32` array of shape `(len(index.csv), 768)` row-aligned 1:1 with `data/index.csv`.

---

## 3. `out/manual.json` & `coord/artifacts/manual.json` (`med_school.py` -> `label.py`, `accept.py`, `narrate.py`, `app.py`)
```json
{
  "procedure": "laparoscopic cholecystectomy",
  "phases": ["str", "..."],
  "structures": [
    {
      "key": "grasper",
      "name": "str",
      "appearance": "str",
      "where_found": "str",
      "when_visible": "str",
      "confused_with": ["str"],
      "how_to_tell_apart": "str"
    }
  ],
  "critical_events": ["str"],
  "general_rules": ["str"]
}
```

---

## 4. `out/student.json` & `out/student_probs.csv` (`student.py` -> `app.py`, `eval/`)
DINOv2 linear probes trained on `split == "train"` using each label source (`single`, `vote`, `verified`, `oracle_upper_bound`), evaluated on `split == "test"` against `gt_<key>`:
```json
{
  "mean_acc_by_source": {
    "single": 0.82,
    "vote": 0.84,
    "verified": 0.88,
    "oracle_upper_bound": 0.91
  },
  "per_label": {
    "grasper": {
      "single": {"n_train": 600, "acc": 0.85, "f1": 0.83, "auroc": 0.91},
      "verified": {"n_train": 390, "acc": 0.89, "f1": 0.88, "auroc": 0.94},
      "oracle_upper_bound": {"n_train": 600, "acc": 0.92, "f1": 0.91, "auroc": 0.96}
    }
  }
}
```
- `out/student_probs.csv`: `frame_id` (test split) + `<key>__<source>` probability columns.

---

## 5. Auxiliary Contracts (`atlas.py`, `narrate.py`, `gem.py` -> `app.py`)
- **`out/atlas/<slug>/dataset.json`**: `{"procedure": str, "source_video": str, "created": str, "schema": {"structures": [str], "phases": [str]}, "timeline": {"phases": [...], "events": [...], "sightings": [...]}, "audit": {"problems": [str], "unverified_items": [str], "overall_confidence": float}, "human_labels_used": 0}`
- **`out/narration_<video>.json`**: `[{"frame_id": str, "narration": str, "unsure_about": [str]}, ...]`
- **`out/trace.jsonl`**: `{"t": float, "agent": str, "msg": str}` per line.
