# SurgProof File Contracts (`docs/DESIGN.md`)

**Project**: **SurgProof** — *"No human labels. Only proof."*
*(Prior art credited on stage and in UI: **SurgAtlas**, **SurgVLP**, **SurgSLOT**, **Med-Gemini**.)*

Canonical schemas between pipeline steps and across machines (`nir-*` and `harald-*`). Do not rename columns or keys without posting `-> <lane>` on Issue #1 and requesting an architect update.

## Shared Label Keys (`config.LABELS`)
`grasper` (5), `l_hook` (9), `liver` (2), `gallbladder` (10), `cystic_duct` (8), `blood` (7).
*(Note: `gt_liver` has `1.000` prevalence across all 1,010 frames; headline pooled metrics average over the 5 informative labels `grasper, l_hook, gallbladder, cystic_duct, blood`, while `liver` is reported per-label.)*

---

## 1. TOP PRIORITY: Labeller -> Verifier -> Human Gate Contract (`label.py` -> `accept.py` -> `app.py` -> `eval/`)

> **STRICT ZERO-LEAKAGE RULE (Enforced by `tests/test_architect_contracts.py` & `eval/audit.py`)**:
> 1. `labels_*.jsonl` MUST NOT contain `truth` or `gt_*`.
> 2. `accept.py` / `verify/` MUST NEVER read `truth` or `gt_*`. `decisions.csv` contains predictions, verification signals, and `status` only.
> 3. Ground truth (`gt_<key>` from `data/index.csv`) is joined ONLY in the post-decision evaluation step (`eval/score_decisions.py` and `eval/audit.py`) AFTER `decisions.csv` is written to disk.

### 1A. `out/labels_*.jsonl` & `coord/artifacts/labels_smoke.jsonl` (`nir-label` -> `harald-verifier`, `nir-student`)
Files:
- `out/labels_manual.jsonl`: main run (`mode="manual"`, `sample=0`, splits `train,calib,test`)
- `coord/artifacts/labels_smoke.jsonl`: 40-frame smoke-test subset committed for `harald-verifier`
- `out/labels_blank.jsonl`: baseline without manual (`mode="blank"`, `split="test"`)
- `out/labels_manual_v2.jsonl`: extra stochastic samples (`votes=2`, `sample in {1, 2}`)

**Row format**: One flat JSON object per `(frame_id, key, sample)` line (NO `truth` or `gt_*` field!):
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
  "critic": "could be clip applier shaft, but fenestrated jaws confirm grasper"
}
```
- **Required fields (`nir-label` emits, `harald-verifier` reads)**:
  - `frame_id` (`str`): unique frame identifier `<clip>_<frame_num>` (e.g. `video01_00080_80`).
  - `video` & `video_id` (`str`): video folder name (e.g. `"video01"`).
  - `split` (`str`): `"train"` | `"calib"` | `"test"`.
  - `key` (`str`): one of the 6 `config.LABELS` keys.
  - `present` (`int`): `1` if predicted present, `0` if absent.
  - `conf` & `confidence` (`float` in `[0.0, 1.0]`): labeller confidence that `present` is correct.
  - `evidence` & `rationale` (`str`): 1-phrase visual rationale from the labeller.
  - `box` (`list[int]`): `[ymin, xmin, ymax, xmax]` normalized `0..1000` when `present == 1`, or `[]` when `present == 0`.
  - `mode` (`str`): `"manual"` | `"blank"`.
  - `sample` (`int`): `0` for deterministic primary run, `1..N` for vote samples.
  - `model` (`str`): Gemini model ID used (e.g. `"gemini-3.8-flash"`).
  - `prompt_hash` (`str`): 16-char SHA-256 hex prefix of the labelling prompt.
  - **Decorrelated check fields** (present when `full_checks=True`, else `null` / `0.5` / `""`):
    - `pv_match` (`bool | null`): blind crop verification agreement (`true`/`false` when `present == 1` and `len(box) == 4`, else `null`). *(Note: in pandas, check `bool(pd.notna(r.pv_match) and r.pv_match)`, never `r.pv_match is True`!)*
    - `pv_conf` (`float | null`): blind crop confidence in `[0.0, 1.0]`.
    - `p_wrong` (`float` in `[0.0, 1.0]`): sceptical critic probability that the claim is wrong.
    - `critic` (`str`): critic counter-argument.

---

### 1B. `out/decisions.csv` & `coord/artifacts/decisions.csv` (`harald-verifier` -> `nir-student`, `nir-demo`)
One row per `(frame_id, key)` (NO `truth`, `correct`, `final_correct`, or `gt_*` columns!):
- **Inherited columns from `labels_manual.jsonl`**:
  - `frame_id, video, video_id, split, key, mode, sample, model, prompt_hash, present, conf, confidence, box, evidence, rationale, pv_match, pv_conf, p_wrong, critic`
- **Added by `accept.py` (`harald-verifier`)**:
  - `frame_path` (`str`): path to frame image from `data/index.csv.path`
  - `trust` (`float` in `[0.0, 1.0]`): continuous zero-label verification score used for risk-coverage ranking
  - `status` (`str`):
    - `"accept"` — high-trust verified label (Green; used for specialist student distillation and grounded narration)
    - `"AMBER"` — uncertain / abstained label queued for the **Human Gate** (#14) (Amber; `app.py` also accepts legacy `"reanalyse"` as `"AMBER"`)
    - `"reject"` — low-trust / rejected claim (Red; `app.py` and `eval/` also accept legacy `"escalate"` as `"reject"`)
  - `final_present` (`int`: `0 | 1`): final predicted presence after verifier / Argon Amber loop
  - `final_by` (`str`): `"flash"` | `"argon"` | `"human"`
  - *(Optional second-pass & loop tracking columns)*: `pass2_present` (`int`), `pass2_conf` (`float`), `pass2_agree` (`bool`), `initial_status` (`"accept" | "AMBER" | "reject"`), `resolved_round` (`int`: `0..3`).

### 1C. Argon Amber Loop Log (`out/amber_rounds.jsonl`, `VISION.md`)
Every label is double-checked by an independent second pass; disagreements or low-confidence items start as `AMBER`. Before any human sees them, Argon re-examines each `AMBER` for up to `N=3` rounds (Round 1: zoomed crop + manual criteria; Round 2: tighter crop + differential vs `confused_with`; Round 3: fresh vote consensus) to shrink `AMBER` -> `0`:
```json
{
  "label_id": "video01_00080_80__cystic_duct",
  "frame_id": "video01_00080_80",
  "video": "video01",
  "split": "test",
  "key": "cystic_duct",
  "round": 1,
  "verdict": "accept",
  "present": 1,
  "conf": 0.91,
  "reason": "Tighter hepatocystic crop confirms tubular continuity into gallbladder neck."
}
```
- `verdict`: `"accept"` | `"reject"` | `"AMBER"`.

### 1D. Human Final Pass & Dataset Release (`pages/2_Human_gate.py` -> `out/verified_dataset.jsonl`, `out/human_gate_log.jsonl`, `out/gate_log.jsonl`)
- **Final Pass rule (`VISION.md`)**: Before anything becomes proper data, the human reviews:
  1. Any residual `AMBER` items remaining after the N-round Argon Amber loop, AND
  2. A random audit sample (e.g. 20 items) of auto-accepted labels with their evidence crops.
- **`Release dataset` / `Export verified dataset` lock**: Disabled until all residual `AMBER` items and the audit sample are signed off.
- **`out/human_gate_log.jsonl` & `out/gate_log.jsonl`**:
```json
{"t": 1730379000.0, "reviewer": "reviewer_id", "frame_id": "video01_00080_80", "key": "cystic_duct", "item_type": "amber | audit_sample", "action": "approve | flip | reject", "human_present": 1}
```
- **`out/verified_dataset.jsonl`**: Released dataset rows (`frame_id, video, split, key, final_present, box, trust, final_by, manual_sources, audit_signed_off_by, released_at`).

---

### 1E. Post-Decision Evaluation: `out/results.json`, `out/eval_audit.json`, `out/eval_report.md` (`eval/score_decisions.py` & `eval/audit.py`)
Runs AFTER `out/decisions.csv` exists, joining `data/index.csv` (`gt_<key>`) on `split == "test"` (`upper_bound_calibrated` fits on `split == "calib"` only):
```json
{
  "n_test_labels": 1680,
  "single_conf": {"curve": [[0.02, 1.0], "..."], "coverage_at_95": 0.42, "acc_at_60": 0.89, "aurc": 0.081},
  "system_zero_label": {"curve": [[0.02, 1.0], "..."], "coverage_at_95": 0.65, "acc_at_60": 0.96, "aurc": 0.042},
  "accepted": {
    "coverage": 0.62,
    "accuracy": 0.954,
    "escalated_share": 0.38,
    "amber_share": 0.20,
    "rejected_share": 0.18
  },
  "blank_single_conf": {"curve": ["..."], "coverage_at_95": 0.25, "acc_at_60": 0.81, "aurc": 0.12},
  "blank_raw_acc": 0.78,
  "manual_raw_acc": 0.85,
  "majority_vote": {"curve": ["..."], "coverage_at_95": 0.50, "acc_at_60": 0.91, "aurc": 0.065},
  "upper_bound_calibrated": {"curve": ["..."], "coverage_at_95": 0.70, "acc_at_60": 0.97, "aurc": 0.035},
  "per_label_accepted_acc": {"grasper": 0.96, "l_hook": 0.95, "liver": 1.0, "gallbladder": 0.94, "cystic_duct": 0.91, "blood": 0.93}
}
```

---

## 2. `data/index.csv` & `data/features.npy` (`prepare_data.py` -> `features.py`, `label.py`, `accept.py`, `student.py`, `eval/`, `app.py`)
One row per sampled frame (`FRAME_STRIDE = 8`, `1010` frames across `17` videos, split strictly by `video` into `train=630`, `calib=100`, `test=280`):
- `frame_id` (`str`, `<clip>_<frame_num>`, e.g. `video01_00080_80`)
- `video` (`str`, e.g. `video01`)
- `clip` (`str`, e.g. `video01_00080`)
- `path` (`str`, path to `*_endo.png`)
- `mask_path` (`str`, path to `*_endo_watershed_mask.png`)
- `split` (`str`: `"train"` | `"calib"` | `"test"`, partitioned at the `video` level with `SEED = 0`)
- `gt_<key>` (`int`: `0 | 1`) and `frac_<key>` (`float` in `[0.0, 1.0]`) for each `<key>` in `config.LABELS` (ONLY read by `eval/` and `student.py` test evaluation / `oracle_upper_bound`)
- **`data/features.npy`**: `float32` array of shape `(len(index.csv), 768)` row-aligned 1:1 with `data/index.csv`.

---

## 3. `out/manual.json` & `coord/artifacts/manual.json` (`med_school.py` -> `label.py`, `accept.py`, `narrate.py`, `app.py`)
Must include explicit source provenance (`sources` with URLs, timestamps, and whether footage is real intraoperative endoscopic video):
```json
{
  "procedure": "laparoscopic cholecystectomy",
  "sources": [
    {
      "url": "https://www.youtube.com/watch?v=...",
      "title": "str",
      "is_real_intraop_footage": true,
      "timestamps": ["01:15 - trocar placement and liver exposure", "03:40 - hepatocystic triangle dissection"]
    }
  ],
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
DINOv2 linear probes trained on `split == "train"` using each label source (`single`, `vote`, `verified`, `oracle_upper_bound`), evaluated on `split == "test"` against `gt_<key>` (headline `mean_acc_by_source` averages over the 5 informative labels `grasper, l_hook, gallbladder, cystic_duct, blood`):
```json
{
  "headline_labels": ["grasper", "l_hook", "gallbladder", "cystic_duct", "blood"],
  "excluded_uninformative_labels": ["liver"],
  "mean_acc_by_source": {
    "single": 0.82,
    "vote": 0.84,
    "verified": 0.88,
    "oracle_upper_bound": 0.91
  },
  "per_label": {
    "grasper": {
      "single": {"n_train": 630, "acc": 0.85, "f1": 0.83, "auroc": 0.91, "is_upper_bound": false},
      "verified": {"n_train": 390, "acc": 0.89, "f1": 0.88, "auroc": 0.94, "is_upper_bound": false},
      "oracle_upper_bound": {"n_train": 630, "acc": 0.92, "f1": 0.91, "auroc": 0.96, "is_upper_bound": true}
    }
  }
}
```
- `out/student_probs.csv`: `frame_id` (test split) + `<key>__<source>` probability columns.

---

## 5. Live Full Annotation Stream (`localize.py` / `verify_localize.py` -> `app.py`)
Beyond presence labels, `nir-localize` draws bounding boxes (`[ymin, xmin, ymax, xmax]` in `0..1000`) for tools and anatomy, instrument tip points (`[y, x]` in `0..1000`), and the surgical phase of each frame, double-checked with an Argon Amber loop and streamed live:
- **`out/annotations_live.jsonl` & `out/annotations.jsonl`**:
```json
{
  "frame_id": "video12_15750_15758",
  "video": "video12",
  "split": "test",
  "img_path": "raw/.../frame_15758_endo.png",
  "label": "grasper",
  "box": [180, 410, 620, 790],
  "tip": [490, 560],
  "phase": "2. Calot's (Hepatocystic) Triangle Dissection",
  "conf": 0.94,
  "pass1": {"box": [180, 410, 620, 790], "tip": [490, 560], "conf": 0.92},
  "pass2": {"box": [185, 415, 615, 785], "tip": [488, 558], "conf": 0.95, "iou": 0.89},
  "round": 0,
  "status": "accept",
  "ts": 1730380500.0
}
```
- **`out/localize_rounds.jsonl`**: per-round disagreement resolution records (`frame_id, label, round, iou, verdict, reason`).
- **`out/localize_results.json`**: post-loop held-out `test` evaluation against watershed-mask-derived bounding boxes and instrument masks (never read during localization/verification).

