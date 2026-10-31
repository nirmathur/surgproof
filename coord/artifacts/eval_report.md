# Independent Evaluation & Leakage Audit Report

## 1. Leakage & Integrity Checks
- **Split Disjointness (Video & Frame level)**: `PASS` (videos: `{'train': 8, 'calib': 3, 'test': 6}`, frames: `{'train': 630, 'calib': 100, 'test': 280}`)
- **GT Separation in `labels_*.jsonl` (12:42 Audit)**: `PASS` (`{'labels_smoke.jsonl': {'n_rows': 240, 'forbidden_gt_cols': []}, 'amber_rounds.jsonl': {'n_rows': 180, 'forbidden_gt_cols': []}, 'labels_blank.jsonl': {'n_rows': 1680, 'forbidden_gt_cols': []}, 'labels_cholect50_blank.jsonl': {'n_rows': 720, 'forbidden_gt_cols': []}, 'labels_cholect50_manual.jsonl': {'n_rows': 720, 'forbidden_gt_cols': []}, 'labels_endoscapes2023_blank.jsonl': {'n_rows': 444, 'forbidden_gt_cols': []}, 'labels_endoscapes2023_manual.jsonl': {'n_rows': 444, 'forbidden_gt_cols': []}, 'labels_manual.jsonl': {'n_rows': 6060, 'forbidden_gt_cols': []}}`)
- **Degenerate & Split-Sparse Label Audit**: `WARN` (100%-present excluded from headline: `['liver']`; sparse on test: `['cystic_duct', 'blood']`; core balanced test labels: `['grasper', 'l_hook', 'gallbladder']`)
- **Zero-Human-Label Verifier Leakage Check**: `PASS` (issues: `[]`)
- **Manual Provenance Check**: `PASS` (Provenance verified in manual.json (7 URLs, 57 timestamps).)

## 2. Raw Labeller Baselines on Held-Out TEST Split (Video-Level Cluster Bootstrap 95% CIs)
- **Blank Baseline (`labels_blank.jsonl`)** (1680 rows, 6 test videos): 5-label headline acc = `0.9250` (95% video CI `[0.8795, 0.9773]`), Core-3 balanced acc = `0.9333` (95% video CI `[0.8989, 0.9758]`), coverage@95% = `0.926`, AURC = `0.0279`
  - Per-label: grasper=0.921 [0.8615, 0.9909] (pos=201/280), l_hook=0.946 [0.8892, 0.996] (pos=92/280), liver=0.993 [0.9806, 1.0] (pos=280/280), gallbladder=0.932 [0.873, 0.9679] (pos=194/280), cystic_duct=0.946 [0.8971, 0.9963] (pos=1/280), blood=0.879 [0.7824, 0.9818] (pos=0/280)
- **Manual Single-Pass (`labels_manual.jsonl`)** (1680 rows, 6 test videos): 5-label headline acc = `0.9229` (95% video CI `[0.8933, 0.9556]`), Core-3 balanced acc = `0.8845` (95% video CI `[0.8378, 0.9315]`), coverage@95% = `0.864`, AURC = `0.0380`
  - Per-label: grasper=0.900 [0.8, 1.0] (pos=201/280), l_hook=0.929 [0.8842, 0.9773] (pos=92/280), liver=0.996 [0.9903, 1.0] (pos=280/280), gallbladder=0.825 [0.6864, 0.9441] (pos=194/280), cystic_duct=0.979 [0.9545, 1.0] (pos=1/280), blood=0.982 [0.975, 0.9913] (pos=0/280)

## 3. Verifier Test Metrics (5 Informative Labels, Video-Level Cluster Bootstrap 95% CIs)
- **single_raw_acc**: `0.9067` (95% video CI: `[0.875, 0.9429]`, n_videos=6)
- **accepted_acc**: `0.9853` (95% video CI: `[0.9577, 1.0]`, n_videos=6)
- **accepted_coverage**: `0.9067` (95% video CI: `[0.875, 0.9429]`, n_videos=6)
- **accepted_minus_raw_acc_delta**: `0.0786` (95% video CI: `[0.0451, 0.1107]`, n_videos=6)
- **Amber Loop Per-Round Resolution on TEST (5 Informative Labels)**:
  - Round `0` (`initial_double_check`): ambers_remaining = `8`, resolved_in_round = `0` (resolved_precision = `None`), cumulative_accepted = `62` (precision = `0.9839`)
  - Round `1` (`amber_loop_round_1`): ambers_remaining = `6`, resolved_in_round = `2` (resolved_precision = `1.0`), cumulative_accepted = `64` (precision = `0.9844`)
  - Round `2` (`amber_loop_round_2`): ambers_remaining = `3`, resolved_in_round = `3` (resolved_precision = `1.0`), cumulative_accepted = `66` (precision = `0.9848`)
  - Round `3` (`amber_loop_round_3`): ambers_remaining = `1`, resolved_in_round = `2` (resolved_precision = `1.0`), cumulative_accepted = `68` (precision = `0.9853`)

## 4. Student Specialist Test Metrics (5 Informative Labels, Video-Level Cluster Bootstrap 95% CIs)
- **single** *(zero human labels)*: `0.8893` (95% video CI: `[0.8439, 0.9537]`, n_videos=6)
- **verified** *(zero human labels)*: `0.9029` (95% video CI: `[0.8626, 0.9543]`, n_videos=6)
- **oracle_upper_bound** **[UPPER BOUND ONLY - human GT]**: `0.8814` (95% video CI: `[0.836, 0.9383]`, n_videos=6)
- **Paired Delta (`verified - single`)**: `+0.0136` (95% video CI: `[-0.0008, 0.0244]`, P(delta > 0) = `0.9655`)

## 5. External Held-Out Datasets (Endoscapes2023 & CholecT50, Video-Level Cluster Bootstrap 95% CIs)
- **cholect50** (120 frames, 40 videos; scorable: `['grasper', 'l_hook']`, pos_counts: `{'grasper': 83, 'l_hook': 76}`)
  - `labels_cholect50_blank.jsonl` (240 test rows, 40 videos): raw acc = `0.9125` (95% video CI `[0.8792, 0.9458]`), pos claim precision = `0.9423`, double-checked pos precision = `None`
    - Per-label: grasper: acc=0.900 [0.85, 0.95], pos_prec=0.9277, ver_acc=None (pos_gt=83/120), l_hook: acc=0.925 [0.8583, 0.9752], pos_prec=0.9589, ver_acc=None (pos_gt=76/120)
  - `labels_cholect50_manual.jsonl` (240 test rows, 40 videos): raw acc = `0.8958` (95% video CI `[0.8583, 0.9292]`), pos claim precision = `0.9408`, double-checked pos precision = `0.9677`, zero-label verified (trust>=0.70) acc = `0.9126` (cov=`0.8583`)
    - Per-label: grasper: acc=0.900 [0.8417, 0.95], pos_prec=0.9494, ver_acc=0.9029 (pos_gt=83/120), l_hook: acc=0.892 [0.825, 0.95], pos_prec=0.9315, ver_acc=0.9223 (pos_gt=76/120)
  - **3-Round Argon Verifier (`decisions_cholect50.csv`)**: accepted_acc = `0.9123` (95% video CI `[0.8767, 0.9437]`), coverage = `0.9500` (`[0.9208, 0.9792]`), delta vs raw = `+0.0164` (`[0.0017, 0.0337]`), Amber loop shrinkage = `[R0:29 -> R1:28 -> R2:13 -> R3:3]`
- **endoscapes2023** (493 frames, 50 videos; scorable: `['gallbladder', 'cystic_duct']`, pos_counts: `{'gallbladder': 466, 'cystic_duct': 374}`)
  - `labels_endoscapes2023_blank.jsonl` (148 test rows, 10 videos): raw acc = `0.7365` (95% video CI `[0.6463, 0.7826]`), pos claim precision = `0.9865`, double-checked pos precision = `None`
    - Per-label: gallbladder: acc=0.973 [0.95, 1.0], pos_prec=0.9828, ver_acc=None (pos_gt=58/74), cystic_duct: acc=0.500 [0.2927, 0.6042], pos_prec=1.0, ver_acc=None (pos_gt=53/74)
  - `labels_endoscapes2023_manual.jsonl` (148 test rows, 10 videos): raw acc = `0.6757` (95% video CI `[0.561, 0.7435]`), pos claim precision = `1.0`, double-checked pos precision = `1.0`, zero-label verified (trust>=0.70) acc = `0.7257` (cov=`0.7635`)
    - Per-label: gallbladder: acc=0.973 [0.9342, 1.0], pos_prec=1.0, ver_acc=0.9833 (pos_gt=58/74), cystic_duct: acc=0.378 [0.1562, 0.4951], pos_prec=1.0, ver_acc=0.434 (pos_gt=53/74)
  - **3-Round Argon Verifier (`decisions_endoscapes2023.csv`)**: accepted_acc = `0.7050` (95% video CI `[0.6061, 0.7581]`), coverage = `0.9392` (`[0.8654, 0.975]`), delta vs raw = `+0.0294` (`[0.0062, 0.0781]`), Amber loop shrinkage = `[R0:28 -> R1:28 -> R2:6 -> R3:2]`

## 6. What a Judge Could Attack & Defenses
### [HIGH] attack_1_pseudo_replication (`MITIGATED`)
- **Judge Attack**: Your 280 test frames come from 6 surgical videos; frames every 8 steps are temporally correlated. Are your confidence intervals inflated by treating frames as i.i.d.?
- **Defense**: We report video-level cluster bootstrap 95% CIs (2,000 resamples at the whole-video level) and paired video-level bootstrap deltas for both verifier precision and student accuracy.

### [HIGH] attack_2_liver_100pct_and_sparse_test_positives (`MITIGATED`)
- **Judge Attack**: In CholecSeg8k, 'liver' is 100% present, and under SEED=0 'blood' (0/280) and 'cystic_duct' (1/280) have almost no positive frames in the 6 test videos. Doesn't that distort macro accuracy?
- **Defense**: 1) We exclude 100%-present `liver` from all headline metrics (`mean_acc_by_source` uses the 5 remaining labels). 2) We report both the 5-label headline AND the Core-3 balanced test subset (`grasper`, `l_hook`, `gallbladder`, which have 201/280, 92/280, and 194/280 positives on TEST), while explicitly framing `cystic_duct` and `blood` on TEST as false-positive suppression (specificity) benchmarks.

### [HIGH] attack_3_zero_label_leakage (`PASS`)
- **Judge Attack**: Did your verification threshold or student training leak ground-truth labels or test videos?
- **Defense**: 1) Splits are strictly disjoint at the video level (0 overlapping videos between train/calib/test). 2) `labels_*.jsonl` files carry zero ground-truth fields (`truth` dropped at generation time); GT is joined only inside the post-hoc evaluation step. 3) `upper_bound_calibrated` and `oracle_upper_bound` are isolated and explicitly marked as UPPER BOUNDS ONLY.

### [MEDIUM] attack_4_coverage_accuracy_tradeoff (`MITIGATED`)
- **Judge Attack**: Anyone can get high accuracy by abstaining on 90% of frames. What happens to downstream utility when you reject uncertain labels?
- **Defense**: We prove utility two ways: (a) Full risk-coverage curves (AURC, coverage@95%, acc@60%) against single-pass confidence and majority vote; (b) Downstream specialist distillation (`student.py`), where the same DINOv2 linear probe trained ONLY on verified labels is evaluated at 100% coverage on all held-out test frames.

### [MEDIUM] attack_5_manual_provenance (`PASS`)
- **Judge Attack**: Did the agent actually learn from public surgical sources/video, or did Gemini just rely on pretraining weights?
- **Defense**: We compare `--mode manual` against `--mode blank` (identical model & prompt without the self-written manual) on the held-out test split and audit `manual.json` / `research_notes.md` for source URLs and timestamped video lessons.
