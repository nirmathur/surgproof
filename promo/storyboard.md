# {{NAME}} Pitch Video Storyboard (60–90s, Target: 75s)

- **Lane**: `nir-promo` (Issue #10)
- **Project Name Placeholder**: `{{NAME}}` (configurable via `PROJECT_NAME` env var; zero occurrences of the old taken name)
- **Primary Video Model**: `veo-3.1-generate-preview` (final shots) / `veo-3.1-fast-generate-preview` (drafts) via `google-genai` `client.models.generate_videos` (`predictLongRunning`)
- **Fallback Video Model**: `gemini-omni-1.1-flash`
- **Output Path**: `out/promo/surgatlas_promo.mp4` (1280×720, 24 fps, H.264 + AAC audio, ~75s total)
- **Individual Shots**: `out/promo/shot_01_med_school.mp4` .. `out/promo/shot_06_narrate.mp4`

## Honest Claims & Compliance Rules (Strictly Enforced)
1. **Mandatory Illustrative Caption**: Every AI-generated clip (`shot_01`, `shot_02`, `shot_03`, `shot_04`, `shot_06`) carries a permanent burned-in upper badge:
   `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
2. **Zero Fake Numbers / Charts / Endorsements**: Prompts and overlays for generated shots forbid fake metrics, fake charts, or simulated surgeon endorsements.
3. **Real Proof Slot (`shot_05_proof_slot.mp4`)**: Reads strictly from real pipeline artifacts (`out/results.json`, `out/student.json`, `out/decisions.csv`, `out/manual.json`) and a live headless capture of `app.py`. Until those files exist, the slot explicitly displays `AWAITING REAL PIPELINE OUTPUTS` with zero fabricated numbers. Also respects the `liver` prevalence finding (reports non-trivial classes / excludes `liver=1.000` from pooled headline metrics).
4. **Honest Attribution on Screen**: Explicitly notes *"No human labels used in training; ground truth evaluates held-out test videos only; calib/oracle reported as upper bounds"* and credits prior work (**SurgVLP**, **SurgSLOT**, **Med-Gemini**).

---

## Shot-by-Shot Breakdown (6 Shots, 75 Seconds Total)

### Shot 1 — Autonomous Medical School (`0:00–0:12`, 12s)
- **File**: `out/promo/shot_01_med_school.mp4`
- **Pipeline Stage**: `python med_school.py --video <URL>` $\rightarrow$ `out/manual.json`
- **Narrative**: Surgical AI normally demands thousands of hours of manual annotation by surgeons. `{{NAME}}` starts with zero human labels: an autonomous Gemini 4 Argon agent studies public narrated surgical lectures and writes its own structured visual manual (`out/manual.json`), cataloguing instruments, anatomy, and common visual confusions.
- **Veo 3.1 Prompt**:
  > Cinematic wide shot inside a dark-mode surgical AI research lab. A sleek glass monitor displays an abstract medical lecture waveform transforming into glowing structured knowledge nodes and anatomical wireframe schematics. Warm rim lighting, shallow depth of field, clean modern interface aesthetic, no readable text or charts.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 1/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 1 · AUTONOMOUS MED SCHOOL — Gemini 4 Argon learns surgical criteria from public video -> out/manual.json (0 human labels)`

### Shot 2 — Zero-Shot Bulk Frame Labelling (`0:12–0:24`, 12s)
- **File**: `out/promo/shot_02_label.mp4`
- **Pipeline Stage**: `python label.py` $\rightarrow$ `out/labels_manual.jsonl`
- **Narrative**: Guided by the reference manual it authored itself, Gemini 3.8 Flash scans surgical video frames in parallel—identifying instruments (`grasper`, `l_hook`) and anatomy (`gallbladder`, `cystic_duct`, `blood`, `liver`) with spatial bounding boxes and confidence scores.
- **Veo 3.1 Prompt**:
  > Cinematic close-up of a high-resolution medical imaging monitor scanning a sequence of synthetic laparoscopic tissue phantom frames. Glowing cyan spatial bounding boxes and coordinate reticles lock onto surgical instruments and anatomical regions in smooth motion. Dark laboratory background, volumetric light.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 2/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 2 · BULK LABELLING — Gemini 3.8 Flash labels anatomy & instruments using its self-written manual`

### Shot 3 — Autonomous Self-Verification Loop (`0:24–0:36`, 12s)
- **File**: `out/promo/shot_03_verify.mp4`
- **Pipeline Stage**: `python accept.py` $\rightarrow$ `out/decisions.csv`, `out/results.json`
- **Narrative**: Prior self-evolving pipelines (SurgSLOT) still required a human in the loop to verify pseudo-labels. `{{NAME}}` replaces the human verifier with an autonomous audit loop: cropping uncertain regions for re-analysis, verifying spatial consistency, accepting trusted labels (green), re-analysing borderline predictions (amber), and escalating ambiguous cases to a surgeon (red).
- **Veo 3.1 Prompt**:
  > Futuristic AI verification workstation screen showing a high-precision optical zoom lens inspecting a highlighted crop of a synthetic surgical frame. Amber inspection rings pulse and lock into emerald-green verified check indicators, while one uncertain frame routes into a crimson escalation lane. Smooth camera dolly, photorealistic lighting.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 3/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 3 · SELF-VERIFICATION — Agent checks its own labels: Accept (green) · Re-analyse (amber) · Escalate (red)`

### Shot 4 — Specialist Student Distillation (`0:36–0:48`, 12s)
- **File**: `out/promo/shot_04_student.mp4`
- **Pipeline Stage**: `python student.py` $\rightarrow$ `out/student.json`
- **Narrative**: Only labels that pass self-verification train a lightweight specialist classifier on visual embeddings (`data/features.npy`). Zero human labels are used in training; human ground-truth masks (`CholecSeg8k`) are touched only once to score held-out test videos.
- **Veo 3.1 Prompt**:
  > Cinematic macro visualization of a neural distillation pipeline: streams of glowing data particles pass through a crystalline verification filter that blocks noisy red sparks and transmits pure emerald-green verified vectors into a compact glowing processor core. Dark sci-fi documentary cinematography.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 4/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 4 · SPECIALIST STUDENT — Trained strictly on self-verified labels (0 human training labels)`

### Shot 5 — Live Pipeline Proof & `app.py` Screen Capture Slot (`0:48–1:02`, 14s)
- **File**: `out/promo/shot_05_proof_slot.mp4`
- **Pipeline Stage**: Live capture of `app.py` + real metrics from `out/results.json`, `out/student.json`, `out/decisions.csv`
- **Narrative**: Real quantitative evaluation on held-out CholecSeg8k test videos and interactive exploration in the Streamlit UI (`app.py`) for `{{NAME}}`.
- **Dynamic Slot Behaviour**:
  - **When `out/results.json` & `out/student.json` are present**: Renders the live `app.py` screen capture alongside exact test-split numbers (Accepted Accuracy, Coverage, Escalated Share, Self-Written Manual vs Blank Baseline $\Delta$, and Specialist Student Test Accuracy vs `oracle_upper_bound`).
  - **Before pipeline outputs land**: Renders the live `app.py` screen capture with an explicit `[SLOT RESERVED FOR REAL PIPELINE NUMBERS — AWAITING out/results.json & out/student.json]` banner. Never substitutes synthetic metrics.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 5/6`
  - Top badge: `REAL PIPELINE TELEMETRY & LIVE APP.PY CAPTURE — NO HUMAN LABELS IN TRAINING`
  - Bottom lower-third: `ACT 5 · HONEST PROOF — Held-out CholecSeg8k test evaluation · Prior work: SurgVLP, SurgSLOT, Med-Gemini`

### Shot 6 — Grounded Surgical Narration (`1:02–1:15`, 13s)
- **File**: `out/promo/shot_06_narrate.mp4`
- **Pipeline Stage**: `python narrate.py` & `python atlas.py` $\rightarrow$ `out/narration_*.json`, `out/atlas/`
- **Narrative**: Finally, `{{NAME}}` narrates surgical video ("Netflix for Surgery") grounded exclusively in verified structures and audited timeline events—refusing to speculate when evidence is uncertain.
- **Veo 3.1 Prompt**:
  > Sleek surgical education theatre display showing a synchronized endoscopic replay timeline with glowing phase markers and clean subtitle waveforms beneath the video frame. A modern clinical viewing room with soft ambient blue lighting, slow camera push-in, cinematic finish.
- **On-Screen Captions**:
  - Top-left pill: `{{NAME}} · SHOT 6/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 6 · GROUNDED NARRATION — {{NAME}} narrates only what it verified (Gemini 4 Argon + 3.8 Flash + Veo 3.1)`
