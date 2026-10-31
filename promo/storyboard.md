# SurgProof Pitch Video Storyboard (60–90s, Target: 75s)

- **Project**: **SurgProof** — *"No human labels. Only proof."*
- **Named Person**: **Nir Mathur**, King's College London (KCL) medical student who would otherwise label these frames by hand, plus a **Supervisor Quote Card Slot** (populated strictly from exact words once posted on Issue #1; never fabricated).
- **Lane**: `nir-promo` (Issue #10)
- **Primary Video Model**: `veo-3.1-generate-preview` (final shots) / `veo-3.1-fast-generate-preview` (drafts) via `google-genai` `client.models.generate_videos` (`predictLongRunning`)
- **Fallback Video Model**: `gemini-omni-1.1-flash`
- **Output Paths**: `out/promo/surgproof_promo.mp4` and `out/promo/surgatlas_promo.mp4` (1280×720, 24 fps, H.264 + AAC audio, ~75s total)
- **Individual Shots**: `out/promo/shot_01_med_school.mp4` .. `out/promo/shot_06_narrate.mp4`

## Honest Claims & Compliance Rules (Strictly Enforced)
1. **Mandatory Illustrative Caption**: Every AI-generated clip (`shot_01`, `shot_02`, `shot_03`, `shot_04`, `shot_06`) carries a permanent burned-in upper badge:
   `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
2. **Zero Fake Numbers / Charts / Endorsements**: Prompts and overlays for generated shots forbid fake metrics, fake charts, or simulated surgeon endorsements. The Supervisor Quote Card Slot on Shot 1 stays explicitly marked `[SUPERVISOR QUOTE SLOT — AWAITING VERBATIM QUOTE]` until real words arrive in `out/supervisor_quote.json` or Issue #1.
3. **Real Proof Slot (`shot_05_proof_slot.mp4`)**: Reads strictly from real pipeline artifacts (`out/results.json`, `out/student.json`, `out/decisions.csv`, `out/manual.json`) and a live headless capture of `app.py`. Until those files exist, the slot explicitly displays `AWAITING REAL PIPELINE OUTPUTS` with zero fabricated numbers. Excludes `liver` (100% prevalence) from pooled headline metrics.
4. **Honest Attribution & Prior Art on Screen**: Explicitly notes *"No human labels. Only proof. Ground truth evaluates held-out test videos only; calib/oracle reported as upper bounds"* and credits prior work (**SurgAtlas** dataset, **SurgVLP**, **SurgSLOT**, **Med-Gemini**; *our step: the agent replaces the human verifier*).

---

## Shot-by-Shot Breakdown (6 Shots, 75 Seconds Total)

### Shot 1 — Nir Mathur (KCL Medical Student) + Supervisor Quote Slot + Autonomous Medical School (`0:00–0:12`, 12s)
- **File**: `out/promo/shot_01_med_school.mp4`
- **Pipeline Stage**: `python med_school.py --video <URL>` $\rightarrow$ `out/manual.json`
- **On-Screen Cards**:
  1. **Nir Mathur — Medical Student, King's College London (KCL)**:
     *"As a KCL medical student, I would otherwise have to label thousands of surgical video frames by hand — so we built SurgProof to learn surgical criteria from public video and verify its own labels."* (Uses `out/promo/raw/nir_intro.mp4` or `nir_photo.png` if present, alongside the Veo 3.1 background).
  2. **Supervisor Quote Card Slot**:
     Reads `out/supervisor_quote.json` / `coord/artifacts/supervisor_quote.json` (`supervisor_name`, `supervisor_role`, `quote`). Until verbatim words arrive, displays: `[SUPERVISOR QUOTE SLOT — AWAITING VERBATIM NAME & WORDS (NEVER FABRICATED)]`.
- **Veo 3.1 Prompt**:
  > Cinematic wide shot inside a dark-mode surgical AI research lab at King's College London style medical workstation. A sleek glass monitor displays an abstract surgical lecture waveform transforming into glowing structured knowledge nodes and anatomical wireframe schematics. Warm rim lighting, shallow depth of field, clean modern interface aesthetic, no readable text or charts.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 1/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 1 · NIR MATHUR (KCL MED STUDENT) & AUTONOMOUS MED SCHOOL — med_school.py -> out/manual.json`

### Shot 2 — Zero-Shot Bulk Frame Labelling (`0:12–0:24`, 12s)
- **File**: `out/promo/shot_02_label.mp4`
- **Pipeline Stage**: `python label.py` $\rightarrow$ `out/labels_manual.jsonl`
- **Narrative**: Guided by the reference manual it authored itself, Gemini 3.8 Flash scans surgical video frames in parallel—identifying instruments (`grasper`, `l_hook`) and anatomy (`gallbladder`, `cystic_duct`, `blood`, `liver`) with spatial bounding boxes and confidence scores.
- **Veo 3.1 Prompt**:
  > Cinematic close-up of a high-resolution medical imaging monitor scanning a sequence of synthetic laparoscopic tissue phantom frames. Glowing cyan spatial bounding boxes and coordinate reticles lock onto surgical instruments and anatomical regions in smooth motion. Dark laboratory background, volumetric light.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 2/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 2 · BULK LABELLING — Gemini 3.8 Flash labels anatomy & instruments using its self-written manual`

### Shot 3 — Autonomous Self-Verification & AMBER Gate (`0:24–0:36`, 12s)
- **File**: `out/promo/shot_03_verify.mp4`
- **Pipeline Stage**: `python accept.py` $\rightarrow$ `out/decisions.csv`, `out/results.json`
- **Narrative**: Prior self-evolving pipelines (SurgSLOT) still required a human in the loop to verify every pseudo-label. **SurgProof** replaces the bulk human verifier with an autonomous audit loop: cropping uncertain regions for re-analysis, accepting high-trust verified labels (green), routing borderline cases to the **AMBER** human gate, and rejecting low-trust noise (red).
- **Veo 3.1 Prompt**:
  > Futuristic AI verification workstation screen showing a high-precision optical zoom lens inspecting a highlighted crop of a synthetic surgical frame. Amber inspection rings pulse and lock into emerald-green verified check indicators, while one uncertain frame routes into an amber review gate. Smooth camera dolly, photorealistic lighting.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 3/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 3 · SELF-VERIFICATION — Agent checks its own labels: Accept (green) · AMBER Gate · Reject (red)`

### Shot 4 — Specialist Student Distillation (`0:36–0:48`, 12s)
- **File**: `out/promo/shot_04_student.mp4`
- **Pipeline Stage**: `python student.py` $\rightarrow$ `out/student.json`
- **Narrative**: Only labels that pass self-verification train a lightweight specialist classifier on visual embeddings (`data/features.npy`). Zero human labels are used in training; human ground-truth masks (`CholecSeg8k` + external `Endoscapes2023` / `CholecT50`) are touched only once to score held-out test videos.
- **Veo 3.1 Prompt**:
  > Cinematic macro visualization of a neural distillation pipeline: streams of glowing data particles pass through a crystalline verification filter that blocks noisy red sparks and transmits pure emerald-green verified vectors into a compact glowing processor core. Dark sci-fi documentary cinematography.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 4/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 4 · SPECIALIST STUDENT — Trained strictly on self-verified labels (0 human training labels)`

### Shot 5 — Live Pipeline Proof & `app.py` Screen Capture Slot (`0:48–1:02`, 14s)
- **File**: `out/promo/shot_05_proof_slot.mp4`
- **Pipeline Stage**: Live capture of `app.py` + real metrics from `out/results.json`, `out/student.json`, `out/decisions.csv`
- **Narrative**: Real quantitative evaluation on held-out CholecSeg8k test videos (excluding 100%-prevalence liver from headline means) and interactive exploration in the **SurgProof** Streamlit UI (`app.py`).
- **Dynamic Slot Behaviour**:
  - **When `out/results.json` & `out/student.json` are present**: Renders the live `app.py` screen capture alongside exact test-split numbers (Accepted Accuracy excl. liver, Coverage, AMBER/Escalated Share, Self-Written Manual vs Blank Baseline $\Delta$, and Specialist Student Test Accuracy vs `oracle_upper_bound`).
  - **Before pipeline outputs land**: Renders the live `app.py` screen capture with an explicit `[SLOT RESERVED FOR REAL PIPELINE NUMBERS — AWAITING out/results.json & out/student.json]` banner. Never substitutes synthetic metrics.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 5/6`
  - Top badge: `REAL PIPELINE TELEMETRY & LIVE APP.PY CAPTURE — NO HUMAN LABELS. ONLY PROOF.`
  - Bottom lower-third: `ACT 5 · HONEST PROOF — Held-out CholecSeg8k test evaluation · Prior art: SurgAtlas, SurgVLP, SurgSLOT, Med-Gemini`

### Shot 6 — Human Gate & Grounded Surgical Proof (`1:02–1:15`, 13s)
- **File**: `out/promo/shot_06_narrate.mp4`
- **Pipeline Stage**: `streamlit run app.py` (Human Gate + Live Mode + Grounded Replay)
- **Narrative**: Finally, **SurgProof** locks dataset export behind the AMBER Human Gate and replays surgical video grounded exclusively in verified structures—giving medical students like Nir Mathur and surgical researchers a self-verifying data engine they can stand behind.
- **Veo 3.1 Prompt**:
  > Sleek surgical education theatre display showing a synchronized endoscopic replay timeline with glowing phase markers and clean audio waveform bars beneath the video frame. A modern clinical viewing room with soft ambient blue lighting, slow camera push-in, cinematic finish.
- **On-Screen Captions**:
  - Top-left pill: `SurgProof · SHOT 6/6`
  - Top badge: `ILLUSTRATIVE — AI-GENERATED VISUALIZATION (NOT CLINICAL FOOTAGE OR REAL RESULTS)`
  - Bottom lower-third: `ACT 6 · GROUNDED PROOF — SurgProof: No human labels. Only proof. (Gemini 4 Argon + 3.8 Flash + Veo 3.1)`
