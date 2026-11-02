STATUS: running
DONE:
- surgproof/vlm.py, pipeline.py (stages 1-4, 7, orchestration), run.py, prompts.py. tests/test_core_*.py: 135 pass, 1 real-model test opt-in.
- All QA findings handled (old job outputs cleared, narration now a whole-clause grammar, far out-of-frame outlines rejected, Read scoped to the call folder, real canary denied). QA's narration and phase-statement cases: 38 pass, run directly against this code. The rest of QA's suite has not been re-run here.
- Prompts fixed on the calib split only. Tuning clips (16 keyframes, 3 calib videos, final presence, resolved labels only):
  before: 0.909 (n=33), 0.882 (n=34), 0.944 (n=18). After: 1.000 (n=36), 1.000 (n=34), 0.947 (n=19). Ambers after: 2 and 5 on two of the videos.
  These are TUNING numbers on the clips the fix was designed from, so they are upper bounds and not evidence the fix generalises.
- Validation of the prompt fix on unseen calib clips (full pipeline with masks lane code, no tuning after seeing these). New segments but the same patients as the tuning clips, so weaker than new videos, and still calib, not a held-out result:
  video27 (3 clips, 10 keyframes): final 0.950 (n=60, 0 amber), detect only 0.933, verify only 0.983, baseline 0.500. Mask IoU counting errors as 0: 0.621 (n=33). Propagated frames, both present: 0.647 (n=690).
  video35 (2 clips, 7 keyframes): final 0.975 (n=40, 2 amber), detect only 0.905, verify only 0.929, baseline 0.525. Mask IoU counting errors as 0: 0.692 (n=19). Propagated frames, both present: 0.644 (n=443).
  Final beats detect alone on both. Verify alone beats final on video27.
  COST OF THE FIX: the "bare shaft counts" definition caused 3 l_hook false positives on video27 (detect and amber both accepted a dark shaft). The COCO export for that clip has 75 false-positive annotations from them. Before the fix calib had none. Not tuned away, so as not to spend more calib on it.
- Real narration against the grammar: both segments kept model sentences (`source` is the model), with 1 and 2 sentences dropped. Their phase is often wrong or uncertain: both clips said Preparation, marked uncertain.
NOT DONE:
- Held-out result with the new prompts: none yet. QA owns that run (video09, video17, video48). QA's held-out 0.613 was on the old prompts.
- Merged cross-lane test run on the latest commits.
- 5-minute video: never run. Cost and time are linear estimates.
BLOCKED: nothing now. claude came back at 19:17 and about 1.6 GB of disk is free.
KNOWN LIMITS:
- Narration grammar has not yet seen real model output, because claude is missing. If real sentences miss the shapes, subtitles fall back to text built from decisions: honest but plainer. Check this on the next real run.
- One outline per label, so a second grasper in a frame is not outlined. Costs mask IoU where two instruments are present.
- Cost is about $0.04 per keyframe, so a 5-minute video at 1 keyframe per second is about $11, and about 20 minutes or more.
- Phase comes from single frames with low confidence (often 0.35-0.6), so subtitles often say the phase is uncertain.
Interfaces for the other lanes:
- masks: pipeline calls `run(job_dir, on_progress=cb)` on all four modules. meta.json has `source_video` and `source`. decisions.jsonl rows carry `polygon`, matching your rule (latest round polygon, else detect polygon).
- web and masks: `phase` in phase.jsonl is the stabilised phase. `raw_phase` and `raw_conf` hold the model's answers. A new run clears earlier outputs from the job folder but keeps upload.* and job.json.
- web: call `surgproof.pipeline.run(video_path, job_dir, kf_fps=1.0)`. On failure it writes status.json error and raises.
- qa: `SURGPROOF_VLM=mock` tags model fields `mock:<model>`. `SURGPROOF_CLAUDE_BIN` swaps the binary.
- orchestrator: please add `/jobs` to .gitignore (each job folder also self-ignores). `python` is not on PATH here, only `python3`.
