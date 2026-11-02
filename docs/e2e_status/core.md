STATUS: running
DONE:
- surgproof/vlm.py, pipeline.py (stages 1-4, 7, orchestration), run.py, prompts.py. tests/test_core_*.py: 119 pass, 1 real-model test opt-in.
- All QA findings handled (old job outputs cleared, narration now a whole-clause grammar, far out-of-frame outlines rejected, Read scoped to the call folder, real canary denied). QA's narration attack list: 27 pass, run directly on this commit. The rest of QA's suite has not been re-run here.
- Prompts fixed on the calib split only. Tuning clips (16 keyframes, 3 calib videos, final presence, resolved labels only):
  before: 0.909 (n=33), 0.882 (n=34), 0.944 (n=18). After: 1.000 (n=36), 1.000 (n=34), 0.947 (n=19). Ambers after: 2 and 5 on two of the videos.
  These are TUNING numbers on the clips the fix was designed from, so they are upper bounds and not evidence the fix generalises.
NOT DONE:
- Validation on unseen calib clips (video27_00400/00480/00640, video35_00940/01100). Run failed on the machine problem below. Only partial data: detect alone 0.889 (n=36) on the video35 clips, which says nothing about final decisions.
- Held-out result with the new prompts: none yet. QA's earlier held-out number (0.613, n=31, equal to baseline, 1 clip) was on the old prompts and still stands until someone re-runs it.
- Mask IoU not measured by core on calib: these runs stop before masks to save disk.
- Merged cross-lane test run after the last two commits. Previous one (at 666bbbc): 156 pass, 1 fail, since fixed in a local change now committed.
BLOCKED:
- Machine: disk is at about 400 MB free, with "No space left on device" during my run, and /opt/homebrew/bin/claude is currently missing, so no real model call can run on this machine, for any lane. Not touched by me: shared infra. Needs Nir or Aether.
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
