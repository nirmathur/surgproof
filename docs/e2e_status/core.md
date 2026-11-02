STATUS: running
DONE: surgproof/vlm.py, pipeline.py (stages 1-4, 7, orchestration), run.py, prompts.py, tests (85 pass, 1 real-model test opt-in).
  Real end-to-end run on a 3.2 s training clip (video01, not held-out), combined with the masks lane's current files in a scratch copy: all 9 stages finished. 8 accept, 16 reject, 0 amber. annotated.mp4 has 80/80 frames. Model cost $0.07 for 4 keyframes.
NEXT: larger real run for JSON failure and amber rates, then merge origin/main and finish.
BLOCKED: nothing.
Interfaces for the other lanes:
- masks: pipeline calls `run(job_dir, on_progress=cb)` on all four of your modules. meta.json has both `source_video` and `source` (both `source.mp4`). decisions.jsonl rows carry `polygon`, equal to your rule (latest round polygon, else detect polygon).
- web and masks: `phase` in phase.jsonl is now the stabilised phase, so the header, Watch page and subtitles agree. The model's own answers are in `raw_phase` and `raw_conf`. `agree` compares the second pass with the stabilised phase.
- web: call `surgproof.pipeline.run(video_path, job_dir, kf_fps=1.0)`. On failure it writes status.json error and raises. status.json also has `stage_index`, `n_stages`, `stage_progress`, `stages`, `cost_usd`. Final stage is `done`.
- qa: `SURGPROOF_VLM=mock` tags every model field `mock:<model>`. `SURGPROOF_CLAUDE_BIN` swaps the binary. `meta.json` has `vlm_provider`.
- orchestrator: please add `/jobs` to .gitignore. Each job folder writes its own `.gitignore` as a guard meanwhile, so frames can't be committed. `python` isn't on PATH on this machine, only `python3`; the acceptance command needs a venv or `python3`.
