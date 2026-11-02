STATUS: running
DONE: real `claude -p` vision call tested. Minimal flags cut a call from $0.12 to $0.013. surgproof/vlm.py and surgproof/prompts.py drafted.
NEXT: surgproof/pipeline.py and surgproof/run.py, tests, a real run on a test clip.
BLOCKED: nothing yet. Interfaces for the other lanes:
- masks: pipeline.py will call `surgproof.masks.run(job_dir)`, `surgproof.propagate.run(job_dir)`, `surgproof.render.run(job_dir)` and `surgproof.export.run(job_dir)`. job_dir is a Path. An optional keyword `progress(fraction, message)` is passed if your function accepts it.
- masks: decisions.jsonl rows also carry `polygon`, the final outline in full-frame pixels. Use it instead of detect.jsonl, because the amber loop can supply or replace an outline. Only draw rows with status accept and final_present true.
- web: call `surgproof.pipeline.run(video_path, job_dir, kf_fps=1.0)`. On failure it writes status.json error and raises. status.json also has `stage_index`, `n_stages`, `stage_progress`, `stages`, `cost_usd`. Final stage is `done`. meta.json `source` names the browser-playable copy (`source.mp4`).
- qa: for the mocked run, set `SURGPROOF_VLM=mock`. Every model field then reads `mock:<model>`. `surgproof.vlm.ask` can also be monkeypatched.
