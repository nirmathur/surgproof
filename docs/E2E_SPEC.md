# SurgProof end-to-end: drop in any surgical video, get a verified, annotated dataset

Owner of this spec: the orchestrator. Builders: four Claude Code sessions (core, masks, web, qa). Each works in its own git worktree on branch `e2e/<lane>`; the orchestrator merges. Read VISION.md, README.md and AGENTS.md (reward-hacking rules apply) first.

## Acceptance test (the orchestrator runs this; it is the definition of done)
1. `make setup && make serve` on a clean clone starts the app at http://localhost:8600.
2. On the **Upload** page, drop any laparoscopic video (mp4/mov, up to 5 minutes). A job page shows live progress through every stage.
3. When finished, the job page offers: **Watch** (the original video at full frame rate with the AI's boxes, masks and phase overlaid, and narration subtitles), **Final pass** (a human approves or corrects ambers plus a random audit sample, and Release stays locked until done), and **Download** (annotated MP4, COCO JSON dataset, provenance JSONL).
4. The same pipeline runs from the CLI: `python -m surgproof.run path/to/video.mp4 --out jobs/<id>`.
5. On a CholecSeg8k clip assembled into a video (`tools/make_test_video.py`), a separate scorer (`python -m surgproof.eval jobs/<id>`) reports presence accuracy and mask IoU vs ground truth with n. The pipeline itself never reads ground truth.
6. `lanes/integrity.py` passes with 0 FAIL; `pytest` passes.

## The model: Claude via headless Claude Code (no API key)
`surgproof/vlm.py` exposes `ask(image_paths, prompt, schema, model)` returning parsed JSON. Default provider `claude_cli`: runs `claude -p "<prompt referencing the image path>" --model <model> --allowedTools Read --output-format json`, extracts the JSON from `result`, validates it against `schema`, retries up to 2 times on invalid JSON, and caches by sha256(image bytes + prompt + model) in `jobs/cache.sqlite`. Concurrency is a semaphore (default 6). Keep the Gemini provider as optional (`SURGPROOF_VLM=gemini`). Models: `claude-sonnet-5-5` for detection, outlines, amber loop and narration; `claude-haiku-5-5` for the independent second pass.
Measured on one held-out frame: Sonnet outlines score IoU about 0.66 on instruments and about 0.3 on large organs, at about 22 s per call. So outlines are seeds, refined by classical edge snapping (below).

## Pipeline stages (each writes to jobs/<id>/ and updates jobs/<id>/status.json: {stage, progress 0-1, message})
1. **ingest**: ffprobe; extract keyframes at `--kf-fps` (default 1) to `frames/kf_XXXXX.jpg`; keep the source video for full-fps playback.
2. **detect** (Sonnet, one call per keyframe): presence of gallbladder, liver, grasper, l_hook, cystic_duct, blood, plus a polygon (pixel coords) for each present object, plus the surgical phase (the 7 Cholec80 phases) with confidence.
3. **verify** (Haiku, independent prompt, one call per keyframe): presence plus box for each label. Agree means accept or reject; disagree or low confidence means AMBER.
4. **amber loop** (Sonnet, crop around the disputed region, up to 2 rounds): resolve or keep AMBER.
5. **masks**: polygon -> GrabCut seeded by the polygon (eroded = sure foreground, outside dilated = sure background) -> keep the main component -> smooth contour. Boxes come from mask bounds.
6. **propagate**: optical flow (Farneback) forward from each keyframe to the next, blended with backward flow; keyframes are always the AI's own masks; propagated frames are flagged `propagated: true`. Then a 5-frame temporal majority and box smoothing (7-frame moving average) for stable playback.
7. **narrate** (Sonnet): a commentary per phase segment, built ONLY from accepted labels; each sentence cites its evidence (keyframe ids). Unsure means it says so.
8. **render**: annotated MP4 at source fps (house style: tools/render_dense.py CLEAN mode: phase header, clean boxes with label chips, optional translucent masks, small footer).
9. **export**: COCO JSON (images, categories, annotations with segmentation polygons, bbox, and attributes {status, source, keyframe|propagated, model, prompt_hash}), plus provenance.jsonl. Ambers are excluded until the human final pass resolves them.

## Lanes (files you own; touch nothing else)
- **core**: `surgproof/vlm.py`, `surgproof/pipeline.py` (stages 1-4, 7 and orchestration), `surgproof/run.py` (CLI), `surgproof/prompts.py`, `tests/test_core_*.py`.
- **masks**: `surgproof/masks.py` (stage 5), `surgproof/propagate.py` (stage 6), `surgproof/render.py` (stage 8, reusing tools/render_dense.py ideas), `surgproof/export.py` (stage 9), `tests/test_masks_*.py`. Write against the stage I/O contract below; use the existing out/dense_smooth data to test without calling the model.
- **web**: `surgproof/server.py` (FastAPI: POST /api/jobs upload, GET /api/jobs, GET /api/jobs/{id} status, GET /api/jobs/{id}/files/..., POST /api/jobs/{id}/final-pass), and `web/app/` (Upload, Job progress, Watch, Final pass, Download pages; design from promo/DESIGN.md, the dark clean style), plus a `Makefile` with `setup` and `serve`. Background jobs via a thread or process pool calling `surgproof.pipeline.run`.
- **qa**: `tools/make_test_video.py` (assemble CholecSeg8k held-out clips into mp4), `surgproof/eval.py` (scorer vs GT, separate from the pipeline), `tests/test_e2e.py` (runs the CLI on a 6-second test video with a mocked VLM, and then once for real), red-teams the others for fabrication, and keeps README "Run it" accurate.

## Stage I/O contract (jobs/<id>/)
- `status.json` `{"stage": str, "progress": float, "message": str, "started": iso, "updated": iso, "error": str|null}`
- `meta.json` `{"fps", "width", "height", "n_frames", "kf_fps", "keyframes": [{"kf": "kf_00012", "frame_index": int, "t": float}]}`
- `detect.jsonl` one row per keyframe and label: `{"kf", "label", "present", "conf", "polygon": [[x,y],...]|null, "model", "prompt_hash", "raw_ref"}`; `phase.jsonl` `{"kf", "phase", "conf"}`
- `verify.jsonl` `{"kf", "label", "present", "box", "model", "prompt_hash"}`; `decisions.jsonl` `{"kf", "label", "status": "accept|reject|AMBER", "final_present", "rounds": [...]}`
- `masks/<kf>_<label>.png`; `tracks/<frame_index>_<label>.png` (propagated, sparse); `tracks.jsonl` `{"frame_index", "label", "box", "propagated"}`
- `narration.json` `{"segments": [{"t0", "t1", "phase", "text", "evidence": [kf...]}]}`
- `annotated.mp4`, `dataset_coco.json`, `provenance.jsonl`, `final_pass.jsonl`, `released_coco.json`

## Rules
No ground truth anywhere in surgproof/ except surgproof/eval.py. No hand-typed boxes, verdicts or numbers in UI or video. Every model output is cached with its raw response, so it is traceable. Post progress in `docs/e2e_status/<lane>.md` (STATUS / DONE / NEXT / BLOCKED) every 15 minutes; the orchestrator reads only that and your commits.
