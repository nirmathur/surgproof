# masks lane status
STATUS: running (interfaces fixed, implementation in progress)

## Interface other lanes can code against now
All functions take the job dir (str or Path). `on_progress(frac, msg)` is optional; without it each stage updates status.json itself.

    surgproof.masks.run(job_dir, on_progress=None)                    # stage 5, needs detect/verify/decisions.jsonl, frames/
    surgproof.propagate.run(job_dir, video=None, on_progress=None)    # stage 6, writes tracks/, tracks.jsonl
    surgproof.render.run(job_dir, video=None, show_masks=True, subtitles=True, on_progress=None)  # stage 8, annotated.mp4, narration.vtt
    surgproof.export.run(job_dir, on_progress=None)                   # stage 9, dataset_coco.json, provenance.jsonl
    # final pass (web lane calls these):
    surgproof.export.final_pass_queue(job_dir, audit_frac=0.10)       # stable list of items, persisted to final_pass_queue.json
    surgproof.export.record_final_pass(job_dir, rows)                 # validates and appends to final_pass.jsonl
    surgproof.export.final_pass_state(job_dir)                        # {"n_required","n_done","remaining":[...],"locked":bool}
    surgproof.export.release(job_dir, reviewer=None)                  # writes released_coco.json; raises export.ReleaseLocked while unresolved

final_pass.jsonl row: {"kf","label","action":"approve|reject|correct","polygon":[[x,y],...] (correct only),"reviewer","ts"}

## Assumptions about core's outputs (core: tell me here or in your status file if any differ)
- Source video: `meta.json` key `source_video` (path relative to the job dir); otherwise I look for `source.*`, `input.*`, `video.*`, `original.*` in the job dir.
- `frame_index` in meta.keyframes is the 0-based decode-order index, and frames/<kf>.jpg is that exact frame. Extract with a select on frame number, not `fps=`, or masks will be misaligned with the video.
- Polygons and boxes are pixel coords of the keyframe image. verify `box` is [x0,y0,x1,y1].
- A decisions row may carry a corrected `polygon` inside `rounds[i]`; the latest one wins over detect.jsonl.
- A detect row with a polygon but no decisions row is treated as AMBER, never as accepted.

## Extra files beyond the contract
`masks.jsonl` (per-mask seed/fallback/area/box), `narration.vtt` (subtitle cues, for the Watch page), `final_pass_queue.json`.

## Output notes for qa/web
PNG masks are 0/255 single-channel at keyframe image size. In tracks.jsonl, `box` is smoothed for playback and `box_raw` is the exact mask bound. Keyframes have rows but no tracks/ PNG: use masks/<kf>_<label>.png.

DONE: read spec, inventory of existing tools.
NEXT: masks.py, propagate.py, render.py, export.py with tests.
BLOCKED: none
