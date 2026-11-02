# masks lane status
STATUS: done (lane tests pass, origin/main merged, which was a no-op). See NOT VERIFIED below.

## DONE
surgproof/masks.py (stage 5), propagate.py (stage 6), render.py (stage 8), export.py (stage 9 and final pass), tests/test_masks_*.py.

## Test evidence
- tests/test_masks_*.py: 95 tests. 94 passed in the full run. test_amber_label_is_visibly_different_from_accepted hit "No space left on device" in that run, then passed when re-run alone. No clean single run of all 95 exists yet.
- Real clip (video12_15750, 80 frames, seeded by stored AI masks, no model calls): 80/80 frames rendered, frames inspected by eye.
- qa tests on latest core, web, qa and masks code in a scratch copy: 28 selected tests passed (masks, release, correction, rerun, ground truth). An earlier wider run was invalidated by disk-full errors, so most of those tests have not run against this code.
- lanes/integrity.py: 0 FAIL, run outside the sandbox because sqlite cannot open the cache inside it. It only scans ~/surgatlas, so it does not cover surgproof/ (qa's finding, orchestrator's to change).

## NUMBERS
GrabCut vs raw Sonnet polygons, calib split only (video20, 27, 35; 20 frames, n=56 masks, one polygon run). Calib was used to choose the parameters, so these are upper bounds, not a held-out result.
| label | n | seed IoU | refined IoU |
| gallbladder | 18 | 0.683 | 0.780 |
| grasper | 14 | 0.601 | 0.749 |
| l_hook | 4 | 0.823 | 0.864 |
| liver | 20 | 0.555 | 0.584 |
Macro mean 0.666 -> 0.744. Micro mean 0.627 -> 0.708. Liver barely improves. Blood and cystic_duct had no calib examples, so their settings are untested.
Held-out scoring is surgproof/eval.py's job. See qa's status for those numbers.

## NOT VERIFIED
- Full-length video: not run. qa asked lanes not to run a 5-minute video because of the disk.
- Speed: measured only at load average about 170, so it says little about a normal machine. Render took 0.39 s per frame before the drawing speedup (not re-timed after it), flow 0.28 s per pair. If that held, a 5-minute 25 fps video would take about 49 min to render and 70 min for flow.
- Full repo pytest: not run on this branch (disk and load). qa reports 2 legacy failures on main plus sandbox-only failures.

## Known limits
- Instrument outlines can barely shrink: erosion is small, chosen because it scored best on calib. An oversized tool outline stays oversized.
- Polygons cannot hold holes: an organ ring around a tool loses the hole in COCO `segmentation`. `area` still uses the true mask.
- A false-positive keyframe spreads to about 25 propagated annotations (qa saw this on video28), and the audit only samples, so catching it depends on the draw. Not changed here. Weighting the audit by downstream annotation count is a possible follow-up.
- Propagated frames fed by an AMBER keyframe stay out of the draft, even ones closer to the accepted end.
- When a human rejects or corrects a keyframe, its propagated frames are dropped and counted, not re-propagated.
- dataset_coco.json is a draft (info.released false). Only released_coco.json has passed the final pass.

## Interface
    surgproof.masks.run(job_dir, on_progress=None)
    surgproof.propagate.run(job_dir, video=None, on_progress=None)
    surgproof.render.run(job_dir, video=None, show_masks=True, subtitles=True, on_progress=None)   # annotated.mp4, narration.vtt
    surgproof.export.run(job_dir, on_progress=None)                                                 # dataset_coco.json, provenance.jsonl
    surgproof.export.final_pass_queue(job_dir, audit_frac=0.10)   # drawn once, persisted to final_pass_queue.json
    surgproof.export.record_final_pass(job_dir, rows)             # ValueError on bad rows
    surgproof.export.final_pass_state(job_dir)
    surgproof.export.release(job_dir, reviewer=None)              # raises ReleaseLocked while unresolved

final_pass.jsonl row: {"kf","label","action":"approve|reject|correct","polygon":[[x,y],...] (correct only),"reviewer","ts"}
Extra files beyond the contract: masks.jsonl, narration.vtt, final_pass_queue.json.
Masks are 0/255 single-channel at keyframe image size. In tracks.jsonl, `box` is smoothed for playback and `box_raw` is the exact mask bound. Keyframes have rows but no tracks/ PNG: use masks/<kf>_<label>.png.

NEXT: re-run all of tests/test_masks_*.py and the wider qa suite once there is disk space.
BLOCKED: disk nearly full on this Mac (under 500 MB free during this work, hit zero at least once). Not caused by this lane's outputs and not cleared by it: Nir or the orchestrator needs to free space.
