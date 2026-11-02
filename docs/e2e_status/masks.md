# masks lane status
STATUS: hardened
Means every masks item in the hardening brief is implemented and my tests and runs pass. The orchestrator's own re-run of the audit's attacks has not happened yet, and the REQUESTS below are open.

## Test evidence
- Regression tests: tests/test_masks_hardening.py, 66 tests. On the pre-hardening code (75107e6): 52 failed, 12 errored, 2 passed. The 2 that pass cover behaviour that was already right: an unaudited keyframe is not human_reviewed, and no bundle exists before release. Many old-code failures are missing fields or functions rather than wrong values.
- All tests/test_masks_*.py on hardened code: 158 passed in one clean run.
- Real CLI, mock model, one calib clip (80 frames): release locked, then released after a final pass; signature passed; editing released_coco.json was detected and `python -m surgproof.export verify` exited 1; all 80 COCO images resolved after running the bundle's extract tool; schema validated. Draft and released frames inspected by eye. The mock model says nothing about annotation quality.
- qa and web tests on latest branches, scratch copy: a 13-test subset covering the release and mask flows: 10 passed, 3 failed. One was my licence text naming a dataset, which trips qa's ground-truth tripwire (fixed; none of its tokens remain in my files, but that test itself was not re-run). The other two are the qa REQUESTS below. A wider run gave 26 failures of 144, mostly "Invalid host header" and the core input gate, and those clusters passed when run as the smaller subset, so I suspect state leaking between test files. Not confirmed, not all 26 re-run. Please re-run on your side.

## Items
1. human_reviewed is true only for the exact annotation a human judged. Added derived_from_reviewed (true only when all source keyframes were reviewed) and reviewed_source_keyframes.
2. Audit stop rule: above 10% failed overall, or 2+ failures above 10% in one label, the job escalates to review of every keyframe label. Sticky: changing verdicts cannot undo it, and hand-written final_pass rows are caught at release. An empty queue never releases.
3. Audit pools, stratified by label: both-pass accepted, amber accepted, AI absent, propagated. verified_absent is gone: ai_absent and human_absent are separate. Reject means "not present", so on an AI-absent item only a correction counts as a failure.
   Amber-accepted labels are all audited, up to 20 per label, per the request after core's report. Chosen because they had no independent agreement and they feed propagation, not tuned to any clip. decision_route is in COCO and provenance.
4. release() re-renders annotated.mp4 from the released data. Before that, the video carries an "AI DRAFT · UNREVIEWED" banner. Audio and container metadata are dropped. Build first, commit last: a failed render or a full disk releases nothing. HMAC-signed release_manifest.json, verify_release(), CLI check. final_pass.jsonl: fsync, torn last line tolerated and repaired, corruption elsewhere is an error.
5. Each propagated frame has fb_overlap and flow_check. Below 0.5, or no second direction to compare, it goes to AMBER and stays out of the dataset; approving a keyframe does not clear it. Every tracked frame stays an image with its exclusions listed; propagated images are never exhaustive.
6. Every mask has mask_quality draft. `source` no longer implies quality, `refinement` records what happened. instance_id per label per unbroken run within a shot.
7. dataset_bundle.zip, written at release: released COCO, provenance, final pass, source.mp4, frames, masks, annotated.mp4, phases, narration, tips, per-label confidence, versions (models actually recorded, commit, source hashes), code/ including prompts.py, prompt text per hash, schema, card, licences, signed inner manifest, frame-extract tool.

## NUMBERS
Flow check on the calib split, real Sonnet seeds at 40-frame gaps, 1222 frame-label rows from 10 clips of 3 videos. Rows within a clip are correlated, so treat as indicative. 0.5 was set before measuring and not tuned.
| overlap | share of frames | mean IoU vs GT | absent from GT |
| >= 0.5 (kept) | 58.6% | 0.647 | 5.4% |
| < 0.5 (AMBER) | 41.4% | 0.425 | 18.8% |
Correlation of overlap with IoU: 0.40. It helps but is a weak filter: **about 40% of propagated frames leave the dataset**, many of them good (one clip's grasper had overlap 0 and IoU 0.75, because a keyframe missed it). 5.4% false positives remain among kept frames.

## Limits to read before re-running the attacks
- The signature detects edits by anyone without the key, which sits beside the job folder (jobs/.surgproof_release_key, mode 600). Someone with the key can re-sign. That is not tamper-proofing on one user's machine.
- After escalation, "full review" means every keyframe label, not every propagated frame.
- Audit review load grows with length: 10% of each keyframe pool plus up to 10 propagated frames and 20 amber accepts per label. Not measured on a long video.
- tips.jsonl is a geometric estimate I added because the model reports no tips. Tested only on a synthetic shaft, never against real tips. Flagged validated false.
- Prompt text: detect and verify hashes are reproduced and checked. Amber and narration prompts depend on run-time data, so they show as unresolved until core records them (REQUESTS). The templates ship in code/prompts.py either way.
- No licence is claimed. Without a rights record the bundle says "not attested".
- Speed and a full-length video remain unmeasured. Release now re-renders, so it takes as long as a render.

## Older tests changed (all deliberate, listed because AGENTS.md treats test edits as suspect)
Removed: test_source_audio_is_kept and test_job_with_nothing_to_review_releases (the brief reverses both; test_masks_hardening asserts the opposite), test_footer_reflects_release_state (used a fake empty release; replaced by a real one).
Updated: verified_absent to ai_absent or human_absent; source names; fixtures gain flow_check and a video; audit test goes through the escalation; re-run test checks the old queue is gone rather than no queue exists; real-clip test now requires all 80 images, each unannotated one saying why (76 are annotated, the last 4 have no second flow).

## REQUESTS
- qa: tests/test_e2e.py::test_masks_only_for_accepted_present should skip masks/prop_*. Those are audit copies of propagated masks, kept in masks/ so web's routes work.
- qa: tests/test_qa_http_redteam.py::test_human_reject_removes_the_label_and_what_flowed_from_it assumes an audit item's kf is a keyframe. It can now be prop_<frame>; use the item's frame_index.
- web: add dataset_bundle.zip and release_manifest.json to DOWNLOADABLE, and show export.verify_release on Download.
- web: finalpass.js says "audit of a label the AI accepted". Items now carry `stratum` (accepted, amber_accepted, ai_absent, propagated) and kind can be `escalation`, so wording and counts need updating.
- web: release now renders video, so it can take minutes inside the request and the job lock. Run it in the background; release(job, reviewer, on_progress) reports progress.
- web: latest_verdicts should tolerate a torn last line of final_pass.jsonl as export does. Worth passing a rights record as rights.json or job.json "rights"; the card reads either.
- core: write prompts.jsonl ({prompt_hash, prompt}) at run time; export prefers it, which resolves the amber and narration prompts. Add release_manifest.json and dataset_bundle.zip to OUTPUT_FILES (export already clears them when stale).

BLOCKED: nothing in this lane. Disk hit zero again during this round, and a full-length run needs space first.
