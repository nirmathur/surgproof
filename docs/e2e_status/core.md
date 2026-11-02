STATUS: hardened
Core section of briefs/HARDENING.md is implemented. "Hardened" here means my tests and real checks pass. The orchestrator's own attack re-run has not happened, and the open items below are not closed by this status.
Scope: single-user, localhost, laparoscopic cholecystectomy, public or simulator footage only.

DONE, by brief item:
1. Input gate: 5 spread-out keyframes are checked first, then all. Fails on mostly non-laparoscopic frames, mostly another procedure, mostly Unknown phase, or nothing accepted. status.json `result` is out_of_scope or nothing_accepted, and the error names the supported procedure. New scene.jsonl.
2. Amber loop cannot settle a reject: a confident absent stays AMBER with `amber_suggests`. A failed or missing pass stays AMBER and skips the loop. Every decision row has `route`.
3. source.mp4 has no audio or metadata. Frames are plain JPEG. No upload name in meta.json, status.json or error text. Random `display_name`. CLI default folder is jobs/job-<random>.
4. Timestamp uniformity check replaces the frame-rate comparison, and non-uniform video is re-encoded. Timeouts on every ffmpeg and ffprobe call, protocol and format whitelists, and resolution and fps caps.
5. kf_fps cap 5. Cost cap (default $25, SURGPROOF_MAX_COST_USD or --max-cost), checked by estimate before any call and on actual spend during the job. Plan usage limit pauses and resumes by itself. After 6 hours the job stops with result paused, and a rerun resumes from cache.
6. Narration: whole-clause grammar stays. Every segment starts "AI draft." and narration.json has `draft` and `notice`. "Verified" wording removed from narration, the narration prompt and the CLI.

7. Added on request: prompts.jsonl in each job folder, rows {prompt_hash, stage, model, prompt}, one per distinct prompt, written before the call so failed calls resolve too. This matches masks' reader in export.py. Prompt text is unchanged.

TESTS:
- tests/test_core_*.py: 210 pass, 1 real-model test opt-in. 50 of the first 51 hardening tests failed on the old code. The disguised-playlist test already passed, so it is a guard, not proof of a fix.
- Merged run with masks, web and qa at f637afe: 358 pass, 7 fail, 2 skipped. All 7 assert old behaviour the brief changed; none is a core defect. Listed below, QA is updating them. So QA's suite is NOT green until that lands.
  test_qa_redteam: test_nothing_seen_means_nothing_drawn, test_unsure_model_stays_amber_and_nothing_is_exported, test_degenerate_polygon_is_never_accepted, test_absurd_coordinates_do_not_become_a_full_frame_mask, test_rerun_into_the_same_folder_leaves_no_stale_results (nothing accepted now exits 1).
  test_e2e: test_masks_only_for_accepted_present (masks' AMBER proposal for cystic_duct now exists), test_scorer_finds_truth_for_remuxed_source_and_frames_align (presence_final n is 30 with 6 amber, not 36).
- Real model: test-pattern video stopped at 5 of 12 frames for $0.06 with a clear message. One calib clip ran the full pipeline, with no audio or identifying tags in source.mp4 or annotated.mp4.

NUMBERS (all calib, none held out, so none of this is a proof result):
- Before the amber rule, unseen calib clips: video27 final 0.950 (n=60, 0 amber), detect only 0.933, baseline 0.500. video35 final 0.975 (n=40, 2 amber), detect only 0.905, baseline 0.525. These are the prompts before the scene field was added.
- After the amber rule, one tuning clip (video27): final 1.000 on resolved labels (n=27), coverage 0.75, 9 of 36 amber, detect only 0.917. The clip was used for tuning and is one clip, so treat this as an upper bound. The rule costs human review.
- Mask IoU counting errors as 0: 0.621 (n=33) and 0.692 (n=19) on unseen calib clips, 0.598 (n=11) on the tuning clip.

OPEN, NOT PROVEN:
- No held-out number exists for this code. QA owns that run. QA's 0.613 (n=31) was on the old prompts.
- 3 l_hook false positives on unseen calib clips from the "bare shaft counts" definition, giving 75 false-positive annotations in that clip's COCO export. The amber rule does not stop them: they were amber accepts. Deliberately not tuned further.
- Blood and cystic duct: calib has almost no positives, so these are effectively unmeasured.
- Phase is often wrong or uncertain from single frames.
- The plan-limit pause was tested with simulated errors. Patterns now use wording QA found in the Claude Code 2.1.316 binary, but which strings reach a headless call is unknown, and no real limit has been hit. An unmatched wording would be retried, then counted as a failure, not silently passed. Credit errors fail visibly on purpose: they never lift.
- 5-minute video never run. About $11 and 20+ minutes are linear estimates.
- One outline per label, so a second instrument of the same kind is not outlined.

BLOCKED / REQUESTS:
- masks: copy decisions `route` into COCO attributes and provenance. Not there yet, so "export the decision route" is only done in decisions.jsonl. Also a confident amber absent is now AMBER, so `verified_absent` only ever comes from both passes agreeing.
- web: server.py line 707 stores the upload filename in job.json, and line 435 returns it in job listings. Brief item 3 forbids this. Use meta.json `display_name`. upload.* also still holds the original file with its metadata and audio. Lower the kf_fps form limit from 30 to 5 (the pipeline now refuses above 5). Show `result` and `paused` from status.json.
- orchestrator: please add /jobs to .gitignore (each job folder also self-ignores). `python` is not on PATH here, only `python3`.

FROZEN for QA's held-out run: prompts.py and the decision logic in pipeline.py, as of this commit.

Interfaces:
- `surgproof.pipeline.run(video_path, job_dir, kf_fps=1.0, max_cost=None)`. On failure it writes status.json error and raises. status.json also has stage_index, n_stages, stage_progress, stages, cost_usd, result, paused.
- phase.jsonl `phase` is the stabilised phase. raw_phase and raw_conf hold the model's answers. A new run clears earlier outputs but keeps upload.* and job.json.
- `SURGPROOF_VLM=mock` tags model fields mock:<model>. `SURGPROOF_CLAUDE_BIN` swaps the binary.
