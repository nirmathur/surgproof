STATUS: hardened

Scope: single-user, localhost, laparoscopic cholecystectomy, public or simulator footage only.

DONE (web section of the hardening brief):
1. Fixed bind to 127.0.0.1: there is no HOST setting and the instructions are gone. Local login: one account, created with a one-time code printed in the terminal; scrypt; mode 600; 12 h HttpOnly SameSite=Strict session; throttled. The reviewer comes from the session and any name in a request body is ignored. TrustedHostMiddleware. Cross-site writes refused. API docs off. File route is an exact-name allowlist. Before release, 403 on annotated.mp4, dataset_coco.json, released_coco.json, provenance.jsonl, final_pass.jsonl, dataset_bundle.zip, release_manifest.json. Everything else in a job folder is 404, always, apart from review frames.
2. Blocking upload notice. The attestation (who, when, footage source, full notice text and hash) is required by the server and stored in job.json. DELETE /api/jobs/{id}, also killing its worker. Retention defaults to 30 days (SURGPROOF_RETENTION_DAYS, 0 turns it off). Both remove cache entries with secure_delete, keeping entries another job still cites. The upload filename is never stored or listed: display name is "Job <id tail>". The original upload is deleted once ingest has written the stripped source.mp4; Retry runs from source.mp4. kf_fps capped at 5, matching the pipeline.
3. Disk preflight (507, needs SURGPROOF_MIN_FREE_MB, default 500, plus 3x the upload). Upload is streamed straight to staging, with the size limit enforced from the header and while streaming. Staging swept at startup. A second server refuses before touching the folder. Every failure in the runner surfaces, including when the disk is too full to record it. Each worker has its own process group, killed on stop, delete, retry and crash. A kernel flock per job gives the single-worker guarantee and instant dead-worker detection. Retry refused after release or once a person has started reviewing.
4. Player: distinct styles and a key for reviewed, AI keyframe, tracked and AI unsure. A visible banner and automatic retry when overlays or video fail. No stale boxes while loading. Download page counts what a person reviewed from the released file itself. /api/health reports disk, tools and model backend.
5. requirements-e2e.lock and requirements-e2e-dev.lock, hash-pinned, with `pip --require-hashes`. Dev dependencies are separate (`make setup-dev`).

EVIDENCE:
- Merged core, masks, qa and web (f30d923), real headless Chrome: 250 tests, 248 passed. The 2 failures: a browser test of mine that assumed the old audit behaviour (fixed; all 12 browser tests then passed on a re-run) and qa's test_human_reject_removes_the_label_and_what_flowed_from_it (see BLOCKED). The non-browser suites were not re-run after that, because only test files changed. The head also has a later merge of qa's test changes, on which I ran only the two tests named here.
- Same attack script against pre-hardening 89b6435 and this code: anonymous GET jobs 200->401; Host evil.example 200->400; annotated.mp4 and provenance before release 200->403; worker.log, detect.jsonl, mask PNG 200->404; no attestation 201->422; kf_fps=20 201->422; filename in listing yes->no; 16 GB Content-Length body read->refused unread; dead worker running->stalled; spoofed reviewer recorded->not; worker cannot start queued->failed. Retry after release showed 409 in both, because the fixture had no upload, so that row proves nothing; test_retry_is_refused_after_release covers it.
- Live `make serve` with HOST=0.0.0.0: listens on 127.0.0.1:8658 only, LAN address refused.
- Real `make setup && make serve` from a `git archive` export of this branch (not a full clone): 199 MB venv, all pipeline modules import, pytest absent, server healthy.
- Audit stop rule through the server: test_audit_stop_rule_holds_through_the_server passes.

NOT VERIFIED:
- The web flow on the real model. Only the mock model and stand-ins went through this server.
- A 5-minute video through Watch. Disk. Overlay for 2 s of 1080p takes about 2.0 s on this loaded machine, so it barely keeps pace.
- A full `git clone`. qa's clean-clone test is the proper check.
- Process-group kill, flock, retention and disk-fill behaviour have regression tests but are not in the before/after script. Those tests were not run against the old code, which has no login route to run them through.

KNOWN LIMITS:
- /api/health checks that the claude CLI is logged in, not usage limits.
- Sessions are in memory: restarting the server signs you out.
- secure_delete overwrites rows in the SQLite file. It does not reach filesystem snapshots or backups.
- Delete keeps uncited cache rows while another job could still resume, and always when SURGPROOF_CACHE points at a shared file. They cannot be attributed to a job.
- A video that fails during ingest keeps its original upload (audio and metadata included) until deleted or expired, because Retry needs it.
- Retention deletes released datasets too, at 30 days by default. The job page shows the date.

DECISIONS TO CHECK:
- annotated.mp4 and provenance.jsonl are 403 until release, per "gate unreleased artifacts". Acceptance step 3's Download therefore offers them only after the final pass; Watch shows the AI draft in the app before that.
- Tests that are not about disk need SURGPROOF_MIN_FREE_MB=1 while the machine is this low.

BLOCKED:
- qa: test_human_reject_removes_the_label_and_what_flowed_from_it still fails on this head. Its selector can pick an ai_absent item; a human reject there agrees with the AI (export._is_failure returns False), so release 200 is correct and the test's expected 409 is wrong for that item. Reported to qa; their file, not edited. Until they change it, that test stays red.
- orchestrator: jobs/ still needs a line in .gitignore, which I do not own. The server and core each write a stopgap .gitignore.
- Nir: disk. Free space hit zero during an earlier test run and was under 500 MB at last check.
