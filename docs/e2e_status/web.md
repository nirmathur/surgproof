STATUS: hardened

Scope: single-user, localhost, laparoscopic cholecystectomy, public or simulator footage only.

DONE (web section of the hardening brief):
1. Fixed bind to 127.0.0.1: there is no HOST setting and the instructions are gone. Local login: one account, created with a one-time code printed in the terminal; scrypt; mode 600; 12 h HttpOnly SameSite=Strict session; throttled. The reviewer comes from the session and any name in a request body is ignored. TrustedHostMiddleware. Cross-site writes refused. API docs off. File route is an exact-name allowlist. Before release, 403 on annotated.mp4, dataset_coco.json, released_coco.json, provenance.jsonl, final_pass.jsonl, dataset_bundle.zip, release_manifest.json. Everything else in a job folder is 404, always, apart from review frames.
2. Blocking upload notice. The attestation (who, when, footage source, full notice text and hash) is required by the server and stored in job.json. DELETE /api/jobs/{id}, also killing its worker. Retention defaults to 30 days (SURGPROOF_RETENTION_DAYS, 0 turns it off). Both remove cache entries with secure_delete, keeping entries another job still cites. The upload filename is never stored or listed: display name is "Job <id tail>". The original upload is deleted once ingest has written the stripped source.mp4; Retry runs from source.mp4. kf_fps capped at 5, matching the pipeline.
3. Disk preflight (507, needs SURGPROOF_MIN_FREE_MB, default 500, plus 3x the upload). Upload is streamed straight to staging, with the size limit enforced from the header and while streaming. Staging swept at startup. A second server refuses before touching the folder. Every failure in the runner surfaces, including when the disk is too full to record it. Each worker has its own process group, killed on stop, delete, retry and crash. A kernel flock per job gives the single-worker guarantee and instant dead-worker detection. Retry refused after release or once a person has started reviewing.
4. Player: distinct styles and a key for reviewed, AI keyframe, tracked and AI unsure. A visible banner and automatic retry when overlays or video fail. No stale boxes while loading. Download page counts what a person reviewed from the released file itself. /api/health reports disk, tools and model backend.
5. requirements-e2e.lock and requirements-e2e-dev.lock, hash-pinned, with `pip --require-hashes`. Dev dependencies are separate (`make setup-dev`).

6. From masks' report: release runs in the background in its own process, under the same per-job lock as the pipeline. POST returns 202; progress is on GET final-pass `release_job` and survives a reload. `wait: true` gives scripts the old blocking answer (200, 409 locked, 500 with reason). Nothing can change during a release (reviews, second release, retry, delete all 409), and the job lock is not held during it. An interrupted release reports failed, with nothing released, and can be tried again. dataset_bundle.zip and release_manifest.json are on Download and the allowlist. Download runs export.verify_release through GET release-check and shows the result. Audit wording follows stratum and kind, and no longer says "random" or "accepted" for everything. Download states the deletion date, released dataset included. If the release had no source video, Download says annotated.mp4 is still the unreviewed draft.

"Invalid host header" (masks' merged run): not state leaking. TrustedHostMiddleware is per app, and every create_app in tests now goes through local_client. Legacy apps, qa and web run in one process at this head gave 0 occurrences, and tests/test_web_hardening.py::test_host_refusal_belongs_to_each_app_and_does_not_leak pins it. The likely cause is that masks' scratch copy had qa's tests from before their migration; I did not re-run masks' exact copy, so that part is inference.

EVIDENCE:
- Merged core, masks, qa and web (622d972), real headless Chrome: tests/test_web_hardening.py + test_web_server.py + test_web_browser.py, 219 of 219 passed, no leftover processes. tests/test_qa_http_redteam.py passes in full on the same head. The wider QA and masks suites were not re-run by me this round.
- Same attack script against pre-hardening 89b6435 and this code: anonymous GET jobs 200->401; Host evil.example 200->400; annotated.mp4 and provenance before release 200->403; worker.log, detect.jsonl, mask PNG 200->404; no attestation 201->422; kf_fps=20 201->422; filename in listing yes->no; 16 GB Content-Length body read->refused unread; dead worker running->stalled; spoofed reviewer recorded->not; worker cannot start queued->failed. Retry after release showed 409 in both, because the fixture had no upload, so that row proves nothing; test_retry_is_refused_after_release covers it.
- Live `make serve` with HOST=0.0.0.0: listens on 127.0.0.1:8658 only, LAN address refused.
- Real `make setup && make serve` from a `git archive` export of this branch (not a full clone): 199 MB venv, all pipeline modules import, pytest absent, server healthy.
- Audit stop rule through the server: test_audit_stop_rule_holds_through_the_server passes.

NOT VERIFIED:
- The web flow on the real model. Only the mock model and stand-ins went through this server.
- A 5-minute video through Watch. Disk. Overlay for 2 s of 1080p takes about 2.0 s on this loaded machine, so it barely keeps pace.
- A full `git clone`. qa's clean-clone test is the proper check.
- The bundle's rights text. Export reads `attestation` from job.json according to its code; no test here checks the wording it produces.
- How long a release takes on a real video: only synthetic clips went through it.
- Process-group kill, flock, retention and disk-fill behaviour have regression tests but are not in the before/after script. Those tests were not run against the old code, which has no login route to run them through.

KNOWN LIMITS:
- /api/health checks that the claude CLI is logged in, not usage limits.
- Sessions are in memory: restarting the server signs you out.
- secure_delete overwrites rows in the SQLite file. It does not reach filesystem snapshots or backups.
- Delete keeps uncited cache rows while another job could still resume, and always when SURGPROOF_CACHE points at a shared file. They cannot be attributed to a job.
- A video that fails during ingest keeps its original upload (audio and metadata included) until deleted or expired, because Retry needs it.
- Retention deletes released datasets too, at 30 days by default. The job page shows the date.

DECISIONS TO CHECK:
- POST release is now 202 by default, so scripted clients need `wait: true` or must poll. qa has migrated.
- annotated.mp4 and provenance.jsonl are 403 until release, per "gate unreleased artifacts". Acceptance step 3's Download therefore offers them only after the final pass; Watch shows the AI draft in the app before that.
- Tests that are not about disk need SURGPROOF_MIN_FREE_MB=1 while the machine is this low.

BLOCKED:
- orchestrator: jobs/ still needs a line in .gitignore, which I do not own. The server and core each write a stopgap .gitignore.
- Nir: disk. Free space hit zero during an earlier test run, ranged from about 200 to 840 MB this round, and was 715 MB at last check. It blocks the 5-minute test.
