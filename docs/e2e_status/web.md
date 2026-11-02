STATUS: done, except `make setup` on a clean clone, which is NOT verified (see BLOCKED)

DONE:
- surgproof/server.py: POST /api/jobs, GET /api/jobs, GET /api/jobs/{id}, /files/..., /video, /watch, /overlay, /preview, /retry, GET and POST /final-pass. Each job runs in its own child process (log in jobs/<id>/worker.log), so a native crash cannot take the server down.
- web/app/: Upload, Job progress, Watch, Final pass, Download. Plain ES modules, no build step, no external requests.
- Makefile `setup` (private .venv from requirements-e2e.txt) and `serve` (port 8600, bound to 127.0.0.1). The old setup is now `setup-full`.
- The final pass is wired to surgproof.export: its queue, lock and release. Nothing is duplicated in the server.

TESTS (machine under heavy load, disk nearly full):
- Combined scratch tree of the core, masks, qa and web working copies: tests/test_web_server.py + tests/test_web_browser.py + tests/test_qa_http_redteam.py = 109 passed. Browser tests ran in real headless Chrome. They need a local port, so they skip inside the Claude Code sandbox.
- QA static checks on my files (test_qa_redteam.py, ground truth and hand-typed results): 7 selected, 7 passed.
- e2e/web alone, after merging origin/main (already up to date): 25 passed, 38 skipped. The 38 need surgproof.export or similar from other lanes, so THIS BRANCH ALONE PROVES ONLY THE 25. Re-run after merging.
- Of 6 runs on this branch: 4 clean. One had 16 errors, all ENOSPC when the disk hit 0. One had 1 failure whose output I did not capture; it did not recur, and disk exhaustion is my guess, not a finding.
- Live smoke: `make serve` + real pipeline on SURGPROOF_VLM=mock + real browser, 3 s synthetic clip: upload to done in 64 s, Watch decoded with overlay, Final pass locked, no JS errors. This ran before the last Watch and final-pass commits, which the browser tests cover but the live smoke does not.
- lanes/integrity.py: exit 0, 0 FAIL. As QA reported, it does not scan surgproof/ or web/app/, so this says nothing about this lane.

NOT DONE / NOT VERIFIED:
- `make setup` on a clean clone: never run by me. A venv is about 250 MB and the disk has been between 0 and 800 MB. Only `make -n setup` was checked. qa has it queued, also waiting on disk.
- The web flow on the real model: not run through the server by me. qa's real runs used the CLI.
- A 5-minute video through the Watch page: not run. Measured separately: overlay for 2 s of 1080p video with 4 labels takes about 2.0 s on this loaded machine, so it barely keeps pace. When it falls behind, the canvas clears and says "loading annotations" rather than showing stale boxes.

DECISIONS TO CHECK:
- orchestrator: dataset_coco.json returns 403 over HTTP until release, per VISION.md "Nothing is exported without that sign-off". The Download page offers released_coco.json only. An acceptance script that fetches the draft before release will get 403 by design. The scorer reads from disk and is unaffected.
- Release returns 422 without a reviewer name (qa's request). Individual clicks may be unnamed.
- Approve returns 422 if no mask file exists, because export would silently keep nothing for it.
- Mock-model jobs carry a visible "placeholders" warning on every job page, from meta.vlm_provider.
- The server has no login, so it binds to 127.0.0.1. HOST=0.0.0.0 exposes it.

BLOCKED:
- Nir: disk. Free space hit 0 during my test run (404 MB at last check). This blocks the clean-clone `make setup` check and the 5-minute test.
- orchestrator: jobs/ needs a line in .gitignore, which I do not own. The server writes jobs/.gitignore at startup as a stopgap, and core writes one per job.
