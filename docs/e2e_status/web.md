STATUS: running
DONE: surgproof/server.py (upload, jobs, status, files, video, overlay, watch, final-pass), web/app/ (Upload, Job, Watch, Final pass, Download), Makefile setup/serve, requirements-e2e.txt. Tests not yet in.
NEXT: tests/test_web_*.py, real browser run, integration against e2e/core and e2e/masks.
BLOCKED: nothing yet. Notes for other lanes:
- masks: the server calls export.final_pass_queue(job, audit_frac=0.10), export.final_pass_state, export.record_final_pass, export.release(job, reviewer=...), and catches export.ReleaseLocked (409) and ValueError from record_final_pass (422). Not yet tested, because export.py is not committed.
- core: the upload is stored at jobs/<id>/upload.<ext>, and I call pipeline.run(video, job_dir, kf_fps=...) in a spawned worker process. Watch plays meta.json `source`.
- orchestrator: jobs/ needs a line in .gitignore. I do not own that file, so the server writes jobs/.gitignore at startup.
