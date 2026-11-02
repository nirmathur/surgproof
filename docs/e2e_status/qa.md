STATUS: blocked. Lane code and tests are finished. The held-out run on the new prompts, the 5-minute test and the clean-clone setup cannot run on this machine right now.

BLOCKED, needs Nir:
1. `claude` is broken on this Mac: /opt/homebrew/bin/claude is missing and @anthropic-ai/claude-code/bin/claude.exe is a 500-byte stub dated 18:58, probably an interrupted auto-update on the full disk (cause not confirmed). No real model call can run. Shared infrastructure, so not touched. /api/health correctly reports claude false.
2. Disk: 380 MB free at last check. It reached 0 at about 19:00 and commands failed outright. Swap is 8.4 GB and the Data volume holds 209.8 GB. All surgproof-e2e worktrees and temp files total under 0.6 GB, so this cannot be fixed from inside the lanes.

NOT VERIFIED (because of the above):
- New prompts on held-out data. All 3 real runs below predate core 7a2549c. Core's calib numbers (1.000, 1.000, 0.947) are on the clips it tuned on, so they are upper bounds. Clean clips reserved for this run, not looked at by anyone: video09, video17, video48. Mine to run.
- tests/test_e2e.py::test_real_run_end_to_end has never executed. The real runs were the same steps by hand (CLI then scorer) on an overlay of the lanes.
- Real narration against core's new grammar. If real sentences miss the shapes, subtitles fall back to text built from decisions (`source: decisions`).
- "Up to 5 minutes" (acceptance step 2). The mock scale test was killed by my disk watchdog at mask 19/299, so no result. A real 5-minute job is an estimated 350-670 MB plus the upload, and about $8-18 by linear estimate (not measured).
- Clean clone `make setup` (acceptance step 1): the venv is about 250 MB. Partial checks passed: `make serve` starts on system Python and serves /, /app.js and /api/health; a pip dry run in a fresh Python 3.14 venv resolves all of requirements-e2e.txt from wheels.
- A browser has not been driven by this lane. The web flow is tested over HTTP only.

DONE:
- tools/make_test_video.py (--split test|calib), surgproof/eval.py, tests/test_e2e.py, tests/test_qa_eval.py, tests/test_qa_make_test_video.py, tests/test_qa_redteam.py, tests/test_qa_http_redteam.py, tests/test_qa_inputs.py, tests/test_qa_readme.py, fake claude harness, README Run it.

TEST RESULTS on core acd86f8, masks 75107e6, web 09c0048 (overlay of committed heads, suites run one at a time because of disk):
- test_qa_redteam 65/65, test_qa_http_redteam 46/46, test_e2e 18 passed and 1 skipped (the real run), test_qa_eval 26/26, test_qa_readme 23/23.
- test_qa_inputs 11/11 and test_qa_make_test_video 8/8 were last run a commit or two earlier (core 91cde25), not repeated, to save disk.
- All of this uses the mock or fake model. No real model sentence has been checked against core's narration grammar.

OPEN FINDINGS: none from the red team against the lanes' code. Still open for the orchestrator:
- orchestrator: lanes/integrity.py does os.chdir(~/surgatlas) and globs web/*.js, so it scans NO file under surgproof/ or web/app/. "0 FAIL" in acceptance step 6 says nothing about the new code. tests/test_qa_redteam.py applies the same patterns to it. integrity.py is yours to change.
- orchestrator: 2 legacy tests fail on main regardless of this work, and step 6 needs pytest to pass: tests/test_coach_fresh_clone.py::test_readme_sections_and_placeholders (asserts README strings removed by the README rewrite) and tests/test_web_ui.py::test_missing_files_are_null_not_invented (web/data bundle fills in the "missing" files). Editing or deleting tests is your call under AGENTS.md, so I have not touched them. 9 more legacy tests fail only inside the Claude Code sandbox (port binding, sqlite). The `make setup` venv lacks the legacy suite's dependencies (pandas, torch, streamlit), so plain `pytest` there fails at collection.
- orchestrator: README top still says "agent: Gemini 4 Argon" and carries the 96.6% badge. Those numbers are from the earlier pipeline, and Run it now says so, but the header reads as if they describe this one. Only "Run it" is my remit.

FIXED BY OTHER LANES AND RE-VERIFIED BY ME: rerun into a used folder keeps no old sign-off (core, masks); out-of-frame polygons no longer accepted (core); Read scoped to the call folder (core); release gate case bypass and nameless release (web); zero-area correction (masks, web); narration perforated, torn, negation tricks, comma tricks, whole-segment claims and unchecked phase statements (core).

RESULT, REAL RUNS, OLD PROMPTS ONLY (before core 7a2549c). 3 held-out clips picked metric-blind, 6 s and 6 keyframes each. A small sample; labels within a clip are correlated:
| clip | final acc (n) | baseline | detect only | verify only | amber | IoU counting errors as 0 (n) | IoU both present (n) | cost |
| video12_19500 | 0.613 (31) | 0.613 | 0.667 | 0.500 | 5 | 0.172 (19) | 0.466 (7) | $0.36 |
| video28_00000 | 0.972 (36) | 0.833 | 0.972 | 0.917 | 0 | 0.573 (7) | 0.668 (6) | $0.16 |
| video55_00508 | 0.938 (32) | 0.594 | 0.944 | 0.694 | 4 | 0.624 (19) | 0.697 (17) | $0.35 |
Pooled: final 0.848 on 99 resolved labels (9 amber, coverage 0.917), detect alone 0.861 on 108, verify alone 0.704 on 108, baseline 0.556. Final recall 0.682 vs detect 0.736, final precision 0.968 vs detect 0.975. So on these clips the double-check did not beat detect alone. These numbers do not reproduce the README's 96.6%, which came from the earlier Gemini pipeline with a different label mix; do not reuse that claim for this pipeline.
These 3 clips have been looked at per label, so they are spent as clean tests.
Errors multiply downstream: on video28, one false-positive gallbladder keyframe became 25 false-positive annotations through propagation, and the audit only samples accepted keyframes, so catching it depends on the draw.
Working as designed: 18/18 keyframes align with ground truth; COCO IoU matches mask IoU on each clip; propagated IoU is close to keyframe IoU (0.667 vs 0.668, 0.620 vs 0.659, 0.404 vs 0.426, each when both say present).
Run time 224-398 s per 6-second clip on a loaded machine.
