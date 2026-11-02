STATUS: hardened, with one item not met: phase accuracy (qa item 2) cannot be scored, because no phase ground truth exists on this machine. Everything else in the qa section passes. See NOT MET.

Scope for this round: single-user localhost research tool, laparoscopic cholecystectomy, public or simulator footage only.

NOT MET
- Phase accuracy (item 2). CholecSeg8k has no phase labels. Cholec80 phase annotations come through a CAMMA request form. Unofficial copies exist on Hugging Face and I did not use them, because that would sidestep the form. Needs Nir: request the annotations, or accept phase as unmeasured. surgproof/eval.py refuses to report a phase accuracy. Only agreement between the passes is shown, which is not accuracy.

RESULTS (docs/e2e_status/qa_results.json; surgproof/eval.py aggregate; CIs are t intervals over per-video values)
Model calls from core 20b9c94, prompts.py sha256 0c717e759fc9. Rebuilt from the cache on core d861281, masks 2d52d80, web 184351c: 606 of 606 decisions identical, no call repeated, keyframe IoU moved by at most 0.0012 on 2 of 14 jobs.
| set | videos / clips / keyframes | final, mean over videos | 95% CI | n decided | coverage | amber | same labels pooled: final / detect / second pass |
| never scored before (primary): video09, 17, 48 | 3 / 10 / 36 | 0.932 | 0.785 to 1.000 | 183 | 0.847 | 33 | 0.929 / 0.934 / 0.940 |
| all unused clips, all 6 held-out videos | 6 / 21 / 74 | 0.957 | 0.906 to 1.000 | 351 | 0.790 | 93 | 0.952 / 0.952 / 0.943 |
| rare labels only: video01, train, not held-out | 1 / 8 / 27 | 0.981 | none, one video | 107 | 0.661 | 55 | 0.981 / 0.972 / 0.935 |

What came out, worst first:
- Cystic duct does not work. Held-out keyframes have no cystic duct, so sensitivity there is unknown. On video01, both passes missed 11 of 11. Final shows only n=1 because the other 10 went amber for a person. This is the label a surgeon would care about most.
- The double-check adds no accuracy. On the labels final decided, final is not better than detect alone: 0.929 vs 0.934 on the primary set, 0.952 vs 0.952 on all 6. It costs a second model pass, and its value is routing about a fifth of labels to a person.
- amber_said_present is the weak route. All 6 held-out videos: 33 of 42 right (9 false accepts), against both_passes_present 141 of 141. Primary set: 13 of 19. Every false accept in the held-out set came from this route. The masks lane now audits this stratum, but these labels are still accepted automatically. On video01 it was 15 of 15, so the rate is not stable.
- both_passes_absent still misses: 8 of 168 held-out were present. These enter the dataset as ai_absent.
- The amber rule is justified: on video01, amber_said_absent was wrong 15 of 28 times. Before this round each of those would have been a reject.
- Blood: no held-out positives, so sensitivity is unknown there. 0 false positives on 74 held-out keyframes. video01 alone: detect 26 of 27, second pass 27 of 27. One video gives no interval.
- Phase: the passes agree on 23 of 74 held-out keyframes (6 of 36 on the primary set, 19 of 27 on video01). Treat phase as a guess. Unknown was never chosen in these runs.
- Masks: IoU counting errors as 0 is 0.555 over 6 videos (CI 0.325 to 0.785) and 0.474 over 3 (CI 0 to 1, meaningless). Worst video 0.198.
- Spent clips are left out, and that flatters the table. video12_19500 and 19580 are the hardest known: coverage 0.417, final 0.933 (n=15), IoU 0.221, from the one real run of test_real_run_end_to_end. Disclosed in the README.
- Review load: one 3.2-second clip produced 36 review items (7 amber, 29 audit). Audit caps have not been tested on a long video.

How to read this:
- The 6-video row contains the 3-video row. Its other 3 videos had other clips examined earlier, so it is not clean. The primary row is 10 clips: a very small sample.
- Spent: brief named video12_15750, video28_00000 and video55_00508. I also excluded the clips used alongside them earlier (video12_19500, 19580, video28_00080, video55_00588), and the 7 video01 clips core read by eye while writing prompts. The scorer refuses all of them.
- Train-split numbers exist only because held-out and calib lack the labels. All blood and duct positives are in one video.
- Numbers from before this round (0.848, 0.613) used old prompts and an old amber rule, on different clips, so they are not comparable.
- Video-level CIs from 3 to 6 videos say nothing about other surgeons, hospitals or cameras.

ITEM 1: REGRESSION TEST FOR EVERY AUDIT FINDING
tests/test_qa_hardening.py: 119 passed on the merge below. They replay attacks through the CLI and HTTP routes, using tests/e2e_helpers/fake_claude.py for model misbehaviour. A pass here is evidence, not proof.
Fails-before, measured on acd86f8, 75107e6, 09c0048, at 92 tests: 44 failed, 45 errors, 3 passed. Weak spots:
- The 45 errors are the fixture failing because the old server had no login or notice route. That shows the feature was missing, not that each individual attack succeeded on the old code.
- The 3 that passed did not fail before: the two disguised-playlist attacks at the CLI, already refused, and plan_limit_once.
- One failure in that run (teleporting object) was my own fake-model bug, fixed afterwards. Tests added after that baseline, such as limit wordings and background release, were not run on the old code.
- A few checks read code rather than behaviour: subprocess timeouts by AST, killpg by text, the banner via render.status_text. The draft banner is not read from video pixels.
Findings this produced, all fixed by the owning lane and re-tested: bare 429 and generic "limit reached" read as a plan limit; real Claude Code wordings missed or misread (found by scanning the installed binary, which of them reach a headless call is unknown, and no real limit has been hit); prompt hashes with no text in the bundle (core d861281).
Existing red-team tests changed because the brief changed behaviour: 7 listed by core, the login migration, the audit stop rule, background release. Each still asserts the attack and adds the new requirement. One narrowing to review: test_eval_is_the_only_ground_truth_reader now allows exactly one verbatim licence sentence in export.py that names CholecSeg8k.

ITEM 3: README AND CLEAN CLONE
- README: 96.6% badge and table removed, along with Critical View of Safety, operative-record, "verified", live claims, launch film and integrity badge. Scope, limits and "no real patient video" are at the top. tests/test_qa_readme.py ties each results-table number to qa_results.json and forbids the retired claims.
- Clean clone, for real (tests/test_qa_clean_clone.py, SURGPROOF_CLEAN_CLONE=1): git clone, make setup, make serve, then real HTTP: bound to 127.0.0.1 even with HOST=0.0.0.0, forged Host 400, 401 before login, setup code, upload, mock job to done, dataset locked. Passed 2 of 2. It clones a local merge of the 4 branches, because main does not hold this code yet. Re-run after merging to main.

TEST RESULTS on a local merge of core d861281, masks 2d52d80, web c909a04, qa ab39658
hardening 119/119, HTTP red team 46/46, red team 65/65, e2e 18 passed and 1 skipped (the real run, listed below), inputs 11/11, scorer 49/49, README 45/45, test video tool 9/9, clean clone 2/2. Run one suite at a time because of disk.
- Found and fixed in this round: a release right after a job said done could get 409 "this job already has a worker" (3 of 13 runs on web e521b2d). Web fixed it at c909a04. test_whole_web_flow_with_mock_model then passed 25 of 25 on this merge, and no retry was added to the test.
Real runs, on the previous merge (web e521b2d; web's only change since is the release fix above):
- test_real_run_end_to_end passed for the first time ($0.51).
- One real job through the real server (cached calls, real claude health check): login, upload, final pass, signed release, release-check ok, bundle served, decisions identical to the CLI run, upload name not stored.

NOT VERIFIED
- A 5-minute video, mock or real. Estimated $17 and up to about 1 GB. The README says it has not been run.
- Which limit wording a real limit produces headlessly. No real limit has been hit.
- The UI in a browser, from this lane. HTTP only here. Styling checks are static.
- Any footage outside CholecSeg8k, including footage of other procedures. The out-of-scope gate is tested with a fake model only.

FOR THE ORCHESTRATOR (not mine to change)
- lanes/integrity.py runs in ~/surgatlas and scans nothing under surgproof/ or web/app/. "0 FAIL" says nothing about this code. tests/test_qa_redteam.py applies the same patterns.
- 2 legacy tests still fail, as they did before this round: tests/test_coach_fresh_clone.py::test_readme_sections_and_placeholders and tests/test_web_ui.py::test_missing_files_are_null_not_invented. Editing them is your call under AGENTS.md.
- amber_said_present staying auto-accepted and cystic duct being undetectable are product decisions. I recommend amber_said_present stays amber, and that cystic duct is removed from the supported labels or marked unreliable.

COST AND DISK
Model spend this round: $6.14, including $1.44 stopped when I found core's prompts were about to change. Free disk fell to about 190 MB during this work, mostly from swap (8.9 GB used) rather than job files. My scratch files were cleared.
