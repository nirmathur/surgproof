# SurgAtlas: rules for every Gemini worker

Six Antigravity (agy) workers on Gemini 4 Argon build this project in parallel. Claude is the orchestrator and reads only your status file. The existing code is a first draft: every layer is assumed imperfect until your research and redesign say otherwise.

## Layout
- Each lane has its own git worktree at ~/surgatlas-lanes/<lane>, on branch lane/<lane>. The architect lane works in ~/surgatlas on main and is the only one that merges into main.
- raw/, data/, out/ and cache.sqlite are symlinks to the shared copies in ~/surgatlas. All lanes see the same data and outputs.
- Status files: out/lanes/<lane>.md (shared). Read other lanes' status files to see what exists.

## FOCUSED BUILD MODE: SUBMISSION 15:15 TODAY (feature freeze 14:30)
- Ship the README run order. No redesign phase. Fix what breaks in your files as you go.
- Research only if it unblocks you within 10 minutes. Only the architect runs deep research, one brief, in the background.
- Work in small, runnable steps. A smaller result that is real beats a perfect plan.

## Keep each other updated (HARD RULE, both machines)
- Shared workspace = GitHub Issues on nirmathur/surgproof. Bulletin = issue #1 (pinned). Your lane's task issues are labelled lane:<lane>.
- Every 10 minutes AND at every milestone, do both:
  1. Rewrite out/lanes/<lane>.md (shape below).
  2. `gh issue comment 1 -R nirmathur/surgproof -b "[nir-<lane>] DONE: ... | NEXT: ... | BLOCKED: ... | NEED: <who> -> <what>"`
- Before each task: `gh issue view 1 -R nirmathur/surgproof --comments | tail -80` and act on anything addressed to you.
- Claim a task: comment on its issue and `gh issue edit <n> -R nirmathur/surgproof --add-label status:claimed --remove-label status:todo`. Close it with a comment naming the artifact path and commit sha.
- Changing a file or schema another lane reads? Post on #1 with `-> <lane>` before you change it.
- Silence is a failure: the orchestrator restarts any lane with no post for 15 minutes.

## Git (proper, not optional)
- Work only in your own worktree and branch. Small commits, imperative messages ("Add mask-convention detector"), each one leaving tests passing.
- `git push -u origin lane/<lane>` after each meaningful commit. Never push to main unless you are the architect.
- Pull in others' merged work with `git merge main`. Do not rebase shared branches. Do not force-push.
- Never commit .env, keys, raw/, data/, out/, cache.sqlite or *.npy. CholecSeg8k is not ours to redistribute.

## Rules
- Never print or log GEMINI_API_KEY.
- Only edit files your lane owns. Shared contracts (config.py, gem.py, schemas) belong to the architect: request changes under REQUESTS.
- The README "Honest claims" bind everyone: no human labels in training; calib/oracle results are upper bounds; ground truth only scores the held-out test split.
- Models: gemini-4-argon for reasoning and code, gemini-3.8-flash for bulk labelling.
- Every lane writes tests (pytest, tests/test_<lane>_*.py) for what it builds.

## Status file shape (keep it short, update at every milestone)
```
STATUS: running | blocked | done
ARTIFACTS: paths, with counts/shapes
NUMBERS: metric = value (how computed, which split)
BLOCKED ON: exact thing, or none
REQUESTS: changes needed from other lanes, or none
```
"done" means the artifact exists and you checked it, not that a script exited 0.

