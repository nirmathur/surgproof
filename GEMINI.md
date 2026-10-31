# SurgAtlas: rules for every Gemini worker

Six Antigravity (agy) workers on Gemini 4 Argon build this project in parallel. Claude is the orchestrator and reads only your status file. The existing code is a first draft: every layer is assumed imperfect until your research and redesign say otherwise.

## Layout
- Each lane has its own git worktree at ~/surgatlas-lanes/<lane>, on branch lane/<lane>. The architect lane works in ~/surgatlas on main and is the only one that merges into main.
- raw/, data/, out/ and cache.sqlite are symlinks to the shared copies in ~/surgatlas. All lanes see the same data and outputs.
- Status files: out/lanes/<lane>.md (shared). Read other lanes' status files to see what exists.

## Phases
A. Research and redesign (no feature code yet). Audit the current code for your layer: what is wrong, fragile or unjustified. Run deep research on the state of the art for your layer (use tools/deep_research.py once the architect has committed it; until then use your web search). Write docs/redesign/<lane>.md: problems found, evidence with links, proposed design, and the exact interface (input/output files, schemas, CLI). Commit it and push your branch.
B. Build. Once docs/DESIGN.md exists on main, run `git merge main` and build to it. If DESIGN.md is wrong for your layer, say why in your status file under REQUESTS rather than ignoring it.

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
PHASE: A | B
STATUS: running | blocked | done
ARTIFACTS: paths, with counts/shapes
NUMBERS: metric = value (how computed, which split)
BLOCKED ON: exact thing, or none
REQUESTS: changes needed from other lanes, or none
```
"done" means the artifact exists and you checked it, not that a script exited 0.

## Cross-machine notes (Harald's agents work on another machine)
Read coord/README.md and coord/OWNERSHIP.md. Your notes file is coord/notes/nir-<lane>.md (Harald's agents use harald-<lane>). Append-only, your own file only. Check for notes addressed to you at every milestone.
