# Harald: getting your agents running

1. Accept the GitHub invite, then `git clone https://github.com/nirmathur/surgatlas && cd surgatlas`.
2. Read README.md, GEMINI.md (rules for every agent), coord/README.md (how notes work) and coord/OWNERSHIP.md (your lanes: demo, narrate, atlas, eval).
3. One git worktree per lane: `git worktree add -b harald/<lane> ../surgatlas-<lane> main`.
4. Start one Antigravity agent per lane in its worktree:
   `agy --model gemini-4-argon-high --dangerously-skip-permissions -i "You are harald-<lane>. Read GEMINI.md, coord/README.md and coord/OWNERSHIP.md. Your notes file is coord/notes/harald-<lane>.md. Start Phase A."`
5. You don't need the dataset. Build against docs/DESIGN.md and coord/artifacts/. Until real artifacts land there, use clearly labelled mock data (never fake numbers in the demo).
6. Push harald/<lane>. Leave a note for nir-architect when a branch is ready to merge.
