# Cross-machine coordination (Nir's agents <-> Harald's agents)

Local status files (out/lanes/*.md) never leave a machine. Anything another machine's agent must know goes here, in git.

## Notes: one file per agent, append-only
- Your file: coord/notes/<machine>-<lane>.md, e.g. nir-label.md, harald-demo.md. Write ONLY to your own file, so there are never merge conflicts.
- Append entries at the bottom, newest last:
  `## 2026-10-31 13:05 [nir-label -> harald-demo] <one line>`
  then 1-5 lines of detail: what changed, file/schema, commit sha.
- Addressing: `-> all`, `-> <machine>-<lane>`, or `-> nir` / `-> harald` for the humans.
- Read: at every milestone, `git fetch origin && git log origin/main -p --since="2 hours ago" -- coord/notes/` and act on anything addressed to you.
- Get your notes onto main: lane agents commit notes on their branch; the architect (Nir's machine) merges to main every ~20 min. If a note is urgent, say so in the first line and the architect will merge at once.

## Contracts
- docs/DESIGN.md (owned by nir-architect) is the single source of truth for file and schema contracts between layers.
- Small cross-machine artifacts (manual.json, results.json, student.json, decisions.csv) are published to coord/artifacts/ by their owner when a version is good. Never raw/, data/, frames, masks or features (CholecSeg8k is not ours to redistribute).

## Ownership
See coord/OWNERSHIP.md. Do not edit files owned by another machine's lane; leave a note instead.
