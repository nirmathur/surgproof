# Autonomous Coding Agents & Multi-Agent Team Best Practices

**Author:** `nir-coach`  
**Date:** 2026-10-31  
**Scope:** Actionable guidance for autonomous coding agents and multi-agent teams under tight deadlines (Antigravity / Gemini 4 Argon / Flash workers + orchestrator), synthesizing official Google Antigravity (`agy`), Google DeepMind / ADK, and Anthropic agentic coding documentation.

---

## 1. Executive Summary: The 7 Rules That Win Time-Boxed Multi-Agent Sprints

1. **Zero Idle Waiting (Mock-First Unblocking):** Never sit blocked waiting on an upstream artifact (`out/manual.json`, `out/labels_manual.jsonl`, `out/decisions.csv`). Immediately synthesize a schema-compliant mock or dry-run on an existing baseline (`out/labels_blank.jsonl`), verify the downstream code path end-to-end, and time wall-clock latency so the real artifact drop is a 1-command rerun.
2. **Use Subagents Only for Data-Parallel Sharding or Fresh-Context Verification:** Avoid "subagent sprawl" for sequential logic. Spawn subagents (`invoke_subagent`) when:
   - **Data-parallel batch runs:** Sharding a 1,010-frame (6,060-row) Gemini API labelling run across 4 video-sharded subagents sharing WAL-mode `cache.sqlite` and writing disjoint JSONL shard files (`out/shards/labels_part{1..4}.jsonl`).
   - **Concurrent prep + test:** Running a background evaluation/pytest/render subagent while editing or preparing the next step.
   - **Fresh-context verification:** Auditing data leakage or contract compliance without self-confirmation bias.
3. **Reactive Wakeup Over Polling Loops:** In Antigravity (`agy`), background tasks (`run_command` with `WaitMsBeforeAsync`) and subagents (`invoke_subagent`) notify the parent agent reactively upon completion. Never waste turns or shell loops (`sleep` polling) waiting for background processes unless a periodic coaching/orchestration timer (`schedule`) is explicitly required.
4. **Worktree Isolation + Small, Test-Gated Commits:** Every lane works in an isolated Git worktree (`~/surgatlas-lanes/<lane>` on branch `lane/<lane>`) sharing heavy data/output directories via symlinks (`raw/`, `data/`, `out/`, `cache.sqlite`). Commit every working increment with an imperative message only after `pytest` passes, and `git push` immediately so the architect can merge continuously.
5. **Hard Verification Gates ("Done = Inspected Artifact, Not Exit Code 0"):** Unit tests alone miss integration and data-leakage bugs. Every pipeline stage requires:
   - **Layer 1 (Contract & AST tests):** `pytest` verifying schema keys and zero ground-truth leakage (`gt_*`, `truth`).
   - **Layer 2 (Artifact inspection):** Checking row counts, NaN rates, class distributions, and wall-clock runtime on real outputs before marking a status `done`.
   - **Layer 3 (Independent audit):** Cross-lane verification (`eval/audit.py`, `eval/score_decisions.py`) joining ground truth strictly post-decision on held-out test splits.
6. **Tiered Model Routing + Shared SQLite WAL Caching:** Use frontier reasoning models (`gemini-4-argon`) for code synthesis, architecture, ambiguous `AMBER` escalations, and verification audits; use high-throughput models (`gemini-3.8-flash`) for bulk frame labelling. Protect shared SQLite caches with `PRAGMA journal_mode=WAL;` and `PRAGMA busy_timeout=30000;` plus per-event-loop semaphores.
7. **Structured, High-Signal Coordination:** Multi-agent teams fail from silent stalls and vague status updates. Every worker updates its machine-readable status file (`out/lanes/<lane>.md`) and posts structured diffs (`DONE | NEXT | BLOCKED | NEED: <lane> -> <exact action>`) on a shared bulletin every 10–15 minutes and at every milestone.

---

## 2. Google Antigravity (`agy`) & Gemini Agent Architecture

### 2.1 Context Engineering: Rules & Progressive Disclosure
- **Hierarchical Rules (`GEMINI.md`, `AGENTS.md`):** Keep non-negotiable operational constraints (branch topology, deadline freeze times, leakage bans, status file schemas) in root `GEMINI.md`. Antigravity deduplicates rules by resolved file path and injects `always_on` rules into every turn.
- **Progressive Disclosure via Skills (`skills/<name>/SKILL.md`):** Do not bloat system prompts (>3,000 tokens degrades instruction following). Store specialized runbooks in `SKILL.md` files; Antigravity injects only skill names/descriptions by default and loads full instructions on demand.

### 2.2 Subagents, Workspaces, and Reactive Execution
- **`invoke_subagent` Workspace Modes:**
  - `inherit` (default): Shares the parent's working directory. Use when subagents write to disjoint output paths (e.g., shard files `out/shards/part_0.jsonl`..`part_3.jsonl`) or perform read-only verification (`research` subagent).
  - `branch` / `share`: Creates an isolated workspace or shared git worktree when subagents need to modify tracked source files independently without colliding.
- **Event-Driven Completion (No Polling):** Antigravity automatically resumes the agent turn when a subagent sends a message or a background task exits. Do not run bash `sleep` loops to poll subagents.
- **Tiered Model Selection:** Default subagents to `inherit`, or explicitly route lightweight parallel extraction/summarization tasks to `flash` / `flash_lite` while reserving `pro` / `gemini-4-argon` for complex debugging and verification.

---

## 3. Anthropic & Google Guidance on Multi-Agent Coding Workflows

### 3.1 When to Use Subagents vs. Single-Loop Execution ("Folding")
Recent engineering guidance from Anthropic (*Building Effective Agents*, *Claude Code Best Practices*, and *Multi-Agent Research System*) and Google (*Agent Development Kit / ADK*) converges on a critical distinction:
- **Avoid Subagent Sprawl for Sequential Code Edits:** Splitting tightly coupled code changes across multiple subagents introduces context fragmentation, interface mismatches, and coordination latency. Keep sequential code-edit-test loops inside a single owning lane.
- **Use Subagents Aggressively for Embarrassingly Parallel I/O & Compute:**
  - **Sharded API Batching:** When running 1,010 frames (6,060 binary decisions) through `label.py`, a single serial loop risks missing the submission window. Splitting the index by video IDs across 4 concurrent subagents cuts wall-clock time by ~3.5x while hitting the same WAL-enabled `cache.sqlite`.
  - **Parallel Asset Generation:** Pre-rendering independent video shots (`shot_01`..`shot_04`, `shot_06` in `promo/`) or extracting external dataset features (`Endoscapes2023` vs. `CholecT50` in `data/`) via subagents keeps the main agent free to wire tests and integration contracts.
  - **Independent Critique / Verification Subagents:** A subagent with a clean context window reviewing `accept.py` or `student.py` against `docs/DESIGN.md` catches subtle leakage bugs that the authoring agent overlooks.

### 3.2 Test-First Development & The "Implement–Verify–Fix" Loop
- **Write Contract Tests Before or Alongside Implementation:** Define fast `pytest` suites (`tests/test_<lane>_*.py`) that check:
  1. **Schema invariants:** Required JSON/CSV columns, valid enum values (`status in {'accept', 'AMBER', 'reject'}`).
  2. **AST / Static Leakage Guards:** Parsing source files with Python's `ast` module to assert that inference and verification modules (`label.py`, `accept.py`) never reference ground-truth columns (`gt_*`, `truth`).
  3. **Edge-Case & NaN Guards:** Empty bounding boxes, out-of-bounds crop coordinates, missing optional keys (`gt=-1` on external datasets), and zero-positive test splits.
- **Beyond Unit Tests — Smoke Tests Before Bulk Runs:** Always run a bounded smoke test (`python label.py --limit 40`) and inspect the output file before launching a 6,060-row bulk run. Measure and log throughput (`rows/min` and `ETA`) from the smoke test.

### 3.3 Avoiding Idle Waiting on Cross-Lane Dependencies
In a multi-lane dependency graph (`data` -> `manual` -> `label` -> `accept` -> `student` -> `demo`/`promo`), upstream delays cascade unless downstream lanes decouple **mechanism verification** from **final data arrival**:
- **Pattern 1 — Baseline / Mock Substitution:** If `student.py` is waiting on `out/decisions.csv` from the `manual` label run, run `accept.py` and `student.py` immediately on `out/labels_blank.jsonl` (or a synthetic `decisions.csv` in scratch) to verify feature alignment, bootstrap CIs, and `out/student.json` serialization.
- **Pattern 2 — Dry-Run Timing:** If `demo` is waiting on an updated `out/manual.json` (`sources[]` field) to record a 20-frame live run, run and time the 20-frame live pipeline using the current `out/manual.json` into a scratch path first. Fix any latency bottlenecks or UI glitches *before* the upstream file arrives.
- **Pattern 3 — Time-Boxed Fallback Triggers:** Set hard cutoffs (e.g., if `out/manual.json` `sources[]` update isn't merged by 13:00, launch `label.py` immediately with existing visual criteria — because `sources[]` metadata does not alter `do_frame` visual prompts, meaning `cache.sqlite` entries remain 100% reusable!).

### 3.4 Git Worktree Hygiene & Continuous Integration
- **One Lane = One Worktree = One Branch:** Isolate parallel agents in `~/surgatlas-lanes/<lane>` (`lane/<lane>`) to prevent index lock contention (`.git/index.lock`) and dirty working tree collisions.
- **Small, Atomic Commits:** Commit immediately after each passing test (`git commit -m "Add ..."` && `git push -u origin lane/<lane>`).
- **Frequent Mainline Sync:** The `architect` lane runs a 10–15 minute fetch-test-merge loop into `main`. Worker lanes pull merged contracts via `git merge main` (never rebase or force-push shared branches).

---

## 4. Concrete Coaching Checklist for SurgProof (12:55 -> 15:15 Submission)

| Risk / Anti-Pattern | Detection Signal in `#1` / `out/lanes/*.md` / `git log` | Coaching Intervention (`[nir-coach] -> nir-<lane>`) |
|---|---|---|
| **Upstream Wait Stall** | Lane marks `BLOCKED ON: waiting for <other-lane>` with no commits for >10 min | Instruct lane to run its pipeline on existing `labels_blank.jsonl` / scratch mock or exploit cache invariance (`cache.sqlite` keys on prompt text, not unused metadata). |
| **Serial Bottleneck on Bulk API Run** | Single process running 6,060 rows sequentially (<150 rows/min risks missing 14:00) | Instruct `nir-label` to shard by video across 4 subagents (`--video` or sharded index CSVs) sharing `cache.sqlite` (WAL). |
| **Silent Worker (>12–15 min)** | No `#1` comment or `out/lanes/<lane>.md` mtime update in >12 min | Flag immediately on `#1` with exact recovery task so orchestrator or lane resumes without losing context. |
| **Unverified Claim ("Script Exited 0")** | Status says `done` without row counts, shapes, or test split metrics in `NUMBERS:` | Require exact artifact verification command (`wc -l`, `jq`, `pytest`, metric value + split name). |
| **Duplicate Implementation** | Two lanes editing overlapping logic (e.g. `accept.py` on `main` vs `lane/label`) | Direct worker lane to `git merge main` and reconcile against architect's merged contract tests before pushing further changes. |
| **Feature Freeze Drift (14:30)** | Lanes adding new features after 14:15 instead of locking `results.json`, `student.json`, and demo/promo recordings | Shift all lanes at 14:15 to end-to-end artifact verification, UI wiring check, and final video/demo capture. |

---

## 5. Sources & References

1. **Google Antigravity (`agy`) Official Documentation:**
   - Main Documentation & Architecture: <https://antigravity.google/docs>
   - CLI Features & Subagents: <https://antigravity.google/docs/cli/features>
   - CLI Best Practices: <https://antigravity.google/docs/cli/best-practices>
   - Rules & Workflows (`GEMINI.md` / `AGENTS.md`): <https://antigravity.google/docs/rules-workflows>
   - Progressive Disclosure & Skills (`SKILL.md`): <https://antigravity.google/docs/skills>
   - Terminal Sandbox & Permissions: <https://antigravity.google/docs/sandbox> and <https://antigravity.google/docs/permissions>
   - Antigravity Python SDK: <https://github.com/google-antigravity/antigravity-sdk-python>
2. **Anthropic Engineering & Agentic Coding Guidance:**
   - *Claude Code: Best Practices for Agentic Coding* (test-first workflows, `CLAUDE.md`/`AGENTS.md`, git worktrees, verification gates, subagent verification): <https://www.anthropic.com/engineering/claude-code-best-practices>
   - *Building Effective Agents* (workflows vs. agents, orchestrator-workers, evaluator-optimizer loops, avoiding over-engineering): <https://www.anthropic.com/research/building-effective-agents>
   - *How We Built Our Multi-Agent Research System* (parallel subagent orchestration, token economics, artifact handoffs, avoiding synchronous blocking): <https://www.anthropic.com/engineering/multi-agent-research-system>
   - *Claude Code Subagents Documentation* (context isolation, parallel execution, fresh-context verification): <https://docs.anthropic.com/en/docs/claude-code/sub-agents>
3. **Google DeepMind & Google Cloud Agentic Systems:**
   - *Google Agent Development Kit (ADK) Documentation — Multi-Agent Systems & Orchestration Patterns*: <https://google.github.io/adk-docs/agents/multi-agent-systems/>
   - *Gemini API Documentation — Thinking & Agentic Workflows*: <https://ai.google.dev/gemini-api/docs/thinking>
   - *Vertex AI Agent Builder & Agent Engine Architecture*: <https://cloud.google.com/vertex-ai/generative-ai/docs/agent-engine/overview>
