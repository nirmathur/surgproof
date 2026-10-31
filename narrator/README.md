# narrator: verified narration for laparoscopic cholecystectomy

**Thesis.** Free VLM narration of surgery hallucinates. This module makes Gemini emit *atomic claims*,
checks each one with separate verifier layers, and only says what it can back with frames, verified
labels and guideline sources. We measure the difference on held-out test videos.

**Framing.** This is a post-hoc review of recorded video for teaching and documentation. It is not intraoperative guidance and not diagnosis.
It never predicts bile duct injury and never judges surgeon quality. The game review reports
"deviation from reference policy", not outcome regret.

## Pipeline (each stage writes usable output on its own)
| stage | command | output |
|---|---|---|
| data | `python -m narrator.data` | `out/gt_presence.parquet` (eval only), `out/mock_labels.parquet` (until real labels) |
| KB | `python -m narrator.kb build` | `kb/*.yaml` (Layer 1 rules), `kb/passages.parquet` (Layer 2), `out/kb_report.json` |
| events | `python -m narrator.events` | `out/events.jsonl` (tool enter/exit/moving/still, anatomy confirmed/uncertain/rejected, blood, smoke hint) |
| claims | `python -m narrator.claims --mode both` | `out/claims_free.jsonl`, `out/claims_grounded.jsonl` |
| check | `python -m narrator.check` | `out/claims.parquet` (all layer results + verdict speak/hedge/drop) |
| render | `python -m narrator.render` | `out/narration.jsonl` |
| eval | `python -m narrator.eval` | `out/eval/*.csv, *.png, RESULTS.md` |
| review | `python -m narrator.review` | `out/review.json` (showcase only, unscored) |
| preview | `python -m narrator.demo_preview --clip video12_15750` | `out/demo_preview.html` |

`make all` in `narrator/` runs everything. Every LLM call goes through `narrator/llm.py`, which handles the disk cache
(`narrator/cache/`), the concurrency semaphore, retries, cost counting with a hard budget cap, and JSON-schema validation with one re-ask.
Reruns are free. The key comes from `ANTIGRAVITY_API_KEY`. Models are in `config.yaml`
(fast `gemini-3.8-flash`, judge `gemini-4-argon`, embed `gemini-embedding-2`, tts `gemini-3.8-flash-tts`).

## Checker layers (`check/`), each pass / fail / abstain with a reason
- **L1 schema**: entities in the KB vocabulary; type/entity/modality consistent.
- **L2 rules**: valid instrument-verb-target; phase transitions; tool fits phase; no "CVS achieved" after clipping; grade needs its findings; not-inferable conditions blocked; no Tokyo grading.
- **L3 temporal**: no unexplained contradictions; no single-frame observations without an event.
- **L4 evidence**: observed claims vs the verified label table + tracker (uncertain label or high tracker suspicion -> hedge).
- **L5 blind visual**: a fresh `fast_model` call sees only the frame or crop and a yes/no question derived from the claim, never the narration.
- **L6 perturbation**: the same question on a blacked-out or shuffled frame. An unchanged "yes" means the answer came from priors.
- **L7 knowledge**: retrieve KB passages; `judge_model` says supported, contradicted or not_found and quotes a span, and code verifies the span exists.
- **P proportionality**: the most conservative grade/condition the confirmed findings support.
Verdict: any fail -> `drop`. Otherwise a hedge or uncertain evidence -> `hedge`. An evidence layer passed -> `speak`. Nothing backed it -> `drop`.

## Schemas
See `coordination/contracts/narrator_outputs.md` (claims.parquet, narration.jsonl, evidence/, review.json).

## Limitations (read before quoting numbers)
- Until the labeller's verified table exists, L4 and the event timeline run on a **mock** label table built from the
  ground-truth masks plus seeded noise (`source=mock`). Checked-mode numbers on mock inputs are therefore **not** a real
  result. They show the plumbing, and they would be optimistic. Free and grounded *raw* numbers do not depend on the mock.
- Only presence/absence of grasper, L-hook, liver, gallbladder, cystic duct and blood can be scored on CholecSeg8k.
  There is no cystic artery or clipper class, and the clips are 80 frames, so phase, CVS, grade and game-review claims are **unscored**.
- Cholec80 videos are likely in Gemini's training data, so scores may be inflated.
- Injected-error results measure detection of *synthetic* errors. Real errors are harder.
- Same-family judges (L5, L7 are Gemini) make correlated errors; `eval/layer_correlation.csv` measures how much.

## Licences
CholecSeg8k: CC BY-NC-SA 4.0, not redistributed (data stays outside git). Cholec80 / CholecT50: non-commercial research licences.
KB quotes are short (<15 words) with source URLs.

## Results
See `out/eval/RESULTS.md` (numbers, n, which inputs were mock, and the bad results). Real-input smoke run: `out/real/`
(`NARRATOR_LABELS=<decisions.csv> NARRATOR_OUT=narrator/out/real`).
