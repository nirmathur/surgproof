# Contract: narrator outputs (producer: narrator)

All files under `narrator/out/`. Paths are relative to the repo root. Every row carries `source`
(`gemini` | `mock` | `injected`) and every eval file says whether its inputs were mock. Showcase output
is under `narrator/out/showcase/` and is always `unscored`.

## claims.parquet (one row per atomic claim, both modes)
| column | type | meaning |
|---|---|---|
| claim_id | str | `<mode>-<clip_id>-<n>` |
| clip_id | str | `video12_15750` |
| video_id, split | str | `test` / `tune` |
| mode | str | `free` (baseline) / `grounded` |
| t_start, t_end | int | absolute frame indices (CholecSeg8k N) |
| type | str | presence, absence, action, phase, cvs_criterion, finding, condition, grade, decision_point, knowledge |
| modality | str | observed, knowledge, inference |
| entities | json str | {instrument, verb, target, finding, condition, grade, phase, cvs_criterion, decision}: KB enums or null |
| polarity | str | asserted, negated, hedged |
| evidence_refs | json list | event ids (`ev:<clip>:<n>`) or label keys (`lab:<clip_id>:<frame_idx>:<label>`) |
| kb_refs | json list | KB rule ids |
| confidence | float | 0..1 |
| text | str | the source phrase |
| source | str | gemini / mock / injected |
| L1..L7, P | str | `pass` / `fail` / `abstain` / `` (not run) |
| reasons | json dict | layer -> reason |
| verdict | str | `speak` / `hedge` / `drop` |
| decided_by | json list | layers that decided the verdict |
| gt_correct | bool/null | test split, scorable classes only (eval only; null elsewhere) |

## narration.jsonl (one line per sentence)
`{"clip_id", "t_start", "t_end", "text", "claim_ids": [..], "mode": "free"|"grounded", "verdicts": [..], "source"}`
In free mode every claim is kept so the UI can mark the failed ones red: `verdicts` aligns with `claim_ids`.

## evidence/ (frame crops for the UI)
`narrator/out/evidence/<clip_id>/<frame_idx>.jpg` full frame with label boxes drawn, plus
`<frame_idx>_<label>.jpg` crops. Index: `narrator/out/evidence/index.json`:
`{"<claim_id>": [{"frame_idx", "path", "label", "box": [x1,y1,x2,y2]}], ...}`.
KB quotes per claim: `narrator/out/kb_quotes.json` `{"<rule_id>": {"statement", "quote", "source_id", "url", "quote_verified", "evidence_grade"}}`.

## review.json (game review, showcase only, unscored)
```json
{"video": "...", "scored": false, "label": "deviation from reference policy",
 "difficulty": {"grade": "parkland_2|unknown", "claim_ids": []},
 "nodes": [{"node_id": "cvs_pause", "t": 1234.0, "verdict": "ok|inaccuracy|blunder|missed_alternative|unknown",
            "observed": "...", "guideline_action": "...", "recommended_alternative": "...",
            "evidence_grade": "...", "source_quote": "...", "source_id": "...", "claim_ids": []}]}
```

## eval (narrator/out/eval/)
`free_vs_grounded.csv`, `hallucination.png`, `injection.csv` (error_type, layer, recall, precision, mcc, n),
`injection.png`, `layer_correlation.csv`, `cost_latency.csv`, `RESULTS.md`. Test videos only.

Changes: posted on coordination/BOARD.md as CONTRACT lines before they happen.
