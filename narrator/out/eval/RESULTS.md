# Narrator results (test videos only: video09, 12, 17, 28, 48, 55)

Generated 2026-10-31 ~14:35 from cache with `make all`. Generator: gemini-3.8-flash. Judge: gemini-4-argon.
**Input caveat:** the main run uses a MOCK labeller table (CholecSeg8k GT masks plus seeded noise: 8% drop,
4% spurious, 4% swap, 10% uncertain) and tracker flags computed on corrupted GT. Any number that depends on L4
or on the grounded prompt's label table is therefore not a real result. Each row below says which inputs it used.

## 1. Free vs grounded, scored against the masks (n = 28 clips per mode)
Only presence/absence of grasper, L-hook, liver, gallbladder, cystic duct and blood is scorable. Precision and
hallucination count **asserted** claims. Hedged claims are scored by hedge correctness.

| condition | asserted scored claims | precision | hallucinated structures / clip | coverage | inputs |
|---|---|---|---|---|---|
| free raw | 122 | 0.951 | 0.071 | 0.765 | frames only |
| free + checker without L4 | 68 | 0.985 | 0.000 | 0.605 | GT-independent |
| grounded raw | 105 | 0.876 | 0.429 | 0.988 | **mock** labels in prompt |
| grounded + checker without L4 | 86 | **1.000** | **0.000** | 0.914 | rules + L5 + L6 + L7 (no GT) |
| grounded + full checker | 84 | 1.000 | 0.000 | 0.901 | **mock** L4: circular |

- On mask-scorable classes, free narration rarely hallucinates (1.8% of positive claims). **The masks cannot show the problem the thesis is about.**
- Raw grounded is *worse* than free (11.5% vs 1.8%) because it faithfully repeats the mock table's injected spurious labels. That is garbage in, garbage out, and it shows that grounding alone is not verification.
- The checker without L4 (independent of GT) removes every hallucinated structure from the grounded output (0.43 per clip, 12 clip-structure pairs, down to 0) at a coverage cost of 0.988 → 0.914.
- Rows using L4 on mock labels are circular (L4 checks against a GT-derived table) and are shown only as plumbing.

## 1b. Safety-critical claims (not mask-scorable; checker effect only)
Free-mode asserted claims about CVS, grades, conditions, clip/cut actions and decision points: **60**.
Checker verdicts: **44 dropped, 12 hedged, 4 spoken**. All 7 "CVS criterion met" claims and all 6 grade claims were dropped,
e.g. "confirming that the critical view of safety (CVS) was already achieved" (video12_15750).
We cannot say these were *wrong*: no GT exists. We can say nothing independent backed them.
Grounded mode made 0 asserted CVS or grade claims.

## 1c. Real inputs (P1.1): verifier labels `coord/artifacts/decisions.csv` @6ed73f1 (n = 6 test clips, 90 label rows)
free raw precision 0.962 (26 claims), grounded raw 1.000 (20), grounded checked 1.000 (14 asserted), coverage 0.938 → 0.812.
**n is tiny: treat as a smoke test, not a result.** Notable case: on video12_15750 the real verifier marked the cystic duct
AMBER. L5 (Gemini) *and* L5c (CLIP) both wrongly said it was visible. The claim was spoken only as a hedge because of the AMBER
status. That is one example of independent evidence catching a same-family error.

## 2. Injected wrong claims (164 errors of 9 types on 169 GT-correct claims)
51 injected claims that GT showed to be accidentally true (e.g. grasper→hook while a hook was in view) were excluded.

| layer | recall | precision | MCC | clean claims falsely dropped |
|---|---|---|---|---|
| L1 schema | 0.006 | 1.00 | 0.06 | 0% |
| L2 rules | 0.640 | 1.00 | 0.69 | 0% |
| L3 temporal | 0.006 | 0.33 | -0.03 | 1.2% |
| L4 evidence (**mock**) | 0.165 | 0.55 | 0.05 | 13.0% (mock noise) |
| P proportionality | 0.293 | 1.00 | 0.42 | 0% |
| L5 blind visual | 0.244 | 0.85 | 0.29 | 4.1% |
| L6 perturbation | 0.000 | n/a | 0.00 | 0% |
| L7 knowledge | 0.585 | 1.00 | 0.65 | 0% |
| L5c OpenCLIP (report-only) | 0.085 | 0.20 | -0.31 | 33.7% |
| **full stack** | **0.988** | **0.853** | **0.830** | **16.6%** |

Recall per error type, full stack: cvs_after_clip, fabricated_knowledge, grade_escalation, impossible_combination,
not_inferable_condition and phase_order_violation all 1.00 (n = 24, 24, 24, 9, 24, 24). absent_structure 0.96 (n = 24),
entity_swap 0.90 (n = 10), shifted_timestamp 1.00 (n = 1, meaningless).
- These are synthetic errors. Rule-detectable types are easy by construction, and real errors will be harder.
- The full stack drops 1 in 6 correct claims. Most of that comes from L4 on mock noise (22 of 28). On real labels it depends on the verifier's error rate.
- L3 and L1 contribute almost nothing to recall here. L6 never fired: no L5 "yes" survived a blacked-out frame (0/120 sampled), so Flash is not answering from priors on these questions.
- Per-type precision in `injection.csv` pools all clean false alarms into every type, so treat it as a lower bound.

## 3. Error correlation between layers (`layer_correlation.csv`)
For each pair on errors both layers judged: observed P(both miss) vs P(A miss)·P(B miss).
- L4 × L5 (label table vs Gemini blind check): ratio 1.60 (n = 32): their misses are positively correlated.
- L5 × L5c (Gemini vs CLIP): ratio 2.0 (n = 22, but CLIP is at chance, so this says little).
- On real Gemini claims that were wrong vs GT (n = 28), every pair is at or below independence, but small n.
- The answer to "Gemini checking Gemini": we see some correlated misses, and our one non-Gemini visual model (zero-shot OpenCLIP)
  scored **52% on 200 decided GT cases = chance**. It is report-only, never decisive. We do **not** yet have a good independent visual verifier.
  A surgical contrastive model (e.g. SurgVLP) is the obvious next step.

## 4. Cost and latency (generation; `cost_latency.csv`)
free: 2.9 calls, $0.020, ~30 s per clip. grounded: 1.3 calls, $0.012, ~30 s per clip (wall clock with 16 concurrent).
Checker over the whole run: L5 $0.96 (619 calls), L6 $0.45, L7 $1.37. About $0.10 per clip all-in. Total API spend so far is under $10.

## KB
39 Layer-1 rules. **15 quote-verified**: the quote appears verbatim in source text fetched directly over HTTP
(Strasberg 2010 PDF, Nassar 2021 PDF, Alius 2023, Yildirim 2022, TG18, arXiv abstracts). 6 matched only Gemini-grounding text, which does
not count. 18 not found. "Verified" means the quote exists verbatim, not that it fully entails the rule (e.g. the R4U rule's quote is weak).
SAGES and StatPearls blocked direct fetch (403 / captcha), so rules citing them stay unverified and are hedged.

## Showcase (unscored): https://www.youtube.com/watch?v=PuC7iJffTpw
Gemini accepts YouTube URLs. 5 phases, 24 claims: 19 hedged (no labels or frames, so nothing independent backs them), 5 dropped.
Game review (`showcase/review.json`, "deviation from reference policy"): CVS pause and clip/divide are **unknown**, because only 1 of 3
CVS criteria was confirmed and 2 were never assessed. An earlier version called this a "blunder" from silence; we changed the rule so
that a safety flag needs an explicit "criterion not met" claim. There is no ground truth for any of this.

## Limitations
Cholec80 frames are likely in Gemini's training data. The masks score only presence/absence. The clips are 80 frames, so phase,
CVS and grade claims are unscored. The main run's labels are mock. agy sub-worker quota ran out mid-build, so some layers were written fast.
