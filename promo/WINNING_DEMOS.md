# Winning Product & Hackathon Demo Patterns (`promo/WINNING_DEMOS.md`)

Compiled by `nir-director` to guide the 45–60s **SurgProof** how-it-works demo (`promo/DESIGN.md`).

---

## 1. Reference Corpus & Links

### Classic Product-Led Launch Films (Apple, Linear, Stripe, Loom, Figma, Arc, Notion AI)
- **Linear Launch & Release Films** ([linear.app/releases](https://linear.app/releases), [linear.app/method](https://linear.app/method)):
  - Near-black `#0A0B0E` canvas, single geometric sans (`Inter` / `SF Pro`), `1px` hairline borders (`#222630`), zero neon glow, one UI focal point per shot, smooth `cubic-bezier(0.22, 1, 0.36, 1)` transitions.
- **Stripe Product Walkthroughs & Sessions Demos** ([stripe.com/sessions](https://stripe.com/sessions), [docs.stripe.com](https://docs.stripe.com)):
  - Show the real artifact (real JSON payload, real webhook response, real dashboard state) side-by-side with the interface action. Never claim a capability abstractly—show the exact input and exact output.
- **Apple Product Design Films** ([apple.com/newsroom](https://www.apple.com/newsroom/)):
  - Quiet confidence. Maximum 5–6 words per title card, generous negative space (>=40% of frame), slow eased motion, zero sci-fi HUD overlays.
- **Loom / Figma / Notion AI Feature Demos** ([figma.com/blog](https://www.figma.com/blog/), [notion.so/product/ai](https://www.notion.so/product/ai)):
  - Immediate product-in-action within 3 seconds. No talking-head intros, no team bios, no slide decks.

### Hackathon Winners & YC Demo Day Guidance
- **Y Combinator Demo Day & Application Video Guide** ([ycombinator.com/library](https://www.ycombinator.com/library)):
  - *"Clarity over polish."* Lead with what the system actually does on real data in the first 5 seconds; end with the single strongest quantitative proof and pause for 2 seconds so the number lands.
- **Google Gemini / Anthropic / Devpost Hackathon Winners (incl. LabTwin & Medical AI Winners)** ([devpost.com/software](https://devpost.com/software), [ai.google.dev/competition](https://ai.google.dev/competition)):
  - Exhausted judges score dozens of submissions in a row. Winners avoid synthetic "AI trailer" stock clips and instead show a **deterministic, end-to-end causal chain on a single real domain artifact**:
    1. Raw unlabelled input $\rightarrow$
    2. Self-authored reference criteria $\rightarrow$
    3. Automated prediction $\rightarrow$
    4. Self-verification filter (what passes, what fails, what escalates) $\rightarrow$
    5. Minimal human gate $\rightarrow$
    6. Held-out quantitative result.

---

## 2. Six Extracted Patterns & How We Apply Them in `promo/DESIGN.md`

### Pattern 1: The First 5–10 Seconds (Immediate Reality, Zero Preamble)
- **Why winners win**: Losers spend 0:00–0:15 on logos, team bios, sci-fi B-roll, or multi-sentence problem statements. Winners drop the viewer directly onto the raw, unlabelled domain artifact in second 1.
- **How we apply it (Shot 1, `0:00–0:07`)**:
  - Open directly on a real laparoscopic cholecystectomy frame (`video12_15750_15758` from `data/index.csv`) on a quiet `#0A0B0E` stage.
  - Caption (4 words): **`Surgical video. No labels.`**

### Pattern 2: Show, Don't Tell (Real JSON & Real UI Over Abstract Metaphors)
- **Why winners win**: Showing a glowing neural network or laser prism tells a judge nothing. Showing 3 real lines of `out/manual.json` next to the public surgical video source proves the agent actually wrote usable anatomical rules.
- **How we apply it (Shot 2, `0:07–0:15` & Shot 3, `0:15–0:23`)**:
  - **Shot 2**: Real `out/manual.json` rules for `grasper`, `l_hook`, and `cystic_duct` (or the `1 — Med School` tab of `app.py`) appear cleanly. Caption (5 words): **`Writes its own visual manual.`**
  - **Shot 3**: Real bounding boxes from `out/labels_manual.jsonl` lock onto `video12_15750_15758` in calm neutral blue (`#3B82F6`). Caption (4 words): **`Labels a real frame.`**

### Pattern 3: One Hero Moment (The Self-Check Turn)
- **Why winners win**: Every great 60-second demo has one unforgettable visual state change that embodies the core technical insight. For **SurgProof**, that insight is *autonomous self-verification without human labels*.
- **How we apply it (Shot 4, `0:23–0:32`)**:
  - On the exact same frame (`video12_15750_15758`), the neutral blue labels self-audit into three semantic verdicts:
    - **Green (`#10B981`, accept)**: `grasper`, `l_hook`, `gallbladder`
    - **Amber (`#F59E0B`, unsure)**: `liver` (`trust = 0.49`)
    - **Red (`#EF4444`, reject)**: `cystic_duct` (`trust = 0.29`, dimmed out)
  - Caption (5 words): **`Checks itself. Green, amber, red.`**

### Pattern 4: Human-in-Control With Minimal Effort (The Gate Beat)
- **Why winners win**: Judges in clinical/safety-critical tracks distrust black-box automation, yet hate manual drudgery. Showing that the human only touches the small **Amber** queue proves both safety and 10× leverage.
- **How we apply it (Shot 5, `0:32–0:40`)**:
  - Show the real Human Gate (`pages/2_Human_gate.py` / `app.py` gate view) with `1 amber remaining`, clearing that single item from Amber to Green (`0 amber remaining`).
  - Caption (5 words): **`Human clears only amber cases.`**

### Pattern 5: The Single-Number Proof Beat (The YC 2-Second Pause)
- **Why winners win**: Four crowded metric cards dilute impact; fake numbers disqualify immediately. One large, honest number read directly from `out/results.json`—held on screen for 6 seconds in silence—projects total integrity.
- **How we apply it (Shot 6, `0:40–0:46`)**:
  - Reads `out/results.json` dynamically. Shows one large number alone on screen (or exact text **`Result: pending`** if `out/results.json` has not landed yet, with zero filler).

### Pattern 6: Restrained Pacing & Crisp Ending
- **Why winners win**: 6–9 seconds per shot gives the eye time to read the image and the 4–6 word caption without rushing. End on the thesis statement, then stop cleanly.
- **How we apply it (Shot 7, `0:46–0:52`)**:
  - Pure `#0A0B0E` background, `SF Pro` type: **`SurgProof. No human labels. Only proof.`** (6 words).
