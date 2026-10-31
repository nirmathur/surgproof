# Verified labels unlock lap chole video

The value case is strong and backed by numbers. Bile duct injury (BDI) affects about 0.3% to 1.5% of cholecystectomies. In the classic analysis, 97% of injuries came from misreading the anatomy, not from poor technique. Video audits find the critical view of safety (CVS) truly achieved in only 9% to 16% of cases. The largest public CVS-labelled set has just 201 videos. Google's own Med-Gemini scored 55.2% zero-shot on CVS against 67.0% for a small supervised model. Together these make "labels the agent has verified, plus narration grounded in evidence" a direct answer to a gap Gemini already has. The chess framing holds up only as **reference-policy regret**: how far the surgeon departed from guideline actions at about 8 to 10 decision points. True outcome regret cannot be identified from video, because nobody sees what the untaken action would have led to. Most narration claims can be checked against closed vocabularies (Cholec80 phases, CholecT50 triplets, the three Endoscapes CVS criteria) and explicit rules. Parkland grade has no public ground truth, and the best CVS dataset needs a signed data use agreement. The biggest risks are Gemini checking Gemini (correlated errors), noisy ground truth (CVS inter-rater kappa 0.38), non-commercial and restricted licences, possible benchmark contamination, and a tool-motion verifier that no source in the notes supports.

**Flags used below**
- **[secondary]**: figure taken from a later paper citing the original; the primary source was not read.
- **[unverified]**: background knowledge that no fetched source confirmed; check it before using on stage or encoding it as a rule.
- **[inference]**: derived by the researchers or by this report, not taken from a source.
- **[low reliability]**: commercial or methodologically opaque source.

## A 0.3% to 1.5% injury rate caused by misperception makes this a vision problem

### Clinical burden

| Metric | Figure | Source |
|---|---|---|
| US cholecystectomies per year | about 750,000 (older, widely repeated figure) | ([Karger, Dig Surg](https://karger.com/dsu/article/37/1/10/118048/Long-Term-Impact-of-Iatrogenic-Bile-Duct-Injury)) |
| England, 2000 to 2009 | 418,214 over 10 years (about 42,000 per year), 83.3% laparoscopic, major BDI 0.4% | ([Oxford NDS](https://www.nds.ox.ac.uk/publication/382934)) |
| Sweden | more than 13,000 per year for about 10M people | ([PMC7944855](https://pmc.ncbi.nlm.nih.gov/articles/PMC7944855)) |
| BDI incidence | 0.3% to 1.5% including minor injuries; major BDI 0.08% to 0.3% | ([Karger](https://karger.com/dsu/article/37/1/10/118048/Long-Term-Impact-of-Iatrogenic-Bile-Duct-Injury)) |
| Major BDI, recent large series | 0.15% to 0.36% | ([SAGES multi-society guideline](https://www.sages.org/publications/guidelines/safe-cholecystectomy-multi-society-practice-guideline/)) |
| US BDI patients per year | about 2,500 | ([Karger](https://karger.com/dsu/article/37/1/10/118048/Long-Term-Impact-of-Iatrogenic-Bile-Duct-Injury)) |
| Cause of BDI | 97% visual perceptual illusion, 3% technical skill (Way 2003, 252 BDIs) **[secondary]** | ([Mascagni et al. 2026](https://arxiv.org/pdf/2606.30657)) |
| Direct cost per BDI, Sweden | EUR 11,200 (minor, found during surgery) to EUR 42,800 (major, found after surgery), against EUR 4,500 for the base operation | ([PMC7944855](https://pmc.ncbi.nlm.nih.gov/articles/PMC7944855)) |
| Quality of life (EQ-5D) | 0.90 with no BDI vs 0.67 with major BDI found after surgery | ([PMC7944855](https://pmc.ncbi.nlm.nih.gov/articles/PMC7944855)) |
| Long-term BDI-related mortality | 1.8% to 4.6% | ([Karger](https://karger.com/dsu/article/37/1/10/118048/Long-Term-Impact-of-Iatrogenic-Bile-Duct-Injury)) |
| Medicare cohort | nearly 8,000 BDIs among more than 1.5M patients; nearly one third of BDI patients died within a year (elderly cohort, so not representative) | ([UW summary of Flum 2003](https://depts.washington.edu/drrpt/clinicalleadership/gallbladdersurgery.htm)) |
| NHS litigation, 1995 to 2009 | 418 lap chole claims; 65% of settled claims won by the claimant; GBP 20.4M total; BDI the most frequent and most successful claim type | ([Univ. of Manchester](https://research.manchester.ac.uk/en/publications/15-years-of-litigation-following-laparoscopic-cholecystectomy-in-/)) |
| Compensation range | UK GBP 2,500 to 216,000; US USD 628,138 to 2,891,421; 19% to 32% of European BDI patients file claims | ([Karger](https://karger.com/dsu/article/37/1/10/118048/Long-Term-Impact-of-Iatrogenic-Bile-Duct-Injury)) |

- **Why it matters for this build:** if 97% of injuries are misidentification, the failure is in perception and checking. An agent that checks anatomy and CVS before clipping goes after the main cause.
- **Gaps:**
  - No verified 2023 to 2026 annual counts for the US or the NHS.
  - The "over 1 million US" and "1.2 million" figures were not confirmed.
  - No US or NHS cost per BDI in local currency.

### What surgeons write differs from what the video shows

- In audits, video showed CVS correctly achieved in only **9% to 16%** of lap choles (Korndorffer 2020; Mascagni 2021) **[secondary]** ([Mascagni et al. 2026](https://arxiv.org/pdf/2606.30657)).
- In a SAGES abstract, 2 of 10 videos (20%) reached an adequate CVS score, but 4 op notes claimed CVS, including 2 of the 8 inadequate cases ([SAGES abstract](https://sages.org/meetings/annual-meeting/abstracts-archive/how-often-do-surgeons-obtain-the-critical-view-of-safety-during-laparoscopic-cholecystectomy)). The sample is tiny, so treat it as an illustration.
- Video-rated skill predicts outcomes. In bariatric surgery (Birkmeyer, NEJM 2013), the lowest-skill surgeons had 14.5% complications against 5.2% for the highest-skill group. Each video needed ratings from 10 or more peers ([AHRQ PSNet](https://psnet.ahrq.gov/issue/surgical-skill-and-complication-rates-after-bariatric-surgery)). This is bariatric surgery, not lap chole.
- **Product angle [inference]:** a CVS record linked to evidence (timestamped frames, checked criterion by criterion) has quality and medico-legal value, because BDI is the main driver of litigation.

### The labelling gap is 3 to 4 orders of magnitude

- **Public lap chole labels are small:**
  - Endoscapes: 201 videos, 11,090 CVS frames, each labelled by three experts ([arXiv 2312.12429](https://arxiv.org/abs/2312.12429v1)).
  - CholecSeg8k: 8,080 frames.
  - Cholec80: 80 videos.
- **The unlabelled supply is far larger:**
  - Theator alone reported more than 30,000 hours of video ([Healthcare IT Today](https://www.healthcareittoday.com/?p=2193893)).
  - Caresyntax covers more than 3M procedures per year in more than 3,500 ORs ([Business Wire](https://www.businesswire.com/news/home/20240815975998/en/surgical.ai-Backs-Caresyntax-in-$180M-Series-C-Extension-to-Accelerate-Adoption-of-AI-enabled-Surgery)).
- Endoscapes' 201 videos are about 0.03% of one year of US lap choles **[inference]**.
- Researchers name annotation as "the most resource-intensive part" of surgical AI ([Mascagni et al. 2026](https://arxiv.org/pdf/2606.30657)). They also name the scarcity of annotated data as the main limit on progress ([SurgeNet, arXiv 2501.09436](https://arxiv.org/pdf/2501.09436)).
- **Gap:** no sourced figure for expert hours or cost per labelled video hour. Do not quote a cost per label on stage.

### Gemini's own results show the gap this project fills

- **Med-Gemini (Google Research / Google DeepMind / Verily, 2024):** 55.2% zero-shot accuracy on 572 Cholec80-CVS clips, against 67.0% for a supervised ResNet3D ([arXiv 2404.18416](https://arxiv.org/abs/2404.18416)).
- **11-VLM benchmark (2025):**
  - On zero-shot CVS, contrastive models (SurgVLP, OpenCLIP, CLIP, PaliGemma) beat GPT-4o and Gemini by at least 160% on average.
  - GPT-4o reached tool F1 0.34 on HeiChole, against 0.62 for the supervised state of the art.
  - Source: ([arXiv 2504.02799](https://arxiv.org/abs/2504.02799)).
- **HeiCo-VQA study:** on CVS and hard scene-graph questions, GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 scored only marginally above random. Medically tuned models did worse than generalists of the same size ([arXiv 2506.06232](https://arxiv.org/abs/2506.06232)).
- **Video reports:** Gemini-1.5-pro reports on CholecT45 cholecystectomy videos were 71.1% correct ([TU Dresden record](https://fis.tu-dresden.de/portal/en/publications/visionlanguage-models-for-automated-video-analysis-and-documentation-in-laparoscopic-surgery(c939fffb-1c32-491c-b697-16ae1fa17ddc).html)).
- **Pitch line [inference]:** a frontier generalist already looks at this video but falls below a small supervised model. The missing pieces are verified labels and grounding.
- **Caveat:** no evaluation of 2026 frontier Gemini models on lap chole was found. Re-measure on a small labelled slice instead of quoting 2024 numbers as current.

### Market

- **Funding:**
  - Caresyntax raised a USD 180M Series C in 2024 and sells to hospitals, device makers and insurers ([Business Wire](https://www.businesswire.com/news/home/20240815975998/en/surgical.ai-Backs-Caresyntax-in-$180M-Series-C-Extension-to-Accelerate-Adoption-of-AI-enabled-Surgery)).
  - Theator had raised USD 42.5M by 2022 ([TechCrunch](https://techcrunch.com/?p=2357973)).
- **Market size:** surgical video analytics at USD 1.12B in 2026, growing to USD 8.66B by 2036 (22.7% CAGR) **[low reliability]**. This is one paywalled report with no visible method ([Meticulous Research](https://www.meticulousresearch.com/product/surgical-video-analytics-market-6781)).
- **Positioning [inference]:** incumbents' moat is their video libraries. Verified labels on that video are the scarce input, so a self-verifying label engine is a picks-and-shovels play.

## Surgery is a chess game only against a reference policy, not against outcomes

### What is rigorous

- **Formal regret:** in extensive-form games (CFR), regret is the gap between the best fixed strategy in hindsight and the strategies actually played. Per decision point, counterfactual regret needs u(σ|I→a), the payoff of the action **not** taken. That payoff comes from a known game tree and known terminal payoffs ([Zinkevich et al. 2007](https://proceedings.nips.cc/paper_files/paper/2007/file/08d98638c6fcd194a4b1e6992063e944-Paper.pdf)).
- **Chess review works because an engine supplies a dense value function:**
  - Centipawn loss is the engine's best move minus the move played.
  - Modern grading converts this to win-probability lost, with inaccuracy, mistake and blunder bands.
  - ACPL depends on the position: a 100 cp loss in a won position barely matters.
  - Source: ([Chess It Up](https://chessitup.com/blog/centipawn-loss-explained)).
- **Surgery has no engine:**
  - The counterfactual is never observed.
  - Outcomes arrive late (leaks show up days later).
  - The terminal "loss" is rare (major BDI 0.15% to 0.36%) ([SAGES guideline](https://www.sages.org/publications/guidelines/safe-cholecystectomy-multi-society-practice-guideline/)).
  - Case difficulty confounds the picture: hard gallbladders get bail-outs and also get complications.
- **Learning from video is early:**
  - A KCL study framed lap chole as RL on CholecT50 (state = frame embeddings, action = triplet, reward = similarity to the expert's next action).
  - It reported sparse rewards and no outcome data.
  - Every RL variant lost to plain imitation learning (IVT mAP at 10 s: IL 29.2 vs world model + RL 3.1).
  - Source: ([Boels et al. 2025](https://arxiv.org/html/2507.05011v3)).
- **Surgery is not zero-sum.** Nash and CFR equilibrium results do not transfer. What carries over is the per-decision bookkeeping of "regret for not taking action a".
- **No precedent:** no published work defines intraoperative regret in the CFR or online-learning sense, and no paper uses a chess-engine analogy for surgery. The framing is novel, so present it as such.

### Existing move-quality signals that stand in for an engine

| Chess concept | Closest surgical analogue | Evidence |
|---|---|---|
| Engine evaluation of a position | CVS criteria score; Go/No-Go zone overlap | GoNoGoNet: BDI cases had 33.6% more tool-tissue interactions outside the Go zone; 11 BDI videos vs 11 controls, not public ([Khalid et al. 2023](https://www.springermedicine.com/artificial-intelligence/cholecystectomy/use-of-artificial-intelligence-for-decision-support-to-avoid-hig/26029182)) |
| Blunder | Clipping or dividing a tubular structure without confirmed CVS | Way 2003: most BDIs come from a single misidentification **[secondary]** ([Mascagni et al.](https://arxiv.org/pdf/2606.30657)) |
| Wrong move order vs bad execution | OCHRA error modes 1 to 6 (between steps, wrong order) vs 7 to 10 (faulty execution within a step) | 99 lap chole videos: 92 errors (median 1 per case); 0.75 per case at difficulty grade 1 vs 1.78 at grade 4; 75% executional ([Schwab et al. 2020](https://link.springer.com/article/10.1007/s00464-019-06961-1)) |
| Position dependence of ACPL | Normalising by difficulty (Parkland, Nassar) | Same OCHRA difficulty gradient |
| Game annotation by a human | Competency assessment tools (COLOR III CAT: 9 steps, 4 domains; generalisability coefficient 0.883) | ([Mascagni et al.](https://arxiv.org/pdf/2606.30657)) |

### Decision nodes with guideline backing

- **Source caveat:** the Q numbers below come from the SAGES multi-society guideline page ([SAGES](https://www.sages.org/publications/guidelines/safe-cholecystectomy-multi-society-practice-guideline/)). A second researcher could verify only the abstract of the paywalled full text ([Brunt et al. 2020](https://link.springer.com/article/10.1007%2Fs00464-020-07568-7)). Check wording and GRADE strength before encoding them.
- **Q1:** use CVS to identify the cystic duct and artery (expert opinion).
- **Q13:** pause before clipping or transecting to confirm CVS (best practice).
- **Q4:** use intraoperative imaging when anatomy is uncertain (strong recommendation, very low certainty).
- **Q2 / Q9:** prefer subtotal over fundus-first total cholecystectomy when CVS cannot be achieved.
- **Q17:** keep a low threshold to call for help. BDI was 0.14% for experienced surgeons vs 0.47% for less experienced ones.
- **Q12:** no recommendation on conversion to open, so regret at that node is undefined even as a proxy.
- **TG18:** bail out when CVS cannot be reached because of scarring or fibrosis. Dissect above the line from the base of Segment 4 to Rouvière's sulcus ([TG18](https://profiles.wustl.edu/en/publications/tokyo-guidelines-2018-surgical-management-of-acute-cholecystitis-/)).

### Rigorous vs speculative

| Claim | Status |
|---|---|
| CFR regret needs utilities of untaken actions | Rigorous (definition) |
| Outcome regret cannot be identified from surgical video alone | Rigorous enough to state; the underlying facts are sourced |
| Imitation learning beats RL on lap chole next-action prediction today | Rigorous (one study, CholecT50) |
| "Clip without CVS" is a well-defined deviation | Defensible: strongest guideline consensus |
| Reference-policy regret r(s,a) = Q_ref(s, a_ref) − Q_ref(s, a) | Speculative framing; Q_ref from guidelines is ordinal, not a calibrated value |
| "Surgical centipawn": change in a risk proxy such as CVS score or No-Go overlap | Speculative; no validated link to outcomes |
| Win% analogue: P(no major complication given the current state) | Speculative; no public outcome-linked video to fit it |
| Predicting BDI from video | Not credible at hackathon scale; no prospectively validated model found |

### Minimal credible version **[inference: design]**

- **Decision tree:** 8 to 10 nodes:
  1. Difficulty grade
  2. Exposure
  3. Dissection plane relative to the R4U line
  4. CVS check / pause
  5. If CVS is not reached: imaging, call for help, or bail-out
  6. Choice of bail-out
  7. Clip and divide
  8. Gallbladder bed and haemostasis
- **Mapping video to nodes:** Cholec80-style phases, then CholecT50 `clip` verb events, then CVS criteria in the window before each clip.
- **Scoring per node:**
  - 0 if the guideline action was taken.
  - "Inaccuracy" if 2 of 3 CVS criteria were met at the clip.
  - "Blunder" if the clip came with 1 or fewer criteria, or with a tool in the No-Go zone.
  - "Missed alternative" if CVS was not reached by time T and no imaging, help or bail-out followed.
- **Display:**
  - A game-review timeline.
  - Each flag shows the recommended alternative and its evidence grade. Show "expert opinion / very low certainty" plainly.
  - Label it "deviation from reference policy", never "outcome regret".
- **Guardrails:**
  - Condition on difficulty grade.
  - Blind the evaluation to outcome to avoid hindsight bias.
  - Expect many flags: with CVS achieved in only 9% to 16% of real cases, a binary detector would flag most of them. Grade the score instead.
  - Do not train on outcomes without difficulty adjustment, or the model will learn "subtotal = bad".

## Closed vocabularies and guideline rules make most narration claims checkable

### Knowledge-base rules

| Rule | Basis | Status |
|---|---|---|
| Phase order P1 Preparation → P2 Calot triangle dissection → P3 Clipping and cutting → P4 Gallbladder dissection → {P5 Packaging, P6 Cleaning and coagulation in either order; P6 optional} → P7 Retraction | ([EndoNet](https://arxiv.org/pdf/1602.03012)) | Verified |
| P2 is usually the longest phase (954±538 s; P3 168±152 s), which bounds plausible timing | ([EndoNet](https://arxiv.org/pdf/1602.03012)) | Verified |
| A tool counts as present only if at least half its tip is visible; 7 tools | ([EndoNet](https://arxiv.org/pdf/1602.03012)) | Verified |
| Tool-phase expectations: clipper in P3, hook in P2/P4, bag in P5 to P7, irrigator/bipolar in P6 | Derived from the phase definitions | [inference] |
| Triplets must fall within the CholecT50 vocabulary: 6 instruments, 10 verbs, 15 targets, 100 classes (94 valid for ranking) | ([Nwoye et al.](https://discovery.ucl.ac.uk/id/eprint/10168171/1/2204.04746%20(1).pdf); [FAQ](https://cholectriplet2021.grand-challenge.org/FAQ/)) | Counts verified; full target list and 100-class list [unverified], pull the `dict/` files |
| Check CVS one criterion at a time: triangle cleared, lower cystic plate exposed, only two structures entering; "two windows" alone is not CVS; prefer anterior plus posterior views over a window | ([Strasberg & Brunt 2010](https://drcesarramirez.com/joomla/files/cirugiaMBE/Critical_View_-_Strasberg.pdf)) | Verified |
| A "CVS achieved" claim made after clipping and cutting is temporally inconsistent | Derived from the CVS workflow | [inference] |
| Assign Parkland grade from the pre-dissection view only (ICC 0.804 across 11 surgeons) | ([Madni 2018](https://www.em-consulte.com/article/1204381/the-parkland-grading-scale-for-cholecystitis)) | Verified; grade wording [secondary] ([Jyoti et al. 2026](https://www.ijsurgery.com/index.php/isj/article/download/10388/7130/59401)) |
| Nassar grade as a prior on CVS failure: about 1% (I to II), 14.4% (III), 55.6% (IV), 92.3% (V) | ([Nassar 2021](https://eprints.gla.ac.uk/223839/1/223839.pdf)) | Verified |
| TG18 Grade II/III cannot be claimed from video (they depend on labs, vitals, duration), except "marked local inflammation" such as gangrene or abscess | ([GPnotebook TG18](https://gpnotebook.com/en-GB/pages/gastroenterology/acute-cholecystitis/severity-of-acute-cholecystitis)) | Criteria verified (secondary site); flag rule [inference] |
| Do not name the "cystic artery" with certainty before CVS: the right hepatic artery crosses the triangle in 15%; caterpillar hump in 1.3% to 13.3% | ([Alius et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10456257); [Kulkarni 2025](https://www.msjonline.org/index.php/ijrms/article/view/14466)) | Verified |
| Dissection below the Rouvière sulcus / R4U line raises a risk flag (sulcus present in 83%) | ([Alius et al. 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10456257)) | Landmark verified; flag [inference] |
| Bail-out claims must match what is seen: fundus-first starts at the fundus; subtotal leaves wall behind (fenestrating means more leaks, reconstituting means more neostones) | ([Yildirim et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8942168)) | Verified |
| Aspiration or irrigation in P6 is normal and does not imply a complication | Derived | [inference] |

### What video can and cannot support

- **Visually supported:**
  - Phase and tool presence.
  - Triplets.
  - Individual CVS criteria (needs a time window).
  - Adhesion extent, hyperemia, distension, pericholecystic fluid.
  - Gangrene or necrosis, perforation, pus.
  - An impacted stone in Hartmann's pouch.
  - Bail-out type.
- **Needs data from outside the video:**
  - TG18 grade.
  - Acute vs chronic histology.
  - Exact Mirizzi (Csendes) type, which needs fistula circumference, usually from cholangiography ([Radiopaedia](https://radiopaedia.org/articles/mirizzi-syndrome?lang=gb)).
  - Porcelain gallbladder, unless overtly calcified.
  - Malignancy.
  - Hidden ductal anatomy and arterial origin.
- Tag every condition claim with one of: "visually supported", "requires non-video data" or "not inferable".

### Ground-truth datasets and access

| Dataset | Validates claims about | Size | Access and licence |
|---|---|---|---|
| Cholec80 | Phases (7), tool presence (7) | 80 videos, 13 surgeons, 25 fps ([EndoNet](https://arxiv.org/pdf/1602.03012)) | CAMMA request form, CC BY-NC-SA 4.0 [unverified] |
| Cholec80-CVS | Whole-video and segment-level CVS | Labels for all 80 videos ([DOAJ](https://doaj.org/article/a4f03ee824f94be69a4cfe6997dd8b44)) | Scoring scheme (0 to 2 per criterion, 5 or more of 6 = achieved) and figshare distribution [unverified]; needs the Cholec80 videos |
| CholecT45 / CholecT50 | Instrument-verb-target triplets | 50 videos, about 100.9K frames, about 161K labels ([Nwoye et al.](https://discovery.ucl.ac.uk/id/eprint/10168171/1/2204.04746%20(1).pdf)) | Challenge entrants signed a contract or NDA ([FAQ](https://cholectriplet2021.grand-challenge.org/FAQ/)); public CAMMA release under CC BY-NC-SA [unverified] |
| Endoscapes2023 | CVS per criterion (3 raters, majority vote), boxes, segmentation of cystic artery and cystic plate | 201 videos; 11,090 CVS frames; 1,933 box frames; 493 seg frames (the abstract says 422, which conflicts) ([arXiv](https://arxiv.org/html/2312.12429v3)) | PhysioNet Restricted Health Data License 1.5.0: register and sign the DUA ([PhysioNet](https://www.physionet.org/content/endoscapes-2023/)) |
| CholecSeg8k | Anatomy and tool pixels (13 classes, including Blood and Cystic Duct; no cystic artery) | 8,080 frames from 17 Cholec80 videos ([Hong et al.](https://ar5iv.labs.arxiv.org/html/2012.12453)) | Kaggle, CC BY-NC-SA 4.0; [Hugging Face mirror](https://huggingface.co/datasets/minwoosun/CholecSeg8k/blob/fbf8b94c674ea97e6adf9836e7f626f3ad3380ab/README.md?code=true). Can be downloaded today |
| HeiChole | Phases (no rigid order), 4 actions, instruments, mGOALS including "operation difficulty" | 33 videos (24 train) ([arXiv 2109.14956](https://arxiv.org/pdf/2109.14956)) | Synapse, CC BY-NC-SA; test set not released |
| CholecInstanceSeg | Tool instance masks, usable to calibrate a motion tracker | 41.9k frames ([arXiv 2406.16039](https://arxiv.org/html/2406.16039v2)) | [Synapse](https://www.synapse.org/Synapse:syn60239970/wiki/628710); dataset licence not stated |
| CholecTrack20 | Tool tracks | Not fetched ([Papers with Code](https://ml.paperswithcode.com/dataset/cholectrack20)) | Unknown |
| Intraoperative Adverse Events | 7 adverse-event types on Cholec80 | Not stated | IEEE DataPort subscription; the page says files have not been uploaded ([IEEE DataPort](https://ieee-dataport.org/documents/intraoperative-adverse-events-dataset)) |
| Parkland / Nassar labels | Difficulty grade | None public found | Would need custom annotation |

- **Ground truth is noisy.** Endoscapes inter-rater kappa is 0.33 (two structures), 0.53 (triangle), 0.44 (cystic plate) and 0.38 overall ([arXiv 2312.12429v3](https://arxiv.org/html/2312.12429v3)). A SAGES abstract is titled "no consensus exists" on video CVS assessment ([SAGES](https://www.sages.org/meetings/annual-meeting/abstracts-archive/video-assessment-of-the-critical-view-of-safety-no-consensus-exists/)). Agreement with these labels has a ceiling well below 100%.

### A verification stack with independent layers

- **L1, schema:** constrained JSON with enums taken from the Cholec80 phases and the CholecT50 vocabulary. This removes non-existent instruments and verbs **[inference]**.
- **L2, rules:** phase state machine, triplet validity, tool-phase consistency and the CVS ordering rule. Rules check consistency, not truth.
- **L3, temporal:** HMM or median smoothing; flag flicker and order violations; tools should not appear for a single frame **[inference]**.
- **L4, tool-motion/physics verifier:**
  - **Status:** the notes contain **no sourced evidence** for it. Physics and motion priors (Kalman filtering, optical-flow consistency) were not researched.
  - **Proposed checks [inference]:**
    - Tool tracks are continuous with bounded velocity.
    - Tools enter from the edge of the frame.
    - A `clip` or `cut` triplet needs the right instrument present, with its tip touching a segmented target.
  - **Calibration:** use CholecInstanceSeg or CholecTrack20 to measure how often real tool motion breaks these priors before trusting the verifier.
- **L5, independent model:**
  - Use a contrastive or supervised model (SurgVLP- or OpenCLIP-style), not a second Gemini call.
  - When two LLMs are both wrong, they give the same wrong answer about 60% of the time, against about 33% by chance. Models from the same developer correlate more ([arXiv 2506.07962](https://arxiv.org/abs/2506.07962)).
  - Contrastive models outperformed Gemini on zero-shot CVS ([arXiv 2504.02799](https://arxiv.org/abs/2504.02799)).
- **L6, perturbation:**
  - Re-run on blacked-out or shuffled frames; labels that stay the same are driven by priors, so drop them.
  - Rephrase the question without naming entities.
  - Basis: in SurgCheck, Action-category answers barely changed when images were removed (79.4 vs 72.6 for fine-tuned LLaVA-OV) ([arXiv 2605.01911](https://arxiv.org/abs/2605.01911)).
- **L7, confident learning:** train the specialist on pseudo-labels, then flag likely label errors from cross-validated probabilities ([arXiv 1911.00068](https://arxiv.org/abs/1911.00068), method not re-checked).
- **Confidence:**
  - Do not filter on the VLM's stated confidence. In medical VQA, expected calibration error (ECE) reached 0.79, and discrimination stayed at chance on hard tasks even after calibration training ([arXiv 2606.27023](https://arxiv.org/abs/2606.27023)).
  - Calibrate on a labelled public slice instead.
- **Few-shot:** in-domain examples more than tripled GPT-4o/Gemini CVS F1, but example bounding boxes made tool detection collapse to near zero ([arXiv 2504.02799](https://arxiv.org/abs/2504.02799)). Validate each task separately.
- **Narration:**
  - Emit atomic structured claims: timestamp, phase, instrument, verb, target, safety assertion.
  - Answer verification questions on the frames without showing the model the draft (Chain-of-Verification, [arXiv 2309.11495](https://arxiv.org/abs/2309.11495), method not re-checked).
  - Escalate when answers disagree across samples. Med-Gemini's search guided by answer entropy raised MedQA from 87.2% to 91.1%, but that was text, not video ([arXiv 2404.18416](https://arxiv.org/abs/2404.18416)).

### Testing the verifier without new human labels

- **Injected-error test:** corrupt one field at a time in correct narrations on labelled clips (phase, instrument, target, a false "CVS achieved", a shifted timestamp). Report the verifier's recall and precision for each error type **[inference]**.
- **Metrics:** report MCC or macro-F1, not accuracy. CVS labels are mostly negative.
- **LLM judges:** calibrate any judge against ground truth. Judges overrate weaker models and models from their own provider ([arXiv 2506.07962](https://arxiv.org/abs/2506.07962)).

## What this means for the build today

- **Frame it as post-hoc education and documentation review of recorded video, not intraoperative guidance.**
  - FDA: the non-device CDS exemption excludes software that analyses medical images or signals, and the January 2026 revision did not change this ([Covington](https://www.cov.com/en/news-and-insights/insights/2026/01/5-key-takeaways-from-fdas-revised-clinical-decision-support-cds-software-guidance)). Applying this to video is an inference.
  - MHRA: intended purpose decides device status, and "not for diagnosis" disclaimers do not override medical claims ([Brodies](https://brodies.com/insights/healthcare-and-life-sciences/mhra-clarifies-that-ai-scribes-are-not-automatically-medical-devices/)).
- **Data:**
  - Download CholecSeg8k now.
  - Submit the PhysioNet Endoscapes DUA and the CAMMA requests immediately, but plan the demo without them.
  - Use Cholec80 or CholecT50 only if a team member already has legitimate access.
- **Claim scope:**
  - Narrate phase, tool presence and triplets with confidence.
  - Narrate CVS only per criterion and hedged.
  - Show Parkland as "rule applied, not validated", since no public labels exist.
  - Drop any TG18 claims from video.
- **Verifier diversity is the Track 1 story.**
  - Show at least three non-Gemini checks: rules or state machine, a supervised or contrastive model, and the motion tracker.
  - Show one ablation where Gemini-only verification lets an error through and the diverse stack catches it.
- **Headline metric:**
  - Injected-error recall and precision per claim type.
  - Agreement with held-out public labels, reported with MCC.
  - A before/after on CVS or phase: raw Gemini labels vs the specialist trained on verified labels. Anchor against Med-Gemini's 55.2% vs 67.0%, but re-measure with today's model.
- **Run one flywheel round only.** No published head-to-head shows self-verified pseudo-labels matching expert labels for lap chole, so do not claim multi-round gains you have not run.
- **Chess review UI:**
  - 8 to 10 nodes, labelled "deviation from reference policy".
  - The blunder is defined by CVS at clip time.
  - Show the evidence grade on every flag.
  - Do not mention BDI prediction.
- **Contamination check:** Cholec80 is public and likely in pretraining data. Evaluate at least partly on HeiChole, the CholecT50 in-house videos or another less-public source, and say so.
- **Safety default:** tune for precision on "CVS achieved" and "safe to clip". Abstain when unsure, because false positives carry the liability.
- **Safe to say on stage:**
  - 0.3% to 1.5% BDI.
  - 97% misperception (cite as secondary).
  - CVS achieved in 9% to 16% (secondary).
  - 201 videos.
  - 55.2% vs 67.0%.
  - Kappa 0.38.
  - EUR 11k to 43k per BDI.
- **Avoid on stage:**
  - "Over 1 million US cases".
  - The market CAGR as fact.
  - Any cost per label.
  - The one-third Medicare mortality figure without the elderly-cohort caveat.

## Open questions

### Risks to resolve before the demo

- **Licence compatibility with the Gemini API:** does the PhysioNet DUA, or the CC BY-NC-SA terms, allow sending frames to a third-party cloud model? The notes do not answer this. Check before uploading any Endoscapes frames.
- **Licence for anything beyond the hackathon:** most sets are non-commercial or restricted, so a commercial product needs separate permissions. YouTube surgical video terms of service were not researched.
- **Correlated errors:** how independent is each verifier layer? Measure pairwise error correlation on a labelled slice instead of assuming independence.
- **Noisy ground truth:** with CVS kappa at 0.38, a mismatch with labels does not necessarily mean the model is wrong. Decide up front how to present agreement ceilings.
- **Base rate:** CVS is achieved in only 9% to 16% of cases, so many cases will raise flags. Judges may read these as false positives unless the base rate is shown.
- **Tool-motion verifier:** there is no sourced evidence that motion priors catch label errors in laparoscopic video. Smoke, occlusion and camera motion may make it noisy. It needs its own injected-error test.
- **Distribution shift:** GPT and Gemini scored under 0.03 F1 on several gastric bypass phases ([arXiv 2504.02799](https://arxiv.org/abs/2504.02799)). The shift from datasets to a real OR (smoke, bleeding, lighting) was not quantified anywhere.
- **Regulatory drift:** narration that names complications or advises on next steps moves toward device territory in both the US and the UK.

### Facts to verify before encoding or quoting

- Current (2023 to 2026) US and NHS annual lap chole counts. NHS figures could come from HES OPCS code J18 or GIRFT.
- Way 2003 (97%) and Korndorffer/Mascagni (9% to 16%) from the primary papers.
- The Brunt 2020 recommendations beyond the abstract: exact wording and GRADE strength.
- The Cholec80 licence and access route.
- The Cholec80-CVS scoring scheme and file format.
- The CholecT50 15-target list and full 100-triplet list.
- m2cai16 details.
- Whether Endoscapes' GitHub route still works after the move to PhysioNet.
- Parkland wording from Madni 2018 Table 1, and the 2019 outcomes validation.
- Strasberg 2016 on subtotal subtypes; ducts of Luschka definition and prevalence.
- Endoscapes segmentation count: 493 or 422.
- Whether Google DeepMind's "Aletheia" generator-verifier agent exists. It was not verified; do not cite it.

### Research questions the build leaves open

- Does a specialist trained on self-verified pseudo-labels reach or beat the 67.0% supervised CVS baseline? How does its precision compare with expert-label training?
- How should regret be defined at nodes with no guideline recommendation (conversion, Q12; choice of bail-out subtype)?
- Can reference-policy regret ever be checked against outcomes? No public lap chole video dataset with linked outcomes was found. Registries such as GallRiks hold outcomes but no video.
- What does one expert-labelled hour of lap chole video actually cost? No sourced figure exists, and it drives the economic case.
- How do 2026 Gemini models perform on Cholec80-CVS, Endoscapes and CholecT45? No current benchmark was found.

## Conclusion

The project's real contribution is not narration but making surgical claims auditable. Each claim becomes atomic, tied to frames, checked against a closed vocabulary or a guideline rule, and confirmed by a model that does not share Gemini's errors. That reframes the DeepMind pitch: Med-Gemini already shows that generalist perception falls short on CVS, and correlated-error research shows that adding more of the same model does not fix it. Diverse verification is the lever, and it is measurable today with injected errors on public labels.

The chess metaphor earns its place only in a restricted form. Surgery offers one well-defined blunder (dividing a structure without CVS) and a handful of guideline branches. Everywhere else, the "engine" is expert opinion with very low certainty. A team that shows this honestly, with evidence grades on screen and difficulty-normalised flags, will look more credible than one claiming to minimise outcome regret. The data needed to compute outcome regret does not yet exist publicly.
