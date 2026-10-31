# SurgProof: what we are showing (confirmed by Nir, updated 13:15 31 Oct)

**Problem.** Surgical AI is bottlenecked by expert labelling. Surgeons cannot spend hours annotating frames, so models never get the data.

**What SurgProof does.** An Argon agent teaches itself a procedure from public teaching video and writes its own manual. It labels real surgical frames, then checks its own work with zero human labels. EVERY label is double-checked by an independent second pass (different prompt, zoomed crop, stronger model). Agreement -> accept or reject; disagreement or low confidence -> AMBER. Ambers are cleared in a LOOP: Argon re-examines each amber (tighter crop, manual criteria, fresh votes) for up to N rounds, resolving it to accept or reject; only what is still amber after the loop goes to a human. A specialist is trained only on what survived.

**Proof.** Shown on laparoscopic cholecystectomy, the one procedure with pixel-level ground truth: 6 held-out CholecSeg8k videos, plus external sets the system never saw (Endoscapes2023, CholecT50) with the same frozen manual. Ground truth is used only to score at the end.

**Demo.** Show the WHOLE pipeline running end to end, visually: public teaching video -> self-written manual -> real frames labelled -> every label double-checked (green accept / red reject / amber unsure) -> amber loop shrinks the ambers round by round -> human clears the few that remain -> specialist trained on verified labels only -> one real number on held-out videos. No judge input needed; every stage visibly runs on real data. "No human labels. Only proof."

**Bigger claim.** "Any surgery" is a labelled showcase, never presented as proof.

**Positioning: remove the human from dataset creation.** Today's best surgical annotation tools (e.g. Supervisely's suite, https://supervisely.com/blog/top-5-ai-tools-for-surgical-video-annotation/) still need a human at every step: a person draws the first box or clicks the first points (MixFormer, SAM-based SmartTool, TAP-Net), a person corrects tracking errors and restarts it, action tags are assigned by hand, and reviewer and domain-expert roles check the result. Their claim is "10x faster", with a human still in every loop (no benchmark given).
SurgProof removes the human from each of those steps: seeding (manual learned from public video), labelling, correction (the amber loop) and review (independent double-check). The output is a **dataset** (out/verified_dataset.jsonl plus provenance). People stay in control with a FINAL PASS before anything becomes proper data: the human reviews any ambers left after the loop plus a random audit sample of the auto-accepted labels (with their evidence crops), and only then releases the dataset. Nothing is exported without that sign-off. Goal: ambers after the loop -> 0.

**Full annotation, by the agent, visible live.** Beyond presence labels, the agent draws bounding boxes for tools and anatomy, points on instrument tips, and the surgical phase of each frame, double-checking each and looping on disagreements. The demo shows these being drawn on real frames in real time, then scored against ground truth (masks-derived boxes on held-out videos).
