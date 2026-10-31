# SurgProof: what we are showing (confirmed by Nir, updated 13:15 31 Oct)

**Problem.** Surgical AI is bottlenecked by expert labelling. Surgeons cannot spend hours annotating frames, so models never get the data.

**What SurgProof does.** An Argon agent teaches itself a procedure from public teaching video and writes its own manual. It labels real surgical frames, then checks its own work with zero human labels: accept / reject / AMBER. Only the ambers go to a human. A specialist is trained only on what survived.

**Proof.** Shown on laparoscopic cholecystectomy, the one procedure with pixel-level ground truth: 6 held-out CholecSeg8k videos, plus external sets the system never saw (Endoscapes2023, CholecT50) with the same frozen manual. Ground truth is used only to score at the end.

**Demo.** Show the WHOLE pipeline running end to end, visually: public teaching video -> self-written manual -> real frames labelled -> each label checked (green accept / red reject / amber unsure) -> human clears the ambers -> specialist trained on verified labels only -> one real number on held-out videos. No judge input needed; every stage visibly runs on real data. "No human labels. Only proof."

**Bigger claim.** "Any surgery" is a labelled showcase, never presented as proof.
