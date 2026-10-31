# Claude Motion prompt (paste into claude.ai > Claude Motion). All numbers verified from out/*.json at 15:20, 31 Oct 2026.

Make three short motion-graphics clips, 1920x1080, dark background #0B0D10, one typeface (Inter or SF Pro), off-white text #ECEEF1, muted grey #8C929C for secondary text. Verdict colours used ONLY for verdicts: accept #40C88C, amber #F0B43C, reject #E64646. Calm, Apple/Linear product-film style: slow cubic easing, staggered reveals (80 ms), no glows, no particles, no 3D, no stock icons. Max 8 words on screen at once. Export each as MP4.

## Clip A, 10 s: "How SurgProof works"
A horizontal pipeline of six labelled nodes that light up left to right, each with a one-line caption fading in under it:
1. Public teaching video: "Learns the procedure"
2. Self-written manual: "Writes its own criteria"
3. Label every frame: "Gemini labels instruments and anatomy"
4. Double-check: "Independent second pass" (node splits into green / amber / red)
5. Amber loop: "Re-checks only the unsure ones"
6. Human final pass: "One sign-off before release"
End state: the six nodes stay lit; caption: "No human labels. Only proof."

## Clip B, 8 s: "Held-out results" (exact numbers, do not round further)
Three stat rows animate in one after another, each number counting up from 0 to its value:
- "96.6%": "accurate on the 93.2% it accepted", small: "n = 1,400 labels · 6 unseen videos"
- "0.76": "box IoU vs ground truth", small: "n = 258"
- "0.61": "mask IoU vs ground truth", small: "n = 599 · 95% CI 0.52 to 0.72"

## Clip C, 6 s: "It corrects itself"
A simple line or bar chart with three points: Round 0 IoU 0.63, Round 1 0.69, Round 2 0.68. Animate the line drawing from round 0 to round 1 (highlight the step up in accept green), round 2 fades in flat. Caption: "Self-correction: +0.06 IoU in one round". Small: "n = 65 masks, scored after the fact; 69% improved, 15% worse".
