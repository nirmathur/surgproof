# SurgProof Product Demo Storyboard (52.0s, 7 Shots)

- **Project**: **SurgProof** — *"No human labels. Only proof."*
- **Spec**: Implements `promo/DESIGN.md` (`nir-director`)
- **Format**: `52.0s` (45–60s target), `1280×720`, `24 fps`, H.264 (`yuv420p`) + stereo AAC
- **Output Paths**: `out/promo/surgproof_promo.mp4`, `out/promo/surgatlas_promo.mp4`, and `out/promo/contact_sheet.png`

## Design Restraint Rules (Strictly Enforced)
1. **100% Real Material**: Uses real `CholecSeg8k` frames from `data/index.csv`, real `out/manual.json`, real `out/labels_manual.jsonl` & `out/decisions.csv`, and real `out/results.json`. Zero AI-generated video spectacle, zero holograms/lasers/glows/HUDs.
2. **No Personal Names**: Zero personal names or supervisor quote slots anywhere.
3. **Max 6 Words Per Screen**: Every shot caption is 6 words or fewer, set in `SF Pro` (`#F4F5F8`) on a calm near-black canvas (`#0A0B0E`).

---

## Seven-Shot Breakdown (52.0 Seconds Total)

| # | File | Duration | Real Data Source | On-Screen Caption (≤6 words) |
|---|---|---:|---|---|
| 1 | `out/promo/shot_01_raw_frame.mp4` | `7.0s` | `data/index.csv` (`video12_15750` frames) | `Surgical video. No labels.` (4w) |
| 2 | `out/promo/shot_02_manual.mp4` | `8.0s` | `out/manual.json` (`grasper`, `l_hook`, `cystic_duct`) | `Writes its own visual manual.` (5w) |
| 3 | `out/promo/shot_03_labels.mp4` | `8.0s` | `out/labels_manual.jsonl` (`video12_15750_15758`) | `Labels a real frame.` (4w) |
| 4 | `out/promo/shot_04_verify.mp4` | `9.0s` | `out/decisions.csv` (`accept` / `unsure` / `reject`) | `Checks itself. Green, amber, red.` (5w) |
| 5 | `out/promo/shot_05_gate.mp4` | `8.0s` | Human Gate clearing the single `AMBER` label | `Human clears only amber cases.` (5w) |
| 6 | `out/promo/shot_06_result.mp4` | `6.0s` | `out/results.json` (or `Result: pending`) | `Held-out test accuracy, zero labels.` (5w) |
| 7 | `out/promo/shot_07_end.mp4` | `6.0s` | End Card | `SurgProof. No human labels. Only proof.` (6w) |
