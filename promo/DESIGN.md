# SurgProof Product Demo — Design & Direction Specification (`promo/DESIGN.md`)

**Director**: `nir-director`  
**Renderer**: `nir-promo` (`python -m promo.render`)  
**Format**: 45–60 seconds (Target: **52.0s**, 7 shots), `1280×720` (or `1920×1080`), `24 fps`, H.264 `.mp4` (`yuv420p`)  
**Output Paths**: `out/promo/surgproof_promo.mp4` and `out/promo/surgatlas_promo.mp4`

---

## 1. Core Mandate: Simple Product Demo, Not a Trailer

The goal is a **simple, restrained 52-second product demo of how SurgProof works**, understood by a judge in a single viewing. Think Apple, Linear, and Stripe product films: calm, confident, typographically precise, and anchored 100% in real surgical frames and real pipeline outputs.

### Hard Prohibitions (Zero Exceptions)
1. **No AI-generated video (No Veo / Omni clips)**: Delete or bypass all Veo / `gemini-omni-1.1-flash` video backgrounds. Every shot uses real `CholecSeg8k` frames from `data/index.csv` and real files from `out/`.
2. **No personal names or quotes**: Remove all references to `Nir Mathur`, `KCL`, and `Supervisor Quote Slot`.
3. **6 words maximum per caption**: Every shot has at most **one** caption of **6 words or fewer**.
4. **No chrome or clutter**:
   - No top-left `SurgProof · SHOT X/6` pills.
   - No `ILLUSTRATIVE — AI-GENERATED VISUALIZATION` banners.
   - No bottom lower-third banners (`ACT 1 · ...`).
   - No CLI commands (`python med_school.py ...`) on screen.
5. **Nothing glows**: Zero outer glows, drop-shadow neon, HUD reticles, holograms, particles, lasers, lens flares, or sci-fi UI.
6. **No fake or placeholder numbers**: Never invent a metric. In Shot 6, read one real number from `out/results.json`. If `out/results.json` does not exist yet, display `"Result: pending"` and nothing else.

---

## 2. Visual System: Typography, Colour, Layout & Motion

### Typography (One Typeface: `SF Pro`)
Load macOS system font `/System/Library/Fonts/SFNS.ttf` via Pillow `ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)` and set weight via `font.set_variation_by_axes([100, opt_size, 400, weight])` (or fallback to `/System/Library/Fonts/HelveticaNeue.ttc`). **Never fallback to Arial.**

| Role | Size (`1280×720`) | Weight | Colour | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Hero Number / End Title** | `72px` (`56px` if pending) | Semibold (`600`) | `#F4F5F8` | Shot 6 metric & Shot 7 title |
| **Shot Caption** | `26px` | Medium (`500`) | `#F4F5F8` | Bottom-centered, `y = 636`, **max 6 words** |
| **Card Body / Manual Quote** | `19px` | Regular (`400`) | `#D8DCE6` | Clean sentence case, generous line height (`1.5×`) |
| **Label Pill / Structure Key** | `15px` | Medium (`500`) | `#F4F5F8` | Inside crisp `6px`-radius pill |
| **Muted Metadata** | `14px` | Regular (`400`) | `#8A909E` | Source URL or frame ID only |

### Colour Palette
- **Canvas Background**: Near-black `#0A0B0E` (`rgb(10, 11, 14)`)
- **Elevated Card Surface**: `#12141C` (`rgb(18, 20, 28)`)
- **Card Hairline Border**: `1px` solid `#232734` (`rgb(35, 39, 52)`)
- **Primary Text**: `#F4F5F8` (`rgb(244, 245, 248)`)
- **Secondary Text**: `#8A909E` (`rgb(138, 144, 158)`)
- **Single Accent Colour** (unverified labels & focus state only): `#3B82F6` (`rgb(59, 130, 246)`), fill `rgba(59, 130, 246, 0.14)`
- **Verdict Colours (Shots 4 & 5 ONLY)**:
  - **Accept (Green)**: Stroke/Text `#10B981` (`rgb(16, 185, 129)`), Pill Fill `#092E24` (`rgb(9, 46, 36)`)
  - **Unsure (Amber)**: Stroke/Text `#F59E0B` (`rgb(245, 158, 11)`), Pill Fill `#362407` (`rgb(54, 36, 7)`)
  - **Reject (Red)**: Stroke/Text `#EF4444` (`rgb(239, 68, 68)`), Pill Fill `#331014` (`rgb(51, 16, 20)`)

### Layout & Margins (`1280×720` Canvas)
- **Outer Safe Margin**: Minimum `96px` left/right, `64px` top/bottom.
- **Content Stage**: `x ∈ [96, 1184]`, `y ∈ [64, 584]`.
- **Caption Baseline**: Centered horizontally at `x = 640`, `y = 636`, always with quiet dark space around it.
- **Whitespace Rule**: At least 40% of every frame must remain unadorned `#0A0B0E` background.

### Motion Principles
- **Easing**: All entrances and state changes use cubic ease-out: `ease_out(t) = 1 - (1 - t)**3`.
- **Shot Transitions**: `0.45s` (`11` frames at `24 fps`) smooth cross-dissolve or fade through `#0A0B0E`.
- **Element Entrances**: `0.4s` opacity fade `0 -> 1` paired with a subtle `10px` upward slide.
- **Zero camera shake, zero zoom-pulsing, zero blinking cursors.**

---

## 3. Seven-Shot Storyboard (54.0 Seconds Total — Matched Beat-for-Beat to `VISION.md`)

Every shot follows `VISION.md` (13:15–13:22 updates): positioning (`Today: a human at every step` vs `SurgProof: none`), public teaching video $\rightarrow$ self-written manual $\rightarrow$ real frames labelled $\rightarrow$ independent double-check (green/amber/red) + multi-round Amber loop shrinking uncertain cases $\rightarrow$ human final pass & `Release dataset` sign-off (`out/verified_dataset.jsonl`) $\rightarrow$ one real held-out number (`out/results.json`) $\rightarrow$ end card.

### Shot 1 — Raw Surgical Video & Positioning (`0:00–0:07`, 7.0s)
- **File**: `out/promo/shot_01_raw_frame.mp4`
- **Real Source**: Real laparoscopic frames from `data/index.csv` — specifically held-out test clip `video12_15750` (`frame_15758_endo.png` in `/Users/nirmathur/surgatlas-lanes/data/raw/CholecSeg8k/video12/video12_15750/`).
- **Composition**:
  - Top positioning header (clean two-line contrast centered at `y = 68`, fading in gently):
    - Line 1 (`18px` SF Pro Regular `#8A909E`): `Today: a human at every step.`
    - Line 2 (`22px` SF Pro Semibold `#F4F5F8`): `SurgProof: none.`
  - Real surgical frame displayed centered at `736×414` (`x = 272, y = 140`), framed by a `1px` `#232734` border (`8px` corner radius).
  - No bounding boxes, no sidebar, no badges.
- **Caption (4 words)**:
  > `Surgical video. No labels.`

---

### Shot 2 — Self-Written Manual (`0:07–0:15`, 8.0s)
- **File**: `out/promo/shot_02_manual.mp4`
- **Real Source**: Reads `out/manual.json` directly (`procedure`, `sources[0]`, and `structures`).
- **Composition**:
  - **Left Column (`x = 112, w = 320, y = 136`)**: Minimal source card (`#12141C`, `1px` border `#232734`) showing the public teaching video metadata read from `out/manual.json`:
    - Small muted label: `PUBLIC TEACHING SOURCE`
    - Title: `Laparoscopic Cholecystectomy` (`sources[0].title`)
    - Muted URL: `youtube.com/watch?v=PNydeaI2WIU`
  - **Right Column (`x = 464, w = 704, y = 112`)**: Clean document card (`#12141C`, `1px` border `#232734`) titled `out/manual.json` in `14px` `#8A909E`. Inside, 3 real structure rules extracted from `out/manual.json` fade in sequentially (`t = 0.8s, 1.8s, 2.8s`):
    1. **`grasper`** — *"Metallic hinged double-jaw tip with horizontal serrations."*
    2. **`l_hook`** — *"Single-piece metallic silver wire bent at a 90-degree right angle."*
    3. **`cystic_duct`** — *"Mark present only when a skeletonized cylindrical duct is isolated."*
- **Caption (5 words)**:
  > `Writes its own visual manual.`

---

### Shot 3 — Zero-Shot Frame Labelling (`0:15–0:22`, 7.0s)
- **File**: `out/promo/shot_03_labels.mp4`
- **Real Source**: Real frame `video12_15750_15758` (`frame_15758_endo.png`) and its real predictions from `out/labels_manual.jsonl`:
  - `grasper`: `box = [420, 140, 522, 380]`
  - `l_hook`: `box = [50, 524, 492, 824]`
  - `gallbladder`: `box = [8, 298, 565, 560]`
  - `liver`: `box = [126, 300, 764, 868]`
  - `cystic_duct`: `box = [445, 445, 748, 810]`
  *(Note: coordinates in `out/labels_manual.jsonl` are `[ymin, xmin, ymax, xmax]` normalized `0..1000` — scale accurately to the rendered frame rectangle).*
- **Composition**:
  - **Left (`x = 112, y = 112, w = 688, h = 387`)**: The real laparoscopic frame. One by one (`t = 0.5s, 1.1s, 1.7s, 2.3s, 2.9s`), clean `2px` `#3B82F6` boxes appear with a small `#3B82F6` pill tag at the top-left of each box (`grasper`, `l_hook`, `gallbladder`, `liver`, `cystic_duct`).
  - **Right (`x = 832, y = 112, w = 336, h = 387`)**: Clean vertical list of the 5 labels appearing in sync with the boxes, each with a calm blue dot `#3B82F6` and structure name (`grasper`, `l_hook`, `gallbladder`, `liver`, `cystic_duct`).
- **Caption (4 words)**:
  > `Labels a real frame.`

---

### Shot 4 — Independent Double-Check & Multi-Round Amber Loop (`0:22–0:32`, 10.0s — Hero Moment)
- **File**: `out/promo/shot_04_verify.mp4`
- **Real Source**: Same frame `video12_15750_15758` from `out/labels_manual.jsonl` + `out/amber_rounds.jsonl` / `out/decisions.csv` scored by `accept.py` rules (zero GT access):
  - `grasper` (`trust = 0.98`, `pv_match = True`) $\rightarrow$ **Accept (Green `#10B981`)**
  - `l_hook` (`trust = 0.98`, `pv_match = True`) $\rightarrow$ **Accept (Green `#10B981`)**
  - `gallbladder` (`trust = 0.97`, `pv_match = True`) $\rightarrow$ **Accept (Green `#10B981`)**
  - `liver` (`trust = 0.49`, `pv_match = False`) $\rightarrow$ **Unsure (Amber `#F59E0B`)**
  - `cystic_duct` (`trust = 0.29`, `pv_match = False`, `p_wrong = 0.35`) $\rightarrow$ **Reject (Red `#EF4444`)**
- **Composition (Two-Beat Hero Sequence)**:
  - **Beat 4A (`t = 0.0s–4.5s`, Double-Check)**: Each bounding box and right-hand row smoothly transitions from neutral blue `#3B82F6` to its double-check verdict colour:
    - `grasper`, `l_hook`, `gallbladder` turn **Green (`#10B981`)** (`accept`).
    - `liver` turns **Amber (`#F59E0B`)** (`amber`).
    - `cystic_duct` turns **Red (`#EF4444`)** (`reject`, box dims to `25%` opacity).
  - **Beat 4B (`t = 4.5s–10.0s`, Shrinking Amber Loop)**: Below the verdict list on the right card, display the real **Amber Loop** progress read from `out/amber_rounds.jsonl` (or `out/decisions.csv` if `amber_rounds.jsonl` is still pending, showing `Round 0 -> Round 1 -> Round 2 -> Round 3` horizontal shrinking amber bars as Argon re-examines tight crops against the manual criteria).
- **Caption (5 words)**:
  > `Checks itself. Green, amber, red.`

---

### Shot 5 — Human Final Pass & "Release dataset" Sign-Off (`0:32–0:40`, 8.0s)
- **File**: `out/promo/shot_05_gate.mp4`
- **Real Source**: Real Human Gate (`pages/2_Human_gate.py`, reading `out/decisions.csv` and `out/amber_rounds.jsonl` $\rightarrow$ `out/verified_dataset.jsonl`).
- **Composition**:
  - Centered minimal Gate review card (`x = 160, y = 96, w = 960, h = 456`, `#12141C`, `1px` border `#232734`).
  - Top bar of card: `Final Pass · Release Dataset` on left; two clean status pills on right: `Ambers left: 1 -> 0` and `Audit sample: 20/20`.
  - Inside left (`w = 480, h = 270`): Real laparoscopic frame with evidence crop showing the residual **Amber (`#F59E0B`)** item.
  - Inside right: Structure label, manual criterion, `Approve` / `Reject` buttons, and the **`Release dataset`** button (`out/verified_dataset.jsonl`).
  - **Action at `t = 3.5s`**: The last Amber item turns **Green (`#10B981`, `verified`)**, `Ambers left: 0`, and the **`Release dataset`** button unlocks in crisp **Green (`#10B981`)** (`out/verified_dataset.jsonl`).
- **Caption (5 words)**:
  > `Human clears only amber cases.`

---

### Shot 6 — The Result (`0:40–0:47`, 7.0s)
- **File**: `out/promo/shot_06_result.mp4`
- **Real Source**: Strictly `out/results.json` (and `out/student.json` if present).
- **Conditional Rule (MANDATORY)**:
  - **Case A — `out/results.json` exists and contains a valid test-split metric**:
    - Display **one** hero number read from `out/results.json` (specifically held-out test accuracy on accepted self-verified labels excluding liver, e.g. `94.2%`), set in `84px` SF Pro Semibold `#F4F5F8`, centered at `(640, 320)`.
    - Below it at `(640, 400)` in `24px` SF Pro Regular `#8A909E`, one caption (5 words): `Held-out test accuracy, zero labels.`
  - **Case B — `out/results.json` does NOT exist yet**:
    - Pure `#0A0B0E` canvas.
    - Display **exact text**:
      > `Result: pending`
    - Set in `52px` SF Pro Medium `#F4F5F8`, centered at `(640, 360)`.
    - **Nothing else on screen.** No subtitle, no boxes, no explanation.

---

### Shot 7 — End Card (`0:47–0:54`, 7.0s)
- **File**: `out/promo/shot_07_end.mp4`
- **Composition**:
  - Pure `#0A0B0E` canvas.
  - Line 1 (centered at `x = 640, y = 330`, `48px` SF Pro Semibold `#F4F5F8`):
    > `SurgProof.`
  - Line 2 (centered at `x = 640, y = 392`, `28px` SF Pro Regular `#8A909E`):
    > `No human labels. Only proof.`
  - Total words on screen: **6 words**.
  - Fades in smoothly over `0.8s`, holds cleanly, fades to `#0A0B0E` over final `0.8s`.

---

## 4. Exact Build Instructions for `nir-promo`

1. **Pure PIL + `ffmpeg` Pipeline (Fast, Deterministic, Zero API Wait)**:
   - Render each shot (`shot_01` .. `shot_07`) at `1280×720`, `24 fps` by generating PIL `Image` frames in Python and piping raw RGB frames directly to `ffmpeg -f rawvideo -pix_fmt rgb24 -s 1280x720 -r 24 -i - -c:v libx264 -pix_fmt yuv420p -preset fast -crf 18` (with silent AAC audio track `-f lavfi -i anullsrc=channel_layout=stereo:sample_rate=44100 -c:a aac -shortest`).
   - Total render time for all 7 shots + stitch will be **under 15 seconds**, enabling rapid review iterations.
2. **Bounding Box Coordinate Mapping**:
   - In `out/labels_manual.jsonl` and `out/decisions.csv`, `box` is `[ymin, xmin, ymax, xmax]` on a `0..1000` scale.
   - Map to pixel coordinates inside the rendered image rectangle `(rx, ry, rw, rh)` via:
     - `x1 = rx + int((xmin / 1000.0) * rw)`
     - `y1 = ry + int((ymin / 1000.0) * rh)`
     - `x2 = rx + int((xmax / 1000.0) * rw)`
     - `y2 = ry + int((ymax / 1000.0) * rh)`
3. **Contact Sheet Generation for Review**:
   - After stitching `out/promo/surgproof_promo.mp4` (and copying to `out/promo/surgatlas_promo.mp4`), always generate the review contact sheet:
     ```bash
     ffmpeg -y -i out/promo/surgproof_promo.mp4 -vf "fps=1/2,scale=480:-1,tile=4x4" out/promo/contact_sheet.png
     ```
4. **Word Count Verification**:
   - Shot 1: `Surgical video. No labels.` (4 words)
   - Shot 2: `Writes its own visual manual.` (5 words)
   - Shot 3: `Labels a real frame.` (4 words)
   - Shot 4: `Checks itself. Green, amber, red.` (5 words)
   - Shot 5: `Human clears only amber cases.` (5 words)
   - Shot 6: `Result: pending` (2 words) — or `<N>%` + `Held-out test accuracy, zero labels.` (6 words)
   - Shot 7: `SurgProof. No human labels. Only proof.` (6 words)

---

## 5. Real Streamlit UI Capture Contract (`app.py` & `origin/harald/demo`)

Per the 13:12 update: Shots 2–6 should showcase our **real product UI** (`app.py` and `pages/2_Human_gate.py`, swapping in Harald's tabs from `origin/harald/demo` as merged):
- **Shot 2 (`0:07–0:15`)**: Can show the real `1 — Med School` tab from `app.py` (`origin/harald/demo`) alongside the 3 clean structure rules (`grasper`, `l_hook`, `cystic_duct`) from `out/manual.json`.
- **Shot 3 (`0:15–0:23`) & Shot 4 (`0:23–0:32`)**: Show the real frame `video12_15750_15758` inside the clean UI container (`2 — Label -> Verify Loop` / `app.py`), first with neutral blue boxes (`#3B82F6`) in Shot 3, then transitioning to **Green (`#10B981`, accept)**, **Amber (`#F59E0B`, unsure)**, and **Red (`#EF4444`, reject)** in Shot 4.
- **Shot 5 (`0:32–0:40`)**: Real capture/render of `pages/2_Human_gate.py` (the Human Gate page) clearing the single **Amber** item to **Green (`verified`)**.
- **UI Hygiene Check**: Before capturing `app.py` from `origin/harald/demo`, ensure the top header (`👤 Built for a Real Person — Nir Mathur` and `🩺 Clinical Supervisor Quote Slot`) is stripped so zero personal names appear in the video.

---

## 6. Verbatim Veo 3.1 Prompts & Negative Prompt (`promo/prompts/`)

Real CholecSeg8k frames and real UI screens are always preferred. If `nir-promo` uses Veo 3.1 (`veo-3.1-fast-generate-preview` for drafts, `veo-3.1-generate-preview` for final) for at most 1–2 quiet, natural establishing shots, it **must** use the exact prompts below (saved in `promo/prompts/`) and place a small, muted corner tag `AI-generated illustration` (`13px` SF Pro Regular `#8A909E`, top-right `(1184, 48)` — never an orange banner).

### Shared `negativePrompt` (`promo/prompts/negative_prompt.txt`)
```text
text, captions, subtitles, numbers, digits, percentages, charts, graphs, gauges, UI widgets, HUD, reticles, crosshairs, hologram, wireframe, glow, bloom, neon, lens flare, particles, sparks, lasers, prism, lightning, portal, sci-fi, dramatic lighting, rim light, fast motion, camera shake, heart, cardiac anatomy, bowel loops, anatomy diagrams, watermark, logo, personal names
```

### Optional Establishing Clip 1 (`promo/prompts/shot_01_establishing.txt`, 151 words)
```text
Static, locked-off medium close-up shot on a 50mm prime lens at f/2.8 with shallow depth of field and zero camera drift. A medical grade surgical display monitor inside a quiet laparoscopic operating theatre shows a clear endoscopic view of a pear-shaped bluish-grey gallbladder tucked beneath the dark reddish-brown inferior edge of the liver while a brushed steel laparoscopic grasper gently holds the fundus stationary. In the softly defocused foreground, a gloved surgeon's hand rests calmly near a stainless steel instrument tray without sudden movement. Soft diffuse overhead clinical illumination at neutral 5600K white balance with gentle ambient fill and no harsh rim lighting or specular glare. Documentary realism, muted natural clinical palette of slate greys, desaturated blues, and warm tissue tones, clean minimalist composition inspired by Apple product cinematography, shot on 35mm film with fine natural grain. Audio: quiet room tone, faint steady ventilation hum, no music.
```

### Optional Establishing Clip 2 (`promo/prompts/shot_02_manual.txt`, 136 words)
```text
Static, locked-off over-the-shoulder medium shot on an 85mm prime lens at f/2.8 with shallow depth of field and zero camera movement. A matte dark-anodized workstation display on a clean surgical research desk shows a calm laparoscopic gallbladder surgery lecture frame on the left half of the screen while clean monospaced reference notes appear quietly on the right half; a researcher's hand rests still on the desk surface beside a dark trackpad. Quiet academic surgical research room with uncluttered matte charcoal desk surface and softly blurred background wall. Soft diffuse key light from the left at neutral 5200K white balance, low-contrast natural shadows, no neon accent lights or screen bloom. Documentary realism, calm minimalist product film aesthetic, muted neutral greys and off-whites, high optical clarity. Audio: quiet room tone, no music, no voiceover.
```

