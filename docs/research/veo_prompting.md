# Veo 3.1 prompting: how to make each shot exact (compiled 13:15 for nir-director / nir-promo)

Sources (third-party guides, broadly consistent; read Google's own first: https://ai.google.dev/gemini-api/docs/video):
- https://ltx.studio/blog/veo-prompt-guide
- https://www.atlabs.ai/blog/ultimate-veo-3-1-prompting-guide
- https://www.dreamhost.com/blog/veo-3-1-prompt-guide/
- https://dhl.veed.io/learn/veo-3-1-prompts
- https://developer.tenten.co/veo-331-pro-json-prompting-step-by-step (structured JSON prompts)

## Structure every prompt in this order (aim ~100-200 words)
1. Camera first: shot type + movement + lens. Use film vocabulary ("locked-off static shot", "slow dolly-in", "rack focus", "85mm, shallow depth of field"). For product/UI-like shots say "static, locked-off camera" to stop drift.
2. Subject and the ONE action in this clip. One scene, one action per clip.
3. Environment / setting, concretely.
4. Lighting, named precisely ("soft diffuse overhead light", "neutral 5600K", "no rim light").
5. Style and mood ("documentary, muted natural colour, clean, minimal, calm"); technical look ("35mm film", "high detail").
6. Audio: describe ambient sound explicitly, or "no music, quiet room tone".
Also pass a negativePrompt (the API supports it) listing everything we never want.

## Our house style for any Veo shot (minimal, not dramatic)
Positive: "static locked-off camera, documentary, natural soft light, muted neutral colours, shallow depth of field, calm, minimal composition, realistic".
negativePrompt: "text, captions, numbers, digits, charts, UI, HUD, hologram, glow, neon, lens flare, particles, lasers, sci-fi, dramatic lighting, fast motion, heart, anatomy diagrams, watermark".

## Example (establishing shot)
"Static, locked-off medium shot, 50mm, shallow depth of field. A laparoscopic monitor in a quiet operating theatre shows a live endoscopic view of a gallbladder beneath the liver edge; a gloved hand rests near the instrument tray, barely moving. Soft diffuse overhead light, neutral white balance, muted blues and greys. Documentary realism, clean and calm, minimal composition. Audio: quiet room tone, faint monitor hum, no music."

## Process
Draft with veo-3.1-fast-generate-preview, review the frames, refine ONE variable at a time, render finals with veo-3.1-generate-preview. Keep the prompt text of every final clip in promo/prompts/<shot>.txt for reproducibility.
