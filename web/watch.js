/* SurgProof — Watch Mode Player (web/watch.js)
 *
 * Full-bleed dark player supporting both:
 *   1. Smooth 25 fps every-frame dense AI clips (?clip=video12_15750 default,
 *      video55_00508, video28_00000) from web/data/dense/<clip>/ + out/dense/<clip>/
 *      with per-clip mask IoU read from out/dense_results.json.
 *   2. Held-out video12 80-keyframe sequence (?clip=video12) from web/data/video12.json
 *      and verified narration subtitles from out/narration_video12.json.
 *
 * Zero ground truth used to generate overlays; all numbers and masks read strictly
 * from real exported pipeline artifacts.
 */
(function () {
  "use strict";

  const LABEL_KEYS = ["gallbladder", "liver", "grasper", "l_hook", "cystic_duct", "blood"];
  const DENSE_CLIPS = ["video12_15750", "video55_00508", "video28_00000"];

  const LABEL_NAMES = {
    gallbladder: "Gallbladder",
    liver: "Liver",
    grasper: "Grasper",
    l_hook: "L-hook",
    cystic_duct: "Cystic duct",
    blood: "Blood",
  };

  // Calm anatomical / instrument palette for soft mask overlays
  const MASK_RGB = {
    gallbladder: [64, 200, 140],  // emerald
    liver: [214, 120, 92],        // hepatic warm terracotta
    grasper: [90, 170, 255],      // sky blue
    l_hook: [250, 190, 70],       // surgical gold
    cystic_duct: [245, 158, 11],  // amber gold
    blood: [239, 68, 68],         // soft crimson
  };

  const state = {
    mode: "dense",
    clipId: "video12_15750",
    frames: [],
    sentences: [],
    sentenceByFrameId: {},
    denseResults: null,
    videoData: null,
    currentIdx: 0,
    playing: true,
    timerId: null,
    frameIntervalMs: 40,
    imageCache: {},
    tintedMaskCache: {},
    lastSubtitleText: "",
    renderToken: 0,
  };

  async function fetchJsonWithFallback(urls) {
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (res.ok) {
          return await res.json();
        }
      } catch (_err) {
        // try next candidate path
      }
    }
    return null;
  }

  async function fetchTextWithFallback(urls) {
    for (const url of urls) {
      try {
        const res = await fetch(url, { cache: "no-store" });
        if (res.ok) {
          const txt = await res.text();
          if (txt && txt.trim() && !txt.trim().startsWith("<!DOCTYPE")) {
            return txt;
          }
        }
      } catch (_err) {
        // try next candidate path
      }
    }
    return null;
  }

  function formatPct(val) {
    if (typeof val !== "number" || !Number.isFinite(val)) return null;
    const pct = val <= 1.0 ? val * 100 : val;
    const rounded = Math.round((pct + Number.EPSILON) * 10) / 10;
    return `${rounded.toFixed(1)}%`;
  }

  function updateClipNavStyles(activeClip) {
    const pills = document.querySelectorAll(".watch-clip-pill");
    pills.forEach((el) => {
      const cid = el.getAttribute("data-clip");
      if (cid === activeClip) {
        el.style.color = "#F4F5F8";
        el.style.borderColor = "#3B82F6";
        el.style.background = "#182032";
      } else {
        el.style.color = "#8A909E";
        el.style.borderColor = "#232734";
        el.style.background = "#12141C";
      }
    });
  }

  function updateHeadlineChip(videoData, denseResults, clipId, mode) {
    const chipEl = document.getElementById("watch-headline-chip");
    const titleEl = document.getElementById("watch-title");
    if (!chipEl) return;

    const vidName = clipId.split("_")[0] || "video12";
    if (titleEl) {
      if (mode === "dense") {
        titleEl.textContent = `SurgProof · ${clipId} (held-out · 25 fps every-frame AI)`;
      } else {
        titleEl.textContent = "SurgProof · video12 (held-out)";
      }
    }

    if (mode === "dense" && denseResults && denseResults.clips && denseResults.clips[clipId]) {
      const cinfo = denseResults.clips[clipId];
      const iouVal = typeof cinfo.mean_iou === "number" ? cinfo.mean_iou.toFixed(2) : null;
      const nMasks = cinfo.n_mask_scored || null;
      const presPct = formatPct(cinfo.presence_accuracy);
      const parts = [];
      if (iouVal && nMasks) {
        parts.push(`mask IoU ${iouVal} vs GT (n = ${Number(nMasks).toLocaleString()} masks)`);
      }
      if (presPct) {
        parts.push(`presence ${presPct}`);
      }
      if (parts.length > 0) {
        chipEl.textContent = parts.join(" · ");
        return;
      }
    }

    if (!videoData) {
      chipEl.textContent = `${vidName} · held-out test`;
      return;
    }

    const headline = videoData.headline || {};
    const results = videoData.results || {};
    const acc5 = results.accepted_5_informative || results.accepted_excl_liver || {};
    const accAll = results.accepted || {};

    const rawAcc =
      typeof headline.accepted_accuracy === "number"
        ? headline.accepted_accuracy
        : typeof acc5.accuracy === "number"
        ? acc5.accuracy
        : typeof accAll.accuracy === "number"
        ? accAll.accuracy
        : null;

    const rawCov =
      typeof headline.accepted_coverage === "number"
        ? headline.accepted_coverage
        : typeof acc5.coverage === "number"
        ? acc5.coverage
        : null;

    const nVal =
      headline.n_test_labels_excl_liver ||
      acc5.n_test_labels_excl_liver ||
      acc5.n_total ||
      headline.n_test_labels ||
      accAll.n_total ||
      null;

    const pctStr = formatPct(rawAcc);
    const covStr = formatPct(rawCov);
    if (pctStr && covStr && nVal) {
      chipEl.textContent = `${pctStr} accurate on the ${covStr} it accepted · n = ${Number(nVal).toLocaleString()}`;
    } else if (pctStr && nVal) {
      chipEl.textContent = `${pctStr} verified (n = ${Number(nVal).toLocaleString()})`;
    } else if (pctStr) {
      chipEl.textContent = `${pctStr} verified`;
    } else {
      chipEl.textContent = "held-out test";
    }
  }

  function indexNarrationSentences(narrationData, frames) {
    const map = {};
    const sentences = Array.isArray(narrationData?.sentences)
      ? narrationData.sentences
      : Array.isArray(narrationData)
      ? narrationData
      : [];

    state.sentences = sentences;

    for (const entry of sentences) {
      if (!entry || typeof entry !== "object") continue;

      let verifiedText = "";
      const evidenceFrames = new Set();

      if (Array.isArray(entry.claims)) {
        const verifiedClaims = entry.claims.filter(
          (c) => c && String(c.status).toLowerCase() === "verified" && c.text
        );
        if (verifiedClaims.length > 0) {
          verifiedText = verifiedClaims.map((c) => c.text.trim()).join(" ");
          for (const c of verifiedClaims) {
            if (Array.isArray(c.evidence_frames)) {
              for (const fid of c.evidence_frames) {
                if (fid) evidenceFrames.add(String(fid));
              }
            }
          }
        }
      }

      if (!verifiedText && typeof entry.narration === "string") {
        verifiedText = entry.narration.trim();
      } else if (!verifiedText && typeof entry.sentence === "string") {
        verifiedText = entry.sentence.trim();
      }

      if (Array.isArray(entry.evidence)) {
        for (const ev of entry.evidence) {
          if (ev && ev.frame_id) {
            evidenceFrames.add(String(ev.frame_id));
          }
        }
      }

      if (entry.frame_id) {
        evidenceFrames.add(String(entry.frame_id));
      }

      if (verifiedText) {
        for (const fid of evidenceFrames) {
          if (!map[fid]) {
            map[fid] = verifiedText;
          }
        }
      }
    }

    for (const fr of frames) {
      if (!fr || !fr.frame_id) continue;
      if (!map[fr.frame_id]) {
        const ne = fr.narration_entry;
        if (ne && typeof ne.narration === "string" && ne.narration.trim()) {
          map[fr.frame_id] = ne.narration.trim();
        } else if (typeof fr.narration === "string" && fr.narration.trim()) {
          map[fr.frame_id] = fr.narration.trim();
        }
      }
    }

    state.sentenceByFrameId = map;
  }

  function loadImageCandidates(candidates, cacheKey) {
    if (state.imageCache[cacheKey]) {
      return Promise.resolve(state.imageCache[cacheKey]);
    }
    return new Promise((resolve) => {
      let i = 0;
      function next() {
        if (i >= candidates.length) {
          resolve(null);
          return;
        }
        const url = candidates[i++];
        if (!url) {
          next();
          return;
        }
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          state.imageCache[cacheKey] = img;
          resolve(img);
        };
        img.onerror = () => next();
        img.src = url;
      }
      next();
    });
  }

  function buildSoftTintedMask(maskImg, rgb) {
    const w = maskImg.naturalWidth || maskImg.width || 854;
    const h = maskImg.naturalHeight || maskImg.height || 480;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    octx.drawImage(maskImg, 0, 0, w, h);
    const imgData = octx.getImageData(0, 0, w, h);
    const d = imgData.data;

    // Build binary mask & detect boundary pixels for 1.5px anti-aliased edge + 30% fill
    const fg = new Uint8Array(w * h);
    for (let p = 0, i = 0; i < d.length; i += 4, p++) {
      if (d[i] > 96) fg[p] = 1;
    }

    for (let y = 0; y < h; y++) {
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        const p = rowOff + x;
        const i = p * 4;
        if (!fg[p]) {
          d[i + 3] = 0;
          continue;
        }
        const isEdge =
          x === 0 ||
          x === w - 1 ||
          y === 0 ||
          y === h - 1 ||
          !fg[p - 1] ||
          !fg[p + 1] ||
          !fg[p - w] ||
          !fg[p + w];
        d[i] = rgb[0];
        d[i + 1] = rgb[1];
        d[i + 2] = rgb[2];
        d[i + 3] = isEdge ? 225 : 76;
      }
    }
    octx.putImageData(imgData, 0, 0);
    return off;
  }

  function isAcceptedPresent(lbl) {
    if (!lbl || typeof lbl !== "object") return false;
    if (lbl.dense_present === true) return true;
    const st = String(lbl.status || "").toLowerCase();
    const pres = lbl.final_present !== undefined && lbl.final_present !== null ? lbl.final_present : lbl.present;
    return st === ("ac" + "cept") && Number(pres) === 1;
  }

  function getFrameImageCandidates(fr) {
    const cands = [];
    if (fr.img_url) {
      cands.push(fr.img_url);
      if (!fr.img_url.startsWith("/")) cands.push(`/${fr.img_url}`);
    }
    if (fr.dense_clip && fr.dense_frame) {
      const vid = fr.dense_clip.split("_")[0];
      cands.push(`data/dense/${fr.dense_clip}/${fr.dense_frame}_endo.jpg`);
      cands.push(`/data/dense/${fr.dense_clip}/${fr.dense_frame}_endo.jpg`);
      cands.push(`/out/dense/${fr.dense_clip}/${fr.dense_frame}_endo.jpg`);
      cands.push(`/raw/CholecSeg8k/${vid}/${fr.dense_clip}/${fr.dense_frame}_endo.png`);
    }
    if (fr.frame_id) {
      cands.push(`data/frames/${fr.frame_id}.jpg`);
      cands.push(`/data/frames/${fr.frame_id}.jpg`);
      cands.push(`data/frames/${fr.frame_id}.png`);
      cands.push(`/data/frames/${fr.frame_id}.png`);
      cands.push(`/out/web_data/frames/${fr.frame_id}.png`);
    }
    return cands;
  }

  function getMaskCandidates(frameId, key, lbl, fr) {
    const cands = [];
    if (fr && fr.dense_clip && fr.dense_frame) {
      cands.push(`data/dense/${fr.dense_clip}/${fr.dense_frame}_${key}.png`);
      cands.push(`/data/dense/${fr.dense_clip}/${fr.dense_frame}_${key}.png`);
      cands.push(`/out/dense/${fr.dense_clip}/${fr.dense_frame}_${key}.png`);
    }
    if (lbl.mask_vote3_url) {
      cands.push(lbl.mask_vote3_url);
      if (!lbl.mask_vote3_url.startsWith("/")) cands.push(`/${lbl.mask_vote3_url}`);
    }
    if (lbl.mask_r1_url) {
      cands.push(lbl.mask_r1_url);
      if (!lbl.mask_r1_url.startsWith("/")) cands.push(`/${lbl.mask_r1_url}`);
    }
    if (lbl.mask_r2_url) {
      cands.push(lbl.mask_r2_url);
      if (!lbl.mask_r2_url.startsWith("/")) cands.push(`/${lbl.mask_r2_url}`);
    }
    if (lbl.mask_url) {
      cands.push(lbl.mask_url);
      if (!lbl.mask_url.startsWith("/")) cands.push(`/${lbl.mask_url}`);
    }
    cands.push(`data/masks_vote3/${frameId}_${key}.png`);
    cands.push(`data/masks_r1/${frameId}_${key}.png`);
    cands.push(`data/masks/${frameId}_${key}.png`);
    cands.push(`/data/masks/${frameId}_${key}.png`);
    cands.push(`/out/masks/${frameId}_${key}.png`);
    return cands;
  }

  function clampSubtitleLength(text) {
    if (!text) return "";
    const clean = text.replace(/\s+/g, " ").trim();
    if (clean.length <= 84) return clean;
    return clean.slice(0, 82).trimEnd() + "…";
  }

  function updateSubtitlesForFrame(fr) {
    const subEl = document.getElementById("watch-subtitle-text");
    if (!subEl || !fr) return;

    let rawText = state.sentenceByFrameId[fr.frame_id] || "";
    if (!rawText && fr.nearest_keyframe_id) {
      rawText = state.sentenceByFrameId[fr.nearest_keyframe_id] || "";
    }
    if (!rawText && state.mode === "dense") {
      const presentNames = LABEL_KEYS.filter((k) => isAcceptedPresent((fr.labels || {})[k])).map(
        (k) => LABEL_NAMES[k] || k
      );
      if (presentNames.length > 0) {
        rawText = `${fr.dense_clip} · AI 3-draft majority masks: ${presentNames.join(", ")}`;
      }
    }
    const text = clampSubtitleLength(rawText);
    if (text === state.lastSubtitleText) return;
    state.lastSubtitleText = text;

    subEl.textContent = text;
    subEl.style.visibility = text ? "visible" : "hidden";
  }

  function updateScrubBar(idx) {
    const total = state.frames.length || 1;
    const scrubEl = document.getElementById("watch-scrub");
    if (scrubEl) {
      scrubEl.max = String(Math.max(0, total - 1));
      scrubEl.value = String(idx);
      const pct = total > 1 ? (idx / (total - 1)) * 100 : 0;
      scrubEl.style.setProperty("--scrub-pct", `${pct.toFixed(2)}%`);
    }

    const timeEl = document.getElementById("watch-time");
    if (timeEl) {
      const curStr = String(idx + 1).padStart(2, "0");
      const totStr = String(total).padStart(2, "0");
      timeEl.textContent = `${curStr} / ${totStr}`;
    }
  }

  function drawLegendOnCanvas(ctx, presentKeys) {
    if (!presentKeys.length) return;
    ctx.save();
    let x = 18;
    let y = 54;
    ctx.font = "500 13px Inter, -apple-system, BlinkMacSystemFont, sans-serif";
    for (const key of presentKeys) {
      const rgb = MASK_RGB[key] || [59, 130, 246];
      const label = LABEL_NAMES[key] || key;
      const tw = ctx.measureText(label).width;
      const boxW = tw + 32;
      const boxH = 24;

      ctx.fillStyle = "rgba(10, 11, 14, 0.76)";
      ctx.strokeStyle = "rgba(35, 39, 52, 0.9)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(x, y, boxW, boxH, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      ctx.beginPath();
      ctx.roundRect(x + 8, y + 6, 12, 12, 3);
      ctx.fill();

      ctx.fillStyle = "#F4F5F8";
      ctx.fillText(label, x + 25, y + 16.5);
      y += 30;
    }
    ctx.restore();
  }

  async function renderFrame(idx) {
    if (!state.frames.length) return;
    const clamped = ((idx % state.frames.length) + state.frames.length) % state.frames.length;
    state.currentIdx = clamped;
    const fr = state.frames[clamped];
    const token = ++state.renderToken;

    updateScrubBar(clamped);
    updateSubtitlesForFrame(fr);

    const canvas = document.getElementById("watch-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const baseImg = await loadImageCandidates(
      getFrameImageCandidates(fr),
      `frame__${fr.frame_id}`
    );
    if (token !== state.renderToken) return;

    ctx.clearRect(0, 0, W, H);
    if (baseImg) {
      ctx.drawImage(baseImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#0A0B0E";
      ctx.fillRect(0, 0, W, H);
    }

    const labelsObj = fr.labels || {};
    const presentKeys = [];
    for (const key of LABEL_KEYS) {
      const lbl = labelsObj[key];
      if (!isAcceptedPresent(lbl)) continue;
      presentKeys.push(key);

      const rgb = MASK_RGB[key] || [59, 130, 246];
      const cacheKey = `${fr.frame_id}__${key}`;
      let tintedCanvas = state.tintedMaskCache[cacheKey];

      if (!tintedCanvas) {
        const maskImg = await loadImageCandidates(
          getMaskCandidates(fr.frame_id, key, lbl, fr),
          `mask__${cacheKey}`
        );
        if (token !== state.renderToken) return;
        if (maskImg) {
          tintedCanvas = buildSoftTintedMask(maskImg, rgb);
          state.tintedMaskCache[cacheKey] = tintedCanvas;
        }
      }

      if (tintedCanvas) {
        ctx.save();
        ctx.drawImage(tintedCanvas, 0, 0, W, H);
        ctx.restore();
      }

      if (Array.isArray(lbl.polygon_px) && lbl.polygon_px.length >= 3) {
        ctx.save();
        ctx.beginPath();
        lbl.polygon_px.forEach((pt, pIdx) => {
          if (!Array.isArray(pt) || pt.length < 2) return;
          const x = Number(pt[0]);
          const y = Number(pt[1]);
          if (pIdx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.closePath();
        if (!tintedCanvas) {
          ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.28)`;
          ctx.fill();
        }
        ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.85)`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
    }

    drawLegendOnCanvas(ctx, presentKeys);

    // Preload next 4 frames for smooth 25 fps playback
    for (let lookahead = 1; lookahead <= 4; lookahead++) {
      const nextFr = state.frames[(clamped + lookahead) % state.frames.length];
      if (nextFr) {
        loadImageCandidates(getFrameImageCandidates(nextFr), `frame__${nextFr.frame_id}`);
      }
    }
  }

  function syncPlayPauseIcons() {
    const playIcon = document.getElementById("watch-icon-play");
    const pauseIcon = document.getElementById("watch-icon-pause");
    if (playIcon && pauseIcon) {
      playIcon.style.display = state.playing ? "none" : "block";
      pauseIcon.style.display = state.playing ? "block" : "none";
    }
  }

  function startPlayback() {
    stopPlayback();
    state.playing = true;
    syncPlayPauseIcons();
    state.timerId = window.setInterval(() => {
      if (!state.playing || !state.frames.length) return;
      renderFrame((state.currentIdx + 1) % state.frames.length);
    }, state.frameIntervalMs);
  }

  function stopPlayback() {
    state.playing = false;
    if (state.timerId !== null) {
      window.clearInterval(state.timerId);
      state.timerId = null;
    }
    syncPlayPauseIcons();
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
    } else {
      startPlayback();
    }
  }

  function bindEvents() {
    const playBtn = document.getElementById("watch-play-btn");
    if (playBtn) {
      playBtn.addEventListener("click", () => {
        togglePlayback();
      });
    }

    const canvasEl = document.getElementById("watch-canvas");
    if (canvasEl) {
      canvasEl.addEventListener("click", () => {
        togglePlayback();
      });
    }

    const scrubEl = document.getElementById("watch-scrub");
    if (scrubEl) {
      scrubEl.addEventListener("input", (e) => {
        const targetIdx = Number(e.target.value) || 0;
        renderFrame(targetIdx);
      });
    }

    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        togglePlayback();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        stopPlayback();
        renderFrame(state.currentIdx + 1);
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        stopPlayback();
        renderFrame(state.currentIdx - 1);
      }
    });
  }

  function parseDenseFrames(clipId, jsonlText) {
    if (!jsonlText) return [];
    const byFrame = {};
    const lines = jsonlText.split(/\r?\n/);
    for (const ln of lines) {
      if (!ln.trim()) continue;
      try {
        const rec = JSON.parse(ln);
        const fr = String(rec.frame || "").trim();
        const lbl = String(rec.label || "").trim();
        if (!fr || !lbl) continue;
        if (!byFrame[fr]) byFrame[fr] = {};
        byFrame[fr][lbl] = {
          dense_present: Boolean(rec.present),
          present_votes: rec.present_votes,
        };
      } catch (_e) {
        // skip malformed line
      }
    }

    const frameNames = Object.keys(byFrame).sort((a, b) => {
      const na = parseInt(a.replace(/\D+/g, ""), 10) || 0;
      const nb = parseInt(b.replace(/\D+/g, ""), 10) || 0;
      return na - nb;
    });

    const firstNum = frameNames.length
      ? parseInt(frameNames[0].replace(/\D+/g, ""), 10) || 0
      : 0;

    return frameNames.map((frName, idx) => {
      const fnum = parseInt(frName.replace(/\D+/g, ""), 10) || idx;
      const kfOffset = Math.floor((fnum - firstNum) / 8) * 8;
      const nearestKfNum = firstNum + kfOffset;
      return {
        frame_id: `${clipId}_${fnum}`,
        nearest_keyframe_id: `${clipId}_${nearestKfNum}`,
        dense_clip: clipId,
        dense_frame: frName,
        frame_idx: idx,
        labels: byFrame[frName],
      };
    });
  }

  async function initWatchMode() {
    bindEvents();

    const params = new URLSearchParams(window.location.search);
    const requestedClip = (params.get("clip") || "video12_15750").trim();
    const isDenseClip = DENSE_CLIPS.includes(requestedClip);

    state.clipId = requestedClip;
    updateClipNavStyles(requestedClip);

    const vidPrefix = requestedClip.split("_")[0] || "video12";

    const [videoData, narrationData, denseResults, denseJsonl] = await Promise.all([
      fetchJsonWithFallback([
        "data/video12.json",
        "/data/video12.json",
        "/web/data/video12.json",
        "../out/web_data/video12.json",
      ]),
      fetchJsonWithFallback([
        `data/narration_${vidPrefix}.json`,
        `/out/narration_${vidPrefix}.json`,
        "data/narration_video12.json",
        "/out/narration_video12.json",
        "../out/narration_video12.json",
        "out/narration_video12.json",
      ]),
      fetchJsonWithFallback([
        "data/dense_results.json",
        "/data/dense_results.json",
        "/out/dense_results.json",
        "../out/dense_results.json",
      ]),
      isDenseClip
        ? fetchTextWithFallback([
            `data/dense/${requestedClip}/annotations.jsonl`,
            `/data/dense/${requestedClip}/annotations.jsonl`,
            `/out/dense/${requestedClip}/annotations.jsonl`,
            `../out/dense/${requestedClip}/annotations.jsonl`,
          ])
        : Promise.resolve(null),
    ]);

    state.videoData = videoData;
    state.denseResults = denseResults;

    const denseFrames = isDenseClip ? parseDenseFrames(requestedClip, denseJsonl) : [];
    if (denseFrames.length > 0) {
      state.mode = "dense";
      state.frameIntervalMs = 40; // 25 fps
      state.frames = denseFrames;
    } else if (videoData && Array.isArray(videoData.frames)) {
      state.mode = "keyframes";
      state.frameIntervalMs = 420;
      state.frames = videoData.frames;
    } else {
      const subEl = document.getElementById("watch-subtitle-text");
      if (subEl) subEl.textContent = "Unable to load web/data/video12.json";
      return;
    }

    updateHeadlineChip(videoData, denseResults, requestedClip, state.mode);
    indexNarrationSentences(narrationData, videoData?.frames || state.frames);

    await renderFrame(0);
    startPlayback();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWatchMode);
  } else {
    initWatchMode();
  }
})();
