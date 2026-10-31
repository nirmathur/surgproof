/* SurgProof — Watch Mode Player (web/watch.js)
 *
 * Full-bleed dark player showing held-out video12 frames in sequence with
 * the AI's accepted masks softly overlaid (from web/data/video12.json),
 * bottom play/pause/scrub bar, and bottom-centre verified narration subtitles
 * (from out/narration_video12.json, each shown while its evidence frames play).
 *
 * Zero ground truth used or displayed; all numbers and masks read strictly
 * from real exported pipeline artifacts.
 */
(function () {
  "use strict";

  const LABEL_KEYS = ["grasper", "l_hook", "liver", "gallbladder", "cystic_duct", "blood"];

  // Calm anatomical / instrument palette for soft mask overlays
  const MASK_RGB = {
    gallbladder: [16, 185, 129],  // emerald
    l_hook: [59, 130, 246],       // surgical blue
    grasper: [56, 189, 248],      // sky cyan
    cystic_duct: [245, 158, 11],  // warm gold
    liver: [139, 92, 246],        // muted violet
    blood: [239, 68, 68],         // soft crimson
  };

  const state = {
    frames: [],
    sentences: [],
    sentenceByFrameId: {},
    currentIdx: 0,
    playing: true,
    timerId: null,
    frameIntervalMs: 420,
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

  function formatPct(val) {
    if (typeof val !== "number" || !Number.isFinite(val)) return null;
    const pct = val <= 1.0 ? val * 100 : val;
    return `${pct.toFixed(1)}%`;
  }

  function updateHeadlineChip(videoData) {
    const chipEl = document.getElementById("watch-headline-chip");
    if (!chipEl || !videoData) return;

    const headline = videoData.headline || {};
    const results = videoData.results || {};
    const acc5 = results.accepted_5_informative || {};
    const accAll = results.accepted || {};

    const rawAcc =
      typeof headline.accepted_accuracy === "number"
        ? headline.accepted_accuracy
        : typeof acc5.accuracy === "number"
        ? acc5.accuracy
        : typeof accAll.accuracy === "number"
        ? accAll.accuracy
        : null;

    const nVal =
      headline.n_test_labels_excl_liver ||
      acc5.n_test_labels_excl_liver ||
      acc5.n_accepted_excl_liver ||
      headline.n_test_labels ||
      accAll.n_total ||
      null;

    const pctStr = formatPct(rawAcc);
    if (pctStr && nVal) {
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

      // Prefer verified claim text if present, otherwise entry.narration
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

    // Fallback to per-frame narration embedded in web/data/video12.json
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

    for (let i = 0; i < d.length; i += 4) {
      const lum = d[i];
      if (lum > 96) {
        // Soft anatomical tint where mask is foreground; transparent background
        const norm = lum / 255;
        d[i] = rgb[0];
        d[i + 1] = rgb[1];
        d[i + 2] = rgb[2];
        d[i + 3] = Math.round(68 * norm);
      } else {
        d[i + 3] = 0;
      }
    }
    octx.putImageData(imgData, 0, 0);
    return off;
  }

  function isAcceptedPresent(lbl) {
    if (!lbl || typeof lbl !== "object") return false;
    const st = String(lbl.status || "").toLowerCase();
    const pres = lbl.final_present !== undefined && lbl.final_present !== null ? lbl.final_present : lbl.present;
    return st === "accept" && Number(pres) === 1;
  }

  function getFrameImageCandidates(fr) {
    const cands = [];
    if (fr.img_url) {
      cands.push(fr.img_url);
      if (!fr.img_url.startsWith("/")) cands.push(`/${fr.img_url}`);
    }
    if (fr.frame_id) {
      cands.push(`data/frames/${fr.frame_id}.png`);
      cands.push(`/data/frames/${fr.frame_id}.png`);
      cands.push(`/out/web_data/frames/${fr.frame_id}.png`);
    }
    return cands;
  }

  function getMaskCandidates(frameId, key, lbl) {
    const cands = [];
    if (lbl.mask_r2_url) cands.push(lbl.mask_r2_url);
    if (lbl.mask_r1_url) cands.push(lbl.mask_r1_url);
    if (lbl.mask_url) {
      cands.push(lbl.mask_url);
      if (!lbl.mask_url.startsWith("/")) cands.push(`/${lbl.mask_url}`);
    }
    cands.push(`data/masks/${frameId}_${key}.png`);
    cands.push(`/data/masks/${frameId}_${key}.png`);
    cands.push(`/out/masks/${frameId}_${key}.png`);
    return cands;
  }

  function updateSubtitlesForFrame(fr) {
    const subEl = document.getElementById("watch-subtitle-text");
    if (!subEl || !fr) return;

    const text = state.sentenceByFrameId[fr.frame_id] || "";
    if (text === state.lastSubtitleText) return;
    state.lastSubtitleText = text;

    subEl.classList.add("sub-fading");
    window.setTimeout(() => {
      subEl.textContent = text;
      subEl.style.visibility = text ? "visible" : "hidden";
      subEl.classList.remove("sub-fading");
    }, 110);
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

    // Softly overlay AI's accepted masks from web/data/video12.json
    const labelsObj = fr.labels || {};
    for (const key of LABEL_KEYS) {
      const lbl = labelsObj[key];
      if (!isAcceptedPresent(lbl)) continue;

      const rgb = MASK_RGB[key] || [59, 130, 246];
      const cacheKey = `${fr.frame_id}__${key}`;
      let tintedCanvas = state.tintedMaskCache[cacheKey];

      if (!tintedCanvas) {
        const maskImg = await loadImageCandidates(
          getMaskCandidates(fr.frame_id, key, lbl),
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

      // Soft contour stroke if polygon_px is present for this accepted mask
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
          ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.22)`;
          ctx.fill();
        }
        ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.58)`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
    }

    // Preload next frame softly in background
    const nextFr = state.frames[(clamped + 1) % state.frames.length];
    if (nextFr) {
      loadImageCandidates(getFrameImageCandidates(nextFr), `frame__${nextFr.frame_id}`);
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

  async function initWatchMode() {
    bindEvents();

    const [videoData, narrationData] = await Promise.all([
      fetchJsonWithFallback([
        "data/video12.json",
        "/data/video12.json",
        "/web/data/video12.json",
        "../out/web_data/video12.json",
      ]),
      fetchJsonWithFallback([
        "/out/narration_video12.json",
        "../out/narration_video12.json",
        "out/narration_video12.json",
      ]),
    ]);

    if (!videoData || !Array.isArray(videoData.frames)) {
      const subEl = document.getElementById("watch-subtitle-text");
      if (subEl) subEl.textContent = "Unable to load web/data/video12.json";
      return;
    }

    state.frames = videoData.frames;
    updateHeadlineChip(videoData);
    indexNarrationSentences(narrationData, state.frames);

    await renderFrame(0);
    startPlayback();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initWatchMode);
  } else {
    initWatchMode();
  }
})();
