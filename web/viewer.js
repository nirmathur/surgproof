/* SurgProof <canvas> Surgical Frame & AI Annotation Viewport (web/viewer.js)
 * Lane: nir-webcanvas
 * Renders the real surgical frame image and overlays real AI annotations from web/data/video12.json
 * and live updates from web/data/live.json:
 *   - Semi-transparent per-label AI segmentation masks (PNG/base64, polygons, or RLE) with contour edge
 *   - AI bounding boxes [ymin, xmin, ymax, xmax] (0-1000 normalized) coloured by verdict
 *   - Instrument tool-tip points [y, x] (0-1000 normalized) with precision crosshair & target ring
 *   - Verdict label chips coloured by real status (accept green #10B981, AMBER #F59E0B, reject red #EF4444)
 *   - Hover inspection card showing label, verdict, confidence, IoU, verbatim evidence, and Amber loop rounds
 *   - Wheel zoom (1.0x - 8.0x anchored at cursor) and click-drag pan + double-click / button reset
 *   - 600ms cubic ease-out draw-in / fade-in animation when newly arrived annotations stream in from live.json
 *   - Zero fabricated data: any frame image, phase, verdict, or spatial annotation not yet produced shows "pending".
 */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) {
    const instance = factory();
    module.exports = instance;
    if (typeof window !== "undefined") {
      window.Viewer = instance;
      window.SurgProofViewer = instance;
    }
  } else {
    const instance = factory();
    root.Viewer = instance;
    root.SurgProofViewer = instance;
  }
})(typeof globalThis !== "undefined" ? globalThis : typeof window !== "undefined" ? window : this, function () {
  "use strict";

  const NATIVE_W = 854;
  const NATIVE_H = 480;
  const ANIM_DURATION_MS = 600;

  const DEFAULT_LABEL_KEYS = ["grasper", "l_hook", "gallbladder", "cystic_duct", "liver", "blood"];

  const LABEL_NAMES = {
    grasper: "Grasper (Instrument)",
    l_hook: "L-Hook Electrocautery",
    gallbladder: "Gallbladder",
    cystic_duct: "Cystic Duct",
    liver: "Liver",
    blood: "Free Blood / Clot",
  };

  // Per-label segmentation mask colours (distinct anatomical & instrument palette)
  const LABEL_RGB = {
    grasper: [56, 189, 248],     // #38BDF8 cyan-blue instrument
    l_hook: [168, 85, 247],      // #A855F7 violet instrument
    gallbladder: [16, 185, 129], // #10B981 emerald organ
    cystic_duct: [250, 204, 21], // #FACC15 gold critical structure
    liver: [249, 115, 22],       // #F97316 warm hepatic terracotta
    blood: [239, 68, 68],        // #EF4444 crimson
  };

  // Verdict design tokens (matching promo/DESIGN.md)
  const VERDICT_TOKENS = {
    accept: {
      stroke: "#10B981",
      fill: "#092E24",
      text: "#F4F5F8",
      badgeText: "#10B981",
      rgb: [16, 185, 129],
      label: "accept",
    },
    AMBER: {
      stroke: "#F59E0B",
      fill: "#362407",
      text: "#F4F5F8",
      badgeText: "#F59E0B",
      rgb: [245, 158, 11],
      label: "AMBER",
    },
    reject: {
      stroke: "#EF4444",
      fill: "#331014",
      text: "#F4F5F8",
      badgeText: "#EF4444",
      rgb: [239, 68, 68],
      label: "reject",
    },
    pending: {
      stroke: "#3B82F6",
      fill: "#12141C",
      text: "#8A909E",
      badgeText: "#8A909E",
      rgb: [59, 130, 246],
      label: "pending",
    },
  };

  // CholecSeg8k watershed pixel value -> label key (used strictly for post-hoc Scoring View GT overlay toggle)
  const WATERSHED_TO_KEY = {
    21: "liver",
    31: "grasper",
    24: "blood",
    25: "cystic_duct",
    32: "l_hook",
    22: "gallbladder",
    // Direct class_id fallback if mask already mapped to class IDs (2, 5, 7, 8, 9, 10)
    2: "liver",
    5: "grasper",
    7: "blood",
    8: "cystic_duct",
    9: "l_hook",
    10: "gallbladder",
  };

  function easeOutCubic(t) {
    const clamped = Math.max(0, Math.min(1, t));
    return 1 - Math.pow(1 - clamped, 3);
  }

  function normalizeVerdict(raw) {
    if (raw === null || raw === undefined) return "pending";
    const s = String(raw).trim();
    if (!s) return "pending";
    const low = s.toLowerCase();
    if (low === "accept" || low === "accepted" || low === "verified") return "accept";
    if (low === "reject" || low === "rejected" || low === "escalate") return "reject";
    if (low === "amber" || low === "unsure" || low === "uncertain") return "AMBER";
    if (low === "pending" || low === "none" || low === "null") return "pending";
    return "AMBER";
  }

  function formatConfidence(val) {
    if (typeof val !== "number" || !Number.isFinite(val)) return "pending";
    return val.toFixed(2);
  }

  function parseFrameNum(frameId) {
    if (!frameId) return 0;
    const parts = String(frameId).split("_");
    const last = parseInt(parts[parts.length - 1], 10);
    return Number.isFinite(last) ? last : 0;
  }

  function deriveCandidateUrls(frameId) {
    const fid = String(frameId || "");
    const parts = fid.split("_");
    if (parts.length === 3) {
      const vid = parts[0];
      const clip = parts[0] + "_" + parts[1];
      const fnum = parts[2];
      return {
        imgCandidates: [
          `data/frames/${fid}.jpg`,
          `/web/data/frames/${fid}.jpg`,
          `data/frames/${fid}.png`,
          `/web/data/frames/${fid}.png`,
          `/data/frames/${fid}.png`,
          `/raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `../raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `/data/cholecseg8k/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
        ],
        gtCandidates: [
          `data/gt_masks/${fid}.png`,
          `/web/data/gt_masks/${fid}.png`,
          `/data/gt_masks/${fid}.png`,
          `/raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
          `../raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
          `/data/cholecseg8k/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
        ],
      };
    }
    return {
      imgCandidates: fid ? [`data/frames/${fid}.png`, `/web/data/frames/${fid}.png`] : [],
      gtCandidates: fid ? [`data/gt_masks/${fid}.png`, `/web/data/gt_masks/${fid}.png`] : [],
    };
  }

  function normalizeBoxCoords(rawBox) {
    if (!Array.isArray(rawBox) || rawBox.length < 4) return null;
    const nums = rawBox.slice(0, 4).map(Number);
    if (nums.some((n) => !Number.isFinite(n))) return null;
    let [y1, x1, y2, x2] = nums;
    const maxVal = Math.max(Math.abs(y1), Math.abs(x1), Math.abs(y2), Math.abs(x2));
    if (maxVal === 0) return null;
    // Auto-detect 0..1 normalized vs 0..1000 normalized
    if (maxVal <= 1.0) {
      y1 *= 1000;
      x1 *= 1000;
      y2 *= 1000;
      x2 *= 1000;
    }
    const ymin = Math.max(0, Math.min(1000, Math.min(y1, y2)));
    const ymax = Math.max(0, Math.min(1000, Math.max(y1, y2)));
    const xmin = Math.max(0, Math.min(1000, Math.min(x1, x2)));
    const xmax = Math.max(0, Math.min(1000, Math.max(x1, x2)));
    if (ymax <= ymin || xmax <= xmin) return null;
    return [ymin, xmin, ymax, xmax];
  }

  function normalizeTipCoords(rawTip) {
    if (!Array.isArray(rawTip) || rawTip.length < 2) return null;
    const y = Number(rawTip[0]);
    const x = Number(rawTip[1]);
    if (!Number.isFinite(y) || !Number.isFinite(x)) return null;
    if (y === 0 && x === 0) return null;
    const maxVal = Math.max(Math.abs(y), Math.abs(x));
    const ny = maxVal <= 1.0 ? y * 1000 : y;
    const nx = maxVal <= 1.0 ? x * 1000 : x;
    return [Math.max(0, Math.min(1000, ny)), Math.max(0, Math.min(1000, nx))];
  }

  function normalizePolygon(rawPoly) {
    if (!Array.isArray(rawPoly) || rawPoly.length === 0) return null;
    // Support either single polygon [[y, x], ...] or multi-polygon [[[y, x], ...], ...]
    const polys = Array.isArray(rawPoly[0]) && Array.isArray(rawPoly[0][0]) ? rawPoly : [rawPoly];
    const out = [];
    for (const ring of polys) {
      if (!Array.isArray(ring) || ring.length < 3) continue;
      let maxVal = 0;
      const pts = [];
      for (const pt of ring) {
        if (Array.isArray(pt) && pt.length >= 2) {
          const py = Number(pt[0]);
          const px = Number(pt[1]);
          if (Number.isFinite(py) && Number.isFinite(px)) {
            maxVal = Math.max(maxVal, Math.abs(py), Math.abs(px));
            pts.push([py, px]);
          }
        } else if (pt && typeof pt === "object" && Number.isFinite(Number(pt.y)) && Number.isFinite(Number(pt.x))) {
          const py = Number(pt.y);
          const px = Number(pt.x);
          maxVal = Math.max(maxVal, Math.abs(py), Math.abs(px));
          pts.push([py, px]);
        }
      }
      if (pts.length >= 3) {
        const scale = maxVal > 0 && maxVal <= 1.0 ? 1000 : 1;
        out.push(pts.map(([py, px]) => [py * scale, px * scale]));
      }
    }
    return out.length > 0 ? out : null;
  }

  // Internal state of the Viewer singleton
  const state = {
    mounted: false,
    rootEl: null,
    stageEl: null,
    canvas: null,
    ctx: null,
    tooltipEl: null,
    zoomPillEl: null,
    zoomLabelEl: null,
    toolbarEl: null,
    options: {},

    video: "video12",
    labelKeys: [...DEFAULT_LABEL_KEYS],
    frames: [],
    frameById: {},
    currentFrameIdx: 0,
    hasReceivedData: false,

    // Layer visibility flags
    layers: {
      masks: true,
      boxes: true,
      dualPass: false,
      tips: true,
      chips: true,
      gt: false,
    },

    // Interactive state
    selectedKey: null,
    hoveredKey: null,
    hoverClientX: 0,
    hoverClientY: 0,
    zoom: 1.0,
    panX: 0,
    panY: 0,
    isDragging: false,
    dragMoved: false,
    dragStartX: 0,
    dragStartY: 0,
    dragOriginPanX: 0,
    dragOriginPanY: 0,

    // Hit-test regions populated on each render pass
    hitRegions: [],
    hudChipRects: [],

    // Caches for images & tinted offscreen mask canvases
    imageCache: {},
    maskRawCache: {},
    maskTintedCache: {},
    gtCanvasCache: {},
    gateDecisions: {},

    // Live animation tracking: key `${frameId}__${label}` -> { startMs, durationMs, kind }
    seenSignatures: new Map(),
    activeAnimations: new Map(),
    rafId: null,
  };

  function injectStyles() {
    if (typeof document === "undefined") return;
    if (document.getElementById("sp-viewer-styles")) return;
    const style = document.createElement("style");
    style.id = "sp-viewer-styles";
    style.textContent = `
      .sp-viewer-host {
        position: relative;
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 100%;
        min-height: 320px;
        background: #0A0B0E;
        color: #F4F5F8;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Inter", "Helvetica Neue", sans-serif;
        user-select: none;
        overflow: hidden;
      }
      .sp-viewer-toolbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 6px 12px;
        background: #12141C;
        border-bottom: 1px solid #232734;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
        z-index: 6;
      }
      .sp-viewer-toolbar-group {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }
      .sp-viewer-chip-toggle {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 8px;
        border-radius: 6px;
        background: #0A0B0E;
        border: 1px solid #232734;
        color: #D8DCE6;
        font-size: 11.5px;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease;
      }
      .sp-viewer-chip-toggle:hover {
        border-color: #3B82F6;
      }
      .sp-viewer-chip-toggle input {
        accent-color: #10B981;
        cursor: pointer;
      }
      .sp-viewer-chip-toggle.sp-gt-toggle {
        border-color: rgba(168, 85, 247, 0.45);
        color: #D8B4FE;
      }
      .sp-viewer-stage {
        position: relative;
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #0A0B0E;
        overflow: hidden;
        min-height: 260px;
      }
      .sp-viewer-canvas {
        max-width: 100%;
        max-height: 100%;
        width: auto;
        height: auto;
        aspect-ratio: 854 / 480;
        border-radius: 8px;
        border: 1px solid #232734;
        background: #0A0B0E;
        cursor: grab;
        display: block;
      }
      .sp-viewer-canvas.sp-dragging {
        cursor: grabbing;
      }
      .sp-viewer-canvas.sp-hovering-target {
        cursor: pointer;
      }
      .sp-viewer-zoom-pill {
        position: absolute;
        right: 14px;
        bottom: 14px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        border-radius: 6px;
        background: rgba(18, 20, 28, 0.92);
        border: 1px solid #232734;
        color: #D8DCE6;
        font-size: 11px;
        font-family: "SF Mono", "JetBrains Mono", ui-monospace, Menlo, monospace;
        z-index: 8;
      }
      .sp-viewer-zoom-btn {
        background: #0A0B0E;
        color: #F4F5F8;
        border: 1px solid #232734;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 11px;
        font-family: inherit;
        cursor: pointer;
        line-height: 1.2;
      }
      .sp-viewer-zoom-btn:hover {
        border-color: #3B82F6;
        color: #3B82F6;
      }
      .sp-viewer-tooltip {
        position: fixed;
        pointer-events: none;
        z-index: 9999;
        width: 320px;
        max-width: 90vw;
        background: #12141C;
        border: 1px solid #232734;
        border-radius: 8px;
        padding: 10px 12px;
        color: #F4F5F8;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif;
        font-size: 12px;
        line-height: 1.45;
        display: none;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.72);
      }
      .sp-viewer-tooltip.sp-visible {
        display: block;
      }
      .sp-tt-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 6px;
        padding-bottom: 5px;
        border-bottom: 1px solid #232734;
      }
      .sp-tt-title {
        font-weight: 600;
        font-size: 13px;
        color: #F4F5F8;
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .sp-tt-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
      }
      .sp-tt-badge {
        font-family: "SF Mono", ui-monospace, monospace;
        font-size: 10.5px;
        font-weight: 600;
        padding: 2px 7px;
        border-radius: 6px;
        text-transform: uppercase;
      }
      .sp-tt-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
        margin-bottom: 6px;
        font-family: "SF Mono", ui-monospace, monospace;
        font-size: 11px;
        color: #8A909E;
      }
      .sp-tt-metrics span {
        background: #0A0B0E;
        border: 1px solid #232734;
        padding: 1px 6px;
        border-radius: 4px;
        color: #D8DCE6;
      }
      .sp-tt-evidence {
        font-size: 12px;
        color: #D8DCE6;
        margin-bottom: 5px;
      }
      .sp-tt-rounds {
        font-family: "SF Mono", ui-monospace, monospace;
        font-size: 10.5px;
        color: #8A909E;
        padding-top: 5px;
        border-top: 1px dashed #232734;
      }
    `;
    document.head.appendChild(style);
  }

  function createEmptyLabelRecord(frameId, key) {
    const fid = String(frameId || "");
    return {
      frame_id: fid,
      key: key,
      label: key,
      present: null,
      final_present: null,
      status: "pending",
      spatial_status: "pending",
      conf: null,
      trust: null,
      evidence: null,
      critic: null,
      pv_match: null,
      p_wrong: null,
      status_round_0: null,
      status_round_1: null,
      status_round_2: null,
      status_round_3: null,
      round_history: [],
      argon_reason: null,
      box: null,
      tip: null,
      pass1: null,
      pass2: null,
      box_iou: null,
      mask_iou: null,
      polygon: null,
      rle: null,
      mask_b64: null,
      mask_url: null,
      maskCandidates: fid
        ? [
            `data/masks/${fid}_${key}.png`,
            `/web/data/masks/${fid}_${key}.png`,
            `/data/masks/${fid}_${key}.png`,
            `/out/masks/${fid}_${key}.png`,
            `../out/masks/${fid}_${key}.png`,
          ]
        : [],
    };
  }

  function ensureFrame(frameId, frameIdxHint) {
    const fid = String(frameId || `frame_${frameIdxHint || 0}`);
    if (!state.frameById[fid]) {
      const urls = deriveCandidateUrls(fid);
      const rec = {
        frame_id: fid,
        frame_idx: typeof frameIdxHint === "number" ? frameIdxHint : Object.keys(state.frameById).length,
        frame_num: parseFrameNum(fid),
        img_url: null,
        gt_mask_url: null,
        imgCandidates: urls.imgCandidates,
        gtCandidates: urls.gtCandidates,
        phase: null,
        phase_conf: null,
        narration: null,
        labels: {},
      };
      for (const k of state.labelKeys) {
        rec.labels[k] = createEmptyLabelRecord(fid, k);
      }
      state.frameById[fid] = rec;
    }
    return state.frameById[fid];
  }

  function computeSignature(lbl) {
    if (!lbl) return "";
    const b = Array.isArray(lbl.box) ? lbl.box.join(",") : "";
    const t = Array.isArray(lbl.tip) ? lbl.tip.join(",") : "";
    const m = lbl.mask_url || (lbl.mask_b64 ? String(lbl.mask_b64).slice(0, 32) : "") || (lbl.polygon ? "poly" : "");
    return `${lbl.status || "pending"}|${lbl.final_present}|${b}|${t}|${m}`;
  }

  function triggerAnnotationAnimation(frameId, key, kind) {
    const animKey = `${frameId}__${key}`;
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    state.activeAnimations.set(animKey, {
      frameId,
      key,
      kind: kind || "draw-in",
      startMs: now,
      durationMs: ANIM_DURATION_MS,
    });
    ensureAnimationLoop();
  }

  function getAnimationProgress(frameId, key) {
    const animKey = `${frameId}__${key}`;
    const entry = state.activeAnimations.get(animKey);
    if (!entry) return 1.0;
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const raw = (now - entry.startMs) / (entry.durationMs || ANIM_DURATION_MS);
    if (raw >= 1.0) {
      state.activeAnimations.delete(animKey);
      return 1.0;
    }
    return easeOutCubic(Math.max(0.02, raw));
  }

  function ensureAnimationLoop() {
    if (state.rafId !== null) return;
    if (typeof requestAnimationFrame === "undefined") return;
    function step() {
      state.rafId = null;
      renderCurrentFrameSync();
      if (state.activeAnimations.size > 0) {
        state.rafId = requestAnimationFrame(step);
      }
    }
    state.rafId = requestAnimationFrame(step);
  }

  function mergeLabelItem(frameRec, rawKey, src, animateIfNew) {
    const k = String(rawKey || src.key || src.label || "").trim();
    if (!k) return false;
    if (!state.labelKeys.includes(k)) {
      state.labelKeys.push(k);
    }
    if (!frameRec.labels[k]) {
      frameRec.labels[k] = createEmptyLabelRecord(frameRec.frame_id, k);
    }
    const dst = frameRec.labels[k];
    const sigKey = `${frameRec.frame_id}__${k}`;
    const prevSig = state.seenSignatures.get(sigKey);

    if (src.present !== undefined && src.present !== null && src.present !== "") {
      dst.present = Number(src.present);
    }
    if (src.final_present !== undefined && src.final_present !== null && src.final_present !== "") {
      dst.final_present = Number(src.final_present);
    } else if (dst.final_present === null && dst.present !== null) {
      dst.final_present = dst.present;
    }

    if (src.status !== undefined && src.status !== null && src.status !== "") {
      dst.status = normalizeVerdict(src.status);
    }
    if (src.spatial_status !== undefined && src.spatial_status !== null && src.spatial_status !== "") {
      dst.spatial_status = normalizeVerdict(src.spatial_status);
    }

    const rawConf = src.conf !== undefined && src.conf !== null && src.conf !== "" ? src.conf : src.confidence;
    if (rawConf !== undefined && rawConf !== null && rawConf !== "") {
      const nConf = Number(rawConf);
      if (Number.isFinite(nConf)) dst.conf = nConf;
    }
    if (src.trust !== undefined && src.trust !== null && src.trust !== "") {
      const nTrust = Number(src.trust);
      if (Number.isFinite(nTrust)) dst.trust = nTrust;
    }

    if (src.evidence || src.rationale) {
      dst.evidence = String(src.evidence || src.rationale);
    }
    if (src.critic || src.critic_reason) {
      dst.critic = String(src.critic || src.critic_reason);
    }
    if (src.pv_match !== undefined && src.pv_match !== null && src.pv_match !== "") {
      if (typeof src.pv_match === "boolean") dst.pv_match = src.pv_match;
      else dst.pv_match = String(src.pv_match).toLowerCase() === "true" || String(src.pv_match) === "1";
    }
    if (src.p_wrong !== undefined && src.p_wrong !== null && src.p_wrong !== "") {
      const pw = Number(src.p_wrong);
      if (Number.isFinite(pw)) dst.p_wrong = pw;
    }

    if (src.status_round_0) dst.status_round_0 = normalizeVerdict(src.status_round_0);
    if (src.status_round_1) dst.status_round_1 = normalizeVerdict(src.status_round_1);
    if (src.status_round_2) dst.status_round_2 = normalizeVerdict(src.status_round_2);
    if (src.status_round_3) dst.status_round_3 = normalizeVerdict(src.status_round_3);
    if (Array.isArray(src.round_history) && src.round_history.length > 0) {
      dst.round_history = src.round_history;
      for (const rh of src.round_history) {
        if (rh && typeof rh.round === "number" && rh.status) {
          dst[`status_round_${rh.round}`] = normalizeVerdict(rh.status);
        }
      }
    }
    if (src.argon_reason) {
      dst.argon_reason = String(src.argon_reason);
    }

    const normBox = normalizeBoxCoords(src.box || src.bbox || src.box_2d);
    if (normBox) {
      dst.box = normBox;
      if (dst.final_present === null) dst.final_present = 1;
    }

    const normTip = normalizeTipCoords(src.tip || src.tip_2d || src.point);
    if (normTip) {
      dst.tip = normTip;
    }

    if (src.pass1 && typeof src.pass1 === "object") {
      dst.pass1 = {
        ...src.pass1,
        box: normalizeBoxCoords(src.pass1.box || src.pass1.box_2d) || src.pass1.box,
        tip: normalizeTipCoords(src.pass1.tip || src.pass1.tip_2d) || src.pass1.tip,
      };
    }
    if (src.pass2 && typeof src.pass2 === "object") {
      dst.pass2 = {
        ...src.pass2,
        box: normalizeBoxCoords(src.pass2.box || src.pass2.box_2d) || src.pass2.box,
        tip: normalizeTipCoords(src.pass2.tip || src.pass2.tip_2d) || src.pass2.tip,
      };
    }

    const rawBoxIou =
      src.box_iou !== undefined && src.box_iou !== null
        ? src.box_iou
        : src.iou !== undefined && src.iou !== null
        ? src.iou
        : dst.pass2 && typeof dst.pass2.iou === "number"
        ? dst.pass2.iou
        : null;
    if (rawBoxIou !== null && rawBoxIou !== undefined && rawBoxIou !== "") {
      const nIou = Number(rawBoxIou);
      if (Number.isFinite(nIou)) dst.box_iou = nIou;
    }

    if (src.mask_iou !== undefined && src.mask_iou !== null && src.mask_iou !== "") {
      const mIou = Number(src.mask_iou);
      if (Number.isFinite(mIou)) dst.mask_iou = mIou;
    }

    // Polygon support
    const poly = normalizePolygon(src.polygon || src.polygons || (Array.isArray(src.mask) ? src.mask : null));
    if (poly) {
      dst.polygon = poly;
    }

    // RLE mask support
    if (src.rle && typeof src.rle === "object") {
      dst.rle = src.rle;
    } else if (src.mask && typeof src.mask === "object" && !Array.isArray(src.mask)) {
      if (src.mask.polygon || src.mask.polygons) {
        dst.polygon = normalizePolygon(src.mask.polygon || src.mask.polygons);
      } else if (src.mask.counts) {
        dst.rle = src.mask;
      }
    }

    // Mask URL / path / base64 support
    const maskStr =
      src.mask_url ||
      src.ai_mask_url ||
      src.mask_path ||
      (typeof src.mask_b64 === "string" ? src.mask_b64 : null) ||
      (typeof src.mask === "string" ? src.mask : null);

    if (maskStr) {
      const s = String(maskStr).trim();
      if (s.startsWith("data:image/")) {
        dst.mask_b64 = s;
        dst.mask_url = s;
        dst.maskCandidates = [s, ...dst.maskCandidates];
      } else if (s.startsWith("iVBOR") || (s.length > 120 && !s.includes("/") && !s.endsWith(".png"))) {
        const dataUri = `data:image/png;base64,${s}`;
        dst.mask_b64 = dataUri;
        dst.mask_url = dataUri;
        dst.maskCandidates = [dataUri, ...dst.maskCandidates];
      } else {
        dst.mask_url = s;
        const clean = s.replace(/^\/+/, "");
        const baseCands = Array.isArray(src.maskCandidates) && src.maskCandidates.length > 0 ? src.maskCandidates : dst.maskCandidates;
        const cands = [s, "/" + clean, "../" + clean];
        for (const c of baseCands) {
          if (!cands.includes(c)) cands.push(c);
        }
        dst.maskCandidates = cands;
      }
    }

    const newSig = computeSignature(dst);
    state.seenSignatures.set(sigKey, newSig);

    const hasSpatialNow = Boolean(dst.box || dst.tip || dst.mask_url || dst.polygon || dst.rle);
    if (animateIfNew && prevSig !== newSig && hasSpatialNow) {
      triggerAnnotationAnimation(frameRec.frame_id, k, "draw-in");
      return true;
    }
    return prevSig !== newSig;
  }

  function ingestFrameObject(frObj, idxHint, animateIfNew) {
    if (!frObj || typeof frObj !== "object") return;
    const fid = frObj.frame_id || frObj.id || `frame_${idxHint}`;
    const f = ensureFrame(fid, typeof frObj.frame_idx === "number" ? frObj.frame_idx : idxHint);
    if (typeof frObj.frame_idx === "number") f.frame_idx = frObj.frame_idx;
    if (typeof frObj.frame_num === "number") f.frame_num = frObj.frame_num;

    const imgUrl = frObj.img_url || frObj.frame_url || frObj.image_url || frObj.img_path || frObj.frame_path || frObj.path;
    if (imgUrl) {
      const s = String(imgUrl).trim();
      f.img_url = s;
      const clean = s.replace(/^\/+/, "");
      f.imgCandidates = [s, "/" + clean, "../" + clean, ...f.imgCandidates.filter((x) => x !== s)];
    }

    const gtUrl = frObj.gt_mask_url || frObj.watershed_mask_url || frObj.mask_gt_url;
    if (gtUrl) {
      const s = String(gtUrl).trim();
      f.gt_mask_url = s;
      const clean = s.replace(/^\/+/, "");
      f.gtCandidates = [s, "/" + clean, "../" + clean, ...f.gtCandidates.filter((x) => x !== s)];
    }

    if (frObj.phase && frObj.phase !== "pending") {
      f.phase = String(frObj.phase);
    }
    if (typeof frObj.phase_conf === "number" && Number.isFinite(frObj.phase_conf)) {
      f.phase_conf = frObj.phase_conf;
    }
    if (frObj.narration) {
      f.narration = typeof frObj.narration === "string" ? frObj.narration : frObj.narration.text || null;
    }

    // Ingest labels / annotations / objects (both dict and array forms)
    const rawSources = [frObj.labels, frObj.labels_list, frObj.annotations, frObj.objects];
    for (const raw of rawSources) {
      if (!raw) continue;
      if (Array.isArray(raw)) {
        for (const item of raw) {
          if (item && typeof item === "object") {
            mergeLabelItem(f, item.key || item.label, item, animateIfNew);
          }
        }
      } else if (typeof raw === "object") {
        for (const [k, item] of Object.entries(raw)) {
          if (item && typeof item === "object") {
            mergeLabelItem(f, k, item, animateIfNew);
          }
        }
      }
    }
  }

  function getEffectiveStatus(frameId, key, lbl) {
    const gk = `${frameId}__${key}`;
    if (state.gateDecisions && state.gateDecisions[gk]) {
      return "accept";
    }
    if (!lbl) return "pending";
    if (lbl.status && lbl.status !== "pending") return normalizeVerdict(lbl.status);
    if (lbl.spatial_status && lbl.spatial_status !== "pending") return normalizeVerdict(lbl.spatial_status);
    return "pending";
  }

  function getEffectivePresent(frameId, key, lbl) {
    const gk = `${frameId}__${key}`;
    if (state.gateDecisions && state.gateDecisions[gk]) {
      return state.gateDecisions[gk].human_present === 1 ? 1 : 0;
    }
    if (!lbl) return null;
    if (lbl.final_present !== null && lbl.final_present !== undefined) return Number(lbl.final_present);
    if (lbl.present !== null && lbl.present !== undefined) return Number(lbl.present);
    if (lbl.box || lbl.tip || lbl.mask_url || lbl.polygon || lbl.rle) return 1;
    return null;
  }

  // Image loading with automatic fallback chain
  function loadImageCandidates(candidates, cacheMap, cacheKey, onLoaded) {
    if (!candidates || candidates.length === 0) return null;
    if (cacheMap[cacheKey] !== undefined) {
      return cacheMap[cacheKey]; // either HTMLImageElement, "loading", or null
    }
    if (typeof Image === "undefined") return null;

    cacheMap[cacheKey] = "loading";
    let idx = 0;

    function tryNext() {
      if (idx >= candidates.length) {
        cacheMap[cacheKey] = null;
        if (onLoaded) onLoaded(null);
        return;
      }
      const url = candidates[idx++];
      if (!url) {
        tryNext();
        return;
      }
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = function () {
        cacheMap[cacheKey] = img;
        if (onLoaded) onLoaded(img);
      };
      img.onerror = function () {
        tryNext();
      };
      img.src = url;
    }

    tryNext();
    return "loading";
  }

  // Build tinted per-label mask offscreen canvas (semi-transparent interior + crisp contour)
  function getTintedMaskCanvas(maskImg, key, cacheKey) {
    if (state.maskTintedCache[cacheKey]) return state.maskTintedCache[cacheKey];
    if (typeof document === "undefined") return null;

    const w = maskImg.naturalWidth || maskImg.width || NATIVE_W;
    const h = maskImg.naturalHeight || maskImg.height || NATIVE_H;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    if (!octx) return null;

    octx.drawImage(maskImg, 0, 0, w, h);
    let imgData;
    try {
      imgData = octx.getImageData(0, 0, w, h);
    } catch (_e) {
      // If cross-origin tainted, fallback to raw image
      state.maskTintedCache[cacheKey] = maskImg;
      return maskImg;
    }

    const d = imgData.data;
    const rgb = LABEL_RGB[key] || [56, 189, 248];
    const fg = new Uint8Array(w * h);

    for (let p = 0; p < w * h; p++) {
      const i = p * 4;
      const r = d[i];
      const g = d[i + 1];
      const b = d[i + 2];
      const a = d[i + 3];
      // Foreground if alpha > 20 and not near-black background
      if (a > 20 && (r > 20 || g > 20 || b > 20)) {
        fg[p] = 1;
      }
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
        // Check 4-neighborhood for contour boundary
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
        d[i + 3] = isEdge ? 235 : 102;
      }
    }

    octx.putImageData(imgData, 0, 0);
    off._fgMask = fg;
    off._fgW = w;
    off._fgH = h;
    state.maskTintedCache[cacheKey] = off;
    return off;
  }

  // Decode RLE mask to tinted offscreen canvas
  function getRleMaskCanvas(rleObj, key, cacheKey) {
    if (state.maskTintedCache[cacheKey]) return state.maskTintedCache[cacheKey];
    if (typeof document === "undefined" || !rleObj || !Array.isArray(rleObj.counts)) return null;

    const size = Array.isArray(rleObj.size) ? rleObj.size : [NATIVE_H, NATIVE_W];
    const h = Number(size[0]) || NATIVE_H;
    const w = Number(size[1]) || NATIVE_W;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    if (!octx) return null;

    const imgData = octx.createImageData(w, h);
    const d = imgData.data;
    const rgb = LABEL_RGB[key] || [56, 189, 248];
    const fg = new Uint8Array(w * h);

    let idx = 0;
    let val = 0;
    for (const count of rleObj.counts) {
      const c = Number(count) || 0;
      if (val === 1) {
        for (let k = 0; k < c && idx + k < w * h; k++) {
          fg[idx + k] = 1;
        }
      }
      idx += c;
      val = 1 - val;
    }

    for (let p = 0; p < w * h; p++) {
      if (!fg[p]) continue;
      const i = p * 4;
      d[i] = rgb[0];
      d[i + 1] = rgb[1];
      d[i + 2] = rgb[2];
      d[i + 3] = 105;
    }
    octx.putImageData(imgData, 0, 0);
    off._fgMask = fg;
    off._fgW = w;
    off._fgH = h;
    state.maskTintedCache[cacheKey] = off;
    return off;
  }

  // Build CholecSeg8k Ground-Truth Watershed Mask Overlay (for Scoring View toggle only)
  function getGtWatershedCanvas(gtImg, frameId) {
    if (state.gtCanvasCache[frameId]) return state.gtCanvasCache[frameId];
    if (typeof document === "undefined") return null;

    const w = gtImg.naturalWidth || gtImg.width || NATIVE_W;
    const h = gtImg.naturalHeight || gtImg.height || NATIVE_H;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    if (!octx) return null;

    octx.drawImage(gtImg, 0, 0, w, h);
    let imgData;
    try {
      imgData = octx.getImageData(0, 0, w, h);
    } catch (_e) {
      return null;
    }

    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const val = d[i];
      if (val === 255 || val === 0 || val === 50) {
        d[i + 3] = 0;
        continue;
      }
      const key = WATERSHED_TO_KEY[val];
      if (key && LABEL_RGB[key]) {
        const rgb = LABEL_RGB[key];
        d[i] = rgb[0];
        d[i + 1] = rgb[1];
        d[i + 2] = rgb[2];
        d[i + 3] = 115;
      } else {
        d[i + 3] = 0;
      }
    }
    octx.putImageData(imgData, 0, 0);
    state.gtCanvasCache[frameId] = off;
    return off;
  }

  function syncExternalCheckboxes() {
    if (typeof document === "undefined") return;
    const map = [
      ["chk-ai-masks", "masks"],
      ["chk-ai-boxes", "boxes"],
      ["chk-dual-pass", "dualPass"],
      ["chk-ai-tips", "tips"],
      ["chk-hud-chips", "chips"],
      ["chk-show-gt", "gt"],
    ];
    for (const [id, layerKey] of map) {
      const el = document.getElementById(id);
      if (el && typeof el.checked === "boolean") {
        state.layers[layerKey] = el.checked;
      }
    }
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    const radius = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + w - radius, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
    ctx.lineTo(x + w, y + h - radius);
    ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    ctx.lineTo(x + radius, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }

  function drawCornerBrackets(ctx, x, y, w, h, len, color, lineWidth) {
    const L = Math.min(len, w * 0.35, h * 0.35);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = "square";
    ctx.beginPath();
    // Top-left
    ctx.moveTo(x, y + L);
    ctx.lineTo(x, y);
    ctx.lineTo(x + L, y);
    // Top-right
    ctx.moveTo(x + w - L, y);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x + w, y + L);
    // Bottom-right
    ctx.moveTo(x + w, y + h - L);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w - L, y + h);
    // Bottom-left
    ctx.moveTo(x + L, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + h - L);
    ctx.stroke();
    ctx.restore();
  }

  // Main synchronous canvas render pass
  function renderCurrentFrameSync() {
    if (!state.canvas || !state.ctx) return;
    syncExternalCheckboxes();

    const canvas = state.canvas;
    const ctx = state.ctx;
    const W = canvas.width || NATIVE_W;
    const H = canvas.height || NATIVE_H;

    state.hitRegions = [];
    state.hudChipRects = [];

    // Clear background with DESIGN.md near-black #0A0B0E
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = "#0A0B0E";
    ctx.fillRect(0, 0, W, H);

    const fr = state.frames[state.currentFrameIdx] || null;
    if (!fr) {
      // No frames loaded yet -> explicit pending screen
      ctx.fillStyle = "#12141C";
      drawRoundedRect(ctx, W * 0.18, H * 0.34, W * 0.64, H * 0.32, 8);
      ctx.fill();
      ctx.strokeStyle = "#232734";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = "#F4F5F8";
      ctx.font = '600 16px -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif';
      ctx.textAlign = "center";
      ctx.fillText("Surgical Frame Viewer · pending", W / 2, H / 2 - 10);

      ctx.fillStyle = "#8A909E";
      ctx.font = '400 13px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
      ctx.fillText("Awaiting web/data/video12.json or Viewer.update(data)...", W / 2, H / 2 + 16);
      ctx.textAlign = "left";
      ctx.restore();
      return;
    }

    // Begin zoomed & panned world space
    ctx.save();
    ctx.translate(state.panX, state.panY);
    ctx.scale(state.zoom, state.zoom);

    // 1. Base Layer: Real Surgical Frame Image
    const baseImg = loadImageCandidates(fr.imgCandidates, state.imageCache, fr.frame_id, () => {
      renderCurrentFrameSync();
    });

    if (baseImg && baseImg !== "loading") {
      ctx.drawImage(baseImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#0D0F14";
      ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#232734";
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
      ctx.fillStyle = "#8A909E";
      ctx.font = '500 14px -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
      const statusTxt = baseImg === "loading" ? "loading real frame..." : "pending";
      ctx.fillText(`Frame ${fr.frame_id} · image ${statusTxt}`, 28, H / 2);
    }

    // 2. Optional Post-Hoc Scoring View: Ground-Truth Watershed Mask Overlay
    if (state.layers.gt) {
      const gtImg = loadImageCandidates(fr.gtCandidates, state.imageCache, `${fr.frame_id}__gt`, () => {
        renderCurrentFrameSync();
      });
      if (gtImg && gtImg !== "loading") {
        const gtCanvas = getGtWatershedCanvas(gtImg, fr.frame_id);
        if (gtCanvas) {
          ctx.drawImage(gtCanvas, 0, 0, W, H);
        }
      }
    }

    // 3. AI Segmentation Masks Layer (semi-transparent, per-label colour)
    let loadedMasksCount = 0;
    let pendingMasksCount = 0;
    if (state.layers.masks) {
      for (const k of state.labelKeys) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const effPres = getEffectivePresent(fr.frame_id, k, lbl);
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const animProg = getAnimationProgress(fr.frame_id, k);
        const isHovered = state.hoveredKey === k;
        const isSelected = state.selectedKey === k;
        const rgb = LABEL_RGB[k] || [56, 189, 248];
        const baseAlpha = effStatus === "reject" ? 0.16 : isHovered || isSelected ? 0.52 : 0.36;

        let maskDrawn = false;

        // 3a. PNG / base64 mask image
        if (lbl.maskCandidates && lbl.maskCandidates.length > 0 && (lbl.mask_url || effPres === 1)) {
          const mCacheKey = `${fr.frame_id}__${k}__${lbl.mask_url || "mask"}`;
          const rawMaskImg = loadImageCandidates(lbl.maskCandidates, state.maskRawCache, mCacheKey, () => {
            renderCurrentFrameSync();
          });
          if (rawMaskImg && rawMaskImg !== "loading") {
            const tinted = getTintedMaskCanvas(rawMaskImg, k, mCacheKey);
            if (tinted) {
              ctx.save();
              ctx.globalAlpha = Math.min(1, baseAlpha * 1.8 * animProg);
              ctx.drawImage(tinted, 0, 0, W, H);
              ctx.restore();
              maskDrawn = true;
              loadedMasksCount++;
              state.hitRegions.push({
                type: "mask",
                key: k,
                maskCanvas: tinted,
                area: W * H * 0.5,
              });
            }
          }
        }

        // 3b. Polygon segmentation mask
        if (!maskDrawn && Array.isArray(lbl.polygon) && lbl.polygon.length > 0) {
          ctx.save();
          ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${(baseAlpha * animProg).toFixed(3)})`;
          ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${(0.9 * animProg).toFixed(3)})`;
          ctx.lineWidth = (isHovered || isSelected ? 2.4 : 1.5) / state.zoom;
          for (const ring of lbl.polygon) {
            if (!Array.isArray(ring) || ring.length < 3) continue;
            ctx.beginPath();
            ring.forEach(([py, px], idx) => {
              const cx = (px * W) / 1000;
              const cy = (py * H) / 1000;
              if (idx === 0) ctx.moveTo(cx, cy);
              else ctx.lineTo(cx, cy);
            });
            ctx.closePath();
            ctx.fill();
            ctx.stroke();
          }
          ctx.restore();
          maskDrawn = true;
          loadedMasksCount++;
          state.hitRegions.push({
            type: "polygon",
            key: k,
            polygon: lbl.polygon,
            area: W * H * 0.25,
          });
        }

        // 3c. RLE segmentation mask
        if (!maskDrawn && lbl.rle) {
          const rleKey = `${fr.frame_id}__${k}__rle`;
          const rleCanvas = getRleMaskCanvas(lbl.rle, k, rleKey);
          if (rleCanvas) {
            ctx.save();
            ctx.globalAlpha = Math.min(1, baseAlpha * 1.8 * animProg);
            ctx.drawImage(rleCanvas, 0, 0, W, H);
            ctx.restore();
            maskDrawn = true;
            loadedMasksCount++;
            state.hitRegions.push({
              type: "mask",
              key: k,
              maskCanvas: rleCanvas,
              area: W * H * 0.5,
            });
          }
        }

        // 3d. Subtle per-label region tint inside AI box when PNG mask is still pending
        if (!maskDrawn && effPres === 1) {
          pendingMasksCount++;
          if (Array.isArray(lbl.box) && lbl.box.length === 4) {
            const [y1, x1, y2, x2] = lbl.box;
            const rx = (x1 * W) / 1000;
            const ry = (y1 * H) / 1000;
            const rw = ((x2 - x1) * W) / 1000;
            const rh = ((y2 - y1) * H) / 1000;
            ctx.save();
            ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${(0.11 * animProg).toFixed(3)})`;
            ctx.fillRect(rx, ry, rw, rh);
            ctx.restore();
          }
        }
      }
    }

    // Update external #mask-status-label if present in host DOM
    if (typeof document !== "undefined") {
      const maskStatusEl = document.getElementById("mask-status-label");
      if (maskStatusEl) {
        maskStatusEl.textContent = loadedMasksCount > 0 ? `${loadedMasksCount} masks` : "pending";
      }
    }

    // 4. AI Bounding Boxes Layer + Anchored Verdict Label Chips
    const placedChipRects = [];
    if (state.layers.boxes) {
      for (const k of state.labelKeys) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const vTok = VERDICT_TOKENS[effStatus] || VERDICT_TOKENS.pending;
        const animProg = getAnimationProgress(fr.frame_id, k);
        const isHovered = state.hoveredKey === k;
        const isSelected = state.selectedKey === k;
        const rejectDim = effStatus === "reject" ? 0.35 : 1.0;

        // Optional Pass-1 (Flash) vs Pass-2 (Argon) Dual Boxes
        if (state.layers.dualPass) {
          if (lbl.pass1 && Array.isArray(lbl.pass1.box) && lbl.pass1.box.length === 4) {
            const [py1, px1, py2, px2] = lbl.pass1.box;
            ctx.save();
            ctx.globalAlpha = 0.85 * animProg;
            ctx.strokeStyle = "#3B82F6";
            ctx.lineWidth = 1.6 / state.zoom;
            ctx.setLineDash([5 / state.zoom, 4 / state.zoom]);
            ctx.strokeRect((px1 * W) / 1000, (py1 * H) / 1000, ((px2 - px1) * W) / 1000, ((py2 - py1) * H) / 1000);
            ctx.restore();
          }
          if (lbl.pass2 && Array.isArray(lbl.pass2.box) && lbl.pass2.box.length === 4) {
            const [qy1, qx1, qy2, qx2] = lbl.pass2.box;
            ctx.save();
            ctx.globalAlpha = 0.85 * animProg;
            ctx.strokeStyle = "#A855F7";
            ctx.lineWidth = 1.6 / state.zoom;
            ctx.setLineDash([3 / state.zoom, 3 / state.zoom]);
            ctx.strokeRect((qx1 * W) / 1000, (qy1 * H) / 1000, ((qx2 - qx1) * W) / 1000, ((qy2 - qy1) * H) / 1000);
            ctx.restore();
          }
        }

        if (Array.isArray(lbl.box) && lbl.box.length === 4) {
          const [y1, x1, y2, x2] = lbl.box;
          const fullRx = (x1 * W) / 1000;
          const fullRy = (y1 * H) / 1000;
          const fullRw = ((x2 - x1) * W) / 1000;
          const fullRh = ((y2 - y1) * H) / 1000;

          // Draw-in scale from box center during live update animation
          const scale = 0.86 + 0.14 * animProg;
          const cx = fullRx + fullRw / 2;
          const cy = fullRy + fullRh / 2;
          const rx = cx - (fullRw * scale) / 2;
          const ry = cy - (fullRh * scale) / 2;
          const rw = fullRw * scale;
          const rh = fullRh * scale;

          ctx.save();
          ctx.globalAlpha = rejectDim * animProg;

          // Subtle highlight fill when hovered or selected
          if (isHovered || isSelected) {
            const [vr, vg, vb] = vTok.rgb;
            ctx.fillStyle = `rgba(${vr}, ${vg}, ${vb}, 0.14)`;
            ctx.fillRect(rx, ry, rw, rh);
          }

          // Main bounding box stroke
          ctx.strokeStyle = vTok.stroke;
          ctx.lineWidth = (isHovered || isSelected ? 2.8 : 1.8) / state.zoom;
          drawRoundedRect(ctx, rx, ry, rw, rh, 4 / state.zoom);
          ctx.stroke();

          // Supervisely / CVAT corner brackets
          drawCornerBrackets(
            ctx,
            rx,
            ry,
            rw,
            rh,
            Math.max(8, Math.min(16, Math.min(rw, rh) * 0.22)),
            vTok.stroke,
            (isHovered || isSelected ? 3.4 : 2.5) / state.zoom
          );

          // Register box hit region
          state.hitRegions.push({
            type: "box",
            key: k,
            x: fullRx,
            y: fullRy,
            w: fullRw,
            h: fullRh,
            area: fullRw * fullRh,
          });

          // Anchored Verdict Label Chip on Box
          const confTxt = typeof lbl.conf === "number" ? ` ${Math.round(lbl.conf * 100)}%` : "";
          const iouVal = typeof lbl.box_iou === "number" ? lbl.box_iou : lbl.pass2 && typeof lbl.pass2.iou === "number" ? lbl.pass2.iou : null;
          const iouTxt = typeof iouVal === "number" ? ` · IoU ${iouVal.toFixed(2)}` : "";
          const chipLabel = `${k} · ${vTok.label}${confTxt}${iouTxt}`;

          const fontPx = Math.max(9, 11.5 / state.zoom);
          ctx.font = `600 ${fontPx}px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif`;
          const textW = ctx.measureText(chipLabel).width;
          const padX = 8 / state.zoom;
          const chipW = textW + padX * 2 + 8 / state.zoom;
          const chipH = 20 / state.zoom;

          // Upward slide during entrance animation
          const slideOffset = (1 - animProg) * (8 / state.zoom);
          let chipX = Math.max(4 / state.zoom, Math.min(W - chipW - 4 / state.zoom, rx));
          let chipY = ry - chipH - 3 / state.zoom + slideOffset;
          if (chipY < 34 / state.zoom) {
            chipY = ry + 3 / state.zoom + slideOffset;
          }

          // Simple vertical collision avoidance against already placed box chips
          for (const prev of placedChipRects) {
            const overlapX = chipX < prev.x + prev.w && chipX + chipW > prev.x;
            const overlapY = chipY < prev.y + prev.h && chipY + chipH > prev.y;
            if (overlapX && overlapY) {
              chipY = prev.y + prev.h + 2 / state.zoom;
            }
          }
          placedChipRects.push({ x: chipX, y: chipY, w: chipW, h: chipH });

          // Draw pill background (DESIGN.md verdict pill fill + 1px verdict border)
          ctx.fillStyle = vTok.fill;
          drawRoundedRect(ctx, chipX, chipY, chipW, chipH, 5 / state.zoom);
          ctx.fill();
          ctx.strokeStyle = vTok.stroke;
          ctx.lineWidth = 1 / state.zoom;
          ctx.stroke();

          // Verdict indicator dot inside pill
          ctx.fillStyle = vTok.stroke;
          ctx.beginPath();
          ctx.arc(chipX + padX, chipY + chipH / 2, 3 / state.zoom, 0, Math.PI * 2);
          ctx.fill();

          // Pill text
          ctx.fillStyle = vTok.text;
          ctx.textBaseline = "middle";
          ctx.fillText(chipLabel, chipX + padX + 6 / state.zoom, chipY + chipH / 2 + 0.5 / state.zoom);
          ctx.textBaseline = "alphabetic";

          state.hitRegions.push({
            type: "chip",
            key: k,
            x: chipX,
            y: chipY,
            w: chipW,
            h: chipH,
            area: chipW * chipH * 0.1, // high hit priority
          });

          ctx.restore();
        }
      }
    }

    // 5. Instrument Tool-Tip Points Layer ([y, x] in 0..1000)
    if (state.layers.tips) {
      for (const k of state.labelKeys) {
        const lbl = fr.labels[k];
        if (!lbl || !Array.isArray(lbl.tip) || lbl.tip.length < 2) continue;

        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const vTok = VERDICT_TOKENS[effStatus] || VERDICT_TOKENS.pending;
        const animProg = getAnimationProgress(fr.frame_id, k);
        const isHovered = state.hoveredKey === k;
        const isSelected = state.selectedKey === k;

        const ty = (lbl.tip[0] * H) / 1000;
        const tx = (lbl.tip[1] * W) / 1000;

        // Target ring contracts from 22px -> 8.5px during draw-in animation
        const ringRadius = ((isHovered || isSelected ? 10.5 : 8.0) + (1 - animProg) * 14) / state.zoom;
        const crossOuter = 14 / state.zoom;
        const crossInner = 3.5 / state.zoom;

        ctx.save();
        ctx.globalAlpha = animProg;

        // Dark backing halo for contrast on bright metallic reflections
        ctx.strokeStyle = "rgba(10, 11, 14, 0.85)";
        ctx.lineWidth = 3.6 / state.zoom;
        ctx.beginPath();
        ctx.arc(tx, ty, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Precision target ring colored by verdict (or cyan #38BDF8)
        const tipColor = isHovered || isSelected ? "#F4F5F8" : vTok.stroke;
        ctx.strokeStyle = tipColor;
        ctx.lineWidth = 2.0 / state.zoom;
        ctx.beginPath();
        ctx.arc(tx, ty, ringRadius, 0, Math.PI * 2);
        ctx.stroke();

        // Gapped crosshair ticks
        ctx.beginPath();
        ctx.moveTo(tx - crossOuter, ty);
        ctx.lineTo(tx - crossInner, ty);
        ctx.moveTo(tx + crossInner, ty);
        ctx.lineTo(tx + crossOuter, ty);
        ctx.moveTo(tx, ty - crossOuter);
        ctx.lineTo(tx, ty - crossInner);
        ctx.moveTo(tx, ty + crossInner);
        ctx.lineTo(tx, ty + crossOuter);
        ctx.stroke();

        // Exact tip pixel dot
        ctx.fillStyle = "#38BDF8";
        ctx.beginPath();
        ctx.arc(tx, ty, 2.5 / state.zoom, 0, Math.PI * 2);
        ctx.fill();

        // Compact TIP pill tag
        const tipTag = `TIP · ${k}`;
        const fontPx = Math.max(8.5, 10.5 / state.zoom);
        ctx.font = `600 ${fontPx}px "SF Mono", ui-monospace, Menlo, monospace`;
        const tw = ctx.measureText(tipTag).width + 10 / state.zoom;
        const th = 17 / state.zoom;
        const tagX = Math.min(W - tw - 4 / state.zoom, tx + 11 / state.zoom);
        const tagY = Math.max(34 / state.zoom, ty - th / 2);

        ctx.fillStyle = "#12141C";
        drawRoundedRect(ctx, tagX, tagY, tw, th, 4 / state.zoom);
        ctx.fill();
        ctx.strokeStyle = tipColor;
        ctx.lineWidth = 1 / state.zoom;
        ctx.stroke();

        ctx.fillStyle = "#F4F5F8";
        ctx.textBaseline = "middle";
        ctx.fillText(tipTag, tagX + 5 / state.zoom, tagY + th / 2 + 0.5 / state.zoom);
        ctx.textBaseline = "alphabetic";

        state.hitRegions.push({
          type: "tip",
          key: k,
          tx,
          ty,
          r: 15 / state.zoom,
          area: 1, // highest hit priority
        });

        ctx.restore();
      }
    }

    // End world-space transform; draw screen-space HUD overlays so text stays crisp at any zoom
    ctx.restore();

    // 6. Top Surgical Phase & Frame Status Bar (Screen Space)
    ctx.fillStyle = "rgba(10, 11, 14, 0.86)";
    ctx.fillRect(0, 0, W, 30);
    ctx.strokeStyle = "#232734";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, 30.5);
    ctx.lineTo(W, 30.5);
    ctx.stroke();

    ctx.fillStyle = "#F4F5F8";
    ctx.font = '600 12px -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif';
    const phaseStr = fr.phase
      ? `SURGICAL PHASE: ${fr.phase}` + (typeof fr.phase_conf === "number" ? ` (${ formatConfidence(fr.phase_conf) })` : "")
      : "SURGICAL PHASE: pending";
    ctx.fillText(phaseStr, 12, 19);

    // Top-right frame & mask status summary or Scoring View banner
    if (state.layers.gt) {
      const gtTag = "SCORING VIEW ONLY · GROUND-TRUTH MASK";
      ctx.font = '600 10.5px "SF Mono", ui-monospace, monospace';
      const gw = ctx.measureText(gtTag).width + 16;
      ctx.fillStyle = "#3B1D5E";
      drawRoundedRect(ctx, W - gw - 10, 5, gw, 20, 5);
      ctx.fill();
      ctx.strokeStyle = "#A855F7";
      ctx.stroke();
      ctx.fillStyle = "#F4F5F8";
      ctx.fillText(gtTag, W - gw - 2, 19);
    } else {
      const maskSummary = loadedMasksCount > 0 ? `masks: ${loadedMasksCount}` : "masks: pending";
      const rightSummary = `${fr.frame_id} · ${maskSummary}`;
      ctx.font = '400 11px "SF Mono", ui-monospace, monospace';
      ctx.fillStyle = "#8A909E";
      const rw = ctx.measureText(rightSummary).width;
      ctx.fillText(rightSummary, W - rw - 12, 19);
    }

    // 7. Left Verdict Label Chips HUD Tray (Screen Space, shows all 6 structures & their verdicts)
    if (state.layers.chips) {
      let hudY = 38;
      for (const k of state.labelKeys) {
        const lbl = fr.labels[k];
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const effPres = getEffectivePresent(fr.frame_id, k, lbl);
        const vTok = VERDICT_TOKENS[effStatus] || VERDICT_TOKENS.pending;
        const isHovered = state.hoveredKey === k;
        const isSelected = state.selectedKey === k;

        const presTxt = effPres === 1 ? "PRESENT" : effPres === 0 ? "absent" : "pending";
        const confTxt = lbl && typeof lbl.conf === "number" ? ` · ${lbl.conf.toFixed(2)}` : "";
        const spatialTags = [];
        if (lbl && (lbl.mask_url || lbl.polygon || lbl.rle)) spatialTags.push("MASK");
        if (lbl && Array.isArray(lbl.box)) spatialTags.push("BOX");
        if (lbl && Array.isArray(lbl.tip)) spatialTags.push("TIP");
        const spSuffix =
          spatialTags.length > 0
            ? ` [${spatialTags.join("+")}]`
            : effPres === 1
            ? " [spatial: pending]"
            : "";

        const chipText = `${k}: ${presTxt} · ${vTok.label}${confTxt}${spSuffix}`;
        ctx.font = '500 11px "SF Mono", ui-monospace, Menlo, monospace';
        const cw = Math.ceil(ctx.measureText(chipText).width + 22);
        const ch = 21;
        const cx = 10;

        ctx.save();
        ctx.fillStyle = isHovered || isSelected ? vTok.fill : "rgba(18, 20, 28, 0.86)";
        drawRoundedRect(ctx, cx, hudY, cw, ch, 5);
        ctx.fill();

        ctx.strokeStyle = isHovered || isSelected ? vTok.stroke : "#232734";
        ctx.lineWidth = isHovered || isSelected ? 1.5 : 1;
        ctx.stroke();

        // Left verdict color bar
        ctx.fillStyle = vTok.stroke;
        drawRoundedRect(ctx, cx + 1, hudY + 1, 4, ch - 2, 2);
        ctx.fill();

        ctx.fillStyle = effStatus === "pending" ? "#8A909E" : "#F4F5F8";
        ctx.fillText(chipText, cx + 11, hudY + 14.5);
        ctx.restore();

        state.hudChipRects.push({
          key: k,
          x: cx,
          y: hudY,
          w: cw,
          h: ch,
        });

        hudY += 24;
      }
    }

    ctx.restore();

    if (state.zoomLabelEl) {
      state.zoomLabelEl.textContent = `${Math.round(state.zoom * 100)}%`;
    }
  }

  // Point-in-polygon helper (ray casting in 0..1000 normalized coords)
  function pointInPolygonRing(ny, nx, ring) {
    let inside = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const yi = ring[i][0];
      const xi = ring[i][1];
      const yj = ring[j][0];
      const xj = ring[j][1];
      const intersect = yi > ny !== yj > ny && nx < ((xj - xi) * (ny - yi)) / (yj - yi + 1e-9) + xi;
      if (intersect) inside = !inside;
    }
    return inside;
  }

  function hitTestAtCanvasPoint(cx, cy) {
    const W = (state.canvas && state.canvas.width) || NATIVE_W;
    const H = (state.canvas && state.canvas.height) || NATIVE_H;

    // 1. Check screen-space HUD chips first
    for (const r of state.hudChipRects) {
      if (cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h) {
        return r.key;
      }
    }

    // Convert screen canvas coords (cx, cy) into world image coords (ix, iy)
    const ix = (cx - state.panX) / state.zoom;
    const iy = (cy - state.panY) / state.zoom;
    if (ix < 0 || iy < 0 || ix > W || iy > H) return null;

    // 2. Check tips first (highest priority)
    for (const reg of state.hitRegions) {
      if (reg.type === "tip") {
        const dx = ix - reg.tx;
        const dy = iy - reg.ty;
        if (dx * dx + dy * dy <= reg.r * reg.r) {
          return reg.key;
        }
      }
    }

    // 3. Check anchored box label chips
    for (const reg of state.hitRegions) {
      if (reg.type === "chip") {
        if (ix >= reg.x && ix <= reg.x + reg.w && iy >= reg.y && iy <= reg.y + reg.h) {
          return reg.key;
        }
      }
    }

    // 4. Check bounding boxes (smallest area first so nested tools win over large organs like liver)
    const matchingBoxes = state.hitRegions
      .filter((reg) => reg.type === "box" && ix >= reg.x && ix <= reg.x + reg.w && iy >= reg.y && iy <= reg.y + reg.h)
      .sort((a, b) => a.area - b.area);
    if (matchingBoxes.length > 0) {
      return matchingBoxes[0].key;
    }

    // 5. Check polygon masks
    const ny = (iy / H) * 1000;
    const nx = (ix / W) * 1000;
    for (const reg of state.hitRegions) {
      if (reg.type === "polygon" && Array.isArray(reg.polygon)) {
        for (const ring of reg.polygon) {
          if (pointInPolygonRing(ny, nx, ring)) return reg.key;
        }
      }
    }

    // 6. Check bitmap mask foreground pixels
    for (const reg of state.hitRegions) {
      if (reg.type === "mask" && reg.maskCanvas && reg.maskCanvas._fgMask) {
        const mw = reg.maskCanvas._fgW || W;
        const mh = reg.maskCanvas._fgH || H;
        const mx = Math.floor((ix / W) * mw);
        const my = Math.floor((iy / H) * mh);
        if (mx >= 0 && mx < mw && my >= 0 && my < mh) {
          if (reg.maskCanvas._fgMask[my * mw + mx]) {
            return reg.key;
          }
        }
      }
    }

    return null;
  }

  function updateHoverTooltip(key, clientX, clientY) {
    if (!state.tooltipEl) return;
    const fr = state.frames[state.currentFrameIdx];
    if (!fr || !key || !fr.labels[key]) {
      state.tooltipEl.classList.remove("sp-visible");
      return;
    }

    const lbl = fr.labels[key];
    const effStatus = getEffectiveStatus(fr.frame_id, key, lbl);
    const effPres = getEffectivePresent(fr.frame_id, key, lbl);
    const vTok = VERDICT_TOKENS[effStatus] || VERDICT_TOKENS.pending;
    const rgb = LABEL_RGB[key] || [56, 189, 248];

    const presStr = effPres === 1 ? "PRESENT" : effPres === 0 ? "ABSENT" : "PENDING";
    const confStr = formatConfidence(lbl.conf);
    const trustStr = formatConfidence(lbl.trust);
    const iouVal = typeof lbl.box_iou === "number" ? lbl.box_iou : lbl.pass2 && typeof lbl.pass2.iou === "number" ? lbl.pass2.iou : null;
    const iouStr = typeof iouVal === "number" ? iouVal.toFixed(2) : "pending";
    const maskIouStr = typeof lbl.mask_iou === "number" ? ` · Mask IoU: ${lbl.mask_iou.toFixed(2)}` : "";
    const pvStr = lbl.pv_match === true ? "match" : lbl.pv_match === false ? "mismatch" : "n/a";
    const pwStr = typeof lbl.p_wrong === "number" ? lbl.p_wrong.toFixed(2) : "n/a";

    const evText = lbl.evidence ? lbl.evidence : "Evidence: pending";
    const roundsArr = [lbl.status_round_0, lbl.status_round_1, lbl.status_round_2, lbl.status_round_3].filter(Boolean);
    const roundsStr = roundsArr.length > 0 ? roundsArr.join(" → ") : effStatus;

    state.tooltipEl.innerHTML = `
      <div class="sp-tt-header">
        <div class="sp-tt-title">
          <span class="sp-tt-dot" style="background:rgb(${rgb[0]},${rgb[1]},${rgb[2]})"></span>
          <span>${LABEL_NAMES[key] || key}</span>
          <code style="font-size:11px;color:#8A909E">(${key})</code>
        </div>
        <span class="sp-tt-badge" style="background:${vTok.fill};color:${vTok.stroke};border:1px solid ${vTok.stroke}">
          ${presStr} · ${vTok.label}
        </span>
      </div>
      <div class="sp-tt-metrics">
        <span>conf: ${confStr}</span>
        <span>trust: ${trustStr}</span>
        <span>Pass1/2 IoU: ${iouStr}${maskIouStr}</span>
        <span>crop: ${pvStr}</span>
        <span>p_wrong: ${pwStr}</span>
      </div>
      <div class="sp-tt-evidence">${evText}</div>
      <div class="sp-tt-rounds">
        <div><strong>Amber Loop:</strong> ${roundsStr}</div>
        ${lbl.argon_reason ? `<div style="color:#F59E0B;margin-top:2px;">Argon: ${lbl.argon_reason}</div>` : ""}
      </div>
    `;

    state.tooltipEl.classList.add("sp-visible");

    // Position tooltip near cursor without overflowing window bounds
    const vw = typeof window !== "undefined" ? window.innerWidth : 1280;
    const vh = typeof window !== "undefined" ? window.innerHeight : 720;
    const rect = state.tooltipEl.getBoundingClientRect();
    const tw = rect.width || 320;
    const th = rect.height || 150;

    let left = clientX + 16;
    let top = clientY + 16;
    if (left + tw + 12 > vw) left = Math.max(8, clientX - tw - 16);
    if (top + th + 12 > vh) top = Math.max(8, clientY - th - 16);

    state.tooltipEl.style.left = `${left}px`;
    state.tooltipEl.style.top = `${top}px`;
  }

  function clampPan() {
    if (!state.canvas) return;
    const W = state.canvas.width || NATIVE_W;
    const H = state.canvas.height || NATIVE_H;
    if (state.zoom <= 1.001) {
      state.zoom = 1.0;
      state.panX = 0;
      state.panY = 0;
      return;
    }
    const minPanX = W - W * state.zoom;
    const minPanY = H - H * state.zoom;
    state.panX = Math.min(0, Math.max(minPanX, state.panX));
    state.panY = Math.min(0, Math.max(minPanY, state.panY));
  }

  function setZoomAtPoint(newZoom, cx, cy) {
    const clampedZoom = Math.max(1.0, Math.min(8.0, newZoom));
    const oldZoom = state.zoom;
    if (Math.abs(clampedZoom - oldZoom) < 1e-4) return;

    state.panX = cx - (cx - state.panX) * (clampedZoom / oldZoom);
    state.panY = cy - (cy - state.panY) * (clampedZoom / oldZoom);
    state.zoom = clampedZoom;
    clampPan();
    renderCurrentFrameSync();
  }

  function getCanvasLogicalPoint(evt) {
    if (!state.canvas) return { cx: 0, cy: 0 };
    const rect = state.canvas.getBoundingClientRect();
    const W = state.canvas.width || NATIVE_W;
    const H = state.canvas.height || NATIVE_H;
    if (!rect.width || !rect.height) {
      return { cx: evt.offsetX || 0, cy: evt.offsetY || 0 };
    }
    const cx = ((evt.clientX - rect.left) / rect.width) * W;
    const cy = ((evt.clientY - rect.top) / rect.height) * H;
    return { cx, cy };
  }

  function dispatchViewerEvent(name, detail) {
    if (typeof CustomEvent === "undefined") return;
    if (state.rootEl && typeof state.rootEl.dispatchEvent === "function") {
      state.rootEl.dispatchEvent(new CustomEvent(name, { detail, bubbles: true }));
    }
    if (typeof window !== "undefined" && typeof window.dispatchEvent === "function") {
      window.dispatchEvent(new CustomEvent(name, { detail }));
    }
  }

  function bindCanvasInteractions(canvas) {
    if (!canvas || canvas._spBound) return;
    canvas._spBound = true;

    canvas.addEventListener(
      "wheel",
      (evt) => {
        evt.preventDefault();
        const { cx, cy } = getCanvasLogicalPoint(evt);
        const factor = evt.deltaY < 0 ? 1.18 : 1 / 1.18;
        setZoomAtPoint(state.zoom * factor, cx, cy);
      },
      { passive: false }
    );

    canvas.addEventListener("mousedown", (evt) => {
      if (evt.button !== 0) return;
      state.isDragging = true;
      state.dragMoved = false;
      state.dragStartX = evt.clientX;
      state.dragStartY = evt.clientY;
      state.dragOriginPanX = state.panX;
      state.dragOriginPanY = state.panY;
      canvas.classList.add("sp-dragging");
    });

    canvas.addEventListener("mousemove", (evt) => {
      const { cx, cy } = getCanvasLogicalPoint(evt);
      state.hoverClientX = evt.clientX;
      state.hoverClientY = evt.clientY;

      if (state.isDragging) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = rect.width ? (canvas.width || NATIVE_W) / rect.width : 1;
        const scaleY = rect.height ? (canvas.height || NATIVE_H) / rect.height : 1;
        const dx = (evt.clientX - state.dragStartX) * scaleX;
        const dy = (evt.clientY - state.dragStartY) * scaleY;
        if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
          state.dragMoved = true;
        }
        if (state.zoom > 1.0) {
          state.panX = state.dragOriginPanX + dx;
          state.panY = state.dragOriginPanY + dy;
          clampPan();
          renderCurrentFrameSync();
        }
        return;
      }

      const hitKey = hitTestAtCanvasPoint(cx, cy);
      if (hitKey !== state.hoveredKey) {
        state.hoveredKey = hitKey;
        if (hitKey) {
          canvas.classList.add("sp-hovering-target");
        } else {
          canvas.classList.remove("sp-hovering-target");
        }
        renderCurrentFrameSync();
        const fr = state.frames[state.currentFrameIdx];
        if (fr && hitKey) {
          const detail = {
            frameIndex: state.currentFrameIdx,
            frameId: fr.frame_id,
            label: hitKey,
            key: hitKey,
            annotation: fr.labels[hitKey],
          };
          dispatchViewerEvent("viewer:hover", detail);
          dispatchViewerEvent("surgproof:hover-label", detail);
        }
      }
      updateHoverTooltip(hitKey, evt.clientX, evt.clientY);
    });

    const endDrag = () => {
      if (state.isDragging) {
        state.isDragging = false;
        canvas.classList.remove("sp-dragging");
      }
    };
    canvas.addEventListener("mouseup", endDrag);
    canvas.addEventListener("mouseleave", () => {
      endDrag();
      if (state.hoveredKey !== null) {
        state.hoveredKey = null;
        canvas.classList.remove("sp-hovering-target");
        renderCurrentFrameSync();
      }
      if (state.tooltipEl) {
        state.tooltipEl.classList.remove("sp-visible");
      }
    });

    canvas.addEventListener("click", (evt) => {
      if (state.dragMoved) return;
      const { cx, cy } = getCanvasLogicalPoint(evt);
      const hitKey = hitTestAtCanvasPoint(cx, cy);
      if (hitKey) {
        state.selectedKey = hitKey;
        renderCurrentFrameSync();
        const fr = state.frames[state.currentFrameIdx];
        const detail = {
          frameIndex: state.currentFrameIdx,
          frameId: fr ? fr.frame_id : null,
          label: hitKey,
          key: hitKey,
          annotation: fr && fr.labels ? fr.labels[hitKey] : null,
        };
        if (typeof state.options.onSelect === "function") {
          state.options.onSelect(hitKey, detail.annotation, fr);
        }
        dispatchViewerEvent("viewer:select", detail);
        dispatchViewerEvent("surgproof:select-label", detail);
      }
    });

    canvas.addEventListener("dblclick", () => {
      Viewer.resetZoom();
    });
  }

  function bindGlobalEventsAndCheckboxes() {
    if (typeof document === "undefined" || state._globalBound) return;
    state._globalBound = true;

    const checkboxIds = [
      ["chk-ai-masks", "masks"],
      ["chk-ai-boxes", "boxes"],
      ["chk-dual-pass", "dualPass"],
      ["chk-ai-tips", "tips"],
      ["chk-hud-chips", "chips"],
      ["chk-show-gt", "gt"],
    ];
    for (const [id, layerKey] of checkboxIds) {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener("change", () => {
          state.layers[layerKey] = Boolean(el.checked);
          renderCurrentFrameSync();
        });
      }
    }

    if (typeof window !== "undefined") {
      window.addEventListener("timeline:seek", (e) => {
        if (e && e.detail) {
          const idx = e.detail.frameIndex !== undefined ? e.detail.frameIndex : e.detail.index;
          if (idx !== undefined) Viewer.show(idx);
        }
      });
      window.addEventListener("surgproof:seek", (e) => {
        if (e && e.detail) {
          const idx = e.detail.frameIndex !== undefined ? e.detail.frameIndex : e.detail.index;
          if (idx !== undefined) Viewer.show(idx);
        }
      });
      window.addEventListener("objects:select", (e) => {
        if (e && e.detail) {
          const k = e.detail.label || e.detail.key;
          if (k) Viewer.selectLabel(k);
        }
      });
      window.addEventListener("surgproof:select", (e) => {
        if (e && e.detail) {
          const k = e.detail.label || e.detail.key;
          if (k) Viewer.selectLabel(k);
        }
      });
    }
  }

  // Public API object
  const Viewer = {
    /**
     * Mount the <canvas> surgical frame viewer into `el`.
     * @param {HTMLElement|string} el - Container element, <canvas> element, or CSS selector
     * @param {Object} [opts] - Optional config or initial video12.json data bundle
     */
    mount(el, opts) {
      injectStyles();
      const target =
        typeof el === "string"
          ? typeof document !== "undefined"
            ? document.querySelector(el)
            : null
          : el;

      if (!target) {
        return Viewer;
      }

      state.options = opts && typeof opts === "object" ? opts : {};

      // Check whether target is directly a <canvas> or contains #frame-canvas
      if (target.tagName && target.tagName.toUpperCase() === "CANVAS") {
        state.canvas = target;
        state.stageEl = target.parentElement || target;
        state.rootEl = state.stageEl;
      } else {
        state.rootEl = target;
        const existingCanvas = target.querySelector ? target.querySelector("canvas") : null;
        if (existingCanvas) {
          state.canvas = existingCanvas;
          state.stageEl = existingCanvas.parentElement || target;
        } else {
          target.innerHTML = "";
          target.classList.add("sp-viewer-host");

          // Build embedded layer toggle bar if external checkboxes are not in DOM
          const hasExternalToggles = typeof document !== "undefined" && Boolean(document.getElementById("chk-ai-masks"));
          if (!hasExternalToggles) {
            const tb = document.createElement("div");
            tb.className = "sp-viewer-toolbar";
            tb.innerHTML = `
              <div class="sp-viewer-toolbar-group">
                <label class="sp-viewer-chip-toggle" title="Toggle semi-transparent per-label AI segmentation masks">
                  <input type="checkbox" data-sp-layer="masks" checked />
                  <span>AI Masks (<span id="mask-status-label">pending</span>)</span>
                </label>
                <label class="sp-viewer-chip-toggle" title="Toggle AI bounding boxes coloured by verdict">
                  <input type="checkbox" data-sp-layer="boxes" checked />
                  <span>Boxes</span>
                </label>
                <label class="sp-viewer-chip-toggle" title="Toggle Pass-1 (Flash) vs Pass-2 (Argon) comparison boxes">
                  <input type="checkbox" data-sp-layer="dualPass" />
                  <span>Pass 1 vs 2</span>
                </label>
                <label class="sp-viewer-chip-toggle" title="Toggle surgical instrument tip crosshairs">
                  <input type="checkbox" data-sp-layer="tips" checked />
                  <span>Tool Tips</span>
                </label>
                <label class="sp-viewer-chip-toggle" title="Toggle structure verdict chips tray">
                  <input type="checkbox" data-sp-layer="chips" checked />
                  <span>Verdict Chips</span>
                </label>
              </div>
              <div class="sp-viewer-toolbar-group">
                <label class="sp-viewer-chip-toggle sp-gt-toggle" title="Scoring View Only: overlay CholecSeg8k ground-truth watershed mask">
                  <input type="checkbox" data-sp-layer="gt" />
                  <span>Scoring View: Ground-Truth Mask</span>
                </label>
              </div>
            `;
            tb.querySelectorAll("input[data-sp-layer]").forEach((inp) => {
              inp.addEventListener("change", () => {
                const lk = inp.getAttribute("data-sp-layer");
                if (lk) {
                  state.layers[lk] = Boolean(inp.checked);
                  renderCurrentFrameSync();
                }
              });
            });
            target.appendChild(tb);
            state.toolbarEl = tb;
          }

          const stage = document.createElement("div");
          stage.className = "sp-viewer-stage";
          const canvas = document.createElement("canvas");
          canvas.id = "frame-canvas";
          canvas.className = "sp-viewer-canvas";
          canvas.width = NATIVE_W;
          canvas.height = NATIVE_H;
          stage.appendChild(canvas);
          target.appendChild(stage);

          state.stageEl = stage;
          state.canvas = canvas;
        }
      }

      if (!state.canvas.width) state.canvas.width = NATIVE_W;
      if (!state.canvas.height) state.canvas.height = NATIVE_H;
      state.canvas.classList.add("sp-viewer-canvas");
      state.ctx = state.canvas.getContext("2d");

      // Create floating zoom/pan HUD pill inside stageEl
      if (state.stageEl && typeof document !== "undefined") {
        const compStyle = typeof window !== "undefined" && window.getComputedStyle ? window.getComputedStyle(state.stageEl) : null;
        if (compStyle && compStyle.position === "static") {
          state.stageEl.style.position = "relative";
        }
        let zoomPill = state.stageEl.querySelector ? state.stageEl.querySelector(".sp-viewer-zoom-pill") : null;
        if (!zoomPill) {
          zoomPill = document.createElement("div");
          zoomPill.className = "sp-viewer-zoom-pill";
          zoomPill.innerHTML = `
            <button type="button" class="sp-viewer-zoom-btn" data-sp-zoom="out" title="Zoom out">−</button>
            <span class="sp-viewer-zoom-val" style="min-width:38px;text-align:center;">100%</span>
            <button type="button" class="sp-viewer-zoom-btn" data-sp-zoom="in" title="Zoom in">+</button>
            <button type="button" class="sp-viewer-zoom-btn" data-sp-zoom="reset" title="Reset zoom & pan (Double-click)">Reset</button>
          `;
          const zOut = zoomPill.querySelector ? zoomPill.querySelector('[data-sp-zoom="out"]') : null;
          const zIn = zoomPill.querySelector ? zoomPill.querySelector('[data-sp-zoom="in"]') : null;
          const zReset = zoomPill.querySelector ? zoomPill.querySelector('[data-sp-zoom="reset"]') : null;
          if (zOut && zOut.addEventListener) zOut.addEventListener("click", () => Viewer.zoomOut());
          if (zIn && zIn.addEventListener) zIn.addEventListener("click", () => Viewer.zoomIn());
          if (zReset && zReset.addEventListener) zReset.addEventListener("click", () => Viewer.resetZoom());
          state.stageEl.appendChild(zoomPill);
        }
        state.zoomPillEl = zoomPill;
        state.zoomLabelEl = zoomPill && zoomPill.querySelector ? zoomPill.querySelector(".sp-viewer-zoom-val") : null;
      }

      // Create floating hover tooltip in document.body
      if (typeof document !== "undefined" && document.body) {
        let tt = document.getElementById("sp-viewer-tooltip");
        if (!tt) {
          tt = document.createElement("div");
          tt.id = "sp-viewer-tooltip";
          tt.className = "sp-viewer-tooltip";
          document.body.appendChild(tt);
        }
        state.tooltipEl = tt;
      }

      bindCanvasInteractions(state.canvas);
      bindGlobalEventsAndCheckboxes();
      state.mounted = true;

      // If opts contains initial data or frames, ingest immediately
      if (opts && typeof opts === "object") {
        if (opts.data) {
          Viewer.update(opts.data);
        } else if (Array.isArray(opts.frames) || Array.isArray(opts.annotations) || Array.isArray(opts)) {
          Viewer.update(opts);
        }
      }

      renderCurrentFrameSync();
      return Viewer;
    },

    /**
     * Switch to and render `frameIndex` (integer index 0..N-1, frame_id string, or frame object).
     * @param {number|string|Object} frameIndex
     */
    show(frameIndex) {
      if (frameIndex && typeof frameIndex === "object" && frameIndex.frame_id) {
        ingestFrameObject(frameIndex, state.frames.length, false);
        state.frames = Object.values(state.frameById).sort((a, b) => a.frame_num - b.frame_num);
        frameIndex = frameIndex.frame_id;
      }

      let targetIdx = 0;
      if (typeof frameIndex === "string") {
        const found = state.frames.findIndex((f) => f.frame_id === frameIndex);
        targetIdx = found >= 0 ? found : 0;
      } else if (typeof frameIndex === "number" && Number.isFinite(frameIndex)) {
        targetIdx = frameIndex;
      }

      if (state.frames.length > 0) {
        state.currentFrameIdx = Math.max(0, Math.min(state.frames.length - 1, Math.floor(targetIdx)));
      } else {
        state.currentFrameIdx = 0;
      }

      renderCurrentFrameSync();

      const fr = state.frames[state.currentFrameIdx] || null;
      if (typeof state.options.onFrameChange === "function" && fr) {
        state.options.onFrameChange(state.currentFrameIdx, fr);
      }
      dispatchViewerEvent("viewer:framechange", {
        frameIndex: state.currentFrameIdx,
        frameId: fr ? fr.frame_id : null,
        frame: fr,
      });
      return Viewer;
    },

    setFrame(frameIndex) {
      return Viewer.show(frameIndex);
    },

    /**
     * Update the viewer with full `web/data/video12.json` or incremental `web/data/live.json` data.
     * Diffs newly arrived annotations and animates their draw-in / fade-in on the canvas.
     * @param {Object|Array} data
     */
    update(data) {
      if (!data) {
        renderCurrentFrameSync();
        return Viewer;
      }

      const animateNew = state.hasReceivedData;

      if (Array.isArray(data)) {
        // Either an array of frame objects or an array of flat live annotation rows
        const looksLikeFrames = data.some((x) => x && (x.labels || x.labels_list || x.img_url));
        if (looksLikeFrames) {
          data.forEach((fr, idx) => ingestFrameObject(fr, idx, animateNew));
        } else {
          for (const ann of data) {
            if (!ann || !ann.frame_id) continue;
            const f = ensureFrame(ann.frame_id);
            if (ann.phase && ann.phase !== "pending") f.phase = String(ann.phase);
            if (typeof ann.phase_conf === "number") f.phase_conf = ann.phase_conf;
            const k = ann.label || ann.key;
            if (k) {
              mergeLabelItem(f, k, ann, true);
            }
          }
        }
      } else if (typeof data === "object") {
        if (data.video) state.video = String(data.video);
        if (Array.isArray(data.label_keys) && data.label_keys.length > 0) {
          for (const k of data.label_keys) {
            if (!state.labelKeys.includes(k)) state.labelKeys.push(k);
          }
        }
        if (Array.isArray(data.frames)) {
          data.frames.forEach((fr, idx) => ingestFrameObject(fr, idx, animateNew));
        }
        if (Array.isArray(data.annotations)) {
          for (const ann of data.annotations) {
            if (!ann || !ann.frame_id) continue;
            const f = ensureFrame(ann.frame_id);
            if (ann.phase && ann.phase !== "pending") f.phase = String(ann.phase);
            if (typeof ann.phase_conf === "number") f.phase_conf = ann.phase_conf;
            const k = ann.label || ann.key;
            if (k) {
              mergeLabelItem(f, k, ann, true);
            }
          }
        }
        // Single frame object passed directly
        if (data.frame_id && (data.labels || data.labels_list || data.annotations)) {
          ingestFrameObject(data, state.frames.length, animateNew);
        }
      }

      state.frames = Object.values(state.frameById).sort((a, b) => a.frame_num - b.frame_num);
      state.frames.forEach((f, idx) => {
        f.frame_idx = idx;
      });

      if (!state.hasReceivedData && state.frames.length > 0) {
        state.hasReceivedData = true;
        // Default to first frame with spatial annotations if current frame is 0 and has none
        const cur = state.frames[state.currentFrameIdx];
        const curHasBox = cur && state.labelKeys.some((k) => cur.labels[k] && cur.labels[k].box);
        if (!curHasBox) {
          const firstWithBox = state.frames.findIndex((fr) =>
            state.labelKeys.some((k) => fr.labels[k] && fr.labels[k].box)
          );
          if (firstWithBox >= 0) {
            state.currentFrameIdx = firstWithBox;
          }
        }
      }

      renderCurrentFrameSync();
      return Viewer;
    },

    selectLabel(key) {
      state.selectedKey = key || null;
      renderCurrentFrameSync();
      return Viewer;
    },

    setSelectedKey(key) {
      return Viewer.selectLabel(key);
    },

    setLayer(layerName, enabled) {
      if (layerName in state.layers) {
        state.layers[layerName] = Boolean(enabled);
        renderCurrentFrameSync();
      }
      return Viewer;
    },

    getLayers() {
      return { ...state.layers };
    },

    setGateDecisions(gateDecisionsMap) {
      if (gateDecisionsMap && typeof gateDecisionsMap === "object") {
        state.gateDecisions = gateDecisionsMap;
        renderCurrentFrameSync();
      }
      return Viewer;
    },

    zoomIn() {
      const W = (state.canvas && state.canvas.width) || NATIVE_W;
      const H = (state.canvas && state.canvas.height) || NATIVE_H;
      setZoomAtPoint(state.zoom * 1.25, W / 2, H / 2);
      return Viewer;
    },

    zoomOut() {
      const W = (state.canvas && state.canvas.width) || NATIVE_W;
      const H = (state.canvas && state.canvas.height) || NATIVE_H;
      setZoomAtPoint(state.zoom / 1.25, W / 2, H / 2);
      return Viewer;
    },

    resetZoom() {
      state.zoom = 1.0;
      state.panX = 0;
      state.panY = 0;
      renderCurrentFrameSync();
      return Viewer;
    },

    getZoom() {
      return state.zoom;
    },

    getCurrentFrame() {
      return state.frames[state.currentFrameIdx] || null;
    },

    getActiveAnimations() {
      return Array.from(state.activeAnimations.values());
    },

    getState() {
      return {
        mounted: state.mounted,
        video: state.video,
        frameCount: state.frames.length,
        currentFrameIdx: state.currentFrameIdx,
        currentFrameId: state.frames[state.currentFrameIdx] ? state.frames[state.currentFrameIdx].frame_id : null,
        selectedKey: state.selectedKey,
        hoveredKey: state.hoveredKey,
        zoom: state.zoom,
        panX: state.panX,
        panY: state.panY,
        layers: { ...state.layers },
        activeAnimationCount: state.activeAnimations.size,
      };
    },

    destroy() {
      if (state.rafId !== null && typeof cancelAnimationFrame !== "undefined") {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
      state.activeAnimations.clear();
      state.frames = [];
      state.frameById = {};
      state.seenSignatures.clear();
      state.hasReceivedData = false;
      state.mounted = false;
    },
  };

  return Viewer;
});
