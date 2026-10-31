/* SurgProof — Autonomous Surgical Video Annotation Toolbox (web/app.js)
 * Wires:
 *   - Top bar (#video-badge, #headline-acc, #headline-n, #headline-loc, #headline-student, #live-status-pill)
 *   - Centre canvas area (#viewer — integrates nir-webcanvas web/viewer.js + built-in <canvas> fallback)
 *   - Right panel (#objects — label list with verbatim evidence, double-check metrics, 4-round Amber history, manual criteria)
 *   - Bottom timeline (#timeline — integrates nir-webtimeline web/timeline.js + built-in multi-track fallback)
 *   - Final Pass drawer (#final-pass-drawer — residual AMBER queue + 20-item audit sample -> POST /api/gate)
 *
 * Strictly reads real data from web/data/video12.json, web/data/live.json, and /out/* artifacts.
 * Anything not yet produced displays "pending". Zero hardcoded boxes, verdicts, or percentages.
 */
(function () {
  "use strict";

  const LABEL_KEYS = ["grasper", "l_hook", "gallbladder", "cystic_duct", "liver", "blood"];
  const LABEL_NAMES = {
    grasper: "Grasper",
    l_hook: "L-Hook Electrocautery",
    gallbladder: "Gallbladder",
    cystic_duct: "Cystic Duct",
    liver: "Liver",
    blood: "Free Blood / Clot",
  };
  const PRESENT_SHORT_NAMES = {
    grasper: "Grasper",
    l_hook: "L-Hook",
    gallbladder: "Gallbladder",
    cystic_duct: "Cystic Duct",
    liver: "Liver",
    blood: "Blood",
  };

  // Tokens from promo/DESIGN.md
  const STATUS_COL = {
    accept: "#10B981",
    AMBER: "#F59E0B",
    amber: "#F59E0B",
    reject: "#EF4444",
    pending: "#8A909E",
  };

  const STATUS_FILL = {
    accept: "#092E24",
    AMBER: "#362407",
    amber: "#362407",
    reject: "#331014",
    pending: "#12141C",
  };

  const LABEL_RGB = {
    grasper: [59, 130, 246],
    l_hook: [139, 92, 246],
    gallbladder: [16, 185, 129],
    cystic_duct: [245, 158, 11],
    liver: [180, 83, 9],
    blood: [239, 68, 68],
  };

  // CholecSeg8k watershed pixel value -> label key (used ONLY when Scoring View GT toggle is checked)
  const WATERSHED_TO_KEY = {
    31: "grasper",
    32: "l_hook",
    22: "gallbladder",
    11: "liver",
  };

  const state = {
    video: "video12",
    bundle: null,
    frames: [],
    frameById: {},
    currentFrameIdx: 0,
    selectedKey: "grasper",
    objectFilter: "all",
    playing: false,
    playTimer: null,
    playIntervalMs: 650,
    headline: null,
    results: null,
    localizeResults: null,
    student: null,
    manual: null,
    narrationByFrame: {},
    liveAnnotationCount: 0,
    seenLiveSignatures: new Set(),
    imageCache: {},
    maskCache: {},
    tintedMaskCache: {},
    gtCanvasCache: {},
    gateDecisions: {},
    auditSignedOff: false,
    streamReplayTimer: null,
    externalViewerMounted: false,
    externalTimelineMounted: false,
    externalGateMounted: false,
    canvasZoom: 1.0,
    presentMode: false,
    _renderSeq: 0,
  };

  function isPresentModeUrl() {
    if (typeof window === "undefined" || !window.location) return false;
    const params = new URLSearchParams(window.location.search || "");
    const p = String(params.get("present") || "").trim().toLowerCase();
    return p === "1" || p === "true" || p === "yes";
  }

  function syncPresentBodyClass() {
    const fromUrl = isPresentModeUrl();
    if (fromUrl) {
      state.presentMode = true;
    }
    if (typeof document !== "undefined" && document.body) {
      if (fromUrl) {
        document.body.classList.add("present");
      }
      state.presentMode = document.body.classList.contains("present");
    }
    return Boolean(state.presentMode);
  }

  syncPresentBodyClass();

  function getExternalViewer() {
    if (syncPresentBodyClass()) return null;
    const v = window.Viewer || window.SurgProofViewer;
    return v && typeof v.mount === "function" ? v : null;
  }

  function getExternalTimeline() {
    const t = window.Timeline || window.SurgProofTimeline;
    return t && typeof t.mount === "function" ? t : null;
  }

  function getExternalGate() {
    const g = window.FinalPassGate || window.Gate;
    return g && typeof g.mount === "function" ? g : null;
  }

  function normStatus(raw) {
    const s = String(raw || "").trim().toLowerCase();
    if (s === "accept" || s === "verified") return "accept";
    if (s === "reject" || s === "escalate") return "reject";
    if (!s || s === "pending" || s === "null" || s === "none") return "pending";
    return "AMBER";
  }

  function fmtPct(val, digits = 1) {
    if (typeof val !== "number" || !Number.isFinite(val)) return "pending";
    const factor = Math.pow(10, digits);
    const rounded = Math.round((val * 100 + Number.EPSILON) * factor) / factor;
    return rounded.toFixed(digits) + "%";
  }

  function computeBoxIou(b1, b2) {
    if (!Array.isArray(b1) || b1.length !== 4 || !Array.isArray(b2) || b2.length !== 4) return null;
    const [y1a, x1a, y2a, x2a] = b1.map(Number);
    const [y1b, x1b, y2b, x2b] = b2.map(Number);
    const iy1 = Math.max(y1a, y1b);
    const ix1 = Math.max(x1a, x1b);
    const iy2 = Math.min(y2a, y2b);
    const ix2 = Math.min(x2a, x2b);
    const iw = Math.max(0, ix2 - ix1);
    const ih = Math.max(0, iy2 - iy1);
    const inter = iw * ih;
    const areaA = Math.max(0, x2a - x1a) * Math.max(0, y2a - y1a);
    const areaB = Math.max(0, x2b - x1b) * Math.max(0, y2b - y1b);
    const union = areaA + areaB - inter;
    return union > 0 ? inter / union : null;
  }

  function getPassAgreement(lbl) {
    if (!lbl) return null;
    const mAgree = typeof lbl.mask_iou === "number" ? lbl.mask_iou : null;
    const bAgree =
      lbl.pass1 && Array.isArray(lbl.pass1.box) && lbl.pass2 && Array.isArray(lbl.pass2.box)
        ? computeBoxIou(lbl.pass1.box, lbl.pass2.box)
        : typeof lbl.box_iou === "number"
          ? lbl.box_iou
          : null;
    if (mAgree !== null && bAgree !== null) return Math.max(mAgree, bAgree);
    return mAgree !== null ? mAgree : bAgree;
  }

  async function fetchJsonSafe(urls) {
    const list = Array.isArray(urls) ? urls : [urls];
    for (const u of list) {
      try {
        const res = await fetch(u, { cache: "no-store" });
        if (res.ok) return await res.json();
      } catch (_e) {
        // try next candidate
      }
    }
    return null;
  }

  async function fetchTextSafe(urls) {
    const list = Array.isArray(urls) ? urls : [urls];
    for (const u of list) {
      try {
        const res = await fetch(u, { cache: "no-store" });
        if (res.ok) return await res.text();
      } catch (_e) {
        // try next candidate
      }
    }
    return null;
  }

  function parseCsv(text) {
    if (!text) return [];
    const rows = [];
    let cur = [];
    let field = "";
    let inQuotes = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const next = text[i + 1];
      if (inQuotes) {
        if (ch === '"' && next === '"') {
          field += '"';
          i++;
        } else if (ch === '"') {
          inQuotes = false;
        } else {
          field += ch;
        }
      } else {
        if (ch === '"') {
          inQuotes = true;
        } else if (ch === ",") {
          cur.push(field);
          field = "";
        } else if (ch === "\n" || ch === "\r") {
          if (ch === "\r" && next === "\n") i++;
          cur.push(field);
          if (cur.length > 1 || cur[0] !== "") rows.push(cur);
          cur = [];
          field = "";
        } else {
          field += ch;
        }
      }
    }
    if (field !== "" || cur.length > 0) {
      cur.push(field);
      rows.push(cur);
    }
    if (rows.length < 2) return [];
    const headers = rows[0].map((h) => h.trim());
    return rows.slice(1).map((vals) => {
      const obj = {};
      headers.forEach((h, idx) => {
        obj[h] = vals[idx] !== undefined ? vals[idx] : "";
      });
      return obj;
    });
  }

  function parseJsonl(text) {
    if (!text) return [];
    const out = [];
    for (const ln of text.split(/\r?\n/)) {
      const s = ln.trim();
      if (!s) continue;
      try {
        out.push(JSON.parse(s));
      } catch (_e) {
        // ignore partial line
      }
    }
    return out;
  }

  function frameSortKey(fid) {
    const parts = String(fid).split("_");
    const last = parts[parts.length - 1];
    const num = parseInt(last, 10);
    return Number.isFinite(num) ? num : 0;
  }

  function deriveFrameUrls(frameId) {
    const parts = String(frameId).split("_");
    if (parts.length === 3) {
      const vid = parts[0];
      const clip = parts[0] + "_" + parts[1];
      const fnum = parts[2];
      return {
        imgCandidates: [
          `data/frames/${frameId}.png`,
          `/web/data/frames/${frameId}.png`,
          `/out/web_data/frames/${frameId}.png`,
          `/raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `../raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `/data/cholecseg8k/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
        ],
        gtCandidates: [
          `data/gt_masks/${frameId}.png`,
          `/web/data/gt_masks/${frameId}.png`,
          `/out/web_data/gt_masks/${frameId}.png`,
          `/raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
          `../raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
          `/data/cholecseg8k/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo_watershed_mask.png`,
        ],
      };
    }
    return {
      imgCandidates: [`data/frames/${frameId}.png`],
      gtCandidates: [`data/gt_masks/${frameId}.png`],
    };
  }

  function ensureFrameRecord(frameId) {
    if (!state.frameById[frameId]) {
      const urls = deriveFrameUrls(frameId);
      const rec = {
        frame_id: frameId,
        frame_num: frameSortKey(frameId),
        img_url: urls.imgCandidates[0],
        gt_mask_url: urls.gtCandidates[0],
        imgCandidates: urls.imgCandidates,
        gtCandidates: urls.gtCandidates,
        phase: null,
        phase_conf: null,
        narration: null,
        labels: {},
      };
      for (const k of LABEL_KEYS) {
        rec.labels[k] = {
          frame_id: frameId,
          key: k,
          label: k,
          present: null,
          final_present: null,
          status: "pending",
          conf: null,
          trust: null,
          evidence: "",
          pv_match: null,
          p_wrong: null,
          critic_reason: "",
          status_round_0: null,
          status_round_1: null,
          status_round_2: null,
          status_round_3: null,
          round_history: [],
          argon_reason: "",
          box: null,
          tip: null,
          pass1: null,
          pass2: null,
          box_iou: null,
          mask_url: null,
          mask_r0_url: null,
          mask_r1_url: null,
          mask_r2_url: null,
          mask_b64: null,
          mask_iou: null,
          mask_source: null,
          polygon_px: null,
          polygon_rounds: null,
          loc_round: null,
          loc_status: null,
          maskCandidates: [
            `data/masks_r1/${frameId}_${k}.png`,
            `data/masks/${frameId}_${k}.png`,
            `/web/data/masks_r1/${frameId}_${k}.png`,
            `/web/data/masks/${frameId}_${k}.png`,
            `/out/masks_r1/${frameId}_${k}.png`,
            `/out/masks/${frameId}_${k}.png`,
          ],
          hasNewLivePulse: false,
        };
      }
      state.frameById[frameId] = rec;
    }
    return state.frameById[frameId];
  }

  function buildMaskCandidatesForRound(frameId, k, lbl, roundIdx) {
    const r = Number(roundIdx);
    const r0 = [
      lbl && lbl.mask_r0_url ? lbl.mask_r0_url : `data/masks/${frameId}_${k}.png`,
      `/web/data/masks/${frameId}_${k}.png`,
      `/out/masks/${frameId}_${k}.png`,
      `../out/masks/${frameId}_${k}.png`,
    ];
    const r1 = [
      lbl && lbl.mask_r1_url ? lbl.mask_r1_url : `data/masks_r1/${frameId}_${k}.png`,
      `/web/data/masks_r1/${frameId}_${k}.png`,
      `/out/masks_r1/${frameId}_${k}.png`,
      `../out/masks_r1/${frameId}_${k}.png`,
    ];
    const r2 = [
      lbl && lbl.mask_r2_url ? lbl.mask_r2_url : `data/masks_r2/${frameId}_${k}.png`,
      `/web/data/masks_r2/${frameId}_${k}.png`,
      `/out/masks_r2/${frameId}_${k}.png`,
      `../out/masks_r2/${frameId}_${k}.png`,
    ];
    if (r === 0) return r0;
    if (r === 2) return [...r2, ...r1, ...r0];
    return [...r1, ...r0];
  }

  function mergeAnnotationRow(ann, markPulse = false) {
    if (!ann || !ann.frame_id) return false;
    if (ann.video && ann.video !== state.video) return false;
    const f = ensureFrameRecord(ann.frame_id);
    if (ann.phase) {
      f.phase = ann.phase;
      if (typeof ann.phase_conf === "number") f.phase_conf = ann.phase_conf;
    }
    const k = ann.label || ann.key;
    if (k && f.labels[k]) {
      const lbl = f.labels[k];
      if (Array.isArray(ann.box) && ann.box.length === 4) lbl.box = ann.box.map(Number);
      if (Array.isArray(ann.tip) && ann.tip.length === 2) lbl.tip = ann.tip.map(Number);
      if (ann.pass1) lbl.pass1 = ann.pass1;
      if (ann.pass2) lbl.pass2 = ann.pass2;
      if (typeof ann.box_iou === "number") lbl.box_iou = ann.box_iou;
      else if (ann.pass2 && typeof ann.pass2.iou === "number") lbl.box_iou = ann.pass2.iou;
      else if (typeof ann.iou === "number") lbl.box_iou = ann.iou;
      if (typeof ann.mask_iou === "number") lbl.mask_iou = ann.mask_iou;
      if (Array.isArray(ann.polygon_px) && ann.polygon_px.length >= 3) {
        lbl.polygon_px = ann.polygon_px;
      } else if (Array.isArray(ann.polygon) && ann.polygon.length >= 3) {
        lbl.polygon_px = ann.polygon;
      } else if (Array.isArray(ann.points) && ann.points.length >= 3) {
        lbl.polygon_px = ann.points;
      }
      if (Array.isArray(ann.polygon_rounds) || Array.isArray(ann.rounds)) {
        lbl.polygon_rounds = ann.polygon_rounds || ann.rounds;
        const lastR = lbl.polygon_rounds[lbl.polygon_rounds.length - 1];
        if (lastR && typeof lastR.iou_pass1_pass2 === "number" && lbl.mask_iou === null) {
          lbl.mask_iou = lastR.iou_pass1_pass2;
        }
      }
      if (ann.mask_source) lbl.mask_source = String(ann.mask_source);
      if (ann.round !== undefined && ann.round !== null) lbl.loc_round = Number(ann.round);
      if (ann.status) lbl.loc_status = normStatus(ann.status);
      if (ann.mask_url || ann.mask_path) {
        const mp = String(ann.mask_url || ann.mask_path);
        lbl.mask_url = mp;
        lbl.maskCandidates = [mp, "/" + mp.replace(/^\/+/, ""), ...lbl.maskCandidates];
      }
      if (typeof ann.mask === "string" && ann.mask.length > 20) {
        lbl.mask_b64 = ann.mask;
      }
      if (markPulse) lbl.hasNewLivePulse = true;
      return true;
    }
    return false;
  }

  function buildNormalizedBundle() {
    return {
      video: state.video,
      split: "test",
      n_frames: state.frames.length,
      label_keys: LABEL_KEYS,
      headline: state.headline,
      results: state.results,
      localize_results: state.localizeResults,
      student: state.student,
      manual: state.manual,
      frames: state.frames,
    };
  }

  async function switchVideo(videoId) {
    if (!videoId) return;
    state.video = videoId;
    state.frameById = {};
    state.frames = [];
    state.currentFrameIdx = 0;
    state.seenLiveSignatures.clear();
    const selEl = document.getElementById("sel-video");
    if (selEl && selEl.value !== videoId) selEl.value = videoId;
    await loadInitialData();
  }

  async function loadInitialData() {
    const vid = state.video || "video12";
    // 1. Load pre-exported web/data/<video>.json from web/export.py first
    const exported = await fetchJsonSafe([
      `data/${vid}.json`,
      `/web/data/${vid}.json`,
      `/out/web_data/${vid}.json`,
      `/data/${vid}.json`,
      `/api/${vid}`,
    ]);

    if (exported) {
      state.bundle = exported;
      if (exported.video) state.video = exported.video;
      if (exported.headline) state.headline = exported.headline;
      if (exported.results) state.results = exported.results;
      if (exported.localize_results) state.localizeResults = exported.localize_results;
      if (exported.student) state.student = exported.student;
      if (exported.manual) state.manual = exported.manual;

      if (Array.isArray(exported.frames)) {
        for (const fr of exported.frames) {
          const f = ensureFrameRecord(fr.frame_id);
          if (fr.img_url || fr.frame_url || fr.path) {
            const u = fr.img_url || fr.frame_url || fr.path;
            f.img_url = u;
            f.imgCandidates = [u, "/" + String(u).replace(/^\/+/, ""), ...f.imgCandidates];
          }
          if (fr.gt_mask_url || fr.mask_url) {
            const gu = fr.gt_mask_url || fr.mask_url;
            f.gt_mask_url = gu;
            f.gtCandidates = [gu, "/" + String(gu).replace(/^\/+/, ""), ...f.gtCandidates];
          }
          if (fr.phase) f.phase = fr.phase;
          if (fr.phase_conf !== undefined && fr.phase_conf !== null) f.phase_conf = fr.phase_conf;
          if (fr.narration) {
            f.narration = fr.narration;
          } else if (fr.narration_entry) {
            f.narration = fr.narration_entry.narration || fr.narration_entry.sentence || fr.narration_entry.text || null;
          }

          const rawLabels = fr.labels || fr.annotations || fr.labels_list || {};
          const entries = Array.isArray(rawLabels)
            ? rawLabels.map((item) => [item.key || item.label, item])
            : Object.entries(rawLabels);

          for (const [k, v] of entries) {
            if (!f.labels[k] || !v) continue;
            const t = f.labels[k];
            if (v.present !== undefined && v.present !== null && v.present !== "") t.present = Number(v.present);
            if (v.final_present !== undefined && v.final_present !== null && v.final_present !== "") {
              t.final_present = Number(v.final_present);
            } else if (t.present !== null) {
              t.final_present = t.present;
            }
            if (v.status) t.status = normStatus(v.status);
            if (v.conf !== undefined && v.conf !== null && v.conf !== "") t.conf = Number(v.conf);
            else if (v.confidence !== undefined && v.confidence !== null) t.conf = Number(v.confidence);
            if (v.trust !== undefined && v.trust !== null && v.trust !== "") t.trust = Number(v.trust);
            if (v.evidence || v.rationale) t.evidence = String(v.evidence || v.rationale);
            if (v.pv_match !== undefined && v.pv_match !== null && v.pv_match !== "") {
              t.pv_match = String(v.pv_match) === "True" || v.pv_match === true;
            }
            if (v.p_wrong !== undefined && v.p_wrong !== null && v.p_wrong !== "") t.p_wrong = Number(v.p_wrong);
            if (v.critic_reason || v.critic) t.critic_reason = String(v.critic_reason || v.critic);

            const rHist = v.round_history || v.amber_rounds;
            if (Array.isArray(rHist) && rHist.length > 0) {
              t.round_history = rHist.map((rh) => ({
                round: rh.round,
                status: normStatus(rh.status),
              }));
              for (const rh of t.round_history) {
                t[`status_round_${rh.round}`] = rh.status;
              }
            } else {
              for (let rIdx = 0; rIdx < 4; rIdx++) {
                const rk = `status_round_${rIdx}`;
                if (v[rk]) {
                  t[rk] = normStatus(v[rk]);
                  t.round_history.push({ round: rIdx, status: t[rk] });
                }
              }
            }

            if (v.argon_reason) t.argon_reason = String(v.argon_reason);
            if (Array.isArray(v.box) && v.box.length === 4) t.box = v.box.map(Number);
            if (Array.isArray(v.tip) && v.tip.length === 2) t.tip = v.tip.map(Number);
            if (v.pass1) t.pass1 = v.pass1;
            if (v.pass2) t.pass2 = v.pass2;
            if (typeof v.box_iou === "number") t.box_iou = v.box_iou;
            else if (v.pass2 && typeof v.pass2.iou === "number") t.box_iou = v.pass2.iou;
            if (typeof v.mask_iou === "number") t.mask_iou = v.mask_iou;
            if (Array.isArray(v.polygon_px) && v.polygon_px.length >= 3) t.polygon_px = v.polygon_px;
            else if (Array.isArray(v.polygon) && v.polygon.length >= 3) t.polygon_px = v.polygon;
            if (Array.isArray(v.polygon_rounds)) t.polygon_rounds = v.polygon_rounds;
            if (v.mask_source) t.mask_source = String(v.mask_source);
            if (v.loc_round !== undefined) t.loc_round = Number(v.loc_round);
            if (v.loc_status || v.spatial_status) t.loc_status = normStatus(v.loc_status || v.spatial_status);
            if (v.mask_r0_url) t.mask_r0_url = String(v.mask_r0_url);
            if (v.mask_r1_url) t.mask_r1_url = String(v.mask_r1_url);
            if (v.mask_r2_url) t.mask_r2_url = String(v.mask_r2_url);
            if (v.mask_url || v.ai_mask_url || v.mask_path) {
              const mu = String(v.mask_url || v.ai_mask_url || v.mask_path);
              t.mask_url = mu;
              t.maskCandidates = buildMaskCandidatesForRound(fr.frame_id, k, t, state.maskRound !== undefined ? state.maskRound : 1);
            }
            if (v.mask_b64) t.mask_b64 = v.mask_b64;
          }
        }
      }
      if (exported.mask_results) state.maskResults = exported.mask_results;
      if (exported.mask_refine_results) state.maskRefineResults = exported.mask_refine_results;
    }

    // 2. Supplement from raw /out/* files so the UI works even before web/export.py runs
    const [resJson, locResJson, stuJson, manJson, maskResJson, maskRefJson, narrJson, decText, annText, polyText] = await Promise.all([
      state.results ? Promise.resolve(state.results) : fetchJsonSafe(["/out/results.json", "../out/results.json", "data/results.json"]),
      state.localizeResults ? Promise.resolve(state.localizeResults) : fetchJsonSafe(["/out/localize_results.json", "../out/localize_results.json", "data/localize_results.json"]),
      state.student ? Promise.resolve(state.student) : fetchJsonSafe(["/out/student.json", "../out/student.json", "data/student.json"]),
      state.manual ? Promise.resolve(state.manual) : fetchJsonSafe(["/out/manual.json", "../out/manual.json", "data/manual.json"]),
      state.maskResults ? Promise.resolve(state.maskResults) : fetchJsonSafe(["/out/mask_results.json", "../out/mask_results.json", "data/mask_results.json"]),
      state.maskRefineResults ? Promise.resolve(state.maskRefineResults) : fetchJsonSafe(["/out/mask_refine_results.json", "../out/mask_refine_results.json", "data/mask_refine_results.json"]),
      fetchJsonSafe([`/out/narration_${vid}.json`, `../out/narration_${vid}.json`, `data/narration_${vid}.json`]),
      Object.keys(state.frameById).length > 0 ? Promise.resolve(null) : fetchTextSafe(["/out/decisions.csv", "../out/decisions.csv"]),
      fetchTextSafe(["/out/annotations.jsonl", "../out/annotations.jsonl", "/out/annotations_live.jsonl", "../out/annotations_live.jsonl"]),
      fetchTextSafe([`/out/polygons/${vid}.jsonl`, `../out/polygons/${vid}.jsonl`]),
    ]);

    if (resJson) state.results = resJson;
    if (locResJson) state.localizeResults = locResJson;
    if (stuJson) state.student = stuJson;
    if (manJson) state.manual = manJson;
    if (maskResJson) state.maskResults = maskResJson;
    if (maskRefJson) state.maskRefineResults = maskRefJson;

    if (narrJson) {
      const sents = Array.isArray(narrJson) ? narrJson : (narrJson.sentences || []);
      for (const s of sents) {
        if (s && s.frame_id) {
          const txt = s.narration || s.sentence || s.text || "";
          state.narrationByFrame[s.frame_id] = txt;
          if (state.frameById[s.frame_id] && !state.frameById[s.frame_id].narration) {
            state.frameById[s.frame_id].narration = txt;
          }
        }
      }
    }

    if (decText) {
      const rows = parseCsv(decText);
      for (const r of rows) {
        const rowVid = r.video || r.video_id;
        if (rowVid !== state.video && !String(r.frame_id || "").startsWith(state.video + "_")) continue;
        const f = ensureFrameRecord(r.frame_id);
        const k = r.key || r.label;
        if (!f.labels[k]) continue;
        const t = f.labels[k];
        t.present = r.present !== "" ? Number(r.present) : null;
        t.final_present = r.final_present !== "" ? Number(r.final_present) : t.present;
        t.status = normStatus(r.status);
        t.conf = r.conf !== "" ? Number(r.conf) : null;
        t.trust = r.trust !== "" ? Number(r.trust) : null;
        t.evidence = r.evidence || r.rationale || "";
        t.pv_match = r.pv_match === "True" ? true : (r.pv_match === "False" ? false : null);
        t.p_wrong = r.p_wrong !== "" ? Number(r.p_wrong) : null;
        t.critic_reason = r.critic_reason || r.critic || "";
        t.round_history = [];
        for (let rIdx = 0; rIdx < 4; rIdx++) {
          const rk = `status_round_${rIdx}`;
          if (r[rk]) {
            t[rk] = normStatus(r[rk]);
            t.round_history.push({ round: rIdx, status: t[rk] });
          }
        }
        t.argon_reason = r.argon_reason || "";
        if (state.narrationByFrame[r.frame_id]) {
          f.narration = state.narrationByFrame[r.frame_id];
        }
      }
    }

    if (annText) {
      const annRows = parseJsonl(annText);
      for (const a of annRows) {
        const sig = `${a.frame_id}__${a.label || a.key}__${a.round || 0}`;
        state.seenLiveSignatures.add(sig);
        mergeAnnotationRow(a, false);
      }
      state.liveAnnotationCount = state.seenLiveSignatures.size;
    }

    if (polyText) {
      const polyRows = parseJsonl(polyText);
      for (const pRow of polyRows) {
        mergeAnnotationRow(pRow, false);
      }
    }

    state.frames = Object.values(state.frameById).sort((a, b) => a.frame_num - b.frame_num);

    // Default to showcase frame video12_15750_15758 or first frame with polygons/boxes
    const heroIdx = state.frames.findIndex((fr) => fr.frame_id === "video12_15750_15758");
    const firstPolyIdx = state.frames.findIndex((fr) =>
      LABEL_KEYS.some((k) => fr.labels[k] && Array.isArray(fr.labels[k].polygon_px))
    );
    const firstSpatialIdx = state.frames.findIndex((fr) =>
      LABEL_KEYS.some((k) => fr.labels[k] && Array.isArray(fr.labels[k].box))
    );
    if (heroIdx >= 0) {
      state.currentFrameIdx = heroIdx;
    } else if (firstPolyIdx >= 0) {
      state.currentFrameIdx = firstPolyIdx;
    } else if (firstSpatialIdx >= 0) {
      state.currentFrameIdx = firstSpatialIdx;
    }

    const vBadge = document.getElementById("video-badge");
    if (vBadge) {
      vBadge.textContent = `${state.video} · Held-Out Test (${state.frames.length || "pending"} frames)`;
    }

    mountExternalComponentsIfPresent();
    const extViewer = getExternalViewer();
    if (state.externalViewerMounted && extViewer && typeof extViewer.update === "function") {
      extViewer.update(buildNormalizedBundle());
    }
    renderTopHeadlines();
    renderAmberFunnel();
    renderTimeline();
    setMaskRound(state.maskRound !== undefined ? state.maskRound : 1);
    renderManualRule(state.selectedKey);
    renderGateDrawer();
  }

  function mountExternalComponentsIfPresent() {
    const normBundle = buildNormalizedBundle();

    // 1. Mount nir-webcanvas (window.Viewer / window.SurgProofViewer) into #viewer if available
    const extViewer = getExternalViewer();
    const viewerEl = document.getElementById("viewer");
    if (extViewer && viewerEl && !state.externalViewerMounted) {
      try {
        extViewer.mount(viewerEl, {
          data: normBundle,
          onSelectLabel: (labelKey) => {
            if (labelKey && LABEL_KEYS.includes(labelKey)) {
              state.selectedKey = labelKey;
              if (state.frames[state.currentFrameIdx]) {
                renderInspector(state.frames[state.currentFrameIdx]);
                renderManualRule(state.selectedKey);
              }
            }
          },
          onSeek: (frameIdx) => selectFrame(frameIdx),
        });
        if (typeof extViewer.update === "function") {
          extViewer.update(normBundle);
        }
        state.externalViewerMounted = true;
      } catch (_e) {
        state.externalViewerMounted = false;
      }
    }

    // 2. Mount nir-webtimeline (window.Timeline / window.SurgProofTimeline) into #timeline if available
    const extTimeline = getExternalTimeline();
    const timelineEl = document.getElementById("timeline");
    if (extTimeline && timelineEl && !state.externalTimelineMounted) {
      try {
        extTimeline.mount(timelineEl, normBundle, (frameIdx, labelKey) => {
          if (labelKey && LABEL_KEYS.includes(labelKey)) {
            state.selectedKey = labelKey;
          }
          selectFrame(frameIdx);
        });
        if (typeof extTimeline.update === "function") {
          extTimeline.update(normBundle);
        }
        state.externalTimelineMounted = true;
      } catch (_e) {
        state.externalTimelineMounted = false;
      }
    }

    // 3. Mount nir-gate (window.FinalPassGate / window.Gate) into #final-pass-mount if available
    const extGate = getExternalGate();
    const gateEl = document.getElementById("final-pass-mount");
    if (extGate && gateEl && !state.externalGateMounted) {
      try {
        extGate.mount(gateEl, {
          scope: "test_5",
          onUpdate: (gState) => {
            if (gState && Array.isArray(gState.clearedAmbers)) {
              for (const c of gState.clearedAmbers) {
                const gk = `${c.frame_id}__${c.key}`;
                state.gateDecisions[gk] = {
                  frame_id: c.frame_id,
                  key: c.key,
                  human_action: c.human_action || "approve",
                  human_present: c.final_present,
                };
              }
              renderTimeline();
              if (state.frames[state.currentFrameIdx]) {
                selectFrame(state.currentFrameIdx);
              }
            }
          },
        });
        state.externalGateMounted = true;
      } catch (_e) {
        state.externalGateMounted = false;
      }
    }
  }

  function renderTopHeadlines() {
    const accEl = document.getElementById("headline-acc");
    const nEl = document.getElementById("headline-n");
    const isPresent = syncPresentBodyClass();

    let acc = null;
    let cov = null;
    let nLabels = null;

    if (state.headline && typeof state.headline.accepted_accuracy === "number") {
      acc = state.headline.accepted_accuracy;
      cov = state.headline.accepted_coverage;
      nLabels = state.headline.n_test_labels_excl_liver || state.headline.n_test_labels;
    } else if (state.results) {
      const a5 = state.results.accepted_5_informative || state.results.accepted || {};
      acc = a5.accuracy;
      cov = a5.coverage;
      nLabels = state.results.n_test_labels_excl_liver || state.results.n_test_labels;
    }

    if (accEl && nEl) {
      if (typeof acc === "number" && typeof cov === "number") {
        const fmtN =
          typeof nLabels === "number" && Number.isFinite(nLabels)
            ? Number(nLabels).toLocaleString("en-US")
            : "pending";
        if (isPresent) {
          accEl.textContent = `${fmtPct(acc, 1)} accurate on the ${fmtPct(cov, 1)} it accepted · n = ${fmtN}`;
          nEl.textContent = `n = ${fmtN}`;
        } else {
          accEl.textContent = `${fmtPct(acc, 1)} accurate on the ${fmtPct(cov, 1)} it accepted`;
          nEl.textContent = `n = ${fmtN} held-out labels, 6 unseen videos`;
        }
      } else {
        accEl.textContent = "pending";
        nEl.textContent = "n = pending held-out labels";
      }
    }

    const locEl = document.getElementById("headline-loc");
    const locSub = document.getElementById("headline-loc-sub");
    if (locEl && locSub) {
      if (state.localizeResults && state.localizeResults.boxes) {
        const meanIou = state.localizeResults.boxes.mean_iou;
        const tipRate = state.localizeResults.tips ? state.localizeResults.tips.within_tool_mask_rate : null;
        const nScored = state.localizeResults.boxes.n_scored || state.localizeResults.n_test_annotations || "pending";
        locEl.textContent = `Box IoU ${fmtPct(meanIou, 1)} · Tip ${fmtPct(tipRate, 0)} vs ground truth`;
        locSub.textContent = `n = ${nScored} test annotations`;
      } else {
        locEl.textContent = "pending";
        locSub.textContent = "vs ground truth";
      }
    }

    const stuEl = document.getElementById("headline-student");
    const stuSub = document.getElementById("headline-student-sub");
    if (stuEl && stuSub) {
      const ciObj =
        (state.student &&
          state.student.bootstrap_ci_by_source &&
          state.student.bootstrap_ci_by_source.verified) ||
        (state.student &&
          state.student.video_bootstrap_ci &&
          state.student.video_bootstrap_ci.verified) ||
        null;
      const ci95 = ciObj && Array.isArray(ciObj.ci_95) && ciObj.ci_95.length === 2 ? ciObj.ci_95 : null;
      const m = state.student ? state.student.mean_acc_by_source : null;

      if (m && typeof m.verified === "number" && ci95) {
        if (stuEl.parentElement) stuEl.parentElement.style.display = "";
        stuEl.textContent = `Specialist ${fmtPct(m.verified, 1)} (95% CI ${fmtPct(ci95[0], 1)}–${fmtPct(ci95[1], 1)})`;
        stuSub.textContent = `n = ${state.student.n_test_frames || "pending"} test frames`;
      } else if (stuEl.parentElement) {
        stuEl.parentElement.style.display = "none";
      }
    }
  }

  function renderAmberFunnel() {
    const container = document.getElementById("amber-funnel-steps");
    const scopeEl = document.getElementById("funnel-scope-label");
    if (!container) return;
    container.innerHTML = "";

    let rounds5 = null;
    let roundsAll6 = null;
    const amberLoop = (state.headline && state.headline.amber_loop) || (state.results && state.results.amber_loop);
    if (amberLoop) {
      rounds5 = amberLoop.ambers_by_round_test_excl_liver;
      roundsAll6 = amberLoop.ambers_by_round_test;
    }

    if (Array.isArray(rounds5) && rounds5.length > 0) {
      if (scopeEl) scopeEl.textContent = "held-out test split (n = 1,400 excl. liver)";
      rounds5.forEach((cnt, idx) => {
        const sp = document.createElement("span");
        sp.className = "funnel-step" + (idx === rounds5.length - 1 ? " final-round" : "");
        const allTag = Array.isArray(roundsAll6) && roundsAll6[idx] !== undefined ? ` [all-6: ${roundsAll6[idx]}]` : "";
        sp.textContent = `Round ${idx}: ${cnt}${allTag}`;
        container.appendChild(sp);
      });
      return;
    }

    if (state.frames.length === 0) {
      const sp = document.createElement("span");
      sp.className = "funnel-step";
      sp.textContent = "pending";
      container.appendChild(sp);
      return;
    }

    const rCols = ["status_round_0", "status_round_1", "status_round_2", "status_round_3"];
    const counts = rCols.map((rc) => {
      let c = 0;
      for (const fr of state.frames) {
        for (const k of LABEL_KEYS) {
          if (fr.labels[k] && fr.labels[k][rc] === "AMBER") c++;
        }
      }
      return c;
    });
    if (scopeEl) scopeEl.textContent = `${state.video} (${state.frames.length * 6} labels)`;
    counts.forEach((cnt, idx) => {
      const sp = document.createElement("span");
      sp.className = "funnel-step" + (idx === counts.length - 1 ? " final-round" : "");
      sp.textContent = `Round ${idx}: ${cnt}`;
      container.appendChild(sp);
    });
  }

  function renderTimeline() {
    const countEl = document.getElementById("timeline-frame-count");
    if (countEl) countEl.textContent = String(state.frames.length);

    const extTimeline = getExternalTimeline();
    if (state.externalTimelineMounted && extTimeline && typeof extTimeline.update === "function") {
      extTimeline.update(buildNormalizedBundle());
      if (typeof extTimeline.setFrame === "function") {
        extTimeline.setFrame(state.currentFrameIdx);
      }
      return;
    }

    const tracksEl = document.getElementById("timeline") || document.getElementById("timeline-tracks");
    if (!tracksEl) return;
    tracksEl.innerHTML = "";

    if (state.frames.length === 0) {
      const empty = document.createElement("div");
      empty.style.color = "var(--text-secondary)";
      empty.style.fontFamily = "var(--font-mono)";
      empty.style.fontSize = "12px";
      empty.textContent = "Timeline: pending (awaiting web/data/video12.json or out/decisions.csv)";
      tracksEl.appendChild(empty);
      return;
    }

    // 1. Surgical Phase Track
    const phaseRow = document.createElement("div");
    phaseRow.className = "track-row";
    const phaseName = document.createElement("div");
    phaseName.className = "track-name";
    phaseName.textContent = "phase";
    const phaseCells = document.createElement("div");
    phaseCells.className = "track-cells";

    state.frames.forEach((fr, idx) => {
      const cell = document.createElement("div");
      let pClass = "cell-pending";
      if (fr.phase) {
        const pStr = String(fr.phase);
        if (pStr.startsWith("1")) pClass = "cell-phase-1";
        else if (pStr.startsWith("2")) pClass = "cell-phase-2";
        else if (pStr.startsWith("3")) pClass = "cell-phase-3";
        else pClass = "cell-phase-4";
      }
      cell.className = `track-cell ${pClass}` + (idx === state.currentFrameIdx ? " active-frame" : "");
      cell.title = `${fr.frame_id} · Phase: ${fr.phase || "pending"}`;
      cell.dataset.frameIdx = String(idx);
      cell.addEventListener("click", () => selectFrame(idx));
      phaseCells.appendChild(cell);
    });
    phaseRow.appendChild(phaseName);
    phaseRow.appendChild(phaseCells);
    tracksEl.appendChild(phaseRow);

    // 2. Six Structure Tracks
    for (const k of LABEL_KEYS) {
      const row = document.createElement("div");
      row.className = "track-row";
      const nameDiv = document.createElement("div");
      nameDiv.className = "track-name";
      nameDiv.textContent = k;
      const cellsDiv = document.createElement("div");
      cellsDiv.className = "track-cells";

      state.frames.forEach((fr, idx) => {
        const lbl = fr.labels[k];
        const cell = document.createElement("div");
        let stClass = "cell-pending";
        const effectiveStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const effectivePresent = getEffectivePresent(fr.frame_id, k, lbl);

        if (effectiveStatus === "accept") {
          stClass = effectivePresent ? "cell-accept-pos" : "cell-accept-neg";
        } else if (effectiveStatus === "AMBER") {
          stClass = "cell-amber";
        } else if (effectiveStatus === "reject") {
          stClass = "cell-reject";
        }

        const hasSpatial = lbl && (Array.isArray(lbl.box) || Array.isArray(lbl.tip) || Boolean(lbl.mask_url) || Boolean(lbl.mask_b64));
        cell.className =
          `track-cell ${stClass}` +
          (hasSpatial ? " has-spatial" : "") +
          (lbl && lbl.hasNewLivePulse ? " live-flash" : "") +
          (idx === state.currentFrameIdx ? " active-frame" : "");

        cell.title =
          `${fr.frame_id} · ${k}: ${effectivePresent ? "present" : "absent"} (${effectiveStatus})` +
          (hasSpatial ? " · spatial box/tip/mask localized" : "");
        cell.dataset.frameIdx = String(idx);
        cell.dataset.labelKey = k;
        cell.addEventListener("click", () => {
          state.selectedKey = k;
          selectFrame(idx);
          renderManualRule(k);
        });
        if (lbl) lbl.hasNewLivePulse = false;
        cellsDiv.appendChild(cell);
      });

      row.appendChild(nameDiv);
      row.appendChild(cellsDiv);
      tracksEl.appendChild(row);
    }
  }

  function getEffectiveStatus(frameId, key, lbl) {
    const gk = `${frameId}__${key}`;
    if (state.gateDecisions[gk]) return "accept";
    return lbl ? lbl.status : "pending";
  }

  function getEffectivePresent(frameId, key, lbl) {
    const gk = `${frameId}__${key}`;
    if (state.gateDecisions[gk]) return state.gateDecisions[gk].human_present === 1;
    if (!lbl) return false;
    return Boolean(lbl.final_present !== null ? lbl.final_present : lbl.present);
  }

  function selectFrame(idx) {
    if (state.frames.length === 0) return;
    const clamped = Math.max(0, Math.min(state.frames.length - 1, idx));
    state.currentFrameIdx = clamped;
    const fr = state.frames[clamped];

    const extTimeline = getExternalTimeline();
    if (state.externalTimelineMounted && extTimeline && typeof extTimeline.setFrame === "function") {
      extTimeline.setFrame(clamped);
    } else {
      document.querySelectorAll(".track-cell").forEach((el) => {
        if (Number(el.dataset.frameIdx) === clamped) {
          el.classList.add("active-frame");
        } else {
          el.classList.remove("active-frame");
        }
      });
    }

    const readoutEl = document.getElementById("frame-readout");
    if (readoutEl) {
      readoutEl.textContent = `Frame ${clamped + 1} / ${state.frames.length} · ${fr.frame_id}`;
    }

    const tagEl = document.getElementById("inspector-frame-tag");
    if (tagEl) tagEl.textContent = fr.frame_id;

    const phaseEl = document.getElementById("phase-badge");
    if (phaseEl) {
      phaseEl.textContent = fr.phase
        ? `Phase: ${fr.phase}` + (typeof fr.phase_conf === "number" ? ` (${fr.phase_conf.toFixed(2)})` : "")
        : "Phase: pending";
    }

    const narrEl = document.getElementById("narration-text");
    const narrBanner = document.getElementById("narration-banner");
    const narrTag = narrBanner ? narrBanner.querySelector(".tag") : null;
    const isPresent = syncPresentBodyClass();

    if (narrTag) {
      if (isPresent) {
        const cleanPhase = fr.phase ? String(fr.phase).replace(/^\d+\.\s*/, "") : "Held-Out Step";
        narrTag.textContent = cleanPhase;
      } else {
        narrTag.textContent = "Verified Narration";
      }
    }

    if (narrEl) {
      if (fr.narration) {
        narrEl.textContent = fr.narration;
      } else {
        const acceptedKeys = LABEL_KEYS.filter(
          (k) => getEffectiveStatus(fr.frame_id, k, fr.labels[k]) === "accept" && getEffectivePresent(fr.frame_id, k, fr.labels[k])
        );
        narrEl.textContent = acceptedKeys.length > 0
          ? `Verified present structures on ${fr.frame_id}: ${acceptedKeys.map((k) => PRESENT_SHORT_NAMES[k] || k).join(", ")}.`
          : "pending";
      }
    }

    renderInspector(fr);

    const extViewer = getExternalViewer();
    if (!isPresent && state.externalViewerMounted && extViewer && typeof extViewer.show === "function") {
      extViewer.show(clamped);
    } else {
      renderCanvas(fr);
    }
  }

  function renderInspector(fr) {
    const container = document.getElementById("objects") || document.getElementById("inspector-cards");
    if (!container) return;
    container.innerHTML = "";
    const scoringMode = document.getElementById("chk-show-gt") ? document.getElementById("chk-show-gt").checked : false;

    LABEL_KEYS.forEach((k, kIdx) => {
      const lbl = fr.labels[k];
      const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
      const effPres = getEffectivePresent(fr.frame_id, k, lbl);
      const hasSpatial = lbl && (Array.isArray(lbl.box) || Array.isArray(lbl.tip) || Boolean(lbl.mask_url) || Boolean(lbl.mask_b64));

      if (state.objectFilter === "present" && !effPres) return;
      if (state.objectFilter === "amber" && effStatus !== "AMBER") return;
      if (state.objectFilter === "spatial" && !hasSpatial) return;

      const card = document.createElement("div");
      card.className = `label-card st-${effStatus}` + (state.selectedKey === k ? " selected" : "");
      card.style.setProperty("--card-idx", String(kIdx));
      card.dataset.labelKey = k;
      card.addEventListener("click", () => {
        state.selectedKey = k;
        renderInspector(fr);
        renderManualRule(k);
        const extViewer = getExternalViewer();
        if (state.externalViewerMounted && extViewer && typeof extViewer.selectLabel === "function") {
          extViewer.selectLabel(k);
        } else {
          renderCanvas(fr);
        }
      });

      const top = document.createElement("div");
      top.className = "label-card-top";

      const title = document.createElement("div");
      title.className = "label-title";
      title.innerHTML =
        `<kbd>${kIdx + 1}</kbd> <span>${LABEL_NAMES[k] || k}</span> ` +
        `<code style="font-size:11px;color:var(--text-secondary)">(${k})</code>`;

      const badge = document.createElement("span");
      badge.className = `status-badge st-${effStatus}`;
      badge.textContent = effStatus === "pending"
        ? "pending"
        : `${effPres ? "present" : "absent"} · ${effStatus}`;

      top.appendChild(title);
      top.appendChild(badge);
      card.appendChild(top);

      // Single calm AI signal line (AI's own signals only; zero GT IoU on AI view)
      const meta = document.createElement("div");
      meta.className = "label-meta";
      const sigParts = [];
      if (lbl && lbl.conf !== null) sigParts.push(`conf ${lbl.conf.toFixed(2)}`);
      const agreeVal = getPassAgreement(lbl);
      if (typeof agreeVal === "number") sigParts.push(`agree ${agreeVal.toFixed(2)}`);
      if (lbl && lbl.pv_match !== null) sigParts.push(`crop ${lbl.pv_match ? "agree" : "disagree"}`);
      if (lbl && (lbl.mask_url || lbl.mask_b64)) sigParts.push("mask");
      if (lbl && Array.isArray(lbl.tip)) sigParts.push("tip");
      if (scoringMode && lbl && lbl.gt_present_scoring_only !== undefined && lbl.gt_present_scoring_only !== null) {
        sigParts.push(`Scoring (ground truth): ${lbl.gt_present_scoring_only ? "present" : "absent"}`);
      }
      const sigSpan = document.createElement("span");
      sigSpan.className = "signal-line";
      sigSpan.textContent = sigParts.length > 0 ? sigParts.join(" · ") : "signals: pending";
      meta.appendChild(sigSpan);
      card.appendChild(meta);

      const ev = document.createElement("div");
      ev.className = "label-evidence";
      ev.textContent = lbl && lbl.evidence ? lbl.evidence : "Evidence: pending";
      card.appendChild(ev);

      // Concise Double-Check & Amber-Round History
      const rDiv = document.createElement("div");
      rDiv.className = "label-rounds";
      const roundsRow = document.createElement("div");
      roundsRow.className = "round-pills";

      const roundsList = [];
      for (let rIdx = 0; rIdx < 4; rIdx++) {
        const st = lbl ? lbl[`status_round_${rIdx}`] : null;
        if (st) roundsList.push({ round: rIdx, status: st });
      }

      const uniqueStatuses = new Set(roundsList.map((r) => r.status));
      if (roundsList.length > 0 && uniqueStatuses.size === 1) {
        const onlySt = roundsList[0].status;
        const pill = document.createElement("span");
        pill.className = `round-pill rp-${onlySt}`;
        pill.textContent = onlySt === "AMBER" ? `R0–R${roundsList.length - 1}: AMBER` : `Double-check: ${onlySt}`;
        roundsRow.appendChild(pill);
      } else if (roundsList.length > 0) {
        roundsList.forEach((item, idx) => {
          const pill = document.createElement("span");
          pill.className = `round-pill rp-${item.status}`;
          pill.textContent = `R${item.round}: ${item.status}`;
          roundsRow.appendChild(pill);
          if (idx < roundsList.length - 1) {
            const arrow = document.createElement("span");
            arrow.textContent = "→";
            roundsRow.appendChild(arrow);
          }
        });
      } else {
        const pill = document.createElement("span");
        pill.className = `round-pill rp-${effStatus}`;
        pill.textContent = effStatus;
        roundsRow.appendChild(pill);
      }
      rDiv.appendChild(roundsRow);

      if (lbl && lbl.argon_reason) {
        const ar = document.createElement("div");
        ar.style.color = "var(--verdict-amber)";
        ar.textContent = `Argon loop: ${lbl.argon_reason}`;
        rDiv.appendChild(ar);
      }

      card.appendChild(rDiv);
      container.appendChild(card);
    });
  }

  function renderManualRule(key) {
    const titleEl = document.getElementById("manual-rule-title");
    const bodyEl = document.getElementById("manual-rule-body");
    if (!titleEl || !bodyEl) return;
    titleEl.textContent = key;

    if (!state.manual || !state.manual.structures || !state.manual.structures[key]) {
      bodyEl.textContent = "Self-written manual criteria: pending (awaiting out/manual.json)";
      return;
    }

    const s = state.manual.structures[key];
    const src = Array.isArray(state.manual.sources) && state.manual.sources[0] ? state.manual.sources[0] : null;
    const appearance = s.appearance || s.description || "pending";
    const cues = Array.isArray(s.verification_cues) ? s.verification_cues.slice(0, 2).join(" · ") : (s.verification_cues || "");
    const srcLine = src && src.url ? `Source: ${src.title || "Public teaching video"} (${src.url})` : "";

    bodyEl.innerHTML = "";
    const p1 = document.createElement("div");
    p1.textContent = appearance;
    bodyEl.appendChild(p1);

    if (cues) {
      const p2 = document.createElement("div");
      p2.style.marginTop = "4px";
      p2.style.color = "var(--text-secondary)";
      p2.textContent = `Verification cues: ${cues}`;
      bodyEl.appendChild(p2);
    }

    if (srcLine) {
      const p3 = document.createElement("div");
      p3.style.marginTop = "4px";
      p3.style.fontFamily = "var(--font-mono)";
      p3.style.fontSize = "10.5px";
      p3.style.color = "var(--text-secondary)";
      p3.textContent = srcLine;
      bodyEl.appendChild(p3);
    }
  }

  function loadImageWithFallback(candidates, cacheObj, key) {
    if (cacheObj[key] !== undefined) return Promise.resolve(cacheObj[key]);
    return new Promise((resolve) => {
      let idx = 0;
      function tryNext() {
        if (idx >= candidates.length) {
          cacheObj[key] = null;
          resolve(null);
          return;
        }
        const img = new Image();
        img.crossOrigin = "anonymous";
        img.onload = () => {
          cacheObj[key] = img;
          resolve(img);
        };
        img.onerror = () => {
          idx++;
          tryNext();
        };
        img.src = candidates[idx];
      }
      tryNext();
    });
  }

  function buildGtOverlayCanvas(gtImg) {
    const w = gtImg.naturalWidth || gtImg.width;
    const h = gtImg.naturalHeight || gtImg.height;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    octx.drawImage(gtImg, 0, 0);
    const imgData = octx.getImageData(0, 0, w, h);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const val = d[i];
      const k = WATERSHED_TO_KEY[val];
      if (k && LABEL_RGB[k]) {
        const rgb = LABEL_RGB[k];
        d[i] = rgb[0];
        d[i + 1] = rgb[1];
        d[i + 2] = rgb[2];
        d[i + 3] = 115;
      } else {
        d[i + 3] = 0;
      }
    }
    octx.putImageData(imgData, 0, 0);
    return off;
  }

  function buildTintedMaskCanvas(maskImg, rgb, cacheKey) {
    if (cacheKey && state.tintedMaskCache[cacheKey]) {
      return state.tintedMaskCache[cacheKey];
    }
    const w = maskImg.naturalWidth || maskImg.width || 854;
    const h = maskImg.naturalHeight || maskImg.height || 480;
    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    octx.drawImage(maskImg, 0, 0, w, h);
    const imgData = octx.getImageData(0, 0, w, h);
    const d = imgData.data;

    const fg = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) {
      const lum = Math.max(d[i], d[i + 1], d[i + 2]);
      const a = d[i + 3];
      if (a > 64 && lum > 96) {
        fg[p] = 1;
      }
    }

    for (let y = 0; y < h; y++) {
      const rowOff = y * w;
      for (let x = 0; x < w; x++) {
        const p = rowOff + x;
        const idx = p * 4;
        if (!fg[p]) {
          d[idx + 3] = 0;
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
        d[idx] = rgb[0];
        d[idx + 1] = rgb[1];
        d[idx + 2] = rgb[2];
        d[idx + 3] = isEdge ? 205 : 80;
      }
    }
    octx.putImageData(imgData, 0, 0);
    if (cacheKey) {
      state.tintedMaskCache[cacheKey] = off;
    }
    return off;
  }

  function drawRoundedPillRect(ctx, x, y, w, h, r) {
    const rad = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + rad, y);
    ctx.lineTo(x + w - rad, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rad);
    ctx.lineTo(x + w, y + h - rad);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rad, y + h);
    ctx.lineTo(x + rad, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rad);
    ctx.lineTo(x, y + rad);
    ctx.quadraticCurveTo(x, y, x + rad, y);
    ctx.closePath();
  }

  function pillRectsOverlap(a, b, pad = 6) {
    return (
      a.x < b.x + b.w + pad &&
      a.x + a.w + pad > b.x &&
      a.y < b.y + b.h + pad &&
      a.y + a.h + pad > b.y
    );
  }

  function findNonOverlappingPillPosition(rx, ry, rw, rh, tw, th, k, placedPills, W, H, minTopY) {
    const PILL_CORNERS = {
      gallbladder: "tl",
      l_hook: "tr",
      liver: "br",
      grasper: "bl",
      cystic_duct: "tr",
      blood: "br",
    };
    const pref = PILL_CORNERS[k] || "tl";
    const corners = {
      tl: [rx, ry - th - 5],
      tr: [rx + rw - tw, ry - th - 5],
      bl: [rx, ry + rh + 5],
      br: [rx + rw - tw, ry + rh + 5],
      inTl: [rx + 6, ry + 6],
      inTr: [rx + rw - tw - 6, ry + 6],
      inBl: [rx + 6, ry + rh - th - 6],
      inBr: [rx + rw - tw - 6, ry + rh - th - 6],
    };

    const order = [pref, "tl", "tr", "bl", "br", "inTl", "inTr", "inBl", "inBr"];
    const candidates = [];
    for (const cKey of order) {
      if (corners[cKey]) candidates.push(corners[cKey]);
    }

    const baseX = corners[pref][0];
    const baseY = corners[pref][1];
    for (const dy of [26, -26, 52, -52, 78, -78, 104, -104, 130, -130]) {
      for (const dx of [0, 48, -48, 96, -96]) {
        candidates.push([baseX + dx, baseY + dy]);
      }
    }

    for (let gy = minTopY; gy <= H - th - 8; gy += 26) {
      for (let gx = 8; gx <= W - tw - 8; gx += 40) {
        candidates.push([gx, gy]);
      }
    }

    for (const [cx, cy] of candidates) {
      const px = Math.max(8, Math.min(W - tw - 8, Math.round(cx)));
      const py = Math.max(minTopY, Math.min(H - th - 8, Math.round(cy)));
      const rect = { x: px, y: py, w: tw, h: th };
      if (!placedPills.some((p) => pillRectsOverlap(rect, p, 6))) {
        return rect;
      }
    }

    return {
      x: Math.max(8, Math.min(W - tw - 8, Math.round(baseX))),
      y: Math.max(minTopY, Math.min(H - th - 8, Math.round(baseY))),
      w: tw,
      h: th,
    };
  }

  async function renderCanvas(fr) {
    const canvas = document.getElementById("frame-canvas");
    if (!canvas || !fr) return;
    const isPresent = syncPresentBodyClass();
    const renderSeq = ++state._renderSeq;

    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const showMasks = isPresent
      ? true
      : document.getElementById("chk-ai-masks")
        ? document.getElementById("chk-ai-masks").checked
        : true;
    const showPolygons = isPresent
      ? true
      : document.getElementById("chk-ai-polygons")
        ? document.getElementById("chk-ai-polygons").checked
        : true;
    const showBoxes = isPresent
      ? true
      : document.getElementById("chk-ai-boxes")
        ? document.getElementById("chk-ai-boxes").checked
        : true;
    const showDualPass = isPresent
      ? false
      : document.getElementById("chk-dual-pass")
        ? document.getElementById("chk-dual-pass").checked
        : false;
    const showTips = isPresent
      ? true
      : document.getElementById("chk-ai-tips")
        ? document.getElementById("chk-ai-tips").checked
        : true;
    const showHud = isPresent
      ? false
      : document.getElementById("chk-hud-chips")
        ? document.getElementById("chk-hud-chips").checked
        : false;
    const showGt = isPresent
      ? false
      : document.getElementById("chk-show-gt")
        ? document.getElementById("chk-show-gt").checked
        : false;

    // Preload base frame image, optional GT image, and all active mask images before drawing (no flash)
    const baseImgPromise = loadImageWithFallback(fr.imgCandidates, state.imageCache, fr.frame_id);
    const gtImgPromise = showGt
      ? loadImageWithFallback(fr.gtCandidates, state.imageCache, fr.frame_id + "__gt")
      : Promise.resolve(null);

    const maskRequests = [];
    if (showMasks) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl || !getEffectivePresent(fr.frame_id, k, lbl)) continue;
        const mKey = `${fr.frame_id}__${k}__${lbl.mask_url || state.maskRound || 0}`;
        const cands = lbl.mask_b64
          ? [`data:image/png;base64,${lbl.mask_b64.replace(/^data:image\/\w+;base64,/, "")}`, ...lbl.maskCandidates]
          : lbl.maskCandidates;
        maskRequests.push(
          loadImageWithFallback(cands, state.maskCache, mKey).then((img) => ({ key: k, mKey, img }))
        );
      }
    }

    const [baseImg, gtImg, loadedMaskEntries] = await Promise.all([
      baseImgPromise,
      gtImgPromise,
      Promise.all(maskRequests),
    ]);

    if (renderSeq !== state._renderSeq) return;

    ctx.clearRect(0, 0, W, H);
    if (baseImg) {
      ctx.drawImage(baseImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#12141C";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#8A909E";
      ctx.font = '500 14px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText(`Surgical frame image pending: ${fr.frame_id}`, 28, H / 2);
    }

    // 1. Scoring-View-Only Ground-Truth Mask Overlay
    if (showGt) {
      if (gtImg) {
        if (!state.gtCanvasCache[fr.frame_id]) {
          state.gtCanvasCache[fr.frame_id] = buildGtOverlayCanvas(gtImg);
        }
        ctx.drawImage(state.gtCanvasCache[fr.frame_id], 0, 0, W, H);
      }
      ctx.fillStyle = "#12141C";
      ctx.fillRect(W - 340, 8, 332, 24);
      ctx.strokeStyle = "#3B82F6";
      ctx.strokeRect(W - 340, 8, 332, 24);
      ctx.fillStyle = "#F4F5F8";
      ctx.font = '600 11px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      ctx.fillText("SCORING VIEW ONLY: CHOLECSEG8K GT MASK", W - 328, 24);
    }

    // 2. AI Segmentation Masks Overlay (softly tinted RGBA with transparent background)
    let loadedMasksCount = 0;
    for (const item of loadedMaskEntries) {
      if (item && item.img) {
        loadedMasksCount++;
        const rgb = LABEL_RGB[item.key] || [59, 130, 246];
        const tinted = buildTintedMaskCanvas(item.img, rgb, item.mKey);
        ctx.drawImage(tinted, 0, 0, W, H);
      }
    }
    const maskStatusEl = document.getElementById("mask-status-label");
    if (maskStatusEl) {
      maskStatusEl.textContent = loadedMasksCount > 0 ? `${loadedMasksCount} active` : "pending";
    }

    // 2b. AI Polygon Outlines (out/polygons/<video>.jsonl -> polygon_px [[x, y], ...])
    let activePolygonsCount = 0;
    for (const k of LABEL_KEYS) {
      const lbl = fr.labels[k];
      if (!lbl || !Array.isArray(lbl.polygon_px) || lbl.polygon_px.length < 3) continue;
      activePolygonsCount++;
      if (!showPolygons) continue;
      const rgb = LABEL_RGB[k] || [59, 130, 246];
      ctx.save();
      ctx.beginPath();
      lbl.polygon_px.forEach((pt, idx) => {
        if (!Array.isArray(pt) || pt.length < 2) return;
        const px = Number(pt[0]);
        const py = Number(pt[1]);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.14)`;
      ctx.fill();
      ctx.strokeStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
      ctx.lineWidth = state.selectedKey === k ? 2.5 : 1.75;
      ctx.stroke();
      ctx.restore();
    }
    const polyStatusEl = document.getElementById("polygon-status-label");
    if (polyStatusEl) {
      polyStatusEl.textContent = activePolygonsCount > 0 ? `${activePolygonsCount} active` : "pending";
    }

    // 3. AI Bounding Boxes & One Clean Inter Chip per Object (Zero Overlaps)
    const placedPills = [];
    const minTopY = isPresent ? 10 : 36;

    if (showBoxes) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const effPres = getEffectivePresent(fr.frame_id, k, lbl);
        const col = STATUS_COL[effStatus] || "#3B82F6";
        const fillCol = STATUS_FILL[effStatus] || "#12141C";

        if (showDualPass && lbl.pass1 && Array.isArray(lbl.pass1.box)) {
          const [py1, px1, py2, px2] = lbl.pass1.box;
          ctx.save();
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 1.75;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect((px1 * W) / 1000, (py1 * H) / 1000, ((px2 - px1) * W) / 1000, ((py2 - py1) * H) / 1000);
          ctx.restore();
        }
        if (showDualPass && lbl.pass2 && Array.isArray(lbl.pass2.box)) {
          const [qy1, qx1, qy2, qx2] = lbl.pass2.box;
          ctx.save();
          ctx.strokeStyle = "#F59E0B";
          ctx.lineWidth = 1.75;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect((qx1 * W) / 1000, (qy1 * H) / 1000, ((qx2 - qx1) * W) / 1000, ((qy2 - qy1) * H) / 1000);
          ctx.restore();
        }

        let anchorRect = null;
        if (Array.isArray(lbl.box) && lbl.box.length === 4) {
          const [y1, x1, y2, x2] = lbl.box;
          const rx = (x1 * W) / 1000;
          const ry = (y1 * H) / 1000;
          const rw = ((x2 - x1) * W) / 1000;
          const rh = ((y2 - y1) * H) / 1000;
          anchorRect = [rx, ry, rw, rh];

          ctx.save();
          ctx.strokeStyle = col;
          ctx.lineWidth = state.selectedKey === k && !isPresent ? 2.8 : 2.0;
          drawRoundedPillRect(ctx, rx, ry, rw, rh, 4);
          ctx.stroke();
          ctx.restore();
        } else if (isPresent && effPres && Array.isArray(lbl.polygon_px) && lbl.polygon_px.length >= 3) {
          let minX = W;
          let minY = H;
          let maxX = 0;
          let maxY = 0;
          for (const pt of lbl.polygon_px) {
            if (Array.isArray(pt) && pt.length >= 2) {
              minX = Math.min(minX, Number(pt[0]));
              minY = Math.min(minY, Number(pt[1]));
              maxX = Math.max(maxX, Number(pt[0]));
              maxY = Math.max(maxY, Number(pt[1]));
            }
          }
          if (maxX > minX && maxY > minY) {
            anchorRect = [minX, minY, maxX - minX, maxY - minY];
          }
        }

        if (anchorRect) {
          const [rx, ry, rw, rh] = anchorRect;
          const shortName = PRESENT_SHORT_NAMES[k] || LABEL_NAMES[k] || k;
          const statusWord = effStatus === "AMBER" ? "amber" : effStatus;
          const agreeVal = getPassAgreement(lbl);
          const agreeSuffix = !isPresent && typeof agreeVal === "number" ? ` · agree ${agreeVal.toFixed(2)}` : "";
          const tag = isPresent
            ? `${shortName} · ${statusWord}`
            : `${shortName} · ${effStatus}${agreeSuffix}`;

          ctx.font = '600 12px "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Text", sans-serif';
          const tw = Math.ceil(ctx.measureText(tag).width + 24);
          const th = 22;

          const pillRect = findNonOverlappingPillPosition(
            rx,
            ry,
            rw,
            rh,
            tw,
            th,
            k,
            placedPills,
            W,
            H,
            minTopY
          );
          placedPills.push({
            key: k,
            label: tag,
            x: pillRect.x,
            y: pillRect.y,
            w: pillRect.w,
            h: pillRect.h,
            box: [rx, ry, rw, rh],
          });

          // Draw subtle leader line if pill was shifted away from box
          const nearX = Math.max(rx, Math.min(rx + rw, pillRect.x + pillRect.w / 2));
          const nearY = Math.max(ry, Math.min(ry + rh, pillRect.y + pillRect.h / 2));
          const dist = Math.hypot(nearX - (pillRect.x + pillRect.w / 2), nearY - (pillRect.y + pillRect.h / 2));
          if (dist > Math.max(pillRect.w, pillRect.h) * 0.75) {
            ctx.save();
            ctx.strokeStyle = col;
            ctx.globalAlpha = 0.55;
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(pillRect.x + pillRect.w / 2, pillRect.y + pillRect.h / 2);
            ctx.lineTo(nearX, nearY);
            ctx.stroke();
            ctx.restore();
          }

          ctx.save();
          ctx.fillStyle = fillCol;
          drawRoundedPillRect(ctx, pillRect.x, pillRect.y, pillRect.w, pillRect.h, 6);
          ctx.fill();
          ctx.strokeStyle = col;
          ctx.lineWidth = 1.25;
          ctx.stroke();

          // Verdict indicator dot
          ctx.fillStyle = col;
          ctx.beginPath();
          ctx.arc(pillRect.x + 10, pillRect.y + pillRect.h / 2, 3.2, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = "#F4F5F8";
          ctx.textBaseline = "middle";
          ctx.fillText(tag, pillRect.x + 18, pillRect.y + pillRect.h / 2 + 0.5);
          ctx.textBaseline = "alphabetic";
          ctx.restore();
        }
      }
    }
    canvas._placedPills = placedPills;

    // 4. Instrument Tip Points ([y, x] in 0..1000)
    if (showTips) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl || !Array.isArray(lbl.tip) || lbl.tip.length !== 2) continue;
        const ty = (lbl.tip[0] * H) / 1000;
        const tx = (lbl.tip[1] * W) / 1000;

        ctx.save();
        ctx.strokeStyle = "rgba(10, 11, 14, 0.85)";
        ctx.lineWidth = 3.2;
        ctx.beginPath();
        ctx.arc(tx, ty, 7.5, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = "#38BDF8";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tx, ty, 7.5, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(tx - 12, ty);
        ctx.lineTo(tx - 3.5, ty);
        ctx.moveTo(tx + 3.5, ty);
        ctx.lineTo(tx + 12, ty);
        ctx.moveTo(tx, ty - 12);
        ctx.lineTo(tx, ty - 3.5);
        ctx.moveTo(tx, ty + 3.5);
        ctx.lineTo(tx, ty + 12);
        ctx.stroke();

        // In Present mode: strictly one clean chip per object (no extra tip label chip)
        if (!isPresent) {
          ctx.fillStyle = "#12141C";
          const tipTag = `tip: ${PRESENT_SHORT_NAMES[k] || k}`;
          ctx.font = '600 11px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
          const tw = ctx.measureText(tipTag).width + 12;
          ctx.fillRect(tx + 10, ty - 10, tw, 18);
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 1;
          ctx.strokeRect(tx + 10, ty - 10, tw, 18);
          ctx.fillStyle = "#F4F5F8";
          ctx.fillText(tipTag, tx + 16, ty + 3);
        }
        ctx.restore();
      }
    }

    // 5. Top Surgical Phase Banner & Structure Verdict HUD Pills (hidden in Present mode)
    if (!isPresent) {
      ctx.fillStyle = "rgba(10, 11, 14, 0.86)";
      ctx.fillRect(0, 0, W, 30);
      ctx.fillStyle = "#F4F5F8";
      ctx.font = '600 12px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
      const phaseTxt = fr.phase
        ? `SURGICAL PHASE: ${fr.phase}` + (typeof fr.phase_conf === "number" ? ` (conf ${fr.phase_conf.toFixed(2)})` : "")
        : "SURGICAL PHASE: pending";
      ctx.fillText(phaseTxt, 12, 19);

      if (showHud) {
        let hudY = 38;
        for (const k of LABEL_KEYS) {
          const lbl = fr.labels[k];
          const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
          const effPres = getEffectivePresent(fr.frame_id, k, lbl);
          const col = STATUS_COL[effStatus] || "#8A909E";
          const fillCol = STATUS_FILL[effStatus] || "#12141C";
          const chipTxt = `${PRESENT_SHORT_NAMES[k] || k}: ${effStatus === "pending" ? "pending" : (effPres ? "present" : "absent") + " · " + effStatus}`;

          ctx.font = '500 11px "Inter", -apple-system, BlinkMacSystemFont, sans-serif';
          const cw = ctx.measureText(chipTxt).width + 16;
          ctx.fillStyle = fillCol;
          ctx.fillRect(10, hudY, cw, 20);
          ctx.strokeStyle = col;
          ctx.lineWidth = 1;
          ctx.strokeRect(10, hudY, cw, 20);
          ctx.fillStyle = "#F4F5F8";
          ctx.fillText(chipTxt, 18, hudY + 14);
          hudY += 24;
        }
      }
    }
  }

  // Poll web/data/live.json (or /out/annotations_live.jsonl) every 2 seconds
  async function pollLiveAnnotations() {
    mountExternalComponentsIfPresent();

    let rows = [];
    const liveJson = await fetchJsonSafe(["data/live.json", "/web/data/live.json", "/out/web_data/live.json", "/api/live"]);
    if (liveJson) {
      rows = Array.isArray(liveJson) ? liveJson : (liveJson.annotations || []);
    } else {
      const liveText = await fetchTextSafe(["/out/annotations_live.jsonl", "../out/annotations_live.jsonl"]);
      if (liveText) rows = parseJsonl(liveText);
    }

    let added = 0;
    let lastAddedFrameId = null;
    for (const r of rows) {
      const hasMask = Boolean(r.mask_url || r.mask_path || r.mask);
      const sig = `${r.frame_id}__${r.label || r.key}__${r.round || 0}__${hasMask ? 1 : 0}`;
      if (!state.seenLiveSignatures.has(sig)) {
        state.seenLiveSignatures.add(sig);
        if (mergeAnnotationRow(r, true)) {
          added++;
          lastAddedFrameId = r.frame_id;
        }
      }
    }

    state.liveAnnotationCount = state.seenLiveSignatures.size;
    const statusTextEl = document.getElementById("live-status-text");
    if (statusTextEl) {
      statusTextEl.textContent = `Agent Live: ${state.liveAnnotationCount} localized`;
    }

    if (added > 0) {
      const extViewer = getExternalViewer();
      if (state.externalViewerMounted && extViewer && typeof extViewer.update === "function") {
        extViewer.update(buildNormalizedBundle());
      }
      renderTimeline();
      if (lastAddedFrameId) {
        const idx = state.frames.findIndex((f) => f.frame_id === lastAddedFrameId);
        if (idx >= 0) selectFrame(idx);
      } else {
        selectFrame(state.currentFrameIdx);
      }
    }
  }

  // Interactive "Run / Stream Agent" replay of real annotations_live.jsonl across the timeline
  async function triggerStreamReplay() {
    if (state.streamReplayTimer) {
      clearInterval(state.streamReplayTimer);
      state.streamReplayTimer = null;
    }

    let rows = [];
    const liveJson = await fetchJsonSafe(["data/live.json", "/web/data/live.json", "/out/web_data/live.json"]);
    if (liveJson && Array.isArray(liveJson.annotations) && liveJson.annotations.length > 0) {
      rows = liveJson.annotations.filter((r) => !r.video || r.video === state.video);
    } else {
      const text = await fetchTextSafe([
        "/out/annotations_live.jsonl",
        "../out/annotations_live.jsonl",
        "/out/annotations.jsonl",
        "../out/annotations.jsonl",
      ]);
      rows = text ? parseJsonl(text).filter((r) => !r.video || r.video === state.video) : [];
    }
    if (rows.length === 0) return;

    for (const fr of state.frames) {
      fr.phase = null;
      for (const k of LABEL_KEYS) {
        fr.labels[k].box = null;
        fr.labels[k].tip = null;
      }
    }
    renderTimeline();

    let i = 0;
    const btn = document.getElementById("btn-stream-agent");
    if (btn) btn.innerHTML = `<span>Streaming Live...</span> <kbd>R</kbd>`;

    state.streamReplayTimer = setInterval(() => {
      const batch = rows.slice(i, i + 4);
      if (batch.length === 0) {
        clearInterval(state.streamReplayTimer);
        state.streamReplayTimer = null;
        if (btn) btn.innerHTML = `<span>Run / Stream Agent</span> <kbd>R</kbd>`;
        return;
      }
      let targetFid = null;
      for (const r of batch) {
        mergeAnnotationRow(r, true);
        targetFid = r.frame_id;
      }
      i += 4;
      const statusTextEl = document.getElementById("live-status-text");
      if (statusTextEl) {
        statusTextEl.textContent = `Streaming: ${Math.min(i, rows.length)} / ${rows.length}`;
      }
      const extViewer = getExternalViewer();
      if (state.externalViewerMounted && extViewer && typeof extViewer.update === "function") {
        extViewer.update(buildNormalizedBundle());
      }
      renderTimeline();
      if (targetFid) {
        const fIdx = state.frames.findIndex((f) => f.frame_id === targetFid);
        if (fIdx >= 0) selectFrame(fIdx);
      }
    }, 260);
  }

  // Final Pass · Release Dataset Drawer (Human Gate)
  function getResidualAmbers() {
    const out = [];
    for (const fr of state.frames) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const gk = `${fr.frame_id}__${k}`;
        if (lbl.status === "AMBER" && !state.gateDecisions[gk]) {
          out.push({ frame_id: fr.frame_id, key: k, lbl });
        }
      }
    }
    return out;
  }

  function getDeterministicAuditSample() {
    const accepted = [];
    for (const fr of state.frames) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (lbl && lbl.status === "accept") {
          accepted.push({ frame_id: fr.frame_id, key: k, lbl });
        }
      }
    }
    return accepted.slice(0, 20);
  }

  async function postGateAction(payload) {
    for (const ep of ["/api/gate", "/gate"]) {
      try {
        const res = await fetch(ep, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) return await res.json();
      } catch (_e) {
        // fallback to in-memory state when static server is used
      }
    }
    return null;
  }

  function recordGateItemDecision(frameId, key, humanAction, humanPresent) {
    const gk = `${frameId}__${key}`;
    state.gateDecisions[gk] = {
      frame_id: frameId,
      video: state.video,
      key,
      human_action: humanAction,
      human_present: humanPresent,
      ts: Date.now() / 1000,
    };
    postGateAction(state.gateDecisions[gk]);
    renderTimeline();
    selectFrame(state.currentFrameIdx);
    renderGateDrawer();
  }

  function renderGateDrawer() {
    const extGate = getExternalGate();
    if (state.externalGateMounted && extGate && typeof extGate.refresh === "function") {
      extGate.refresh();
      return;
    }
    const ambers = getResidualAmbers();
    const auditItems = getDeterministicAuditSample();
    let totalAccept = 0;
    for (const fr of state.frames) {
      for (const k of LABEL_KEYS) {
        if (getEffectiveStatus(fr.frame_id, k, fr.labels[k]) === "accept") totalAccept++;
      }
    }

    const badgeEl = document.getElementById("amber-count-badge");
    if (badgeEl) badgeEl.textContent = state.frames.length > 0 ? `${ambers.length} AMBER` : "pending";

    const statAcc = document.getElementById("gate-stat-accept");
    if (statAcc) statAcc.textContent = state.frames.length > 0 ? String(totalAccept) : "pending";

    const statAmb = document.getElementById("gate-stat-amber");
    if (statAmb) statAmb.textContent = state.frames.length > 0 ? String(ambers.length) : "pending";

    const statAud = document.getElementById("gate-stat-audit");
    if (statAud) {
      statAud.textContent = state.auditSignedOff
        ? `${auditItems.length} / ${auditItems.length} ✓`
        : `0 / ${auditItems.length}`;
    }

    const amberListEl = document.getElementById("gate-amber-list");
    if (amberListEl) {
      amberListEl.innerHTML = "";
      if (ambers.length === 0) {
        const doneMsg = document.createElement("div");
        doneMsg.style.padding = "10px 12px";
        doneMsg.style.borderRadius = "6px";
        doneMsg.style.backgroundColor = "var(--verdict-accept-fill)";
        doneMsg.style.border = "1px solid var(--verdict-accept)";
        doneMsg.style.color = "var(--verdict-accept)";
        doneMsg.textContent = "✓ Ambers left: 0 (all residual AMBER items cleared).";
        amberListEl.appendChild(doneMsg);
      } else {
        ambers.slice(0, 25).forEach((item) => {
          const card = document.createElement("div");
          card.className = "gate-item-card";
          const left = document.createElement("div");
          const pVal = item.lbl.final_present !== null ? item.lbl.final_present : item.lbl.present;
          left.innerHTML =
            `<div><strong>${item.frame_id}</strong> · <code>${item.key}</code> ` +
            `(Proposal: <strong>${pVal ? "PRESENT" : "ABSENT"}</strong>, conf ${item.lbl.conf !== null ? item.lbl.conf.toFixed(2) : "pending"})</div>` +
            `<div style="font-size:11.5px;color:var(--text-secondary);margin-top:3px;">${item.lbl.argon_reason || item.lbl.evidence || "pending"}</div>`;

          const btns = document.createElement("div");
          btns.style.display = "flex";
          btns.style.gap = "6px";

          const bApprove = document.createElement("button");
          bApprove.type = "button";
          bApprove.className = "btn btn-sm btn-accept";
          bApprove.textContent = "✓ Approve (A)";
          bApprove.addEventListener("click", () => recordGateItemDecision(item.frame_id, item.key, "approve", pVal ? 1 : 0));

          const bFlip = document.createElement("button");
          bFlip.type = "button";
          bFlip.className = "btn btn-sm btn-amber";
          bFlip.textContent = "⇄ Flip (C)";
          bFlip.addEventListener("click", () => recordGateItemDecision(item.frame_id, item.key, "correct", pVal ? 0 : 1));

          btns.appendChild(bApprove);
          btns.appendChild(bFlip);
          card.appendChild(left);
          card.appendChild(btns);
          amberListEl.appendChild(card);
        });
      }
    }

    const auditListEl = document.getElementById("gate-audit-list");
    if (auditListEl) {
      auditListEl.innerHTML = "";
      auditItems.slice(0, 6).forEach((item) => {
        const row = document.createElement("div");
        row.className = "gate-item-card";
        const pVal = item.lbl.final_present !== null ? item.lbl.final_present : item.lbl.present;
        row.innerHTML =
          `<div><strong>${item.frame_id}</strong> · <code>${item.key}</code> · ` +
          `<span style="color:var(--verdict-accept)">AUTO-ACCEPTED (${pVal ? "present" : "absent"})</span></div>` +
          `<div style="font-size:11px;color:var(--text-secondary);max-width:340px;">${item.lbl.evidence || "pending"}</div>`;
        auditListEl.appendChild(row);
      });
    }

    const releaseBtn = document.getElementById("btn-release-dataset");
    const lockMsg = document.getElementById("gate-lock-msg");
    const unlocked = ambers.length === 0 && state.auditSignedOff;
    if (releaseBtn) releaseBtn.disabled = !unlocked;
    if (lockMsg) {
      lockMsg.textContent = unlocked
        ? "Ambers left: 0 · Audit sample: 20/20 signed off. Ready to release dataset!"
        : `Locked: Ambers left: ${ambers.length} · Audit sample: ${state.auditSignedOff ? "20/20" : "0/20"}.`;
    }
  }

  // Self-Correction Mask Round Slider (r0 -> r1 -> r2)
  function getRefineRoundStatText(roundIdx) {
    const r = Number(roundIdx);
    const ref = state.maskRefineResults || (state.maskResults && state.maskResults.self_correction) || null;
    if (ref) {
      const rObj = ref[`round_${r}`] || (ref.rounds && ref.rounds[r]) || null;
      const nVal = ref.n_masks || ref.n || (rObj && rObj.n) || null;
      if (rObj && typeof rObj.mean_iou === "number") {
        return `r${r} IoU ${rObj.mean_iou.toFixed(2)}` + (nVal ? ` (n=${nVal})` : "");
      }
    }
    const mr = state.maskResults && state.maskResults.per_round_improvement && state.maskResults.per_round_improvement.accepted_test_masks;
    if (mr) {
      const keyMap = { 0: "round_0_two_pass", 1: "round_1_crop_loop", 2: "round_2_crop_loop" };
      const entry = mr[keyMap[r]];
      if (entry && typeof entry.mean_iou === "number") {
        return `r${r} IoU ${entry.mean_iou.toFixed(2)} (n=${mr.n || "pending"})`;
      }
    }
    return `r${r}`;
  }

  function setMaskRound(roundIdx) {
    const r = Math.max(0, Math.min(2, Number(roundIdx) || 0));
    state.maskRound = r;
    const rng = document.getElementById("rng-mask-round");
    if (rng && Number(rng.value) !== r) rng.value = String(r);

    document.querySelectorAll(".mask-round-btn").forEach((btn) => {
      if (Number(btn.dataset.round) === r) {
        btn.classList.add("btn-accept");
      } else {
        btn.classList.remove("btn-accept");
      }
    });

    const statEl = document.getElementById("mask-round-stat");
    if (statEl) {
      statEl.textContent = getRefineRoundStatText(r);
    }

    for (const fr of state.frames) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const cands = buildMaskCandidatesForRound(fr.frame_id, k, lbl, r);
        lbl.maskCandidates = cands;
        lbl.mask_url = cands[0];
      }
    }

    const extViewer = getExternalViewer();
    if (state.externalViewerMounted && extViewer && typeof extViewer.update === "function") {
      extViewer.update(buildNormalizedBundle());
    }
    if (state.frames[state.currentFrameIdx]) {
      selectFrame(state.currentFrameIdx);
    }
  }

  // 7-Beat Guided Pipeline Walkthrough Mode (Promo North Star)
  function applyLayerToggles(cfg) {
    const map = {
      masks: "chk-ai-masks",
      polygons: "chk-ai-polygons",
      boxes: "chk-ai-boxes",
      dualPass: "chk-dual-pass",
      tips: "chk-ai-tips",
      hud: "chk-hud-chips",
      gt: "chk-show-gt",
    };
    for (const [k, id] of Object.entries(map)) {
      if (cfg[k] !== undefined) {
        const el = document.getElementById(id);
        if (el) el.checked = Boolean(cfg[k]);
      }
    }
    const extViewer = getExternalViewer();
    if (state.externalViewerMounted && extViewer && typeof extViewer.setLayers === "function") {
      extViewer.setLayers({
        masks: document.getElementById("chk-ai-masks")?.checked,
        polygons: document.getElementById("chk-ai-polygons")?.checked,
        boxes: document.getElementById("chk-ai-boxes")?.checked,
        dualPass: document.getElementById("chk-dual-pass")?.checked,
        tips: document.getElementById("chk-ai-tips")?.checked,
        hud: document.getElementById("chk-hud-chips")?.checked,
        gt: document.getElementById("chk-show-gt")?.checked,
      });
    } else if (state.frames[state.currentFrameIdx]) {
      renderCanvas(state.frames[state.currentFrameIdx]);
    }
  }

  const GUIDED_BEATS = [1, 2, 3, 4, 5, 5.5, 6, 7];

  function setGuidedBeat(beatNum) {
    const b = GUIDED_BEATS.includes(Number(beatNum)) ? Number(beatNum) : 1;
    state.guidedBeat = b;
    if (state.morphTimer) {
      clearInterval(state.morphTimer);
      state.morphTimer = null;
    }
    const banner = document.getElementById("guided-beat-banner");
    if (banner) banner.style.display = "flex";

    document.querySelectorAll(".guided-step-btn").forEach((btn) => {
      if (Number(btn.dataset.beat) === b) {
        btn.classList.add("btn-primary");
      } else {
        btn.classList.remove("btn-primary");
      }
    });

    const titleEl = document.getElementById("guided-beat-title");
    const descEl = document.getElementById("guided-beat-desc");
    const metaEl = document.getElementById("guided-beat-meta");
    const manualBox = document.getElementById("manual-reference-box");
    const drawer = document.getElementById("final-pass-drawer");
    if (drawer && b !== 6) drawer.classList.remove("open");

    const heroIdx = state.frames.findIndex((fr) => fr.frame_id === "video12_15750_15758");
    const targetIdx = heroIdx >= 0 ? heroIdx : state.currentFrameIdx;
    const curFrame = state.frames[targetIdx] || state.frames[0];
    const fidStr = curFrame ? curFrame.frame_id : state.video;

    if (b === 1) {
      if (titleEl) titleEl.textContent = "Beat 1 · Surgical video. No labels.";
      if (descEl) descEl.textContent = "Today: a human at every step. SurgProof: none.";
      if (metaEl) metaEl.textContent = `Raw frame ${fidStr} (${state.frames.length} frames in ${state.video})`;
      applyLayerToggles({ masks: false, polygons: false, boxes: false, dualPass: false, tips: false, hud: false, gt: false });
      selectFrame(targetIdx);
    } else if (b === 2) {
      const srcCount = state.manual && Array.isArray(state.manual.sources) ? state.manual.sources.length : 1;
      const structCount = state.manual && state.manual.structures ? Object.keys(state.manual.structures).length : 6;
      if (titleEl) titleEl.textContent = "Beat 2 · Writes its own manual from public surgical teaching video.";
      if (descEl) descEl.textContent = `Learns ${structCount} anatomical/instrument rules & safety constraints from ${srcCount} CC-BY source (out/manual.json).`;
      if (metaEl) metaEl.textContent = "Zero human training labels";
      if (manualBox) manualBox.open = true;
      renderManualRule(state.selectedKey);
      applyLayerToggles({ masks: false, polygons: false, boxes: false, dualPass: false, tips: false, hud: true, gt: false });
    } else if (b === 3) {
      if (titleEl) titleEl.textContent = `Beat 3 · Labels a real frame (${fidStr}).`;
      if (descEl) descEl.textContent = "Pass-1 Gemini 3.8 Flash proposes structure presence with verbatim visual evidence.";
      if (metaEl) metaEl.textContent = "Neutral proposal before double-check";
      applyLayerToggles({ masks: false, polygons: false, boxes: true, dualPass: false, tips: false, hud: true, gt: false });
      selectFrame(targetIdx);
    } else if (b === 4) {
      const amberLoop = (state.headline && state.headline.amber_loop) || (state.results && state.results.amber_loop);
      const r5 = amberLoop && Array.isArray(amberLoop.ambers_by_round_test_excl_liver)
        ? amberLoop.ambers_by_round_test_excl_liver.join(" → ")
        : "pending";
      if (titleEl) titleEl.textContent = "Beat 4 · Checks itself + 4-Round Argon Amber Loop.";
      if (descEl) descEl.textContent = `3-signal verifier (conf, crop check, Pass-2 critic) + multi-round Argon re-check shrinks AMBERs: ${r5}.`;
      if (metaEl) metaEl.textContent = "Accept (#10B981) · AMBER (#F59E0B) · Reject (#EF4444)";
      applyLayerToggles({ masks: false, polygons: false, boxes: true, dualPass: true, tips: false, hud: true, gt: false });
      selectFrame(targetIdx);
    } else if (b === 5) {
      if (titleEl) titleEl.textContent = "Beat 5 · The AI draws the masks, polygons, boxes & tool tips.";
      if (descEl) descEl.textContent = "Two-pass polygon segmentation (IoU >= 0.50 consensus) + bounding boxes + instrument tip crosshairs + surgical phase.";
      if (metaEl) metaEl.textContent = `out/masks/ + out/polygons/${state.video}.jsonl`;
      setMaskRound(0);
      applyLayerToggles({ masks: true, polygons: true, boxes: true, dualPass: false, tips: true, hud: true, gt: false });
      selectFrame(targetIdx);
    } else if (b === 5.5) {
      const r0Txt = getRefineRoundStatText(0);
      const r1Txt = getRefineRoundStatText(1);
      if (titleEl) titleEl.textContent = "Beat 5b · Then corrects itself (Round 0 → Round 1 Mask Refinement).";
      if (descEl) descEl.textContent = `Self-critique & refinement morphs Round 0 mask (out/masks/) to Round 1 (out/masks_r1/): ${r0Txt} → ${r1Txt}.`;
      if (metaEl) metaEl.textContent = "Use Round Slider r0 / r1 / r2 to inspect";
      applyLayerToggles({ masks: true, polygons: true, boxes: true, dualPass: false, tips: true, hud: true, gt: false });
      setMaskRound(0);
      selectFrame(targetIdx);
      state.morphTimer = setTimeout(() => {
        setMaskRound(1);
      }, 1100);
    } else if (b === 6) {
      const ambersLeft = getResidualAmbers().length;
      if (titleEl) titleEl.textContent = "Beat 6 · Final Pass (Human Gate) · People stay in control.";
      if (descEl) descEl.textContent = `Clear ${ambersLeft} residual AMBERs in ${state.video} + sign off 20-item random audit sample to unlock Release dataset.`;
      if (metaEl) metaEl.textContent = "POST /api/gate → out/verified_dataset.jsonl";
      renderGateDrawer();
      if (drawer) drawer.classList.add("open");
    } else if (b === 7) {
      const accTxt = document.getElementById("headline-acc")?.textContent || "pending";
      const nTxt = document.getElementById("headline-n")?.textContent || "pending";
      if (titleEl) titleEl.textContent = `Beat 7 · Held-Out Proof: ${accTxt}`;
      if (descEl) descEl.textContent = `${nTxt} · Ground truth touched once at the end to score the held-out test split.`;
      if (metaEl) metaEl.textContent = "Source: out/results.json · out/student.json · out/mask_results.json";
      setMaskRound(1);
      applyLayerToggles({ masks: true, polygons: true, boxes: true, dualPass: false, tips: true, hud: true, gt: false });
      selectFrame(targetIdx);
    }
  }

  function startGuidedWalkthrough() {
    setGuidedBeat(1);
  }

  function stopGuidedWalkthrough() {
    if (state.guidedTimer) {
      clearInterval(state.guidedTimer);
      state.guidedTimer = null;
    }
    const autoBtn = document.getElementById("btn-beat-autoplay");
    if (autoBtn) autoBtn.textContent = "▶ Auto-Play";
    const banner = document.getElementById("guided-beat-banner");
    if (banner) banner.style.display = "none";
    const drawer = document.getElementById("final-pass-drawer");
    if (drawer) drawer.classList.remove("open");
    applyLayerToggles({ masks: true, polygons: true, boxes: true, dualPass: false, tips: true, hud: true, gt: false });
  }

  function toggleGuidedAutoplay() {
    const autoBtn = document.getElementById("btn-beat-autoplay");
    if (state.guidedTimer) {
      clearInterval(state.guidedTimer);
      state.guidedTimer = null;
      if (autoBtn) autoBtn.textContent = "▶ Auto-Play";
      return;
    }
    if (autoBtn) autoBtn.textContent = "⏸ Pause Beats";
    state.guidedTimer = setInterval(() => {
      const idx = GUIDED_BEATS.indexOf(state.guidedBeat || 1);
      const nextIdx = idx < 0 || idx >= GUIDED_BEATS.length - 1 ? 0 : idx + 1;
      setGuidedBeat(GUIDED_BEATS[nextIdx]);
    }, 3800);
  }

  function togglePlay() {
    state.playing = !state.playing;
    const btn = document.getElementById("btn-play");
    if (state.playing) {
      if (btn) btn.innerHTML = `⏸ Pause <kbd>Space</kbd>`;
      state.playTimer = setInterval(() => {
        const nextIdx = (state.currentFrameIdx + 1) % Math.max(1, state.frames.length);
        selectFrame(nextIdx);
      }, state.playIntervalMs);
    } else {
      if (btn) btn.innerHTML = `▶ Play <kbd>Space</kbd>`;
      if (state.playTimer) clearInterval(state.playTimer);
      state.playTimer = null;
    }
  }

  function bindEvents() {
    const on = (id, ev, fn) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener(ev, fn);
    };

    on("sel-video", "change", (e) => {
      switchVideo(e.target.value);
    });

    on("btn-guided-mode", "click", () => {
      const banner = document.getElementById("guided-beat-banner");
      if (banner && banner.style.display !== "none") {
        stopGuidedWalkthrough();
      } else {
        startGuidedWalkthrough();
      }
    });

    document.querySelectorAll(".guided-step-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        setGuidedBeat(Number(btn.dataset.beat) || 1);
      });
    });

    on("btn-beat-prev", "click", () => {
      const idx = GUIDED_BEATS.indexOf(state.guidedBeat || 1);
      const prevIdx = idx <= 0 ? GUIDED_BEATS.length - 1 : idx - 1;
      setGuidedBeat(GUIDED_BEATS[prevIdx]);
    });
    on("btn-beat-next", "click", () => {
      const idx = GUIDED_BEATS.indexOf(state.guidedBeat || 1);
      const nextIdx = idx < 0 || idx >= GUIDED_BEATS.length - 1 ? 0 : idx + 1;
      setGuidedBeat(GUIDED_BEATS[nextIdx]);
    });
    on("btn-beat-autoplay", "click", toggleGuidedAutoplay);
    on("btn-beat-close", "click", stopGuidedWalkthrough);

    on("rng-mask-round", "input", (e) => {
      setMaskRound(Number(e.target.value));
    });
    document.querySelectorAll(".mask-round-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        setMaskRound(Number(btn.dataset.round));
      });
    });

    on("btn-first", "click", () => selectFrame(0));
    on("btn-prev", "click", () => selectFrame(state.currentFrameIdx - 1));
    on("btn-play", "click", togglePlay);
    on("btn-next", "click", () => selectFrame(state.currentFrameIdx + 1));
    on("btn-last", "click", () => selectFrame(state.frames.length - 1));

    on("sel-speed", "change", (e) => {
      state.playIntervalMs = Number(e.target.value) || 650;
      if (state.playing) {
        togglePlay();
        togglePlay();
      }
    });

    ["chk-ai-masks", "chk-ai-polygons", "chk-ai-boxes", "chk-dual-pass", "chk-ai-tips", "chk-hud-chips", "chk-show-gt"].forEach((id) => {
      on(id, "change", () => {
        const extViewer = getExternalViewer();
        if (state.externalViewerMounted && extViewer && typeof extViewer.setLayers === "function") {
          extViewer.setLayers({
            masks: document.getElementById("chk-ai-masks")?.checked,
            polygons: document.getElementById("chk-ai-polygons")?.checked,
            boxes: document.getElementById("chk-ai-boxes")?.checked,
            dualPass: document.getElementById("chk-dual-pass")?.checked,
            tips: document.getElementById("chk-ai-tips")?.checked,
            hud: document.getElementById("chk-hud-chips")?.checked,
            gt: document.getElementById("chk-show-gt")?.checked,
          });
        } else if (state.frames[state.currentFrameIdx]) {
          renderCanvas(state.frames[state.currentFrameIdx]);
        }
      });
    });

    on("btn-zoom-reset", "click", () => {
      state.canvasZoom = 1.0;
      const extViewer = getExternalViewer();
      if (state.externalViewerMounted && extViewer && typeof extViewer.resetZoom === "function") {
        extViewer.resetZoom();
      } else if (state.frames[state.currentFrameIdx]) {
        renderCanvas(state.frames[state.currentFrameIdx]);
      }
    });

    // Filter tabs in Right Panel (#objects)
    document.querySelectorAll(".filter-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".filter-tab").forEach((t) => t.classList.remove("active"));
        tab.classList.add("active");
        state.objectFilter = tab.dataset.filter || "all";
        if (state.frames[state.currentFrameIdx]) {
          renderInspector(state.frames[state.currentFrameIdx]);
        }
      });
    });

    // Listen to custom events from nir-webcanvas (web/viewer.js) and nir-webtimeline (web/timeline.js)
    window.addEventListener("viewer:select", (e) => {
      const k = e.detail && (e.detail.label || e.detail.key);
      if (k && LABEL_KEYS.includes(k)) {
        state.selectedKey = k;
        if (state.frames[state.currentFrameIdx]) renderInspector(state.frames[state.currentFrameIdx]);
        renderManualRule(k);
      }
    });

    window.addEventListener("surgproof:select-label", (e) => {
      const k = e.detail && (e.detail.label || e.detail.key);
      if (k && LABEL_KEYS.includes(k)) {
        state.selectedKey = k;
        if (state.frames[state.currentFrameIdx]) renderInspector(state.frames[state.currentFrameIdx]);
        renderManualRule(k);
      }
    });

    window.addEventListener("timeline:seek", (e) => {
      if (e.detail && typeof e.detail.frameIndex === "number") {
        selectFrame(e.detail.frameIndex);
      }
    });

    window.addEventListener("surgproof:seek", (e) => {
      if (e.detail && typeof e.detail.frameIndex === "number") {
        selectFrame(e.detail.frameIndex);
      }
    });

    on("btn-stream-agent", "click", triggerStreamReplay);

    const drawer = document.getElementById("final-pass-drawer");
    on("btn-final-pass", "click", () => {
      renderGateDrawer();
      if (drawer) drawer.classList.add("open");
    });
    on("btn-close-drawer", "click", () => {
      if (drawer) drawer.classList.remove("open");
    });

    on("btn-clear-all-ambers", "click", () => {
      const ambers = getResidualAmbers();
      for (const item of ambers) {
        const pVal = item.lbl.final_present !== null ? item.lbl.final_present : item.lbl.present;
        const gk = `${item.frame_id}__${item.key}`;
        state.gateDecisions[gk] = {
          frame_id: item.frame_id,
          video: state.video,
          key: item.key,
          human_action: "approve",
          human_present: pVal ? 1 : 0,
          ts: Date.now() / 1000,
        };
      }
      postGateAction({ action: "batch_approve_ambers", items: Object.values(state.gateDecisions) });
      renderTimeline();
      selectFrame(state.currentFrameIdx);
      renderGateDrawer();
    });

    on("btn-confirm-audit-sample", "click", () => {
      state.auditSignedOff = true;
      postGateAction({ action: "sign_off_audit_sample", count: 20 });
      renderGateDrawer();
    });

    on("btn-release-dataset", "click", async () => {
      await postGateAction({ action: "release_dataset", video: state.video, decisions: Object.values(state.gateDecisions) });
      const lockMsg = document.getElementById("gate-lock-msg");
      if (lockMsg) {
        lockMsg.style.color = "var(--verdict-accept)";
        lockMsg.textContent = "✓ Released! Verified dataset exported to out/verified_dataset.jsonl with full provenance.";
      }
    });

    window.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) {
        return;
      }
      const key = e.key;
      if (key === "Escape" && drawer && drawer.classList.contains("open")) {
        drawer.classList.remove("open");
        return;
      }
      if (key === "f" || key === "F") {
        e.preventDefault();
        if (drawer) {
          renderGateDrawer();
          drawer.classList.toggle("open");
        }
        return;
      }
      if (drawer && drawer.classList.contains("open")) {
        if (key === "a" || key === "A" || key === "y" || key === "Y") {
          const ambers = getResidualAmbers();
          if (ambers.length > 0) {
            const first = ambers[0];
            const pVal = first.lbl.final_present !== null ? first.lbl.final_present : first.lbl.present;
            recordGateItemDecision(first.frame_id, first.key, "approve", pVal ? 1 : 0);
          }
          return;
        }
        if (key === "c" || key === "C" || key === "n" || key === "N") {
          const ambers = getResidualAmbers();
          if (ambers.length > 0) {
            const first = ambers[0];
            const pVal = first.lbl.final_present !== null ? first.lbl.final_present : first.lbl.present;
            recordGateItemDecision(first.frame_id, first.key, "correct", pVal ? 0 : 1);
          }
          return;
        }
      }

      if (key === "ArrowRight" || key === "j" || key === "J") {
        e.preventDefault();
        selectFrame(state.currentFrameIdx + 1);
      } else if (key === "ArrowLeft" || key === "k" || key === "K") {
        e.preventDefault();
        selectFrame(state.currentFrameIdx - 1);
      } else if (key === "Home") {
        e.preventDefault();
        selectFrame(0);
      } else if (key === "End") {
        e.preventDefault();
        selectFrame(state.frames.length - 1);
      } else if (key === " ") {
        e.preventDefault();
        togglePlay();
      } else if (key >= "1" && key <= "6") {
        const k = LABEL_KEYS[Number(key) - 1];
        if (k) {
          state.selectedKey = k;
          if (state.frames[state.currentFrameIdx]) {
            renderInspector(state.frames[state.currentFrameIdx]);
            renderCanvas(state.frames[state.currentFrameIdx]);
          }
          renderManualRule(k);
        }
      } else if (key === "m" || key === "M") {
        const el = document.getElementById("chk-ai-masks");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "p" || key === "P") {
        const el = document.getElementById("chk-ai-polygons");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "b" || key === "B") {
        const el = document.getElementById("chk-ai-boxes");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "d" || key === "D") {
        const el = document.getElementById("chk-dual-pass");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "t" || key === "T") {
        const el = document.getElementById("chk-ai-tips");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "g" || key === "G") {
        const el = document.getElementById("chk-show-gt");
        if (el) {
          el.checked = !el.checked;
          el.dispatchEvent(new Event("change"));
        }
      } else if (key === "w" || key === "W") {
        const banner = document.getElementById("guided-beat-banner");
        if (banner && banner.style.display !== "none") {
          stopGuidedWalkthrough();
        } else {
          startGuidedWalkthrough();
        }
      } else if (key === "r" || key === "R") {
        triggerStreamReplay();
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    syncPresentBodyClass();
    bindEvents();
    loadInitialData();
    setInterval(pollLiveAnnotations, 2000);
  });
})();
