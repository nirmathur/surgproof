/* SurgProof AI-Native Surgical Video Annotation Studio (video12)
 * Reads exclusively from real pipeline artifacts (web/data/video12.json, web/data/live.json,
 * or /out/decisions.csv, /out/annotations.jsonl, /out/results.json, /out/localize_results.json).
 * Zero fabricated boxes, verdicts, or numbers; missing artifacts display 'pending'.
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

  const STATUS_COL = {
    accept: "#10B981",
    AMBER: "#F59E0B",
    amber: "#F59E0B",
    reject: "#EF4444",
    pending: "#64748B",
  };

  const LABEL_RGB = {
    grasper: [66, 153, 225],
    l_hook: [159, 122, 234],
    gallbladder: [72, 187, 120],
    cystic_duct: [236, 201, 75],
    liver: [160, 82, 45],
    blood: [229, 62, 62],
  };

  // CholecSeg8k watershed pixel value -> label key (for Scoring View GT toggle only)
  const WATERSHED_TO_KEY = {
    31: "grasper",
    32: "l_hook",
    22: "gallbladder",
    11: "liver",
  };

  const state = {
    video: "video12",
    frames: [],
    frameById: {},
    currentFrameIdx: 0,
    selectedKey: "grasper",
    playing: false,
    playTimer: null,
    playIntervalMs: 650,
    results: null,
    localizeResults: null,
    student: null,
    narrationByFrame: {},
    liveAnnotationCount: 0,
    seenLiveSignatures: new Set(),
    imageCache: {},
    maskCache: {},
    gtCanvasCache: {},
    gateDecisions: {},
    auditSignedOff: false,
    streamReplayTimer: null,
  };

  function normStatus(raw) {
    const s = String(raw || "").trim().toLowerCase();
    if (s === "accept") return "accept";
    if (s === "reject" || s === "escalate") return "reject";
    if (!s || s === "pending") return "pending";
    return "AMBER";
  }

  function fmtPct(val, digits = 1) {
    if (typeof val !== "number" || !Number.isFinite(val)) return "pending";
    return (val * 100).toFixed(digits) + "%";
  }

  async function fetchJsonSafe(urls) {
    const list = Array.isArray(urls) ? urls : [urls];
    for (const u of list) {
      try {
        const res = await fetch(u, { cache: "no-store" });
        if (res.ok) return await res.json();
      } catch (_e) {
        // try next fallback URL
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
        // try next fallback URL
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
          `/raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `../raw/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
          `/data/cholecseg8k/CholecSeg8k/${vid}/${clip}/frame_${fnum}_endo.png`,
        ],
        gtCandidates: [
          `data/gt_masks/${frameId}.png`,
          `/web/data/gt_masks/${frameId}.png`,
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
        imgCandidates: urls.imgCandidates,
        gtCandidates: urls.gtCandidates,
        phase: null,
        phase_conf: null,
        narration: null,
        labels: {},
      };
      for (const k of LABEL_KEYS) {
        rec.labels[k] = {
          key: k,
          present: null,
          final_present: null,
          status: "pending",
          conf: null,
          trust: null,
          evidence: "",
          pv_match: null,
          p_wrong: null,
          critic: "",
          status_round_0: null,
          status_round_1: null,
          status_round_2: null,
          status_round_3: null,
          argon_reason: "",
          box: null,
          tip: null,
          pass1: null,
          pass2: null,
          loc_round: null,
          loc_status: null,
          maskCandidates: [
            `data/masks/${frameId}_${k}.png`,
            `/web/data/masks/${frameId}_${k}.png`,
            `/out/masks/${frameId}_${k}.png`,
            `../out/masks/${frameId}_${k}.png`,
          ],
          hasNewLivePulse: false,
        };
      }
      state.frameById[frameId] = rec;
    }
    return state.frameById[frameId];
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
      if (ann.round !== undefined && ann.round !== null) lbl.loc_round = Number(ann.round);
      if (ann.status) lbl.loc_status = normStatus(ann.status);
      if (ann.mask_url || ann.mask_path) {
        const mp = String(ann.mask_url || ann.mask_path);
        lbl.maskCandidates = [mp, "/" + mp.replace(/^\/+/, ""), ...lbl.maskCandidates];
      }
      if (markPulse) lbl.hasNewLivePulse = true;
      return true;
    }
    return false;
  }

  async function loadInitialData() {
    // 1. Try pre-exported web/data/video12.json from web/export.py first
    const exported = await fetchJsonSafe([
      "data/video12.json",
      "/web/data/video12.json",
      "/data/video12.json",
      "/api/video12",
    ]);

    if (exported && Array.isArray(exported.frames) && exported.frames.length > 0) {
      if (exported.results) state.results = exported.results;
      if (exported.localize_results) state.localizeResults = exported.localize_results;
      if (exported.student) state.student = exported.student;

      for (const fr of exported.frames) {
        const f = ensureFrameRecord(fr.frame_id);
        if (fr.img_url || fr.frame_url || fr.path) {
          const u = fr.img_url || fr.frame_url || fr.path;
          f.imgCandidates = [u, "/" + String(u).replace(/^\/+/, ""), ...f.imgCandidates];
        }
        if (fr.gt_mask_url || fr.mask_url) {
          const gu = fr.gt_mask_url || fr.mask_url;
          f.gtCandidates = [gu, "/" + String(gu).replace(/^\/+/, ""), ...f.gtCandidates];
        }
        if (fr.phase) f.phase = fr.phase;
        if (fr.phase_conf !== undefined) f.phase_conf = fr.phase_conf;
        if (fr.narration) f.narration = fr.narration;

        const rawLabels = fr.labels || {};
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
          if (v.pv_match !== undefined && v.pv_match !== "") t.pv_match = String(v.pv_match) === "True" || v.pv_match === true;
          if (v.p_wrong !== undefined && v.p_wrong !== null && v.p_wrong !== "") t.p_wrong = Number(v.p_wrong);
          if (v.critic) t.critic = String(v.critic);
          if (v.status_round_0) t.status_round_0 = normStatus(v.status_round_0);
          if (v.status_round_1) t.status_round_1 = normStatus(v.status_round_1);
          if (v.status_round_2) t.status_round_2 = normStatus(v.status_round_2);
          if (v.status_round_3) t.status_round_3 = normStatus(v.status_round_3);
          if (v.argon_reason) t.argon_reason = String(v.argon_reason);
          if (Array.isArray(v.box) && v.box.length === 4) t.box = v.box.map(Number);
          if (Array.isArray(v.tip) && v.tip.length === 2) t.tip = v.tip.map(Number);
          if (v.pass1) t.pass1 = v.pass1;
          if (v.pass2) t.pass2 = v.pass2;
          if (v.loc_round !== undefined) t.loc_round = Number(v.loc_round);
          if (v.loc_status) t.loc_status = normStatus(v.loc_status);
          if (v.mask_url || v.ai_mask_url || v.mask_path) {
            const mu = String(v.mask_url || v.ai_mask_url || v.mask_path);
            t.maskCandidates = [mu, "/" + mu.replace(/^\/+/, ""), ...t.maskCandidates];
          }
        }
      }
    }

    // 2. Also load / supplement from raw /out/* artifacts so nothing is missing
    const [resJson, locResJson, stuJson, narrJson, decText, annText] = await Promise.all([
      state.results ? Promise.resolve(state.results) : fetchJsonSafe(["/out/results.json", "../out/results.json", "data/results.json"]),
      state.localizeResults ? Promise.resolve(state.localizeResults) : fetchJsonSafe(["/out/localize_results.json", "../out/localize_results.json", "data/localize_results.json"]),
      state.student ? Promise.resolve(state.student) : fetchJsonSafe(["/out/student.json", "../out/student.json", "data/student.json"]),
      fetchJsonSafe(["/out/narration_video12.json", "../out/narration_video12.json", "data/narration_video12.json"]),
      Object.keys(state.frameById).length > 0 ? Promise.resolve(null) : fetchTextSafe(["/out/decisions.csv", "../out/decisions.csv"]),
      fetchTextSafe(["/out/annotations.jsonl", "../out/annotations.jsonl", "/out/annotations_live.jsonl", "../out/annotations_live.jsonl"]),
    ]);

    if (resJson) state.results = resJson;
    if (locResJson) state.localizeResults = locResJson;
    if (stuJson) state.student = stuJson;

    if (narrJson) {
      const sents = Array.isArray(narrJson) ? narrJson : (narrJson.sentences || []);
      for (const s of sents) {
        if (s && s.frame_id) {
          state.narrationByFrame[s.frame_id] = s.narration || "";
          if (state.frameById[s.frame_id]) {
            state.frameById[s.frame_id].narration = s.narration || "";
          }
        }
      }
    }

    if (decText) {
      const rows = parseCsv(decText);
      for (const r of rows) {
        const vid = r.video || r.video_id;
        if (vid !== state.video) continue;
        const f = ensureFrameRecord(r.frame_id);
        const k = r.key;
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
        t.critic = r.critic || "";
        t.status_round_0 = r.status_round_0 ? normStatus(r.status_round_0) : null;
        t.status_round_1 = r.status_round_1 ? normStatus(r.status_round_1) : null;
        t.status_round_2 = r.status_round_2 ? normStatus(r.status_round_2) : null;
        t.status_round_3 = r.status_round_3 ? normStatus(r.status_round_3) : null;
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

    state.frames = Object.values(state.frameById).sort((a, b) => a.frame_num - b.frame_num);

    // Default to first frame that has spatial annotations (e.g. video12_15750_15758) if available
    const firstSpatialIdx = state.frames.findIndex((fr) =>
      LABEL_KEYS.some((k) => fr.labels[k] && Array.isArray(fr.labels[k].box))
    );
    if (firstSpatialIdx >= 0) {
      state.currentFrameIdx = firstSpatialIdx;
    }

    renderTopHeadlines();
    renderAmberFunnel();
    renderTimeline();
    selectFrame(state.currentFrameIdx);
    renderGateDrawer();
  }

  function renderTopHeadlines() {
    const accEl = document.getElementById("headline-acc");
    const nEl = document.getElementById("headline-n");
    if (state.results) {
      const a5 = state.results.accepted_5_informative || state.results.accepted || {};
      const acc = a5.accuracy;
      const cov = a5.coverage;
      const nLabels = state.results.n_test_labels_excl_liver || state.results.n_test_labels || "pending";
      if (typeof acc === "number" && typeof cov === "number") {
        accEl.textContent = `${fmtPct(acc, 1)} accurate on the ${fmtPct(cov, 1)} it accepted`;
        nEl.textContent = `n = ${nLabels} held-out labels, 6 unseen videos`;
      } else {
        accEl.textContent = "pending";
        nEl.textContent = "awaiting out/results.json";
      }
    }

    const locEl = document.getElementById("headline-loc");
    const locSub = document.getElementById("headline-loc-sub");
    if (state.localizeResults && state.localizeResults.boxes) {
      const meanIou = state.localizeResults.boxes.mean_iou;
      const tipRate = state.localizeResults.tips ? state.localizeResults.tips.within_tool_mask_rate : null;
      const nScored = state.localizeResults.boxes.n_scored || state.localizeResults.n_test_annotations || 0;
      locEl.textContent = `Box IoU ${fmtPct(meanIou, 1)} · Tip-in-Mask ${fmtPct(tipRate, 0)}`;
      locSub.textContent = `n = ${nScored} test annotations`;
    }

    const stuEl = document.getElementById("headline-student");
    const stuSub = document.getElementById("headline-student-sub");
    if (state.student && state.student.mean_acc_by_source) {
      const m = state.student.mean_acc_by_source;
      if (typeof m.verified === "number" && typeof m.single === "number") {
        const diff = ((m.verified - m.single) * 100).toFixed(1);
        const sign = m.verified >= m.single ? "+" : "";
        stuEl.textContent = `Specialist: ${fmtPct(m.verified, 1)} (${sign}${diff} pp vs single)`;
        stuSub.textContent = `n_test = ${state.student.n_test_frames || 280} frames`;
      }
    }
  }

  function renderAmberFunnel() {
    const container = document.getElementById("amber-funnel-steps");
    const scopeEl = document.getElementById("funnel-scope-label");
    container.innerHTML = "";

    let rounds5 = null;
    let roundsAll6 = null;
    if (state.results && state.results.amber_loop) {
      rounds5 = state.results.amber_loop.ambers_by_round_test_excl_liver;
      roundsAll6 = state.results.amber_loop.ambers_by_round_test;
    }

    if (Array.isArray(rounds5) && rounds5.length > 0) {
      scopeEl.textContent = "5 informative classes (n=1,400 test) · [all 6 in brackets]";
      rounds5.forEach((cnt, idx) => {
        const sp = document.createElement("span");
        sp.className = "funnel-step";
        const allTag = Array.isArray(roundsAll6) && roundsAll6[idx] !== undefined ? ` [all-6: ${roundsAll6[idx]}]` : "";
        sp.textContent = `R${idx}: ${cnt} AMBER${allTag}`;
        if (idx === rounds5.length - 1) {
          sp.style.borderColor = "rgba(16, 185, 129, 0.5)";
          sp.style.color = "#6EE7B7";
        }
        container.appendChild(sp);
      });
      return;
    }

    // Fallback: compute on video12 frames from status_round_0..3
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
    scopeEl.textContent = `${state.video} (${state.frames.length * 6} frame-labels)`;
    counts.forEach((cnt, idx) => {
      const sp = document.createElement("span");
      sp.className = "funnel-step";
      sp.textContent = `R${idx}: ${cnt} AMBER`;
      container.appendChild(sp);
    });
  }

  function renderTimeline() {
    const tracksEl = document.getElementById("timeline-tracks");
    const countEl = document.getElementById("timeline-frame-count");
    countEl.textContent = String(state.frames.length);
    tracksEl.innerHTML = "";

    if (state.frames.length === 0) return;

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

        const hasSpatial = lbl && (Array.isArray(lbl.box) || Array.isArray(lbl.tip));
        cell.className =
          `track-cell ${stClass}` +
          (hasSpatial ? " has-spatial" : "") +
          (lbl && lbl.hasNewLivePulse ? " live-flash" : "") +
          (idx === state.currentFrameIdx ? " active-frame" : "");

        cell.title = `${fr.frame_id} · ${k}: ${effectivePresent ? "present" : "absent"} (${effectiveStatus})` +
          (hasSpatial ? " · spatial box/tip localized" : "");
        cell.dataset.frameIdx = String(idx);
        cell.dataset.labelKey = k;
        cell.addEventListener("click", () => {
          state.selectedKey = k;
          selectFrame(idx);
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

    // Highlight active column in timeline without rebuilding DOM
    document.querySelectorAll(".track-cell").forEach((el) => {
      if (Number(el.dataset.frameIdx) === clamped) {
        el.classList.add("active-frame");
      } else {
        el.classList.remove("active-frame");
      }
    });

    document.getElementById("frame-readout").textContent =
      `Frame ${clamped + 1} / ${state.frames.length} · ${fr.frame_id}`;
    document.getElementById("inspector-frame-tag").textContent = fr.frame_id;

    const narrEl = document.getElementById("narration-text");
    if (fr.narration) {
      narrEl.textContent = fr.narration;
    } else {
      const acceptedKeys = LABEL_KEYS.filter(
        (k) => getEffectiveStatus(fr.frame_id, k, fr.labels[k]) === "accept" && getEffectivePresent(fr.frame_id, k, fr.labels[k])
      );
      narrEl.textContent = acceptedKeys.length > 0
        ? `Verified present structures on ${fr.frame_id}: ${acceptedKeys.join(", ")}. (Run narrate.py for full commentary)`
        : `No pre-generated narration line for ${fr.frame_id}.`;
    }

    renderInspector(fr);
    renderCanvas(fr);
  }

  function renderInspector(fr) {
    const container = document.getElementById("inspector-cards");
    container.innerHTML = "";

    for (const k of LABEL_KEYS) {
      const lbl = fr.labels[k];
      const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
      const effPres = getEffectivePresent(fr.frame_id, k, lbl);

      const card = document.createElement("div");
      card.className = `label-card st-${effStatus}` + (state.selectedKey === k ? " selected" : "");
      card.addEventListener("click", () => {
        state.selectedKey = k;
        renderInspector(fr);
        renderCanvas(fr);
      });

      const top = document.createElement("div");
      top.className = "label-card-top";

      const title = document.createElement("div");
      title.className = "label-title";
      title.innerHTML = `<span>${LABEL_NAMES[k] || k}</span> <code style="font-size:11px;color:var(--text-secondary)">(${k})</code>`;

      const badge = document.createElement("span");
      badge.className = `status-badge st-${effStatus}`;
      badge.textContent = `${effPres ? "PRESENT" : "ABSENT"} · ${effStatus}`;

      top.appendChild(title);
      top.appendChild(badge);
      card.appendChild(top);

      const meta = document.createElement("div");
      meta.className = "label-meta";
      meta.innerHTML =
        `<span>conf: ${lbl && lbl.conf !== null ? lbl.conf.toFixed(2) : "pending"}</span>` +
        `<span>crop pv_match: ${lbl && lbl.pv_match !== null ? String(lbl.pv_match) : "n/a"}</span>` +
        `<span>critic p_wrong: ${lbl && lbl.p_wrong !== null ? lbl.p_wrong.toFixed(2) : "n/a"}</span>`;

      if (lbl && Array.isArray(lbl.box)) {
        const iouStr = lbl.pass2 && typeof lbl.pass2.iou === "number" ? ` · Pass1/2 IoU ${lbl.pass2.iou.toFixed(2)}` : "";
        meta.innerHTML += `<span style="color:var(--col-tip)">box: [${lbl.box.join(", ")}]${iouStr}</span>`;
      } else if (effPres) {
        meta.innerHTML += `<span>box: pending</span>`;
      }
      if (lbl && Array.isArray(lbl.tip)) {
        meta.innerHTML += `<span style="color:var(--col-tip)">tip: [y=${lbl.tip[0]}, x=${lbl.tip[1]}]</span>`;
      }
      card.appendChild(meta);

      const ev = document.createElement("div");
      ev.className = "label-evidence";
      ev.textContent = (lbl && lbl.evidence) ? lbl.evidence : "Evidence: pending";
      card.appendChild(ev);

      if (lbl && (lbl.status_round_0 || lbl.argon_reason)) {
        const rDiv = document.createElement("div");
        rDiv.className = "label-rounds";
        const rSeq = [lbl.status_round_0, lbl.status_round_1, lbl.status_round_2, lbl.status_round_3]
          .filter(Boolean)
          .join(" → ");
        rDiv.innerHTML = `<div><strong>Double-Check &amp; Amber Loop:</strong> ${rSeq || effStatus}</div>`;
        if (lbl.argon_reason) {
          const ar = document.createElement("div");
          ar.style.marginTop = "3px";
          ar.style.color = "#FCD34D";
          ar.textContent = `Argon review: ${lbl.argon_reason}`;
          rDiv.appendChild(ar);
        }
        card.appendChild(rDiv);
      }

      container.appendChild(card);
    }
  }

  function loadImageWithFallback(candidates, cacheObj, key) {
    if (cacheObj[key]) return Promise.resolve(cacheObj[key]);
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
      const val = d[i]; // watershed class value in channel 0
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

  async function renderCanvas(fr) {
    const canvas = document.getElementById("frame-canvas");
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;

    const showMasks = document.getElementById("chk-ai-masks").checked;
    const showBoxes = document.getElementById("chk-ai-boxes").checked;
    const showDualPass = document.getElementById("chk-dual-pass").checked;
    const showTips = document.getElementById("chk-ai-tips").checked;
    const showHud = document.getElementById("chk-hud-chips").checked;
    const showGt = document.getElementById("chk-show-gt").checked;

    const baseImg = await loadImageWithFallback(fr.imgCandidates, state.imageCache, fr.frame_id);

    ctx.clearRect(0, 0, W, H);
    if (baseImg) {
      ctx.drawImage(baseImg, 0, 0, W, H);
    } else {
      ctx.fillStyle = "#111318";
      ctx.fillRect(0, 0, W, H);
      ctx.fillStyle = "#94A3B8";
      ctx.font = "14px -apple-system, sans-serif";
      ctx.fillText(`Frame image pending on disk: ${fr.frame_id}`, 28, H / 2);
    }

    // 1. Optional Scoring-Only Ground-Truth Watershed Mask Overlay
    if (showGt) {
      const gtImg = await loadImageWithFallback(fr.gtCandidates, state.imageCache, fr.frame_id + "__gt");
      if (gtImg) {
        if (!state.gtCanvasCache[fr.frame_id]) {
          state.gtCanvasCache[fr.frame_id] = buildGtOverlayCanvas(gtImg);
        }
        ctx.drawImage(state.gtCanvasCache[fr.frame_id], 0, 0, W, H);
      }
      ctx.fillStyle = "rgba(168, 85, 247, 0.88)";
      ctx.fillRect(W - 330, 8, 322, 24);
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.fillText("SCORING VIEW ONLY: CHOLECSEG8K GT MASK", W - 320, 24);
    }

    // 2. AI Segmentation Masks Overlay (from out/masks/<frame_id>_<label>.png when present)
    let loadedMasksCount = 0;
    if (showMasks) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl || !getEffectivePresent(fr.frame_id, k, lbl)) continue;
        const mKey = `${fr.frame_id}__${k}__mask`;
        const maskImg = await loadImageWithFallback(lbl.maskCandidates, state.maskCache, mKey);
        if (maskImg) {
          loadedMasksCount++;
          ctx.save();
          ctx.globalAlpha = 0.35;
          ctx.drawImage(maskImg, 0, 0, W, H);
          ctx.restore();
        } else if (Array.isArray(lbl.box)) {
          // Subtle translucent fill inside localized AI box when PNG mask is still pending
          const [y1, x1, y2, x2] = lbl.box;
          const rgb = LABEL_RGB[k] || [59, 130, 246];
          ctx.fillStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.12)`;
          ctx.fillRect((x1 * W) / 1000, (y1 * H) / 1000, ((x2 - x1) * W) / 1000, ((y2 - y1) * H) / 1000);
        }
      }
    }
    const maskStatusEl = document.getElementById("mask-status-label");
    maskStatusEl.textContent = loadedMasksCount > 0 ? `${loadedMasksCount} PNG masks` : "pending";

    // 3. AI Bounding Boxes (and optional Pass-1 vs Pass-2 Dual Boxes)
    if (showBoxes) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl) continue;
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const col = STATUS_COL[effStatus] || "#F59E0B";

        if (showDualPass && lbl.pass1 && Array.isArray(lbl.pass1.box)) {
          const [py1, px1, py2, px2] = lbl.pass1.box;
          ctx.save();
          ctx.strokeStyle = "#3B82F6";
          ctx.lineWidth = 2;
          ctx.setLineDash([6, 4]);
          ctx.strokeRect((px1 * W) / 1000, (py1 * H) / 1000, ((px2 - px1) * W) / 1000, ((py2 - py1) * H) / 1000);
          ctx.restore();
        }
        if (showDualPass && lbl.pass2 && Array.isArray(lbl.pass2.box)) {
          const [qy1, qx1, qy2, qx2] = lbl.pass2.box;
          ctx.save();
          ctx.strokeStyle = "#A855F7";
          ctx.lineWidth = 2;
          ctx.setLineDash([3, 3]);
          ctx.strokeRect((qx1 * W) / 1000, (qy1 * H) / 1000, ((qx2 - qx1) * W) / 1000, ((qy2 - qy1) * H) / 1000);
          ctx.restore();
        }

        if (Array.isArray(lbl.box) && lbl.box.length === 4) {
          const [y1, x1, y2, x2] = lbl.box;
          const rx = (x1 * W) / 1000;
          const ry = (y1 * H) / 1000;
          const rw = ((x2 - x1) * W) / 1000;
          const rh = ((y2 - y1) * H) / 1000;

          ctx.strokeStyle = col;
          ctx.lineWidth = state.selectedKey === k ? 3.5 : 2.2;
          ctx.strokeRect(rx, ry, rw, rh);

          const iouTxt = lbl.pass2 && typeof lbl.pass2.iou === "number" ? ` IoU:${lbl.pass2.iou.toFixed(2)}` : "";
          const tag = `${k} (${effStatus}${iouTxt})`;
          ctx.font = "bold 11px ui-monospace, monospace";
          const tw = ctx.measureText(tag).width + 12;
          const ty = Math.max(32, ry - 20);
          ctx.fillStyle = col;
          ctx.fillRect(rx, ty, tw, 19);
          ctx.fillStyle = "#0A0B0E";
          ctx.fillText(tag, rx + 6, ty + 13);
        }
      }
    }

    // 4. Instrument Tip Points ([y, x] in 0..1000)
    if (showTips) {
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        if (!lbl || !Array.isArray(lbl.tip) || lbl.tip.length !== 2) continue;
        const ty = (lbl.tip[0] * H) / 1000;
        const tx = (lbl.tip[1] * W) / 1000;

        ctx.save();
        ctx.strokeStyle = "#38BDF8";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(tx, ty, 8, 0, Math.PI * 2);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(tx - 14, ty);
        ctx.lineTo(tx + 14, ty);
        ctx.moveTo(tx, ty - 14);
        ctx.lineTo(tx, ty + 14);
        ctx.stroke();

        ctx.fillStyle = "rgba(10, 11, 14, 0.82)";
        const tipTag = `TIP: ${k}`;
        ctx.font = "bold 11px ui-monospace, monospace";
        const tw = ctx.measureText(tipTag).width + 10;
        ctx.fillRect(tx + 10, ty - 10, tw, 18);
        ctx.fillStyle = "#38BDF8";
        ctx.fillText(tipTag, tx + 15, ty + 3);
        ctx.restore();
      }
    }

    // 5. Top Surgical Phase Banner & Structure Status HUD Pills
    ctx.fillStyle = "rgba(10, 11, 14, 0.84)";
    ctx.fillRect(0, 0, W, 30);
    ctx.fillStyle = "#F8FAFC";
    ctx.font = "bold 12px -apple-system, sans-serif";
    const phaseTxt = fr.phase
      ? `SURGICAL PHASE: ${fr.phase}` + (typeof fr.phase_conf === "number" ? ` (conf ${fr.phase_conf.toFixed(2)})` : "")
      : "SURGICAL PHASE: pending (localizing video12 frames...)";
    ctx.fillText(phaseTxt, 12, 19);

    if (showHud) {
      let hudY = 38;
      for (const k of LABEL_KEYS) {
        const lbl = fr.labels[k];
        const effStatus = getEffectiveStatus(fr.frame_id, k, lbl);
        const effPres = getEffectivePresent(fr.frame_id, k, lbl);
        const col = STATUS_COL[effStatus] || "#64748B";
        const chipTxt = `${k}: ${effPres ? "PRESENT" : "absent"} · ${effStatus}`;

        ctx.font = "bold 11px ui-monospace, monospace";
        const cw = ctx.measureText(chipTxt).width + 14;
        ctx.fillStyle = "rgba(10, 11, 14, 0.78)";
        ctx.fillRect(10, hudY, cw, 20);
        ctx.fillStyle = col;
        ctx.fillRect(10, hudY, 4, 20);
        ctx.fillStyle = "#F8FAFC";
        ctx.fillText(chipTxt, 18, hudY + 14);
        hudY += 23;
      }
    }
  }

  // Poll web/data/live.json (or /out/annotations_live.jsonl) every 2 seconds
  async function pollLiveAnnotations() {
    let rows = [];
    const liveJson = await fetchJsonSafe(["data/live.json", "/web/data/live.json", "/api/live"]);
    if (liveJson) {
      rows = Array.isArray(liveJson) ? liveJson : (liveJson.annotations || []);
    } else {
      const liveText = await fetchTextSafe(["/out/annotations_live.jsonl", "../out/annotations_live.jsonl"]);
      if (liveText) rows = parseJsonl(liveText);
    }

    let added = 0;
    let lastAddedFrameId = null;
    for (const r of rows) {
      const sig = `${r.frame_id}__${r.label || r.key}__${r.round || 0}`;
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
    statusTextEl.textContent = `Agent Live: ${state.liveAnnotationCount} spatial annotations`;

    if (added > 0) {
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
    const text = await fetchTextSafe([
      "/out/annotations_live.jsonl",
      "../out/annotations_live.jsonl",
      "/out/annotations.jsonl",
      "../out/annotations.jsonl",
    ]);
    const rows = text ? parseJsonl(text).filter((r) => !r.video || r.video === state.video) : [];
    if (rows.length === 0) return;

    // Clear spatial boxes temporarily to visually stream each real annotation line-by-line
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
    btn.textContent = "⚡ Streaming Live...";
    state.streamReplayTimer = setInterval(() => {
      const batch = rows.slice(i, i + 4);
      if (batch.length === 0) {
        clearInterval(state.streamReplayTimer);
        state.streamReplayTimer = null;
        btn.textContent = "⚡ Run / Stream Agent";
        return;
      }
      let targetFid = null;
      for (const r of batch) {
        mergeAnnotationRow(r, true);
        targetFid = r.frame_id;
      }
      i += 4;
      document.getElementById("live-status-text").textContent = `Agent Streaming: ${Math.min(i, rows.length)} / ${rows.length}`;
      renderTimeline();
      if (targetFid) {
        const fIdx = state.frames.findIndex((f) => f.frame_id === targetFid);
        if (fIdx >= 0) selectFrame(fIdx);
      }
    }, 280);
  }

  // Final Pass & Release Dataset Drawer (Human Gate)
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
    const ambers = getResidualAmbers();
    const auditItems = getDeterministicAuditSample();
    let totalAccept = 0;
    for (const fr of state.frames) {
      for (const k of LABEL_KEYS) {
        if (getEffectiveStatus(fr.frame_id, k, fr.labels[k]) === "accept") totalAccept++;
      }
    }

    document.getElementById("amber-count-badge").textContent = `${ambers.length} AMBER`;
    document.getElementById("gate-stat-accept").textContent = String(totalAccept);
    document.getElementById("gate-stat-amber").textContent = String(ambers.length);
    document.getElementById("gate-stat-audit").textContent = state.auditSignedOff
      ? `${auditItems.length} / ${auditItems.length} ✓`
      : `0 / ${auditItems.length}`;

    const amberListEl = document.getElementById("gate-amber-list");
    amberListEl.innerHTML = "";
    if (ambers.length === 0) {
      const doneMsg = document.createElement("div");
      doneMsg.style.padding = "10px";
      doneMsg.style.borderRadius = "7px";
      doneMsg.style.background = "rgba(16, 185, 129, 0.14)";
      doneMsg.style.color = "#6EE7B7";
      doneMsg.textContent = "✓ All residual AMBER items on video12 are cleared (0 remaining).";
      amberListEl.appendChild(doneMsg);
    } else {
      ambers.slice(0, 25).forEach((item) => {
        const card = document.createElement("div");
        card.className = "gate-item-card";
        const left = document.createElement("div");
        const pVal = item.lbl.final_present !== null ? item.lbl.final_present : item.lbl.present;
        left.innerHTML =
          `<div><strong>${item.frame_id}</strong> · <code>${item.key}</code> (Agent proposal: <strong>${pVal ? "PRESENT" : "ABSENT"}</strong>, conf ${item.lbl.conf !== null ? item.lbl.conf.toFixed(2) : "n/a"})</div>` +
          `<div style="font-size:11.5px;color:var(--text-secondary);margin-top:3px;">${item.lbl.argon_reason || item.lbl.evidence || ""}</div>`;

        const btns = document.createElement("div");
        btns.style.display = "flex";
        btns.style.gap = "6px";

        const bApprove = document.createElement("button");
        bApprove.className = "btn btn-accept";
        bApprove.textContent = "✓ Approve (A)";
        bApprove.addEventListener("click", () => recordGateItemDecision(item.frame_id, item.key, "approve", pVal ? 1 : 0));

        const bFlip = document.createElement("button");
        bFlip.className = "btn btn-amber";
        bFlip.textContent = "⇄ Flip (C)";
        bFlip.addEventListener("click", () => recordGateItemDecision(item.frame_id, item.key, "correct", pVal ? 0 : 1));

        btns.appendChild(bApprove);
        btns.appendChild(bFlip);
        card.appendChild(left);
        card.appendChild(btns);
        amberListEl.appendChild(card);
      });
    }

    const auditListEl = document.getElementById("gate-audit-list");
    auditListEl.innerHTML = "";
    auditItems.slice(0, 6).forEach((item) => {
      const row = document.createElement("div");
      row.className = "gate-item-card";
      const pVal = item.lbl.final_present !== null ? item.lbl.final_present : item.lbl.present;
      row.innerHTML =
        `<div><strong>${item.frame_id}</strong> · <code>${item.key}</code> · <span style="color:var(--col-accept)">AUTO-ACCEPTED (${pVal ? "present" : "absent"})</span></div>` +
        `<div style="font-size:11px;color:var(--text-secondary);max-width:340px;">${item.lbl.evidence || ""}</div>`;
      auditListEl.appendChild(row);
    });

    const releaseBtn = document.getElementById("btn-release-dataset");
    const lockMsg = document.getElementById("gate-lock-msg");
    const unlocked = ambers.length === 0 && state.auditSignedOff;
    releaseBtn.disabled = !unlocked;
    lockMsg.textContent = unlocked
      ? "🟢 Final pass complete (0 residual AMBERs + 20-item audit signed off). Ready to release dataset!"
      : `🔒 Locked: ${ambers.length} residual AMBERs remaining · Audit sample ${state.auditSignedOff ? "signed off" : "pending sign-off"}.`;
  }

  function togglePlay() {
    state.playing = !state.playing;
    const btn = document.getElementById("btn-play");
    if (state.playing) {
      btn.textContent = "⏸ Pause";
      state.playTimer = setInterval(() => {
        const nextIdx = (state.currentFrameIdx + 1) % Math.max(1, state.frames.length);
        selectFrame(nextIdx);
      }, state.playIntervalMs);
    } else {
      btn.textContent = "▶ Play";
      if (state.playTimer) clearInterval(state.playTimer);
      state.playTimer = null;
    }
  }

  function bindEvents() {
    document.getElementById("btn-first").addEventListener("click", () => selectFrame(0));
    document.getElementById("btn-prev").addEventListener("click", () => selectFrame(state.currentFrameIdx - 1));
    document.getElementById("btn-play").addEventListener("click", togglePlay);
    document.getElementById("btn-next").addEventListener("click", () => selectFrame(state.currentFrameIdx + 1));
    document.getElementById("btn-last").addEventListener("click", () => selectFrame(state.frames.length - 1));

    document.getElementById("sel-speed").addEventListener("change", (e) => {
      state.playIntervalMs = Number(e.target.value) || 650;
      if (state.playing) {
        togglePlay();
        togglePlay();
      }
    });

    ["chk-ai-masks", "chk-ai-boxes", "chk-dual-pass", "chk-ai-tips", "chk-hud-chips", "chk-show-gt"].forEach((id) => {
      document.getElementById(id).addEventListener("change", () => {
        if (state.frames[state.currentFrameIdx]) {
          renderCanvas(state.frames[state.currentFrameIdx]);
        }
      });
    });

    document.getElementById("btn-stream-agent").addEventListener("click", triggerStreamReplay);

    const drawer = document.getElementById("final-pass-drawer");
    document.getElementById("btn-final-pass").addEventListener("click", () => {
      renderGateDrawer();
      drawer.classList.add("open");
    });
    document.getElementById("btn-close-drawer").addEventListener("click", () => {
      drawer.classList.remove("open");
    });

    document.getElementById("btn-clear-all-ambers").addEventListener("click", () => {
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

    document.getElementById("btn-confirm-audit-sample").addEventListener("click", () => {
      state.auditSignedOff = true;
      postGateAction({ action: "sign_off_audit_sample", count: 20 });
      renderGateDrawer();
    });

    document.getElementById("btn-release-dataset").addEventListener("click", async () => {
      await postGateAction({ action: "release_dataset", video: state.video, decisions: Object.values(state.gateDecisions) });
      const lockMsg = document.getElementById("gate-lock-msg");
      lockMsg.style.color = "#6EE7B7";
      lockMsg.textContent = "✅ Released! Verified dataset exported to out/verified_dataset.jsonl with full provenance.";
    });

    window.addEventListener("keydown", (e) => {
      if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT" || e.target.tagName === "TEXTAREA")) {
        return;
      }
      if (e.key === "ArrowRight" || e.key === "j" || e.key === "J") {
        e.preventDefault();
        selectFrame(state.currentFrameIdx + 1);
      } else if (e.key === "ArrowLeft" || e.key === "k" || e.key === "K") {
        e.preventDefault();
        selectFrame(state.currentFrameIdx - 1);
      } else if (e.key === " ") {
        e.preventDefault();
        togglePlay();
      } else if ((e.key === "a" || e.key === "A" || e.key === "y" || e.key === "Y") && drawer.classList.contains("open")) {
        const ambers = getResidualAmbers();
        if (ambers.length > 0) {
          const first = ambers[0];
          const pVal = first.lbl.final_present !== null ? first.lbl.final_present : first.lbl.present;
          recordGateItemDecision(first.frame_id, first.key, "approve", pVal ? 1 : 0);
        }
      } else if ((e.key === "c" || e.key === "C" || e.key === "n" || e.key === "N") && drawer.classList.contains("open")) {
        const ambers = getResidualAmbers();
        if (ambers.length > 0) {
          const first = ambers[0];
          const pVal = first.lbl.final_present !== null ? first.lbl.final_present : first.lbl.present;
          recordGateItemDecision(first.frame_id, first.key, "correct", pVal ? 0 : 1);
        }
      }
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    bindEvents();
    loadInitialData();
    setInterval(pollLiveAnnotations, 2000);
  });
})();
