/**
 * SurgProof Multi-Track Video Annotation Timeline (web/timeline.js)
 * Lane: nir-webtimeline
 *
 * Professional reference: Supervisely Video Toolbox, CVAT, Encord, V7 Darwin, Labelbox.
 *
 * Public API:
 *   Timeline.mount(el, data, onSeek)
 *   Timeline.update(data)
 *   Timeline.setFrame(i)
 *
 * Strictly renders real pipeline data from web/data/video12.json and web/data/live.json.
 * Anything not yet produced is explicitly shown as "pending".
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    var api = factory();
    root.Timeline = api;
    root.SurgProofTimeline = api;
  }
})(typeof window !== "undefined" ? window : globalThis, function () {
  "use strict";

  var DEFAULT_LABELS = [
    "grasper",
    "l_hook",
    "gallbladder",
    "cystic_duct",
    "liver",
    "blood"
  ];

  var LABEL_META = {
    grasper: { name: "Grasper", kind: "tool", short: "Grasper" },
    l_hook: { name: "L-Hook Cautery", kind: "tool", short: "L-Hook" },
    gallbladder: { name: "Gallbladder", kind: "anatomy", short: "Gallbladder" },
    cystic_duct: { name: "Cystic Duct", kind: "anatomy", short: "Cystic Duct" },
    liver: { name: "Liver", kind: "anatomy", short: "Liver" },
    blood: { name: "Blood", kind: "finding", short: "Blood" }
  };

  var PHASE_PALETTE = [
    { bg: "#132038", border: "#2563EB", text: "#93C5FD" },
    { bg: "#1A1836", border: "#6366F1", text: "#A5B4FC" },
    { bg: "#11282B", border: "#0D9488", text: "#5EEAD4" },
    { bg: "#27192C", border: "#9333EA", text: "#D8B4FE" },
    { bg: "#2A1F14", border: "#D97706", text: "#FCD34D" },
    { bg: "#17251E", border: "#059669", text: "#6EE7B7" },
    { bg: "#281822", border: "#DB2777", text: "#F9A8D4" }
  ];

  var DEFAULT_FRAME_COUNT = 80;
  var LIVE_POLL_INTERVAL_MS = 2000;
  var FRESH_HIGHLIGHT_MS = 1600;

  var state = {
    container: null,
    onSeek: null,
    video: "video12",
    nFrames: DEFAULT_FRAME_COUNT,
    labels: DEFAULT_LABELS.slice(),
    frames: [],
    frameIndexById: {},
    currentFrame: 0,
    selectedLabel: null,
    isPlaying: false,
    fps: 6,
    playTimer: null,
    liveUrl: "data/live.json",
    liveTimer: null,
    livePolling: true,
    liveLastCount: 0,
    liveLastTs: null,
    liveFetchInFlight: false,
    zoomLevel: 1,
    isScrubbing: false,
    cellSignatures: {},
    freshCells: {},
    freshCleanupTimer: null,
    dom: {},
    boundKeyHandler: null,
    boundMoveHandler: null,
    boundUpHandler: null
  };

  function injectStyles() {
    if (typeof document === "undefined") return;
    if (document.getElementById("surgproof-timeline-styles")) return;
    var style = document.createElement("style");
    style.id = "surgproof-timeline-styles";
    style.textContent = [
      ".sp-timeline {",
      "  --tl-bg: #0A0B0E;",
      "  --tl-surface: #12141C;",
      "  --tl-surface-2: #171A24;",
      "  --tl-border: #232734;",
      "  --tl-border-subtle: #1A1D28;",
      "  --tl-text: #F4F5F8;",
      "  --tl-muted: #8A909E;",
      "  --tl-accent: #3B82F6;",
      "  --tl-accept-fg: #10B981;",
      "  --tl-accept-bg: #092E24;",
      "  --tl-amber-fg: #F59E0B;",
      "  --tl-amber-bg: #362407;",
      "  --tl-reject-fg: #EF4444;",
      "  --tl-reject-bg: #331014;",
      "  --tl-absent-fg: #4B5262;",
      "  --tl-absent-bg: #141720;",
      "  --tl-pending-fg: #646B7A;",
      "  --tl-pending-bg: #0E1016;",
      "  background: var(--tl-bg);",
      "  color: var(--tl-text);",
      "  border-top: 1px solid var(--tl-border);",
      "  font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'SF Pro Display', 'Inter', sans-serif;",
      "  font-size: 12px;",
      "  line-height: 1.35;",
      "  user-select: none;",
      "  display: flex;",
      "  flex-direction: column;",
      "  width: 100%;",
      "  box-sizing: border-box;",
      "  position: relative;",
      "}",
      ".sp-timeline *, .sp-timeline *::before, .sp-timeline *::after {",
      "  box-sizing: border-box;",
      "}",
      ".timeline-pane:has(.sp-timeline) > .timeline-header {",
      "  display: none;",
      "}",
      ".tl-toolbar {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 8px;",
      "  padding: 3px 10px;",
      "  background: var(--tl-surface);",
      "  border-bottom: 1px solid var(--tl-border);",
      "  min-height: 28px;",
      "  flex-wrap: nowrap;",
      "  overflow-x: auto;",
      "}",
      ".tl-transport, .tl-legend, .tl-controls {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 6px;",
      "}",
      ".tl-btn {",
      "  background: var(--tl-surface-2);",
      "  color: var(--tl-text);",
      "  border: 1px solid var(--tl-border);",
      "  border-radius: 5px;",
      "  padding: 3px 8px;",
      "  font-size: 11px;",
      "  font-weight: 500;",
      "  cursor: pointer;",
      "  display: inline-flex;",
      "  align-items: center;",
      "  gap: 4px;",
      "  transition: background 120ms ease, border-color 120ms ease;",
      "}",
      ".tl-btn:hover {",
      "  background: #1E2330;",
      "  border-color: #343A4D;",
      "}",
      ".tl-btn--active {",
      "  background: rgba(59, 130, 246, 0.16);",
      "  border-color: var(--tl-accent);",
      "  color: #93C5FD;",
      "}",
      ".tl-btn--amber {",
      "  background: var(--tl-amber-bg);",
      "  border-color: rgba(245, 158, 11, 0.45);",
      "  color: var(--tl-amber-fg);",
      "}",
      ".tl-btn--amber:hover {",
      "  background: #473009;",
      "  border-color: var(--tl-amber-fg);",
      "}",
      ".tl-readout {",
      "  font-variant-numeric: tabular-nums;",
      "  font-weight: 600;",
      "  color: var(--tl-text);",
      "  padding: 2px 8px;",
      "  background: #0D0F15;",
      "  border: 1px solid var(--tl-border);",
      "  border-radius: 4px;",
      "  display: inline-flex;",
      "  align-items: center;",
      "  gap: 8px;",
      "}",
      ".tl-readout-id {",
      "  color: var(--tl-muted);",
      "  font-family: 'SF Mono', 'JetBrains Mono', Menlo, monospace;",
      "  font-size: 11px;",
      "  font-weight: 400;",
      "}",
      ".tl-phase-pill {",
      "  padding: 2px 8px;",
      "  border-radius: 4px;",
      "  font-size: 11px;",
      "  font-weight: 500;",
      "  background: #132038;",
      "  border: 1px solid #2563EB;",
      "  color: #93C5FD;",
      "  max-width: 260px;",
      "  white-space: nowrap;",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "}",
      ".tl-phase-pill--pending {",
      "  background: var(--tl-pending-bg);",
      "  border-color: var(--tl-border);",
      "  color: var(--tl-muted);",
      "}",
      ".tl-stat-chip {",
      "  display: inline-flex;",
      "  align-items: center;",
      "  gap: 5px;",
      "  padding: 2px 7px;",
      "  border-radius: 4px;",
      "  font-size: 11px;",
      "  font-variant-numeric: tabular-nums;",
      "  background: #0E1017;",
      "  border: 1px solid var(--tl-border-subtle);",
      "  color: var(--tl-muted);",
      "}",
      ".tl-dot {",
      "  width: 7px;",
      "  height: 7px;",
      "  border-radius: 2px;",
      "  display: inline-block;",
      "}",
      ".tl-dot--accept { background: var(--tl-accept-fg); }",
      ".tl-dot--amber { background: var(--tl-amber-fg); }",
      ".tl-dot--reject { background: var(--tl-reject-fg); }",
      ".tl-dot--absent { background: var(--tl-absent-fg); }",
      ".tl-dot--pending { background: var(--tl-pending-fg); }",
      ".tl-live-pill {",
      "  display: inline-flex;",
      "  align-items: center;",
      "  gap: 5px;",
      "  padding: 2px 8px;",
      "  border-radius: 999px;",
      "  font-size: 11px;",
      "  border: 1px solid var(--tl-border);",
      "  background: #0E1118;",
      "  color: var(--tl-muted);",
      "  cursor: pointer;",
      "}",
      ".tl-live-pill--on {",
      "  border-color: rgba(16, 185, 129, 0.4);",
      "  color: #A7F3D0;",
      "}",
      ".tl-live-dot {",
      "  width: 6px;",
      "  height: 6px;",
      "  border-radius: 50%;",
      "  background: var(--tl-muted);",
      "}",
      ".tl-live-pill--on .tl-live-dot {",
      "  background: var(--tl-accept-fg);",
      "}",
      ".tl-workspace {",
      "  display: flex;",
      "  width: 100%;",
      "  overflow: hidden;",
      "  background: var(--tl-bg);",
      "}",
      ".tl-sidebar {",
      "  width: 176px;",
      "  min-width: 176px;",
      "  background: var(--tl-surface);",
      "  border-right: 1px solid var(--tl-border);",
      "  display: flex;",
      "  flex-direction: column;",
      "  z-index: 3;",
      "}",
      ".tl-sidebar-row {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  padding: 0 10px;",
      "  border-bottom: 1px solid var(--tl-border-subtle);",
      "  font-size: 11px;",
      "  color: var(--tl-text);",
      "}",
      ".tl-sidebar-row--ruler {",
      "  height: 20px;",
      "  font-size: 10px;",
      "  font-weight: 600;",
      "  letter-spacing: 0.05em;",
      "  text-transform: uppercase;",
      "  color: var(--tl-muted);",
      "  background: #0E1016;",
      "}",
      ".tl-sidebar-row--phase {",
      "  height: 22px;",
      "  font-weight: 600;",
      "  color: #93C5FD;",
      "  background: #10131C;",
      "}",
      ".tl-sidebar-row--label {",
      "  height: 20px;",
      "  cursor: pointer;",
      "  transition: background 120ms ease;",
      "}",
      ".tl-sidebar-row--label:hover {",
      "  background: #191D28;",
      "}",
      ".tl-sidebar-row--selected {",
      "  background: rgba(59, 130, 246, 0.14);",
      "  border-left: 2px solid var(--tl-accent);",
      "  padding-left: 8px;",
      "}",
      ".tl-track-title {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 6px;",
      "  overflow: hidden;",
      "  white-space: nowrap;",
      "  text-overflow: ellipsis;",
      "}",
      ".tl-track-kind {",
      "  font-size: 9px;",
      "  text-transform: uppercase;",
      "  padding: 1px 4px;",
      "  border-radius: 3px;",
      "  background: #1A1E2B;",
      "  color: var(--tl-muted);",
      "}",
      ".tl-track-counts {",
      "  font-variant-numeric: tabular-nums;",
      "  font-size: 10px;",
      "  color: var(--tl-muted);",
      "  display: flex;",
      "  gap: 4px;",
      "}",
      ".tl-stage-scroll {",
      "  flex: 1;",
      "  overflow-x: auto;",
      "  overflow-y: hidden;",
      "  position: relative;",
      "  background: var(--tl-bg);",
      "}",
      ".tl-stage {",
      "  position: relative;",
      "  min-width: 100%;",
      "  display: flex;",
      "  flex-direction: column;",
      "}",
      ".tl-ruler {",
      "  height: 20px;",
      "  display: grid;",
      "  background: #0E1016;",
      "  border-bottom: 1px solid var(--tl-border-subtle);",
      "  cursor: ew-resize;",
      "  position: relative;",
      "}",
      ".tl-tick {",
      "  position: relative;",
      "  height: 100%;",
      "  border-left: 1px solid rgba(255, 255, 255, 0.05);",
      "  display: flex;",
      "  align-items: flex-start;",
      "  padding-top: 2px;",
      "  padding-left: 3px;",
      "  font-size: 9px;",
      "  font-variant-numeric: tabular-nums;",
      "  color: var(--tl-muted);",
      "  pointer-events: none;",
      "}",
      ".tl-tick--major {",
      "  border-left-color: rgba(255, 255, 255, 0.18);",
      "  color: #C4C9D4;",
      "  font-weight: 500;",
      "}",
      ".tl-tick::after {",
      "  content: '';",
      "  position: absolute;",
      "  bottom: 0;",
      "  left: 0;",
      "  width: 1px;",
      "  height: 4px;",
      "  background: rgba(255, 255, 255, 0.14);",
      "}",
      ".tl-tick--major::after {",
      "  height: 6px;",
      "  background: rgba(255, 255, 255, 0.32);",
      "}",
      ".tl-phase-track {",
      "  height: 22px;",
      "  display: grid;",
      "  position: relative;",
      "  background: #0C0E14;",
      "  border-bottom: 1px solid var(--tl-border-subtle);",
      "  padding: 2px 0;",
      "}",
      ".tl-phase-band {",
      "  height: 17px;",
      "  margin: 0 1px;",
      "  border-radius: 3px;",
      "  border: 1px solid #2563EB;",
      "  background: #132038;",
      "  color: #93C5FD;",
      "  font-size: 10px;",
      "  font-weight: 500;",
      "  display: flex;",
      "  align-items: center;",
      "  padding: 0 6px;",
      "  overflow: hidden;",
      "  white-space: nowrap;",
      "  text-overflow: ellipsis;",
      "  cursor: pointer;",
      "  transition: filter 120ms ease;",
      "}",
      ".tl-phase-band:hover {",
      "  filter: brightness(1.15);",
      "}",
      ".tl-phase-band--pending {",
      "  background: repeating-linear-gradient(-45deg, #0E1016, #0E1016 5px, #141720 5px, #141720 10px);",
      "  border-color: var(--tl-border);",
      "  color: var(--tl-pending-fg);",
      "  font-style: italic;",
      "}",
      ".tl-track {",
      "  height: 20px;",
      "  display: grid;",
      "  border-bottom: 1px solid var(--tl-border-subtle);",
      "  background: var(--tl-bg);",
      "  position: relative;",
      "}",
      ".tl-cell {",
      "  position: relative;",
      "  height: 100%;",
      "  border-right: 1px solid rgba(10, 11, 14, 0.65);",
      "  cursor: pointer;",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  transition: transform 220ms cubic-bezier(0.22, 1, 0.36, 1), filter 160ms ease, background-color 240ms ease;",
      "}",
      ".tl-cell:hover {",
      "  filter: brightness(1.22);",
      "  z-index: 2;",
      "}",
      ".tl-cell--accept {",
      "  background: var(--tl-accept-bg);",
      "  box-shadow: inset 0 2px 0 var(--tl-accept-fg), inset 0 -1px 0 rgba(16, 185, 129, 0.35);",
      "}",
      ".tl-cell--amber {",
      "  background: var(--tl-amber-bg);",
      "  box-shadow: inset 0 2px 0 var(--tl-amber-fg), inset 0 -2px 0 var(--tl-amber-fg);",
      "}",
      ".tl-cell--reject {",
      "  background: var(--tl-reject-bg);",
      "  box-shadow: inset 0 2px 0 var(--tl-reject-fg);",
      "}",
      ".tl-cell--absent {",
      "  background: var(--tl-absent-bg);",
      "}",
      ".tl-cell--pending {",
      "  background: repeating-linear-gradient(-45deg, #0C0E14, #0C0E14 3px, #12151E 3px, #12151E 6px);",
      "}",
      ".tl-cell--current, .tl-cell.active-frame {",
      "  outline: 1px solid rgba(244, 245, 248, 0.55);",
      "  z-index: 1;",
      "}",
      ".tl-cell--fresh {",
      "  animation: tlCellArrive 1.5s cubic-bezier(0.22, 1, 0.36, 1);",
      "}",
      "@keyframes tlCellArrive {",
      "  0% { filter: brightness(2.1); transform: scaleY(0.35); }",
      "  45% { filter: brightness(1.45); transform: scaleY(1.0); }",
      "  100% { filter: brightness(1.0); transform: scaleY(1.0); }",
      "}",
      ".tl-spatial-dot {",
      "  width: 4px;",
      "  height: 4px;",
      "  border-radius: 50%;",
      "  background: rgba(244, 245, 248, 0.85);",
      "  pointer-events: none;",
      "}",
      ".tl-cell--amber .tl-spatial-dot {",
      "  background: var(--tl-amber-fg);",
      "}",
      ".tl-playhead {",
      "  position: absolute;",
      "  top: 0;",
      "  bottom: 0;",
      "  width: 2px;",
      "  background: var(--tl-accent);",
      "  pointer-events: none;",
      "  z-index: 5;",
      "  transform: translateX(-1px);",
      "}",
      ".tl-playhead-cap {",
      "  position: absolute;",
      "  top: 0;",
      "  left: 50%;",
      "  transform: translate(-50%, 0);",
      "  width: 10px;",
      "  height: 12px;",
      "  background: var(--tl-accent);",
      "  clip-path: polygon(0 0, 100% 0, 100% 55%, 50% 100%, 0 55%);",
      "}",
      ".tl-tooltip {",
      "  position: fixed;",
      "  z-index: 9999;",
      "  background: var(--tl-surface);",
      "  border: 1px solid var(--tl-border);",
      "  border-radius: 6px;",
      "  padding: 8px 10px;",
      "  color: var(--tl-text);",
      "  font-size: 11px;",
      "  max-width: 320px;",
      "  pointer-events: none;",
      "  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);",
      "  display: none;",
      "}",
      ".tl-tooltip-head {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 8px;",
      "  margin-bottom: 4px;",
      "  font-weight: 600;",
      "}",
      ".tl-tooltip-badge {",
      "  padding: 1px 6px;",
      "  border-radius: 3px;",
      "  font-size: 10px;",
      "  font-weight: 600;",
      "  text-transform: uppercase;",
      "}",
      ".tl-tooltip-badge--accept { background: var(--tl-accept-bg); color: var(--tl-accept-fg); border: 1px solid var(--tl-accept-fg); }",
      ".tl-tooltip-badge--amber { background: var(--tl-amber-bg); color: var(--tl-amber-fg); border: 1px solid var(--tl-amber-fg); }",
      ".tl-tooltip-badge--reject { background: var(--tl-reject-bg); color: var(--tl-reject-fg); border: 1px solid var(--tl-reject-fg); }",
      ".tl-tooltip-badge--absent { background: var(--tl-absent-bg); color: var(--tl-muted); border: 1px solid var(--tl-border); }",
      ".tl-tooltip-badge--pending { background: var(--tl-pending-bg); color: var(--tl-muted); border: 1px solid var(--tl-border); }",
      ".tl-tooltip-meta {",
      "  color: var(--tl-muted);",
      "  font-size: 10px;",
      "  margin-bottom: 4px;",
      "}",
      ".tl-tooltip-ev {",
      "  color: #D8DCE6;",
      "  font-size: 11px;",
      "  border-top: 1px solid var(--tl-border-subtle);",
      "  padding-top: 4px;",
      "  margin-top: 4px;",
      "}"
    ].join("\n");
    document.head.appendChild(style);
  }

  function normalizeVerdict(cell) {
    if (!cell || typeof cell !== "object") return "pending";
    var rawStatus = cell.status || cell.verdict || "";
    var st = String(rawStatus).trim().toLowerCase();
    var pres = cell.final_present !== undefined && cell.final_present !== null
      ? cell.final_present
      : cell.present;
    var hasPres = pres !== undefined && pres !== null && pres !== "";
    var isPresent = hasPres ? Number(pres) === 1 || pres === true : null;

    if (!st || st === "pending" || st === "null" || st === "none") {
      return "pending";
    }
    if (st === "absent") {
      return "absent";
    }
    if (st === "amber" || st === "reanalyse" || st === "unsure") {
      return "amber";
    }
    if (st === "reject" || st === "rejected" || st === "escalate") {
      return "reject";
    }
    if (st === "accept" || st === "accepted" || st === "verified") {
      if (isPresent === false) return "absent";
      return "accept";
    }
    return "pending";
  }

  function createEmptyFrame(idx, video) {
    var v = video || state.video || "video12";
    var num = 15750 + idx * 8;
    return {
      frame_id: idx >= 0 ? v + "_15750_" + num : "pending_" + idx,
      frame_idx: idx,
      video: v,
      phase: null,
      phase_conf: null,
      annotations: {}
    };
  }

  function normalizeAnnotationsMap(frameObj) {
    var map = {};
    if (!frameObj || typeof frameObj !== "object") return map;
    var src = frameObj.annotations || frameObj.labels || frameObj.decisions || frameObj.objects;
    if (!src) return map;

    if (Array.isArray(src)) {
      for (var i = 0; i < src.length; i++) {
        var item = src[i];
        if (!item || typeof item !== "object") continue;
        var k = item.key || item.label;
        if (k) map[String(k)] = Object.assign({}, item, { key: String(k) });
      }
    } else if (typeof src === "object") {
      var keys = Object.keys(src);
      for (var j = 0; j < keys.length; j++) {
        var key = keys[j];
        var val = src[key];
        if (val && typeof val === "object") {
          map[key] = Object.assign({}, val, { key: key });
        } else if (typeof val === "string") {
          map[key] = { key: key, status: val };
        }
      }
    }
    return map;
  }

  function ensureFramesInitialized(count) {
    var target = Math.max(1, Number(count) || state.nFrames || DEFAULT_FRAME_COUNT);
    state.nFrames = target;
    while (state.frames.length < target) {
      state.frames.push(createEmptyFrame(state.frames.length, state.video));
    }
    rebuildFrameIndexMap();
  }

  function rebuildFrameIndexMap() {
    state.frameIndexById = {};
    for (var i = 0; i < state.frames.length; i++) {
      var f = state.frames[i];
      if (f && f.frame_id) {
        state.frameIndexById[f.frame_id] = i;
      }
    }
  }

  function computeCellSignature(cell) {
    if (!cell) return "pending";
    var v = normalizeVerdict(cell);
    var hasBox = Array.isArray(cell.box) && cell.box.length === 4 ? "B" : "";
    var hasTip = Array.isArray(cell.tip) && cell.tip.length === 2 ? "T" : "";
    var hasMask = cell.mask || cell.mask_url || cell.mask_path || cell.mask_b64 ? "M" : "";
    var r = cell.round !== undefined ? String(cell.round) : (cell.loc_round !== undefined ? String(cell.loc_round) : "");
    return [v, hasBox, hasTip, hasMask, r].join(":");
  }

  function markCellFresh(frameIdx, key, prevSig, nextSig, isInitialLoad) {
    var id = frameIdx + "__" + key;
    state.cellSignatures[id] = nextSig;
    if (isInitialLoad) return;
    if (prevSig && prevSig !== nextSig && nextSig.indexOf("pending") !== 0) {
      state.freshCells[id] = Date.now();
    } else if (!prevSig && nextSig.indexOf("pending") !== 0) {
      state.freshCells[id] = Date.now();
    }
  }

  function mergeFrameObject(idx, incomingFrame, isInitialLoad) {
    if (!state.frames[idx]) {
      state.frames[idx] = createEmptyFrame(idx, state.video);
    }
    var target = state.frames[idx];
    if (incomingFrame.frame_id) target.frame_id = incomingFrame.frame_id;
    if (incomingFrame.frame_idx !== undefined) target.frame_idx = Number(incomingFrame.frame_idx);
    if (incomingFrame.img_url) target.img_url = incomingFrame.img_url;
    if (incomingFrame.path) target.path = incomingFrame.path;
    if (incomingFrame.gt_mask_url) target.gt_mask_url = incomingFrame.gt_mask_url;

    if (incomingFrame.phase !== undefined && incomingFrame.phase !== null && incomingFrame.phase !== "") {
      var prevPhase = target.phase || "pending";
      target.phase = incomingFrame.phase;
      if (incomingFrame.phase_conf !== undefined) {
        target.phase_conf = incomingFrame.phase_conf;
      }
      markCellFresh(idx, "__phase__", prevPhase, String(target.phase), isInitialLoad);
    }

    var incomingMap = normalizeAnnotationsMap(incomingFrame);
    var keys = Object.keys(incomingMap);
    for (var i = 0; i < keys.length; i++) {
      var k = keys[i];
      if (state.labels.indexOf(k) === -1) {
        state.labels.push(k);
      }
      var prevCell = target.annotations[k];
      var prevSig = state.cellSignatures[idx + "__" + k] || computeCellSignature(prevCell);
      var merged = Object.assign({}, prevCell || {}, incomingMap[k]);
      target.annotations[k] = merged;
      var nextSig = computeCellSignature(merged);
      markCellFresh(idx, k, prevSig, nextSig, isInitialLoad);
      if (incomingMap[k] && (incomingMap[k].hasNewLivePulse || incomingMap[k].is_new)) {
        state.freshCells[idx + "__" + k] = Date.now();
      }
    }
  }

  function mergeLiveAnnotationRow(row, isInitialLoad) {
    if (!row || typeof row !== "object") return;
    var fid = row.frame_id;
    var idx = -1;
    if (fid && state.frameIndexById[fid] !== undefined) {
      idx = state.frameIndexById[fid];
    } else if (row.frame_idx !== undefined && row.frame_idx !== null) {
      idx = Number(row.frame_idx);
    } else if (fid) {
      var match = String(fid).match(/_(\d+)$/);
      if (match) {
        var fnum = parseInt(match[1], 10);
        var guess = Math.round((fnum - 15750) / 8);
        if (guess >= 0 && guess < state.nFrames) idx = guess;
      }
    }
    if (idx < 0 || idx >= state.frames.length) return;

    var target = state.frames[idx];
    if (fid) {
      target.frame_id = fid;
      state.frameIndexById[fid] = idx;
    }
    if (row.phase && row.phase !== "pending") {
      var prevPhase = target.phase || "pending";
      target.phase = row.phase;
      if (row.phase_conf !== undefined) target.phase_conf = row.phase_conf;
      markCellFresh(idx, "__phase__", prevPhase, String(target.phase), isInitialLoad);
    }

    var key = row.key || row.label;
    if (!key) return;
    if (state.labels.indexOf(key) === -1) {
      state.labels.push(key);
    }
    var prevCell = target.annotations[key];
    var prevSig = state.cellSignatures[idx + "__" + key] || computeCellSignature(prevCell);
    var merged = Object.assign({}, prevCell || {}, row, { key: key });
    if (merged.present === undefined && merged.final_present === undefined && merged.box) {
      merged.present = 1;
    }
    target.annotations[key] = merged;
    var nextSig = computeCellSignature(merged);
    markCellFresh(idx, key, prevSig, nextSig, isInitialLoad);
  }

  function ingestData(data, isInitialLoad) {
    if (!data) {
      ensureFramesInitialized(state.nFrames || DEFAULT_FRAME_COUNT);
      return;
    }

    if (Array.isArray(data)) {
      if (data.length > 0 && (data[0].annotations || data[0].labels || data[0].frame_idx !== undefined)) {
        ensureFramesInitialized(Math.max(state.nFrames, data.length));
        for (var i = 0; i < data.length; i++) {
          var fIdx = data[i].frame_idx !== undefined ? Number(data[i].frame_idx) : i;
          mergeFrameObject(fIdx, data[i], isInitialLoad);
        }
        rebuildFrameIndexMap();
      } else {
        ensureFramesInitialized(state.nFrames);
        for (var j = 0; j < data.length; j++) {
          mergeLiveAnnotationRow(data[j], isInitialLoad);
        }
      }
      return;
    }

    if (typeof data === "object") {
      if (data.video) state.video = data.video;
      var incomingLabels = data.labels || data.label_keys;
      if (Array.isArray(incomingLabels) && incomingLabels.length > 0) {
        state.labels = incomingLabels.slice();
      }
      var frameCount = Number(data.n_frames) ||
        (Array.isArray(data.frames) && data.frames.length) ||
        state.nFrames ||
        DEFAULT_FRAME_COUNT;
      ensureFramesInitialized(frameCount);

      if (Array.isArray(data.frames)) {
        for (var k = 0; k < data.frames.length; k++) {
          var f = data.frames[k];
          var idx = f && f.frame_idx !== undefined ? Number(f.frame_idx) : k;
          mergeFrameObject(idx, f, isInitialLoad);
        }
        rebuildFrameIndexMap();
      }

      var liveRows = data.annotations || data.items || data.rows;
      if (Array.isArray(liveRows)) {
        for (var m = 0; m < liveRows.length; m++) {
          mergeLiveAnnotationRow(liveRows[m], isInitialLoad);
        }
        state.liveLastCount = liveRows.length;
      }
      if (data.ts) state.liveLastTs = data.ts;
    }
  }

  function computePhaseSegments() {
    var segments = [];
    var n = state.frames.length;
    if (n === 0) return segments;

    var phaseColorMap = {};
    var nextColorIdx = 0;

    function getPhaseStyle(phaseName) {
      if (!phaseName || phaseName === "pending") return null;
      if (!phaseColorMap[phaseName]) {
        phaseColorMap[phaseName] = PHASE_PALETTE[nextColorIdx % PHASE_PALETTE.length];
        nextColorIdx++;
      }
      return phaseColorMap[phaseName];
    }

    var start = 0;
    var currentPhase = state.frames[0] && state.frames[0].phase ? String(state.frames[0].phase) : "pending";
    var confSum = state.frames[0] && state.frames[0].phase_conf ? Number(state.frames[0].phase_conf) : 0;
    var confCount = state.frames[0] && state.frames[0].phase_conf ? 1 : 0;

    for (var i = 1; i <= n; i++) {
      var ph = i < n && state.frames[i] && state.frames[i].phase
        ? String(state.frames[i].phase)
        : "pending";
      if (i < n && ph === currentPhase) {
        if (state.frames[i] && state.frames[i].phase_conf) {
          confSum += Number(state.frames[i].phase_conf);
          confCount++;
        }
      } else {
        segments.push({
          start: start,
          end: i - 1,
          span: i - start,
          phase: currentPhase,
          isPending: currentPhase === "pending",
          meanConf: confCount > 0 ? confSum / confCount : null,
          style: getPhaseStyle(currentPhase)
        });
        if (i < n) {
          start = i;
          currentPhase = ph;
          confSum = state.frames[i] && state.frames[i].phase_conf ? Number(state.frames[i].phase_conf) : 0;
          confCount = state.frames[i] && state.frames[i].phase_conf ? 1 : 0;
        }
      }
    }
    return segments;
  }

  function computeSummaryCounts() {
    var totals = {
      accept: 0,
      amber: 0,
      reject: 0,
      absent: 0,
      pending: 0,
      phasedFrames: 0,
      amberFrames: [],
      perLabel: {}
    };

    for (var li = 0; li < state.labels.length; li++) {
      totals.perLabel[state.labels[li]] = {
        accept: 0,
        amber: 0,
        reject: 0,
        absent: 0,
        pending: 0
      };
    }

    for (var fi = 0; fi < state.frames.length; fi++) {
      var f = state.frames[fi];
      if (f && f.phase && f.phase !== "pending") {
        totals.phasedFrames++;
      }
      var frameHasAmber = false;
      for (var k = 0; k < state.labels.length; k++) {
        var key = state.labels[k];
        var cell = f && f.annotations ? f.annotations[key] : null;
        var v = normalizeVerdict(cell);
        totals[v] = (totals[v] || 0) + 1;
        totals.perLabel[key][v] = (totals.perLabel[key][v] || 0) + 1;
        if (v === "amber") frameHasAmber = true;
      }
      if (frameHasAmber) totals.amberFrames.push(fi);
    }
    return totals;
  }

  function formatPhaseShort(phase) {
    if (!phase || phase === "pending") return "pending";
    var clean = String(phase).split(":")[0].trim();
    return clean;
  }

  function showTooltip(evt, html) {
    var tip = state.dom.tooltip;
    if (!tip) return;
    tip.innerHTML = html;
    tip.style.display = "block";
    positionTooltip(evt);
  }

  function positionTooltip(evt) {
    var tip = state.dom.tooltip;
    if (!tip || tip.style.display === "none") return;
    var pad = 12;
    var x = (evt.clientX || 0) + pad;
    var y = (evt.clientY || 0) - tip.offsetHeight - pad;
    if (typeof window !== "undefined") {
      if (x + tip.offsetWidth > window.innerWidth - 8) {
        x = Math.max(8, window.innerWidth - tip.offsetWidth - 8);
      }
      if (y < 8) {
        y = (evt.clientY || 0) + pad;
      }
    }
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hideTooltip() {
    var tip = state.dom.tooltip;
    if (tip) tip.style.display = "none";
  }

  function escapeHtml(str) {
    return String(str === undefined || str === null ? "" : str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function buildCellTooltipHtml(frameIdx, key) {
    var frame = state.frames[frameIdx] || createEmptyFrame(frameIdx, state.video);
    var cell = frame.annotations ? frame.annotations[key] : null;
    var verdict = normalizeVerdict(cell);
    var meta = LABEL_META[key] || { name: key, kind: "label" };

    var confVal = cell && (cell.conf !== undefined ? cell.conf : cell.confidence);
    var trustVal = cell && cell.trust !== undefined ? cell.trust : null;
    var iouVal = cell && (cell.iou !== undefined ? cell.iou : (cell.pass2 && cell.pass2.iou));
    var hasBox = cell && Array.isArray(cell.box) && cell.box.length === 4;
    var hasTip = cell && Array.isArray(cell.tip) && cell.tip.length === 2;
    var hasMask = cell && Boolean(cell.mask || cell.mask_url || cell.mask_path);

    var metaParts = ["Frame " + (frameIdx + 1) + "/" + state.frames.length, escapeHtml(frame.frame_id)];
    if (confVal !== undefined && confVal !== null && confVal !== "") {
      metaParts.push("conf " + Number(confVal).toFixed(2));
    }
    if (trustVal !== undefined && trustVal !== null && trustVal !== "") {
      metaParts.push("trust " + Number(trustVal).toFixed(2));
    }
    if (iouVal !== undefined && iouVal !== null && iouVal !== "") {
      metaParts.push("IoU " + Number(iouVal).toFixed(2));
    }

    var spatialParts = [];
    if (hasMask) spatialParts.push("Mask");
    if (hasBox) spatialParts.push("Box");
    if (hasTip) spatialParts.push("Tip");

    var ev = cell && (cell.evidence || cell.rationale || cell.argon_reason);
    var roundsText = "";
    if (cell && (cell.status_round_0 || cell.round !== undefined)) {
      var rParts = [];
      if (cell.status_round_0) rParts.push("R0:" + cell.status_round_0);
      if (cell.status_round_1) rParts.push("R1:" + cell.status_round_1);
      if (cell.status_round_2) rParts.push("R2:" + cell.status_round_2);
      if (cell.status_round_3) rParts.push("R3:" + cell.status_round_3);
      if (rParts.length === 0 && cell.round !== undefined) {
        rParts.push("Round " + cell.round);
      }
      roundsText = rParts.join(" -> ");
    }

    var html = [
      '<div class="tl-tooltip-head">',
      "  <span>" + escapeHtml(meta.name) + " <small style=\"color:#8A909E\">(" + escapeHtml(key) + ")</small></span>",
      '  <span class="tl-tooltip-badge tl-tooltip-badge--' + verdict + '">' + escapeHtml(verdict) + "</span>",
      "</div>",
      '<div class="tl-tooltip-meta">' + metaParts.join(" · ") + "</div>"
    ];

    if (spatialParts.length > 0) {
      html.push('<div class="tl-tooltip-meta">Spatial: ' + spatialParts.join(" + ") + "</div>");
    }
    if (roundsText) {
      html.push('<div class="tl-tooltip-meta">Loop: ' + escapeHtml(roundsText) + "</div>");
    }
    if (ev) {
      html.push('<div class="tl-tooltip-ev">' + escapeHtml(ev) + "</div>");
    } else if (verdict === "pending") {
      html.push('<div class="tl-tooltip-ev">Annotation pending</div>');
    }
    return html.join("");
  }

  function renderDOM() {
    if (!state.container) return;
    injectStyles();

    var n = Math.max(1, state.frames.length);
    var counts = computeSummaryCounts();
    var phaseSegments = computePhaseSegments();
    var curFrame = state.frames[state.currentFrame] || createEmptyFrame(state.currentFrame, state.video);
    var curPhase = curFrame.phase ? formatPhaseShort(curFrame.phase) : "pending";
    var curPhasePending = !curFrame.phase || curFrame.phase === "pending";
    var now = Date.now();

    var gridCols = "repeat(" + n + ", minmax(" + (state.zoomLevel > 1 ? 22 : 8) + "px, 1fr))";

    var rootEl = state.dom.root;
    if (!rootEl || rootEl.parentNode !== state.container) {
      state.container.innerHTML = "";
      rootEl = document.createElement("div");
      rootEl.className = "sp-timeline";
      rootEl.setAttribute("tabindex", "0");
      rootEl.setAttribute("role", "region");
      rootEl.setAttribute("aria-label", "Surgical Video Annotation Timeline");
      state.container.appendChild(rootEl);
      state.dom.root = rootEl;

      var tooltip = document.createElement("div");
      tooltip.className = "tl-tooltip";
      rootEl.appendChild(tooltip);
      state.dom.tooltip = tooltip;
    }

    var toolbarHtml = [
      '<div class="tl-toolbar">',
      '  <div class="tl-transport">',
      '    <button type="button" class="tl-btn" data-tl-action="prev" title="Previous frame (ArrowLeft / J)">&#9664;&#9664;</button>',
      '    <button type="button" class="tl-btn ' + (state.isPlaying ? "tl-btn--active" : "") + '" data-tl-action="play" title="Play / Pause (Space)">' + (state.isPlaying ? "&#10074;&#10074; Pause" : "&#9654; Play") + "</button>",
      '    <button type="button" class="tl-btn" data-tl-action="next" title="Next frame (ArrowRight / K)">&#9654;&#9654;</button>',
      '    <div class="tl-readout">',
      '      <span data-tl-frame-counter>Frame ' + (state.currentFrame + 1) + " / " + n + "</span>",
      '      <span class="tl-readout-id" data-tl-frame-id>' + escapeHtml(curFrame.frame_id || "pending") + "</span>",
      "    </div>",
      '    <span class="tl-phase-pill ' + (curPhasePending ? "tl-phase-pill--pending" : "") + '" data-tl-phase-pill title="' + escapeHtml(curFrame.phase || "Phase: pending") + '">',
      "      Phase: " + escapeHtml(curPhase),
      "    </span>",
      "  </div>",
      '  <div class="tl-legend">',
      '    <span class="tl-stat-chip" title="Accepted presence annotations"><span class="tl-dot tl-dot--accept"></span>Accept <strong>' + counts.accept + "</strong></span>",
      '    <span class="tl-stat-chip" title="Amber annotations queued for loop / human gate"><span class="tl-dot tl-dot--amber"></span>Amber <strong>' + counts.amber + "</strong></span>",
      '    <span class="tl-stat-chip" title="Rejected annotations"><span class="tl-dot tl-dot--reject"></span>Reject <strong>' + counts.reject + "</strong></span>",
      '    <span class="tl-stat-chip" title="Verified absent structures"><span class="tl-dot tl-dot--absent"></span>Absent <strong>' + counts.absent + "</strong></span>",
      '    <span class="tl-stat-chip" title="Annotations not yet produced"><span class="tl-dot tl-dot--pending"></span>Pending <strong>' + counts.pending + "</strong></span>",
      counts.amberFrames.length > 0
        ? '    <button type="button" class="tl-btn tl-btn--amber" data-tl-action="next-amber" title="Jump to next frame with an AMBER label">Next Amber (' + counts.amberFrames.length + "f) &#8594;</button>"
        : "",
      "  </div>",
      '  <div class="tl-controls">',
      '    <button type="button" class="tl-live-pill ' + (state.livePolling ? "tl-live-pill--on" : "") + '" data-tl-action="toggle-live" title="Polls web/data/live.json every 2s for incoming annotations">',
      '      <span class="tl-live-dot"></span>',
      '      <span>' + (state.livePolling ? "Live 2s" : "Live paused") + (state.liveLastCount ? " · " + state.liveLastCount : "") + "</span>",
      "    </button>",
      '    <button type="button" class="tl-btn ' + (state.zoomLevel > 1 ? "tl-btn--active" : "") + '" data-tl-action="zoom" title="Toggle timeline zoom">' + (state.zoomLevel > 1 ? "Fit 80f" : "Zoom 2x") + "</button>",
      "  </div>",
      "</div>"
    ].join("\n");

    var sidebarRows = [
      '<div class="tl-sidebar-row tl-sidebar-row--ruler">',
      "  <span>Track / Structure</span>",
      "  <span>Acc/Amb</span>",
      "</div>",
      '<div class="tl-sidebar-row tl-sidebar-row--phase">',
      '  <span class="tl-track-title">Surgical Phase</span>',
      '  <span class="tl-track-counts">' + (counts.phasedFrames > 0 ? counts.phasedFrames + "/" + n : "pending") + "</span>",
      "</div>"
    ];

    for (var li = 0; li < state.labels.length; li++) {
      var key = state.labels[li];
      var meta = LABEL_META[key] || { name: key, kind: "label", short: key };
      var lCounts = counts.perLabel[key] || { accept: 0, amber: 0, pending: n };
      var isSel = state.selectedLabel === key;
      var countBadge = lCounts.pending === n
        ? "<span>pending</span>"
        : '<span style="color:#10B981">' + lCounts.accept + '</span>/<span style="color:#F59E0B">' + lCounts.amber + "</span>";
      sidebarRows.push([
        '<div class="tl-sidebar-row tl-sidebar-row--label ' + (isSel ? "tl-sidebar-row--selected" : "") + '" data-tl-select-label="' + escapeHtml(key) + '">',
        '  <span class="tl-track-title">',
        '    <span class="tl-track-kind">' + escapeHtml(meta.kind) + "</span>",
        "    <span>" + escapeHtml(meta.short || meta.name) + "</span>",
        "  </span>",
        '  <span class="tl-track-counts">' + countBadge + "</span>",
        "</div>"
      ].join(""));
    }

    var rulerTicks = [];
    for (var ti = 0; ti < n; ti++) {
      var isMajor = ti === 0 || ti % 5 === 0 || ti === n - 1;
      rulerTicks.push(
        '<div class="tl-tick ' + (isMajor ? "tl-tick--major" : "") + '" data-frame-idx="' + ti + '">' +
        (isMajor ? ti : "") +
        "</div>"
      );
    }

    var phaseBandsHtml = [];
    for (var pi = 0; pi < phaseSegments.length; pi++) {
      var seg = phaseSegments[pi];
      var colSpan = (seg.start + 1) + " / " + (seg.end + 2);
      var inlineStyle = "grid-column: " + colSpan + ";";
      if (!seg.isPending && seg.style) {
        inlineStyle += "background:" + seg.style.bg + ";border-color:" + seg.style.border + ";color:" + seg.style.text + ";";
      }
      var labelText = seg.isPending ? "pending" : formatPhaseShort(seg.phase);
      phaseBandsHtml.push(
        '<div class="tl-phase-band ' + (seg.isPending ? "tl-phase-band--pending" : "") + '" ' +
        'style="' + inlineStyle + '" ' +
        'data-phase-start="' + seg.start + '" data-phase-end="' + seg.end + '" ' +
        ' data-phase-name="' + escapeHtml(seg.phase) + '">' +
        escapeHtml(labelText) +
        "</div>"
      );
    }

    var tracksHtml = [];
    for (var tk = 0; tk < state.labels.length; tk++) {
      var tKey = state.labels[tk];
      var cellsHtml = [];
      for (var fi = 0; fi < n; fi++) {
        var frameObj = state.frames[fi];
        var cellObj = frameObj && frameObj.annotations ? frameObj.annotations[tKey] : null;
        var verdict = normalizeVerdict(cellObj);
        var isCur = fi === state.currentFrame;
        var freshTs = state.freshCells[fi + "__" + tKey];
        var isFresh = freshTs && (now - freshTs < FRESH_HIGHLIGHT_MS);
        var hasSpatial = cellObj && (
          (Array.isArray(cellObj.box) && cellObj.box.length === 4) ||
          (Array.isArray(cellObj.tip) && cellObj.tip.length === 2) ||
          Boolean(cellObj.mask || cellObj.mask_url || cellObj.mask_path || cellObj.mask_b64)
        );

        cellsHtml.push(
          '<div class="tl-cell track-cell tl-cell--' + verdict +
          (isCur ? " tl-cell--current active-frame" : "") +
          (isFresh ? " tl-cell--fresh" : "") + '" ' +
          'data-frame-idx="' + fi + '" data-label-key="' + escapeHtml(tKey) + '" data-verdict="' + verdict + '">' +
          (hasSpatial ? '<span class="tl-spatial-dot"></span>' : "") +
          "</div>"
        );
      }
      tracksHtml.push(
        '<div class="tl-track" data-track-key="' + escapeHtml(tKey) + '" style="grid-template-columns: ' + gridCols + ';">' +
        cellsHtml.join("") +
        "</div>"
      );
    }

    var playheadPct = n > 0 ? ((state.currentFrame + 0.5) / n) * 100 : 0;

    var workspaceHtml = [
      '<div class="tl-workspace">',
      '  <div class="tl-sidebar">' + sidebarRows.join("") + "</div>",
      '  <div class="tl-stage-scroll">',
      '    <div class="tl-stage" style="' + (state.zoomLevel > 1 ? "width:" + (state.zoomLevel * 100) + "%;" : "width:100%;") + '">',
      '      <div class="tl-ruler" style="grid-template-columns: ' + gridCols + ';">' + rulerTicks.join("") + "</div>",
      '      <div class="tl-phase-track" style="grid-template-columns: ' + gridCols + ';">' + phaseBandsHtml.join("") + "</div>",
      tracksHtml.join(""),
      '      <div class="tl-playhead" data-tl-playhead style="left: ' + playheadPct.toFixed(4) + '%;">',
      '        <div class="tl-playhead-cap"></div>',
      "      </div>",
      "    </div>",
      "  </div>",
      "</div>"
    ].join("\n");

    var tooltipEl = state.dom.tooltip;
    rootEl.innerHTML = toolbarHtml + workspaceHtml;
    if (tooltipEl) rootEl.appendChild(tooltipEl);

    state.dom.stage = rootEl.querySelector(".tl-stage");
    state.dom.stageScroll = rootEl.querySelector(".tl-stage-scroll");
    state.dom.playhead = rootEl.querySelector("[data-tl-playhead]");
    state.dom.frameCounter = rootEl.querySelector("[data-tl-frame-counter]");
    state.dom.frameId = rootEl.querySelector("[data-tl-frame-id]");
    state.dom.phasePill = rootEl.querySelector("[data-tl-phase-pill]");

    if (typeof document !== "undefined" && typeof document.getElementById === "function") {
      var extCount = document.getElementById("timeline-frame-count");
      if (extCount) extCount.textContent = String(n);
    }
  }

  function updatePlayheadDOM() {
    var rootEl = state.dom.root;
    if (!rootEl) return;
    var n = Math.max(1, state.frames.length);
    var pct = ((state.currentFrame + 0.5) / n) * 100;

    if (state.dom.playhead) {
      state.dom.playhead.style.left = pct.toFixed(4) + "%";
    }
    var curFrame = state.frames[state.currentFrame] || createEmptyFrame(state.currentFrame, state.video);
    if (state.dom.frameCounter) {
      state.dom.frameCounter.textContent = "Frame " + (state.currentFrame + 1) + " / " + n;
    }
    if (state.dom.frameId) {
      state.dom.frameId.textContent = curFrame.frame_id || "pending";
    }
    if (state.dom.phasePill) {
      var curPhase = curFrame.phase ? formatPhaseShort(curFrame.phase) : "pending";
      var isPending = !curFrame.phase || curFrame.phase === "pending";
      state.dom.phasePill.textContent = "Phase: " + curPhase;
      state.dom.phasePill.title = curFrame.phase || "Phase: pending";
      state.dom.phasePill.classList.toggle("tl-phase-pill--pending", isPending);
    }

    var prevCur = rootEl.querySelectorAll(".tl-cell--current, .tl-cell.active-frame");
    for (var i = 0; i < prevCur.length; i++) {
      prevCur[i].classList.remove("tl-cell--current");
      prevCur[i].classList.remove("active-frame");
    }
    var nextCur = rootEl.querySelectorAll('.tl-cell[data-frame-idx="' + state.currentFrame + '"]');
    for (var j = 0; j < nextCur.length; j++) {
      nextCur[j].classList.add("tl-cell--current");
      nextCur[j].classList.add("active-frame");
    }

    if (state.dom.stageScroll && state.dom.stage && state.zoomLevel > 1) {
      var scrollEl = state.dom.stageScroll;
      var stageWidth = state.dom.stage.clientWidth;
      var targetX = (pct / 100) * stageWidth;
      var viewLeft = scrollEl.scrollLeft;
      var viewRight = viewLeft + scrollEl.clientWidth;
      if (targetX < viewLeft + 24 || targetX > viewRight - 24) {
        scrollEl.scrollLeft = Math.max(0, targetX - scrollEl.clientWidth / 2);
      }
    }
  }

  function dispatchTimelineEvent(name, detail) {
    if (typeof CustomEvent === "undefined") return;
    if (state.container && state.container.dispatchEvent) {
      state.container.dispatchEvent(new CustomEvent(name, { bubbles: true, detail: detail }));
    } else if (typeof window !== "undefined" && window.dispatchEvent) {
      window.dispatchEvent(new CustomEvent(name, { detail: detail }));
    }
  }

  function setFrameInternal(target, notifySeek, labelKey) {
    var n = Math.max(1, state.frames.length);
    var idx = state.currentFrame;

    if (typeof target === "string" && state.frameIndexById[target] !== undefined) {
      idx = state.frameIndexById[target];
    } else if (typeof target === "number" || !isNaN(Number(target))) {
      idx = Math.round(Number(target));
    }

    idx = Math.max(0, Math.min(n - 1, idx));
    state.currentFrame = idx;
    updatePlayheadDOM();

    if (notifySeek) {
      var frameObj = state.frames[idx] || createEmptyFrame(idx, state.video);
      if (typeof state.onSeek === "function") {
        state.onSeek(idx, labelKey || frameObj, frameObj);
      }
      var payload = {
        index: idx,
        frameIndex: idx,
        frameId: frameObj.frame_id,
        frame: frameObj,
        key: labelKey || state.selectedLabel || null,
        label: labelKey || state.selectedLabel || null
      };
      dispatchTimelineEvent("timeline:seek", payload);
      dispatchTimelineEvent("surgproof:seek", payload);
    }
    return idx;
  }

  function frameIndexFromClientX(clientX) {
    var stage = state.dom.stage;
    if (!stage) return state.currentFrame;
    var rect = stage.getBoundingClientRect();
    if (!rect || rect.width <= 0) return state.currentFrame;
    var ratio = (clientX - rect.left) / rect.width;
    var clamped = Math.max(0, Math.min(0.999999, ratio));
    return Math.floor(clamped * state.frames.length);
  }

  function bindEvents() {
    var rootEl = state.dom.root;
    if (!rootEl) return;

    rootEl.addEventListener("click", function (evt) {
      var actionBtn = evt.target.closest("[data-tl-action]");
      if (actionBtn) {
        var action = actionBtn.getAttribute("data-tl-action");
        if (action === "prev") {
          pausePlayback();
          setFrameInternal(state.currentFrame - 1, true);
        } else if (action === "next") {
          pausePlayback();
          setFrameInternal(state.currentFrame + 1, true);
        } else if (action === "play") {
          togglePlayback();
        } else if (action === "next-amber") {
          pausePlayback();
          jumpToNextAmber();
        } else if (action === "toggle-live") {
          if (state.livePolling) {
            stopLivePolling();
          } else {
            startLivePolling(state.liveUrl, LIVE_POLL_INTERVAL_MS);
          }
          renderDOM();
        } else if (action === "zoom") {
          state.zoomLevel = state.zoomLevel > 1 ? 1 : 2;
          renderDOM();
        }
        return;
      }

      var selectLabelRow = evt.target.closest("[data-tl-select-label]");
      if (selectLabelRow) {
        var key = selectLabelRow.getAttribute("data-tl-select-label");
        state.selectedLabel = state.selectedLabel === key ? null : key;
        renderDOM();
        var frameObj = state.frames[state.currentFrame];
        var detail = {
          index: state.currentFrame,
          frameIndex: state.currentFrame,
          frameId: frameObj ? frameObj.frame_id : null,
          key: key,
          label: key
        };
        dispatchTimelineEvent("timeline:select-label", detail);
        dispatchTimelineEvent("surgproof:select-label", detail);
        return;
      }

      var cellEl = evt.target.closest(".tl-cell");
      if (cellEl) {
        var fIdx = Number(cellEl.getAttribute("data-frame-idx"));
        var lKey = cellEl.getAttribute("data-label-key");
        state.selectedLabel = lKey;
        setFrameInternal(fIdx, true, lKey);
        var fObj = state.frames[state.currentFrame];
        var cellDetail = {
          index: state.currentFrame,
          frameIndex: state.currentFrame,
          frameId: fObj ? fObj.frame_id : null,
          key: lKey,
          label: lKey
        };
        dispatchTimelineEvent("timeline:select-label", cellDetail);
        dispatchTimelineEvent("surgproof:select-label", cellDetail);
        return;
      }

      var phaseBand = evt.target.closest(".tl-phase-band");
      if (phaseBand) {
        var seekIdx = frameIndexFromClientX(evt.clientX);
        setFrameInternal(seekIdx, true);
      }
    });

    rootEl.addEventListener("mousedown", function (evt) {
      if (evt.button !== 0) return;
      var inStage = evt.target.closest(".tl-stage");
      if (!inStage) return;
      state.isScrubbing = true;
      pausePlayback();
      var idx = frameIndexFromClientX(evt.clientX);
      setFrameInternal(idx, true);
    });

    rootEl.addEventListener("mousemove", function (evt) {
      var cellEl = evt.target.closest(".tl-cell");
      if (cellEl) {
        var fIdx = Number(cellEl.getAttribute("data-frame-idx"));
        var lKey = cellEl.getAttribute("data-label-key");
        showTooltip(evt, buildCellTooltipHtml(fIdx, lKey));
        return;
      }
      var phaseBand = evt.target.closest(".tl-phase-band");
      if (phaseBand) {
        var pName = phaseBand.getAttribute("data-phase-name") || "pending";
        var pStart = Number(phaseBand.getAttribute("data-phase-start")) + 1;
        var pEnd = Number(phaseBand.getAttribute("data-phase-end")) + 1;
        showTooltip(
          evt,
          '<div class="tl-tooltip-head"><span>Surgical Phase</span></div>' +
          '<div class="tl-tooltip-meta">Frames ' + pStart + "–" + pEnd + "</div>" +
          '<div class="tl-tooltip-ev">' + escapeHtml(pName) + "</div>"
        );
        return;
      }
      hideTooltip();
    });

    rootEl.addEventListener("mouseleave", function () {
      hideTooltip();
    });

    if (typeof window !== "undefined") {
      if (state.boundMoveHandler) window.removeEventListener("mousemove", state.boundMoveHandler);
      if (state.boundUpHandler) window.removeEventListener("mouseup", state.boundUpHandler);
      if (state.boundKeyHandler) window.removeEventListener("keydown", state.boundKeyHandler);

      state.boundMoveHandler = function (evt) {
        if (!state.isScrubbing) return;
        var idx = frameIndexFromClientX(evt.clientX);
        if (idx !== state.currentFrame) {
          setFrameInternal(idx, true);
        }
      };

      state.boundUpHandler = function () {
        state.isScrubbing = false;
      };

      state.boundKeyHandler = function (evt) {
        var tag = evt.target && evt.target.tagName ? evt.target.tagName.toLowerCase() : "";
        if (tag === "input" || tag === "textarea" || tag === "select" || (evt.target && evt.target.isContentEditable)) {
          return;
        }
        if (evt.key === "ArrowLeft" || evt.key === "j" || evt.key === "J") {
          evt.preventDefault();
          pausePlayback();
          setFrameInternal(state.currentFrame - 1, true);
        } else if (evt.key === "ArrowRight" || evt.key === "k" || evt.key === "K") {
          evt.preventDefault();
          pausePlayback();
          setFrameInternal(state.currentFrame + 1, true);
        } else if (evt.key === " ") {
          evt.preventDefault();
          togglePlayback();
        } else if (evt.key === "Home") {
          evt.preventDefault();
          pausePlayback();
          setFrameInternal(0, true);
        } else if (evt.key === "End") {
          evt.preventDefault();
          pausePlayback();
          setFrameInternal(state.frames.length - 1, true);
        }
      };

      window.addEventListener("mousemove", state.boundMoveHandler);
      window.addEventListener("mouseup", state.boundUpHandler);
      window.addEventListener("keydown", state.boundKeyHandler);
    }
  }

  function startPlayback() {
    if (state.isPlaying) return;
    state.isPlaying = true;
    var interval = Math.max(60, Math.round(1000 / (state.fps || 6)));
    state.playTimer = setInterval(function () {
      var next = (state.currentFrame + 1) % Math.max(1, state.frames.length);
      setFrameInternal(next, true);
    }, interval);
    renderDOM();
  }

  function pausePlayback() {
    if (!state.isPlaying && !state.playTimer) return;
    state.isPlaying = false;
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
    renderDOM();
  }

  function togglePlayback() {
    if (state.isPlaying) {
      pausePlayback();
    } else {
      startPlayback();
    }
  }

  function jumpToNextAmber() {
    var counts = computeSummaryCounts();
    var ambers = counts.amberFrames;
    if (!ambers || ambers.length === 0) return state.currentFrame;
    for (var i = 0; i < ambers.length; i++) {
      if (ambers[i] > state.currentFrame) {
        return setFrameInternal(ambers[i], true);
      }
    }
    return setFrameInternal(ambers[0], true);
  }

  function pollLiveOnce() {
    if (typeof fetch !== "function" || state.liveFetchInFlight) return Promise.resolve(null);
    state.liveFetchInFlight = true;
    var urlsToTry = [state.liveUrl, "web/data/live.json", "/data/live.json", "/web/data/live.json"];
    var uniqueUrls = [];
    for (var i = 0; i < urlsToTry.length; i++) {
      if (urlsToTry[i] && uniqueUrls.indexOf(urlsToTry[i]) === -1) {
        uniqueUrls.push(urlsToTry[i]);
      }
    }

    function tryIndex(uIdx) {
      if (uIdx >= uniqueUrls.length) {
        state.liveFetchInFlight = false;
        return Promise.resolve(null);
      }
      var u = uniqueUrls[uIdx];
      var sep = u.indexOf("?") === -1 ? "?" : "&";
      return fetch(u + sep + "_t=" + Date.now(), { cache: "no-store" })
        .then(function (res) {
          if (!res || !res.ok) return tryIndex(uIdx + 1);
          state.liveUrl = u;
          return res.json();
        })
        .then(function (payload) {
          if (payload) {
            update(payload);
          }
          state.liveFetchInFlight = false;
          return payload;
        })
        .catch(function () {
          return tryIndex(uIdx + 1);
        });
    }

    return tryIndex(0);
  }

  function startLivePolling(url, intervalMs) {
    stopLivePolling();
    if (url) state.liveUrl = url;
    state.livePolling = true;
    var ms = Number(intervalMs) || LIVE_POLL_INTERVAL_MS;
    state.liveTimer = setInterval(function () {
      pollLiveOnce();
    }, ms);
  }

  function stopLivePolling() {
    state.livePolling = false;
    if (state.liveTimer) {
      clearInterval(state.liveTimer);
      state.liveTimer = null;
    }
  }

  function scheduleFreshCleanup() {
    if (state.freshCleanupTimer) clearTimeout(state.freshCleanupTimer);
    if (Object.keys(state.freshCells).length === 0) return;
    state.freshCleanupTimer = setTimeout(function () {
      var now = Date.now();
      var keys = Object.keys(state.freshCells);
      for (var i = 0; i < keys.length; i++) {
        if (now - state.freshCells[keys[i]] >= FRESH_HIGHLIGHT_MS) {
          delete state.freshCells[keys[i]];
        }
      }
    }, FRESH_HIGHLIGHT_MS + 60);
  }

  /**
   * Mounts the multi-track timeline inside container element `el`.
   * @param {HTMLElement|string} el - DOM Element or selector string
   * @param {Object|Array|Function} [data] - Initial video12.json / frames data (or options object)
   * @param {Function} [onSeek] - Callback invoked as onSeek(frameIndex, frameObj) when user seeks
   */
  function mount(el, data, onSeek) {
    var targetEl = typeof el === "string" && typeof document !== "undefined"
      ? document.querySelector(el)
      : el;
    if (!targetEl) {
      throw new Error("Timeline.mount: container element is required");
    }

    var initialData = data;
    var seekCb = onSeek;
    var pollLive = true;
    var liveUrl = "data/live.json";

    if (typeof data === "function" && !onSeek) {
      initialData = null;
      seekCb = data;
    } else if (
      data &&
      typeof data === "object" &&
      !Array.isArray(data) &&
      !data.frames &&
      !data.annotations &&
      (data.data !== undefined || data.onSeek !== undefined || data.pollLive !== undefined)
    ) {
      initialData = data.data || null;
      seekCb = data.onSeek || onSeek || null;
      if (data.pollLive !== undefined) pollLive = Boolean(data.pollLive);
      if (data.liveUrl) liveUrl = data.liveUrl;
    }

    if (
      state.container === targetEl &&
      state.dom.root &&
      state.dom.root.parentNode === targetEl &&
      initialData
    ) {
      if (typeof seekCb === "function") state.onSeek = seekCb;
      return update(initialData);
    }

    pausePlayback();
    stopLivePolling();

    state.container = targetEl;
    state.onSeek = typeof seekCb === "function" ? seekCb : null;
    state.frames = [];
    state.frameIndexById = {};
    state.cellSignatures = {};
    state.freshCells = {};
    state.currentFrame = 0;
    state.selectedLabel = null;
    state.dom = {};

    ingestData(initialData, true);
    renderDOM();
    bindEvents();

    if (pollLive && typeof window !== "undefined") {
      startLivePolling(liveUrl, LIVE_POLL_INTERVAL_MS);
    }

    return api;
  }

  /**
   * Updates timeline tracks & cells from full video12.json or incremental live.json data.
   * Newly arrived annotations animate in smoothly; missing annotations remain "pending".
   * @param {Object|Array} data
   */
  function update(data) {
    var wasEmpty = Object.keys(state.cellSignatures).length === 0;
    ingestData(data, wasEmpty);
    renderDOM();
    scheduleFreshCleanup();
    return api;
  }

  /**
   * Moves the playhead to frame index `i` (or `frame_id` string) without triggering a circular onSeek loop.
   * @param {number|string} i
   */
  function setFrame(i) {
    setFrameInternal(i, false);
    return api;
  }

  function destroy() {
    pausePlayback();
    stopLivePolling();
    if (state.freshCleanupTimer) {
      clearTimeout(state.freshCleanupTimer);
      state.freshCleanupTimer = null;
    }
    if (typeof window !== "undefined") {
      if (state.boundMoveHandler) window.removeEventListener("mousemove", state.boundMoveHandler);
      if (state.boundUpHandler) window.removeEventListener("mouseup", state.boundUpHandler);
      if (state.boundKeyHandler) window.removeEventListener("keydown", state.boundKeyHandler);
    }
    if (state.container) {
      state.container.innerHTML = "";
    }
    state.container = null;
    state.dom = {};
  }

  function getState() {
    return {
      video: state.video,
      nFrames: state.frames.length,
      currentFrame: state.currentFrame,
      selectedLabel: state.selectedLabel,
      isPlaying: state.isPlaying,
      livePolling: state.livePolling,
      labels: state.labels.slice(),
      counts: computeSummaryCounts(),
      phaseSegments: computePhaseSegments()
    };
  }

  var api = {
    mount: mount,
    update: update,
    setFrame: setFrame,
    seek: function (i) { return setFrameInternal(i, true); },
    show: setFrame,
    play: startPlayback,
    pause: pausePlayback,
    togglePlay: togglePlayback,
    nextFrame: function () { pausePlayback(); return setFrameInternal(state.currentFrame + 1, true); },
    prevFrame: function () { pausePlayback(); return setFrameInternal(state.currentFrame - 1, true); },
    nextAmber: jumpToNextAmber,
    startLivePoll: startLivePolling,
    stopLivePoll: stopLivePolling,
    pollLiveOnce: pollLiveOnce,
    getState: getState,
    destroy: destroy
  };

  return api;
});
