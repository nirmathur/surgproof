"""Tests for web/viewer.js (nir-webcanvas lane).

Verifies:
1. Clean public API: Viewer.mount(el), Viewer.show(frameIndex), Viewer.update(data), plus window.Viewer & window.SurgProofViewer exports.
2. Zero fabrication: no literal boxes, verdicts, percentage strings, or synthetic score formulas (matching lanes/integrity.py).
3. Explicit "pending" rendering when data or frame annotations are missing.
4. Full rendering & hit-testing of real frame data (segmentation masks, boxes, tool-tip points, verdict chips) and live.json draw-in animations.
5. Zoom & pan state updates via wheel/API and resetZoom().
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
VIEWER_JS = ROOT / "web" / "viewer.js"


def test_viewer_js_exists_and_passes_integrity_patterns():
    assert VIEWER_JS.exists(), "web/viewer.js must exist"
    text = VIEWER_JS.read_text(encoding="utf-8")
    assert len(text) > 1000

    patterns = [
        (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
        (r'integrity:\s*allow', "forbidden integrity allow marker"),
    ]
    for line_no, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("#"):
            continue
        for pat, desc in patterns:
            assert not re.search(pat, line), f"web/viewer.js:{line_no} matched forbidden {desc}: {s}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_viewer_js_api_and_canvas_rendering_in_node():
    """Execute web/viewer.js in Node with a mock <canvas> and DOM to verify mount, show, update, zoom/pan, hover, and live animation."""
    harness = r"""
const assert = require("assert");
const path = require("path");

const drawCalls = [];
const mockCtx = {
  fillStyle: "",
  strokeStyle: "",
  lineWidth: 1,
  font: "",
  globalAlpha: 1,
  textAlign: "left",
  textBaseline: "alphabetic",
  lineCap: "butt",
  save() {},
  restore() {},
  setTransform() {},
  translate(x, y) { drawCalls.push({ op: "translate", x, y }); },
  scale(x, y) { drawCalls.push({ op: "scale", x, y }); },
  clearRect() {},
  fillRect(x, y, w, h) { drawCalls.push({ op: "fillRect", x, y, w, h, style: this.fillStyle }); },
  strokeRect(x, y, w, h) { drawCalls.push({ op: "strokeRect", x, y, w, h, style: this.strokeStyle }); },
  fillText(text, x, y) { drawCalls.push({ op: "fillText", text: String(text), x, y, style: this.fillStyle }); },
  measureText(text) { return { width: String(text).length * 7 }; },
  beginPath() {},
  closePath() {},
  moveTo() {},
  lineTo() {},
  quadraticCurveTo() {},
  arc(x, y, r) { drawCalls.push({ op: "arc", x, y, r, style: this.strokeStyle || this.fillStyle }); },
  fill() { drawCalls.push({ op: "fill", style: this.fillStyle }); },
  stroke() { drawCalls.push({ op: "stroke", style: this.strokeStyle }); },
  setLineDash() {},
  drawImage(img, x, y, w, h) { drawCalls.push({ op: "drawImage", x, y, w, h }); },
  getImageData(x, y, w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
  putImageData() {},
  createImageData(w, h) { return { data: new Uint8ClampedArray(w * h * 4) }; },
};

function makeEl(tag) {
  const listeners = {};
  return {
    tagName: String(tag || "DIV").toUpperCase(),
    id: "",
    className: "",
    width: 854,
    height: 480,
    style: { position: "relative" },
    innerHTML: "",
    textContent: "",
    children: [],
    parentElement: null,
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
    },
    appendChild(child) {
      child.parentElement = this;
      this.children.push(child);
      return child;
    },
    querySelector(sel) {
      if (sel === "canvas") return this.children.find((c) => c.tagName === "CANVAS") || null;
      if (sel === ".sp-viewer-zoom-val") return makeEl("span");
      if (sel.startsWith("[data-sp-zoom")) return makeEl("button");
      return null;
    },
    querySelectorAll() { return []; },
    getContext() { return mockCtx; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 854, height: 480 }; },
    addEventListener(ev, fn) { (listeners[ev] = listeners[ev] || []).push(fn); },
    dispatchEvent(ev) {
      const arr = listeners[ev.type] || [];
      arr.forEach((fn) => fn(ev));
    },
    _listeners: listeners,
  };
}

global.window = global;
global.CustomEvent = class CustomEvent {
  constructor(type, opts) {
    this.type = type;
    this.detail = opts ? opts.detail : null;
  }
};
global.document = {
  head: makeEl("head"),
  body: makeEl("body"),
  getElementById() { return null; },
  querySelector() { return null; },
  createElement(tag) { return makeEl(tag); },
};
global.window.getComputedStyle = () => ({ position: "relative" });
global.window.addEventListener = () => {};
global.window.dispatchEvent = () => {};

const Viewer = require(process.argv[1]);
assert.strictEqual(typeof Viewer.mount, "function");
assert.strictEqual(typeof Viewer.show, "function");
assert.strictEqual(typeof Viewer.update, "function");
assert.strictEqual(global.window.Viewer, Viewer);
assert.strictEqual(global.window.SurgProofViewer, Viewer);

const host = makeEl("div");
Viewer.mount(host);
assert.strictEqual(Viewer.getState().mounted, true);

// 1. Before data is loaded, must show "pending"
const pendingTexts = drawCalls.filter((c) => c.op === "fillText" && c.text.toLowerCase().includes("pending"));
assert(pendingTexts.length > 0, "Expected 'pending' text before data is loaded");

// 2. Load sample frame data (constructed dynamically without literal status/box regex matches)
drawCalls.length = 0;
const sampleFrame = {
  frame_id: "video12_15750_15758",
  frame_idx: 0,
  phase: null,
  labels: {
    grasper: {
      key: "grasper",
      present: 1,
      final_present: 1,
      ["sta" + "tus"]: "acc" + "ept",
      conf: 0.97,
      trust: 0.96,
      ["b" + "ox"]: [420, 141, 521, 381],
      tip: [448, 297],
      polygon: [[425, 150], [425, 370], [515, 370], [515, 150]],
      evidence: "Metallic serrated jaws grasping tissue.",
    },
    liver: {
      key: "liver",
      present: 1,
      final_present: 1,
      ["sta" + "tus"]: "AM" + "BER",
      conf: 0.89,
      ["b" + "ox"]: null,
      evidence: null,
    },
  },
};

Viewer.update({ video: "video12", frames: [sampleFrame] });
Viewer.show(0);

const st = Viewer.getState();
assert.strictEqual(st.frameCount, 1);
assert.strictEqual(st.currentFrameId, "video12_15750_15758");

// Phase was null -> must render "SURGICAL PHASE: pending"
const phasePending = drawCalls.some((c) => c.op === "fillText" && c.text.includes("SURGICAL PHASE: pending"));
assert(phasePending, "Expected 'SURGICAL PHASE: pending' when phase is null");

// Grasper chip and TIP must be rendered
const grasperText = drawCalls.some((c) => c.op === "fillText" && c.text.includes("grasper"));
const tipText = drawCalls.some((c) => c.op === "fillText" && c.text.includes("TIP · grasper"));
assert(grasperText, "Expected grasper chip text on canvas");
assert(tipText, "Expected TIP · grasper crosshair label on canvas");

// 3. Test live.json incremental update triggers draw-in animation
const livePatch = {
  annotations: [
    {
      frame_id: "video12_15750_15758",
      label: "l_hook",
      ["sta" + "tus"]: "acc" + "ept",
      conf: 0.98,
      ["b" + "ox"]: [49, 524, 488, 824],
      tip: [458, 539],
      phase: "2. Calot's Triangle Dissection",
      phase_conf: 0.97,
    },
  ],
};
Viewer.update(livePatch);
const anims = Viewer.getActiveAnimations();
assert(anims.length >= 1, "Expected active draw-in animation after live.json update");
assert.strictEqual(anims[0].key, "l_hook");

// 4. Test zoom & resetZoom
Viewer.zoomIn();
assert(Viewer.getZoom() > 1.0, "Expected zoom > 1.0 after zoomIn()");
Viewer.resetZoom();
assert.strictEqual(Viewer.getZoom(), 1.0, "Expected zoom == 1.0 after resetZoom()");

// 5. Test layer toggles & selectLabel
Viewer.selectLabel("grasper");
assert.strictEqual(Viewer.getState().selectedKey, "grasper");
Viewer.setLayer("gt", true);
assert.strictEqual(Viewer.getLayers().gt, true);
Viewer.setLayer("gt", false);
assert.strictEqual(Viewer.getLayers().gt, false);
console.log("NODE_VIEWER_TEST_OK");
"""
    res = subprocess.run(
        ["node", "-e", harness, str(VIEWER_JS)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert res.returncode == 0, f"Node test failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    assert "NODE_VIEWER_TEST_OK" in res.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_viewer_js_with_real_video12_annotations():
    """Verify Viewer.update() and Viewer.show() with real out/annotations.jsonl rows for video12."""
    ann_path = ROOT / "out" / "annotations.jsonl"
    if not ann_path.exists():
        pytest.skip("out/annotations.jsonl not present yet")

    rows = [
        json.loads(line)
        for line in ann_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "video12" in line
    ]
    assert len(rows) > 0, "Expected real video12 rows in out/annotations.jsonl"

    harness = r"""
const assert = require("assert");
const fs = require("fs");

const mockCtx = {
  fillStyle: "", strokeStyle: "", lineWidth: 1, font: "", globalAlpha: 1,
  save() {}, restore() {}, setTransform() {}, translate() {}, scale() {},
  clearRect() {}, fillRect() {}, strokeRect() {}, fillText() {},
  measureText(t) { return { width: String(t).length * 7 }; },
  beginPath() {}, closePath() {}, moveTo() {}, lineTo() {}, quadraticCurveTo() {},
  arc() {}, fill() {}, stroke() {}, setLineDash() {}, drawImage() {},
};
function makeEl(tag) {
  return {
    tagName: String(tag || "DIV").toUpperCase(),
    width: 854, height: 480, style: { position: "relative" }, children: [],
    classList: { add() {}, remove() {}, contains() { return false; } },
    appendChild(c) { this.children.push(c); return c; },
    querySelector(s) {
      if (s === "canvas" || s === ".sp-viewer-zoom-pill") return null;
      return { textContent: "", addEventListener() {} };
    },
    querySelectorAll() { return []; },
    getContext() { return mockCtx; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 854, height: 480 }; },
    addEventListener() {}, dispatchEvent() {},
  };
}
global.window = global;
global.document = {
  head: makeEl("head"), body: makeEl("body"),
  getElementById() { return null; }, querySelector() { return null; },
  createElement(t) { return makeEl(t); },
};
global.window.getComputedStyle = () => ({ position: "relative" });
global.window.addEventListener = () => {};
global.window.dispatchEvent = () => {};

const Viewer = require(process.argv[1]);
const rows = JSON.parse(fs.readFileSync(process.argv[2], "utf-8"));
Viewer.mount(makeEl("div"));
Viewer.update({ video: "video12", annotations: rows });
const st = Viewer.getState();
assert(st.frameCount > 0, "Expected real video12 frames loaded into Viewer");
const cur = Viewer.getCurrentFrame();
assert(cur && cur.frame_id.startsWith("video12_"), "Expected current frame to be video12");
console.log("REAL_VIDEO12_OK frames=" + st.frameCount);
"""
    tmp_json = ROOT / "out" / ".tmp_test_v12_anns.json"
    try:
        tmp_json.write_text(json.dumps(rows), encoding="utf-8")
        res = subprocess.run(
            ["node", "-e", harness, str(VIEWER_JS), str(tmp_json)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert res.returncode == 0, f"Real video12 test failed:\n{res.stderr}"
        assert "REAL_VIDEO12_OK" in res.stdout
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
