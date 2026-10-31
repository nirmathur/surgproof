import csv
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIMELINE_JS = ROOT / "web" / "timeline.js"


def test_timeline_js_exists_and_passes_integrity_patterns():
    assert TIMELINE_JS.exists(), "web/timeline.js must exist"
    text = TIMELINE_JS.read_text(encoding="utf-8")

    # Required public API symbols
    for sym in ["mount", "update", "setFrame"]:
        assert sym in text, f"Missing required API method {sym} in web/timeline.js"

    # Strict integrity.py patterns must never appear anywhere in web/timeline.js
    patterns = [
        (r'["\']?box["\']?\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
    ]
    for lineno, line in enumerate(text.splitlines(), 1):
        assert "integrity: allow" not in line, f"Line {lineno} uses forbidden marker"
        for pat, desc in patterns:
            assert not re.search(pat, line), f"Line {lineno} matched {desc}: {line}"


def test_timeline_js_mount_update_setframe_with_real_video12_data():
    decisions_path = ROOT / "out" / "decisions.csv"
    annotations_path = ROOT / "out" / "annotations.jsonl"

    # Build a real video12 payload from out/decisions.csv and out/annotations.jsonl
    frames_by_id = {}
    if decisions_path.exists():
        with open(decisions_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("video") != "video12":
                    continue
                fid = row["frame_id"]
                if fid not in frames_by_id:
                    frames_by_id[fid] = {
                        "frame_id": fid,
                        "video": "video12",
                        "phase": None,
                        "annotations": {},
                    }
                frames_by_id[fid]["annotations"][row["key"]] = {
                    "key": row["key"],
                    "status": row.get("status"),
                    "present": int(row["present"]) if row.get("present") in ("0", "1") else 0,
                    "final_present": int(row["final_present"]) if row.get("final_present") in ("0", "1") else None,
                    "conf": float(row["conf"]) if row.get("conf") else None,
                    "trust": float(row["trust"]) if row.get("trust") else None,
                    "evidence": row.get("evidence", ""),
                }

    live_rows = []
    if annotations_path.exists():
        with open(annotations_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("video") == "video12":
                    live_rows.append(rec)

    sorted_frames = [
        dict(v, frame_idx=i)
        for i, (_, v) in enumerate(sorted(frames_by_id.items()))
    ]
    payload = {
        "video": "video12",
        "n_frames": len(sorted_frames) or 80,
        "labels": ["grasper", "l_hook", "gallbladder", "cystic_duct", "liver", "blood"],
        "frames": sorted_frames[:10],  # Mount with first 10 frames populated, remaining 70 pending
    }
    live_payload = {
        "annotations": live_rows[:12],
        "count": len(live_rows[:12]),
    }

    node_script = """
const Timeline = require(process.env.TIMELINE_JS_PATH);

function makeEl(tag) {
  return {
    tagName: (tag || 'DIV').toUpperCase(),
    className: '',
    id: '',
    innerHTML: '',
    textContent: '',
    style: {},
    attributes: {},
    children: [],
    parentNode: null,
    classList: {
      toggle() {},
      remove() {},
      add() {}
    },
    setAttribute(k, v) { this.attributes[k] = String(v); },
    getAttribute(k) { return this.attributes[k] || null; },
    appendChild(c) { c.parentNode = this; this.children.push(c); return c; },
    querySelector() { return null; },
    querySelectorAll() { return []; },
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() { return true; }
  };
}

global.document = {
  head: makeEl('HEAD'),
  getElementById() { return null; },
  createElement(tag) { return makeEl(tag); },
  querySelector() { return makeEl('DIV'); }
};

const container = makeEl('DIV');
const seekEvents = [];

// 1. Mount with null data -> everything must be pending across 80 frames
Timeline.mount(container, null, (idx, frame) => {
  seekEvents.push({ idx, frameId: frame.frame_id });
});
let st0 = Timeline.getState();
if (st0.nFrames !== 80) throw new Error('Expected 80 frames on empty mount, got ' + st0.nFrames);
if (st0.counts.pending !== 80 * 6) throw new Error('Expected all 480 cells pending on empty mount');
if (!container.children[0].innerHTML.includes('pending')) {
  throw new Error('Expected pending text in rendered HTML');
}

// 2. Update with partial real video12 payload (10 frames populated, 70 pending)
const payload = JSON.parse(process.env.TEST_PAYLOAD);
Timeline.update(payload);
let st1 = Timeline.getState();
if (payload.frames.length > 0) {
  if (st1.counts.pending !== (st1.nFrames - payload.frames.length) * 6) {
    throw new Error('Expected remaining frames to stay pending, got ' + JSON.stringify(st1.counts));
  }
}

// 3. setFrame(i) updates currentFrame without firing circular onSeek
Timeline.setFrame(5);
if (Timeline.getState().currentFrame !== 5) throw new Error('setFrame(5) failed');
if (seekEvents.length !== 0) throw new Error('setFrame must not trigger onSeek callback');

// 4. nextFrame() and seek(i) trigger onSeek
Timeline.nextFrame();
if (Timeline.getState().currentFrame !== 6) throw new Error('nextFrame() failed');
if (seekEvents.length !== 1 || seekEvents[0].idx !== 6) throw new Error('nextFrame did not call onSeek(6)');

// 5. Incremental live.json update populates phases & spatial indicators and marks fresh cells
const livePayload = JSON.parse(process.env.TEST_LIVE_PAYLOAD);
Timeline.update(livePayload);
let st2 = Timeline.getState();
if (livePayload.annotations.length > 0 && st2.counts.phasedFrames < 1) {
  throw new Error('Expected phasedFrames >= 1 after live update');
}
if (livePayload.annotations.length > 0 && !container.children[0].innerHTML.includes('tl-cell--fresh')) {
  throw new Error('Expected tl-cell--fresh animation class on newly arrived live cells');
}

Timeline.destroy();
console.log('TIMELINE_NODE_TEST_OK', JSON.stringify(st2.counts));
"""
    env = os.environ.copy()
    env["TIMELINE_JS_PATH"] = str(TIMELINE_JS)
    env["TEST_PAYLOAD"] = json.dumps(payload)
    env["TEST_LIVE_PAYLOAD"] = json.dumps(live_payload)

    res = subprocess.run(
        ["node", "-e", node_script],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert res.returncode == 0, f"Node test failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "TIMELINE_NODE_TEST_OK" in res.stdout
