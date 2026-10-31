#!/usr/bin/env python3
"""Quick keyframe test: annotate only every s-th frame with the AI, propagate masks forward with optical flow,
score every frame vs per-frame ground truth (scoring only). usage: python3 tools/keyframe_quick.py video28_00000"""
import glob, json, os, sys
import cv2, numpy as np
R = os.path.expanduser("~/surgatlas"); WV = {"liver": 21, "grasper": 31, "l_hook": 32, "gallbladder": 22}
clip = sys.argv[1]; video = clip.split("_")[0]
frames = sorted(glob.glob(f"{R}/raw/CholecSeg8k/{video}/{clip}/*_endo.png"))
names = [os.path.basename(f).replace("_endo.png", "") for f in frames]
ann = {}
for l in open(f"{R}/out/dense/{clip}/annotations.jsonl"):
    a = json.loads(l); ann[(a["frame"], a["label"])] = a
gray = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY) for f in frames]
H, W = gray[0].shape
def ai_mask(i, lab):
    a = ann.get((names[i], lab))
    return (cv2.imread(a["mask_path"], 0) > 0) if a and a.get("present") and a.get("mask_path") else np.zeros((H, W), bool)
def gt(i, lab):
    g = cv2.imread(f"{R}/raw/CholecSeg8k/{video}/{clip}/{names[i]}_endo_watershed_mask.png", 0); return g == WV[lab]
def iou(a, b):
    u = (a | b).sum(); return float((a & b).sum() / u) if u else None
gx, gy = np.meshgrid(np.arange(W), np.arange(H))
res = {}
for s in (1, 5, 10):
    scores = []
    for lab in WV:
        cur = None
        for i in range(len(frames)):
            if i % s == 0: cur = ai_mask(i, lab).astype(np.float32)
            else:
                flow = cv2.calcOpticalFlowFarneback(gray[i], gray[i - 1], None, 0.5, 3, 21, 3, 5, 1.1, 0)
                cur = cv2.remap(cur, (gx + flow[..., 0]).astype(np.float32), (gy + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR)
            g = gt(i, lab)
            if g.sum() > 0.002 * g.size:
                v = iou(cur > 0.5, g)
                if v is not None: scores.append(v)
    res[s] = {"mean_iou_all_frames": round(float(np.mean(scores)), 4), "n_frame_masks": len(scores),
              "ai_calls_per_second_of_video": round(12 * 25 / s, 1)}
out = {"clip": clip, "method": "keyframes every s-th frame (AI, 3-draft majority), forward optical-flow propagation between; every frame scored vs GT afterwards", "results": res}
json.dump(out, open(f"{R}/out/keyframe_quick_{clip}.json", "w"), indent=1); print(json.dumps(out["results"], indent=1))
