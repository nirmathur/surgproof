#!/usr/bin/env python3
"""Clean, temporally stable masks from the AI's own every-frame masks (no ground truth used).
Per label: per-pixel majority over a 5-frame window of the AI masks (+-2 frames), then morphological
open/close and a soft blur-threshold for smooth edges. Writes out/dense_smooth/<clip>/ with the same layout
as out/dense/<clip>/ (annotations.jsonl + PNGs), so score_dense/render_dense can run on it.
usage: python3 tools/smooth_dense.py video12_15750 video55_00508 video28_00000"""
import glob, json, os, sys
import cv2, numpy as np
R = os.path.expanduser("~/surgatlas")
for clip in sys.argv[1:]:
    src = f"{R}/out/dense/{clip}"; dst = f"{R}/out/dense_smooth/{clip}"; os.makedirs(dst, exist_ok=True)
    ann = [json.loads(l) for l in open(f"{src}/annotations.jsonl")]
    frames = sorted({a["frame"] for a in ann}, key=lambda f: int(f.split("_")[-1]))
    labels = sorted({a["label"] for a in ann})
    shape = None
    m = {}
    for a in ann:
        if a.get("present") and a.get("mask_path"):
            x = cv2.imread(a["mask_path"], 0) > 0; m[(a["frame"], a["label"])] = x; shape = x.shape
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    out = open(f"{dst}/annotations.jsonl", "w")
    for lab in labels:
        stack = [m.get((f, lab), np.zeros(shape, bool)) for f in frames]
        for i, f in enumerate(frames):
            win = stack[max(0, i - 2): i + 3]
            maj = (np.sum(win, axis=0) * 2 > len(win)).astype(np.uint8)
            maj = cv2.morphologyEx(cv2.morphologyEx(maj, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
            maj = cv2.GaussianBlur(maj.astype(np.float32), (0, 0), 3) > 0.5
            rec = {"frame": f, "label": lab, "present": bool(maj.any()),
                   "source": "AI masks, 5-frame temporal majority + morphological smoothing (no ground truth)"}
            if maj.any():
                p = f"{dst}/{f}_{lab}.png"; cv2.imwrite(p, maj.astype(np.uint8) * 255); rec["mask_path"] = p
            out.write(json.dumps(rec) + "\n")
    out.close(); print("smoothed", clip)
