#!/usr/bin/env python3
"""Snap the AI's masks to real image edges with GrabCut, seeded only by the AI's own (temporally smoothed) mask.
No ground truth. Eroded AI mask = sure foreground, outside the dilated AI mask = sure background, the band between
= let GrabCut decide from colour/edges. Then a soft blur-threshold for clean contours.
usage: python3 tools/refine_edges.py video12_15750 ...  -> out/dense_clean/<clip>/ (same layout as out/dense/<clip>/)"""
import concurrent.futures as cf, glob, json, os, sys
import cv2, numpy as np
R = os.path.expanduser("~/surgatlas")


def refine(img, m):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.erode(m, k, iterations=1); band = cv2.dilate(m, k, iterations=2)
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[band > 0] = cv2.GC_PR_BGD; gc[m > 0] = cv2.GC_PR_FGD; gc[fg > 0] = cv2.GC_FGD
    if (gc == cv2.GC_FGD).sum() < 50: return m > 0
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(img, gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m > 0
    out = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out)
    if n > 1: out = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)  # keep the main object
    return cv2.GaussianBlur(out.astype(np.float32), (0, 0), 2.5) > 0.5


def job(a, video, clip, dst):
    rec = dict(a); rec["source"] = a.get("source", "") + " + GrabCut edge refinement seeded by the AI mask"
    if a.get("present") and a.get("mask_path"):
        img = cv2.imread(f"{R}/raw/CholecSeg8k/{video}/{clip}/{a['frame']}_endo.png")
        m = (cv2.imread(a["mask_path"], 0) > 0).astype(np.uint8)
        r = refine(img, m)
        p = f"{dst}/{a['frame']}_{a['label']}.png"; cv2.imwrite(p, r.astype(np.uint8) * 255); rec["mask_path"] = p
        rec["present"] = bool(r.any())
    return rec


for clip in sys.argv[1:]:
    video = clip.split("_")[0]; src = f"{R}/out/dense_smooth/{clip}"; dst = f"{R}/out/dense_clean/{clip}"
    os.makedirs(dst, exist_ok=True)
    ann = [json.loads(l) for l in open(f"{src}/annotations.jsonl")]
    with cf.ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(job, ann, [video] * len(ann), [clip] * len(ann), [dst] * len(ann)))
    with open(f"{dst}/annotations.jsonl", "w") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    print("refined", clip, flush=True)
