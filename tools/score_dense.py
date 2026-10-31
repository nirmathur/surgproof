#!/usr/bin/env python3
"""Score dense (every-frame) AI masks against CholecSeg8k per-frame ground truth. Scoring only; runs after annotation.
usage: python3 tools/score_dense.py  -> out/dense_results.json"""
import glob, json, os
import numpy as np
from PIL import Image
R = os.path.expanduser("~/surgatlas")
WV = {"liver": 21, "grasper": 31, "l_hook": 32, "gallbladder": 22}
res = {}
for ann in glob.glob(f"{R}/out/dense/*/annotations.jsonl"):
    clip = os.path.basename(os.path.dirname(ann)); video = clip.split("_")[0]
    rows = [json.loads(l) for l in open(ann)]
    per = {l: [] for l in WV}; pres = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for r in rows:
        g = np.array(Image.open(f"{R}/raw/CholecSeg8k/{video}/{clip}/{r['frame']}_watershed_mask.png")); g = g[..., 0] if g.ndim == 3 else g
        gt = g == WV[r["label"]]; gt_present = gt.sum() > 0.002 * gt.size
        if r["present"] and gt_present: pres["tp"] += 1
        elif r["present"]: pres["fp"] += 1
        elif gt_present: pres["fn"] += 1
        else: pres["tn"] += 1
        if r["present"] and gt_present:
            m = np.array(Image.open(r["mask_path"])) > 0; u = (m | gt).sum()
            per[r["label"]].append(float((m & gt).sum() / u) if u else 0.0)
    allv = [v for l in per.values() for v in l]
    res[clip] = {"frames": len({r["frame"] for r in rows}), "n_mask_scored": len(allv),
                 "mean_iou": round(float(np.mean(allv)), 4) if allv else None,
                 "per_label_iou": {l: (round(float(np.mean(v)), 4), len(v)) for l, v in per.items() if v},
                 "presence_accuracy": round((pres["tp"] + pres["tn"]) / max(sum(pres.values()), 1), 4), "presence_counts": pres}
allm = [res[c]["mean_iou"] for c in res if res[c]["mean_iou"] is not None]
out = {"clips": res, "note": "every frame of each clip annotated by the AI (no human); scored afterwards vs CholecSeg8k per-frame masks"}
json.dump(out, open(f"{R}/out/dense_results.json", "w"), indent=1)
print(json.dumps({c: {k: v for k, v in d.items() if k != "presence_counts"} for c, d in res.items()}, indent=1))
