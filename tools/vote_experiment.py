#!/usr/bin/env python3
"""Do more independent drawing passes improve AI masks? Ground truth used ONLY for scoring at the end.

For a sample of accepted objects on held-out videos, draw K=5 independent polygons (different phrasing and
image scale), then compare against the CholecSeg8k ground-truth mask:
  single pass, 2-pass intersection (current), 3-vote majority, 5-vote majority.
writes out/vote_experiment.json
"""
import concurrent.futures as cf, csv, json, os, random, sys
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from polygon_worker import NAMES, R, call, poly_mask, jpeg

WV = {"liver": 21, "grasper": 31, "blood": 24, "cystic_duct": 25, "l_hook": 32, "gallbladder": 22}
VARIANTS = [(0, 512), (1, 640), (0, 768), (1, 896), (0, 1024)]
PROMPTS = [
    lambda n: f'Outline {n} in this laparoscopic image as a polygon. Return JSON {{"polygon":[[y,x],...]}} with 12-40 points tracing its visible boundary, coordinates normalised 0-1000. If it is not visible return {{"polygon":[]}}.',
    lambda n: f'Trace the visible outline of {n} in this surgical video frame. Respond as JSON {{"polygon":[[y,x],...]}}: 12-40 boundary points in order, each [y,x] normalised to 0-1000. Empty list if absent.',
]


def draw(im, label, ph, width):
    raw = call(jpeg(im, width), PROMPTS[ph](NAMES[label]))
    pts = json.loads(raw).get("polygon", [])
    m, _ = poly_mask(pts, (0, 0) + im.size, im.size)
    return m


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def one(rec, idx):
    im = Image.open(rec["frame_path"]).convert("RGB")
    masks = [draw(im, rec["label"], ph, w) for ph, w in VARIANTS]
    r = idx[rec["frame_id"]]
    mp = r["mask_path"] if os.path.isabs(r["mask_path"]) else os.path.join(R, r["mask_path"])
    g = np.array(Image.open(mp)); g = g[..., 0] if g.ndim == 3 else g
    gt = g == WV[rec["label"]]
    stack = np.stack(masks).astype(int)
    return {"frame_id": rec["frame_id"], "label": rec["label"],
            "single": iou(masks[0], gt),
            "two_pass_intersection": iou(masks[0] & masks[1], gt),
            "vote3": iou(stack[:3].sum(0) >= 2, gt),
            "vote5": iou(stack.sum(0) >= 3, gt),
            "union5": iou(stack.sum(0) >= 1, gt)}


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    idx = {r["frame_id"]: r for r in csv.DictReader(open(f"{R}/data/index.csv"))}
    paths = {r["frame_id"]: r["frame_path"] for r in csv.DictReader(open(f"{R}/out/decisions.csv"))}
    recs = []
    split = sys.argv[2] if len(sys.argv) > 2 else "test"
    for r in csv.DictReader(open(f"{R}/out/decisions.csv")):
        if r["split"] == split and r["status"] == "accept" and r["key"] in WV and str(r["final_present"]).strip() in ("1", "1.0", "True"):
            fp = r["frame_path"]; recs.append({"frame_id": r["frame_id"], "label": r["key"], "frame_path": fp if os.path.isabs(fp) else os.path.join(R, fp)})
    recs = random.Random(0).sample(recs, min(n, len(recs)))
    rows = []
    with cf.ThreadPoolExecutor(12) as ex:
        for fut in cf.as_completed([ex.submit(one, r, idx) for r in recs]):
            try: rows.append(fut.result())
            except Exception as e: print("err", str(e)[:80], flush=True)
    keys = ["single", "two_pass_intersection", "vote3", "vote5", "union5"]
    summ = {k: {"mean_iou": round(float(np.mean([r[k] for r in rows])), 4),
                "ge_0_5": round(float(np.mean([r[k] >= 0.5 for r in rows])), 3)} for k in keys}
    out = {"n_objects": len(rows), "split": split, "summary": summ, "rows": rows}
    json.dump(out, open(f"{R}/out/vote_experiment_{split}.json", "w"), indent=1)
    print(json.dumps({"n": len(rows), **summ}, indent=1))


if __name__ == "__main__":
    main()
