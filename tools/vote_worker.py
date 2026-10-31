#!/usr/bin/env python3
"""Production masks by 3-vote majority (standard ensemble rule, not tuned on test).
Three independent Gemini polygon drafts per accepted object (different phrasing/scale); final mask = pixels in >=2 drafts.
No ground truth. writes out/masks_vote3/<frame>_<label>.png and out/polygons/vote3_<video>.jsonl
usage: python3 tools/vote_worker.py --video video55
"""
import argparse, concurrent.futures as cf, csv, json, os, sys, threading, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from polygon_worker import NAMES, R
from vote_experiment import draw, VARIANTS, iou
lock = threading.Lock()

def job(fr, label, fp, video):
    im = Image.open(fp).convert("RGB")
    ms = [draw(im, label, ph, w) for ph, w in VARIANTS[:3]]
    final = np.stack(ms).astype(int).sum(0) >= 2
    agree = float(np.mean([iou(ms[i], ms[j]) for i, j in ((0, 1), (0, 2), (1, 2))]))
    status = "accept" if agree >= 0.5 and final.any() else "AMBER"
    os.makedirs(f"{R}/out/masks_vote3", exist_ok=True)
    path = f"{R}/out/masks_vote3/{fr}_{label}.png"
    Image.fromarray((final * 255).astype("uint8")).save(path)
    rec = {"frame_id": fr, "video": video, "label": label, "status": status, "mean_pairwise_iou": round(agree, 4),
           "mask_path": path, "mask_source": "gemini-3.8-flash polygons, 3 drafts, majority vote", "ts": time.time()}
    with lock, open(f"{R}/out/polygons/vote3_{video}.jsonl", "a") as f: f.write(json.dumps(rec) + "\n")
    with lock, open(f"{R}/out/annotations_live.jsonl", "a") as f: f.write(json.dumps({**rec, "kind": "mask_vote3"}) + "\n")
    return status

ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--workers", type=int, default=8); a = ap.parse_args()
done = set()
fo = f"{R}/out/polygons/vote3_{a.video}.jsonl"
if os.path.exists(fo): done = {(j["frame_id"], j["label"]) for j in map(json.loads, open(fo))}
jobs = []
for r in csv.DictReader(open(f"{R}/out/decisions.csv")):
    if r["video"] == a.video and r["key"] in NAMES and r["status"] == "accept" and str(r["final_present"]).strip() in ("1", "1.0", "True") and (r["frame_id"], r["key"]) not in done:
        fp = r["frame_path"]; jobs.append((r["frame_id"], r["key"], fp if os.path.isabs(fp) else os.path.join(R, fp), a.video))
print(f"{a.video}: {len(jobs)} objects", flush=True)
with cf.ThreadPoolExecutor(a.workers) as ex:
    for n, fut in enumerate(cf.as_completed([ex.submit(job, *j) for j in jobs]), 1):
        try: st = fut.result()
        except Exception as e: st = f"error {str(e)[:60]}"
        if n % 20 == 0 or n == len(jobs): print(f"{a.video}: {n}/{len(jobs)} {st}", flush=True)
