#!/usr/bin/env python3
"""Dense AI annotation: EVERY frame of a held-out clip (25 fps), so the demo is real smooth video.

For each frame and each structure, Gemini draws 3 independent polygon drafts (different phrasing/scale);
a pixel is in the final mask if >=2 drafts include it; the object is 'absent' if >=2 drafts return empty.
No ground truth here. Scoring vs the per-frame CholecSeg8k masks is done afterwards by tools/score_dense.py.

usage: python3 tools/dense_worker.py --clip raw/CholecSeg8k/video12/video12_15750 [--workers 24]
writes: out/dense/<clip>/<frame>_<label>.png, out/dense/<clip>/annotations.jsonl
"""
import argparse, concurrent.futures as cf, glob, json, os, sys, threading, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from polygon_worker import R
from vote_experiment import draw, VARIANTS

LABELS = ["gallbladder", "liver", "grasper", "l_hook"]
lock = threading.Lock()


def job(frame_path, label, outdir):
    fr = os.path.basename(frame_path).replace("_endo.png", "")
    im = Image.open(frame_path).convert("RGB")
    ms = [draw(im, label, ph, w) for ph, w in VARIANTS[:3]]
    present_votes = sum(int(m.any()) for m in ms)
    final = np.stack(ms).astype(int).sum(0) >= 2
    rec = {"frame": fr, "label": label, "present_votes": present_votes, "present": present_votes >= 2 and bool(final.any()),
           "source": "gemini-3.8-flash polygons, 3 drafts, majority", "ts": time.time()}
    if rec["present"]:
        p = f"{outdir}/{fr}_{label}.png"; Image.fromarray((final * 255).astype("uint8")).save(p); rec["mask_path"] = p
    with lock, open(f"{outdir}/annotations.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec["present"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--clip", required=True); ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()
    clip = a.clip.rstrip("/"); name = os.path.basename(clip)
    outdir = f"{R}/out/dense/{name}"; os.makedirs(outdir, exist_ok=True)
    done = set()
    if os.path.exists(f"{outdir}/annotations.jsonl"):
        done = {(j["frame"], j["label"]) for j in map(json.loads, open(f"{outdir}/annotations.jsonl"))}
    frames = sorted(glob.glob(f"{R}/{clip}/*_endo.png") if not clip.startswith("/") else glob.glob(f"{clip}/*_endo.png"))
    jobs = [(f, l, outdir) for f in frames for l in LABELS
            if (os.path.basename(f).replace("_endo.png", ""), l) not in done]
    print(f"{name}: {len(frames)} frames, {len(jobs)} jobs", flush=True)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, fut in enumerate(cf.as_completed([ex.submit(job, *j) for j in jobs]), 1):
            try: fut.result()
            except Exception as e: print("err", str(e)[:80], flush=True)
            if n % 40 == 0 or n == len(jobs): print(f"{name}: {n}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
