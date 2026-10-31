#!/usr/bin/env python3
"""AI-drawn masks: Gemini traces a polygon outline for every object the verifier accepted as present.

No human and no ground truth anywhere in this file. Each object is drawn twice (two independent
passes with different phrasing and image scale). If the two masks agree (IoU >= 0.5) the mask is
accepted (pixel-wise intersection of the two); otherwise it goes AMBER and is re-drawn on a tighter
crop for up to 2 more rounds. Every call's raw response is logged for provenance.

usage: python3 tools/polygon_worker.py --video video12 [--workers 8]
writes: out/polygons/<video>.jsonl, out/masks/<frame_id>_<label>.png, out/polygons/raw_<video>.jsonl
"""
import argparse, base64, concurrent.futures as cf, csv, io, json, os, threading, time, urllib.request
import numpy as np
from PIL import Image, ImageDraw

R = os.path.expanduser("~/surgatlas")
KEY = [l.split("=", 1)[1].strip() for l in open(f"{R}/.env") if l.startswith("GEMINI_API_KEY=")][0]
MODEL = "gemini-3.8-flash"
NAMES = {"grasper": "the laparoscopic grasper instrument (shaft and jaws)",
         "l_hook": "the L-hook electrocautery instrument (shaft and hook tip)",
         "liver": "the liver", "gallbladder": "the gallbladder",
         "cystic_duct": "the cystic duct", "blood": "the free blood or blood clot"}
lock = threading.Lock()


def call(img_bytes, prompt, timeout=120):
    body = {"contents": [{"parts": [{"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()}},
                                    {"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "thinkingConfig": {"thinkingBudget": 0}}}
    req = urllib.request.Request(f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
                                 data=json.dumps(body).encode(), headers={"x-goog-api-key": KEY, "Content-Type": "application/json"})
    for attempt in range(3):
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=timeout))
            return resp["candidates"][0]["content"]["parts"][-1]["text"]
        except Exception as e:
            err = str(e)
            time.sleep(2 + 3 * attempt)
    raise RuntimeError(err)


def jpeg(im, width):
    im = im.resize((width, int(width * im.size[1] / im.size[0])))
    b = io.BytesIO(); im.save(b, "JPEG", quality=88); return b.getvalue()


def poly_mask(points, box, size):
    """points in 0-1000 of the image region `box`=(x0,y0,x1,y1) in full-image pixels."""
    W, H = size; x0, y0, x1, y1 = box
    m = Image.new("L", (W, H), 0)
    pts = [(x0 + p[1] / 1000 * (x1 - x0), y0 + p[0] / 1000 * (y1 - y0)) for p in points if len(p) == 2]
    if len(pts) > 2:
        ImageDraw.Draw(m).polygon(pts, fill=1)
    return np.array(m) > 0, pts


def draw(im, label, region, phrasing):
    x0, y0, x1, y1 = region
    crop = im.crop(region)
    width = 640 if phrasing == 0 else 896
    if phrasing == 0:
        prompt = (f"Outline {NAMES[label]} in this laparoscopic image as a polygon. Return JSON "
                  f'{{"polygon":[[y,x],...]}} with 12-40 points tracing its visible boundary, coordinates normalised 0-1000. '
                  f'If it is not visible return {{"polygon":[]}}.')
    else:
        prompt = (f"Trace the visible outline of {NAMES[label]} in this surgical video frame. Respond as JSON "
                  f'{{"polygon":[[y,x],...]}}: 12-40 boundary points in order, each [y,x] normalised to 0-1000. '
                  f'Empty list if absent.')
    raw = call(jpeg(crop, width), prompt)
    pts = json.loads(raw).get("polygon", [])
    mask, pix = poly_mask(pts, region, im.size)
    return mask, pix, raw


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def job(fr, label, frame_path, video):
    im = Image.open(frame_path).convert("RGB"); W, H = im.size
    region = (0, 0, W, H); rounds = []
    status, final = "AMBER", None
    for rnd in range(3):
        m1, p1, raw1 = draw(im, label, region, 0)
        m2, p2, raw2 = draw(im, label, region, 1)
        s = iou(m1, m2)
        rounds.append({"round": rnd, "region": region, "iou_pass1_pass2": round(s, 4), "px1": int(m1.sum()), "px2": int(m2.sum())})
        with lock, open(f"{R}/out/polygons/raw_{video}.jsonl", "a") as f:
            f.write(json.dumps({"frame_id": fr, "label": label, "round": rnd, "region": region, "raw_pass1": raw1, "raw_pass2": raw2}) + "\n")
        if s >= 0.5 and m1.sum() > 0:
            status, final = "accept", (m1 & m2) if (m1 & m2).sum() else m1
            poly = p1
            break
        u = m1 | m2
        if not u.any():
            status = "absent_by_both"; break
        ys, xs = np.where(u); pad = 40
        region = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(W, xs.max() + pad), min(H, ys.max() + pad))
    rec = {"frame_id": fr, "video": video, "label": label, "status": status, "rounds": rounds,
           "mask_source": f"{MODEL} polygon, 2 independent passes, IoU>=0.5", "ts": time.time()}
    if final is not None:
        path = f"{R}/out/masks/{fr}_{label}.png"
        Image.fromarray((final * 255).astype("uint8")).save(path)
        rec["mask_path"] = path; rec["polygon_px"] = [[round(x, 1), round(y, 1)] for x, y in poly]
    with lock, open(f"{R}/out/polygons/{video}.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    with lock, open(f"{R}/out/annotations_live.jsonl", "a") as f:
        f.write(json.dumps({**{k: rec[k] for k in ("frame_id", "video", "label", "status", "ts")}, "kind": "mask", "mask_path": rec.get("mask_path")}) + "\n")
    return status


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(f"{R}/out/polygons", exist_ok=True); os.makedirs(f"{R}/out/masks", exist_ok=True)
    done = set()
    if os.path.exists(f"{R}/out/polygons/{a.video}.jsonl"):
        done = {(j["frame_id"], j["label"]) for j in map(json.loads, open(f"{R}/out/polygons/{a.video}.jsonl"))}
    jobs = []
    for r in csv.DictReader(open(f"{R}/out/decisions.csv")):
        if r["video"] != a.video or r["key"] not in NAMES or r["status"] != "accept":
            continue
        if str(r["final_present"]).strip() not in ("1", "1.0", "True"):
            continue
        if (r["frame_id"], r["key"]) in done:
            continue
        fp = r["frame_path"] if os.path.isabs(r["frame_path"]) else os.path.join(R, r["frame_path"])
        jobs.append((r["frame_id"], r["key"], fp, a.video))
    print(f"{a.video}: {len(jobs)} objects to draw", flush=True)
    n = 0
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for fut in cf.as_completed([ex.submit(job, *j) for j in jobs]):
            n += 1
            try:
                st = fut.result()
            except Exception as e:
                st = f"error {str(e)[:80]}"
            if n % 10 == 0 or n == len(jobs):
                print(f"{a.video}: {n}/{len(jobs)} last={st}", flush=True)


if __name__ == "__main__":
    main()
