#!/usr/bin/env python3
"""Render a dense-annotated clip as smooth 25 fps video: real frames + the AI's own masks on every frame.
Only reads AI outputs (out/dense/<clip>) and real frames; the optional score line is read from out/dense_results.json.
usage: python3 tools/render_dense.py video12_15750 [...]  -> out/promo/dense_<clip>.mp4"""
import glob, json, os, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
R = os.path.expanduser("~/surgatlas"); W, H = 1920, 1080
COL = {"gallbladder": (64, 200, 140), "liver": (214, 120, 92), "grasper": (90, 170, 255), "l_hook": (250, 190, 70)}
NAME = {"gallbladder": "Gallbladder", "liver": "Liver", "grasper": "Grasper", "l_hook": "L-hook"}


def font(s):
    for f in ["/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try: return ImageFont.truetype(f, s)
        except Exception: pass
    return ImageFont.load_default()


def render(clip):
    video = clip.split("_")[0]
    ann = [json.loads(l) for l in open(f"{R}/out/{os.environ.get('DENSE','dense')}/{clip}/annotations.jsonl")]
    by = {}
    for a in ann: by.setdefault(a["frame"], []).append(a)
    frames = sorted(glob.glob(f"{R}/raw/CholecSeg8k/{video}/{clip}/*_endo.png"))
    score = None
    if os.path.exists(f"{R}/out/{os.environ.get('DENSE','dense')}_results.json"):
        d = json.load(open(f"{R}/out/{os.environ.get('DENSE','dense')}_results.json"))["clips"].get(clip)
        if d and d.get("mean_iou") is not None: score = d
    tmp = tempfile.mkdtemp()
    f_t, f_s, f_c = font(30), font(24), font(22)
    for i, fp in enumerate(frames):
        fr = os.path.basename(fp).replace("_endo.png", "")
        img = Image.open(fp).convert("RGBA")
        for a in by.get(fr, []):
            if not a.get("present") or not a.get("mask_path"): continue
            m = Image.open(a["mask_path"]).convert("L").filter(ImageFilter.GaussianBlur(1.2))
            c = COL[a["label"]]
            fill = Image.new("RGBA", img.size, c + (0,)); fill.putalpha(m.point(lambda v: int(v * 0.32)))
            img = Image.alpha_composite(img, fill)
            edge = m.filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.GaussianBlur(0.8))
            e = Image.new("RGBA", img.size, c + (255,)); e.putalpha(edge); img = Image.alpha_composite(img, e)
        img = img.convert("RGB")
        canvas = Image.new("RGB", (W, H), (8, 9, 11))
        fw = int(H * 0.86 * img.size[0] / img.size[1]); fh = int(H * 0.86)
        canvas.paste(img.resize((fw, fh), Image.LANCZOS), ((W - fw) // 2, int(H * 0.04)))
        d = ImageDraw.Draw(canvas)
        # labels for objects present on this frame (from the AI only), as a quiet legend
        x = (W - fw) // 2 + 24; y = int(H * 0.04) + 24
        for a in sorted(by.get(fr, []), key=lambda a: a["label"]):
            if a.get("present"):
                d.rounded_rectangle((x, y, x + 18, y + 18), 4, fill=COL[a["label"]]); d.text((x + 28, y - 4), NAME[a["label"]], font=f_c, fill=(236, 238, 241)); y += 34
        d.text((40, H - 70), "SurgProof", font=f_t, fill=(236, 238, 241))
        d.text((210, H - 64), f"{video} · held-out · every frame annotated by the AI · no human labels", font=f_s, fill=(140, 146, 156))
        if score:
            t = f"mask IoU vs human ground truth {score['mean_iou']:.2f} · n={score['n_mask_scored']} masks, scored afterwards"
            tw = d.textlength(t, font=f_s); d.text((W - tw - 40, H - 64), t, font=f_s, fill=(140, 146, 156))
        canvas.save(f"{tmp}/{i:04d}.png")
    out = f"{R}/out/promo/{os.environ.get('DENSE','dense')}_{clip}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "25", "-i", f"{tmp}/%04d.png",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out], check=True)
    print(out)


for c in sys.argv[1:]: render(c)
