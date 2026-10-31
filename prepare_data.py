"""Step 1: extract CholecSeg8k, derive presence labels from masks, sample frames, split by video.

Usage:
  python prepare_data.py --inspect      # print mask values for 3 frames, sanity check mapping
  python prepare_data.py                # build data/index.csv
"""
import argparse
import random
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import config as C


def find_root():
    """Return the folder containing videoXX dirs, extracting the zip if needed."""
    hits = list(C.RAW_DIR.rglob("video01"))
    if hits:
        return hits[0].parent
    zips = list(C.RAW_DIR.rglob("*.zip"))
    if not zips:
        raise SystemExit(f"No zip or extracted data under {C.RAW_DIR}. Download first.")
    print(f"Extracting {zips[0]} ... (a few minutes)")
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(C.RAW_DIR)
    return list(C.RAW_DIR.rglob("video01"))[0].parent


def mask_to_classes(mask_path):
    m = np.array(Image.open(mask_path))
    if m.ndim == 3:
        m = m[..., 0]
    vals = np.unique(m)
    # auto-detect convention
    if set(vals.tolist()) <= set(C.WATERSHED_VALUE_TO_CLASS):
        lut = np.zeros(256, dtype=np.int16) - 1
        for v, c in C.WATERSHED_VALUE_TO_CLASS.items():
            lut[v] = c
        cls = lut[m]
    else:
        cls = m.astype(np.int16)  # assume value == class id (paper convention)
    return cls


def presence(cls_map):
    total = cls_map.size
    out = {}
    for key, (cid, _) in C.LABELS.items():
        frac = float((cls_map == cid).sum()) / total
        out[f"gt_{key}"] = int(frac >= C.MIN_FRAC.get(key, C.MIN_FRAC["default"]))
        out[f"frac_{key}"] = round(frac, 5)
    return out


def inspect(root):
    frames = sorted(root.rglob("*_endo.png"))[:3]
    for f in frames:
        wm = f.with_name(f.name.replace("_endo.png", "_endo_watershed_mask.png"))
        m = np.array(Image.open(wm))
        m = m[..., 0] if m.ndim == 3 else m
        print(f.name, "watershed values:", Counter(m.flatten().tolist()).most_common(15))
    print("If values look like 50/11/21/... the default mapping applies. If 0..12, paper convention.")


def build(root):
    rows = []
    for clip_dir in sorted(p for p in root.glob("video*/video*_*") if p.is_dir()):
        video = clip_dir.parent.name
        frames = sorted(clip_dir.glob("*_endo.png"),
                        key=lambda p: int(p.name.split("_")[1]))
        for i, f in enumerate(frames):
            if i % C.FRAME_STRIDE:
                continue
            wm = f.with_name(f.name.replace("_endo.png", "_endo_watershed_mask.png"))
            cls = mask_to_classes(wm)
            rows.append({"frame_id": f"{clip_dir.name}_{f.name.split('_')[1]}",
                         "video": video, "clip": clip_dir.name,
                         "path": str(f), "mask_path": str(wm), **presence(cls)})
    df = pd.DataFrame(rows)

    videos = sorted(df.video.unique())
    rng = random.Random(C.SEED)
    rng.shuffle(videos)
    test = set(videos[:C.N_TEST_VIDEOS])
    calib = set(videos[C.N_TEST_VIDEOS:C.N_TEST_VIDEOS + C.N_CALIB_VIDEOS])
    df["split"] = df.video.map(lambda v: "test" if v in test else "calib" if v in calib else "train")
    df.to_csv(C.DATA_DIR / "index.csv", index=False)

    print(f"{len(df)} frames, {df.video.nunique()} videos")
    print(df.groupby("split").size().to_string())
    print("\nPrevalence (fraction of frames where present):")
    print(df[[f"gt_{k}" for k in C.LABELS]].mean().round(3).to_string())
    print("\nIf a label is ~1.0 or ~0.0 it will not show abstention well; swap it in config.LABELS.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    r = find_root()
    inspect(r) if a.inspect else build(r)
