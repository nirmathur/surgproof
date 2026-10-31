"""Step 2 (run in background as soon as index.csv exists): frozen DINOv2 features for the student.

  python features.py                                  # writes data/features.npy aligned with data/index.csv
  python features.py --external endoscapes2023        # writes data/external/endoscapes2023/features.npy
  python features.py --external cholect50             # writes data/external/cholect50/features.npy
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config as C

MODEL = "facebook/dinov2-small"


def extract_for_dir(target_dir):
    df = pd.read_csv(target_dir / "index.csv")
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(dev).eval()
    feats = []
    bs = 32
    for i in range(0, len(df), bs):
        imgs = [Image.open(p).convert("RGB") for p in df.path[i:i + bs]]
        with torch.no_grad():
            out = model(**proc(images=imgs, return_tensors="pt").to(dev))
        cls = out.last_hidden_state[:, 0]
        mean = out.last_hidden_state[:, 1:].mean(1)
        feats.append(torch.cat([cls, mean], 1).cpu().numpy())
        print(f"{min(i + bs, len(df))}/{len(df)}", end="\r")
    out_path = target_dir / "features.npy"
    np.save(out_path, np.concatenate(feats))
    print(f"\nsaved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", choices=["endoscapes2023", "cholect50"], default=None)
    a = ap.parse_args()
    target_dir = (C.DATA_DIR / "external" / a.external) if a.external else C.DATA_DIR
    extract_for_dir(target_dir)


if __name__ == "__main__":
    main()
