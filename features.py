"""Step 2 (run in background as soon as index.csv exists): frozen DINOv2 features for the student.

  python features.py        # writes data/features.npy aligned with data/index.csv
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config as C

MODEL = "facebook/dinov2-small"


def main():
    df = pd.read_csv(C.DATA_DIR / "index.csv")
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
    np.save(C.DATA_DIR / "features.npy", np.concatenate(feats))
    print("\nsaved data/features.npy")


if __name__ == "__main__":
    main()
