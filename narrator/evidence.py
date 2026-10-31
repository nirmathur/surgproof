"""Evidence export for the UI: frames with label boxes drawn + crops, and an index per claim.

python -m narrator.evidence [--clips video12_15750,...]   (default: clips with grounded speak/hedge claims, test split)
-> narrator/out/evidence/<clip_id>/<frame>.jpg, <frame>_<label>.jpg, narrator/out/evidence/index.json
"""
from __future__ import annotations

import argparse
import json

import cv2
import pandas as pd

from narrator import cfg as C
from narrator import data as D

COLORS = {"grasper": (60, 180, 255), "l_hook": (255, 160, 40), "liver": (80, 80, 220), "gallbladder": (60, 200, 60),
          "cystic_duct": (220, 60, 220), "blood": (30, 30, 200)}


def _refs(c) -> list[tuple[str, int, str]]:
    refs = json.loads(c["evidence_refs"]) if isinstance(c["evidence_refs"], str) else c["evidence_refs"]
    out = []
    for r in refs or []:
        if r.startswith("lab:"):
            _, clip, fr, lab = r.split(":")
            out.append((clip, int(fr), lab))
    return out


def export(df: pd.DataFrame, labels: pd.DataFrame, events: dict, clips=None, max_frames_per_claim=3) -> dict:
    base = C.out_dir("evidence")
    idx = {}
    g = df[df.verdict.isin(["speak", "hedge"]) | (df["mode"] == "free")]
    if clips:
        g = g[g.clip_id.isin(clips)]
    drawn = set()
    for c in g.to_dict("records"):
        refs = _refs(c)
        if not refs:  # event refs -> their label keys
            evr = json.loads(c["evidence_refs"]) if isinstance(c["evidence_refs"], str) else c["evidence_refs"]
            for e in events.get(c["clip_id"], []):
                if e["event_id"] in (evr or []):
                    refs += [(e["clip_id"], int(k.split(":")[2]), e["label"]) for k in e["label_keys"][:2]]
        items = []
        for clip, fr, lab in refs[:max_frames_per_claim]:
            p = D.frame_path(clip, fr)
            if not p.exists():
                continue
            d = base / clip
            d.mkdir(exist_ok=True)
            full = d / f"{fr}.jpg"
            rows = labels[(labels.clip_id == clip) & (labels.frame_idx == fr) & labels.present]
            if (clip, fr) not in drawn:
                im = cv2.imread(str(p))
                for r in rows.itertuples():
                    for x1, y1, x2, y2 in json.loads(r.boxes):
                        cv2.rectangle(im, (x1, y1), (x2, y2), COLORS.get(r.label, (255, 255, 255)), 2)
                        cv2.putText(im, f"{r.label} ({r.status})", (x1 + 3, max(14, y1 + 14)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS.get(r.label, (255, 255, 255)), 1)
                cv2.imwrite(str(full), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
                drawn.add((clip, fr))
            box = None
            r = rows[rows.label == lab]
            if len(r):
                bxs = json.loads(r.iloc[0].boxes)
                if bxs:
                    box = max(bxs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    im = cv2.imread(str(p))
                    x1, y1, x2, y2 = box
                    crop = im[max(0, y1 - 20):y2 + 20, max(0, x1 - 20):x2 + 20]
                    if crop.size:
                        cv2.imwrite(str(d / f"{fr}_{lab}.jpg"), crop)
            items.append({"frame_idx": fr, "path": str(full.relative_to(C.ROOT)), "label": lab, "box": box,
                          "crop": str((d / f"{fr}_{lab}.jpg").relative_to(C.ROOT)) if box else None})
        idx[c["claim_id"]] = items
    (base / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


if __name__ == "__main__":
    from narrator import events as E
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default=None)
    a = ap.parse_args()
    df = pd.read_parquet(C.out_dir() / "claims.parquet")
    df = df[df.split == "test"]
    idx = export(df, D.load_labels(), E.build(), a.clips.split(",") if a.clips else None)
    print(f"evidence for {len(idx)} claims, {sum(len(v) for v in idx.values())} frames")
