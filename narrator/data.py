"""Dataset access (read-only), ground-truth presence from masks, and the mock label table.

python -m narrator.data            # builds out/gt_presence.parquet and out/mock_labels.parquet

Ground truth: CholecSeg8k watershed mask channel 0, value -> class via the tracker's config.yaml
mapping (verified 1:1 by the tracker lane), 255 = 1-px border, ignored.
GT is used ONLY to (a) score test videos and (b) build the clearly-labelled mock label table
until the real labeller output exists.
"""
from __future__ import annotations

import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from narrator import cfg as C

VALUE_TO_CLASS = {50: "background", 11: "abdominal_wall", 21: "liver", 13: "gi_tract", 12: "fat",
                  31: "grasper", 23: "connective_tissue", 24: "blood", 25: "cystic_duct", 32: "l_hook",
                  22: "gallbladder", 33: "hepatic_vein", 5: "liver_ligament"}
CLASSES = [c for c in VALUE_TO_CLASS.values() if c != "background"]
LABEL_KEY = ["video_id", "clip_id", "frame_idx", "label"]


def data_root() -> Path:
    return C.path("data_root")


def list_clips() -> pd.DataFrame:
    rows = []
    for vdir in sorted(data_root().glob("video*")):
        for cdir in sorted(vdir.glob(f"{vdir.name}_*")):
            frames = sorted(int(p.name.split("_")[1]) for p in cdir.glob("frame_*_endo.png"))
            if frames:
                rows.append({"video_id": vdir.name, "clip_id": cdir.name, "first": frames[0],
                             "last": frames[-1], "n_frames": len(frames), "split": C.split_of(vdir.name)})
    return pd.DataFrame(rows)


def frame_path(clip_id: str, frame_idx: int, kind: str = "endo") -> Path:
    vid = clip_id.split("_")[0]
    suffix = {"endo": "_endo.png", "watershed": "_endo_watershed_mask.png"}[kind]
    return data_root() / vid / clip_id / f"frame_{frame_idx}{suffix}"


def sampled_frames(first: int, n: int, stride: int | None = None) -> list[int]:
    stride = stride or C.cfg()["events"]["frame_stride"]
    return list(range(first, first + n, stride))


def _frame_gt(args) -> list[dict]:
    """Per frame: fraction and connected-component boxes of every class."""
    import cv2
    video_id, clip_id, f = args
    m = cv2.imread(str(frame_path(clip_id, f, "watershed")), cv2.IMREAD_UNCHANGED)
    if m is None:
        return []
    if m.ndim == 3:
        m = m[..., 0]
    total = m.size
    min_cc = C.cfg()["presence"]["min_cc_frac"] * total
    out = []
    for val, cls in VALUE_TO_CLASS.items():
        if cls == "background":
            continue
        mask = (m == val).astype(np.uint8)
        frac = float(mask.sum()) / total
        boxes = []
        if frac > 0:
            n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                x, y, w, h, a = stats[i]
                if a >= min_cc:
                    boxes.append([int(x), int(y), int(x + w), int(y + h)])
        out.append({"video_id": video_id, "clip_id": clip_id, "frame_idx": f, "label": cls,
                    "frac": frac, "present": frac > C.cfg()["presence"]["min_frac"],
                    "boxes": json.dumps(boxes)})
    return out


def build_gt(force: bool = False, stride: int | None = None) -> pd.DataFrame:
    p = C.out_dir() / "gt_presence.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    clips = list_clips()
    jobs = [(r.video_id, r.clip_id, f) for r in clips.itertuples()
            for f in sampled_frames(r.first, r.n_frames, stride)]
    with ThreadPoolExecutor(8) as ex:
        rows = [x for res in ex.map(_frame_gt, jobs) for x in res]
    df = pd.DataFrame(rows)
    df["split"] = df.video_id.map(C.split_of)
    df.to_parquet(p, index=False)
    return df


SWAPS = {"grasper": "l_hook", "l_hook": "grasper", "liver": "gallbladder", "gallbladder": "liver",
         "cystic_duct": "connective_tissue", "blood": "connective_tissue"}


def build_mock_labels(gt: pd.DataFrame | None = None, force: bool = False) -> pd.DataFrame:
    """Labeller-shaped table derived from GT plus seeded noise. source='mock' on every row.

    Columns: video_id, clip_id, frame_idx, label, status (accepted|uncertain|rejected),
    confidence, boxes (json list of [x1,y1,x2,y2] px), present (bool, labeller's claim),
    noise (none|drop|spurious|swap|uncertain), source.
    """
    p = C.out_dir() / "mock_labels.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    gt = build_gt() if gt is None else gt
    mc = C.cfg()["mock"]
    rng = random.Random(mc["seed"])
    scored = C.cfg()["scored_classes"]
    rows = []
    for (vid, clip, f), g in gt[gt.label.isin(scored)].groupby(["video_id", "clip_id", "frame_idx"]):
        present = {r.label: r for r in g.itertuples() if r.present}
        for lab in scored:
            r = present.get(lab)
            u = rng.random()
            row = None
            if r is not None:
                if u < mc["p_drop"]:
                    row = dict(label=lab, present=False, status="rejected", confidence=0.4, boxes="[]", noise="drop")
                elif u < mc["p_drop"] + mc["p_swap"] and SWAPS[lab] in scored:
                    row = dict(label=SWAPS[lab], present=True, status="accepted", confidence=0.8,
                               boxes=r.boxes, noise="swap")
                elif u < mc["p_drop"] + mc["p_swap"] + mc["p_uncertain"]:
                    row = dict(label=lab, present=True, status="uncertain", confidence=0.6, boxes=r.boxes,
                               noise="uncertain")
                else:
                    row = dict(label=lab, present=True, status="accepted",
                               confidence=round(0.85 + 0.15 * rng.random(), 3), boxes=r.boxes, noise="none")
            elif u < mc["p_spurious"]:
                x, y = rng.randint(50, 700), rng.randint(50, 380)
                row = dict(label=lab, present=True, status=rng.choice(["accepted", "uncertain"]),
                           confidence=0.7, boxes=json.dumps([[x, y, x + 100, y + 80]]), noise="spurious")
            if row is not None:
                row.update(video_id=vid, clip_id=clip, frame_idx=int(f))
                rows.append(row)
    df = pd.DataFrame(rows)
    df["frame_idx"] = df.frame_idx.astype(int)
    # a swap can create a duplicate (frame, label); keep the first
    df = df.drop_duplicates(LABEL_KEY, keep="first")
    df["source"] = "mock"
    df["split"] = df.video_id.map(C.split_of)
    df.to_parquet(p, index=False)
    return df


def load_labels() -> pd.DataFrame:
    """Real labeller table if one exists (out/labels_*.jsonl), else the mock table."""
    real = sorted(C.ROOT.glob(C.cfg()["paths"]["labels_glob"]))
    if (C.ROOT / "out" / "decisions.csv").exists() or (C.ROOT / "out" / "labels_manual.jsonl").exists():
        from narrator import adapters  # noqa: F401  (written when the real schema is known)
        return adapters.load_real_labels(real)
    return build_mock_labels()


if __name__ == "__main__":
    gt = build_gt()
    print("gt rows", len(gt), "frames", gt.groupby(["clip_id", "frame_idx"]).ngroups)
    ml = build_mock_labels(gt)
    print("mock labels", len(ml), ml.noise.value_counts().to_dict())


_CLIPS_CACHE: dict | None = None


def list_clips_cached() -> dict:
    """clip_id -> (first, last), cached."""
    global _CLIPS_CACHE
    if _CLIPS_CACHE is None:
        _CLIPS_CACHE = {r.clip_id: (int(r.first), int(r.last)) for r in list_clips().itertuples()}
    return _CLIPS_CACHE
