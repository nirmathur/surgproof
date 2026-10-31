"""Step 1: index frames -> out/tracker/frames.parquet. Asserts frames within each clip are consecutive."""
import re

import pandas as pd

from tracker.common import ROOT, cli, load_cfg, skip, work


def build(cfg):
    root = ROOT / cfg["paths"]["data_root"]
    rows = []
    for clip_dir in sorted(p for p in root.glob("video*/video*_*") if p.is_dir()):
        for f in clip_dir.glob("frame_*_endo.png"):
            n = int(re.match(r"frame_(\d+)_endo\.png$", f.name).group(1))
            m = f.with_name(f"frame_{n}_endo_watershed_mask.png")
            rows.append({"video_id": clip_dir.parent.name, "clip_id": clip_dir.name, "frame_idx": n,
                         "image_path": str(f.relative_to(ROOT)),
                         "mask_path": str(m.relative_to(ROOT)) if m.exists() else None})
    if not rows:
        raise SystemExit(f"No frames under {root}")
    df = pd.DataFrame(rows).sort_values(["video_id", "clip_id", "frame_idx"]).reset_index(drop=True)
    test, tune = set(cfg["split"]["test"]), set(cfg["split"]["tune"])
    assert not test & tune, "video in both splits"
    unknown = set(df.video_id) - test - tune
    assert not unknown, f"videos not in split: {unknown}"
    df["split"] = df.video_id.map(lambda v: "test" if v in test else "tune")

    gaps = []
    for clip, g in df.groupby("clip_id"):
        d = g.frame_idx.diff().dropna()
        for i in d.index[d != 1]:
            gaps.append((clip, int(df.frame_idx[i - 1]), int(df.frame_idx[i])))
    missing_masks = int(df.mask_path.isna().sum())
    print(f"{len(df)} frames, {df.clip_id.nunique()} clips, {df.video_id.nunique()} videos; "
          f"split frames: {df.split.value_counts().to_dict()}; missing masks: {missing_masks}")
    print(f"gaps within clips: {len(gaps)}" + "".join(f"\n  {c}: {a} -> {b}" for c, a, b in gaps[:20]))
    assert not gaps, "frames within a clip are not consecutive"
    return df


if __name__ == "__main__":
    a = cli("Index CholecSeg8k frames", source=False).parse_args()
    cfg = load_cfg()
    out = work(cfg, "frames.parquet")
    if not skip(out, a.force):
        build(cfg).to_parquet(out, index=False)
        print("wrote", out)
