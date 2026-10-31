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
    non_ignore = set(vals.tolist()) - {0, 255}
    # auto-detect convention: released watershed masks use WATERSHED_VALUE_TO_CLASS
    # plus 255 for the watershed border (and rare 0 pixels), whereas paper convention is 0..12.
    if non_ignore and non_ignore <= set(C.WATERSHED_VALUE_TO_CLASS) and (max(non_ignore) > 12 or 0 not in vals):
        lut = np.zeros(256, dtype=np.int16) - 1
        for v, c in C.WATERSHED_VALUE_TO_CLASS.items():
            lut[v] = c
        cls = lut[m]
    else:
        cls = np.where(m == 255, -1, m.astype(np.int16))  # assume value == class id (paper convention)
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


# Endoscapes2023 (CAMMA Endoscapes-Seg50) semantic mask class IDs (verified against COCO categories):
# 0: background, 1: cystic_plate, 2: calot_triangle, 3: cystic_artery,
# 4: cystic_duct, 5: gallbladder, 6: tool (pooled instruments), 255: ignore border.
ENDOSCAPES_CLASS_MAP = {
    "gallbladder": 5,
    "cystic_duct": 4,
}


def endoscapes_presence(mask_arr):
    """Derive presence & pixel fraction from an Endoscapes2023 semantic mask array."""
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    total = mask_arr.size
    out = {}
    for key in C.LABELS:
        if key in ENDOSCAPES_CLASS_MAP:
            cid = ENDOSCAPES_CLASS_MAP[key]
            frac = float((mask_arr == cid).sum()) / total
            out[f"gt_{key}"] = int(frac >= C.MIN_FRAC.get(key, C.MIN_FRAC["default"]))
            out[f"frac_{key}"] = round(frac, 5)
        else:
            out[f"gt_{key}"] = -1
            out[f"frac_{key}"] = -1.0
    frac_tool = float((mask_arr == 6).sum()) / total
    out["gt_tool_any"] = int(frac_tool >= C.MIN_FRAC["default"])
    out["frac_tool_any"] = round(frac_tool, 5)
    return out


def build_endoscapes2023():
    """Download MedOtter/Endoscapes2023 (CC BY-NC-SA 4.0, 50 non-Cholec80 videos, 493 frames)
    and build data/external/endoscapes2023/index.csv with the standard index contract.
    """
    import io
    from huggingface_hub import hf_hub_download

    raw_ext = C.RAW_DIR / "external" / "endoscapes2023"
    frames_dir = raw_ext / "frames"
    masks_dir = raw_ext / "masks"
    frames_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    out_ext = C.DATA_DIR / "external" / "endoscapes2023"
    out_ext.mkdir(parents=True, exist_ok=True)

    split_map = {"test": "test", "val": "calib", "train": "train"}
    rows = []
    for hf_split in ["test", "val", "train"]:
        pq_path = hf_hub_download(
            "MedOtter/Endoscapes2023",
            f"data/{hf_split}-00000-of-00001.parquet",
            repo_type="dataset",
            local_dir=str(raw_ext),
        )
        df_pq = pd.read_parquet(pq_path)
        for _, r in df_pq.iterrows():
            img_id = str(r["image_id"])
            vid = f"endoscapes_{r['video_id']}"
            frame_id = f"endoscapes_{img_id}"
            f_path = frames_dir / f"{img_id}.png"
            m_path = masks_dir / f"{img_id}_mask.png"
            if not f_path.exists():
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(f_path)
            mask_im = Image.open(io.BytesIO(r["label"]["bytes"]))
            if not m_path.exists():
                mask_im.save(m_path)
            pres = endoscapes_presence(np.array(mask_im))
            rows.append({
                "frame_id": frame_id,
                "video": vid,
                "clip": vid,
                "path": str(f_path),
                "mask_path": str(m_path),
                **{k: pres[k] for k in [
                    "gt_grasper", "frac_grasper",
                    "gt_l_hook", "frac_l_hook",
                    "gt_liver", "frac_liver",
                    "gt_gallbladder", "frac_gallbladder",
                    "gt_cystic_duct", "frac_cystic_duct",
                    "gt_blood", "frac_blood",
                ]},
                "split": split_map[hf_split],
                "source_split": hf_split,
                "dataset": "endoscapes2023",
                "scorable_keys": "gallbladder,cystic_duct",
                "gt_tool_any": pres["gt_tool_any"],
                "frac_tool_any": pres["frac_tool_any"],
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_ext / "index.csv", index=False)
    print(f"Endoscapes2023: {len(df_out)} frames, {df_out.video.nunique()} videos -> {out_ext / 'index.csv'}")
    print(df_out.groupby(["split", "source_split"]).size().to_string())
    for s in ["test", "calib", "train"]:
        sub = df_out[df_out.split == s]
        print(
            f"  {s:5s} (n={len(sub):3d}, videos={sub.video.nunique():2d}): "
            f"gallbladder={sub.gt_gallbladder.mean():.3f} ({int(sub.gt_gallbladder.sum())}/{len(sub)}), "
            f"cystic_duct={sub.gt_cystic_duct.mean():.3f} ({int(sub.gt_cystic_duct.sum())}/{len(sub)}), "
            f"tool_any={sub.gt_tool_any.mean():.3f} ({int(sub.gt_tool_any.sum())}/{len(sub)})"
        )
    return df_out


# All 17 Cholec80 videos present in CholecSeg8k (train, calib, and test) — excluded from CholecT50 external test.
CHOLECSEG8K_VIDEOS = {
    "video01", "video09", "video12", "video17", "video18", "video20",
    "video24", "video25", "video26", "video27", "video28", "video35",
    "video37", "video43", "video48", "video52", "video55",
}
CHOLECSEG8K_VID_IDS = {f"VID{int(v.replace('video', '')):02d}" for v in CHOLECSEG8K_VIDEOS}


def cholect50_presence(instruments):
    """Map CholecT50 frame-level instrument list onto config.LABELS.
    Only instrument presence ('grasper' -> grasper, 'hook' -> l_hook) is a true visibility label;
    triplet targets are active action targets (not visibility), so anatomy keys are set to -1.
    """
    inst_set = {str(x).strip().lower() for x in instruments}
    out = {}
    for key in C.LABELS:
        if key == "grasper":
            present = int("grasper" in inst_set)
            out[f"gt_{key}"] = present
            out[f"frac_{key}"] = float(present)
        elif key == "l_hook":
            present = int("hook" in inst_set)
            out[f"gt_{key}"] = present
            out[f"frac_{key}"] = float(present)
        else:
            out[f"gt_{key}"] = -1
            out[f"frac_{key}"] = -1.0
    return out


def _fetch_cholect50_meta(raw_c50):
    """Fetch lightweight frame metadata from Voxel51/cholect50 samples.json using parallel byte-range requests."""
    import concurrent.futures
    import json
    import re
    import requests
    from huggingface_hub import hf_hub_url

    cache_file = raw_c50 / "samples_meta.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = hf_hub_url("Voxel51/cholect50", "samples.json", repo_type="dataset")
    head = requests.head(url, allow_redirects=True, timeout=20)
    total_size = int(head.headers["Content-Length"])
    n_chunks = 16
    chunk_size = (total_size + n_chunks - 1) // n_chunks
    overlap = 65536

    def fetch_chunk(idx):
        start = idx * chunk_size
        end = min(total_size - 1, (idx + 1) * chunk_size - 1 + overlap)
        r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=90)
        text = r.text
        out = []
        # Split on '"filepath":' so inner Classification '{"_id": ...}' objects never split a sample block
        for block in text.split('"filepath":')[1:]:
            if '"clip_embeddings"' not in block:
                continue
            m_fp = re.match(r'\s*"([^"]+)"', block)
            m_vid = re.search(r'"video_id":\s*"([^"]+)"', block)
            m_fn = re.search(r'"frame_number":\s*(\d+)', block)
            m_inst = re.search(r'"instruments":\s*\{(.*?)"targets":', block, re.DOTALL)
            if m_fp and m_vid and m_fn and m_inst:
                inst_labels = re.findall(r'"label":\s*"([^"]+)"', m_inst.group(1))
                out.append((m_fp.group(1), m_vid.group(1), int(m_fn.group(1)), sorted(set(inst_labels))))
        return out

    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for chunk_out in ex.map(fetch_chunk, range(n_chunks)):
            for fp, vid, fn, inst in chunk_out:
                records[(vid, fn)] = {
                    "filepath": fp,
                    "video_id": vid,
                    "frame_number": fn,
                    "instruments": inst,
                }

    meta_list = [records[k] for k in sorted(records)]
    cache_file.write_text(json.dumps(meta_list))
    return meta_list


def build_cholect50(frames_per_video=3):
    """Build data/external/cholect50/index.csv from Voxel51/cholect50 (CC BY-NC-SA 4.0),
    strictly excluding all Cholec80 videos that appear in CholecSeg8k.
    """
    import concurrent.futures
    from huggingface_hub import hf_hub_download

    raw_c50 = C.RAW_DIR / "external" / "cholect50"
    raw_c50.mkdir(parents=True, exist_ok=True)
    out_ext = C.DATA_DIR / "external" / "cholect50"
    out_ext.mkdir(parents=True, exist_ok=True)

    meta_list = _fetch_cholect50_meta(raw_c50)
    by_vid = {}
    for item in meta_list:
        vid = item["video_id"]
        if vid in CHOLECSEG8K_VID_IDS:
            continue
        by_vid.setdefault(vid, []).append(item)

    def download_valid_for_video(vid_items):
        vid, items = vid_items
        items = sorted(items, key=lambda x: x["frame_number"])
        # Prefer interior frames (15%..85% of surgery) where active dissection occurs
        n = len(items)
        anchors = [int(round(n * (0.20 + 0.60 * k / max(1, frames_per_video - 1)))) for k in range(frames_per_video)]
        chosen = []
        used_fn = set()
        for anchor in anchors:
            # Walk outward from anchor until a non-blank frame (>20KB, std>5) is found
            for offset in range(0, n, 5):
                for cand_idx in (anchor + offset, anchor - offset):
                    if 0 <= cand_idx < n:
                        cand = items[cand_idx]
                        if cand["frame_number"] in used_fn:
                            continue
                        rel_fp = cand["filepath"].lstrip("./")
                        local_p = hf_hub_download(
                            "Voxel51/cholect50",
                            rel_fp,
                            repo_type="dataset",
                            local_dir=str(raw_c50),
                        )
                        from pathlib import Path as _P
                        if _P(local_p).stat().st_size >= 20_000:
                            arr = np.array(Image.open(local_p))
                            if float(arr.std()) > 5.0:
                                used_fn.add(cand["frame_number"])
                                chosen.append((cand, local_p))
                                break
                else:
                    continue
                break
        return chosen

    selected_with_paths = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        for vid_chosen in ex.map(download_valid_for_video, sorted(by_vid.items())):
            selected_with_paths.extend(vid_chosen)

    rows = []
    for item, local_p in selected_with_paths:
        vid = item["video_id"]
        fn = item["frame_number"]
        vid_slug = f"cholect50_{vid.lower()}"
        frame_id = f"{vid_slug}_{fn:06d}"
        pres = cholect50_presence(item["instruments"])
        rows.append({
            "frame_id": frame_id,
            "video": vid_slug,
            "clip": vid_slug,
            "path": str(local_p),
            "mask_path": "",
            **{k: pres[k] for k in [
                "gt_grasper", "frac_grasper",
                "gt_l_hook", "frac_l_hook",
                "gt_liver", "frac_liver",
                "gt_gallbladder", "frac_gallbladder",
                "gt_cystic_duct", "frac_cystic_duct",
                "gt_blood", "frac_blood",
            ]},
            "split": "test",
            "source_split": "test",
            "dataset": "cholect50",
            "scorable_keys": "grasper,l_hook",
            "instruments_raw": ",".join(item["instruments"]),
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_ext / "index.csv", index=False)
    print(f"CholecT50 (disjoint from CholecSeg8k): {len(df_out)} frames, {df_out.video.nunique()} videos -> {out_ext / 'index.csv'}")
    print(
        f"  test (n={len(df_out)}, videos={df_out.video.nunique()}): "
        f"grasper={df_out.gt_grasper.mean():.3f} ({int(df_out.gt_grasper.sum())}/{len(df_out)}), "
        f"l_hook={df_out.gt_l_hook.mean():.3f} ({int(df_out.gt_l_hook.sum())}/{len(df_out)})"
    )
    return df_out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--external", choices=["endoscapes2023", "cholect50", "all"], default=None)
    a = ap.parse_args()
    if a.external == "endoscapes2023":
        build_endoscapes2023()
    elif a.external == "cholect50":
        build_cholect50()
    elif a.external == "all":
        build_endoscapes2023()
        build_cholect50()
    else:
        r = find_root()
        inspect(r) if a.inspect else build(r)
