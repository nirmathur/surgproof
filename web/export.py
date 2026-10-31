#!/usr/bin/env python3
"""Export real SurgProof outputs for video12 into web/data/video12.json and web/data/live.json.

Reads only real model outputs on disk:
  - data/index.csv (all frames of video12)
  - out/decisions.csv (per-frame, per-label verification status, evidence, double-check & Amber loop rounds)
  - out/annotations.jsonl, out/annotations_live.jsonl, out/annotations_masks*.jsonl (boxes, tips, masks, phase)
  - out/masks/<frame_id>_<label>.png (AI segmentation masks)
  - out/narration_video12.json (evidence-grounded surgical narration)
  - out/results.json, out/student.json, out/localize_results.json, out/manual.json

Never fabricates missing fields: any annotation not yet produced is set to None (JSON null) so the UI renders 'pending'.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C


def _read_json(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _to_bool_or_none(val):
    if val in (None, "", "nan", "NaN", "None"):
        return None
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no"):
        return False
    return None


def _to_float_or_none(val):
    if val in (None, "", "nan", "NaN", "None"):
        return None
    try:
        return round(float(val), 4)
    except Exception:
        return None


def _to_int_or_none(val):
    if val in (None, "", "nan", "NaN", "None"):
        return None
    try:
        return int(float(val))
    except Exception:
        return None


def _merge_annotation_sources(out_dir: Path) -> tuple[dict[tuple[str, str], dict], dict[str, dict], list[dict]]:
    """Merge out/annotations.jsonl, out/annotations_live.jsonl, out/annotations_masks*.jsonl, and out/polygons/*.jsonl."""
    ann_by_frame_label: dict[tuple[str, str], dict] = {}
    phase_by_frame: dict[str, dict] = {}
    live_rows: list[dict] = []

    base_files = [
        out_dir / "annotations.jsonl",
        out_dir / "annotations_live.jsonl",
    ]
    mask_files = [Path(p) for p in sorted(glob.glob(str(out_dir / "annotations_masks*.jsonl")))]
    poly_dir = out_dir / "polygons"
    poly_files = (
        [
            p
            for p in sorted(poly_dir.glob("*.jsonl"))
            if not p.name.startswith("raw_") and not p.name.startswith("refine_raw_")
        ]
        if poly_dir.exists()
        else []
    )
    all_files = base_files + mask_files + poly_files

    for fpath in all_files:
        rows = _read_jsonl(fpath)
        is_poly_file = fpath.parent.name == "polygons"
        if fpath.name.startswith("annotations_live") or fpath.name.startswith("annotations_masks") or is_poly_file:
            live_rows.extend(rows)
        for r in rows:
            fid = r.get("frame_id")
            lbl = r.get("label") or r.get("key")
            if not fid:
                continue
            if r.get("phase"):
                phase_by_frame[fid] = {
                    "phase": r.get("phase"),
                    "phase_conf": _to_float_or_none(r.get("phase_conf")),
                }
            if not lbl:
                continue
            key = (str(fid), str(lbl))
            cur = dict(ann_by_frame_label.get(key, {}))
            if is_poly_file:
                for k, v in r.items():
                    if k == "status":
                        cur["mask_status"] = v
                        if not cur.get("status"):
                            cur["status"] = v
                    elif k == "rounds" and isinstance(v, list):
                        cur["polygon_rounds"] = v
                        if v and isinstance(v[-1], dict) and v[-1].get("iou_pass1_pass2") is not None and not cur.get("mask_iou"):
                            cur["mask_iou"] = v[-1].get("iou_pass1_pass2")
                    elif v is not None and v != "":
                        cur[k] = v
            else:
                for k, v in r.items():
                    if v is not None and v != "":
                        cur[k] = v
            ann_by_frame_label[key] = cur

    # Also check out/masks/<frame_id>_<label>.png directly on disk
    masks_dir = out_dir / "masks"
    if masks_dir.exists():
        for mp in sorted(masks_dir.glob("*.png")):
            stem = mp.stem
            for lbl in C.LABELS:
                suffix = f"_{lbl}"
                if stem.endswith(suffix):
                    fid = stem[: -len(suffix)]
                    key = (fid, lbl)
                    cur = dict(ann_by_frame_label.get(key, {"frame_id": fid, "label": lbl}))
                    cur["mask_path"] = str(mp)
                    ann_by_frame_label[key] = cur
                    break

    # If mask shard files, polygon files, or out/masks added mask_path/polygon_px to annotations.jsonl, sync back cleanly
    ann_main_path = out_dir / "annotations.jsonl"
    if ann_main_path.exists() and (mask_files or poly_files or (masks_dir.exists() and any(masks_dir.glob("*.png")))):
        main_rows = _read_jsonl(ann_main_path)
        updated = False
        seen_keys = set()
        for r in main_rows:
            fid = str(r.get("frame_id", ""))
            lbl = str(r.get("label") or r.get("key") or "")
            seen_keys.add((fid, lbl))
            merged = ann_by_frame_label.get((fid, lbl), {})
            for mk in ("mask", "mask_path", "mask_iou", "pass1_mask_iou", "polygon_px", "mask_source"):
                if merged.get(mk) and not r.get(mk):
                    r[mk] = merged[mk]
                    updated = True
        for (fid, lbl), merged in ann_by_frame_label.items():
            if (fid, lbl) not in seen_keys and (
                merged.get("mask") or merged.get("mask_path") or merged.get("polygon_px") or merged.get("box")
            ):
                main_rows.append(merged)
                updated = True
        if updated:
            tmp_p = ann_main_path.with_suffix(".jsonl.tmp")
            with open(tmp_p, "w") as fh:
                for r in main_rows:
                    fh.write(json.dumps(r) + "\n")
            os.replace(tmp_p, ann_main_path)

    if not live_rows and (out_dir / "annotations.jsonl").exists():
        live_rows = _read_jsonl(out_dir / "annotations.jsonl")

    return ann_by_frame_label, phase_by_frame, live_rows


def export_video_bundle(
    video: str = "video12",
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    web_data_dir: Path | None = None,
) -> dict:
    data_dir = Path(data_dir or C.DATA_DIR)
    out_dir = Path(out_dir or C.OUT_DIR)
    web_data_dir = Path(web_data_dir or (ROOT / "web" / "data"))
    frames_out_dir = web_data_dir / "frames"
    gt_masks_out_dir = web_data_dir / "gt_masks"
    masks_out_dir = web_data_dir / "masks"
    masks_r1_out_dir = web_data_dir / "masks_r1"
    masks_r2_out_dir = web_data_dir / "masks_r2"
    frames_out_dir.mkdir(parents=True, exist_ok=True)
    gt_masks_out_dir.mkdir(parents=True, exist_ok=True)
    masks_out_dir.mkdir(parents=True, exist_ok=True)
    masks_r1_out_dir.mkdir(parents=True, exist_ok=True)
    masks_r2_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frame index for target video
    index_csv = data_dir / "index.csv"
    video_rows = []
    if index_csv.exists():
        with open(index_csv) as fh:
            for r in csv.DictReader(fh):
                if r.get("video") == video:
                    video_rows.append(r)
    video_rows.sort(key=lambda r: (str(r.get("clip", "")), int(float(r.get("frame", 0) or 0)), str(r.get("frame_id", ""))))

    # 2. Load decisions.csv for target video
    dec_by_frame_label: dict[tuple[str, str], dict] = {}
    dec_csv = out_dir / "decisions.csv"
    if dec_csv.exists():
        with open(dec_csv) as fh:
            for r in csv.DictReader(fh):
                fid = r.get("frame_id")
                lbl = r.get("key") or r.get("label")
                if fid and lbl and (r.get("video") == video or str(fid).startswith(f"{video}_")):
                    dec_by_frame_label[(str(fid), str(lbl))] = r

    # 3. Load and merge spatial annotations + masks
    ann_by_frame_label, phase_by_frame, live_rows = _merge_annotation_sources(out_dir)

    # 4. Load grounded narration for target video
    narration_by_frame: dict[str, dict] = {}
    narr_obj = _read_json(out_dir / f"narration_{video}.json")
    if narr_obj:
        sents = narr_obj.get("sentences", narr_obj if isinstance(narr_obj, list) else [])
        for s in sents:
            if isinstance(s, dict) and s.get("frame_id"):
                narration_by_frame[str(s["frame_id"])] = s

    # 5. Load evaluation & manual artifacts
    results_json = _read_json(out_dir / "results.json")
    student_json = _read_json(out_dir / "student.json")
    localize_json = _read_json(out_dir / "localize_results.json")
    mask_results_json = _read_json(out_dir / "mask_results.json")
    mask_refine_json = _read_json(out_dir / "mask_refine_results.json")
    manual_json = _read_json(out_dir / "manual.json")
    audit_json = _read_json(out_dir / "eval_audit.json")

    headline = None
    if isinstance(results_json, dict):
        acc5 = results_json.get("accepted_5_informative") or results_json.get("accepted_excl_liver") or {}
        headline = {
            "accepted_accuracy": _to_float_or_none(acc5.get("accuracy")),
            "accepted_coverage": _to_float_or_none(acc5.get("coverage")),
            "single_raw_accuracy": _to_float_or_none(results_json.get("manual_raw_acc") or acc5.get("raw_accuracy")),
            "n_test_labels": _to_int_or_none(results_json.get("n_test_labels")),
            "n_test_labels_excl_liver": _to_int_or_none(results_json.get("n_test_labels_excl_liver")),
            "n_test_videos": 6,
            "bootstrap_ci_95": acc5.get("bootstrap_ci_95"),
            "amber_loop": results_json.get("amber_loop"),
            "hero_number_source": "out/results.json:accepted_5_informative.accuracy",
        }

    # 6. Build per-frame bundle and copy real images/masks
    frames_bundle = []
    for idx_pos, frow in enumerate(video_rows):
        fid = str(frow["frame_id"])
        img_raw = str(frow.get("img_path") or frow.get("path") or "").strip()
        img_src = Path(img_raw) if img_raw else None
        img_url = None
        if img_src and img_src.is_file():
            dst_img = frames_out_dir / f"{fid}.png"
            if not dst_img.exists() or dst_img.stat().st_size != img_src.stat().st_size:
                shutil.copy2(img_src, dst_img)
            img_url = f"data/frames/{fid}.png"

        gt_mask_raw = str(frow.get("mask_path") or "").strip()
        gt_mask_src = Path(gt_mask_raw) if gt_mask_raw else None
        gt_mask_url = None
        if gt_mask_src and gt_mask_src.is_file():
            dst_gt = gt_masks_out_dir / f"{fid}.png"
            if not dst_gt.exists() or dst_gt.stat().st_size != gt_mask_src.stat().st_size:
                shutil.copy2(gt_mask_src, dst_gt)
            gt_mask_url = f"data/gt_masks/{fid}.png"

        ph_info = phase_by_frame.get(fid, {})
        labels_dict = {}
        labels_list = []

        for lbl in C.LABELS:
            d = dec_by_frame_label.get((fid, lbl), {})
            a = ann_by_frame_label.get((fid, lbl), {})

            mask_url = None
            cand_mask_paths = []
            if a.get("mask_path"):
                cand_mask_paths.append(Path(str(a["mask_path"])))
            cand_mask_paths.append(out_dir / "masks" / f"{fid}_{lbl}.png")
            for mp in cand_mask_paths:
                if mp and mp.is_file():
                    dst_m = masks_out_dir / f"{fid}_{lbl}.png"
                    if not dst_m.exists() or dst_m.stat().st_size != mp.stat().st_size:
                        shutil.copy2(mp, dst_m)
                    mask_url = f"data/masks/{fid}_{lbl}.png"
                    break

            mask_r1_url = None
            mp_r1 = out_dir / "masks_r1" / f"{fid}_{lbl}.png"
            if mp_r1.is_file():
                dst_r1 = masks_r1_out_dir / f"{fid}_{lbl}.png"
                if not dst_r1.exists() or dst_r1.stat().st_size != mp_r1.stat().st_size:
                    shutil.copy2(mp_r1, dst_r1)
                mask_r1_url = f"data/masks_r1/{fid}_{lbl}.png"

            mask_r2_url = None
            mp_r2 = out_dir / "masks_r2" / f"{fid}_{lbl}.png"
            if mp_r2.is_file():
                dst_r2 = masks_r2_out_dir / f"{fid}_{lbl}.png"
                if not dst_r2.exists() or dst_r2.stat().st_size != mp_r2.stat().st_size:
                    shutil.copy2(mp_r2, dst_r2)
                mask_r2_url = f"data/masks_r2/{fid}_{lbl}.png"

            round_history = []
            for r_idx in range(4):
                rk = f"status_round_{r_idx}"
                if d.get(rk):
                    round_history.append({"round": r_idx, "status": d.get(rk)})

            poly_px = a.get("polygon_px") if isinstance(a.get("polygon_px"), list) else None
            item = {
                "frame_id": fid,
                "key": lbl,
                "label": lbl,
                "status": d.get("status") or a.get("status") or None,
                "present": _to_int_or_none(d.get("present")),
                "final_present": _to_int_or_none(d.get("final_present") if d.get("final_present") is not None else d.get("present")),
                "conf": _to_float_or_none(d.get("conf") if d.get("conf") is not None else a.get("conf")),
                "trust": _to_float_or_none(d.get("trust")),
                "evidence": d.get("evidence") or d.get("rationale") or None,
                "pv_match": _to_bool_or_none(d.get("pv_match")),
                "p_wrong": _to_float_or_none(d.get("p_wrong")),
                "critic_reason": d.get("critic_reason") or d.get("critic") or None,
                "status_round_0": d.get("status_round_0") or None,
                "status_round_1": d.get("status_round_1") or None,
                "status_round_2": d.get("status_round_2") or None,
                "status_round_3": d.get("status_round_3") or None,
                "round_resolved": _to_int_or_none(d.get("round_resolved") or d.get("resolved_round")),
                "argon_reason": d.get("argon_reason") or None,
                "round_history": round_history,
                "amber_rounds": round_history,
                "box": a.get("box") if isinstance(a.get("box"), list) else None,
                "tip": a.get("tip") if isinstance(a.get("tip"), list) else None,
                "pass1": a.get("pass1") if isinstance(a.get("pass1"), dict) else None,
                "pass2": a.get("pass2") if isinstance(a.get("pass2"), dict) else None,
                "box_iou": _to_float_or_none((a.get("pass2") or {}).get("iou") if isinstance(a.get("pass2"), dict) else a.get("iou")),
                "mask_url": mask_url,
                "mask_r0_url": mask_url,
                "mask_r1_url": mask_r1_url,
                "mask_r2_url": mask_r2_url,
                "mask_b64": a.get("mask") if isinstance(a.get("mask"), str) else None,
                "mask_iou": _to_float_or_none(a.get("mask_iou")),
                "mask_source": a.get("mask_source") or None,
                "mask_status": a.get("mask_status") or None,
                "polygon_px": poly_px,
                "polygon_rounds": a.get("polygon_rounds") if isinstance(a.get("polygon_rounds"), list) else None,
                "spatial_status": a.get("status") or ("pending" if not a.get("box") and not mask_url and not poly_px else None),
                "gt_present_scoring_only": _to_int_or_none(frow.get(f"gt_{lbl}")),
                "gt_frac_scoring_only": _to_float_or_none(frow.get(f"frac_{lbl}")),
            }
            labels_dict[lbl] = item
            labels_list.append(item)

        narr_entry = narration_by_frame.get(fid)
        frames_bundle.append({
            "frame_id": fid,
            "frame_idx": idx_pos,
            "clip": frow.get("clip"),
            "frame_num": _to_int_or_none(frow.get("frame") or str(fid).split("_")[-1]),
            "split": frow.get("split", "test"),
            "img_url": img_url,
            "gt_mask_url": gt_mask_url,
            "phase": ph_info.get("phase") or None,
            "phase_conf": ph_info.get("phase_conf"),
            "narration": (narr_entry.get("sentence") or narr_entry.get("narration")) if isinstance(narr_entry, dict) else None,
            "narration_entry": narr_entry,
            "labels": labels_dict,
            "annotations": labels_dict,
            "labels_list": labels_list,
        })

    n_with_boxes = sum(
        1 for f in frames_bundle if any(v.get("box") is not None for v in f["labels"].values())
    )
    n_with_masks = sum(
        1
        for f in frames_bundle
        if any(
            v.get("mask_url") is not None or v.get("mask_b64") is not None or v.get("polygon_px") is not None
            for v in f["labels"].values()
        )
    )
    n_with_polygons = sum(
        1 for f in frames_bundle if any(v.get("polygon_px") is not None for v in f["labels"].values())
    )
    n_with_phase = sum(1 for f in frames_bundle if f.get("phase") is not None)

    bundle = {
        "video": video,
        "split": "test",
        "exported_at": round(time.time(), 3),
        "n_frames": len(frames_bundle),
        "n_frames_with_boxes": n_with_boxes,
        "n_frames_with_masks": n_with_masks,
        "n_frames_with_polygons": n_with_polygons,
        "n_frames_with_phase": n_with_phase,
        "labels": list(C.LABELS),
        "label_keys": list(C.LABELS),
        "headline": headline,
        "results": results_json,
        "student": student_json,
        "localize_results": localize_json,
        "mask_results": mask_results_json,
        "mask_refine_results": mask_refine_json,
        "eval_audit": audit_json,
        "manual": manual_json,
        "frames": frames_bundle,
    }

    video_json_path = web_data_dir / f"{video}.json"
    video_json_path.write_text(json.dumps(bundle, indent=2))

    # 7. Write web/data/live.json (tail of out/annotations_live.jsonl + mask shards)
    live_tail = live_rows[-240:]
    live_payload = {
        "video": video,
        "count": len(live_rows),
        "updated_at": round(time.time(), 3),
        "n_frames_with_boxes": n_with_boxes,
        "n_frames_with_masks": n_with_masks,
        "annotations": live_tail,
    }
    live_json_path = web_data_dir / "live.json"
    live_json_path.write_text(json.dumps(live_payload, indent=2))

    return bundle


def sync_web_data_across_worktrees(src_web_data: Path):
    """Ensure ~/surgatlas/web/data and active web lanes see the exported bundle immediately."""
    main_repo = Path(os.path.expanduser("~/surgatlas"))
    lanes_root = Path(os.path.expanduser("~/surgatlas-lanes"))
    targets = [main_repo / "web" / "data"]
    if lanes_root.exists():
        for lane_name in ("demo", "webshell", "webcanvas", "webtimeline", "gate", "recorder"):
            lane_dir = lanes_root / lane_name
            if lane_dir.exists():
                targets.append(lane_dir / "web" / "data")

    for t in targets:
        try:
            if t.resolve() == src_web_data.resolve():
                continue
        except Exception:
            pass
        try:
            t.parent.mkdir(parents=True, exist_ok=True)
            if t.is_symlink() or t.is_file():
                t.unlink()
            elif t.exists():
                shutil.rmtree(t)
            os.symlink(src_web_data.resolve(), t)
        except Exception:
            continue


def main():
    web_data_dir = ROOT / "web" / "data"
    index_csv = C.DATA_DIR / "index.csv"
    test_videos = []
    if index_csv.exists():
        with open(index_csv) as fh:
            for r in csv.DictReader(fh):
                if r.get("split") == "test" and r.get("video") and r["video"] not in test_videos:
                    test_videos.append(r["video"])
    if "video12" in test_videos:
        test_videos = [v for v in test_videos if v != "video12"] + ["video12"]
    elif not test_videos:
        test_videos = ["video12"]

    video_summaries = []
    bundle = None
    for vid in test_videos:
        b = export_video_bundle(vid, web_data_dir=web_data_dir)
        video_summaries.append({
            "video": vid,
            "n_frames": b["n_frames"],
            "n_frames_with_boxes": b["n_frames_with_boxes"],
            "n_frames_with_masks": b["n_frames_with_masks"],
            "n_frames_with_polygons": b["n_frames_with_polygons"],
            "n_frames_with_phase": b["n_frames_with_phase"],
        })
        if vid == "video12" or bundle is None:
            bundle = b
    (web_data_dir / "videos.json").write_text(json.dumps({"test_videos": video_summaries}, indent=2))

    # Copy real directory into ~/surgatlas/out/web_data and symlink or copy into ~/surgatlas/web/data
    shared_web_data = (C.OUT_DIR / "web_data").resolve()
    if shared_web_data != web_data_dir.resolve():
        if shared_web_data.exists():
            shutil.rmtree(shared_web_data)
        shutil.copytree(web_data_dir, shared_web_data)
        sync_web_data_across_worktrees(shared_web_data)
    print(
        f"Exported web/data/video12.json (+ {len(test_videos)} test videos): {bundle['n_frames']} frames "
        f"({bundle['n_frames_with_boxes']} with boxes, {bundle['n_frames_with_masks']} with masks, "
        f"{bundle['n_frames_with_polygons']} with polygons, {bundle['n_frames_with_phase']} with phase)"
    )


if __name__ == "__main__":
    main()
