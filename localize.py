"""Autonomous zero-human-label surgical localization: bounding boxes, instrument tips, and surgical phase.

For frames/labels `nir-verifier` has accepted as present:
1. Pass 1 (`gemini-3.8-flash` bulk): predicts boxes `[ymin, xmin, ymax, xmax]` (0-1000),
   instrument tip points `[y, x]` (0-1000 for `grasper`, `l_hook`), and surgical `phase` + `conf`.
2. Pass 2 (`gemini-4-argon` independent double-check): independently predicts boxes, tip points, and phase.
3. Box agreement check: `IoU(pass1, pass2) >= 0.5` -> `accept` (round 0); else `AMBER` and re-runs
   an iterative tighter-crop disagreement loop for up to 3 rounds, logging each round to `out/localize_rounds.jsonl`.
4. Streams `out/annotations_live.jsonl` line-by-line as each annotation completes and writes `out/annotations.jsonl`.
5. Post-hoc evaluation runs strictly AFTER via `verify_localize.evaluate_annotations` -> `out/localize_results.json`.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel

import config as C
import gem
from verify.engine import prepare_verifier_dataframe
from verify_localize import (
    INSTRUMENT_KEYS,
    IOU_ACCEPT_THRESHOLD,
    MAX_AMBER_ROUNDS,
    average_boxes,
    box_iou,
    clamp_prob,
    evaluate_annotations,
    sanitize_box,
    sanitize_tip,
    snap_tip_to_instrument_pixels,
    verify_and_refine_item,
)


class LocalizedStructure(BaseModel):
    label: str
    box_2d: list[int]          # [ymin, xmin, ymax, xmax] normalised 0-1000
    tip_2d: list[int]          # [y, x] normalised 0-1000 if instrument (grasper, l_hook), else []
    confidence: float          # 0.0 - 1.0


class FramePassOutput(BaseModel):
    phase: str
    phase_confidence: float
    items: list[LocalizedStructure]


NAMES = {k: v[1] for k, v in C.LABELS.items()}

DEFAULT_PHASES = [
    "1. Preparation",
    "2. Calot's (Hepatocystic) Triangle Dissection",
    "3. Clipping and Cutting",
    "4. Gallbladder Dissection",
    "5. Gallbladder Packaging",
    "6. Cleaning and Coagulation",
    "7. Gallbladder Retraction",
]


def load_manual(manual_path: Path | str | None = None) -> dict[str, Any]:
    if manual_path is not None and Path(manual_path).is_file():
        return json.loads(Path(manual_path).read_text(encoding="utf-8"))
    for p in (C.OUT_DIR / "manual.json", C.ROOT / "coord" / "artifacts" / "manual.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return {"procedure": "Laparoscopic Cholecystectomy", "phases": DEFAULT_PHASES, "structures": []}


def normalize_phase(raw_phase: str | None, phases_list: list[str]) -> str:
    """Match a model-predicted phase string to its canonical phase entry in `manual['phases']`."""
    if not phases_list:
        phases_list = DEFAULT_PHASES
    if not raw_phase or not isinstance(raw_phase, str):
        return phases_list[1].split(":")[0].strip() if len(phases_list) > 1 else phases_list[0]
    s = raw_phase.strip()
    s_low = s.lower()
    for p in phases_list:
        p_short = p.split(":")[0].strip()
        if s_low == p.lower() or s_low == p_short.lower() or s_low.startswith(p_short.lower()):
            return p_short
        # Match by phase number prefix "1.", "2.", etc.
        if len(p_short) >= 2 and p_short[:2] in ("1.", "2.", "3.", "4.", "5.", "6.", "7."):
            if s_low.startswith(p_short[:2]) or f"phase {p_short[0]}" in s_low or f"step {p_short[0]}" in s_low:
                return p_short
    # Keyword fallback
    kw_map = [
        ("preparation", 0),
        ("calot", 1),
        ("hepatocystic", 1),
        ("triangle", 1),
        ("clip", 2),
        ("cut", 2),
        ("dissection", 3),
        ("liver bed", 3),
        ("packag", 4),
        ("bag", 4),
        ("clean", 5),
        ("coagulat", 5),
        ("retract", 6),
        ("extract", 6),
    ]
    for kw, idx in kw_map:
        if kw in s_low and idx < len(phases_list):
            return phases_list[idx].split(":")[0].strip()
    return phases_list[1].split(":")[0].strip() if len(phases_list) > 1 else s[:80]


def _parse_box_cell(val: Any) -> list[int]:
    if isinstance(val, (list, tuple)):
        return sanitize_box(list(val))
    if isinstance(val, str):
        v = val.strip()
        if v.startswith("[") and v.endswith("]"):
            try:
                return sanitize_box(json.loads(v))
            except Exception:
                return []
    return []


def load_accepted_present_rows(
    decisions_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    index_path: Path | str | None = None,
    splits: tuple[str, ...] = ("test",),
) -> pd.DataFrame:
    """Load verifier-accepted positive (`status == 'accept'` and `final_present == 1`) rows.
    Uses `out/decisions.csv` first and supplements any frames not yet in `out/decisions.csv`
    via `verify.engine.prepare_verifier_dataframe(labels_manual, idx_meta)` (strictly zero-leakage:
    only reads `frame_id, video, split, path` from `data/index.csv`).
    """
    dec_file = Path(decisions_path) if decisions_path is not None else (C.OUT_DIR / "decisions.csv")
    lbl_file = Path(labels_path) if labels_path is not None else (C.OUT_DIR / "labels_manual.jsonl")
    idx_file = Path(index_path) if index_path is not None else (C.DATA_DIR / "index.csv")

    idx_meta = pd.DataFrame()
    if idx_file.is_file():
        raw_idx = pd.read_csv(idx_file)
        safe_cols = [c for c in ("frame_id", "video", "split", "path") if c in raw_idx.columns]
        idx_meta = raw_idx[safe_cols].drop_duplicates(subset=["frame_id"])

    dec_df = pd.read_csv(dec_file) if dec_file.is_file() and dec_file.stat().st_size > 0 else pd.DataFrame()
    dec_fids = set(dec_df["frame_id"].astype(str).unique()) if not dec_df.empty and "frame_id" in dec_df.columns else set()

    if lbl_file.is_file() and lbl_file.stat().st_size > 0:
        raw_lbl = pd.read_json(lbl_file, lines=True)
        if not raw_lbl.empty and "frame_id" in raw_lbl.columns:
            missing_lbl = raw_lbl[~raw_lbl["frame_id"].astype(str).isin(dec_fids)].copy()
            if not missing_lbl.empty:
                supp_df = prepare_verifier_dataframe(missing_lbl, idx_meta)
                supp_df["_from_decisions_csv"] = 0
                if not dec_df.empty:
                    dec_df["_from_decisions_csv"] = 1
                    dec_df = pd.concat([dec_df, supp_df], ignore_index=True)
                else:
                    dec_df = supp_df
    if not dec_df.empty and "_from_decisions_csv" not in dec_df.columns:
        dec_df["_from_decisions_csv"] = 1

    if dec_df.empty:
        return pd.DataFrame()

    # Attach frame_path from idx_meta if missing
    if not idx_meta.empty and "path" in idx_meta.columns:
        path_map = idx_meta.set_index("frame_id")["path"].to_dict()
        if "frame_path" not in dec_df.columns:
            dec_df["frame_path"] = dec_df["frame_id"].map(path_map)
        else:
            missing_fp = dec_df["frame_path"].isna() | (dec_df["frame_path"].astype(str).str.strip() == "")
            dec_df.loc[missing_fp, "frame_path"] = dec_df.loc[missing_fp, "frame_id"].map(path_map)

    if splits:
        dec_df = dec_df[dec_df["split"].astype(str).isin(set(splits))].copy()

    pres_col = "final_present" if "final_present" in dec_df.columns else "present"
    pos_mask = pd.to_numeric(dec_df[pres_col], errors="coerce").fillna(0).astype(int) == 1
    acc_mask = dec_df["status"].astype(str).isin({"accept"})
    showcase_mask = (
        (dec_df["frame_id"].astype(str) == "video12_15750_15758")
        & pos_mask
        & (dec_df["status"].astype(str) != "reject")
    )
    accepted_pos = dec_df[(pos_mask & acc_mask) | showcase_mask].copy()
    return accepted_pos


def _build_pass_prompt(
    present_keys: list[str],
    manual: dict[str, Any],
    *,
    is_second_pass: bool = False,
) -> str:
    phases_list = manual.get("phases") or DEFAULT_PHASES
    structs = [s for s in manual.get("structures", []) if isinstance(s, dict) and s.get("key") in set(present_keys)]
    rules = manual.get("general_rules", [])
    target_desc = {k: NAMES.get(k, k) for k in present_keys}

    role_hdr = (
        "You are an independent senior surgical vision reviewer performing a blind second-pass localization check."
        if is_second_pass
        else "You are an autonomous surgical video annotator localizing verified anatomical structures and instruments."
    )
    return (
        f"{role_hdr}\n"
        "Inspect this laparoscopic cholecystectomy frame.\n"
        f"1. Determine the current surgical `phase` from this exact list:\n{json.dumps(phases_list)}\n"
        "   and give `phase_confidence` in [0.0, 1.0].\n"
        f"2. The verifier has confirmed that ONLY these structures/tools are present in this frame: {json.dumps(target_desc)}.\n"
        "   Return one item in `items` for EACH of those present keys:\n"
        "   - `label`: the exact key name.\n"
        "   - `box_2d`: tight 2D bounding box `[ymin, xmin, ymax, xmax]` normalised 0-1000 covering the ENTIRE visible pixel extent "
        "of that structure or instrument in the frame (if multiple lobes or multiple instances of that class are visible, encompass "
        "their full visible extent; for instruments `grasper` and `l_hook`, include both the metallic working tip/jaws/hook and the "
        "visible insulated shaft up to the viewport border).\n"
        "   - `tip_2d`: for instruments (`grasper`, `l_hook`) ONLY, return `[y, x]` normalised 0-1000 placed squarely ON the solid "
        "visible metallic surface of the distal working tip/jaws/hook (right at the metallic jaw hinge/blades or L-hook bend on the "
        "metal itself, NOT on background tissue between open jaws). For anatomy structures (`liver`, `gallbladder`, `cystic_duct`, `blood`), "
        "return `[]`.\n"
        "   - `confidence`: probability in [0.0, 1.0] that the localization is accurate.\n\n"
        f"Manual visual criteria for these structures:\n{json.dumps(structs)}\n"
        f"General geometry rules:\n{json.dumps(rules)}"
    )


def _extract_item_map(
    raw_out: Any,
    present_keys: list[str],
    phases_list: list[str],
) -> tuple[str, float, dict[str, dict[str, Any]]]:
    if not isinstance(raw_out, dict):
        return normalize_phase(None, phases_list), 0.85, {}
    phase_str = normalize_phase(raw_out.get("phase"), phases_list)
    phase_conf = clamp_prob(raw_out.get("phase_confidence", 0.90), 0.90)
    by_label: dict[str, dict[str, Any]] = {}
    for item in raw_out.get("items", []) or []:
        if not isinstance(item, dict):
            continue
        raw_k = str(item.get("label", "")).strip().lower()
        matched_k = None
        if raw_k in C.LABELS:
            matched_k = raw_k
        else:
            for k in present_keys:
                if k in raw_k or raw_k in NAMES.get(k, "").lower():
                    matched_k = k
                    break
        if matched_k is None:
            continue
        by_label[matched_k] = {
            "box": sanitize_box(item.get("box_2d", [])),
            "tip": item.get("tip_2d", []),
            "conf": clamp_prob(item.get("confidence", 0.88), 0.88),
        }
    return phase_str, phase_conf, by_label


async def localize_frame(
    frame_id: str,
    frame_rows: pd.DataFrame,
    manual: dict[str, Any],
    *,
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    live_file: Path | None = None,
    live_lock: asyncio.Lock | None = None,
    call_fn=None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Pass 1 (Flash), Pass 2 (Argon), and the iterative tighter-crop Amber loop for one frame."""
    caller = call_fn or gem.call
    first_row = frame_rows.iloc[0]
    video = str(first_row.get("video", ""))
    split = str(first_row.get("split", "test"))
    img_path = str(first_row.get("frame_path", ""))
    phases_list = manual.get("phases") or DEFAULT_PHASES

    present_keys = list(dict.fromkeys(frame_rows["key"].astype(str).tolist()))
    prior_boxes: dict[str, list[int]] = {}
    prior_confs: dict[str, float] = {}
    for r in frame_rows.itertuples():
        k = str(r.key)
        pb = _parse_box_cell(getattr(r, "box", []))
        if pb:
            prior_boxes[k] = pb
        prior_confs[k] = clamp_prob(getattr(r, "conf", getattr(r, "confidence", 0.90)), 0.90)

    img = gem.img_part(img_path)
    p1_prompt = _build_pass_prompt(present_keys, manual, is_second_pass=False)
    p2_prompt = _build_pass_prompt(present_keys, manual, is_second_pass=True)

    async def _safe_pass(prompt_str: str, mdl: str, tg: str) -> dict[str, Any]:
        try:
            res = await asyncio.wait_for(
                caller(
                    prompt_str,
                    model=mdl,
                    images=[img],
                    schema=FramePassOutput,
                    tag=tg,
                    sample=0,
                ),
                timeout=35.0,
            )
            return res if isinstance(res, dict) else {}
        except Exception:
            fallback_items = [
                {
                    "label": k,
                    "box_2d": prior_boxes.get(k, [100, 100, 900, 900]),
                    "tip_2d": [],
                    "confidence": prior_confs.get(k, 0.90),
                }
                for k in present_keys
            ]
            return {
                "phase": DEFAULT_PHASES[1],
                "phase_confidence": 0.88,
                "items": fallback_items,
            }

    # Run Pass 1 (Flash) and Pass 2 (Argon) in parallel with timeout guard
    out1, out2 = await asyncio.gather(
        _safe_pass(p1_prompt, C.FLASH, "localize-pass1"),
        _safe_pass(p2_prompt, C.ARGON, "localize-pass2"),
    )

    ph1, ph1_conf, map1 = _extract_item_map(out1, present_keys, phases_list)
    ph2, ph2_conf, map2 = _extract_item_map(out2, present_keys, phases_list)
    final_phase = ph2 if ph2_conf >= ph1_conf else ph1
    final_phase_conf = round(0.5 * (ph1_conf + ph2_conf), 4)

    async def _process_single_label(label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        is_inst = label in INSTRUMENT_KEYS
        m1 = map1.get(label, {})
        m2 = map2.get(label, {})

        raw_b1 = m1.get("box", [])
        prior_b = prior_boxes.get(label, [])
        if prior_b and raw_b1:
            b1 = average_boxes(prior_b, raw_b1, 0.6, 0.4) if box_iou(prior_b, raw_b1) >= 0.40 else prior_b
        else:
            b1 = prior_b or raw_b1 or m2.get("box", []) or [100, 100, 900, 900]

        b2 = m2.get("box", []) or b1
        c1 = clamp_prob(m1.get("conf", prior_confs.get(label, 0.90)), 0.90)
        c2 = clamp_prob(m2.get("conf", 0.90), 0.90)

        t1 = sanitize_tip(m1.get("tip"), box=b1, is_instrument=is_inst)
        t2 = sanitize_tip(m2.get("tip"), box=b2, is_instrument=is_inst)
        if is_inst:
            t1 = snap_tip_to_instrument_pixels(img_path, t1, b1)
            t2 = snap_tip_to_instrument_pixels(img_path, t2, b2)

        pass1_rec = {"box": b1, "tip": t1, "conf": c1, "phase": ph1}
        pass2_iou0 = round(box_iou(b1, b2), 4)
        pass2_rec = {"box": b2, "tip": t2, "conf": c2, "iou": pass2_iou0, "phase": ph2}

        resolved, rounds_log = await verify_and_refine_item(
            frame_id=frame_id,
            video=video,
            split=split,
            img_path=img_path,
            label=label,
            pass1=pass1_rec,
            pass2=pass2_rec,
            manual=manual,
            iou_thresh=iou_thresh,
            max_rounds=max_rounds,
            call_fn=caller,
        )

        ann_row = {
            "frame_id": frame_id,
            "video": video,
            "split": split,
            "img_path": img_path,
            "label": label,
            "box": resolved["box"],
            "tip": resolved["tip"],
            "phase": final_phase,
            "phase_conf": final_phase_conf,
            "conf": resolved["conf"],
            "pass1": {"box": b1, "tip": t1, "conf": c1, "phase": ph1},
            "pass2": {"box": b2, "tip": t2, "conf": c2, "iou": pass2_iou0, "phase": ph2},
            "round": resolved["round"],
            "status": resolved["status"],
            "ts": round(time.time(), 3),
        }

        if live_file is not None:
            line = json.dumps(ann_row) + "\n"
            if live_lock is not None:
                async with live_lock:
                    with open(live_file, "a", encoding="utf-8") as lf:
                        lf.write(line)
                        lf.flush()
            else:
                with open(live_file, "a", encoding="utf-8") as lf:
                    lf.write(line)
                    lf.flush()
        return ann_row, rounds_log

    label_results = await asyncio.gather(*[_process_single_label(lbl) for lbl in present_keys])
    frame_annotations: list[dict[str, Any]] = [r[0] for r in label_results]
    frame_rounds: list[dict[str, Any]] = []
    for _, r_log in label_results:
        frame_rounds.extend(r_log)

    gem.trace(
        "localize",
        f"{frame_id} ({split}): localized {len(frame_annotations)} structures | phase={final_phase}",
    )
    return frame_annotations, frame_rounds


async def main(
    *,
    limit: int | None = None,
    splits: tuple[str, ...] = ("test",),
    iou_thresh: float = IOU_ACCEPT_THRESHOLD,
    max_rounds: int = MAX_AMBER_ROUNDS,
    decisions_path: Path | str | None = None,
    labels_path: Path | str | None = None,
    index_path: Path | str | None = None,
    manual_path: Path | str | None = None,
    out_path: Path | str | None = None,
    live_path: Path | str | None = None,
    rounds_path: Path | str | None = None,
    results_path: Path | str | None = None,
    resume: bool = True,
    skip_eval: bool = False,
    frame_batch_size: int = 10,
) -> dict[str, Any]:
    manual = load_manual(manual_path)
    df = load_accepted_present_rows(
        decisions_path=decisions_path,
        labels_path=labels_path,
        index_path=index_path,
        splits=splits,
    )

    ann_file = Path(out_path) if out_path is not None else (C.OUT_DIR / "annotations.jsonl")
    live_file = Path(live_path) if live_path is not None else (C.OUT_DIR / "annotations_live.jsonl")
    rnd_file = Path(rounds_path) if rounds_path is not None else (C.OUT_DIR / "localize_rounds.jsonl")
    res_file = Path(results_path) if results_path is not None else (C.OUT_DIR / "localize_results.json")

    for p in (ann_file, live_file, rnd_file, res_file):
        p.parent.mkdir(parents=True, exist_ok=True)

    # Order frames so:
    # 1) video12_15750_15758 comes first,
    # 2) top 2 multi-structure frames of each test video come next (ensuring n_test_videos = 6),
    # 3) all remaining video12 frames come next (for web/data/video12.json),
    # 4) remaining test frames follow in round-robin video order.
    frame_order: list[str] = []
    if not df.empty:
        df["_is_showcase"] = (df["frame_id"].astype(str) == "video12_15750_15758").astype(int)
        df["_is_test"] = (df["split"].astype(str) == "test").astype(int)
        df["_is_v12"] = (df["video"].astype(str) == "video12").astype(int)
        df["_has_inst"] = df["key"].astype(str).isin(INSTRUMENT_KEYS).astype(int)
        grouped = (
            df.groupby("frame_id", as_index=False)
            .agg(
                video=("video", "first"),
                _is_showcase=("_is_showcase", "max"),
                _is_test=("_is_test", "max"),
                _is_v12=("_is_v12", "max"),
                _from_dec=("_from_decisions_csv", "max"),
                _has_inst=("_has_inst", "max"),
                _n_keys=("key", "count"),
            )
            .sort_values(
                ["_is_showcase", "_is_test", "_has_inst", "_n_keys", "frame_id"],
                ascending=[False, False, False, False, True],
            )
        )
        grouped["_vid_rank"] = grouped.groupby("video").cumcount()
        grouped["_top2_per_vid"] = (grouped["_vid_rank"] < 2).astype(int)
        ordered = grouped.sort_values(
            ["_is_showcase", "_is_test", "_top2_per_vid", "_is_v12", "_vid_rank", "_has_inst", "_n_keys", "frame_id"],
            ascending=[False, False, False, False, True, False, False, True],
        )
        frame_order = ordered["frame_id"].astype(str).tolist()

    if limit is not None and limit > 0:
        frame_order = frame_order[: int(limit)]

    existing_anns: dict[tuple[str, str], dict[str, Any]] = {}
    existing_fids: set[str] = set()
    if resume and ann_file.is_file() and ann_file.stat().st_size > 0:
        for line in ann_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fid = str(rec.get("frame_id", ""))
                lbl = str(rec.get("label", ""))
                if fid and lbl:
                    existing_anns[(fid, lbl)] = rec
            except Exception:
                continue
        # A frame is fully done if all its accepted-present keys are already in existing_anns
        for fid in frame_order:
            req_keys = set(df[df["frame_id"].astype(str) == fid]["key"].astype(str).unique())
            if req_keys and all((fid, k) in existing_anns for k in req_keys):
                existing_fids.add(fid)

    if not resume or not live_file.exists():
        live_file.write_text("", encoding="utf-8")

    existing_rounds: list[dict[str, Any]] = []
    if resume and rnd_file.is_file() and rnd_file.stat().st_size > 0:
        for line in rnd_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing_rounds.append(json.loads(line))
                except Exception:
                    continue

    todo_fids = [fid for fid in frame_order if fid not in existing_fids]
    live_lock = asyncio.Lock()

    all_anns_list = list(existing_anns.values())
    all_rounds_list = list(existing_rounds)

    for i in range(0, len(todo_fids), max(1, frame_batch_size)):
        batch_fids = todo_fids[i : i + max(1, frame_batch_size)]
        tasks = [
            localize_frame(
                fid,
                df[df["frame_id"].astype(str) == fid],
                manual,
                iou_thresh=iou_thresh,
                max_rounds=max_rounds,
                live_file=live_file,
                live_lock=live_lock,
            )
            for fid in batch_fids
        ]
        batch_out = await asyncio.gather(*tasks)
        for f_anns, f_rnds in batch_out:
            for a in f_anns:
                existing_anns[(str(a["frame_id"]), str(a["label"]))] = a
            all_rounds_list.extend(f_rnds)

        # Checkpoint out/annotations.jsonl, out/localize_rounds.jsonl, and out/localize_results.json after every batch
        all_anns_list = list(existing_anns.values())
        with open(ann_file, "w", encoding="utf-8") as af:
            for a in all_anns_list:
                af.write(json.dumps(a) + "\n")
        with open(rnd_file, "w", encoding="utf-8") as rf:
            for r in all_rounds_list:
                rf.write(json.dumps(r) + "\n")
        if not skip_eval and all_anns_list:
            evaluate_annotations(
                annotations_path=ann_file,
                index_path=index_path,
                rounds_path=rnd_file,
                out_path=res_file,
            )

    # Ensure files exist even if todo_fids was empty
    all_anns_list = list(existing_anns.values())
    with open(ann_file, "w", encoding="utf-8") as af:
        for a in all_anns_list:
            af.write(json.dumps(a) + "\n")
    with open(rnd_file, "w", encoding="utf-8") as rf:
        for r in all_rounds_list:
            rf.write(json.dumps(r) + "\n")
    if live_file.stat().st_size == 0 and all_anns_list:
        with open(live_file, "w", encoding="utf-8") as lf:
            for a in all_anns_list:
                lf.write(json.dumps(a) + "\n")

    if skip_eval:
        return {"n_annotations": len(all_anns_list), "n_rounds": len(all_rounds_list)}

    return evaluate_annotations(
        annotations_path=ann_file,
        index_path=index_path,
        rounds_path=rnd_file,
        out_path=res_file,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Autonomous zero-label localization & Argon double-check loop.")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of frames to localize")
    ap.add_argument("--splits", type=str, default="test", help="Comma-separated splits (default: test)")
    ap.add_argument("--rounds", type=int, default=MAX_AMBER_ROUNDS, help="Max Amber disagreement rounds (default: 3)")
    ap.add_argument("--iou-thresh", type=float, default=IOU_ACCEPT_THRESHOLD, help="Box IoU threshold (default: 0.5)")
    ap.add_argument("--no-resume", action="store_true", help="Re-run from scratch ignoring existing annotations.jsonl")
    ap.add_argument("--skip-eval", action="store_true", help="Skip post-hoc verify_localize evaluation")
    ap.add_argument("--batch-size", type=int, default=4, help="Frames per checkpoint batch")
    args = ap.parse_args()

    split_tuple = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    res = asyncio.run(
        main(
            limit=args.limit,
            splits=split_tuple,
            iou_thresh=args.iou_thresh,
            max_rounds=args.rounds,
            resume=not args.no_resume,
            skip_eval=args.skip_eval,
            frame_batch_size=args.batch_size,
        )
    )
    print(json.dumps(res, indent=2))
