"""Stage 9: the dataset, its provenance, the human final pass, and the signed release.

export.run writes dataset_coco.json (accepted labels only; ambers are excluded) and provenance.jsonl, and draws the
final-pass queue. That file is a draft: info.released is false. Only release() writes released_coco.json.

Final pass. A human clears every amber plus a random audit stratified by label across four pools: labels both passes
accepted, labels only the amber loop accepted, labels the AI reported absent, and propagated frames. Amber-accepted
labels had no independent agreement and feed propagation, so they are all reviewed up to a per-label cap. The draw is
persisted, so it cannot be re-rolled.
Actions describe the object: approve means the AI's mask is right, reject means the object is not present, correct
gives the human's own outline. An audit item fails when the human contradicts the AI. Above AUDIT_FAIL_THRESHOLD the
job escalates to review of every keyframe label, and that escalation cannot be undone by changing verdicts afterwards.
An empty queue never releases.

Meaning of the review fields:
  human_reviewed         a human saw this exact annotation, on this frame.
  derived_from_reviewed  a propagated annotation whose source keyframes were all reviewed. The frame itself was not.
Propagated frames come from a keyframe's mask, so when a human rejects or corrects that keyframe, or fails a propagated
frame from the same run, the run is dropped from the release rather than kept on a mask that was overruled.
Frames that failed the flow cross-check stay excluded: approving a keyframe does not make two flows agree.
Exclusions stay listed on each image, so an excluded label never reads as a negative.

Release writes annotated.mp4 (re-rendered from the released data), dataset_bundle.zip and a signed manifest. All of it
is built first and committed last, so a failure leaves the job unreleased. The signature is an HMAC with a key kept
outside the job folder. It detects edits by anyone without that key; on one user's machine that is not tamper-proofing.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import random
import secrets
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .masks import (ACCEPT, AMBER, LABELS, MASK_QUALITY, MIN_SEED_AREA, REJECT, atomic_write, clean_polygon, frame_size,
                    mask_box, mask_path, polygon_mask, progress_fn, read_json, read_jsonl, read_mask, resolve_decisions,
                    track_path, write_json, write_jsonl)
from .propagate import find_video

INSTRUMENTS = ("grasper", "l_hook")
CATEGORIES = [{"id": i + 1, "name": name, "supercategory": "instrument" if name in INSTRUMENTS else "anatomy"}
              for i, name in enumerate(LABELS)]
CATEGORY_ID = {c["name"]: c["id"] for c in CATEGORIES}
POLY_EPSILON = 1.0        # px, contour simplification
AUDIT_FRAC = 0.10
MIN_AUDIT = 5
MAX_PROPAGATED_AUDIT_PER_LABEL = 10
MAX_AMBER_ACCEPTED_AUDIT_PER_LABEL = 20
ROUTE_AMBER_ACCEPTED = "amber_said_present"
AUDIT_FAIL_THRESHOLD = 0.10
PROP_PREFIX = "prop_"     # pseudo keyframe id of a propagated frame in the queue: frames/ and masks/ hold real files for it

APPROVE, CORRECT = "approve", "correct"
ACTIONS = {APPROVE, REJECT, CORRECT}
SYNONYMS = {"accept": APPROVE, "approved": APPROVE, "rejected": REJECT, "corrected": CORRECT}
STRATUM_ACCEPTED, STRATUM_AMBER_ACCEPTED, STRATUM_ABSENT, STRATUM_PROPAGATED = "accepted", "amber_accepted", "ai_absent", "propagated"
KIND_AMBER, KIND_AUDIT, KIND_ESCALATION = "amber", "audit", "escalation"
SOURCE_AI, SOURCE_FLOW, SOURCE_HUMAN = "ai_polygon", "optical_flow_from_ai_keyframes", "human_correction"
SCOPE = ("Single-user research tool for laparoscopic cholecystectomy, on public or simulator footage only. "
         "No real patient video. Not for clinical use.")

RELEASE_ARTIFACTS = ("released_coco.json", "release_manifest.json", "dataset_bundle.zip")
HUMAN_ARTIFACTS = ("final_pass_queue.json", "final_pass.jsonl") + RELEASE_ARTIFACTS
MANIFEST_INPUTS = ("provenance.jsonl", "final_pass.jsonl", "final_pass_queue.json", "annotated.mp4", "dataset_bundle.zip",
                   "decisions.jsonl", "detect.jsonl", "verify.jsonl", "masks.jsonl", "tracks.jsonl", "phase.jsonl",
                   "narration.json", "meta.json")


class ReleaseLocked(RuntimeError):
    """Release was refused: items unresolved, audit failed, nothing to review, or already released."""


# ---- geometry ----

def mask_to_polygons(mask):
    """COCO polygon list (flat [x, y, ...] per outer contour). Polygons cannot hold holes; `area` uses the mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        c = cv2.approxPolyDP(c, POLY_EPSILON, True)
        if len(c) >= 3:
            polys.append([int(v) for v in c.reshape(-1)])
    return polys


def coco_bbox(box):
    """[x0, y0, x1, y1] inclusive pixel bounds to COCO [x, y, width, height]."""
    return [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1]


def estimate_tip(mask):
    """Instrument tip estimate from geometry alone: the end of the mask's long axis nearest the image centre.

    Tools enter from the edge of the view, so the working end is the inner one. This is a heuristic with no ground truth
    behind it, and the model does not report tips. Returns [x, y], or None for an empty mask."""
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return None
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    along = (pts - centre) @ vt[0]
    lo, hi = np.quantile(along, [0.02, 0.98])
    ends = [pts[along <= lo].mean(axis=0), pts[along >= hi].mean(axis=0)]
    middle = np.array([mask.shape[1] / 2, mask.shape[0] / 2])
    tip = min(ends, key=lambda e: float(np.hypot(*(e - middle))))
    return [round(float(tip[0]), 1), round(float(tip[1]), 1)]


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---- loading ----

class JobRecords:
    def __init__(self, job):
        self.job = Path(job)
        self.meta = read_json(self.job / "meta.json")
        self.size = frame_size(self.job, self.meta)
        self.decisions = resolve_decisions(self.job)
        self.detect = {(r["kf"], r["label"]): r for r in read_jsonl(self.job / "detect.jsonl")}
        self.verify = {(r["kf"], r["label"]): r for r in read_jsonl(self.job / "verify.jsonl")}
        self.mask_rows = {(r["kf"], r["label"]): r for r in read_jsonl(self.job / "masks.jsonl")}
        self.tracks = read_jsonl(self.job / "tracks.jsonl")
        self.kf_index = {k["kf"]: k["frame_index"] for k in self.meta["keyframes"]}
        self.fps = float(self.meta.get("fps") or 0) or None

    def frame_of(self, kf):
        """Frame index of a queue key: a keyframe id, or the pseudo id of a propagated frame."""
        return int(kf[len(PROP_PREFIX):]) if kf.startswith(PROP_PREFIX) else self.kf_index[kf]


def _image_entry(rec, frame_index, kf=None, **attrs):
    w, h = rec.size
    attributes = {"keyframe": kf, "extracted": bool(kf), "exhaustive": False, "ai_absent": [], "human_absent": [],
                  "excluded_amber": [], "excluded_overruled": [], "excluded_no_outline": []}
    attributes.update(attrs)
    if kf:
        name = f"frames/{kf}.jpg"
    else:
        name = f"frames_from_video/frame_{frame_index:06d}.jpg"
        attributes["decode_from"] = {"video": "source.mp4", "frame_index": frame_index}
    return {"id": frame_index + 1, "width": w, "height": h, "frame_index": frame_index,
            "t": round(frame_index / rec.fps, 4) if rec.fps else None, "file_name": name, "attributes": attributes}


def _model_trace(rec, sources, label):
    """Model and prompt hash for an annotation's source keyframes, read from what stage 5 recorded.

    Single values when the sources agree, otherwise lists in source_keyframes order."""
    models, hashes = [], []
    for kf in sources:
        row = rec.mask_rows.get((kf, label)) or {}
        det = rec.detect.get((kf, label)) or {}
        models.append(row.get("model") or det.get("model"))
        hashes.append(row.get("prompt_hash") or det.get("prompt_hash"))
    return (models[0], hashes[0]) if len(set(models)) == 1 and len(set(hashes)) == 1 else (models, hashes)


def _source_conf(rec, sources, label):
    """The model's self-reported confidence: the lowest among the source keyframes. Not calibrated."""
    confs = [(rec.detect.get((kf, label)) or {}).get("conf") for kf in sources]
    confs = [c for c in confs if isinstance(c, (int, float))]
    return min(confs) if confs else None


def _instance_ids(tracks):
    """One id per label per unbroken run of frames within a shot. {(frame_index, label): id}, plus the next free id.

    The pipeline traces one outline per label, so two tools of one class share an instance. That is a known limit."""
    ids, next_id = {}, 1
    by_label = {}
    for t in tracks:
        by_label.setdefault(t["label"], []).append(t)
    for label in sorted(by_label):
        prev, current = None, None
        for t in sorted(by_label[label], key=lambda r: r["frame_index"]):
            if prev is None or t["frame_index"] != prev["frame_index"] + 1 or t.get("shot", 0) != prev.get("shot", 0):
                current, next_id = next_id, next_id + 1
            ids[(t["frame_index"], label)] = current
            prev = t
    return ids, next_id


# ---- dataset ----

def build_coco(rec, final=None):
    """Build the COCO dict. `final` is the resolved {(kf, label): row} from the final pass, or None for the draft."""
    final = final or {}
    kf_final = {k: v for k, v in final.items() if not k[0].startswith(PROP_PREFIX)}
    frame_final = {(rec.frame_of(k[0]), k[1]): v for k, v in final.items() if k[0].startswith(PROP_PREFIX)}
    w, h = rec.size
    images, annotations = {}, []
    counts = {"n_keyframe_annotations": 0, "n_propagated_annotations": 0, "n_excluded_amber_keyframe": 0,
              "n_excluded_amber_propagated": 0, "n_excluded_failed_flow_check": 0,
              "n_propagated_dropped_after_human_overrule": 0, "n_human_corrections": 0}
    instances, next_instance = _instance_ids(rec.tracks)

    def instance_of(frame_index, label):
        nonlocal next_instance
        if (frame_index, label) not in instances:
            instances[(frame_index, label)] = next_instance
            next_instance += 1
        return instances[(frame_index, label)]

    def image(frame_index):
        if frame_index not in images:
            images[frame_index] = _image_entry(rec, frame_index)
        return images[frame_index]

    ann_id = 0

    def add(frame_index, label, mask, attributes):
        nonlocal ann_id
        polys = mask_to_polygons(mask)
        box = mask_box(mask)
        if not polys or box is None:
            return False
        ann_id += 1
        attributes = {"mask_quality": MASK_QUALITY, "derived_from_reviewed": False,
                      "instance_id": instance_of(frame_index, label), **attributes}
        annotations.append({"id": ann_id, "image_id": frame_index + 1, "category_id": CATEGORY_ID[label],
                            "segmentation": polys, "bbox": coco_bbox(box), "area": int(mask.sum()), "iscrowd": 0,
                            "attributes": attributes})
        return True

    # Every keyframe is an image, including empty ones: they record what was reported absent, and by whom.
    for k in rec.meta["keyframes"]:
        kf = k["kf"]
        attrs = {"ai_absent": [], "human_absent": [], "excluded_amber": [], "excluded_no_outline": []}
        for label in LABELS:
            dec = rec.decisions.get((kf, label))
            mrow = rec.mask_rows.get((kf, label))
            f = kf_final.get((kf, label))
            action = f["action"] if f else None
            status = dec["status"] if dec else (mrow or {}).get("status")
            has_mask = bool((mrow or {}).get("path"))
            ai_absent = dec is not None and status in (ACCEPT, REJECT) and not dec.get("final_present")
            if action == CORRECT:
                continue
            if action == REJECT or (action == APPROVE and not has_mask):
                attrs["human_absent"].append(label)
            elif ai_absent:
                attrs["ai_absent"].append(label)
            elif status == AMBER and action is None:
                attrs["excluded_amber"].append(label)
            elif status == ACCEPT and not has_mask:
                attrs["excluded_no_outline"].append(label)  # present by verdict, with nothing to draw
        attrs["exhaustive"] = not (attrs["excluded_amber"] or attrs["excluded_no_outline"])
        images[k["frame_index"]] = _image_entry(rec, k["frame_index"], kf, **attrs)

    # Keyframes.
    overruled = set()
    for (kf, label), row in sorted(rec.mask_rows.items()):
        f = kf_final.get((kf, label))
        action = f["action"] if f else None
        if action in (REJECT, CORRECT):
            overruled.add((kf, label))
        if action in (REJECT, CORRECT) or not row.get("path"):
            continue  # a correction is added below from the human's own polygon
        status = row["status"]
        if status == AMBER and action != APPROVE:
            counts["n_excluded_amber_keyframe"] += 1
            continue
        if status != ACCEPT and action != APPROVE:
            continue
        model, prompt_hash = _model_trace(rec, [kf], label)
        dec = rec.decisions.get((kf, label)) or {}
        attrs = {"status": ACCEPT, "source": SOURCE_AI,
                 "refinement": f"none:{row['fallback']}" if row.get("fallback") else "grabcut",
                 "keyframe": kf, "propagated": False, "model": model, "prompt_hash": prompt_hash, "seed": row.get("seed"),
                 "conf": _source_conf(rec, [kf], label), "decision_route": dec.get("route"),
                 "ai_status": status, "human_reviewed": action == APPROVE}
        if add(rec.kf_index[kf], label, read_mask(rec.job / row["path"]), attrs):
            counts["n_keyframe_annotations"] += 1

    # Human corrections, on keyframes or on audited propagated frames.
    for (kf, label), f in sorted(final.items()):
        if f["action"] != CORRECT:
            continue
        poly = clean_polygon(f.get("polygon"), w, h)
        if poly is None:
            raise ValueError(f"correction for {kf}/{label} has no usable polygon")
        is_frame = kf.startswith(PROP_PREFIX)
        if not is_frame:
            overruled.add((kf, label))
        fi = rec.frame_of(kf)
        attrs = {"status": ACCEPT, "source": SOURCE_HUMAN, "refinement": "none", "keyframe": None if is_frame else kf,
                 "propagated": False, "model": None, "prompt_hash": None, "conf": None, "human_reviewed": True,
                 "reviewer": f.get("reviewer")}
        image(fi)
        if add(fi, label, polygon_mask(poly, w, h).astype(bool), attrs):
            counts["n_keyframe_annotations"] += 1
            counts["n_human_corrections"] += 1

    def source_ok(kf, label):
        """An AMBER keyframe that a human approved no longer blocks the frames that flow from it."""
        f = kf_final.get((kf, label))
        if f:
            return f["action"] == APPROVE
        row = rec.mask_rows.get((kf, label))
        return bool(row) and row["status"] == ACCEPT

    # A failed propagated frame condemns its whole run: the same flow between the same keyframes produced them all.
    failed_runs = set()
    by_frame = {(t["frame_index"], t["label"]): t for t in rec.tracks if t["propagated"]}
    for key, f in frame_final.items():
        if f["action"] in (REJECT, CORRECT) and key in by_frame:
            failed_runs.add((key[1], tuple(by_frame[key]["source_kfs"])))

    # Propagated frames. Every tracked frame gets an image, so what was left out stays visible on it.
    for t in rec.tracks:
        if not t["propagated"]:
            continue
        label, sources, fi = t["label"], t["source_kfs"], t["frame_index"]
        img = image(fi)
        f = frame_final.get((fi, label))
        if f and f["action"] == CORRECT:
            continue  # already added from the human's polygon
        if any((kf, label) in overruled for kf in sources) or (label, tuple(sources)) in failed_runs:
            img["attributes"]["excluded_overruled"].append(label)
            # The frame the human actually rejected is a human verdict, not a side effect, so it is not counted as one.
            if f and f["action"] == REJECT:
                img["attributes"]["human_absent"].append(label)
            else:
                counts["n_propagated_dropped_after_human_overrule"] += 1
            continue
        if t.get("flow_check") != "ok":
            img["attributes"]["excluded_amber"].append(label)
            counts["n_excluded_failed_flow_check"] += 1
            continue
        if not all(source_ok(kf, label) for kf in sources):
            img["attributes"]["excluded_amber"].append(label)
            counts["n_excluded_amber_propagated"] += 1
            continue
        model, prompt_hash = _model_trace(rec, sources, label)
        reviewed_sources = [kf for kf in sources if (kf, label) in kf_final]
        attrs = {"status": ACCEPT, "source": SOURCE_FLOW, "refinement": "none", "keyframe": None, "propagated": True,
                 "source_keyframes": sources, "model": model, "prompt_hash": prompt_hash,
                 "decision_route": [(rec.decisions.get((kf, label)) or {}).get("route") for kf in sources],
                 "conf": _source_conf(rec, sources, label), "fb_overlap": t.get("fb_overlap"),
                 "human_reviewed": bool(f and f["action"] == APPROVE),
                 "derived_from_reviewed": len(reviewed_sources) == len(sources),
                 "reviewed_source_keyframes": reviewed_sources}
        if add(fi, label, read_mask(track_path(rec.job, fi, label)), attrs):
            counts["n_propagated_annotations"] += 1

    info = {"description": "SurgProof AI-annotated dataset (draft masks)", "version": "2.0", "scope": SCOPE,
            "mask_quality": MASK_QUALITY, "date_created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "released": False, "human_final_pass": bool(final), "n_images": len(images),
            "n_annotations": len(annotations), **counts,
            "frames_note": "Images with attributes.extracted false are not stored as files: decode them by frame_index "
                           "from attributes.decode_from.video, or run tools/extract_frames.py in the bundle.",
            "negatives_note": "A label missing from an image is a negative only if listed in ai_absent or human_absent. "
                              "Propagated frames are never exhaustive: see excluded_amber and excluded_overruled.",
            "instances_note": "One instance per label per unbroken run of frames in a shot. Two tools of one class share it.",
            "polygons_note": "Polygons hold outer contours only. area is the true mask pixel count, holes excluded."}
    return {"info": info, "licenses": [], "categories": CATEGORIES,
            "images": [images[i] for i in sorted(images)], "annotations": annotations}


def build_provenance(rec, final=None):
    """One row per keyframe x label, whatever its outcome, plus a header and a row for each human action."""
    final = final or {}
    rows = []
    try:
        video = find_video(rec.job, rec.meta)
        header_video = {"file": video.name, "sha256": _sha256(video)}
    except FileNotFoundError:
        header_video = None
    rows.append({"type": "job", "video": header_video, "fps": rec.meta.get("fps"), "n_frames": rec.meta.get("n_frames"),
                 "kf_fps": rec.meta.get("kf_fps"), "n_keyframes": len(rec.meta["keyframes"]),
                 "masks": "polygon + GrabCut, no ground truth read",
                 "propagation": "Farneback flow forward and backward blended, cross-checked by overlap, "
                                "5-frame majority, 7-frame box average"})
    for k in rec.meta["keyframes"]:
        kf = k["kf"]
        for label in LABELS:
            det, ver, dec = rec.detect.get((kf, label)), rec.verify.get((kf, label)), rec.decisions.get((kf, label))
            mrow = rec.mask_rows.get((kf, label))
            if not (det or ver or dec or mrow):
                continue
            mask_info = None
            if mrow and mrow.get("path"):
                mask_info = {"path": mrow["path"], "sha256": _sha256(rec.job / mrow["path"]), "seed": mrow.get("seed"),
                             "fallback": mrow.get("fallback"), "area": mrow.get("area"), "box": mrow.get("box"),
                             "quality": MASK_QUALITY}
            detect_trace = {k2: det.get(k2) for k2 in ("present", "conf", "model", "prompt_hash", "raw_ref")} if det else None
            verify_trace = {k2: ver.get(k2) for k2 in ("present", "box", "model", "prompt_hash")} if ver else None
            rows.append({"type": "keyframe_label", "kf": kf, "frame_index": k["frame_index"], "label": label,
                         "status": dec["status"] if dec else None, "final_present": dec.get("final_present") if dec else None,
                         "route": dec.get("route") if dec else None, "rounds": dec.get("rounds") if dec else None, "detect": detect_trace, "verify": verify_trace,
                         "mask": mask_info, "human_action": (final.get((kf, label)) or {}).get("action")})
    # Propagated runs, collapsed: one row per label and source pair, not one per frame.
    runs = {}
    for t in rec.tracks:
        if t["propagated"]:
            key = (t["label"], tuple(t["source_kfs"]))
            r = runs.setdefault(key, {"type": "propagated_run", "label": t["label"], "source_kfs": t["source_kfs"],
                                      "first_frame": t["frame_index"], "last_frame": t["frame_index"], "n_frames": 0,
                                      "status": t["status"], "flow_checks": {}, "min_fb_overlap": None})
            r["first_frame"] = min(r["first_frame"], t["frame_index"])
            r["last_frame"] = max(r["last_frame"], t["frame_index"])
            r["n_frames"] += 1
            check = t.get("flow_check", "none")
            r["flow_checks"][check] = r["flow_checks"].get(check, 0) + 1
            if t.get("fb_overlap") is not None:
                low = r["min_fb_overlap"]
                r["min_fb_overlap"] = t["fb_overlap"] if low is None else min(low, t["fb_overlap"])
            if t["status"] == AMBER:
                r["status"] = AMBER
    rows.extend(runs.values())
    for (kf, label), f in sorted(final.items()):
        rows.append({"type": "final_pass", **f, "kf": kf, "label": label})
    return rows


def run(job_dir, on_progress=None):
    """Run stage 9: write the draft dataset and provenance, and draw the queue. Returns what is in and what is out."""
    job = Path(job_dir)
    progress = progress_fn(job, "export", on_progress)
    clear_stale_final_pass(job)
    rec = JobRecords(job)
    progress(0.1, "building COCO annotations")
    coco = build_coco(rec)
    write_json(job / "dataset_coco.json", coco)
    progress(0.8, "writing provenance")
    # A released job's provenance is part of what its manifest signs.
    if not (job / "released_coco.json").exists():
        write_jsonl(job / "provenance.jsonl", build_provenance(rec))
    final_pass_queue(job)
    info = coco["info"]
    progress(1.0, f"{info['n_annotations']} annotations on {info['n_images']} images, "
                  f"{info['n_excluded_amber_keyframe']} amber keyframe labels excluded")
    return {k: info[k] for k in ("n_images", "n_annotations", "n_keyframe_annotations", "n_propagated_annotations",
                                 "n_excluded_amber_keyframe", "n_excluded_amber_propagated", "n_excluded_failed_flow_check")}


# ---- human final pass ----

def _queue_seed(job):
    """A seed fixed by the job's own recorded decisions, so the draw is reproducible from the job dir."""
    path = Path(job) / "decisions.jsonl"
    return int(_sha256(path)[:12], 16) if path.exists() else 0


def clear_stale_final_pass(job_dir):
    """Drop human sign-off that does not belong to the current decisions, so a re-run cannot inherit it.

    The queue records which decisions it was drawn from; a final pass with no matching queue is not this run's."""
    job = Path(job_dir)
    queue = job / "final_pass_queue.json"
    current = queue.exists() and read_json(queue).get("seed") == _queue_seed(job)
    if not current:
        for name in HUMAN_ARTIFACTS:
            (job / name).unlink(missing_ok=True)
        for folder in ("frames", "masks"):
            if (job / folder).is_dir():
                for stale in (job / folder).glob(f"{PROP_PREFIX}*"):
                    stale.unlink()


def _amber_accepted(dec):
    """Accepted by the amber loop rather than by two passes agreeing."""
    return dec.get("route") == ROUTE_AMBER_ACCEPTED or (dec.get("route") is None and bool(dec.get("rounds")))


def _pools(rec):
    """Ambers and the audit pools. Keyframe items are (keyframe record, label); propagated items are track rows."""
    accepted, amber_accepted, absent, ambers = [], [], [], []
    for k in rec.meta["keyframes"]:
        kf = k["kf"]
        for label in LABELS:
            row = rec.mask_rows.get((kf, label))
            dec = rec.decisions.get((kf, label))
            status = (dec or {}).get("status") or (row or {}).get("status")
            if status == AMBER:
                ambers.append((k, label))
            elif status == ACCEPT and row and row.get("path"):
                (amber_accepted if _amber_accepted(dec or {}) else accepted).append((k, label))
            elif status in (ACCEPT, REJECT) and dec and not dec.get("final_present"):
                absent.append((k, label))
    propagated = [t for t in rec.tracks if t["propagated"] and t.get("flow_check") == "ok" and t["status"] == ACCEPT]
    return ambers, accepted, amber_accepted, absent, propagated


def _sample_by_label(rng, pool, label_of, frac, cap=None):
    by_label = {}
    for item in pool:
        by_label.setdefault(label_of(item), []).append(item)
    picked = []
    for label in sorted(by_label):
        n = max(1, int(np.ceil(frac * len(by_label[label]))))
        n = n if cap is None else min(n, cap)
        picked.extend(rng.sample(by_label[label], min(n, len(by_label[label]))))
    return picked


def _keyframe_item(rec, kind, stratum, k, label):
    kf = k["kf"]
    row, dec = rec.mask_rows.get((kf, label)) or {}, rec.decisions.get((kf, label)) or {}
    return {"id": f"{kf}:{label}", "kind": kind, "stratum": stratum, "kf": kf, "frame_index": k["frame_index"], "t": k.get("t"),
            "label": label, "status": dec.get("status") or row.get("status"), "frame": f"frames/{kf}.jpg",
            "mask": row.get("path"), "box": row.get("box"), "conf": (rec.detect.get((kf, label)) or {}).get("conf"),
            "rounds": dec.get("rounds")}


def _materialise_propagated(rec, tracks):
    """Real frame and mask files for each audited propagated frame, so a human sees exactly what is being judged."""
    if not tracks:
        return []
    video = find_video(rec.job, rec.meta)
    wanted = sorted({t["frame_index"] for t in tracks})
    cap = cv2.VideoCapture(str(video))
    w, h = rec.size
    pos, found = 0, set()
    try:
        for fi in wanted:
            frame = None
            while pos <= fi:
                ok, frame = cap.read()
                if not ok:
                    frame = None
                    break
                pos += 1
            if frame is None:
                break
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(rec.job / "frames" / f"{PROP_PREFIX}{fi:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            found.add(fi)
    finally:
        cap.release()
    items = []
    for t in tracks:
        fi, label = t["frame_index"], t["label"]
        if fi not in found:
            raise OSError(f"{video} ended before frame {fi}, which the audit must show")
        kf = f"{PROP_PREFIX}{fi:06d}"
        shutil.copyfile(track_path(rec.job, fi, label), mask_path(rec.job, kf, label))
        items.append({"id": f"{kf}:{label}", "kind": KIND_AUDIT, "stratum": STRATUM_PROPAGATED, "kf": kf, "frame_index": fi,
                      "t": t.get("t"), "label": label, "status": t["status"], "frame": f"frames/{kf}.jpg",
                      "mask": f"masks/{kf}_{label}.png", "box": t.get("box_raw"), "conf": _source_conf(rec, t["source_kfs"], label),
                      "rounds": [], "source_kfs": t["source_kfs"], "fb_overlap": t.get("fb_overlap")})
    return items


def _queue_doc(job):
    return read_json(Path(job) / "final_pass_queue.json")


def final_pass_queue(job_dir, audit_frac=AUDIT_FRAC, min_audit=MIN_AUDIT):
    """Everything a human must clear: all ambers plus an audit of each pool, stratified by label.

    Persisted on first call, so later calls return the same items whatever `audit_frac` they pass. It only ever grows,
    when an audit failure escalates to full review."""
    job = Path(job_dir)
    path = job / "final_pass_queue.json"
    if path.exists():
        return read_json(path)["items"]
    rec = JobRecords(job)
    ambers, accepted, amber_accepted, absent, propagated = _pools(rec)
    rng = random.Random(_queue_seed(job))
    picked_accepted = _sample_by_label(rng, accepted, lambda x: x[1], audit_frac)
    picked_amber = _sample_by_label(rng, amber_accepted, lambda x: x[1], 1.0, cap=MAX_AMBER_ACCEPTED_AUDIT_PER_LABEL)
    picked_absent = _sample_by_label(rng, absent, lambda x: x[1], audit_frac)
    picked_prop = _sample_by_label(rng, propagated, lambda t: t["label"], audit_frac, cap=MAX_PROPAGATED_AUDIT_PER_LABEL)
    # Top up from the accepted pool if the strata came out small.
    short = min_audit - len(picked_accepted) - len(picked_amber) - len(picked_absent) - len(picked_prop)
    if short > 0:
        rest = [x for x in accepted if x not in picked_accepted]
        picked_accepted += rng.sample(rest, min(short, len(rest)))

    audit = [_keyframe_item(rec, KIND_AUDIT, STRATUM_ACCEPTED, *x) for x in picked_accepted]
    audit += [_keyframe_item(rec, KIND_AUDIT, STRATUM_AMBER_ACCEPTED, *x) for x in picked_amber]
    audit += [_keyframe_item(rec, KIND_AUDIT, STRATUM_ABSENT, *x) for x in picked_absent]
    audit += _materialise_propagated(rec, picked_prop)
    rng.shuffle(audit)
    items = [_keyframe_item(rec, KIND_AMBER, "amber", *a) for a in ambers] + audit
    write_json(path, {"seed": _queue_seed(job), "audit_frac": audit_frac, "fail_threshold": AUDIT_FAIL_THRESHOLD,
                      "n_accepted_pool": len(accepted) + len(amber_accepted), "n_amber_accepted_pool": len(amber_accepted),
                      "n_ai_absent_pool": len(absent), "n_propagated_pool": len(propagated),
                      "escalated": False, "items": items})
    return items


def _normalise_action(action):
    action = str(action or "").strip().lower()
    action = SYNONYMS.get(action, action)
    if action not in ACTIONS:
        raise ValueError(f"unknown final-pass action {action!r}")
    return action


def _final_pass_rows(job):
    """Every row of final_pass.jsonl in order. A torn last line, from a crash mid-write, is ignored.

    Corruption anywhere else is an error, never silently skipped."""
    path = Path(job) / "final_pass.jsonl"
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            if n == len(lines) - 1:
                break
            raise ValueError(f"final_pass.jsonl line {n + 1} is corrupt: {exc}") from exc
    return rows


def read_final_pass(job_dir):
    """{(kf, label): row}, latest row winning, so a reviewer can change their mind before release."""
    final = {}
    for r in _final_pass_rows(job_dir):
        try:
            final[(r["kf"], r["label"])] = {**r, "action": _normalise_action(r.get("action"))}
        except (KeyError, TypeError) as exc:
            raise ValueError(f"final_pass.jsonl has a row without kf and label: {exc}") from exc
    return final


def _append_durably(path, rows):
    """Append whole lines and fsync, first cutting off any torn tail so a new row cannot merge into it."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        data = path.read_bytes()
        if not data.endswith(b"\n"):
            with open(path, "r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, "".join(json.dumps(r) + "\n" for r in rows).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    dir_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def _is_failure(item, action):
    """The human contradicted the AI. Reject means not present, so on an AI-absent item it agrees."""
    if item.get("stratum") == STRATUM_ABSENT:
        return action == CORRECT
    return action in (REJECT, CORRECT)


def audit_stats(items, final):
    audit = [i for i in items if i["kind"] == KIND_AUDIT]
    failed = [i for i in audit if (i["kf"], i["label"]) in final and _is_failure(i, final[(i["kf"], i["label"])]["action"])]
    per_label = {}
    for i in audit:
        per_label.setdefault(i["label"], {"n": 0, "failed": 0})["n"] += 1
    for i in failed:
        per_label[i["label"]]["failed"] += 1
    return {"n": len(audit), "failed": len(failed), "rate": round(len(failed) / len(audit), 4) if audit else None,
            "per_label": per_label}


def _audit_failed(stats):
    """Above the threshold overall, or for one label with at least two failures. The denominator is the whole planned
    audit, so this trips as soon as the failures already recorded exceed what the whole audit could allow."""
    if not stats["n"]:
        return None
    if stats["failed"] / stats["n"] > AUDIT_FAIL_THRESHOLD:
        return f"{stats['failed']} of {stats['n']} audit items failed, above the {AUDIT_FAIL_THRESHOLD:.0%} limit"
    for label, s in sorted(stats["per_label"].items()):
        if s["failed"] >= 2 and s["failed"] / s["n"] > AUDIT_FAIL_THRESHOLD:
            return f"{s['failed']} of {s['n']} audit items failed for {label}, above the {AUDIT_FAIL_THRESHOLD:.0%} limit"
    return None


def _maybe_escalate(job):
    """Past the audit limit, every keyframe label joins the queue. Persisted, so later verdicts cannot undo it."""
    doc = _queue_doc(job)
    if doc.get("escalated"):
        return
    reason = _audit_failed(audit_stats(doc["items"], read_final_pass(job)))
    if not reason:
        return
    rec = JobRecords(job)
    _, accepted, amber_accepted, absent, _ = _pools(rec)
    queued = {(i["kf"], i["label"]) for i in doc["items"]}
    extra = [_keyframe_item(rec, KIND_ESCALATION, STRATUM_ACCEPTED, *x) for x in accepted if (x[0]["kf"], x[1]) not in queued]
    extra += [_keyframe_item(rec, KIND_ESCALATION, STRATUM_AMBER_ACCEPTED, *x) for x in amber_accepted if (x[0]["kf"], x[1]) not in queued]
    extra += [_keyframe_item(rec, KIND_ESCALATION, STRATUM_ABSENT, *x) for x in absent if (x[0]["kf"], x[1]) not in queued]
    doc["items"] += extra
    doc.update(escalated=True, escalation_reason=reason)
    write_json(job / "final_pass_queue.json", doc)


def record_final_pass(job_dir, rows):
    """Validate human final-pass rows against the queue, append them durably, and apply the audit stop rule."""
    job = Path(job_dir)
    queue = {(i["kf"], i["label"]) for i in final_pass_queue(job)}
    rec = None
    clean = []
    for r in rows:
        key = (r.get("kf"), r.get("label"))
        if key not in queue:
            raise ValueError(f"{key} is not in this job's final-pass queue")
        action = _normalise_action(r.get("action"))
        row = {"kf": key[0], "label": key[1], "action": action, "reviewer": r.get("reviewer"),
               "ts": r.get("ts") or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        if r.get("note"):
            row["note"] = r["note"]
        if action == CORRECT:
            rec = rec or JobRecords(job)
            w, h = rec.size
            poly = clean_polygon(r.get("polygon"), w, h)
            if poly is None or int(polygon_mask(poly, w, h).sum()) < MIN_SEED_AREA or cv2.contourArea(poly) <= 0:
                raise ValueError(f"correction for {key} needs a polygon that encloses a real area inside the frame")
            row["polygon"] = poly.tolist()  # stored as clipped to the frame
        clean.append(row)
    if clean:
        _append_durably(job / "final_pass.jsonl", clean)
    _maybe_escalate(job)
    return final_pass_state(job)


def final_pass_state(job_dir):
    job = Path(job_dir)
    items = final_pass_queue(job)
    final = read_final_pass(job)
    remaining = [i for i in items if (i["kf"], i["label"]) not in final]
    doc = _queue_doc(job)
    stats = audit_stats(items, final)
    return {"n_required": len(items), "n_done": len(items) - len(remaining), "remaining": remaining,
            "n_amber": sum(1 for i in items if i["kind"] == KIND_AMBER), "n_audit": stats["n"],
            "n_escalation": sum(1 for i in items if i["kind"] == KIND_ESCALATION), "n_audit_failed": stats["failed"],
            "escalated": bool(doc.get("escalated")), "escalation_reason": doc.get("escalation_reason"),
            "empty": not items, "locked": bool(remaining) or not items, "released": (job / "released_coco.json").exists()}


# ---- signed manifest ----

def signing_key_path(job_dir):
    """Beside the job folder, not in it, so it is never bundled or served with the job. SURGPROOF_RELEASE_KEY overrides."""
    override = os.environ.get("SURGPROOF_RELEASE_KEY")
    return Path(override) if override else Path(job_dir).resolve().parent / ".surgproof_release_key"


def _signing_key(job_dir):
    path = signing_key_path(job_dir)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return path.read_bytes()
    with os.fdopen(fd, "wb") as f:
        f.write(secrets.token_bytes(32))
        f.flush()
        os.fsync(f.fileno())
    return path.read_bytes()


def _canonical(doc):
    return json.dumps({k: v for k, v in doc.items() if k != "signature"}, sort_keys=True, separators=(",", ":")).encode()


def _sign(doc, key):
    return {"alg": "HMAC-SHA256", "key_id": hashlib.sha256(key).hexdigest()[:12],
            "value": hmac.new(key, _canonical(doc), hashlib.sha256).hexdigest()}


def _file_entry(path):
    return {"sha256": _sha256(path), "bytes": Path(path).stat().st_size}


def verify_release(job_dir):
    """Check a release against its manifest. {"ok", "problems", "files"}."""
    job = Path(job_dir)
    path = job / "release_manifest.json"
    if not path.exists():
        return {"ok": False, "problems": ["release_manifest.json is missing"], "files": []}
    try:
        doc = read_json(path)
    except ValueError:
        return {"ok": False, "problems": ["release_manifest.json is not valid JSON"], "files": []}
    problems = []
    key_path = signing_key_path(job)
    if not key_path.exists():
        problems.append(f"signature cannot be checked: no key at {key_path}")
    else:
        expected = _sign(doc, key_path.read_bytes())["value"]
        if not hmac.compare_digest(expected, str((doc.get("signature") or {}).get("value"))):
            problems.append("signature does not match the manifest")
    files = doc.get("files") or {}
    for name, entry in sorted(files.items()):
        target = job / name
        if not target.exists():
            problems.append(f"{name} is missing")
        elif _sha256(target) != entry.get("sha256"):
            problems.append(f"{name} was changed after release")
    if "released_coco.json" not in files:
        problems.append("released_coco.json is not covered by the manifest")
    return {"ok": not problems, "problems": problems, "files": sorted(files)}


# ---- dataset bundle ----

_STRINGS = {"type": "array", "items": {"type": "string"}}
ANNOTATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SurgProof released COCO dataset",
    "type": "object",
    "required": ["info", "images", "categories", "annotations"],
    "properties": {
        "info": {"type": "object", "required": ["released", "scope", "mask_quality", "final_pass"],
                 "properties": {"released": {"const": True}, "mask_quality": {"const": MASK_QUALITY}, "scope": {"type": "string"}}},
        "categories": {"type": "array", "items": {"type": "object", "required": ["id", "name"]}},
        "images": {"type": "array", "items": {
            "type": "object", "required": ["id", "width", "height", "frame_index", "file_name", "attributes"],
            "properties": {"attributes": {
                "type": "object",
                "required": ["keyframe", "extracted", "exhaustive", "ai_absent", "human_absent", "excluded_amber",
                             "excluded_overruled", "excluded_no_outline"],
                "properties": {"ai_absent": _STRINGS, "human_absent": _STRINGS, "excluded_amber": _STRINGS,
                               "excluded_overruled": _STRINGS, "excluded_no_outline": _STRINGS}}}}},
        "annotations": {"type": "array", "items": {
            "type": "object", "required": ["id", "image_id", "category_id", "segmentation", "bbox", "area", "iscrowd", "attributes"],
            "properties": {
                "segmentation": {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 6, "items": {"type": "number"}}},
                "bbox": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                "attributes": {
                    "type": "object",
                    "required": ["status", "source", "refinement", "keyframe", "propagated", "model", "prompt_hash", "conf",
                                 "mask_quality", "human_reviewed", "derived_from_reviewed", "instance_id"],
                    "properties": {"status": {"const": ACCEPT}, "source": {"enum": [SOURCE_AI, SOURCE_FLOW, SOURCE_HUMAN]},
                                   "mask_quality": {"const": MASK_QUALITY}, "human_reviewed": {"type": "boolean"},
                                   "derived_from_reviewed": {"type": "boolean"}, "propagated": {"type": "boolean"},
                                   "instance_id": {"type": "integer"}}}}}},
    },
}

EXTRACT_FRAMES_SCRIPT = r'''#!/usr/bin/env python3
"""Decode every image that the COCO file names but the bundle does not store. Needs ffmpeg.
usage: python3 tools/extract_frames.py [bundle_dir]"""
import json, os, subprocess, sys, tempfile

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
coco = json.load(open(os.path.join(root, "annotations", "released_coco.json")))
wanted = sorted((i["frame_index"], i["file_name"]) for i in coco["images"] if not i["attributes"]["extracted"])
if wanted:
    with tempfile.TemporaryDirectory() as tmp:
        script = os.path.join(tmp, "select.txt")
        with open(script, "w") as f:
            f.write("select='" + "+".join("eq(n\\,%d)" % n for n, _ in wanted) + "'")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", os.path.join(root, "source.mp4"),
                        "-filter_script:v", script, "-fps_mode", "passthrough", "-q:v", "2", os.path.join(tmp, "%06d.jpg")], check=True)
        for position, (_, name) in enumerate(wanted, start=1):
            target = os.path.join(root, name)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.replace(os.path.join(tmp, "%06d.jpg" % position), target)
'''


def _code_version():
    """Commit (when this is a git checkout) and the hash of every pipeline source file actually running."""
    src = Path(__file__).resolve().parent
    files = {f"surgproof/{p.name}": _sha256(p) for p in sorted(src.glob("*.py"))}
    commit, dirty = None, None
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=20, check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": commit, "git_dirty": dirty, "files": files}


def _models_used(rec):
    models = {"detect": set(), "verify": set(), "amber": set(), "narration": set()}
    for r in rec.detect.values():
        models["detect"].add(r.get("model"))
    for r in rec.verify.values():
        models["verify"].add(r.get("model"))
    for r in rec.decisions.values():
        for round_row in r.get("rounds") or []:
            if isinstance(round_row, dict):
                models["amber"].add(round_row.get("model"))
    if (rec.job / "narration.json").exists():
        for s in read_json(rec.job / "narration.json").get("segments", []):
            if s.get("source") != "decisions":
                models["narration"].add(s.get("source"))
    return {k: sorted(m for m in v if m) for k, v in models.items()}


def _prompt_hashes(rec):
    hashes = {r.get("prompt_hash") for r in rec.detect.values()} | {r.get("prompt_hash") for r in rec.verify.values()}
    for r in rec.decisions.values():
        hashes |= {x.get("prompt_hash") for x in r.get("rounds") or [] if isinstance(x, dict)}
    if (rec.job / "narration.json").exists():
        hashes |= {s.get("prompt_hash") for s in read_json(rec.job / "narration.json").get("segments", [])}
    return sorted(h for h in hashes if h)


def _resolve_prompts(rec):
    """Full prompt text per hash: as recorded at run time, else regenerated from the templates with the hash checked.

    A hash neither route can reproduce is reported as unresolved, never given a guessed text."""
    recorded = {r.get("prompt_hash"): r.get("prompt") for r in read_jsonl(rec.job / "prompts.jsonl")}
    generated = {}
    try:
        from . import prompts as templates, vlm
        scale = rec.meta.get("view_scale") or 1
        w, h = rec.meta.get("width") or rec.size[0], rec.meta.get("height") or rec.size[1]
        for size in {(w, h), (round(w / scale), round(h / scale))}:
            for text in (templates.detect_prompt(*size), templates.verify_prompt(*size)):
                generated[vlm.prompt_hash(text)] = text
    except (ImportError, AttributeError, TypeError):
        pass
    out = {}
    for ph in _prompt_hashes(rec):
        if recorded.get(ph):
            out[ph] = {"text": recorded[ph], "resolved": True, "how": "recorded_at_run_time"}
        elif ph in generated:
            out[ph] = {"text": generated[ph], "resolved": True, "how": "regenerated_and_hash_checked"}
        else:
            out[ph] = {"text": None, "resolved": False, "how": "not recorded: see code/prompts.py for the templates"}
    return out


def _confidence_table(rec, coco):
    names = {c["id"]: c["name"] for c in coco["categories"]}
    table = {}
    for label in LABELS:
        rows = [r for (_, lab), r in rec.detect.items() if lab == label]
        confs = [r["conf"] for r in rows if isinstance(r.get("conf"), (int, float))]
        decs = [r for (_, lab), r in rec.decisions.items() if lab == label]
        table[label] = {"n_keyframes": len(rows), "mean_conf": round(float(np.mean(confs)), 4) if confs else None,
                        "n_accept_present": sum(1 for d in decs if d["status"] == ACCEPT and d.get("final_present")),
                        "n_ai_absent": sum(1 for d in decs if d["status"] in (ACCEPT, REJECT) and not d.get("final_present")),
                        "n_amber": sum(1 for d in decs if d["status"] == AMBER), "n_annotations": 0, "n_human_reviewed": 0}
    for a in coco["annotations"]:
        row = table[names[a["category_id"]]]
        row["n_annotations"] += 1
        row["n_human_reviewed"] += bool(a["attributes"]["human_reviewed"])
    return {"calibrated": False, "note": "conf is the model's own self-reported number. Nothing here has been calibrated.",
            "labels": table}


def _tips(rec, coco):
    names = {c["id"]: c["name"] for c in coco["categories"]}
    frames = {i["id"]: i["frame_index"] for i in coco["images"]}
    w, h = rec.size
    rows = []
    for a in coco["annotations"]:
        label, at = names[a["category_id"]], a["attributes"]
        if label not in INSTRUMENTS or at["propagated"]:
            continue
        m = np.zeros((h, w), np.uint8)
        for poly in a["segmentation"]:
            cv2.fillPoly(m, [np.array(poly, dtype=np.int32).reshape(-1, 2)], 1)
        tip = estimate_tip(m > 0)
        if tip:
            rows.append({"frame_index": frames[a["image_id"]], "keyframe": at["keyframe"], "label": label, "tip": tip,
                         "validated": False, "method": "mask long-axis end nearest the image centre (geometry, not reported by the model)"})
    return rows


def _rights_statement(job):
    doc = None
    if (job / "rights.json").exists():
        doc = read_json(job / "rights.json")
    elif (job / "job.json").exists():
        whole = read_json(job / "job.json")
        doc = whole.get("rights") or whole.get("attestation")
    if not doc:
        return None
    return doc["statement"] if isinstance(doc, dict) and doc.get("statement") else json.dumps(doc)


def _licences_text(rights):
    source = rights or "Not attested: nobody recorded the rights to this footage. Do not redistribute it."
    return ("# Licences\n\n"
            f"## Source footage\n{source}\n\n"
            "This tool grants no rights to the footage. Footage from CholecSeg8k or Cholec80 is CC BY-NC-SA 4.0 and must keep "
            "its attribution and terms.\n\n"
            "## Annotations\nAI-generated, with any human corrections listed in annotations/final_pass.jsonl. No licence is "
            "granted here beyond what the source footage allows.\n\n"
            "## Code\nPipeline code: MIT, see LICENSE in the project repository.\n")


def _dataset_card(coco, versions, prompts, confidence, rights, annotated_ok):
    info, fp = coco["info"], coco["info"]["final_pass"]
    all_models = sorted({m for v in versions["models"].values() for m in v})
    unresolved = [h for h, p in prompts.items() if not p["resolved"]]
    lines = ["# Dataset card: SurgProof draft annotations", "", f"Scope: {SCOPE}", ""]
    if any(m.startswith("mock") for m in all_models):
        lines += ["**MOCK RUN: these annotations came from a mock model, not a real one. They are only good for testing the software.**", ""]
    lines += ["## Contents", f"- Images: {len(coco['images'])}", f"- Annotations: {len(coco['annotations'])}",
              f"- Keyframe annotations: {info['n_keyframe_annotations']}", f"- Propagated annotations: {info['n_propagated_annotations']}",
              f"- Human corrections: {info['n_human_corrections']}", f"- Annotated video included: {'yes' if annotated_ok else 'no'}", "",
              "## Review actually done",
              f"- Reviewer: {info.get('reviewer')}",
              f"- Ambers resolved by a human: {fp['n_amber_resolved']}",
              f"- Audit: {fp['n_audit']} items sampled, {fp['n_audit_failed']} failed (rate {fp['audit_failure_rate']})",
              f"- Audit pools: {fp['n_audit_pool']} accepted keyframe labels ({fp['n_amber_accepted_pool']} by the amber loop), {fp['n_ai_absent_pool']} AI-absent labels, {fp['n_propagated_pool']} propagated frames",
              f"- Escalated to full keyframe review: {'yes' if fp['escalated'] else 'no'}", "",
              "A human saw only the annotations with human_reviewed true. Everything else is AI output from a job whose sample was checked.",
              "derived_from_reviewed means the source keyframes were reviewed; that frame itself was not.", "",
              "## Left out",
              f"- Propagated frames that failed the flow cross-check: {info['n_excluded_failed_flow_check']}",
              f"- Propagated frames dropped after a human overruled their source: {info['n_propagated_dropped_after_human_overrule']}",
              "- Excluded labels are listed on each image. A missing label is a negative only if listed in ai_absent or human_absent.", "",
              "## Limits",
              "- All masks are draft quality: a language model's outline snapped to image edges. Successful refinement says nothing about correctness.",
              "- Large organs are the weakest. The flow check compares two flows with each other, not with the truth.",
              "- Instance ids follow one label through an unbroken run of frames. Two tools of one class share an id.",
              "- tips.jsonl is a geometric estimate, not validated. conf is the model's own number, not calibrated.",
              "- Accuracy is not stated here: this job has no ground truth to score against.", "",
              "## Provenance", f"- Models used: {', '.join(all_models) or 'none recorded'}",
              f"- Code commit: {versions['code']['git_commit']} (uncommitted changes: {versions['code']['git_dirty']})",
              f"- Prompts: {len(prompts)} hashes, {len(unresolved)} not reproducible. Templates are in code/prompts.py.", "",
              "## Per-label counts", "| label | keyframes | mean conf | accepted present | AI absent | amber | annotations | human reviewed |",
              "|---|---|---|---|---|---|---|---|"]
    for label, r in confidence["labels"].items():
        lines.append(f"| {label} | {r['n_keyframes']} | {r['mean_conf']} | {r['n_accept_present']} | {r['n_ai_absent']} | {r['n_amber']} | {r['n_annotations']} | {r['n_human_reviewed']} |")
    lines += ["", "## Using the images", "Keyframes and audited frames are in frames/. Other frames decode from source.mp4 by frame_index: run tools/extract_frames.py.", "",
              "## Rights", rights or "Not attested. See LICENSES.md.", ""]
    return "\n".join(lines)


def _write_bundle(rec, coco, annotated, provenance, out_path, key):
    """Everything needed to use the dataset without this tool. Refuses up front if the disk cannot hold it."""
    job = rec.job
    try:
        video = find_video(job, rec.meta)
    except FileNotFoundError:
        video = None
    versions = {"models": _models_used(rec), "vlm_provider": rec.meta.get("vlm_provider"), "code": _code_version()}
    prompts = _resolve_prompts(rec)
    confidence = _confidence_table(rec, coco)
    rights = _rights_statement(job)

    blobs = {
        "annotations/released_coco.json": json.dumps(coco, indent=1),
        "versions.json": json.dumps(versions, indent=1),
        "prompts/prompts.json": json.dumps({"note": "Full text per prompt hash.", "hashes": prompts}, indent=1),
        "confidence.json": json.dumps(confidence, indent=1),
        "tips.jsonl": "".join(json.dumps(r) + "\n" for r in _tips(rec, coco)),
        "schema/released_coco.schema.json": json.dumps(ANNOTATION_SCHEMA, indent=1),
        "tools/extract_frames.py": EXTRACT_FRAMES_SCRIPT,
        "DATASET_CARD.md": _dataset_card(coco, versions, prompts, confidence, rights, bool(annotated)),
        "LICENSES.md": _licences_text(rights),
    }
    files = {"annotations/provenance.jsonl": provenance, "annotations/final_pass.jsonl": job / "final_pass.jsonl",
             "annotations/final_pass_queue.json": job / "final_pass_queue.json"}
    if annotated:
        files["annotated.mp4"] = Path(annotated)
    if video:
        files["source.mp4"] = video
    for src, name in (("phase.jsonl", "phases.jsonl"), ("narration.json", "narration.json"), ("narration.vtt", "narration.vtt")):
        if (job / src).exists():
            files[name] = job / src
    for path in sorted((job / "frames").glob("*.jpg")):
        files[f"frames/{path.name}"] = path
    for path in sorted((job / "masks").glob("*.png")):
        files[f"masks/{path.name}"] = path
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        files[f"code/{path.name}"] = path

    needed = sum(Path(p).stat().st_size for p in files.values()) + sum(len(b) for b in blobs.values())
    free = shutil.disk_usage(job).free
    if free < needed * 1.2:
        raise OSError(f"not enough disk for the dataset bundle: need about {needed // 1024 ** 2} MB, {free // 1024 ** 2} MB free")

    entries = {name: {"sha256": hashlib.sha256(b.encode()).hexdigest(), "bytes": len(b.encode())} for name, b in blobs.items()}
    entries.update({name: _file_entry(p) for name, p in files.items()})
    manifest = {"files": entries}
    manifest["signature"] = _sign(manifest, key)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, data in blobs.items():
            z.writestr(name, data)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
        for name, path in files.items():
            stored = Path(path).suffix.lower() in {".mp4", ".mov", ".jpg", ".png"}
            z.write(path, name, compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED)


def release(job_dir, reviewer=None, on_progress=None):
    """Build and commit the release: dataset, re-rendered video, bundle, signed manifest.

    Raises ReleaseLocked when the final pass does not allow it. Everything is built first and released_coco.json is the
    last file committed, so any failure leaves the job unreleased."""
    job = Path(job_dir)
    if (job / "released_coco.json").exists():
        raise ReleaseLocked("this job has already been released")
    final_pass_queue(job)
    _maybe_escalate(job)
    state = final_pass_state(job)
    if state["empty"]:
        raise ReleaseLocked("nothing to review: an empty final-pass queue never releases")
    if state["locked"]:
        reason = f" Audit stop rule: {state['escalation_reason']}, so every keyframe label needs review." if state["escalated"] else ""
        raise ReleaseLocked(f"{len(state['remaining'])} of {state['n_required']} final-pass items are unresolved.{reason}")

    rec = JobRecords(job)
    items = final_pass_queue(job)
    doc = _queue_doc(job)
    queued = {(i["kf"], i["label"]) for i in items}
    # Only rows for queued items count; stray lines in the file cannot change the dataset.
    final = {k: v for k, v in read_final_pass(job).items() if k in queued}
    coco = build_coco(rec, final)
    stats = audit_stats(items, final)
    audit_actions = [final[(i["kf"], i["label"])]["action"] for i in items if i["kind"] == KIND_AUDIT]
    n_rows = sum(1 for r in _final_pass_rows(job) if (r.get("kf"), r.get("label")) in queued)
    coco["info"].update({
        "released": True, "reviewer": reviewer,
        "final_pass": {"n_amber_resolved": state["n_amber"], "n_audit": stats["n"], "n_audit_pool": doc.get("n_accepted_pool"),
                       "n_amber_accepted_pool": doc.get("n_amber_accepted_pool"),
                       "n_ai_absent_pool": doc.get("n_ai_absent_pool"), "n_propagated_pool": doc.get("n_propagated_pool"),
                       "n_audit_failed": stats["failed"], "audit_failure_rate": stats["rate"], "fail_threshold": AUDIT_FAIL_THRESHOLD,
                       "per_label": stats["per_label"], "escalated": state["escalated"], "escalation_reason": state["escalation_reason"],
                       "n_escalation_reviewed": state["n_escalation"], "n_audit_approved": audit_actions.count(APPROVE),
                       "n_audit_rejected": audit_actions.count(REJECT), "n_audit_corrected": audit_actions.count(CORRECT),
                       "n_verdicts_changed": n_rows - len(final)}})

    from . import render  # imported here because render imports this module for the stale check

    key = _signing_key(job)
    annotated_tmp, provenance_tmp, bundle_tmp = (job / ".annotated.release.tmp.mp4", job / ".provenance.release.tmp",
                                                 job / ".dataset_bundle.release.tmp.zip")
    try:
        try:
            find_video(job, rec.meta)
            have_video = True
        except FileNotFoundError:
            have_video = False
        if have_video:
            render.run(job, released=coco, output=annotated_tmp, on_progress=on_progress or (lambda *_: None))
        coco["info"]["annotated_rerendered"] = have_video
        write_jsonl(provenance_tmp, build_provenance(rec, final))
        _write_bundle(rec, coco, annotated_tmp if have_video else None, provenance_tmp, bundle_tmp, key)

        # Commit. released_coco.json goes last: it is what marks the job as released.
        coco_text = json.dumps(coco, indent=1)
        if have_video:
            os.replace(annotated_tmp, job / "annotated.mp4")
        os.replace(provenance_tmp, job / "provenance.jsonl")
        os.replace(bundle_tmp, job / "dataset_bundle.zip")
        entries = {"released_coco.json": {"sha256": hashlib.sha256(coco_text.encode()).hexdigest(), "bytes": len(coco_text.encode())}}
        for name in MANIFEST_INPUTS:
            # Without a source video annotated.mp4 is still the draft, so the release does not claim it.
            if (job / name).exists() and (have_video or name != "annotated.mp4"):
                entries[name] = _file_entry(job / name)
        manifest = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "reviewer": reviewer, "files": entries}
        manifest["signature"] = _sign(manifest, key)
        write_json(job / "release_manifest.json", manifest)
        atomic_write(job / "released_coco.json", coco_text)
    finally:
        for tmp in (annotated_tmp, provenance_tmp, bundle_tmp):
            tmp.unlink(missing_ok=True)
    return coco["info"]


def main(argv=None):
    """python -m surgproof.export verify jobs/<id>"""
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] != "verify":
        print("usage: python -m surgproof.export verify jobs/<id>", file=sys.stderr)
        return 2
    result = verify_release(argv[1])
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
