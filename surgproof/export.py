"""Stage 9: the dataset, its provenance, and the human final pass that unlocks release.

export.run writes dataset_coco.json (accepted labels only; ambers are excluded) and provenance.jsonl.
That file is a draft: info.released is false. Only release() writes released_coco.json, and it refuses until a human
has resolved every amber plus a random audit sample of the auto-accepted labels.

The audit sample is drawn once and persisted to final_pass_queue.json, so nobody can re-roll it for an easier one.
A human correction is recorded as human work (source human_correction), never as the AI's.
Propagated frames are derived from a keyframe's mask, so when a human rejects or corrects that keyframe they are
dropped from the release rather than kept on the strength of a mask that was overruled. Those drops are counted.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

from .masks import (ACCEPT, AMBER, LABELS, MIN_SEED_AREA, REJECT, clean_polygon, frame_size, mask_box, polygon_mask, progress_fn,
                    read_json, read_jsonl, read_mask, resolve_decisions, track_path, write_json, write_jsonl)
from .propagate import find_video

CATEGORIES = [{"id": i + 1, "name": name, "supercategory": "instrument" if name in ("grasper", "l_hook") else "anatomy"}
              for i, name in enumerate(LABELS)]
CATEGORY_ID = {c["name"]: c["id"] for c in CATEGORIES}
POLY_EPSILON = 1.0        # px, contour simplification
AUDIT_FRAC = 0.10
MIN_AUDIT = 5

APPROVE, CORRECT = "approve", "correct"
ACTIONS = {APPROVE, REJECT, CORRECT}
SYNONYMS = {"accept": APPROVE, "approved": APPROVE, "rejected": REJECT, "corrected": CORRECT}


class ReleaseLocked(RuntimeError):
    """Release was attempted while ambers or audit items are still unresolved."""


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


def _image_entry(rec, frame_index, kf=None, **attrs):
    w, h = rec.size
    entry = {"id": frame_index + 1, "width": w, "height": h, "frame_index": frame_index,
             "t": round(frame_index / rec.fps, 4) if rec.fps else None,
             "file_name": f"frames/{kf}.jpg" if kf else f"frame_{frame_index:06d}.jpg",
             "attributes": {"keyframe": kf, "extracted": bool(kf), **attrs}}
    return entry


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


# ---- dataset ----

def build_coco(rec, final=None):
    """Build the COCO dict. `final` is the resolved {(kf, label): row} from the final pass, or None for the draft."""
    final = final or {}
    w, h = rec.size
    images, annotations = {}, []
    counts = {"n_keyframe_annotations": 0, "n_propagated_annotations": 0, "n_excluded_amber_keyframe": 0,
              "n_excluded_amber_propagated": 0, "n_propagated_dropped_after_human_overrule": 0,
              "n_human_corrections": 0}

    # Every keyframe is an image, including empty ones. They record verified absence, which a dataset needs.
    for k in rec.meta["keyframes"]:
        kf = k["kf"]
        absent, excluded = [], []
        for label in LABELS:
            dec = rec.decisions.get((kf, label))
            f = final.get((kf, label))
            action = f["action"] if f else None
            status = dec["status"] if dec else None
            has_mask = bool((rec.mask_rows.get((kf, label)) or {}).get("path"))
            if action == REJECT or (action == APPROVE and not has_mask) or (
                    status in (ACCEPT, REJECT) and not dec.get("final_present") and action != CORRECT):
                absent.append(label)
            elif (status == AMBER or status is None and (kf, label) in rec.mask_rows) and action is None:
                excluded.append(label)
        images[k["frame_index"]] = _image_entry(rec, k["frame_index"], kf, verified_absent=absent, excluded_amber=excluded)

    ann_id = 0

    def add(frame_index, label, mask, attributes):
        nonlocal ann_id
        polys = mask_to_polygons(mask)
        box = mask_box(mask)
        if not polys or box is None:
            return False
        ann_id += 1
        annotations.append({"id": ann_id, "image_id": frame_index + 1, "category_id": CATEGORY_ID[label],
                            "segmentation": polys, "bbox": coco_bbox(box), "area": int(mask.sum()), "iscrowd": 0,
                            "attributes": attributes})
        return True

    # Keyframes.
    overruled = set()
    for (kf, label), row in sorted(rec.mask_rows.items()):
        f = final.get((kf, label))
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
        source = "ai_polygon_grabcut" if not row.get("fallback") else "ai_polygon_unrefined"
        attrs = {"status": ACCEPT, "source": source, "keyframe": kf, "propagated": False, "model": model,
                 "prompt_hash": prompt_hash, "seed": row.get("seed"), "human_reviewed": action == APPROVE}
        if add(rec.kf_index[kf], label, read_mask(rec.job / row["path"]), attrs):
            counts["n_keyframe_annotations"] += 1

    # Human corrections.
    for (kf, label), f in sorted(final.items()):
        if f["action"] != CORRECT:
            continue
        poly = clean_polygon(f.get("polygon"), w, h)
        if poly is None:
            raise ValueError(f"correction for {kf}/{label} has no usable polygon")
        overruled.add((kf, label))
        attrs = {"status": ACCEPT, "source": "human_correction", "keyframe": kf, "propagated": False, "model": None,
                 "prompt_hash": None, "human_reviewed": True, "reviewer": f.get("reviewer")}
        if add(rec.kf_index[kf], label, polygon_mask(poly, w, h).astype(bool), attrs):
            counts["n_keyframe_annotations"] += 1
            counts["n_human_corrections"] += 1

    # An AMBER keyframe that a human approved no longer blocks its propagated frames.
    def resolved_ok(kf, label):
        f = final.get((kf, label))
        if f:
            return f["action"] == APPROVE
        row = rec.mask_rows.get((kf, label))
        return bool(row) and row["status"] == ACCEPT

    # Propagated frames.
    for t in rec.tracks:
        if not t["propagated"]:
            continue
        label, sources = t["label"], t["source_kfs"]
        if any((kf, label) in overruled for kf in sources):
            counts["n_propagated_dropped_after_human_overrule"] += 1
            continue
        if not all(resolved_ok(kf, label) for kf in sources):
            counts["n_excluded_amber_propagated"] += 1
            continue
        fi = t["frame_index"]
        model, prompt_hash = _model_trace(rec, sources, label)
        attrs = {"status": ACCEPT, "source": "optical_flow_from_ai_keyframes", "keyframe": None, "propagated": True,
                 "source_keyframes": sources, "model": model, "prompt_hash": prompt_hash,
                 "human_reviewed": any(final.get((kf, label)) for kf in sources)}
        if add(fi, label, read_mask(track_path(rec.job, fi, label)), attrs):
            counts["n_propagated_annotations"] += 1
            if fi not in images:
                images[fi] = _image_entry(rec, fi)

    info = {"description": "SurgProof AI-annotated dataset", "version": "1.0",
            "date_created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "released": False, "human_final_pass": bool(final), "n_images": len(images),
            "n_annotations": len(annotations), **counts,
            "frames_note": "Images with attributes.extracted false are source-video frames not stored as files: "
                           "decode them by frame_index from the source video.",
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
                 "propagation": "Farneback flow forward and backward blended, 5-frame majority, 7-frame box average"})
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
                             "fallback": mrow.get("fallback"), "area": mrow.get("area"), "box": mrow.get("box")}
            detect_trace = {k2: det.get(k2) for k2 in ("present", "conf", "model", "prompt_hash", "raw_ref")} if det else None
            verify_trace = {k2: ver.get(k2) for k2 in ("present", "box", "model", "prompt_hash")} if ver else None
            rows.append({"type": "keyframe_label", "kf": kf, "frame_index": k["frame_index"], "label": label,
                         "status": dec["status"] if dec else None, "final_present": dec.get("final_present") if dec else None,
                         "rounds": dec.get("rounds") if dec else None, "detect": detect_trace, "verify": verify_trace,
                         "mask": mask_info, "human_action": (final.get((kf, label)) or {}).get("action")})
    # Propagated runs, collapsed: one row per label and source pair, not one per frame.
    runs = {}
    for t in rec.tracks:
        if t["propagated"]:
            key = (t["label"], tuple(t["source_kfs"]))
            r = runs.setdefault(key, {"type": "propagated_run", "label": t["label"], "source_kfs": t["source_kfs"],
                                      "first_frame": t["frame_index"], "last_frame": t["frame_index"], "n_frames": 0,
                                      "status": t["status"]})
            r["first_frame"] = min(r["first_frame"], t["frame_index"])
            r["last_frame"] = max(r["last_frame"], t["frame_index"])
            r["n_frames"] += 1
    rows.extend(runs.values())
    for (kf, label), f in sorted(final.items()):
        rows.append({"type": "final_pass", **f, "kf": kf, "label": label})
    return rows


def clear_stale_final_pass(job_dir):
    """Drop human sign-off that does not belong to the current decisions, so a re-run cannot inherit it.

    The queue records which decisions it was drawn from; a final pass with no matching queue is not this run's."""
    job = Path(job_dir)
    queue = job / "final_pass_queue.json"
    current = queue.exists() and read_json(queue).get("seed") == _queue_seed(job)
    if not current:
        for name in ("final_pass_queue.json", "final_pass.jsonl", "released_coco.json"):
            (job / name).unlink(missing_ok=True)


def run(job_dir, on_progress=None):
    """Run stage 9: write the draft dataset and provenance. Returns a summary of what is in and what is out."""
    job = Path(job_dir)
    progress = progress_fn(job, "export", on_progress)
    clear_stale_final_pass(job)
    rec = JobRecords(job)
    progress(0.1, "building COCO annotations")
    coco = build_coco(rec)
    write_json(job / "dataset_coco.json", coco)
    progress(0.8, "writing provenance")
    write_jsonl(job / "provenance.jsonl", build_provenance(rec))
    info = coco["info"]
    progress(1.0, f"{info['n_annotations']} annotations on {info['n_images']} images, "
                  f"{info['n_excluded_amber_keyframe']} amber keyframe labels excluded")
    return {k: info[k] for k in ("n_images", "n_annotations", "n_keyframe_annotations", "n_propagated_annotations",
                                 "n_excluded_amber_keyframe", "n_excluded_amber_propagated")}


# ---- human final pass ----

def _queue_seed(job):
    """A seed fixed by the job's own recorded decisions, so the draw is reproducible from the job dir."""
    path = Path(job) / "decisions.jsonl"
    return int(_sha256(path)[:12], 16) if path.exists() else 0


def final_pass_queue(job_dir, audit_frac=AUDIT_FRAC, min_audit=MIN_AUDIT):
    """Everything a human must clear: all ambers plus a random audit sample of accepted labels.

    Persisted on first call, so later calls return the same items whatever `audit_frac` they pass."""
    job = Path(job_dir)
    path = job / "final_pass_queue.json"
    if path.exists():
        return read_json(path)["items"]
    rec = JobRecords(job)
    ambers, accepted = [], []
    for k in rec.meta["keyframes"]:
        kf = k["kf"]
        for label in LABELS:
            row = rec.mask_rows.get((kf, label))
            dec = rec.decisions.get((kf, label))
            status = (row or {}).get("status") or (dec or {}).get("status")
            if status == AMBER:
                ambers.append((k, label, row, dec))
            elif status == ACCEPT and row and row.get("path"):
                accepted.append((k, label, row, dec))

    # Stratified by label, so a rare label is audited at all.
    by_label = {}
    for item in accepted:
        by_label.setdefault(item[1], []).append(item)
    n_target = min(len(accepted), max(min_audit, int(np.ceil(audit_frac * len(accepted))))) if accepted else 0
    rng = random.Random(_queue_seed(job))
    audit = []
    for label in sorted(by_label):
        share = max(1, round(n_target * len(by_label[label]) / len(accepted)))
        audit.extend(rng.sample(by_label[label], min(share, len(by_label[label]))))
    rng.shuffle(audit)

    def item(kind, k, label, row, dec):
        return {"id": f"{k['kf']}:{label}", "kind": kind, "kf": k["kf"], "frame_index": k["frame_index"], "t": k.get("t"),
                "label": label, "status": (row or {}).get("status") or (dec or {}).get("status"),
                "frame": f"frames/{k['kf']}.jpg", "mask": (row or {}).get("path"), "box": (row or {}).get("box"),
                "conf": (rec.detect.get((k["kf"], label)) or {}).get("conf"),
                "rounds": (dec or {}).get("rounds")}

    items = [item("amber", *a) for a in ambers] + [item("audit", *a) for a in audit]
    write_json(path, {"seed": _queue_seed(job), "audit_frac": audit_frac, "n_accepted_pool": len(accepted), "items": items})
    return items


def _normalise_action(action):
    action = str(action or "").strip().lower()
    action = SYNONYMS.get(action, action)
    if action not in ACTIONS:
        raise ValueError(f"unknown final-pass action {action!r}")
    return action


def read_final_pass(job_dir):
    """{(kf, label): row}, latest row winning, so a reviewer can change their mind before release."""
    final = {}
    for r in read_jsonl(Path(job_dir) / "final_pass.jsonl"):
        final[(r["kf"], r["label"])] = {**r, "action": _normalise_action(r.get("action"))}
    return final


def record_final_pass(job_dir, rows):
    """Validate human final-pass rows against the queue and append them to final_pass.jsonl."""
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
        if action == CORRECT:
            rec = rec or JobRecords(job)
            w, h = rec.size
            poly = clean_polygon(r.get("polygon"), w, h)
            if poly is None or int(polygon_mask(poly, w, h).sum()) < MIN_SEED_AREA or cv2.contourArea(poly) <= 0:
                raise ValueError(f"correction for {key} needs a polygon that encloses a real area inside the frame")
            row["polygon"] = poly.tolist()  # stored as clipped to the frame
        clean.append(row)
    with open(job / "final_pass.jsonl", "a") as f:
        for row in clean:
            f.write(json.dumps(row) + "\n")
    return final_pass_state(job)


def final_pass_state(job_dir):
    items = final_pass_queue(job_dir)
    final = read_final_pass(job_dir)
    remaining = [i for i in items if (i["kf"], i["label"]) not in final]
    return {"n_required": len(items), "n_done": len(items) - len(remaining), "remaining": remaining,
            "n_amber": sum(1 for i in items if i["kind"] == "amber"), "n_audit": sum(1 for i in items if i["kind"] == "audit"),
            "locked": bool(remaining), "released": (Path(job_dir) / "released_coco.json").exists()}


def release(job_dir, reviewer=None):
    """Write released_coco.json. Raises ReleaseLocked until every queued item has a human action."""
    job = Path(job_dir)
    state = final_pass_state(job)
    if state["locked"]:
        raise ReleaseLocked(f"{len(state['remaining'])} of {state['n_required']} final-pass items are unresolved")
    rec = JobRecords(job)
    final = read_final_pass(job)
    queued = {(i["kf"], i["label"]): i for i in final_pass_queue(job)}
    # Only rows for queued items count; stray lines in the file cannot change the dataset.
    final = {k: v for k, v in final.items() if k in queued}
    coco = build_coco(rec, final)
    audit = [final[k]["action"] for k, i in queued.items() if i["kind"] == "audit"]
    pool = read_json(job / "final_pass_queue.json").get("n_accepted_pool")
    coco["info"].update({
        "released": True, "reviewer": reviewer,
        "final_pass": {"n_amber_resolved": state["n_amber"], "n_audit": len(audit), "n_audit_pool": pool,
                       "n_audit_approved": audit.count(APPROVE), "n_audit_rejected": audit.count(REJECT),
                       "n_audit_corrected": audit.count(CORRECT)}})
    write_json(job / "released_coco.json", coco)
    write_jsonl(job / "provenance.jsonl", build_provenance(rec, final))
    return coco["info"]
