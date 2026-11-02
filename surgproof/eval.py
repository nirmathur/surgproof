"""Score a finished job against CholecSeg8k ground truth. Separate from the pipeline.

    python -m surgproof.eval jobs/<id>

This is the ONLY module under surgproof/ allowed to read ground truth, and nothing in the pipeline
may import it (tests/test_qa_redteam.py enforces both). It reads job outputs and never changes them,
apart from writing its own jobs/<id>/eval_report.json.

Finding the ground truth: tools/make_test_video.py registers each test video by the sha256 of the
mp4. The scorer hashes the source video in the job dir and looks it up. A video that is not registered
has no ground truth, and the scorer exits 2 rather than producing a number.

Alignment is checked, not assumed: every keyframe jpg is compared with the dataset frame it claims
to be, and with its neighbours. An off-by-N in the pipeline would otherwise score every mask against
the wrong frame and look like a model problem (or, worse, not look like anything). On a mismatch the
scorer exits 3 unless --allow-misaligned, and the report records it either way.

Nothing is silently dropped. Rows the pipeline failed to produce count against coverage, an accepted
object with no mask file scores IoU 0, and every number travels with its n.

Conventions:
  presence GT: prepare_data.presence (class fraction >= config.MIN_FRAC), same as the rest of the repo.
  IoU: watershed border pixels (value 255, class -1) are ignored in both prediction and GT.
  iou_when_both_present: mean over objects both AI and GT call present. Flattering on its own.
  iou_with_errors: also counts a false positive or a false negative as IoU 0. Use this as the headline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as C  # noqa: E402
from prepare_data import mask_to_classes, presence  # noqa: E402

LABELS = list(C.LABELS)
DEFAULT_RAW = ROOT / "raw" / "CholecSeg8k"
DEFAULT_REGISTRY = ROOT / "test_videos" / ".gt_registry"
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
NOT_SOURCE = {"annotated.mp4"}
ALIGN_TOLERANCE = 0.5     # grey levels; the claimed frame may be this much worse than its best neighbour
ALIGN_ABS_MAX = 12.0      # mean abs difference above this is simply not the same image
ALIGN_MAX_BAD_FRAC = 0.1
BOOTSTRAP = 1000


class NoGroundTruth(Exception):
    """exit code 2"""


class Misaligned(Exception):
    """exit code 3"""


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_manifest(job: Path, registry: Path, explicit: Path | None = None) -> dict:
    if explicit:
        return json.loads(Path(explicit).read_text())
    candidates = [p for p in job.rglob("*") if p.suffix.lower() in VIDEO_EXT and p.name not in NOT_SOURCE and p.is_file()]
    meta_path = job / "meta.json"
    if meta_path.exists():
        declared = json.loads(meta_path.read_text()).get("source_video")
        if declared:  # may be relative to the job dir, may point outside it (CLI runs on a path)
            d = Path(declared) if Path(declared).is_absolute() else job / declared
            if d.is_file():
                candidates.insert(0, d)
    for p in candidates:
        m = registry / f"{_sha256(p)}.json"
        if m.exists():
            return json.loads(m.read_text())
    raise NoGroundTruth(
        f"no registered test video in {job} (looked at {[p.name for p in candidates]}, registry {registry}). "
        "Only videos built by tools/make_test_video.py have ground truth. If the pipeline re-encodes the "
        "source, pass --manifest explicitly; the alignment check still applies."
    )


class GroundTruth:
    """Frame index in the assembled video -> dataset frame -> class map."""

    def __init__(self, manifest: dict, raw: Path):
        self.raw = Path(raw)
        self.by_index = {int(r["frame_index"]): r for r in manifest["frames"]}
        self.n_frames = len(self.by_index)
        self._cache: dict[int, tuple[np.ndarray, dict]] = {}

    def _dir(self, r: dict) -> Path:
        return self.raw / r["video"] / r["clip"]

    def image_path(self, frame_index: int) -> Path:
        r = self.by_index[frame_index]
        return self._dir(r) / f"frame_{r['frame']}_endo.png"

    def clip_of(self, frame_index: int) -> str:
        return self.by_index[frame_index]["clip"]

    def classes(self, frame_index: int) -> tuple[np.ndarray, dict]:
        if frame_index not in self._cache:
            r = self.by_index[frame_index]
            cls = mask_to_classes(self._dir(r) / f"frame_{r['frame']}_endo_watershed_mask.png")
            self._cache[frame_index] = (cls, presence(cls))
            if len(self._cache) > 64:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[frame_index]

    def present(self, frame_index: int, label: str) -> bool:
        return bool(self.classes(frame_index)[1][f"gt_{label}"])

    def mask(self, frame_index: int, label: str) -> tuple[np.ndarray, np.ndarray]:
        """(object mask, valid pixels). Border pixels are invalid."""
        cls, _ = self.classes(frame_index)
        return cls == C.LABELS[label][0], cls != -1


# ---------- metrics ----------

def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """pairs of (predicted, truth). Accuracy comes with the always-say-the-majority baseline."""
    tp = sum(1 for p, t in pairs if p and t)
    fp = sum(1 for p, t in pairs if p and not t)
    fn = sum(1 for p, t in pairs if t and not p)
    tn = sum(1 for p, t in pairs if not p and not t)
    n = tp + fp + fn + tn
    if not n:
        return {"n": 0, "accuracy": None}
    recall = tp / (tp + fn) if tp + fn else None
    precision = tp / (tp + fp) if tp + fp else None
    spec = tn / (tn + fp) if tn + fp else None
    return {
        "n": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": round((tp + tn) / n, 4),
        "majority_baseline": round(max(tp + fn, fp + tn) / n, 4),
        "balanced_accuracy": round((recall + spec) / 2, 4) if recall is not None and spec is not None else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
    }


def cluster_ci(values: list[tuple[object, float]]) -> list[float] | None:
    """95% bootstrap CI of the mean, resampling whole frames: labels on one frame are not independent."""
    clusters: dict[object, list[float]] = {}
    for k, v in values:
        clusters.setdefault(k, []).append(v)
    if len(clusters) < 2:
        return None
    arr = list(clusters.values())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP, len(arr)))
    sums = np.array([sum(a) for a in arr], dtype=float)
    counts = np.array([len(a) for a in arr], dtype=float)
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p, g = pred & valid, gt & valid
    union = (p | g).sum()
    return float((p & g).sum() / union) if union else 0.0


def box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def mask_box(mask: np.ndarray):
    ys, xs = np.where(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1] if len(xs) else None


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("L")
    if im.size != (shape[1], shape[0]):
        im = im.resize((shape[1], shape[0]), Image.NEAREST)
    return np.array(im) > 0


def mean_summary(items: list[tuple[object, float]]) -> dict:
    if not items:
        return {"n": 0, "mean": None}
    return {"n": len(items), "n_frames": len({k for k, _ in items}),
            "mean": round(float(np.mean([v for _, v in items])), 4), "ci95_by_frame": cluster_ci(items)}


def _by_label(rows: list[tuple[str, object, float]]) -> dict:
    return {lb: mean_summary([(k, v) for l, k, v in rows if l == lb]) for lb in sorted({r[0] for r in rows})}


# ---------- alignment ----------

def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def check_alignment(job: Path, meta: dict, gt: GroundTruth) -> dict:
    per = []
    for k in meta.get("keyframes", []):
        fi = int(k["frame_index"])
        jpg = job / "frames" / f"{k['kf']}.jpg"
        if not jpg.exists() or fi not in gt.by_index:
            per.append({"kf": k["kf"], "ok": False, "reason": "missing frame or index out of range"})
            continue
        im = Image.open(jpg).convert("RGB")
        im_arr = np.array(im)
        h, w = im_arr.shape[:2]
        scores = {}
        for off in range(-3, 4):
            if fi + off in gt.by_index:
                src = Image.open(gt.image_path(fi + off)).convert("RGB")
                if src.size != (w, h):
                    src = src.resize((w, h), Image.BILINEAR)
                scores[off] = _mae(im_arr, np.array(src))
        best = min(scores, key=scores.get)
        ok = scores[0] <= scores[best] + ALIGN_TOLERANCE and scores[0] <= ALIGN_ABS_MAX
        per.append({"kf": k["kf"], "ok": ok, "mae": round(scores[0], 2), "best_offset": best})
    bad = [p for p in per if not p["ok"]]
    return {
        "n_checked": len(per), "n_bad": len(bad),
        "ok": bool(per) and len(bad) / len(per) <= ALIGN_MAX_BAD_FRAC,
        "bad": bad[:10],
        "mean_mae": round(float(np.mean([p["mae"] for p in per if "mae" in p])), 2) if any("mae" in p for p in per) else None,
    }


# ---------- presence ----------

def score_presence(rows: list[dict], kf_index: dict, gt: GroundTruth, present_field: str) -> dict:
    """One prediction per (kf, label). Rows with no boolean prediction are undecided."""
    latest: dict[tuple[str, str], dict] = {}
    dupes = 0
    for r in rows:
        key = (r["kf"], r["label"])
        dupes += key in latest
        latest[key] = r
    expected = [(kf, lb) for kf in kf_index for lb in LABELS]
    pairs, per_label, clusters, undecided = [], {lb: [] for lb in LABELS}, [], 0
    missing = 0
    for kf, lb in expected:
        r = latest.get((kf, lb))
        if r is None:
            missing += 1
            continue
        pred = r.get(present_field)
        if not isinstance(pred, bool):
            undecided += 1
            continue
        t = gt.present(kf_index[kf], lb)
        pairs.append((pred, t))
        per_label[lb].append((pred, t))
        clusters.append((kf, float(pred == t)))
    out = confusion(pairs)
    out.update({"n_expected": len(expected), "n_missing_rows": missing, "n_undecided": undecided,
                "n_duplicate_rows": dupes, "coverage": round(len(pairs) / len(expected), 4) if expected else None,
                "ci95_by_frame": cluster_ci(clusters),
                "per_label": {lb: confusion(v) for lb, v in per_label.items()}})
    return out


def score_decisions(rows: list[dict], kf_index: dict, gt: GroundTruth) -> dict:
    """Resolved labels only (what the dataset would contain), plus how many stayed amber and how they'd have scored."""
    resolved = [r for r in rows if r.get("status") in ("accept", "reject")]
    amber = [r for r in rows if r.get("status") == "AMBER"]
    other = [r for r in rows if r.get("status") not in ("accept", "reject", "AMBER")]
    out = score_presence(resolved, kf_index, gt, "final_present")
    # coverage above counts ambers as missing rows; split them out so the arithmetic is visible
    out["n_missing_rows"] -= len({(r["kf"], r["label"]) for r in amber})
    out["n_amber"] = len(amber)
    out["n_unknown_status"] = len(other)
    # A label "accepted as present" that is also final_present False would be a contract bug
    out["n_status_contradictions"] = sum(
        1 for r in rows if (r.get("status") == "reject" and r.get("final_present") is True)
    )
    return out


# ---------- masks ----------

def score_keyframe_masks(job: Path, decisions: list[dict], kf_index: dict, gt: GroundTruth) -> dict:
    both, with_errors, boxes = [], [], []
    missing, fp, fn, excluded_amber = [], 0, 0, 0
    for r in decisions:
        kf, lb = r["kf"], r["label"]
        if kf not in kf_index or lb not in C.LABELS:
            continue
        fi = kf_index[kf]
        t = gt.present(fi, lb)
        if r.get("status") == "AMBER":
            excluded_amber += 1
            continue
        pred = r.get("final_present") is True and r.get("status") in ("accept", "reject")
        if pred and t:
            g, valid = gt.mask(fi, lb)
            path = job / "masks" / f"{kf}_{lb}.png"
            if path.exists():
                m = load_mask(path, g.shape)
                v = iou(m, g, valid)
                gb, mb = mask_box(g & valid), mask_box(m & valid)
                if gb and mb:
                    boxes.append((lb, kf, box_iou(mb, gb)))
            else:
                missing.append(f"{kf}_{lb}")
                v = 0.0
            both.append((lb, kf, v))
            with_errors.append((lb, kf, v))
        elif pred and not t:
            fp += 1
            with_errors.append((lb, kf, 0.0))
        elif t and not pred:
            fn += 1
            with_errors.append((lb, kf, 0.0))
    return {
        "iou_when_both_present": mean_summary([(k, v) for _, k, v in both]),
        "iou_with_errors": mean_summary([(k, v) for _, k, v in with_errors]),
        "box_iou_from_mask": mean_summary([(k, v) for _, k, v in boxes]),
        "per_label_when_both_present": _by_label(both),
        "per_label_with_errors": _by_label(with_errors),
        "n_false_positive_objects": fp, "n_false_negative_objects": fn,
        "n_excluded_amber": excluded_amber,
        "n_missing_mask_files": len(missing), "missing_mask_files": missing[:20],
    }


_TRACK = re.compile(r"^0*(\d+)_([a-z_]+)\.png$")


def score_tracks(job: Path, gt: GroundTruth) -> dict | None:
    """Every frame of the video, split into keyframe and propagated, since they are very different claims."""
    rows = _read_jsonl(job / "tracks.jsonl")
    if not rows:
        return None
    files = {}
    if (job / "tracks").is_dir():
        for p in (job / "tracks").iterdir():
            m = _TRACK.match(p.name)
            if m:
                files[(int(m.group(1)), m.group(2))] = p
    # Keyframes have tracks.jsonl rows but no tracks/ png: their mask is masks/<kf>_<label>.png.
    meta = json.loads((job / "meta.json").read_text())
    for k in meta.get("keyframes", []):
        for lb in LABELS:
            p = job / "masks" / f"{k['kf']}_{lb}.png"
            if p.exists():
                files.setdefault((int(k["frame_index"]), lb), p)
    track = {(int(r["frame_index"]), r["label"]): r for r in rows}
    pres: dict[str, list] = {"keyframe": [], "propagated": [], "untracked": []}
    ious: dict[str, list] = {"keyframe": [], "propagated": []}
    box_ious: dict[str, list] = {"keyframe": [], "propagated": []}
    kf_frames = {int(r["frame_index"]) for r in rows if r.get("propagated") is False}
    no_png = 0
    for fi in sorted(gt.by_index):
        for lb in LABELS:
            t = gt.present(fi, lb)
            r = track.get((fi, lb))
            has_box = bool(r and r.get("box"))
            kind = "keyframe" if fi in kf_frames else "propagated"
            pres[kind].append((has_box, t))
            if not (has_box and t):
                continue
            g, valid = gt.mask(fi, lb)
            gb = mask_box(g & valid)
            if gb:  # "box" is what playback shows (smoothed), so that is the claim being scored
                box_ious[kind].append((lb, fi, box_iou(r["box"], gb)))
            p = files.get((fi, lb))
            if p is None:
                no_png += 1
            else:
                ious[kind].append((lb, fi, iou(load_mask(p, g.shape), g, valid)))
    return {
        "n_video_frames": gt.n_frames,
        "n_frames_with_any_track": len({fi for fi, _ in track}),
        "presence": {k: confusion(v) for k, v in pres.items() if v},
        "mask_iou_when_both_present": {k: mean_summary([(f, x) for _, f, x in v]) for k, v in ious.items()},
        "mask_iou_per_label": {k: _by_label(v) for k, v in ious.items()},
        "box_iou_when_both_present": {k: mean_summary([(f, x) for _, f, x in v]) for k, v in box_ious.items()},
        "n_tracks_without_mask_png": no_png,
    }


# ---------- exported dataset ----------

def _coco_frame_index(img: dict) -> int | None:
    if isinstance(img.get("frame_index"), int):
        return img["frame_index"]
    return None


def _rasterise(segmentation, shape: tuple[int, int]) -> np.ndarray:
    im = Image.new("L", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(im)
    for poly in segmentation or []:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
    return np.array(im, dtype=bool)


def score_coco(path: Path, gt: GroundTruth) -> dict | None:
    """Score what is actually handed over. If the export corrupts masks, this differs from the keyframe IoU."""
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    images = {i["id"]: _coco_frame_index(i) for i in d.get("images", [])}
    cats = {c["id"]: c["name"] for c in d.get("categories", [])}
    if images and all(v is None for v in images.values()):
        return {"scored": False, "reason": "coco images carry no integer frame_index", "n_annotations": len(d.get("annotations", []))}
    both, fp, n_amber, unknown = [], 0, 0, 0
    for a in d.get("annotations", []):
        attrs = a.get("attributes") or {}
        n_amber += str(attrs.get("status", "")).upper() == "AMBER"
        fi, lb = images.get(a["image_id"]), cats.get(a["category_id"])
        if fi not in gt.by_index or lb not in C.LABELS:
            unknown += 1
            continue
        if gt.present(fi, lb):
            g, valid = gt.mask(fi, lb)
            both.append((f"{fi}", iou(_rasterise(a.get("segmentation"), g.shape), g, valid)))
        else:
            fp += 1
    return {"scored": True, "n_images": len(images), "n_annotations": len(d.get("annotations", [])),
            "n_amber_annotations": n_amber, "n_false_positive_annotations": fp, "n_unmatched": unknown,
            "iou_when_both_present": mean_summary(both)}


# ---------- entry ----------

def score_job(job: Path, raw: Path = DEFAULT_RAW, registry: Path = DEFAULT_REGISTRY, manifest_path: Path | None = None,
              allow_misaligned: bool = False) -> dict:
    job = Path(job)
    meta_path = job / "meta.json"
    if not meta_path.exists():
        raise NoGroundTruth(f"{meta_path} missing: not a job directory, or the job never ingested")
    meta = json.loads(meta_path.read_text())
    manifest = find_manifest(job, Path(registry), manifest_path)
    gt = GroundTruth(manifest, Path(raw))
    alignment = check_alignment(job, meta, gt)
    alignment["n_frames_job"] = meta.get("n_frames")
    alignment["n_frames_truth"] = gt.n_frames
    alignment["fps_job"] = meta.get("fps")
    alignment["fps_truth"] = manifest.get("fps")
    if not alignment["ok"] and not allow_misaligned:
        raise Misaligned(f"keyframes do not match the dataset frames they claim to be: {json.dumps(alignment)}")

    kf_index = {k["kf"]: int(k["frame_index"]) for k in meta["keyframes"] if int(k["frame_index"]) in gt.by_index}
    decisions = _read_jsonl(job / "decisions.jsonl")
    report = {
        "job": str(job),
        "clips": manifest.get("clips"),
        "n_clips": len(manifest.get("clips", [])),
        "n_keyframes": len(kf_index),
        "alignment": alignment,
        "presence_detect_raw": score_presence(_read_jsonl(job / "detect.jsonl"), kf_index, gt, "present"),
        "presence_verify_raw": score_presence(_read_jsonl(job / "verify.jsonl"), kf_index, gt, "present"),
        "presence_final": score_decisions(decisions, kf_index, gt),
        "masks_keyframe": score_keyframe_masks(job, decisions, kf_index, gt),
        "all_frames": score_tracks(job, gt),
        "dataset_coco": score_coco(job / "dataset_coco.json", gt),
        "released_coco": score_coco(job / "released_coco.json", gt),
        "caveat": "Keyframes one second apart inside one surgery are strongly correlated. Treat n as an upper "
                  "bound on independent evidence; n_clips is closer to the real sample size.",
    }
    return report


def _fmt(s: dict | None, key: str = "accuracy") -> str:
    if not s or s.get(key) is None:
        return "no data (n=0)"
    ci = s.get("ci95_by_frame")
    return f"{s[key]:.3f} (n={s['n']}" + (f", 95% CI {ci[0]:.3f}-{ci[1]:.3f}" if ci else "") + ")"


def summary_lines(r: dict) -> list[str]:
    p = r["presence_final"]
    m = r["masks_keyframe"]
    lines = [
        f"clips: {r['clips']}  keyframes: {r['n_keyframes']}",
        f"alignment: {'ok' if r['alignment']['ok'] else 'FAILED'} ({r['alignment']['n_bad']}/{r['alignment']['n_checked']} bad)",
        f"presence accuracy, final decisions: {_fmt(p)}, coverage {p.get('coverage')}, amber {p.get('n_amber')}, "
        f"always-majority baseline {p.get('majority_baseline')}",
        f"presence accuracy, detect only:     {_fmt(r['presence_detect_raw'])}",
        f"presence accuracy, verify only:     {_fmt(r['presence_verify_raw'])}",
        f"mask IoU counting errors as 0:      {_fmt(m['iou_with_errors'], 'mean')}",
        f"mask IoU when both say present:     {_fmt(m['iou_when_both_present'], 'mean')}",
    ]
    for lb, s in m["per_label_with_errors"].items():
        lines.append(f"  {lb:12s} IoU with errors {_fmt(s, 'mean')}")
    if m["n_missing_mask_files"]:
        lines.append(f"WARNING: {m['n_missing_mask_files']} accepted objects have no mask file (scored 0)")
    t = r["all_frames"]
    if t:
        for k, s in t["mask_iou_when_both_present"].items():
            lines.append(f"all-frames mask IoU, {k}: {_fmt(s, 'mean')}")
        for k, s in t["presence"].items():
            lines.append(f"all-frames presence, {k}: {_fmt(s)}")
    for name in ("dataset_coco", "released_coco"):
        c = r[name]
        if c and c.get("scored"):
            lines.append(f"{name}: {c['n_annotations']} annotations, IoU when both present {_fmt(c['iou_when_both_present'], 'mean')}, "
                         f"{c['n_amber_annotations']} amber, {c['n_false_positive_annotations']} false positives")
    lines.append(r["caveat"])
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score a job against ground truth. Never called by the pipeline.")
    ap.add_argument("job", type=Path)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--manifest", type=Path, help="explicit frame map, if the job's source video was re-encoded")
    ap.add_argument("--allow-misaligned", action="store_true", help="score anyway; the report records the failure")
    a = ap.parse_args(argv)
    try:
        report = score_job(a.job, a.raw, a.registry, a.manifest, a.allow_misaligned)
    except NoGroundTruth as e:
        print(f"cannot score: {e}", file=sys.stderr)
        return 2
    except Misaligned as e:
        print(f"refusing to score: {e}", file=sys.stderr)
        return 3
    (a.job / "eval_report.json").write_text(json.dumps(report, indent=1))
    print("\n".join(summary_lines(report)))
    print(f"full report: {a.job / 'eval_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
