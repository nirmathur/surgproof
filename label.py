"""Step 4: label frames with Flash, then verify with decorrelated checks.

Checks per frame:
  1. labeller   - full frame, presence + confidence + box for each label
  2. point-and-verify - crop each claimed box, ask a FRESH call what it is (blind to the claim)
  3. critic     - argue each claim is wrong, return p_wrong

  python label.py --mode manual            # main run (all splits), uses out/manual.json
  python label.py --mode blank --splits test   # baseline: no manual
  python label.py --mode manual --votes 2 --splits train,test   # extra samples for majority vote
  python label.py --limit 40               # quick smoke test
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

import config as C
import gem

try:
    gem._db.execute("PRAGMA journal_mode=WAL")
    gem._db.execute("PRAGMA busy_timeout=30000")
except Exception:
    pass


class Item(BaseModel):
    key: str
    present: bool
    confidence: float          # 0-1 that `present` is correct
    box_2d: list[int]          # [ymin, xmin, ymax, xmax] 0-1000, empty if absent
    evidence: str


class FrameLabels(BaseModel):
    items: list[Item]


class CropID(BaseModel):
    answer: str
    confidence: float


class Rebuttal(BaseModel):
    key: str
    argument: str
    p_wrong: float


class Critique(BaseModel):
    rebuttals: list[Rebuttal]


NAMES = {k: v[1] for k, v in C.LABELS.items()}

KEY_ALIASES = {}
for _k, _name in NAMES.items():
    KEY_ALIASES[_k.lower()] = _k
    KEY_ALIASES[_k.lower().replace("_", " ")] = _k
    KEY_ALIASES[_k.lower().replace("_", "-")] = _k
    KEY_ALIASES[_name.lower()] = _k
    KEY_ALIASES[_name.lower().split(" (")[0].strip()] = _k


def normalize_key(raw_key):
    if not isinstance(raw_key, str):
        return None
    s = raw_key.strip().lower()
    if s in KEY_ALIASES:
        return KEY_ALIASES[s]
    s_prefix = s.split(" (")[0].strip()
    return KEY_ALIASES.get(s_prefix)


def clamp_prob(val, default=0.5):
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    if 1.0 < f <= 100.0:
        f = f / 100.0
    return round(min(1.0, max(0.0, f)), 4)


def sanitize_box(box, present):
    if not present or not isinstance(box, (list, tuple)) or len(box) != 4:
        return []
    try:
        coords = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return []
    y0, x0, y1, x1 = [min(1000, max(0, c)) for c in coords]
    y0, y1 = min(y0, y1), max(y0, y1)
    x0, x1 = min(x0, x1), max(x0, x1)
    if (y1 - y0) < 5 or (x1 - x0) < 5:
        return []
    return [y0, x0, y1, x1]


def matches_crop_answer(answer, key):
    if not isinstance(answer, str):
        return False
    ans = answer.strip().lower()
    if not ans or "none" in ans or "unclear" in ans:
        return False
    norm = normalize_key(ans)
    if norm == key:
        return True
    target_name = NAMES[key].lower()
    short_name = target_name.split(" (")[0].strip()
    if ans == target_name or short_name in ans or key.lower().replace("_", " ") in ans:
        return True
    if key == "l_hook" and ("l-hook" in ans or "l_hook" in ans or "hook" in ans):
        return True
    return False


def prompt_hash(prompt_text):
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]


def label_prompt(manual):
    base = ("You are labelling a single frame from laparoscopic cholecystectomy video.\n"
            f"For EACH of these keys decide if it is visible: {json.dumps(NAMES)}.\n"
            "Return one item per key. confidence = probability your present/absent call is correct. "
            "If present, give a tight box_2d [ymin,xmin,ymax,xmax] normalised 0-1000; if absent, []. "
            "Evidence: one short phrase on what you saw.")
    if manual:
        base += "\n\nUse this labelling manual you wrote earlier:\n" + json.dumps(manual)
    return base


async def verify_box(path, key, box):
    part = gem.img_part(path, crop=box)
    options = list(NAMES.values()) + ["none / unclear"]
    r = await gem.call(
        "This is a crop from a laparoscopic surgery frame. What is the main structure or instrument "
        f"in it? Answer with exactly one of: {json.dumps(options)}.",
        images=[part], schema=CropID, tag="verify")
    ans = r.get("answer", "") if isinstance(r, dict) else ""
    conf = clamp_prob(r.get("confidence", 0.5) if isinstance(r, dict) else 0.5, default=0.5)
    return matches_crop_answer(ans, key), conf


async def critique(path, items):
    claims = [{"key": i["key"], "claim": "present" if i["present"] else "absent"} for i in items]
    r = await gem.call(
        "You are a sceptical surgical reviewer. For each claim about this frame, make the strongest "
        "case that it is WRONG, then give p_wrong (0-1) honestly; low if the claim is clearly right.\n"
        f"Claims: {json.dumps(claims)}",
        images=[gem.img_part(path)], schema=Critique, tag="critic")
    rebuttals = r.get("rebuttals", []) if isinstance(r, dict) else []
    out = {}
    for x in rebuttals:
        if not isinstance(x, dict):
            continue
        k = normalize_key(x.get("key"))
        if k:
            out[k] = x
    return out


async def do_frame(row, mode, manual, sample, full_checks):
    temp = 0.0 if sample == 0 else 0.7
    p_text = label_prompt(manual if mode == "manual" else None)
    p_hash = prompt_hash(p_text)
    lab = await gem.call(p_text,
                         images=[gem.img_part(row.path)], schema=FrameLabels,
                         temperature=temp, sample=sample, tag=f"label-{mode}")
    raw_items = lab.get("items", []) if isinstance(lab, dict) else []
    items = {}
    for i in raw_items:
        if not isinstance(i, dict):
            continue
        k = normalize_key(i.get("key"))
        if k and k not in items:
            present = bool(i.get("present", False))
            items[k] = {
                "key": k,
                "present": present,
                "confidence": clamp_prob(i.get("confidence", 0.0), default=0.0),
                "box_2d": sanitize_box(i.get("box_2d", []), present),
                "evidence": str(i.get("evidence", "") or ""),
            }
    for k in NAMES:
        if k not in items:
            items[k] = {"key": k, "present": False, "confidence": 0.0, "box_2d": [], "evidence": "missing"}

    ordered_items = [items[k] for k in NAMES]
    pv_results = {}
    if full_checks:
        pv_keys = [k for k in NAMES if items[k]["present"] and len(items[k]["box_2d"]) == 4]
        crit_coro = critique(row.path, ordered_items)
        pv_coros = [verify_box(row.path, k, items[k]["box_2d"]) for k in pv_keys]
        gathered = await asyncio.gather(crit_coro, *pv_coros)
        crit = gathered[0]
        for k, res in zip(pv_keys, gathered[1:]):
            pv_results[k] = res
    else:
        crit = {}

    out = []
    video_str = str(row.video)
    for key in NAMES:
        it = items[key]
        conf_val = float(it["confidence"])
        ev_val = it["evidence"]
        rec = {
            "frame_id": row.frame_id,
            "video": video_str,
            "video_id": video_str,
            "split": row.split,
            "key": key,
            "mode": mode,
            "sample": int(sample),
            "model": C.FLASH,
            "prompt_hash": p_hash,
            "present": int(it["present"]),
            "conf": conf_val,
            "confidence": conf_val,
            "box": it["box_2d"],
            "evidence": ev_val,
            "rationale": ev_val,
        }
        if full_checks:
            if key in pv_results:
                rec["pv_match"], rec["pv_conf"] = pv_results[key]
            else:
                rec["pv_match"], rec["pv_conf"] = (None, None)
            c_entry = crit.get(key, {})
            rec["p_wrong"] = clamp_prob(c_entry.get("p_wrong", 0.5), default=0.5)
            rec["critic"] = str(c_entry.get("argument", "") or "")
        else:
            rec["pv_match"], rec["pv_conf"] = (None, None)
            rec["p_wrong"] = 0.5
            rec["critic"] = ""
        out.append(rec)
    return out


def select_frames(df, splits, limit, videos=None):
    df = df[df.split.isin(splits)].copy()
    if videos:
        df = df[df.video.astype(str).isin(videos)].copy()
    if limit and limit < len(df):
        df["_rk"] = df.groupby("video").cumcount()
        df = df.sort_values(["_rk", "video", "frame_id"]).head(limit).drop(columns=["_rk"])
        df = df.sort_values(["video", "frame_id"]).reset_index(drop=True)
    return df


def evaluate_against_index(d, idx_df):
    """Join GT from index_df in-memory only for console summary (never written to labels JSONL)."""
    gt_cols = {k: f"gt_{k}" for k in NAMES if f"gt_{k}" in idx_df.columns}
    if not gt_cols or len(d) == 0:
        return None
    gt_long = idx_df[["frame_id"] + list(gt_cols.values())].melt(
        id_vars=["frame_id"], var_name="gt_col", value_name="gt"
    )
    inv_map = {v: k for k, v in gt_cols.items()}
    gt_long["key"] = gt_long["gt_col"].map(inv_map)
    merged = d.merge(gt_long[["frame_id", "key", "gt"]], on=["frame_id", "key"], how="inner")
    merged = merged[merged["gt"].notna() & (merged["gt"] >= 0)]
    if len(merged) == 0:
        return None
    merged["ok"] = merged["present"] == merged["gt"].astype(int)
    acc_all = float(merged["ok"].mean())
    no_liver = merged[merged["key"] != "liver"]
    acc_no_liver = float(no_liver["ok"].mean()) if len(no_liver) else acc_all
    per_key = merged.groupby("key")["ok"].mean().round(3).to_dict()
    return {"acc_excl_liver": round(acc_no_liver, 4), "acc_all": round(acc_all, 4), "per_key": per_key}


async def run(mode, splits, votes, limit, out_path=None, index_path=None, videos=None, manual_path=None):
    idx_file = Path(index_path) if index_path else (C.DATA_DIR / "index.csv")
    df_full = pd.read_csv(idx_file)
    df = select_frames(df_full, splits, limit, videos=videos)
    m_file = Path(manual_path) if manual_path else (C.OUT_DIR / "manual.json")
    manual = json.loads(m_file.read_text()) if mode == "manual" else None
    samples = range(1, votes + 1) if votes else [0]
    jobs = [do_frame(r, mode, manual, s, full_checks=(s == 0 and mode == "manual"))
            for r in df.itertuples() for s in samples]
    gem.trace("flash", f"Labelling {len(df)} frames, mode={mode}, samples={list(samples)}")
    recs = []
    for i, coro in enumerate(asyncio.as_completed(jobs), 1):
        recs += await coro
        if i % 20 == 0 or i == len(jobs):
            print(f"{i}/{len(jobs)} frames done", flush=True)
    key_order = {k: idx for idx, k in enumerate(NAMES)}
    recs.sort(key=lambda r: (r["frame_id"], key_order.get(r["key"], 99), r["sample"]))
    tag = f"{mode}_v{votes}" if votes else mode
    path = Path(out_path) if out_path else (C.OUT_DIR / f"labels_{tag}.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            assert "truth" not in r and not any(k.startswith("gt_") for k in r), "Leakage guard: no GT in labels output"
            f.write(json.dumps(r) + "\n")
    if limit and mode == "manual" and not votes and not out_path:
        smoke_path = C.ROOT / "coord" / "artifacts" / "labels_smoke.jsonl"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        with open(smoke_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"saved smoke artifact {smoke_path} ({len(recs)} rows, 0 GT fields)", flush=True)
    d = pd.DataFrame(recs)
    metrics = evaluate_against_index(d, df)
    if metrics:
        print(
            f"saved {path} ({len(recs)} rows, {d.frame_id.nunique()} frames, 0 GT fields in file). "
            f"headline accuracy (excl. liver): {metrics['acc_excl_liver']:.3f} | all-6 raw acc: {metrics['acc_all']:.3f}",
            flush=True,
        )
        print(pd.Series(metrics["per_key"], name="ok").to_string(), flush=True)
    else:
        print(f"saved {path} ({len(recs)} rows, {d.frame_id.nunique() if len(d) else 0} frames, 0 GT fields in file).", flush=True)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["manual", "blank"], default="manual")
    ap.add_argument("--splits", default="train,calib,test")
    ap.add_argument("--votes", type=int, default=0, help="extra sampled runs for majority vote")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None, help="optional explicit output jsonl path")
    ap.add_argument("--index", default=None, help="optional index csv path (e.g. external test set)")
    ap.add_argument("--video", default=None, help="optional comma-separated video filter")
    ap.add_argument("--manual", default=None, help="optional manual.json path")
    a = ap.parse_args()
    vids = [v.strip() for v in a.video.split(",") if v.strip()] if a.video else None
    asyncio.run(run(a.mode, a.splits.split(","), a.votes, a.limit, a.out, a.index, vids, a.manual))
