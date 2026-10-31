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
import numpy as np
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

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
    core3 = merged[merged["key"].isin(["grasper", "l_hook", "gallbladder"])]
    acc_core3 = float(core3["ok"].mean()) if len(core3) else acc_no_liver
    per_key = merged.groupby("key")["ok"].mean().round(3).to_dict()
    return {
        "acc_excl_liver": round(acc_no_liver, 4),
        "acc_core3": round(acc_core3, 4),
        "acc_all": round(acc_all, 4),
        "per_key": per_key,
    }


def risk_coverage(correct, score):
    correct_arr = np.asarray(correct, dtype=float)
    score_arr = np.asarray(score, dtype=float)
    if len(correct_arr) == 0:
        return {"curve": [], "coverage_at_95": 0.0, "acc_at_60": 0.0, "aurc": 0.0}
    order = np.argsort(-score_arr, kind="mergesort")
    c = correct_arr[order]
    n = len(c)
    cov = np.arange(1, n + 1) / n
    acc = np.cumsum(c) / np.arange(1, n + 1)
    step = max(1, n // 50)
    pts = [(round(float(x), 3), round(float(y), 4)) for x, y in zip(cov[::step], acc[::step])]
    if pts[-1][0] != 1.0:
        pts.append((1.0, round(float(acc[-1]), 4)))
    at95 = float(cov[acc >= 0.95].max()) if (acc >= 0.95).any() else 0.0
    idx60 = max(0, min(n - 1, int(round(0.6 * n)) - 1))
    return {
        "curve": pts,
        "coverage_at_95": round(at95, 3),
        "acc_at_60": round(float(acc[idx60]), 4),
        "aurc": round(float((1.0 - acc).mean()), 4),
    }


def score_decisions_file(decisions_path=None, index_path=None, out_results_path=None):
    """Post-decision evaluation step: joins ground truth from index.csv AFTER decisions.csv exists."""
    dec_file = Path(decisions_path) if decisions_path else (C.OUT_DIR / "decisions.csv")
    idx_file = Path(index_path) if index_path else (C.DATA_DIR / "index.csv")
    res_file = Path(out_results_path) if out_results_path else (C.OUT_DIR / "results.json")

    d = pd.read_csv(dec_file)
    idx_df = pd.read_csv(idx_file)
    gt_cols = {k: f"gt_{k}" for k in NAMES if f"gt_{k}" in idx_df.columns}
    gt_long = idx_df[["frame_id"] + list(gt_cols.values())].melt(
        id_vars=["frame_id"], var_name="gt_col", value_name="gt"
    )
    inv_map = {v: k for k, v in gt_cols.items()}
    gt_long["key"] = gt_long["gt_col"].map(inv_map)

    if "gt" in d.columns:
        d = d.drop(columns=["gt"])
    d = d.merge(gt_long[["frame_id", "key", "gt"]], on=["frame_id", "key"], how="left")
    d = d[d["gt"].notna() & (d["gt"] >= 0)].copy()
    d["gt"] = d["gt"].astype(int)
    d["correct"] = (d["present"].astype(int) == d["gt"]).astype(int)
    d["final_correct"] = (d["final_present"].astype(int) == d["gt"]).astype(int)

    test_all = d[d.split == "test"].copy()
    test_5 = test_all[test_all.key != "liver"].copy()
    test = test_5 if len(test_5) else test_all

    acc_t = test[test.status == "accept"]
    acc_t_all = test_all[test_all.status == "accept"]
    amber_mask = test.status.isin(["AMBER", "reanalyse", "escalate"])
    reject_mask = test.status == "reject"

    res = {
        "headline_note": "Headline metrics exclude 100%-prevalence liver (evaluated on the 5 informative classes on held-out test videos).",
        "n_test_labels": int(len(test)),
        "n_test_labels_all6": int(len(test_all)),
        "single_conf": risk_coverage(test.correct, test.conf),
        "system_zero_label": risk_coverage(test.final_correct, test.trust),
        "accepted": {
            "coverage": round(len(acc_t) / len(test), 3) if len(test) else 0.0,
            "accuracy": round(float(acc_t.final_correct.mean()), 4) if len(acc_t) else None,
            "amber_share": round(float(amber_mask.mean()), 3) if len(test) else 0.0,
            "rejected_share": round(float(reject_mask.mean()), 3) if len(test) else 0.0,
            "escalated_share": round(float((test.status != "accept").mean()), 3) if len(test) else 0.0,
        },
        "accepted_all6": {
            "coverage": round(len(acc_t_all) / len(test_all), 3) if len(test_all) else 0.0,
            "accuracy": round(float(acc_t_all.final_correct.mean()), 4) if len(acc_t_all) else None,
            "escalated_share": round(float((test_all.status != "accept").mean()), 3) if len(test_all) else 0.0,
        },
        "manual_raw_acc": round(float(test.correct.mean()), 4) if len(test) else None,
        "manual_raw_acc_all6": round(float(test_all.correct.mean()), 4) if len(test_all) else None,
    }

    try:
        b = pd.read_json(C.OUT_DIR / "labels_blank.jsonl", lines=True)
        b = b[b.split == "test"].merge(gt_long[["frame_id", "key", "gt"]], on=["frame_id", "key"], how="inner")
        b = b[b["gt"].notna() & (b["gt"] >= 0)].copy()
        b["correct"] = (b["present"].astype(int) == b["gt"].astype(int)).astype(int)
        b5 = b[b.key != "liver"]
        b_eval = b5 if len(b5) else b
        res["blank_single_conf"] = risk_coverage(b_eval.correct, b_eval.conf)
        res["blank_raw_acc"] = round(float(b_eval.correct.mean()), 4)
        res["blank_raw_acc_all6"] = round(float(b.correct.mean()), 4)
    except (FileNotFoundError, ValueError):
        pass

    try:
        v = pd.read_json(C.OUT_DIR / "labels_manual_v2.jsonl", lines=True)
        allv = pd.concat([
            d[["frame_id", "key", "present", "conf", "split"]],
            v[["frame_id", "key", "present", "conf", "split"]],
        ])
        g = (
            allv[allv.split == "test"]
            .groupby(["frame_id", "key"], as_index=False)
            .agg(p=("present", "mean"), c=("conf", "mean"))
            .merge(gt_long[["frame_id", "key", "gt"]], on=["frame_id", "key"], how="inner")
        )
        g = g[g.key != "liver"] if (g.key != "liver").any() else g
        g["vote"] = (g.p >= 0.5).astype(int)
        g["agree"] = np.maximum(g.p, 1.0 - g.p) * g.c
        res["majority_vote"] = risk_coverage((g.vote == g.gt.astype(int)).astype(int), g.agree)
    except (FileNotFoundError, ValueError):
        pass

    cal = d[(d.split == "calib") & (d.key != "liver")]
    if len(cal) > 20 and cal.correct.nunique() == 2 and len(test) > 0:
        def _feats(x):
            pvm = pd.to_numeric(x.pv_match, errors="coerce").fillna(0.5).to_numpy(dtype=float)
            pvc = pd.to_numeric(x.pv_conf, errors="coerce").fillna(0.5).to_numpy(dtype=float)
            return np.c_[x.conf.to_numpy(dtype=float), x.p_wrong.to_numpy(dtype=float),
                         x.present.to_numpy(dtype=float), pvm, pvc]
        lr = LogisticRegression().fit(_feats(cal), cal.correct)
        res["upper_bound_calibrated"] = risk_coverage(test.correct, lr.predict_proba(_feats(test))[:, 1])

    res["per_label_accepted_acc"] = (
        acc_t_all.groupby("key").final_correct.mean().round(3).to_dict() if len(acc_t_all) else {}
    )
    res["per_label_raw_acc"] = (
        test_all.groupby("key").correct.mean().round(3).to_dict() if len(test_all) else {}
    )
    res["per_label_accepted_coverage"] = (
        test_all.assign(_acc=(test_all.status == "accept").astype(int)).groupby("key")["_acc"].mean().round(3).to_dict()
        if len(test_all) else {}
    )

    res_file.parent.mkdir(parents=True, exist_ok=True)
    res_file.write_text(json.dumps(res, indent=2))
    coord_res = C.ROOT / "coord" / "artifacts" / "results.json"
    coord_res.parent.mkdir(parents=True, exist_ok=True)
    coord_res.write_text(json.dumps(res, indent=2))
    return res


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
            f"headline acc (excl. liver): {metrics['acc_excl_liver']:.3f} | "
            f"core-3 acc: {metrics['acc_core3']:.3f} | all-6 acc: {metrics['acc_all']:.3f}",
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
