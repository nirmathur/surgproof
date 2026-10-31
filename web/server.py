"""SurgProof web UI backend: serves real pipeline outputs as JSON and real CholecSeg8k frames.

Run:  .venv/bin/python -m uvicorn web.server:app --port 8600
Reads only files other lanes write (out/ first, then coord/artifacts/). Never invents numbers:
a missing file is returned as null and the UI shows it as pending.
"""
from __future__ import annotations

import io
import json
import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
COORD = ROOT / "coord" / "artifacts"
CHOLEC = ROOT / "data" / "cholecseg8k" / "CholecSeg8k"
NARRATOR_OUT = ROOT / "narrator" / "out"
DIST = Path(__file__).resolve().parent / "dist"
TEST_VIDEOS = ["video09", "video12", "video17", "video28", "video48", "video55"]
DECISION_COLS = [
    "label_id", "frame_id", "video", "split", "key", "present", "conf", "box", "evidence",
    "pv_match", "p_wrong", "critic", "second_pass_present", "second_pass_conf", "second_pass_agree",
    "status", "final_present", "final_by", "status_round_0", "status_round_1", "status_round_2",
    "status_round_3", "resolved_round", "argon_reason", "model",
]

app = FastAPI(title="SurgProof UI")


def artifact(name: str) -> Path | None:
    for base in (OUT, COORD):
        p = base / name
        if p.exists():
            return p
    return None


def read_json(name: str):
    p = artifact(name)
    if not p:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def read_jsonl(name: str) -> list[dict] | None:
    p = artifact(name)
    if not p:
        return None
    rows = []
    for ln in p.read_text().splitlines():
        try:
            rows.append(json.loads(ln))
        except Exception:
            continue
    return rows


def clean(v):
    """JSON-safe: NaN -> None, numpy scalars -> python."""
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):
        v = v.item()
        if isinstance(v, float) and math.isnan(v):
            return None
    return v


def parse_box(raw):
    if isinstance(raw, list):
        return raw if len(raw) == 4 else None
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            b = json.loads(raw)
            return b if isinstance(b, list) and len(b) == 4 else None
        except Exception:
            return None
    return None


def frame_png(frame_id: str) -> Path | None:
    parts = frame_id.split("_")
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    p = CHOLEC / parts[0] / f"{parts[0]}_{parts[1]}" / f"frame_{int(parts[2])}_endo.png"
    return p if p.is_file() else None


def load_labels() -> tuple[pd.DataFrame | None, str | None]:
    p = artifact("decisions.csv")
    if p:
        return pd.read_csv(p), p.name
    for name in ("labels_manual.jsonl", "labels_smoke.jsonl"):
        p = artifact(name)
        if p:
            d = pd.read_json(p, lines=True)
            if "sample" in d.columns:
                d = d[d["sample"] == 0]
            return d, p.name
    return None, None


@app.get("/api/overview")
def overview():
    res = read_json("results.json")
    stu = read_json("student.json")
    man = read_json("manual.json")
    curve = None
    bc = OUT / "eval" / "budget_curve.csv"
    if bc.exists():
        curve = json.loads(pd.read_csv(bc).to_json(orient="records"))
    flags_src = None
    fp = OUT / "flags.parquet"
    if fp.exists():
        try:
            flags_src = str(pd.read_parquet(fp, columns=["source"])["source"].iloc[0])
        except Exception:
            flags_src = None
    amber = read_jsonl("amber_rounds.jsonl")
    verified = artifact("verified_dataset.jsonl")
    n_verified = sum(1 for _ in verified.open()) if verified else None
    atlases = []
    for ds in sorted((OUT / "atlas").glob("*/dataset.json")) if (OUT / "atlas").exists() else []:
        try:
            atlases.append({"slug": ds.parent.name, **json.loads(ds.read_text())})
        except Exception:
            pass
    return JSONResponse({
        "manual": man,
        "results": res,
        "student": stu,
        "amber_rounds": amber,
        "n_verified_labels": n_verified,
        "tracker": {
            "source": flags_src,
            "summary": json.loads((OUT / "eval" / "summary.json").read_text()) if (OUT / "eval" / "summary.json").exists() else None,
            "budget_curve": curve,
        },
        "atlas": atlases,
        "test_videos": TEST_VIDEOS,
    })


@app.get("/api/labels")
def labels():
    d, src = load_labels()
    if d is None:
        return {"source": None, "rows": []}
    cols = [c for c in DECISION_COLS if c in d.columns]
    rows = []
    for r in d[cols].to_dict(orient="records"):
        r = {k: clean(v) for k, v in r.items()}
        r["box"] = parse_box(r.get("box"))
        rows.append(r)
    return {"source": src, "has_status": "status" in d.columns, "rows": rows}


@app.get("/api/narration")
def narration():
    p = NARRATOR_OUT / "claims.parquet"
    if not p.exists():
        return {"source": None, "claims": []}
    c = pd.read_parquet(p)
    if "source" in c.columns:
        if set(c["source"].unique()) <= {"mock"}:
            return {"source": "mock", "claims": []}
        c = c[c["source"] != "mock"]
    layers = [l for l in ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "P"] if l in c.columns]
    keep = [k for k in ["claim_id", "clip_id", "video_id", "mode", "t_start", "t_end", "text", "verdict", "source"] if k in c.columns]
    out = []
    for r in c.to_dict(orient="records"):
        failed = [l for l in layers if str(r.get(l)) == "fail"]
        v = str(r.get("verdict", "")).lower()
        tag = "verified" if v == "speak" else ("contradicted" if v == "drop" and failed else "unsupported")
        out.append({**{k: clean(r[k]) for k in keep}, "tag": tag, "failed_layers": failed})
    return {"source": "narrator", "claims": out}


@lru_cache(maxsize=512)
def _jpeg(path: str) -> bytes:
    buf = io.BytesIO()
    Image.open(path).convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue()


@app.get("/api/frame/{frame_id}.jpg")
def frame(frame_id: str):
    p = frame_png(frame_id)
    if not p:
        raise HTTPException(404, f"No CholecSeg8k frame for {frame_id}")
    return Response(_jpeg(str(p)), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})



# ---------- Clip player + report (real inputs only: narrator/out/real + decisions.csv) ----------
NARR_REAL = NARRATOR_OUT / "real"
GT_PRESENCE = NARRATOR_OUT / "gt_presence.parquet"


def _jl(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _real_claims() -> pd.DataFrame | None:
    p = NARR_REAL / "claims.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _clip_frames(clip_id: str) -> list[int]:
    vid = clip_id.split("_")[0]
    d = CHOLEC / vid / clip_id
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("frame_*_endo.png"):
        try:
            out.append(int(f.name.split("_")[1]))
        except ValueError:
            pass
    return sorted(out)


@app.get("/api/clips")
def clips():
    d, _ = load_labels()
    c = _real_claims()
    ids = sorted(set(c["clip_id"])) if c is not None else []
    out = []
    for cid in ids:
        fr = _clip_frames(cid)
        if not fr:
            continue
        rows = d[d["frame_id"].str.startswith(cid + "_")] if d is not None else pd.DataFrame()
        st = rows["status"].map(lambda s: str(s)) if "status" in rows else pd.Series(dtype=str)
        cc = c[c["clip_id"] == cid]
        out.append({
            "clip_id": cid, "video": cid.split("_")[0], "split": "test" if cid.split("_")[0] in TEST_VIDEOS else "train",
            "first_frame": fr[0], "last_frame": fr[-1], "n_frames": len(fr),
            "n_labels": int(len(rows)), "n_accept": int((st == "accept").sum()),
            "n_claims": int(len(cc)), "n_spoken": int(((cc["mode"] == "grounded") & (cc["verdict"] == "speak")).sum()),
        })
    return {"clips": out}


@app.get("/api/clip/{clip_id}")
def clip(clip_id: str):
    fr = _clip_frames(clip_id)
    if not fr:
        raise HTTPException(404, f"No frames for {clip_id}")
    d, src = load_labels()
    labels = []
    if d is not None:
        rows = d[d["frame_id"].str.startswith(clip_id + "_")]
        cols = [k for k in DECISION_COLS + ["trust", "amber_rounds_used"] if k in rows.columns]
        for r in rows[cols].to_dict(orient="records"):
            r = {k: clean(v) for k, v in r.items()}
            r["box"] = parse_box(r.get("box"))
            r["frame"] = int(str(r["frame_id"]).split("_")[-1])
            labels.append(r)
    gt = {}
    if GT_PRESENCE.exists() and clip_id.split("_")[0] in TEST_VIDEOS:
        g = pd.read_parquet(GT_PRESENCE, columns=["clip_id", "frame_idx", "label", "present"])
        g = g[g["clip_id"] == clip_id]
        for r in g.to_dict(orient="records"):
            gt[f"{int(r['frame_idx'])}:{r['label']}"] = int(bool(r["present"]))
    for r in labels:
        k = f"{r['frame']}:{r['key']}"
        fp = r.get("final_present") if r.get("final_present") is not None else r.get("present")
        r["gt_present"] = gt.get(k)
        r["gt_correct"] = None if k not in gt or fp is None else bool(int(fp) == gt[k])
    events = []
    ep = NARR_REAL / "events.jsonl"
    if ep.exists():
        for ln in ep.read_text().splitlines():
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("clip_id") == clip_id:
                events.append({k: e.get(k) for k in ("event_id", "type", "label", "t_start", "t_end", "status", "frames", "source")})
    claims_out = []
    c = _real_claims()
    if c is not None:
        sc = NARR_REAL / "claims_scored.parquet"
        gtc = {}
        if sc.exists():
            s = pd.read_parquet(sc, columns=["claim_id", "gt_correct"])
            gtc = {r["claim_id"]: clean(r["gt_correct"]) for r in s.to_dict(orient="records")}
        layers = [l for l in ["L1", "L2", "L3", "L4", "P", "L5", "L5c", "L6", "L7"] if l in c.columns]
        for r in c[c["clip_id"] == clip_id].to_dict(orient="records"):
            claims_out.append({
                "claim_id": r["claim_id"], "mode": r.get("mode"), "type": r.get("type"), "text": r.get("text"),
                "statement": r.get("statement"), "t_start": clean(r.get("t_start")), "t_end": clean(r.get("t_end")),
                "confidence": clean(r.get("confidence")), "verdict": r.get("verdict"),
                "decided_by": _jl(r.get("decided_by")), "reasons": _jl(r.get("reasons")),
                "layers": {l: r.get(l) for l in layers}, "evidence_refs": _jl(r.get("evidence_refs")),
                "gt_correct": gtc.get(r["claim_id"]) if isinstance(gtc.get(r["claim_id"]), bool) else None,
            })
    return {"clip_id": clip_id, "video": clip_id.split("_")[0], "frames": fr, "label_source": src,
            "labels": labels, "events": events, "claims": claims_out}


@lru_cache(maxsize=256)
def _crop(path: str, box: tuple) -> bytes:
    im = Image.open(path).convert("RGB")
    W, H = im.size
    y0, x0, y1, x1 = box
    pad = 30
    l, t = max(0, x0 * W / 1000 - pad), max(0, y0 * H / 1000 - pad)
    r, b = min(W, x1 * W / 1000 + pad), min(H, y1 * H / 1000 + pad)
    buf = io.BytesIO()
    im.crop((int(l), int(t), int(r), int(b))).save(buf, "JPEG", quality=88)
    return buf.getvalue()


@app.get("/api/crop/{frame_id}/{y0}/{x0}/{y1}/{x1}.jpg")
def crop(frame_id: str, y0: int, x0: int, y1: int, x1: int):
    p = frame_png(frame_id)
    if not p or not (0 <= y0 < y1 <= 1000 and 0 <= x0 < x1 <= 1000):
        raise HTTPException(404, "bad crop")
    return Response(_crop(str(p), (y0, x0, y1, x1)), media_type="image/jpeg", headers={"Cache-Control": "max-age=3600"})


if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = DIST / path
        return FileResponse(f if path and f.is_file() else DIST / "index.html")
