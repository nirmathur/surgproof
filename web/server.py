"""SurgProof unified web UI backend: serves real pipeline outputs as JSON, real CholecSeg8k frames,
AI masks/polygons, and the Final Pass Human Gate (/api/gate GET & POST).

Run:  python3 -m uvicorn web.server:app --port 8600   (or ./run.sh / make run)
Configurable via environment variables:
  - SURGPROOF_ROOT          (default: repo root)
  - SURGPROOF_OUT_DIR       (default: <root>/out)
  - SURGPROOF_COORD_DIR     (default: <root>/coord/artifacts)
  - SURGPROOF_CHOLEC_DIR    (default: auto-detected from <root>/raw/CholecSeg8k or <root>/data/cholecseg8k/CholecSeg8k)
  - SURGPROOF_NARRATOR_DIR  (default: <root>/narrator/out)

Reads only files other lanes write (out/ first, then coord/artifacts/). Never invents numbers:
a missing file is returned as null and the UI shows it as pending.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(os.environ.get("SURGPROOF_ROOT", Path(__file__).resolve().parents[1])).resolve()
WEB_DIR = Path(__file__).resolve().parent
OUT = Path(os.environ.get("SURGPROOF_OUT_DIR", ROOT / "out"))
COORD = Path(os.environ.get("SURGPROOF_COORD_DIR", ROOT / "coord" / "artifacts"))
NARRATOR_OUT = Path(os.environ.get("SURGPROOF_NARRATOR_DIR", ROOT / "narrator" / "out"))
DIST = WEB_DIR / "dist"


def _resolve_default_cholec_dir() -> Path:
    env_cholec = os.environ.get("SURGPROOF_CHOLEC_DIR") or os.environ.get("SURGPROOF_RAW_DIR")
    if env_cholec:
        p = Path(env_cholec)
        if (p / "CholecSeg8k").is_dir():
            return p / "CholecSeg8k"
        return p
    for cand in (
        ROOT / "raw" / "CholecSeg8k",
        ROOT / "data" / "cholecseg8k" / "CholecSeg8k",
        ROOT / "raw",
    ):
        if cand.is_dir():
            return cand
    return ROOT / "raw" / "CholecSeg8k"


CHOLEC = _resolve_default_cholec_dir()
TEST_VIDEOS = ["video09", "video12", "video17", "video28", "video48", "video55"]
DECISION_COLS = [
    "label_id", "frame_id", "video", "split", "key", "present", "conf", "box", "evidence",
    "pv_match", "p_wrong", "critic", "second_pass_present", "second_pass_conf", "second_pass_agree",
    "status", "final_present", "final_by", "status_round_0", "status_round_1", "status_round_2",
    "status_round_3", "resolved_round", "argon_reason", "model",
]

app = FastAPI(title="SurgProof Unified UI")


def _cholec_roots() -> list[Path]:
    roots = [CHOLEC]
    for fallback in (
        ROOT / "raw" / "CholecSeg8k",
        ROOT / "data" / "cholecseg8k" / "CholecSeg8k",
        ROOT / "raw",
    ):
        if fallback not in roots:
            roots.append(fallback)
    return roots


@lru_cache(maxsize=4)
def _index_frame_map(index_csv_str: str) -> dict[str, str]:
    p = Path(index_csv_str)
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        with p.open("r", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                fid = (r.get("frame_id") or "").strip()
                fpath = (r.get("path") or r.get("img_path") or "").strip()
                if fid and fpath:
                    out[fid] = fpath
    except Exception:
        pass
    return out


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
    if len(parts) == 3 and parts[2].isdigit():
        for root_dir in _cholec_roots():
            p = root_dir / parts[0] / f"{parts[0]}_{parts[1]}" / f"frame_{int(parts[2])}_endo.png"
            if p.is_file():
                return p
    # Fallback to data/index.csv path column
    idx_map = _index_frame_map(str(ROOT / "data" / "index.csv"))
    mapped = idx_map.get(frame_id)
    if mapped:
        mp = Path(mapped)
        if mp.is_file():
            return mp
        rel = ROOT / mapped.lstrip("/")
        if rel.is_file():
            return rel
    return None


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


@app.get("/api/health")
@app.get("/health")
def health():
    return {"ok": True, "service": "surgproof-web", "cholec_dir": str(CHOLEC), "out_dir": str(OUT)}


@app.get("/api/overview")
def overview():
    res = read_json("results.json")
    stu = read_json("student.json")
    man = read_json("manual.json")
    loc_res = read_json("localize_results.json")
    mask_res = read_json("mask_results.json")
    mask_ref = read_json("mask_refine_results.json")
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
        "localize_results": loc_res,
        "mask_results": mask_res,
        "mask_refine_results": mask_ref,
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


# ---------- Clip player + report (real inputs only: narrator/out/real + decisions.csv + data/index.csv fallback) ----------
def _jl(v):
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except Exception:
            return v
    return v


def _real_claims() -> pd.DataFrame | None:
    p = NARRATOR_OUT / "real" / "claims.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _clip_frames(clip_id: str) -> list[int]:
    vid = clip_id.split("_")[0]
    for root_dir in _cholec_roots():
        d = root_dir / vid / clip_id
        if d.is_dir():
            out = []
            for f in d.glob("frame_*_endo.png"):
                try:
                    out.append(int(f.name.split("_")[1]))
                except ValueError:
                    pass
            if out:
                return sorted(out)
    # Fallback to data/index.csv
    idx_csv = ROOT / "data" / "index.csv"
    if idx_csv.is_file():
        out = []
        try:
            with idx_csv.open("r", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("clip") == clip_id:
                        fid = str(r.get("frame_id") or "")
                        tail = fid.split("_")[-1]
                        if tail.isdigit():
                            out.append(int(tail))
        except Exception:
            pass
        if out:
            return sorted(set(out))
    return []


@app.get("/api/clips")
def clips():
    d, _ = load_labels()
    c = _real_claims()
    ids_set: set[str] = set()
    if c is not None and "clip_id" in c.columns:
        ids_set.update(str(x) for x in c["clip_id"].dropna().unique())
    if d is not None and "frame_id" in d.columns:
        for fid in d["frame_id"].dropna().unique():
            parts = str(fid).split("_")
            if len(parts) == 3 and parts[0] in TEST_VIDEOS:
                ids_set.add(f"{parts[0]}_{parts[1]}")
    idx_csv = ROOT / "data" / "index.csv"
    if not ids_set and idx_csv.is_file():
        try:
            with idx_csv.open("r", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    cid = (r.get("clip") or "").strip()
                    if cid:
                        ids_set.add(cid)
        except Exception:
            pass
    if not ids_set:
        for root_dir in _cholec_roots():
            for vid in TEST_VIDEOS:
                vdir = root_dir / vid
                if vdir.is_dir():
                    for cdir in vdir.iterdir():
                        if cdir.is_dir():
                            ids_set.add(cdir.name)
    ids = sorted(ids_set)
    out = []
    for cid in ids:
        fr = _clip_frames(cid)
        if not fr:
            continue
        rows = d[d["frame_id"].str.startswith(cid + "_")] if d is not None else pd.DataFrame()
        st = rows["status"].map(lambda s: str(s)) if "status" in rows else pd.Series(dtype=str)
        cc = c[c["clip_id"] == cid] if (c is not None and "clip_id" in c.columns) else pd.DataFrame()
        n_spoken = (
            int(((cc["mode"] == "grounded") & (cc["verdict"] == "speak")).sum())
            if ("mode" in cc.columns and "verdict" in cc.columns)
            else 0
        )
        out.append({
            "clip_id": cid,
            "video": cid.split("_")[0],
            "split": "test" if cid.split("_")[0] in TEST_VIDEOS else "train",
            "first_frame": fr[0],
            "last_frame": fr[-1],
            "n_frames": len(fr),
            "n_labels": int(len(rows)),
            "n_accept": int((st == "accept").sum()),
            "n_claims": int(len(cc)),
            "n_spoken": n_spoken,
        })
    return {"clips": out}


@app.get("/api/clip/{clip_id}")
def clip(clip_id: str):
    fr = _clip_frames(clip_id)
    if not fr:
        raise HTTPException(404, f"No frames for {clip_id}")
    d, src = load_labels()
    labels_list = []
    if d is not None:
        rows = d[d["frame_id"].str.startswith(cid_prefix := (clip_id + "_"))]
        cols = [k for k in DECISION_COLS + ["trust", "amber_rounds_used"] if k in rows.columns]
        for r in rows[cols].to_dict(orient="records"):
            r = {k: clean(v) for k, v in r.items()}
            r["box"] = parse_box(r.get("box"))
            r["frame"] = int(str(r["frame_id"]).split("_")[-1])
            labels_list.append(r)
    gt: dict[str, int] = {}
    gt_presence = NARRATOR_OUT / "gt_presence.parquet"
    if gt_presence.exists() and clip_id.split("_")[0] in TEST_VIDEOS:
        g = pd.read_parquet(gt_presence, columns=["clip_id", "frame_idx", "label", "present"])
        g = g[g["clip_id"] == clip_id]
        for r in g.to_dict(orient="records"):
            gt[f"{int(r['frame_idx'])}:{r['label']}"] = int(bool(r["present"]))
    elif (ROOT / "data" / "index.csv").is_file() and clip_id.split("_")[0] in TEST_VIDEOS:
        try:
            with (ROOT / "data" / "index.csv").open("r", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    if r.get("clip") == clip_id:
                        fidx = int(str(r["frame_id"]).split("_")[-1])
                        for k in ("grasper", "l_hook", "liver", "gallbladder", "cystic_duct", "blood"):
                            v = r.get(f"gt_{k}")
                            if v not in (None, "", "nan", "NaN"):
                                gt[f"{fidx}:{k}"] = int(float(v) > 0)
        except Exception:
            pass
    for r in labels_list:
        k = f"{r['frame']}:{r['key']}"
        fp = r.get("final_present") if r.get("final_present") is not None else r.get("present")
        r["gt_present"] = gt.get(k)
        r["gt_correct"] = None if k not in gt or fp is None else bool(int(fp) == gt[k])
    events = []
    ep = NARRATOR_OUT / "real" / "events.jsonl"
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
    if c is not None and "clip_id" in c.columns:
        sc = NARRATOR_OUT / "real" / "claims_scored.parquet"
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
    return {
        "clip_id": clip_id,
        "video": clip_id.split("_")[0],
        "frames": fr,
        "label_source": src,
        "labels": labels_list,
        "events": events,
        "claims": claims_out,
    }


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


# ---------- Final Pass Human Gate (/api/gate GET & POST) ----------
@app.get("/api/gate")
@app.get("/gate")
def get_gate_state(scope: str = Query(default="test_5")):
    from web.serve import build_gate_state_payload
    dec_path = artifact("decisions.csv") or (OUT / "decisions.csv")
    return JSONResponse(
        build_gate_state_payload(
            scope=scope,
            decisions_csv_path=dec_path,
            gate_log_path=OUT / "gate_log.jsonl",
            index_csv_path=ROOT / "data" / "index.csv",
        )
    )


@app.post("/api/gate")
@app.post("/gate")
async def post_gate_action(request: Request):
    from web.serve import handle_gate_post
    try:
        payload = await request.json()
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"Invalid JSON body: {exc}"}, status_code=400)
    dec_path = artifact("decisions.csv") or (OUT / "decisions.csv")
    status_code, resp = handle_gate_post(
        payload,
        decisions_csv_path=dec_path,
        gate_log_path=OUT / "gate_log.jsonl",
        human_gate_log_path=OUT / "human_gate_log.jsonl",
        verified_jsonl_path=OUT / "verified_dataset.jsonl",
        index_csv_path=ROOT / "data" / "index.csv",
    )
    return JSONResponse(resp, status_code=int(status_code))


# ---------- Static assets (Studio web/*, Watch mode, out/, data/, raw/, and React web/dist) ----------
if DIST.exists() and (DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/studio")
def studio_index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/{path:path}")
def spa(path: str):
    if not path:
        if (WEB_DIR / "index.html").is_file():
            return FileResponse(WEB_DIR / "index.html")
        if (DIST / "index.html").is_file():
            return FileResponse(DIST / "index.html")
    clean_rel = path.lstrip("/")
    # 1. Direct match inside web/ (index.html, watch.html, app.js, viewer.js, timeline.js, gate.js, style.css, data/*)
    cand_web = (WEB_DIR / clean_rel).resolve()
    if str(cand_web).startswith(str(WEB_DIR.resolve())) and cand_web.is_file():
        return FileResponse(cand_web)
    # 2. Match inside web/dist/
    if DIST.exists():
        cand_dist = (DIST / clean_rel).resolve()
        if str(cand_dist).startswith(str(DIST.resolve())) and cand_dist.is_file():
            return FileResponse(cand_dist)
    # 3. Safe fallback to repo root for out/, data/, raw/, coord/
    for allowed_prefix in ("out/", "data/", "raw/", "coord/"):
        if clean_rel.startswith(allowed_prefix):
            cand_repo = (ROOT / clean_rel).resolve()
            if cand_repo.is_file():
                return FileResponse(cand_repo)
    if (DIST / "index.html").is_file():
        return FileResponse(DIST / "index.html")
    if (WEB_DIR / "index.html").is_file():
        return FileResponse(WEB_DIR / "index.html")
    raise HTTPException(404, f"Not found: {path}")
