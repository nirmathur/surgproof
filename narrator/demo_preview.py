"""Static HTML preview for one clip: frames with boxes, free narration (dropped claims red),
grounded narration with each sentence linked to its evidence frames, layer reasons and KB quotes.

python -m narrator.demo_preview --clip video12_15750 [--out narrator/out/demo_preview.html]
Fallback demo + layout reference for the UI agent.
"""
from __future__ import annotations

import argparse
import base64
import html
import json

import cv2
import pandas as pd

from narrator import cfg as C
from narrator import data as D

COLORS = {"grasper": (60, 180, 255), "l_hook": (255, 160, 40), "liver": (80, 80, 220), "gallbladder": (60, 200, 60),
          "cystic_duct": (220, 60, 220), "blood": (30, 30, 200)}
VCOL = {"speak": "#1a7f37", "hedge": "#9a6700", "drop": "#cf222e"}


def frame_b64(clip_id, f, labels, width=300) -> str:
    im = cv2.imread(str(D.frame_path(clip_id, f)))
    if im is None:
        return ""
    for r in labels[(labels.clip_id == clip_id) & (labels.frame_idx == f) & labels.present].itertuples():
        for x1, y1, x2, y2 in json.loads(r.boxes):
            cv2.rectangle(im, (x1, y1), (x2, y2), COLORS.get(r.label, (255, 255, 255)), 2)
            cv2.putText(im, r.label, (x1 + 3, max(14, y1 + 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLORS.get(r.label, (255, 255, 255)), 2)
    h, w = im.shape[:2]
    im = cv2.resize(im, (width, int(h * width / w)))
    ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode()


def _j(v):
    return json.loads(v) if isinstance(v, str) else (v or {})


def build(clip_id: str) -> str:
    out = C.out_dir()
    claims = pd.read_parquet(out / "claims.parquet")
    claims = claims[claims.clip_id == clip_id]
    by_id = {c["claim_id"]: c for c in claims.to_dict("records")}
    lines = [json.loads(l) for l in (out / "narration.jsonl").read_text().splitlines() if l.strip()]
    lines = [l for l in lines if l["clip_id"] == clip_id]
    kbq = json.loads((out / "kb_quotes.json").read_text()) if (out / "kb_quotes.json").exists() else {}
    labels = D.load_labels()
    clip = D.list_clips().set_index("clip_id").loc[clip_id]
    frames = D.sampled_frames(int(clip["first"]), int(clip["n_frames"]), 16)
    src = str(labels.source.iloc[0])

    def reasons(c):
        r = _j(c.get("reasons"))
        return "; ".join(f"{k}: {html.escape(str(v))}" for k, v in r.items() if v and c.get(k) in ("fail", "pass"))

    def claim_block(cid):
        c = by_id.get(cid)
        if not c:
            return ""
        ev = [r for r in _j(c.get("evidence_refs")) or [] if isinstance(r, str)]
        kb = "".join(f"<div class='kb'>{html.escape(k)}: \"{html.escape(kbq[k].get('quote', ''))}\" "
                     f"({'verified quote' if kbq[k].get('quote_verified') else 'quote not verified'})</div>"
                     for k in _j(c.get("kb_refs")) or [] if k in kbq)
        return (f"<div class='claim'><b style='color:{VCOL.get(c['verdict'])}'>{c['verdict']}</b> "
                f"[{c['type']}/{c['modality']}] {html.escape(str(c.get('text') or ''))}"
                f"<div class='why'>decided by {html.escape(str(_j(c.get('decided_by'))))}. {reasons(c)}</div>"
                f"<div class='ev'>evidence: {html.escape(', '.join(ev[:6]))}</div>{kb}</div>")

    def section(mode):
        rows = []
        for l in sorted([l for l in lines if l["mode"] == mode], key=lambda l: l["t_start"]):
            vs = l.get("verdicts") or []
            bad = mode == "free" and "drop" in vs
            col = VCOL["drop"] if bad else (VCOL["hedge"] if "hedge" in vs and mode == "grounded" else "#1f2328")
            inner = "".join(claim_block(cid) for cid in l["claim_ids"])
            rows.append(f"<details><summary style='color:{col}'><span class='t'>{l['t_start']}-{l['t_end']}</span> "
                        f"{html.escape(l['text'])}{' ✗' if bad else ''}</summary>{inner}</details>")
        return "\n".join(rows) or "<i>no narration</i>"

    imgs = "".join(f"<figure><img src='data:image/jpeg;base64,{frame_b64(clip_id, f, labels)}'/>"
                   f"<figcaption>frame {f}</figcaption></figure>" for f in frames)
    n_free = (claims["mode"] == "free").sum()
    n_drop = ((claims["mode"] == "free") & (claims.verdict == "drop")).sum()
    warn = ("<p class='warn'>Evidence layer L4 ran on a MOCK label table (ground truth + noise): this preview shows the "
            "plumbing, not a validated result.</p>") if src == "mock" else ""
    return f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>Narrator preview {clip_id}</title><style>
body{{font:14px/1.45 -apple-system,Segoe UI,sans-serif;color:#1f2328;max-width:1200px;margin:24px auto;padding:0 16px}}
.frames{{display:flex;gap:8px;overflow-x:auto}} figure{{margin:0}} figcaption{{font-size:11px;color:#6e7781}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:24px}} summary{{cursor:pointer;padding:3px 0}}
.t{{font:11px monospace;color:#6e7781;margin-right:6px}} .claim{{margin:4px 0 8px 16px;font-size:12px}}
.why,.ev,.kb{{color:#6e7781;font-size:11px}} .kb{{color:#0550ae}} .warn{{color:#9a6700}} h2{{font-size:16px}}
</style></head><body><h1>{clip_id}: free vs verified narration</h1>
<p>Post-hoc review of recorded video for teaching. Not intraoperative guidance. Label source: <b>{src}</b>.</p>{warn}
<div class='frames'>{imgs}</div><div class='cols'>
<div><h2>Free VLM narration (baseline): {n_drop}/{n_free} claims dropped by the checker, in red</h2>{section('free')}</div>
<div><h2>Grounded narration: only checked claims, each linked to evidence</h2>{section('grounded')}</div>
</div></body></html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="video12_15750")
    ap.add_argument("--out", default=str(C.out_dir() / "demo_preview.html"))
    a = ap.parse_args()
    open(a.out, "w").write(build(a.clip))
    print("wrote", a.out)
