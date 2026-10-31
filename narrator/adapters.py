"""Adapters from the real labeller/verifier outputs (docs/DESIGN.md 1A/1B) to the narrator label table.

Narrator table: video_id, clip_id, frame_idx, label, present, status (accepted|uncertain|rejected),
confidence, boxes (json [[x1,y1,x2,y2]] px), source.
Preferred input: out/decisions.csv (verifier status accept/AMBER/reject, final_present).
Fallback: out/labels_manual.jsonl (labeller only: conf >= 0.85 -> accepted, else uncertain).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from narrator import cfg as C

W, H = 854, 480
STATUS = {"accept": "accepted", "amber": "uncertain", "reanalyse": "uncertain", "reject": "rejected",
          "escalate": "rejected"}


def box_to_px(box) -> str:
    if isinstance(box, str):
        try:
            box = json.loads(box)
        except Exception:  # noqa: BLE001
            return "[]"
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return "[]"
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    return json.dumps([[int(xmin * W / 1000), int(ymin * H / 1000), int(xmax * W / 1000), int(ymax * H / 1000)]])


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    df["clip_id"] = df.frame_id.str.rsplit("_", n=1).str[0]
    df["frame_idx"] = df.frame_id.str.rsplit("_", n=1).str[1].astype(int)
    df["video_id"] = df.clip_id.str.split("_").str[0]
    df["label"] = df["key"]
    return df


def from_decisions(p: Path) -> pd.DataFrame:
    d = _keys(pd.read_csv(p))
    d["status"] = d.status.astype(str).str.lower().map(STATUS).fillna("uncertain")
    pres = d["final_present"] if "final_present" in d else d["present"]
    d["present"] = pres.fillna(0).astype(int) == 1
    d["confidence"] = d.get("trust", d.get("conf"))
    d["boxes"] = d.box.apply(box_to_px)
    d["source"] = "verifier"
    return d


def from_labels(p: Path) -> pd.DataFrame:
    d = _keys(pd.DataFrame([json.loads(l) for l in p.read_text().splitlines() if l.strip()]))
    if "sample" in d:
        d = d[d["sample"] == 0]
    d["present"] = d.present.astype(int) == 1
    d["confidence"] = d.get("conf", 0.5)
    d["status"] = (d.confidence >= 0.85).map({True: "accepted", False: "uncertain"})
    d["boxes"] = d.box.apply(box_to_px)
    d["source"] = "labeller"
    return d


def load_real_labels(paths=None) -> pd.DataFrame:
    dec = C.ROOT / "out" / "decisions.csv"
    d = from_decisions(dec) if dec.exists() else from_labels(C.ROOT / "out" / "labels_manual.jsonl")
    d["split"] = d.video_id.map(C.split_of)
    cols = ["video_id", "clip_id", "frame_idx", "label", "present", "status", "confidence", "boxes", "source", "split"]
    return d[cols].drop_duplicates(["video_id", "clip_id", "frame_idx", "label"])
