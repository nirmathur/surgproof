"""Step 7 (stretch): 'Netflix for surgery'. Narrate a test video using ONLY verified labels.
Pre-generated, so the demo plays back with no latency.

  python narrate.py --video video12      # writes out/narration_video12.json
  python narrate.py --video auto         # narrates first test video in decisions.csv
"""
import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

import config as C
import gem


class Line(BaseModel):
    narration: str        # 1-2 sentences, expert commentator voice
    unsure_about: list[str]


def _resolve_artifact(filename: str) -> Path:
    primary = C.OUT_DIR / filename
    if primary.exists():
        return primary
    fallback = C.ROOT / "coord" / "artifacts" / filename
    if fallback.exists():
        return fallback
    return primary


def _frame_sort_key(frame_id: str):
    parts = str(frame_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (str(frame_id), 0)


def _resolve_image_path(frame_id: str, rows: pd.DataFrame, idx: pd.DataFrame | None) -> Path | None:
    candidates = []
    if idx is not None and frame_id in idx.index:
        val = idx.loc[frame_id, "path"]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        if pd.notna(val):
            candidates.append(str(val))
    if "frame_path" in rows.columns:
        fp = rows["frame_path"].dropna()
        if not fp.empty:
            candidates.append(str(fp.iloc[0]))
    for raw in candidates:
        p = Path(raw)
        if p.is_file():
            return p
        rel = C.ROOT / raw
        if rel.is_file():
            return rel
    return None


async def main(video: str, limit: int = 0):
    dec_path = _resolve_artifact("decisions.csv")
    dec = pd.read_csv(dec_path)
    vid_col = "video" if "video" in dec.columns else "video_id"

    if video == "auto":
        test_vids = sorted(dec[dec["split"] == "test"][vid_col].dropna().unique()) if "split" in dec.columns else []
        all_vids = sorted(dec[vid_col].dropna().unique())
        video = (test_vids or all_vids)[0]

    idx_path = C.DATA_DIR / "index.csv"
    idx = pd.read_csv(idx_path).set_index("frame_id") if idx_path.exists() else None

    manual_path = _resolve_artifact("manual.json")
    manual = json.loads(manual_path.read_text()) if manual_path.exists() else {}

    frames = sorted(dec[dec[vid_col] == video]["frame_id"].unique(), key=_frame_sort_key)
    if limit and limit > 0:
        frames = frames[:limit]

    lines, prev = [], ""
    for f in frames:
        rows = dec[dec["frame_id"] == f]
        verified = {r.key: bool(r.final_present) for r in rows.itertuples() if r.status == "accept"}
        unsure = [r.key for r in rows.itertuples() if r.status != "accept"]
        img_path = _resolve_image_path(f, rows, idx)
        images = [gem.img_part(img_path)] if img_path is not None else []
        out = await gem.call(
            "You are narrating a laparoscopic cholecystectomy for trainees, like an expert commentator. "
            "Describe what is happening in 1-2 sentences. You may ONLY state facts supported by the "
            f"VERIFIED labels: {json.dumps(verified)}. For anything in UNSURE {json.dumps(unsure)}, say "
            "you are not yet confident rather than guessing. Use the manual for context on phases and "
            f"safety (e.g. critical view of safety). Previous line: {prev!r}\nManual: {json.dumps(manual)}",
            images=images, schema=Line, temperature=0.3, tag="narrate")
        prev = out["narration"]
        evidence_list = [
            {"frame_id": str(f), "key": str(r.key), "present": bool(r.final_present)}
            for r in rows.itertuples()
            if r.status == "accept"
        ]
        unsure_list = out.get("unsure_about", unsure)
        claims_list = [
            {"text": out["narration"], "status": "verified", "evidence_frames": [str(f)]}
        ]
        for u in unsure_list:
            claims_list.append({"text": f"{u} (withheld: not verified)", "status": "unsupported", "evidence_frames": []})
        rec = {
            "frame_id": f,
            "narration": out["narration"],
            "unsure_about": unsure_list,
            "evidence": evidence_list,
            "claims": claims_list,
        }
        lines.append(rec)
        gem.trace("flash", f"Narrated {f}: {prev[:120]}")
        print(f, "-", prev)

    out_path = C.OUT_DIR / f"narration_{video}.json"
    out_path.write_text(json.dumps({"video": video, "sentences": lines}, indent=2))
    return lines


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="auto", help="Test video ID (e.g. video12) or 'auto'")
    ap.add_argument("--limit", type=int, default=0, help="Max frames to narrate (0 = all)")
    args = ap.parse_args()
    asyncio.run(main(args.video, args.limit))
