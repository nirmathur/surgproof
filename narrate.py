"""Step 7 (stretch): 'Netflix for surgery'. Narrate a test video using ONLY verified labels.
Pre-generated, so the demo plays back with no latency.

  python narrate.py --video video12      # writes out/narration_video12.json
"""
import argparse
import asyncio
import json

import pandas as pd
from pydantic import BaseModel

import config as C
import gem


class Line(BaseModel):
    narration: str        # 1-2 sentences, expert commentator voice
    unsure_about: list[str]


async def main(video):
    dec = pd.read_csv(C.OUT_DIR / "decisions.csv")
    idx = pd.read_csv(C.DATA_DIR / "index.csv").set_index("frame_id")
    manual = json.loads((C.OUT_DIR / "manual.json").read_text())
    frames = sorted(dec[dec.video == video].frame_id.unique(),
                    key=lambda f: (f.rsplit("_", 1)[0], int(f.rsplit("_", 1)[1])))
    lines, prev = [], ""
    for f in frames:
        rows = dec[dec.frame_id == f]
        verified = {r.key: bool(r.final_present) for r in rows.itertuples() if r.status == "accept"}
        unsure = [r.key for r in rows.itertuples() if r.status != "accept"]
        out = await gem.call(
            "You are narrating a laparoscopic cholecystectomy for trainees, like an expert commentator. "
            "Describe what is happening in 1-2 sentences. You may ONLY state facts supported by the "
            f"VERIFIED labels: {json.dumps(verified)}. For anything in UNSURE {json.dumps(unsure)}, say "
            "you are not yet confident rather than guessing. Use the manual for context on phases and "
            f"safety (e.g. critical view of safety). Previous line: {prev!r}\nManual: {json.dumps(manual)}",
            images=[gem.img_part(idx.loc[f, "path"])], schema=Line, temperature=0.3, tag="narrate")
        prev = out["narration"]
        lines.append({"frame_id": f, **out})
        print(f, "-", prev)
    (C.OUT_DIR / f"narration_{video}.json").write_text(json.dumps(lines, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    asyncio.run(main(ap.parse_args().video))
