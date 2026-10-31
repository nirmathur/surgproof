"""Tests for narrate.py (grounded surgical narration strictly from verified labels)."""
import asyncio
import json

import pandas as pd
from PIL import Image

import config as C
import narrate


def test_narrate_uses_only_verified_labels_and_writes_json(tmp_path, monkeypatch):
    out_dir = tmp_path / "out"
    data_dir = tmp_path / "data"
    out_dir.mkdir()
    data_dir.mkdir()

    monkeypatch.setattr(C, "OUT_DIR", out_dir)
    monkeypatch.setattr(C, "DATA_DIR", data_dir)

    img_file = tmp_path / "video12_00080_08.png"
    Image.new("RGB", (160, 120), color=(90, 20, 20)).save(img_file)

    pd.DataFrame([
        {"frame_id": "video12_00080_08", "video": "video12", "split": "test", "path": str(img_file)},
        {"frame_id": "video12_00080_16", "video": "video12", "split": "test", "path": str(tmp_path / "missing.png")},
    ]).to_csv(data_dir / "index.csv", index=False)

    (out_dir / "manual.json").write_text(json.dumps({"procedure": "laparoscopic cholecystectomy", "phases": ["Dissection"]}))

    pd.DataFrame([
        {"frame_id": "video12_00080_08", "video": "video12", "split": "test", "key": "grasper", "final_present": 1, "status": "accept"},
        {"frame_id": "video12_00080_08", "video": "video12", "split": "test", "key": "cystic_duct", "final_present": 1, "status": "escalate"},
        {"frame_id": "video12_00080_16", "video": "video12", "split": "test", "key": "liver", "final_present": 1, "status": "accept"},
        {"frame_id": "video12_00080_16", "video": "video12", "split": "test", "key": "blood", "final_present": 1, "status": "reanalyse"},
    ]).to_csv(out_dir / "decisions.csv", index=False)

    captured_prompts = []

    async def fake_call(prompt, *, model=None, images=(), schema=None, tools=None, video_url=None, temperature=0.0, tag="", sample=0):
        captured_prompts.append((prompt, len(images)))
        return {"narration": "Grasper holds tissue; cystic duct is not yet confirmed.", "unsure_about": ["cystic_duct"]}

    monkeypatch.setattr(narrate.gem, "call", fake_call)
    monkeypatch.setattr(narrate.gem, "trace", lambda *a, **kw: None)

    lines = asyncio.run(narrate.main("auto", limit=2))
    assert len(lines) == 2
    assert (out_dir / "narration_video12.json").is_file()

    # Frame 1 has an existing image file -> 1 image part passed; verified={'grasper': True}, unsure=['cystic_duct']
    p0, n_img0 = captured_prompts[0]
    assert n_img0 == 1
    assert '"grasper": true' in p0
    assert '"cystic_duct"' in p0

    # Frame 2 has a missing image file -> gracefully falls back to 0 image parts without crashing
    _, n_img1 = captured_prompts[1]
    assert n_img1 == 0
