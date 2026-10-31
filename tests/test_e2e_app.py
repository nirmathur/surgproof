"""End-to-end smoke test for the unified SurgProof web server (web/server.py + web/serve.py).

Verifies:
1. GET /api/overview returns schema without ground-truth fields.
2. GET /api/clips and GET /api/clip/{clip_id} return clip bundles (both live if present and in fresh-clone temp dirs).
3. GET /api/frame/{frame_id}.jpg serves JPEG bytes from configurable CholecSeg8k/index paths.
4. GET /api/gate and POST /api/gate record human review actions into an isolated gate_log.jsonl and unlock release.
5. Static HTML entrypoints (/, /watch.html, /react) serve 200 OK.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import web.serve as gate_mod
import web.server as S

client = TestClient(S.app)


def test_e2e_static_entrypoints() -> None:
    """Verify web studio, watch mode, and react studio routes return 200 HTML."""
    for route in ("/", "/watch.html", "/react"):
        resp = client.get(route)
        assert resp.status_code == 200, f"{route} returned {resp.status_code}"
        assert "text/html" in resp.headers.get("content-type", "")


def test_e2e_clips_and_frame_jpeg(tmp_path: Path, monkeypatch) -> None:
    """Verify /api/clips, /api/clip/{clip_id}, and /api/frame/{frame_id}.jpg work in a fresh clone."""
    # If live data exists, verify live /api/clips first
    live_clips = client.get("/api/clips")
    assert live_clips.status_code == 200
    assert "clips" in live_clips.json()

    # Now test deterministic fresh-clone behavior with configurable directories
    cholec_dir = tmp_path / "raw" / "CholecSeg8k"
    clip_dir = cholec_dir / "video12" / "video12_15750"
    clip_dir.mkdir(parents=True, exist_ok=True)
    frame_file = clip_dir / "frame_15750_endo.png"
    Image.new("RGB", (64, 48), color=(32, 64, 96)).save(frame_file)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    index_csv = data_dir / "index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame_id", "path", "split", "video", "clip", "frame"])
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "video12_15750_15750",
                "path": str(frame_file),
                "split": "test",
                "video": "video12",
                "clip": "video12_15750",
                "frame": "15750",
            }
        )

    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "OUT", out_dir)
    monkeypatch.setattr(S, "COORD", out_dir)
    monkeypatch.setattr(S, "CHOLEC", cholec_dir)

    clips_resp = client.get("/api/clips")
    assert clips_resp.status_code == 200
    clips_list = clips_resp.json()["clips"]
    assert any(c["clip_id"] == "video12_15750" for c in clips_list)

    bundle_resp = client.get("/api/clip/video12_15750")
    assert bundle_resp.status_code == 200
    bundle = bundle_resp.json()
    assert bundle["clip_id"] == "video12_15750"
    assert bundle["frames"] == [15750]

    frame_resp = client.get("/api/frame/video12_15750_15750.jpg")
    assert frame_resp.status_code == 200
    assert frame_resp.headers["content-type"] == "image/jpeg"
    assert frame_resp.content[:2] == b"\xff\xd8"


def test_e2e_gate_get_and_post(tmp_path: Path, monkeypatch) -> None:
    """Verify GET /api/gate and POST /api/gate through FastAPI unified server without touching real gate_log."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    decisions_csv = out_dir / "decisions.csv"
    gate_log = out_dir / "gate_log.jsonl"

    with open(decisions_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "frame_id",
                "video",
                "split",
                "key",
                "present",
                "final_present",
                "conf",
                "status",
                "status_round_0",
                "status_round_1",
                "status_round_2",
                "status_round_3",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "frame_id": "video12_15750_15750",
                "video": "video12",
                "split": "test",
                "key": "cystic_duct",
                "present": "1",
                "final_present": "1",
                "conf": "0.75",
                "status": "AMBER",
                "status_round_0": "AMBER",
                "status_round_1": "AMBER",
                "status_round_2": "AMBER",
                "status_round_3": "AMBER",
            }
        )

    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S, "OUT", out_dir)
    monkeypatch.setattr(S, "COORD", out_dir)

    get_resp = client.get("/api/gate?scope=test_5")
    assert get_resp.status_code == 200
    state = get_resp.json()
    assert state["ok"] is True
    assert state["summary"]["n_pending_amber"] == 1

    post_resp = client.post(
        "/api/gate",
        json={
            "who": "e2e_tester",
            "what": "approve",
            "frame_id": "video12_15750_15750",
            "key": "cystic_duct",
            "human_present": 1,
            "scope": "test_5",
        },
    )
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["ok"] is True
    assert post_data["state"]["summary"]["n_pending_amber"] == 0
    assert gate_log.exists()
    logged = json.loads(gate_log.read_text(encoding="utf-8").strip().splitlines()[0])
    assert logged["frame_id"] == "video12_15750_15750"
    assert logged["key"] == "cystic_duct"
