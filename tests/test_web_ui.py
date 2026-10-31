"""web/server.py: serves real outputs only, no ground truth, frames from CholecSeg8k."""
import json

from fastapi.testclient import TestClient

import web.server as S

c = TestClient(S.app)


def test_overview_and_labels_have_no_ground_truth():
    o = c.get("/api/overview").json()
    assert set(o) >= {"manual", "results", "student", "tracker", "test_videos"}
    assert o["test_videos"] == ["video09", "video12", "video17", "video28", "video48", "video55"]
    rows = c.get("/api/labels").json()["rows"]
    for r in rows[:50]:
        assert not any(k.startswith("gt_") or k in ("truth", "correct", "final_correct", "trust") for k in r)


def test_missing_files_are_null_not_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "OUT", tmp_path)
    monkeypatch.setattr(S, "COORD", tmp_path)
    o = c.get("/api/overview").json()
    assert o["results"] is None and o["student"] is None and o["manual"] is None
    assert c.get("/api/labels").json()["rows"] == []


def test_frame_endpoint():
    assert c.get("/api/frame/not_a_frame.jpg").status_code == 404
    if S.frame_png("video12_15750_15750"):
        r = c.get("/api/frame/video12_15750_15750.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"


def test_narration_hides_mock(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setattr(S, "NARRATOR_OUT", tmp_path)
    pd.DataFrame([{"claim_id": "a", "clip_id": "c", "text": "x", "verdict": "speak", "source": "mock"}]).to_parquet(tmp_path / "claims.parquet")
    assert c.get("/api/narration").json() == {"source": "mock", "claims": []}
    pd.DataFrame([
        {"claim_id": "a", "clip_id": "c", "text": "x", "verdict": "speak", "source": "gemini", "L3": "pass"},
        {"claim_id": "b", "clip_id": "c", "text": "y", "verdict": "drop", "source": "gemini", "L3": "fail"},
        {"claim_id": "d", "clip_id": "c", "text": "z", "verdict": "hedge", "source": "gemini", "L3": "abstain"},
    ]).to_parquet(tmp_path / "claims.parquet")
    tags = [x["tag"] for x in c.get("/api/narration").json()["claims"]]
    assert tags == ["verified", "contradicted", "unsupported"]
