"""Unit and contract tests for med_school.py and the generated manual.json."""
import asyncio
import json
from pathlib import Path

import pytest

import config as C
import med_school


def _sample_cue(key: str, name: str) -> dict:
    return {
        "key": key,
        "name": name,
        "appearance": f"Distinct visual colour, texture, and shape for {name} on endoscopic video.",
        "where_found": f"Typical anatomical region or quadrant for {name}.",
        "when_visible": "Calot's triangle dissection and gallbladder detachment phases.",
        "confused_with": ["adjacent fat", "other instrument"],
        "how_to_tell_apart": f"Distinguish {name} by its rigid shaft/jaw or ductal continuity.",
    }


def _sample_manual_dict(keys=None) -> dict:
    keys = keys or list(C.LABELS)
    return {
        "procedure": "laparoscopic cholecystectomy",
        "sources": med_school.DEFAULT_LAP_CHOLE_SOURCES,
        "phases": [
            "Port placement and diagnostic laparoscopy",
            "Exposure and fundus retraction",
            "Calot's triangle dissection (hepatocystic triangle)",
            "Critical View of Safety confirmation",
            "Clipping and division of cystic duct and artery",
            "Gallbladder dissection from liver bed",
            "Hemostasis, irrigation, and specimen extraction",
        ],
        "structures": [_sample_cue(k, C.LABELS.get(k, (0, k))[1]) for k in keys],
        "critical_events": [
            "Critical View of Safety achieved prior to clipping",
            "Cystic duct clipped and divided",
            "Cystic artery clipped and divided",
        ],
        "general_rules": [
            "Verify instrument tip shape before distinguishing grasper from L-hook.",
            "Only call cystic_duct when a distinct tubular structure connects to the gallbladder infundibulum.",
        ],
    }


def test_pydantic_schemas_and_validate_manual():
    d = _sample_manual_dict()
    m = med_school.Manual.model_validate(d)
    assert m.procedure == "laparoscopic cholecystectomy"
    assert len(m.structures) == len(C.LABELS)
    assert len(m.sources) >= 1
    assert m.sources[0].url == med_school.DEFAULT_LAP_CHOLE_VIDEO
    assert m.sources[0].is_real_intraop_footage is False
    assert set(C.LABELS).issubset(m.sources[0].structure_timestamps.keys())

    validated = med_school.validate_manual(d, list(C.LABELS))
    assert [s["key"] for s in validated["structures"]] == list(C.LABELS)
    assert len(validated["sources"]) >= 1

    # Missing required key should raise ValueError
    broken = _sample_manual_dict(keys=["grasper", "liver"])
    with pytest.raises(ValueError, match="missing required structure keys"):
        med_school.validate_manual(broken, list(C.LABELS))

    # Empty appearance should raise ValueError
    bad_cue = _sample_manual_dict()
    bad_cue["structures"][0]["appearance"] = "   "
    with pytest.raises(ValueError, match="empty visual criteria"):
        med_school.validate_manual(bad_cue, list(C.LABELS))

    vl = med_school.VideoLesson.model_validate(
        {"rules": ["Identify Rouviere's sulcus first"], "moments": ["02:15 - Critical view shown"]}
    )
    assert len(vl.rules) == 1
    assert len(vl.moments) == 1


def test_build_manual_mocked(monkeypatch):
    calls = []

    async def fake_call(prompt, *, model=C.FLASH, images=(), schema=None, tools=None, video_url=None,
                        temperature=0.0, tag="", sample=0):
        calls.append({"tag": tag, "model": model, "video_url": video_url, "schema": schema, "prompt": prompt})
        if tag == "research":
            return "Dense research notes on laparoscopic cholecystectomy structures and CVS."
        if tag == "watch":
            return {"rules": ["Rule 1: clear hepatocystic triangle"], "moments": ["01:20 - fundus retracted"]}
        if tag == "manual":
            d = _sample_manual_dict()
            d.pop("sources", None)
            return d
        raise AssertionError(f"Unexpected tag: {tag}")

    monkeypatch.setattr(med_school.gem, "call", fake_call)
    monkeypatch.setattr(med_school.gem, "trace", lambda *a, **kw: None)

    keys = list(C.LABELS)
    manual, notes = asyncio.run(
        med_school.build_manual(
            "laparoscopic cholecystectomy",
            keys=keys,
            video_urls=["https://www.youtube.com/watch?v=PNydeaI2WIU"],
        )
    )
    assert "Dense research notes" in notes
    assert [s["key"] for s in manual["structures"]] == keys
    assert len(manual["sources"]) >= 2
    assert manual["sources"][0]["url"] == "https://www.youtube.com/watch?v=PNydeaI2WIU"
    assert manual["sources"][0]["is_real_intraop_footage"] is False
    assert set(keys).issubset(manual["sources"][0]["structure_timestamps"].keys())
    tags = [c["tag"] for c in calls]
    assert "research" in tags
    assert "watch" in tags
    assert "manual" in tags


def test_generated_manual_artifact_if_present():
    candidates = [
        C.OUT_DIR / "manual.json",
        Path(__file__).resolve().parent.parent / "coord" / "artifacts" / "manual.json",
    ]
    present = [p for p in candidates if p.exists()]
    if not present:
        pytest.skip("manual.json not generated yet")
    for path in present:
        data = json.loads(path.read_text())
        m = med_school.Manual.model_validate(data)
        assert len(m.sources) >= 2, f"Missing sources[] in {path}"
        video_sources = [s for s in m.sources if "PNydeaI2WIU" in s.url]
        assert len(video_sources) == 1, f"Missing PNydeaI2WIU video source in {path}"
        vs = video_sources[0]
        assert vs.title == "Ellie Scrubs In: Laparoscopic Cholecystectomy"
        assert vs.is_real_intraop_footage is False
        assert "static" in vs.real_intraop_footage_note.lower()
        assert set(C.LABELS).issubset(vs.structure_timestamps.keys())
        by_key = {s.key: s for s in m.structures}
        assert set(C.LABELS).issubset(by_key.keys()), f"Missing keys in {path}: {set(C.LABELS) - by_key.keys()}"
        for k in C.LABELS:
            cue = by_key[k]
            assert len(cue.appearance.strip()) >= 20, f"Weak appearance for {k}"
            assert len(cue.where_found.strip()) >= 15, f"Weak where_found for {k}"
            assert len(cue.when_visible.strip()) >= 10, f"Weak when_visible for {k}"
            assert len(cue.confused_with) >= 1, f"Empty confused_with for {k}"
            assert len(cue.how_to_tell_apart.strip()) >= 20, f"Weak how_to_tell_apart for {k}"
        assert len(m.phases) >= 4
        assert len(m.critical_events) >= 2
        assert len(m.general_rules) >= 2
