"""Verification tests for nir-coach: foolproof fresh-clone demo bundle, README, licenses, and zero absolute /Users paths in served code/data."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

import web.server as S

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "web"
WEB_DATA = WEB_DIR / "data"
COORD_DIR = REPO_ROOT / "coord" / "artifacts"


def test_required_repo_files_and_contributing_free() -> None:
    assert (REPO_ROOT / "README.md").is_file()
    assert (REPO_ROOT / "LICENSE").is_file()
    assert (REPO_ROOT / "DATA_LICENSE.md").is_file()
    assert (REPO_ROOT / ".env.example").is_file()
    assert (REPO_ROOT / "Makefile").is_file()
    assert not (REPO_ROOT / "CONTRIBUTING.md").exists()
    assert not (REPO_ROOT / "CONTRIBUTING").exists()

    license_txt = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "MIT License" in license_txt

    data_license_txt = (REPO_ROOT / "DATA_LICENSE.md").read_text(encoding="utf-8")
    assert "CC BY-NC-SA 4.0" in data_license_txt
    assert "CholecSeg8k" in data_license_txt

    makefile_txt = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "setup:" in makefile_txt
    assert "demo:" in makefile_txt
    assert "test:" in makefile_txt


def test_readme_sections_and_placeholders() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "<!-- VIDEO -->" in readme
    assert "Try it in 60 seconds" in readme
    assert "make setup" in readme
    assert "make demo" in readme
    assert "```mermaid" in readme
    assert "out/results.json" in readme
    assert "out/dense_results.json" in readme
    assert "Honest Limits" in readme or "Known limits" in readme


def test_demo_bundle_exists_and_within_size_limits() -> None:
    assert (WEB_DATA / "video12.json").is_file()
    assert (WEB_DATA / "dense_results.json").is_file()
    assert (WEB_DATA / "results.json").is_file()

    v12 = json.loads((WEB_DATA / "video12.json").read_text(encoding="utf-8"))
    assert v12.get("video") == "video12"
    assert len(v12.get("frames", [])) == 80

    # Verify all 80 JPEG frames for video12 exist
    jpg_frames = list((WEB_DATA / "frames").glob("video12_*.jpg"))
    assert len(jpg_frames) == 80

    # Verify all 3 dense clips have 80 JPEGs + AI masks + annotations.jsonl
    for clip in ("video12_15750", "video28_00000", "video55_00508"):
        cdir = WEB_DATA / "dense" / clip
        assert cdir.is_dir(), f"Missing dense clip dir {cdir}"
        assert (cdir / "annotations.jsonl").is_file()
        clip_jpgs = list(cdir.glob("frame_*_endo.jpg"))
        assert len(clip_jpgs) == 80, f"Expected 80 JPEGs in {cdir}, got {len(clip_jpgs)}"
        clip_masks = list(cdir.glob("frame_*_*.png"))
        assert len(clip_masks) > 50, f"Expected AI masks in {cdir}, got {len(clip_masks)}"

    # Check total size < 80 MB and no single file > 50 MB
    max_single_bytes = 50 * 1024 * 1024
    max_total_bytes = 80 * 1024 * 1024
    total_bytes = 0
    for root_dir in (WEB_DATA, COORD_DIR):
        for f in root_dir.rglob("*"):
            if f.is_file():
                sz = f.stat().st_size
                assert sz < max_single_bytes, f"File {f} exceeds 50 MB ({sz} bytes)"
                total_bytes += sz
    assert total_bytes < max_total_bytes, f"Total demo bundle {total_bytes / (1024*1024):.1f} MB exceeds 80 MB"


def test_no_absolute_users_paths_or_secrets_in_served_code_or_data() -> None:
    text_exts = {".py", ".js", ".html", ".css", ".json", ".jsonl", ".csv", ".md", ".tsx", ".ts"}
    for scan_dir in (WEB_DIR, COORD_DIR):
        for f in scan_dir.rglob("*"):
            if f.is_file() and f.suffix in text_exts:
                txt = f.read_text(encoding="utf-8", errors="ignore")
                assert "/Users/" not in txt, f"Found absolute /Users/ path in served file {f}"
                assert not re.search(r"AIza[0-9A-Za-z_-]{35}", txt), f"Found API key pattern in {f}"


def test_served_demo_endpoints_and_watch_clips() -> None:
    client = TestClient(S.app)
    for route in (
        "/",
        "/watch.html",
        "/watch.html?clip=video12_15750",
        "/watch.html?clip=video55_00508",
        "/watch.html?clip=video28_00000",
        "/data/video12.json",
        "/data/dense_results.json",
        "/data/dense/video12_15750/annotations.jsonl",
        "/data/dense/video12_15750/frame_15750_endo.jpg",
        "/data/frames/video12_15750_15750.jpg",
        "/data/frames/video12_15750_15750.png",
        "/api/overview",
        "/api/gate?scope=video12",
    ):
        resp = client.get(route)
        assert resp.status_code == 200, f"GET {route} failed with {resp.status_code}"
