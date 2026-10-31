"""Tests for promo/DESIGN.md specification and director constraints."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent
DESIGN_MD = ROOT / "promo" / "DESIGN.md"


def test_design_md_exists_and_has_all_seven_shots():
    assert DESIGN_MD.is_file(), "promo/DESIGN.md must exist"
    text = DESIGN_MD.read_text(encoding="utf-8")
    for shot_num in range(1, 8):
        assert f"### Shot {shot_num} " in text, f"Missing Shot {shot_num} section in promo/DESIGN.md"


def test_design_md_captions_respect_six_word_limit():
    text = DESIGN_MD.read_text(encoding="utf-8")
    required_captions = [
        "Surgical video. No labels.",
        "Writes its own visual manual.",
        "Labels a real frame.",
        "Checks itself. Green, amber, red.",
        "Human clears only amber cases.",
        "Result: pending",
        "SurgProof. No human labels. Only proof.",
    ]
    for cap in required_captions:
        assert cap in text, f"Expected caption {cap!r} in promo/DESIGN.md"
        words = [w for w in re.split(r"\s+", cap.strip()) if w]
        assert len(words) <= 6, f"Caption {cap!r} exceeds 6 words ({len(words)})"


def test_design_md_enforces_restraint_and_real_data():
    text = DESIGN_MD.read_text(encoding="utf-8")
    assert "/System/Library/Fonts/SFNS.ttf" in text
    assert "#0A0B0E" in text
    assert "#10B981" in text  # Green (accept)
    assert "#F59E0B" in text  # Amber (unsure)
    assert "#EF4444" in text  # Red (reject)
    assert "data/index.csv" in text
    assert "out/manual.json" in text
    assert "out/results.json" in text
    assert "45–60 seconds" in text or "45-60 seconds" in text


def test_winning_demos_and_veo_prompts_exist():
    winning = ROOT / "promo" / "WINNING_DEMOS.md"
    assert winning.is_file(), "promo/WINNING_DEMOS.md must exist"
    w_text = winning.read_text(encoding="utf-8")
    assert "https://" in w_text
    assert "Pattern 1" in w_text and "Pattern 6" in w_text

    prompts_dir = ROOT / "promo" / "prompts"
    s1 = prompts_dir / "shot_01_establishing.txt"
    s2 = prompts_dir / "shot_02_manual.txt"
    neg = prompts_dir / "negative_prompt.txt"
    for p in (s1, s2, neg):
        assert p.is_file(), f"Missing {p}"
    for p in (s1, s2):
        words = p.read_text(encoding="utf-8").split()
        assert 100 <= len(words) <= 220, f"Prompt {p.name} should be ~100-200 words, got {len(words)}"
        assert "Static, locked-off" in p.read_text(encoding="utf-8")
    neg_text = neg.read_text(encoding="utf-8")
    for forbidden in ("text", "numbers", "charts", "HUD", "glow", "lasers", "sci-fi", "heart"):
        assert forbidden in neg_text


def test_rendered_promo_cut_passes_director_checks():
    promo_mp4 = ROOT / "out" / "promo" / "surgproof_promo.mp4"
    manifest_json = ROOT / "out" / "promo" / "manifest.json"
    if not promo_mp4.is_file() or not manifest_json.is_file():
        return
    import json
    m = json.loads(manifest_json.read_text(encoding="utf-8"))
    dur = float(m.get("final_video", {}).get("duration_sec", 0.0))
    assert 45.0 <= dur <= 60.0, f"Expected promo duration in 45..60s, got {dur}"
    assert len(m.get("shots", {})) == 7, "Expected 7 shots in manifest.json"
    for sid, sinfo in m["shots"].items():
        wc = int(sinfo.get("word_count", 0))
        assert wc <= 6, f"Shot {sid} caption exceeds 6 words ({wc})"


def test_promo_code_has_zero_fabricated_boxes_or_verdicts():
    patterns = [
        (r'["\']box["\']\s*:\s*\[\s*\d', "literal bounding box"),
        (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
        (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
        (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
        (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
    ]
    promo_dirs = [ROOT / "promo"]
    lane_promo = Path("/Users/nirmathur/surgatlas-lanes/promo/promo")
    if lane_promo.is_dir():
        promo_dirs.append(lane_promo)
    for pdir in promo_dirs:
        for py_file in sorted(pdir.glob("*.py")):
            for i, line in enumerate(py_file.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                s = line.strip()
                if s.startswith("#") or "integrity: allow" in s:
                    continue
                for pat, what in patterns:
                    assert not re.search(pat, line), f"{py_file}:{i}: {what}: {s[:110]}"



