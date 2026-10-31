"""Unit tests for narrator.kb module."""
import pytest
import yaml

from narrator import cfg as C
from narrator.kb import KB, load, normalize_text, verify_quote


def test_verify_quote_normalisation():
    # Exact match
    text = "The hepatocystic triangle is cleared of fat and fibrous tissue."
    assert verify_quote("hepatocystic triangle is cleared", text)

    # Case difference
    assert verify_quote("HEPATOCYSTIC TRIANGLE IS CLEARED", text)

    # Smart quotes and punctuation
    text_with_quotes = 'He stated, "the hepatocystic triangle is cleared."'
    assert verify_quote('“the hepatocystic triangle is cleared”', text_with_quotes)
    assert verify_quote("‘the hepatocystic triangle is cleared’", text_with_quotes)

    # Whitespace and newlines
    text_newlines = "The hepatocystic\n\n  triangle   is   cleared."
    assert verify_quote("hepatocystic triangle is cleared", text_newlines)

    # Negative case
    assert not verify_quote("gallbladder was removed immediately", text)
    assert not verify_quote("", text)
    assert not verify_quote("hepatocystic", "")


def test_every_rule_has_required_keys():
    kb = load()
    assert len(kb.rules) > 0, "KB should contain rules"
    required_keys = {"id", "statement", "source_id", "quote", "quote_verified", "evidence_grade", "seed"}

    for rid, rule in kb.rules.items():
        missing = required_keys - set(rule.keys())
        assert not missing, f"Rule {rid} in {rule.get('file')} missing keys: {missing}"
        assert isinstance(rule["id"], str) and rule["id"]
        assert isinstance(rule["statement"], str) and rule["statement"]
        assert isinstance(rule["source_id"], str) and rule["source_id"] in kb.sources, (
            f"Rule {rid} source_id '{rule['source_id']}' not in sources.yaml"
        )
        assert isinstance(rule["quote"], str)
        assert len(rule["quote"].split()) <= 15, f"Rule {rid} quote has >=15 words: '{rule['quote']}'"
        assert isinstance(rule["quote_verified"], bool)
        assert isinstance(rule["evidence_grade"], str) and rule["evidence_grade"]
        assert isinstance(rule["seed"], bool)


def test_every_enum_in_phases_grading_conditions():
    kb = load()
    vocab = kb.yaml("vocab")

    valid_phases = vocab["phases"]
    valid_instruments = list(vocab["instruments"].keys()) if isinstance(vocab["instruments"], dict) else vocab["instruments"]
    valid_findings = list(vocab["findings"].keys()) if isinstance(vocab["findings"], dict) else vocab["findings"]
    valid_conditions = vocab["conditions"]

    # 1. Phases
    phases_data = kb.yaml("phases")
    assert "phases" in phases_data
    for p in phases_data["phases"]:
        assert p["name"] in valid_phases, f"Phase name {p['name']} not in vocab"
        assert isinstance(p["tools"], list)
        for t in p["tools"]:
            assert t in valid_instruments, f"Instrument {t} in phase {p['name']} not in vocab"
        assert len(p["typical_duration_s"]) == 2
        assert p["typical_duration_s"][0] < p["typical_duration_s"][1]

    # 2. Grading
    grading_data = kb.yaml("grading")
    assert "grades" in grading_data
    grades = grading_data["grades"]
    for g_id, g_info in grades.items():
        if "required_findings" in g_info:
            for f in g_info["required_findings"]:
                assert f in valid_findings, f"Finding {f} in grade {g_id} not in vocab"
    assert "tokyo" in grading_data
    assert grading_data["tokyo"]["inferable_from_video"] is False

    # 3. Conditions
    cond_data = kb.yaml("conditions")
    assert "conditions" in cond_data
    for c_id, c_info in cond_data["conditions"].items():
        assert c_id in valid_conditions, f"Condition {c_id} not in vocab"
        assert c_info["visibility"] in {"visible", "needs_non_video_data", "not_inferable"}
        for f in c_info.get("supporting_findings", []):
            assert f in valid_findings, f"Finding {f} in condition {c_id} not in vocab"
        for f in c_info.get("against_findings", []):
            assert f in valid_findings, f"Finding {f} in condition {c_id} not in vocab"


def test_transitions_include_p5_p6_swap():
    kb = load()
    phases_data = kb.yaml("phases")
    assert "transitions" in phases_data
    transitions = [tuple(t) for t in phases_data["transitions"]]

    # Both directions between packaging (P5) and cleaning_coagulation (P6) must exist
    assert ("gallbladder_packaging", "cleaning_coagulation") in transitions
    assert ("cleaning_coagulation", "gallbladder_packaging") in transitions
