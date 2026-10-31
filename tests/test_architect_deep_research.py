"""Unit tests for tools/deep_research.py (no live API calls)."""
from types import SimpleNamespace
from pathlib import Path

import pytest

from tools import deep_research as dr


class DummyInteractions:
    def __init__(self, fail_agents=(), poll_statuses=("in_progress", "completed")):
        self.fail_agents = set(fail_agents)
        self.poll_statuses = list(poll_statuses)
        self.created_with = []
        self.poll_idx = 0

    def create(self, *, agent, input, background):
        self.created_with.append((agent, input, background))
        if agent in self.fail_agents:
            raise RuntimeError(f"Agent {agent} unavailable")
        return SimpleNamespace(
            id="int_123",
            status=self.poll_statuses[0],
            outputs=[],
        )

    def get(self, interaction_id):
        assert interaction_id == "int_123"
        self.poll_idx = min(self.poll_idx + 1, len(self.poll_statuses) - 1)
        st = self.poll_statuses[self.poll_idx]
        outputs = []
        if st == "completed":
            ann = SimpleNamespace(url="https://example.org/paper", title="SurgPaper 2025")
            outputs = [
                SimpleNamespace(
                    type="text",
                    text="Report body with [Inline Link](https://example.org/inline).",
                    annotations=[ann],
                )
            ]
        return SimpleNamespace(id=interaction_id, status=st, outputs=outputs)


class DummyClient:
    def __init__(self, **kw):
        self.interactions = DummyInteractions(**kw)


def test_parse_args_variations():
    a1 = dr.parse_args(["architect_method", "Can an LLM replace the human verifier?"])
    assert a1.resolved_slug == "architect_method"
    assert "replace the human verifier" in a1.resolved_prompt

    a2 = dr.parse_args(["--slug", "my_slug", "--prompt", "Some query"])
    assert a2.resolved_slug == "my_slug"
    assert a2.resolved_prompt == "Some query"

    a3 = dr.parse_args(["Some query only"])
    assert a3.resolved_slug == "some-query-only"
    assert a3.resolved_prompt == "Some query only"


def test_fallback_and_report_saving(tmp_path: Path):
    client = DummyClient(fail_agents={"deep-research-max-preview-04-2026"})
    out_file = tmp_path / "test_report.md"
    saved = dr.run_deep_research(
        "Investigate surgical VLM verification",
        "test_report",
        out_path=out_file,
        poll_interval=0.01,
        client=client,
    )
    assert saved == out_file
    assert saved.is_file()
    content = saved.read_text(encoding="utf-8")
    # Should have fallen back to deep-research-preview-04-2026
    assert "deep-research-preview-04-2026" in content
    assert "Report body with" in content
    assert "https://example.org/paper" in content
    assert "https://example.org/inline" in content
    assert [c[0] for c in client.interactions.created_with] == [
        "deep-research-max-preview-04-2026",
        "deep-research-preview-04-2026",
    ]


def test_extract_report_from_steps_schema():
    inter = SimpleNamespace(
        outputs=None,
        steps=[
            {"type": "user_input", "content": [{"type": "text", "text": "User prompt"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "Part 1 [Ref](https://example.org/ref1)"}]},
            {"type": "model_output", "content": [{"type": "image"}, {"type": "text", "text": "Part 2"}]},
        ],
    )
    txt, sources = dr.extract_report_and_sources(inter)
    assert "Part 1" in txt and "Part 2" in txt
    assert "User prompt" not in txt
    assert ("Ref", "https://example.org/ref1") in sources
