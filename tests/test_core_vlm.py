"""vlm.py: the headless Claude client, tested against a fake `claude` binary so no model is called."""
import json
import os
import stat
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from surgproof import vlm

SCHEMA = {"type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}

FAKE = textwrap.dedent("""\
    #!{python}
    import json, os, sys, time
    log = os.environ["FAKE_LOG"]
    n = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    with open(log, "a") as f:
        f.write(json.dumps({{"argv": sys.argv[1:], "cwd": os.getcwd(), "files": sorted(os.listdir("."))}}) + "\\n")
    replies = json.load(open(os.environ["FAKE_REPLIES"]))
    reply = replies[min(n, len(replies) - 1)]
    time.sleep(float(os.environ.get("FAKE_SLEEP", 0)))
    if reply.get("crash"):
        sys.stderr.write("boom")
        sys.exit(3)
    print(json.dumps({{"type": "result", "is_error": reply.get("is_error", False), "result": reply["result"], "total_cost_usd": 0.01}}))
""")


@pytest.fixture
def fake(tmp_path, monkeypatch):
    binary = tmp_path / "fake_claude"
    binary.write_text(FAKE.format(python=sys.executable))
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    log, replies = tmp_path / "log.jsonl", tmp_path / "replies.json"
    monkeypatch.setenv("SURGPROOF_CLAUDE_BIN", str(binary))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_REPLIES", str(replies))
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    monkeypatch.delenv("SURGPROOF_CACHE", raising=False)
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "cache.sqlite")

    def script(*items):
        replies.write_text(json.dumps(list(items)))

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []

    return script, calls


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "video12_secret_name.jpg"
    p.write_bytes(b"pretend image bytes")
    return p


def test_extract_json_handles_fences_and_prose():
    assert vlm.extract_json('{"a": 1}') == {"a": 1}
    assert vlm.extract_json('Here you go:\n```json\n{"a": [1, 2]}\n```\nDone.') == {"a": [1, 2]}
    assert vlm.extract_json('note {not json} then {"a": 2}') == {"a": 2}
    with pytest.raises(ValueError):
        vlm.extract_json("no json at all")


@pytest.mark.parametrize("value, schema", [
    ({"a": 1}, {"type": "object", "required": ["b"]}),
    ("x", {"type": "number"}),
    (True, {"type": "number"}),
    (1.5, {"type": "number", "maximum": 1}),
    ("z", {"enum": ["a"]}),
    ([1], {"type": "array", "minItems": 2}),
    ([1, 2, 3], {"type": "array", "maxItems": 2}),
    ({"a": "x"}, {"type": "object", "properties": {"a": {"type": ["array", "null"]}}}),
    ({"a": 1, "b": 2}, {"type": "object", "properties": {"a": {}}, "additionalProperties": False}),
])
def test_validate_rejects(value, schema):
    with pytest.raises(ValueError):
        vlm.validate(value, schema)


def test_validate_accepts_nullable_and_nested():
    schema = {"type": "object", "properties": {"p": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "number"}}}}}
    vlm.validate({"p": None}, schema)
    vlm.validate({"p": [[1, 2.5]]}, schema)


def test_command_has_required_flags_and_isolates_the_model():
    cmd = vlm.claude_command("hello", "claude-sonnet-5-5", "/tmp/call_1")
    assert cmd[1:3] == ["-p", "hello"]
    for flag, value in [("--model", "claude-sonnet-5-5"), ("--allowedTools", "Read(//tmp/call_1/**)"), ("--output-format", "json"),
                        ("--tools", "Read"), ("--setting-sources", "")]:
        assert cmd[cmd.index(flag) + 1] == value
    assert "--strict-mcp-config" in cmd and "--no-session-persistence" in cmd


def test_ask_returns_parsed_and_caches_with_raw(fake, image):
    script, calls = fake
    script({"result": 'Sure.\n{"ok": true}'})
    assert vlm.ask([image], "question", SCHEMA, "m") == {"ok": True}
    assert vlm.ask([image], "question", SCHEMA, "m") == {"ok": True}
    assert len(calls()) == 1
    record = vlm.get_raw(vlm.cache_key([image], "question", "m"))
    assert record["raw"] == 'Sure.\n{"ok": true}' and record["model"] == "m" and record["prompt"] == "question"
    assert record["cost_usd"] == pytest.approx(0.01)


def test_call_runs_in_empty_folder_with_neutral_image_name(fake, image):
    script, calls = fake
    script({"result": '{"ok": true}'})
    vlm.ask([image], "question", SCHEMA, "m")
    call = calls()[0]
    assert call["files"] == ["image_1.jpg"]
    prompt = call["argv"][1]
    assert "image_1.jpg" in prompt and "secret_name" not in " ".join(call["argv"])
    assert str(Path(call["cwd"]).resolve()) in prompt
    assert not Path(call["cwd"]).exists()  # removed after the call


def test_invalid_json_retries_twice_then_succeeds(fake, image):
    script, calls = fake
    script({"result": "nope"}, {"result": '{"wrong": 1}'}, {"result": '{"ok": false}'})
    assert vlm.ask([image], "q", SCHEMA, "m") == {"ok": False}
    assert len(calls()) == 3
    assert "rejected" in calls()[1]["argv"][1]


def test_invalid_json_gives_up_and_caches_nothing(fake, image):
    script, calls = fake
    script({"result": "nope"})
    with pytest.raises(vlm.VlmError):
        vlm.ask([image], "q", SCHEMA, "m")
    assert len(calls()) == 3
    assert vlm.get_raw(vlm.cache_key([image], "q", "m")) is None


def test_cli_failure_is_retried_then_raised(fake, image):
    script, calls = fake
    script({"crash": True, "result": ""}, {"result": "x", "is_error": True}, {"result": '{"ok": true}'})
    assert vlm.ask([image], "q", SCHEMA, "m") == {"ok": True}
    assert len(calls()) == 3
    script({"crash": True, "result": ""})
    with pytest.raises(vlm.VlmError):
        vlm.ask([image], "another", SCHEMA, "m")


def test_not_logged_in_stops_immediately(fake, image):
    script, calls = fake
    script({"result": "Not logged in · Please run /login", "is_error": True})
    with pytest.raises(vlm.VlmAuthError):
        vlm.ask([image], "q", SCHEMA, "m")
    assert len(calls()) == 1


def test_missing_binary_is_an_error_not_a_crash(fake, image, monkeypatch):
    monkeypatch.setenv("SURGPROOF_CLAUDE_BIN", "/nonexistent/claude")
    with pytest.raises(vlm.VlmError):
        vlm.ask([image], "q", SCHEMA, "m")


def test_cache_key_depends_on_bytes_prompt_model_not_path(tmp_path):
    a, b, c = tmp_path / "a.jpg", tmp_path / "b.jpg", tmp_path / "c.jpg"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    c.write_bytes(b"other")
    key = vlm.cache_key([a], "p", "m")
    assert key == vlm.cache_key([b], "p", "m")
    assert len({key, vlm.cache_key([c], "p", "m"), vlm.cache_key([a], "q", "m"), vlm.cache_key([a], "p", "n")}) == 4


def test_mock_is_tagged_and_cannot_share_a_cache_entry_with_real(monkeypatch, tmp_path, image):
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "c.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    real_key = vlm.cache_key([image], "p", "m")
    monkeypatch.setenv("SURGPROOF_VLM", "mock")
    assert vlm.model_tag("m") == "mock:m"
    assert vlm.cache_key([image], "p", "m") != real_key


def test_unknown_provider_is_refused(monkeypatch, tmp_path, image):
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "c.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "something_else")
    monkeypatch.setenv("SURGPROOF_VLM_BACKOFF_S", "0")
    with pytest.raises(vlm.VlmError):
        vlm.ask([image], "q", SCHEMA, "m")


def test_semaphore_limits_concurrent_calls(fake, tmp_path, monkeypatch):
    script, _ = fake
    script({"result": '{"ok": true}'})
    monkeypatch.setattr(vlm, "_sem", threading.BoundedSemaphore(2))
    active, peak, guard = [0], [0], threading.Lock()
    def provider(paths, prompt, model):
        with guard:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        with guard:
            active[0] -= 1
        return '{"ok": true}', 0.0
    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", provider)
    threads = [threading.Thread(target=vlm.ask, args=([], f"p{i}", SCHEMA, "m")) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert peak[0] == 2


def test_fresh_cache_survives_racing_threads_and_processes(tmp_path, monkeypatch):
    # A first run on a clean clone: many workers open a cache file that does not exist yet.
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "fresh" / "cache.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", lambda *a: ('{"ok": true}', 0.0))
    errors = []

    def go(i):
        try:
            vlm.ask([], f"p{i}", SCHEMA, "m")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(24)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors

    code = (
        "import sys; from pathlib import Path; from surgproof import vlm\n"
        "vlm._cache_path = Path(sys.argv[1]); vlm.PROVIDERS['claude_cli'] = lambda *a: ('{\"ok\": true}', 0.0)\n"
        "[vlm.ask([], f'q{i}-{sys.argv[2]}', {'type': 'object'}, 'm') for i in range(5)]\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    target = tmp_path / "fresh2" / "cache.sqlite"
    procs = [subprocess.Popen([sys.executable, "-c", code, str(target), str(i)], env=env, stderr=subprocess.PIPE, text=True) for i in range(6)]
    results = [(p.communicate()[1], p.returncode) for p in procs]
    assert all(rc == 0 for _, rc in results), results


def test_meter_counts_this_context_only(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "c.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    monkeypatch.setitem(vlm.PROVIDERS, "claude_cli", lambda *a: ('{"ok": true}', 0.25))
    import contextvars

    def job(prompt):
        meter = vlm.start_meter()
        vlm.ask([], prompt, SCHEMA, "m")
        vlm.ask([], prompt, SCHEMA, "m")
        return meter
    meter = contextvars.copy_context().run(job, "one")
    assert meter == {"calls": 1, "cache_hits": 1, "cost_usd": 0.25}


def test_gemini_provider_is_optional_and_maps_models(monkeypatch, tmp_path, image):
    import types
    seen = {}

    class Models:
        def generate_content(self, model, contents, config):
            seen.update(model=model, n=len(contents))
            return types.SimpleNamespace(text='{"ok": true}')

    genai = types.SimpleNamespace(Client=lambda: types.SimpleNamespace(models=Models()))
    gtypes = types.SimpleNamespace(
        Part=types.SimpleNamespace(from_bytes=lambda data, mime_type: "part"),
        GenerateContentConfig=lambda **k: k)
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", types.SimpleNamespace(types=gtypes, Client=genai.Client))
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "g.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(vlm.VlmAuthError):
        vlm.ask([image], "q", SCHEMA, vlm.SONNET)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    assert vlm.ask([image], "q", SCHEMA, vlm.SONNET) == {"ok": True}
    assert seen == {"model": "gemini-4-argon", "n": 2}
    assert vlm.model_tag(vlm.SONNET) == f"gemini:{vlm.SONNET}"


@pytest.mark.skipif(not os.environ.get("SURGPROOF_REAL"), reason="calls the real model; set SURGPROOF_REAL=1")
def test_real_call(tmp_path, monkeypatch):
    monkeypatch.setattr(vlm, "_cache_path", tmp_path / "c.sqlite")
    monkeypatch.setenv("SURGPROOF_VLM", "claude_cli")
    frame = Path(__file__).resolve().parent.parent / "docs/media/fin55.jpg"
    reply = vlm.ask([frame], 'Is this a photo? Reply with only JSON {"ok": true or false}.', SCHEMA, vlm.HAIKU)
    assert isinstance(reply["ok"], bool)
