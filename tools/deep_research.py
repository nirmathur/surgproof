#!/usr/bin/env python3
"""CLI and helper to run Gemini Deep Research via the google-genai Interactions API.

Usage examples (all supported):
  python tools/deep_research.py <slug> "<query>"
  python tools/deep_research.py --slug <slug> --prompt "<query>"
  python tools/deep_research.py --slug <slug> --query "<query>"
  python tools/deep_research.py "<query>" --slug <slug>
  python tools/deep_research.py "<query>"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Sequence

warnings.filterwarnings("ignore", message=".*Interactions usage is experimental.*")

ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = ROOT / "docs" / "research"

DEFAULT_AGENTS: tuple[str, ...] = (
    "deep-research-max-preview-04-2026",
    "deep-research-preview-04-2026",
)


def _ensure_api_key() -> None:
    """Load GEMINI_API_KEY from .env if not already present in environment (never logs key)."""
    if os.environ.get("GEMINI_API_KEY"):
        return
    candidates = [
        ROOT / ".env",
        Path.home() / "surgatlas" / ".env",
    ]
    for env_path in candidates:
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                if k == "GEMINI_API_KEY" and v:
                    os.environ["GEMINI_API_KEY"] = v
                    return


def slugify(text: str, max_len: int = 48) -> str:
    """Convert arbitrary text into a clean filesystem slug."""
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    s = s[:max_len].strip("-")
    return s or "research"


def extract_report_and_sources(interaction: object) -> tuple[str, list[tuple[str, str]]]:
    """Extract markdown report text and (title, url) source tuples from an Interaction."""
    outputs = getattr(interaction, "outputs", None) or []
    text_blocks: list[str] = []
    sources: list[tuple[str, str]] = []
    seen_urls: set[str] = set()

    def _add_source(url: Optional[str], title: Optional[str] = None) -> None:
        if not url or url in seen_urls:
            return
        seen_urls.add(url)
        sources.append((title or url, url))

    for item in outputs:
        item_type = getattr(item, "type", None)
        if item_type == "text":
            txt = getattr(item, "text", None)
            if txt:
                text_blocks.append(txt)
            for ann in getattr(item, "annotations", None) or []:
                _add_source(getattr(ann, "url", None), getattr(ann, "title", None))
        elif item_type == "url_context_result":
            for res in getattr(item, "result", None) or []:
                _add_source(getattr(res, "url", None), None)

    # Also support google-genai Interactions SDK `steps` (type == "model_output") and `output_text`
    if not text_blocks:
        for step in getattr(interaction, "steps", None) or []:
            step_type = step.get("type") if isinstance(step, dict) else getattr(step, "type", None)
            if step_type in ("model_output", "assistant", None):
                contents = (step.get("content") if isinstance(step, dict) else getattr(step, "content", None)) or []
                for c in contents:
                    c_type = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
                    txt = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)
                    if txt and (c_type in ("text", None)):
                        text_blocks.append(txt)
                    anns = (c.get("annotations") if isinstance(c, dict) else getattr(c, "annotations", None)) or []
                    for ann in anns:
                        u = ann.get("url") if isinstance(ann, dict) else getattr(ann, "url", None)
                        t = ann.get("title") if isinstance(ann, dict) else getattr(ann, "title", None)
                        _add_source(u, t)
        if not text_blocks:
            out_txt = getattr(interaction, "output_text", None)
            if out_txt:
                text_blocks.append(out_txt)

    report_text = "\n\n".join(text_blocks).strip()

    # Also harvest inline markdown links [title](https://...) from the report text
    for match in re.finditer(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", report_text):
        _add_source(match.group(2), match.group(1))
    for match in re.finditer(r"(?<!\()(https?://[^\s)\]>\"']+)", report_text):
        _add_source(match.group(1), None)

    return report_text, sources


def format_markdown(
    *,
    slug: str,
    prompt: str,
    agent_used: str,
    interaction_id: str,
    report_text: str,
    sources: Sequence[tuple[str, str]],
) -> str:
    """Format the saved markdown file with metadata header, report body, and sources."""
    lines = [
        f"# Deep Research: {slug}",
        "",
        f"- **Agent**: `{agent_used}`",
        f"- **Interaction ID**: `{interaction_id}`",
        f"- **Query**: {prompt}",
        "",
        "---",
        "",
        report_text or "_No text output returned._",
    ]
    if sources:
        lines.extend(["", "---", "", "## Sources (Extracted Metadata & Links)", ""])
        for title, url in sources:
            clean_title = title.replace("\n", " ").strip()
            if clean_title == url:
                lines.append(f"- <{url}>")
            else:
                lines.append(f"- [{clean_title}]({url})")
    lines.append("")
    return "\n".join(lines)


def create_with_fallback(
    client: object,
    prompt: str,
    agents: Iterable[str] = DEFAULT_AGENTS,
) -> tuple[object, str]:
    """Create a background Deep Research interaction, falling back across agent IDs."""
    last_err: Optional[Exception] = None
    for agent_id in agents:
        try:
            print(f"[deep_research] Starting background interaction with agent={agent_id} ...", flush=True)
            interaction = client.interactions.create(
                agent=agent_id,
                input=prompt,
                background=True,
            )
            return interaction, agent_id
        except Exception as exc:
            last_err = exc
            print(
                f"[deep_research] Agent {agent_id} failed on create ({type(exc).__name__}: {str(exc)[:160]}), trying fallback...",
                flush=True,
            )
    raise RuntimeError(f"All Deep Research agents failed to start. Last error: {last_err}") from last_err


def run_deep_research(
    prompt: str,
    slug: str,
    *,
    out_path: Optional[Path] = None,
    agents: Sequence[str] = DEFAULT_AGENTS,
    poll_interval: float = 10.0,
    timeout: float = 2400.0,
    client: Optional[object] = None,
) -> Path:
    """Run Deep Research in the background, poll until completion, and write markdown output."""
    _ensure_api_key()
    if client is None:
        from google import genai
        client = genai.Client()

    target = Path(out_path) if out_path else (RESEARCH_DIR / f"{slug}.md")
    target.parent.mkdir(parents=True, exist_ok=True)

    interaction, agent_used = create_with_fallback(client, prompt, agents=agents)
    interaction_id = getattr(interaction, "id", None)
    if not interaction_id:
        raise RuntimeError("Interaction created without an ID")

    print(f"[deep_research] Interaction id={interaction_id} status={getattr(interaction, 'status', 'unknown')}", flush=True)
    start_t = time.monotonic()

    # If the initial create already failed immediately on the backend, try fallback agent if available
    while True:
        status = getattr(interaction, "status", None)
        if status == "completed":
            break
        if status in ("failed", "cancelled", "incomplete"):
            # Check if we can fallback to the next agent if it failed quickly
            remaining = [a for a in agents if a != agent_used]
            if status == "failed" and remaining and (time.monotonic() - start_t) < 60.0:
                print(f"[deep_research] Agent {agent_used} failed quickly; falling back to {remaining[0]}...", flush=True)
                return run_deep_research(
                    prompt,
                    slug,
                    out_path=target,
                    agents=remaining,
                    poll_interval=poll_interval,
                    timeout=timeout,
                    client=client,
                )
            raise RuntimeError(f"Deep Research interaction {interaction_id} ended with status={status}")

        elapsed = time.monotonic() - start_t
        if elapsed > timeout:
            raise TimeoutError(f"Deep Research interaction {interaction_id} timed out after {elapsed:.0f}s")

        time.sleep(poll_interval)
        interaction = client.interactions.get(interaction_id)
        print(
            f"[deep_research] [{elapsed:.0f}s] id={interaction_id} status={getattr(interaction, 'status', 'unknown')}",
            flush=True,
        )

    report_text, sources = extract_report_and_sources(interaction)
    md = format_markdown(
        slug=slug,
        prompt=prompt,
        agent_used=agent_used,
        interaction_id=interaction_id,
        report_text=report_text,
        sources=sources,
    )
    target.write_text(md, encoding="utf-8")
    print(f"[deep_research] Saved report ({len(report_text)} chars, {len(sources)} sources) -> {target}", flush=True)
    return target


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemini Deep Research via Interactions API.")
    parser.add_argument("pos", nargs="*", help="Positional [slug] [prompt] or [prompt]")
    parser.add_argument("--slug", "-s", help="Output slug (saves to docs/research/<slug>.md)")
    parser.add_argument("--prompt", "--query", "-p", "-q", dest="prompt", help="Research query / prompt")
    parser.add_argument("--out", "-o", help="Explicit output path (overrides docs/research/<slug>.md)")
    parser.add_argument("--agent", help="Override agent ID (otherwise tries max then preview)")
    parser.add_argument("--poll-interval", type=float, default=10.0, help="Seconds between status polls")
    parser.add_argument("--timeout", type=float, default=2400.0, help="Max wait time in seconds")
    args = parser.parse_args(argv)

    slug = args.slug
    prompt = args.prompt
    pos = list(args.pos)

    if len(pos) >= 2:
        # Decide which is slug and which is prompt: slug is shorter and has no spaces
        if not slug and " " not in pos[0] and len(pos[0]) <= 64:
            slug = pos.pop(0)
            if not prompt:
                prompt = " ".join(pos)
        elif not slug and " " not in pos[-1] and len(pos[-1]) <= 64:
            slug = pos.pop(-1)
            if not prompt:
                prompt = " ".join(pos)
        else:
            if not prompt:
                prompt = " ".join(pos)
    elif len(pos) == 1:
        if prompt and not slug:
            slug = pos[0]
        else:
            prompt = pos[0]

    if not prompt:
        parser.error("A research prompt/query is required.")

    if not slug:
        if args.out:
            slug = Path(args.out).stem
        else:
            slug = slugify(prompt)

    slug = re.sub(r"\.md$", "", slug)
    args.resolved_slug = slug
    args.resolved_prompt = prompt
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    agents = (args.agent,) if args.agent else DEFAULT_AGENTS
    run_deep_research(
        prompt=args.resolved_prompt,
        slug=args.resolved_slug,
        out_path=Path(args.out) if args.out else None,
        agents=agents,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
