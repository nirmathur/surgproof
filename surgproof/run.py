"""CLI: python -m surgproof.run path/to/video.mp4 --out jobs/<id>"""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

from surgproof import pipeline, vlm


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m surgproof.run", description="Laparoscopic cholecystectomy video in, an AI-draft annotated dataset out, for a human to review. Public or simulator footage only.")
    ap.add_argument("video", help="path to an mp4 or mov")
    ap.add_argument("--out", help="job folder (default: jobs/job-<random>)")
    ap.add_argument("--kf-fps", type=float, default=1.0, help="keyframes per second sent to the model (default 1)")
    ap.add_argument("--max-minutes", type=float, default=pipeline.MAX_MINUTES, help="refuse longer videos; 0 removes the limit")
    ap.add_argument("--concurrency", type=int, default=None, help=f"model calls at once (default {vlm.DEFAULT_CONCURRENCY})")
    ap.add_argument("--max-cost", type=float, default=None, help=f"cost cap in USD (default {pipeline.DEFAULT_MAX_COST_USD:g}, or SURGPROOF_MAX_COST_USD)")
    ap.add_argument("--stop-after", choices=["ingest", "detect", "amber"], help="stop early, for testing the model stages")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path("jobs") / f"job-{secrets.token_hex(4)}"  # a file name can identify someone

    if vlm.provider() != "claude_cli":
        print(f"NOTE: provider is {vlm.provider()}; outputs are tagged {vlm.model_tag(vlm.SONNET)}", file=sys.stderr)

    last = {}

    def show(s):
        line = (s["stage"], s["message"])
        if line != last.get("line"):
            last["line"] = line
            print(f"[{s['progress'] * 100:5.1f}%] {s['stage']:<10} {s['message']}", flush=True)

    try:
        final = pipeline.run(args.video, out, kf_fps=args.kf_fps, max_minutes=args.max_minutes,
                             stop_after=args.stop_after, concurrency=args.concurrency, on_status=show, max_cost=args.max_cost)
    except (pipeline.PipelineError, vlm.VlmError) as e:
        print(f"FAILED: {pipeline._hide_name(str(e), args.video)}", file=sys.stderr)
        return 1
    print(f"{final['message']}\nmodel cost: ${final['cost_usd']:.2f}\njob folder: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
