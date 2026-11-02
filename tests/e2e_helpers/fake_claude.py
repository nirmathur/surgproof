#!/usr/bin/env python3
"""A stand-in for the `claude` binary, used through SURGPROOF_CLAUDE_BIN.

Unlike SURGPROOF_VLM=mock this exercises the real claude_cli code path: argv, working directory, envelope
parsing, retries, validation and caching. FAKE_CLAUDE_MODE picks a misbehaviour, so the tests can ask what
the pipeline does when the model is dead, lying, unsure or returning junk. Every call is logged to
FAKE_CLAUDE_LOG as one JSON line: argv, cwd and directory listing.

Answers are plain shapes for plumbing tests. Rows are tagged with the fake's --model argument, so nothing
downstream is told this is mock data; these runs live in temp dirs and never reach a report.
"""
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def arg(name, default=None):
    a = sys.argv
    return a[a.index(name) + 1] if name in a else default


def strip_prefix(prompt):
    """vlm._claude_cli prepends the image list; the prompts module matches on what follows."""
    m = re.match(r"Use the Read tool to open: (.*?)\n\n", prompt, re.S)
    return (m.group(1).split(", "), prompt[m.end():]) if m else ([], prompt)


def base_reply(prompt, images, model):
    from surgproof.prompts import mock_reply

    return mock_reply(prompt, images, model)


def apply_mode(mode, reply, prompt, images):
    """Change what a normal answer says. Returns the object the model would have sent."""
    labels = ["gallbladder", "liver", "grasper", "l_hook", "cystic_duct", "blood"]
    is_detect = prompt.startswith("This is one frame")
    is_verify = prompt.startswith("You are auditing")
    is_amber = "magnified crop" in prompt[:80]
    is_narrate = prompt.startswith("You are writing subtitle")
    if mode == "all_absent" and (is_detect or is_verify):
        for k in labels:
            reply["objects"][k] = {"present": False, "conf": 0.95, "polygon" if is_detect else "box": None}
    elif mode == "agree_all":  # both passes agree everything in the mock's present set is there
        if is_verify:
            reply["objects"]["cystic_duct"] = {"present": False, "conf": 0.9, "box": None}
    elif mode == "unsure" and (is_detect or is_verify):
        for k in labels:
            reply["objects"][k]["conf"] = 0.3
    elif mode == "unsure" and is_amber:
        reply.update(present=True, conf=0.3, polygon=None, reason="cannot tell")
    elif mode == "amber_flips" and is_amber:
        h = hash(tuple(images)) % 2 == 0
        reply.update(present=h, conf=0.4, polygon=None, reason="flip")
    elif mode == "amber_stuck" and is_amber:  # the loop cannot settle cystic_duct, so it goes to the human
        reply.update(present=True, conf=0.3, polygon=None, reason="cannot tell")
    elif mode == "narrate_lies" and is_narrate:
        kf = re.search(r"kf_\d+", prompt)
        reply["sentences"] = [{"text": "Blood is visible and the cystic duct has been clipped.", "evidence": [kf.group(0)] if kf else []}]
    elif mode == "narrate_fake_evidence" and is_narrate:
        reply["sentences"] = [{"text": "The grasper is visible.", "evidence": ["kf_99999"]}]
    elif mode == "narrate_no_evidence" and is_narrate:
        reply["sentences"] = [{"text": "The grasper is visible.", "evidence": []}]
    elif mode == "polygon_out_of_bounds" and is_detect:
        for k in labels:
            if reply["objects"][k]["present"]:
                reply["objects"][k]["polygon"] = [[-5000, -5000], [90000, -10], [90000, 90000]]
    elif mode == "polygon_degenerate" and is_detect:
        for k in labels:
            if reply["objects"][k]["present"]:
                reply["objects"][k]["polygon"] = [[10, 10], [10, 10]]
    elif mode == "shifted_right" and is_detect:
        for k in labels:
            if reply["objects"][k]["present"]:
                reply["objects"][k]["polygon"] = [[x + 150, y] for x, y in reply["objects"][k]["polygon"]]
    return reply


def main():
    prompt = arg("-p", "")
    model = arg("--model", "")
    mode = os.environ.get("FAKE_CLAUDE_MODE", "normal")
    log = os.environ.get("FAKE_CLAUDE_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "listdir": sorted(os.listdir(".")), "mode": mode}) + "\n")
    if mode == "dead":
        sys.stderr.write("connection refused\n")
        return 1
    if mode == "logged_out":
        print(json.dumps({"is_error": True, "result": "Not logged in. Please run /login"}))
        return 1
    images, body = strip_prefix(prompt)
    for p in images:
        if not os.path.exists(p):
            print(json.dumps({"is_error": True, "result": f"cannot read {p}"}))
            return 1
    if mode == "garbage":
        print(json.dumps({"result": "I can't help with that, sorry.", "total_cost_usd": 0}))
        return 0
    if mode == "wrong_schema":
        print(json.dumps({"result": json.dumps({"hello": "world"}), "total_cost_usd": 0}))
        return 0
    reply = apply_mode(mode, base_reply(body, images, model), body, images)
    print(json.dumps({"result": "Here is the JSON:\n```json\n" + json.dumps(reply) + "\n```", "total_cost_usd": 0.001, "is_error": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
