#!/usr/bin/env python3
"""SurgProof fabrication checker. Exit 1 if anything shown to a judge could be invented.

Checks:
  1. Display code (app.py, pages/, promo/, narrate.py) contains no literal boxes, verdicts,
     metric percentages or synthetic score formulas.
  2. Labeller output carries no ground truth; the verifier/localizer never read it.
  3. manual.json cites sources; results/student report their sample size.
  4. Every narration sentence cites accepted decision rows that exist.
Run: python3 lanes/integrity.py   (prints FAIL lines; exit 1 on any FAIL)
"""
import csv, glob, json, os, re, sys

R = os.path.expanduser("~/surgatlas")
os.chdir(R)
fails, notes = [], []


def fail(msg):
    fails.append(msg)


DISPLAY = [f for f in ["app.py", "narrate.py"] + glob.glob("pages/*.py") + glob.glob("promo/*.py") if os.path.exists(f)]
PATTERNS = [
    (r'["\']box["\']\s*:\s*\[\s*\d', "literal bounding box"),
    (r'["\']status["\']\s*:\s*["\'](accept|reject|AMBER|amber)["\']', "literal verdict"),
    (r'["\']\d{1,3}\.\d%["\']', "literal percentage string"),
    (r'0\.7\d?\s*\+\s*0\.\d+\s*\*', "synthetic score formula"),
    (r'(accuracy|acc|precision|coverage)\s*=\s*0\.\d{2,}', "hard-coded metric"),
]
for f in DISPLAY:
    for i, line in enumerate(open(f, errors="ignore"), 1):
        s = line.strip()
        if s.startswith("#") or "integrity: allow" in s:
            continue
        for pat, what in PATTERNS:
            if re.search(pat, line):
                fail(f"{f}:{i}: {what}: {s[:110]}")

for f in glob.glob("out/labels_*.jsonl") + glob.glob("coord/artifacts/labels_*.jsonl"):
    with open(f) as fh:
        for n, line in enumerate(fh):
            if n > 2000:
                break
            r = json.loads(line)
            if "truth" in r or any(k.startswith("gt_") for k in r):
                fail(f"{f}: ground truth inside labeller output (row {n})")
                break

for f in ["accept.py", "localize.py", "verify_localize.py"] + glob.glob("verify/*.py"):
    if os.path.exists(f):
        for i, line in enumerate(open(f, errors="ignore"), 1):
            if re.search(r'gt_|\btruth\b|mask_path', line) and not line.strip().startswith("#") and "integrity: eval-only" not in line:
                fail(f"{f}:{i}: verifier/localizer touches ground truth: {line.strip()[:100]}")

if os.path.exists("out/manual.json"):
    m = json.load(open("out/manual.json"))
    if not re.search(r"https?://", json.dumps(m.get("sources", ""))):
        fail("out/manual.json: no source URLs")
for f in ["out/results.json", "out/student.json"]:
    if os.path.exists(f):
        s = json.dumps(json.load(open(f)))
        if not re.search(r'"n_[a-z_]*"\s*:\s*\d', s):
            fail(f"{f}: no sample size (n_*) reported")
        else:
            notes.append(f"{f}: n fields present")

dec = {}
if os.path.exists("out/decisions.csv"):
    for r in csv.DictReader(open("out/decisions.csv")):
        dec[(r.get("frame_id"), r.get("key") or r.get("label"))] = r.get("status")
for f in glob.glob("out/narration_*.json"):
    n = json.load(open(f))
    sents = n.get("sentences", n if isinstance(n, list) else [])
    for s in sents:
        ev = s.get("evidence", []) if isinstance(s, dict) else []
        if not ev:
            fail(f"{f}: sentence without evidence: {str(s)[:90]}")
            continue
        for e in ev:
            st = dec.get((e.get("frame_id"), e.get("key") or e.get("label")))
            if st != "accept":
                fail(f"{f}: evidence not an accepted decision: {e}")


SUPPORTED = {"endoscapes2023": {"gallbladder", "cystic_duct"}, "cholect50": {"grasper", "l_hook"}}
for name, ok in SUPPORTED.items():
    f = f"data/external/{name}/index.csv"
    if os.path.exists(f):
        rows = list(csv.DictReader(open(f)))
        for k in ["grasper", "l_hook", "liver", "gallbladder", "cystic_duct", "blood"]:
            if k in ok:
                continue
            vals = [r.get(f"gt_{k}", "") for r in rows]
            if any(v not in ("", "nan", "NaN", "None") for v in vals):
                fail(f"{f}: gt_{k} is filled but {name} never annotates {k} (must be blank / unscored)")
for f in glob.glob("out/*results*.json") + glob.glob("out/eval*.json"):
    s2 = json.dumps(json.load(open(f))).lower()
    for name, ok in SUPPORTED.items():
        for k in {"grasper", "l_hook", "liver", "gallbladder", "cystic_duct", "blood"} - ok:
            if re.search(rf'"{name}[^"]*{k}|"{k}[^"]*{name}', s2):
                fail(f"{f}: reports {k} on {name}, which has no {k} ground truth")

# ---- PROVENANCE: every model output must trace to a real recorded Gemini response ----
import random, sqlite3
if os.path.exists("cache.sqlite"):
    con = sqlite3.connect("cache.sqlite")
    blob = "\n".join(v or "" for (v,) in con.execute("select v from cache"))
    con.close()
    norm = lambda t: re.sub(r"\s+", " ", str(t)).strip().lower()
    blob_n = norm(blob)
    def traced(row):
        ev = row.get("evidence") or row.get("rationale") or ""
        return len(ev) < 12 or norm(ev)[:60] in blob_n
    for f in glob.glob("out/labels_*.jsonl"):
        rows = [json.loads(l) for l in open(f)]
        sample = random.Random(0).sample(rows, min(300, len(rows)))
        bad = [r for r in sample if not traced(r)]
        if bad:
            fail(f"{f}: {len(bad)}/{len(sample)} sampled labels have evidence text found in NO recorded Gemini response (e.g. {bad[0].get('frame_id')}/{bad[0].get('key')})")
        else:
            notes.append(f"{f}: {len(sample)} sampled labels traced to recorded model responses")
    if os.path.exists("out/decisions.csv"):
        drows = list(csv.DictReader(open("out/decisions.csv")))
        sample = random.Random(1).sample(drows, min(300, len(drows)))
        bad = [r for r in sample if not traced(r)]
        if bad:
            fail(f"out/decisions.csv: {len(bad)}/{len(sample)} sampled decisions not traceable to a recorded model response")
        else:
            notes.append(f"out/decisions.csv: {len(sample)} sampled decisions traced to recorded model responses")

# ---- PROMO/APP must declare their sources, and the declared values must match the data ----
man = "out/promo/manifest.json"
if os.path.exists(man):
    m = json.load(open(man))
    dec_idx = {}
    if os.path.exists("out/decisions.csv"):
        for r in csv.DictReader(open("out/decisions.csv")):
            dec_idx[(r["frame_id"], r["key"])] = r
    for shot, s in m.get("shots", {}).items():
        for item in s.get("shown_labels", []) or []:
            d = dec_idx.get((item.get("frame_id"), item.get("key")))
            if d is None:
                fail(f"{man}: {shot} shows {item} but that frame/label is not in decisions.csv")
            elif item.get("status") and item["status"].lower() != str(d.get("status", "")).lower():
                fail(f"{man}: {shot} shows status {item['status']} for {item.get('key')} but decisions.csv says {d.get('status')}")
        hn = s.get("hero_number")
        if hn not in (None, "", "pending") and not s.get("hero_number_source"):
            fail(f"{man}: {shot} shows a number ({hn}) with no hero_number_source (file + key)")

print(f"integrity: {len(fails)} FAIL, scanned {len(DISPLAY)} display files")
for x in fails:
    print("FAIL", x)
for x in notes:
    print("ok  ", x)
sys.exit(1 if fails else 0)
