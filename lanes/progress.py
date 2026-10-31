#!/usr/bin/env python3
"""Prints SurgProof progress as JSON for the Claude Code pane. Evidence = files on disk + posts on issue #1."""
import json, os, glob, time, subprocess, re, datetime as dt
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", "")
R = os.path.expanduser("~/surgatlas"); now = time.time()
def p(*a): return os.path.join(R, *a)
def age(f): return f"{int((now-os.path.getmtime(f))//60)}m ago" if os.path.exists(f) else "missing"
def rows(f):
    try: return sum(1 for _ in open(f))
    except Exception: return 0
def issue_state(n):
    try:
        o = json.loads(subprocess.run(["gh","issue","view",str(n),"-R","nirmathur/surgproof","--json","state,labels"],capture_output=True,text=True,timeout=20).stdout)
        ls = [l["name"] for l in o["labels"]]
        return 100 if o["state"]=="CLOSED" else 50 if "status:claimed" in ls else 10
    except Exception: return 0
S = []
def stage(name, pct, detail): S.append({"name": name, "pct": max(0, min(100, int(pct))), "detail": detail})
def jload(f):
    try: return json.load(open(p(f)))
    except Exception: return None
def nrows(f):
    try: return sum(1 for _ in open(p(f)))
    except Exception: return 0
# 1 presence labels: full CholecSeg8k run
lm = nrows("out/labels_manual.jsonl"); stage("labels: full run", lm/6060*100, f"{lm}/6060 rows")
# 2 verification on FULL test (n>=1400 excl liver) + amber loop log
r = jload("out/results.json") or {}; acc = (r.get("accepted_5_informative") or {})
ok_n = (r.get("n_test_labels") or 0) >= 1680
stage("verify + amber loop", (60 if ok_n else 20) + (40 if nrows("out/amber_rounds.jsonl") else 0),
      f"n={r.get('n_test_labels')}, acc {acc.get('accuracy')} @ cov {acc.get('coverage')}, amber rounds {nrows('out/amber_rounds.jsonl')}")
# 3 student + confidence intervals
st = jload("out/student.json"); sts = json.dumps(st or {}).lower()
stage("student + CIs", (50 if st else 0) + (50 if ("ci" in sts and "boot" in sts) else 0), "CIs present" if "boot" in sts else "NO bootstrap CIs yet")
# 4 AI masks/boxes on video12
ann = [json.loads(l) for l in open(p("out/annotations.jsonl"))] if os.path.exists(p("out/annotations.jsonl")) else []
v12 = sum(1 for _ in open(p("data/index.csv")) if "video12" in _)
m12 = len({a.get("frame_id") for a in ann if "video12" in str(a.get("frame_id")) and (a.get("mask") or a.get("mask_path"))})
b12 = len({a.get("frame_id") for a in ann if "video12" in str(a.get("frame_id")) and a.get("box")})
stage("AI masks (video12)", m12/max(v12,1)*100, f"{m12}/{v12} frames with masks, {b12} with boxes")
lr = jload("out/localize_results.json"); stage("mask scoring vs GT", 0 if m12 == 0 else 100 if lr and re.search(r'"n[_a-z]*"\s*:\s*\d', json.dumps(lr)) else 0, "IoU/Dice with n" if lr else "missing")
ph = len({a.get("frame_id") for a in ann if a.get("phase") and "video12" in str(a.get("frame_id"))}); stage("phases per frame", ph/max(v12,1)*100, f"{ph}/{v12} video12 frames")
# 5 web UI actually serving real data
import urllib.request
def up(u):
    try: return urllib.request.urlopen(u, timeout=3).status == 200
    except Exception: return False
w_files = all(os.path.exists(p(f)) for f in ["web/index.html", "web/app.js", "web/style.css"])
w_data = os.path.exists(p("web/data/video12.json")); w_up = up("http://localhost:8600/")
stage("web toolbox UI", 30*w_files + 30*w_data + 40*w_up, f"files {w_files}, data {w_data}, serving :8600 {w_up}")
srv = open(p("web/serve.py")).read() if os.path.exists(p("web/serve.py")) else ""
stage("final pass (human)", (40 if "POST" in srv else 0) + (60 if nrows("out/gate_log.jsonl") else 0), f"POST endpoint {'POST' in srv}, gate_log rows {nrows('out/gate_log.jsonl')}")
# 6 integrity: canonical checker must be 0 FAIL
try:
    o = subprocess.run(["python3", os.path.expanduser("/private/tmp/claude-501/-Users-nirmathur/434c1202-c152-485b-8e37-ec0e91406d8b/scratchpad/integrity_canonical.py")], capture_output=True, text=True, timeout=120).stdout
    nf = int(re.search(r"(\d+) FAIL", o).group(1))
except Exception: nf = -1
stage("NOTHING FABRICATED", 100 if nf == 0 else 0, f"{nf} FAIL (canonical checker)")
nar = glob.glob(p("out/narration_*.json")); stage("narration (evidence-checked)", 100 if nar and nf == 0 else 50 if nar else 0, f"{len(nar)} files")
blk = 0
try: blk = sum(1 for l in open(p("out/labels_endoscapes2023_manual.jsonl")) if re.search(r"black (frame|image|screen)", l, re.I))
except Exception: pass
stage("external test (Endoscapes)", 0 if blk else 100 if os.path.exists(p("out/decisions_endoscapes2023.csv")) else 0, f"{blk} labels on black frames" if blk else "clean")
rec = glob.glob(p("out/recording*.mp4")) + glob.glob(p("out/*backup*.mp4")); stage("backup recording", 100 if rec else 0, f"{len(rec)} files")
A = {}
try:
    cs = json.loads(subprocess.run(["gh","issue","view","1","-R","nirmathur/surgproof","--json","comments"],capture_output=True,text=True,timeout=20).stdout)["comments"]
    for c in cs:
        mm = re.match(r"\[((?:nir|harald)-[a-z]+)\]", c["body"])
        if mm and mm.group(1) != "nir-orchestrator":
            t = dt.datetime.fromisoformat(c["createdAt"].replace("Z","+00:00")).timestamp()
            txt = re.sub(r"\s+"," ",c["body"][len(mm.group(0)):]).strip()
            nx = re.search(r"NEXT:\s*([^|]+)", txt)
            A[mm.group(1)] = {"mins": int((now-t)//60), "doing": (nx.group(1) if nx else txt)[:110]}
except Exception: pass
mp4 = glob.glob(p("out/promo/*.mp4"))
_latest = max([os.path.getmtime(f) for f in mp4] or [0])
appr = any(c["body"].startswith("[nir-director]") and re.search(r"\bAPPROVED:\s*\S*out/promo/\S+\.mp4", c["body"]) and dt.datetime.fromisoformat(c["createdAt"].replace("Z","+00:00")).timestamp() > _latest for c in (cs if 'cs' in dir() else []))
stage("promo video", 100 if appr else min(len(mp4)/8*60, 60), f"{len(mp4)} clips; " + ("director APPROVED" if appr else "not approved by nir-director"))
OWN = {"nir-label":"labels: full run","nir-verifier":"verify + amber loop","nir-student":"student + CIs","nir-localize":"AI masks (video12)",
 "nir-masks2":"AI masks (video12)","nir-masks3":"AI masks (video12)","nir-webshell":"web toolbox UI","nir-webcanvas":"web toolbox UI",
 "nir-webtimeline":"web toolbox UI","nir-demo":"web toolbox UI","nir-gate":"final pass (human)","nir-director":"promo video",
 "nir-promo":"promo video","nir-manual":"narration (evidence-checked)","nir-data":"external test (Endoscapes)","nir-recorder":"backup recording",
 "nir-architect":"NOTHING FABRICATED","nir-coach":"NOTHING FABRICATED","harald-ui":"web toolbox UI"}
_pct = {x["name"]: x["pct"] for x in S}
dl = dt.datetime.now().replace(hour=15, minute=15, second=0)
mins = int((dl - dt.datetime.now()).total_seconds() // 60)
try:
    bk = subprocess.run(["rclone","cat","gdrive:SurgProof/backup/LAST_BACKUP"],capture_output=True,text=True,timeout=12).stdout.strip()
except Exception:
    bk = "unknown (Drive slow)"
print(json.dumps({"at": dt.datetime.now().strftime("%H:%M:%S"), "minsLeft": mins, "overall": int(sum(s["pct"] for s in S)/len(S)), "stages": S,
  "agents": [{"lane": k, **v, "task": OWN.get(k, "-"), "pct": _pct.get(OWN.get(k, ""), -1)} for k, v in sorted(A.items(), key=lambda kv: _pct.get(OWN.get(kv[0], ""), 101))], "backup": bk or "none yet"}))
