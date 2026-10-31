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
stage("data index", 100 if os.path.exists(p("data/index.csv")) else 0, f"{max(rows(p('data/index.csv'))-1,0)} frames, {age(p('data/index.csv'))}")
stage("features", 100 if os.path.exists(p("data/features.npy")) else 0, age(p("data/features.npy")))
m = p("out/manual.json"); src = os.path.exists(m) and "http" in open(m).read()
stage("manual (sourced)", 100 if src else 50 if os.path.exists(m) else 0, ("has source URLs" if src else "NO source URLs") + f", {age(m)}")
lm = p("out/labels_manual.jsonl"); stage("labels (manual)", rows(lm)/6060*100, f"{rows(lm)}/6060 rows, {age(lm)}")
lb = p("out/labels_blank.jsonl"); stage("blank baseline", rows(lb)/1680*100, f"{rows(lb)}/1680 test rows")
d = p("out/decisions.csv"); r = p("out/results.json")
stage("verifier decisions", (50 if os.path.exists(d) else 0) + (50 if os.path.exists(r) else 0), f"decisions {age(d)}, results {age(r)}")
stage("student", 100 if os.path.exists(p("out/student.json")) else 0, age(p("out/student.json")))
ev = [f for f in glob.glob(p("out/eval*.json")) + glob.glob(p("out/eval/*.json")) if "mock" not in f]; stage("eval + CIs", 100 if ev else 0, ", ".join(os.path.basename(f) for f in ev) or "no real eval json")
ext = glob.glob(p("data/external/*/index.csv")); stage("external test sets", len(ext)/2*100, f"{len(ext)}/2 indexed")
stage("human gate (#14)", issue_state(14), "issue #14")
stage("live mode (#15)", issue_state(15), "issue #15")
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
appr = any(c["body"].startswith("[nir-director]") and "APPROVED" in c["body"] for c in (cs if 'cs' in dir() else []))
stage("promo video", 100 if appr else min(len(mp4)/8*60, 60), f"{len(mp4)} clips; " + ("director APPROVED" if appr else "not approved by nir-director"))
dl = dt.datetime.now().replace(hour=15, minute=15, second=0)
mins = int((dl - dt.datetime.now()).total_seconds() // 60)
try:
    bk = subprocess.run(["rclone","cat","gdrive:SurgProof/backup/LAST_BACKUP"],capture_output=True,text=True,timeout=12).stdout.strip()
except Exception:
    bk = "unknown (Drive slow)"
print(json.dumps({"at": dt.datetime.now().strftime("%H:%M:%S"), "minsLeft": mins, "overall": int(sum(s["pct"] for s in S)/len(S)), "stages": S,
  "agents": [{"lane": k, **v} for k, v in sorted(A.items())], "backup": bk or "none yet"}))
