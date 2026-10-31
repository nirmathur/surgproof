"""Demo UI.  streamlit run app.py"""
import json
import time

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import config as C

st.set_page_config(page_title="Surgical Atlas", layout="wide")
O = C.OUT_DIR
COL = {"accept": "#1f9d55", "reanalyse": "#d69e2e", "escalate": "#e53e3e"}


def load_json(p, default=None):
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def draw(path, rows, reveal):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    d = ImageDraw.Draw(im)
    for r in rows.itertuples():
        box = json.loads(r.box) if isinstance(r.box, str) else r.box
        if r.final_present and isinstance(box, list) and len(box) == 4:
            y0, x0, y1, x1 = box
            colour = COL[r.status]
            if reveal:
                colour = "#1f9d55" if r.final_correct else "#e53e3e"
            d.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], outline=colour, width=4)
            d.text((x0 * w / 1000 + 4, y0 * h / 1000 + 4), r.key, fill=colour)
    return im


# ---------------- sidebar: live agent reasoning ----------------
with st.sidebar:
    st.subheader("Agent reasoning")
    try:
        lines = (O / "trace.jsonl").read_text().splitlines()[-25:]
        for ln in reversed(lines):
            t = json.loads(ln)
            st.markdown(f"**{t['agent']}** {t['msg']}")
    except FileNotFoundError:
        st.caption("No trace yet.")

tab1, tab2, tab3 = st.tabs(["1 · Any surgery", "2 · Proof", "3 · Watch (Netflix for surgery)"])

# ---------------- Act 1 ----------------
with tab1:
    st.header("Any surgery, zero setup")
    atlases = sorted((O / "atlas").glob("*/dataset.json")) if (O / "atlas").exists() else []
    if not atlases:
        st.info("Run atlas.py first.")
    else:
        pick = st.selectbox("Procedure", atlases, format_func=lambda p: p.parent.name)
        ds = load_json(pick)
        man = load_json(pick.parent / "manual.json", {})
        c1, c2 = st.columns([1, 1])
        with c1:
            st.video(ds["source_video"])
            st.metric("Human labels used", ds["human_labels_used"])
            st.metric("Audit confidence", f"{ds['audit']['overall_confidence']:.2f}")
        with c2:
            st.subheader("Manual the agent wrote itself")
            for s in man.get("structures", []):
                st.markdown(f"**{s['name']}**: {s['appearance']} *(confused with {', '.join(s['confused_with'])})*")
        st.subheader("Machine-readable timeline")
        st.dataframe(pd.DataFrame(ds["timeline"]["phases"]), width="stretch")
        st.dataframe(pd.DataFrame(ds["timeline"]["events"]), width="stretch")
        if ds["audit"]["problems"]:
            st.warning("Flagged by Argon audit: " + "; ".join(ds["audit"]["problems"]))
        st.download_button("Download dataset.json", pick.read_text(), file_name="dataset.json")

# ---------------- Act 2 ----------------
with tab2:
    st.header("Proof: CholecSeg8k, labels hidden from the agent")
    res = load_json(O / "results.json")
    stu = load_json(O / "student.json")
    if not res:
        st.info("Run accept.py first.")
    else:
        a = res["accepted"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy on accepted labels", f"{a['accuracy'] * 100:.1f}%" if a["accuracy"] else "n/a")
        c2.metric("Labels accepted (coverage)", f"{a['coverage'] * 100:.0f}%")
        c3.metric("Sent to surgeon", f"{a['escalated_share'] * 100:.0f}%")
        if "blank_raw_acc" in res:
            c4.metric("Self-written manual vs blank", f"{res['manual_raw_acc'] * 100:.1f}%",
                      f"{(res['manual_raw_acc'] - res['blank_raw_acc']) * 100:+.1f} pts")
        curves = {}
        for k, v in res.items():
            if isinstance(v, dict) and "curve" in v:
                for cov, acc in v["curve"]:
                    curves.setdefault(cov, {})[k] = acc
        st.subheader("Accuracy vs share of labels accepted (higher line = better)")
        st.line_chart(pd.DataFrame(curves).T.sort_index())
        if stu:
            st.subheader("Student trained on the system's own labels (test videos, real GT)")
            st.bar_chart(pd.Series(stu["mean_acc_by_source"]))
            st.caption("oracle_upper_bound uses human labels and is shown for reference only.")

# ---------------- Act 3 ----------------
with tab3:
    st.header("Watch")
    try:
        dec = pd.read_csv(O / "decisions.csv")
    except FileNotFoundError:
        st.info("Run accept.py first.")
        st.stop()
    idx = pd.read_csv(C.DATA_DIR / "index.csv").set_index("frame_id")
    vids = sorted(dec[dec.split == "test"].video.unique())
    v = st.selectbox("Test video", vids)
    narr = {x["frame_id"]: x for x in load_json(O / f"narration_{v}.json", [])}
    frames = sorted(dec[dec.video == v].frame_id.unique(),
                    key=lambda f: (f.rsplit("_", 1)[0], int(f.rsplit("_", 1)[1])))
    reveal = st.toggle("Reveal ground truth")
    play = st.button("▶ Play")
    i = st.slider("Frame", 0, len(frames) - 1, 0)
    slot_img, slot_chips, slot_narr = st.empty(), st.empty(), st.empty()

    def show(f):
        rows = dec[dec.frame_id == f]
        slot_img.image(draw(idx.loc[f, "path"], rows, reveal), width="stretch")
        chips = []
        for r in rows.itertuples():
            colour = COL[r.status]
            label = f"{r.key}: {'yes' if r.final_present else 'no'}"
            if reveal:
                label += " ✓" if r.final_correct else " ✗"
            chips.append(f"<span style='background:{colour};color:white;padding:4px 10px;"
                         f"border-radius:12px;margin:3px;display:inline-block'>{label}</span>")
        slot_chips.markdown(" ".join(chips), unsafe_allow_html=True)
        if f in narr:
            slot_narr.markdown(f"### {narr[f]['narration']}")

    if play:
        for f in frames[i:]:
            show(f)
            time.sleep(1.2)
    else:
        show(frames[i])
