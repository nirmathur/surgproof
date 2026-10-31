"""Demo UI.  streamlit run app.py --server.headless true"""
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

import config as C

st.set_page_config(page_title="Surgical Atlas", layout="wide")
O = C.OUT_DIR
COORD_ARTIFACTS = C.ROOT / "coord" / "artifacts"
COL = {"accept": "#1f9d55", "reanalyse": "#d69e2e", "escalate": "#e53e3e"}


def resolve_artifact(rel_path: str | Path) -> Path:
    """Resolve an artifact from out/ first, falling back to coord/artifacts/."""
    p = O / rel_path
    if p.exists():
        return p
    alt = COORD_ARTIFACTS / Path(rel_path).name
    if alt.exists():
        return alt
    return p


def load_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return default


def parse_box(raw_box) -> list[float] | None:
    """Safely parse a normalized [ymin, xmin, ymax, xmax] bounding box in 0..1000."""
    if raw_box is None or (isinstance(raw_box, float) and pd.isna(raw_box)):
        return None
    box = raw_box
    if isinstance(raw_box, str):
        s = raw_box.strip()
        if not s or s == "[]":
            return None
        try:
            box = json.loads(s)
        except Exception:
            return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        y0, x0, y1, x1 = [ float(v) for v in box ]
    except (TypeError, ValueError):
        return None
    y0, y1 = max(0.0, min(1000.0, min(y0, y1))), max(0.0, min(1000.0, max(y0, y1)))
    x0, x1 = max(0.0, min(1000.0, min(x0, x1))), max(0.0, min(1000.0, max(x0, x1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [y0, x0, y1, x1]


def resolve_frame_image(path_str: str | Path | None, frame_id: str = "") -> Image.Image:
    """Load frame image from disk, or return a clearly labelled placeholder canvas if unavailable."""
    candidates = []
    if path_str and not (isinstance(path_str, float) and pd.isna(path_str)):
        p = Path(str(path_str))
        candidates.extend([p, C.ROOT / p])
    for cand in candidates:
        if cand.is_file():
            try:
                return Image.open(cand).convert("RGB")
            except Exception:
                pass
    im = Image.new("RGB", (640, 480), color=(28, 32, 38))
    d = ImageDraw.Draw(im)
    d.rectangle([12, 12, 628, 468], outline=(90, 100, 115), width=2)
    title = f"[PLACEHOLDER CANVAS — Frame file unavailable: {frame_id or 'awaiting data/index.csv'}]"
    d.text((28, 220), title, fill=(210, 215, 225))
    d.text((28, 248), "Real endoscopic frame renders here once raw/ & data/index.csv are present.", fill=(150, 160, 175))
    return im


def draw(path, rows: pd.DataFrame, reveal: bool, frame_id: str = "") -> Image.Image:
    """Render frame image with verification-status (or ground-truth reveal) bounding boxes."""
    im = resolve_frame_image(path, frame_id=frame_id)
    w, h = im.size
    d = ImageDraw.Draw(im)
    if rows is None or rows.empty:
        return im
    for r in rows.itertuples():
        box = parse_box(getattr(r, "box", None))
        final_present = bool(getattr(r, "final_present", getattr(r, "present", 0)))
        if final_present and box is not None:
            y0, x0, y1, x1 = box
            status_val = getattr(r, "status", "reanalyse")
            colour = COL.get(status_val, "#d69e2e")
            if reveal and hasattr(r, "final_correct"):
                colour = "#1f9d55" if getattr(r, "final_correct", 0) else "#e53e3e"
            d.rectangle([x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000], outline=colour, width=4)
            d.text((x0 * w / 1000 + 4, y0 * h / 1000 + 4), str(getattr(r, "key", "")), fill=colour)
    return im


def frame_sort_key(frame_id: str):
    parts = str(frame_id).rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return (parts[0], int(parts[1]))
    return (str(frame_id), 0)


def read_trace_lines(limit: int = 25) -> list[dict]:
    trace_path = resolve_artifact("trace.jsonl")
    if not trace_path.exists():
        return []
    out = []
    try:
        for ln in trace_path.read_text().splitlines()[-limit:]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
                if isinstance(rec, dict) and "agent" in rec and "msg" in rec:
                    out.append(rec)
            except Exception:
                continue
    except Exception:
        return []
    return list(reversed(out))


# ---------------- Top Header: Honest Claims & Prior Work (Always on screen) ----------------
st.title("Surgical Atlas")
st.caption(
    "An autonomous Gemini agent that teaches itself a surgery from public sources, labels surgical video, "
    "checks its own labels so no human has to, trains a specialist only on what it verified, and narrates "
    "the operation using only what it can stand behind."
)

hc_col, pw_col = st.columns(2)
with hc_col:
    st.markdown(
        """
**Honest Claims (Strict Evaluation Protocol)**
- **"No human labels used in training."** Ground truth is hidden from the agent and used **only** to evaluate on held-out test videos (`split == "test"`).
- **Upper bounds explicitly marked:** The `calib` split (`upper_bound_calibrated`) and `oracle_upper_bound` student use human labels and are reported strictly as **upper bounds** for reference only.
- **Act 1 (any surgery) has no ground truth:** Showcase of zero-setup generalization, not accuracy proof. Act 2 (CholecSeg8k) is the blind proof.
"""
    )
with pw_col:
    st.markdown(
        """
**Prior Work & Our Step**
- **SurgVLP** *(CAMMA Strasbourg)*: Learning representations from narrated surgical video lectures.
- **SurgSLOT**: Self-evolving surgical data engine with **human verification** in the loop.
- **Med-Gemini**: Zero-shot Critical View of Safety (CVS) assessment on surgical video.
- **Our Step**: An autonomous Gemini agent **replaces the human verifier** (via blind crop check + sceptical critic + Argon revision), and verified labels train a better specialist student.
"""
    )

st.divider()

# ---------------- Sidebar: Pipeline Status, Promo Video & Live Agent Reasoning ----------------
with st.sidebar:
    st.subheader("Pipeline Artifacts")
    artifact_checks = {
        "data/index.csv": (C.DATA_DIR / "index.csv").exists(),
        "out/manual.json": resolve_artifact("manual.json").exists(),
        "out/labels_manual.jsonl": resolve_artifact("labels_manual.jsonl").exists() or resolve_artifact("labels_smoke.jsonl").exists(),
        "out/decisions.csv": resolve_artifact("decisions.csv").exists(),
        "out/results.json": resolve_artifact("results.json").exists(),
        "out/student.json": resolve_artifact("student.json").exists(),
        "out/atlas/*/dataset.json": bool((O / "atlas").exists() and list((O / "atlas").glob("*/dataset.json"))),
        "out/narration_*.json": bool(list(O.glob("narration_*.json")) or list(COORD_ARTIFACTS.glob("narration_*.json"))),
    }
    status_lines = [
        f"- {'🟢 **LIVE**' if ok else '🟡 **PLACEHOLDER**'} `{name}`"
        for name, ok in artifact_checks.items()
    ]
    st.markdown("\n".join(status_lines))
    if st.button("🔄 Refresh outputs"):
        st.rerun()

    promo_path = O / "promo" / "surgatlas_promo.mp4"
    if promo_path.is_file():
        st.divider()
        st.subheader("Pitch Video (Illustrative + Live Capture)")
        st.video(str(promo_path))
        st.caption("Generated shots are ILLUSTRATIVE; metrics come strictly from real pipeline outputs.")

    st.divider()
    st.subheader("Agent reasoning")
    trace_items = read_trace_lines(limit=25)
    if trace_items:
        for t in trace_items:
            st.markdown(f"**{t['agent']}** {t['msg']}")
    else:
        st.caption("[PLACEHOLDER — No trace yet in out/trace.jsonl. Agent trace streams live as med_school.py, label.py, accept.py, and narrate.py run.]")

tab1, tab2, tab3 = st.tabs(["1 · Any surgery", "2 · Proof", "3 · Watch (Netflix for surgery)"])

# ---------------- Act 1 ----------------
with tab1:
    st.header("Any surgery, zero setup")
    st.info(
        "**Honest claim:** Act 1 (any surgery) has **no ground truth** — this is a zero-setup capability showcase, not accuracy proof. "
        "See Tab 2 (**Proof**) for blind evaluation against held-out CholecSeg8k ground truth."
    )

    atlases = sorted((O / "atlas").glob("*/dataset.json")) if (O / "atlas").exists() else []
    if not atlases:
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/atlas/<procedure>/dataset.json` from `atlas.py`]**\n\n"
            "Run `python atlas.py --procedure \"laparoscopic appendectomy\" --video <URL>` to populate Act 1. "
            "No fake numbers are shown while awaiting run output."
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Human labels used", "0 (by design)")
        p2.metric("Audit confidence [PLACEHOLDER]", "PENDING")
        p3.metric("Timeline phases / events [PLACEHOLDER]", "PENDING")

        st.subheader("Machine-readable timeline [PLACEHOLDER SCHEMA]")
        st.dataframe(
            pd.DataFrame(columns=["name", "start", "end", "evidence", "confidence"]),
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(columns=["time", "event", "evidence", "confidence"]),
            width="stretch",
        )
    else:
        pick = st.selectbox("Procedure", atlases, format_func=lambda p: p.parent.name)
        ds = load_json(pick, {})
        man = load_json(pick.parent / "manual.json", {})
        c1, c2 = st.columns([1, 1])
        with c1:
            if ds.get("source_video"):
                st.video(ds["source_video"])
            st.metric("Human labels used", ds.get("human_labels_used", 0))
            audit_obj = ds.get("audit", {})
            conf_val = audit_obj.get("overall_confidence")
            st.metric("Audit confidence", f"{conf_val:.2f}" if isinstance(conf_val, (int, float)) else "n/a")
        with c2:
            st.subheader("Manual the agent wrote itself")
            for s in man.get("structures", []):
                confused = ", ".join(s.get("confused_with", [])) or "none listed"
                st.markdown(f"**{s.get('name', s.get('key', ''))}**: {s.get('appearance', '')} *(confused with {confused})*")
        st.subheader("Machine-readable timeline")
        tl = ds.get("timeline", {})
        st.dataframe(pd.DataFrame(tl.get("phases", [])), width="stretch")
        st.dataframe(pd.DataFrame(tl.get("events", [])), width="stretch")
        if tl.get("sightings"):
            with st.expander("Structure sightings timeline"):
                st.dataframe(pd.DataFrame(tl["sightings"]), width="stretch")
        if audit_obj.get("problems"):
            st.warning("Flagged by Argon audit: " + "; ".join(audit_obj["problems"]))
        if audit_obj.get("unverified_items"):
            st.caption("Unverified items flagged by Argon: " + ", ".join(audit_obj["unverified_items"]))
        st.download_button("Download dataset.json", pick.read_text(), file_name="dataset.json")

    st.divider()
    st.subheader("Autonomous Medical School: Laparoscopic Cholecystectomy Manual (`out/manual.json`)")
    chole_manual_path = resolve_artifact("manual.json")
    chole_manual = load_json(chole_manual_path)
    if chole_manual:
        mc1, mc2 = st.columns([1, 2])
        with mc1:
            st.markdown(f"**Procedure:** {chole_manual.get('procedure', 'laparoscopic cholecystectomy')}")
            st.markdown("**Surgical Phases:**")
            for ph in chole_manual.get("phases", []):
                st.markdown(f"- {ph}")
            st.markdown("**Critical Safety Events:**")
            for ev in chole_manual.get("critical_events", []):
                st.markdown(f"- {ev}")
            if chole_manual.get("general_rules"):
                st.markdown("**General Visual Rules:**")
                for gr in chole_manual.get("general_rules", []):
                    st.markdown(f"- {gr}")
        with mc2:
            st.markdown("**Self-Learned Visual Cues for the 6 Evaluated Labels:**")
            struct_rows = []
            for s in chole_manual.get("structures", []):
                struct_rows.append({
                    "key": s.get("key", ""),
                    "name": s.get("name", ""),
                    "appearance": s.get("appearance", ""),
                    "where_found": s.get("where_found", ""),
                    "when_visible": s.get("when_visible", ""),
                    "confused_with": ", ".join(s.get("confused_with", [])),
                    "how_to_tell_apart": s.get("how_to_tell_apart", ""),
                })
            st.dataframe(pd.DataFrame(struct_rows), width="stretch")
        notes_path = resolve_artifact("research_notes.md")
        if notes_path.exists():
            with st.expander("View Argon Google Search Research Notes (`out/research_notes.md`)"):
                st.markdown(notes_path.read_text())
    else:
        st.info(
            "**[PLACEHOLDER — Awaiting `out/manual.json` from `med_school.py`]**\n\n"
            "When `med_school.py` completes, the agent's self-written visual manual for all 6 CholecSeg8k structures appears here."
        )
        placeholder_cues = [
            {"key": k, "class_id": cid, "name": name, "visual_criteria": "[PLACEHOLDER — awaiting med_school.py]"}
            for k, (cid, name) in C.LABELS.items()
        ]
        st.dataframe(pd.DataFrame(placeholder_cues), width="stretch")

# ---------------- Act 2 ----------------
with tab2:
    st.header("Proof: CholecSeg8k, labels hidden from the agent")
    st.info(
        "**Honest claims:**\n"
        "- **No human labels used in training.** Ground truth is used **only** to score held-out test videos (`split == 'test'`).\n"
        "- **`calib` split (`upper_bound_calibrated`)** and **`oracle_upper_bound` student** use human labels and are reported strictly as **upper bounds**."
    )

    # Show live upstream progress if index.csv or labels exist
    idx_file = C.DATA_DIR / "index.csv"
    lbl_file = resolve_artifact("labels_manual.jsonl")
    smoke_file = resolve_artifact("labels_smoke.jsonl")
    if idx_file.exists() or lbl_file.exists() or smoke_file.exists():
        prog_cols = st.columns(3)
        if idx_file.exists():
            try:
                idx_df = pd.read_csv(idx_file)
                prog_cols[0].metric("Indexed frames (data/index.csv)", len(idx_df))
                prog_cols[1].metric("Held-out test videos", int(idx_df[idx_df["split"] == "test"]["video"].nunique()))
            except Exception:
                pass
        active_lbl = lbl_file if lbl_file.exists() else smoke_file
        if active_lbl.exists():
            try:
                n_lines = sum(1 for ln in active_lbl.read_text().splitlines() if ln.strip())
                prog_cols[2].metric(f"Label rows ({active_lbl.name})", n_lines)
            except Exception:
                pass

    res = load_json(resolve_artifact("results.json"))
    stu = load_json(resolve_artifact("student.json"))

    if not res:
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/results.json` from `accept.py`]**\n\n"
            "Metrics below are placeholders (`PENDING`) until `accept.py` finishes scoring on held-out test videos. "
            "No numbers are fabricated."
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy on accepted labels [PLACEHOLDER]", "PENDING")
        c2.metric("Labels accepted (coverage) [PLACEHOLDER]", "PENDING")
        c3.metric("Sent to surgeon [PLACEHOLDER]", "PENDING")
        c4.metric("Self-written manual vs blank [PLACEHOLDER]", "PENDING")

        st.subheader("Selective Prediction Comparison [PLACEHOLDER SCHEMA]")
        placeholder_methods = pd.DataFrame([
            {"method": "blank_single_conf (no manual)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "single_conf (Flash + manual)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "majority_vote (multi-sample)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "system_zero_label (Ours: checks + Argon)", "uses_human_labels": "No (0 labels)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
            {"method": "upper_bound_calibrated [UPPER BOUND]", "uses_human_labels": "Yes (calib split GT)", "coverage_at_95": "PENDING", "acc_at_60": "PENDING", "aurc": "PENDING"},
        ])
        st.dataframe(placeholder_methods, width="stretch")
    else:
        a = res.get("accepted", {})
        c1, c2, c3, c4 = st.columns(4)
        acc_val = a.get("accuracy")
        cov_val = a.get("coverage")
        esc_val = a.get("escalated_share")
        c1.metric("Accuracy on accepted labels", f"{acc_val * 100:.1f}%" if isinstance(acc_val, (int, float)) else "n/a")
        c2.metric("Labels accepted (coverage)", f"{cov_val * 100:.0f}%" if isinstance(cov_val, (int, float)) else "n/a")
        c3.metric("Sent to surgeon", f"{esc_val * 100:.0f}%" if isinstance(esc_val, (int, float)) else "n/a")
        if "blank_raw_acc" in res and "manual_raw_acc" in res:
            c4.metric(
                "Self-written manual vs blank",
                f"{res['manual_raw_acc'] * 100:.1f}%",
                f"{(res['manual_raw_acc'] - res['blank_raw_acc']) * 100:+.1f} pts",
            )
        elif "manual_raw_acc" in res:
            c4.metric("Raw accuracy (with self-written manual)", f"{res['manual_raw_acc'] * 100:.1f}%")
        else:
            c4.metric("Held-out test label evaluations", str(res.get("n_test_labels", "n/a")))

        curves = {}
        summary_rows = []
        label_map = {
            "blank_single_conf": "blank_single_conf (no manual)",
            "single_conf": "single_conf (Flash + manual)",
            "majority_vote": "majority_vote",
            "system_zero_label": "system_zero_label (Ours: zero human labels)",
            "upper_bound_calibrated": "upper_bound_calibrated (UPPER BOUND - calib GT)",
        }
        for k, v in res.items():
            if isinstance(v, dict) and "curve" in v:
                display_name = label_map.get(k, k)
                for pt in v["curve"]:
                    if isinstance(pt, (list, tuple)) and len(pt) == 2:
                        cov, acc = pt
                        curves.setdefault(cov, {})[display_name] = acc
                summary_rows.append({
                    "method": display_name,
                    "human_labels_used": "Yes (calib split upper bound)" if "upper_bound" in k else "No (0 human labels)",
                    "coverage_at_95": v.get("coverage_at_95"),
                    "acc_at_60": v.get("acc_at_60"),
                    "aurc (lower=better)": v.get("aurc"),
                })
        if curves:
            st.subheader("Accuracy vs share of labels accepted (higher line = better)")
            st.line_chart(pd.DataFrame(curves).T.sort_index().interpolate(method="index"))
            st.caption("Note: `upper_bound_calibrated` fits a logistic regression on the small `calib` split using human labels and is shown strictly as an upper bound.")
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch")
        if res.get("per_label_accepted_acc"):
            st.subheader("Accepted label accuracy by anatomical structure / instrument")
            per_lbl_s = pd.Series(res["per_label_accepted_acc"], name="Accepted Accuracy")
            st.bar_chart(per_lbl_s)

    st.divider()
    st.subheader("Student trained on the system's own labels (test videos, real GT)")
    if not stu:
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/student.json` from `student.py`]**\n\n"
            "DINOv2 linear probe comparison (`single` vs `vote` vs `verified` vs `oracle_upper_bound`) populates here once `student.py` completes."
        )
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Single pseudo-labels [PLACEHOLDER]", "PENDING")
        sc2.metric("Majority vote [PLACEHOLDER]", "PENDING")
        sc3.metric("Verified only (Ours) [PLACEHOLDER]", "PENDING")
        sc4.metric("Oracle GT (UPPER BOUND) [PLACEHOLDER]", "PENDING")
    else:
        mean_acc = stu.get("mean_acc_by_source", {})
        if mean_acc:
            display_series = {
                ("oracle_upper_bound (UPPER BOUND - human GT)" if k == "oracle_upper_bound" else k): v
                for k, v in mean_acc.items()
            }
            sc_cols = st.columns(len(display_series))
            single_base = mean_acc.get("single")
            for idx_c, (k, v) in enumerate(mean_acc.items()):
                lbl = "oracle_upper_bound (UPPER BOUND)" if k == "oracle_upper_bound" else k
                delta = None
                if k == "verified" and isinstance(single_base, (int, float)) and isinstance(v, (int, float)):
                    delta = f"{(v - single_base) * 100:+.1f} pts vs single"
                sc_cols[idx_c].metric(lbl, f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a", delta=delta)
            st.bar_chart(pd.Series(display_series))
        st.caption("oracle_upper_bound uses human labels on the train split and is shown as an upper bound for reference only.")
        if stu.get("per_label"):
            rows_stu = []
            for key, by_src in stu["per_label"].items():
                if not isinstance(by_src, dict):
                    continue
                for src_name, metrics in by_src.items():
                    if isinstance(metrics, dict):
                        rows_stu.append({
                            "label": key,
                            "source": "oracle_upper_bound (UPPER BOUND)" if src_name == "oracle_upper_bound" else src_name,
                            "n_train": metrics.get("n_train"),
                            "test_acc": metrics.get("acc"),
                            "test_f1": metrics.get("f1"),
                            "test_auroc": metrics.get("auroc"),
                        })
            if rows_stu:
                with st.expander("Per-structure specialist breakdown (test split)"):
                    st.dataframe(pd.DataFrame(rows_stu), width="stretch")

# ---------------- Act 3 ----------------
with tab3:
    st.header("Watch: Grounded Surgical Narration")
    st.markdown(
        "**Verification badge legend:** "
        "<span style='background:#1f9d55;color:white;padding:3px 9px;border-radius:10px;margin-right:6px'>accept (verified — safe to narrate)</span>"
        "<span style='background:#d69e2e;color:white;padding:3px 9px;border-radius:10px;margin-right:6px'>reanalyse (uncertain — must disclose uncertainty)</span>"
        "<span style='background:#e53e3e;color:white;padding:3px 9px;border-radius:10px'>escalate (sent to surgeon — do not claim)</span>",
        unsafe_allow_html=True,
    )

    dec_path = resolve_artifact("decisions.csv")
    idx_path = C.DATA_DIR / "index.csv"
    if not dec_path.exists():
        st.warning(
            "**[PLACEHOLDER — Awaiting `out/decisions.csv` from `accept.py` and `out/narration_<video>.json` from `narrate.py`]**\n\n"
            "Once `accept.py` finishes, you can scrub or play held-out test videos with verified bounding boxes, status chips, "
            "ground-truth reveal, and grounded commentary."
        )
        if idx_path.exists():
            try:
                idx_preview = pd.read_csv(idx_path)
                test_preview = idx_preview[idx_preview["split"] == "test"] if "split" in idx_preview.columns else idx_preview
                if not test_preview.empty:
                    st.info("Previewing unlabelled held-out test frames from `data/index.csv` while awaiting `out/decisions.csv`:")
                    vids_p = sorted(test_preview["video"].unique())
                    v_p = st.selectbox("Test video (raw preview)", vids_p, key="raw_preview_vid")
                    frames_p = sorted(test_preview[test_preview["video"] == v_p]["frame_id"].unique(), key=frame_sort_key)
                    i_p = st.slider("Frame (raw preview)", 0, len(frames_p) - 1, 0, key="raw_preview_slider") if len(frames_p) > 1 else 0
                    f_p = frames_p[i_p]
                    row_p = test_preview[test_preview["frame_id"] == f_p].iloc[0]
                    st.image(draw(row_p["path"], pd.DataFrame(), False, frame_id=f_p), width="stretch")
            except Exception:
                pass
        else:
            st.image(draw(None, pd.DataFrame(), False, frame_id="placeholder_preview"), width="stretch")
    else:
        dec = pd.read_csv(dec_path)
        vid_col = "video" if "video" in dec.columns else "video_id"
        idx = pd.read_csv(idx_path).set_index("frame_id") if idx_path.exists() else None

        test_dec = dec[dec["split"] == "test"] if "split" in dec.columns else dec
        if test_dec.empty:
            test_dec = dec
        vids = sorted(test_dec[vid_col].dropna().unique())
        v = st.selectbox("Test video", vids)

        narr_list = load_json(resolve_artifact(f"narration_{v}.json"), [])
        narr = {x["frame_id"]: x for x in narr_list if isinstance(x, dict) and "frame_id" in x}

        frames = sorted(dec[dec[vid_col] == v]["frame_id"].unique(), key=frame_sort_key)
        reveal = st.toggle("Reveal ground truth")
        play = st.button("▶ Play")
        if len(frames) > 1:
            i = st.slider("Frame", 0, len(frames) - 1, 0)
        else:
            i = 0
            if frames:
                st.caption(f"Showing single sampled frame: `{frames[0]}`")
        slot_img, slot_chips, slot_narr = st.empty(), st.empty(), st.empty()

        def lookup_frame_path(f_id: str, rows_f: pd.DataFrame):
            if idx is not None and f_id in idx.index:
                val = idx.loc[f_id, "path"]
                return val.iloc[0] if isinstance(val, pd.Series) else val
            if "frame_path" in rows_f.columns:
                fp = rows_f["frame_path"].dropna()
                if not fp.empty:
                    return fp.iloc[0]
            return None

        def show(f):
            rows = dec[dec["frame_id"] == f]
            f_path = lookup_frame_path(f, rows)
            slot_img.image(draw(f_path, rows, reveal, frame_id=f), width="stretch")
            chips = []
            for r in rows.itertuples():
                status_val = getattr(r, "status", "reanalyse")
                colour = COL.get(status_val, "#d69e2e")
                final_pres = bool(getattr(r, "final_present", getattr(r, "present", 0)))
                label = f"{r.key}: {'yes' if final_pres else 'no'} ({status_val})"
                if reveal and hasattr(r, "final_correct"):
                    label += " ✓" if r.final_correct else " ✗"
                chips.append(
                    f"<span style='background:{colour};color:white;padding:4px 10px;"
                    f"border-radius:12px;margin:3px;display:inline-block'>{label}</span>"
                )
            slot_chips.markdown(" ".join(chips), unsafe_allow_html=True)
            if f in narr:
                n_obj = narr[f]
                unsure_items = n_obj.get("unsure_about", [])
                unsure_txt = (
                    f"\n\n*Disclosed uncertainty (not claimed as fact): {', '.join(unsure_items)}*"
                    if unsure_items else ""
                )
                slot_narr.markdown(f"### {n_obj.get('narration', '')}{unsure_txt}")
            else:
                acc_present = [r.key for r in rows.itertuples() if getattr(r, "status", "") == "accept" and getattr(r, "final_present", 0)]
                acc_absent = [r.key for r in rows.itertuples() if getattr(r, "status", "") == "accept" and not getattr(r, "final_present", 0)]
                unsure_keys = [r.key for r in rows.itertuples() if getattr(r, "status", "") != "accept"]
                slot_narr.info(
                    f"**[PLACEHOLDER — Run `python narrate.py --video {v}` for pre-generated voiceover]**\n\n"
                    f"- **Verified present (`accept`):** {', '.join(acc_present) or 'none'}\n"
                    f"- **Verified absent (`accept`):** {', '.join(acc_absent) or 'none'}\n"
                    f"- **Uncertain / withheld (`reanalyse` / `escalate`):** {', '.join(unsure_keys) or 'none'}"
                )

        if play:
            for f in frames[i:]:
                show(f)
                time.sleep(1.2)
        elif frames:
            show(frames[i])
            with st.expander("Frame verification audit details (`decisions.csv`)"):
                cols_to_show = [
                    c for c in ["key", "present", "conf", "pv_match", "pv_conf", "p_wrong", "trust", "status", "final_present", "final_by", "evidence", "critic"]
                    if c in dec.columns
                ]
                st.dataframe(dec[dec["frame_id"] == frames[i]][cols_to_show], width="stretch")
