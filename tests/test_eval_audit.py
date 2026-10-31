"""Unit and integration tests for independent evaluation and leakage audit (eval/audit.py)."""
import json

import numpy as np
import pandas as pd

import config as C
from eval import audit


def test_audit_splits_pass_and_fail():
    df_ok = pd.DataFrame([
        {"frame_id": "v1_1", "video": "v1", "split": "train"},
        {"frame_id": "v2_1", "video": "v2", "split": "calib"},
        {"frame_id": "v3_1", "video": "v3", "split": "test"},
    ])
    res_ok = audit.audit_splits(df_ok, None)
    assert res_ok["status"] == "PASS"

    df_leak = pd.DataFrame([
        {"frame_id": "v1_1", "video": "v1", "split": "train"},
        {"frame_id": "v1_2", "video": "v1", "split": "test"},
    ])
    res_leak = audit.audit_splits(df_leak, None)
    assert res_leak["status"] == "FAIL"
    assert "v1" in res_leak["video_overlaps"]["train_test"]


def test_audit_prevalence_flags_liver():
    rows = []
    for i in range(20):
        r = {f"gt_{k}": i % 2 for k in C.LABELS}
        r["gt_liver"] = 1  # 100% present
        rows.append(r)
    idx = pd.DataFrame(rows)
    res = audit.audit_prevalence_and_liver(idx, None)
    assert res["status"] == "WARN"
    assert "liver" in res["degenerate_labels"]
    assert "liver" not in res["headline_labels"]


def test_audit_verifier_leakage_detects_truth_leak():
    # Clean zero-label trust
    clean_dec = pd.DataFrame([
        {"frame_id": f"f{i}", "key": "grasper", "present": 1, "conf": 0.9, "p_wrong": 0.1,
         "pv_match": True, "trust": 0.81, "truth": i % 2, "final_by": "flash"}
        for i in range(20)
    ])
    res_clean = audit.audit_verifier_leakage(clean_dec)
    assert res_clean["status"] == "PASS"
    assert res_clean["zero_human_label_verified"] is True

    # Leaked trust that directly boosts trust when truth == 1
    leaked_dec = clean_dec.copy()
    leaked_dec["trust"] = np.where(leaked_dec.truth == 1, 0.99, 0.20)
    res_leaked = audit.audit_verifier_leakage(leaked_dec)
    assert res_leaked["status"] == "FAIL"
    assert res_leaked["zero_human_label_verified"] is False


def test_video_cluster_bootstrap_ci_and_run_audit(tmp_path):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    coord_dir = tmp_path / "coord"
    for d in (data_dir, out_dir, coord_dir):
        d.mkdir()

    # Build synthetic index.csv, decisions.csv, student.json, student_probs.csv
    idx_rows, dec_rows, prob_rows = [], [], []
    for v_i, (sp, vid) in enumerate([
        ("train", "video01"), ("train", "video02"),
        ("calib", "video03"),
        ("test", "video04"), ("test", "video05"), ("test", "video06"),
    ]):
        for f_j in range(6):
            fid = f"{vid}_00080_{f_j}"
            ir = {"frame_id": fid, "video": vid, "split": sp, "path": f"/tmp/{fid}.png"}
            pr = {"frame_id": fid}
            for k_i, k in enumerate(C.LABELS):
                gt = 1 if k == "liver" else int((v_i + f_j + k_i) % 2 == 0)
                ir[f"gt_{k}"] = gt
                is_err = (f_j == 0)
                pred = (1 - gt) if is_err else gt
                status = "escalate" if is_err else "accept"
                dec_rows.append({
                    "frame_id": fid, "video": vid, "split": sp, "key": k,
                    "present": pred, "conf": 0.5 if is_err else 0.9,
                    "p_wrong": 0.6 if is_err else 0.05, "pv_match": not is_err,
                    "trust": 0.1 if is_err else 0.855, "status": status,
                    "final_present": gt if status == "accept" else pred,
                    "final_by": "flash", "truth": gt,
                    "correct": int(pred == gt), "final_correct": int((gt if status == "accept" else pred) == gt),
                })
                if sp == "test":
                    pr[f"{k}__single"] = 0.8 if pred == 1 else 0.2
                    pr[f"{k}__verified"] = 0.9 if gt == 1 else 0.1
                    pr[f"{k}__oracle_upper_bound"] = 0.95 if gt == 1 else 0.05
            idx_rows.append(ir)
            if sp == "test":
                prob_rows.append(pr)

    pd.DataFrame(idx_rows).to_csv(data_dir / "index.csv", index=False)
    pd.DataFrame(dec_rows).to_csv(out_dir / "decisions.csv", index=False)
    pd.DataFrame(prob_rows).to_csv(out_dir / "student_probs.csv", index=False)
    (out_dir / "student.json").write_text(json.dumps({
        "mean_acc_by_source": {"single": 0.8333, "verified": 1.0, "oracle_upper_bound": 1.0},
        "per_label": {},
    }))
    (out_dir / "manual.json").write_text(json.dumps({"procedure": "lap chole", "source": "https://youtube.com/watch?v=abc"}))

    res = audit.run_audit(data_dir=data_dir, out_dir=out_dir, coord_dir=coord_dir)
    assert (out_dir / "eval_audit.json").exists()
    assert (out_dir / "eval_report.md").exists()
    assert res["split_audit"]["status"] == "PASS"
    assert "liver" in res["prevalence_audit"]["degenerate_labels"]
    v5 = res["verifier_test_evaluation"]["headline_5_labels"]
    assert v5["accepted_acc"]["point"] > v5["single_raw_acc"]["point"]
    s5 = res["student_test_evaluation"]["video_bootstrap_acc_5_labels"]
    assert s5["verified"]["point"] > s5["single"]["point"]
    assert s5["oracle_upper_bound"]["is_upper_bound"] is True


def test_attach_ground_truth_ignores_unscorable_minus_one():
    idx = pd.DataFrame([
        {
            "frame_id": "ext_1", "video": "ev1", "split": "test",
            "gt_grasper": 1, "gt_l_hook": 0, "gt_liver": -1,
            "gt_gallbladder": -1, "gt_cystic_duct": -1, "gt_blood": -1,
        }
    ])
    lbl = pd.DataFrame([
        {"frame_id": "ext_1", "key": "grasper", "present": 1},
        {"frame_id": "ext_1", "key": "l_hook", "present": 0},
        {"frame_id": "ext_1", "key": "liver", "present": 1},
    ])
    joined = audit.attach_ground_truth(lbl, idx)
    assert set(joined["key"]) == {"grasper", "l_hook"}
    assert (joined["correct"] == 1).all()
