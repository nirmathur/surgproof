"""Unit and integration tests for student.py and mock_test.py."""
import json

import numpy as np
import pandas as pd

import config as C
import student


def test_fit_eval_normal_and_upper_bound():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(size=(40, 16))
    y_tr = (X_tr[:, 0] > 0).astype(int)
    X_te = rng.normal(size=(20, 16))
    y_te = (X_te[:, 0] > 0).astype(int)

    res_zero, p_zero = student.fit_eval(X_tr, y_tr, X_te, y_te, is_upper_bound=False)
    assert res_zero is not None and p_zero is not None
    assert res_zero["n_train"] == 40
    assert res_zero["is_upper_bound"] is False
    assert res_zero["acc"] >= 0.75
    assert "f1" in res_zero and "auroc" in res_zero

    res_ub, _ = student.fit_eval(X_tr, y_tr, X_te, y_te, is_upper_bound=True)
    assert res_ub["is_upper_bound"] is True


def test_fit_eval_single_class_and_empty():
    X_tr = np.ones((10, 8))
    y_tr_one = np.zeros(10, dtype=int)
    X_te = np.ones((6, 8))
    y_te = np.array([0, 0, 0, 1, 1, 0])

    res, p = student.fit_eval(X_tr, y_tr_one, X_te, y_te)
    assert res is not None
    assert res.get("single_class_fallback") is True
    assert np.allclose(p, 0.0)
    assert res["acc"] == round(4 / 6, 4)

    res_empty, p_empty = student.fit_eval(np.empty((0, 8)), np.array([]), X_te, y_te)
    assert res_empty is None and p_empty is None


def test_student_main_strict_splits_and_upper_bound_marking(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "out"
    data_dir.mkdir()
    out_dir.mkdir()
    monkeypatch.setattr(C, "DATA_DIR", data_dir)
    monkeypatch.setattr(C, "OUT_DIR", out_dir)

    rng = np.random.default_rng(7)
    rows = []
    # 30 train, 10 calib, 20 test frames across distinct videos
    splits = [("train", f"video{i:02d}", 10) for i in range(1, 4)] + [
        ("calib", "video04", 10),
        ("test", "video05", 10),
        ("test", "video06", 10),
    ]
    fid_counter = 0
    for sp, vid, count in splits:
        for j in range(count):
            fid_counter += 1
            r = {"frame_id": f"{vid}_00080_{fid_counter}", "video": vid, "split": sp, "path": f"/tmp/{fid_counter}.png"}
            for k_idx, k in enumerate(C.LABELS):
                r[f"gt_{k}"] = int((fid_counter + k_idx) % 2 == 0)
            rows.append(r)
    idx = pd.DataFrame(rows)
    idx.to_csv(data_dir / "index.csv", index=False)

    # Features with strong signal for ground truth
    X = rng.normal(0, 0.3, size=(len(idx), 64)).astype(np.float32)
    for k_idx, k in enumerate(C.LABELS):
        sign = idx[f"gt_{k}"].values * 2 - 1
        X[:, k_idx * 8:(k_idx + 1) * 8] += sign[:, None] * 1.5
    np.save(data_dir / "features.npy", X)

    # Build decisions.csv where single has systematic false-positive noise on train, verified filters it out
    dec_rows = []
    for i_r, r in enumerate(idx.itertuples()):
        for k_idx, k in enumerate(C.LABELS):
            truth = int(getattr(r, f"gt_{k}"))
            is_noisy = ((i_r + k_idx) % 5) < 3  # 60% error rate in single on train
            single_pred = (1 - truth) if is_noisy else truth
            status = "escalate" if is_noisy else "accept"
            dec_rows.append({
                "frame_id": r.frame_id,
                "video": r.video,
                "split": r.split,
                "key": k,
                "present": single_pred,
                "final_present": truth if status == "accept" else single_pred,
                "status": status,
                "truth": truth,
            })
    pd.DataFrame(dec_rows).to_csv(out_dir / "decisions.csv", index=False)

    out = student.main()
    assert (out_dir / "student.json").exists()
    assert (out_dir / "student_probs.csv").exists()

    # Verify upper bound metadata and ordering
    assert "oracle_upper_bound" in out["upper_bounds"]
    assert out["source_metadata"]["oracle_upper_bound"]["is_upper_bound"] is True
    assert out["source_metadata"]["oracle_upper_bound"]["uses_human_labels"] is True
    assert out["source_metadata"]["verified"]["is_upper_bound"] is False
    assert out["source_metadata"]["verified"]["uses_human_labels"] is False

    # Verified should beat single and use fewer training labels
    assert out["mean_acc_by_source"]["verified"] > out["mean_acc_by_source"]["single"]
    assert out["mean_n_train_by_source"]["verified"] < out["mean_n_train_by_source"]["single"]

    probs = pd.read_csv(out_dir / "student_probs.csv")
    test_fids = set(idx[idx.split == "test"].frame_id)
    assert set(probs.frame_id) == test_fids
