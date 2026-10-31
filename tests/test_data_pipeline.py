"""Unit and integration tests for prepare_data.py and features.py (lane/data)."""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import config as C
import prepare_data


def test_mask_to_classes_watershed_with_255_border(tmp_path):
    """Watershed masks in CholecSeg8k contain 255 border pixels and rare 0 pixels.
    Verify mask_to_classes still detects WATERSHED_VALUE_TO_CLASS and maps accurately.
    """
    arr = np.full((20, 20, 3), 50, dtype=np.uint8)  # 50 -> class 0 (background)
    arr[0:5, 0:5] = 21    # 21 -> class 2 (liver)
    arr[5:10, 5:10] = 31  # 31 -> class 5 (grasper)
    arr[10:15, 0:5] = 22  # 22 -> class 10 (gallbladder)
    arr[19, :] = 255      # 255 -> -1 (watershed border ignore)
    arr[0, 19] = 0        # 0 -> -1 (unseeded ignore)
    p = tmp_path / "frame_100_endo_watershed_mask.png"
    Image.fromarray(arr).save(p)

    cls = prepare_data.mask_to_classes(p)
    assert cls.shape == (20, 20)
    assert (cls[0:5, 0:5] == 2).all()
    assert (cls[5:10, 5:10] == 5).all()
    assert (cls[10:15, 0:5] == 10).all()
    assert (cls[19, :] == -1).all()
    assert cls[0, 19] == -1


def test_mask_to_classes_paper_convention(tmp_path):
    """Paper-convention masks (values 0..12 directly) should stay 0..12."""
    arr = np.zeros((20, 20), dtype=np.uint8)
    arr[0:5, 0:5] = 2   # liver
    arr[5:10, 5:10] = 5  # grasper
    arr[10:15, 0:5] = 10  # gallbladder
    p = tmp_path / "paper_mask.png"
    Image.fromarray(arr).save(p)

    cls = prepare_data.mask_to_classes(p)
    assert (cls[0:5, 0:5] == 2).all()
    assert (cls[5:10, 5:10] == 5).all()
    assert (cls[10:15, 0:5] == 10).all()


def test_presence_thresholds():
    """Verify presence() computes exact fractions and applies MIN_FRAC thresholds."""
    cls = np.zeros((100, 100), dtype=np.int16)  # 10,000 pixels
    # grasper (class 5): default threshold 0.002 -> 20 pixels
    cls.flat[:20] = 5
    # blood (class 7): threshold 0.001 -> 10 pixels; give 9 pixels (below threshold)
    cls.flat[20:29] = 7
    # cystic_duct (class 8): threshold 0.001 -> 10 pixels; give 10 pixels (at threshold)
    cls.flat[30:40] = 8

    out = prepare_data.presence(cls)
    assert out["gt_grasper"] == 1
    assert out["frac_grasper"] == 0.002
    assert out["gt_blood"] == 0
    assert out["frac_blood"] == 0.0009
    assert out["gt_cystic_duct"] == 1
    assert out["frac_cystic_duct"] == 0.001


def test_index_csv_contract_if_built():
    """Verify built data/index.csv satisfies video-level split disjointness and schema."""
    idx_path = C.DATA_DIR / "index.csv"
    if not idx_path.exists():
        return
    df = pd.read_csv(idx_path)
    assert len(df) == 1010
    assert df["video"].nunique() == 17
    # Video-level splits must be strictly disjoint
    vids_by_split = {
        s: set(df[df.split == s]["video"].unique())
        for s in ["train", "calib", "test"]
    }
    assert len(vids_by_split["test"]) == C.N_TEST_VIDEOS
    assert len(vids_by_split["calib"]) == C.N_CALIB_VIDEOS
    assert not (vids_by_split["train"] & vids_by_split["test"])
    assert not (vids_by_split["train"] & vids_by_split["calib"])
    assert not (vids_by_split["calib"] & vids_by_split["test"])
    for k in C.LABELS:
        assert f"gt_{k}" in df.columns
        assert f"frac_{k}" in df.columns
        assert set(df[f"gt_{k}"].unique()) <= {0, 1}


def test_features_npy_contract_if_built():
    """Verify built data/features.npy is aligned with data/index.csv and finite."""
    idx_path = C.DATA_DIR / "index.csv"
    feat_path = C.DATA_DIR / "features.npy"
    if not idx_path.exists() or not feat_path.exists():
        return
    df = pd.read_csv(idx_path)
    X = np.load(feat_path)
    assert X.shape == (len(df), 768)
    assert X.dtype == np.float32
    assert np.isfinite(X).all()
    assert float(X.std()) > 0.01


def test_endoscapes_presence_mapping():
    """Endoscapes2023 maps class 5 -> gallbladder, class 4 -> cystic_duct, class 6 -> tool_any,
    and sets unscorable keys (grasper, l_hook, liver, blood) to -1.
    """
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask.flat[:30] = 5   # gallbladder (0.003 >= 0.002)
    mask.flat[30:45] = 4  # cystic_duct (0.0015 >= 0.001)
    mask.flat[45:70] = 6  # tool (0.0025 >= 0.002)
    out = prepare_data.endoscapes_presence(mask)
    assert out["gt_gallbladder"] == 1
    assert out["gt_cystic_duct"] == 1
    assert out["gt_tool_any"] == 1
    for k in ("grasper", "l_hook", "liver", "blood"):
        assert out[f"gt_{k}"] is None or pd.isna(out[f"gt_{k}"])
        assert out[f"frac_{k}"] is None or pd.isna(out[f"frac_{k}"])


def test_cholect50_presence_mapping():
    """CholecT50 maps instrument presence ('grasper', 'hook' -> l_hook) and sets anatomy keys to None/NaN."""
    out = prepare_data.cholect50_presence(["grasper", "hook", "bipolar"])
    assert out["gt_grasper"] == 1
    assert out["gt_l_hook"] == 1
    for k in ("liver", "gallbladder", "cystic_duct", "blood"):
        assert out[f"gt_{k}"] is None or pd.isna(out[f"gt_{k}"])

    out_empty = prepare_data.cholect50_presence(["irrigator"])
    assert out_empty["gt_grasper"] == 0
    assert out_empty["gt_l_hook"] == 0


def test_external_datasets_contract_if_built():
    """Verify external held-out datasets (Endoscapes2023, CholecT50) schema & zero video leakage."""
    endo_dir = C.DATA_DIR / "external" / "endoscapes2023"
    if (endo_dir / "index.csv").exists():
        df_e = pd.read_csv(endo_dir / "index.csv")
        assert len(df_e) == 471
        assert df_e["video"].nunique() == 50
        assert (df_e.split == "test").sum() == 59
        assert set(df_e["gt_gallbladder"].unique()) <= {0, 1}
        assert set(df_e["gt_cystic_duct"].unique()) <= {0, 1}
        assert df_e["gt_grasper"].isna().all()
        for p in df_e.loc[df_e.split == "test", "path"]:
            arr = np.array(Image.open(p))
            assert float(arr.mean()) >= 10.0 and float(arr.std()) >= 5.0
        if (endo_dir / "features.npy").exists():
            Xe = np.load(endo_dir / "features.npy")
            assert Xe.shape == (len(df_e), 768)
            assert np.isfinite(Xe).all()

    c50_dir = C.DATA_DIR / "external" / "cholect50"
    if (c50_dir / "index.csv").exists():
        df_c = pd.read_csv(c50_dir / "index.csv")
        assert len(df_c) >= 50
        # Zero overlap with any CholecSeg8k video
        for v in df_c["video"].unique():
            vid_num = int(v.split("vid")[-1])
            assert f"video{vid_num:02d}" not in prepare_data.CHOLECSEG8K_VIDEOS
        assert set(df_c["gt_grasper"].unique()) <= {0, 1}
        assert set(df_c["gt_l_hook"].unique()) <= {0, 1}
        assert df_c["gt_liver"].isna().all()
        if (c50_dir / "features.npy").exists():
            Xc = np.load(c50_dir / "features.npy")
            assert Xc.shape == (len(df_c), 768)
            assert np.isfinite(Xc).all()


