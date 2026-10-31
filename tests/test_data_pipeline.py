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

