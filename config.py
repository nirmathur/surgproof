"""Central config. Check model IDs against your Argon builder guide before running."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DIR = ROOT / "raw"                 # where the HF zip lands / is extracted
DATA_DIR = ROOT / "data"               # index csv, features, splits
CACHE_DB = ROOT / "cache.sqlite"       # every Gemini response is cached here
OUT_DIR = ROOT / "out"                 # labels, results, manual, narration, exports
for d in (DATA_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

# ---- Models (verify IDs in the builder guide) ----
FLASH = os.environ.get("FLASH_MODEL", "gemini-3.8-flash")   # bulk labeller / verifier
ARGON = os.environ.get("ARGON_MODEL", "gemini-4-argon")     # planner, reviser, manual writer
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))        # parallel Gemini calls

# ---- Labels we evaluate on (CholecSeg8k class IDs from the paper, Table I) ----
# key -> (class_id, human name)
LABELS = {
    "grasper":     (5,  "grasper (surgical instrument)"),
    "l_hook":      (9,  "L-hook electrocautery (instrument)"),
    "liver":       (2,  "liver"),
    "gallbladder": (10, "gallbladder"),
    "cystic_duct": (8,  "cystic duct"),
    "blood":       (7,  "blood"),
}

# Watershed mask pixel values. The paper says pixel value == class id, but the released
# masks are widely reported to use the values below. prepare_data.py auto-detects which
# convention your copy uses; run `python prepare_data.py --inspect` to confirm by eye.
WATERSHED_VALUE_TO_CLASS = {
    50: 0, 11: 1, 21: 2, 13: 3, 12: 4, 31: 5, 23: 6,
    24: 7, 25: 8, 32: 9, 22: 10, 33: 11, 5: 12,
}

# A class counts as "present" if it covers at least this fraction of the frame.
MIN_FRAC = {"default": 0.002, "blood": 0.001, "cystic_duct": 0.001}

# ---- Sampling / splits ----
FRAME_STRIDE = 8          # every 8th frame of each 80-frame clip -> 10 per clip, ~1,010 total
N_TEST_VIDEOS = 6         # held out: final evaluation only
N_CALIB_VIDEOS = 3        # optional small labelled calibration set (upper-bound mode only)
SEED = 0

# ---- Acceptance (zero-label rule) ----
ACCEPT_CONF = 0.85        # labeller confidence needed before checks are even considered
