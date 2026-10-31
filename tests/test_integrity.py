"""Pytest wrapper for lanes/integrity.py fabrication checker."""
import subprocess
import sys
from pathlib import Path

import config as C


def test_fabrication_integrity_checker():
    script = Path(C.ROOT) / "lanes" / "integrity.py"
    res = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"lanes/integrity.py failed:\n{res.stdout}\n{res.stderr}"
