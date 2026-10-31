#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PORT="${SURGPROOF_PORT:-${PORT:-8600}}"
HOST="${HOST:-127.0.0.1}"

# On a fresh clone (where out/ is not a worktree symlink), materialize out/ from committed demo bundle
python3 - << 'PYEOF'
import pathlib, shutil

root = pathlib.Path(".")
out = root / "out"
coord = root / "coord" / "artifacts"
wdata = root / "web" / "data"

if not out.is_symlink():
    out.mkdir(parents=True, exist_ok=True)
    if coord.is_dir():
        for f in coord.iterdir():
            if f.is_file() and f.name != ".gitkeep" and not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
    if wdata.is_dir():
        for f in wdata.glob("*.json"):
            if f.is_file() and not (out / f.name).exists():
                shutil.copy2(f, out / f.name)
        if (wdata / "dense").is_dir() and not (out / "dense").exists():
            shutil.copytree(wdata / "dense", out / "dense", dirs_exist_ok=True)
        for mdir in ("masks", "masks_r1", "masks_r2", "masks_vote3"):
            if (wdata / mdir).is_dir() and not (out / mdir).exists():
                shutil.copytree(wdata / mdir, out / mdir, dirs_exist_ok=True)
PYEOF

if [ -f "web/export.py" ] && [ -f "out/annotations.jsonl" ]; then
  python3 web/export.py >/dev/null 2>&1 || true
fi

if [ "${1:-}" = "--check" ]; then
  python3 -m pytest -q tests/test_e2e_app.py
  exit 0
fi

if python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
  exec python3 -m uvicorn web.server:app --host "$HOST" --port "$PORT"
else
  exec python3 web/serve.py --host "$HOST" --port "$PORT"
fi
