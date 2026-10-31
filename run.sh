#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CLI_PORT="${PORT:-}"
CLI_HOST="${HOST:-}"

if [ -f ".env" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
fi

PORT="${CLI_PORT:-${SURGPROOF_PORT:-${PORT:-8600}}}"
HOST="${CLI_HOST:-${HOST:-127.0.0.1}}"

# On a fresh clone (where out/ is not a worktree symlink), materialize out/ instantly from committed demo bundle
python3 - << 'PYEOF'
import os, pathlib, shutil

root = pathlib.Path(".").resolve()
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
        for sub in ("dense", "masks", "masks_r1", "masks_r2", "masks_vote3"):
            src_sub = wdata / sub
            dst_sub = out / sub
            if src_sub.is_dir() and not dst_sub.exists() and not dst_sub.is_symlink():
                try:
                    os.symlink(src_sub, dst_sub)
                except Exception:
                    shutil.copytree(src_sub, dst_sub, dirs_exist_ok=True)
PYEOF

if [ -f "web/export.py" ] && [ -f "out/annotations.jsonl" ]; then
  python3 web/export.py >/dev/null 2>&1 || true
fi

if [ "${1:-}" = "--check" ]; then
  python3 -m pytest -q tests/test_e2e_app.py
  exit 0
fi

# Free stale listener on target port if occupied so fresh-clone 'make demo' binds cleanly
if command -v lsof >/dev/null 2>&1; then
  STALE_PIDS="$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$STALE_PIDS" ]; then
    kill $STALE_PIDS 2>/dev/null || true
    sleep 0.3
  fi
fi

if python3 -c "import uvicorn, fastapi" >/dev/null 2>&1; then
  exec python3 -m uvicorn web.server:app --host "$HOST" --port "$PORT"
else
  exec python3 web/serve.py --host "$HOST" --port "$PORT"
fi
