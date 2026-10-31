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

if [ -f "web/export.py" ] && [ -d "out" ]; then
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
