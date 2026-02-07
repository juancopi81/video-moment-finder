#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
echo "[check_all] Running backend tests..."
.venv/bin/pytest -q

echo "[check_all] Running frontend lint and build..."
(
  cd "$ROOT_DIR/frontend"
  npm run lint
  npm run build
)

echo "[check_all] All checks passed."
