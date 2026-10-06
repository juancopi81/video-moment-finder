#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "$ROOT_DIR"
echo "[check_all] Running backend tests..."
uv run --frozen --no-sync pytest -q

echo "[check_all] Checking portable plugin package..."
uv run --frozen --no-sync python -m unittest discover -s scripts/plugin -p 'test_*.py'
uv run --frozen --no-sync python scripts/plugin/build_package.py

echo "[check_all] Running frontend lint and build..."
(
  cd "$ROOT_DIR/frontend"
  npm run lint
  node tests/account-and-source-links.test.mjs
  npm run build
)

echo "[check_all] All checks passed."
