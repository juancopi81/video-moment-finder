#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATIONS_DIR="$ROOT_DIR/supabase/migrations"

if [[ -z "${SUPABASE_DB_URL:-}" ]]; then
  echo "SUPABASE_DB_URL is required."
  echo "Add it to your .env and export it before running this script."
  exit 1
fi

if ! command -v psql >/dev/null 2>&1; then
  echo "psql is required but not installed."
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed."
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "npm is required but not installed."
  exit 1
fi

echo "Applying Supabase migrations..."
shopt -s nullglob
migration_files=("$MIGRATIONS_DIR"/*.sql)
if (( ${#migration_files[@]} == 0 )); then
  echo "No migration files found in $MIGRATIONS_DIR"
  exit 1
fi

for file in "${migration_files[@]}"; do
  echo "  -> $(basename "$file")"
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f "$file"
done

echo "Installing backend dependencies..."
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" uv sync --frozen --group dev

echo "Installing frontend dependencies..."
(
  cd "$ROOT_DIR/frontend"
  npm ci
)

echo "Local setup complete."
