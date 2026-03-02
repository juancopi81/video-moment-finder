#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATIONS_DIR="$ROOT_DIR/supabase/migrations"
MIGRATIONS_TABLE="public.schema_migrations"
BASELINE_EXISTING_DB=false

usage() {
  cat <<'EOF'
Usage: ./scripts/setup_local.sh [--baseline-existing-db]

Options:
  --baseline-existing-db   Mark current migration files as applied without replaying SQL.
                           Use this once on an existing DB that was already migrated before
                           migration tracking was introduced.
EOF
}

while (( "$#" )); do
  case "$1" in
    --baseline-existing-db)
      BASELINE_EXISTING_DB=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

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
mapfile -t migration_files < <(printf '%s\n' "${migration_files[@]}" | sort)

psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 <<SQL
create table if not exists $MIGRATIONS_TABLE (
  filename text primary key,
  applied_at timestamptz not null default now()
);
SQL

applied_count="$(
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -At -c "select count(*) from $MIGRATIONS_TABLE"
)"
has_core_schema="$(
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -At -c "
    select exists (
      select 1
      from information_schema.tables
      where table_schema = 'public'
        and table_name in ('videos', 'credits', 'video_jobs')
    );
  "
)"

if [[ "$applied_count" == "0" && "$has_core_schema" == "t" && "$BASELINE_EXISTING_DB" != "true" ]]; then
  echo "Detected an existing DB schema but no migration history in $MIGRATIONS_TABLE."
  echo "To avoid replaying old migrations on populated data, run once with:"
  echo "  ./scripts/setup_local.sh --baseline-existing-db"
  exit 1
fi

if [[ "$BASELINE_EXISTING_DB" == "true" && "$applied_count" == "0" ]]; then
  echo "Baselining existing DB: marking current migration files as applied..."
  for file in "${migration_files[@]}"; do
    filename="$(basename "$file")"
    escaped_filename="${filename//\'/\'\'}"
    psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 \
      -c "insert into $MIGRATIONS_TABLE (filename) values ('$escaped_filename') on conflict (filename) do nothing;"
  done
fi

for file in "${migration_files[@]}"; do
  filename="$(basename "$file")"
  escaped_filename="${filename//\'/\'\'}"
  already_applied="$(
    psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -At \
      -c "select 1 from $MIGRATIONS_TABLE where filename = '$escaped_filename' limit 1"
  )"

  if [[ "$already_applied" == "1" ]]; then
    echo "  -> $filename (already applied)"
    continue
  fi

  echo "  -> $filename"
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f "$file"
  psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 \
    -c "insert into $MIGRATIONS_TABLE (filename) values ('$escaped_filename') on conflict (filename) do nothing;"
done

echo "Installing backend dependencies..."
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" uv sync --frozen --group dev

echo "Installing frontend dependencies..."
(
  cd "$ROOT_DIR/frontend"
  npm ci
)

echo "Local setup complete."
