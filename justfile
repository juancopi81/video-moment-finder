set shell := ["bash", "-cu"]

default:
    @just --list

# Start local Supabase and Qdrant for development.
dev-services:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v supabase >/dev/null 2>&1; then
        echo "supabase CLI is required: brew install supabase/tap/supabase"
        exit 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "docker is required for local Supabase and Qdrant"
        exit 1
    fi
    echo "Starting local Supabase (DB + Auth + API)..."
    supabase start
    echo ""
    echo "Starting local Qdrant on port 6333..."
    if docker ps -q -f name='^qdrant-dev$' | grep -q .; then
        echo "  qdrant-dev container already running"
    elif docker ps -aq -f name='^qdrant-dev$' | grep -q .; then
        docker start qdrant-dev
        echo "  qdrant-dev container restarted"
    else
        docker run -d --name qdrant-dev -p 6333:6333 -p 6334:6334 \
            -v qdrant-dev-data:/qdrant/storage \
            qdrant/qdrant:latest
        echo "  qdrant-dev container started"
    fi
    if [[ ! -f .env.local ]]; then
        echo ""
        echo "Copy .env.local.example to .env.local:"
        echo "  cp .env.local.example .env.local"
    fi
    echo ""
    echo "Local Supabase and Qdrant are running."
    echo "Start API + worker as usual:"
    echo "  just api"
    echo "  just worker"

# Stop local Supabase and Qdrant services.
dev-services-stop:
    supabase stop
    -docker stop qdrant-dev 2>/dev/null; docker rm qdrant-dev 2>/dev/null; true

# Start the FastAPI app with autoreload on port 8000.
api:
    uv run uvicorn src.api.app:app --reload --port 8000

# Start the durable queue worker.
worker:
    uv run python -m src.worker.runner

# Run the backend test suite.
test:
    uv run pytest -q

# Run backend tests plus frontend lint/build checks.
check:
    ./scripts/workflow/check_all.sh

# Start the Next.js frontend in development mode.
frontend-dev:
    cd frontend && npm run dev

# Show Railway project/service status in JSON.
rw-status:
    railway status --json

# List Railway service variables in JSON.
rw-vars service="API":
    railway variable list --service "{{service}}" --json

# List Railway service variable names only (no values).
rw-vars-keys service="API":
    railway variable list --service "{{service}}" --json | jq -r 'keys[]'

# Set a Railway service variable.
rw-set key value confirm service="API":
    if [[ "{{confirm}}" != "CONFIRM_PROD" ]]; then echo "Refusing mutation. Pass CONFIRM_PROD as the confirmation token."; echo "Usage: just rw-set <key> <value> CONFIRM_PROD [service]"; exit 2; fi; railway variable set --service "{{service}}" "{{key}}={{value}}"

# Show recent Railway logs for a service.
rw-logs service="API" lines="200":
    railway logs --service "{{service}}" --lines "{{lines}}"

# Redeploy the latest Railway deployment for a service.
rw-redeploy confirm service="API":
    if [[ "{{confirm}}" != "CONFIRM_PROD" ]]; then echo "Refusing redeploy. Pass CONFIRM_PROD as the confirmation token."; echo "Usage: just rw-redeploy CONFIRM_PROD [service]"; exit 2; fi; railway redeploy --service "{{service}}" --yes

# Probe browser preflight behavior for the authenticated videos endpoint.
cors-preflight api_url origin:
    base="{{api_url}}"; base="${base%/}"; curl -i -X OPTIONS "${base}/users/me/videos" -H "Origin: {{origin}}" -H "Access-Control-Request-Method: GET" -H "Access-Control-Request-Headers: authorization"

# Fetch current bucket CORS config (works with R2 S3 endpoint).
r2-cors-get bucket endpoint:
    aws s3api get-bucket-cors --bucket "{{bucket}}" --endpoint-url "{{endpoint}}"
