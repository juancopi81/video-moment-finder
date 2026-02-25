set shell := ["bash", "-cu"]

default:
    @just --list

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

# Set a Railway service variable.
rw-set service key value:
    railway variable set --service "{{service}}" "{{key}}={{value}}"

# Show recent Railway logs for a service.
rw-logs service="API" lines="200":
    railway logs --service "{{service}}" --lines "{{lines}}"

# Redeploy the latest Railway deployment for a service.
rw-redeploy service="API":
    railway redeploy --service "{{service}}" --yes

# Probe browser preflight behavior for the authenticated videos endpoint.
cors-preflight api_url origin:
    base="{{api_url}}"; base="${base%/}"; curl -i -X OPTIONS "${base}/users/me/videos" -H "Origin: {{origin}}" -H "Access-Control-Request-Method: GET" -H "Access-Control-Request-Headers: authorization"

# Fetch current bucket CORS config (works with R2 S3 endpoint).
r2-cors-get bucket endpoint:
    aws s3api get-bucket-cors --bucket "{{bucket}}" --endpoint-url "{{endpoint}}"
