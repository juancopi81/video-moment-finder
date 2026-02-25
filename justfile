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
