# Video Moment Finder

Semantic video frame search. Paste a YouTube URL, process it, and search moments with text queries.

## Documentation

- [PROJECT_SPEC.md](./PROJECT_SPEC.md) - Product and architecture spec
- [ROADMAP.md](./ROADMAP.md) - Phased implementation plan
- [STATUS.md](./STATUS.md) - Progress log and metrics
- [RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md) - Comprehensive payment-provider research for Colombia-based global launch
- [CLAUDE.md](./CLAUDE.md) - Agent guidance and developer commands

## Local Setup (One Command)

1. Copy `.env.example` to `.env` and fill backend/infrastructure values (including `SUPABASE_DB_URL`, `CLERK_ISSUER`, and `CORS_ALLOWED_ORIGINS`).
2. Copy `frontend/.env.example` to `frontend/.env.local` and set frontend values (especially `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`).
3. Run:

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

```bash
set -a && source .env && set +a
./scripts/setup_local.sh
```

This applies SQL migrations in `supabase/migrations`, installs Python deps with `uv`, and installs frontend deps with `npm ci`.

Required auth/CORS env for local API + frontend:

Backend (`.env`):
- `CLERK_ISSUER` (JWT issuer verification)
- `CORS_ALLOWED_ORIGINS` (comma-separated frontend origins; default `http://localhost:3000`)

Frontend (`frontend/.env.local`):
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` (Clerk initialization)
- `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`)

If `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is missing in the frontend env, Clerk can run in keyless mode and API calls will fail with `Invalid authentication token`.

## Run Services

Run API:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Protected API routes (`POST /videos`, `GET /videos/{id}`, `POST /videos/{id}/search`) now require `Authorization: Bearer <Clerk JWT>`.

Run worker (required for processing queue):

```bash
uv run python -m src.worker.runner
```

Optional reliability overrides:

```bash
uv run python -m src.worker.runner --max-attempts 3 --stale-lock-timeout 600
```

- Failed jobs are retried up to `VIDEO_JOB_MAX_ATTEMPTS` (default `3`).
- `processing` jobs with stale locks older than `VIDEO_JOB_STALE_LOCK_TIMEOUT_S` seconds
  (default `600`) are recovered and requeued.

Run frontend:

```bash
cd frontend && npm run dev
```

## Quality Checks

Run all checks with one command:

```bash
./scripts/workflow/check_all.sh
```

Equivalent manual commands:

Backend tests:

```bash
.venv/bin/pytest -q
```

Frontend lint/build:

```bash
cd frontend && npm run lint && npm run build
```

## Search Latency Benchmark

Deploy the current embedding app code before benchmarking:

```bash
uv run modal deploy src/embedding/modal_app.py
```

After a video reaches `ready`, run:

```bash
.venv/bin/python scripts/phase3/search_latency_benchmark.py --video-id <VIDEO_ID>
```

Optional warm-container comparison (benchmark only):

```bash
MODAL_TEXT_EMBED_MIN_CONTAINERS=1 .venv/bin/python scripts/phase3/search_latency_benchmark.py --video-id <VIDEO_ID>
```

Leave `MODAL_TEXT_EMBED_MIN_CONTAINERS` unset for normal development and default-cost behavior.

If you test warm containers, redeploy with env set:

```bash
MODAL_TEXT_EMBED_MIN_CONTAINERS=1 uv run modal deploy src/embedding/modal_app.py
```
