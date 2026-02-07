# Video Moment Finder

Semantic video frame search. Paste a YouTube URL, process it, and search moments with text queries.

## Documentation

- [PROJECT_SPEC.md](./PROJECT_SPEC.md) - Product and architecture spec
- [ROADMAP.md](./ROADMAP.md) - Phased implementation plan
- [STATUS.md](./STATUS.md) - Progress log and metrics
- [CLAUDE.md](./CLAUDE.md) - Agent guidance and developer commands

## Local Setup (One Command)

1. Copy `.env.example` to `.env` and fill values (including `SUPABASE_DB_URL`).
2. Run:

```bash
set -a && source .env && set +a
./scripts/setup_local.sh
```

This applies SQL migrations in `supabase/migrations`, installs Python deps with `uv`, and installs frontend deps with `npm ci`.

## Run Services

Run API:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Run worker (required for processing queue):

```bash
uv run python -m src.worker.runner
```

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

After a video reaches `ready`, run:

```bash
.venv/bin/python scripts/phase3/search_latency_benchmark.py --video-id <VIDEO_ID>
```

Optional warm-container comparison (benchmark only):

```bash
MODAL_TEXT_EMBED_MIN_CONTAINERS=1 .venv/bin/python scripts/phase3/search_latency_benchmark.py --video-id <VIDEO_ID>
```

Leave `MODAL_TEXT_EMBED_MIN_CONTAINERS` unset for normal development and default-cost behavior.
