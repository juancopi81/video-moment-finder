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

## Temporary Benchmark Checklist (Delete After Tomorrow)

1. Pick one `video_id` that is already `ready` and reuse it for all runs.
2. Create a separate `main` worktree (one time):

```bash
cd /Users/juanpineros/juancopi81/video-moment-finder
git worktree add /tmp/vmf-main main
```

3. Run **BEFORE (main)** API (Terminal A):

```bash
cd /tmp/vmf-main
set -a && source /Users/juanpineros/juancopi81/video-moment-finder/.env && set +a
/Users/juanpineros/juancopi81/video-moment-finder/.venv/bin/uvicorn src.api.app:app --port 8001
```

4. Run benchmark against `main` (Terminal B):

```bash
cd /Users/juanpineros/juancopi81/video-moment-finder
.venv/bin/python /Users/juanpineros/juancopi81/video-moment-finder/scripts/phase3/search_latency_benchmark.py \
  --api-url http://localhost:8001 \
  --video-id <VIDEO_ID> \
  --runs-per-query 5 \
  --json-output /tmp/search_before.json
```

5. Stop Terminal A, then run **AFTER (feature branch)** API (Terminal A):

```bash
cd /Users/juanpineros/juancopi81/video-moment-finder
set -a && source .env && set +a
.venv/bin/uvicorn src.api.app:app --port 8000
```

6. Run benchmark against feature branch (Terminal B):

```bash
cd /Users/juanpineros/juancopi81/video-moment-finder
.venv/bin/python /Users/juanpineros/juancopi81/video-moment-finder/scripts/phase3/search_latency_benchmark.py \
  --api-url http://localhost:8000 \
  --video-id <VIDEO_ID> \
  --runs-per-query 5 \
  --json-output /tmp/search_after.json
```

7. Optional warm-container cost experiment (after only):

```bash
MODAL_TEXT_EMBED_MIN_CONTAINERS=1 .venv/bin/uvicorn src.api.app:app --port 8002
```

```bash
.venv/bin/python /Users/juanpineros/juancopi81/video-moment-finder/scripts/phase3/search_latency_benchmark.py \
  --api-url http://localhost:8002 \
  --video-id <VIDEO_ID> \
  --runs-per-query 5 \
  --json-output /tmp/search_after_warm.json
```
