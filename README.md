# Video Moment Finder

Semantic video frame search. Paste a YouTube URL or upload a video, process it, and search moments with text queries.

## Documentation

- [PROJECT_SPEC.md](./PROJECT_SPEC.md) - Stable product charter (vision, user, scope, constraints, success metrics, risks, high-level architecture)
- [ROADMAP.md](./ROADMAP.md) - Planned future work only (phases, tasks, gates)
- [STATUS.md](./STATUS.md) - Execution history only (progress log, blockers, decisions, metrics)
- [RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md) - Time-bounded payment-provider research snapshot for Colombia-based global launch
- [CLAUDE.md](./CLAUDE.md) - Contributor/agent workflow guidance and developer commands

Maintenance rule: update only the document that owns the change type above to avoid duplicate sources of truth.

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
- `VIDEO_MAX_DURATION_S` (reject videos longer than this many seconds; default `1800`)
- `VIDEO_SOURCE_URL_TTL_S` (signed URL lifetime in seconds for uploaded video playback; default `3600`)
- `VIDEO_LOCAL_VIDEO_DIR` (optional local cache for pre-downloaded videos, named `<youtube_id>.<ext>`)
- `R2_*` (required for uploaded video ingest and thumbnails)

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

Upload a video (requires R2 env configured):

```bash
curl -X POST "http://localhost:8000/videos/upload" \
  -H "Authorization: Bearer <CLERK_JWT>" \
  -F "file=@/path/to/video.mp4"
```

Run worker (required for processing queue):

```bash
uv run python -m src.worker.runner
```

YouTube download workaround (for cloud IP bot detection):

1. Download a video locally with `yt-dlp`:

```bash
uv run yt-dlp -f "best[height<=720]" -o "abc123xyz45.mp4" "https://www.youtube.com/watch?v=abc123xyz45"
```

2. Set `VIDEO_LOCAL_VIDEO_DIR` to the folder containing the file.
3. Ensure the filename matches the YouTube video ID (`abc123xyz45.mp4`).

Optional reliability overrides:

```bash
uv run python -m src.worker.runner --max-attempts 3 --stale-lock-timeout 600
```

- Failed jobs are retried up to `VIDEO_JOB_MAX_ATTEMPTS` (default `3`).
- `processing` jobs with stale locks older than `VIDEO_JOB_STALE_LOCK_TIMEOUT_S` seconds
  (default `600`) are recovered and requeued.

## R2 Object Namespaces + Lifecycle Rules

R2 stores different object types in separate namespaces:

- Thumbnails: `thumb/<video_id>/thumb_00001.jpg`
- Uploaded source videos: `source/<video_id>/<filename>`

Lifecycle setup (Cloudflare dashboard):

1. Open **R2 → Buckets → <your bucket> → Settings → Lifecycle rules**.
2. Add a rule with prefix `source/`.
3. Set the expiration window that matches your retention policy (for example, delete after 7 days).
4. Save the rule.

Verification steps:

1. Upload a small test object to `source/<video_id>/test.txt` and another to `thumb/<video_id>/thumb_00000.jpg`.
2. Confirm the lifecycle rule targets only the `source/` prefix in the bucket UI.
3. Optional CLI check (uses the same R2 credentials as the app):

```bash
aws s3api get-bucket-lifecycle-configuration \
  --endpoint-url "$R2_ENDPOINT_URL" \
  --bucket "$R2_BUCKET_NAME"
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
