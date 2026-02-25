# Video Moment Finder

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

Semantic video frame search. Paste a YouTube URL or upload a video, process it, and search moments with text queries.

## Documentation

- [PROJECT_SPEC.md](./PROJECT_SPEC.md) - Stable product charter (vision, user, scope, constraints, success metrics, risks, high-level architecture)
- [ROADMAP.md](./ROADMAP.md) - Planned future work only (phases, tasks, gates)
- [STATUS.md](./STATUS.md) - Execution history only (progress log, blockers, decisions, metrics)
- [RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md) - Time-bounded payment-provider research snapshot for Colombia-based global launch
- [CLAUDE.md](./CLAUDE.md) - Contributor/agent workflow guidance and developer commands

Maintenance rule: update only the document that owns the change type above to avoid duplicate sources of truth.

## Next PR Targets

- **PR 1 — Lemon Squeezy checkout session creation**: expose backend endpoint to create signed Lemon Squeezy checkout links with user/credit metadata.
- **PR 2 — Paid pricing CTA go-live**: replace waitlist CTAs with live checkout URLs after checkout endpoint and store products are wired.

## Production Deployment

**Live at [videomomentfinder.com](https://videomomentfinder.com)**

Production environment variables are managed in the **Railway** (API + worker) and **Vercel** (frontend) dashboards — not in the local `.env` file. The local `.env` is for development only.

This repo ships with Railway-ready Dockerfiles for the API + worker and a Vercel config for the Next.js frontend.

### Railway (API)

- Service root: repository root
- Dockerfile: `Dockerfile`
- Runtime packages: includes `ffmpeg` for video frame extraction
- Start command: provided by `CMD`
- Port: Railway injects `$PORT` (default 8000)

Required environment (API + worker):

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `SUPABASE_DB_URL`
- `CLERK_ISSUER`
- `CLERK_AUDIENCE` (optional)
- `CLERK_JWKS_URL` (optional override)
- `CORS_ALLOWED_ORIGINS` (comma-separated; supports exact origins plus `*` wildcards like `https://video-moment-finder-*.vercel.app`)
- `CORS_ALLOWED_ORIGIN_REGEX` (optional regex for dynamic origins when wildcards are not enough)
- `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`
- `R2_PUBLIC_URL` (optional, for public thumbnail URLs)
- `QDRANT_URL`
- `QDRANT_API_KEY` (if required by your Qdrant deployment)
- `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`
- `LEMON_SQUEEZY_WEBHOOK_SECRET` (for webhook signature verification)

Optional production tuning:

- `VIDEO_MAX_DURATION_S`
- `VIDEO_MAX_FREE_VIDEOS`
- `VIDEO_SOURCE_URL_TTL_S`
- `VIDEO_UPLOAD_URL_TTL_S`

### Railway (Worker)

- Service root: repository root
- Dockerfile: `Dockerfile.worker`
- Runtime packages: includes `ffmpeg` for video frame extraction
- Start command: provided by `CMD`

### Vercel (Frontend)

Vercel reads `vercel.json` at repo root and builds from `frontend/`.

Required environment:

- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`

### Production/Preview Environment Checklist

Use this checklist before enabling or changing auto-deploy behavior:

- Vercel:
  - Set `NEXT_PUBLIC_API_URL` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` in both **Production** and **Preview** env scopes.
  - Ensure production branch is `main`.
- Railway (API + worker):
  - Keep the same backend env contract in both services (`SUPABASE_*`, `CLERK_*`, `R2_*`, `QDRANT_*`, `MODAL_*`, `CORS_ALLOWED_ORIGINS`).
  - Ensure `CORS_ALLOWED_ORIGINS` includes production plus preview origins (for example `https://videomomentfinder.com,https://video-moment-finder-*.vercel.app`).
  - Ensure both services track the intended branch (`main`) and have auto-deploy enabled.
- Clerk:
  - Keep production deployments wired to the **Production** Clerk instance (`CLERK_ISSUER` and `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` must match that instance).
  - Use Clerk **Development** instance keys only for local development.
- Supabase:
  - Keep production services on production project credentials.
  - If/when adding a staging environment, use a separate Supabase project and keys.

### Initial Deployment (Completed)

The first production deploy has been completed. For reference, the steps were:

- Applied Supabase migrations in `supabase/migrations` to production.
- Configured Railway services (API + worker) with the environment variables listed above.
- Set `CORS_ALLOWED_ORIGINS` to include the Vercel production domain and preview wildcard (or configure `CORS_ALLOWED_ORIGIN_REGEX`).
- Configured Clerk production instance with Google OAuth (test mode).
- Set up Cloudflare DNS (A record + www CNAME for Vercel, CNAMEs for Clerk).
- Updated R2 CORS for the production frontend origin.

Production env vars live in: **Railway dashboard** (API + worker) and **Vercel dashboard** (frontend).

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
- `CORS_ALLOWED_ORIGINS` (comma-separated frontend origins; supports `*` wildcard entries; default `http://localhost:3000`)
- `CORS_ALLOWED_ORIGIN_REGEX` (optional regex to match dynamic preview origins)
- `VIDEO_MAX_DURATION_S` (reject videos longer than this many seconds; default `1800`)
- `VIDEO_MAX_FREE_VIDEOS` (max free videos per user before payments; default `1`)
- `VIDEO_SOURCE_URL_TTL_S` (signed URL lifetime in seconds for uploaded video playback; default `3600`)
- `VIDEO_UPLOAD_URL_TTL_S` (signed upload URL lifetime in seconds for direct-to-R2 uploads; default `900`)
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

Protected API routes (`POST /videos`, `GET /videos/{id}`, `GET /users/me/videos`, `POST /videos/{id}/search`) now require `Authorization: Bearer <Clerk JWT>`.

Public billing webhook route:

```bash
POST /webhooks/lemonsqueezy
```

Webhook contract (current V0):

- Signature header: `X-Signature` (HMAC SHA-256 of raw body, using `LEMON_SQUEEZY_WEBHOOK_SECRET`).
- Grant events (default): `order_created`, `subscription_payment_success` (`BILLING_GRANT_EVENT_NAMES` override supported).
- Credit metadata source: `meta.custom_data.user_id` and `meta.custom_data.credits`.
- Idempotency key: `<event_name>:<data.id>` fallback to raw payload SHA-256 when `data.id` is absent.

Upload a video via presigned direct-to-R2 flow (requires R2 env configured):

```bash
curl -X POST "http://localhost:8000/videos/upload/init" \
  -H "Authorization: Bearer <CLERK_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"filename":"video.mp4","content_type":"video/mp4"}'

curl -X PUT "<UPLOAD_URL_FROM_RESPONSE>" \
  -H "Content-Type: video/mp4" \
  --data-binary "@/path/to/video.mp4"

curl -X POST "http://localhost:8000/videos/upload/complete" \
  -H "Authorization: Bearer <CLERK_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"video_id":"<VIDEO_ID_FROM_RESPONSE>","filename":"video.mp4"}'
```

Ensure your R2 bucket CORS allows `PUT` from the frontend origin for direct uploads.

Small files can still use the API upload endpoint:

```bash
curl -X POST "http://localhost:8000/videos/upload" \
  -H "Authorization: Bearer <CLERK_JWT>" \
  -F "file=@/path/to/video.mp4"
```

List your videos:

```bash
curl -X GET "http://localhost:8000/users/me/videos" \
  -H "Authorization: Bearer <CLERK_JWT>"
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

## License

This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE).

You are free to use, modify, and distribute this software under the terms of the AGPL-3.0. If you run a modified version as a network service, you must make the complete source code available to its users.

The hosted service at [videomomentfinder.com](https://videomomentfinder.com) is operated by the project author and subject to its own [Terms of Service](https://videomomentfinder.com/terms).
