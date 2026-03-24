# Video Moment Finder

Semantic and transcript-aware video moment search for creators and researchers.
Process a video, run text or image queries, and jump to matching timestamps.

**Live site:** [videomomentfinder.com](https://videomomentfinder.com)

## What It Does

- Accepts direct video uploads as the reliable ingest path.
- Supports best-effort YouTube URL import for videos you own or are authorized to use.
- Processes video frames asynchronously.
- Stores YouTube transcript segments when subtitle or automatic caption tracks are available.
- Embeds frames into a vector index for semantic retrieval.
- Runs text queries across visual retrieval and semantically indexed transcript retrieval.
- Returns top timestamped matches for text or example-image queries, with thumbnails when visual matches are available.
- For text queries, the `limit` applies per result group: up to `limit` visual matches and up to `limit` spoken matches.
- Exposes the upload, poll, and text-search happy path through a thin CLI over `/api/v1`.

Current transcript scope:

- YouTube videos use existing subtitle tracks or automatic caption tracks when available and index them for semantic transcript retrieval.
- YouTube videos currently do not fall back to Whisper when no caption track is available.
- Direct uploads extract speech with Whisper large-v3-turbo via `faster-whisper` and index those transcript segments for spoken-text queries.

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- Node.js 18+
- `uv` and `npm`
- Supabase project and database credentials

### 1) Configure environment

```bash
cp .env.example .env
cp frontend/.env.example frontend/.env.local
```

Fill required values in both files.
For service ownership and operations detail, see [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md).

### 2) One-command setup

```bash
set -a && source .env && set +a
./scripts/setup_local.sh
```

### 3) Run services

Backend API:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Worker:

```bash
uv run python -m src.worker.runner
```

Frontend:

```bash
cd frontend && npm run dev
```

### Local Development (Supabase + Qdrant Isolation)

To develop with a local database and vector store (no production queue
or index interference):

1. Install the Supabase CLI: `brew install supabase/tap/supabase`
2. Copy the local config: `cp .env.local.example .env.local`
3. Start local services: `just dev-services`
4. Run API and worker as usual: `just api` / `just worker`

Local Supabase runs on port 54321/54322. Local Qdrant runs on port 6333
via Docker. Modal GPU functions and R2 storage are still shared with
production (stateless).

### 4) Verify quality checks

```bash
./scripts/workflow/check_all.sh
```

## High-Level Architecture

```text
Next.js frontend -> FastAPI API -> Supabase-backed queue -> Python worker -> Modal GPU
                                            |                  |                |
                                            v                  v                v
                                         Supabase           Qdrant      Cloudflare R2
```

Core flow:

1. User uploads a video or optionally submits an owned YouTube URL.
2. API enqueues a processing job.
3. Worker extracts frames and sends embedding work to Modal.
4. Embeddings are stored in Qdrant and thumbnails in R2.
5. User searches with text or an example image and receives timestamped matches.

Reliable production ingest path: browser upload -> direct-to-R2 storage -> worker processing.
YouTube URL import remains best effort and may be blocked by server-side restrictions.

## Documentation Map

- [PROJECT_SPEC.md](./PROJECT_SPEC.md): stable product charter.
- [ROADMAP.md](./ROADMAP.md): future work only.
- [STATUS.md](./STATUS.md): execution history only.
- [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md): deployment env ownership, webhook contract, and upload flow reference.
- [docs/CLI_API_GUIDE.md](./docs/CLI_API_GUIDE.md): public CLI and external API happy-path guide.

## Deployment (At a Glance)

- Frontend: Vercel (`frontend/`)
- API + worker: Railway (`Dockerfile`, `Dockerfile.worker`)
- Data: Supabase + Qdrant + Cloudflare R2

Detailed operational runbooks are intentionally kept out of README; use `docs/DEPLOYMENT.md` for operations reference.

## License

This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE).

If you run a modified version as a network service, you must make the complete source code available to its users.
