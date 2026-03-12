# Video Moment Finder

Semantic video moment search for creators and researchers.
Process a video, run text or image queries, and jump to matching timestamps.

**Live site:** [videomomentfinder.com](https://videomomentfinder.com)

## What It Does

- Accepts direct video uploads as the reliable ingest path.
- Supports best-effort YouTube URL import for videos you own or are authorized to use.
- Processes video frames asynchronously.
- Embeds frames into a vector index for semantic retrieval.
- Returns top timestamped matches with thumbnails for text or example-image queries.

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

### Local Development (Isolated)

To develop without touching production services:

1. Install the Supabase CLI: `brew install supabase/tap/supabase`
2. Copy the local config: `cp .env.local.example .env.local`
3. Start local services: `just dev-services`
4. Run API and worker as usual: `just api` / `just worker`

Local Supabase runs on port 54321/54322. Qdrant runs in-memory inside the
Python process. Modal GPU functions are shared (stateless).

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

## Deployment (At a Glance)

- Frontend: Vercel (`frontend/`)
- API + worker: Railway (`Dockerfile`, `Dockerfile.worker`)
- Data: Supabase + Qdrant + Cloudflare R2

Detailed operational runbooks are intentionally kept out of README; use `docs/DEPLOYMENT.md` for operations reference.

## License

This project is licensed under the [GNU Affero General Public License v3.0](./LICENSE).

If you run a modified version as a network service, you must make the complete source code available to its users.
