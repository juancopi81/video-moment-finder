# Video Moment Finder - Status

## Current Phase

**Phase 3: Real Integrations + Reliability Hardening** - In Progress

## Progress Log


| Date       | Phase   | Task                              | Status | Notes                                                                                                                                                                 |
| ---------- | ------- | --------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-01-19 | Setup   | Create ROADMAP.md                 | Done   | Phased plan with gates                                                                                                                                                |
| 2026-01-19 | Setup   | Create STATUS.md                  | Done   | Progress tracking                                                                                                                                                     |
| 2026-01-19 | Setup   | Update README.md & CLAUDE.md      | Done   | Added doc links                                                                                                                                                       |
| 2026-01-20 | Phase 0 | 0.1 Modal + Qwen3 Setup           | Done   | A10G smoke test OK. Qwen3-VL-Embedding-2B loads and embeds text+image.                                                                                                |
| 2026-01-20 | Phase 0 | 0.2 Video Processing Test         | Done   | **GATE PASSED.** Initial: 0.766s/frame (batch=1). Later optimized to 0.146s/frame (batch=8).                                                                          |
| 2026-01-21 | Phase 0 | 0.3 Vector Search Validation      | Done   | **GATE PASSED.** Recall@5 = 90% (9/10 queries). In-memory Qdrant validated.                                                                                           |
| 2026-01-21 | Phase 1 | 1.1 Video Download Module         | Done   | Added fail-fast yt-dlp wrapper in src/video.                                                                                                                          |
| 2026-01-21 | Phase 1 | 1.2 Frame Extraction Module       | Done   | Added fail-fast ffmpeg wrapper with timestamps + thumbnails; refactored tests to use it.                                                                              |
| 2026-01-21 | Phase 1 | 1.3 Batch Embedding Pipeline      | Done   | Modal-only validation passed with batch=8.                                                                                                                            |
| 2026-01-22 | Phase 1 | Code cleanup                      | Done   | Removed dead code, DRY refactor: -36 lines. Created src/utils/subprocess.py utility.                                                                                  |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration           | Done   | Added Qdrant + R2 storage modules, pipeline orchestrator, cleanup utils. Local test passed.                                                                           |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration Tests     | Done   | Added unit tests for Qdrant/R2/cleanup/orchestrator. pytest: 16 passed.                                                                                               |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration (Cloud)   | Done   | Full Qdrant+R2 integration test passed via scripts/phase1/storage_integration_test.py                                                                                 |
| 2026-01-23 | Phase 1 | Phase 1 Gate Check (Cost)         | Done   | Extrapolated 30-min cost $0.1263 (1871s video, 0.1317s/frame). **GATE PASSED**                                                                                        |
| 2026-01-23 | Phase 1 | Phase 1 Gate Check (End-to-End)   | Done   | 1800 frames processed end-to-end (Qdrant+R2). Embed 357.36s, process 889.62s, total 1254.02s.                                                                         |
| 2026-01-26 | Phase 1 | R2 Parallel Uploads               | Done   | 1800 thumbnails uploaded in 68.25s (26.37 thumbs/s) with 16 workers. Total pipeline 429.74s.                                                                          |
| 2026-01-27 | Phase 2 | 2.2 Backend API (FastAPI)         | Done   | Mock FastAPI app with /videos and /search endpoints validated locally.                                                                                                |
| 2026-01-27 | Phase 2 | 2.3 Frontend Shell (scaffold)     | Done   | Next.js 14 with 2 routes: `/` (landing) and `/video/[id]` (status/search/results).                                                                                    |
| 2026-01-27 | Phase 2 | 2.3 Landing Page Form Wiring      | Done   | Form submits to backend API, navigates to /video/{id} on success, handles errors.                                                                                     |
| 2026-01-27 | Phase 2 | 2.3 Video Page API Wiring         | Done   | Status polling (2s interval), search form submission, results grid with thumbnails/timestamps.                                                                        |
| 2026-01-27 | Phase 2 | 2.1 Supabase Database Schema      | Done   | Created videos/credits tables, Python client with CRUD. Manual + unit tests (5 passing).                                                                              |
| 2026-01-28 | Phase 2 | 2.4 Connect the Pieces            | Done   | **GATE PASSED.** Full flow verified: submit URL → poll status → search → mock results.                                                                                |
| 2026-01-28 | Phase 1 | End-to-End Timing Recheck         | Done   | 1800 frames end-to-end: total 450.36s (~7.5 min). Extract 7.86s, embed 388.77s, process 53.73s. Test video: `https://www.youtube.com/watch?v=02YLwsCKUww` (31:11).    |
| 2026-01-28 | Phase 3 | 3.1 Wire FastAPI to Real Services | Done   | Added `embed_text` Modal function, video processing service, search service. Replaced mock endpoints with Supabase/Modal/Qdrant integrations.                         |
| 2026-02-06 | Phase 3 | Durable Queue + Worker Lifecycle  | Done   | Replaced FastAPI `BackgroundTasks` with Supabase-backed `video_jobs` queue and dedicated worker (`src/worker/runner.py`) with `queued -> processing -> ready/failed`. |
| 2026-02-06 | Phase 3 | Thumbnail URL Contract Fix        | Done   | Made thumbnail URL nullable end-to-end (storage + API response model) so non-R2 setups do not fail response validation.                                               |
| 2026-02-06 | Phase 3 | API + Queue Test Coverage         | Done   | Added API tests for submit/status/search plus worker and queue unit tests (`tests/api`, `tests/worker`, `tests/db/test_video_jobs.py`).                               |
| 2026-02-06 | Phase 3 | CI + Setup Hardening              | Done   | Added GitHub Actions CI (pytest, frontend lint/build), Supabase migrations, and one-command local setup script (`scripts/setup_local.sh`).                            |
| 2026-02-06 | Phase 3 | Search Latency Instrumentation    | Done   | Added search-stage timing logs, per-container text embedder cache, optional `MODAL_TEXT_EMBED_MIN_CONTAINERS` (default disabled), benchmark CLI, and unified `check_all` script. |
| 2026-02-07 | Phase 3 | Search Latency Iteration (Preload) | Done  | Switched text embedding to Modal class with container-start preload (`@modal.enter`), default `MODAL_TEXT_EMBED_MAX_CONTAINERS=1` for cache reuse, and benchmark docs now require explicit Modal deploy per mode. |
| 2026-02-07 | Phase 3 | Modal Class Lookup Deprecation Fix | Done  | Switched search text embedding handle to `modal.Cls.from_name(...).embed.remote(...)` to remove class-method lookup deprecation warning; removed temporary benchmark checklist notes from README. |
| 2026-02-07 | Phase 3 | Queue Reliability Hardening         | Done   | Added worker retry cap (`max_attempts`), stale-lock recovery/requeue for stuck `processing` jobs, structured worker metric logs (`job_attempt_started`, `job_requeued`, terminal failures), and crash/restart-oriented queue tests. |
| 2026-02-13 | Phase 3 | Search Result Deep Links            | Done   | API now returns `youtube_url` on search; frontend renders "Open at timestamp" links for each result.                                                                    |
| 2026-02-13 | Phase 4 | Auth Ownership + RLS Hardening      | Done   | Added Clerk JWT auth on create/get/search routes, owner-scoped video access (404 for non-owner), env-driven CORS origins, stricter YouTube URL normalization, frontend Bearer token wiring, and Supabase migration for RLS + function `search_path`. |
| 2026-02-16 | Phase 4 | 4.2 Payment Provider Feasibility (Colombia) | In Progress | Added launch gate to validate Stripe account availability for Colombia and choose documented workaround path before implementing payments. See [`RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md`](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md). |


## Blockers

- YouTube bot detection blocks yt-dlp from Modal IPs. Workaround: download videos locally first, then upload to Modal.
- Apply latest Supabase migration in each deployed environment to activate RLS policies and the hardened `public.set_updated_at` `search_path`.
- Phase 4.2 payments implementation is blocked pending payment-provider feasibility for Colombia (confirm direct Stripe availability or approve workaround path). Reference: [`RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md`](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md).

## Decisions Made

- **Batch=8 is optimal for A10G** - Tested batch sizes 1, 4, 8, 16, 32. All fit in 24GB VRAM. Batch=8 fastest (0.146s/frame), larger batches plateau (~0.15s/frame) due to GPU saturation.
- **Qwen3-VL-Embedding-2B is suitable for semantic search** - 90% Recall@5 validates the model for finding video moments via text queries.
- **2-route frontend structure** - Landing (`/`) + Video page (`/video/[id]`) that handles processing, search, and results based on state. Simpler than 3 separate pages.
- **Queue durability before feature expansion** - Added durable queue state in Supabase before continuing feature work to avoid fragile in-process job execution.
- **Warm containers are opt-in for cost control** - Search latency now supports optional `MODAL_TEXT_EMBED_MIN_CONTAINERS` (plus legacy alias), but default behavior keeps it unset to avoid always-on GPU cost in dev.

## Metrics / Measurements


| Metric                    | Target  | Actual             | Notes                                                                      |
| ------------------------- | ------- | ------------------ | -------------------------------------------------------------------------- |
| Search quality (Recall@5) | >70%    | **90%**            | **GATE PASSED** - 9/10 queries found correct frame in top 5                |
| Cost per 30-min video     | <$1     | **~$0.13**         | **GATE PASSED** - Extrapolated from 1871s video, 0.1317s/frame, 25.7s load |
| Processing time (30-min)  | <20 min | **~7.5 min**       | Full pipeline: 1800 frames end-to-end (extract+embed+upload+store)         |
| Single embed latency      | -       | **0.1317s**        | Per-frame with batch=8 (1871s video sample)                                |
| Model load time           | -       | ~25.7s             | Cold load in container                                                     |
| Frame extraction          | -       | 10.46s/1871 frames | ffmpeg at 1 fps                                                            |
| GPU device                | -       | NVIDIA A10         | Modal A10G (24GB VRAM)                                                     |
| Embedding dim             | -       | 2048               | Qwen3-VL-Embedding-2B                                                      |
| GPU cost rate             | -       | $0.000463/s        | A10G pricing                                                               |
| Qdrant store time         | -       | 0.05s              | 40 vectors (in-memory)                                                     |
| Query embedding time      | -       | ~0.2s              | Per text query                                                             |
