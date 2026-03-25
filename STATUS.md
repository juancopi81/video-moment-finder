# Video Moment Finder - Status

## Current Phase

**Phase 7: Acquire, Activate, Monetize** - In Progress

## Progress Log

Authoring rules:

- Keep one-line notes per row.
- Do not use raw `|` in notes; escape it as `\|` when needed.
- Add new rows in chronological order.

| Date       | Phase   | Milestone                                           | Status      | Notes |
| ---------- | ------- | --------------------------------------------------- | ----------- | ----- |
| 2026-01-19 | Setup   | Documentation baseline                              | Done        | Created initial roadmap/status docs and linked core documentation. |
| 2026-01-20 | Phase 0 | Modal + Qwen setup                                  | Done        | A10G smoke test succeeded for multimodal embedding paths. |
| 2026-01-20 | Phase 0 | Video processing validation                          | Done        | Phase-0 gate passed after batch-size tuning. |
| 2026-01-21 | Phase 0 | Vector search validation                             | Done        | Recall@5 reached 90 percent in validation set. |
| 2026-01-21 | Phase 1 | Download and frame extraction modules                | Done        | Added fail-fast wrappers for yt-dlp and ffmpeg pipelines. |
| 2026-01-22 | Phase 1 | Storage integration and tests                        | Done        | Landed Qdrant, R2, cleanup orchestration, and focused unit coverage. |
| 2026-01-23 | Phase 1 | Cost and end-to-end gate checks                      | Done        | Phase-1 gate passed for cost and full processing flow. |
| 2026-01-26 | Phase 1 | R2 parallel uploads                                  | Done        | Improved upload throughput with parallelized workers. |
| 2026-01-27 | Phase 2 | Backend and frontend scaffolds                       | Done        | Added mock API plus initial Next.js shell and wiring. |
| 2026-01-28 | Phase 2 | End-to-end skeleton gate                             | Done        | URL submit, status polling, and search mock flow validated. |
| 2026-01-28 | Phase 3 | Real service wiring                                  | Done        | Replaced mocks with Supabase, Modal, and Qdrant integration. |
| 2026-02-06 | Phase 3 | Durable queue and CI hardening                       | Done        | Added Supabase-backed queue worker lifecycle and CI/setup hardening. |
| 2026-02-07 | Phase 3 | Search latency and queue reliability                 | Done        | Added latency instrumentation, preload tuning, and reliability controls. |
| 2026-02-13 | Phase 3 | Search deep links                                    | Done        | Search results now include direct open-at-timestamp links. |
| 2026-02-13 | Phase 4 | Auth ownership and DB policy hardening               | Done        | Enforced owner-scoped access with Clerk auth and DB policy updates. |
| 2026-02-17 | Phase 4 | Payments provider decision                            | Done        | Selected Lemon Squeezy first path with Paddle as fallback. |
| 2026-02-18 | Phase 3 | Upload ingest path                                   | Done        | Added authenticated upload ingest and local cache fallback path. |
| 2026-02-19 | Phase 3 | Frontend upload UX                                   | Done        | Added signed-in upload flow with progress and mode toggle. |
| 2026-02-20 | Phase 3 | Playback jump links                                  | Done        | Added playback controls tied to timestamped results. |
| 2026-02-23 | Phase 3 | Presigned direct upload flow                         | Done        | Added init and complete endpoints with R2 storage checks. |
| 2026-02-23 | Phase 4 | Public payment-onboarding pages                      | Done        | Added marketing/legal/support pages required for activation readiness. |
| 2026-02-24 | Phase 4 | Dashboard and free-tier guardrail                    | Done        | Added user dashboard and enforced per-user free video cap. |
| 2026-02-24 | Phase 4 | Production deployment                                | Done        | Deployed frontend, API, and worker with production infrastructure wiring. |
| 2026-02-25 | Phase 4 | Payment webhook grants                               | Done        | Added signed webhook handling with idempotent credit grants. |
| 2026-02-25 | Phase 4 | Tooling safety and preview CORS                      | Done        | Hardened helper commands and added wildcard or regex preview CORS support. |
| 2026-03-02 | Phase 4 | Checkout sessions and paid CTAs                      | Done        | Added checkout endpoint and wired paid pricing CTAs. |
| 2026-03-03 | Phase 4 | Worker transport resilience                           | Done        | Added transient transport retries and idle poll backoff. |
| 2026-03-04 | Phase 4 | Billing enforcement and status UX                    | Done        | Enforced paid-credit deduction and added billing summary UX feedback. |
| 2026-03-04 | Phase 4 | Launch baseline hardening                             | Done        | Added rate limiting and completed DRY refactor series milestone. |
| 2026-03-05 | Phase 4 | Runtime monitoring and upload admission validation   | Done        | Added Sentry integration and ffprobe-based upload duration checks. |
| 2026-03-06 | Phase 4 | Security advisor closure and launch copy pass         | Done        | Cleared Supabase Security Advisor, verified backups, removed unimplemented feature claims, added AI disclaimer. |
| 2026-03-06 | Phase 4 | Image query search                                    | Done        | Added uploaded-image query search across API, Modal, frontend, and docs. |
| 2026-03-06 | Phase 5 | Analytics baseline (product events)                   | Done        | First-party event tracking via analytics_events table. |
| 2026-03-06 | Phase 5 | Transcript-backed text search                         | Done        | Added YouTube subtitle extraction plus grouped spoken and visual text results. |
| 2026-03-10 | Phase 5 | Upload-first ingest positioning                       | Done        | Made direct upload the primary product story, simplified YouTube fallback UX, and updated owning docs. |
| 2026-03-10 | Phase 5 | Semantic transcript retrieval                         | Done        | Added Qdrant transcript embeddings for captioned videos and kept Supabase transcript search as fallback. |
| 2026-03-12 | Phase 5 | Parallel transcript and visual processing             | Done        | Parallelized independent worker branches, validated captioned YouTube retrieval end to end locally, and kept the search response contract unchanged. |
| 2026-03-12 | Phase 5 | Upload ASR-backed spoken retrieval                    | Done        | Added Whisper large-v3-turbo via faster-whisper for direct uploads, reused the transcript storage/indexing path, and kept the public search response schema unchanged. |
| 2026-03-14 | Phase 6 | External API contract (v1 routes)                     | Done        | Productized versioned /api/v1/ routes with Clerk JWT auth. |
| 2026-03-14 | Phase 6 | API keys and usage controls                            | Done        | Added vmf_ API key auth, key management endpoints, and quota enforcement. |
| 2026-03-17 | Phase 5 | Qdrant visual upsert batching                         | Done        | Batched frame-vector writes to stay under Qdrant payload limits, marked payload-limit failures terminal, and attached richer worker Sentry context. |
| 2026-03-17 | Phase 6 | Agent CLI and public guide                            | Done        | Added the stdlib `vmf` CLI over `/api/v1` for upload, poll, key bootstrap, and text search, plus a public usage guide. |
| 2026-03-18 | Phase 6 | Dashboard API access and billing separation            | Done        | Added /dashboard/api, separated API billing with unit-based model, /developers page, API section on pricing page. |
| 2026-03-23 | Phase 7 | Phase 7 planning baseline                              | Done        | Archived the completed Phase 5/6 plan, created the Phase 7 checklist, and reset the active roadmap toward acquisition, activation, and monetization. |
| 2026-03-23 | Phase 7 | ICP selection and pursuit playbook                     | Done        | Chose knowledge-heavy creators with owned long-form video libraries as the primary ICP and added a dedicated pursuit playbook. |
| 2026-03-24 | Phase 7 | PostHog funnel instrumentation (7.1)                   | Done        | Added PostHog for client-side product analytics; added processing_failure backend event; kept existing Supabase backend events. |

## Blockers

- Best-effort YouTube URL import can be blocked from cloud IP ranges; direct upload is the supported reliable path.
- Current traffic volume is too low to validate retention or pricing confidently.
- ~~Funnel instrumentation is incomplete on the signed-out and pre-auth path, so top-of-funnel conclusions are still weak.~~ Resolved 2026-03-25 (PostHog covers signed-out pageviews, CTAs, and full acquisition funnel).
- ~~Final launch hardening depends on clearing remaining Supabase Security Advisor findings in every deployed environment.~~ Resolved 2026-03-06.
- ~~Ops readiness checklist (backup verification and monitoring alert routing) remains open for launch gate completion.~~ Resolved 2026-03-06 (manual backup path verified; Sentry active for monitoring).

## Decisions Made

- **Batch=8 on A10G** is the default throughput baseline for embedding jobs.
- **Qwen3-VL-Embedding-2B** is the retrieval model baseline for semantic search.
- **Lemon Squeezy first** is the provider path, with Paddle retained as fallback.
- **Durable queue before expansion** was prioritized to stabilize processing reliability.
- **Warm containers remain opt-in** to control default development and production cost.
- **Direct upload is the primary ingest path**; YouTube URL import remains best effort.
- **Phase 7 prioritizes acquisition, activation, and monetization learning** before broader platform expansion.

## Metrics / Measurements

| Metric                    | Target  | Actual      | Notes |
| ------------------------- | ------- | ----------- | ----- |
| Search quality (Recall@5) | >70%    | 90%         | Validation set gate passed. |
| Cost per 30-min video     | <$1     | ~$0.13      | Extrapolated from A10G benchmark runs. |
| Processing time (30-min)  | <20 min | ~7.5 min    | End-to-end pipeline timing baseline. |
| Single embed latency      | -       | 0.1317s     | Batch=8 benchmark sample. |
| Model load time           | -       | ~25.7s      | Cold container start baseline. |
| Embedding dimension       | -       | 2048        | Qwen3-VL-Embedding-2B output vector size. |
