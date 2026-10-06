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
| 2026-03-25 | Phase 7 | Activation and onboarding UX (7.2)                     | Done        | Added signed-out preview card, search suggestion chips, empty-results and error-state guidance, YouTube demotion to secondary link, dashboard empty-state copy. |
| 2026-03-27 | Phase 7 | Remote MCP OAuth connector (7.6)                      | Done        | Shipped `/mcp` with four MCP tools, OAuth PKCE auth, connector approval flow, and public Claude-facing docs. |
| 2026-07-09 | Phase 7 | Max video duration raised to 90 minutes               | Done        | Raised default duration cap with derived frame limit, extended stale-lock timeout to cover long jobs, and updated frontend copy truthfully. |
| 2026-07-09 | Phase 7 | Lecture-notes primitives (transcript, frames, prompt) | Done        | Added transcript and frames REST endpoints, get_transcript/get_frames MCP tools with image content, the lecture_notes MCP prompt, per-call unit pricing, and published recipe docs. |
| 2026-07-09 | Phase 7 | Retrieval hardening (review follow-up)                 | Done        | Paginated transcript fetch past PostgREST row caps, added bill-then-compensate to metered retrieval, ready-status gating, process-wide ffmpeg concurrency bound, rate limiting on retrieval routes, finite-timestamp validation, and per-video duration persistence for frame clamping. |
| 2026-07-09 | Phase 7 | MCP connector one-time re-consent                      | Done        | Versioned the approved tool list on OAuth grants, rejected pre-expansion tokens and refreshes so existing connections reconnect once, and disclosed exact per-tool unit costs on the approval screen. |
| 2026-07-09 | Phase 7 | Upload size limit                                      | Done        | Added an 8 GiB upload cap enforced mid-stream on the multipart path and via HEAD check at presigned complete, with frontend pre-check and bounded temp-disk usage. |
| 2026-07-10 | Phase 7 | Client-neutral MCP guidance                            | Done        | First real lecture-notes dogfood run passed end to end; embedded the workflow in MCP server instructions for clients without prompt support, neutralized Claude-only copy, documented prompt-invocation reality and Codex setup. |
| 2026-10-02 | Phase 7 | Portable lecture-learning plugin                      | Implemented | Added four skills, offline HTML/CSV renderers, original examples, reproducible archive, and a saved private release; real-lecture excerpt artifacts and claim audits remain private. |
| 2026-10-02 | Phase 7 | Neutral onboarding and verified-account trial          | Implemented, inactive | Prepared default-off 600-unit trial with trusted Clerk verification, one-time enrollment, shared website/API accounting, historical reconciliation and retry/refund fixes; no production migration or grant. |
| 2026-10-02 | Phase 7 | Learning release validation and independent review     | Passed locally | Full check passed: 672 Python tests including 17 isolated PostgreSQL cases, 13 archive tests, seven frontend tests, lint and normal 18-page build; accounting review findings resolved. |
| 2026-10-02 | Phase 7 | Learning experience checks                             | Partial live coverage | Audited one real lecture excerpt across all four skills; three HTML outputs passed desktop/mobile browser checks and tutoring passed a seven-turn agent harness; ingestion, second lecture and installed-host OAuth remain pending. |
| 2026-10-02 | Phase 7 | Public review preparation                              | Blocked externally | Listing, five positive/three negative cases and recording script prepared; current official policies and live listing URLs blocked by destination policy, while reviewer/publisher/recording facts remain outstanding in docs/plugin/SUBMISSION.md. |
| 2026-10-02 | Phase 7 | Private release follow-up                             | Validated with release gates | Saved 0.1.1 under the same private identity with publisher/country metadata; official portable schemas passed. Indexed one original narrated lesson, verified full transcript/high-resolution frames/playback, tested a second lecture and reconnected the existing ChatGPT connector. Live policy/page access is restored; production pages still need the prepared updates. Exact remaining gates are in docs/plugin/SUBMISSION.md. |
| 2026-10-03 | Phase 7 | Playground and Presentation expansion | Implemented | Added a computed vector playground with predictions, pin/reset and observation export; reusable presentation authoring with HTML and editable PPTX, source notes and original work fixtures. Reused cached lecture evidence with zero new VMF calls. Release preparation remains in docs/plugin/SUBMISSION.md. |
| 2026-10-02 | Phase 7 | Local follow-up validation                            | Passed; isolated DB run in CI | 669 Python tests passed and 17 opt-in PostgreSQL tests skipped locally because Docker was unavailable; 13 archive tests, seven frontend checks and lint passed. Normal frontend build passed after the supported network permission allowed its existing Google Fonts requests. Added nine fixture portability/empty-audio tests; no production dependency added. |
| 2026-10-05 | Phase 7 | Hosted staging database verification | Passed; API staging pending | Applied all 20 migrations atomically to an empty separate Supabase project; all 17 existing PostgreSQL integration cases passed, including concurrency and client-role restrictions. A separate schema preflight and independent cleanup audit passed; all 18 application tables are empty afterward. No production migration, API deployment or real-user trial grant. |
| 2026-10-05 | Phase 7 | Hosted staging API verification | Passed; real-account OAuth pending | Separate Railway API deployed with a public HTTPS endpoint and OAuth health check; all 20 HTTP checks passed, including real Supabase SDK reads, PKCE, access restrictions and disabled trials. Disposable records removed. Real Clerk sign-in and approved OAuth/reconnect remain; no production rollout or trial activation. |
| 2026-10-05 | Phase 7 | Staging OAuth reconnect and revocation compatibility | Reconnect passed; fix awaiting live validation | Real Clerk sign-in and a second approved PKCE connection passed 17 checks, including five-tool account isolation, code single use and refresh rotation. Revocation exposed MCP SDK 1.26's required nullable secret field for public clients; a narrow normalization fix and five regression cases passed full local validation (691 backend tests, 13 archive tests, seven frontend cases, lint and build). Scoped OAuth/balance/video fixtures were removed and audited; no production changes, indexing or trial activation. |
| 2026-10-05 | Phase 7 | Live staging revocation verification | Passed; production approval next | Commit d3c3826 passed both CI runs (703 tracked backend/PostgreSQL cases, including all 17 database cases) and deployed to staging. All 11 independent synthetic-token revocation checks passed, covering access/refresh invalidation, cross-client ownership and unknown tokens; fixture cleanup passed. Production's read-only preflight found exactly two missing migrations, saved current billing definitions and confirmed trial enablement is absent. The guarded atomic migration draft is prepared but unexecuted; no production deployment or trial grant. |

## Blockers

- Best-effort YouTube URL import can be blocked from cloud IP ranges; direct upload is the supported reliable path.
- Current traffic volume is too low to validate retention or pricing confidently.
- Learning plugin release gates and exact remaining checks are tracked in `docs/plugin/SUBMISSION.md`; private creation does not establish installation, submission or publication.
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
- **Study-notes generation stays agent-side first**: transcript/frames primitives and the lecture_notes MCP prompt ship with flat per-call unit pricing; a web-UI notes feature waits for dogfooding evidence.
- **Playground expands the experimental Assumption Lab** with real computed models when justified by the evidence. Finite case comparison remains a mode. Presentation adds a fifth workflow for learning, teaching and work. Neither extension establishes measured demand or learning gains; see `docs/plugin/fourth-skill-evaluation.md`.
- **Trial enrollment is permanent per verified account** and replaces the remaining legacy free indexing entitlement; new grants remain disabled until separately approved. Deployment and historical-account caveats live in `docs/DEPLOYMENT.md`.

## Metrics / Measurements

| Metric                    | Target  | Actual      | Notes |
| ------------------------- | ------- | ----------- | ----- |
| Search quality (Recall@5) | >70%    | 90%         | Validation set gate passed. |
| Cost per 30-min video     | <$1     | ~$0.13      | Extrapolated from A10G benchmark runs. |
| Processing time (30-min)  | <20 min | ~7.5 min    | End-to-end pipeline timing baseline. |
| Single embed latency      | -       | 0.1317s     | Batch=8 benchmark sample. |
| Model load time           | -       | ~25.7s      | Cold container start baseline. |
| Embedding dimension       | -       | 2048        | Qwen3-VL-Embedding-2B output vector size. |
| Learning development API use (2026-10-02) | ≤700 units, ≤1 new index | 513 actual units, 1 new index | Live ledger reconciled readiness 4 + second lecture 3 + original lesson 506. Reconnect added no grant; available balance 5,437. |
| Portable package reproducibility (0.1.1) | Identical bytes | 28 files | SHA256 `ddca03a0939c310a98b738323baaa67f164455d8a062a7a06c156c9af7f004a5`; stored portable and generated compatibility manifests verified at 0.1.1. |
| Portable package reproducibility (0.2.0) | Identical bytes | 44 files, 5 skills | SHA256 `4e05430e8b8a626b53d87dc4fd650d822e63960c8858b6255143ee5c16ece3a4`; private release read back, original PPTX included; zero additional VMF calls. |
