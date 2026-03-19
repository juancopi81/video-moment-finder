# Phase 5-6 Plan

## Purpose

Phase 5 and Phase 6 should be treated as one staged expansion:

- strengthen the core search product first,
- then expose the same capabilities through an external API,
- then ship a thin CLI over that API once the contract is stable.

The web app remains the primary product. Agent access is an additional interface, not a separate product path. For this phase, the CLI is the agent-facing wrapper and MCP is explicitly out of scope.

## Working Rules

- Keep business logic in shared domain services used by both web and API flows.
- Keep this document at decision level; implementation details belong in code and tests.
- Prefer a small number of milestone-based PRs over one large rewrite or many tiny PRs.
- Each milestone should end at a reviewable gate that can be checked with tests, a short manual flow, or both.

## Phase 5 Outcomes

- Ship transcript-backed or hybrid retrieval for the core product.
- Keep existing text and image search behavior working while search expands.
- Stabilize the search result contract that later API clients will consume.

Suggested milestone gates:

1. Transcript-aware retrieval returns useful timestamped results for spoken-content queries.
2. Existing visual and image search paths keep passing regression checks.
3. Search responses expose a stable set of fields needed by later API clients.

## Phase 6 Outcomes

- Expose authenticated REST access for the same core capabilities.
- Add account-scoped API keys, usage attribution, quotas, and revocation.
- Keep room for a separate API pricing track and usage ledger instead of coupling long-term API billing to the web app's creator-facing credit packs.
- Add a thin CLI over the external API only after the API surface is stable enough to wrap cleanly.
- Add an agent-readable markdown usage guide only after the CLI and API happy path are stable enough to describe clearly.

Suggested milestone gates:

1. Account ownership, key scope, and revocation are covered by tests.
2. Usage is attributed correctly and retries do not create double billing.
3. The API can process a video and run search end-to-end in a reviewable happy path with stable request and response shapes.
4. CLI smoke checks are added only after the API milestone is stable and prove the wrapper does not add product-only logic beyond the API contract.
5. An agent can complete a documented happy path entirely through the CLI without reading implementation code.
6. The public agent-readable markdown guide matches the real CLI and API behavior, including auth setup, command examples, expected outputs, and failure cases.

## Delivery Shape

- Default target: one focused PR for the main Phase 5 work and a small sequence of focused PRs for Phase 6.
- Inside each PR, milestone commits should still land at clear review checkpoints.
- Phase 6 should split at clear product boundaries instead of combining API contract, account controls, CLI behavior, and docs into one large PR.
- Keep the repository-required `codex/` branch prefix, but use conventional `feat` or `docs` naming in the branch suffix and PR title when it improves clarity.
- A milestone is done only when its review gate is explicit and repeatable.

## Phase 5 Technical Decisions

### ASR Model: Whisper large-v3-turbo via faster-whisper

Chosen over IBM Granite 4.0 1B Speech (5.52% WER, 1B params) and NVIDIA Canary-Qwen 2.5B (5.63% WER, English-only).

- **Word-level timestamps** are critical for moment search; Whisper has native, battle-tested support. Granite and Canary do not document this.
- **99 languages** vs Granite's 6 and Canary's 1 (English-only).
- **Auto-chunking** for long audio built in. Canary has a hard 40s input limit.
- **Tiny container image** (`faster-whisper` is a single pip install) means fast cold starts on Modal. NeMo (Canary) is 10-15 GB.
- **INT8 quantization** brings VRAM to ~2-3 GB, fitting on a **T4 GPU** ($0.59/hr) — nearly half the cost of the A10G used for embeddings.
- WER (~7-9%) is slightly behind Granite/Canary but more than sufficient for search indexing.

Runs on Modal using the existing $30/month free tier. Budget covers ~500-1,000+ videos/month.

### Transcript Retrieval: Semantic Embeddings in Qdrant

Chosen over adding BM25 or staying with Supabase FTS keyword matching alone.

- **Problem with keyword matching**: a query like "when does the speaker explain machine learning?" won't match transcript text saying "deep learning is a subset of AI" — there is no word overlap. BM25 does not solve this; it still requires lexical overlap.
- **Solution**: embed transcript segments with the same **Qwen3-VL-Embedding-2B** model already used for frame embeddings and query embeddings. Store in the same Qdrant collection with `source="transcript"`. One query embedding → one Qdrant search returns both visual and spoken matches ranked by cosine similarity.
- **Why not BM25**: Supabase `ts_rank` already approximates BM25. Adding Elasticsearch would mean new infrastructure for a problem semantic embeddings solve more completely.
- **Supabase FTS stays** as a free, zero-GPU bonus layer for exact keyword hits (proper nouns, technical terms). It is no longer the primary spoken-content search path.
- **Cost**: ~200 text segments per video adds <1s of A10G time (~$0.0003 per video). Negligible.

### Processing Pipeline Shape

After downloading the video, the visual and spoken branches are fully independent and run in parallel via `ThreadPoolExecutor` (threads are fine since the heavy work is remote Modal calls, not local CPU):

- **Visual branch**: extract frames → embed via Qwen3-VL (A10G) → upload thumbnails → store in Qdrant.
- **Spoken branch**: extract audio (ffmpeg) → ASR via faster-whisper (T4) → embed transcript segments via Qwen3-VL (A10G) → store in Qdrant + Supabase.

For YouTube videos, caption fetch also runs in parallel as a simple network call.

At search time: embed query once → single Qdrant search returns both visual and transcript matches. Supabase FTS stays as an optional keyword boost.

## Phase 6 Product Direction

### API Billing Direction

The current web billing model is intentionally simple:

- creator-facing credit packs bought on the pricing page
- `1 credit = 1 processed video` up to the current duration limit
- search on already-processed videos is part of the product, not a separate billable event

That model works for the web app, but it is not the right long-term product shape for the public API.

API billing should differ in these ways:

- **Separate product track**: API access should not be sold as just "the same web credits through a different auth method." It should become a developer-facing offer with its own packaging and messaging.
- **Separate accounting**: API usage should be attributed to a distinct API usage ledger, even if the same account owns both web and API access. This keeps creator usage and automation usage from being mixed into one opaque balance.
- **Key-aware attribution**: usage reporting should roll up by account and by API key so teams can understand which integration is consuming spend.
- **Different packaging**: the web app can keep simple credit packs for creators, while the API can move toward a more developer-friendly model such as larger prepaid API bundles or usage-based billing once real demand is understood.
- **Separate UX**: dashboard copy, limits, and billing summaries for API users should read like a developer product, not like the current creator upload workflow.

This document deliberately does **not** pick final API price numbers yet. The goal for the next PR is to separate the product and accounting model clearly enough that pricing can change later without rewriting the onboarding story again.

## Suggested PR Sequence

The intent is that a developer can work in order, with each PR ending at a reviewable boundary and each commit still mapping cleanly to milestone progress.

### Phase 5 PR

1. Add transcript-backed or hybrid retrieval behind the existing search flow.
   Review gate: spoken-content queries produce useful timestamped results.
2. Add regression coverage so current text and image search paths still pass.
   Review gate: existing search modes still behave correctly after retrieval changes.
3. Normalize the search response shape needed by later API clients.
   Review gate: result fields are stable enough to reuse without web-specific assumptions.

### Phase 6 PR 1: API Contract

Suggested branch: `codex/feat-phase6-api-contract`
Suggested PR title: `feat(api): productize external API contract`

1. Productize the authenticated REST happy path over the same core capabilities.
2. Stabilize the external request and response shapes the CLI will depend on.
3. Keep the initial happy path focused on submit or upload, poll status, and text search end to end.
   Review gate: a stable external API flow can submit or upload a video, poll status, and run text search end to end without web-specific assumptions.

### Phase 6 PR 2: API Keys and Usage Controls

Suggested branch: `codex/feat-phase6-api-keys`
Suggested PR title: `feat(auth): add API keys and usage controls`

1. Add API keys with account ownership, scope, revocation, and quota checks.
2. Add retry-safe usage attribution and accounting rules over the stabilized API contract.
3. Keep billing and quota behavior aligned with the same ownership and idempotency guarantees already expected in the web product.
   Review gate: ownership, key lifecycle, quota enforcement, and idempotent accounting rules are covered by tests.

### Phase 6 PR 3: CLI and Agent Guide

Suggested branch: `codex/feat-phase6-cli-docs`
Suggested PR title: `feat(cli): add agent CLI and usage guide`

1. Add a thin CLI wrapper over the external API.
2. Cover the supported CLI happy path with smoke checks, including image search only if the API contract is stable enough to expose it cleanly.
3. Add a public agent-readable markdown guide for the CLI and API happy path.
   Review gate: the CLI can authenticate, submit or complete uploads, poll status, and run supported search flows without diverging from API behavior, and the markdown guide is sufficient for an agent to use the product without reading implementation code.

### Phase 6 PR 4: Dashboard API Access and Billing Split

Suggested branch: `codex/feat-phase6-api-access-billing`
Suggested PR title: `feat(api): add dashboard API access and billing separation`

1. Add a dashboard API access area where a signed-in user can create, list, and revoke API keys without needing to understand Clerk bearer tokens.
2. Add product-grade onboarding for CLI users: website copy should explain where to get a key, how CLI install works, and how API usage differs from the main web app flow.
3. Separate API billing presentation from the current web credit packs: keep creator credits as-is for the web product, but introduce a distinct API balance or API entitlement view and API usage reporting.
4. Prepare the system for different API pricing later by keeping API usage accounting and dashboard summaries separate from the current web-credit UX, even if checkout fulfillment still reuses the existing billing provider in the short term.
   Review gate: a signed-in user can buy or hold API capacity, create a key from the dashboard, follow the documented CLI install path, and complete a first upload and search flow without raw tokens or implementation-level knowledge.
