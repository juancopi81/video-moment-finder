# Phase 5-6 Plan

## Purpose

Phase 5 and Phase 6 should be treated as one staged expansion:

- strengthen the core search product first,
- then expose the same capabilities through an external API,
- add MCP or CLI only after the API contract is stable.

The web app remains the primary product. Agent access is an additional interface, not a separate product path.

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
- Keep room for separate API pricing and usage accounting if later usage patterns justify a different billing model.
- Add MCP or CLI only after the API surface is stable enough to wrap cleanly.
- Add agent-facing docs and discovery only after the interface is stable enough to describe clearly.

Suggested milestone gates:

1. Account ownership, key scope, and revocation are covered by tests.
2. Usage is attributed correctly and retries do not create double billing.
3. The API can process a video and run search end-to-end in a reviewable happy path.
4. MCP or CLI smoke checks are added only after the API milestone is stable.
5. An agent can discover the supported interface and complete a documented happy path without reading implementation code.

## Delivery Shape

- Default target: one focused PR for the main Phase 5 work and one focused PR for the main Phase 6 work.
- Inside each PR, milestone commits should land at clear review checkpoints.
- If a phase grows too large or gets blocked, split at a milestone boundary instead of mixing partial work into one PR.
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

## Suggested Commit Sequence

The intent is that a developer can work in order and stop after any milestone commit for review.

### Phase 5 PR

1. Add transcript-backed or hybrid retrieval behind the existing search flow.
   Review gate: spoken-content queries produce useful timestamped results.
2. Add regression coverage so current text and image search paths still pass.
   Review gate: existing search modes still behave correctly after retrieval changes.
3. Normalize the search response shape needed by later API clients.
   Review gate: result fields are stable enough to reuse without web-specific assumptions.

### Phase 6 PR

1. Add authenticated REST endpoints over the same core capabilities.
   Review gate: a happy-path API flow can process a video and run search.
2. Add API keys with account ownership, scope, revocation, and quota checks.
   Review gate: ownership and key lifecycle rules are covered by tests.
3. Add usage attribution rules that avoid double billing on retries.
   Review gate: usage accounting is correct for normal calls and idempotent retries.
4. Add MCP or CLI only if the API contract is stable after the earlier milestones.
   Review gate: wrapper smoke checks pass without adding new product-only logic.
5. Add agent-facing docs and discovery updates only after the earlier milestones are stable.
   Review gate: `llms.txt` and a public `/skill.md` or equivalent agent-onboarding doc match the real interface and support a documented happy path.
