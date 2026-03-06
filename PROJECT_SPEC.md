# Video Moment Finder

> One-line pitch: Process a video and find moments using semantic search.

## Document Boundaries

- **`PROJECT_SPEC.md`**: Stable product charter (vision, users, scope, constraints, metrics, risks, high-level architecture).
- **`ROADMAP.md`**: Future work only.
- **`STATUS.md`**: Execution history only.
- **`docs/archive/`**: Historical research and retired planning snapshots.
- **`AGENTS.md`**: Canonical repository workflow for coding agents.

## Goals

- Build a paid SaaS product for semantic video moment search.
- Learn and apply multimodal embeddings in a production-style system.
- Keep the architecture reliable enough for paid-user expectations.

## Problem

Finding specific moments inside long videos is slow and manual.
Traditional transcript search misses visual-only content.

Opportunity: map frames and queries into a shared embedding space so users can search by meaning, not only exact words.

## Users

- **Primary**: creators/editors searching their own footage.
- **Secondary**: researchers analyzing long-form video.
- **Tertiary**: anyone trying to locate moments quickly in long videos.

## Product Inputs

- Video source: YouTube URL or direct upload.
- Query source: text and uploaded example images.

## Product Outputs

- Top timestamped matches for a query.
- Relevance scores.
- Jump-to-moment playback links.

## MVP Scope

### In Scope

- Video processing and indexing pipeline for submitted videos.
- Semantic retrieval over indexed frames.
- Timestamped results with thumbnails.
- Authenticated user workflows.
- Credit-based billing model.

### Out of Scope (Current)

- Team collaboration/workspaces.
- Public API productization.
- Full video editing/export feature set.

## Constraints

- Cost discipline for GPU-heavy processing.
- Reliable async processing path (queue + worker).
- Security and ownership boundaries for user data.
- Launch-phase operational readiness (monitoring, backups, incident response).

## High-Level Architecture

```text
+-------------------+    +----------------+    +----------------+    +-----------+
| Next.js Frontend | -> | FastAPI API    | -> | Queue Worker   | -> | Modal GPU |
+-------------------+    +----------------+    +----------------+    +-----------+
          |                      |                     |
          v                      v                     v
     +----------+          +-----------+         +-------------+
     | Supabase |          | Qdrant    |         | Cloudflare  |
     | DB/Jobs  |          | Vectors   |         | R2 Storage  |
     +----------+          +-----------+         +-------------+
```

## Monetization

- Credit-based model with free trial allowance.
- Paid plans fund GPU processing and storage costs.
- Unit economics target: keep processing cost materially below per-video revenue.

## Success Metrics

| Signal          | Target  | Notes |
| --------------- | ------- | ----- |
| Search quality  | >70%    | Relevant result near the top for benchmark queries. |
| Processing time | <20 min | For a 30-minute video under baseline settings. |
| Conversion      | >5%     | Free trial to paid conversion indicator. |
| Cost per video  | <$1     | GPU plus storage operating cost envelope. |

## Risks & Unknowns

- Video-source reliability constraints (for example download restrictions).
- Query quality variance across different content styles.
- Cost/latency tradeoffs as usage scales.
- Billing and operational edge cases during launch hardening.
