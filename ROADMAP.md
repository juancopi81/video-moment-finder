# Video Moment Finder - Roadmap

## How to Use

- This file is **future work only**.
- Execution history belongs in [STATUS.md](./STATUS.md).
- Stable product charter belongs in [PROJECT_SPEC.md](./PROJECT_SPEC.md).
- Historical roadmap snapshot: [docs/archive/roadmap/ROADMAP_2026-03-05.md](./docs/archive/roadmap/ROADMAP_2026-03-05.md)

---

## Current Phase: Phase 4 - Product Launch Hardening

**Launch Gate:** ready for paid users with security, reliability, and ops readiness.

### ~~4.3 Security Advisor Closure~~ ✅

- ~~Verify Supabase Security Advisor is clean in all deployed environments.~~
- ~~Confirm RLS coverage is enabled and policies are least-privilege for user-owned data paths.~~
- ~~Confirm all custom DB functions use explicit `search_path` and no mutable warnings remain.~~

### ~~4.4 Launch Ops Readiness~~ ✅

- ~~Confirm automated database backups and restore drill documentation.~~
- ~~Finalize monitoring dashboards and production alert routing.~~
- ~~Complete release/incident checklist for API, worker, and frontend deployments.~~
- ~~Final copy pass on public pages for pricing/support/legal consistency.~~

### 4.5 Production Deployment Guardrail

- Stop automatic deploy-to-production on every merge to `main`.
- Require explicit production promotion/manual deploy step.
- Keep feature branches + PR review as the default development flow.

---

## Next: Phase 5 - Launch and Learn

**Gate:** app is publicly shareable, stable for early users, and instrumented to learn from usage.

### ~~5.1 Analytics Baseline (Behavior, Not Just Telemetry)~~ ✅

- ~~Use existing logs + DB counts immediately to track early signal (`new users`, `videos submitted`, `jobs completed`, `searches run`).~~
- ~~Add a minimal product analytics event set in week 1:~~
  - ~~`landing_visit`~~
  - ~~`signup_complete`~~
  - ~~`video_submitted`~~
  - ~~`video_ready`~~
  - ~~`search_run`~~
  - ~~`search_success`~~
  - ~~`checkout_started`~~
  - ~~`checkout_success`~~
- ~~Keep telemetry/monitoring (Sentry) as system health, separate from product behavior analytics.~~

### ~~5.2 Discovery Basics (SEO)~~ ✅

- ~~Add/verify `robots.txt`.~~
- ~~Add/verify `sitemap.xml`.~~
- ~~Ensure public pages have clear title/meta/OG metadata and canonical URLs.~~
- ~~Add `llms.txt` for AI-crawler discoverability.~~

### 5.3 Core Feature Priorities After Launch

- Ship transcript-backed text search first (`where was this said?` queries).
- Keep UI refinements scoped to issues surfaced by early-user feedback.

---

## Later: Phase 6 - Agent Access and Scale

**Gate:** early usage validates demand and justifies broader platform investment.

### 6.1 Agents-First Integration Path

- Provide authenticated REST access for agent use cases (index video, query indexed video).
- Add per-user API keys, quotas, and key revocation.
- Add CLI/MCP wrapper after API contracts are stable.

### 6.2 Environment and Cost Maturity

- Keep lightweight infra while validating; avoid full staging complexity too early.
- Introduce separate dev/staging environments when usage, team size, or release risk justifies it.
- Continue embedding/storage cost optimization and adaptive frame-sampling evaluation as volume grows.
