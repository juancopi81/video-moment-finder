# Video Moment Finder - Roadmap

## How to Use

- This file is **future work only**.
- Execution history belongs in [STATUS.md](./STATUS.md).
- Stable product charter belongs in [PROJECT_SPEC.md](./PROJECT_SPEC.md).
- Historical roadmap snapshot: [docs/archive/roadmap/ROADMAP_2026-03-05.md](./docs/archive/roadmap/ROADMAP_2026-03-05.md)

---

## Current Phase: Phase 4 - Product Launch Hardening

**Launch Gate:** ready for paid users with security, reliability, and ops readiness.

### 4.3 Security Advisor Closure

- Verify Supabase Security Advisor is clean in all deployed environments.
- Confirm RLS coverage is enabled and policies are least-privilege for user-owned data paths.
- Confirm all custom DB functions use explicit `search_path` and no mutable warnings remain.

### 4.4 Launch Ops Readiness

- Confirm automated database backups and restore drill documentation.
- Finalize monitoring dashboards and production alert routing.
- Complete release/incident checklist for API, worker, and frontend deployments.
- Final copy pass on public pages for pricing/support/legal consistency.

---

## Next: Phase 5 - Search Quality and Product UX

**Gate:** users can find moments with stronger precision and richer query options.

### 5.1 Search Quality

- Add image-query search path (reference-image to frame retrieval).
- Introduce optional reranking step for top-k candidate results.
- Add query quality benchmark set and periodic regression checks.

### 5.2 Retrieval UX

- Add richer filtering/sorting for results (score and timestamp windows).
- Improve result explanations (why a moment matched).
- Add better empty-state guidance for weak/ambiguous queries.

### 5.3 Processing Experience

- Improve user-facing progress visibility for long processing jobs.
- Add notification path for job completion (email or in-app).

---

## Later: Phase 6 - Growth and Platform Extensions

**Gate:** expansion features are validated against cost and reliability constraints.

### 6.1 Product Expansion

- Evaluate support for videos beyond current upload/YouTube flow.
- Evaluate clip export/sharing workflows for selected moments.
- Evaluate collaboration features for multi-user workspaces.

### 6.2 Platform and Cost

- Optimize embedding/storage cost controls for larger usage.
- Evaluate adaptive frame sampling strategies by content type.
- Add monthly cost/performance report for capacity planning.
