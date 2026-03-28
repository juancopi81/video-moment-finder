# Video Moment Finder - Roadmap

## How to Use

- This file is **future work only**.
- Execution history belongs in [STATUS.md](./STATUS.md).
- Stable product charter belongs in [PROJECT_SPEC.md](./PROJECT_SPEC.md).
- Historical roadmap snapshot: [docs/archive/roadmap/ROADMAP_2026-03-05.md](./docs/archive/roadmap/ROADMAP_2026-03-05.md)
- Historical Phase 5/6 plan snapshot: [docs/archive/PHASE_5_6_PLAN_2026-03-23.md](./docs/archive/PHASE_5_6_PLAN_2026-03-23.md)

---

## Current Phase: Phase 7 - Acquire, Activate, Monetize

**Gate:** the product is measured clearly enough to trust funnel decisions, one ICP shows repeated activation, and early revenue signal exists without adding broad new product surface area.

- Active execution checklist: [docs/PHASE_7_PLAN.md](./docs/PHASE_7_PLAN.md)
- Selected ICP pursuit playbook: [docs/ICP_KNOWLEDGE_CREATORS_PLAYBOOK.md](./docs/ICP_KNOWLEDGE_CREATORS_PLAYBOOK.md)
- Phase 5/6 archive: [docs/archive/PHASE_5_6_PLAN_2026-03-23.md](./docs/archive/PHASE_5_6_PLAN_2026-03-23.md)

### 7.1 Funnel Instrumentation and Truth

- Add the missing anonymous and pre-auth events needed to measure real traffic and drop-off.
- Separate creator and API activation, checkout, and failure states in analytics.
- Produce one repeatable weekly funnel view used to review progress.

### 7.2 Activation and Onboarding

- Reduce signed-out friction so users can understand the value before auth.
- Keep direct upload as the only primary ingest path and keep YouTube clearly secondary.
- Improve first-search guidance, empty states, and failure-state UX.

### 7.3 ICP and Positioning

- Primary ICP: knowledge-heavy creators with owned long-form video libraries, such as course creators, coaches, consultants, and niche educators.
- Rewrite homepage and developer messaging around one job-to-be-done: find the exact teaching or explanation moment inside a private archive.
- Keep agents as an interface advantage for power users, not the whole product story.

### 7.4 Distribution and Customer Discovery

- Run manual outbound and onboarding sessions with a tight target list.
- Track acquisition source and objections for each serious user.
- Prefer niche demos and case studies over broad launch-directory posting.

### 7.5 Monetization Validation

- Validate whether the current free-to-paid path can convert once activation improves.
- Test lighter entry points only after measurement is trustworthy.
- Keep creator and developer pricing tracks separate unless evidence says otherwise.

### 7.6 Agent Productization on Evidence

- Improve the current API and CLI onboarding alongside the thin manual-auth MCP path.
- Add non-repo-clone quickstarts and end-to-end examples for REST, CLI, and the four-tool MCP flow.
- Add OAuth account linking and connector-specific docs before broad Claude promotion.

---

## Later: Phase 8 - Expand Proven Demand

**Gate:** repeated usage and early revenue justify broader platform and workflow expansion.

### 8.1 Agent Surface Expansion

- Expand beyond the thin four-tool MCP surface only after repeated end-to-end use justifies broader connector investment.
- Expand image-search and richer automation flows only after the core text-search path has traction.
- Add more developer onboarding assets when evidence shows the API wedge is working.

### 8.2 Environment and Cost Maturity

- Introduce separate staging only when release risk or team size justifies it.
- Continue embedding/storage cost optimization and adaptive frame-sampling evaluation as volume grows.
