# Phase 7 Plan - Acquire, Activate, Monetize

## Purpose

Phase 7 shifts the project from "can we build it?" to "can this become a real side project with revenue?"

The product already has the core search, API, and agent foundations. The next phase is to:

- improve funnel measurement,
- reduce activation friction,
- choose one primary ICP,
- run direct customer discovery,
- and validate the first repeatable revenue signal.

## Phase Thesis

The current bottleneck is not missing infrastructure. It is low traffic, incomplete funnel truth, and too much onboarding friction before a user sees value.

Agents remain important, but they should be treated as a distribution and workflow advantage layered on top of a clear core use case, not as the only product story.

## Exit Gate

- [ ] Funnel measurement is trustworthy enough to explain where users drop.
- [ ] At least 10 activated users complete the full flow: signup -> video submitted -> video ready -> search run.
- [ ] At least 3 users complete a paid checkout or Developer Pack purchase.
- [ ] At least 10 live onboarding or user-discovery sessions are completed and documented.
- [ ] One primary ICP is chosen based on evidence and reflected in the homepage and developer messaging.
- [ ] At least 5 successful end-to-end API or agent sessions are completed by real users.

## Baseline Snapshot (2026-03-23)

- [X] Phase 5 and Phase 6 implementation work is complete and archived.
- [X] Core product paths exist across web app, `/api/v1`, CLI, API keys, `/developers`, and `/skill.md`.
- [X] Initial funnel review was completed manually.
- [X] Known baseline: 49 landing visits, 3 signups, 3 video submitters, 2 users reaching successful search, 1 checkout start, 1 active API key.
- [ ] Replace the manual snapshot with a repeatable weekly funnel report.

## 7.1 Funnel Instrumentation and Truth

Goal: trust the numbers before making more growth bets.

- [X] Audit the current analytics implementation and identify missing events.
- [X] Record anonymous `landing_visit` from the signed-out homepage. (PostHog auto-pageview)
- [X] Record CTA click events for hero, pricing, developers, and sign-in entry points. (PostHog `cta_click` on homepage and pricing CTAs; developers page not yet instrumented)
- [X] Record upload init, upload complete, upload failure, and processing failure states. (PostHog client events + backend `processing_failure`)
- [X] Record empty-search-result states separately from successful search states. (Existing `search_success` with `result_count` metadata)
- [ ] Record checkout cancel and checkout failure states for both creator and API billing flows. (PostHog `checkout_started_client` tracks checkout starts; cancel and failure events not yet captured because Lemon Squeezy does not surface them)
- [X] Add acquisition-source attribution so serious users can be traced back to channel or campaign. (PostHog auto-captures UTM params)
- [X] Create one repeatable weekly funnel query or dashboard view. (Saved as "Weekly Acquisition Funnel" in PostHog)

Review gate:
Can answer "where are users dropping?" from one repeatable report instead of manual inspection.

## 7.2 Activation and Onboarding

Goal: let users see value faster and remove avoidable friction.

- [X] Add a signed-out demo path so users can see real output before auth. (Static preview card with mock visual and spoken results)
- [X] Keep direct upload as the only primary ingest CTA on the homepage. (YouTube demoted to secondary text link)
- [X] Move YouTube import behind a clearly labeled secondary or beta path. (Secondary text link, not equal tab)
- [X] Add first-search suggestions once a video is ready. (4 ICP-relevant suggestion chips below text search input)
- [X] Improve empty states for "what should I search for?" (Empty-results block with refinement guidance, dashboard empty state with ICP copy)
- [X] Improve error and recovery copy for upload, processing, and search failure states. (Processing failed card, upload error card, search error hint)
- [ ] Review the full onboarding flow with at least 3 live users and fix the biggest friction points.

Review gate:
A new user can understand the product, upload a video, and run a first search with less confusion and fewer dead ends.

## 7.3 ICP and Positioning

Goal: stop speaking to everyone at once.

- [X] Choose one primary ICP for the next cycle.
- [X] Write a one-sentence positioning statement for that ICP.
- [ ] Rewrite homepage hero and supporting copy around that job-to-be-done.
- [ ] Rewrite `/developers` around the best agent or API use case instead of generic API access.
- [ ] Add 3 concrete use cases and sample queries tied to the chosen ICP.
- [X] Decide explicitly whether agents/API is the primary acquisition wedge or a secondary expansion path.

Selected ICP:
knowledge-heavy creators with owned long-form video libraries, especially course creators, coaches, consultants, and niche educators.

Positioning statement:
Video Moment Finder helps knowledge-heavy creators search their private video archives and instantly find the exact teaching moment they need.

Wedge decision:
Lead with the web product and direct archive search outcome. Treat agents and API as a power-user workflow advantage, not the primary acquisition story.

Execution playbook:
[docs/ICP_KNOWLEDGE_CREATORS_PLAYBOOK.md](./ICP_KNOWLEDGE_CREATORS_PLAYBOOK.md)

Review gate:
The product can be described in one sentence, for one user, with one clear job-to-be-done.

## 7.4 Manual Distribution and Customer Discovery

Goal: replace broad posting with tighter outbound and direct feedback loops.

- [ ] Build a target list of 30 prospects in the chosen ICP.
- [ ] Run 10 live onboarding or discovery sessions.
- [ ] Send focused outbound with one short message, one demo, and one CTA.
- [ ] Publish 3 niche demos or case studies tailored to the chosen ICP.
- [ ] Track acquisition source, objections, and reasons for non-conversion for each serious lead.
- [ ] Review weekly which channel produces the best activation, not just the most traffic.

Review gate:
There is a clear list of objections, activation blockers, and acquisition channels ranked by quality.

## 7.5 Monetization Validation

Goal: learn whether the current pricing and purchase timing can convert.

- [ ] Audit the current free-to-paid path and document where trust, price, or timing blocks conversion.
- [ ] Decide whether to test a lower-friction paid entry point.
- [ ] Decide whether the API should get a free sandbox or trial allocation.
- [ ] Add clear purchase triggers at moments of real product value.
- [ ] Review whether creator pricing and developer pricing should stay separate.
- [ ] Define the minimum revenue signal that justifies further expansion work.

Review gate:
There is a documented revenue hypothesis with a measured conversion path, not just a pricing page.

## 7.6 Agent Productization

Goal: improve the agent path without outrunning demand.

- [X] Ship the API, API keys, CLI, `/developers`, and `/skill.md`.
- [ ] Add a copy-paste `curl` quickstart for users who did not clone the repo.
- [ ] Add one canonical end-to-end agent example from prompt to result.
- [ ] Gather 5 successful real-user API or agent sessions.
- [ ] Review support burden and setup friction for the current agent flow.
- [ ] Keep MCP out of scope unless the current API or CLI path shows repeated successful use.

Review gate:
Real users can complete the agent or API happy path without repo-specific knowledge, and the setup friction is understood.

## Out of Scope for Phase 7

- Major model swaps without evidence that retrieval quality is the current bottleneck.
- Broad feature expansion unrelated to acquisition, activation, or monetization.
- Full team or workspace collaboration features.
- Staging-environment expansion unless release risk clearly justifies it.
