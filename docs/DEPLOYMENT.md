# Deployment and Operations Reference

This file is the concise operations reference that was intentionally removed from `README.md`.

## Source of Truth for Environment Variables

- Backend and infrastructure variables: `.env.example`
- Frontend variables: `frontend/.env.example`

Use those files as the canonical variable list and defaults. This document explains **where each group belongs**.

## Environment Ownership by Service

| Variable Group | API Service | Worker Service | Frontend | Notes |
| --- | --- | --- | --- | --- |
| `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `SUPABASE_DB_URL` | Required | Required | - | Database and Supabase API access. |
| `QDRANT_URL`, `QDRANT_API_KEY` | Required | Required | - | Query path uses API; indexing path uses worker. |
| `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME`, `R2_PUBLIC_URL` | Required | Required | - | API handles upload/presign; worker handles processing outputs. |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | Required | Required | - | Required for Modal calls from both services. |
| `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_RELEASE` | Optional | Optional | - | Runtime monitoring for API and worker. |
| `CLERK_ISSUER`, `CLERK_AUDIENCE`, `CLERK_JWKS_URL` | Required | - | - | API JWT verification only. |
| `CLERK_SECRET_KEY`, `API_TRIAL_ENABLED`, `API_TRIAL_UNITS` | Optional | - | - | Server-verified, once-per-account trial. Grants default OFF; secret required only to enroll new eligible accounts. |
| `CORS_ALLOWED_ORIGINS`, `CORS_ALLOWED_ORIGIN_REGEX` | Required | - | - | API CORS policy only. Include Claude web origins and localhost callback origins for MCP browser auth. |
| `FRONTEND_BASE_URL`, `MCP_OAUTH_ISSUER_URL`, `MCP_OAUTH_RESOURCE_URL`, `MCP_OAUTH_CLIENT_ID`, `MCP_OAUTH_CLIENT_SECRET` | Required | - | - | Claude connector OAuth issuer, protected resource metadata, DCR support, optional static reviewer client validation, and approval-page redirects. |
| `LEMON_SQUEEZY_API_KEY`, `LEMON_SQUEEZY_STORE_ID`, `LEMON_SQUEEZY_VARIANT_ID_STARTER`, `LEMON_SQUEEZY_VARIANT_ID_PRO`, `LEMON_SQUEEZY_VARIANT_ID_DEVELOPER`, `LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL`, `LEMON_SQUEEZY_CHECKOUT_TEST_MODE`, `LEMON_SQUEEZY_WEBHOOK_SECRET`, `BILLING_GRANT_EVENT_NAMES`, `API_UNIT_COST_INDEX_VIDEO`, `API_UNIT_COST_TEXT_QUERY` | Required | - | - | API billing checkout, webhook handling, and API unit pricing. |
| `API_UNIT_COST_TRANSCRIPT_FETCH`, `API_UNIT_COST_FRAMES_THUMB`, `API_UNIT_COST_FRAMES_HIGH` | Optional | - | - | API unit pricing for transcript fetch and frame retrieval (defaults 1, 1, 5). |
| `FRAMES_FFMPEG_MAX_CONCURRENCY` | Optional | - | - | Process-wide cap on concurrent high-res frame ffmpeg extractions (default 4), independent of the per-request thread pool. |
| `RATE_LIMIT_*` | Optional | - | - | API rate limit tuning; the search limiter also covers transcript fetch and frame retrieval. |
| `VIDEO_MAX_FREE_VIDEOS`, `VIDEO_UPLOAD_URL_TTL_S`, `VIDEO_SOURCE_URL_TTL_S` | Optional | - | - | API admission and signed URL behavior. |
| `VIDEO_MAX_DURATION_S` | Optional | Optional | - | Duration checks are used in API admission and processing path validation. |
| `VIDEO_MAX_UPLOAD_BYTES` | Optional | - | - | API-only: enforced at multipart upload (mid-stream) and presigned upload completion (HEAD size check), before the video is admitted for processing. Not read by the worker. |
| `VIDEO_JOB_MAX_ATTEMPTS`, `VIDEO_JOB_STALE_LOCK_TIMEOUT_S`, `VIDEO_JOB_IDLE_BACKOFF_MAX_S`, `VIDEO_JOB_DB_RETRY_BASE_DELAY_S`, `VIDEO_JOB_DB_RETRY_MAX_DELAY_S` | - | Optional | - | Worker queue behavior tuning. |
| `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | - | - | Required | Frontend runtime config in Vercel/frontend env. |
| `NEXT_PUBLIC_POSTHOG_KEY`, `NEXT_PUBLIC_POSTHOG_HOST` | - | - | Required | PostHog client-side product analytics. |

### Deployment Placement

- Railway API service: API-required groups + shared API/worker groups.
- Railway worker service: worker-required groups + shared API/worker groups.
- Vercel frontend: `NEXT_PUBLIC_*` variables only.
- Packaging note: Railway installs the default shared runtime dependencies from `pyproject.toml`, while the service image must also include `ffmpeg` plus a JavaScript runtime for best-effort `yt-dlp` YouTube import. The Modal image installs the additional `modal` dependency group. After merging dependency-group changes or renaming Modal objects, redeploy Modal with `uv run modal deploy src/embedding/modal_app.py`.

## Worker Database Recovery

The continuous worker loop retries Supabase transport failures and PostgREST HTTP
errors `429`, `500`, `502`, `503`, and `504`. It resets the database client and
uses exponential backoff controlled by `VIDEO_JOB_DB_RETRY_BASE_DELAY_S` (default
1 second) and `VIDEO_JOB_DB_RETRY_MAX_DELAY_S` (default 30 seconds). A successful
queue iteration resets the retry delay. Other API errors still propagate so
authentication, schema, and query failures remain visible. Restarting the worker
does not resolve an upstream database outage; these retries keep it running until
requests succeed again.

## Modal Deploy-Time Variables

- `MODAL_QUERY_EMBED_MIN_CONTAINERS` and `MODAL_QUERY_EMBED_MAX_CONTAINERS` are optional deploy-time knobs for the Modal app.
- They are not Railway or Vercel service variables, so they are intentionally not listed in the service ownership table above.

## Billing Webhook Contract (Lemon Squeezy)

- Endpoint: `POST /webhooks/lemonsqueezy`
- Signature header: `X-Signature` (HMAC-SHA256 of raw request body).
- Secret env: `LEMON_SQUEEZY_WEBHOOK_SECRET`
- Grant events default: `order_created`, `subscription_payment_success`
- Override events with: `BILLING_GRANT_EVENT_NAMES`
- Credit metadata source: `meta.custom_data.user_id`, `meta.custom_data.credits`
- Idempotency key:
  - Primary: `<event_name>:<data.id>`
  - Fallback: `<event_name>:sha256:<raw_payload_hash>`

Expected behavior:

- Missing/invalid signature -> `401`
- Invalid JSON payload -> `400`
- Event outside configured grant set -> ignored response (`processed=false`)
- Duplicate event -> idempotent no-op (`granted=false`)

## Analytics Event Contract

Client-side product analytics (pageviews, CTA clicks, upload lifecycle, checkout tracking, UTM attribution) are handled by PostHog via `posthog-js` in the frontend. PostHog is initialized in `frontend/src/components/posthog-provider.tsx` and gracefully disabled when `NEXT_PUBLIC_POSTHOG_KEY` is unset.

Vercel Web Analytics continues to run alongside PostHog for lightweight traffic data in the Vercel dashboard.

Server-side product events use a first-party Supabase table:

- Endpoint: `POST /analytics/event`
- Body: `{"event_name": "...", "metadata": {...}}`
- Allowed frontend events: `signup_complete`
- Auth: required
- Table: `analytics_events` in Supabase (service_role only RLS)
- Backend events (`video_submitted`, `video_ready`, `processing_failure`, `search_run`, `search_success`, `checkout_started`, `checkout_success`) are inserted directly by API and worker handlers.

Example verification query:
```sql
SELECT event_name, count(*), count(distinct user_id) FROM analytics_events GROUP BY 1;
```

## Upload Flow Contract

Preferred large-file flow (direct-to-R2, reliable production ingest path):

1. `POST /videos/upload/init` with auth + filename/content type.
2. `PUT` binary payload to returned presigned `upload_url`.
3. `POST /videos/upload/complete` with auth + `video_id` + filename.

Small-file convenience flow:

- `POST /videos/upload` (multipart file upload).

Notes:

- Upload endpoints enforce admission checks and duration validation.
- Missing R2 configuration or failed storage verification returns `503`.
- Missing uploaded object on complete returns `400`.

## VMF Connector OAuth Contract

- Protected resource endpoint: `https://api.videomomentfinder.com/mcp`
- OAuth discovery endpoints:
  - `GET /.well-known/oauth-authorization-server`
  - `GET /.well-known/oauth-protected-resource/mcp`
- OAuth transaction endpoints:
  - `GET|POST /authorize`
  - `POST /register`
  - `POST /token`
  - `POST /revoke`
- Frontend approval page:
  - `https://www.videomomentfinder.com/connectors/claude?request_id=...`
- Approved redirect URIs:
  - `https://claude.ai/api/mcp/auth_callback`
  - `https://claude.com/api/mcp/auth_callback`
  - `http://localhost:6274/oauth/callback`
  - `http://localhost:6274/oauth/callback/debug`

Behavior notes:

- `/mcp` only accepts OAuth bearer tokens. Legacy `vmf_` API keys remain valid for REST and CLI, not for MCP.
- Connector usage bills against the shared API unit balance (existing paid units plus any trial grant) and records `api_usage_events.api_key_id = null`.
- The connect page blocks approval when `api_units_balance <= 0`, explains the unavailable operation neutrally, and offers denial. It does not advertise or link to digital-credit purchases. Ordinary website checkout remains independent.
- `HEAD /mcp` must stay tokenless for Claude client compatibility checks.
- Public clients registered with `token_endpoint_auth_method=none` may omit
  `client_secret` on token and revocation requests. The authenticator supplies an
  empty form field after public-client validation for MCP SDK 1.26's revocation
  model; confidential-client secret checks and token ownership checks still apply.
  Revoking either token invalidates both access and refresh tokens for that connection.

## Verified-account trial (inactive proposal)

Apply `20261002120000_api_billing_retry_safety.sql` and
`20261002121000_verified_account_trial.sql` before deploying the updated API.
These migrations create schema/functions only and grant no trial units. Keep
`API_TRIAL_ENABLED=false` until activation is separately approved. No production
migration, activation, or live grant is part of the implementation validation.

With grants enabled, an authenticated account is enrolled on a billing summary,
connector approval, indexing admission, or metered API operation. The backend
fetches that same immutable Clerk user ID from `https://api.clerk.com/v1/users/`
using the server-only `CLERK_SECRET_KEY`. Only a verified **primary** email on an
unlocked, unbanned account qualifies. Client booleans, editable metadata, and an
unverified primary email with a verified secondary email do not qualify. Clerk
unavailability grants nothing; paid access and the existing unverified-account
website allowance remain usable. Only the opaque email ID is recorded, not its
address. This is once per Clerk account, not a claim of one trial per human.

`API_TRIAL_UNITS` starts at **600**. A permanent `api_trial_grants.user_id` primary
key and one database transaction serialize competing requests, add the grant to
`api_credits`, and record the grant event. Reconnects, repeated requests, changed
configuration, and a spent balance cannot grant again. `API_TRIAL_ENABLED` controls
**new** grants only: disabling it preserves existing enrollment and balances,
and cannot revive a second legacy free-video allowance. Deleting grant records
would break this protection and is not a rollback strategy.

The website's old free indexing allowance and the new trial do not stack. Prior
nonfailed videos without an API indexing debit are conservatively treated as
legacy free usage, up to `VIDEO_MAX_FREE_VIDEOS`. Their offset uses the configured
`API_UNIT_COST_INDEX_VIDEO`, capped at the grant: with default pricing, one prior
free video reduces a 600-unit grant to **100**. Failed videos, explicitly
API-funded videos, and unlimited-access accounts are excluded from this inference.
Historical API indexing also sometimes omitted its video ID. Those unattributed
debits conservatively exclude the same number of otherwise unmatched uploads from
the free-usage inference. Historical website debits were not recorded, so
imported/manual videos and past changes to the free quota cannot be distinguished
perfectly. Review those cohorts
before activation; this migration does not rewrite their balances or assume all
existing videos were free. Grant reconciliation runs before a new video insert.

For enrolled accounts, website indexing atomically uses enough shared API units
first, otherwise one existing paid website processing credit. Concurrent retries
for the same video charge once. A queued row alone is not proof of a charge:
trial/API retries with no committed debit return a neutral `409` while admission
is pending, instead of starting unbilled work. An upload with a proven debit whose
enqueue failed can retry without another balance precheck or charge. New API
indexing debits retain the video ID and a stable request ID. Paid API balances, paid website balances and
checkout, and unlimited-access overrides are preserved. Existing website search
remains free; the trial does not silently change that entitlement. API/MCP search,
transcript, and frame calls keep the configured unit costs. Listing and status
remain free. At default costs, 600 units cover one 500-unit index and **up to 100
units of retrieval**; this is tariff arithmetic, not a measured infrastructure
cost or a promise of 100 complete learning outputs.

Both billing summaries add `trial_enabled`, `trial_status`,
`trial_allowance_units`, `trial_units_granted`, and `trial_legacy_units_offset`.
Statuses are `disabled`, `verification_required`, `verification_unavailable`,
`granted`, and `exhausted`. The latter two describe an existing grant and whether
the **combined** API balance is positive/zero; historical granted units are not a
separate remaining-trial balance. The web summary also exposes `api_units_balance`
and `unit_cost_index_video`; enrolled accounts report zero legacy free quota.

The billing reliability migration additionally fixes failed-call compensation:
refunds now reference a matching original debit, use a distinct refund key, and
cannot exceed or duplicate that debit or cross account boundaries. Compensated
requests need a new request ID when retried. No existing ledger rows are changed
or retroactively refunded.

Isolated validation (never uses `SUPABASE_DB_URL`):

```bash
docker pull postgres:16-alpine
VMF_RUN_POSTGRES_TESTS=1 uv run pytest -q tests/db/test_trial_postgres.py
uv run pytest -q tests/billing/test_trial.py tests/api/test_api_trial.py
```

The PostgreSQL tests create a disposable container with no network and no exposed
ports, apply every repository migration, exercise real concurrent transactions,
and remove the container. Ordinary tests skip this opt-in integration suite.

## Quick Troubleshooting

- `Billing webhook is not configured` -> set `LEMON_SQUEEZY_WEBHOOK_SECRET` in API service.
- `Upload storage is not configured` -> set R2 variables in API and worker services.
- Modal auth failures -> set both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in both services.
- `MCP OAuth is not configured` -> set `FRONTEND_BASE_URL`, `MCP_OAUTH_ISSUER_URL`, `MCP_OAUTH_RESOURCE_URL`, `MCP_OAUTH_CLIENT_ID`, and `MCP_OAUTH_CLIENT_SECRET` on the API service.
- YouTube metadata or import fails with bot/sign-in challenges -> direct upload is the supported reliable path. Use the `yt-dlp` deploy/runtime notes only when explicitly debugging the best-effort YouTube import path.
