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
| `CORS_ALLOWED_ORIGINS`, `CORS_ALLOWED_ORIGIN_REGEX` | Required | - | - | API CORS policy only. |
| `LEMON_SQUEEZY_API_KEY`, `LEMON_SQUEEZY_STORE_ID`, `LEMON_SQUEEZY_VARIANT_ID_STARTER`, `LEMON_SQUEEZY_VARIANT_ID_PRO`, `LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL`, `LEMON_SQUEEZY_CHECKOUT_TEST_MODE`, `LEMON_SQUEEZY_WEBHOOK_SECRET`, `BILLING_GRANT_EVENT_NAMES` | Required | - | - | API billing checkout and webhook handling. |
| `RATE_LIMIT_*` | Optional | - | - | API rate limit tuning. |
| `VIDEO_MAX_FREE_VIDEOS`, `VIDEO_UPLOAD_URL_TTL_S`, `VIDEO_SOURCE_URL_TTL_S` | Optional | - | - | API admission and signed URL behavior. |
| `VIDEO_MAX_DURATION_S` | Optional | Optional | - | Duration checks are used in API admission and processing path validation. |
| `VIDEO_JOB_MAX_ATTEMPTS`, `VIDEO_JOB_STALE_LOCK_TIMEOUT_S`, `VIDEO_JOB_IDLE_BACKOFF_MAX_S`, `VIDEO_JOB_DB_RETRY_BASE_DELAY_S`, `VIDEO_JOB_DB_RETRY_MAX_DELAY_S` | - | Optional | - | Worker queue behavior tuning. |
| `NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | - | - | Required | Frontend runtime config in Vercel/frontend env. |

### Deployment Placement

- Railway API service: API-required groups + shared API/worker groups.
- Railway worker service: worker-required groups + shared API/worker groups.
- Vercel frontend: `NEXT_PUBLIC_*` variables only.
- Packaging note: Railway installs the default shared runtime dependencies from `pyproject.toml`, while the Modal image installs the additional `modal` dependency group. After merging dependency-group changes or renaming Modal objects, redeploy Modal with `uv run modal deploy src/embedding/modal_app.py`.

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

## Upload Flow Contract

Preferred large-file flow (direct-to-R2):

1. `POST /videos/upload/init` with auth + filename/content type.
2. `PUT` binary payload to returned presigned `upload_url`.
3. `POST /videos/upload/complete` with auth + `video_id` + filename.

Small-file convenience flow:

- `POST /videos/upload` (multipart file upload).

Notes:

- Upload endpoints enforce admission checks and duration validation.
- Missing R2 configuration or failed storage verification returns `503`.
- Missing uploaded object on complete returns `400`.

## Quick Troubleshooting

- `Billing webhook is not configured` -> set `LEMON_SQUEEZY_WEBHOOK_SECRET` in API service.
- `Upload storage is not configured` -> set R2 variables in API and worker services.
- Modal auth failures -> set both `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in both services.
