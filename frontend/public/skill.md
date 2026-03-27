---
name: video-moment-finder
description: Public agent bootstrap for indexing a video and searching it by text with Video Moment Finder.
homepage: https://www.videomomentfinder.com
api_host: https://api.videomomentfinder.com
openapi: https://api.videomomentfinder.com/openapi.json
---

# Video Moment Finder

Use this file when you need one public entrypoint for the Video Moment Finder API and CLI.
This is the current agent-ready REST API and CLI surface. It is not an MCP server.

## Canonical URLs

- Site: `https://www.videomomentfinder.com`
- Developers overview: `https://www.videomomentfinder.com/developers`
- Public skill file: `https://www.videomomentfinder.com/skill.md`
- API host: `https://api.videomomentfinder.com`
- Public REST prefix: `https://api.videomomentfinder.com/api/v1`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`

## Supported Happy Path

- create or reuse a `vmf_` API key
- upload a video with the one-shot multipart route
- poll until processing reaches `ready`
- search a processed video by text

## Not The Primary Documented Agent Flow

- image search
- YouTube submit

Those capabilities may exist elsewhere in the product or API surface, but the canonical documented agent flow is one-shot upload plus text search.
They are intentionally excluded from the public OpenAPI schema.

## Security Rules

- Only send `vmf_` API keys to `https://api.videomomentfinder.com`.
- Only send temporary Clerk bearer tokens to Video Moment Finder API routes that explicitly require them, such as `POST /api/v1/keys`.
- If you choose the lower-level direct-upload flow, do not forward your `Authorization` header to the returned presigned `upload_url`.
- Treat the raw API key returned by key creation as secret material. It is returned once.

## Fastest Path

1. If you already have a `vmf_` API key, use it.
2. Otherwise create one from `https://www.videomomentfinder.com/dashboard/api`.
3. Use the CLI if you already cloned this repository; otherwise call the REST API directly.

## REST Happy Path

All authenticated API requests use:

```text
Authorization: Bearer <token>
```

Use the `/api/v1` routes for the public contract:

1. Upload one video with the one-shot multipart route.
   The response includes the `id` you use for status polling and search.

```bash
curl -X POST https://api.videomomentfinder.com/api/v1/videos/upload \
  -H "Authorization: Bearer <vmf_api_key>" \
  -H "Idempotency-Key: sample-v1" \
  -F "file=@sample.mp4;type=video/mp4"
```

2. Poll status until `status` is `ready`:

```http
GET https://api.videomomentfinder.com/api/v1/videos/<video_id>
Authorization: Bearer <vmf_api_key>
```

3. Search by text:

```http
POST https://api.videomomentfinder.com/api/v1/videos/<video_id>/search
Content-Type: application/json
Authorization: Bearer <vmf_api_key>

{"query_text":"when do they explain the model?","limit":3}
```

Lower-level direct upload is still available if your client specifically needs a presigned URL flow:

1. `POST /api/v1/videos/upload/init`
2. `PUT` the raw bytes to the returned `upload_url`
3. `POST /api/v1/videos/upload/complete`

## CLI Happy Path

If you already cloned this repository, the CLI wraps the same `/api/v1` flow:

```bash
uv sync
uv run vmf --version
uv run vmf auth set \
  --api-base-url https://api.videomomentfinder.com \
  --api-key vmf_YOUR_KEY
uv run vmf videos upload ./sample.mp4
uv run vmf videos list
uv run vmf videos wait <video_id>
uv run vmf videos search <video_id> --query-text "when do they explain the model?"
```

Retry-safe upload:

```bash
uv run vmf videos upload ./sample.mp4 --idempotency-key sample-v1
```

Upload from stdin:

```bash
cat sample.mp4 | uv run vmf videos upload - \
  --filename sample.mp4 \
  --content-type video/mp4
```

The CLI keeps stdout machine-readable:

- success payloads are JSON on stdout
- dry runs are JSON on stdout
- prompts and errors go to stderr

Destructive commands are safe by default:

- `uv run vmf auth clear --dry-run`
- `uv run vmf auth clear --yes`
- `uv run vmf keys revoke <key_id> --dry-run`
- `uv run vmf keys revoke <key_id> --yes --bearer-token <clerk-token>`

## References

- Developers overview: `https://www.videomomentfinder.com/developers`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`
- Pricing and API key dashboard: `https://www.videomomentfinder.com/dashboard/api`
- Support: `https://www.videomomentfinder.com/support`
