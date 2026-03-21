---
name: video-moment-finder
description: Public agent bootstrap for indexing a video and searching it by text with Video Moment Finder.
homepage: https://www.videomomentfinder.com
api_host: https://api.videomomentfinder.com
openapi: https://api.videomomentfinder.com/openapi.json
---

# Video Moment Finder

Use this file when you need one public entrypoint for the Video Moment Finder API and CLI.

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
- upload a video with the direct upload flow
- poll until processing reaches `ready`
- search a processed video by text

## Not The Primary Documented Agent Flow

- image search
- YouTube submit

Those capabilities may exist elsewhere in the product or API surface, but the canonical documented agent flow is direct upload plus text search.

## Security Rules

- Only send `vmf_` API keys to `https://api.videomomentfinder.com`.
- Only send temporary Clerk bearer tokens to Video Moment Finder API routes that explicitly require them, such as `POST /api/v1/keys`.
- Do not forward your `Authorization` header when uploading file bytes to the returned presigned `upload_url`.
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

1. Initialize a direct upload:

```http
POST https://api.videomomentfinder.com/api/v1/videos/upload/init
Content-Type: application/json
Authorization: Bearer <vmf_api_key>

{"filename":"sample.mp4","content_type":"video/mp4"}
```

2. `PUT` the raw file bytes to the returned `upload_url`.
   Do not send your VMF auth header to that storage URL.

3. Finalize the upload:

```http
POST https://api.videomomentfinder.com/api/v1/videos/upload/complete
Content-Type: application/json
Authorization: Bearer <vmf_api_key>

{"video_id":"<video_id>","filename":"sample.mp4"}
```

4. Poll status until `status` is `ready`:

```http
GET https://api.videomomentfinder.com/api/v1/videos/<video_id>
Authorization: Bearer <vmf_api_key>
```

5. Search by text:

```http
POST https://api.videomomentfinder.com/api/v1/videos/<video_id>/search
Content-Type: application/json
Authorization: Bearer <vmf_api_key>

{"query_text":"when do they explain the model?","limit":3}
```

## CLI Happy Path

If you already cloned this repository, the CLI wraps the same `/api/v1` flow:

```bash
uv sync
uv run vmf auth set \
  --api-base-url https://api.videomomentfinder.com \
  --api-key vmf_YOUR_KEY
uv run vmf videos upload ./sample.mp4
uv run vmf videos wait <video_id>
uv run vmf videos search <video_id> --query-text "when do they explain the model?"
```

## References

- Developers overview: `https://www.videomomentfinder.com/developers`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`
- Pricing and API key dashboard: `https://www.videomomentfinder.com/dashboard/api`
- Support: `https://www.videomomentfinder.com/support`
