# CLI and API Guide

Need a one-link public entrypoint for agents? Start with `https://www.videomomentfinder.com/skill.md`.
This guide remains the detailed reference for the same happy path.
Video Moment Finder also ships a remote MCP connector, but this guide is specifically for the REST API plus CLI happy path. For the Claude connector flow, see `docs/MCP_GUIDE.md`.

This guide covers the supported external happy path for Video Moment Finder:

- create or manage API keys with a temporary Clerk bearer token
- save API base URL and `vmf_` API key locally
- upload a video
- wait for processing
- run text search on the processed video

Current scope:

- one-shot CLI upload and direct-upload REST flow
- text search only
- no image-search CLI yet
- no YouTube-submit CLI yet

Public references:

- skill file: `https://www.videomomentfinder.com/skill.md`
- developers page: `https://www.videomomentfinder.com/developers`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`

## Prerequisites

- Python 3.11+
- `uv`
- a reachable Video Moment Finder API base URL, for example `http://localhost:8000`
- a temporary Clerk bearer token if you need to bootstrap a new API key

Install the CLI from this repo:

```bash
uv sync
```

You can then run commands with either `uv run vmf ...` or `uv run python -m src.cli ...`.

Check the installed CLI version:

```bash
uv run vmf --version
```

## Auth Setup

The CLI resolves configuration in this order:

1. command-line flags
2. environment variables
3. local config file

Environment variables:

- `VMF_API_BASE_URL`
- `VMF_API_KEY`
- `VMF_BEARER_TOKEN`
- `VMF_CONFIG_PATH` (optional override for the config file path)

Default config file path:

- `$XDG_CONFIG_HOME/video-moment-finder/config.json`
- fallback: `~/.config/video-moment-finder/config.json`

The local config stores only:

- `api_base_url`
- `api_key`

It does not store the Clerk bearer token.

### Bootstrap a New API Key

Export your API base URL and temporary Clerk bearer token:

```bash
export VMF_API_BASE_URL=http://localhost:8000
export VMF_BEARER_TOKEN=<your-clerk-bearer-token>
```

Create a key:

```bash
uv run vmf keys create --name agent
```

Expected output:

```json
{
  "id": "00000000-0000-4000-8000-aaaaaaaaaaaa",
  "name": "agent",
  "key_prefix": "vmf_dead",
  "created_at": "2026-03-17T12:00:00+00:00",
  "last_used_at": null,
  "key": "vmf_deadbeef12345678deadbeef12345678"
}
```

By default, `keys create` also saves the returned API key to the local config file. To skip that write, pass `--no-save`.

### Save an Existing API Key

If you already have a `vmf_` key, save it explicitly:

```bash
uv run vmf auth set \
  --api-base-url http://localhost:8000 \
  --api-key vmf_deadbeef12345678deadbeef12345678
```

Inspect the resolved config:

```bash
uv run vmf auth status
```

Preview or clear the saved config non-interactively:

```bash
uv run vmf auth clear --dry-run
uv run vmf auth clear --yes
```

### List or Revoke Keys

These commands require an explicit Clerk bearer token:

```bash
uv run vmf keys list --bearer-token <your-clerk-bearer-token>
uv run vmf keys revoke <key_id> --dry-run
uv run vmf keys revoke <key_id> --yes --bearer-token <your-clerk-bearer-token>
```

## Happy Path

### 1. Upload a Video

```bash
uv run vmf videos upload ./sample.mp4
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

Expected output:

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "youtube_url": null,
  "status": "queued",
  "source_type": "upload",
  "source_filename": "sample.mp4",
  "source_url": null,
  "created_at": "2026-03-17T12:05:00+00:00",
  "error_message": null
}
```

The CLI uses the one-shot multipart upload route:

1. `POST /api/v1/videos/upload`
2. optional `Idempotency-Key` header for retry-safe uploads

The lower-level direct upload flow remains available in the REST API:

1. `POST /api/v1/videos/upload/init`
2. `PUT` the file to the returned presigned `upload_url`
3. `POST /api/v1/videos/upload/complete`

### 2. Wait for Processing

```bash
uv run vmf videos wait 11111111-1111-4111-8111-111111111111
```

If the wait fails or times out, the CLI still prints the last JSON payload to stdout and includes the last status or error detail on stderr.

Expected output when the video is ready:

```json
{
  "id": "11111111-1111-4111-8111-111111111111",
  "youtube_url": null,
  "status": "ready",
  "source_type": "upload",
  "source_filename": "sample.mp4",
  "source_url": "https://example.com/source/11111111-1111-4111-8111-111111111111/sample.mp4",
  "created_at": "2026-03-17T12:05:00+00:00",
  "error_message": null
}
```

Defaults:

- `--interval-seconds 2`
- `--timeout-seconds 1200`

To fetch status once instead of polling:

```bash
uv run vmf videos get 11111111-1111-4111-8111-111111111111
```

To list your uploaded videos:

```bash
uv run vmf videos list
```

### 3. Search by Text

```bash
uv run vmf videos search 11111111-1111-4111-8111-111111111111 \
  --query-text "when do they explain the model?" \
  --limit 3
```

Search from stdin:

```bash
printf 'when do they explain the model?' | \
  uv run vmf videos search 11111111-1111-4111-8111-111111111111 --query-text -
```

The CLI allows up to 120 seconds for a search request before timing out.

Expected output:

```json
{
  "video_id": "11111111-1111-4111-8111-111111111111",
  "youtube_url": null,
  "source_url": "https://example.com/source/11111111-1111-4111-8111-111111111111/sample.mp4",
  "status": "ready",
  "results": [
    {
      "timestamp_s": 42.5,
      "thumbnail_url": null,
      "score": 0.91,
      "source": "transcript",
      "transcript_text": "The model is trained on..."
    }
  ]
}
```

## Underlying API Contract

The CLI wraps these existing `/api/v1` routes:

| CLI command | Method | Path | Auth |
| --- | --- | --- | --- |
| `vmf keys create` | `POST` | `/api/v1/keys` | Clerk bearer token |
| `vmf keys list` | `GET` | `/api/v1/keys` | Clerk bearer token |
| `vmf keys revoke` | `DELETE` | `/api/v1/keys/{key_id}` | Clerk bearer token |
| `vmf videos upload` | `POST` | `/api/v1/videos/upload` | API key |
| `vmf videos list` | `GET` | `/api/v1/videos` | API key |
| `vmf videos get` / `wait` | `GET` | `/api/v1/videos/{video_id}` | API key |
| `vmf videos search` | `POST` | `/api/v1/videos/{video_id}/search` | API key |

Additional public REST routes that are part of the curated schema:

| Purpose | Method | Path | Auth |
| --- | --- | --- | --- |
| API billing summary | `GET` | `/api/v1/billing/units/summary` | Clerk bearer token or API key |
| API billing usage | `GET` | `/api/v1/billing/units/usage` | Clerk bearer token or API key |
| Developer Pack checkout | `POST` | `/api/v1/billing/units/checkout` | Clerk bearer token |

The direct upload API remains available for lower-level clients:

| REST route | Method | Path |
| --- | --- | --- |
| direct upload init | `POST` | `/api/v1/videos/upload/init` |
| direct upload bytes | `PUT` | presigned `upload_url` |
| direct upload complete | `POST` | `/api/v1/videos/upload/complete` |

Web-only routes such as YouTube submit, creator credit billing, and image search are intentionally excluded from the published OpenAPI schema.

All authenticated requests use:

```text
Authorization: Bearer <token>
```

## CLI Safety And Output Contract

- Success payloads are always JSON on stdout.
- Dry runs are also JSON on stdout.
- Prompts and errors are written to stderr.
- Destructive commands use a safe default:
  - `vmf auth clear`
  - `vmf keys revoke`
- In a TTY, destructive commands prompt unless you pass `--yes`.
- In non-interactive mode, destructive commands fail fast unless you pass `--yes`.
- Use `--dry-run` to preview a destructive command without mutating anything.

Exit codes:

- `0`: success
- `1`: runtime failure or user-cancelled confirmation
- `2`: usage, validation, or config error, including missing `--yes` in non-interactive mode
- `130`: interrupted

## Failure Cases

- `HTTP 401: Missing Authorization header` or `HTTP 401: Invalid authentication token`
  - Check whether you passed a valid Clerk bearer token for `keys ...` commands or a valid `vmf_` key for `videos ...` commands.
- `HTTP 400: Video not ready for search (status: queued)` or `processing`
  - Wait for the video to reach `ready` before running `videos search`.
- `HTTP 503: Failed to verify upload`
  - The API could not confirm the uploaded object or inspect the uploaded video.
- `HTTP 503: Search is temporarily unavailable. Please try again.`
  - The search backend is temporarily unavailable. Retry later.
- `Request timed out after 120s`
  - The first search against a cold backend can take longer than usual. Retry the request.
- `API key created but failed to save config: ...`
  - The key was created successfully and is still printed to stdout. Save it manually with `vmf auth set`.
- `Refusing to revoke an API key in non-interactive mode without --yes.`
  - Re-run the same command with `--yes`, or use `--dry-run` to preview it first.
- `stdin uploads require --filename` / `--content-type`
  - When `videos upload` reads bytes from stdin, pass both flags so the API can name and validate the upload.

## API Billing

API usage is billed in units, separate from web credits:

- **500 units** per indexed video
- **1 unit** per text query (launch pricing, configurable via `API_UNIT_COST_INDEX_VIDEO` / `API_UNIT_COST_TEXT_QUERY`)

Purchase a Developer Pack ($20, 10,000 units) from `/dashboard/api` or `/pricing`.

When API units are insufficient, requests return `HTTP 402` with `{"code": "insufficient_api_units", ...}`.

## Data Retention

Source video uploads are temporary and may be auto-deleted after processing. Indexed search data (embeddings, transcripts, thumbnails) remains available while your account is active.

## Notes for Agents

- Commands write JSON to stdout on success.
- Errors are written to stderr and return a non-zero exit code.
- `videos wait` returns exit code `1` if the video fails processing or the wait times out. In those cases it still prints the last video payload to stdout before exiting.
