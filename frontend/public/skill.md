---
name: video-moment-finder
description: Public bootstrap for the Video Moment Finder REST API, CLI, and client-neutral remote MCP connection.
homepage: https://www.videomomentfinder.com
api_host: https://api.videomomentfinder.com
openapi: https://api.videomomentfinder.com/openapi.json
---

# Video Moment Finder

Use this file when you need the canonical public entrypoints for Video Moment Finder.

Video Moment Finder exposes two integration surfaces:

- REST API + CLI, authenticated with `vmf_` API keys
- Remote MCP at `https://api.videomomentfinder.com/mcp`, authenticated with OAuth 2.0 authorization-code + PKCE

## Canonical URLs

- Site: `https://www.videomomentfinder.com`
- Developers overview: `https://www.videomomentfinder.com/developers`
- MCP approval page: `https://www.videomomentfinder.com/connectors/claude` (existing compatibility URL; start the flow in your app)
- Public skill file: `https://www.videomomentfinder.com/skill.md`
- Privacy policy: `https://www.videomomentfinder.com/privacy`
- Support: `https://www.videomomentfinder.com/support`
- API host: `https://api.videomomentfinder.com`
- Public REST prefix: `https://api.videomomentfinder.com/api/v1`
- Remote MCP endpoint: `https://api.videomomentfinder.com/mcp`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`

## Supported Happy Path

- upload a video
- wait for indexing
- list videos
- search a processed video by text
- fetch the full transcript with per-segment timestamps
- fetch frames at specific timestamps (stored thumbnails or on-demand high-resolution)

REST is the canonical public contract for these capabilities. The CLI wraps only upload, status polling, and search — there is no CLI wrapper yet for transcript or frame retrieval; use REST directly for those two.

## Connect Video Moment Finder

Remote MCP tools:

- `upload_video`
- `get_video_status`
- `list_videos`
- `search_video`
- `get_transcript`
- `get_frames` (returns frames as image content; defaults to high resolution with automatic thumbnail fallback)

The server also ships the `lecture_notes` MCP prompt, a guided workflow that turns an indexed lecture video into Markdown study notes from its transcript and board-moment frames. MCP prompts are user-invoked (not callable by agents as tools); clients without prompt support get the same workflow from the server's MCP instructions.

How the connection works:

1. Add Video Moment Finder in a supported app, or use the remote server URL `https://api.videomomentfinder.com/mcp`.
2. Start `Connect` and follow guided OAuth. Clients that support dynamic client registration can register automatically.
3. Sign in to the Video Moment Finder account that holds your videos. Do not paste API keys or tokens into chat.
4. Review the requesting app, six tools, current allowance, and operation costs, then approve access. Approval requires a positive API-unit balance. Approval itself does not run a video operation.
5. Return to your app and ask: `List my ready videos, then help me understand one key idea from a lecture, with a timestamp and a frame where available.`

Access is scoped to the connected account. Use a returned video ID and wait for `ready` status. A public video URL alone does not make its content available through MCP. If the video is missing, inaccessible, or failed, explain that state and choose another available video. Do not repeatedly retry a failed or inaccessible video.

## Units and account limits

Website video credits and API units are separate balances. MCP and API operations use API units. Default costs are:

| Operation | API units |
| --- | --- |
| Index a new video | 500 |
| Search a ready video | 1 per call |
| Retrieve a transcript or range | 1 per call |
| Retrieve thumbnail frames | 1 per call, up to 25 timestamps |
| Retrieve high-resolution frames | 5 per call, up to 8 timestamps |
| List videos or check status | 0 |

The consent screen reports the server's configured rates. Batch timestamps, reuse retrieved evidence, and explain the expected cost before indexing. One transcript plus one thumbnail call is 2 units at the default rates; one transcript plus one high-resolution call is 6. Treat these as estimates, not charge receipts. Do not assume a high-resolution request with thumbnail fallback is free.

An available verified-account trial is granted at most once per account. Its actual grant and status appear after sign-in; prior free website processing can count toward the allowance. Trial availability is deployment-controlled, so do not promise a grant or a fixed amount before the server reports it. Reconnecting, signing in again, or creating a new API key does not reset a trial.

When the account has insufficient units, stop metered calls and explain the limitation without steering to a purchase or upgrade. Continue teaching from evidence already retrieved when useful. If the connection is still active, video listing and status checks remain free. Do not make paid operations to test whether an allowance changed.

If transcripts or frames are unavailable, identify that gap. Inspect actual returned images before making visual claims; report unreadable labels instead of guessing. Keep video IDs and timestamps as source references. Link to `https://www.videomomentfinder.com/video/<video_id>?t=<seconds>` for an account-scoped source location; playback may be unavailable after source cleanup. Do not embed expiring or signed storage URLs in saved learning artifacts.

## File upload

Current MCP upload behavior:

- `upload_video(action="start")` returns a presigned `upload_url`
- write the file bytes to that URL with a plain `PUT`
- `upload_video(action="complete")` finalizes the upload

This requires a client capable of transferring file bytes; a tool call alone does not upload the file. Upload only material the user owns or is authorized to use, with their instruction to index it and awareness of the indexing cost.

## MCP client example

The MCP endpoint is client-neutral: Streamable HTTP with OAuth authorization-code + PKCE and dynamic client registration. Any MCP client supporting remote OAuth servers can connect. For example, Codex:

```bash
codex mcp add vmf --url https://api.videomomentfinder.com/mcp
codex mcp login vmf
```

Clients without MCP prompt support receive the lecture-notes workflow via the server's MCP instructions.

## REST Happy Path

All authenticated REST API requests use:

```text
Authorization: Bearer <vmf_api_key>
```

1. Upload one video with the one-shot multipart route.

```bash
curl -X POST https://api.videomomentfinder.com/api/v1/videos/upload \
  -H "Authorization: Bearer <vmf_api_key>" \
  -H "Idempotency-Key: sample-v1" \
  -F "file=@sample.mp4;type=video/mp4"
```

2. Poll until processing reaches `ready`:

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

4. Fetch the full transcript (optionally range-filtered with `start_s`/`end_s`):

```http
GET https://api.videomomentfinder.com/api/v1/videos/<video_id>/transcript
Authorization: Bearer <vmf_api_key>
```

5. Fetch frames at timestamps (`thumb` returns presigned thumbnail URLs, up to 25 timestamps; `high` returns base64 JPEG frames extracted from the retained source, up to 8 timestamps, 409 `source_not_retained` when no source is retained):

```http
POST https://api.videomomentfinder.com/api/v1/videos/<video_id>/frames
Content-Type: application/json
Authorization: Bearer <vmf_api_key>

{"timestamps":[312.0, 754.5], "resolution":"high"}
```

## CLI Happy Path

```bash
uv sync
uv run vmf auth set \
  --api-base-url https://api.videomomentfinder.com \
  --api-key vmf_YOUR_KEY
uv run vmf videos upload ./sample.mp4
uv run vmf videos wait <video_id>
uv run vmf videos search <video_id> --query-text "when do they explain the model?"
```

## Example Prompts

1. `List my recent videos in Video Moment Finder.`
   Expected behavior: the connected app returns recent video IDs, status, and source details from the authorized account.

2. `Check whether video <video_id> is ready and tell me if it failed.`
   Expected behavior: status polling plus failure detail when present.

3. `Search video <video_id> for the moment they explain the model.`
   Expected behavior: timestamped text-search results from the indexed video.

4. `Use the lecture_notes prompt to turn video <video_id> into study notes.`
   Expected behavior: the connector's `lecture_notes` prompt guides the agent through transcript retrieval, board-moment frame inspection, and structured Markdown notes generation.

## Security Rules

- Only send OAuth tokens, Clerk JWTs, or `vmf_` API keys to `https://api.videomomentfinder.com`.
- Do not forward your `Authorization` header to a presigned `upload_url`.
- Treat raw API keys and confidential OAuth client secrets as secret material.
- Keep REST/CLI auth and MCP auth separate:
  - REST + CLI use `vmf_` API keys
  - `/mcp` uses OAuth bearer tokens

## References

- Developers overview: `https://www.videomomentfinder.com/developers`
- OpenAPI schema: `https://api.videomomentfinder.com/openapi.json`
- Swagger UI: `https://api.videomomentfinder.com/docs`
- Privacy policy: `https://www.videomomentfinder.com/privacy`
- Support: `https://www.videomomentfinder.com/support`
