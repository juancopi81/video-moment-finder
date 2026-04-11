# Remote MCP Guide

This document covers the shipped Claude-compatible remote MCP server, its OAuth flow, reviewer setup, and the validation gates required before submission.

## What Ships

- Remote MCP endpoint: `https://api.videomomentfinder.com/mcp`
- Transport: Streamable HTTP
- Auth: OAuth 2.0 authorization-code + PKCE
- OAuth discovery:
  - `GET /.well-known/oauth-authorization-server`
  - `GET /.well-known/oauth-protected-resource/mcp`
- OAuth transaction endpoints:
  - `GET|POST /authorize`
  - `POST /register`
  - `POST /token`
  - `POST /revoke`
- Frontend approval page:
  - `https://www.videomomentfinder.com/connectors/claude?request_id=...`
- Current MCP tools:
  - `upload_video`
  - `get_video_status`
  - `list_videos`
  - `search_video`

Behavior notes:

- `/mcp` is OAuth-only.
- REST API and CLI keep their existing JWT + `vmf_` API-key behavior.
- Connector usage bills against Developer Pack API units.
- MCP tool execution records `api_usage_events.api_key_id = null` for OAuth calls.
- `upload_video` is annotated as a write tool; the other three tools are read-only.

## Current Tool Scope

- `upload_video` is a two-step presigned upload flow:
  - `action="start"` returns `video_id` + `upload_url`
  - `action="complete"` finalizes the upload after the file bytes are written
- `search_video` is text-only in MCP
- YouTube submit is not part of the MCP tool surface
- REST remains the canonical public contract for one-shot multipart upload, image search, and non-MCP programmatic usage

## Claude Connect Flow

1. Add the custom connector in Claude with server URL `https://api.videomomentfinder.com/mcp`.
2. Use guided OAuth. Claude web/Desktop can self-register through DCR, and internal review flows may still use the static client credentials from the secure review/test configuration.
3. Click `Connect`.
4. The user lands on `https://www.videomomentfinder.com/connectors/claude?request_id=...`.
5. If signed out, they sign in or create an account.
6. If `api_units_balance <= 0`, they buy a Developer Pack and return to the same connector page.
7. They review the four MCP tools and explicitly approve access.
8. Claude receives the authorization code callback, exchanges it for tokens, and begins using the connector.

Supported redirect URIs:

- `https://claude.ai/api/mcp/auth_callback`
- `https://claude.com/api/mcp/auth_callback`
- `http://localhost:6274/oauth/callback`
- `http://localhost:6274/oauth/callback/debug`

## Reviewer Setup

Do not publish the confidential client secret in public docs or UI copy.

For Anthropic review and internal testing:

- Claude web/Desktop and other DCR-capable surfaces can self-register a public client through `POST /register`.
- Keep the static reviewer client (`MCP_OAUTH_CLIENT_ID` / `MCP_OAUTH_CLIENT_SECRET`) available for review flows that still expect explicit credentials.
- Prepare a live review account before submission:
  - valid login
  - positive Developer Pack API-unit balance
  - at least one ready video
  - at least three documented example prompts

Related public pages:

- Developers: `https://www.videomomentfinder.com/developers`
- Privacy: `https://www.videomomentfinder.com/privacy`
- Support: `https://www.videomomentfinder.com/support`

## Example Prompts

1. `List my recent videos in Video Moment Finder.`
   Expected behavior: Claude calls `list_videos` and returns recent video IDs, status, and upload source details.

2. `Check whether video <video_id> is ready yet.`
   Expected behavior: Claude calls `get_video_status` and reports the current processing state or failure reason.

3. `Search video <video_id> for the moment they explain the model and give me timestamps.`
   Expected behavior: Claude calls `search_video` and returns timestamped matches from the indexed video.

4. `Upload this MP4 to Video Moment Finder and then poll until processing starts.`
   Expected behavior: Claude uses `upload_video(action="start")`, writes the file to the presigned upload URL, then uses `upload_video(action="complete")` and `get_video_status`.

## Local Validation

Start the API:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

Run the targeted backend test suite:

```bash
.venv/bin/pytest -q \
  tests/api/test_mcp.py \
  tests/api/test_mcp_oauth.py \
  tests/api/test_api_billing.py \
  tests/api/test_openapi.py \
  tests/billing/test_lemonsqueezy.py
```

Full validation before merge:

```bash
./scripts/workflow/check_all.sh
```

## Submission Gates

These checks must pass before Anthropic directory submission:

- Claude.ai custom connector completes connect/auth successfully
- Claude Desktop completes connect/auth successfully
- MCP Inspector or Claude Code completes connect/auth successfully
- `upload_video`, `get_video_status`, `list_videos`, and `search_video` all work end to end
- Privacy and support URLs are reachable over HTTPS
- Public docs contain no preview, deferred, or manual-auth wording

## Upload Validation Gate

The current MCP upload path remains the two-step presigned flow.

Before submission, verify in a real Claude client that:

- Claude can complete `upload_video(action="start")`
- Claude can write the file bytes to the returned `upload_url` without forwarding `Authorization`
- Claude can call `upload_video(action="complete")` and continue with status + search
