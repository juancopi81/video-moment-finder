# Remote MCP Guide

This document covers the thin remote MCP server shipped in this repository and how to review, test, deploy, and merge it safely.

## What This Ships

- Remote MCP endpoint: `https://api.videomomentfinder.com/mcp`
- Transport: Streamable HTTP
- Auth today: manual `Authorization: Bearer <vmf_...>` API key
- Tools:
  - `upload_video`
  - `get_video_status`
  - `list_videos`
  - `search_video`

Current scope:

- `upload_video` is a two-phase presigned-upload flow:
  - `action="start"` returns `video_id` + `upload_url`
  - `action="complete"` finalizes the upload after the file bytes are written
- `search_video` is text-only in MCP
- OAuth account linking is intentionally deferred to the next PR

## What Is Deferred

Not part of this shipped flow:

- Claude-native OAuth install/connect
- Protected resource metadata for MCP OAuth discovery
- Account-link approval UX
- Connector-specific pricing/connect UI
- Image search in MCP
- YouTube submit in MCP

## Local Test Flow

1. Start the API:

```bash
uv run uvicorn src.api.app:app --reload --port 8000
```

2. Use a real `vmf_` API key.

3. Run a transport smoke test:

```bash
uv run python - <<'PY'
import asyncio
import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8000/mcp"
API_KEY = "vmf_YOUR_KEY"


async def main():
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {API_KEY}"},
    ) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                print([tool.name for tool in tools.tools])
                result = await session.call_tool(
                    "upload_video",
                    {
                        "action": "start",
                        "filename": "sample.mp4",
                        "content_type": "video/mp4",
                    },
                )
                print(result.structuredContent)


asyncio.run(main())
PY
```

4. Upload the file bytes to the returned `upload_url` with a plain `PUT`.
   Do not forward the `Authorization` header to the presigned upload URL.

5. Finalize and search:

```bash
uv run python - <<'PY'
import asyncio
import httpx

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8000/mcp"
API_KEY = "vmf_YOUR_KEY"
VIDEO_ID = "VIDEO_ID_FROM_START"


async def main():
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {API_KEY}"},
    ) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                completed = await session.call_tool(
                    "upload_video",
                    {
                        "action": "complete",
                        "video_id": VIDEO_ID,
                        "filename": "sample.mp4",
                    },
                )
                print(completed.structuredContent)
                status = await session.call_tool("get_video_status", {"video_id": VIDEO_ID})
                print(status.structuredContent)


asyncio.run(main())
PY
```

## Production Test After Merge

Once the Railway API deploy is live, test production with the same flow against:

- `https://api.videomomentfinder.com/mcp`

Recommended smoke test:

1. Connect with a valid `vmf_` API key
2. Confirm tool discovery returns exactly:
   - `upload_video`
   - `get_video_status`
   - `list_videos`
   - `search_video`
3. Run `upload_video` with `action="start"`
4. `PUT` a small test file to the returned `upload_url`
5. Run `upload_video` with `action="complete"`
6. Poll with `get_video_status`
7. Run `search_video` after the video reaches `ready`

Recommended clients:

- Anthropic Messages API with a remote MCP server definition and bearer auth
- A generic MCP inspector/client for transport debugging

## Validation Included In This PR

Targeted validation:

- `uv run pytest -q tests/api/test_mcp.py tests/api/test_openapi.py tests/api/v1/test_router.py`

Full validation before merge:

- `./scripts/workflow/check_all.sh`

The MCP tests cover:

- missing auth -> `401`
- invalid `vmf_` key -> `401`
- Clerk-style bearer token rejection on `/mcp`
- tool discovery
- upload start
- upload complete
- status lookup
- video listing
- text search
- representative tool error paths
- `/mcp` exclusion from the curated REST OpenAPI schema

## Merge and Deploy Notes

Merge path:

- Normal squash merge is fine once validation is green.

Deploy effects after merge:

- Railway API must rebuild because backend code and Python dependencies changed
- Frontend redeploy is needed if the docs pages changed in the same PR

What is *not* required for this PR:

- no Supabase migrations
- no new environment variables
- no Modal redeploy

If deployment fails after merge, investigate:

- Python dependency install/build on Railway
- MCP route wiring or transport behavior
- existing auth/storage/billing runtime config already used by the API
