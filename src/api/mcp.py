"""Remote MCP server mounted into the main FastAPI app."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.auth import AuthIdentity, TokenVerificationError, verify_api_key

_IDENTITY_STATE_KEY = "vmf_mcp_identity"
_mcp_session_manager_cm: Any | None = None


class McpVideoRecord(BaseModel):
    id: str
    youtube_url: str | None
    status: Literal["queued", "processing", "ready", "failed"]
    source_type: Literal["youtube", "upload"]
    source_filename: str | None = None
    source_url: str | None = None
    created_at: str
    error_message: str | None = None


class McpSearchResult(BaseModel):
    timestamp_s: float
    thumbnail_url: str | None = None
    score: float
    source: Literal["visual", "transcript"]
    transcript_text: str | None = None


class UploadVideoResult(BaseModel):
    action: Literal["start", "complete"]
    video_id: str
    upload_url: str | None = None
    method: Literal["PUT"] | None = None
    expires_in_seconds: int | None = None
    do_not_send_headers: list[str] | None = None
    next_action: Literal["complete"] | None = None
    video: McpVideoRecord | None = None


class ListVideosResult(BaseModel):
    returned_count: int
    videos: list[McpVideoRecord]


class SearchVideoResult(BaseModel):
    video_id: str
    youtube_url: str | None
    source_url: str | None = None
    status: Literal["queued", "processing", "ready", "failed"]
    results: list[McpSearchResult]


def _unauthorized(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": detail},
        headers={"WWW-Authenticate": "Bearer"},
    )


# TODO(next PR: OAuth connector): replace this manual vmf_ API-key gate with
# MCP protected resource metadata + OAuth account linking, then remove the
# manual-key testing instructions from the docs.
class ManualApiKeyAuthApp:
    """Require Bearer vmf_ API keys before forwarding to the mounted MCP app."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        authorization = headers.get("authorization")
        if authorization is None:
            await _unauthorized("Missing Authorization header")(scope, receive, send)
            return

        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            await _unauthorized("Invalid Authorization header")(scope, receive, send)
            return

        raw_token = token.strip()
        if not raw_token.startswith("vmf_"):
            await _unauthorized("MCP endpoints require a vmf_ API key")(scope, receive, send)
            return

        try:
            identity = verify_api_key(raw_token)
        except TokenVerificationError as exc:
            await _unauthorized(str(exc))(scope, receive, send)
            return

        scope.setdefault("state", {})
        scope["state"][_IDENTITY_STATE_KEY] = identity
        await self.app(scope, receive, send)


class StreamableHttpMcpEndpoint:
    """Dispatch ASGI requests into the current FastMCP session manager."""

    def __init__(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.mcp_server.session_manager.handle_request(scope, receive, send)


def _request_identity(ctx: Context) -> AuthIdentity:
    request_context = ctx.request_context
    if request_context is None or request_context.request is None:
        raise RuntimeError("MCP request context is unavailable")
    identity = getattr(request_context.request.state, _IDENTITY_STATE_KEY, None)
    if not isinstance(identity, AuthIdentity):
        raise RuntimeError("MCP authentication context is unavailable")
    return identity


def _video_record_from_response(response: Any) -> McpVideoRecord:
    created_at = response.created_at.isoformat() if hasattr(response.created_at, "isoformat") else str(response.created_at)
    youtube_url = str(response.youtube_url) if response.youtube_url is not None else None
    source_url = str(response.source_url) if response.source_url is not None else None
    return McpVideoRecord(
        id=response.id,
        youtube_url=youtube_url,
        status=response.status,
        source_type=response.source_type,
        source_filename=response.source_filename,
        source_url=source_url,
        created_at=created_at,
        error_message=response.error_message,
    )


def _search_result_from_response(response: Any) -> SearchVideoResult:
    return SearchVideoResult(
        video_id=response.video_id,
        youtube_url=str(response.youtube_url) if response.youtube_url is not None else None,
        source_url=str(response.source_url) if response.source_url is not None else None,
        status=response.status,
        results=[
            McpSearchResult(
                timestamp_s=result.timestamp_s,
                thumbnail_url=str(result.thumbnail_url) if result.thumbnail_url is not None else None,
                score=result.score,
                source=result.source,
                transcript_text=result.transcript_text,
            )
            for result in response.results
        ],
    )


# TODO(next PR: OAuth connector): publish connector-grade metadata here once the
# authorization server, approval UI, and scope model are in place.
vmf_mcp = FastMCP(
    name="Video Moment Finder",
    instructions=(
        "Remote MCP server for Video Moment Finder. "
        "Supports presigned upload bootstrap, upload completion, video status, "
        "video listing, and text search. Manual vmf_ API-key auth is supported "
        "for private testing. OAuth account linking is not shipped yet."
    ),
    website_url="https://www.videomomentfinder.com",
    host="0.0.0.0",
    json_response=True,
    stateless_http=True,
    streamable_http_path="/",
)


@vmf_mcp.tool()
def upload_video(
    action: Annotated[
        Literal["start", "complete"],
        Field(description="Use start to get a presigned upload URL, then complete after the file upload finishes."),
    ],
    filename: Annotated[
        str,
        Field(min_length=1, max_length=200, description="Video filename used for upload bookkeeping."),
    ],
    content_type: Annotated[
        str | None,
        Field(default=None, max_length=200, description="Optional MIME type for the start action, for example video/mp4."),
    ] = None,
    video_id: Annotated[
        str | None,
        Field(default=None, description="Required for the complete action."),
    ] = None,
    ctx: Context | None = None,
) -> UploadVideoResult:
    """Start a presigned upload or complete one after the file bytes are uploaded."""
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import UploadCompleteRequest, UploadInitRequest, v1_complete_upload, v1_init_upload

    identity = _request_identity(ctx)

    if action == "start":
        response = v1_init_upload(
            UploadInitRequest(filename=filename, content_type=content_type),
            identity=identity,
        )
        return UploadVideoResult(
            action="start",
            video_id=response.video_id,
            upload_url=str(response.upload_url),
            method="PUT",
            expires_in_seconds=response.expires_in,
            do_not_send_headers=["Authorization"],
            next_action="complete",
        )

    if video_id is None:
        raise ValueError("video_id is required when action is complete")

    response = v1_complete_upload(
        UploadCompleteRequest(video_id=video_id, filename=filename),
        identity=identity,
    )
    return UploadVideoResult(
        action="complete",
        video_id=response.id,
        video=_video_record_from_response(response),
    )


@vmf_mcp.tool()
def get_video_status(
    video_id: Annotated[str, Field(min_length=1, description="Video UUID returned from upload_video.")],
    ctx: Context | None = None,
) -> McpVideoRecord:
    """Get the current processing status for one indexed video."""
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import v1_get_video

    identity = _request_identity(ctx)
    response = v1_get_video(video_id=video_id, identity=identity)
    return _video_record_from_response(response)


@vmf_mcp.tool()
def list_videos(
    limit: Annotated[
        int,
        Field(default=20, ge=1, le=50, description="Maximum number of recent videos to return."),
    ] = 20,
    ctx: Context | None = None,
) -> ListVideosResult:
    """List recent videos for the authenticated API-key owner."""
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import v1_list_my_videos

    identity = _request_identity(ctx)
    response = v1_list_my_videos(identity=identity)
    videos = [_video_record_from_response(video) for video in response[:limit]]
    return ListVideosResult(returned_count=len(videos), videos=videos)


@vmf_mcp.tool()
def search_video(
    video_id: Annotated[str, Field(min_length=1, description="Video UUID to search.")],
    query_text: Annotated[
        str,
        Field(min_length=1, max_length=500, description="Natural-language text query to run against the video."),
    ],
    limit: Annotated[
        int,
        Field(default=5, ge=1, le=20, description="Per-source result cap, matching the public API contract."),
    ] = 5,
    ctx: Context | None = None,
) -> SearchVideoResult:
    """Search a ready video by text and return timestamped matches."""
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import VideoSearchRequest, v1_search_video

    identity = _request_identity(ctx)
    response = v1_search_video(
        video_id=video_id,
        request=VideoSearchRequest(query_text=query_text, limit=limit),
        identity=identity,
    )
    return _search_result_from_response(response)


def build_mcp_asgi_app() -> ASGIApp:
    """Return the `/mcp` ASGI endpoint protected by manual API-key auth."""
    return ManualApiKeyAuthApp(StreamableHttpMcpEndpoint(vmf_mcp))


async def startup_mcp_session_manager() -> None:
    """Create and start a fresh FastMCP session manager for this app lifespan."""
    global _mcp_session_manager_cm
    if _mcp_session_manager_cm is not None:
        return

    # Initialize a fresh session manager for the current app lifespan.
    vmf_mcp._session_manager = None  # type: ignore[attr-defined]
    vmf_mcp.streamable_http_app()
    _mcp_session_manager_cm = vmf_mcp.session_manager.run()
    await _mcp_session_manager_cm.__aenter__()


async def shutdown_mcp_session_manager() -> None:
    """Stop the active FastMCP session manager and reset it for reuse in tests."""
    global _mcp_session_manager_cm
    if _mcp_session_manager_cm is None:
        return

    await _mcp_session_manager_cm.__aexit__(None, None, None)
    _mcp_session_manager_cm = None
    vmf_mcp._session_manager = None  # type: ignore[attr-defined]
