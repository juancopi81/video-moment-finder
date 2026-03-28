"""Remote MCP server mounted into the main FastAPI app."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.auth import AuthIdentity
from src.api.mcp_oauth import (
    McpOAuthConfigError,
    load_mcp_oauth_access_token,
    mcp_oauth_resource_url,
    mcp_oauth_scope,
    mcp_oauth_www_authenticate,
)
from src.db.supabase import SourceType, VideoStatus

_IDENTITY_STATE_KEY = "vmf_mcp_identity"
_mcp_session_manager_cm: Any | None = None


class McpVideoRecord(BaseModel):
    id: str
    youtube_url: str | None
    status: VideoStatus
    source_type: SourceType
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
    status: VideoStatus
    results: list[McpSearchResult]


def _auth_error(*, status_code: int, error: str, description: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "error_description": description},
        headers={
            "WWW-Authenticate": mcp_oauth_www_authenticate(
                error=error,
                description=description,
            )
        },
    )


class McpOAuthResourceApp:
    """Require OAuth Bearer tokens before forwarding to the mounted MCP app."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope["method"] == "HEAD":
            await Response(status_code=204)(scope, receive, send)
            return

        headers = Headers(scope=scope)
        authorization = headers.get("authorization")
        if authorization is None:
            await _auth_error(
                status_code=401,
                error="invalid_token",
                description="Authentication required",
            )(scope, receive, send)
            return

        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            await _auth_error(
                status_code=401,
                error="invalid_token",
                description="Invalid Authorization header",
            )(scope, receive, send)
            return

        raw_token = token.strip()
        try:
            configured_resource = mcp_oauth_resource_url()
            access_token = await load_mcp_oauth_access_token(raw_token)
        except McpOAuthConfigError:
            await JSONResponse(
                status_code=503,
                content={"detail": "MCP OAuth is not configured"},
            )(scope, receive, send)
            return
        if access_token is None or access_token.resource.rstrip("/") != configured_resource:
            await _auth_error(
                status_code=401,
                error="invalid_token",
                description="Invalid authentication token",
            )(scope, receive, send)
            return

        if mcp_oauth_scope() not in access_token.scopes:
            await _auth_error(
                status_code=403,
                error="insufficient_scope",
                description=f"Required scope: {mcp_oauth_scope()}",
            )(scope, receive, send)
            return

        scope.setdefault("state", {})
        scope["state"][_IDENTITY_STATE_KEY] = AuthIdentity(
            user_id=access_token.user_id,
            auth_method="mcp_oauth",
        )
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


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None


def _video_record_from_response(response: Any) -> McpVideoRecord:
    return McpVideoRecord(
        id=response.id,
        youtube_url=_optional_str(response.youtube_url),
        status=response.status,
        source_type=response.source_type,
        source_filename=response.source_filename,
        source_url=_optional_str(response.source_url),
        created_at=response.created_at.isoformat(),
        error_message=response.error_message,
    )


def _search_result_from_response(response: Any) -> SearchVideoResult:
    return SearchVideoResult(
        video_id=response.video_id,
        youtube_url=_optional_str(response.youtube_url),
        source_url=_optional_str(response.source_url),
        status=response.status,
        results=[
            McpSearchResult(
                timestamp_s=result.timestamp_s,
                thumbnail_url=_optional_str(result.thumbnail_url),
                score=result.score,
                source=result.source,
                transcript_text=result.transcript_text,
            )
            for result in response.results
        ],
    )


vmf_mcp = FastMCP(
    name="Video Moment Finder",
    instructions=(
        "OAuth-protected remote MCP server for Video Moment Finder. "
        "Supports presigned upload bootstrap, upload completion, video status, "
        "video listing, and text search for your connected account."
    ),
    website_url="https://www.videomomentfinder.com",
    host="0.0.0.0",
    json_response=True,
    stateless_http=True,
    streamable_http_path="/",
)


def mcp_tool_approval_items() -> list[dict[str, str]]:
    return [
        {
            "name": "upload_video",
            "title": "Upload Video",
            "description": "Start a presigned video upload or complete it after the file bytes are uploaded.",
        },
        {
            "name": "get_video_status",
            "title": "Get Video Status",
            "description": "Check the processing status for one indexed video.",
        },
        {
            "name": "list_videos",
            "title": "List Videos",
            "description": "List your recent indexed videos.",
        },
        {
            "name": "search_video",
            "title": "Search Video",
            "description": "Run a text search against a ready video and return timestamped matches.",
        },
    ]


@vmf_mcp.tool(
    title="Upload Video",
    annotations=ToolAnnotations(
        title="Upload Video",
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
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


@vmf_mcp.tool(
    title="Get Video Status",
    annotations=ToolAnnotations(
        title="Get Video Status",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
)
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


@vmf_mcp.tool(
    title="List Videos",
    annotations=ToolAnnotations(
        title="List Videos",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
)
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


@vmf_mcp.tool(
    title="Search Video",
    annotations=ToolAnnotations(
        title="Search Video",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
)
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
    """Return the `/mcp` ASGI endpoint protected by OAuth bearer auth."""
    return McpOAuthResourceApp(StreamableHttpMcpEndpoint(vmf_mcp))


async def startup_mcp_session_manager() -> None:
    """Create and start a fresh FastMCP session manager for this app lifespan."""
    global _mcp_session_manager_cm
    if _mcp_session_manager_cm is not None:
        return

    # FastMCP caches its session manager on a private attribute. Reset it so the
    # parent FastAPI lifespan and repeated TestClient runs get a fresh manager.
    # Revisit this workaround when upgrading the pinned MCP SDK.
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
