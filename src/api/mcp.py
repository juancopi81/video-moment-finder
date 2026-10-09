"""Remote MCP server mounted into the main FastAPI app."""
from __future__ import annotations

import base64
import binascii
import io
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import ToolAnnotations
from src.api.workspace import LearningView, UI_MIME, UI_PATH, UI_URI, origin, result as workspace_result
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from src.api.auth import AuthIdentity
from src.api.frames import (
    FramePlanItem,
    build_high_res_frame_plan,
    build_thumb_frame_plan,
    extract_high_res_frames,
    unique_dedupe_keys,
    validate_frame_request,
)
from src.api.mcp_oauth import (
    MCP_APPROVED_TOOLS_VERSION,
    MCP_TOOLS_REAPPROVAL_DESCRIPTION,
    McpOAuthConfigError,
    load_mcp_oauth_access_token,
    mcp_oauth_resource_url,
    mcp_oauth_scope,
    mcp_oauth_www_authenticate,
)
from src.db.supabase import SourceType, VideoStatus
from src.storage.config import R2Config, StorageConfigError
from src.storage.r2 import R2Store, R2StorageError, thumbnail_key
from src.utils.logging import get_logger

logger = get_logger(__name__)

_IDENTITY_STATE_KEY = "vmf_mcp_identity"
_mcp_session_manager_cm: Any | None = None
MAX_SEARCH_IMAGE_BYTES = 512 * 1024
MAX_SEARCH_IMAGE_BASE64 = 4 * ((MAX_SEARCH_IMAGE_BYTES + 2) // 3)


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


class McpTranscriptSegment(BaseModel):
    segment_index: int
    start_s: float
    end_s: float
    text: str


class GetTranscriptResult(BaseModel):
    video_id: str
    has_transcript: bool
    language_code: str | None = None
    segment_count: int
    segments: list[McpTranscriptSegment]


def _auth_error(*, status_code: int, error: str, description: str) -> JSONResponse:
    www_authenticate = "Bearer"
    try:
        www_authenticate = mcp_oauth_www_authenticate(
            error=error,
            description=description,
        )
    except McpOAuthConfigError:
        pass
    return JSONResponse(
        status_code=status_code,
        content={"error": error, "error_description": description},
        headers={"WWW-Authenticate": www_authenticate},
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

        try:
            configured_resource = mcp_oauth_resource_url()
        except McpOAuthConfigError:
            await JSONResponse(
                status_code=503,
                content={"detail": "MCP OAuth is not configured"},
            )(scope, receive, send)
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
        access_token = await load_mcp_oauth_access_token(raw_token)
        if access_token is None or access_token.resource.rstrip("/") != configured_resource:
            await _auth_error(
                status_code=401,
                error="invalid_token",
                description="Invalid authentication token",
            )(scope, receive, send)
            return

        if access_token.approved_tools_version < MCP_APPROVED_TOOLS_VERSION:
            # The grant behind this token was approved against an older tool
            # list. Reject with invalid_token so Claude clients re-run the
            # OAuth flow and the user re-consents to the current tools.
            await _auth_error(
                status_code=401,
                error="invalid_token",
                description=MCP_TOOLS_REAPPROVAL_DESCRIPTION,
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
        "video listing, text search, full transcript retrieval, and frame "
        "retrieval (returned as image content) for your connected account. "
        "Lecture-to-study-notes workflow (for clients without MCP prompt "
        "support; also available as the lecture_notes prompt): 1) confirm the "
        "video is ready via get_video_status; 2) fetch the full transcript "
        "with get_transcript; 3) identify the lecture's sections and 'board "
        "moments' where the speaker references visuals without describing "
        "them; 4) call get_frames (high resolution) for the 5-15 most "
        "important board moments, using timestamps near the END of each "
        "explanation since board content accumulates while the speaker "
        "writes; 5) transcribe the visuals into LaTeX/text and write one "
        "structured Markdown document with a source-status disclosure and a "
        "Main Takeaways list. Full recipe: "
        "https://github.com/juancopi81/video-moment-finder/blob/main/docs/LECTURE_NOTES_RECIPE.md"
    ),
    website_url="https://www.videomomentfinder.com",
    host="0.0.0.0",
    json_response=True,
    stateless_http=True,
    streamable_http_path="/",
)


def _unit_cost_label(units: int) -> str:
    return f"{units} unit" if units == 1 else f"{units} units"


def mcp_tool_approval_items() -> list[dict[str, str]]:
    """Tool list shown on the connector approval screen.

    The ``cost`` strings are derived from the same ``API_UNIT_COST_*``
    constants that the billing paths consume, so the approval screen cannot
    drift from what is actually metered. Changing this displayed surface in a
    way that requires re-consent must bump ``MCP_APPROVED_TOOLS_VERSION``
    (see src/api/mcp_oauth.py).
    """
    # Deferred import: src.api.app imports mcp_tool_approval_items at module
    # load, so importing the cost constants at the top level would be circular.
    from src.api.app import (
        API_UNIT_COST_FRAMES_HIGH,
        API_UNIT_COST_FRAMES_THUMB,
        API_UNIT_COST_INDEX_VIDEO,
        API_UNIT_COST_IMAGE_QUERY,
        API_UNIT_COST_TEXT_QUERY,
        API_UNIT_COST_TRANSCRIPT_FETCH,
    )

    return [
        {
            "name": "open_workspace", "title": "Video Workspace",
            "description": "Open your video library, current allowance and learning workspace.",
            "cost": "No units",
        },
        {
            "name": "get_workspace_video", "title": "Open Video Source",
            "description": "Open an owned video's retained source in the workspace.",
            "cost": "No units",
        },
        {
            "name": "render_learning_view", "title": "Show Learning View",
            "description": "Display a cited guide, cards, Playground or presentation prepared from video evidence.",
            "cost": "No units; evidence retrieval uses the costs below",
        },
        {
            "name": "upload_video",
            "title": "Upload Video",
            "description": "Start a presigned video upload or complete it after the file bytes are uploaded.",
            "cost": f"{_unit_cost_label(API_UNIT_COST_INDEX_VIDEO)} per indexed video",
        },
        {
            "name": "get_video_status",
            "title": "Get Video Status",
            "description": "Check the processing status for one indexed video.",
            "cost": "No units",
        },
        {
            "name": "list_videos",
            "title": "List Videos",
            "description": "List your recent indexed videos.",
            "cost": "No units",
        },
        {
            "name": "search_video",
            "title": "Search Video",
            "description": "Run a text search against a ready video and return timestamped matches.",
            "cost": f"{_unit_cost_label(API_UNIT_COST_TEXT_QUERY)} per search query",
        },
        {
            "name": "search_video_image",
            "title": "Find Similar Frames",
            "description": "Search an owned ready video's visual frames using a locally resized reference image. The workspace sends the chosen image for inference only when you press Search; it is not saved in your VMF library.",
            "cost": f"{_unit_cost_label(API_UNIT_COST_IMAGE_QUERY)} per image search",
        },
        {
            "name": "get_transcript",
            "title": "Get Transcript",
            "description": "Fetch the full spoken transcript with per-segment timestamps for a ready video.",
            "cost": f"{_unit_cost_label(API_UNIT_COST_TRANSCRIPT_FETCH)} per call",
        },
        {
            "name": "get_frames",
            "title": "Get Frames",
            "description": "Fetch video frames as images at specific timestamps for visual inspection.",
            "cost": (
                f"{_unit_cost_label(API_UNIT_COST_FRAMES_THUMB)} per thumbnail call, "
                f"{_unit_cost_label(API_UNIT_COST_FRAMES_HIGH)} per high-res call"
            ),
        },
    ]


def _workspace_video(response: Any) -> dict:
    video = _video_record_from_response(response).model_dump(mode="json")
    # Only the UI receives ephemeral playback capabilities.
    video.pop("source_url", None)
    return video


def _workspace_config() -> dict:
    import os
    from src.api.app import (
        API_UNIT_COST_FRAMES_HIGH, API_UNIT_COST_FRAMES_THUMB,
        API_UNIT_COST_TRANSCRIPT_FETCH, VIDEO_MAX_UPLOAD_BYTES,
        API_UNIT_COST_TEXT_QUERY, API_UNIT_COST_IMAGE_QUERY,
    )
    endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    return {
        "media_origin": origin(endpoint) if endpoint else None,
        "max_upload_bytes": VIDEO_MAX_UPLOAD_BYTES,
        "transcript_units": API_UNIT_COST_TRANSCRIPT_FETCH,
        "frame_thumb_units": API_UNIT_COST_FRAMES_THUMB,
        "frame_high_units": API_UNIT_COST_FRAMES_HIGH,
        "text_search_units": API_UNIT_COST_TEXT_QUERY,
        "image_search_units": API_UNIT_COST_IMAGE_QUERY,
        "max_search_image_bytes": MAX_SEARCH_IMAGE_BYTES,
    }


def _workspace_resource_metadata() -> dict:
    import os
    # Resource descriptors are registered during import, before app tariffs exist.
    endpoint = os.environ.get("R2_ENDPOINT_URL", "")
    origins = [origin(endpoint)] if endpoint else []
    return {
        "ui": {
            "domain": origin(mcp_oauth_resource_url()),
            "prefersBorder": False,
            "csp": {"connectDomains": origins, "resourceDomains": origins},
        },
        "openai/ui": {"availableDisplayModes": ["fullscreen"], "preferredDisplayMode": "fullscreen"},
    }


@vmf_mcp.resource(
    UI_URI, name="vmf-workspace", title="Video Workspace", mime_type=UI_MIME,
    meta={"openai/ui": {"availableDisplayModes": ["fullscreen"], "preferredDisplayMode": "fullscreen"}},
)
def workspace_resource() -> str:
    """Return a bundled component, with an exact storage-origin CSP."""
    return UI_PATH.read_text(encoding="utf-8")


@vmf_mcp._mcp_server.list_resources()
async def _list_workspace_resources():
    resources = await vmf_mcp.list_resources()
    for resource in resources:
        if str(resource.uri) == UI_URI:
            resource.meta = _workspace_resource_metadata()
    return resources


# FastMCP resource descriptors carry metadata; ChatGPT also reads it on each
# resources/read content item. Register a low-level reader to preserve both.
@vmf_mcp._mcp_server.read_resource()
async def _read_workspace_resource(uri):
    from mcp.server.lowlevel.helper_types import ReadResourceContents

    if str(uri) != UI_URI:
        # Preserve FastMCP's normal resource and template lookup.
        resource = await vmf_mcp._resource_manager.get_resource(str(uri))
        return [ReadResourceContents(content=await resource.read(), mime_type=resource.mime_type)]
    return [ReadResourceContents(
        content=workspace_resource(), mime_type=UI_MIME,
        meta=_workspace_resource_metadata(),
    )]


@vmf_mcp.tool(
    title="Video Workspace",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
    meta={"ui": {"resourceUri": UI_URI}, "openai/ui": {"entrypoints": [{"type": "global"}, {"type": "thread"}]}},
)
def open_workspace(ctx: Context | None = None):
    """Open VMF beside this conversation. Accepts {}. Listing and allowance are free.

    Use the initial result to populate the UI; do not immediately call again.
    Native results reuse evidence, never index another video automatically.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")
    from src.api.app import v1_api_billing_summary, v1_list_my_videos

    identity = _request_identity(ctx)
    videos = [_workspace_video(v) for v in v1_list_my_videos(identity=identity)[:50]]
    balance = v1_api_billing_summary(identity=identity).model_dump(mode="json")
    return workspace_result({"videos": videos, "allowance": balance}, config=_workspace_config())


@vmf_mcp.tool(
    title="Open Video Source",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
)
def get_workspace_video(video_id: str, ctx: Context | None = None):
    """Refresh playback for an owned video, without retrieving/billing evidence.

    Source URL is ephemeral and UI-only; do not save it in an artifact.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")
    from src.api.app import v1_get_video

    response = v1_get_video(video_id=video_id, identity=_request_identity(ctx))
    return workspace_result({"video": _workspace_video(response)}, source_url=_optional_str(response.source_url), config=_workspace_config())


@vmf_mcp.tool(
    title="Show Learning View",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
    meta={"ui": {"resourceUri": UI_URI}},
)
def render_learning_view(
    view: LearningView,
    companion_views: Annotated[list[LearningView], Field(max_length=3)] | None = None,
    ctx: Context | None = None,
):
    """Display prepared, cited content in VMF's native workspace. No new units.

    Retrieve/inspect evidence first; reuse cached evidence across views. This
    validates shape and citations, not educational correctness. Exactly one of
    sections, cards, slides or playground matches kind. No HTML, scripts, signed
    URLs, local image paths or third-party exports. Tutor remains in chat.
    Pass other prepared views from this conversation in companion_views to keep
    them in one workspace. Every view must refer to the same owned video.
    Offline exports are prepared separately by the packaged skills. Views belong
    to the current conversation; this does not save a cross-session learner profile.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")
    companions = companion_views or []
    if any(v.video_id != view.video_id for v in companions):
        raise ValueError("Companion views must refer to the same video")
    if len({v.kind for v in [view, *companions]}) != len(companions) + 1:
        raise ValueError("Include only one view per workflow")
    from src.api.app import v1_get_video

    response = v1_get_video(video_id=str(view.video_id), identity=_request_identity(ctx))
    if response.status != "ready":
        raise ValueError("Choose a ready video before preparing a learning view")
    return workspace_result(
        {"view": view.model_dump(mode="json"), "companion_views": [v.model_dump(mode="json") for v in companions], "video": _workspace_video(response)},
        source_url=_optional_str(response.source_url), config=_workspace_config(),
    )


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


@vmf_mcp.tool(
    title="Find Similar Frames",
    annotations=ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False),
    meta={"ui": {"visibility": ["app"]}},
)
def search_video_image(
    video_id: Annotated[str, Field(min_length=1)],
    image_base64: Annotated[str, Field(min_length=4, max_length=MAX_SEARCH_IMAGE_BASE64,
        description="Base64 JPEG, PNG or WebP from the workspace file picker; at most 512 KiB and 2048 pixels per side. No URL or data-URI prefix.")],
    limit: Annotated[int, Field(ge=1, le=20)] = 5,
    ctx: Context | None = None,
):
    """Find visually similar moments in the selected owned video.

    Only the workspace calls this tool. Images are used for inference, not
    library ingestion or person identification. Similarity is not proof of an
    exact match. Signed thumbnail capabilities are returned in UI-only metadata.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")
    from PIL import Image as PILImage, UnidentifiedImageError
    from src.api.app import mcp_search_video_by_image

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
        if not image_bytes or len(image_bytes) > MAX_SEARCH_IMAGE_BYTES:
            raise ValueError("Choose an image within the 512 KiB search limit")
        with PILImage.open(io.BytesIO(image_bytes)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"} or max(image.size) > 2048:
                raise ValueError("Choose a JPEG, PNG or WebP of at most 2048 pixels per side")
            image.verify()
    except (binascii.Error, UnidentifiedImageError, OSError, SyntaxError, PILImage.DecompressionBombError) as exc:
        raise ValueError("Choose a valid JPEG, PNG or WebP image") from exc
    response = mcp_search_video_by_image(video_id, image_bytes, limit, _request_identity(ctx))
    matches = [r.model_dump(exclude={"thumbnail_url"}) for r in response.results]
    return workspace_result(
        {"video_id": response.video_id, "status": response.status, "results": matches},
        search_thumbnails=[r.thumbnail_url for r in response.results],
        config=_workspace_config(),
    )


@vmf_mcp.tool(
    title="Get Transcript",
    annotations=ToolAnnotations(
        title="Get Transcript",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
)
def get_transcript(
    video_id: Annotated[str, Field(min_length=1, description="Video UUID to fetch the transcript for.")],
    start_s: Annotated[
        float | None,
        Field(default=None, description="Optional inclusive start time in seconds to filter segments."),
    ] = None,
    end_s: Annotated[
        float | None,
        Field(default=None, description="Optional inclusive end time in seconds to filter segments."),
    ] = None,
    ctx: Context | None = None,
) -> GetTranscriptResult:
    """Return the FULL spoken transcript for a ready video, with per-segment start/end timestamps.

    Use this to read everything the speaker said. Segment ``start_s``/``end_s``
    timestamps can be fed directly into ``get_frames`` to pull the visual
    board/slide content that was on screen at that moment in the video.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import v1_get_video_transcript

    identity = _request_identity(ctx)
    response = v1_get_video_transcript(
        video_id=video_id, start_s=start_s, end_s=end_s, identity=identity,
    )
    return GetTranscriptResult(
        video_id=response.video_id,
        has_transcript=response.has_transcript,
        language_code=response.language_code,
        segment_count=response.segment_count,
        segments=[
            McpTranscriptSegment(
                segment_index=segment.segment_index,
                start_s=segment.start_s,
                end_s=segment.end_s,
                text=segment.text,
            )
            for segment in response.segments
        ],
    )


def _frames_summary_content(
    *,
    video_id: str,
    resolution_requested: str,
    resolution_used: str,
    fallback_used: bool,
    note: str | None,
    plan: list[FramePlanItem],
    resolved_by_key: dict[int, tuple[bytes | None, str | None]],
) -> list[Any]:
    """Build the mixed [summary_dict, *images] tool result shared by both resolutions.

    ``resolved_by_key`` maps each plan item's dedupe key to either
    ``(jpeg_bytes, None)`` on success or ``(None, error_message)`` on failure.
    Frames that share a dedupe key (duplicate rounded timestamps) reuse the same
    image content block instead of duplicating image bytes.
    """
    frame_summaries: list[dict[str, Any]] = []
    images: list[Image] = []
    image_index_by_key: dict[int, int] = {}

    for item in plan:
        key = item.dedupe_key
        if key in image_index_by_key:
            frame_summaries.append(
                {
                    "requested_timestamp_s": item.requested_timestamp_s,
                    "actual_timestamp_s": item.actual_timestamp_s,
                    "image_index": image_index_by_key[key],
                    "error": None,
                }
            )
            continue

        data, error = resolved_by_key.get(key, (None, "Frame not available"))
        if data is not None:
            image_index = len(images)
            images.append(Image(data=data, format="jpeg"))
            image_index_by_key[key] = image_index
            frame_summaries.append(
                {
                    "requested_timestamp_s": item.requested_timestamp_s,
                    "actual_timestamp_s": item.actual_timestamp_s,
                    "image_index": image_index,
                    "error": None,
                }
            )
        else:
            frame_summaries.append(
                {
                    "requested_timestamp_s": item.requested_timestamp_s,
                    "actual_timestamp_s": item.actual_timestamp_s,
                    "image_index": None,
                    "error": error,
                }
            )

    summary = {
        "video_id": video_id,
        "resolution_requested": resolution_requested,
        "resolution_used": resolution_used,
        "fallback_used": fallback_used,
        "note": note,
        "frames": frame_summaries,
    }
    return [summary, *images]


def _resolved_high_res_frames(
    source_url: str, dedupe_keys: list[int],
) -> dict[int, tuple[bytes | None, str | None]]:
    extracted_by_key = extract_high_res_frames(source_url, dedupe_keys)
    resolved: dict[int, tuple[bytes | None, str | None]] = {}
    for key, extracted in extracted_by_key.items():
        if extracted.image_base64:
            resolved[key] = (base64.b64decode(extracted.image_base64), None)
        else:
            resolved[key] = (None, extracted.error or "Frame extraction failed")
    return resolved


def _resolved_thumb_frames(
    video_id: str, dedupe_keys: list[int],
) -> dict[int, tuple[bytes | None, str | None]]:
    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise RuntimeError("Frame storage is not configured") from exc
    store = R2Store(r2_config)

    resolved: dict[int, tuple[bytes | None, str | None]] = {}
    for frame_index in dedupe_keys:
        try:
            resolved[frame_index] = (
                store.download_object_bytes(thumbnail_key(video_id, frame_index)),
                None,
            )
        except R2StorageError as exc:
            logger.warning(
                "Failed to download thumbnail bytes for video_id=%s frame_index=%d: %s",
                video_id,
                frame_index,
                exc,
            )
            resolved[frame_index] = (None, "Failed to download stored thumbnail")
    return resolved


@vmf_mcp.tool(
    title="Get Frames",
    annotations=ToolAnnotations(
        title="Get Frames",
        readOnlyHint=True,
        destructiveHint=False,
        openWorldHint=False,
    ),
    structured_output=False,
)
def get_frames(
    video_id: Annotated[str, Field(min_length=1, description="Video UUID to fetch frames from.")],
    timestamps: Annotated[
        list[float],
        Field(
            min_length=1,
            description=(
                "Timestamps in seconds to extract frames at. Up to 8 for "
                "resolution='high', up to 25 for resolution='thumb'."
            ),
        ),
    ],
    resolution: Annotated[
        Literal["thumb", "high"],
        Field(
            default="high",
            description=(
                "'high' extracts sharp on-demand frames from the retained source "
                "video (best for reading board/slide text; falls back to 'thumb' "
                "automatically if the source isn't retained). 'thumb' returns the "
                "lower-resolution 1-fps thumbnail already stored for the video."
            ),
        ),
    ] = "high",
    ctx: Context | None = None,
) -> list[Any]:
    """Return video frames as actual image content blocks (not URLs) for the model to view.

    Returns a list of content blocks: first a JSON object with fields
    ``video_id``, ``resolution_requested``, ``resolution_used``,
    ``fallback_used``, ``note``, and ``frames`` (a list of
    ``{requested_timestamp_s, actual_timestamp_s, image_index, error}`` in the
    same order as the requested ``timestamps``). ``image_index`` is the
    0-based position of that frame's image among the image content blocks
    that follow this JSON object (``null`` when extraction/download failed for
    that timestamp, with ``error`` describing why).

    Board content accumulates while a speaker writes, so when illustrating a
    transcript moment, prefer a timestamp a few seconds after the speaker
    finishes describing the visual rather than the moment they start.
    """
    if ctx is None:
        raise RuntimeError("MCP context is required")

    from src.api.app import (
        API_UNIT_COST_FRAMES_HIGH,
        API_UNIT_COST_FRAMES_THUMB,
        _api_usage_key_id,
        _bill_metered_call,
        _enforce_search_rate_limit,
        _require_ready_owned_video_or_404,
        _require_retained_source_url,
        _uses_api_unit_billing,
    )

    validate_frame_request(timestamps, resolution)

    identity = _request_identity(ctx)
    user_id = identity.user_id
    _enforce_search_rate_limit(user_id)
    record = _require_ready_owned_video_or_404(video_id, user_id)

    resolution_used = resolution
    fallback_used = False
    note: str | None = None

    if resolution == "high":
        try:
            source_url = _require_retained_source_url(video_id, record)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            resolution_used = "thumb"
            fallback_used = True
            note = (
                "Source video is not retained for this video; served stored "
                "thumbnails instead of extracting high-resolution frames."
            )
        else:
            plan = build_high_res_frame_plan(timestamps)
            dedupe_keys = unique_dedupe_keys(plan)

            def _extract_high_res() -> dict[int, tuple[bytes | None, str | None]]:
                return _resolved_high_res_frames(source_url, dedupe_keys)

            if _uses_api_unit_billing(identity):
                resolved_by_key = _bill_metered_call(
                    user_id=user_id,
                    api_key_id=_api_usage_key_id(identity),
                    event_type="frames_high",
                    units=API_UNIT_COST_FRAMES_HIGH,
                    video_id=video_id,
                    work=_extract_high_res,
                )
            else:
                resolved_by_key = _extract_high_res()
            return _frames_summary_content(
                video_id=video_id,
                resolution_requested=resolution,
                resolution_used="high",
                fallback_used=False,
                note=None,
                plan=plan,
                resolved_by_key=resolved_by_key,
            )

    plan = build_thumb_frame_plan(timestamps, duration_s=record.duration_s)
    dedupe_keys = unique_dedupe_keys(plan)

    def _resolve_thumbs() -> dict[int, tuple[bytes | None, str | None]]:
        return _resolved_thumb_frames(video_id, dedupe_keys)

    if _uses_api_unit_billing(identity):
        resolved_by_key = _bill_metered_call(
            user_id=user_id,
            api_key_id=_api_usage_key_id(identity),
            event_type="frames_thumb",
            units=API_UNIT_COST_FRAMES_THUMB,
            video_id=video_id,
            work=_resolve_thumbs,
        )
    else:
        resolved_by_key = _resolve_thumbs()
    return _frames_summary_content(
        video_id=video_id,
        resolution_requested=resolution,
        resolution_used=resolution_used,
        fallback_used=fallback_used,
        note=note,
        plan=plan,
        resolved_by_key=resolved_by_key,
    )


_LECTURE_NOTES_OWN_NOTES_DISCLOSURE = " and merged with the user's own notes"

_LECTURE_NOTES_OWN_NOTES_STEP = (
    "7. Treat the user's own notes below as the primary skeleton: use the video "
    "to verify, correct, complete, and enrich them, and flag any conflict "
    "between the notes and the video explicitly.\n\n"
    "{own_notes}"
)


@vmf_mcp.prompt(
    name="lecture_notes",
    title="Lecture Notes",
    description=(
        "Turn an indexed lecture video into polished Markdown study notes, "
        "using its transcript and board-moment frames."
    ),
)
def lecture_notes(
    video_id: Annotated[str, Field(description="Video UUID of the indexed lecture to turn into notes.")],
    course_context: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional course name, lecture number, and/or related links.",
        ),
    ] = None,
    own_notes: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Optional text of the user's own handwritten notes to merge in. "
                "Leave empty if the user will paste photos of their notes in chat instead."
            ),
        ),
    ] = None,
) -> str:
    """Workflow prompt: turn an indexed lecture video into polished Markdown study notes."""
    course_context_block = f"\n{course_context}\n" if course_context else ""
    own_notes_disclosure = _LECTURE_NOTES_OWN_NOTES_DISCLOSURE if own_notes else ""
    own_notes_step = (
        "\n" + _LECTURE_NOTES_OWN_NOTES_STEP.format(own_notes=own_notes) + "\n"
        if own_notes
        else ""
    )

    return (
        f"Turn the indexed lecture video {video_id} into polished Markdown study notes.\n"
        f"{course_context_block}\n"
        "Workflow:\n"
        "1. Call get_video_status to confirm the video is ready.\n"
        "2. Call get_transcript for the full transcript.\n"
        "3. Read the transcript and identify (a) the lecture's natural sections and "
        "(b) \"board moments\" — places where the speaker references something "
        "visual without fully describing it (\"let's draw...\", \"as you can see "
        "here...\", \"this diagram...\", \"this picture of...\").\n"
        "4. For the 5-15 most important board moments, call get_frames (default "
        "high resolution). Board content accumulates while the speaker writes, so "
        "request a timestamp near the END of each explanation, a few seconds after "
        "the drawing is finished. If a frame is unclear, request 2-3 nearby "
        "timestamps and use the clearest.\n"
        "5. Describe each important visual faithfully in the notes as text, LaTeX, "
        "or a described diagram — never as \"see figure\" references.\n"
        "6. Write one Markdown document:\n"
        "   - Title + lecture/source metadata at the top.\n"
        "   - A source-status line disclosing the notes are AI-assisted, generated "
        f"from the video transcript and frames{own_notes_disclosure}.\n"
        "   - Numbered sections following the lecture's own structure.\n"
        "   - All math in LaTeX ($$ blocks), with key results in \\boxed{}.\n"
        "   - A final \"Main Takeaways\" bullet list.\n"
        f"{own_notes_step}\n"
        "Keep notation consistent throughout; clean up spoken-language artifacts; "
        "do not invent content that is in neither the transcript nor the frames — "
        "where you add a bridging explanation, make sure it is standard material "
        "implied by the surrounding derivation."
    )


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
