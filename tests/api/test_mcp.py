from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.api.app import app
from src.api.frames import ExtractedFrame
from src.api.mcp import mcp_tool_approval_items
from src.api.mcp_oauth import MCP_APPROVED_TOOLS_VERSION
from src.db.supabase import (
    ApiCreditRecord,
    ApiUnitConsumeResult,
    McpOAuthAccessTokenRecord,
    TranscriptSegmentRecord,
    VideoRecord,
)
from src.storage.qdrant import SearchResult
from tests.api.conftest import (
    UPLOAD_VIDEO_ID,
    InMemoryMcpOAuthStore,
    _authenticate,
    _upload_video_record,
    _video_record,
)

MCP_RESOURCE_URL = "https://api.videomomentfinder.com/mcp"
CLAUDE_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
CLAUDE_CLIENT_ID = "claude-static-client"
CLAUDE_CLIENT_SECRET = "claude-static-secret"


def _pkce_pair() -> tuple[str, str]:
    verifier = "vmf-test-code-verifier"
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("utf-8")).digest()
    ).decode("utf-8").rstrip("=")
    return verifier, challenge


def _authorized_headers(access_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token}"}


def _run_mcp_session(headers: dict[str, str], callback):
    async def _runner():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=headers,
            follow_redirects=True,
        ) as http_client:
            async with streamable_http_client(
                "http://testserver/mcp",
                http_client=http_client,
            ) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await callback(session)

    with TestClient(app):
        return anyio.run(_runner)


def _issue_access_token(
    monkeypatch,
    *,
    user_id: str = "user_123",
) -> str:
    _authenticate(user_id)
    monkeypatch.setattr(
        "src.api.app.db_get_api_credits",
        lambda _uid: ApiCreditRecord(
            user_id=user_id,
            balance=10_000,
            created_at=None,
            updated_at=None,
        ),
    )

    verifier, challenge = _pkce_pair()

    with TestClient(app) as client:
        authorize_response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": CLAUDE_CLIENT_ID,
                "redirect_uri": CLAUDE_REDIRECT_URI,
                "scope": "vmf:mcp",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": MCP_RESOURCE_URL,
                "state": "state-123",
            },
            follow_redirects=False,
        )
        assert authorize_response.status_code == 302

        request_id = parse_qs(
            urlparse(authorize_response.headers["location"]).query
        )["request_id"][0]

        approve_response = client.post(f"/oauth/mcp/requests/{request_id}/approve")
        assert approve_response.status_code == 200

        code = parse_qs(urlparse(approve_response.json()["redirect_url"]).query)["code"][
            0
        ]
        token_response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "client_id": CLAUDE_CLIENT_ID,
                "client_secret": CLAUDE_CLIENT_SECRET,
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": CLAUDE_REDIRECT_URI,
            },
        )
        assert token_response.status_code == 200
        return token_response.json()["access_token"]


def test_mcp_requires_authorization_header(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    with TestClient(app) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Authentication required",
    }
    assert "resource_metadata=" in response.headers["WWW-Authenticate"]


def test_mcp_rejects_invalid_oauth_token(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": "Bearer not-a-real-token"},
            json={},
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Invalid authentication token",
    }


def test_mcp_rejects_old_manual_vmf_key(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": "Bearer vmf_deadbeef123456"},
            json={},
        )

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Invalid authentication token",
    }


def test_mcp_rejects_insufficient_scope(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    raw_token = "scope-mismatch-token"
    mcp_oauth_store.access_tokens["access-scope-mismatch"] = McpOAuthAccessTokenRecord(
        id="access-scope-mismatch",
        connection_id="conn-scope-mismatch",
        user_id="user_123",
        client_id=CLAUDE_CLIENT_ID,
        token_hash=hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
        scopes=["videos:read"],
        resource=MCP_RESOURCE_URL,
        approved_tools_version=MCP_APPROVED_TOOLS_VERSION,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        revoked_at=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {raw_token}"},
            json={},
        )

    assert response.status_code == 403
    assert response.json() == {
        "error": "insufficient_scope",
        "error_description": "Required scope: vmf:mcp",
    }


def test_mcp_rejects_access_token_from_old_tools_version_grant(
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    raw_token = "old-tools-version-token"
    mcp_oauth_store.access_tokens["access-old-version"] = McpOAuthAccessTokenRecord(
        id="access-old-version",
        connection_id="conn-old-version",
        user_id="user_123",
        client_id=CLAUDE_CLIENT_ID,
        token_hash=hashlib.sha256(raw_token.encode("utf-8")).hexdigest(),
        scopes=["vmf:mcp"],
        resource=MCP_RESOURCE_URL,
        approved_tools_version=MCP_APPROVED_TOOLS_VERSION - 1,
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        revoked_at=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {raw_token}"},
            json={},
        )

    assert response.status_code == 401
    payload = response.json()
    assert payload["error"] == "invalid_token"
    assert payload["error_description"] == (
        "This connection was approved for an older tool list. "
        "Reconnect Video Moment Finder in Claude to approve the updated tools."
    )
    www_authenticate = response.headers["WWW-Authenticate"]
    assert 'error="invalid_token"' in www_authenticate
    assert "Reconnect Video Moment Finder" in www_authenticate
    assert "resource_metadata=" in www_authenticate


def test_mcp_head_allows_tokenless_probe(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    with TestClient(app) as client:
        response = client.head("/mcp")

    assert response.status_code == 204
    assert response.text == ""


def test_mcp_returns_503_when_oauth_not_configured(monkeypatch) -> None:
    for key in (
        "MCP_OAUTH_ISSUER_URL",
        "MCP_OAUTH_RESOURCE_URL",
        "MCP_OAUTH_CLIENT_ID",
        "MCP_OAUTH_CLIENT_SECRET",
        "FRONTEND_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 503
    assert response.json() == {"detail": "MCP OAuth is not configured"}


def test_mcp_lists_only_expected_tools(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.list_tools()

    tools = _run_mcp_session(_authorized_headers(access_token), _callback)

    tool_names = {tool.name for tool in tools.tools}
    assert tool_names == {
        "upload_video",
        "get_video_status",
        "list_videos",
        "search_video",
        "get_transcript",
        "get_frames",
    }

    upload_tool = next(tool for tool in tools.tools if tool.name == "upload_video")
    search_tool = next(tool for tool in tools.tools if tool.name == "search_video")
    transcript_tool = next(tool for tool in tools.tools if tool.name == "get_transcript")
    frames_tool = next(tool for tool in tools.tools if tool.name == "get_frames")
    assert upload_tool.title == "Upload Video"
    assert upload_tool.annotations.destructiveHint is True
    assert upload_tool.annotations.readOnlyHint is False
    assert search_tool.annotations.readOnlyHint is True
    assert transcript_tool.annotations.readOnlyHint is True
    assert frames_tool.annotations.readOnlyHint is True


def test_mcp_upload_video_start_returns_presigned_payload(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate_presigned_upload_url(self, key, content_type=None, expires_in=900):
            return f"https://signed.example.com/{key}"

    monkeypatch.setattr(
        "src.api.app.db_get_api_credits",
        lambda _uid: ApiCreditRecord(user_id="user_123", balance=600),
    )
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "upload_video",
            {
                "action": "start",
                "filename": "upload.mp4",
                "content_type": "video/mp4",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["action"] == "start"
    assert payload["video_id"]
    assert payload["upload_url"].startswith("https://signed.example.com/")
    assert payload["method"] == "PUT"
    assert payload["expires_in_seconds"] == 900
    assert payload["do_not_send_headers"] == ["Authorization"]
    assert payload["next_action"] == "complete"


def test_mcp_upload_video_complete_returns_video_record(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return True

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda video_id, user_id, source_r2_key, source_filename, duration_s=None: (
            VideoRecord(
                id=video_id,
                youtube_url=None,
                status="queued",
                user_id=user_id,
                error_message=None,
                source_type="upload",
                source_r2_key=source_r2_key,
                source_filename=source_filename,
                created_at=datetime.now(timezone.utc).isoformat(),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ),
            True,
        ),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda video_id: object())
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=100),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "upload_video",
            {
                "action": "complete",
                "video_id": UPLOAD_VIDEO_ID,
                "filename": "upload.mp4",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["action"] == "complete"
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["video"]["id"] == UPLOAD_VIDEO_ID
    assert payload["video"]["status"] == "queued"
    assert payload["video"]["source_type"] == "upload"


def test_mcp_get_video_status_returns_owned_video(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool("get_video_status", {"video_id": UPLOAD_VIDEO_ID})

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "ready"
    assert payload["source_type"] == "upload"


def test_mcp_list_videos_respects_limit(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_list_videos",
        lambda user_id=None: [
            _upload_video_record("video-1"),
            _upload_video_record("video-2"),
            _upload_video_record("video-3"),
        ],
    )
    monkeypatch.setattr(
        "src.api.app._source_url_for_record",
        lambda record: "https://example.com/source.mp4",
    )

    async def _callback(session: ClientSession):
        return await session.call_tool("list_videos", {"limit": 2})

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["returned_count"] == 2
    assert [video["id"] for video in payload["videos"]] == ["video-1", "video-2"]


def test_mcp_search_video_returns_results(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=100),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_by_text_service",
        lambda video_id, query_text, limit=5: [
            SearchResult(
                video_id=video_id,
                frame_index=0,
                timestamp_s=12.5,
                thumbnail_url=None,
                score=0.92,
                source="transcript",
                transcript_text="The model is explained here.",
            )
        ],
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "search_video",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "query_text": "when do they explain the model?",
                "limit": 3,
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "ready"
    assert payload["results"][0]["timestamp_s"] == 12.5
    assert payload["results"][0]["source"] == "transcript"


def test_mcp_upload_video_complete_requires_video_id(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "upload_video",
            {
                "action": "complete",
                "filename": "upload.mp4",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert "video_id is required when action is complete" in result.content[0].text


def test_mcp_search_video_surfaces_insufficient_units(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "search_video",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "query_text": "needle in the video",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert "Insufficient API units" in result.content[0].text


def test_mcp_tool_approval_items_lists_six_tools() -> None:
    items = mcp_tool_approval_items()

    assert [item["name"] for item in items] == [
        "upload_video",
        "get_video_status",
        "list_videos",
        "search_video",
        "get_transcript",
        "get_frames",
    ]


def test_mcp_tool_approval_items_list_exact_unit_costs() -> None:
    costs = {item["name"]: item["cost"] for item in mcp_tool_approval_items()}

    assert costs == {
        "upload_video": "500 units per indexed video",
        "get_video_status": "No units",
        "list_videos": "No units",
        "search_video": "1 unit per search query",
        "get_transcript": "1 unit per call",
        "get_frames": "1 unit per thumbnail call, 5 units per high-res call",
    }


# ---------------------------------------------------------------------------
# get_transcript
# ---------------------------------------------------------------------------


def test_mcp_get_transcript_returns_segments_and_bills_transcript_fetch(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.db_get_video_transcript_segments",
        lambda video_id, start_s=None, end_s=None: [
            TranscriptSegmentRecord(
                video_id=video_id, segment_index=0, start_s=0.0, end_s=2.0,
                text="Hello", language_code="en",
            ),
            TranscriptSegmentRecord(
                video_id=video_id, segment_index=1, start_s=2.0, end_s=4.5,
                text="World", language_code="en",
            ),
        ],
    )
    consumed: dict = {}

    def mock_consume(**kwargs):
        consumed.update(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

    async def _callback(session: ClientSession):
        return await session.call_tool("get_transcript", {"video_id": UPLOAD_VIDEO_ID})

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["has_transcript"] is True
    assert payload["language_code"] == "en"
    assert payload["segment_count"] == 2
    assert payload["segments"][0] == {
        "segment_index": 0,
        "start_s": 0.0,
        "end_s": 2.0,
        "text": "Hello",
    }
    assert consumed["event_type"] == "transcript_fetch"
    assert consumed["units"] == 1


def test_mcp_get_transcript_no_segments_returns_empty_not_error(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.db_get_video_transcript_segments",
        lambda video_id, start_s=None, end_s=None: [],
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool("get_transcript", {"video_id": UPLOAD_VIDEO_ID})

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload["has_transcript"] is False
    assert payload["segment_count"] == 0
    assert payload["segments"] == []


# ---------------------------------------------------------------------------
# get_frames
# ---------------------------------------------------------------------------


class _FakeRetainedSourceR2Store:
    """Fake for src.api.app.R2Store used by _require_retained_source_url."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def source_exists(self, key: str) -> bool:
        return True

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return "https://source.example.com/presigned"


def _fake_thumb_bytes_r2_store(thumb_bytes: bytes):
    class _FakeThumbBytesR2Store:
        """Fake for src.api.mcp.R2Store used by _resolved_thumb_frames."""

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def download_object_bytes(self, key: str) -> bytes:
            return thumb_bytes

    return _FakeThumbBytesR2Store


def test_mcp_get_frames_high_returns_image_content_and_bills_frames_high(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", _FakeRetainedSourceR2Store)

    jpeg_bytes = b"\xff\xd8\xff\xe0fakejpegbytes"
    monkeypatch.setattr(
        "src.api.mcp.extract_high_res_frames",
        lambda source_url, dedupe_keys: {
            key: ExtractedFrame(
                image_base64=base64.b64encode(jpeg_bytes).decode("ascii"),
                width=640,
                height=360,
            )
            for key in dedupe_keys
        },
    )
    consumed: dict = {}

    def mock_consume(**kwargs):
        consumed.update(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [3.2], "resolution": "high"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    assert result.structuredContent is None
    assert len(result.content) == 2
    summary_block, image_block = result.content
    assert summary_block.type == "text"
    payload = json.loads(summary_block.text)
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["resolution_requested"] == "high"
    assert payload["resolution_used"] == "high"
    assert payload["fallback_used"] is False
    assert payload["frames"][0]["actual_timestamp_s"] == 3.0
    assert payload["frames"][0]["image_index"] == 0
    assert payload["frames"][0]["error"] is None
    assert image_block.type == "image"
    assert image_block.mimeType == "image/jpeg"
    assert base64.b64decode(image_block.data) == jpeg_bytes
    assert consumed["event_type"] == "frames_high"
    assert consumed["units"] == 5


def test_mcp_get_frames_thumb_returns_image_content_and_bills_frames_thumb(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.mcp.R2Config.from_env", lambda: object())
    thumb_bytes = b"thumb-jpeg-bytes"
    monkeypatch.setattr("src.api.mcp.R2Store", _fake_thumb_bytes_r2_store(thumb_bytes))
    consumed: dict = {}

    def mock_consume(**kwargs):
        consumed.update(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "thumb"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    summary_block, image_block = result.content
    payload = json.loads(summary_block.text)
    assert payload["resolution_requested"] == "thumb"
    assert payload["resolution_used"] == "thumb"
    assert payload["fallback_used"] is False
    assert image_block.type == "image"
    assert base64.b64decode(image_block.data) == thumb_bytes
    assert consumed["event_type"] == "frames_thumb"
    assert consumed["units"] == 1


def test_mcp_get_frames_high_falls_back_to_thumb_when_source_not_retained(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    # Youtube-sourced video: no retained R2 source, so the high-res path 409s.
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.mcp.R2Config.from_env", lambda: object())
    thumb_bytes = b"fallback-thumb-bytes"
    monkeypatch.setattr("src.api.mcp.R2Store", _fake_thumb_bytes_r2_store(thumb_bytes))
    consumed: dict = {}

    def mock_consume(**kwargs):
        consumed.update(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "high"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    summary_block, image_block = result.content
    payload = json.loads(summary_block.text)
    assert payload["resolution_requested"] == "high"
    assert payload["resolution_used"] == "thumb"
    assert payload["fallback_used"] is True
    assert payload["note"] is not None and "not retained" in payload["note"]
    assert image_block.type == "image"
    assert base64.b64decode(image_block.data) == thumb_bytes
    # Only the (cheaper) thumb event is billed -- no double billing on fallback.
    assert consumed["event_type"] == "frames_thumb"
    assert consumed["units"] == 1


def test_mcp_get_frames_high_cap_exceeded_is_tool_error(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "timestamps": [float(i) for i in range(9)],
                "resolution": "high",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert "At most 8 timestamps" in result.content[0].text


def test_mcp_get_frames_thumb_cap_exceeded_is_tool_error(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "timestamps": [float(i) for i in range(26)],
                "resolution": "thumb",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert "At most 25 timestamps" in result.content[0].text


def test_mcp_get_frames_per_frame_error_does_not_fail_whole_call(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", _FakeRetainedSourceR2Store)

    jpeg_bytes = b"\xff\xd8\xff\xe0okframe"

    def fake_extract(source_url, dedupe_keys):
        result = {}
        for key in dedupe_keys:
            if key == 500:
                result[key] = ExtractedFrame(
                    image_base64=None, width=None, height=None,
                    error="No frame was produced (timestamp may be past the end of the video)",
                )
            else:
                result[key] = ExtractedFrame(
                    image_base64=base64.b64encode(jpeg_bytes).decode("ascii"),
                    width=100,
                    height=50,
                )
        return result

    monkeypatch.setattr("src.api.mcp.extract_high_res_frames", fake_extract)
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "timestamps": [5.0, 500.0],
                "resolution": "high",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    summary_block, image_block = result.content
    payload = json.loads(summary_block.text)
    assert payload["frames"][0]["image_index"] == 0
    assert payload["frames"][0]["error"] is None
    assert payload["frames"][1]["image_index"] is None
    assert "past the end" in payload["frames"][1]["error"]
    assert image_block.type == "image"


def test_mcp_get_frames_rate_limit_enforced_same_as_search(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.mcp.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.mcp.R2Store", _fake_thumb_bytes_r2_store(b"thumb-bytes"))
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
    )

    async def _callback(session: ClientSession):
        first = await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "thumb"},
        )
        second = await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "thumb"},
        )
        return first, second

    first, second = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert first.isError is False
    assert second.isError is True
    assert "Rate limit exceeded" in second.content[0].text


def test_mcp_get_frames_not_ready_video_is_tool_error_without_billing(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="processing"),
    )
    consumed = {"called": False}

    def mock_consume(**kwargs):
        consumed["called"] = True
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "thumb"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert "Video not ready" in result.content[0].text
    assert consumed["called"] is False


def test_mcp_get_frames_wholesale_extraction_failure_compensates_units(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", _FakeRetainedSourceR2Store)

    consumed: list[dict] = []
    compensated: list[dict] = []

    def mock_consume(**kwargs):
        consumed.append(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    def mock_compensate(**kwargs):
        compensated.append(kwargs)

    def fake_extract(*_args, **_kwargs):
        raise RuntimeError("extraction subsystem crashed")

    monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
    monkeypatch.setattr("src.api.app.db_compensate_api_units", mock_compensate)
    monkeypatch.setattr("src.api.mcp.extract_high_res_frames", fake_extract)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [1.0], "resolution": "high"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is True
    assert len(consumed) == 1
    assert len(compensated) == 1
    assert consumed[0]["request_id"] == compensated[0]["request_id"]
    assert compensated[0]["units"] == 5
    assert compensated[0]["metadata"]["event_type"] == "frames_high_failed"


def test_mcp_get_frames_per_frame_error_does_not_compensate_units(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", _FakeRetainedSourceR2Store)

    def fake_extract(source_url, dedupe_keys):
        return {
            key: (
                ExtractedFrame(image_base64=None, width=None, height=None, error="No frame")
                if key == 500
                else ExtractedFrame(image_base64=base64.b64encode(b"jpeg").decode("ascii"), width=1, height=1)
            )
            for key in dedupe_keys
        }

    monkeypatch.setattr("src.api.mcp.extract_high_res_frames", fake_extract)
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
    )
    compensated = {"called": False}
    monkeypatch.setattr(
        "src.api.app.db_compensate_api_units",
        lambda **kwargs: compensated.update(called=True),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [5.0, 500.0], "resolution": "high"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    assert compensated["called"] is False


def test_mcp_get_frames_thumb_clamps_to_video_duration_s(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready", duration_s=10.0),
    )
    monkeypatch.setattr("src.api.mcp.R2Config.from_env", lambda: object())
    thumb_bytes = b"thumb-bytes"
    downloaded_keys: list[str] = []

    class _RecordingThumbStore:
        def __init__(self, *_a, **_kw) -> None:
            pass

        def download_object_bytes(self, key: str) -> bytes:
            downloaded_keys.append(key)
            return thumb_bytes

    monkeypatch.setattr("src.api.mcp.R2Store", _RecordingThumbStore)
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "get_frames",
            {"video_id": UPLOAD_VIDEO_ID, "timestamps": [999999.0], "resolution": "thumb"},
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert result.isError is False
    summary_block, _image_block = result.content
    payload = json.loads(summary_block.text)
    # A 10s video's last full sampled frame is index 9, not the global cap.
    assert payload["frames"][0]["actual_timestamp_s"] == 9.0
    assert downloaded_keys == [f"thumb/{UPLOAD_VIDEO_ID}/thumb_00009.jpg"]


# ---------------------------------------------------------------------------
# lecture_notes prompt
# ---------------------------------------------------------------------------


def test_mcp_lecture_notes_prompt_renders_without_optional_args(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.get_prompt("lecture_notes", {"video_id": UPLOAD_VIDEO_ID})

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    assert len(result.messages) == 1
    text = result.messages[0].content.text
    assert f"lecture video {UPLOAD_VIDEO_ID}" in text
    assert "get_video_status" in text
    assert "get_transcript" in text
    assert "get_frames" in text
    assert "Main Takeaways" in text
    assert "primary skeleton" not in text
    assert "merged with the user's own notes" not in text


def test_mcp_lecture_notes_prompt_renders_with_optional_args(
    monkeypatch,
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    access_token = _issue_access_token(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.get_prompt(
            "lecture_notes",
            {
                "video_id": UPLOAD_VIDEO_ID,
                "course_context": "CS229, Lecture 3",
                "own_notes": "Gradient descent notes from class.",
            },
        )

    result = _run_mcp_session(_authorized_headers(access_token), _callback)

    text = result.messages[0].content.text
    assert "CS229, Lecture 3" in text
    assert "merged with the user's own notes" in text
    assert "primary skeleton" in text
    assert "Gradient descent notes from class." in text
