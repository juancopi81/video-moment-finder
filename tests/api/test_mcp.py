from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
from urllib.parse import parse_qs, urlparse

import anyio
import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.api.app import app
from src.db.supabase import (
    ApiCreditRecord,
    ApiUnitConsumeResult,
    McpOAuthAccessTokenRecord,
    VideoRecord,
)
from src.storage.qdrant import SearchResult
from tests.api.conftest import (
    UPLOAD_VIDEO_ID,
    InMemoryMcpOAuthStore,
    _authenticate,
    _upload_video_record,
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


def test_mcp_head_allows_tokenless_probe(mcp_oauth_store: InMemoryMcpOAuthStore) -> None:
    with TestClient(app) as client:
        response = client.head("/mcp")

    assert response.status_code == 204
    assert response.text == ""


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
    }

    upload_tool = next(tool for tool in tools.tools if tool.name == "upload_video")
    search_tool = next(tool for tool in tools.tools if tool.name == "search_video")
    assert upload_tool.title == "Upload Video"
    assert upload_tool.annotations.destructiveHint is True
    assert upload_tool.annotations.readOnlyHint is False
    assert search_tool.annotations.readOnlyHint is True


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
        lambda video_id, user_id, source_r2_key, source_filename: (
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
