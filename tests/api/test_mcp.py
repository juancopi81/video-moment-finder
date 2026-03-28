from __future__ import annotations

from datetime import datetime, timezone

import anyio
import httpx
from fastapi.testclient import TestClient
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.api.app import app
from src.db.supabase import ApiCreditRecord, ApiUnitConsumeResult, VideoRecord
from src.storage.qdrant import SearchResult
from tests.api.conftest import UPLOAD_VIDEO_ID, _setup_api_key_auth, _upload_video_record


def _authorized_headers(raw_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {raw_key}"}


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


def test_mcp_requires_authorization_header() -> None:
    with TestClient(app) as client:
        response = client.post("/mcp", json={})

    assert response.status_code == 401
    assert response.json() == {"detail": "Missing Authorization header"}


def test_mcp_rejects_non_vmf_bearer_token() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": "Bearer clerk_jwt_token"},
            json={},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "MCP endpoints require a vmf_ API key"}


def test_mcp_rejects_invalid_vmf_key(monkeypatch) -> None:
    monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda _hash: None)
    monkeypatch.setattr("src.api.auth.touch_api_key_last_used", lambda _kid: None)

    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            headers={"Authorization": "Bearer vmf_invalid"},
            json={},
        )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid authentication token"}


def test_mcp_lists_only_expected_tools(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.list_tools()

    tools = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert {tool.name for tool in tools.tools} == {
        "upload_video",
        "get_video_status",
        "list_videos",
        "search_video",
    }


def test_mcp_upload_video_start_returns_presigned_payload(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate_presigned_upload_url(self, key, content_type=None, expires_in=900):
            return f"https://signed.example.com/{key}"

    monkeypatch.setattr("src.api.app.db_get_api_credits", lambda _uid: ApiCreditRecord(user_id="user_123", balance=600))
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

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

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


def test_mcp_upload_video_complete_returns_video_record(monkeypatch) -> None:
    raw_key, api_key_record = _setup_api_key_auth(monkeypatch)

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
                user_id=api_key_record.user_id,
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

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["action"] == "complete"
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["video"]["id"] == UPLOAD_VIDEO_ID
    assert payload["video"]["status"] == "queued"
    assert payload["video"]["source_type"] == "upload"


def test_mcp_get_video_status_returns_owned_video(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )

    async def _callback(session: ClientSession):
        return await session.call_tool("get_video_status", {"video_id": UPLOAD_VIDEO_ID})

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "ready"
    assert payload["source_type"] == "upload"


def test_mcp_list_videos_respects_limit(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)
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

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["returned_count"] == 2
    assert [video["id"] for video in payload["videos"]] == ["video-1", "video-2"]


def test_mcp_search_video_returns_results(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)
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

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is False
    payload = result.structuredContent
    assert payload is not None
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "ready"
    assert payload["results"][0]["timestamp_s"] == 12.5
    assert payload["results"][0]["source"] == "transcript"


def test_mcp_upload_video_complete_requires_video_id(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)

    async def _callback(session: ClientSession):
        return await session.call_tool(
            "upload_video",
            {
                "action": "complete",
                "filename": "upload.mp4",
            },
        )

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is True
    assert "video_id is required when action is complete" in result.content[0].text


def test_mcp_search_video_surfaces_insufficient_units(monkeypatch) -> None:
    raw_key, _record = _setup_api_key_auth(monkeypatch)
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

    result = _run_mcp_session(_authorized_headers(raw_key), _callback)

    assert result.isError is True
    assert "Insufficient API units" in result.content[0].text


def test_openapi_does_not_include_mcp_mount() -> None:
    with TestClient(app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/mcp" not in response.json()["paths"]
