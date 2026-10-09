"""Reference-image search must validate, isolate owners and meter/refund correctly."""
from __future__ import annotations

import base64
import io
import json

import pytest
from PIL import Image

from src.db.supabase import ApiUnitConsumeResult
from src.storage.qdrant import SearchResult
from tests.api.conftest import UPLOAD_VIDEO_ID, _upload_video_record
from tests.api.test_mcp import _authorized_headers, _issue_access_token, _run_mcp_session


def image_data(size=(32, 24), format="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", size, "white").save(stream, format=format)
    return base64.b64encode(stream.getvalue()).decode()


@pytest.fixture
def image_search(monkeypatch, mcp_oauth_store):
    token = _issue_access_token(monkeypatch)
    charges, refunds, searches, owners = [], [], [], []

    def owned(video_id, user_id):
        owners.append(user_id)
        return _upload_video_record(video_id, status="ready")

    def consume(**kwargs):
        charges.append(kwargs)
        return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

    def search(**kwargs):
        searches.append(kwargs)
        return [SearchResult(video_id=UPLOAD_VIDEO_ID, frame_index=0, timestamp_s=6,
                             score=.87, source="visual", thumbnail_url="https://legacy.test/frame.jpg")]

    monkeypatch.setattr("src.api.app.db_get_video", owned)
    monkeypatch.setattr("src.api.app.db_consume_api_units", consume)
    monkeypatch.setattr("src.api.app.db_compensate_api_units", lambda **kw: refunds.append(kw))
    monkeypatch.setattr("src.api.app.search_video_by_image_service", search)
    monkeypatch.setattr("src.api.app._search_thumbnail_urls", lambda *_: {0: "https://storage.test/frame?X-Amz-Signature=secret"})
    monkeypatch.setattr("src.api.app._source_url_for_record", lambda *_: "https://storage.test/video?X-Amz-Signature=source")
    return token, charges, refunds, searches, owners


def call_image(token, data, **extra):
    async def callback(session):
        return await session.call_tool("search_video_image", {
            "video_id": UPLOAD_VIDEO_ID, "image_base64": data, **extra,
        })
    return _run_mcp_session(_authorized_headers(token), callback)


def test_image_search_bills_current_tariff_and_hides_media_capabilities(image_search, monkeypatch):
    token, charges, refunds, searches, owners = image_search
    monkeypatch.setattr("src.api.app.API_UNIT_COST_IMAGE_QUERY", 3)
    result = call_image(token, image_data(), limit=2)
    assert not result.isError
    assert result.structuredContent["results"][0]["timestamp_s"] == 6
    assert "X-Amz" not in json.dumps(result.structuredContent)
    assert "X-Amz" not in result.content[0].text
    assert "X-Amz-Signature=secret" in result.meta["vmf"]["search_thumbnails"][0]
    assert result.meta["vmf"]["config"]["image_search_units"] == 3
    assert owners == ["user_123"]
    assert charges[0]["event_type"] == "image_query"
    assert charges[0]["units"] == 3
    assert charges[0]["api_key_id"] is None
    assert charges[0]["video_id"] == UPLOAD_VIDEO_ID
    assert searches[0]["query_image_bytes"] == base64.b64decode(image_data())
    assert searches[0]["limit"] == 2
    assert refunds == []


@pytest.mark.parametrize("data", ["bad!", base64.b64encode(b"not an image").decode(),
    "data:image/png;base64," + image_data(), image_data((2049, 10)),
    image_data(format="GIF"), base64.b64encode(b"x" * (512 * 1024 + 1)).decode()])
def test_invalid_image_never_bills_or_invokes_inference(image_search, data):
    token, charges, refunds, searches, _ = image_search
    assert call_image(token, data).isError
    assert charges == refunds == searches == []


@pytest.mark.parametrize("status", [None, "processing"])
def test_unowned_or_unready_video_never_bills(image_search, monkeypatch, status):
    token, charges, refunds, searches, _ = image_search
    monkeypatch.setattr("src.api.app.db_get_video", lambda *_args, **_kw: None if status is None else _upload_video_record(UPLOAD_VIDEO_ID, status=status))
    assert call_image(token, image_data()).isError
    assert charges == refunds == searches == []


def test_insufficient_allowance_does_not_run_image_inference(image_search, monkeypatch):
    token, _, refunds, searches, _ = image_search
    monkeypatch.setattr("src.api.app.db_consume_api_units", lambda **_: ApiUnitConsumeResult(allowed=False, remaining_balance=0))
    result = call_image(token, image_data())
    assert result.isError and "Insufficient API units" in result.content[0].text
    assert refunds == searches == []


def test_failed_image_search_refunds_exact_original_debit(image_search, monkeypatch):
    token, charges, refunds, _, _ = image_search
    def fail(**_):
        raise RuntimeError("inference unavailable")
    monkeypatch.setattr("src.api.app.search_video_by_image_service", fail)
    result = call_image(token, image_data())
    assert result.isError
    assert len(charges) == len(refunds) == 1
    assert refunds[0]["request_id"] == charges[0]["request_id"]
    assert refunds[0]["units"] == charges[0]["units"]
    assert refunds[0]["metadata"] == {"event_type": "image_query_failed"}


def test_image_tool_is_app_only_with_bounded_payload(image_search):
    async def callback(session):
        return await session.list_tools()
    tools = _run_mcp_session(_authorized_headers(image_search[0]), callback)
    tool = next(t for t in tools.tools if t.name == "search_video_image")
    assert tool.meta["ui"]["visibility"] == ["app"]
    assert tool.inputSchema["properties"]["image_base64"]["maxLength"] <= 700000
