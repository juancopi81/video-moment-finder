"""Exercise privacy boundaries on actual MCP wire responses, not just DTOs."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tests.api.conftest import UPLOAD_VIDEO_ID, _upload_video_record
from tests.api.test_image_search_mcp import image_data
from tests.api.test_mcp import _authorized_headers, _issue_access_token, _run_mcp_session
from tests.api.test_workspace import sample_view

DIAGNOSTIC = "internal_user_private person@example.invalid trace_private /srv/private/database"
MEDIA = "https://storage.example.test/source?X-Amz-Credential=private&X-Amz-Signature=secret"


def call(token, name, arguments):
    async def callback(session):
        return await session.call_tool(name, arguments)
    return _run_mcp_session(_authorized_headers(token), callback)


def visible(result):
    return json.dumps({"content": [c.model_dump(mode="json") for c in result.content],
                       "structuredContent": result.structuredContent})


@pytest.fixture
def privacy_token(monkeypatch, mcp_oauth_store):
    return _issue_access_token(monkeypatch)


@pytest.mark.parametrize("name,arguments", [
    ("open_workspace", {}),
    ("list_videos", {}),
    ("get_video_status", {"video_id": UPLOAD_VIDEO_ID}),
    ("get_workspace_video", {"video_id": UPLOAD_VIDEO_ID}),
    ("render_learning_view", {"view": sample_view()}),
    ("upload_video", {"action": "complete", "filename": "lesson.mp4", "video_id": UPLOAD_VIDEO_ID}),
])
def test_video_responses_allowlist_fields_and_keep_playback_in_ui_metadata(privacy_token, monkeypatch, name, arguments):
    response = SimpleNamespace(
        id=UPLOAD_VIDEO_ID, status="ready", source_type="upload", source_filename="lesson.mp4",
        youtube_url=None, source_url=MEDIA, created_at="2026-10-01T12:34:56Z",
        error_message=DIAGNOSTIC, user_id="internal_user_private", debug={"identity": DIAGNOSTIC},
    )
    monkeypatch.setattr("src.api.app.v1_list_my_videos", lambda **_: [response])
    monkeypatch.setattr("src.api.app.v1_get_video", lambda **_: response)
    monkeypatch.setattr("src.api.app.v1_complete_upload", lambda *_args, **_: response)
    monkeypatch.setattr("src.api.app.v1_api_billing_summary", lambda **_: SimpleNamespace(model_dump=lambda **_: {
        "api_units_balance": 600, "unit_cost_index_video": 500, "user_id": "internal_user_private",
        "trial_units_granted": 600, "trial_legacy_units_offset": 100, "debug": DIAGNOSTIC,
    }))
    result = call(privacy_token, name, arguments)
    assert not result.isError
    data = result.structuredContent
    videos = data.get("videos", [data.get("video", data)])
    for video in videos:
        assert set(video) == {"id", "status", "source_type", "source_filename", "youtube_url", "error_message"}
        assert video["source_filename"] == "lesson.mp4"  # Required to select the source.
        assert video["error_message"] is None
    if name == "open_workspace":
        assert data["allowance"] == {"api_units_balance": 600, "unit_cost_index_video": 500}
    if name in {"get_workspace_video", "render_learning_view"}:
        assert result.meta["vmf"]["source_url"] == MEDIA
    for secret in (DIAGNOSTIC, "internal_user_private", "X-Amz-", "2026-10-01T12:34:56Z"):
        assert secret not in visible(result)


def test_failed_video_reports_actionable_status_without_stored_exception(privacy_token, monkeypatch):
    response = SimpleNamespace(id=UPLOAD_VIDEO_ID, status="failed", source_type="upload",
                               source_filename="lesson.mp4", youtube_url=None, error_message=DIAGNOSTIC)
    monkeypatch.setattr("src.api.app.v1_get_video", lambda **_: response)
    result = call(privacy_token, "get_video_status", {"video_id": UPLOAD_VIDEO_ID})
    assert not result.isError
    assert result.structuredContent["status"] == "failed"
    assert "processing failed" in result.structuredContent["error_message"]
    assert DIAGNOSTIC not in result.model_dump_json()


@pytest.mark.parametrize("name", ["search_video", "search_video_image"])
def test_search_matches_are_allowlisted_and_capabilities_are_ui_only(privacy_token, monkeypatch, name):
    match = SimpleNamespace(timestamp_s=7, score=.9, source="visual", transcript_text=None,
                            thumbnail_url=MEDIA, debug=DIAGNOSTIC, user_id="internal_user_private")
    response = SimpleNamespace(video_id=UPLOAD_VIDEO_ID, status="ready", results=[match], source_url=MEDIA)
    monkeypatch.setattr("src.api.app.v1_search_video", lambda **_: response)
    monkeypatch.setattr("src.api.app.mcp_search_video_by_image", lambda *_: response)
    arguments = {"video_id": UPLOAD_VIDEO_ID}
    arguments.update({"query_text": "a diagram"} if name == "search_video" else {"image_base64": image_data()})
    result = call(privacy_token, name, arguments)
    assert not result.isError
    assert set(result.structuredContent) == {"video_id", "status", "results"}
    assert result.structuredContent["results"] == [{"timestamp_s": 7, "score": .9, "source": "visual", "transcript_text": None}]
    assert result.meta["vmf"]["search_thumbnails"] == [MEDIA]
    assert DIAGNOSTIC not in result.model_dump_json()
    assert "X-Amz-" not in visible(result)


@pytest.mark.parametrize("error", [RuntimeError(DIAGNOSTIC), ValueError(DIAGNOSTIC),
                                      HTTPException(503, {"debug": DIAGNOSTIC}), HTTPException(404, DIAGNOSTIC)])
def test_tool_exceptions_never_echo_backend_details(privacy_token, monkeypatch, error):
    def fail(**_):
        raise error
    monkeypatch.setattr("src.api.app.v1_list_my_videos", fail)
    result = call(privacy_token, "list_videos", {})
    assert result.isError
    assert DIAGNOSTIC not in result.model_dump_json()
    assert result.structuredContent is None
    assert result.meta is None
    if isinstance(error, HTTPException) and error.status_code == 404:
        assert "Video not found" in result.content[0].text


def test_validation_errors_and_unknown_tool_names_never_echo_inputs(privacy_token):
    invalid = call(privacy_token, "get_video_status", {"video_id": {"private": DIAGNOSTIC}})
    unknown = call(privacy_token, DIAGNOSTIC, {})
    for result in (invalid, unknown):
        assert result.isError
        assert DIAGNOSTIC not in result.model_dump_json()


def test_per_frame_errors_do_not_return_ffmpeg_urls_or_identifiers(privacy_token, monkeypatch):
    monkeypatch.setattr("src.api.app.db_get_video", lambda *_args, **_kw: _upload_video_record(UPLOAD_VIDEO_ID, status="ready"))
    monkeypatch.setattr("src.api.app._require_retained_source_url", lambda *_: MEDIA)
    monkeypatch.setattr("src.api.app._uses_api_unit_billing", lambda *_: False)
    monkeypatch.setattr("src.api.mcp._resolved_high_res_frames", lambda *_: {4: (None, DIAGNOSTIC + MEDIA)})
    result = call(privacy_token, "get_frames", {"video_id": UPLOAD_VIDEO_ID, "timestamps": [4], "resolution": "high"})
    assert not result.isError
    summary = json.loads(result.content[0].text)
    assert summary["frames"][0]["image_index"] is None
    assert "Frame unavailable" in summary["frames"][0]["error"]
    assert DIAGNOSTIC not in result.model_dump_json()
    assert "X-Amz-" not in visible(result)
