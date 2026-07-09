"""Smoke tests verifying v1 API routes are reachable with correct response shapes."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.db.supabase import VideoRecord
from src.storage.qdrant import SearchResult
from src.video.download import VideoMetadata

from tests.api.conftest import UPLOAD_VIDEO_ID, _authenticate, _video_record


# ------------------------------------------------------------------
# Individual route smoke tests
# ------------------------------------------------------------------


def test_v1_submit_video(monkeypatch) -> None:
    """POST /api/v1/videos returns 200 with VideoResponse shape."""
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda url, uid: (_video_record("vid_1", status="queued"), True),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda video_id: object())
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: True)

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://youtu.be/abc123xyz45"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "vid_1"
    assert payload["status"] == "queued"
    assert payload["source_type"] == "youtube"


def test_v1_get_video(monkeypatch) -> None:
    """GET /api/v1/videos/{id} returns 200 with VideoResponse."""
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    response = client.get("/api/v1/videos/vid_1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "vid_1"
    assert payload["status"] == "ready"


def test_v1_list_videos(monkeypatch) -> None:
    """GET /api/v1/videos returns 200 with a list."""
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.db_list_videos",
        lambda user_id=None: [_video_record("v1"), _video_record("v2")],
    )
    monkeypatch.setattr("src.api.app._source_url_for_record", lambda record: None)

    response = client.get("/api/v1/videos")

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert len(payload) == 2


def test_v1_text_search(monkeypatch) -> None:
    """POST /api/v1/videos/{id}/search returns 200 with VideoSearchResponse."""
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
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
            )
        ],
    )

    response = client.post(
        "/api/v1/videos/vid_ready/search",
        json={"query_text": "robot in blue hoodie"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == "vid_ready"
    assert payload["status"] == "ready"
    assert len(payload["results"]) == 1
    assert payload["results"][0]["timestamp_s"] == 12.5


def test_v1_upload_init(monkeypatch) -> None:
    """POST /api/v1/videos/upload/init returns 200 with UploadInitResponse."""
    client = TestClient(app)
    _authenticate("user_123")

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def generate_presigned_upload_url(self, key, content_type=None, expires_in=900):
            return f"https://signed.example.com/{key}"

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)

    response = client.post(
        "/api/v1/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "video_id" in payload
    assert "upload_url" in payload
    assert payload["expires_in"] == 900


def test_v1_upload_complete(monkeypatch) -> None:
    """POST /api/v1/videos/upload/complete returns 200 with VideoResponse."""
    client = TestClient(app)
    _authenticate("user_123")

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
        lambda video_id, user_id, source_r2_key, source_filename, duration_s=None: (VideoRecord(
            id=video_id,
            youtube_url=None,
            status="queued",
            user_id="user_123",
            error_message=None,
            source_type="upload",
            source_r2_key=source_r2_key,
            source_filename=source_filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ), True),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda video_id: object())

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"


# ------------------------------------------------------------------
# Auth gate
# ------------------------------------------------------------------

V1_ENDPOINTS = [
    ("POST", "/api/v1/videos", {"youtube_url": "https://youtu.be/abc123xyz45"}),
    ("GET", "/api/v1/videos/some_id", None),
    ("GET", "/api/v1/videos", None),
    ("POST", "/api/v1/videos/some_id/search", {"query_text": "hello"}),
    ("POST", "/api/v1/videos/upload/init", {"filename": "f.mp4"}),
    ("POST", "/api/v1/videos/upload/complete", {"video_id": UPLOAD_VIDEO_ID, "filename": "f.mp4"}),
]


@pytest.mark.parametrize("method,path,body", V1_ENDPOINTS)
def test_v1_auth_required(method: str, path: str, body: dict | None) -> None:
    """All v1 endpoints return 401 without Authorization header."""
    client = TestClient(app)
    if method == "GET":
        response = client.get(path)
    else:
        response = client.post(path, json=body)

    assert response.status_code == 401


# ------------------------------------------------------------------
# End-to-end happy path: submit -> poll -> search
# ------------------------------------------------------------------


def test_v1_end_to_end_happy_path(monkeypatch) -> None:
    """Submit a video via v1, poll status, then run text search."""
    client = TestClient(app)
    _authenticate("user_123")

    video_state: dict[str, str] = {}

    def fake_insert_idempotent(url, uid):
        record = _video_record("e2e_vid", status="queued")
        video_state["e2e_vid"] = "queued"
        return record, True

    def fake_get_video(video_id, user_id=None):
        current_status = video_state.get(video_id, "ready")
        return _video_record(video_id, status=current_status)

    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", fake_insert_idempotent)
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda video_id: object())
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=60.0, is_live=False),
    )
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: True)
    monkeypatch.setattr("src.api.app.db_get_video", fake_get_video)
    monkeypatch.setattr(
        "src.api.app.search_video_by_text_service",
        lambda video_id, query_text, limit=5: [
            SearchResult(
                video_id=video_id,
                frame_index=0,
                timestamp_s=5.0,
                thumbnail_url=None,
                score=0.85,
            )
        ],
    )

    # Step 1: submit
    resp = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://youtu.be/abc123xyz45"},
    )
    assert resp.status_code == 200
    vid_id = resp.json()["id"]
    assert resp.json()["status"] == "queued"

    # Step 2: poll (simulate processing complete)
    video_state[vid_id] = "ready"
    resp = client.get(f"/api/v1/videos/{vid_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"

    # Step 3: search
    resp = client.post(
        f"/api/v1/videos/{vid_id}/search",
        json={"query_text": "something interesting"},
    )
    assert resp.status_code == 200
    assert len(resp.json()["results"]) == 1
    assert resp.json()["results"][0]["timestamp_s"] == 5.0
