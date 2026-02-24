from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, _allowed_cors_origins
from src.api.auth import get_current_user_id
from src.db.supabase import VideoRecord
from src.storage.config import StorageConfigError
from src.storage.qdrant import SearchResult
from src.video.download import VideoMetadata
from src.video.youtube import extract_youtube_video_id

UPLOAD_VIDEO_ID = "00000000-0000-4000-8000-000000000123"


def _video_record(video_id: str, *, status: str = "queued") -> VideoRecord:
    return VideoRecord(
        id=video_id,
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status=status,  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="youtube",
        source_r2_key=None,
        source_filename=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def _upload_video_record(video_id: str, *, status: str = "ready") -> VideoRecord:
    return VideoRecord(
        id=video_id,
        youtube_url=None,
        status=status,  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="upload",
        source_r2_key="source/video_123/upload.mp4",
        source_filename="upload.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture(autouse=True)
def _clear_dependency_overrides() -> None:
    app.dependency_overrides.clear()
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _mock_free_video_count(monkeypatch) -> None:
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 0)


def _authenticate(user_id: str = "user_123") -> None:
    app.dependency_overrides[get_current_user_id] = lambda: user_id


def test_create_video_requires_authentication(monkeypatch) -> None:
    client = TestClient(app)
    called = False

    def fake_create_video(*args, **kwargs) -> VideoRecord:
        nonlocal called
        called = True
        return _video_record("video_unauth")

    monkeypatch.setattr("src.api.app.db_create_video", fake_create_video)

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert called is False


def test_create_video_enqueues_job_for_authenticated_owner(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    enqueue_calls: list[str] = []
    create_calls: list[tuple[str, str | None, str]] = []

    def fake_create_video(youtube_url: str, user_id=None, status="queued") -> VideoRecord:
        create_calls.append((youtube_url, user_id, status))
        return _video_record("video_123", status=status)

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.db_create_video", fake_create_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )

    response = client.post(
        "/videos",
        json={"youtube_url": "https://youtu.be/abc123xyz45"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "video_123"
    assert payload["status"] == "queued"
    assert payload["youtube_url"] == "https://www.youtube.com/watch?v=abc123xyz45"
    assert payload["source_type"] == "youtube"
    assert enqueue_calls == ["video_123"]
    assert create_calls == [
        ("https://www.youtube.com/watch?v=abc123xyz45", "user_123", "queued")
    ]


def test_create_video_returns_500_when_enqueue_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    failure_updates: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        "src.api.app.db_create_video",
        lambda youtube_url, user_id=None, status="queued": _video_record(
            "video_500",
            status=status,
        ),
    )

    def fake_update(video_id: str, status: str, error_message: str | None = None):
        failure_updates.append((video_id, status, error_message))
        return _video_record(video_id, status="failed")

    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: (_ for _ in ()).throw(RuntimeError("queue down")),
    )
    monkeypatch.setattr("src.api.app.update_video_status", fake_update)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to enqueue processing job"
    assert failure_updates
    assert failure_updates[0][0] == "video_500"
    assert failure_updates[0][1] == "failed"


def test_create_video_rejects_non_video_youtube_url() -> None:
    client = TestClient(app)
    _authenticate("user_123")

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/channel/UC12345"},
    )

    assert response.status_code == 422


def test_create_video_rejects_live_stream(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=600.0, is_live=True),
    )

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Live streams are not supported"


def test_create_video_rejects_long_video(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_DURATION_S", "60")
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Video exceeds 1-minute limit"


def test_create_video_rejects_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: (_ for _ in ()).throw(AssertionError("metadata call should not run")),
    )

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Free video limit reached"


def test_get_video_requires_owner_scope(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    calls: list[tuple[str, str | None]] = []

    def fake_get_video(video_id: str, user_id: str | None = None) -> VideoRecord | None:
        calls.append((video_id, user_id))
        return _video_record(video_id, status="processing")

    monkeypatch.setattr("src.api.app.db_get_video", fake_get_video)

    response = client.get("/videos/video_status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "video_status"
    assert payload["status"] == "processing"
    assert payload["source_type"] == "youtube"
    assert payload["source_url"] is None
    assert calls == [("video_status", "user_123")]


def test_get_video_returns_404_for_non_owner(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: None,
    )

    response = client.get("/videos/video_status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found"


def test_get_video_requires_authentication() -> None:
    client = TestClient(app)
    response = client.get("/videos/video_status")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_list_my_videos_requires_authentication() -> None:
    client = TestClient(app)
    response = client.get("/users/me/videos")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_list_my_videos_scopes_to_user(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    calls: list[str | None] = []

    def fake_list_videos(user_id: str | None = None) -> list[VideoRecord]:
        calls.append(user_id)
        return [
            _video_record("video_1", status="ready"),
            _upload_video_record("video_2", status="ready"),
        ]

    monkeypatch.setattr("src.api.app.db_list_videos", fake_list_videos)
    monkeypatch.setattr(
        "src.api.app._source_url_for_record",
        lambda record: "https://example.com/source.mp4"
        if record.source_type == "upload"
        else None,
    )

    response = client.get("/users/me/videos")

    assert response.status_code == 200
    payload = response.json()
    assert calls == ["user_123"]
    assert [item["id"] for item in payload] == ["video_1", "video_2"]
    assert payload[0]["source_url"] is None
    assert payload[1]["source_url"] == "https://example.com/source.mp4"


def test_get_video_includes_source_url_for_uploaded_video(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app._source_url_for_record",
        lambda record: "https://example.com/source.mp4?token=abc",
    )

    response = client.get("/videos/video_upload")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_type"] == "upload"
    assert payload["source_url"] == "https://example.com/source.mp4?token=abc"


def test_search_video_accepts_nullable_thumbnail_url(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_service",
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
        "/videos/video_ready/search",
        json={"query_text": "robot in blue hoodie"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["youtube_url"] == "https://www.youtube.com/watch?v=abc123xyz45"
    assert payload["source_url"] is None
    assert payload["results"][0]["thumbnail_url"] is None
    assert payload["results"][0]["timestamp_s"] == 12.5


def test_upload_video_requires_authentication(monkeypatch) -> None:
    client = TestClient(app)
    called = False

    def fake_upload(*args, **kwargs):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr("src.api.app.R2Store.upload_source_video", fake_upload)

    response = client.post(
        "/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert called is False


def test_init_upload_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_init_upload_returns_503_when_r2_missing(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(StorageConfigError("missing")),
    )

    response = client.post(
        "/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Upload storage is not configured"


def test_init_upload_rejects_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("R2 config should not load")),
    )

    response = client.post(
        "/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Free video limit reached"


def test_init_upload_returns_presigned_url(monkeypatch) -> None:
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
        "/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["key"].startswith("source/")
    assert payload["key"].endswith("/upload.mp4")
    assert payload["upload_url"].endswith(payload["key"])
    assert payload["expires_in"] == 900


def test_upload_video_returns_503_when_r2_missing(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(StorageConfigError("missing")),
    )

    response = client.post(
        "/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Upload storage is not configured"


def test_upload_video_rejects_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("R2 config should not load")),
    )

    response = client.post(
        "/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Free video limit reached"


def test_complete_upload_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_complete_upload_rejects_non_uuid_video_id() -> None:
    client = TestClient(app)
    _authenticate("user_123")

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": "video_123", "filename": "upload.mp4"},
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Value error, video_id must be a valid UUID"


def test_complete_upload_returns_400_when_source_missing(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return False

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded source not found"


def test_complete_upload_rejects_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("R2 config should not load")),
    )

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Free video limit reached"


def test_complete_upload_enqueues_job(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    enqueue_calls: list[str] = []
    create_calls: list[tuple[str, str, str | None, str]] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return True

    def fake_create_uploaded_video(
        video_id: str,
        source_r2_key: str,
        source_filename: str | None = None,
        user_id: str | None = None,
        status: str = "queued",
    ) -> VideoRecord:
        create_calls.append((video_id, source_r2_key, source_filename, status))
        return VideoRecord(
            id=video_id,
            youtube_url=None,
            status=status,  # type: ignore[arg-type]
            user_id=user_id,
            error_message=None,
            source_type="upload",
            source_r2_key=source_r2_key,
            source_filename=source_filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr("src.api.app.db_create_uploaded_video", fake_create_uploaded_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"
    assert payload["source_filename"] == "upload.mp4"
    assert enqueue_calls == [payload["id"]]
    assert create_calls


def test_complete_upload_is_idempotent_for_matching_retry(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    existing = VideoRecord(
        id=UPLOAD_VIDEO_ID,
        youtube_url=None,
        status="queued",  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="upload",
        source_r2_key=f"source/{UPLOAD_VIDEO_ID}/upload.mp4",
        source_filename="upload.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, _key: str) -> bool:
            raise AssertionError("source_exists should not run for matching retry")

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: existing)
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run for matching retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(
            AssertionError("enqueue should not run for matching retry")
        ),
    )

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"


def test_complete_upload_returns_409_for_mismatched_existing_upload(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    existing = VideoRecord(
        id=UPLOAD_VIDEO_ID,
        youtube_url=None,
        status="queued",  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="upload",
        source_r2_key=f"source/{UPLOAD_VIDEO_ID}/other.mp4",
        source_filename="other.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, _key: str) -> bool:
            raise AssertionError("source_exists should not run for mismatched retry")

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: existing)
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run for mismatched retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(
            AssertionError("enqueue should not run for mismatched retry")
        ),
    )

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "video_id already exists with different upload metadata"


def test_complete_upload_handles_create_race_as_idempotent_retry(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    existing = VideoRecord(
        id=UPLOAD_VIDEO_ID,
        youtube_url=None,
        status="queued",  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="upload",
        source_r2_key=f"source/{UPLOAD_VIDEO_ID}/upload.mp4",
        source_filename="upload.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    get_calls = {"count": 0}

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

    def fake_get_video(video_id: str, user_id: str | None = None) -> VideoRecord | None:
        _ = video_id, user_id
        get_calls["count"] += 1
        if get_calls["count"] == 1:
            return None
        return existing

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", fake_get_video)
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("duplicate key value violates unique constraint videos_pkey")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(
            AssertionError("enqueue should not run when create raced")
        ),
    )

    response = client.post(
        "/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "queued"
    assert get_calls["count"] == 2


def test_upload_video_enqueues_job(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    enqueue_calls: list[str] = []
    create_calls: list[tuple[str, str, str | None, str]] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def upload_source_video(self, *args, **kwargs):
            class Result:
                key = "source/video_123/upload.mp4"

            return Result()

    def fake_create_uploaded_video(
        video_id: str,
        source_r2_key: str,
        source_filename: str | None = None,
        user_id: str | None = None,
        status: str = "queued",
    ) -> VideoRecord:
        create_calls.append((video_id, source_r2_key, source_filename, status))
        return VideoRecord(
            id=video_id,
            youtube_url=None,
            status=status,  # type: ignore[arg-type]
            user_id=user_id,
            error_message=None,
            source_type="upload",
            source_r2_key=source_r2_key,
            source_filename=source_filename,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_create_uploaded_video", fake_create_uploaded_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)

    response = client.post(
        "/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"
    assert payload["source_filename"] == "upload.mp4"
    assert enqueue_calls == [payload["id"]]
    assert create_calls


def test_search_video_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_search_video_returns_404_when_video_not_owned(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: None,
    )

    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found"


def test_search_video_returns_503_when_backend_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )

    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Search is temporarily unavailable. Please try again."


def test_search_video_rejects_blank_query_text(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide query_text or query_image_url"


@pytest.mark.parametrize(
    ("url", "expected_video_id"),
    [
        ("https://www.youtube.com/watch?v=abc123xyz45", "abc123xyz45"),
        ("https://youtu.be/abc123xyz45", "abc123xyz45"),
        ("https://www.youtube.com/shorts/abc123xyz45", "abc123xyz45"),
        ("https://www.youtube.com/live/abc123xyz45", "abc123xyz45"),
    ],
)
def test_extract_youtube_video_id_supported_formats(
    url: str,
    expected_video_id: str,
) -> None:
    assert extract_youtube_video_id(url) == expected_video_id


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=too_short",
        "https://www.youtube.com/channel/UC12345",
        "https://www.youtube.com/playlist?list=abc123xyz45",
        "https://vimeo.com/123456",
    ],
)
def test_extract_youtube_video_id_rejects_invalid_urls(url: str) -> None:
    assert extract_youtube_video_id(url) is None


def test_allowed_cors_origins_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://app.example.com, https://staging.example.com ",
    )

    assert _allowed_cors_origins() == [
        "https://app.example.com",
        "https://staging.example.com",
    ]


def test_allowed_cors_origins_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert _allowed_cors_origins() == ["http://localhost:3000"]
