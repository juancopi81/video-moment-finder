from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import hmac
import json
import re

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from src.api.app import (
    UploadDurationLimitExceededError,
    UploadDurationProbeUnavailableError,
    _is_youtube_bot_challenge_error,
    _allowed_cors_origin_regex,
    _allowed_cors_origins,
    _video_record_to_response,
    app,
    report_unhandled_exceptions,
)
from src.api.auth import get_current_user_id, get_optional_user_id
from src.billing.lemonsqueezy import LemonSqueezyProviderError
from src.db.supabase import CreditRecord, ProcessingCreditConsumeResult, VideoRecord
from src.storage.config import StorageConfigError
from src.storage.qdrant import SearchResult
from src.video.download import VideoMetadata, VideoMetadataError
from src.video.youtube import extract_youtube_video_id

from tests.api.conftest import (
    UPLOAD_VIDEO_ID,
    _authenticate,
    _upload_video_record,
    _video_record,
)

QUERY_IMAGE_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDAT\x08\xd7c\xf8\xcf"
    b"\xc0\x00\x00\x03\x01\x01\x00\xc9\xfe\x92\xef\x00\x00\x00\x00IEND\xaeB`\x82"
)
OVERSIZED_QUERY_IMAGE_BYTES = b"x" * ((10 * 1024 * 1024) + 1)


def _credit_record(balance: int) -> CreditRecord:
    return CreditRecord(
        id="credit_123",
        user_id="user_123",
        balance=balance,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_create_video_requires_authentication(monkeypatch) -> None:
    client = TestClient(app)
    called = False

    def fake_insert(*args, **kwargs) -> tuple[VideoRecord, bool]:
        nonlocal called
        called = True
        return _video_record("video_unauth"), True

    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", fake_insert)

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert called is False


def test_create_video_enqueues_job_for_authenticated_owner(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    enqueue_calls: list[str] = []
    insert_calls: list[tuple[str, str | None]] = []

    def fake_insert(youtube_url: str, user_id=None) -> tuple[VideoRecord, bool]:
        insert_calls.append((youtube_url, user_id))
        return _video_record("video_123", status="queued"), True

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", fake_insert)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://youtu.be/abc123xyz45"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "video_123"
    assert payload["status"] == "queued"
    assert payload["youtube_url"] == "https://www.youtube.com/watch?v=abc123xyz45"
    assert payload["source_type"] == "youtube"
    assert enqueue_calls == ["video_123"]
    assert insert_calls == [
        ("https://www.youtube.com/watch?v=abc123xyz45", "user_123")
    ]


def test_create_video_returns_500_when_enqueue_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    failure_updates: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (_video_record("video_500", status="queued"), True),
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
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to enqueue processing job"
    assert failure_updates == []


def test_create_video_retry_returns_existing_without_double_billing(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    existing = _video_record("video_retry", status="queued")

    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (existing, False),
    )
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: object())
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("credit consume should not run for matching retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(
            AssertionError("enqueue should not run when job history exists")
        ),
    )

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "video_retry"


def test_create_video_retry_reenqueues_stranded_video(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    stranded = _video_record("video_stranded", status="queued")
    enqueue_calls: list[str] = []

    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (stranded, False),
    )
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: None)
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: enqueue_calls.append(video_id) or object(),
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("credit consume should not run for queued retry")
        ),
    )

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "video_stranded"
    assert enqueue_calls == ["video_stranded"]


def test_create_video_retry_returns_503_when_reenqueue_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    stranded = _video_record("video_stranded_fail", status="queued")

    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (stranded, False),
    )
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: None)
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(RuntimeError("queue down")),
    )

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Video was created but processing could not be started. Please retry."
    )


def test_create_video_rejects_non_video_youtube_url() -> None:
    client = TestClient(app)
    _authenticate("user_123")

    response = client.post(
        "/api/v1/videos",
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
        "/api/v1/videos",
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
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Video exceeds 1-minute limit"


def test_create_video_returns_actionable_503_for_youtube_bot_challenge(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    def _raise(_youtube_url: str) -> VideoMetadata:
        raise VideoMetadataError(
            "ERROR: [youtube] abc123xyz45: Sign in to confirm you're not a bot. "
            "Use --cookies-from-browser or --cookies for the authentication."
        )

    monkeypatch.setattr("src.api.app.fetch_video_metadata", _raise)

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == {
        "code": "youtube_server_blocked",
        "message": (
            "Upload a video file instead. If this is your own YouTube video, "
            "download it from YouTube Studio or Google Takeout, then upload it here."
        ),
    }
    assert "retry-after" not in response.headers


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "ERROR: [youtube] abc123xyz45: Sign in to confirm you're not a bot.",
            True,
        ),
        (
            "Use --cookies-from-browser for the authentication.",
            True,
        ),
        (
            "ERROR: [youtube] abc123xyz45: HTTP Error 429: Too Many Requests",
            True,
        ),
        (
            "Too many requests. Please try again later.",
            True,
        ),
        (
            "Use --cookies for the authentication.",
            False,
        ),
        (
            "yt-dlp returned invalid JSON metadata",
            False,
        ),
    ],
)
def test_is_youtube_bot_challenge_error(message: str, expected: bool) -> None:
    assert _is_youtube_bot_challenge_error(message) is expected


def test_create_video_keeps_generic_metadata_fetch_failures_as_400(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    def _raise(_youtube_url: str) -> VideoMetadata:
        raise VideoMetadataError("yt-dlp returned invalid JSON metadata")

    monkeypatch.setattr("src.api.app.fetch_video_metadata", _raise)

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Unable to fetch YouTube metadata for this URL"


def test_create_video_rejects_when_no_paid_credits_after_free_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: None)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    status_updates: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda *args, **kwargs: (_video_record("video_credit_denied", status="queued"), True),
    )
    monkeypatch.setattr(
        "src.api.app.update_video_status",
        lambda video_id, status, error_message=None: status_updates.append(
            (video_id, status, error_message)
        ),
    )

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 402
    assert response.json()["detail"] == "Insufficient credits. Buy credits to process another video."
    assert status_updates == [("video_credit_denied", "failed", "Insufficient credits")]


def test_create_video_consumes_paid_credit_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: ProcessingCreditConsumeResult(
            allowed=True,
            remaining_balance=2,
        ),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (_video_record("video_paid", status="queued"), True),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _video_id: object())

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 200
    assert response.json()["id"] == "video_paid"


def test_create_video_returns_429_when_write_rate_limit_exceeded(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_USER_WRITE_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (_video_record("video_rate_limited", status="queued"), True),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _video_id: object())

    first = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )
    second = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Please retry later."
    assert second.headers.get("retry-after") is not None


def test_create_video_rate_limit_is_isolated_per_user(monkeypatch) -> None:
    client = TestClient(app)
    current_user = {"id": "user_a"}
    app.dependency_overrides[get_current_user_id] = lambda: current_user["id"]
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_USER_WRITE_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    create_calls: list[str] = []

    def _fake_insert(youtube_url: str, user_id=None) -> tuple[VideoRecord, bool]:
        _ = youtube_url
        create_calls.append(user_id or "")
        return _video_record(f"video_{user_id or 'unknown'}", status="queued"), True

    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", _fake_insert)
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _video_id: object())

    first = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )
    second = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )
    current_user["id"] = "user_b"
    third = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert third.status_code == 200
    assert create_calls == ["user_a", "user_b"]


def test_create_video_allows_unlimited_user_override(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: True)
    monkeypatch.setattr(
        "src.api.app.db_count_videos_for_user",
        lambda _user_id: (_ for _ in ()).throw(AssertionError("count should not run")),
    )
    monkeypatch.setattr(
        "src.api.app.fetch_video_metadata",
        lambda _: VideoMetadata(duration_s=120.0, is_live=False),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_youtube_video_idempotent",
        lambda youtube_url, user_id=None: (_video_record("video_123", status="queued"), True),
    )
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _video_id: object())

    response = client.post(
        "/api/v1/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 200


def test_get_video_requires_owner_scope(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    calls: list[tuple[str, str | None]] = []

    def fake_get_video(video_id: str, user_id: str | None = None) -> VideoRecord | None:
        calls.append((video_id, user_id))
        return _video_record(video_id, status="processing")

    monkeypatch.setattr("src.api.app.db_get_video", fake_get_video)

    response = client.get("/api/v1/videos/video_status")

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

    response = client.get("/api/v1/videos/video_status")

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found"


def test_get_video_requires_authentication() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/videos/video_status")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_list_my_videos_requires_authentication() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/videos")

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

    response = client.get("/api/v1/videos")

    assert response.status_code == 200
    payload = response.json()
    assert calls == ["user_123"]
    assert [item["id"] for item in payload] == ["video_1", "video_2"]
    assert payload[0]["source_url"] is None
    assert payload[1]["source_url"] == "https://example.com/source.mp4"


def test_billing_summary_requires_authentication() -> None:
    client = TestClient(app)
    response = client.get("/api/v1/billing/credits/summary")

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_billing_summary_returns_usage_and_balance(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: _credit_record(7))
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 3)
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: False)
    monkeypatch.setattr("src.api.app._max_free_videos", lambda: 5)

    response = client.get("/api/v1/billing/credits/summary")

    assert response.status_code == 200
    assert response.json() == {
        "credits_balance": 7,
        "free_videos_limit": 5,
        "free_videos_used": 3,
        "free_videos_remaining": 2,
        "has_unlimited_access": False,
    }


def test_billing_summary_clamps_non_positive_remaining(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: None)
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 6)
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: True)
    monkeypatch.setattr("src.api.app._max_free_videos", lambda: 2)

    response = client.get("/api/v1/billing/credits/summary")

    assert response.status_code == 200
    assert response.json() == {
        "credits_balance": 0,
        "free_videos_limit": 2,
        "free_videos_used": 6,
        "free_videos_remaining": 0,
        "has_unlimited_access": True,
    }


def test_billing_summary_clamps_negative_credit_balance(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: _credit_record(-3))
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: False)
    monkeypatch.setattr("src.api.app._max_free_videos", lambda: 5)

    response = client.get("/api/v1/billing/credits/summary")

    assert response.status_code == 200
    assert response.json() == {
        "credits_balance": 0,
        "free_videos_limit": 5,
        "free_videos_used": 1,
        "free_videos_remaining": 4,
        "has_unlimited_access": False,
    }


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

    response = client.get("/api/v1/videos/video_upload")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_type"] == "upload"
    assert payload["source_url"] == "https://example.com/source.mp4?token=abc"


def test_get_video_returns_null_source_url_when_source_expired(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )

    presign_called = False

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, _key: str) -> bool:
            return False

        def generate_presigned_url(self, *_args, **_kwargs) -> str:
            nonlocal presign_called
            presign_called = True
            return "https://example.com/should-not-be-used"

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)

    response = client.get("/api/v1/videos/video_upload")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source_url"] is None
    assert not presign_called, "generate_presigned_url should not be called for expired source"


def test_search_video_accepts_nullable_thumbnail_url(monkeypatch) -> None:
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
        "/api/v1/videos/video_ready/search",
        json={"query_text": "robot in blue hoodie"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["youtube_url"] == "https://www.youtube.com/watch?v=abc123xyz45"
    assert payload["source_url"] is None
    assert payload["results"][0]["thumbnail_url"] is None
    assert payload["results"][0]["timestamp_s"] == 12.5
    assert payload["results"][0]["source"] == "visual"
    assert payload["results"][0]["transcript_text"] is None


def test_search_video_includes_transcript_result_metadata(monkeypatch) -> None:
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
                frame_index=-1,
                timestamp_s=21.0,
                thumbnail_url=None,
                score=0.81,
                source="transcript",
                transcript_text="The host explains the launch plan.",
            ),
            SearchResult(
                video_id=video_id,
                frame_index=4,
                timestamp_s=25.0,
                thumbnail_url="https://cdn.example.com/thumb.jpg",
                score=0.7,
            ),
        ],
    )

    response = client.post(
        "/api/v1/videos/video_ready/search",
        json={"query_text": "where does he explain the launch plan"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["source"] == "transcript"
    assert payload["results"][0]["transcript_text"] == "The host explains the launch plan."
    assert payload["results"][1]["source"] == "visual"
    assert payload["results"][1]["transcript_text"] is None


def test_search_uploaded_video_can_return_transcript_matches(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _upload_video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_by_text_service",
        lambda video_id, query_text, limit=5: [
            SearchResult(
                video_id=video_id,
                frame_index=-1,
                timestamp_s=18.5,
                thumbnail_url=None,
                score=0.9,
                source="transcript",
                transcript_text="Here the instructor explains tritone substitution.",
            )
        ],
    )

    response = client.post(
        f"/api/v1/videos/{UPLOAD_VIDEO_ID}/search",
        json={"query_text": "where does the instructor explain tritone substitution"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["video_id"] == UPLOAD_VIDEO_ID
    assert payload["youtube_url"] is None
    assert payload["source_url"] is None
    assert payload["results"] == [
        {
            "timestamp_s": 18.5,
            "thumbnail_url": None,
            "score": 0.9,
            "source": "transcript",
            "transcript_text": "Here the instructor explains tritone substitution.",
        }
    ]


def test_search_video_limit_is_per_result_source(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    def _fake_search_video_service(*, video_id: str, query_text: str, limit: int):
        captured["limit"] = limit
        return [
            SearchResult(
                video_id=video_id,
                frame_index=2,
                timestamp_s=8.0,
                thumbnail_url="https://cdn.example.com/visual-1.jpg",
                score=0.93,
            ),
            SearchResult(
                video_id=video_id,
                frame_index=3,
                timestamp_s=9.0,
                thumbnail_url="https://cdn.example.com/visual-2.jpg",
                score=0.91,
            ),
            SearchResult(
                video_id=video_id,
                frame_index=-1,
                timestamp_s=14.0,
                thumbnail_url=None,
                score=0.87,
                source="transcript",
                transcript_text="spoken match one",
            ),
            SearchResult(
                video_id=video_id,
                frame_index=-1,
                timestamp_s=18.0,
                thumbnail_url=None,
                score=0.8,
                source="transcript",
                transcript_text="spoken match two",
            ),
        ]

    monkeypatch.setattr(
        "src.api.app.search_video_by_text_service",
        _fake_search_video_service,
    )

    response = client.post(
        "/api/v1/videos/video_ready/search",
        json={"query_text": "where is the cat", "limit": 2},
    )

    assert response.status_code == 200
    assert captured["limit"] == 2
    payload = response.json()
    assert len(payload["results"]) == 4
    assert [item["source"] for item in payload["results"]] == [
        "visual",
        "visual",
        "transcript",
        "transcript",
    ]


def test_upload_video_requires_authentication(monkeypatch) -> None:
    client = TestClient(app)
    called = False

    def fake_upload(*args, **kwargs):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr("src.api.app.R2Store.upload_source_video", fake_upload)

    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"
    assert called is False


def test_init_upload_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/upload/init",
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
        "/api/v1/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Upload storage is not configured"


def test_init_upload_rejects_when_no_paid_credits_after_free_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("R2 config should not load")),
    )

    response = client.post(
        "/api/v1/videos/upload/init",
        json={"filename": "upload.mp4", "content_type": "video/mp4"},
    )

    assert response.status_code == 402
    assert response.json()["detail"] == "Insufficient credits. Buy credits to process another video."


def test_init_upload_allows_paid_user_without_consuming(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: _credit_record(3))
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("init precheck must not consume credits")
        ),
    )

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
        "/api/v1/videos/upload/init",
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
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Upload storage is not configured"


def test_upload_video_rejects_when_no_paid_credits_after_free_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)

    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(AssertionError("R2 config should not load")),
    )
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run when credits are unavailable")
        ),
    )

    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 402
    assert response.json()["detail"] == "Insufficient credits. Buy credits to process another video."


def test_upload_video_cleans_up_source_on_post_upload_credit_denial(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: _credit_record(1))
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: ProcessingCreditConsumeResult(
            allowed=False,
            remaining_balance=0,
        ),
    )
    cleanup_calls: list[str] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def upload_source_video(self, *args, **kwargs):
            class Result:
                key = "source/video_123/upload.mp4"

            return Result()

        def delete_source_object(self, key: str) -> None:
            cleanup_calls.append(key)

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run when credit consume denies")
        ),
    )

    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 402
    assert cleanup_calls == ["source/video_123/upload.mp4"]


def test_upload_video_rejects_when_uploaded_duration_exceeds_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    consume_calls: list[str] = []
    create_called = {"value": False}
    enqueue_calls: list[str] = []

    def _raise_duration(_file) -> None:
        raise UploadDurationLimitExceededError("Video exceeds 30-minute limit")

    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("R2 config should not load when duration validation fails")
        ),
    )
    monkeypatch.setattr(
        "src.api.app._validate_upload_file_duration_or_raise",
        _raise_duration,
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda user_id: consume_calls.append(user_id)
        or ProcessingCreditConsumeResult(allowed=True, remaining_balance=0),
    )
    monkeypatch.setattr(
        "src.api.app.db_create_uploaded_video",
        lambda *args, **kwargs: create_called.update(value=True),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: enqueue_calls.append(video_id),
    )

    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Video exceeds 30-minute limit"
    assert consume_calls == []
    assert create_called["value"] is False
    assert enqueue_calls == []


def test_upload_video_returns_503_when_duration_probe_unavailable(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    def _raise_probe(_file) -> None:
        raise UploadDurationProbeUnavailableError("ffprobe unavailable")

    monkeypatch.setattr(
        "src.api.app.R2Config.from_env",
        lambda: (_ for _ in ()).throw(
            AssertionError("R2 config should not load when duration probe is unavailable")
        ),
    )
    monkeypatch.setattr(
        "src.api.app._validate_upload_file_duration_or_raise",
        _raise_probe,
    )

    response = client.post(
        "/api/v1/videos/upload",
        files={"file": ("upload.mp4", b"data", "video/mp4")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to verify upload"


def test_complete_upload_requires_authentication() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_complete_upload_rejects_non_uuid_video_id() -> None:
    client = TestClient(app)
    _authenticate("user_123")

    response = client.post(
        "/api/v1/videos/upload/complete",
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
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded source not found"


def test_complete_upload_rejects_when_no_paid_credits_after_free_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: None)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, _key: str) -> bool:
            return True

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr(
        "src.api.app._validate_uploaded_source_duration_with_cleanup",
        lambda store, key, user_id: None,
    )
    status_updates: list[tuple[str, str, str | None]] = []
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda *args, **kwargs: (_upload_video_record(UPLOAD_VIDEO_ID), True),
    )
    monkeypatch.setattr(
        "src.api.app.update_video_status",
        lambda video_id, status, error_message=None: status_updates.append(
            (video_id, status, error_message)
        ),
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 402
    assert response.json()["detail"] == "Insufficient credits. Buy credits to process another video."
    assert status_updates == [(UPLOAD_VIDEO_ID, "failed", "Insufficient credits")]


def test_complete_upload_rejects_when_uploaded_duration_exceeds_limit(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    consume_calls: list[str] = []
    create_called = {"value": False}
    enqueue_calls: list[str] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

        def delete_source_object(self, key: str) -> None:
            return None

    def _raise_duration(*_args, **_kwargs) -> None:
        raise UploadDurationLimitExceededError("Video exceeds 30-minute limit")

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr(
        "src.api.app._validate_uploaded_source_duration_with_cleanup",
        _raise_duration,
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda user_id: consume_calls.append(user_id)
        or ProcessingCreditConsumeResult(allowed=True, remaining_balance=0),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda *args, **kwargs: create_called.update(value=True),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: enqueue_calls.append(video_id),
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Video exceeds 30-minute limit"
    assert consume_calls == []
    assert create_called["value"] is False
    assert enqueue_calls == []


def test_complete_upload_returns_503_when_duration_probe_unavailable(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

    def _raise_probe(*_args, **_kwargs) -> None:
        raise UploadDurationProbeUnavailableError("ffprobe unavailable")

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr(
        "src.api.app._validate_uploaded_source_duration_with_cleanup",
        _raise_probe,
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Failed to verify upload"


def test_complete_upload_enqueues_job(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    enqueue_calls: list[str] = []
    insert_calls: list[tuple[str, str, str, str]] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return True

    def fake_insert_uploaded_video(
        video_id: str,
        user_id: str,
        source_r2_key: str,
        source_filename: str,
    ) -> tuple[VideoRecord, bool]:
        insert_calls.append((video_id, user_id, source_r2_key, source_filename))
        return VideoRecord(
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
        ), True

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr("src.api.app.db_insert_uploaded_video_idempotent", fake_insert_uploaded_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"
    assert payload["source_filename"] == "upload.mp4"
    assert enqueue_calls == [payload["id"]]
    assert insert_calls


def test_complete_upload_enqueues_job_with_paid_credit_when_free_limit_reached(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 1)

    enqueue_calls: list[str] = []
    insert_calls: list[tuple[str, str, str, str]] = []
    consume_calls: list[str] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

    def fake_consume(user_id: str) -> ProcessingCreditConsumeResult:
        consume_calls.append(user_id)
        return ProcessingCreditConsumeResult(
            allowed=True,
            remaining_balance=0,
        )

    def fake_insert_uploaded_video(
        video_id: str,
        user_id: str,
        source_r2_key: str,
        source_filename: str,
    ) -> tuple[VideoRecord, bool]:
        insert_calls.append((video_id, user_id, source_r2_key, source_filename))
        return VideoRecord(
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
        ), True

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: _credit_record(1))
    monkeypatch.setattr("src.api.app.db_consume_processing_credit", fake_consume)
    monkeypatch.setattr("src.api.app.db_insert_uploaded_video_idempotent", fake_insert_uploaded_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "queued"
    assert consume_calls == ["user_123"]
    assert enqueue_calls == [payload["id"]]
    assert insert_calls


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
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: object())
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("credit consume should not run for matching retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
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
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "queued"
    assert payload["source_type"] == "upload"


def test_complete_upload_reenqueues_matching_retry_when_job_missing(monkeypatch) -> None:
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
    enqueue_calls: list[str] = []

    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: existing)
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: None)
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run for matching retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: enqueue_calls.append(video_id) or object(),
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    assert enqueue_calls == [UPLOAD_VIDEO_ID]


def test_complete_upload_recovers_failed_retry_without_job_history(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")

    existing = VideoRecord(
        id=UPLOAD_VIDEO_ID,
        youtube_url=None,
        status="failed",  # type: ignore[arg-type]
        user_id="user_123",
        error_message="Failed to enqueue processing job",
        source_type="upload",
        source_r2_key=f"source/{UPLOAD_VIDEO_ID}/upload.mp4",
        source_filename="upload.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    enqueue_calls: list[str] = []
    status_updates: list[tuple[str, str]] = []

    class FakeR2Store:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

    def fake_update(video_id: str, status: str, error_message: str | None = None):
        status_updates.append((video_id, status))
        existing.status = status  # type: ignore[assignment]
        existing.error_message = error_message
        return existing

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: existing)
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: None)
    monkeypatch.setattr("src.api.app.update_video_status", fake_update)
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("create should not run for recoverable retry")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: enqueue_calls.append(video_id) or object(),
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    assert status_updates == [(UPLOAD_VIDEO_ID, "queued")]
    assert enqueue_calls == [UPLOAD_VIDEO_ID]
    assert response.json()["status"] == "queued"


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
        "src.api.app.db_insert_uploaded_video_idempotent",
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
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "video_id already exists with different upload metadata"


def test_complete_upload_handles_insert_race_as_idempotent_retry(monkeypatch) -> None:
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

        def source_exists(self, key: str) -> bool:
            return key == f"source/{UPLOAD_VIDEO_ID}/upload.mp4"

    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _video_id: object())
    monkeypatch.setattr(
        "src.api.app.db_insert_uploaded_video_idempotent",
        lambda *args, **kwargs: (existing, False),
    )
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: (_ for _ in ()).throw(
            AssertionError("credit consume should not run when insert raced")
        ),
    )
    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda _video_id: (_ for _ in ()).throw(
            AssertionError("enqueue should not run when insert raced")
        ),
    )

    response = client.post(
        "/api/v1/videos/upload/complete",
        json={"video_id": UPLOAD_VIDEO_ID, "filename": "upload.mp4"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == UPLOAD_VIDEO_ID
    assert payload["status"] == "queued"


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
        "/api/v1/videos/upload",
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
        "/api/v1/videos/video_ready/search",
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
        "/api/v1/videos/video_ready/search",
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
        "src.api.app.search_video_by_text_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search",
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
        "/api/v1/videos/video_ready/search",
        json={"query_text": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide query_text"


def test_search_video_returns_429_when_rate_limit_exceeded(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    search_calls: list[dict] = []

    def _fake_search_video_service(*, video_id: str, query_text: str, limit: int):
        search_calls.append(
            {
                "video_id": video_id,
                "query_text": query_text,
                "limit": limit,
            }
        )
        return [
            SearchResult(
                video_id=video_id,
                frame_index=0,
                timestamp_s=12.0,
                thumbnail_url=None,
                score=0.9,
            )
        ]

    monkeypatch.setattr("src.api.app.search_video_by_text_service", _fake_search_video_service)

    first = client.post(
        "/api/v1/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )
    second = client.post(
        "/api/v1/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Please retry later."
    assert second.headers.get("retry-after") is not None
    assert len(search_calls) == 1


def test_search_video_by_image_requires_authentication() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_search_video_by_image_returns_404_when_video_not_owned(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Video not found"


def test_search_video_by_image_returns_400_when_video_not_ready(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="processing"),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Video not ready for search (status: processing)"


def test_search_video_by_image_rejects_non_image_content_type(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.txt", b"abc", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only image uploads are supported"


def test_search_video_by_image_rejects_empty_upload(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded image is empty"


def test_search_video_by_image_rejects_large_upload(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_by_image_service",
        lambda **kwargs: pytest.fail("should not be called for oversized upload"),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", OVERSIZED_QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded image exceeds 10 MB limit"


def test_search_video_by_image_rejects_invalid_image(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", b"not-an-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image"


def test_search_video_by_image_returns_results(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    def _fake_search_video_by_image_service(
        *,
        video_id: str,
        query_image_bytes: bytes,
        limit: int,
    ):
        captured["video_id"] = video_id
        captured["query_image_bytes"] = query_image_bytes
        captured["limit"] = limit
        return [
            SearchResult(
                video_id=video_id,
                frame_index=0,
                timestamp_s=9.5,
                thumbnail_url="https://cdn.example.com/thumb.jpg",
                score=0.88,
            )
        ]

    monkeypatch.setattr(
        "src.api.app.search_video_by_image_service",
        _fake_search_video_by_image_service,
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        data={"limit": "3"},
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 200
    assert captured == {
        "video_id": "video_ready",
        "query_image_bytes": QUERY_IMAGE_BYTES,
        "limit": 3,
    }
    payload = response.json()
    assert payload["results"][0]["timestamp_s"] == 9.5
    assert payload["results"][0]["thumbnail_url"] == "https://cdn.example.com/thumb.jpg"
    assert payload["results"][0]["source"] == "visual"


def test_search_video_by_image_returns_429_when_rate_limit_exceeded(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )

    monkeypatch.setattr(
        "src.api.app.search_video_by_image_service",
        lambda **kwargs: [
            SearchResult(
                video_id=kwargs["video_id"],
                frame_index=0,
                timestamp_s=12.0,
                thumbnail_url=None,
                score=0.9,
            )
        ],
    )

    first = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )
    second = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Please retry later."


def test_search_video_by_image_returns_503_when_backend_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id, user_id=None: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_by_image_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )

    response = client.post(
        "/api/v1/videos/video_ready/search/image",
        files={"query_image": ("query.png", QUERY_IMAGE_BYTES, "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Search is temporarily unavailable. Please try again."


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


def test_billing_checkout_requires_authentication() -> None:
    client = TestClient(app)

    response = client.post("/api/v1/billing/credits/checkout", json={"plan": "starter"})

    assert response.status_code == 401
    assert response.headers.get("www-authenticate") == "Bearer"


def test_billing_checkout_rejects_invalid_plan() -> None:
    client = TestClient(app)
    _authenticate("user_123")

    response = client.post("/api/v1/billing/credits/checkout", json={"plan": "enterprise"})

    assert response.status_code == 422


def test_billing_checkout_returns_503_when_variant_config_missing(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.delenv("LEMON_SQUEEZY_VARIANT_ID_STARTER", raising=False)

    response = client.post("/api/v1/billing/credits/checkout", json={"plan": "starter"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Billing checkout is not configured"


def test_billing_checkout_creates_starter_session(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("LEMON_SQUEEZY_VARIANT_ID_STARTER", "var_starter_123")
    calls: list[dict] = []

    class _Session:
        url = "https://app.lemonsqueezy.com/checkout/buy/session_123"
        test_mode = True

    def fake_create_checkout_session(**kwargs):
        calls.append(kwargs)
        return _Session()

    monkeypatch.setattr("src.api.app.create_checkout_session", fake_create_checkout_session)

    response = client.post("/api/v1/billing/credits/checkout", json={"plan": "starter"})

    assert response.status_code == 200
    assert response.json() == {
        "provider": "lemonsqueezy",
        "plan": "starter",
        "credits": 5,
        "checkout_url": "https://app.lemonsqueezy.com/checkout/buy/session_123",
        "test_mode": True,
    }
    assert calls == [
        {
            "user_id": "user_123",
            "plan": "starter",
            "credits": 5,
            "variant_id": "var_starter_123",
        }
    ]


def test_billing_checkout_returns_502_when_provider_fails(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate("user_123")
    monkeypatch.setenv("LEMON_SQUEEZY_VARIANT_ID_PRO", "var_pro_123")

    def _raise_provider_error(**_kwargs):
        raise LemonSqueezyProviderError("upstream unavailable")

    monkeypatch.setattr("src.api.app.create_checkout_session", _raise_provider_error)

    response = client.post("/api/v1/billing/credits/checkout", json={"plan": "pro"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Billing checkout is temporarily unavailable"


def _lemonsqueezy_signature(secret: str, payload: dict) -> tuple[bytes, str]:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()
    return raw, signature


def test_lemonsqueezy_webhook_rejects_missing_signature(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")

    response = client.post("/webhooks/lemonsqueezy", content=b"{}")

    assert response.status_code == 401
    assert response.json()["detail"] == "Missing webhook signature"


def test_lemonsqueezy_webhook_rejects_invalid_signature(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    raw, _ = _lemonsqueezy_signature("secret_123", {"meta": {"event_name": "order_created"}})

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": "bad-signature"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid webhook signature"


def test_lemonsqueezy_webhook_ignores_untracked_events(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": {"event_name": "order_refunded"},
        "data": {"id": "evt_1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    assert response.json()["processed"] is False
    assert response.json()["granted"] is False


def test_lemonsqueezy_webhook_applies_credit_grant(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "user_123", "credits": 5},
        },
        "data": {"id": "order_1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)
    calls: list[dict] = []

    class _Result:
        applied = True

    def fake_apply(**kwargs):
        calls.append(kwargs)
        return _Result()

    monkeypatch.setattr("src.api.app.db_apply_billing_credit_grant", fake_apply)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "received": True,
        "processed": True,
        "granted": True,
        "reason": None,
    }
    assert calls
    assert calls[0]["provider"] == "lemonsqueezy"
    assert calls[0]["event_id"] == "order_created:order_1"
    assert calls[0]["user_id"] == "user_123"
    assert calls[0]["credits"] == 5


def test_lemonsqueezy_webhook_returns_duplicate_as_not_granted(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "user_123", "credits": 5},
        },
        "data": {"id": "order_1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    class _Result:
        applied = False

    monkeypatch.setattr("src.api.app.db_apply_billing_credit_grant", lambda **_kwargs: _Result())

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["processed"] is True
    assert body["granted"] is False
    assert body["reason"] == "Event already applied"


def test_lemonsqueezy_webhook_returns_429_when_ip_rate_limit_exceeded(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
    monkeypatch.setenv("RATE_LIMIT_WEBHOOK_REQUESTS_PER_WINDOW", "1")
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "user_123", "credits": 5},
        },
        "data": {"id": "order_123"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    class _Result:
        applied = True

    monkeypatch.setattr("src.api.app.db_apply_billing_credit_grant", lambda **_kwargs: _Result())

    first = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature, "x-forwarded-for": "198.51.100.10"},
    )
    second = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature, "x-forwarded-for": "198.51.100.10"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["detail"] == "Rate limit exceeded. Please retry later."
    assert second.headers.get("retry-after") is not None


def _lemonsqueezy_raw_signature(secret: str, raw: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), raw, hashlib.sha256).hexdigest()


def test_lemonsqueezy_webhook_rejects_non_object_json(monkeypatch) -> None:
    """Non-dict top-level JSON (e.g. []) returns 400."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    raw = b"[]"
    signature = _lemonsqueezy_raw_signature("secret_123", raw)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid webhook payload"


def test_lemonsqueezy_webhook_malformed_meta_with_top_level_event_name(
    monkeypatch,
) -> None:
    """Malformed meta with top-level event_name degrades to non-grant, not 400."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": "x",
        "event_name": "order_created",
        "data": {"id": "e1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["processed"] is False
    assert body["granted"] is False
    assert body["reason"] == "No credit grant metadata found in meta.custom_data"


def test_lemonsqueezy_webhook_rejects_bool_credits(monkeypatch) -> None:
    """Bool credits are rejected (int(str(True)) raises ValueError)."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    monkeypatch.setattr(
        "src.api.app.db_apply_billing_credit_grant",
        lambda **kw: pytest.fail("should not be called"),
    )
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "u1", "credits": True},
        },
        "data": {"id": "e1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["processed"] is False
    assert body["granted"] is False
    assert body["reason"] == "No credit grant metadata found in meta.custom_data"


def test_lemonsqueezy_webhook_rejects_float_string_credits(monkeypatch) -> None:
    """Float-string credits are rejected (int('5.0') raises ValueError)."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    monkeypatch.setattr(
        "src.api.app.db_apply_billing_credit_grant",
        lambda **kw: pytest.fail("should not be called"),
    )
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "u1", "credits": "5.0"},
        },
        "data": {"id": "e1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["processed"] is False
    assert body["granted"] is False


def test_lemonsqueezy_webhook_valid_custom_data_invalid_data_type(
    monkeypatch,
) -> None:
    """Valid meta.custom_data + invalid data type still grants with SHA event_id."""
    from unittest.mock import MagicMock

    from src.db.supabase import BillingCreditGrantResult

    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    mock_grant = MagicMock(return_value=BillingCreditGrantResult(applied=True))
    monkeypatch.setattr("src.api.app.db_apply_billing_credit_grant", mock_grant)
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "u1", "credits": 5},
        },
        "data": [],
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["processed"] is True
    assert body["granted"] is True
    call_kwargs = mock_grant.call_args.kwargs
    assert call_kwargs["event_id"].startswith("order_created:sha256:")


def test_lemonsqueezy_webhook_rejects_non_string_user_id(monkeypatch) -> None:
    """Non-string user_id (e.g. int) does not grant."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    monkeypatch.setattr(
        "src.api.app.db_apply_billing_credit_grant",
        lambda **kw: pytest.fail("should not be called"),
    )
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": 123, "credits": 5},
        },
        "data": {"id": "e1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["processed"] is False
    assert body["granted"] is False
    assert body["reason"] == "No credit grant metadata found in meta.custom_data"


def test_lemonsqueezy_webhook_valid_event_name_malformed_custom_data(
    monkeypatch,
) -> None:
    """Valid meta.event_name + malformed custom_data degrades to non-grant, not 400."""
    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": "bad",
        },
        "data": {"id": "e1"},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["received"] is True
    assert body["processed"] is False
    assert body["granted"] is False
    assert body["reason"] == "No credit grant metadata found in meta.custom_data"


def test_lemonsqueezy_webhook_bool_data_id_uses_sha_fallback(monkeypatch) -> None:
    """Bool data.id falls back to SHA event_id (intentional tightening)."""
    from unittest.mock import MagicMock

    from src.db.supabase import BillingCreditGrantResult

    client = TestClient(app)
    monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", "secret_123")
    mock_grant = MagicMock(return_value=BillingCreditGrantResult(applied=True))
    monkeypatch.setattr("src.api.app.db_apply_billing_credit_grant", mock_grant)
    payload = {
        "meta": {
            "event_name": "order_created",
            "custom_data": {"user_id": "u1", "credits": 5},
        },
        "data": {"id": True},
    }
    raw, signature = _lemonsqueezy_signature("secret_123", payload)

    response = client.post(
        "/webhooks/lemonsqueezy",
        content=raw,
        headers={"x-signature": signature},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["processed"] is True
    assert body["granted"] is True
    call_kwargs = mock_grant.call_args.kwargs
    assert call_kwargs["event_id"].startswith("order_created:sha256:")


def test_allowed_cors_origins_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://app.example.com, https://staging.example.com ",
    )

    assert _allowed_cors_origins() == [
        "https://app.example.com",
        "https://staging.example.com",
    ]


def test_allowed_cors_origins_normalize_trailing_slashes(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://app.example.com/, https://staging.example.com/path ",
    )

    assert _allowed_cors_origins() == [
        "https://app.example.com",
        "https://staging.example.com",
    ]


def test_allowed_cors_origin_regex_supports_wildcards(monkeypatch) -> None:
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGINS",
        "https://videomomentfinder.com, https://video-moment-finder-*.vercel.app",
    )

    regex = _allowed_cors_origin_regex()

    assert _allowed_cors_origins() == ["https://videomomentfinder.com"]
    assert regex is not None
    assert re.match(regex, "https://video-moment-finder-git-pr-24-juancopi81.vercel.app")
    assert not re.match(regex, "https://other-project-git-main-juancopi81.vercel.app")


def test_allowed_cors_origin_regex_supports_explicit_regex(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv(
        "CORS_ALLOWED_ORIGIN_REGEX",
        r"^https://preview-[a-z0-9-]+\.example\.com$",
    )

    regex = _allowed_cors_origin_regex()

    assert regex is not None
    assert re.match(regex, "https://preview-pr-24.example.com")
    assert not re.match(regex, "https://api.example.com")


def test_allowed_cors_origins_defaults_to_localhost(monkeypatch) -> None:
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert _allowed_cors_origins() == ["http://localhost:3000"]


def test_video_record_to_response_with_malformed_created_at(caplog) -> None:
    record = VideoRecord(
        id="vid_malformed",
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status="ready",
        user_id="user_123",
        error_message=None,
        source_type="youtube",
        source_r2_key=None,
        source_filename=None,
        created_at="not-a-date",
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    import logging

    dt_logger = logging.getLogger("src.utils.datetime")
    dt_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="src.utils.datetime"):
            response = _video_record_to_response(record)
        assert isinstance(response.created_at, datetime)
        assert "Unparseable ISO datetime" in caplog.text
    finally:
        dt_logger.propagate = False


def test_report_unhandled_exceptions_uses_to_thread(monkeypatch) -> None:
    to_thread_calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    capture_calls: list[tuple[BaseException, dict[str, object] | None]] = []

    async def fake_to_thread(func, *args, **kwargs):
        to_thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    def fake_capture_exception(exc: BaseException, *, context=None) -> None:
        capture_calls.append((exc, context))

    async def fake_call_next(_request: Request):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.api.app.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr("src.api.app.capture_exception", fake_capture_exception)

    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/videos",
            "headers": [],
            "query_string": b"",
        }
    )

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(report_unhandled_exceptions(request, fake_call_next))

    assert len(to_thread_calls) == 1
    assert len(capture_calls) == 1
    assert capture_calls[0][1] == {"path": "/api/v1/videos", "method": "POST"}


# ---------------------------------------------------------------------------
# Analytics endpoint tests
# ---------------------------------------------------------------------------


def _authenticate_optional(user_id: str = "user_123") -> None:
    app.dependency_overrides[get_optional_user_id] = lambda: user_id


def _authenticate_optional_anonymous() -> None:
    app.dependency_overrides[get_optional_user_id] = lambda: None


def test_analytics_event_signup_complete_requires_auth() -> None:
    client = TestClient(app)
    _authenticate_optional_anonymous()

    response = client.post("/analytics/event", json={"event_name": "signup_complete"})

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required for signup_complete"


def test_analytics_event_signup_complete_with_auth(monkeypatch) -> None:
    client = TestClient(app)
    _authenticate_optional("user_123")
    track_calls: list[dict] = []
    monkeypatch.setattr(
        "src.api.app.track",
        lambda event_name, **kwargs: track_calls.append({"event_name": event_name, **kwargs}),
    )

    response = client.post("/analytics/event", json={"event_name": "signup_complete"})

    assert response.status_code == 204
    assert track_calls == [{"event_name": "signup_complete", "user_id": "user_123", "metadata": None}]


def test_analytics_event_rejects_disallowed_event_name() -> None:
    client = TestClient(app)
    _authenticate_optional_anonymous()

    response = client.post("/analytics/event", json={"event_name": "video_submitted"})

    assert response.status_code == 422


def test_analytics_event_rejects_unknown_event_name() -> None:
    client = TestClient(app)
    _authenticate_optional_anonymous()

    response = client.post("/analytics/event", json={"event_name": "unknown"})

    assert response.status_code == 422
