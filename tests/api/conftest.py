"""Shared fixtures and helpers for API tests."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest

from src.api.app import app
from src.api.auth import AuthIdentity, get_current_user, get_current_user_id
from src.api.rate_limit import SlidingWindowRateLimiter
from src.db.supabase import ApiKeyRecord, ProcessingCreditConsumeResult, VideoRecord

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


def _authenticate(user_id: str = "user_123") -> None:
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_current_user] = lambda: AuthIdentity(
        user_id=user_id, auth_method="jwt"
    )


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(monkeypatch) -> None:
    app.dependency_overrides.clear()
    monkeypatch.setattr("src.api.app.USER_WRITE_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.SEARCH_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.WEBHOOK_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.track", lambda *args, **kwargs: None)
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _mock_free_video_count(monkeypatch) -> None:
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: False)
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 0)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: None)
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: ProcessingCreditConsumeResult(
            allowed=False,
            remaining_balance=0,
        ),
    )


@pytest.fixture(autouse=True)
def _mock_upload_duration_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.app._validate_upload_file_duration_or_raise",
        lambda file: None,
    )
    monkeypatch.setattr(
        "src.api.app._validate_uploaded_source_duration_with_cleanup",
        lambda store, key, user_id: None,
    )


def _make_api_key_record(
    user_id: str = "user_123",
    *,
    raw_key: str = "vmf_deadbeef12345678deadbeef12345678",
    name: str = "test-key",
    key_id: str = "00000000-0000-4000-8000-aaaaaaaaaaaa",
) -> tuple[str, ApiKeyRecord]:
    """Return (raw_key, ApiKeyRecord) pair for testing API key auth."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = "vmf_" + raw_key[4:8]
    record = ApiKeyRecord(
        id=key_id,
        user_id=user_id,
        name=name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return raw_key, record


def _mock_api_key_auth(monkeypatch, record: ApiKeyRecord) -> None:
    """Monkeypatch API key DB lookups so the given record is found on auth."""
    monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda h: record)
    monkeypatch.setattr("src.api.auth.touch_api_key_last_used", lambda kid: None)


def _setup_api_key_auth(
    monkeypatch, user_id: str = "user_123", **kwargs,
) -> tuple[str, ApiKeyRecord]:
    """Create an API key record and wire up auth mocks in one call."""
    raw_key, record = _make_api_key_record(user_id=user_id, **kwargs)
    _mock_api_key_auth(monkeypatch, record)
    return raw_key, record
