"""Unit tests for Supabase CRUD operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.db.supabase as supabase_module
from src.db.supabase import (
    VideoRecord,
    CreditRecord,
    _row_to_video,
    _row_to_credit,
    apply_billing_credit_grant,
    count_videos_for_user,
    create_video,
    get_video,
    has_unlimited_video_access,
    reset_client,
    update_credits,
)


def test_row_to_video_converts_correctly() -> None:
    """Test that database rows are converted to VideoRecord."""
    row = {
        "id": "abc-123",
        "youtube_url": "https://youtube.com/watch?v=test",
        "status": "processing",
        "user_id": "user_456",
        "error_message": None,
        "source_type": "youtube",
        "source_r2_key": None,
        "source_filename": None,
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
    }
    video = _row_to_video(row)

    assert video.id == "abc-123"
    assert video.youtube_url == "https://youtube.com/watch?v=test"
    assert video.status == "processing"
    assert video.user_id == "user_456"
    assert video.error_message is None


def test_row_to_credit_converts_correctly() -> None:
    """Test that database rows are converted to CreditRecord."""
    row = {
        "id": "credit-123",
        "user_id": "user_456",
        "balance": 100,
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
    }
    credit = _row_to_credit(row)

    assert credit.id == "credit-123"
    assert credit.user_id == "user_456"
    assert credit.balance == 100


@patch("src.db.supabase.get_client")
def test_create_video_returns_record(mock_get_client: MagicMock) -> None:
    """Test that create_video calls Supabase and returns VideoRecord."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_client.table.return_value.insert.return_value.execute.return_value.data = [
        {
            "id": "new-video-id",
            "youtube_url": "https://youtube.com/watch?v=abc",
            "status": "processing",
            "user_id": None,
            "error_message": None,
            "source_type": "youtube",
            "source_r2_key": None,
            "source_filename": None,
            "created_at": "2026-01-27T10:00:00Z",
            "updated_at": "2026-01-27T10:00:00Z",
        }
    ]

    video = create_video("https://youtube.com/watch?v=abc")

    assert video.id == "new-video-id"
    assert video.status == "processing"
    mock_client.table.assert_called_with("videos")


@patch("src.db.supabase.get_client")
def test_get_video_returns_none_when_not_found(mock_get_client: MagicMock) -> None:
    """Test that get_video returns None for non-existent video."""
    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.execute.return_value.data = []
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value = mock_query
    mock_get_client.return_value = mock_client

    result = get_video("non-existent-id")

    assert result is None


@patch("src.db.supabase.get_client")
def test_get_video_scopes_by_user_id_when_provided(mock_get_client: MagicMock) -> None:
    """Test that get_video applies user_id filter when provided."""
    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.execute.return_value.data = []
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value = mock_query
    mock_get_client.return_value = mock_client

    _ = get_video("video-123", user_id="user_456")

    assert mock_query.eq.call_args_list[0].args == ("id", "video-123")
    assert mock_query.eq.call_args_list[1].args == ("user_id", "user_456")


@patch("src.db.supabase.get_client")
def test_count_videos_for_user_excludes_failed(mock_get_client: MagicMock) -> None:
    """Test that free-cap counting ignores failed videos."""
    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.neq.return_value = mock_query
    mock_query.execute.return_value.count = 2
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value = mock_query
    mock_get_client.return_value = mock_client

    count = count_videos_for_user("user_456")

    assert count == 2
    assert mock_query.eq.call_args.args == ("user_id", "user_456")
    assert mock_query.neq.call_args.args == ("status", "failed")


@patch("src.db.supabase.get_client")
def test_has_unlimited_video_access_returns_true_for_override(mock_get_client: MagicMock) -> None:
    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value.data = [{"user_id": "user_456"}]
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value = mock_query
    mock_get_client.return_value = mock_client

    result = has_unlimited_video_access("user_456")

    assert result is True
    assert mock_query.eq.call_args_list[0].args == ("user_id", "user_456")
    assert mock_query.eq.call_args_list[1].args == ("unlimited_videos", True)


@patch("src.db.supabase.get_client")
def test_has_unlimited_video_access_returns_false_without_override(
    mock_get_client: MagicMock,
) -> None:
    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.limit.return_value = mock_query
    mock_query.execute.return_value.data = []
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value = mock_query
    mock_get_client.return_value = mock_client

    assert has_unlimited_video_access("user_456") is False


def test_update_credits_rejects_negative_balance() -> None:
    """Test that update_credits raises ValueError for negative balance."""
    with pytest.raises(ValueError, match="cannot be negative"):
        update_credits("user_123", -10)


def test_apply_billing_credit_grant_rejects_non_positive_credits() -> None:
    with pytest.raises(ValueError, match="credits must be > 0"):
        apply_billing_credit_grant(
            provider="lemonsqueezy",
            event_id="order_created:1",
            event_type="order_created",
            user_id="user_123",
            credits=0,
        )


@patch("src.db.supabase.get_client")
def test_apply_billing_credit_grant_calls_rpc(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = True

    result = apply_billing_credit_grant(
        provider="lemonsqueezy",
        event_id="order_created:123",
        event_type="order_created",
        user_id="user_123",
        credits=5,
        payload={"meta": {"event_name": "order_created"}},
    )

    assert result.applied is True
    mock_client.rpc.assert_called_once()
    assert mock_client.rpc.call_args.args[0] == "apply_billing_credit_grant"


def test_reset_client_clears_singleton_and_closes_sessions(monkeypatch) -> None:
    postgrest_session = MagicMock()
    storage_session = MagicMock()

    fake_client = MagicMock()
    fake_client.postgrest = MagicMock(session=postgrest_session)
    fake_client.storage = MagicMock(session=storage_session)

    monkeypatch.setattr(supabase_module, "_client", fake_client)

    reset_client()

    postgrest_session.close.assert_called_once()
    storage_session.close.assert_called_once()
    assert supabase_module._client is None


def test_reset_client_ignores_session_close_errors(monkeypatch) -> None:
    broken_session = MagicMock()
    broken_session.close.side_effect = RuntimeError("close failed")

    fake_client = MagicMock()
    fake_client.postgrest = MagicMock(session=broken_session)
    fake_client.storage = MagicMock(session=MagicMock())

    monkeypatch.setattr(supabase_module, "_client", fake_client)

    reset_client()

    assert supabase_module._client is None
