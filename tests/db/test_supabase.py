"""Unit tests for Supabase CRUD operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.db.supabase import (
    VideoRecord,
    CreditRecord,
    _row_to_video,
    _row_to_credit,
    create_video,
    get_video,
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


def test_update_credits_rejects_negative_balance() -> None:
    """Test that update_credits raises ValueError for negative balance."""
    with pytest.raises(ValueError, match="cannot be negative"):
        update_credits("user_123", -10)
