"""Tests for src.analytics.events."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.analytics.events import track


@pytest.fixture(autouse=True)
def _mock_supabase(monkeypatch):
    """Provide a mock Supabase client for all tests."""
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_insert = MagicMock()
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.return_value = MagicMock(data=[{"id": "test"}])
    monkeypatch.setattr("src.analytics.events.get_client", lambda: mock_client)
    return mock_client


def test_track_inserts_valid_event(_mock_supabase):
    track("video_submitted", user_id="user_1", metadata={"source_type": "youtube"})
    _mock_supabase.table.assert_called_once_with("analytics_events")
    insert_call = _mock_supabase.table.return_value.insert
    insert_call.assert_called_once()
    payload = insert_call.call_args[0][0]
    assert payload["event_name"] == "video_submitted"
    assert payload["user_id"] == "user_1"
    assert payload["metadata"] == {"source_type": "youtube"}


def test_track_rejects_unknown_event(_mock_supabase):
    track("unknown_event", user_id="user_1")
    _mock_supabase.table.assert_not_called()


def test_track_rejects_signup_complete_without_user(_mock_supabase):
    track("signup_complete", user_id=None)
    _mock_supabase.table.assert_not_called()


def test_track_allows_signup_complete_with_user(_mock_supabase):
    track("signup_complete", user_id="user_1")
    _mock_supabase.table.assert_called_once_with("analytics_events")


def test_track_swallows_exceptions(monkeypatch):
    mock_client = MagicMock()
    mock_client.table.side_effect = RuntimeError("db down")
    monkeypatch.setattr("src.analytics.events.get_client", lambda: mock_client)
    # Should not raise
    track("video_ready")


def test_track_swallows_duplicate_signup(monkeypatch):
    mock_client = MagicMock()
    mock_table = MagicMock()
    mock_client.table.return_value = mock_table
    mock_insert = MagicMock()
    mock_table.insert.return_value = mock_insert
    mock_insert.execute.side_effect = Exception(
        "duplicate key value violates unique constraint "
        '"analytics_events_signup_complete_user_unique"'
    )
    monkeypatch.setattr("src.analytics.events.get_client", lambda: mock_client)
    # Should not raise — treated as no-op
    track("signup_complete", user_id="user_1")


def test_track_defaults_metadata_to_empty_dict(_mock_supabase):
    track("video_ready")
    payload = _mock_supabase.table.return_value.insert.call_args[0][0]
    assert payload["metadata"] == {}
    assert payload["user_id"] is None
