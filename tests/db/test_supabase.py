"""Unit tests for Supabase CRUD operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.db.supabase as supabase_module
from src.db.supabase import (
    TRANSCRIPT_SEGMENTS_PAGE_SIZE,
    VideoRecord,
    CreditRecord,
    TranscriptSegmentRecord,
    _row_to_video,
    _row_to_credit,
    _row_to_transcript_segment,
    apply_billing_credit_grant,
    consume_processing_credit,
    count_videos_for_user,
    create_uploaded_video,
    create_video,
    get_video,
    get_video_transcript_segments,
    has_unlimited_video_access,
    insert_uploaded_video_idempotent,
    insert_youtube_video_idempotent,
    replace_video_transcript_segments,
    reset_client,
    search_video_transcript_segments,
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
    assert video.duration_s is None


def test_row_to_video_converts_duration_s_when_present() -> None:
    """Uploaded videos with a probed duration expose it as a float."""
    row = {
        "id": "abc-123",
        "youtube_url": None,
        "status": "ready",
        "user_id": "user_456",
        "error_message": None,
        "source_type": "upload",
        "source_r2_key": "source/abc-123/upload.mp4",
        "source_filename": "upload.mp4",
        "created_at": "2026-01-27T10:00:00Z",
        "updated_at": "2026-01-27T10:00:00Z",
        "duration_s": 342.5,
    }

    video = _row_to_video(row)

    assert video.duration_s == 342.5


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


def test_row_to_transcript_segment_converts_correctly() -> None:
    row = {
        "video_id": "video-123",
        "segment_index": 4,
        "start_s": 12.5,
        "end_s": 15.0,
        "text": "Discussing the benchmark results.",
        "language_code": "en",
        "score": 0.77,
    }

    segment = _row_to_transcript_segment(row)

    assert segment.video_id == "video-123"
    assert segment.segment_index == 4
    assert segment.start_s == 12.5
    assert segment.end_s == 15.0
    assert segment.text == "Discussing the benchmark results."
    assert segment.language_code == "en"
    assert segment.score == 0.77


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
def test_insert_youtube_video_idempotent_handles_list_wrapped_rpc_output(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = [
        {
            "row_data": {
                "id": "video-list-shape",
                "youtube_url": "https://youtube.com/watch?v=abc",
                "status": "queued",
                "user_id": "user_123",
                "error_message": None,
                "source_type": "youtube",
                "source_r2_key": None,
                "source_filename": None,
                "created_at": "2026-03-17T00:00:00Z",
                "updated_at": "2026-03-17T00:00:00Z",
            },
            "was_created": True,
        }
    ]

    record, was_created = insert_youtube_video_idempotent(
        "https://youtube.com/watch?v=abc",
        "user_123",
    )

    assert record.id == "video-list-shape"
    assert was_created is True


@patch("src.db.supabase.get_client")
def test_insert_uploaded_video_idempotent_handles_dict_rpc_output(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = {
        "row_data": {
            "id": "video-dict-shape",
            "youtube_url": None,
            "status": "queued",
            "user_id": "user_123",
            "error_message": None,
            "source_type": "upload",
            "source_r2_key": "source/video-dict-shape/upload.mp4",
            "source_filename": "upload.mp4",
            "created_at": "2026-03-17T00:00:00Z",
            "updated_at": "2026-03-17T00:00:00Z",
        },
        "was_created": False,
    }

    record, was_created = insert_uploaded_video_idempotent(
        "video-dict-shape",
        "user_123",
        "source/video-dict-shape/upload.mp4",
        "upload.mp4",
    )

    assert record.id == "video-dict-shape"
    assert record.source_type == "upload"
    assert was_created is False


@patch("src.db.supabase.get_client")
def test_insert_uploaded_video_idempotent_passes_duration_s_to_rpc(
    mock_get_client: MagicMock,
) -> None:
    """The probed duration is forwarded to the RPC so it's persisted at admission."""
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = {
        "row_data": {
            "id": "video-dur",
            "youtube_url": None,
            "status": "queued",
            "user_id": "user_123",
            "error_message": None,
            "source_type": "upload",
            "source_r2_key": "source/video-dur/upload.mp4",
            "source_filename": "upload.mp4",
            "created_at": "2026-03-17T00:00:00Z",
            "updated_at": "2026-03-17T00:00:00Z",
            "duration_s": 125.0,
        },
        "was_created": True,
    }

    record, was_created = insert_uploaded_video_idempotent(
        "video-dur",
        "user_123",
        "source/video-dur/upload.mp4",
        "upload.mp4",
        duration_s=125.0,
    )

    mock_client.rpc.assert_called_once_with(
        "insert_uploaded_video_idempotent",
        {
            "p_video_id": "video-dur",
            "p_user_id": "user_123",
            "p_source_r2_key": "source/video-dur/upload.mp4",
            "p_source_filename": "upload.mp4",
            "p_duration_s": 125.0,
        },
    )
    assert record.duration_s == 125.0
    assert was_created is True


@patch("src.db.supabase.get_client")
def test_create_uploaded_video_includes_duration_s_when_provided(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [
        {
            "id": "video-created",
            "youtube_url": None,
            "status": "queued",
            "user_id": "user_123",
            "error_message": None,
            "source_type": "upload",
            "source_r2_key": "source/video-created/upload.mp4",
            "source_filename": "upload.mp4",
            "created_at": "2026-03-17T00:00:00Z",
            "updated_at": "2026-03-17T00:00:00Z",
            "duration_s": 60.0,
        }
    ]

    video = create_uploaded_video(
        video_id="video-created",
        source_r2_key="source/video-created/upload.mp4",
        source_filename="upload.mp4",
        user_id="user_123",
        duration_s=60.0,
    )

    inserted_payload = mock_client.table.return_value.insert.call_args[0][0]
    assert inserted_payload["duration_s"] == 60.0
    assert video.duration_s == 60.0


@patch("src.db.supabase.get_client")
def test_create_uploaded_video_omits_duration_s_when_not_provided(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.table.return_value.insert.return_value.execute.return_value.data = [
        {
            "id": "video-no-dur",
            "youtube_url": None,
            "status": "queued",
            "user_id": "user_123",
            "error_message": None,
            "source_type": "upload",
            "source_r2_key": "source/video-no-dur/upload.mp4",
            "source_filename": "upload.mp4",
            "created_at": "2026-03-17T00:00:00Z",
            "updated_at": "2026-03-17T00:00:00Z",
        }
    ]

    create_uploaded_video(
        video_id="video-no-dur",
        source_r2_key="source/video-no-dur/upload.mp4",
        source_filename="upload.mp4",
        user_id="user_123",
    )

    inserted_payload = mock_client.table.return_value.insert.call_args[0][0]
    assert "duration_s" not in inserted_payload


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
def test_has_unlimited_video_access_returns_true_for_override(
    mock_get_client: MagicMock,
) -> None:
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


@patch("src.db.supabase.get_client")
def test_replace_video_transcript_segments_replaces_rows(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = 1

    inserted = replace_video_transcript_segments(
        "video_123",
        [
            TranscriptSegmentRecord(
                video_id="video_123",
                segment_index=0,
                start_s=1.0,
                end_s=2.0,
                text="hello world",
                language_code="en",
            )
        ],
    )

    assert inserted == 1
    mock_client.rpc.assert_called_once_with(
        "replace_video_transcript_segments",
        {
            "p_video_id": "video_123",
            "p_segments": [
                {
                    "segment_index": 0,
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "text": "hello world",
                    "language_code": "en",
                }
            ],
        },
    )


@patch("src.db.supabase.get_client")
def test_search_video_transcript_segments_calls_rpc(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = [
        {
            "video_id": "video_123",
            "segment_index": 2,
            "start_s": 8.0,
            "end_s": 10.0,
            "text": "we discuss onboarding here",
            "language_code": "en",
            "score": 0.61,
        }
    ]

    segments = search_video_transcript_segments(
        "video_123",
        "onboarding",
        limit=3,
    )

    assert segments == [
        TranscriptSegmentRecord(
            video_id="video_123",
            segment_index=2,
            start_s=8.0,
            end_s=10.0,
            text="we discuss onboarding here",
            language_code="en",
            score=0.61,
        )
    ]
    mock_client.rpc.assert_called_once_with(
        "search_video_transcript_segments",
        {
            "p_video_id": "video_123",
            "p_query": "onboarding",
            "p_limit": 3,
        },
    )


@patch("src.db.supabase.get_client")
def test_get_video_transcript_segments_orders_by_index(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.range.return_value = mock_query
    mock_query.execute.return_value.data = [
        {
            "video_id": "video_123",
            "segment_index": 0,
            "start_s": 0.0,
            "end_s": 2.0,
            "text": "Hello",
            "language_code": "en",
        },
        {
            "video_id": "video_123",
            "segment_index": 1,
            "start_s": 2.0,
            "end_s": 4.5,
            "text": "World",
            "language_code": "en",
        },
    ]
    mock_client.table.return_value.select.return_value = mock_query

    segments = get_video_transcript_segments("video_123")

    assert segments == [
        TranscriptSegmentRecord(
            video_id="video_123", segment_index=0, start_s=0.0, end_s=2.0,
            text="Hello", language_code="en",
        ),
        TranscriptSegmentRecord(
            video_id="video_123", segment_index=1, start_s=2.0, end_s=4.5,
            text="World", language_code="en",
        ),
    ]
    mock_client.table.assert_called_with("video_transcript_segments")
    mock_query.eq.assert_called_once_with("video_id", "video_123")
    mock_query.order.assert_called_once_with("segment_index")
    mock_query.range.assert_called_once_with(0, TRANSCRIPT_SEGMENTS_PAGE_SIZE - 1)
    mock_query.gte.assert_not_called()
    mock_query.lte.assert_not_called()


@patch("src.db.supabase.get_client")
def test_get_video_transcript_segments_applies_range_filter(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_query = MagicMock()
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.range.return_value = mock_query
    mock_query.gte.return_value = mock_query
    mock_query.lte.return_value = mock_query
    mock_query.execute.return_value.data = []
    mock_client.table.return_value.select.return_value = mock_query

    get_video_transcript_segments("video_123", start_s=10.0, end_s=20.0)

    mock_query.gte.assert_called_once_with("end_s", 10.0)
    mock_query.lte.assert_called_once_with("start_s", 20.0)


@patch("src.db.supabase.get_client")
def test_get_video_transcript_segments_paginates_across_multiple_pages(
    mock_get_client: MagicMock,
) -> None:
    """A transcript longer than one PostgREST page must not be truncated.

    Simulates a page-sized first page (which alone would look "complete" to
    a naive caller) followed by a smaller final page, and asserts every
    segment across both pages is returned in order.
    """
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    def _segment_row(idx: int) -> dict:
        return {
            "video_id": "video_123",
            "segment_index": idx,
            "start_s": float(idx),
            "end_s": float(idx) + 1.0,
            "text": f"segment {idx}",
            "language_code": "en",
        }

    page_size = 3
    with patch.object(supabase_module, "TRANSCRIPT_SEGMENTS_PAGE_SIZE", page_size):
        first_page = [_segment_row(i) for i in range(page_size)]
        second_page = [_segment_row(i) for i in range(page_size, page_size + 2)]

        mock_query = MagicMock()
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        range_calls: list[tuple[int, int]] = []

        def _range(start: int, end: int):
            range_calls.append((start, end))
            return mock_query

        mock_query.range.side_effect = _range
        mock_query.execute.side_effect = [
            MagicMock(data=first_page),
            MagicMock(data=second_page),
        ]
        mock_client.table.return_value.select.return_value = mock_query

        segments = get_video_transcript_segments("video_123")

    assert range_calls == [(0, page_size - 1), (page_size, 2 * page_size - 1)]
    assert [s.segment_index for s in segments] == [0, 1, 2, 3, 4]
    assert [s.text for s in segments] == [f"segment {i}" for i in range(5)]


def test_get_video_transcript_segments_rejects_empty_video_id() -> None:
    with pytest.raises(ValueError, match="video_id"):
        get_video_transcript_segments("")


def test_update_credits_rejects_negative_balance() -> None:
    """Test that update_credits raises ValueError for negative balance."""
    with pytest.raises(ValueError, match="cannot be negative"):
        update_credits("user_123", -10)


def test_consume_processing_credit_rejects_empty_user_id() -> None:
    with pytest.raises(ValueError, match="user_id must be non-empty"):
        consume_processing_credit("   ")


@patch("src.db.supabase.get_client")
def test_consume_processing_credit_calls_rpc(mock_get_client: MagicMock) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = [
        {"allowed": True, "charged": True, "remaining_balance": 4}
    ]

    result = consume_processing_credit("user_123")

    assert result.allowed is True
    assert result.remaining_balance == 4
    mock_client.rpc.assert_called_once_with(
        "consume_processing_credit",
        {"p_user_id": "user_123"},
    )


@patch("src.db.supabase.get_client")
def test_consume_processing_credit_parses_denied_response(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = [
        {"allowed": False, "charged": False, "remaining_balance": 0}
    ]

    result = consume_processing_credit("user_123")

    assert result.allowed is False
    assert result.remaining_balance == 0


@patch("src.db.supabase.get_client")
def test_consume_processing_credit_warns_on_unexpected_response_shape(
    mock_get_client: MagicMock,
) -> None:
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.rpc.return_value.execute.return_value.data = "unexpected"
    warning = MagicMock()
    original_warning = supabase_module.logger.warning
    supabase_module.logger.warning = warning
    try:
        result = consume_processing_credit("user_123")
    finally:
        supabase_module.logger.warning = original_warning

    warning.assert_called_once()
    assert (
        "Unexpected consume_processing_credit RPC response shape"
        in warning.call_args.args[0]
    )
    assert result.allowed is False
    assert result.remaining_balance == 0


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
    auth_client = MagicMock()

    fake_client = MagicMock()
    fake_client.postgrest = MagicMock(session=postgrest_session)
    fake_client.storage = MagicMock(session=storage_session)
    fake_client.auth = auth_client

    monkeypatch.setattr(supabase_module, "_client", fake_client)

    reset_client()

    postgrest_session.close.assert_called_once()
    storage_session.close.assert_called_once()
    auth_client.close.assert_called_once()
    assert supabase_module._client is None


def test_reset_client_ignores_session_close_errors(monkeypatch) -> None:
    broken_session = MagicMock()
    broken_session.close.side_effect = RuntimeError("close failed")
    broken_auth = MagicMock()
    broken_auth.close.side_effect = RuntimeError("auth close failed")

    fake_client = MagicMock()
    fake_client.postgrest = MagicMock(session=broken_session)
    fake_client.storage = MagicMock(session=MagicMock())
    fake_client.auth = broken_auth

    monkeypatch.setattr(supabase_module, "_client", fake_client)

    reset_client()

    assert supabase_module._client is None
