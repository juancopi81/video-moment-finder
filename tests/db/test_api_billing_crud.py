"""Unit tests for API billing CRUD operations."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.db.supabase import (
    ApiCreditRecord,
    ApiBillingCreditGrantResult,
    ApiUnitConsumeResult,
    ApiUsageEventRecord,
    _row_to_api_credit,
    _row_to_api_usage_event,
    apply_api_billing_credit_grant,
    compensate_api_units,
    consume_api_units,
    get_api_credits,
    list_api_usage_events,
)


def _mock_client(rpc_return=None, table_data=None):
    """Build a mock Supabase client."""
    client = MagicMock()

    if rpc_return is not None:
        rpc_result = MagicMock()
        rpc_result.data = rpc_return
        client.rpc.return_value.execute.return_value = rpc_result

    if table_data is not None:
        table_result = MagicMock()
        table_result.data = table_data
        # Support chained calls: client.table(...).select(...).eq(...).execute()
        chain = client.table.return_value.select.return_value
        chain.eq.return_value = chain
        chain.order.return_value = chain
        chain.limit.return_value = chain
        chain.execute.return_value = table_result

    return client


class TestGetApiCredits:
    def test_returns_none_for_new_user(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.db.supabase.get_client", lambda: _mock_client(table_data=[])
        )
        result = get_api_credits("user_new")
        assert result is None

    def test_returns_record_when_exists(self, monkeypatch) -> None:
        row = {
            "user_id": "user_123",
            "balance": 5000,
            "created_at": "2026-03-18T00:00:00Z",
            "updated_at": "2026-03-18T00:00:00Z",
        }
        monkeypatch.setattr(
            "src.db.supabase.get_client", lambda: _mock_client(table_data=[row])
        )
        result = get_api_credits("user_123")
        assert result is not None
        assert result.balance == 5000


class TestApplyApiBillingCreditGrant:
    def test_applied_true_on_first_call(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.db.supabase.get_client", lambda: _mock_client(rpc_return=[True])
        )
        result = apply_api_billing_credit_grant(
            provider="test",
            event_id="evt_1",
            event_type="order_created",
            user_id="user_123",
            credits=10000,
        )
        assert result.applied is True

    def test_applied_false_on_duplicate(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.db.supabase.get_client", lambda: _mock_client(rpc_return=[False])
        )
        result = apply_api_billing_credit_grant(
            provider="test",
            event_id="evt_1",
            event_type="order_created",
            user_id="user_123",
            credits=10000,
        )
        assert result.applied is False

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="provider"):
            apply_api_billing_credit_grant(
                provider="", event_id="e", event_type="t", user_id="u", credits=1
            )
        with pytest.raises(ValueError, match="credits"):
            apply_api_billing_credit_grant(
                provider="p", event_id="e", event_type="t", user_id="u", credits=0
            )


class TestConsumeApiUnits:
    def test_allowed_when_balance_sufficient(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.db.supabase.get_client",
            lambda: _mock_client(rpc_return=[{"allowed": True, "remaining_balance": 9500}]),
        )
        result = consume_api_units(
            user_id="user_123",
            api_key_id="key_abc",
            event_type="index_video",
            units=500,
        )
        assert result.allowed is True
        assert result.remaining_balance == 9500

    def test_rejected_when_insufficient(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "src.db.supabase.get_client",
            lambda: _mock_client(rpc_return=[{"allowed": False, "remaining_balance": 100}]),
        )
        result = consume_api_units(
            user_id="user_123",
            api_key_id=None,
            event_type="index_video",
            units=500,
        )
        assert result.allowed is False
        assert result.remaining_balance == 100

    def test_rejects_empty_user_id(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            consume_api_units(user_id="", api_key_id=None, event_type="index_video", units=1)

    def test_rejects_zero_units(self) -> None:
        with pytest.raises(ValueError, match="units"):
            consume_api_units(user_id="u", api_key_id=None, event_type="index_video", units=0)


class TestCompensateApiUnits:
    def test_calls_rpc(self, monkeypatch) -> None:
        client = _mock_client(rpc_return=None)
        rpc_result = MagicMock()
        rpc_result.data = None
        client.rpc.return_value.execute.return_value = rpc_result
        monkeypatch.setattr("src.db.supabase.get_client", lambda: client)

        compensate_api_units(user_id="user_123", units=100)
        client.rpc.assert_called_once_with(
            "compensate_api_units",
            {
                "p_user_id": "user_123",
                "p_units": 100,
                "p_video_id": None,
                "p_request_id": None,
                "p_metadata": {},
            },
        )

    def test_rejects_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="user_id"):
            compensate_api_units(user_id="  ", units=50)
        with pytest.raises(ValueError, match="units"):
            compensate_api_units(user_id="u", units=-1)


class TestListApiUsageEvents:
    def test_returns_events(self, monkeypatch) -> None:
        rows = [
            {
                "id": "evt_1",
                "user_id": "user_123",
                "api_key_id": "key_abc",
                "event_type": "index_video",
                "units": 500,
                "video_id": None,
                "request_id": None,
                "metadata": {},
                "created_at": "2026-03-18T12:00:00Z",
            },
        ]
        monkeypatch.setattr(
            "src.db.supabase.get_client", lambda: _mock_client(table_data=rows)
        )
        result = list_api_usage_events(user_id="user_123")
        assert len(result) == 1
        assert result[0].event_type == "index_video"


class TestRowConversions:
    def test_row_to_api_credit(self) -> None:
        row = {"user_id": "u", "balance": 42, "created_at": None, "updated_at": None}
        record = _row_to_api_credit(row)
        assert record.user_id == "u"
        assert record.balance == 42

    def test_row_to_api_usage_event(self) -> None:
        row = {
            "id": "evt",
            "user_id": "u",
            "api_key_id": None,
            "event_type": "text_query",
            "units": 1,
            "video_id": None,
            "request_id": None,
            "metadata": {},
            "created_at": None,
        }
        record = _row_to_api_usage_event(row)
        assert record.event_type == "text_query"
        assert record.units == 1
