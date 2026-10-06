"""Serialization and parameter validation at the trial RPC boundary."""
from unittest.mock import MagicMock

import pytest

from src.db.supabase import apply_api_trial_grant, consume_trial_or_processing_credit, get_api_trial_grant


ROW = {"user_id": "user_1", "verified_email_id": "email_1", "allowance_units": 600,
       "legacy_units_offset": 500, "granted_units": 100, "created_at": "2026-10-02T00:00:00Z"}


def test_trial_grant_rpc_uses_account_and_server_verified_email_id(monkeypatch):
    client = MagicMock()
    client.rpc.return_value.execute.return_value.data = [ROW]
    monkeypatch.setattr("src.db.supabase.get_client", lambda: client)
    grant = apply_api_trial_grant(user_id="user_1", verified_email_id="email_1",
                                 allowance_units=600, legacy_free_videos=1, index_cost_units=500)
    assert grant.granted_units == 100
    client.rpc.assert_called_once_with("apply_api_trial_grant", {
        "p_user_id": "user_1", "p_verified_email_id": "email_1", "p_allowance_units": 600,
        "p_legacy_free_videos": 1, "p_index_cost_units": 500,
    })


def test_missing_trial_record_does_not_infer_grant_from_paid_balance(monkeypatch):
    client = MagicMock()
    client.table.return_value.select.return_value.eq.return_value.execute.return_value.data = []
    monkeypatch.setattr("src.db.supabase.get_client", lambda: client)
    assert get_api_trial_grant("user_1") is None
    client.table.assert_called_once_with("api_trial_grants")


@pytest.mark.parametrize("overrides", [{"user_id": ""}, {"verified_email_id": ""},
                                       {"allowance_units": 0}, {"legacy_free_videos": -1}, {"index_cost_units": 0}])
def test_invalid_trial_inputs_are_rejected_before_db_access(overrides):
    args = dict(user_id="u", verified_email_id="email_1", allowance_units=600,
                legacy_free_videos=1, index_cost_units=500) | overrides
    with pytest.raises(ValueError):
        apply_api_trial_grant(**args)


def test_shared_charge_uses_one_atomic_rpc(monkeypatch):
    client = MagicMock()
    client.rpc.return_value.execute.return_value.data = [{"allowed": True, "remaining_balance": 100}]
    monkeypatch.setattr("src.db.supabase.get_client", lambda: client)
    result = consume_trial_or_processing_credit(user_id="user_1", video_id="video_1", units=500)
    assert result.allowed and result.remaining_balance == 100
    client.rpc.assert_called_once_with("consume_trial_or_processing_credit", {
        "p_user_id": "user_1", "p_video_id": "video_1", "p_units": 500,
    })
