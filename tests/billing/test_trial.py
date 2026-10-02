"""Eligibility is trusted server data; reconnects never create a new grant."""
from __future__ import annotations

import io
import json
from unittest.mock import Mock

import pytest

from src.billing import trial
from src.db.supabase import ApiTrialGrantRecord


def _grant(**overrides):
    values = dict(user_id="user_a", allowance_units=600, legacy_units_offset=0,
                  granted_units=600, verified_email_id="email_primary")
    return ApiTrialGrantRecord(**(values | overrides))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("API_TRIAL_ENABLED", "true")
    monkeypatch.setenv("API_TRIAL_UNITS", "600")
    monkeypatch.setenv("CLERK_SECRET_KEY", "test-server-secret")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
    monkeypatch.setenv("API_UNIT_COST_INDEX_VIDEO", "500")
    monkeypatch.setattr(trial, "get_api_trial_grant", lambda _: None)


def _clerk(monkeypatch, payload):
    request = Mock(return_value=io.BytesIO(json.dumps(payload).encode()))
    monkeypatch.setattr(trial.request, "urlopen", request)
    return request


def _verified_payload():
    return {"id": "user_a", "primary_email_address_id": "email_primary",
            "email_addresses": [{"id": "email_primary", "verification": {"status": "verified"}}]}


def test_verified_primary_email_uses_fixed_clerk_server_endpoint(monkeypatch):
    request = _clerk(monkeypatch, _verified_payload())
    assert trial.verified_primary_email_id("user_a") == "email_primary"
    req = request.call_args.args[0]
    assert req.full_url == "https://api.clerk.com/v1/users/user_a"
    assert req.get_header("Authorization") == "Bearer test-server-secret"


@pytest.mark.parametrize("payload", [
    {"id": "user_a", "email_verified": True, "unsafe_metadata": {"verified": True}},
    {"id": "user_a", "primary_email_address_id": "primary", "email_addresses": [
        {"id": "secondary", "verification": {"status": "verified"}}]},
    {"id": "user_a", "primary_email_address_id": "primary", "email_addresses": [
        {"id": "primary", "verification": {"status": "unverified"}}]},
    _verified_payload() | {"banned": True},
    _verified_payload() | {"locked": True},
])
def test_ineligible_accounts_never_grant(monkeypatch, payload):
    _clerk(monkeypatch, payload)
    apply = Mock()
    monkeypatch.setattr(trial, "apply_api_trial_grant", apply)
    state = trial.ensure_trial_state("user_a")
    assert state.status == "verification_required"
    assert not state.enrolled
    apply.assert_not_called()


def test_mismatched_clerk_user_is_not_trusted(monkeypatch):
    _clerk(monkeypatch, _verified_payload() | {"id": "user_other"})
    assert trial.ensure_trial_state("user_a").status == "verification_unavailable"


def test_unavailable_clerk_does_not_grant(monkeypatch):
    monkeypatch.delenv("CLERK_SECRET_KEY")
    apply = Mock()
    monkeypatch.setattr(trial, "apply_api_trial_grant", apply)
    assert trial.ensure_trial_state("user_a").status == "verification_unavailable"
    apply.assert_not_called()


def test_disabled_flag_never_verifies_or_grants_new_account(monkeypatch):
    monkeypatch.delenv("API_TRIAL_ENABLED")
    verify = Mock(side_effect=AssertionError("no Clerk call while disabled"))
    monkeypatch.setattr(trial, "verified_primary_email_id", verify)
    assert trial.ensure_trial_state("user_a").status == "disabled"
    verify.assert_not_called()


def test_one_grant_on_reconnect_even_with_changed_configuration(monkeypatch):
    stored = {}
    monkeypatch.setattr(trial, "get_api_trial_grant", lambda uid: stored.get(uid))
    monkeypatch.setattr(trial, "verified_primary_email_id", lambda _: "email_primary")
    def grant(**kwargs):
        stored[kwargs["user_id"]] = _grant()
        return stored[kwargs["user_id"]]
    apply = Mock(side_effect=grant)
    monkeypatch.setattr(trial, "apply_api_trial_grant", apply)
    assert trial.ensure_trial_state("user_a").enrolled
    monkeypatch.setenv("API_TRIAL_UNITS", "900")
    assert trial.ensure_trial_state("user_a").summary(100)["trial_allowance_units"] == 600
    apply.assert_called_once_with(user_id="user_a", verified_email_id="email_primary",
                                 allowance_units=600, legacy_free_videos=1, index_cost_units=500)


def test_pausing_grants_preserves_enrollment_and_exhaustion(monkeypatch):
    monkeypatch.setenv("API_TRIAL_ENABLED", "false")
    monkeypatch.setattr(trial, "get_api_trial_grant", lambda _: _grant())
    state = trial.ensure_trial_state("user_a")
    assert state.enrolled and not state.enabled
    assert state.summary(0)["trial_status"] == "exhausted"
    assert state.summary(100)["trial_units_granted"] == 600


def test_legacy_offset_uses_the_same_effective_env_defaults_as_api_billing(monkeypatch):
    monkeypatch.setenv("API_UNIT_COST_INDEX_VIDEO", "0")
    monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "invalid")
    monkeypatch.setattr(trial, "verified_primary_email_id", lambda _: "email_primary")
    apply = Mock(return_value=_grant())
    monkeypatch.setattr(trial, "apply_api_trial_grant", apply)
    trial.ensure_trial_state("user_a")
    assert apply.call_args.kwargs["index_cost_units"] == 500
    assert apply.call_args.kwargs["legacy_free_videos"] == 1


@pytest.mark.parametrize("value", ["0", "-1", "100001", "bad"])
def test_invalid_allowance_fails_closed(monkeypatch, value):
    monkeypatch.setenv("API_TRIAL_UNITS", value)
    with pytest.raises(ValueError):
        trial.ensure_trial_state("user_a")
