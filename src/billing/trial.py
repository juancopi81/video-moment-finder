"""Server-authorized, once-per-Clerk-account trial enrollment.

The flag controls new grants only. Existing enrollment remains authoritative
when grants are paused, so reconnects and rollbacks cannot restore a second
legacy website allowance.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib import error, parse, request

from src.db.supabase import (
    ApiTrialGrantRecord,
    apply_api_trial_grant,
    get_api_trial_grant,
)
from src.utils.env import get_env_int
from src.utils.logging import get_logger

logger = get_logger(__name__)


def trial_grants_enabled() -> bool:
    return os.environ.get("API_TRIAL_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }


def trial_allowance_units() -> int:
    """Invalid settings fail closed instead of silently changing the offer."""
    units = int(os.environ.get("API_TRIAL_UNITS", "600"))
    if not 1 <= units <= 100_000:
        raise ValueError("API_TRIAL_UNITS must be between 1 and 100000")
    return units


class TrialVerificationUnavailable(RuntimeError):
    """Clerk could not provide authoritative account verification."""


def verified_primary_email_id(user_id: str) -> str | None:
    """Read verification from Clerk, never from client metadata or a boolean.

    A verified secondary email does not qualify an unverified primary email.
    Only the email object's opaque ID is retained as audit evidence.
    """
    secret = os.environ.get("CLERK_SECRET_KEY", "").strip()
    if not secret:
        raise TrialVerificationUnavailable("Clerk server access is not configured")
    req = request.Request(
        "https://api.clerk.com/v1/users/" + parse.quote(user_id, safe=""),
        headers={"Authorization": f"Bearer {secret}", "Accept": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=5) as response:
            payload = json.load(response)
    except (error.URLError, TimeoutError, ValueError) as exc:
        # Avoid recording request headers, the secret, or email addresses.
        raise TrialVerificationUnavailable("Account verification is unavailable") from exc
    if not isinstance(payload, dict) or payload.get("id") != user_id:
        raise TrialVerificationUnavailable("Account verification response is invalid")
    if payload.get("banned") or payload.get("locked"):
        return None
    primary_id = payload.get("primary_email_address_id")
    emails = payload.get("email_addresses")
    if not isinstance(primary_id, str) or not isinstance(emails, list):
        return None
    for email in emails:
        if not isinstance(email, dict) or email.get("id") != primary_id:
            continue
        verification = email.get("verification")
        if isinstance(verification, dict) and verification.get("status") == "verified":
            return primary_id
    return None


@dataclass(frozen=True)
class TrialState:
    enabled: bool
    status: str
    allowance_units: int
    grant: ApiTrialGrantRecord | None = None

    @property
    def enrolled(self) -> bool:
        return self.grant is not None

    def summary(self, api_balance: int) -> dict:
        status = self.status
        if self.enrolled:
            status = "granted" if api_balance > 0 else "exhausted"
        return {
            "trial_enabled": self.enabled,
            "trial_status": status,
            "trial_allowance_units": self.grant.allowance_units if self.grant else self.allowance_units,
            "trial_units_granted": self.grant.granted_units if self.grant else 0,
            "trial_legacy_units_offset": self.grant.legacy_units_offset if self.grant else 0,
        }


def existing_trial_state(user_id: str) -> TrialState:
    """Read enrollment without initiating identity checks or a new grant."""
    enabled = trial_grants_enabled()
    allowance = trial_allowance_units()
    grant = get_api_trial_grant(user_id)
    return TrialState(enabled, "granted" if grant else "disabled", allowance, grant)


def ensure_trial_state(user_id: str) -> TrialState:
    """Enroll eligible authenticated accounts; preserve paid access on failure.

    Call before inserting a new video, so reconciliation only sees historical
    website usage. A database unique key, not this read, prevents double grants.
    """
    state = existing_trial_state(user_id)
    if state.enrolled or not state.enabled:
        return state
    allowance = state.allowance_units
    try:
        email_id = verified_primary_email_id(user_id)
    except TrialVerificationUnavailable:
        logger.warning("Trial identity verification is unavailable")
        return TrialState(True, "verification_unavailable", allowance)
    if email_id is None:
        return TrialState(True, "verification_required", allowance)
    grant = apply_api_trial_grant(
        user_id=user_id,
        verified_email_id=email_id,
        allowance_units=allowance,
        legacy_free_videos=get_env_int("VIDEO_MAX_FREE_VIDEOS", 1),
        index_cost_units=get_env_int("API_UNIT_COST_INDEX_VIDEO", 500),
    )
    return TrialState(True, "granted", allowance, grant)
