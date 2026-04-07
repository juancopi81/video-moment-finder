"""Lemon Squeezy checkout integration helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib import error, request


class LemonSqueezyConfigError(RuntimeError):
    """Raised when Lemon Squeezy checkout configuration is missing/invalid."""


class LemonSqueezyProviderError(RuntimeError):
    """Raised when Lemon Squeezy checkout API request fails."""


@dataclass(frozen=True)
class LemonSqueezyCheckoutSession:
    """Checkout session response normalized for API handlers."""

    url: str
    test_mode: bool


@dataclass(frozen=True)
class LemonSqueezyCheckoutSettings:
    """Environment-backed Lemon Squeezy checkout settings."""

    api_key: str
    store_id: str
    redirect_url: str
    test_mode: bool


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise LemonSqueezyConfigError(f"{name} environment variable is required")
    return value


def _parse_bool_env(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise LemonSqueezyConfigError(f"{name} must be true/false")


def _settings_from_env() -> LemonSqueezyCheckoutSettings:
    return LemonSqueezyCheckoutSettings(
        api_key=_required_env("LEMON_SQUEEZY_API_KEY"),
        store_id=_required_env("LEMON_SQUEEZY_STORE_ID"),
        redirect_url=_required_env("LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL"),
        test_mode=_parse_bool_env("LEMON_SQUEEZY_CHECKOUT_TEST_MODE", default=False),
    )


def _checkout_payload(
    *,
    store_id: str,
    variant_id: str,
    user_id: str,
    plan: str,
    credits: int,
    redirect_url: str,
    test_mode: bool,
    grant_target: str = "web",
) -> dict:
    return {
        "data": {
            "type": "checkouts",
            "attributes": {
                "checkout_data": {
                    "custom": {
                        "user_id": user_id,
                        "credits": str(credits),
                        "plan": plan,
                        "grant_target": grant_target,
                        "variant_id": variant_id,
                    }
                },
                "product_options": {
                    "redirect_url": redirect_url,
                },
                "test_mode": test_mode,
            },
            "relationships": {
                "store": {
                    "data": {
                        "type": "stores",
                        "id": store_id,
                    }
                },
                "variant": {
                    "data": {
                        "type": "variants",
                        "id": variant_id,
                    }
                },
            },
        }
    }


def create_checkout_session(
    *,
    user_id: str,
    plan: str,
    credits: int,
    variant_id: str,
    redirect_url: str | None = None,
    timeout_s: float = 20.0,
    grant_target: str = "web",
) -> LemonSqueezyCheckoutSession:
    """Create a Lemon Squeezy checkout and return its URL."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if not plan.strip():
        raise ValueError("plan must be non-empty")
    if credits <= 0:
        raise ValueError("credits must be > 0")
    if not variant_id.strip():
        raise LemonSqueezyConfigError("variant_id must be configured")

    settings = _settings_from_env()
    payload = _checkout_payload(
        store_id=settings.store_id,
        variant_id=variant_id.strip(),
        user_id=user_id.strip(),
        plan=plan.strip(),
        credits=credits,
        redirect_url=(redirect_url or settings.redirect_url).strip(),
        test_mode=settings.test_mode,
        grant_target=grant_target,
    )
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = request.Request(
        "https://api.lemonsqueezy.com/v1/checkouts",
        data=body,
        method="POST",
        headers={
            "Accept": "application/vnd.api+json",
            "Content-Type": "application/vnd.api+json",
            "Authorization": f"Bearer {settings.api_key}",
        },
    )

    try:
        with request.urlopen(req, timeout=timeout_s) as response:
            raw_response = response.read()
    except error.HTTPError as exc:
        _ = exc.read()
        raise LemonSqueezyProviderError(
            f"Lemon Squeezy checkout request failed with status {exc.code}"
        ) from exc
    except error.URLError as exc:
        raise LemonSqueezyProviderError(
            "Lemon Squeezy checkout request failed"
        ) from exc

    try:
        parsed = json.loads(raw_response.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LemonSqueezyProviderError(
            "Lemon Squeezy returned a non-JSON checkout response"
        ) from exc

    data = parsed.get("data")
    if not isinstance(data, dict):
        raise LemonSqueezyProviderError(
            "Lemon Squeezy checkout response missing data object"
        )
    attributes = data.get("attributes")
    if not isinstance(attributes, dict):
        raise LemonSqueezyProviderError(
            "Lemon Squeezy checkout response missing data.attributes object"
        )
    checkout_url = attributes.get("url")
    if not isinstance(checkout_url, str) or not checkout_url.strip():
        raise LemonSqueezyProviderError(
            "Lemon Squeezy checkout response missing data.attributes.url"
        )

    return LemonSqueezyCheckoutSession(
        url=checkout_url.strip(),
        test_mode=settings.test_mode,
    )
