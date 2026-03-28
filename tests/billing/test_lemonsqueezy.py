from __future__ import annotations

import io
import json
from urllib import error

import pytest

from src.billing.lemonsqueezy import (
    LemonSqueezyConfigError,
    LemonSqueezyProviderError,
    create_checkout_session,
)


class _MockResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _set_checkout_env(monkeypatch, *, test_mode: str = "true") -> None:
    monkeypatch.setenv("LEMON_SQUEEZY_API_KEY", "api_key_123")
    monkeypatch.setenv("LEMON_SQUEEZY_STORE_ID", "store_456")
    monkeypatch.setenv(
        "LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL",
        "https://videomomentfinder.com/dashboard?checkout=success",
    )
    monkeypatch.setenv("LEMON_SQUEEZY_CHECKOUT_TEST_MODE", test_mode)


def _header_value(req, name: str) -> str:
    for key, value in req.header_items():
        if key.lower() == name.lower():
            return value
    raise AssertionError(f"missing header {name}")


@pytest.mark.parametrize(
    ("plan", "credits", "variant_id"),
    [
        ("starter", 5, "var_starter_1"),
        ("pro", 20, "var_pro_1"),
    ],
)
def test_create_checkout_session_builds_expected_payload(
    monkeypatch,
    plan: str,
    credits: int,
    variant_id: str,
) -> None:
    _set_checkout_env(monkeypatch, test_mode="true")
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["request"] = req
        captured["timeout"] = timeout
        response_payload = {
            "data": {
                "attributes": {
                    "url": "https://app.lemonsqueezy.com/checkout/buy/session_123"
                }
            }
        }
        return _MockResponse(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr("src.billing.lemonsqueezy.request.urlopen", fake_urlopen)

    result = create_checkout_session(
        user_id="user_abc",
        plan=plan,
        credits=credits,
        variant_id=variant_id,
    )

    assert result.url == "https://app.lemonsqueezy.com/checkout/buy/session_123"
    assert result.test_mode is True
    assert captured["timeout"] == 20.0

    req = captured["request"]
    assert req is not None
    payload = json.loads(req.data.decode("utf-8"))

    assert req.full_url == "https://api.lemonsqueezy.com/v1/checkouts"
    assert req.get_method() == "POST"
    assert _header_value(req, "Authorization") == "Bearer api_key_123"
    assert _header_value(req, "Content-Type") == "application/vnd.api+json"

    assert payload["data"]["relationships"]["store"]["data"]["id"] == "store_456"
    assert payload["data"]["relationships"]["variant"]["data"]["id"] == variant_id
    assert payload["data"]["attributes"]["checkout_data"]["custom"] == {
        "user_id": "user_abc",
        "credits": str(credits),
        "plan": plan,
        "grant_target": "web",
    }
    assert (
        payload["data"]["attributes"]["product_options"]["redirect_url"]
        == "https://videomomentfinder.com/dashboard?checkout=success"
    )
    assert payload["data"]["attributes"]["test_mode"] is True


def test_create_checkout_session_raises_for_http_error(monkeypatch) -> None:
    _set_checkout_env(monkeypatch)

    def fake_urlopen(req, timeout):
        raise error.HTTPError(
            url=req.full_url,
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=io.BytesIO(b'{"errors":[{"detail":"boom"}]}'),
        )

    monkeypatch.setattr("src.billing.lemonsqueezy.request.urlopen", fake_urlopen)

    with pytest.raises(
        LemonSqueezyProviderError,
        match="failed with status 500",
    ):
        create_checkout_session(
            user_id="user_abc",
            plan="starter",
            credits=5,
            variant_id="var_starter_1",
        )


def test_create_checkout_session_overrides_redirect_url(monkeypatch) -> None:
    _set_checkout_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured["request"] = req
        response_payload = {
            "data": {
                "attributes": {
                    "url": "https://app.lemonsqueezy.com/checkout/buy/session_456"
                }
            }
        }
        return _MockResponse(json.dumps(response_payload).encode("utf-8"))

    monkeypatch.setattr("src.billing.lemonsqueezy.request.urlopen", fake_urlopen)

    result = create_checkout_session(
        user_id="user_abc",
        plan="developer",
        credits=10_000,
        variant_id="var_dev_1",
        redirect_url="https://www.videomomentfinder.com/connectors/claude?request_id=req-123",
        grant_target="api",
    )

    assert result.url == "https://app.lemonsqueezy.com/checkout/buy/session_456"
    req = captured["request"]
    assert req is not None
    payload = json.loads(req.data.decode("utf-8"))
    assert (
        payload["data"]["attributes"]["product_options"]["redirect_url"]
        == "https://www.videomomentfinder.com/connectors/claude?request_id=req-123"
    )
    assert payload["data"]["attributes"]["checkout_data"]["custom"]["grant_target"] == "api"


def test_create_checkout_session_raises_for_non_json_response(monkeypatch) -> None:
    _set_checkout_env(monkeypatch)
    monkeypatch.setattr(
        "src.billing.lemonsqueezy.request.urlopen",
        lambda _req, timeout: _MockResponse(b"<html>not-json</html>"),
    )

    with pytest.raises(
        LemonSqueezyProviderError,
        match="non-JSON checkout response",
    ):
        create_checkout_session(
            user_id="user_abc",
            plan="starter",
            credits=5,
            variant_id="var_starter_1",
        )


def test_create_checkout_session_raises_for_missing_checkout_url(monkeypatch) -> None:
    _set_checkout_env(monkeypatch)
    monkeypatch.setattr(
        "src.billing.lemonsqueezy.request.urlopen",
        lambda _req, timeout: _MockResponse(
            json.dumps({"data": {"attributes": {}}}).encode("utf-8")
        ),
    )

    with pytest.raises(
        LemonSqueezyProviderError,
        match="missing data.attributes.url",
    ):
        create_checkout_session(
            user_id="user_abc",
            plan="starter",
            credits=5,
            variant_id="var_starter_1",
        )


def test_create_checkout_session_requires_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LEMON_SQUEEZY_API_KEY", raising=False)
    monkeypatch.setenv("LEMON_SQUEEZY_STORE_ID", "store_456")
    monkeypatch.setenv(
        "LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL",
        "https://videomomentfinder.com/dashboard?checkout=success",
    )

    with pytest.raises(
        LemonSqueezyConfigError,
        match="LEMON_SQUEEZY_API_KEY environment variable is required",
    ):
        create_checkout_session(
            user_id="user_abc",
            plan="starter",
            credits=5,
            variant_id="var_starter_1",
        )


def test_create_checkout_session_rejects_invalid_test_mode(monkeypatch) -> None:
    _set_checkout_env(monkeypatch, test_mode="not-a-bool")

    with pytest.raises(
        LemonSqueezyConfigError,
        match="LEMON_SQUEEZY_CHECKOUT_TEST_MODE must be true/false",
    ):
        create_checkout_session(
            user_id="user_abc",
            plan="starter",
            credits=5,
            variant_id="var_starter_1",
        )
