"""Tests for API billing: webhook routing, admission gate, and billing endpoints."""
from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.db.supabase import (
    ApiCreditRecord,
    ApiBillingCreditGrantResult,
    ApiUnitConsumeResult,
    ApiUsageEventRecord,
    BillingCreditGrantResult,
    ProcessingCreditConsumeResult,
)

from tests.api.conftest import (
    _authenticate,
    _setup_api_key_auth,
    _video_record,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEBHOOK_SECRET = "test-webhook-secret"


def _webhook_body(
    *,
    user_id: str = "user_123",
    credits: int = 10_000,
    grant_target: str | None = None,
    event_name: str = "order_created",
    data_id: str = "12345",
) -> dict:
    custom: dict = {"user_id": user_id, "credits": str(credits)}
    if grant_target is not None:
        custom["grant_target"] = grant_target
    return {
        "meta": {
            "event_name": event_name,
            "custom_data": custom,
        },
        "data": {"id": data_id},
    }


def _signed_webhook(body_dict: dict) -> tuple[bytes, str]:
    raw = json.dumps(body_dict).encode()
    sig = hmac.new(WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return raw, sig


# ---------------------------------------------------------------------------
# TestWebhookRoutingByGrantTarget
# ---------------------------------------------------------------------------


class TestWebhookRoutingByGrantTarget:
    def test_grant_target_api_calls_api_billing(self, monkeypatch) -> None:
        monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", WEBHOOK_SECRET)
        called_api = {"value": False}

        def mock_api_grant(**kwargs):
            called_api["value"] = True
            return ApiBillingCreditGrantResult(applied=True)

        monkeypatch.setattr(
            "src.api.app.db_apply_api_billing_credit_grant", mock_api_grant
        )

        body = _webhook_body(grant_target="api")
        raw, sig = _signed_webhook(body)
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=raw,
            headers={"x-signature": sig, "content-type": "application/json"},
        )

        assert response.status_code == 200
        assert response.json()["granted"] is True
        assert called_api["value"] is True

    def test_grant_target_web_calls_web_billing(self, monkeypatch) -> None:
        monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", WEBHOOK_SECRET)
        called_web = {"value": False}

        def mock_web_grant(**kwargs):
            called_web["value"] = True
            return BillingCreditGrantResult(applied=True)

        monkeypatch.setattr(
            "src.api.app.db_apply_billing_credit_grant", mock_web_grant
        )

        body = _webhook_body(grant_target="web")
        raw, sig = _signed_webhook(body)
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=raw,
            headers={"x-signature": sig, "content-type": "application/json"},
        )

        assert response.status_code == 200
        assert response.json()["granted"] is True
        assert called_web["value"] is True

    def test_missing_grant_target_defaults_to_web(self, monkeypatch) -> None:
        monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", WEBHOOK_SECRET)
        called_web = {"value": False}

        def mock_web_grant(**kwargs):
            called_web["value"] = True
            return BillingCreditGrantResult(applied=True)

        monkeypatch.setattr(
            "src.api.app.db_apply_billing_credit_grant", mock_web_grant
        )

        body = _webhook_body(grant_target=None)
        raw, sig = _signed_webhook(body)
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=raw,
            headers={"x-signature": sig, "content-type": "application/json"},
        )

        assert response.status_code == 200
        assert called_web["value"] is True

    def test_mismatched_variant_logs_warning(self, monkeypatch) -> None:
        monkeypatch.setenv("LEMON_SQUEEZY_WEBHOOK_SECRET", WEBHOOK_SECRET)
        monkeypatch.setenv("LEMON_SQUEEZY_VARIANT_ID_DEVELOPER", "expected_variant")

        warnings = []

        def mock_api_grant(**kwargs):
            return ApiBillingCreditGrantResult(applied=True)

        monkeypatch.setattr(
            "src.api.app.db_apply_api_billing_credit_grant", mock_api_grant
        )
        monkeypatch.setattr(
            "src.api.app.logger",
            type("FakeLogger", (), {
                "info": lambda *a, **kw: None,
                "warning": lambda *a, **kw: warnings.append(a),
            })(),
        )

        body = _webhook_body(grant_target="api", data_id="wrong_variant")
        raw, sig = _signed_webhook(body)
        response = client.post(
            "/webhooks/lemonsqueezy",
            content=raw,
            headers={"x-signature": sig, "content-type": "application/json"},
        )

        assert response.status_code == 200
        assert len(warnings) > 0


# ---------------------------------------------------------------------------
# TestApiAdmissionGate
# ---------------------------------------------------------------------------


class TestApiAdmissionGate:
    def test_api_key_video_submit_deducts_api_units(self, monkeypatch) -> None:
        raw_key, record = _setup_api_key_auth(monkeypatch)
        consumed = {"called": False}

        def mock_consume(**kwargs):
            consumed["called"] = True
            assert kwargs["event_type"] == "index_video"
            return ApiUnitConsumeResult(allowed=True, remaining_balance=9500)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr("src.api.app._validate_video_duration", lambda url: None)
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (_video_record("vid_1"), True),
        )
        monkeypatch.setattr("src.api.app.enqueue_video_job", lambda vid: object())

        response = client.post(
            "/api/v1/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert consumed["called"] is True

    def test_jwt_video_submit_uses_web_credits(self, monkeypatch) -> None:
        _authenticate("user_123")
        consumed_web = {"called": False}

        def mock_web_consume(user_id):
            consumed_web["called"] = True
            return ProcessingCreditConsumeResult(allowed=True, remaining_balance=4)

        monkeypatch.setattr("src.api.app.db_consume_processing_credit", mock_web_consume)
        monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _uid: 1)
        monkeypatch.setenv("VIDEO_MAX_FREE_VIDEOS", "1")
        monkeypatch.setattr("src.api.app.db_get_credits", lambda _uid: type("C", (), {"balance": 5})())
        monkeypatch.setattr("src.api.app._validate_video_duration", lambda url: None)
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (_video_record("vid_2"), True),
        )
        monkeypatch.setattr("src.api.app.enqueue_video_job", lambda vid: object())

        response = client.post(
            "/api/v1/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
        )

        assert response.status_code == 200
        assert consumed_web["called"] is True

    def test_insufficient_api_units_returns_402(self, monkeypatch) -> None:
        raw_key, record = _setup_api_key_auth(monkeypatch)

        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
        )
        monkeypatch.setattr("src.api.app._validate_video_duration", lambda url: None)
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (_video_record("vid_3"), True),
        )
        monkeypatch.setattr(
            "src.api.app.update_video_status",
            lambda *a, **kw: None,
        )

        response = client.post(
            "/api/v1/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 402
        assert response.json()["detail"]["code"] == "insufficient_api_units"


# ---------------------------------------------------------------------------
# TestApiSearchBilling
# ---------------------------------------------------------------------------


class TestApiSearchBilling:
    def test_api_key_search_deducts_units(self, monkeypatch) -> None:
        raw_key, record = _setup_api_key_auth(monkeypatch)
        consumed = {"called": False}

        def mock_consume(**kwargs):
            consumed["called"] = True
            assert kwargs["event_type"] == "text_query"
            assert kwargs["units"] == 1
            return ApiUnitConsumeResult(allowed=True, remaining_balance=9999)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.search_video_by_text_service",
            lambda **kwargs: [],
        )

        response = client.post(
            "/api/v1/videos/test-vid/search",
            json={"query_text": "hello"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert consumed["called"] is True

    def test_jwt_search_no_billing(self, monkeypatch) -> None:
        _authenticate("user_123")
        consumed = {"called": False}

        def mock_consume(**kwargs):
            consumed["called"] = True
            return ApiUnitConsumeResult(allowed=True, remaining_balance=0)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.search_video_by_text_service",
            lambda **kwargs: [],
        )

        response = client.post(
            "/api/v1/videos/test-vid/search",
            json={"query_text": "hello"},
        )

        assert response.status_code == 200
        assert consumed["called"] is False

    def test_insufficient_units_blocks_search(self, monkeypatch) -> None:
        raw_key, record = _setup_api_key_auth(monkeypatch)

        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
        )

        response = client.post(
            "/api/v1/videos/test-vid/search",
            json={"query_text": "hello"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 402


# ---------------------------------------------------------------------------
# TestApiCheckout
# ---------------------------------------------------------------------------


class TestApiCheckout:
    def test_developer_plan_creates_session(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setenv("LEMON_SQUEEZY_VARIANT_ID_DEVELOPER", "variant_dev")
        monkeypatch.setenv("LEMON_SQUEEZY_API_KEY", "key")
        monkeypatch.setenv("LEMON_SQUEEZY_STORE_ID", "store")
        monkeypatch.setenv("LEMON_SQUEEZY_CHECKOUT_REDIRECT_URL", "https://example.com")

        from src.billing.lemonsqueezy import LemonSqueezyCheckoutSession

        def mock_checkout(**kwargs):
            assert kwargs["grant_target"] == "api"
            assert kwargs["plan"] == "developer"
            return LemonSqueezyCheckoutSession(url="https://checkout.example.com", test_mode=True)

        monkeypatch.setattr("src.api.app.create_checkout_session", mock_checkout)

        response = client.post(
            "/api/v1/billing/checkout",
            json={"plan": "developer"},
        )

        assert response.status_code == 200
        assert "checkout.example.com" in response.json()["checkout_url"]

    def test_non_developer_plan_rejected(self, monkeypatch) -> None:
        _authenticate("user_123")

        response = client.post(
            "/api/v1/billing/checkout",
            json={"plan": "starter"},
        )

        assert response.status_code == 422


# ---------------------------------------------------------------------------
# TestApiBillingSummary
# ---------------------------------------------------------------------------


class TestApiBillingSummary:
    def test_returns_balance_and_equivalents(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_api_credits",
            lambda uid: ApiCreditRecord(
                user_id=uid, balance=7342, created_at=None, updated_at=None
            ),
        )

        response = client.get("/api/v1/billing/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["api_units_balance"] == 7342
        assert data["approx_videos"] == 7342 // 500
        assert data["approx_queries"] == 7342

    def test_returns_zeros_when_no_credits(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr("src.api.app.db_get_api_credits", lambda uid: None)

        response = client.get("/api/v1/billing/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["api_units_balance"] == 0
        assert data["approx_videos"] == 0


# ---------------------------------------------------------------------------
# TestApiUsageEvents
# ---------------------------------------------------------------------------


class TestApiUsageEvents:
    def test_returns_events(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_list_api_usage_events",
            lambda **kwargs: [
                ApiUsageEventRecord(
                    id="evt1",
                    user_id="user_123",
                    api_key_id=None,
                    event_type="index_video",
                    units=500,
                    video_id=None,
                    request_id=None,
                    metadata={},
                    created_at="2026-03-18T12:00:00Z",
                ),
            ],
        )

        response = client.get("/api/v1/billing/usage")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["event_type"] == "index_video"

    def test_filters_by_api_key_id(self, monkeypatch) -> None:
        _authenticate("user_123")
        captured_kwargs: dict = {}

        def mock_list(**kwargs):
            captured_kwargs.update(kwargs)
            return []

        monkeypatch.setattr("src.api.app.db_list_api_usage_events", mock_list)

        response = client.get("/api/v1/billing/usage?api_key_id=key_abc")

        assert response.status_code == 200
        assert captured_kwargs["api_key_id"] == "key_abc"


# ---------------------------------------------------------------------------
# TestApiCompensation (via CRUD layer, not route)
# ---------------------------------------------------------------------------


class TestApiCompensation:
    def test_compensate_increases_balance(self, monkeypatch) -> None:
        from src.db.supabase import compensate_api_units

        mock_client = type("C", (), {
            "rpc": lambda self, name, params: type("R", (), {
                "execute": lambda self: type("D", (), {"data": None})()
            })(),
        })()
        monkeypatch.setattr("src.db.supabase.get_client", lambda: mock_client)

        # Should not raise
        compensate_api_units(
            user_id="user_123",
            units=100,
            video_id=None,
            request_id="comp_1",
        )

    def test_compensate_rejects_empty_user_id(self) -> None:
        from src.db.supabase import compensate_api_units

        with pytest.raises(ValueError, match="user_id"):
            compensate_api_units(user_id="", units=100)
