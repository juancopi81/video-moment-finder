"""Tests for API key management and authentication."""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.db.supabase import ApiKeyRecord, ProcessingCreditConsumeResult
from src.video.download import VideoMetadata

from tests.api.conftest import _authenticate, _make_api_key_record, _video_record

client = TestClient(app)

V1 = "/api/v1"


# ---------------------------------------------------------------------------
# Key management (ownership)
# ---------------------------------------------------------------------------


class TestCreateApiKey:
    def test_returns_raw_key_and_prefix(self, monkeypatch):
        _authenticate()
        created_records: list[ApiKeyRecord] = []

        def fake_create(user_id, name, key_hash, key_prefix):
            record = ApiKeyRecord(
                id="new-key-id",
                user_id=user_id,
                name=name,
                key_hash=key_hash,
                key_prefix=key_prefix,
            )
            created_records.append(record)
            return record

        monkeypatch.setattr("src.api.app.db_create_api_key", fake_create)

        resp = client.post(f"{V1}/keys", json={"name": "my-key"})
        assert resp.status_code == 201
        body = resp.json()
        assert body["key"].startswith("vmf_")
        assert len(body["key"]) == 4 + 32  # vmf_ + 32 hex chars
        assert body["key_prefix"].startswith("vmf_")
        assert body["name"] == "my-key"
        assert body["id"] == "new-key-id"

    def test_create_key_requires_auth(self):
        resp = client.post(f"{V1}/keys", json={})
        assert resp.status_code == 401

    def test_create_key_cap_exceeded(self, monkeypatch):
        _authenticate()

        def fake_create(_user_id, _name, _key_hash, _key_prefix):
            raise ValueError("Maximum of 10 active API keys per user")

        monkeypatch.setattr("src.api.app.db_create_api_key", fake_create)

        resp = client.post(f"{V1}/keys", json={})
        assert resp.status_code == 422


class TestListApiKeys:
    def test_lists_own_keys(self, monkeypatch):
        _authenticate("user_A")
        _, record = _make_api_key_record(user_id="user_A", name="k1")

        monkeypatch.setattr("src.api.app.db_list_api_keys", lambda uid: [record])

        resp = client.get(f"{V1}/keys")
        assert resp.status_code == 200
        keys = resp.json()
        assert len(keys) == 1
        assert keys[0]["name"] == "k1"
        # Raw key must never appear in list responses.
        assert "key" not in keys[0] or keys[0].get("key") is None

    def test_list_requires_auth(self):
        resp = client.get(f"{V1}/keys")
        assert resp.status_code == 401


class TestRevokeApiKey:
    def test_revoke_own_key(self, monkeypatch):
        _authenticate("user_A")
        monkeypatch.setattr("src.api.app.db_revoke_api_key", lambda kid, uid: True)

        resp = client.delete(f"{V1}/keys/some-key-id")
        assert resp.status_code == 204

    def test_revoke_other_users_key_returns_404(self, monkeypatch):
        _authenticate("user_A")
        monkeypatch.setattr("src.api.app.db_revoke_api_key", lambda kid, uid: False)

        resp = client.delete(f"{V1}/keys/other-key-id")
        assert resp.status_code == 404

    def test_revoke_requires_auth(self):
        resp = client.delete(f"{V1}/keys/some-key-id")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Auth lifecycle
# ---------------------------------------------------------------------------


class TestApiKeyAuth:
    def test_api_key_authenticates_v1_routes(self, monkeypatch):
        raw_key, record = _make_api_key_record(user_id="user_apikey")

        monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda h: record)
        monkeypatch.setattr("src.api.auth.touch_api_key_last_used", lambda kid: None)
        monkeypatch.setattr("src.api.app.db_list_videos", lambda user_id: [])

        resp = client.get(
            f"{V1}/videos",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200

    def test_revoked_key_returns_401(self, monkeypatch):
        raw_key, _ = _make_api_key_record()

        monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda h: None)

        resp = client.get(
            f"{V1}/videos",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------


class TestApiKeyQuota:
    def test_api_key_auth_triggers_credit_check(self, monkeypatch):
        raw_key, record = _make_api_key_record(user_id="user_quota")

        monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda h: record)
        monkeypatch.setattr("src.api.auth.touch_api_key_last_used", lambda kid: None)

        # Simulate credit check failure (free tier, no credits).
        monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: False)
        monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _uid: 1)
        monkeypatch.setattr("src.api.app.db_get_credits", lambda _uid: None)
        monkeypatch.setattr(
            "src.api.app.db_consume_processing_credit",
            lambda _uid: ProcessingCreditConsumeResult(allowed=False, remaining_balance=0),
        )

        # Submitting a video should fail with credit gate.
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 402
