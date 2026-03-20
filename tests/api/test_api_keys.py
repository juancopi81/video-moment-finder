"""Tests for API key management and authentication."""
from __future__ import annotations

from datetime import datetime, timezone
import io

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import AuthIdentity
from src.db.supabase import ApiKeyRecord, ProcessingCreditConsumeResult, VideoRecord
from src.video.download import VideoMetadata

from tests.api.conftest import (
    _authenticate,
    _make_api_key_record,
    _mock_api_key_auth,
    _setup_api_key_auth,
    _video_record,
)

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
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_apikey")
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
# Scope restriction: API keys rejected on legacy / internal routes
# ---------------------------------------------------------------------------


class TestApiKeyScopeRestriction:
    """API keys must not authenticate legacy @app routes."""

    @pytest.mark.parametrize("method,path,payload", [
        ("get", "/users/me/videos", None),
        ("get", "/users/me/billing-summary", None),
        ("post", "/billing/checkout", {"plan": "starter"}),
        ("post", "/videos", {"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"}),
    ])
    def test_api_key_rejected_on_legacy_route(self, monkeypatch, method, path, payload):
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_scope")

        kwargs: dict = {"headers": {"Authorization": f"Bearer {raw_key}"}}
        if payload is not None:
            kwargs["json"] = payload
        resp = getattr(client, method)(path, **kwargs)
        assert resp.status_code == 401
        assert "not accepted" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Auth identity attribution
# ---------------------------------------------------------------------------


class TestAuthIdentityAttribution:
    def test_verify_api_key_returns_identity_with_key_id(self, monkeypatch):
        """verify_api_key must return AuthIdentity with api_key_id for attribution."""
        from src.api.auth import verify_api_key

        raw_key, _ = _setup_api_key_auth(
            monkeypatch, user_id="user_attr", key_id="key-id-for-attribution",
        )

        identity = verify_api_key(raw_key)
        assert isinstance(identity, AuthIdentity)
        assert identity.user_id == "user_attr"
        assert identity.auth_method == "api_key"
        assert identity.api_key_id == "key-id-for-attribution"

    def test_jwt_auth_returns_identity_without_key_id(self, monkeypatch):
        """JWT path must return AuthIdentity with auth_method='jwt' and no api_key_id."""
        from src.api.auth import _verify_token

        monkeypatch.setattr(
            "src.api.auth.verify_bearer_token",
            lambda token: "user_jwt",
        )

        identity = _verify_token("eyJhbGciOiJSUzI1NiJ9.fake.token")
        assert isinstance(identity, AuthIdentity)
        assert identity.user_id == "user_jwt"
        assert identity.auth_method == "jwt"
        assert identity.api_key_id is None


# ---------------------------------------------------------------------------
# Quota enforcement
# ---------------------------------------------------------------------------


class TestApiKeyQuota:
    def test_api_key_auth_triggers_credit_check(self, monkeypatch):
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_quota")

        # Simulate credit check failure (free tier, no credits).
        monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: False)
        monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _uid: 1)
        monkeypatch.setattr("src.api.app.db_get_credits", lambda _uid: None)
        monkeypatch.setattr(
            "src.api.app.db_consume_processing_credit",
            lambda _uid: ProcessingCreditConsumeResult(allowed=False, remaining_balance=0),
        )

        # RPC inserts successfully, then billing fails.
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (_video_record("quota-vid-id", status="queued"), True),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        monkeypatch.setattr("src.api.app.update_video_status", lambda vid, status, error_message=None: None)

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 402


# ---------------------------------------------------------------------------
# Retry-safe idempotency (no double billing)
# ---------------------------------------------------------------------------


class TestRetryIdempotency:
    def test_youtube_retry_returns_existing_without_credit(self, monkeypatch):
        """Retrying a YouTube video submit must return the existing record
        without consuming another credit (atomic RPC-level dedup)."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_retry")

        existing = _video_record("existing-vid-id", status="queued")

        # RPC returns (existing_record, was_created=False) — simulates a retry
        # where the first request already inserted the row.
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (existing, False),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        # If credit consumption were called, it would fail — proving no double billing.
        monkeypatch.setattr(
            "src.api.app.db_consume_processing_credit",
            lambda _uid: ProcessingCreditConsumeResult(allowed=False, remaining_balance=0),
        )
        # Job exists — normal retry, no re-enqueue needed.
        monkeypatch.setattr("src.api.app.db_get_video_job", lambda vid: object())

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "existing-vid-id"

    def test_youtube_new_url_creates_and_bills(self, monkeypatch):
        """A new YouTube URL inserts via RPC and consumes credit."""
        from src.db.supabase import ApiUnitConsumeResult
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_new")

        created_video = _video_record("new-vid-id", status="queued")

        # RPC returns (new_record, was_created=True)
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (created_video, True),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        monkeypatch.setattr("src.api.app.enqueue_video_job", lambda vid: None)
        monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: True)
        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=9500),
        )

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "new-vid-id"

    def test_youtube_billing_failure_marks_video_failed(self, monkeypatch):
        """If billing fails after atomic insert, the video is marked failed."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_fail")

        created_video = _video_record("fail-vid-id", status="queued")
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (created_video, True),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: False)
        monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _uid: 1)
        monkeypatch.setattr("src.api.app.db_get_credits", lambda _uid: None)
        monkeypatch.setattr(
            "src.api.app.db_consume_processing_credit",
            lambda _uid: ProcessingCreditConsumeResult(allowed=False, remaining_balance=0),
        )

        status_updates: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "src.api.app.update_video_status",
            lambda vid, status, error_message=None: status_updates.append((vid, status)),
        )

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 402
        assert ("fail-vid-id", "failed") in status_updates

    def test_youtube_enqueue_failure_preserves_dedupe_anchor(self, monkeypatch):
        """Enqueue failure must NOT mark the row failed, so retries still find
        the dedupe anchor and don't consume another credit."""
        from src.db.supabase import ApiUnitConsumeResult
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_enq_fail")

        created_video = _video_record("enq-fail-vid", status="queued")
        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (created_video, True),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _uid: True)
        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=9500),
        )

        # Enqueue fails — should NOT mark row failed.
        monkeypatch.setattr(
            "src.api.app.enqueue_video_job",
            lambda vid: (_ for _ in ()).throw(RuntimeError("queue down")),
        )

        status_updates: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "src.api.app.update_video_status",
            lambda vid, status, error_message=None: status_updates.append((vid, status)),
        )

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 500
        # Critical: the row must NOT be marked failed.
        assert ("enq-fail-vid", "failed") not in status_updates

    def test_youtube_retry_re_enqueues_stranded_video(self, monkeypatch):
        """A retry that finds a queued video with no job row must re-enqueue it."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_strand")

        stranded = _video_record("stranded-vid", status="queued")

        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (stranded, False),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        # No existing job row — the original enqueue failed.
        monkeypatch.setattr("src.api.app.db_get_video_job", lambda vid: None)

        enqueued: list[str] = []
        monkeypatch.setattr("src.api.app.enqueue_video_job", lambda vid: enqueued.append(vid))

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == "stranded-vid"
        assert "stranded-vid" in enqueued

    def test_youtube_retry_returns_503_when_re_enqueue_fails(self, monkeypatch):
        """If a retry finds a stranded video and re-enqueue also fails, return 503."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_strand2")

        stranded = _video_record("stranded-vid-2", status="queued")

        monkeypatch.setattr(
            "src.api.app.db_insert_youtube_video_idempotent",
            lambda url, uid: (stranded, False),
        )
        monkeypatch.setattr(
            "src.api.app.fetch_video_metadata",
            lambda _: VideoMetadata(duration_s=60.0, is_live=False),
        )
        monkeypatch.setattr("src.api.app.db_get_video_job", lambda vid: None)
        monkeypatch.setattr(
            "src.api.app.enqueue_video_job",
            lambda vid: (_ for _ in ()).throw(RuntimeError("queue down")),
        )

        resp = client.post(
            f"{V1}/videos",
            json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert resp.status_code == 503

    def test_upload_idempotency_key_returns_existing(self, monkeypatch):
        """Upload retry with same Idempotency-Key must return existing record
        without doing R2 upload or billing (fast-path early exit)."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_upload_retry")

        from uuid import uuid5, NAMESPACE_URL

        expected_vid_id = str(uuid5(NAMESPACE_URL, "user_upload_retry:upload-key-1"))
        existing = VideoRecord(
            id=expected_vid_id,
            youtube_url=None,
            status="queued",  # type: ignore[arg-type]
            user_id="user_upload_retry",
            error_message=None,
            source_type="upload",
            source_r2_key=f"source/{expected_vid_id}/test.mp4",
            source_filename="test.mp4",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Fast-path: db_get_video returns existing record before any heavy work.
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: existing if vid == expected_vid_id else None,
        )
        # Job exists — normal retry, no re-enqueue needed.
        monkeypatch.setattr("src.api.app.db_get_video_job", lambda vid: object())

        fake_file = io.BytesIO(b"fake video data")
        resp = client.post(
            f"{V1}/videos/upload",
            files={"file": ("test.mp4", fake_file, "video/mp4")},
            headers={
                "Authorization": f"Bearer {raw_key}",
                "Idempotency-Key": "upload-key-1",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == expected_vid_id

    def test_upload_insert_race_does_not_delete_shared_source_key(self, monkeypatch):
        """Loser-side cleanup must not delete the winner's shared R2 object."""
        raw_key, _ = _setup_api_key_auth(monkeypatch, user_id="user_upload_race")

        from uuid import NAMESPACE_URL, uuid5

        video_id = str(uuid5(NAMESPACE_URL, "user_upload_race:upload-key-race"))
        source_r2_key = f"source/{video_id}/upload.mp4"
        existing = VideoRecord(
            id=video_id,
            youtube_url=None,
            status="queued",  # type: ignore[arg-type]
            user_id="user_upload_race",
            error_message=None,
            source_type="upload",
            source_r2_key=source_r2_key,
            source_filename="upload.mp4",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        cleanup_calls: list[str] = []

        class FakeR2Store:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def upload_source_video(self, *args, **kwargs):
                class Result:
                    key = source_r2_key

                return Result()

            def delete_source_object(self, key: str) -> None:
                cleanup_calls.append(key)

        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeR2Store)
        monkeypatch.setattr("src.api.app.db_get_video", lambda video_id, user_id=None: None)
        monkeypatch.setattr(
            "src.api.app.db_insert_uploaded_video_idempotent",
            lambda *args, **kwargs: (existing, False),
        )
        monkeypatch.setattr("src.api.app.db_get_video_job", lambda vid: object())

        resp = client.post(
            f"{V1}/videos/upload",
            files={"file": ("upload.mp4", io.BytesIO(b"fake video data"), "video/mp4")},
            headers={
                "Authorization": f"Bearer {raw_key}",
                "Idempotency-Key": "upload-key-race",
            },
        )
        assert resp.status_code == 200
        assert resp.json()["id"] == video_id
        assert cleanup_calls == []
