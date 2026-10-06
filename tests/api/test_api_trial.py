"""Trial integration: shared indexing balance, legacy compatibility, consent."""
from __future__ import annotations

from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

from fastapi.testclient import TestClient

from src.api.app import app
from src.billing.trial import TrialState
from src.db.supabase import ApiCreditRecord, ApiTrialGrantRecord, ProcessingCreditConsumeResult
from tests.api.conftest import _authenticate, _video_record, _upload_video_record, UPLOAD_VIDEO_ID

client = TestClient(app)


def _enrolled(monkeypatch, *, enabled=True, api_balance=600, offset=0):
    grant = ApiTrialGrantRecord("user_123", 600, offset, 600 - offset, "email_1")
    state = TrialState(enabled, "granted", 600, grant)
    monkeypatch.setattr("src.api.app.ensure_trial_state", lambda _: state)
    monkeypatch.setattr("src.api.app.existing_trial_state", lambda _: state)
    monkeypatch.setattr("src.api.app.db_get_api_credits", lambda _: ApiCreditRecord("user_123", api_balance))
    return state


def test_summary_shows_shared_balance_and_no_second_free_video(monkeypatch):
    _authenticate()
    _enrolled(monkeypatch, api_balance=100, offset=500)
    response = client.get("/api/v1/billing/credits/summary")
    assert response.status_code == 200
    data = response.json()
    assert (data["free_videos_limit"], data["free_videos_remaining"]) == (0, 0)
    assert data["trial_units_granted"] == 100
    assert data["trial_legacy_units_offset"] == 500
    assert data["api_units_balance"] == 100


def test_paused_grants_do_not_restore_legacy_free_allowance(monkeypatch):
    _authenticate()
    _enrolled(monkeypatch, enabled=False, api_balance=0)
    response = client.post("/api/v1/videos/upload/init", json={"filename": "lecture.mp4", "content_type": "video/mp4"})
    assert response.status_code == 402
    data = client.get("/api/v1/billing/credits/summary").json()
    assert data["trial_enabled"] is False
    assert data["trial_status"] == "exhausted"
    assert data["free_videos_remaining"] == 0


def test_enrolled_website_indexing_uses_atomic_shared_charge(monkeypatch):
    _authenticate()
    _enrolled(monkeypatch)
    monkeypatch.setattr("src.api.app._validate_video_duration", lambda _: None)
    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", lambda *args: (_video_record("vid_1"), True))
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _: object())
    debit = Mock(return_value=ProcessingCreditConsumeResult(True, 100))
    monkeypatch.setattr("src.api.app.db_consume_trial_or_processing_credit", debit)
    legacy = Mock(side_effect=AssertionError("must not charge both ledgers"))
    monkeypatch.setattr("src.api.app.db_consume_processing_credit", legacy)
    response = client.post("/api/v1/videos", json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"})
    assert response.status_code == 200
    debit.assert_called_once_with(user_id="user_123", video_id="vid_1", units=500)
    legacy.assert_not_called()


def test_enrollment_happens_before_new_video_is_inserted(monkeypatch):
    _authenticate()
    calls = []
    state = _enrolled(monkeypatch)
    monkeypatch.setattr("src.api.app.ensure_trial_state", lambda _: (calls.append("trial"), state)[1])
    monkeypatch.setattr("src.api.app._validate_video_duration", lambda _: None)
    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", lambda *args: (calls.append("insert"), (_video_record("vid_1"), False))[1])
    monkeypatch.setattr("src.api.app._ensure_enqueued", lambda _: None)
    response = client.post("/api/v1/videos", json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"})
    assert response.status_code == 200
    assert calls.index("trial") < calls.index("insert")
    assert calls == ["trial", "insert"]


def test_unverified_account_keeps_legacy_website_allowance(monkeypatch):
    _authenticate()
    monkeypatch.setattr("src.api.app.ensure_trial_state", lambda _: TrialState(True, "verification_required", 600))
    data = client.get("/api/v1/billing/credits/summary").json()
    assert data["free_videos_remaining"] == 1
    assert data["trial_status"] == "verification_required"
    assert data["trial_units_granted"] == 0


def test_enrolled_unlimited_account_keeps_free_website_uploads(monkeypatch):
    _authenticate()
    _enrolled(monkeypatch)
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _: True)
    monkeypatch.setattr("src.api.app._validate_and_upload_file", lambda *args: (
        Mock(), type("Upload", (), {"key": "video.mp4"})(), "video.mp4", False, 10,
    ))
    monkeypatch.setattr("src.api.app.db_create_uploaded_video", lambda **kwargs: _video_record("vid_1"))
    monkeypatch.setattr("src.api.app.enqueue_video_job", lambda _: object())
    debit = Mock(side_effect=AssertionError("Unlimited website entitlement is preserved"))
    monkeypatch.setattr("src.api.app.db_consume_trial_or_processing_credit", debit)
    response = client.post("/api/v1/videos/upload", files={"file": ("video.mp4", b"video", "video/mp4")})
    assert response.status_code == 200
    debit.assert_not_called()


def test_insufficient_units_response_is_neutral(monkeypatch):
    from src.api.app import _consume_api_units_or_raise
    from fastapi import HTTPException
    import pytest
    _enrolled(monkeypatch, api_balance=0)
    with pytest.raises(HTTPException) as caught:
        _consume_api_units_or_raise("user_123", None, "text_query", 1)
    assert caught.value.status_code == 402
    detail = caught.value.detail
    assert detail["code"] == "insufficient_api_units"
    assert not any(word in detail["message"].lower() for word in ("buy", "purchase", "upgrade", "checkout"))


def test_summary_then_two_connector_approvals_share_one_account_grant(monkeypatch):
    _authenticate()
    monkeypatch.setenv("API_TRIAL_ENABLED", "true")
    stored = {}
    monkeypatch.setattr("src.billing.trial.get_api_trial_grant", lambda uid: stored.get(uid))
    verification = Mock(return_value="server_verified_email")
    monkeypatch.setattr("src.billing.trial.verified_primary_email_id", verification)
    def grant(**kwargs):
        stored[kwargs["user_id"]] = ApiTrialGrantRecord("user_123", 600, 0, 600, kwargs["verified_email_id"])
        return stored[kwargs["user_id"]]
    grant_call = Mock(side_effect=grant)
    monkeypatch.setattr("src.billing.trial.apply_api_trial_grant", grant_call)
    monkeypatch.setattr("src.api.app.db_get_api_credits", lambda _: ApiCreditRecord("user_123", 600) if stored else None)
    provider = Mock()
    provider.approve_authorization_request.return_value = "https://example.com/callback?code=example"
    monkeypatch.setattr("src.api.app._mcp_oauth_provider_or_raise", lambda: provider)

    assert client.get("/api/v1/billing/units/summary").json()["trial_units_granted"] == 600
    for request_id in ("first_connection", "reconnect"):
        assert client.post(f"/oauth/mcp/requests/{request_id}/approve").status_code == 200
    grant_call.assert_called_once()
    verification.assert_called_once_with("user_123")
    assert provider.approve_authorization_request.call_count == 2


def test_racing_retry_cannot_enqueue_before_trial_debit_commits(monkeypatch):
    _authenticate()
    _enrolled(monkeypatch)
    monkeypatch.setattr("src.api.app._validate_video_duration", lambda _: None)
    monkeypatch.setattr("src.api.app.db_get_video_job", lambda _: None)
    monkeypatch.setattr("src.api.app.update_video_status", lambda *args, **kwargs: None)
    record = _video_record("pending_video")
    inserts = []
    def insert(*args):
        inserts.append(1)
        return record, len(inserts) == 1
    monkeypatch.setattr("src.api.app.db_insert_youtube_video_idempotent", insert)
    billing_started, release_billing = Event(), Event()
    def debit(**kwargs):
        billing_started.set()
        assert release_billing.wait(5)
        return ProcessingCreditConsumeResult(False, 100)
    monkeypatch.setattr("src.api.app.db_consume_trial_or_processing_credit", debit)
    enqueue = Mock()
    monkeypatch.setattr("src.api.app.enqueue_video_job", enqueue)
    body = {"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"}
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(lambda: TestClient(app).post("/api/v1/videos", json=body))
        try:
            assert billing_started.wait(5)
            assert client.post("/api/v1/videos", json=body).status_code == 409
            enqueue.assert_not_called()
        finally:
            release_billing.set()
        assert first.result(timeout=5).status_code == 402
    enqueue.assert_not_called()


def test_paid_failed_upload_retry_does_not_require_another_500_units(monkeypatch):
    from src.api.app import _complete_upload_core, FAILED_ERROR_ENQUEUE
    _enrolled(monkeypatch, api_balance=100)
    queued = _upload_video_record(UPLOAD_VIDEO_ID, status="queued")
    failed = replace(queued, status="failed", error_message=FAILED_ERROR_ENQUEUE)
    monkeypatch.setattr("src.api.app._get_idempotent_upload_record", lambda **kwargs: failed)
    monkeypatch.setattr("src.api.app._should_retry_failed_upload", lambda _: True)
    monkeypatch.setattr("src.api.app.db_has_video_processing_charge", lambda *_: True)
    monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
    store = Mock()
    store.source_exists.return_value = True
    monkeypatch.setattr("src.api.app.R2Store", lambda _: store)
    monkeypatch.setattr("src.api.app._reset_upload_retry_record", lambda _: queued)
    enqueue = Mock()
    monkeypatch.setattr("src.api.app._enqueue_video_or_fail", enqueue)
    admission = Mock(side_effect=AssertionError("A committed charge needs no fresh balance"))
    bill = Mock(side_effect=AssertionError("A retry must not debit twice"))
    result = _complete_upload_core(UPLOAD_VIDEO_ID, "upload.mp4", "user_123", bill, admit=admission)
    assert result.status == "queued"
    admission.assert_not_called()
    bill.assert_not_called()
    enqueue.assert_called_once_with(UPLOAD_VIDEO_ID)
