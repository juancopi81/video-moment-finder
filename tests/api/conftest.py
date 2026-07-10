"""Shared fixtures and helpers for API tests."""
from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from datetime import datetime, timezone

import pytest

os.environ.setdefault("MCP_OAUTH_ISSUER_URL", "https://api.videomomentfinder.com")
os.environ.setdefault("MCP_OAUTH_RESOURCE_URL", "https://api.videomomentfinder.com/mcp")
os.environ.setdefault("MCP_OAUTH_CLIENT_ID", "claude-static-client")
os.environ.setdefault("MCP_OAUTH_CLIENT_SECRET", "claude-static-secret")
os.environ.setdefault("FRONTEND_BASE_URL", "https://www.videomomentfinder.com")
os.environ.setdefault(
    "CORS_ALLOWED_ORIGINS",
    ",".join(
        [
            "http://localhost:3000",
            "http://localhost:6274",
            "http://127.0.0.1:6274",
            "https://claude.ai",
            "https://claude.com",
        ]
    ),
)

from src.api.app import app
from src.api.auth import AuthIdentity, get_current_user, get_current_user_id
from src.db.supabase import (
    McpOAuthAccessTokenRecord,
    McpOAuthAuthorizationCodeRecord,
    McpOAuthClientRecord,
    McpOAuthAuthorizationRequestRecord,
    McpOAuthRefreshTokenRecord,
)
from src.api.rate_limit import SlidingWindowRateLimiter
from src.db.supabase import (
    ApiKeyRecord,
    ApiUnitConsumeResult,
    ProcessingCreditConsumeResult,
    VideoRecord,
)

UPLOAD_VIDEO_ID = "00000000-0000-4000-8000-000000000123"


def _video_record(
    video_id: str, *, status: str = "queued", duration_s: float | None = None,
) -> VideoRecord:
    return VideoRecord(
        id=video_id,
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status=status,  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="youtube",
        source_r2_key=None,
        source_filename=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        duration_s=duration_s,
    )


def _upload_video_record(
    video_id: str, *, status: str = "ready", duration_s: float | None = None,
) -> VideoRecord:
    return VideoRecord(
        id=video_id,
        youtube_url=None,
        status=status,  # type: ignore[arg-type]
        user_id="user_123",
        error_message=None,
        source_type="upload",
        source_r2_key="source/video_123/upload.mp4",
        source_filename="upload.mp4",
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
        duration_s=duration_s,
    )


def _authenticate(user_id: str = "user_123") -> None:
    app.dependency_overrides[get_current_user_id] = lambda: user_id
    app.dependency_overrides[get_current_user] = lambda: AuthIdentity(
        user_id=user_id, auth_method="jwt"
    )


@pytest.fixture(autouse=True)
def _mock_mcp_oauth_env(monkeypatch) -> None:
    monkeypatch.setenv("MCP_OAUTH_ISSUER_URL", "https://api.videomomentfinder.com")
    monkeypatch.setenv("MCP_OAUTH_RESOURCE_URL", "https://api.videomomentfinder.com/mcp")
    monkeypatch.setenv("MCP_OAUTH_CLIENT_ID", "claude-static-client")
    monkeypatch.setenv("MCP_OAUTH_CLIENT_SECRET", "claude-static-secret")
    monkeypatch.setenv("FRONTEND_BASE_URL", "https://www.videomomentfinder.com")


@pytest.fixture(autouse=True)
def _clear_dependency_overrides(monkeypatch) -> None:
    app.dependency_overrides.clear()
    monkeypatch.setattr("src.api.app.USER_WRITE_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.SEARCH_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.WEBHOOK_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.OAUTH_RATE_LIMITER", SlidingWindowRateLimiter())
    monkeypatch.setattr("src.api.app.track", lambda *args, **kwargs: None)
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _mock_free_video_count(monkeypatch) -> None:
    monkeypatch.setattr("src.api.app.db_has_unlimited_video_access", lambda _user_id: False)
    monkeypatch.setattr("src.api.app.db_count_videos_for_user", lambda _user_id: 0)
    monkeypatch.setattr("src.api.app.db_get_credits", lambda _user_id: None)
    monkeypatch.setattr(
        "src.api.app.db_consume_processing_credit",
        lambda _user_id: ProcessingCreditConsumeResult(
            allowed=False,
            remaining_balance=0,
        ),
    )


@pytest.fixture(autouse=True)
def _mock_api_billing(monkeypatch) -> None:
    monkeypatch.setattr("src.api.app.db_get_api_credits", lambda _uid: None)
    monkeypatch.setattr(
        "src.api.app.db_consume_api_units",
        lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
    )
    monkeypatch.setattr(
        "src.api.app.db_list_api_usage_events",
        lambda **kwargs: [],
    )


@pytest.fixture(autouse=True)
def _mock_upload_duration_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.api.app._validate_upload_file_duration_or_raise",
        lambda file: None,
    )
    monkeypatch.setattr(
        "src.api.app._validate_uploaded_source_duration_with_cleanup",
        lambda store, key, user_id: None,
    )


def _make_api_key_record(
    user_id: str = "user_123",
    *,
    raw_key: str = "vmf_deadbeef12345678deadbeef12345678",
    name: str = "test-key",
    key_id: str = "00000000-0000-4000-8000-aaaaaaaaaaaa",
) -> tuple[str, ApiKeyRecord]:
    """Return (raw_key, ApiKeyRecord) pair for testing API key auth."""
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    key_prefix = "vmf_" + raw_key[4:8]
    record = ApiKeyRecord(
        id=key_id,
        user_id=user_id,
        name=name,
        key_hash=key_hash,
        key_prefix=key_prefix,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    return raw_key, record


def _mock_api_key_auth(monkeypatch, record: ApiKeyRecord) -> None:
    """Monkeypatch API key DB lookups so the given record is found on auth."""
    monkeypatch.setattr("src.api.auth.get_api_key_by_hash", lambda h: record)
    monkeypatch.setattr("src.api.auth.touch_api_key_last_used", lambda kid: None)


def _setup_api_key_auth(
    monkeypatch, user_id: str = "user_123", **kwargs,
) -> tuple[str, ApiKeyRecord]:
    """Create an API key record and wire up auth mocks in one call."""
    raw_key, record = _make_api_key_record(user_id=user_id, **kwargs)
    _mock_api_key_auth(monkeypatch, record)
    return raw_key, record


class InMemoryMcpOAuthStore:
    def __init__(self) -> None:
        self._counter = 0
        self.clients: dict[str, McpOAuthClientRecord] = {}
        self.requests: dict[str, McpOAuthAuthorizationRequestRecord] = {}
        self.codes: dict[str, McpOAuthAuthorizationCodeRecord] = {}
        self.access_tokens: dict[str, McpOAuthAccessTokenRecord] = {}
        self.refresh_tokens: dict[str, McpOAuthRefreshTokenRecord] = {}

    def _next_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create_client(
        self,
        *,
        client_id: str,
        client_secret: str | None,
        client_id_issued_at: int | None,
        client_secret_expires_at: int | None,
        redirect_uris: list[str],
        token_endpoint_auth_method: str,
        grant_types: list[str],
        response_types: list[str],
        scope: str | None = None,
        client_name: str | None = None,
        client_uri: str | None = None,
        logo_uri: str | None = None,
        contacts: list[str] | None = None,
        tos_uri: str | None = None,
        policy_uri: str | None = None,
        jwks_uri: str | None = None,
        jwks=None,
        software_id: str | None = None,
        software_version: str | None = None,
    ) -> McpOAuthClientRecord:
        record = McpOAuthClientRecord(
            client_id=client_id,
            client_secret=client_secret,
            client_id_issued_at=client_id_issued_at,
            client_secret_expires_at=client_secret_expires_at,
            redirect_uris=redirect_uris,
            token_endpoint_auth_method=token_endpoint_auth_method,
            grant_types=grant_types,
            response_types=response_types,
            scope=scope,
            client_name=client_name,
            client_uri=client_uri,
            logo_uri=logo_uri,
            contacts=contacts,
            tos_uri=tos_uri,
            policy_uri=policy_uri,
            jwks_uri=jwks_uri,
            jwks=jwks,
            software_id=software_id,
            software_version=software_version,
            created_at=self._now(),
        )
        self.clients[client_id] = record
        return record

    def get_client(self, client_id: str) -> McpOAuthClientRecord | None:
        return self.clients.get(client_id)

    def create_request(
        self,
        *,
        client_id: str,
        redirect_uri: str,
        redirect_uri_provided_explicitly: bool,
        state: str | None,
        scopes: list[str],
        code_challenge: str,
        resource: str,
        expires_at: str,
    ) -> McpOAuthAuthorizationRequestRecord:
        now = self._now()
        record = McpOAuthAuthorizationRequestRecord(
            id=self._next_id("req"),
            client_id=client_id,
            redirect_uri=redirect_uri,
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            state=state,
            scopes=scopes,
            code_challenge=code_challenge,
            resource=resource,
            status="pending",
            expires_at=expires_at,
            created_at=now,
            updated_at=now,
        )
        self.requests[record.id] = record
        return record

    def get_request(self, request_id: str) -> McpOAuthAuthorizationRequestRecord | None:
        return self.requests.get(request_id)

    def update_request_resolution(
        self,
        request_id: str,
        *,
        status: str,
        user_id: str | None = None,
    ) -> McpOAuthAuthorizationRequestRecord | None:
        record = self.requests.get(request_id)
        if record is None or record.status != "pending":
            return None
        now = self._now()
        updated = replace(
            record,
            status=status,
            user_id=user_id if user_id is not None else record.user_id,
            approved_at=now if status == "approved" else record.approved_at,
            denied_at=now if status == "denied" else record.denied_at,
            resolved_at=now,
            updated_at=now,
        )
        self.requests[request_id] = updated
        return updated

    def create_code(
        self,
        *,
        authorization_request_id: str | None,
        user_id: str,
        client_id: str,
        code_hash: str,
        redirect_uri: str,
        redirect_uri_provided_explicitly: bool,
        scopes: list[str],
        code_challenge: str,
        resource: str,
        approved_tools_version: int,
        expires_at: str,
    ) -> McpOAuthAuthorizationCodeRecord:
        record = McpOAuthAuthorizationCodeRecord(
            id=self._next_id("code"),
            authorization_request_id=authorization_request_id,
            user_id=user_id,
            client_id=client_id,
            code_hash=code_hash,
            redirect_uri=redirect_uri,
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            scopes=scopes,
            code_challenge=code_challenge,
            resource=resource,
            approved_tools_version=approved_tools_version,
            expires_at=expires_at,
            created_at=self._now(),
        )
        self.codes[record.id] = record
        return record

    def get_code_by_hash(self, code_hash: str) -> McpOAuthAuthorizationCodeRecord | None:
        for record in self.codes.values():
            if (
                record.code_hash == code_hash
                and record.used_at is None
                and record.revoked_at is None
            ):
                return record
        return None

    def mark_code_used(self, code_id: str) -> None:
        record = self.codes[code_id]
        self.codes[code_id] = replace(record, used_at=self._now())

    def consume_code(self, code_id: str) -> bool:
        record = self.codes.get(code_id)
        if record is None or record.used_at is not None or record.revoked_at is not None:
            return False
        self.codes[code_id] = replace(record, used_at=self._now())
        return True

    def create_tokens(
        self,
        *,
        connection_id: str,
        user_id: str,
        client_id: str,
        access_token_hash: str,
        refresh_token_hash: str,
        scopes: list[str],
        resource: str,
        approved_tools_version: int,
        access_expires_at: str,
        refresh_expires_at: str,
    ) -> tuple[McpOAuthAccessTokenRecord, McpOAuthRefreshTokenRecord]:
        now = self._now()
        access_record = McpOAuthAccessTokenRecord(
            id=self._next_id("access"),
            connection_id=connection_id,
            user_id=user_id,
            client_id=client_id,
            token_hash=access_token_hash,
            scopes=scopes,
            resource=resource,
            approved_tools_version=approved_tools_version,
            expires_at=access_expires_at,
            created_at=now,
        )
        refresh_record = McpOAuthRefreshTokenRecord(
            id=self._next_id("refresh"),
            connection_id=connection_id,
            user_id=user_id,
            client_id=client_id,
            token_hash=refresh_token_hash,
            scopes=scopes,
            resource=resource,
            approved_tools_version=approved_tools_version,
            expires_at=refresh_expires_at,
            created_at=now,
        )
        self.access_tokens[access_record.id] = access_record
        self.refresh_tokens[refresh_record.id] = refresh_record
        return access_record, refresh_record

    def get_access_by_hash(self, token_hash: str) -> McpOAuthAccessTokenRecord | None:
        for record in self.access_tokens.values():
            if record.token_hash == token_hash and record.revoked_at is None:
                return record
        return None

    def get_refresh_by_hash(self, token_hash: str) -> McpOAuthRefreshTokenRecord | None:
        for record in self.refresh_tokens.values():
            if record.token_hash == token_hash and record.revoked_at is None:
                return record
        return None

    def revoke_access_tokens_for_connection(self, connection_id: str) -> None:
        now = self._now()
        for token_id, record in list(self.access_tokens.items()):
            if record.connection_id == connection_id and record.revoked_at is None:
                self.access_tokens[token_id] = replace(record, revoked_at=now)

    def revoke_refresh_token(self, refresh_token_id: str) -> None:
        record = self.refresh_tokens[refresh_token_id]
        if record.revoked_at is None:
            self.refresh_tokens[refresh_token_id] = replace(
                record, revoked_at=self._now()
            )

    def revoke_tokens_by_connection_id(self, connection_id: str) -> None:
        self.revoke_access_tokens_for_connection(connection_id)
        now = self._now()
        for token_id, record in list(self.refresh_tokens.items()):
            if record.connection_id == connection_id and record.revoked_at is None:
                self.refresh_tokens[token_id] = replace(record, revoked_at=now)


@pytest.fixture
def mcp_oauth_store(monkeypatch) -> InMemoryMcpOAuthStore:
    store = InMemoryMcpOAuthStore()
    monkeypatch.setattr(
        "src.api.mcp_oauth.create_mcp_oauth_client",
        store.create_client,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.get_mcp_oauth_client",
        store.get_client,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.create_mcp_oauth_authorization_request",
        store.create_request,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.delete_expired_mcp_oauth_authorization_requests",
        lambda _expires_before: None,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.get_mcp_oauth_authorization_request",
        store.get_request,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.update_mcp_oauth_authorization_request_resolution",
        store.update_request_resolution,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.create_mcp_oauth_authorization_code",
        store.create_code,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.get_mcp_oauth_authorization_code_by_hash",
        store.get_code_by_hash,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.consume_mcp_oauth_authorization_code",
        store.consume_code,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.create_mcp_oauth_tokens",
        store.create_tokens,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.get_mcp_oauth_access_token_by_hash",
        store.get_access_by_hash,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.get_mcp_oauth_refresh_token_by_hash",
        store.get_refresh_by_hash,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.revoke_mcp_oauth_access_tokens_for_connection",
        store.revoke_access_tokens_for_connection,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.revoke_mcp_oauth_refresh_token",
        store.revoke_refresh_token,
    )
    monkeypatch.setattr(
        "src.api.mcp_oauth.revoke_mcp_oauth_tokens_by_connection_id",
        store.revoke_tokens_by_connection_id,
    )
    return store
