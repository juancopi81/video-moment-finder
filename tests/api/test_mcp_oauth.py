from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from urllib.parse import parse_qs, urlparse

import anyio
from fastapi.testclient import TestClient
from mcp.server.auth.provider import TokenError

from src.api.app import app
from src.api.mcp_oauth import get_mcp_oauth_provider
from src.db.supabase import ApiCreditRecord
from tests.api.conftest import InMemoryMcpOAuthStore, _authenticate

MCP_RESOURCE_URL = "https://api.videomomentfinder.com/mcp"
CLAUDE_REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
LOCALHOST_REDIRECT_URI = "http://localhost:6274/oauth/callback"
CLIENT_ID = "claude-static-client"
CLIENT_SECRET = "claude-static-secret"


def _pkce_pair(verifier: str = "vmf-oauth-test-verifier") -> tuple[str, str]:
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("utf-8")).digest()
    ).decode("utf-8").rstrip("=")
    return verifier, challenge


def _start_authorization_request(
    client: TestClient,
    *,
    redirect_uri: str = CLAUDE_REDIRECT_URI,
    scope: str = "vmf:mcp",
    resource: str = MCP_RESOURCE_URL,
    state: str = "oauth-state",
) -> tuple[object, str, str]:
    verifier, challenge = _pkce_pair()
    response = client.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": resource,
            "state": state,
        },
        follow_redirects=False,
    )
    request_id = parse_qs(urlparse(response.headers["location"]).query)["request_id"][0]
    return response, request_id, verifier


def _approve_request(
    client: TestClient,
    monkeypatch,
    request_id: str,
    *,
    user_id: str = "user_123",
    balance: int = 10_000,
) -> str:
    _authenticate(user_id)
    monkeypatch.setattr(
        "src.api.app.db_get_api_credits",
        lambda _uid: ApiCreditRecord(
            user_id=user_id,
            balance=balance,
            created_at=None,
            updated_at=None,
        ),
    )
    response = client.post(f"/oauth/mcp/requests/{request_id}/approve")
    assert response.status_code == 200
    return response.json()["redirect_url"]


def _exchange_code(
    client: TestClient,
    *,
    code: str,
    verifier: str,
    redirect_uri: str = CLAUDE_REDIRECT_URI,
    client_secret: str = CLIENT_SECRET,
) -> object:
    return client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "client_secret": client_secret,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
    )


def test_oauth_authorization_server_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/.well-known/oauth-authorization-server")

    assert response.status_code == 200
    payload = response.json()
    assert payload["issuer"].rstrip("/") == "https://api.videomomentfinder.com"
    assert payload["authorization_endpoint"] == "https://api.videomomentfinder.com/authorize"
    assert payload["token_endpoint"] == "https://api.videomomentfinder.com/token"
    assert payload["revocation_endpoint"] == "https://api.videomomentfinder.com/revoke"
    assert payload["scopes_supported"] == ["vmf:mcp"]


def test_oauth_protected_resource_metadata() -> None:
    with TestClient(app) as client:
        response = client.get("/.well-known/oauth-protected-resource/mcp")

    assert response.status_code == 200
    payload = response.json()
    assert payload["resource"] == MCP_RESOURCE_URL
    assert [value.rstrip("/") for value in payload["authorization_servers"]] == [
        "https://api.videomomentfinder.com"
    ]
    assert payload["scopes_supported"] == ["vmf:mcp"]


def test_authorize_rejects_invalid_client_id() -> None:
    verifier, challenge = _pkce_pair()
    with TestClient(app) as client:
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": "unknown-client",
                "redirect_uri": CLAUDE_REDIRECT_URI,
                "scope": "vmf:mcp",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": MCP_RESOURCE_URL,
                "state": "oauth-state",
            },
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert "Client ID" in response.json()["error_description"]


def test_authorize_rejects_invalid_redirect_uri() -> None:
    verifier, challenge = _pkce_pair()
    _ = verifier
    with TestClient(app) as client:
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": "https://evil.example.com/callback",
                "scope": "vmf:mcp",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": MCP_RESOURCE_URL,
                "state": "oauth-state",
            },
            follow_redirects=False,
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert "Redirect URI" in response.json()["error_description"]


def test_authorize_rejects_invalid_scope() -> None:
    verifier, challenge = _pkce_pair()
    _ = verifier
    with TestClient(app) as client:
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": CLAUDE_REDIRECT_URI,
                "scope": "videos:read",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": MCP_RESOURCE_URL,
                "state": "oauth-state",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    location = response.headers["location"]
    redirect_params = parse_qs(urlparse(location).query)
    assert urlparse(location).scheme == "https"
    assert urlparse(location).netloc == "claude.ai"
    assert redirect_params["error"] == ["invalid_scope"]


def test_authorize_rejects_invalid_resource() -> None:
    verifier, challenge = _pkce_pair()
    _ = verifier
    with TestClient(app) as client:
        response = client.get(
            "/authorize",
            params={
                "response_type": "code",
                "client_id": CLIENT_ID,
                "redirect_uri": CLAUDE_REDIRECT_URI,
                "scope": "vmf:mcp",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "resource": "https://api.videomomentfinder.com/not-mcp",
                "state": "oauth-state",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    redirect_params = parse_qs(urlparse(response.headers["location"]).query)
    assert redirect_params["error"] == ["invalid_request"]


def test_authorize_creates_request_and_redirects_to_frontend(
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    with TestClient(app) as client:
        response, request_id, _verifier = _start_authorization_request(client)

        public_response = client.get(f"/oauth/mcp/requests/{request_id}")

    assert response.status_code == 302
    assert response.headers["location"].startswith(
        "https://www.videomomentfinder.com/connectors/claude?request_id="
    )
    assert request_id in mcp_oauth_store.requests
    payload = public_response.json()
    assert public_response.status_code == 200
    assert payload["request_id"] == request_id
    assert payload["status"] == "pending"
    assert payload["scope"] == "vmf:mcp"
    assert {tool["name"] for tool in payload["tools"]} == {
        "upload_video",
        "get_video_status",
        "list_videos",
        "search_video",
    }


def test_authorize_accepts_localhost_callback(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        response, request_id, _verifier = _start_authorization_request(
            client, redirect_uri=LOCALHOST_REDIRECT_URI
        )
        redirect_url = _approve_request(client, monkeypatch, request_id)

    assert response.status_code == 302
    assert urlparse(redirect_url).scheme == "http"
    assert redirect_url.startswith(f"{LOCALHOST_REDIRECT_URI}?")


def test_connector_approve_blocks_zero_api_balance(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    _authenticate("user_123")
    monkeypatch.setattr("src.api.app.db_get_api_credits", lambda _uid: None)

    with TestClient(app) as client:
        _response, request_id, _verifier = _start_authorization_request(client)
        response = client.post(f"/oauth/mcp/requests/{request_id}/approve")

    assert response.status_code == 402
    assert response.json()["detail"]["code"] == "insufficient_api_units"


def test_connector_deny_returns_access_denied(
    mcp_oauth_store: InMemoryMcpOAuthStore,
) -> None:
    _authenticate("user_123")

    with TestClient(app) as client:
        _response, request_id, _verifier = _start_authorization_request(client)
        response = client.post(f"/oauth/mcp/requests/{request_id}/deny")

    assert response.status_code == 200
    redirect_url = response.json()["redirect_url"]
    redirect_params = parse_qs(urlparse(redirect_url).query)
    assert redirect_params["error"] == ["access_denied"]


def test_expired_connector_request_is_reported_and_cannot_be_approved(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _response, request_id, _verifier = _start_authorization_request(client)

        expired = replace(
            mcp_oauth_store.requests[request_id],
            expires_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
        )
        mcp_oauth_store.requests[request_id] = expired

        public_response = client.get(f"/oauth/mcp/requests/{request_id}")
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_api_credits",
            lambda _uid: ApiCreditRecord(
                user_id="user_123",
                balance=10_000,
                created_at=None,
                updated_at=None,
            ),
        )
        approve_response = client.post(f"/oauth/mcp/requests/{request_id}/approve")

    assert public_response.status_code == 200
    assert public_response.json()["status"] == "expired"
    assert approve_response.status_code == 410


def test_token_exchange_refresh_rotation_and_revoke(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _response, request_id, verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]

        token_response = _exchange_code(client, code=code, verifier=verifier)
        assert token_response.status_code == 200
        token_payload = token_response.json()
        assert token_payload["token_type"] == "Bearer"
        assert token_payload["refresh_token"]
        assert token_payload["scope"] == "vmf:mcp"

        code_reuse_response = _exchange_code(client, code=code, verifier=verifier)
        assert code_reuse_response.status_code == 400
        assert code_reuse_response.json()["error"] == "invalid_grant"

        refresh_response = client.post(
            "/token",
            data={
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "refresh_token": token_payload["refresh_token"],
            },
        )
        assert refresh_response.status_code == 200
        refresh_payload = refresh_response.json()
        assert refresh_payload["access_token"] != token_payload["access_token"]
        assert refresh_payload["refresh_token"] != token_payload["refresh_token"]

        revoke_response = client.post(
            "/revoke",
            data={
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
                "token": refresh_payload["access_token"],
                "token_type_hint": "access_token",
            },
        )
        assert revoke_response.status_code == 200

        mcp_response = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {refresh_payload['access_token']}"},
            json={},
        )

    assert mcp_response.status_code == 401


def test_token_exchange_rejects_wrong_code_verifier(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _response, request_id, _verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        response = _exchange_code(
            client,
            code=code,
            verifier="wrong-code-verifier",
        )

    assert response.status_code == 400
    assert response.json()["error"] == "invalid_grant"
    assert "code_verifier" in response.json()["error_description"]


def test_token_exchange_accepts_client_secret_basic_without_form_client_id(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    basic_auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")).decode("utf-8")

    with TestClient(app) as client:
        _response, request_id, verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]

        response = client.post(
            "/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": CLAUDE_REDIRECT_URI,
            },
            headers={"Authorization": f"Basic {basic_auth}"},
        )

    assert response.status_code == 200
    assert response.json()["token_type"] == "Bearer"


def test_token_exchange_rejects_wrong_client_secret(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _response, request_id, verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        response = _exchange_code(
            client,
            code=code,
            verifier=verifier,
            client_secret="wrong-secret",
        )

    assert response.status_code == 401
    assert response.json()["error"] == "unauthorized_client"


def test_revoke_accepts_client_secret_basic_without_form_client_id(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    basic_auth = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode("utf-8")).decode("utf-8")

    with TestClient(app) as client:
        _response, request_id, verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]
        token_response = _exchange_code(client, code=code, verifier=verifier)
        access_token = token_response.json()["access_token"]

        response = client.post(
            "/revoke",
            data={
                "token": access_token,
                "token_type_hint": "access_token",
            },
            headers={"Authorization": f"Basic {basic_auth}"},
        )

        mcp_response = client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {access_token}"},
            json={},
        )

    assert response.status_code == 200
    assert mcp_response.status_code == 401


def test_exchange_rejects_second_preloaded_copy_of_same_authorization_code(
    mcp_oauth_store: InMemoryMcpOAuthStore,
    monkeypatch,
) -> None:
    with TestClient(app) as client:
        _response, request_id, verifier = _start_authorization_request(client)
        redirect_url = _approve_request(client, monkeypatch, request_id)
        code = parse_qs(urlparse(redirect_url).query)["code"][0]

    async def _runner() -> None:
        provider = get_mcp_oauth_provider()
        oauth_client = await provider.get_client(CLIENT_ID)
        assert oauth_client is not None

        first_loaded = await provider.load_authorization_code(oauth_client, code)
        second_loaded = await provider.load_authorization_code(oauth_client, code)
        assert first_loaded is not None
        assert second_loaded is not None

        first_token = await provider.exchange_authorization_code(oauth_client, first_loaded)
        assert first_token.access_token

        try:
            await provider.exchange_authorization_code(oauth_client, second_loaded)
        except TokenError as exc:
            assert exc.error == "invalid_grant"
            return
        raise AssertionError("Second preloaded authorization code redemption unexpectedly succeeded")

    anyio.run(_runner)


def test_mcp_cors_preflight_allows_claude_origins() -> None:
    with TestClient(app) as client:
        response = client.options(
            "/mcp",
            headers={
                "Origin": "https://claude.ai",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://claude.ai"
