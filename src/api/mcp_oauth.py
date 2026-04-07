"""OAuth service for the Claude MCP connector."""
from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets
from typing import Any
from urllib.parse import unquote
from uuid import uuid4

from starlette.datastructures import FormData
from mcp.server.auth.middleware.client_auth import AuthenticationError
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    AuthorizeError,
    OAuthAuthorizationServerProvider,
    RegistrationError,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.server.auth.routes import build_metadata, build_resource_metadata_url
from mcp.server.auth.settings import ClientRegistrationOptions, RevocationOptions
from mcp.shared.auth import OAuthClientInformationFull, OAuthMetadata, OAuthToken, ProtectedResourceMetadata
from pydantic import AnyHttpUrl, TypeAdapter

from src.db.supabase import (
    McpOAuthAuthorizationRequestRecord,
    consume_mcp_oauth_authorization_code,
    create_mcp_oauth_client,
    create_mcp_oauth_authorization_code,
    create_mcp_oauth_authorization_request,
    create_mcp_oauth_tokens,
    delete_expired_mcp_oauth_authorization_requests,
    get_mcp_oauth_access_token_by_hash,
    get_mcp_oauth_client,
    get_mcp_oauth_authorization_code_by_hash,
    get_mcp_oauth_authorization_request,
    get_mcp_oauth_refresh_token_by_hash,
    revoke_mcp_oauth_access_tokens_for_connection,
    revoke_mcp_oauth_refresh_token,
    revoke_mcp_oauth_tokens_by_connection_id,
    update_mcp_oauth_authorization_request_resolution,
)
from src.utils.datetime import parse_iso_datetime

DEFAULT_MCP_OAUTH_SCOPE = "vmf:mcp"
AUTHORIZATION_REQUEST_TTL_S = 3600
AUTHORIZATION_CODE_TTL_S = 600
ACCESS_TOKEN_TTL_S = 3600
REFRESH_TOKEN_TTL_S = 30 * 24 * 3600
SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS = (
    "none",
    "client_secret_post",
    "client_secret_basic",
)
CLAUDE_REDIRECT_URIS = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "http://localhost:6274/oauth/callback",
    "http://localhost:6274/oauth/callback/debug",
]


class McpOAuthConfigError(RuntimeError):
    """Raised when required OAuth configuration is missing."""


@dataclass(frozen=True)
class McpOAuthFlowError(RuntimeError):
    """Structured runtime error surfaced by connector approval endpoints."""

    status_code: int
    detail: str


@dataclass(frozen=True)
class McpOAuthSettings:
    issuer_url: str
    resource_url: str
    client_id: str
    client_secret: str
    frontend_base_url: str


class StoredAuthorizationCode(AuthorizationCode):
    user_id: str
    authorization_request_id: str | None = None
    record_id: str


class StoredRefreshToken(RefreshToken):
    user_id: str
    connection_id: str
    resource: str
    record_id: str


class StoredAccessToken(AccessToken):
    user_id: str
    connection_id: str
    record_id: str


def _any_http_url(value: str) -> AnyHttpUrl:
    return TypeAdapter(AnyHttpUrl).validate_python(value)


def _required_env(name: str) -> str:
    import os

    value = os.environ.get(name, "").strip()
    if not value:
        raise McpOAuthConfigError(f"{name} environment variable is required")
    return value


def get_mcp_oauth_settings() -> McpOAuthSettings:
    return McpOAuthSettings(
        issuer_url=_required_env("MCP_OAUTH_ISSUER_URL").rstrip("/"),
        resource_url=_required_env("MCP_OAUTH_RESOURCE_URL").rstrip("/"),
        client_id=_required_env("MCP_OAUTH_CLIENT_ID"),
        client_secret=_required_env("MCP_OAUTH_CLIENT_SECRET"),
        frontend_base_url=_required_env("FRONTEND_BASE_URL").rstrip("/"),
    )


def mcp_oauth_scope() -> str:
    return DEFAULT_MCP_OAUTH_SCOPE


def mcp_oauth_resource_url() -> str:
    return get_mcp_oauth_settings().resource_url


def mcp_oauth_resource_metadata_url() -> str:
    return str(build_resource_metadata_url(_any_http_url(mcp_oauth_resource_url())))


def mcp_oauth_client_registration_options() -> ClientRegistrationOptions:
    return ClientRegistrationOptions(
        enabled=True,
        valid_scopes=[DEFAULT_MCP_OAUTH_SCOPE],
        default_scopes=[DEFAULT_MCP_OAUTH_SCOPE],
    )


def mcp_oauth_authorization_metadata() -> OAuthMetadata:
    settings = get_mcp_oauth_settings()
    metadata = build_metadata(
        issuer_url=_any_http_url(settings.issuer_url),
        service_documentation_url=_any_http_url(f"{settings.frontend_base_url}/developers"),
        client_registration_options=mcp_oauth_client_registration_options(),
        revocation_options=RevocationOptions(enabled=True),
    )
    metadata.token_endpoint_auth_methods_supported = list(SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS)
    metadata.revocation_endpoint_auth_methods_supported = list(SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS)
    return metadata


def mcp_oauth_protected_resource_metadata() -> ProtectedResourceMetadata:
    settings = get_mcp_oauth_settings()
    return ProtectedResourceMetadata(
        resource=_any_http_url(settings.resource_url),
        authorization_servers=[_any_http_url(settings.issuer_url)],
        scopes_supported=[DEFAULT_MCP_OAUTH_SCOPE],
        resource_name="Video Moment Finder MCP",
        resource_documentation=_any_http_url(f"{settings.frontend_base_url}/developers"),
    )


def mcp_oauth_www_authenticate(
    *,
    error: str | None = None,
    description: str | None = None,
) -> str:
    parts = []
    if error is not None:
        parts.append(f'error="{error}"')
    if description is not None:
        parts.append(f'error_description="{description}"')
    parts.append(f'resource_metadata="{mcp_oauth_resource_metadata_url()}"')
    return "Bearer " + ", ".join(parts)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _expiry_iso(ttl_s: int) -> str:
    return (_now_utc() + timedelta(seconds=ttl_s)).isoformat()


def _hash_secret(raw_value: str) -> str:
    return hashlib.sha256(raw_value.encode()).hexdigest()


def _set_request_form_field(request: Any, form_data: FormData, *, name: str, value: str) -> FormData:
    items = list(form_data.multi_items())
    items.append((name, value))
    updated = FormData(items)
    request._form = updated
    return updated


def _normalize_scopes(scopes: list[str] | None) -> list[str]:
    if not scopes:
        return [DEFAULT_MCP_OAUTH_SCOPE]
    normalized = [scope.strip() for scope in scopes if scope.strip()]
    if normalized != [DEFAULT_MCP_OAUTH_SCOPE]:
        raise AuthorizeError("invalid_scope", f"Only scope `{DEFAULT_MCP_OAUTH_SCOPE}` is supported")
    return normalized


def _normalize_resource(resource: str | None) -> str:
    configured = mcp_oauth_resource_url()
    if resource is None:
        return configured
    candidate = resource.strip().rstrip("/")
    if candidate != configured:
        raise AuthorizeError("invalid_request", "Unsupported resource indicator")
    return configured


def _redirect_to_connect_page(request_id: str) -> str:
    settings = get_mcp_oauth_settings()
    return f"{settings.frontend_base_url}/connectors/claude?request_id={request_id}"


def _request_is_expired(record: McpOAuthAuthorizationRequestRecord) -> bool:
    expires_at = parse_iso_datetime(record.expires_at)
    return expires_at is None or expires_at <= _now_utc()


def _token_is_expired(expires_at: str | None) -> bool:
    parsed = parse_iso_datetime(expires_at)
    return parsed is None or parsed <= _now_utc()


def _connection_redirect_with_error(
    record: McpOAuthAuthorizationRequestRecord,
    *,
    error: str,
    error_description: str | None = None,
) -> str:
    return construct_redirect_uri(
        record.redirect_uri,
        error=error,
        error_description=error_description,
        state=record.state,
    )


def _static_client() -> OAuthClientInformationFull:
    settings = get_mcp_oauth_settings()
    return OAuthClientInformationFull(
        client_id=settings.client_id,
        client_secret=settings.client_secret,
        client_name="Claude",
        redirect_uris=CLAUDE_REDIRECT_URIS,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        scope=DEFAULT_MCP_OAUTH_SCOPE,
        token_endpoint_auth_method="client_secret_post",
    )


def _client_from_record(record) -> OAuthClientInformationFull:
    return OAuthClientInformationFull(
        client_id=record.client_id,
        client_secret=record.client_secret,
        client_id_issued_at=record.client_id_issued_at,
        client_secret_expires_at=record.client_secret_expires_at,
        redirect_uris=record.redirect_uris,
        token_endpoint_auth_method=record.token_endpoint_auth_method,
        grant_types=record.grant_types,
        response_types=record.response_types,
        scope=record.scope,
        client_name=record.client_name,
        client_uri=record.client_uri,
        logo_uri=record.logo_uri,
        contacts=record.contacts,
        tos_uri=record.tos_uri,
        policy_uri=record.policy_uri,
        jwks_uri=record.jwks_uri,
        jwks=record.jwks,
        software_id=record.software_id,
        software_version=record.software_version,
    )


class FlexibleClientAuthenticator:
    """Accept form or Basic auth while allowing public DCR clients."""

    def __init__(self, provider: "McpOAuthProvider"):
        self.provider = provider

    async def authenticate_request(self, request) -> OAuthClientInformationFull:
        form_data = await request.form()
        client_id_raw = form_data.get("client_id")
        form_client_id = client_id_raw if isinstance(client_id_raw, str) and client_id_raw else None
        request_client_secret: str | None = None
        basic_client_id: str | None = None
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                if ":" not in decoded:
                    raise ValueError("Invalid Basic auth format")
                basic_client_id, request_client_secret = decoded.split(":", 1)
                basic_client_id = unquote(basic_client_id)
                request_client_secret = unquote(request_client_secret)
                if form_client_id is not None and basic_client_id != form_client_id:
                    raise AuthenticationError("Client ID mismatch in Basic auth")
            except (ValueError, UnicodeDecodeError, binascii.Error):
                raise AuthenticationError("Invalid Basic authentication header")
        else:
            client_secret_raw = form_data.get("client_secret")
            if isinstance(client_secret_raw, str) and client_secret_raw:
                request_client_secret = client_secret_raw

        effective_client_id = form_client_id or basic_client_id
        if not effective_client_id:
            raise AuthenticationError("Missing client_id")
        if form_client_id is None and basic_client_id is not None:
            form_data = _set_request_form_field(
                request,
                form_data,
                name="client_id",
                value=basic_client_id,
            )
        form_client_secret = form_data.get("client_secret")
        if (
            basic_client_id is not None
            and request_client_secret is not None
            and not isinstance(form_client_secret, str)
        ):
            form_data = _set_request_form_field(
                request,
                form_data,
                name="client_secret",
                value=request_client_secret,
            )

        client = await self.provider.get_client(effective_client_id)
        if client is None:
            raise AuthenticationError("Invalid client_id")

        token_auth_method = client.token_endpoint_auth_method or "none"
        if token_auth_method not in SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS:
            raise AuthenticationError(f"Unsupported auth method: {token_auth_method}")

        if client.client_secret:
            if not request_client_secret:
                raise AuthenticationError("Client secret is required")
            if not hmac.compare_digest(client.client_secret.encode(), request_client_secret.encode()):
                raise AuthenticationError("Invalid client_secret")

        return client


class McpOAuthProvider(
    OAuthAuthorizationServerProvider[StoredAuthorizationCode, StoredRefreshToken, StoredAccessToken]
):
    """Static-client OAuth provider backed by Supabase tables."""

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        static_client = _static_client()
        if client_id == static_client.client_id:
            return static_client

        record = get_mcp_oauth_client(client_id)
        if record is None:
            return None
        return _client_from_record(record)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        auth_method = client_info.token_endpoint_auth_method or "none"
        if auth_method not in SUPPORTED_TOKEN_ENDPOINT_AUTH_METHODS:
            raise RegistrationError(
                "invalid_client_metadata",
                f"Unsupported token_endpoint_auth_method `{auth_method}`",
            )

        create_mcp_oauth_client(
            client_id=client_info.client_id or "",
            client_secret=client_info.client_secret,
            client_id_issued_at=client_info.client_id_issued_at,
            client_secret_expires_at=client_info.client_secret_expires_at,
            redirect_uris=[str(uri) for uri in client_info.redirect_uris or []],
            token_endpoint_auth_method=auth_method,
            grant_types=list(client_info.grant_types),
            response_types=list(client_info.response_types),
            scope=client_info.scope,
            client_name=client_info.client_name,
            client_uri=str(client_info.client_uri) if client_info.client_uri is not None else None,
            logo_uri=str(client_info.logo_uri) if client_info.logo_uri is not None else None,
            contacts=list(client_info.contacts) if client_info.contacts is not None else None,
            tos_uri=str(client_info.tos_uri) if client_info.tos_uri is not None else None,
            policy_uri=str(client_info.policy_uri) if client_info.policy_uri is not None else None,
            jwks_uri=str(client_info.jwks_uri) if client_info.jwks_uri is not None else None,
            jwks=client_info.jwks,
            software_id=client_info.software_id,
            software_version=client_info.software_version,
        )

    async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
        scopes = _normalize_scopes(params.scopes)
        resource = _normalize_resource(params.resource)
        delete_expired_mcp_oauth_authorization_requests(_now_utc().isoformat())
        record = create_mcp_oauth_authorization_request(
            client_id=client.client_id or "",
            redirect_uri=str(params.redirect_uri),
            redirect_uri_provided_explicitly=params.redirect_uri_provided_explicitly,
            state=params.state,
            scopes=scopes,
            code_challenge=params.code_challenge,
            resource=resource,
            expires_at=_expiry_iso(AUTHORIZATION_REQUEST_TTL_S),
        )
        return _redirect_to_connect_page(record.id)

    async def load_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: str,
    ) -> StoredAuthorizationCode | None:
        record = get_mcp_oauth_authorization_code_by_hash(_hash_secret(authorization_code))
        if record is None or record.client_id != client.client_id or _token_is_expired(record.expires_at):
            return None
        expires_at = parse_iso_datetime(record.expires_at)
        if expires_at is None:
            return None
        return StoredAuthorizationCode(
            code=authorization_code,
            scopes=record.scopes,
            expires_at=expires_at.timestamp(),
            client_id=record.client_id,
            code_challenge=record.code_challenge,
            redirect_uri=record.redirect_uri,
            redirect_uri_provided_explicitly=record.redirect_uri_provided_explicitly,
            resource=record.resource,
            user_id=record.user_id,
            authorization_request_id=record.authorization_request_id,
            record_id=record.id,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: StoredAuthorizationCode,
    ) -> OAuthToken:
        if authorization_code.client_id != client.client_id:
            raise TokenError("invalid_grant", "authorization code does not belong to this client")
        if not consume_mcp_oauth_authorization_code(authorization_code.record_id):
            raise TokenError("invalid_grant", "authorization code does not exist")

        connection_id = str(uuid4())
        raw_access_token = secrets.token_urlsafe(32)
        raw_refresh_token = secrets.token_urlsafe(32)
        create_mcp_oauth_tokens(
            connection_id=connection_id,
            user_id=authorization_code.user_id,
            client_id=authorization_code.client_id,
            access_token_hash=_hash_secret(raw_access_token),
            refresh_token_hash=_hash_secret(raw_refresh_token),
            scopes=authorization_code.scopes,
            resource=authorization_code.resource or mcp_oauth_resource_url(),
            access_expires_at=_expiry_iso(ACCESS_TOKEN_TTL_S),
            refresh_expires_at=_expiry_iso(REFRESH_TOKEN_TTL_S),
        )
        return OAuthToken(
            access_token=raw_access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_S,
            refresh_token=raw_refresh_token,
            scope=" ".join(authorization_code.scopes),
        )

    async def load_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: str,
    ) -> StoredRefreshToken | None:
        record = get_mcp_oauth_refresh_token_by_hash(_hash_secret(refresh_token))
        if record is None or record.client_id != client.client_id or _token_is_expired(record.expires_at):
            return None
        expires_at = parse_iso_datetime(record.expires_at)
        return StoredRefreshToken(
            token=refresh_token,
            client_id=record.client_id,
            scopes=record.scopes,
            expires_at=int(expires_at.timestamp()) if expires_at is not None else None,
            user_id=record.user_id,
            connection_id=record.connection_id,
            resource=record.resource,
            record_id=record.id,
        )

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: StoredRefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        if refresh_token.client_id != client.client_id:
            raise TokenError("invalid_grant", "refresh token does not belong to this client")

        revoke_mcp_oauth_access_tokens_for_connection(refresh_token.connection_id)
        revoke_mcp_oauth_refresh_token(refresh_token.record_id)

        raw_access_token = secrets.token_urlsafe(32)
        raw_refresh_token = secrets.token_urlsafe(32)
        create_mcp_oauth_tokens(
            connection_id=refresh_token.connection_id,
            user_id=refresh_token.user_id,
            client_id=refresh_token.client_id,
            access_token_hash=_hash_secret(raw_access_token),
            refresh_token_hash=_hash_secret(raw_refresh_token),
            scopes=scopes,
            resource=refresh_token.resource,
            access_expires_at=_expiry_iso(ACCESS_TOKEN_TTL_S),
            refresh_expires_at=_expiry_iso(REFRESH_TOKEN_TTL_S),
        )
        return OAuthToken(
            access_token=raw_access_token,
            token_type="Bearer",
            expires_in=ACCESS_TOKEN_TTL_S,
            refresh_token=raw_refresh_token,
            scope=" ".join(scopes),
        )

    async def load_access_token(self, token: str) -> StoredAccessToken | None:
        record = get_mcp_oauth_access_token_by_hash(_hash_secret(token))
        if record is None or _token_is_expired(record.expires_at):
            return None
        expires_at = parse_iso_datetime(record.expires_at)
        return StoredAccessToken(
            token=token,
            client_id=record.client_id,
            scopes=record.scopes,
            expires_at=int(expires_at.timestamp()) if expires_at is not None else None,
            resource=record.resource,
            user_id=record.user_id,
            connection_id=record.connection_id,
            record_id=record.id,
        )

    async def revoke_token(self, token: StoredAccessToken | StoredRefreshToken) -> None:
        revoke_mcp_oauth_tokens_by_connection_id(token.connection_id)

    def get_authorization_request(self, request_id: str) -> McpOAuthAuthorizationRequestRecord | None:
        return get_mcp_oauth_authorization_request(request_id)

    def approve_authorization_request(self, request_id: str, *, user_id: str) -> str:
        record = get_mcp_oauth_authorization_request(request_id)
        if record is None:
            raise McpOAuthFlowError(404, "Connector request not found")
        if record.status != "pending":
            raise McpOAuthFlowError(409, "Connector request has already been resolved")
        if _request_is_expired(record):
            raise McpOAuthFlowError(410, "Connector request has expired")

        raw_code = secrets.token_urlsafe(32)
        create_mcp_oauth_authorization_code(
            authorization_request_id=record.id,
            user_id=user_id,
            client_id=record.client_id,
            code_hash=_hash_secret(raw_code),
            redirect_uri=record.redirect_uri,
            redirect_uri_provided_explicitly=record.redirect_uri_provided_explicitly,
            scopes=record.scopes,
            code_challenge=record.code_challenge,
            resource=record.resource,
            expires_at=_expiry_iso(AUTHORIZATION_CODE_TTL_S),
        )
        updated = update_mcp_oauth_authorization_request_resolution(
            request_id,
            status="approved",
            user_id=user_id,
        )
        if updated is None:
            raise McpOAuthFlowError(409, "Connector request could not be approved")

        return construct_redirect_uri(
            record.redirect_uri,
            code=raw_code,
            state=record.state,
        )

    def deny_authorization_request(self, request_id: str) -> str:
        record = get_mcp_oauth_authorization_request(request_id)
        if record is None:
            raise McpOAuthFlowError(404, "Connector request not found")
        if record.status != "pending":
            raise McpOAuthFlowError(409, "Connector request has already been resolved")
        if _request_is_expired(record):
            raise McpOAuthFlowError(410, "Connector request has expired")

        updated = update_mcp_oauth_authorization_request_resolution(
            request_id,
            status="denied",
        )
        if updated is None:
            raise McpOAuthFlowError(409, "Connector request could not be denied")
        return _connection_redirect_with_error(record, error="access_denied")


def get_mcp_oauth_provider() -> McpOAuthProvider:
    return McpOAuthProvider()


async def load_mcp_oauth_access_token(token: str) -> StoredAccessToken | None:
    return await get_mcp_oauth_provider().load_access_token(token)


def mcp_oauth_request_public_payload(
    record: McpOAuthAuthorizationRequestRecord,
) -> dict[str, Any]:
    return {
        "request_id": record.id,
        "client_id": record.client_id,
        "resource": record.resource,
        "scope": " ".join(record.scopes),
        "scopes": record.scopes,
        "status": record.status,
        "expires_at": record.expires_at,
    }
