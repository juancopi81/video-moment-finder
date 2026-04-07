"""Authentication utilities for API endpoints."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

import jwt
from fastapi import Header, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.db.supabase import get_api_key_by_hash, touch_api_key_last_used
from src.utils.logging import get_logger

logger = get_logger(__name__)

AuthMethod = Literal["jwt", "api_key", "mcp_oauth"]

# Throttle last_used_at writes: at most one DB update per key per 60 s.
_LAST_USED_WRITE_INTERVAL_S = 60
_last_used_writes: dict[str, float] = {}


class AuthConfigError(RuntimeError):
    """Raised when auth configuration is missing or invalid."""


class TokenVerificationError(RuntimeError):
    """Raised when a bearer token cannot be verified."""


@dataclass(frozen=True)
class AuthIdentity:
    """Authenticated caller identity with attribution metadata."""

    user_id: str
    auth_method: AuthMethod
    api_key_id: str | None = None


@dataclass(frozen=True)
class AuthSettings:
    issuer: str
    audience: str | None
    jwks_url: str


@lru_cache(maxsize=1)
def get_auth_settings() -> AuthSettings:
    """Load and validate Clerk auth configuration from environment."""
    issuer = os.environ.get("CLERK_ISSUER", "").strip().rstrip("/")
    if not issuer:
        raise AuthConfigError("CLERK_ISSUER environment variable is required")

    audience_raw = os.environ.get("CLERK_AUDIENCE")
    audience = audience_raw.strip() if audience_raw is not None else None
    if audience == "":
        audience = None

    jwks_url = os.environ.get("CLERK_JWKS_URL", "").strip()
    if not jwks_url:
        jwks_url = f"{issuer}/.well-known/jwks.json"

    return AuthSettings(issuer=issuer, audience=audience, jwks_url=jwks_url)


@lru_cache(maxsize=8)
def _get_jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def verify_bearer_token(token: str) -> str:
    """Verify bearer token and return authenticated user id (JWT sub)."""
    if not token.strip():
        raise TokenVerificationError("Invalid authentication token")

    settings = get_auth_settings()
    try:
        signing_key = _get_jwks_client(settings.jwks_url).get_signing_key_from_jwt(token)
        decode_kwargs: dict[str, Any] = {
            "key": signing_key.key,
            "algorithms": ["RS256"],
            "issuer": settings.issuer,
        }
        if settings.audience:
            decode_kwargs["audience"] = settings.audience
        else:
            decode_kwargs["options"] = {"verify_aud": False}
        payload = jwt.decode(token, **decode_kwargs)
    except (jwt.InvalidTokenError, jwt.PyJWKClientError, ValueError) as exc:
        raise TokenVerificationError("Invalid authentication token") from exc

    user_id = payload.get("sub")
    if not isinstance(user_id, str) or not user_id.strip():
        raise TokenVerificationError("Invalid authentication token")
    return user_id


def hash_api_key(token: str) -> str:
    """Return the SHA-256 hex digest of a raw API key."""
    return hashlib.sha256(token.encode()).hexdigest()


def verify_api_key(token: str) -> AuthIdentity:
    """Verify an API key token and return an AuthIdentity with attribution."""
    key_hash = hash_api_key(token)
    record = get_api_key_by_hash(key_hash)
    if record is None:
        raise TokenVerificationError("Invalid authentication token")

    now = time.monotonic()
    if now - _last_used_writes.get(record.id, 0) >= _LAST_USED_WRITE_INTERVAL_S:
        touch_api_key_last_used(record.id)
        _last_used_writes[record.id] = now

    return AuthIdentity(
        user_id=record.user_id,
        auth_method="api_key",
        api_key_id=record.id,
    )


def _verify_token(token: str) -> AuthIdentity:
    """Route token to API-key or JWT verification, returning full identity."""
    if token.startswith("vmf_"):
        return verify_api_key(token)
    user_id = verify_bearer_token(token)
    return AuthIdentity(user_id=user_id, auth_method="jwt")


_bearer_scheme = HTTPBearer(auto_error=False)


def _run_auth(verify_fn, *args):
    """Call a verification function, translating auth exceptions to HTTP errors."""
    try:
        return verify_fn(*args)
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc


def _extract_bearer_token(authorization: str) -> str:
    """Parse raw Authorization header and return the bearer token."""
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Invalid Authorization header")
    return token.strip()


# ---------------------------------------------------------------------------
# JWT-only dependencies (legacy / internal routes)
# ---------------------------------------------------------------------------


def get_current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_bearer_scheme)
    ] = None,
) -> str:
    """FastAPI dependency: JWT-only auth, returns user id.

    Used by legacy and internal routes (billing, checkout, unversioned
    handlers). API keys are rejected to limit key scope to /api/v1/.
    """
    if credentials is None:
        raise _unauthorized("Missing Authorization header")

    token = credentials.credentials
    if token.startswith("vmf_"):
        raise _unauthorized("API keys are not accepted on this endpoint")

    return _run_auth(verify_bearer_token, token)


def get_optional_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """FastAPI dependency: JWT-only optional auth, returns user id or None.

    Raises 401 if the header is present but the token is invalid.
    API keys are rejected to limit key scope to /api/v1/.
    """
    if authorization is None:
        return None

    token = _extract_bearer_token(authorization)
    if token.startswith("vmf_"):
        raise _unauthorized("API keys are not accepted on this endpoint")

    return _run_auth(verify_bearer_token, token)


# ---------------------------------------------------------------------------
# JWT + API key dependencies (v1 external routes)
# ---------------------------------------------------------------------------


def get_current_user(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_bearer_scheme)
    ] = None,
) -> AuthIdentity:
    """FastAPI dependency: JWT or API key auth, returns full AuthIdentity.

    Used by /api/v1/ routes where API keys are accepted.
    """
    if credentials is None:
        raise _unauthorized("Missing Authorization header")

    return _run_auth(_verify_token, credentials.credentials)
