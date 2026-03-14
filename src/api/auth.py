"""Authentication utilities for API endpoints."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any, Literal

import jwt
from fastapi import Header, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.db.supabase import get_api_key_by_hash, touch_api_key_last_used
from src.utils.logging import get_logger

logger = get_logger(__name__)

AuthMethod = Literal["jwt", "api_key"]


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


def verify_api_key(token: str) -> AuthIdentity:
    """Verify an API key token and return an AuthIdentity with attribution."""
    key_hash = hashlib.sha256(token.encode()).hexdigest()
    record = get_api_key_by_hash(key_hash)
    if record is None:
        raise TokenVerificationError("Invalid authentication token")

    touch_api_key_last_used(record.id)
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

    try:
        return verify_bearer_token(token)
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc


def get_optional_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """FastAPI dependency: JWT-only optional auth, returns user id or None.

    Raises 401 if the header is present but the token is invalid.
    API keys are rejected to limit key scope to /api/v1/.
    """
    if authorization is None:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Invalid Authorization header")

    token = token.strip()
    if token.startswith("vmf_"):
        raise _unauthorized("API keys are not accepted on this endpoint")

    try:
        return verify_bearer_token(token)
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc


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

    try:
        return _verify_token(credentials.credentials)
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc


def get_optional_user(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthIdentity | None:
    """FastAPI dependency: JWT or API key optional auth, returns AuthIdentity or None.

    Raises 401 if the header is present but the token is invalid.
    """
    if authorization is None:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Invalid Authorization header")

    try:
        return _verify_token(token.strip())
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc
