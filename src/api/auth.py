"""Authentication utilities for API endpoints."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Annotated, Any

import jwt
from fastapi import Header, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AuthConfigError(RuntimeError):
    """Raised when auth configuration is missing or invalid."""


class TokenVerificationError(RuntimeError):
    """Raised when a bearer token cannot be verified."""


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


_bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user_id(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Security(_bearer_scheme)
    ] = None,
) -> str:
    """FastAPI dependency that returns the authenticated Clerk user id."""
    if credentials is None:
        raise _unauthorized("Missing Authorization header")

    try:
        return verify_bearer_token(credentials.credentials)
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc


def get_optional_user_id(
    authorization: Annotated[str | None, Header()] = None,
) -> str | None:
    """FastAPI dependency: returns user id when token present, None when header absent.

    Raises 401 if the header is present but the token is invalid (prevents
    broken tokens from being silently treated as anonymous).
    """
    if authorization is None:
        return None

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _unauthorized("Invalid Authorization header")

    try:
        return verify_bearer_token(token.strip())
    except AuthConfigError as exc:
        logger.error("Auth configuration error: %s", exc)
        raise HTTPException(status_code=500, detail="Authentication is not configured") from exc
    except TokenVerificationError as exc:
        raise _unauthorized(str(exc)) from exc
