"""Sentry monitoring helpers for API and worker runtimes."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _SentryConfig:
    endpoint: str
    auth_header: str
    environment: str
    release: str | None
    service: str


_SENTRY_CONFIG: _SentryConfig | None = None


def _build_config(dsn: str, *, service: str) -> _SentryConfig | None:
    parsed = urlsplit(dsn)
    if not parsed.scheme or not parsed.hostname or not parsed.username:
        logger.warning("SENTRY_DSN is invalid; monitoring disabled")
        return None

    path_parts = [part for part in parsed.path.split("/") if part]
    if not path_parts:
        logger.warning("SENTRY_DSN is missing project id; monitoring disabled")
        return None

    project_id = path_parts[-1]
    base_path = "/".join(path_parts[:-1])
    prefix = f"/{base_path}" if base_path else ""
    endpoint = f"{parsed.scheme}://{parsed.hostname}{prefix}/api/{project_id}/store/"
    if parsed.port:
        endpoint = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}{prefix}/api/{project_id}/store/"

    auth_parts = [
        "Sentry sentry_version=7",
        f"sentry_key={parsed.username}",
        "sentry_client=video-moment-finder/1.0",
    ]
    if parsed.password:
        auth_parts.append(f"sentry_secret={parsed.password}")

    environment = os.environ.get("SENTRY_ENVIRONMENT", "").strip() or "development"
    release = os.environ.get("SENTRY_RELEASE", "").strip() or None
    return _SentryConfig(
        endpoint=endpoint,
        auth_header=", ".join(auth_parts),
        environment=environment,
        release=release,
        service=service,
    )


def init_sentry(*, service: str, include_fastapi_integration: bool = False) -> bool:
    """Initialize Sentry if SENTRY_DSN is configured.

    `include_fastapi_integration` is accepted for API call-site parity and ignored.
    """
    del include_fastapi_integration
    global _SENTRY_CONFIG

    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    if _SENTRY_CONFIG is not None:
        return True

    config = _build_config(dsn, service=service)
    if config is None:
        return False

    _SENTRY_CONFIG = config
    logger.info(
        "Sentry monitoring enabled service=%s environment=%s",
        config.service,
        config.environment,
    )
    return True


def capture_exception(exc: BaseException, *, context: dict[str, Any] | None = None) -> None:
    config = _SENTRY_CONFIG
    if config is None:
        return

    event = {
        "event_id": uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": "error",
        "platform": "python",
        "environment": config.environment,
        "release": config.release,
        "logger": "video-moment-finder",
        "tags": {"service": config.service},
        "exception": {
            "values": [
                {
                    "type": type(exc).__name__,
                    "value": str(exc),
                }
            ]
        },
        "extra": context or {},
    }
    payload = json.dumps(event).encode("utf-8")
    request = Request(
        config.endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Sentry-Auth": config.auth_header,
        },
    )
    try:
        with urlopen(request, timeout=2):
            pass
    except Exception as send_exc:
        logger.warning("Failed to send Sentry event: %s", send_exc)
