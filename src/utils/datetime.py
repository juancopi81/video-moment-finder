"""Shared ISO 8601 datetime parsing utilities."""
from __future__ import annotations

from datetime import datetime, timezone

from src.utils.logging import get_logger

logger = get_logger(__name__)


def parse_iso_datetime(iso_value: str | None) -> datetime | None:
    """Parse an ISO 8601 datetime string, returning UTC-normalised datetime.

    Returns ``None`` for falsy input. Logs a warning when a non-empty string
    cannot be parsed (surfaces bad DB data).
    """
    if not iso_value:
        return None
    try:
        dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Unparseable ISO datetime value: %r", iso_value)
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
