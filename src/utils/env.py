"""Shared environment variable parsing utilities."""
from __future__ import annotations

import math
import os

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_env_int(name: str, default: int, *, strict: bool = False) -> int:
    """Parse a positive integer from an environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is missing/empty (or invalid
            when ``strict=False``).
        strict: When *True*, raise ``ValueError`` on invalid or non-positive
            values instead of falling back to *default*.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        if strict:
            raise ValueError(f"{name}={raw!r} is not a valid integer") from None
        logger.warning("Invalid %s=%r; using default %d", name, raw, default)
        return default
    if value <= 0:
        if strict:
            raise ValueError(f"{name} must be > 0, got {value}")
        logger.warning("%s must be positive (%d); using default %d", name, value, default)
        return default
    return value


def get_env_float(name: str, default: float, *, strict: bool = False) -> float:
    """Parse a positive, finite float from an environment variable.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is missing/empty (or invalid
            when ``strict=False``).
        strict: When *True*, raise ``ValueError`` on invalid, non-positive, or
            non-finite values instead of falling back to *default*.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        if strict:
            raise ValueError(f"{name}={raw!r} is not a valid float") from None
        logger.warning("Invalid %s=%r; using default %s", name, raw, default)
        return default
    if not math.isfinite(value) or value <= 0:
        if strict:
            raise ValueError(f"{name} must be positive and finite, got {value}")
        logger.warning("%s must be positive and finite (%s); using default %s", name, value, default)
        return default
    return value
