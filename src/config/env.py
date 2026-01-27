"""Environment loading helpers."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None


def load_env(*, required: bool = False) -> None:
    """Load .env if present for local development.

    When required=True and python-dotenv is missing, raise a clear error so
    local scripts don't silently skip env loading.
    """
    if load_dotenv is None:
        if required:
            raise RuntimeError(
                "python-dotenv is required to load .env; install it or disable "
                "load_env(required=True)."
            )
        return
    load_dotenv(override=False)
