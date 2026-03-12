"""Environment loading helpers."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    load_dotenv = None


def load_env(*, required: bool = False) -> None:
    """Load .env.local (if present) then .env for local development.

    ``.env.local`` values take precedence over ``.env`` defaults so that
    developers can point at local Supabase / in-memory Qdrant without
    editing the shared ``.env`` file.

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
    load_dotenv(dotenv_path=".env.local", override=True)
    load_dotenv(override=False)
