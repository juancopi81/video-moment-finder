"""Environment loading helpers."""

from __future__ import annotations

from dotenv import load_dotenv


def load_env() -> None:
    """Load .env if present for local development."""
    load_dotenv(override=False)
