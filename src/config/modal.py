"""Shared Modal naming constants and helper accessors."""
from __future__ import annotations

import os
from functools import lru_cache

import modal

EMBEDDING_MODAL_APP_NAME = "video-moment-finder-embed"
EMBED_IMAGES_FUNCTION_NAME = "embed_images_in_batches"
EMBED_QUERY_CLASS_NAME = "QueryEmbedder"
_MODAL_AUTH_ERROR_SNIPPETS = (
    "token missing",
    "could not authenticate client",
    "modal token",
)


class ModalAuthError(RuntimeError):
    """Raised when Modal credentials are missing or invalid."""


def _validate_modal_token_pair() -> None:
    token_id = os.environ.get("MODAL_TOKEN_ID", "").strip()
    token_secret = os.environ.get("MODAL_TOKEN_SECRET", "").strip()
    if (token_id and token_secret) or (not token_id and not token_secret):
        return
    raise ModalAuthError(
        "Incomplete Modal credentials: set both MODAL_TOKEN_ID and "
        "MODAL_TOKEN_SECRET for this service."
    )


def is_modal_auth_error(exc: BaseException) -> bool:
    """Return True when an exception message matches common Modal auth failures."""
    message = str(exc).lower()
    return any(snippet in message for snippet in _MODAL_AUTH_ERROR_SNIPPETS)


def raise_modal_auth_error(exc: BaseException, *, context: str) -> None:
    """Raise ModalAuthError when a Modal auth failure is detected."""
    if not is_modal_auth_error(exc):
        return
    raise ModalAuthError(
        f"Modal authentication failed while {context}. "
        "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET for this service."
    ) from exc


@lru_cache(maxsize=None)
def get_embedding_modal_function(function_name: str) -> modal.Function:
    """Return a Modal function handle for the embedding app."""
    _validate_modal_token_pair()
    try:
        return modal.Function.from_name(EMBEDDING_MODAL_APP_NAME, function_name)
    except Exception as exc:
        raise_modal_auth_error(
            exc,
            context=f"looking up Modal function '{function_name}'",
        )
        raise


@lru_cache(maxsize=1)
def get_query_embedder_class() -> modal.Cls:
    """Return the Modal class handle used for query embeddings."""
    _validate_modal_token_pair()
    try:
        return modal.Cls.from_name(EMBEDDING_MODAL_APP_NAME, EMBED_QUERY_CLASS_NAME)
    except Exception as exc:
        raise_modal_auth_error(
            exc,
            context=f"looking up Modal class '{EMBED_QUERY_CLASS_NAME}'",
        )
        raise
