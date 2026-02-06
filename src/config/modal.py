"""Shared Modal naming constants and helper accessors."""
from __future__ import annotations

import modal

EMBEDDING_MODAL_APP_NAME = "video-moment-finder-embed"
EMBED_IMAGES_FUNCTION_NAME = "embed_images_in_batches"
EMBED_TEXT_FUNCTION_NAME = "embed_text"


def get_embedding_modal_function(function_name: str) -> modal.Function:
    """Return a Modal function handle for the embedding app."""
    return modal.Function.from_name(EMBEDDING_MODAL_APP_NAME, function_name)
