"""Shared Modal naming constants and helper accessors."""
from __future__ import annotations

from functools import lru_cache

import modal

EMBEDDING_MODAL_APP_NAME = "video-moment-finder-embed"
EMBED_IMAGES_FUNCTION_NAME = "embed_images_in_batches"
EMBED_TEXT_CLASS_NAME = "TextEmbedder"


@lru_cache(maxsize=None)
def get_embedding_modal_function(function_name: str) -> modal.Function:
    """Return a Modal function handle for the embedding app."""
    return modal.Function.from_name(EMBEDDING_MODAL_APP_NAME, function_name)


@lru_cache(maxsize=1)
def get_text_embedder_class() -> modal.Cls:
    """Return the Modal class handle used for text embeddings."""
    return modal.Cls.from_name(EMBEDDING_MODAL_APP_NAME, EMBED_TEXT_CLASS_NAME)
