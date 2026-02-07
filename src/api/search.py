"""Search service for video moment finder."""
from __future__ import annotations

import time

from src.config.modal import EMBED_TEXT_FUNCTION_NAME, get_embedding_modal_function
from src.storage.config import QdrantConfig
from src.storage.qdrant import QdrantStore, SearchResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


def search_video(video_id: str, query_text: str, limit: int = 5) -> list[SearchResult]:
    """
    Embed query text via Modal, search Qdrant, return results.

    Args:
        video_id: The ID of the video to search within.
        query_text: The text query to search for.
        limit: Maximum number of results to return.

    Returns:
        List of SearchResult with timestamp, thumbnail URL, and score.
    """
    logger.info(
        "Searching video_id=%s query=%r limit=%d", video_id, query_text[:50], limit
    )

    total_start = time.perf_counter()

    embed_start = time.perf_counter()
    embed_fn = get_embedding_modal_function(EMBED_TEXT_FUNCTION_NAME)
    query_vector = embed_fn.remote(query_text)
    embed_ms = (time.perf_counter() - embed_start) * 1000
    logger.debug("Got query embedding with %d dimensions", len(query_vector))

    qdrant_setup_start = time.perf_counter()
    qdrant_config = QdrantConfig.from_env()
    qdrant_store = QdrantStore(qdrant_config)
    qdrant_setup_ms = (time.perf_counter() - qdrant_setup_start) * 1000

    qdrant_search_start = time.perf_counter()
    results = qdrant_store.search(
        query_vector=query_vector,
        video_id=video_id,
        limit=limit,
    )
    qdrant_search_ms = (time.perf_counter() - qdrant_search_start) * 1000
    total_ms = (time.perf_counter() - total_start) * 1000

    logger.info(
        (
            "Search timing video_id=%s embed_ms=%.1f qdrant_setup_ms=%.1f "
            "qdrant_search_ms=%.1f total_ms=%.1f limit=%d results=%d"
        ),
        video_id,
        embed_ms,
        qdrant_setup_ms,
        qdrant_search_ms,
        total_ms,
        limit,
        len(results),
    )

    return results
