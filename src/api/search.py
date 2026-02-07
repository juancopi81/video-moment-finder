"""Search service for video moment finder."""
from __future__ import annotations

from src.config.modal import EMBED_TEXT_FUNCTION_NAME, get_embedding_modal_function
from src.storage.config import QdrantConfig
from src.storage.qdrant import QdrantStore, SearchResult
from src.utils.logging import Timer, get_logger

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

    with Timer("Search total", logger, level="debug") as total_timer:
        with Timer("Search embed query", logger, level="debug") as embed_timer:
            embed_fn = get_embedding_modal_function(EMBED_TEXT_FUNCTION_NAME)
            query_vector = embed_fn.remote(query_text)
        logger.debug("Got query embedding with %d dimensions", len(query_vector))

        with Timer("Search qdrant setup", logger, level="debug") as qdrant_setup_timer:
            qdrant_config = QdrantConfig.from_env()
            qdrant_store = QdrantStore(qdrant_config)

        with Timer("Search qdrant query", logger, level="debug") as qdrant_search_timer:
            results = qdrant_store.search(
                query_vector=query_vector,
                video_id=video_id,
                limit=limit,
            )

    logger.info(
        (
            "Search timing video_id=%s embed_ms=%.1f qdrant_setup_ms=%.1f "
            "qdrant_search_ms=%.1f total_ms=%.1f limit=%d results=%d"
        ),
        video_id,
        (embed_timer.elapsed or 0.0) * 1000,
        (qdrant_setup_timer.elapsed or 0.0) * 1000,
        (qdrant_search_timer.elapsed or 0.0) * 1000,
        (total_timer.elapsed or 0.0) * 1000,
        limit,
        len(results),
    )

    return results
