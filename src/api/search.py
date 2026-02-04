"""Search service for video moment finder."""
from __future__ import annotations

from src.embedding.modal_app import embed_text
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

    query_vector = embed_text.remote(query_text)
    logger.debug("Got query embedding with %d dimensions", len(query_vector))

    qdrant_config = QdrantConfig.from_env()
    qdrant_store = QdrantStore(qdrant_config)

    results = qdrant_store.search(
        query_vector=query_vector,
        video_id=video_id,
        limit=limit,
    )
    logger.info("Found %d results in Qdrant", len(results))

    return results
