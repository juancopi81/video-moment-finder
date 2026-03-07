"""Search service for video moment finder."""
from __future__ import annotations

import io
import re

from PIL import Image, UnidentifiedImageError

from src.config.modal import get_query_embedder_class, raise_modal_auth_error
from src.db.supabase import search_video_transcript_segments
from src.storage.config import QdrantConfig
from src.storage.qdrant import QdrantStore, SearchResult
from src.utils.logging import Timer, get_logger

logger = get_logger(__name__)

_TRANSCRIPT_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_TRANSCRIPT_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "at",
    "be",
    "been",
    "being",
    "did",
    "do",
    "does",
    "find",
    "for",
    "he",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "mention",
    "mentioned",
    "mentions",
    "of",
    "on",
    "or",
    "said",
    "say",
    "says",
    "she",
    "show",
    "talk",
    "talked",
    "talking",
    "that",
    "the",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
}
_TRANSCRIPT_INTENT_TOKENS = {
    "explain",
    "explains",
    "explained",
    "explanation",
    "mention",
    "mentions",
    "mentioned",
    "say",
    "says",
    "said",
    "talk",
    "talks",
    "talked",
    "talking",
    "discuss",
    "discusses",
    "discussed",
    "describe",
    "describes",
    "described",
    "tell",
    "tells",
    "told",
    "speech",
    "speaks",
    "speaking",
}


class QueryImageValidationError(ValueError):
    """Raised when an uploaded search image is invalid."""


def search_video_by_text(
    video_id: str,
    query_text: str,
    limit: int = 5,
) -> list[SearchResult]:
    """Search text queries across transcript and visual retrieval paths."""
    logger.info(
        "Searching video_id=%s query_type=text query=%r limit=%d",
        video_id,
        query_text[:50],
        limit,
    )

    transcript_results: list[SearchResult] = []
    transcript_timer: Timer | None = None
    transcript_error: Exception | None = None
    embed_timer: Timer | None = None
    visual_results: list[SearchResult] = []
    qdrant_setup_timer: Timer | None = None
    qdrant_search_timer: Timer | None = None

    with Timer("Search total", logger, level="debug") as total_timer:
        transcript_query = _normalize_transcript_query(query_text)
        if transcript_query is not None:
            try:
                with Timer(
                    "Search transcript query",
                    logger,
                    level="debug",
                ) as transcript_timer:
                    transcript_results = _search_transcript_results(
                        video_id=video_id,
                        query_text=transcript_query,
                        limit=limit,
                    )
                if not transcript_results:
                    transcript_timer = None
            except Exception as exc:
                transcript_error = exc
                transcript_timer = None
                logger.warning(
                    "Transcript search unavailable for video_id=%s: %s",
                    video_id,
                    exc,
                )

        try:
            with Timer("Search embed query", logger, level="debug") as embed_timer:
                query_embedder = get_query_embedder_class()
                query_vector = query_embedder().embed_text.remote(query_text)
        except Exception as exc:
            if not transcript_results:
                raise_modal_auth_error(exc, context="embedding a text search query")
                raise
            logger.warning(
                "Visual text embedding unavailable for video_id=%s: %s",
                video_id,
                exc,
            )
        else:
            logger.debug(
                "Got text query embedding with %d dimensions",
                len(query_vector),
            )

            try:
                (
                    visual_results,
                    qdrant_setup_timer,
                    qdrant_search_timer,
                ) = _search_with_query_vector(
                    video_id=video_id,
                    query_vector=query_vector,
                    limit=limit,
                )
            except Exception as exc:
                if not transcript_results:
                    raise
                logger.warning(
                    "Visual text search unavailable for video_id=%s: %s",
                    video_id,
                    exc,
                )

    results = _merge_text_search_results(
        query_text=query_text,
        transcript_results=transcript_results,
        visual_results=visual_results,
    )
    if not results and transcript_error is not None:
        raise transcript_error

    logger.info(
        (
            "Search timing video_id=%s query_type=text transcript_ms=%.1f "
            "embed_ms=%.1f qdrant_setup_ms=%.1f qdrant_search_ms=%.1f "
            "total_ms=%.1f limit=%d transcript_results=%d visual_results=%d "
            "results=%d"
        ),
        video_id,
        _elapsed_ms(transcript_timer),
        _elapsed_ms(embed_timer),
        _elapsed_ms(qdrant_setup_timer),
        _elapsed_ms(qdrant_search_timer),
        (total_timer.elapsed or 0.0) * 1000,
        limit,
        len(transcript_results),
        len(visual_results),
        len(results),
    )

    return results


def search_video_by_image(
    video_id: str,
    query_image_bytes: bytes,
    limit: int = 5,
) -> list[SearchResult]:
    """Embed a query image via Modal, search Qdrant, and return results."""
    _validate_query_image_bytes(query_image_bytes)
    logger.info("Searching video_id=%s query_type=image limit=%d", video_id, limit)

    with Timer("Search total", logger, level="debug") as total_timer:
        with Timer("Search embed query", logger, level="debug") as embed_timer:
            query_embedder = get_query_embedder_class()
            try:
                query_vector = query_embedder().embed_image.remote(query_image_bytes)
            except Exception as exc:
                raise_modal_auth_error(exc, context="embedding an image search query")
                raise
        logger.debug("Got image query embedding with %d dimensions", len(query_vector))

        results, qdrant_setup_timer, qdrant_search_timer = _search_with_query_vector(
            video_id=video_id,
            query_vector=query_vector,
            limit=limit,
        )

    _log_search_timing(
        video_id=video_id,
        query_type="image",
        embed_timer=embed_timer,
        qdrant_setup_timer=qdrant_setup_timer,
        qdrant_search_timer=qdrant_search_timer,
        total_timer=total_timer,
        limit=limit,
        results_count=len(results),
    )

    return results


def _validate_query_image_bytes(image_bytes: bytes) -> None:
    if not image_bytes:
        raise QueryImageValidationError("Uploaded image is empty")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.verify()
    except (SyntaxError, UnidentifiedImageError, OSError) as exc:
        raise QueryImageValidationError("Uploaded file is not a valid image") from exc


def _search_transcript_results(
    *,
    video_id: str,
    query_text: str,
    limit: int,
) -> list[SearchResult]:
    segments = search_video_transcript_segments(
        video_id,
        query_text,
        limit=limit,
    )
    return [
        SearchResult(
            video_id=segment.video_id,
            frame_index=-1,
            timestamp_s=segment.start_s,
            thumbnail_url=None,
            score=segment.score or 0.0,
            source="transcript",
            transcript_text=segment.text,
        )
        for segment in segments
    ]


def _search_with_query_vector(
    *,
    video_id: str,
    query_vector: list[float],
    limit: int,
) -> tuple[list[SearchResult], Timer, Timer]:
    with Timer("Search qdrant setup", logger, level="debug") as qdrant_setup_timer:
        qdrant_config = QdrantConfig.from_env()
        qdrant_store = QdrantStore(qdrant_config)

    with Timer("Search qdrant query", logger, level="debug") as qdrant_search_timer:
        results = qdrant_store.search(
            query_vector=query_vector,
            video_id=video_id,
            limit=limit,
        )

    return results, qdrant_setup_timer, qdrant_search_timer


def _log_search_timing(
    *,
    video_id: str,
    query_type: str,
    embed_timer: Timer,
    qdrant_setup_timer: Timer,
    qdrant_search_timer: Timer,
    total_timer: Timer,
    limit: int,
    results_count: int,
) -> None:
    logger.info(
        (
            "Search timing video_id=%s query_type=%s embed_ms=%.1f "
            "qdrant_setup_ms=%.1f qdrant_search_ms=%.1f total_ms=%.1f "
            "limit=%d results=%d"
        ),
        video_id,
        query_type,
        (embed_timer.elapsed or 0.0) * 1000,
        (qdrant_setup_timer.elapsed or 0.0) * 1000,
        (qdrant_search_timer.elapsed or 0.0) * 1000,
        (total_timer.elapsed or 0.0) * 1000,
        limit,
        results_count,
    )


def _elapsed_ms(timer: Timer | None) -> float:
    return (timer.elapsed or 0.0) * 1000 if timer is not None else 0.0


def _merge_text_search_results(
    *,
    query_text: str,
    transcript_results: list[SearchResult],
    visual_results: list[SearchResult],
) -> list[SearchResult]:
    if _prefers_transcript_first(query_text):
        ordered_groups = (transcript_results, visual_results)
    else:
        ordered_groups = (visual_results, transcript_results)
    return [result for group in ordered_groups for result in group]


def _prefers_transcript_first(query_text: str) -> bool:
    normalized = query_text.lower()
    return any(token in normalized for token in _TRANSCRIPT_INTENT_TOKENS)


def _normalize_transcript_query(query_text: str) -> str | None:
    tokens = []
    seen: set[str] = set()

    for raw_token in _TRANSCRIPT_QUERY_TOKEN_RE.findall(query_text.lower()):
        token = raw_token.strip("'")
        if len(token) < 2 or token in _TRANSCRIPT_STOP_WORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)

    if not tokens:
        return None
    return " ".join(tokens)
