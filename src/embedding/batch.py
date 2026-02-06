from __future__ import annotations

from src.config.modal import (
    EMBED_IMAGES_FUNCTION_NAME,
    get_embedding_modal_function,
)
from src.utils.logging import Timer, get_logger
from src.video.frames import FrameInfo


class EmbeddingError(RuntimeError):
    """Raised when embedding fails."""

logger = get_logger(__name__)


def embed_frames(
    frames: list[FrameInfo],
    *,
    batch_size: int = 8,
) -> list[list[float]]:
    """
    Embed frames using Modal in fixed-size batches.

    Fail-fast behavior:
    - Raises EmbeddingError if any frame path is missing.
    - Raises EmbeddingError if Modal call fails.
    - Raises EmbeddingError if result size mismatches input.
    """
    if not frames:
        raise EmbeddingError("frames must be a non-empty list")

    for frame in frames:
        if not frame.path.exists():
            raise EmbeddingError(f"Frame not found: {frame.path}")

    images = [frame.path.read_bytes() for frame in frames]

    try:
        logger.info("Embedding %d frames (batch_size=%d)", len(images), batch_size)
        with Timer("Modal embedding call", logger) as embed_timer:
            embed_fn = get_embedding_modal_function(EMBED_IMAGES_FUNCTION_NAME)
            embeddings = embed_fn.remote(images, batch_size=batch_size)
    except Exception as exc:  # modal can raise various transport errors
        raise EmbeddingError("Modal embedding failed") from exc

    logger.info(
        "Received %d embeddings in %.2fs",
        len(embeddings),
        embed_timer.elapsed or 0.0,
    )

    if len(embeddings) != len(frames):
        raise EmbeddingError(
            f"Embedding count mismatch: {len(embeddings)} != {len(frames)}"
        )

    return embeddings
