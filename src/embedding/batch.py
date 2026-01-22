from __future__ import annotations

from src.embedding.modal_app import embed_images_in_batches
from src.video.frames import FrameInfo


class EmbeddingError(RuntimeError):
    """Raised when embedding fails."""


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
        embeddings = embed_images_in_batches.remote(images, batch_size=batch_size)
    except Exception as exc:  # modal can raise various transport errors
        raise EmbeddingError("Modal embedding failed") from exc

    if len(embeddings) != len(frames):
        raise EmbeddingError(
            f"Embedding count mismatch: {len(embeddings)} != {len(frames)}"
        )

    return embeddings
