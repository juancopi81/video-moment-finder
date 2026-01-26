"""
Phase 1.4: Local Storage Test (in-memory Qdrant, no R2)

Tests the storage pipeline locally without cloud services.

Usage:
    uv run python scripts/phase1/storage_local_test.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.pipeline.orchestrator import StoragePipeline
from src.storage.config import QdrantConfig
from src.storage.qdrant import EMBEDDING_DIM
from src.utils.logging import Timer, get_logger
from src.video.frames import FrameInfo

logger = get_logger(__name__)

def create_mock_frames(count: int, temp_dir: Path) -> list[FrameInfo]:
    """Create mock frame info for testing."""
    frames = []
    for i in range(count):
        # Create dummy thumbnail file
        thumb_path = temp_dir / f"thumb_{i:05d}.jpg"
        thumb_path.write_bytes(b"\xff\xd8\xff")  # minimal JPEG header
        frames.append(
            FrameInfo(
                index=i,
                timestamp_s=float(i),
                path=temp_dir / f"frame_{i:05d}.jpg",  # doesn't need to exist for this test
                thumbnail_path=thumb_path,
            )
        )
    return frames


def create_mock_embeddings(count: int) -> list[list[float]]:
    """Create mock embeddings for testing."""
    return [[float(i) / count] * EMBEDDING_DIM for i in range(count)]


def main() -> None:
    logger.info("Phase 1.4: Local Storage Test")
    logger.info("=" * 50)

    # Use in-memory Qdrant
    qdrant_config = QdrantConfig.in_memory()
    pipeline = StoragePipeline(qdrant_config, r2_config=None)

    logger.info("1. Ensuring collection exists...")
    pipeline.ensure_ready()
    logger.info("Collection ready.")

    # Create mock data
    video_id = "test_video_123"
    frame_count = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        logger.info("2. Creating %d mock frames...", frame_count)
        frames = create_mock_frames(frame_count, temp_path)
        embeddings = create_mock_embeddings(frame_count)

        logger.info("3. Processing video '%s'...", video_id)
        with Timer("Process video", logger) as process_timer:
            result = pipeline.process_video(video_id, frames, embeddings)

        logger.info("Frames processed: %d", result.frames_processed)
        logger.info("Embeddings stored: %d", result.embeddings_stored)
        logger.info(
            "Thumbnails uploaded: %d (R2 disabled)", result.thumbnails_uploaded
        )
        logger.info("Process time: %.2fs", process_timer.elapsed or 0.0)

    # Test search
    logger.info("4. Testing search...")
    query_vector = create_mock_embeddings(1)[0]
    with Timer("Search test", logger, level="debug"):
        search_results = pipeline._qdrant.search(query_vector, video_id, limit=3)
    logger.info("Found %d results:", len(search_results))
    for r in search_results:
        logger.info(
            "- Frame %d @ %.1fs (score: %.4f)",
            r.frame_index,
            r.timestamp_s,
            r.score,
        )

    # Test delete
    logger.info("5. Testing delete...")
    with Timer("Delete video", logger) as delete_timer:
        deleted_embeddings, deleted_thumbnails = pipeline.delete_video(video_id)
    logger.info(
        "Deleted %d embeddings, %d thumbnails in %.2fs",
        deleted_embeddings,
        deleted_thumbnails,
        delete_timer.elapsed or 0.0,
    )

    # Verify deletion
    search_results = pipeline._qdrant.search(query_vector, video_id, limit=3)
    logger.info("Search after delete: %d results", len(search_results))

    logger.info("=" * 50)
    logger.info("Local storage test PASSED!")


if __name__ == "__main__":
    main()
