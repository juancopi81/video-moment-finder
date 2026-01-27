"""
Phase 1.4: Full Storage Integration Test (Qdrant Cloud + R2)

Tests the complete flow: extract frames -> embed -> upload thumbnails -> store vectors -> search -> cleanup

Prerequisites:
    - Set environment variables (see ROADMAP.md for details):
      - QDRANT_URL, QDRANT_API_KEY
      - R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME
    - Have a test video at test_video.mp4

Usage:
    uv run python scripts/phase1/storage_integration_test.py
    uv run python scripts/phase1/storage_integration_test.py --max-frames 30
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from src.embedding.batch import embed_frames
from src.embedding.modal_app import app
from src.pipeline.orchestrator import StoragePipeline
from src.config.env import load_env
from src.storage.config import QdrantConfig, R2Config, StorageConfigError
from src.utils.logging import Timer, get_logger
from src.video.frames import extract_frames

LOCAL_VIDEO_PATH = Path("test_video.mp4")

load_env(required=True)
logger = get_logger(__name__)


def load_configs() -> tuple[QdrantConfig, R2Config | None]:
    """Load storage configs from environment."""
    try:
        qdrant_config = QdrantConfig.from_env()
        logger.info("Qdrant URL: %s", qdrant_config.url)
    except StorageConfigError as e:
        logger.error("Qdrant config error: %s", e)
        logger.error("Set QDRANT_URL and optionally QDRANT_API_KEY")
        sys.exit(1)

    r2_config = None
    try:
        r2_config = R2Config.from_env()
        logger.info("R2 bucket: %s", r2_config.bucket_name)
    except StorageConfigError as e:
        logger.warning("R2 not configured (%s)", e)
        logger.warning("Thumbnails will not be uploaded.")

    return qdrant_config, r2_config


def main(max_frames: int = 30) -> None:
    logger.info("Phase 1.4: Full Storage Integration Test")
    logger.info("=" * 60)

    if not LOCAL_VIDEO_PATH.exists():
        logger.error("No test video found at %s", LOCAL_VIDEO_PATH)
        logger.error(
            'Download a video first: uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "URL"'
        )
        sys.exit(1)

    # Load configs
    qdrant_config, r2_config = load_configs()
    # Initialize pipeline
    pipeline = StoragePipeline(qdrant_config, r2_config)
    pipeline.ensure_ready()
    logger.info("Storage backends ready.")

    video_id = f"integration_test_{int(time.time())}"
    logger.info("Video ID: %s", video_id)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        thumbs_dir = temp_path / "thumbnails"

        # Step 1: Extract frames
        logger.info("1. Extracting frames...")
        with Timer("Extract frames", logger) as extract_timer:
            frames = extract_frames(
                LOCAL_VIDEO_PATH,
                frames_dir,
                fps=1.0,
                thumbnail_dir=thumbs_dir,
            )
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        extract_time = extract_timer.elapsed or 0.0
        logger.info("Extracted %d frames in %.2fs", len(frames), extract_time)

        # Step 2: Embed frames
        logger.info("2. Embedding frames (Modal GPU)...")
        with Timer("Embed frames", logger) as embed_timer:
            with app.run():
                embeddings = embed_frames(frames, batch_size=8)
        embed_time = embed_timer.elapsed or 0.0
        logger.info("Embedded %d frames in %.2fs", len(embeddings), embed_time)

        # Step 3: Process (upload + store)
        logger.info("3. Processing (upload thumbnails, store embeddings)...")
        with Timer("Store embeddings + thumbnails", logger) as process_timer:
            result = pipeline.process_video(video_id, frames, embeddings)
        process_time = process_timer.elapsed or 0.0
        logger.info("Frames processed: %d", result.frames_processed)
        logger.info("Embeddings stored: %d", result.embeddings_stored)
        logger.info("Thumbnails uploaded: %d", result.thumbnails_uploaded)
        logger.info("Processing time: %.2fs", process_time)

        # Step 4: Test search
        logger.info("4. Testing search...")
        query_vector = embeddings[0]  # Use first frame as query
        with Timer("Search test", logger, level="debug"):
            search_results = pipeline._qdrant.search(query_vector, video_id, limit=5)
        logger.info("Found %d results:", len(search_results))
        for r in search_results:
            url_preview = (
                r.thumbnail_url[:50] + "..."
                if len(r.thumbnail_url) > 50
                else r.thumbnail_url
            )
            logger.info(
                "- Frame %d @ %.1fs (score: %.4f)",
                r.frame_index,
                r.timestamp_s,
                r.score,
            )
            logger.info("  URL: %s", url_preview)

        # Step 5: Cleanup
        logger.info("5. Cleaning up test data...")
        with Timer("Cleanup test data", logger) as cleanup_timer:
            deleted_embeddings, deleted_thumbnails = pipeline.delete_video(video_id)
        logger.info(
            "Deleted %d embeddings, %d thumbnails in %.2fs",
            deleted_embeddings,
            deleted_thumbnails,
            cleanup_timer.elapsed or 0.0,
        )

    # Summary
    logger.info("=" * 60)
    total_time = extract_time + embed_time + process_time
    logger.info("Total time: %.2fs", total_time)
    if total_time > 0:
        logger.info("  - Extract: %.2fs (%.0f%%)", extract_time, extract_time / total_time * 100)
        logger.info("  - Embed: %.2fs (%.0f%%)", embed_time, embed_time / total_time * 100)
        logger.info("  - Process: %.2fs (%.0f%%)", process_time, process_time / total_time * 100)
    logger.info("Integration test PASSED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage integration test")
    parser.add_argument(
        "--max-frames", type=int, default=30, help="Maximum frames to process"
    )
    args = parser.parse_args()
    main(max_frames=args.max_frames)
