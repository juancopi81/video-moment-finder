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
from src.pipeline.orchestrator import StoragePipeline
from src.storage.config import QdrantConfig, R2Config, StorageConfigError
from src.video.frames import extract_frames


LOCAL_VIDEO_PATH = Path("test_video.mp4")


def load_configs() -> tuple[QdrantConfig, R2Config | None]:
    """Load storage configs from environment."""
    try:
        qdrant_config = QdrantConfig.from_env()
        print(f"Qdrant URL: {qdrant_config.url}")
    except StorageConfigError as e:
        print(f"ERROR: {e}")
        print("Set QDRANT_URL and optionally QDRANT_API_KEY")
        sys.exit(1)

    r2_config = None
    try:
        r2_config = R2Config.from_env()
        print(f"R2 bucket: {r2_config.bucket_name}")
    except StorageConfigError as e:
        print(f"WARNING: R2 not configured ({e})")
        print("Thumbnails will not be uploaded.")

    return qdrant_config, r2_config


def main(max_frames: int = 30) -> None:
    print("Phase 1.4: Full Storage Integration Test")
    print("=" * 60)

    if not LOCAL_VIDEO_PATH.exists():
        print(f"ERROR: No test video found at {LOCAL_VIDEO_PATH}")
        print('Download a video first: uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "URL"')
        sys.exit(1)

    # Load configs
    qdrant_config, r2_config = load_configs()
    print()

    # Initialize pipeline
    pipeline = StoragePipeline(qdrant_config, r2_config)
    pipeline.ensure_ready()
    print("Storage backends ready.\n")

    video_id = f"integration_test_{int(time.time())}"
    print(f"Video ID: {video_id}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        frames_dir = temp_path / "frames"
        thumbs_dir = temp_path / "thumbnails"

        # Step 1: Extract frames
        print("\n1. Extracting frames...")
        start = time.time()
        frames = extract_frames(
            LOCAL_VIDEO_PATH,
            frames_dir,
            fps=1.0,
            thumbnail_dir=thumbs_dir,
        )
        if len(frames) > max_frames:
            frames = frames[:max_frames]
        extract_time = time.time() - start
        print(f"   Extracted {len(frames)} frames in {extract_time:.2f}s")

        # Step 2: Embed frames
        print("\n2. Embedding frames (Modal GPU)...")
        start = time.time()
        embeddings = embed_frames(frames, batch_size=8)
        embed_time = time.time() - start
        print(f"   Embedded {len(embeddings)} frames in {embed_time:.2f}s")

        # Step 3: Process (upload + store)
        print("\n3. Processing (upload thumbnails, store embeddings)...")
        start = time.time()
        result = pipeline.process_video(video_id, frames, embeddings)
        process_time = time.time() - start
        print(f"   Frames processed: {result.frames_processed}")
        print(f"   Embeddings stored: {result.embeddings_stored}")
        print(f"   Thumbnails uploaded: {result.thumbnails_uploaded}")
        print(f"   Processing time: {process_time:.2f}s")

        # Step 4: Test search
        print("\n4. Testing search...")
        query_vector = embeddings[0]  # Use first frame as query
        search_results = pipeline._qdrant.search(query_vector, video_id, limit=5)
        print(f"   Found {len(search_results)} results:")
        for r in search_results:
            url_preview = r.thumbnail_url[:50] + "..." if len(r.thumbnail_url) > 50 else r.thumbnail_url
            print(f"   - Frame {r.frame_index} @ {r.timestamp_s:.1f}s (score: {r.score:.4f})")
            print(f"     URL: {url_preview}")

        # Step 5: Cleanup
        print("\n5. Cleaning up test data...")
        deleted_embeddings, deleted_thumbnails = pipeline.delete_video(video_id)
        print(f"   Deleted {deleted_embeddings} embeddings, {deleted_thumbnails} thumbnails")

    # Summary
    print("\n" + "=" * 60)
    total_time = extract_time + embed_time + process_time
    print(f"Total time: {total_time:.2f}s")
    print(f"  - Extract: {extract_time:.2f}s ({extract_time/total_time*100:.0f}%)")
    print(f"  - Embed: {embed_time:.2f}s ({embed_time/total_time*100:.0f}%)")
    print(f"  - Process: {process_time:.2f}s ({process_time/total_time*100:.0f}%)")
    print("\nIntegration test PASSED!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Storage integration test")
    parser.add_argument("--max-frames", type=int, default=30, help="Maximum frames to process")
    args = parser.parse_args()
    main(max_frames=args.max_frames)
