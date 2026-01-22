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
from src.video.frames import FrameInfo


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
    print("Phase 1.4: Local Storage Test")
    print("=" * 50)

    # Use in-memory Qdrant
    qdrant_config = QdrantConfig.in_memory_config()
    pipeline = StoragePipeline(qdrant_config, r2_config=None)

    print("\n1. Ensuring collection exists...")
    pipeline.ensure_ready()
    print("   Collection ready.")

    # Create mock data
    video_id = "test_video_123"
    frame_count = 10

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n2. Creating {frame_count} mock frames...")
        frames = create_mock_frames(frame_count, temp_path)
        embeddings = create_mock_embeddings(frame_count)

        print(f"\n3. Processing video '{video_id}'...")
        result = pipeline.process_video(video_id, frames, embeddings)

        print(f"   Frames processed: {result.frames_processed}")
        print(f"   Embeddings stored: {result.embeddings_stored}")
        print(f"   Thumbnails uploaded: {result.thumbnails_uploaded} (R2 disabled)")

    # Test search
    print("\n4. Testing search...")
    query_vector = create_mock_embeddings(1)[0]
    search_results = pipeline._qdrant.search(query_vector, video_id, limit=3)
    print(f"   Found {len(search_results)} results:")
    for r in search_results:
        print(f"   - Frame {r.frame_index} @ {r.timestamp_s:.1f}s (score: {r.score:.4f})")

    # Test delete
    print("\n5. Testing delete...")
    deleted_embeddings, deleted_thumbnails = pipeline.delete_video(video_id)
    print(f"   Deleted {deleted_embeddings} embeddings, {deleted_thumbnails} thumbnails")

    # Verify deletion
    search_results = pipeline._qdrant.search(query_vector, video_id, limit=3)
    print(f"   Search after delete: {len(search_results)} results")

    print("\n" + "=" * 50)
    print("Local storage test PASSED!")


if __name__ == "__main__":
    main()
