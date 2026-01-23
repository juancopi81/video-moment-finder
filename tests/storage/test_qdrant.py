from __future__ import annotations

from src.storage.config import QdrantConfig
from src.storage.qdrant import (
    EMBEDDING_DIM,
    FrameVector,
    QdrantStore,
    generate_point_id,
)


def _vector(value: float) -> list[float]:
    return [value] * EMBEDDING_DIM


def test_generate_point_id_deterministic() -> None:
    first = generate_point_id("video_a", 1)
    second = generate_point_id("video_a", 1)
    different = generate_point_id("video_a", 2)
    assert first == second
    assert first != different


def test_upsert_search_delete_in_memory() -> None:
    config = QdrantConfig.in_memory(collection_name="test_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    frames = [
        FrameVector(
            video_id="video_a",
            frame_index=0,
            timestamp_s=0.0,
            vector=_vector(0.1),
            thumbnail_url="https://cdn/video_a/thumb_00000.jpg",
        ),
        FrameVector(
            video_id="video_a",
            frame_index=1,
            timestamp_s=1.0,
            vector=_vector(0.2),
            thumbnail_url="https://cdn/video_a/thumb_00001.jpg",
        ),
        FrameVector(
            video_id="video_b",
            frame_index=0,
            timestamp_s=0.0,
            vector=_vector(0.3),
            thumbnail_url="https://cdn/video_b/thumb_00000.jpg",
        ),
    ]

    upserted = store.upsert_frames(frames)
    assert upserted == 3

    results = store.search(_vector(0.1), video_id="video_a", limit=5)
    assert results
    assert all(result.video_id == "video_a" for result in results)

    deleted = store.delete_video("video_a")
    assert deleted == 2

    results_after_delete = store.search(_vector(0.1), video_id="video_a", limit=5)
    assert results_after_delete == []


def test_upsert_empty_returns_zero() -> None:
    config = QdrantConfig.in_memory(collection_name="empty_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    assert store.upsert_frames([]) == 0
