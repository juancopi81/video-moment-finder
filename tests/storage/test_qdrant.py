from __future__ import annotations

from src.storage.config import QdrantConfig
from src.storage.qdrant import (
    EMBEDDING_DIM,
    FrameVector,
    QdrantStore,
    generate_point_id,
    generate_transcript_point_id,
    TranscriptVector,
)


def _vector(value: float) -> list[float]:
    return [value] * EMBEDDING_DIM


def test_generate_point_id_deterministic() -> None:
    first = generate_point_id("video_a", 1)
    second = generate_point_id("video_a", 1)
    different = generate_point_id("video_a", 2)
    assert first == second
    assert first != different


def test_generate_transcript_point_id_deterministic() -> None:
    first = generate_transcript_point_id("video_a", 1)
    second = generate_transcript_point_id("video_a", 1)
    different = generate_transcript_point_id("video_a", 2)
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
            thumbnail_url="https://cdn/thumb/video_a/thumb_00000.jpg",
        ),
        FrameVector(
            video_id="video_a",
            frame_index=1,
            timestamp_s=1.0,
            vector=_vector(0.2),
            thumbnail_url="https://cdn/thumb/video_a/thumb_00001.jpg",
        ),
        FrameVector(
            video_id="video_b",
            frame_index=0,
            timestamp_s=0.0,
            vector=_vector(0.3),
            thumbnail_url="https://cdn/thumb/video_b/thumb_00000.jpg",
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


def test_upsert_transcripts_search_in_memory() -> None:
    config = QdrantConfig.in_memory(collection_name="test_transcripts")
    store = QdrantStore(config)
    store.ensure_collection()

    transcripts = [
        TranscriptVector(
            video_id="video_a",
            segment_index=0,
            timestamp_s=5.0,
            end_s=7.0,
            vector=_vector(0.1),
            text="first transcript hit",
            language_code="en",
        ),
        TranscriptVector(
            video_id="video_a",
            segment_index=1,
            timestamp_s=9.0,
            end_s=12.0,
            vector=_vector(0.2),
            text="second transcript hit",
            language_code="en",
        ),
    ]

    upserted = store.upsert_transcripts(transcripts)
    assert upserted == 2

    results = store.search(
        _vector(0.1),
        video_id="video_a",
        limit=5,
        source="transcript",
    )
    assert len(results) == 2
    assert all(result.source == "transcript" for result in results)
    assert results[0].frame_index == -1
    assert results[0].transcript_text is not None


def test_upsert_empty_returns_zero() -> None:
    config = QdrantConfig.in_memory(collection_name="empty_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    assert store.upsert_frames([]) == 0


def test_ensure_collection_creates_video_id_payload_index(monkeypatch) -> None:
    config = QdrantConfig.in_memory(collection_name="index_frames")
    store = QdrantStore(config)

    calls: list[tuple[str, str]] = []
    original_create_payload_index = store._client.create_payload_index

    def tracked_create_payload_index(*args, **kwargs):
        calls.append((kwargs["collection_name"], kwargs["field_name"]))
        return original_create_payload_index(*args, **kwargs)

    monkeypatch.setattr(store._client, "create_payload_index", tracked_create_payload_index)

    store.ensure_collection()

    assert calls
    assert ("index_frames", "video_id") in calls
    assert ("index_frames", "source") in calls
