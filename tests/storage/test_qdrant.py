from __future__ import annotations

import pytest
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse

from src.storage.config import QdrantConfig
from src.storage.qdrant import (
    EMBEDDING_DIM,
    FrameVector,
    QDRANT_UPSERT_BATCH_SIZE,
    QdrantPayloadTooLargeError,
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


def test_replace_transcripts_removes_orphans_and_preserves_visual_points() -> None:
    config = QdrantConfig.in_memory(collection_name="replace_transcripts")
    store = QdrantStore(config)
    store.ensure_collection()

    store.upsert_frames(
        [
            FrameVector(
                video_id="video_a",
                frame_index=0,
                timestamp_s=0.0,
                vector=_vector(0.9),
                thumbnail_url="https://cdn/thumb/video_a/thumb_00000.jpg",
            )
        ]
    )
    store.upsert_transcripts(
        [
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
                text="orphan transcript hit",
                language_code="en",
            ),
        ]
    )

    replaced = store.replace_transcripts(
        "video_a",
        [
            TranscriptVector(
                video_id="video_a",
                segment_index=0,
                timestamp_s=6.0,
                end_s=8.0,
                vector=_vector(0.3),
                text="replacement transcript hit",
                language_code="en",
            )
        ],
    )

    assert replaced == 1

    transcript_results = store.search(
        _vector(0.3),
        video_id="video_a",
        limit=5,
        source="transcript",
    )
    assert len(transcript_results) == 1
    assert transcript_results[0].transcript_text == "replacement transcript hit"

    visual_results = store.search(
        _vector(0.9),
        video_id="video_a",
        limit=5,
        source="visual",
    )
    assert len(visual_results) == 1
    assert visual_results[0].source == "visual"


def test_replace_transcripts_empty_clears_only_transcript_points() -> None:
    config = QdrantConfig.in_memory(collection_name="replace_transcripts_empty")
    store = QdrantStore(config)
    store.ensure_collection()

    store.upsert_frames(
        [
            FrameVector(
                video_id="video_a",
                frame_index=0,
                timestamp_s=0.0,
                vector=_vector(0.9),
                thumbnail_url="https://cdn/thumb/video_a/thumb_00000.jpg",
            )
        ]
    )
    store.upsert_transcripts(
        [
            TranscriptVector(
                video_id="video_a",
                segment_index=0,
                timestamp_s=5.0,
                end_s=7.0,
                vector=_vector(0.1),
                text="first transcript hit",
                language_code="en",
            )
        ]
    )

    replaced = store.replace_transcripts("video_a", [])

    assert replaced == 0
    assert (
        store.search(
            _vector(0.1),
            video_id="video_a",
            limit=5,
            source="transcript",
        )
        == []
    )
    visual_results = store.search(
        _vector(0.9),
        video_id="video_a",
        limit=5,
        source="visual",
    )
    assert len(visual_results) == 1


def test_upsert_empty_returns_zero() -> None:
    config = QdrantConfig.in_memory(collection_name="empty_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    assert store.upsert_frames([]) == 0


def test_upsert_frames_batches_large_payloads(monkeypatch) -> None:
    config = QdrantConfig.in_memory(collection_name="batched_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    call_sizes: list[int] = []
    original_upsert = store._client.upsert
    shared_vector = _vector(0.1)

    def tracked_upsert(*args, **kwargs):
        call_sizes.append(len(kwargs["points"]))
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store._client, "upsert", tracked_upsert)

    frames = [
        FrameVector(
            video_id="video_batch",
            frame_index=idx,
            timestamp_s=float(idx),
            vector=shared_vector,
        )
        for idx in range(QDRANT_UPSERT_BATCH_SIZE + 1)
    ]

    upserted = store.upsert_frames(frames)

    assert upserted == QDRANT_UPSERT_BATCH_SIZE + 1
    assert call_sizes == [QDRANT_UPSERT_BATCH_SIZE, 1]


def test_upsert_frames_uses_single_call_below_batch_limit(monkeypatch) -> None:
    config = QdrantConfig.in_memory(collection_name="single_batch_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    call_sizes: list[int] = []
    original_upsert = store._client.upsert

    def tracked_upsert(*args, **kwargs):
        call_sizes.append(len(kwargs["points"]))
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store._client, "upsert", tracked_upsert)

    frames = [
        FrameVector(
            video_id="video_small",
            frame_index=idx,
            timestamp_s=float(idx),
            vector=_vector(0.2),
        )
        for idx in range(3)
    ]

    upserted = store.upsert_frames(frames)

    assert upserted == 3
    assert call_sizes == [3]


def test_upsert_frames_raises_specific_error_for_payload_limit(monkeypatch) -> None:
    config = QdrantConfig.in_memory(collection_name="payload_limit_frames")
    store = QdrantStore(config)
    store.ensure_collection()

    def raising_upsert(*args, **kwargs):
        raise UnexpectedResponse(
            status_code=400,
            reason_phrase="Bad Request",
            content=(
                b'{"status":{"error":"Payload error: JSON payload '
                b'(58419779 bytes) is larger than allowed '
                b'(limit: 33554432 bytes)."},"time":0.0}'
            ),
            headers=Headers(),
        )

    monkeypatch.setattr(store._client, "upsert", raising_upsert)

    with pytest.raises(QdrantPayloadTooLargeError, match="larger than allowed"):
        store.upsert_frames(
            [
                FrameVector(
                    video_id="video_payload",
                    frame_index=0,
                    timestamp_s=0.0,
                    vector=_vector(0.3),
                )
            ]
        )


def test_upsert_transcripts_batches_large_payloads(monkeypatch) -> None:
    config = QdrantConfig.in_memory(collection_name="batched_transcripts")
    store = QdrantStore(config)
    store.ensure_collection()

    call_sizes: list[int] = []
    original_upsert = store._client.upsert
    shared_vector = _vector(0.4)

    def tracked_upsert(*args, **kwargs):
        call_sizes.append(len(kwargs["points"]))
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store._client, "upsert", tracked_upsert)

    transcripts = [
        TranscriptVector(
            video_id="video_transcript_batch",
            segment_index=idx,
            timestamp_s=float(idx),
            end_s=float(idx + 1),
            vector=shared_vector,
            text=f"segment {idx}",
            language_code="en",
        )
        for idx in range(QDRANT_UPSERT_BATCH_SIZE + 1)
    ]

    upserted = store.upsert_transcripts(transcripts)

    assert upserted == QDRANT_UPSERT_BATCH_SIZE + 1
    assert call_sizes == [QDRANT_UPSERT_BATCH_SIZE, 1]


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
