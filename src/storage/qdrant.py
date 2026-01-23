"""Qdrant vector database storage for frame embeddings."""
from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.storage.config import QdrantConfig


EMBEDDING_DIM = 2048  # Qwen3-VL-Embedding-2B
NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class QdrantStorageError(RuntimeError):
    """Raised when Qdrant operations fail."""


@dataclass(frozen=True)
class FrameVector:
    """Frame embedding data for storage."""

    video_id: str
    frame_index: int
    timestamp_s: float
    vector: list[float]
    thumbnail_url: str


@dataclass(frozen=True)
class SearchResult:
    """Search result from Qdrant."""

    video_id: str
    frame_index: int
    timestamp_s: float
    thumbnail_url: str
    score: float


def generate_point_id(video_id: str, frame_index: int) -> str:
    """Generate deterministic UUID5 point ID from video_id and frame_index."""
    name = f"{video_id}_{frame_index}"
    return str(uuid.uuid5(NAMESPACE_UUID, name))


class QdrantStore:
    """Qdrant vector storage for frame embeddings."""

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        if config.in_memory:
            self._client = QdrantClient(":memory:")
        else:
            self._client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
            )

    def ensure_collection(self) -> None:
        """Create collection if it doesn't exist (idempotent)."""
        collections = self._client.get_collections().collections
        exists = any(c.name == self._config.collection_name for c in collections)

        if not exists:
            try:
                self._client.create_collection(
                    collection_name=self._config.collection_name,
                    vectors_config=models.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=models.Distance.COSINE,
                    ),
                )
            except Exception as exc:
                raise QdrantStorageError(f"Failed to create collection: {exc}") from exc

    def upsert_frames(self, frames: list[FrameVector]) -> int:
        """Upsert frame vectors to Qdrant. Returns count of upserted points."""
        if not frames:
            return 0

        points = [
            models.PointStruct(
                id=generate_point_id(frame.video_id, frame.frame_index),
                vector=frame.vector,
                payload={
                    "video_id": frame.video_id,
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "thumbnail_url": frame.thumbnail_url,
                },
            )
            for frame in frames
        ]

        try:
            self._client.upsert(
                collection_name=self._config.collection_name,
                points=points,
            )
        except Exception as exc:
            raise QdrantStorageError(f"Failed to upsert frames: {exc}") from exc

        return len(points)

    def search(
        self,
        query_vector: list[float],
        video_id: str,
        limit: int = 5,
    ) -> list[SearchResult]:
        """Search for similar frames within a video."""
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="video_id",
                    match=models.MatchValue(value=video_id),
                )
            ]
        )

        try:
            if hasattr(self._client, "query_points"):
                response = self._client.query_points(
                    collection_name=self._config.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
                points = response.points
            else:
                points = self._client.search(
                    collection_name=self._config.collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    with_payload=True,
                )
        except Exception as exc:
            raise QdrantStorageError(f"Search failed: {exc}") from exc

        return [
            SearchResult(
                video_id=point.payload["video_id"],  # type: ignore[index]
                frame_index=point.payload["frame_index"],  # type: ignore[index]
                timestamp_s=point.payload["timestamp_s"],  # type: ignore[index]
                thumbnail_url=point.payload["thumbnail_url"],  # type: ignore[index]
                score=point.score,
            )
            for point in points
        ]

    def delete_video(self, video_id: str) -> int:
        """Delete all vectors for a video. Returns count of deleted points."""
        try:
            # First count existing points for this video
            count_result = self._client.count(
                collection_name=self._config.collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id),
                        )
                    ]
                ),
            )
            count = count_result.count

            if count > 0:
                self._client.delete(
                    collection_name=self._config.collection_name,
                    points_selector=models.FilterSelector(
                        filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="video_id",
                                    match=models.MatchValue(value=video_id),
                                )
                            ]
                        )
                    ),
                )

            return count
        except Exception as exc:
            raise QdrantStorageError(f"Failed to delete video: {exc}") from exc
