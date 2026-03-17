"""Qdrant vector database storage for frame embeddings."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import NoReturn, Literal

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http import models

from src.storage.config import QdrantConfig
from src.utils.logging import get_logger


EMBEDDING_DIM = 2048  # Qwen3-VL-Embedding-2B
QDRANT_UPSERT_BATCH_SIZE = 512
NAMESPACE_UUID = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
logger = get_logger(__name__)


class QdrantStorageError(RuntimeError):
    """Raised when Qdrant operations fail."""


class QdrantPayloadTooLargeError(QdrantStorageError):
    """Raised when a Qdrant request exceeds the upstream payload limit."""


@dataclass(frozen=True)
class FrameVector:
    """Frame embedding data for storage."""

    video_id: str
    frame_index: int
    timestamp_s: float
    vector: list[float]
    thumbnail_url: str | None = None


@dataclass(frozen=True)
class TranscriptVector:
    """Transcript embedding data for storage."""

    video_id: str
    segment_index: int
    timestamp_s: float
    end_s: float
    vector: list[float]
    text: str
    language_code: str | None = None


@dataclass(frozen=True)
class SearchResult:
    """Search result from Qdrant."""

    video_id: str
    frame_index: int
    timestamp_s: float
    score: float
    thumbnail_url: str | None = None
    source: Literal["visual", "transcript"] = "visual"
    transcript_text: str | None = None


def _point_name(video_id: str, *, source: Literal["visual", "transcript"], index: int) -> str:
    return f"{video_id}_{source}_{index}"


def generate_point_id(video_id: str, frame_index: int) -> str:
    """Generate deterministic UUID5 point ID for one visual frame."""
    name = _point_name(video_id, source="visual", index=frame_index)
    return str(uuid.uuid5(NAMESPACE_UUID, name))


def generate_transcript_point_id(video_id: str, segment_index: int) -> str:
    """Generate deterministic UUID5 point ID for one transcript segment."""
    name = _point_name(video_id, source="transcript", index=segment_index)
    return str(uuid.uuid5(NAMESPACE_UUID, name))


def _video_filter(
    video_id: str,
    *,
    source: Literal["visual", "transcript"] | None = None,
) -> models.Filter:
    """Build a Qdrant filter matching a single video_id and optional source."""
    must: list[models.FieldCondition] = [
        models.FieldCondition(
            key="video_id",
            match=models.MatchValue(value=video_id),
        )
    ]
    if source is not None:
        must.append(
            models.FieldCondition(
                key="source",
                match=models.MatchValue(value=source),
            )
        )
    return models.Filter(must=must)


def _unwrap_qdrant_error(exc: Exception) -> Exception:
    if isinstance(exc, ResponseHandlingException):
        return exc.source
    return exc


def _is_payload_too_large_error(exc: Exception) -> bool:
    candidate = _unwrap_qdrant_error(exc)
    if isinstance(candidate, UnexpectedResponse):
        if candidate.status_code != 400:
            return False
        try:
            structured = candidate.structured()
            error = structured.get("status", {}).get("error", "")
        except Exception:
            error = candidate.content.decode("utf-8", errors="ignore")
        message = str(error).casefold()
        return "payload error" in message and "larger than allowed" in message

    message = str(candidate).casefold()
    return "payload error" in message and "larger than allowed" in message


def _raise_upsert_error(kind: str, exc: Exception) -> NoReturn:
    error_cls = (
        QdrantPayloadTooLargeError if _is_payload_too_large_error(exc) else QdrantStorageError
    )
    raise error_cls(f"Failed to upsert {kind}: {exc}") from exc


class QdrantStore:
    """Qdrant vector storage for frame embeddings."""

    def __init__(self, config: QdrantConfig) -> None:
        self._config = config
        if config.use_in_memory:
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

        # Cloud clusters can require payload indexes for filtered queries.
        try:
            self._client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name="video_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            raise QdrantStorageError(f"Failed to ensure payload index for video_id: {exc}") from exc

        try:
            self._client.create_payload_index(
                collection_name=self._config.collection_name,
                field_name="source",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            raise QdrantStorageError(f"Failed to ensure payload index for source: {exc}") from exc

    def _upsert_points(
        self,
        *,
        kind: str,
        points: list[models.PointStruct],
    ) -> int:
        if not points:
            return 0

        batch_count = (len(points) + QDRANT_UPSERT_BATCH_SIZE - 1) // QDRANT_UPSERT_BATCH_SIZE
        logger.info(
            "Upserting %d %s vectors to Qdrant in %d batches (batch_size=%d)",
            len(points),
            kind,
            batch_count,
            QDRANT_UPSERT_BATCH_SIZE,
        )

        # Multi-request upserts are not atomic across batches. We intentionally rely on
        # deterministic UUID5 point ids so retries overwrite any partial prior write
        # instead of creating duplicates.
        try:
            for start in range(0, len(points), QDRANT_UPSERT_BATCH_SIZE):
                self._client.upsert(
                    collection_name=self._config.collection_name,
                    points=points[start : start + QDRANT_UPSERT_BATCH_SIZE],
                )
        except Exception as exc:
            _raise_upsert_error(kind, exc)

        return len(points)

    def upsert_frames(self, frames: list[FrameVector]) -> int:
        """Upsert frame vectors to Qdrant. Returns count of upserted points."""
        points = [
            models.PointStruct(
                id=generate_point_id(frame.video_id, frame.frame_index),
                vector=frame.vector,
                payload={
                    "video_id": frame.video_id,
                    "frame_index": frame.frame_index,
                    "timestamp_s": frame.timestamp_s,
                    "thumbnail_url": frame.thumbnail_url,
                    "source": "visual",
                },
            )
            for frame in frames
        ]
        return self._upsert_points(kind="frame", points=points)

    def upsert_transcripts(self, transcripts: list[TranscriptVector]) -> int:
        """Upsert transcript vectors to Qdrant. Returns count of upserted points."""
        points = [
            models.PointStruct(
                id=generate_transcript_point_id(transcript.video_id, transcript.segment_index),
                vector=transcript.vector,
                payload={
                    "video_id": transcript.video_id,
                    "frame_index": -1,
                    "segment_index": transcript.segment_index,
                    "timestamp_s": transcript.timestamp_s,
                    "end_s": transcript.end_s,
                    "thumbnail_url": None,
                    "source": "transcript",
                    "transcript_text": transcript.text,
                    "language_code": transcript.language_code,
                },
            )
            for transcript in transcripts
        ]
        return self._upsert_points(kind="transcript", points=points)

    def replace_transcripts(
        self,
        video_id: str,
        transcripts: list[TranscriptVector],
    ) -> int:
        """Replace transcript vectors for one video and return current count."""
        for transcript in transcripts:
            if transcript.video_id != video_id:
                raise QdrantStorageError(
                    "Transcript vector video_id does not match replace target"
                )

        self.delete_video_transcripts(video_id)
        if not transcripts:
            return 0
        return self.upsert_transcripts(transcripts)

    def search(
        self,
        query_vector: list[float],
        video_id: str,
        limit: int = 5,
        source: Literal["visual", "transcript"] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors within a video."""
        query_filter = _video_filter(video_id, source=source)

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
                frame_index=point.payload.get("frame_index", -1),  # type: ignore[union-attr]
                timestamp_s=point.payload["timestamp_s"],  # type: ignore[index]
                thumbnail_url=point.payload.get("thumbnail_url"),  # type: ignore[union-attr]
                score=point.score,
                source=point.payload.get("source", "visual"),  # type: ignore[union-attr]
                transcript_text=point.payload.get("transcript_text"),  # type: ignore[union-attr]
            )
            for point in points
        ]

    def _delete_points(
        self,
        video_id: str,
        *,
        source: Literal["visual", "transcript"] | None = None,
    ) -> int:
        query_filter = _video_filter(video_id, source=source)
        try:
            count_result = self._client.count(
                collection_name=self._config.collection_name,
                count_filter=query_filter,
            )
            count = int(count_result.count or 0)

            if count > 0:
                self._client.delete(
                    collection_name=self._config.collection_name,
                    points_selector=models.FilterSelector(
                        filter=query_filter
                    ),
                )

            return count
        except Exception as exc:
            raise QdrantStorageError(f"Failed to delete points: {exc}") from exc

    def delete_video(self, video_id: str) -> int:
        """Delete all vectors for a video. Returns count of deleted points."""
        return self._delete_points(video_id)

    def delete_video_transcripts(self, video_id: str) -> int:
        """Delete transcript vectors for a video. Returns count of deleted points."""
        return self._delete_points(video_id, source="transcript")
