"""Storage pipeline orchestrator for video processing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.storage.config import QdrantConfig, R2Config
from src.storage.qdrant import FrameVector, QdrantStore
from src.storage.r2 import R2Store
from src.utils.cleanup import cleanup_paths
from src.video.frames import FrameInfo


class PipelineError(RuntimeError):
    """Raised when pipeline operations fail."""


@dataclass(frozen=True)
class ProcessingResult:
    """Result of video processing."""

    video_id: str
    frames_processed: int
    embeddings_stored: int
    thumbnails_uploaded: int


class StoragePipeline:
    """Orchestrates storage of embeddings and thumbnails."""

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        r2_config: R2Config | None = None,
    ) -> None:
        self._qdrant = QdrantStore(qdrant_config)
        self._r2 = R2Store(r2_config) if r2_config is not None else None

    def ensure_ready(self) -> None:
        """Prepare storage backends (create collections, etc)."""
        self._qdrant.ensure_collection()

    def process_video(
        self,
        video_id: str,
        frames: list[FrameInfo],
        embeddings: list[list[float]],
        *,
        cleanup_temp_dirs: list[Path] | None = None,
    ) -> ProcessingResult:
        """
        Process a video by uploading thumbnails and storing embeddings.

        Args:
            video_id: Unique identifier for the video
            frames: List of extracted frames with paths and timestamps
            embeddings: List of embedding vectors (same length as frames)
            cleanup_temp_dirs: Optional list of directories to clean up after processing

        Returns:
            ProcessingResult with counts of processed items
        """
        if len(frames) != len(embeddings):
            raise PipelineError(
                f"Frame count ({len(frames)}) != embedding count ({len(embeddings)})"
            )

        if not frames:
            return ProcessingResult(
                video_id=video_id,
                frames_processed=0,
                embeddings_stored=0,
                thumbnails_uploaded=0,
            )

        try:
            # Upload thumbnails if R2 is configured and thumbnails exist
            thumbnail_urls: dict[int, str] = {}
            thumbnails_uploaded = 0

            if self._r2 is not None:
                thumbnails_to_upload = [
                    (frame.index, frame.thumbnail_path)
                    for frame in frames
                    if frame.thumbnail_path is not None
                ]

                if thumbnails_to_upload:
                    results = self._r2.upload_thumbnails(
                        video_id,
                        [(idx, path) for idx, path in thumbnails_to_upload],
                    )
                    thumbnails_uploaded = len(results)
                    for (idx, _), result in zip(thumbnails_to_upload, results):
                        thumbnail_urls[idx] = result.url

            # Build frame vectors with thumbnail URLs
            frame_vectors = [
                FrameVector(
                    video_id=video_id,
                    frame_index=frame.index,
                    timestamp_s=frame.timestamp_s,
                    vector=embedding,
                    thumbnail_url=thumbnail_urls.get(frame.index, ""),
                )
                for frame, embedding in zip(frames, embeddings)
            ]

            # Store embeddings in Qdrant
            try:
                embeddings_stored = self._qdrant.upsert_frames(frame_vectors)
            except Exception:
                if self._r2 is not None and thumbnails_uploaded:
                    try:
                        self._r2.delete_video_thumbnails(video_id)
                    except Exception:
                        pass
                raise

            return ProcessingResult(
                video_id=video_id,
                frames_processed=len(frames),
                embeddings_stored=embeddings_stored,
                thumbnails_uploaded=thumbnails_uploaded,
            )
        finally:
            # Clean up temp directories if requested
            if cleanup_temp_dirs:
                cleanup_paths(cleanup_temp_dirs, ignore_errors=True)

    def delete_video(self, video_id: str) -> tuple[int, int]:
        """
        Delete all data for a video.

        Returns:
            Tuple of (embeddings_deleted, thumbnails_deleted)
        """
        embeddings_deleted = self._qdrant.delete_video(video_id)

        thumbnails_deleted = 0
        if self._r2 is not None:
            thumbnails_deleted = self._r2.delete_video_thumbnails(video_id)

        return embeddings_deleted, thumbnails_deleted
