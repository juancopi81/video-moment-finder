"""Video processing pipeline implementation."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from src.config.modal import (
    EMBED_IMAGES_FUNCTION_NAME,
    get_embedding_modal_function,
)
from src.pipeline.orchestrator import ProcessingResult, StoragePipeline
from src.storage.config import QdrantConfig, R2Config, StorageConfigError
from src.utils.logging import get_logger
from src.video.download import download_video
from src.video.frames import extract_frames

logger = get_logger(__name__)

MAX_FRAMES = 1800  # 30 min at 1fps


class VideoProcessingError(RuntimeError):
    """Raised when the processing pipeline fails."""


def process_video(video_id: str, youtube_url: str) -> ProcessingResult:
    """
    Processing pipeline: download -> extract -> embed -> store.

    Returns:
        ProcessingResult with processed counts.

    Raises:
        VideoProcessingError: If any stage fails.
    """
    logger.info("Starting processing for video_id=%s url=%s", video_id, youtube_url)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_dir = temp_path / "video"
            frames_dir = temp_path / "frames"
            thumbnails_dir = temp_path / "thumbnails"

            local_video_dir = _local_video_dir()
            if local_video_dir is not None:
                logger.info(
                    "Checking local video cache for video_id=%s in %s",
                    video_id,
                    local_video_dir,
                )
            logger.info("Downloading video for video_id=%s", video_id)
            video_path = download_video(
                youtube_url,
                video_dir,
                local_video_dir=local_video_dir,
            )
            if local_video_dir is not None and _is_within_dir(video_path, local_video_dir):
                logger.info("Using local video file %s", video_path)
            logger.info("Downloaded video to %s", video_path)

            logger.info("Extracting frames for video_id=%s", video_id)
            frames = extract_frames(
                video_path,
                frames_dir,
                fps=1.0,
                thumbnail_dir=thumbnails_dir,
            )
            logger.info("Extracted %d frames", len(frames))

            if len(frames) > MAX_FRAMES:
                logger.warning(
                    "Truncating frames from %d to %d (30-min limit)",
                    len(frames),
                    MAX_FRAMES,
                )
                frames = frames[:MAX_FRAMES]

            logger.info("Embedding %d frames via Modal", len(frames))
            frame_bytes = [frame.path.read_bytes() for frame in frames]
            embed_fn = get_embedding_modal_function(EMBED_IMAGES_FUNCTION_NAME)
            embeddings = embed_fn.remote(frame_bytes, batch_size=8)
            logger.info("Got %d embeddings", len(embeddings))

            logger.info("Storing embeddings and thumbnails for video_id=%s", video_id)
            qdrant_config = QdrantConfig.from_env()
            try:
                r2_config = R2Config.from_env()
            except StorageConfigError:
                logger.warning("R2 not configured, thumbnails will not be uploaded")
                r2_config = None

            pipeline = StoragePipeline(qdrant_config, r2_config)
            pipeline.ensure_ready()
            result = pipeline.process_video(video_id, frames, embeddings)
            logger.info(
                "Stored %d embeddings, %d thumbnails for video_id=%s",
                result.embeddings_stored,
                result.thumbnails_uploaded,
                video_id,
            )
            logger.info("Video processing complete for video_id=%s", video_id)
            return result

    except Exception as exc:
        logger.exception("Failed to process video_id=%s: %s", video_id, exc)
        raise VideoProcessingError(str(exc)) from exc


def _local_video_dir() -> Path | None:
    raw = os.environ.get("VIDEO_LOCAL_VIDEO_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _is_within_dir(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True
