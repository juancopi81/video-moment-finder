"""Background video processing task."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.db.supabase import update_video_status
from src.embedding.modal_app import embed_images_in_batches
from src.pipeline.orchestrator import StoragePipeline
from src.storage.config import QdrantConfig, R2Config, StorageConfigError
from src.utils.logging import get_logger
from src.video.download import download_video
from src.video.frames import extract_frames

logger = get_logger(__name__)

MAX_FRAMES = 1800  # 30 min at 1fps


def process_video_task(video_id: str, youtube_url: str) -> None:
    """
    Background task: download -> extract -> embed -> store -> update status.

    This task runs in the background via FastAPI BackgroundTasks.
    On success, updates video status to "ready".
    On failure, updates video status to "failed" with error message.
    """
    logger.info("Starting processing for video_id=%s url=%s", video_id, youtube_url)

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_dir = temp_path / "video"
            frames_dir = temp_path / "frames"
            thumbnails_dir = temp_path / "thumbnails"

            # 1. Download video locally (yt-dlp)
            logger.info("Downloading video for video_id=%s", video_id)
            video_path = download_video(youtube_url, video_dir)
            logger.info("Downloaded video to %s", video_path)

            # 2. Extract frames at 1fps (ffmpeg) with thumbnails
            logger.info("Extracting frames for video_id=%s", video_id)
            frames = extract_frames(
                video_path,
                frames_dir,
                fps=1.0,
                thumbnail_dir=thumbnails_dir,
            )
            logger.info("Extracted %d frames", len(frames))

            # 3. Enforce 30-min limit
            if len(frames) > MAX_FRAMES:
                logger.warning(
                    "Truncating frames from %d to %d (30-min limit)",
                    len(frames),
                    MAX_FRAMES,
                )
                frames = frames[:MAX_FRAMES]

            # 4. Embed frames via Modal
            logger.info("Embedding %d frames via Modal", len(frames))
            frame_bytes = [frame.path.read_bytes() for frame in frames]
            embeddings = embed_images_in_batches.remote(frame_bytes, batch_size=8)
            logger.info("Got %d embeddings", len(embeddings))

            # 5. Store in Qdrant + R2 (StoragePipeline.process_video)
            logger.info("Storing embeddings and thumbnails for video_id=%s", video_id)
            qdrant_config = QdrantConfig.from_env()

            # R2 config is optional - if not configured, thumbnails won't be stored
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

            # 6. Update Supabase: update_video_status(video_id, "ready")
            update_video_status(video_id, "ready")
            logger.info("Video processing complete for video_id=%s", video_id)

    except Exception as e:
        logger.exception("Failed to process video_id=%s: %s", video_id, e)
        update_video_status(video_id, "failed", error_message=str(e))
