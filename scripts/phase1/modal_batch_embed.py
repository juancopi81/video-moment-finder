"""
Phase 1.3: Batch Embedding Pipeline

Validates batch=8 embedding on Modal GPU.

Usage:
    uv run modal run scripts/phase1/modal_batch_embed.py --max-frames 64
"""

from pathlib import Path

from src.embedding.modal_app import app, embed_images_in_batches, extract_frame_bytes
from src.utils.logging import get_logger

LOCAL_VIDEO_PATH = Path("test_video.mp4")
logger = get_logger(__name__)


def _print_progress(done: int, total: int) -> None:
    logger.info("Embedded %d/%d frames...", done, total)


@app.local_entrypoint()
def main(max_frames: int = 64) -> None:
    if not LOCAL_VIDEO_PATH.exists():
        logger.error("No test video found!")
        logger.error(
            'Download a video first: uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "URL"'
        )
        return

    video_bytes = LOCAL_VIDEO_PATH.read_bytes()
    frame_bytes = extract_frame_bytes.remote(video_bytes, fps=1.0, max_frames=max_frames)

    logger.info("Embedding %d frames with batch_size=8...", len(frame_bytes))
    embeddings = embed_images_in_batches.remote(frame_bytes, batch_size=8)
    _print_progress(len(frame_bytes), len(frame_bytes))

    if not embeddings:
        raise RuntimeError("No embeddings returned")

    logger.info("Done. Embedding count: %d", len(embeddings))
