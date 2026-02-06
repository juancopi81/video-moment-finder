"""Queue worker for durable video processing jobs."""
from __future__ import annotations

import argparse
import os
import socket
import time

from src.api.processing import process_video
from src.config.env import load_env
from src.db.supabase import (
    VideoJobRecord,
    claim_next_video_job,
    complete_video_job,
    get_video,
    update_video_status,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)
DEFAULT_POLL_INTERVAL_S = 3.0


def _default_worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    return f"{host}:{pid}"


def _process_claimed_job(job: VideoJobRecord) -> bool:
    video = get_video(job.video_id)
    if video is None:
        message = f"Video not found for job {job.id}"
        logger.error(message)
        complete_video_job(job.id, "failed", error_message=message)
        return False

    logger.info("Processing job_id=%s video_id=%s", job.id, video.id)
    update_video_status(video.id, "processing")

    try:
        process_video(video.id, video.youtube_url)
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Job failed job_id=%s video_id=%s: %s", job.id, video.id, exc)
        update_video_status(video.id, "failed", error_message=error_message)
        complete_video_job(job.id, "failed", error_message=error_message)
        return False

    update_video_status(video.id, "ready")
    complete_video_job(job.id, "completed")
    logger.info("Job complete job_id=%s video_id=%s", job.id, video.id)
    return True


def run_once(*, worker_id: str | None = None) -> bool:
    """Claim and process a single job.

    Returns:
        True if a job was processed, False if queue was empty.
    """
    current_worker_id = worker_id or _default_worker_id()
    job = claim_next_video_job(current_worker_id)
    if job is None:
        logger.debug("No queued jobs available")
        return False

    _process_claimed_job(job)
    return True


def run_forever(
    *,
    worker_id: str | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
) -> None:
    """Continuously process jobs from queue."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")

    current_worker_id = worker_id or _default_worker_id()
    logger.info(
        "Starting worker loop worker_id=%s poll_interval_s=%.1f",
        current_worker_id,
        poll_interval_s,
    )
    while True:
        processed = run_once(worker_id=current_worker_id)
        if not processed:
            time.sleep(poll_interval_s)


def main() -> None:
    load_env()

    parser = argparse.ArgumentParser(description="Video job queue worker")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process at most one job then exit",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL_S,
        help=f"Polling interval in seconds (default: {DEFAULT_POLL_INTERVAL_S})",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Optional stable worker identifier",
    )
    args = parser.parse_args()

    if args.once:
        run_once(worker_id=args.worker_id)
        return

    run_forever(worker_id=args.worker_id, poll_interval_s=args.poll_interval)


if __name__ == "__main__":
    main()
