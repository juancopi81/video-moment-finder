"""Queue worker for durable video processing jobs."""
from __future__ import annotations

import argparse
import os
import socket
import time
from datetime import datetime, timedelta, timezone

from src.api.processing import process_video
from src.config.env import load_env
from src.db.supabase import (
    VideoJobRecord,
    claim_next_video_job,
    complete_video_job,
    get_video,
    list_stale_processing_video_jobs,
    requeue_video_job,
    update_video_status,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)
DEFAULT_POLL_INTERVAL_S = 3.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_STALE_LOCK_TIMEOUT_S = 600.0
STALE_RECOVERY_BATCH_LIMIT = 25


def _default_worker_id() -> str:
    host = socket.gethostname()
    pid = os.getpid()
    return f"{host}:{pid}"


def _log_worker_metric(event: str, **fields: object) -> None:
    parts = [f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={value}")
    logger.info("worker_metric %s", " ".join(parts))


def _parse_iso_datetime(iso_value: str | None) -> datetime | None:
    if not iso_value:
        return None
    try:
        dt = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _recover_stale_jobs(
    *,
    worker_id: str,
    max_attempts: int,
    stale_lock_timeout_s: float,
) -> int:
    stale_before = datetime.now(timezone.utc) - timedelta(seconds=stale_lock_timeout_s)
    stale_jobs = list_stale_processing_video_jobs(
        stale_before.isoformat(),
        limit=STALE_RECOVERY_BATCH_LIMIT,
    )
    if not stale_jobs:
        return 0

    recovered = 0
    for job in stale_jobs:
        locked_at = _parse_iso_datetime(job.locked_at)
        lock_age_s = (
            round((datetime.now(timezone.utc) - locked_at).total_seconds(), 3)
            if locked_at is not None
            else None
        )

        video = get_video(job.video_id)
        if job.attempt_count >= max_attempts:
            error_message = (
                "Job exceeded max attempts during stale-lock recovery: "
                f"{job.attempt_count}/{max_attempts}"
            )
            complete_video_job(job.id, "failed", error_message=error_message)
            if video is not None:
                update_video_status(video.id, "failed", error_message=error_message)
            logger.warning(
                "Terminal stale-lock failure job_id=%s video_id=%s attempts=%d/%d",
                job.id,
                job.video_id,
                job.attempt_count,
                max_attempts,
            )
            _log_worker_metric(
                "job_terminal_failure",
                worker_id=worker_id,
                job_id=job.id,
                video_id=job.video_id,
                attempt=job.attempt_count,
                max_attempts=max_attempts,
                reason="stale_lock_max_attempts",
                lock_age_s=lock_age_s,
            )
            continue

        requeued = requeue_video_job(job.id, error_message="Recovered stale processing lock")
        if requeued is None:
            logger.warning(
                "Stale job no longer recoverable job_id=%s video_id=%s",
                job.id,
                job.video_id,
            )
            continue

        if video is not None:
            update_video_status(video.id, "queued")

        recovered += 1
        logger.warning(
            "Recovered stale lock job_id=%s video_id=%s attempts=%d/%d",
            job.id,
            job.video_id,
            job.attempt_count,
            max_attempts,
        )
        _log_worker_metric(
            "job_requeued",
            worker_id=worker_id,
            job_id=job.id,
            video_id=job.video_id,
            attempt=job.attempt_count,
            max_attempts=max_attempts,
            reason="stale_lock",
            lock_age_s=lock_age_s,
        )
    return recovered


def _process_claimed_job(
    job: VideoJobRecord,
    *,
    worker_id: str,
    max_attempts: int,
) -> bool:
    video = get_video(job.video_id)
    if video is None:
        message = f"Video not found for job {job.id}"
        logger.error(message)
        complete_video_job(job.id, "failed", error_message=message)
        _log_worker_metric(
            "job_terminal_failure",
            worker_id=worker_id,
            job_id=job.id,
            video_id=job.video_id,
            attempt=job.attempt_count,
            max_attempts=max_attempts,
            reason="video_not_found",
        )
        return False

    logger.info(
        "Processing job_id=%s video_id=%s attempt=%d/%d",
        job.id,
        video.id,
        job.attempt_count,
        max_attempts,
    )
    update_video_status(video.id, "processing")

    try:
        process_video(video.id, video.youtube_url)
    except Exception as exc:
        error_message = str(exc)
        logger.exception("Job failed job_id=%s video_id=%s: %s", job.id, video.id, exc)
        if job.attempt_count < max_attempts:
            update_video_status(video.id, "queued")
            requeued = requeue_video_job(job.id, error_message=error_message)
            if requeued is None:
                logger.warning(
                    "Failed to requeue job (state changed) job_id=%s video_id=%s",
                    job.id,
                    video.id,
                )
            else:
                logger.warning(
                    "Requeued failed attempt job_id=%s video_id=%s attempts=%d/%d",
                    job.id,
                    video.id,
                    job.attempt_count,
                    max_attempts,
                )
            _log_worker_metric(
                "job_requeued",
                worker_id=worker_id,
                job_id=job.id,
                video_id=video.id,
                attempt=job.attempt_count,
                max_attempts=max_attempts,
                reason="attempt_failure",
            )
            return False

        update_video_status(video.id, "failed", error_message=error_message)
        complete_video_job(job.id, "failed", error_message=error_message)
        _log_worker_metric(
            "job_terminal_failure",
            worker_id=worker_id,
            job_id=job.id,
            video_id=video.id,
            attempt=job.attempt_count,
            max_attempts=max_attempts,
            reason="max_attempts",
        )
        return False

    update_video_status(video.id, "ready")
    complete_video_job(job.id, "completed")
    logger.info(
        "Job complete job_id=%s video_id=%s attempts=%d",
        job.id,
        video.id,
        job.attempt_count,
    )
    _log_worker_metric(
        "job_completed",
        worker_id=worker_id,
        job_id=job.id,
        video_id=video.id,
        attempt=job.attempt_count,
        max_attempts=max_attempts,
    )
    return True


def run_once(
    *,
    worker_id: str | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> bool:
    """Claim and process a single job.

    Returns:
        True if a job was processed, False if queue was empty.
    """
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if stale_lock_timeout_s <= 0:
        raise ValueError("stale_lock_timeout_s must be > 0")

    current_worker_id = worker_id or _default_worker_id()
    recovered = _recover_stale_jobs(
        worker_id=current_worker_id,
        max_attempts=max_attempts,
        stale_lock_timeout_s=stale_lock_timeout_s,
    )
    job = claim_next_video_job(current_worker_id)
    if job is None:
        logger.debug("No queued jobs available recovered_stale_jobs=%d", recovered)
        return False

    _log_worker_metric(
        "job_attempt_started",
        worker_id=current_worker_id,
        job_id=job.id,
        video_id=job.video_id,
        attempt=job.attempt_count,
        max_attempts=max_attempts,
    )
    _process_claimed_job(
        job,
        worker_id=current_worker_id,
        max_attempts=max_attempts,
    )
    return True


def run_forever(
    *,
    worker_id: str | None = None,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    stale_lock_timeout_s: float = DEFAULT_STALE_LOCK_TIMEOUT_S,
) -> None:
    """Continuously process jobs from queue."""
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be > 0")
    if stale_lock_timeout_s <= 0:
        raise ValueError("stale_lock_timeout_s must be > 0")

    current_worker_id = worker_id or _default_worker_id()
    logger.info(
        (
            "Starting worker loop worker_id=%s poll_interval_s=%.1f "
            "max_attempts=%d stale_lock_timeout_s=%.1f"
        ),
        current_worker_id,
        poll_interval_s,
        max_attempts,
        stale_lock_timeout_s,
    )
    while True:
        processed = run_once(
            worker_id=current_worker_id,
            max_attempts=max_attempts,
            stale_lock_timeout_s=stale_lock_timeout_s,
        )
        if not processed:
            time.sleep(poll_interval_s)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = float(raw)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def main() -> None:
    load_env()
    default_max_attempts = _env_int("VIDEO_JOB_MAX_ATTEMPTS", DEFAULT_MAX_ATTEMPTS)
    default_stale_lock_timeout_s = _env_float(
        "VIDEO_JOB_STALE_LOCK_TIMEOUT_S",
        DEFAULT_STALE_LOCK_TIMEOUT_S,
    )

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
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=default_max_attempts,
        help=(
            "Maximum attempts per job before terminal failure "
            f"(default: {default_max_attempts}; env VIDEO_JOB_MAX_ATTEMPTS)"
        ),
    )
    parser.add_argument(
        "--stale-lock-timeout",
        type=float,
        default=default_stale_lock_timeout_s,
        help=(
            "Lock age in seconds before stale processing jobs are recovered "
            "(default: "
            f"{default_stale_lock_timeout_s}; env VIDEO_JOB_STALE_LOCK_TIMEOUT_S)"
        ),
    )
    args = parser.parse_args()

    if args.once:
        run_once(
            worker_id=args.worker_id,
            max_attempts=args.max_attempts,
            stale_lock_timeout_s=args.stale_lock_timeout,
        )
        return

    run_forever(
        worker_id=args.worker_id,
        poll_interval_s=args.poll_interval,
        max_attempts=args.max_attempts,
        stale_lock_timeout_s=args.stale_lock_timeout,
    )


if __name__ == "__main__":
    main()
