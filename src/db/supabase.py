"""Supabase client and CRUD operations for videos and credits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from supabase import create_client, Client

# Video status type
VideoStatus = Literal["queued", "processing", "ready", "failed"]
JobStatus = Literal["queued", "processing", "completed", "failed"]
TerminalJobStatus = Literal["completed", "failed"]


@dataclass
class VideoRecord:
    """Video database record."""

    id: str
    youtube_url: str
    status: VideoStatus
    user_id: str | None = None
    error_message: str | None = None
    created_at: str | None = None  # ISO 8601 string from Supabase
    updated_at: str | None = None  # ISO 8601 string from Supabase


@dataclass
class CreditRecord:
    """Credit database record."""

    id: str
    user_id: str
    balance: int
    created_at: str | None = None  # ISO 8601 string from Supabase
    updated_at: str | None = None  # ISO 8601 string from Supabase


@dataclass
class VideoJobRecord:
    """Video processing job record."""

    id: str
    video_id: str
    status: JobStatus
    worker_id: str | None = None
    attempt_count: int = 0
    error_message: str | None = None
    locked_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


# Singleton client
_client: Client | None = None


def get_client() -> Client:
    """Get or create Supabase client singleton.

    Requires SUPABASE_URL and SUPABASE_SECRET_KEY environment variables.
    """
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SECRET_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY environment variables required"
            )
        _client = create_client(url, key)
    return _client


def _row_to_video(row: dict) -> VideoRecord:
    """Convert database row to VideoRecord."""
    return VideoRecord(
        id=row["id"],
        youtube_url=row["youtube_url"],
        status=row["status"],
        user_id=row.get("user_id"),
        error_message=row.get("error_message"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_credit(row: dict) -> CreditRecord:
    """Convert database row to CreditRecord."""
    return CreditRecord(
        id=row["id"],
        user_id=row["user_id"],
        balance=row["balance"],
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_video_job(row: dict) -> VideoJobRecord:
    """Convert database row to VideoJobRecord."""
    return VideoJobRecord(
        id=row["id"],
        video_id=row["video_id"],
        status=row["status"],
        worker_id=row.get("worker_id"),
        attempt_count=row.get("attempt_count", 0),
        error_message=row.get("error_message"),
        locked_at=row.get("locked_at"),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Video CRUD
# ---------------------------------------------------------------------------


def create_video(
    youtube_url: str,
    user_id: str | None = None,
    status: VideoStatus = "queued",
) -> VideoRecord:
    """Create a new video record.

    Args:
        youtube_url: YouTube video URL.
        user_id: Optional Clerk user ID.

    Returns:
        Created VideoRecord with generated ID.
    """
    client = get_client()
    data = {"youtube_url": youtube_url, "status": status}
    if user_id is not None:
        data["user_id"] = user_id

    result = client.table("videos").insert(data).execute()
    if not result.data:
        raise RuntimeError("Failed to create video record")
    return _row_to_video(result.data[0])


def get_video(video_id: str) -> VideoRecord | None:
    """Get video by ID.

    Args:
        video_id: UUID of the video.

    Returns:
        VideoRecord if found, None otherwise.
    """
    client = get_client()
    result = client.table("videos").select("*").eq("id", video_id).execute()
    if not result.data:
        return None
    return _row_to_video(result.data[0])


def update_video_status(
    video_id: str,
    status: VideoStatus,
    error_message: str | None = None,
) -> VideoRecord | None:
    """Update video status.

    Args:
        video_id: UUID of the video.
        status: New status ('processing', 'ready', or 'failed').
        error_message: Optional error message (typically for 'failed' status).

    Returns:
        Updated VideoRecord if found, None otherwise.
    """
    client = get_client()
    data: dict = {"status": status}
    if status == "failed":
        data["error_message"] = error_message
    else:
        data["error_message"] = None

    result = client.table("videos").update(data).eq("id", video_id).execute()
    if not result.data:
        return None
    return _row_to_video(result.data[0])


def list_videos(user_id: str | None = None) -> list[VideoRecord]:
    """List videos, optionally filtered by user.

    Args:
        user_id: Optional Clerk user ID to filter by.

    Returns:
        List of VideoRecords, ordered by created_at descending.
    """
    client = get_client()
    query = client.table("videos").select("*").order("created_at", desc=True)
    if user_id is not None:
        query = query.eq("user_id", user_id)

    result = query.execute()
    return [_row_to_video(row) for row in result.data]


# ---------------------------------------------------------------------------
# Video jobs (durable queue)
# ---------------------------------------------------------------------------


def enqueue_video_job(video_id: str) -> VideoJobRecord:
    """Create a queued processing job for a video."""
    client = get_client()
    result = (
        client.table("video_jobs")
        .insert({"video_id": video_id, "status": "queued"})
        .execute()
    )
    if not result.data:
        raise RuntimeError("Failed to enqueue video job")
    return _row_to_video_job(result.data[0])


def list_queued_video_jobs(limit: int = 10) -> list[VideoJobRecord]:
    """List queued jobs in FIFO order."""
    if limit <= 0:
        raise ValueError("limit must be > 0")

    client = get_client()
    result = (
        client.table("video_jobs")
        .select("*")
        .eq("status", "queued")
        .order("created_at")
        .limit(limit)
        .execute()
    )
    return [_row_to_video_job(row) for row in result.data]


def claim_next_video_job(worker_id: str) -> VideoJobRecord | None:
    """Claim the next queued job for processing.

    Uses optimistic claiming with status guard to avoid duplicate claims.
    """
    if not worker_id.strip():
        raise ValueError("worker_id must be non-empty")

    client = get_client()
    candidates = list_queued_video_jobs(limit=25)
    now = _utc_now_iso()

    for job in candidates:
        result = (
            client.table("video_jobs")
            .update(
                {
                    "status": "processing",
                    "worker_id": worker_id,
                    "attempt_count": job.attempt_count + 1,
                    "locked_at": now,
                    "started_at": now,
                    "error_message": None,
                }
            )
            .eq("id", job.id)
            .eq("status", "queued")
            .execute()
        )
        if result.data:
            return _row_to_video_job(result.data[0])
    return None


def complete_video_job(
    job_id: str,
    status: TerminalJobStatus,
    *,
    error_message: str | None = None,
) -> VideoJobRecord | None:
    """Mark a processing job as completed or failed."""
    client = get_client()
    data = {
        "status": status,
        "completed_at": _utc_now_iso(),
        "locked_at": None,
    }
    if status == "failed":
        data["error_message"] = error_message
    else:
        data["error_message"] = None

    result = (
        client.table("video_jobs")
        .update(data)
        .eq("id", job_id)
        .eq("status", "processing")
        .execute()
    )
    if not result.data:
        return None
    return _row_to_video_job(result.data[0])


def get_video_job(video_id: str) -> VideoJobRecord | None:
    """Get latest job record for a video."""
    client = get_client()
    result = (
        client.table("video_jobs")
        .select("*")
        .eq("video_id", video_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_video_job(result.data[0])


# ---------------------------------------------------------------------------
# Credits CRUD
# ---------------------------------------------------------------------------


def get_credits(user_id: str) -> CreditRecord | None:
    """Get credit record for a user.

    Args:
        user_id: Clerk user ID.

    Returns:
        CreditRecord if found, None otherwise.
    """
    client = get_client()
    result = client.table("credits").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return None
    return _row_to_credit(result.data[0])


def update_credits(user_id: str, balance: int) -> CreditRecord:
    """Update or create credit balance for a user.

    Uses upsert to create the record if it doesn't exist.

    Args:
        user_id: Clerk user ID.
        balance: New credit balance (must be >= 0).

    Returns:
        Updated or created CreditRecord.

    Raises:
        ValueError: If balance is negative.
    """
    if balance < 0:
        raise ValueError("Credit balance cannot be negative")

    client = get_client()
    result = (
        client.table("credits")
        .upsert({"user_id": user_id, "balance": balance}, on_conflict="user_id")
        .execute()
    )
    if not result.data:
        raise RuntimeError("Failed to upsert credit record")
    return _row_to_credit(result.data[0])
