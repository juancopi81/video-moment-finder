"""Supabase client and CRUD operations for videos and credits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from supabase import create_client, Client

# Video status type
VideoStatus = Literal["processing", "ready", "failed"]


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


# ---------------------------------------------------------------------------
# Video CRUD
# ---------------------------------------------------------------------------


def create_video(youtube_url: str, user_id: str | None = None) -> VideoRecord:
    """Create a new video record with status='processing'.

    Args:
        youtube_url: YouTube video URL.
        user_id: Optional Clerk user ID.

    Returns:
        Created VideoRecord with generated ID.
    """
    client = get_client()
    data = {"youtube_url": youtube_url}
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
    if error_message is not None:
        data["error_message"] = error_message

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
