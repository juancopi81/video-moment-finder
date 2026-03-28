"""Supabase client and CRUD operations for videos and credits."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from supabase import create_client, Client
from src.utils.logging import get_logger

# Video status type
VideoStatus = Literal["queued", "processing", "ready", "failed"]
SourceType = Literal["youtube", "upload"]
JobStatus = Literal["queued", "processing", "completed", "failed"]
TerminalJobStatus = Literal["completed", "failed"]
McpOAuthRequestStatus = Literal["pending", "approved", "denied"]
logger = get_logger(__name__)


@dataclass
class VideoRecord:
    """Video database record."""

    id: str
    youtube_url: str | None
    status: VideoStatus
    user_id: str | None = None
    error_message: str | None = None
    source_type: SourceType = "youtube"
    source_r2_key: str | None = None
    source_filename: str | None = None
    created_at: str | None = None  # ISO 8601 string from Supabase
    updated_at: str | None = None  # ISO 8601 string from Supabase


@dataclass
class TranscriptSegmentRecord:
    """Transcript segment database record."""

    video_id: str
    segment_index: int
    start_s: float
    end_s: float
    text: str
    language_code: str | None = None
    score: float | None = None


@dataclass
class CreditRecord:
    """Credit database record."""

    id: str
    user_id: str
    balance: int
    created_at: str | None = None  # ISO 8601 string from Supabase
    updated_at: str | None = None  # ISO 8601 string from Supabase


@dataclass
class BillingCreditGrantResult:
    """Result of applying a billing event credit grant."""

    applied: bool


@dataclass
class ProcessingCreditConsumeResult:
    """Result of consuming one processing credit."""

    allowed: bool
    remaining_balance: int


@dataclass
class ApiCreditRecord:
    """API credit balance record."""

    user_id: str
    balance: int
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class ApiUnitConsumeResult:
    """Result of consuming API units."""

    allowed: bool
    remaining_balance: int


@dataclass
class ApiBillingCreditGrantResult:
    """Result of applying an API billing credit grant."""

    applied: bool


@dataclass
class ApiUsageEventRecord:
    """API usage event ledger entry."""

    id: str
    user_id: str
    api_key_id: str | None
    event_type: str
    units: int
    video_id: str | None
    request_id: str | None
    metadata: dict | None = None
    created_at: str | None = None


@dataclass
class ApiKeyRecord:
    """API key database record."""

    id: str
    user_id: str
    name: str
    key_hash: str
    key_prefix: str
    revoked_at: str | None = None
    last_used_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class McpOAuthAuthorizationRequestRecord:
    """Stored OAuth authorization request for MCP connector linking."""

    id: str
    client_id: str
    redirect_uri: str
    redirect_uri_provided_explicitly: bool
    state: str | None
    scopes: list[str]
    code_challenge: str
    resource: str
    status: McpOAuthRequestStatus
    user_id: str | None = None
    expires_at: str | None = None
    approved_at: str | None = None
    denied_at: str | None = None
    resolved_at: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


@dataclass
class McpOAuthAuthorizationCodeRecord:
    """Stored OAuth authorization code."""

    id: str
    user_id: str
    client_id: str
    code_hash: str
    redirect_uri: str
    redirect_uri_provided_explicitly: bool
    scopes: list[str]
    code_challenge: str
    resource: str
    authorization_request_id: str | None = None
    expires_at: str | None = None
    used_at: str | None = None
    revoked_at: str | None = None
    created_at: str | None = None


@dataclass
class McpOAuthAccessTokenRecord:
    """Stored OAuth access token."""

    id: str
    connection_id: str
    user_id: str
    client_id: str
    token_hash: str
    scopes: list[str]
    resource: str
    expires_at: str | None = None
    revoked_at: str | None = None
    created_at: str | None = None


@dataclass
class McpOAuthRefreshTokenRecord:
    """Stored OAuth refresh token."""

    id: str
    connection_id: str
    user_id: str
    client_id: str
    token_hash: str
    scopes: list[str]
    resource: str
    expires_at: str | None = None
    revoked_at: str | None = None
    created_at: str | None = None


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


def reset_client() -> None:
    """Drop cached Supabase client and best-effort close HTTP sessions."""
    global _client
    client = _client
    _client = None
    if client is None:
        return

    for attr_name in ("postgrest", "storage"):
        sub_client = getattr(client, attr_name, None)
        session = getattr(sub_client, "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:
                logger.debug(
                    "Failed to close Supabase %s session: %s",
                    attr_name,
                    exc,
                    exc_info=True,
                )

    auth_client = getattr(client, "auth", None)
    auth_close = getattr(auth_client, "close", None)
    if callable(auth_close):
        try:
            auth_close()
        except Exception as exc:
            logger.debug("Failed to close Supabase auth client: %s", exc, exc_info=True)


def _row_to_video(row: dict) -> VideoRecord:
    """Convert database row to VideoRecord."""
    return VideoRecord(
        id=row["id"],
        youtube_url=row.get("youtube_url"),
        status=row["status"],
        user_id=row.get("user_id"),
        error_message=row.get("error_message"),
        source_type=row.get("source_type") or "youtube",
        source_r2_key=row.get("source_r2_key"),
        source_filename=row.get("source_filename"),
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


def _row_to_transcript_segment(row: dict) -> TranscriptSegmentRecord:
    """Convert database row to TranscriptSegmentRecord."""
    score_raw = row.get("score")
    score = float(score_raw) if score_raw is not None else None
    return TranscriptSegmentRecord(
        video_id=row["video_id"],
        segment_index=int(row["segment_index"]),
        start_s=float(row["start_s"]),
        end_s=float(row["end_s"]),
        text=row["text"],
        language_code=row.get("language_code"),
        score=score,
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


def _row_to_mcp_oauth_authorization_request(row: dict) -> McpOAuthAuthorizationRequestRecord:
    return McpOAuthAuthorizationRequestRecord(
        id=row["id"],
        client_id=row["client_id"],
        redirect_uri=row["redirect_uri"],
        redirect_uri_provided_explicitly=bool(row.get("redirect_uri_provided_explicitly")),
        state=row.get("state"),
        scopes=list(row.get("scopes") or []),
        code_challenge=row["code_challenge"],
        resource=row["resource"],
        status=row.get("status", "pending"),
        user_id=row.get("user_id"),
        expires_at=row.get("expires_at"),
        approved_at=row.get("approved_at"),
        denied_at=row.get("denied_at"),
        resolved_at=row.get("resolved_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_mcp_oauth_authorization_code(row: dict) -> McpOAuthAuthorizationCodeRecord:
    return McpOAuthAuthorizationCodeRecord(
        id=row["id"],
        authorization_request_id=row.get("authorization_request_id"),
        user_id=row["user_id"],
        client_id=row["client_id"],
        code_hash=row["code_hash"],
        redirect_uri=row["redirect_uri"],
        redirect_uri_provided_explicitly=bool(row.get("redirect_uri_provided_explicitly")),
        scopes=list(row.get("scopes") or []),
        code_challenge=row["code_challenge"],
        resource=row["resource"],
        expires_at=row.get("expires_at"),
        used_at=row.get("used_at"),
        revoked_at=row.get("revoked_at"),
        created_at=row.get("created_at"),
    )


def _row_to_mcp_oauth_access_token(row: dict) -> McpOAuthAccessTokenRecord:
    return McpOAuthAccessTokenRecord(
        id=row["id"],
        connection_id=row["connection_id"],
        user_id=row["user_id"],
        client_id=row["client_id"],
        token_hash=row["token_hash"],
        scopes=list(row.get("scopes") or []),
        resource=row["resource"],
        expires_at=row.get("expires_at"),
        revoked_at=row.get("revoked_at"),
        created_at=row.get("created_at"),
    )


def _row_to_mcp_oauth_refresh_token(row: dict) -> McpOAuthRefreshTokenRecord:
    return McpOAuthRefreshTokenRecord(
        id=row["id"],
        connection_id=row["connection_id"],
        user_id=row["user_id"],
        client_id=row["client_id"],
        token_hash=row["token_hash"],
        scopes=list(row.get("scopes") or []),
        resource=row["resource"],
        expires_at=row.get("expires_at"),
        revoked_at=row.get("revoked_at"),
        created_at=row.get("created_at"),
    )


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _rpc_first_item(raw: object) -> object:
    """Return the first RPC payload item when Supabase wraps responses in a list."""
    if isinstance(raw, list):
        return raw[0] if raw else None
    return raw


# ---------------------------------------------------------------------------
# Video CRUD
# ---------------------------------------------------------------------------


def create_video(
    youtube_url: str,
    user_id: str | None = None,
    status: VideoStatus = "queued",
    *,
    video_id: str | None = None,
) -> VideoRecord:
    """Create a new video record.

    Args:
        youtube_url: YouTube video URL.
        user_id: Optional Clerk user ID.

    Returns:
        Created VideoRecord with generated ID.
    """
    client = get_client()
    data: dict = {
        "youtube_url": youtube_url,
        "status": status,
        "source_type": "youtube",
    }
    if user_id is not None:
        data["user_id"] = user_id
    if video_id is not None:
        data["id"] = video_id

    result = client.table("videos").insert(data).execute()
    if not result.data:
        raise RuntimeError("Failed to create video record")
    return _row_to_video(result.data[0])


def create_uploaded_video(
    *,
    video_id: str,
    source_r2_key: str,
    source_filename: str | None = None,
    user_id: str | None = None,
    status: VideoStatus = "queued",
) -> VideoRecord:
    """Create a new uploaded-video record."""
    if not video_id.strip():
        raise ValueError("video_id must be non-empty")
    if not source_r2_key.strip():
        raise ValueError("source_r2_key must be non-empty")

    client = get_client()
    data: dict = {
        "id": video_id,
        "youtube_url": None,
        "status": status,
        "source_type": "upload",
        "source_r2_key": source_r2_key,
        "source_filename": source_filename,
    }
    if user_id is not None:
        data["user_id"] = user_id

    result = client.table("videos").insert(data).execute()
    if not result.data:
        raise RuntimeError("Failed to create video record")
    return _row_to_video(result.data[0])


def _call_idempotent_video_rpc(rpc_name: str, params: dict) -> tuple[VideoRecord, bool]:
    """Call an idempotent insert-or-get RPC and return (record, was_created)."""
    client = get_client()
    result = client.rpc(rpc_name, params).execute()
    if not result.data:
        raise RuntimeError(f"{rpc_name} returned no data")
    item = _rpc_first_item(result.data)
    if not isinstance(item, dict):
        raise RuntimeError(f"{rpc_name} returned unexpected data shape")

    row_data = item.get("row_data")
    was_created = item.get("was_created")
    if not isinstance(row_data, dict) or not isinstance(was_created, bool):
        raise RuntimeError(f"{rpc_name} returned unexpected payload")

    return _row_to_video(row_data), was_created


def insert_youtube_video_idempotent(
    youtube_url: str, user_id: str
) -> tuple[VideoRecord, bool]:
    """Atomically insert a YouTube video or return the existing non-failed record.

    Uses a Postgres RPC backed by a unique partial index on (user_id, youtube_url)
    so concurrent retries are serialized by the DB — only one INSERT wins.
    """
    return _call_idempotent_video_rpc(
        "insert_youtube_video_idempotent",
        {"p_youtube_url": youtube_url, "p_user_id": user_id},
    )


def insert_uploaded_video_idempotent(
    video_id: str,
    user_id: str,
    source_r2_key: str,
    source_filename: str,
) -> tuple[VideoRecord, bool]:
    """Atomically insert an uploaded video or return the existing record.

    The caller supplies a deterministic video_id (derived from Idempotency-Key)
    so concurrent retries collide on the PK and only one INSERT wins.
    """
    return _call_idempotent_video_rpc(
        "insert_uploaded_video_idempotent",
        {
            "p_video_id": video_id,
            "p_user_id": user_id,
            "p_source_r2_key": source_r2_key,
            "p_source_filename": source_filename,
        },
    )


def get_video(video_id: str, user_id: str | None = None) -> VideoRecord | None:
    """Get video by ID.

    Args:
        video_id: UUID of the video.
        user_id: Optional Clerk user ID to enforce ownership.

    Returns:
        VideoRecord if found, None otherwise.
    """
    client = get_client()
    query = client.table("videos").select("*").eq("id", video_id)
    if user_id is not None:
        query = query.eq("user_id", user_id)
    result = query.execute()
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


def count_videos_for_user(user_id: str) -> int:
    """Count non-failed videos owned by a user."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")

    client = get_client()
    result = (
        client.table("videos")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .neq("status", "failed")
        .execute()
    )
    if result.count is not None:
        return int(result.count)
    return len(result.data or [])


def has_unlimited_video_access(user_id: str) -> bool:
    """Return whether a user bypasses the free video cap."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")

    client = get_client()
    result = (
        client.table("video_access_overrides")
        .select("user_id")
        .eq("user_id", user_id)
        .eq("unlimited_videos", True)
        .limit(1)
        .execute()
    )
    return bool(result.data)


def replace_video_transcript_segments(
    video_id: str,
    segments: list[TranscriptSegmentRecord],
) -> int:
    """Replace all transcript segments for one video atomically."""
    if not video_id.strip():
        raise ValueError("video_id must be non-empty")

    client = get_client()
    rows = [
        {
            "segment_index": segment.segment_index,
            "start_s": segment.start_s,
            "end_s": segment.end_s,
            "text": segment.text,
            "language_code": segment.language_code,
        }
        for segment in segments
    ]

    result = client.rpc(
        "replace_video_transcript_segments",
        {
            "p_video_id": video_id,
            "p_segments": rows,
        },
    ).execute()

    inserted_raw = _rpc_first_item(result.data)
    if inserted_raw is None:
        return 0
    return int(inserted_raw)


def search_video_transcript_segments(
    video_id: str,
    query: str,
    *,
    limit: int = 5,
) -> list[TranscriptSegmentRecord]:
    """Search transcript segments for a single video."""
    if not video_id.strip():
        raise ValueError("video_id must be non-empty")
    if limit <= 0:
        raise ValueError("limit must be > 0")

    client = get_client()
    result = client.rpc(
        "search_video_transcript_segments",
        {
            "p_video_id": video_id,
            "p_query": query,
            "p_limit": limit,
        },
    ).execute()

    rows = result.data or []
    return [_row_to_transcript_segment(row) for row in rows]


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


def list_stale_processing_video_jobs(
    stale_before_iso: str,
    limit: int = 25,
) -> list[VideoJobRecord]:
    """List processing jobs with stale locks in oldest-lock order."""
    if not stale_before_iso.strip():
        raise ValueError("stale_before_iso must be non-empty")
    if limit <= 0:
        raise ValueError("limit must be > 0")

    client = get_client()
    result = (
        client.table("video_jobs")
        .select("*")
        .eq("status", "processing")
        .lt("locked_at", stale_before_iso)
        .order("locked_at")
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


def requeue_video_job(
    job_id: str,
    *,
    error_message: str | None = None,
) -> VideoJobRecord | None:
    """Move a processing job back to queued state for retry."""
    client = get_client()
    result = (
        client.table("video_jobs")
        .update(
            {
                "status": "queued",
                "worker_id": None,
                "locked_at": None,
                "started_at": None,
                "completed_at": None,
                "error_message": error_message,
            }
        )
        .eq("id", job_id)
        .eq("status", "processing")
        .execute()
    )
    if not result.data:
        return None
    return _row_to_video_job(result.data[0])


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


def consume_processing_credit(user_id: str) -> ProcessingCreditConsumeResult:
    """Consume one processing credit atomically."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")

    client = get_client()
    result = client.rpc(
        "consume_processing_credit",
        {"p_user_id": user_id},
    ).execute()
    first_item = _rpc_first_item(result.data)
    if isinstance(first_item, dict):
        row = first_item
    else:
        logger.warning(
            "Unexpected consume_processing_credit RPC response shape: %s",
            type(first_item).__name__,
        )
        row = {}

    remaining_raw = row.get("remaining_balance", 0)
    try:
        remaining = int(remaining_raw)
    except (TypeError, ValueError):
        remaining = 0

    return ProcessingCreditConsumeResult(
        allowed=bool(row.get("allowed")),
        remaining_balance=remaining,
    )


def apply_billing_credit_grant(
    *,
    provider: str,
    event_id: str,
    event_type: str,
    user_id: str,
    credits: int,
    payload: dict | None = None,
) -> BillingCreditGrantResult:
    """Apply billing credit grants atomically and idempotently.

    Returns:
        BillingCreditGrantResult where ``applied`` is False on duplicate events.
    """
    if not provider.strip():
        raise ValueError("provider must be non-empty")
    if not event_id.strip():
        raise ValueError("event_id must be non-empty")
    if not event_type.strip():
        raise ValueError("event_type must be non-empty")
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if credits <= 0:
        raise ValueError("credits must be > 0")

    client = get_client()
    result = client.rpc(
        "apply_billing_credit_grant",
        {
            "p_provider": provider,
            "p_event_id": event_id,
            "p_event_type": event_type,
            "p_user_id": user_id,
            "p_credits": credits,
            "p_payload": payload or {},
        },
    ).execute()
    applied = bool(_rpc_first_item(result.data))
    return BillingCreditGrantResult(applied=applied)


# ---------------------------------------------------------------------------
# API keys CRUD
# ---------------------------------------------------------------------------

MAX_API_KEYS_PER_USER = 10


def _row_to_api_key(row: dict) -> ApiKeyRecord:
    """Convert database row to ApiKeyRecord."""
    return ApiKeyRecord(
        id=row["id"],
        user_id=row["user_id"],
        name=row.get("name", ""),
        key_hash=row["key_hash"],
        key_prefix=row["key_prefix"],
        revoked_at=row.get("revoked_at"),
        last_used_at=row.get("last_used_at"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def create_api_key(
    user_id: str, name: str, key_hash: str, key_prefix: str
) -> ApiKeyRecord:
    """Create a new API key record atomically with per-user cap."""
    client = get_client()

    try:
        result = client.rpc(
            "create_api_key_atomic",
            {
                "p_user_id": user_id,
                "p_name": name,
                "p_key_hash": key_hash,
                "p_key_prefix": key_prefix,
                "p_max_keys": MAX_API_KEYS_PER_USER,
            },
        ).execute()
    except Exception as exc:
        msg = str(exc)
        if "Maximum of" in msg and "active API keys per user" in msg:
            raise ValueError(
                f"Maximum of {MAX_API_KEYS_PER_USER} active API keys per user"
            ) from exc
        raise

    if not result.data:
        raise RuntimeError("Failed to create API key record")
    row = _rpc_first_item(result.data)
    if not isinstance(row, dict):
        raise RuntimeError("Failed to create API key record")
    return _row_to_api_key(row)


def get_api_key_by_hash(key_hash: str) -> ApiKeyRecord | None:
    """Look up an active API key by its SHA-256 hash."""
    client = get_client()
    result = (
        client.table("api_keys")
        .select("*")
        .eq("key_hash", key_hash)
        .is_("revoked_at", "null")
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_api_key(result.data[0])


def list_api_keys(user_id: str) -> list[ApiKeyRecord]:
    """List active API keys for a user, newest first."""
    client = get_client()
    result = (
        client.table("api_keys")
        .select("*")
        .eq("user_id", user_id)
        .is_("revoked_at", "null")
        .order("created_at", desc=True)
        .execute()
    )
    return [_row_to_api_key(row) for row in result.data]


def revoke_api_key(key_id: str, user_id: str) -> bool:
    """Soft-revoke an API key. Returns True if a key was revoked."""
    client = get_client()
    result = (
        client.table("api_keys")
        .update({"revoked_at": _utc_now_iso()})
        .eq("id", key_id)
        .eq("user_id", user_id)
        .is_("revoked_at", "null")
        .execute()
    )
    return bool(result.data)


def touch_api_key_last_used(key_id: str) -> None:
    """Update last_used_at timestamp. Fire-and-forget — logs errors, never raises."""
    try:
        client = get_client()
        client.table("api_keys").update(
            {"last_used_at": _utc_now_iso()}
        ).eq("id", key_id).execute()
    except Exception as exc:
        logger.debug("Failed to update api_key last_used_at: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# MCP OAuth CRUD
# ---------------------------------------------------------------------------


def create_mcp_oauth_authorization_request(
    *,
    client_id: str,
    redirect_uri: str,
    redirect_uri_provided_explicitly: bool,
    state: str | None,
    scopes: list[str],
    code_challenge: str,
    resource: str,
    expires_at: str,
) -> McpOAuthAuthorizationRequestRecord:
    client = get_client()
    result = client.table("mcp_oauth_authorization_requests").insert(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "redirect_uri_provided_explicitly": redirect_uri_provided_explicitly,
            "state": state,
            "scopes": scopes,
            "code_challenge": code_challenge,
            "resource": resource,
            "expires_at": expires_at,
        }
    ).execute()
    if not result.data:
        raise RuntimeError("Failed to create MCP OAuth authorization request")
    return _row_to_mcp_oauth_authorization_request(result.data[0])


def get_mcp_oauth_authorization_request(
    request_id: str,
) -> McpOAuthAuthorizationRequestRecord | None:
    client = get_client()
    result = (
        client.table("mcp_oauth_authorization_requests")
        .select("*")
        .eq("id", request_id)
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_mcp_oauth_authorization_request(result.data[0])


def update_mcp_oauth_authorization_request_resolution(
    request_id: str,
    *,
    status: McpOAuthRequestStatus,
    user_id: str | None = None,
) -> McpOAuthAuthorizationRequestRecord | None:
    now = _utc_now_iso()
    update: dict[str, object] = {
        "status": status,
        "resolved_at": now,
    }
    if user_id is not None:
        update["user_id"] = user_id
    if status == "approved":
        update["approved_at"] = now
    if status == "denied":
        update["denied_at"] = now

    client = get_client()
    result = (
        client.table("mcp_oauth_authorization_requests")
        .update(update)
        .eq("id", request_id)
        .eq("status", "pending")
        .execute()
    )
    if not result.data:
        return None
    return _row_to_mcp_oauth_authorization_request(result.data[0])


def create_mcp_oauth_authorization_code(
    *,
    authorization_request_id: str | None,
    user_id: str,
    client_id: str,
    code_hash: str,
    redirect_uri: str,
    redirect_uri_provided_explicitly: bool,
    scopes: list[str],
    code_challenge: str,
    resource: str,
    expires_at: str,
) -> McpOAuthAuthorizationCodeRecord:
    client = get_client()
    result = client.table("mcp_oauth_authorization_codes").insert(
        {
            "authorization_request_id": authorization_request_id,
            "user_id": user_id,
            "client_id": client_id,
            "code_hash": code_hash,
            "redirect_uri": redirect_uri,
            "redirect_uri_provided_explicitly": redirect_uri_provided_explicitly,
            "scopes": scopes,
            "code_challenge": code_challenge,
            "resource": resource,
            "expires_at": expires_at,
        }
    ).execute()
    if not result.data:
        raise RuntimeError("Failed to create MCP OAuth authorization code")
    return _row_to_mcp_oauth_authorization_code(result.data[0])


def get_mcp_oauth_authorization_code_by_hash(
    code_hash: str,
) -> McpOAuthAuthorizationCodeRecord | None:
    client = get_client()
    result = (
        client.table("mcp_oauth_authorization_codes")
        .select("*")
        .eq("code_hash", code_hash)
        .is_("used_at", "null")
        .is_("revoked_at", "null")
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_mcp_oauth_authorization_code(result.data[0])


def mark_mcp_oauth_authorization_code_used(code_id: str) -> None:
    client = get_client()
    client.table("mcp_oauth_authorization_codes").update(
        {"used_at": _utc_now_iso()}
    ).eq("id", code_id).execute()


def create_mcp_oauth_tokens(
    *,
    connection_id: str,
    user_id: str,
    client_id: str,
    access_token_hash: str,
    refresh_token_hash: str,
    scopes: list[str],
    resource: str,
    access_expires_at: str,
    refresh_expires_at: str,
) -> tuple[McpOAuthAccessTokenRecord, McpOAuthRefreshTokenRecord]:
    client = get_client()
    access_result = client.table("mcp_oauth_access_tokens").insert(
        {
            "connection_id": connection_id,
            "user_id": user_id,
            "client_id": client_id,
            "token_hash": access_token_hash,
            "scopes": scopes,
            "resource": resource,
            "expires_at": access_expires_at,
        }
    ).execute()
    refresh_result = client.table("mcp_oauth_refresh_tokens").insert(
        {
            "connection_id": connection_id,
            "user_id": user_id,
            "client_id": client_id,
            "token_hash": refresh_token_hash,
            "scopes": scopes,
            "resource": resource,
            "expires_at": refresh_expires_at,
        }
    ).execute()
    if not access_result.data or not refresh_result.data:
        raise RuntimeError("Failed to create MCP OAuth tokens")
    return (
        _row_to_mcp_oauth_access_token(access_result.data[0]),
        _row_to_mcp_oauth_refresh_token(refresh_result.data[0]),
    )


def get_mcp_oauth_access_token_by_hash(
    token_hash: str,
) -> McpOAuthAccessTokenRecord | None:
    client = get_client()
    result = (
        client.table("mcp_oauth_access_tokens")
        .select("*")
        .eq("token_hash", token_hash)
        .is_("revoked_at", "null")
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_mcp_oauth_access_token(result.data[0])


def get_mcp_oauth_refresh_token_by_hash(
    token_hash: str,
) -> McpOAuthRefreshTokenRecord | None:
    client = get_client()
    result = (
        client.table("mcp_oauth_refresh_tokens")
        .select("*")
        .eq("token_hash", token_hash)
        .is_("revoked_at", "null")
        .limit(1)
        .execute()
    )
    if not result.data:
        return None
    return _row_to_mcp_oauth_refresh_token(result.data[0])


def revoke_mcp_oauth_access_tokens_for_connection(connection_id: str) -> None:
    client = get_client()
    client.table("mcp_oauth_access_tokens").update(
        {"revoked_at": _utc_now_iso()}
    ).eq("connection_id", connection_id).is_("revoked_at", "null").execute()


def revoke_mcp_oauth_refresh_token(refresh_token_id: str) -> None:
    client = get_client()
    client.table("mcp_oauth_refresh_tokens").update(
        {"revoked_at": _utc_now_iso()}
    ).eq("id", refresh_token_id).is_("revoked_at", "null").execute()


def revoke_mcp_oauth_tokens_by_connection_id(connection_id: str) -> None:
    now = _utc_now_iso()
    client = get_client()
    client.table("mcp_oauth_access_tokens").update({"revoked_at": now}).eq(
        "connection_id", connection_id
    ).is_("revoked_at", "null").execute()
    client.table("mcp_oauth_refresh_tokens").update({"revoked_at": now}).eq(
        "connection_id", connection_id
    ).is_("revoked_at", "null").execute()


# ---------------------------------------------------------------------------
# API billing CRUD
# ---------------------------------------------------------------------------


def _row_to_api_credit(row: dict) -> ApiCreditRecord:
    return ApiCreditRecord(
        user_id=row["user_id"],
        balance=row.get("balance", 0),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _row_to_api_usage_event(row: dict) -> ApiUsageEventRecord:
    return ApiUsageEventRecord(
        id=row["id"],
        user_id=row["user_id"],
        api_key_id=row.get("api_key_id"),
        event_type=row["event_type"],
        units=row["units"],
        video_id=row.get("video_id"),
        request_id=row.get("request_id"),
        metadata=row.get("metadata"),
        created_at=row.get("created_at"),
    )


def get_api_credits(user_id: str) -> ApiCreditRecord | None:
    """Get API credit record for a user."""
    client = get_client()
    result = client.table("api_credits").select("*").eq("user_id", user_id).execute()
    if not result.data:
        return None
    return _row_to_api_credit(result.data[0])


def apply_api_billing_credit_grant(
    *,
    provider: str,
    event_id: str,
    event_type: str,
    user_id: str,
    credits: int,
    payload: dict | None = None,
) -> ApiBillingCreditGrantResult:
    """Apply API billing credit grants atomically and idempotently."""
    if not provider.strip():
        raise ValueError("provider must be non-empty")
    if not event_id.strip():
        raise ValueError("event_id must be non-empty")
    if not event_type.strip():
        raise ValueError("event_type must be non-empty")
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if credits <= 0:
        raise ValueError("credits must be > 0")

    client = get_client()
    result = client.rpc(
        "apply_api_credit_grant",
        {
            "p_provider": provider,
            "p_event_id": event_id,
            "p_event_type": event_type,
            "p_user_id": user_id,
            "p_credits": credits,
            "p_payload": payload or {},
        },
    ).execute()
    applied = bool(_rpc_first_item(result.data))
    return ApiBillingCreditGrantResult(applied=applied)


def consume_api_units(
    user_id: str,
    api_key_id: str | None,
    event_type: str,
    units: int,
    video_id: str | None = None,
    request_id: str | None = None,
    metadata: dict | None = None,
) -> ApiUnitConsumeResult:
    """Consume API units atomically with optional idempotency."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if units <= 0:
        raise ValueError("units must be > 0")

    client = get_client()
    result = client.rpc(
        "consume_api_units",
        {
            "p_user_id": user_id,
            "p_api_key_id": api_key_id,
            "p_event_type": event_type,
            "p_units": units,
            "p_video_id": video_id,
            "p_request_id": request_id,
            "p_metadata": metadata or {},
        },
    ).execute()
    first_item = _rpc_first_item(result.data)
    if isinstance(first_item, dict):
        row = first_item
    else:
        logger.warning(
            "Unexpected consume_api_units RPC response shape: %s",
            type(first_item).__name__,
        )
        row = {}

    remaining_raw = row.get("remaining_balance", 0)
    try:
        remaining = int(remaining_raw)
    except (TypeError, ValueError):
        remaining = 0

    return ApiUnitConsumeResult(
        allowed=bool(row.get("allowed")),
        remaining_balance=remaining,
    )


def compensate_api_units(
    user_id: str,
    units: int,
    video_id: str | None = None,
    request_id: str | None = None,
    metadata: dict | None = None,
) -> None:
    """Compensate (refund) API units. Idempotent on request_id."""
    if not user_id.strip():
        raise ValueError("user_id must be non-empty")
    if units <= 0:
        raise ValueError("units must be > 0")

    client = get_client()
    client.rpc(
        "compensate_api_units",
        {
            "p_user_id": user_id,
            "p_units": units,
            "p_video_id": video_id,
            "p_request_id": request_id,
            "p_metadata": metadata or {},
        },
    ).execute()


def list_api_usage_events(
    user_id: str,
    api_key_id: str | None = None,
    limit: int = 50,
) -> list[ApiUsageEventRecord]:
    """List API usage events for a user, newest first."""
    client = get_client()
    query = (
        client.table("api_usage_events")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
    )
    if api_key_id is not None:
        query = query.eq("api_key_id", api_key_id)
    result = query.execute()
    return [_row_to_api_usage_event(row) for row in result.data]
