"""FastAPI app with real Supabase, Modal, and Qdrant integrations."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
from math import ceil
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Literal, TypeVar
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from mcp.server.auth.handlers.authorize import AuthorizationHandler
from mcp.server.auth.handlers.metadata import MetadataHandler, ProtectedResourceMetadataHandler
from mcp.server.auth.handlers.register import RegistrationHandler
from mcp.server.auth.handlers.revoke import RevocationHandler
from mcp.server.auth.handlers.token import TokenHandler
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, field_validator
from starlette.requests import ClientDisconnect

from src.analytics.events import track
from src.api.auth import AuthIdentity, get_current_user, get_current_user_id, get_optional_user_id, hash_api_key
from src.api.frames import (
    FrameRequestValidationError,
    build_high_res_frame_plan,
    build_thumb_frame_plan,
    extract_high_res_frames,
    unique_dedupe_keys,
    validate_frame_request,
)
from src.api.mcp import (
    build_mcp_asgi_app,
    mcp_tool_approval_items,
    shutdown_mcp_session_manager,
    startup_mcp_session_manager,
)
from src.api.mcp_oauth import (
    FlexibleClientAuthenticator,
    McpOAuthConfigError,
    McpOAuthFlowError,
    get_mcp_oauth_settings,
    get_mcp_oauth_provider,
    mcp_oauth_authorization_metadata,
    mcp_oauth_client_registration_options,
    mcp_oauth_protected_resource_metadata,
    mcp_oauth_request_public_payload,
)
from src.api.rate_limit import SlidingWindowRateLimiter
from src.api.search import (
    QueryImageValidationError,
    search_video_by_image as search_video_by_image_service,
    search_video_by_text as search_video_by_text_service,
)
from src.billing.lemonsqueezy import (
    LemonSqueezyConfigError,
    LemonSqueezyProviderError,
    create_checkout_session,
)
from src.config.env import load_env
from src.monitoring.sentry import capture_exception, init_sentry
from src.db.supabase import (
    VideoRecord,
    apply_api_billing_credit_grant as db_apply_api_billing_credit_grant,
    apply_billing_credit_grant as db_apply_billing_credit_grant,
    compensate_api_units as db_compensate_api_units,
    consume_api_units as db_consume_api_units,
    consume_processing_credit as db_consume_processing_credit,
    count_videos_for_user as db_count_videos_for_user,
    create_api_key as db_create_api_key,
    create_uploaded_video as db_create_uploaded_video,
    enqueue_video_job,
    get_api_credits as db_get_api_credits,
    get_credits as db_get_credits,
    get_video_job as db_get_video_job,
    get_video as db_get_video,
    get_video_transcript_segments as db_get_video_transcript_segments,
    insert_uploaded_video_idempotent as db_insert_uploaded_video_idempotent,
    insert_youtube_video_idempotent as db_insert_youtube_video_idempotent,
    has_unlimited_video_access as db_has_unlimited_video_access,
    list_api_keys as db_list_api_keys,
    list_api_usage_events as db_list_api_usage_events,
    list_videos as db_list_videos,
    revoke_api_key as db_revoke_api_key,
    update_video_status,
)
from src.storage.config import R2Config, StorageConfigError
from src.storage.r2 import R2Store, R2StorageError, source_key, thumbnail_key
from src.storage.qdrant import QdrantStorageError
from src.utils.datetime import parse_iso_datetime
from src.utils.env import get_env_int
from src.utils.logging import get_logger
from src.video.download import VideoMetadataError, fetch_video_metadata
from src.video.metadata import VideoMetadataProbeError, max_video_duration_s, probe_video_duration_s
from src.video.youtube import normalize_youtube_url
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

load_env()
logger = get_logger(__name__)
init_sentry(service="api")

StatusType = Literal["queued", "processing", "ready", "failed"]
BillingPlanType = Literal["starter", "pro", "developer"]
DEFAULT_MAX_FREE_VIDEOS = 1
DEFAULT_BILLING_GRANT_EVENTS = {
    "order_created",
    "subscription_payment_success",
}
INSUFFICIENT_CREDITS_DETAIL = "Insufficient credits. Buy credits to process another video."
INSUFFICIENT_API_UNITS_DETAIL = {
    "code": "insufficient_api_units",
    "message": "Insufficient API units. Purchase a Developer Pack to add units.",
}
DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:6274",
    "http://127.0.0.1:6274",
    "https://claude.ai",
    "https://claude.com",
]
DEFAULT_BILLING_PLAN_CREDITS: dict[BillingPlanType, int] = {
    "starter": 5,
    "pro": 20,
    "developer": 10_000,
}
DEFAULT_RATE_LIMIT_WINDOW_S = 60
DEFAULT_RATE_LIMIT_USER_WRITE_REQUESTS_PER_WINDOW = 12
DEFAULT_RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW = 30
DEFAULT_RATE_LIMIT_WEBHOOK_REQUESTS_PER_WINDOW = 60
DEFAULT_RATE_LIMIT_OAUTH_REQUESTS_PER_WINDOW = 60
MAX_QUERY_IMAGE_BYTES = 10 * 1024 * 1024
# Comfortably admits a 90-minute 1080p H.264 upload (~4-6 GB at typical
# bitrates; 8 GiB supports up to ~12.7 Mbps average bitrate over 5400s).
VIDEO_MAX_UPLOAD_BYTES = get_env_int("VIDEO_MAX_UPLOAD_BYTES", 8 * 1024**3)
BILLING_PLAN_VARIANT_ENV: dict[BillingPlanType, str] = {
    "starter": "LEMON_SQUEEZY_VARIANT_ID_STARTER",
    "pro": "LEMON_SQUEEZY_VARIANT_ID_PRO",
    "developer": "LEMON_SQUEEZY_VARIANT_ID_DEVELOPER",
}
USER_WRITE_RATE_LIMITER = SlidingWindowRateLimiter()
SEARCH_RATE_LIMITER = SlidingWindowRateLimiter()
WEBHOOK_RATE_LIMITER = SlidingWindowRateLimiter()
OAUTH_RATE_LIMITER = SlidingWindowRateLimiter()
YOUTUBE_SERVER_BLOCKED_ERROR_CODE = "youtube_server_blocked"
YOUTUBE_METADATA_BOT_CHALLENGE_DETAIL = (
    "Upload a video file instead. If this is your own YouTube video, "
    "download it from YouTube Studio or Google Takeout, then upload it here."
)
FAILED_ERROR_INSUFFICIENT_CREDITS = "Insufficient credits"
FAILED_ERROR_ENQUEUE = "Failed to enqueue processing job"
RECOVERABLE_UPLOAD_RETRY_ERRORS = {
    FAILED_ERROR_INSUFFICIENT_CREDITS,
    FAILED_ERROR_ENQUEUE,
}


class UploadDurationValidationError(RuntimeError):
    """Raised when uploaded source duration validation fails."""


class UploadDurationLimitExceededError(UploadDurationValidationError):
    """Raised when uploaded source exceeds configured duration limit."""


class UploadDurationProbeUnavailableError(UploadDurationValidationError):
    """Raised when uploaded source duration cannot be determined."""


class UploadSizeLimitExceededError(RuntimeError):
    """Raised when an uploaded source exceeds the configured max upload size."""


class UploadSizeProbeUnavailableError(RuntimeError):
    """Raised when an uploaded source's size cannot be determined."""


def _normalize_cors_origin(origin: str) -> str:
    value = origin.strip()
    if not value:
        return ""
    parsed = urlsplit(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return value.rstrip("/")


def _configured_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_CORS_ORIGINS.copy()
    origins = [_normalize_cors_origin(origin) for origin in raw.split(",")]
    cleaned = [origin for origin in origins if origin]
    return cleaned or DEFAULT_CORS_ORIGINS.copy()


def _allowed_cors_origins() -> list[str]:
    return [origin for origin in _configured_cors_origins() if "*" not in origin]


def _wildcard_origin_to_regex(origin: str) -> str:
    escaped = re.escape(origin)
    wildcard_pattern = escaped.replace(r"\*", r"[^/]+")
    return f"^{wildcard_pattern}$"


def _allowed_cors_origin_regex() -> str | None:
    patterns: list[str] = []

    explicit_regex = os.environ.get("CORS_ALLOWED_ORIGIN_REGEX", "").strip()
    if explicit_regex:
        patterns.append(f"(?:{explicit_regex})")

    wildcard_patterns = [
        _wildcard_origin_to_regex(origin)
        for origin in _configured_cors_origins()
        if "*" in origin
    ]
    patterns.extend(wildcard_patterns)

    if not patterns:
        return None
    return "|".join(patterns)


def _max_free_videos() -> int:
    return get_env_int("VIDEO_MAX_FREE_VIDEOS", DEFAULT_MAX_FREE_VIDEOS)


def _upload_duration_limit_detail() -> str:
    max_minutes = max_video_duration_s() // 60
    return f"Video exceeds {max_minutes}-minute limit"


def _upload_size_limit_detail() -> str:
    limit_gib = VIDEO_MAX_UPLOAD_BYTES / (1024**3)
    limit_label = (
        f"{int(limit_gib)} GiB" if limit_gib == int(limit_gib) else f"{limit_gib:.2f} GiB"
    )
    return f"Video exceeds {limit_label} upload size limit"


def _is_youtube_bot_challenge_error(message: str) -> bool:
    normalized = message.casefold()
    return (
        ("sign in" in normalized and "not a bot" in normalized)
        or ("--cookies-from-browser" in normalized)
        or ("http error 429" in normalized)
        or ("too many requests" in normalized)
    )


def _youtube_server_blocked_error() -> dict[str, str]:
    return {
        "code": YOUTUBE_SERVER_BLOCKED_ERROR_CODE,
        "message": YOUTUBE_METADATA_BOT_CHALLENGE_DETAIL,
    }


def _validate_video_duration(youtube_url: str) -> None:
    try:
        metadata = fetch_video_metadata(youtube_url)
    except VideoMetadataError as exc:
        logger.warning("Failed to fetch YouTube metadata for %s: %s", youtube_url, exc)
        if _is_youtube_bot_challenge_error(str(exc)):
            raise HTTPException(
                status_code=503,
                detail=_youtube_server_blocked_error(),
            ) from exc
        raise HTTPException(
            status_code=400,
            detail="Unable to fetch YouTube metadata for this URL",
        ) from exc

    if metadata.is_live:
        raise HTTPException(status_code=400, detail="Live streams are not supported")
    if metadata.duration_s is None:
        raise HTTPException(status_code=400, detail="Unable to determine video duration")

    max_duration_s = max_video_duration_s()
    if metadata.duration_s > max_duration_s:
        raise HTTPException(status_code=400, detail=_upload_duration_limit_detail())


def _validate_duration_limit_or_raise(duration_s: float) -> None:
    if duration_s > max_video_duration_s():
        raise UploadDurationLimitExceededError(_upload_duration_limit_detail())


def _rewind_upload_stream(file: UploadFile) -> None:
    try:
        file.file.seek(0)
    except Exception as exc:
        raise UploadDurationProbeUnavailableError(
            "Failed to read uploaded video for duration validation"
        ) from exc


_UPLOAD_SIZE_CHECK_CHUNK_BYTES = 1024 * 1024


def _copy_upload_stream_with_size_limit(
    file: UploadFile,
    destination,
    max_bytes: int,
    *,
    chunk_size: int = _UPLOAD_SIZE_CHECK_CHUNK_BYTES,
) -> None:
    """Stream ``file`` into ``destination``, aborting as soon as the running
    total exceeds ``max_bytes``.

    Unlike ``shutil.copyfileobj``, the size limit is enforced after every
    chunk rather than after buffering the entire upload — this keeps disk
    usage for a rejected oversized upload bounded to a few chunks instead of
    the full (potentially unbounded) file size.
    """
    total_bytes = 0
    while True:
        chunk = file.file.read(chunk_size)
        if not chunk:
            break
        total_bytes += len(chunk)
        if total_bytes > max_bytes:
            raise UploadSizeLimitExceededError(_upload_size_limit_detail())
        destination.write(chunk)


def _probe_upload_file_duration_s(file: UploadFile) -> float:
    suffix = Path(file.filename or "").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        _rewind_upload_stream(file)
        try:
            _copy_upload_stream_with_size_limit(file, temp_file, VIDEO_MAX_UPLOAD_BYTES)
            temp_file.flush()
        except UploadSizeLimitExceededError:
            raise
        except Exception as exc:
            raise UploadDurationProbeUnavailableError(
                "Failed to read uploaded video for duration validation"
            ) from exc
        finally:
            _rewind_upload_stream(file)

        try:
            return probe_video_duration_s(Path(temp_file.name))
        except VideoMetadataProbeError as exc:
            raise UploadDurationProbeUnavailableError(str(exc)) from exc


def _probe_uploaded_source_duration_s(store: R2Store, key: str) -> float:
    with tempfile.TemporaryDirectory() as temp_dir:
        probe_path = Path(temp_dir) / "upload_probe"
        try:
            store.download_source_video(key, probe_path)
        except R2StorageError as exc:
            logger.exception("Failed to download uploaded source for duration validation: %s", exc)
            raise UploadDurationProbeUnavailableError("Failed to verify upload") from exc

        try:
            return probe_video_duration_s(probe_path)
        except VideoMetadataProbeError as exc:
            raise UploadDurationProbeUnavailableError(str(exc)) from exc


def _validate_upload_file_duration_or_raise(file: UploadFile) -> float:
    """Probe and validate an uploaded file's duration, returning it for persistence."""
    duration_s = _probe_upload_file_duration_s(file)
    _validate_duration_limit_or_raise(duration_s)
    return duration_s


def _validate_uploaded_source_duration_or_raise(store: R2Store, key: str) -> float:
    """Probe and validate a stored source's duration, returning it for persistence."""
    duration_s = _probe_uploaded_source_duration_s(store, key)
    _validate_duration_limit_or_raise(duration_s)
    return duration_s


def _delete_uploaded_source_best_effort(store: R2Store, key: str, user_id: str) -> None:
    try:
        store.delete_source_object(key)
    except R2StorageError as cleanup_exc:
        logger.warning(
            "Failed to cleanup invalid uploaded source for user_id=%s key=%s: %s",
            user_id,
            key,
            cleanup_exc,
        )


def _validate_uploaded_source_duration_with_cleanup(
    store: R2Store,
    key: str,
    user_id: str,
) -> float:
    try:
        return _validate_uploaded_source_duration_or_raise(store, key)
    except UploadDurationLimitExceededError:
        _delete_uploaded_source_best_effort(store, key, user_id)
        raise


def _validate_uploaded_source_size_or_raise(store: R2Store, key: str) -> int:
    """Probe and validate a stored source's size via a cheap HEAD request.

    Runs before duration validation so an oversized upload is rejected
    without ever downloading the full object to probe it with ffprobe.
    """
    try:
        size_bytes = store.object_size(key)
    except R2StorageError as exc:
        logger.exception("Failed to check uploaded source size: %s", exc)
        raise UploadSizeProbeUnavailableError("Failed to verify upload") from exc
    if size_bytes > VIDEO_MAX_UPLOAD_BYTES:
        raise UploadSizeLimitExceededError(_upload_size_limit_detail())
    return size_bytes


def _validate_uploaded_source_size_with_cleanup(
    store: R2Store,
    key: str,
    user_id: str,
) -> int:
    try:
        return _validate_uploaded_source_size_or_raise(store, key)
    except UploadSizeLimitExceededError:
        _delete_uploaded_source_best_effort(store, key, user_id)
        raise


def _is_video_processing_free_for_user(user_id: str) -> bool:
    if db_has_unlimited_video_access(user_id):
        return True
    max_videos = _max_free_videos()
    current_count = db_count_videos_for_user(user_id)
    return current_count < max_videos


def _precheck_video_processing_admission(user_id: str) -> bool:
    """Raise 402 when admission fails; return whether paid credit consume is needed."""
    if _is_video_processing_free_for_user(user_id):
        return False
    credit_record = db_get_credits(user_id)
    if credit_record is not None and credit_record.balance > 0:
        return True
    raise HTTPException(
        status_code=402,
        detail=INSUFFICIENT_CREDITS_DETAIL,
    )


def _consume_processing_credit_or_raise(user_id: str) -> None:
    credit_result = db_consume_processing_credit(user_id)
    if credit_result.allowed:
        return

    raise HTTPException(
        status_code=402,
        detail=INSUFFICIENT_CREDITS_DETAIL,
    )


def _consume_and_admit_video_processing(user_id: str) -> None:
    if _is_video_processing_free_for_user(user_id):
        return
    _consume_processing_credit_or_raise(user_id)


# ---------------------------------------------------------------------------
# API unit billing helpers
# ---------------------------------------------------------------------------


API_UNIT_COST_INDEX_VIDEO = get_env_int("API_UNIT_COST_INDEX_VIDEO", 500)
API_UNIT_COST_TEXT_QUERY = get_env_int("API_UNIT_COST_TEXT_QUERY", 1)
API_UNIT_COST_TRANSCRIPT_FETCH = get_env_int("API_UNIT_COST_TRANSCRIPT_FETCH", 1)
API_UNIT_COST_FRAMES_THUMB = get_env_int("API_UNIT_COST_FRAMES_THUMB", 1)
API_UNIT_COST_FRAMES_HIGH = get_env_int("API_UNIT_COST_FRAMES_HIGH", 5)
SOURCE_NOT_RETAINED_DETAIL = {
    "code": "source_not_retained",
    "message": (
        "Original source video is not retained for this video. "
        "Retry with resolution=\"thumb\"."
    ),
}


def _uses_api_unit_billing(identity: AuthIdentity) -> bool:
    return identity.auth_method in {"api_key", "mcp_oauth"}


def _api_usage_key_id(identity: AuthIdentity) -> str | None:
    if identity.auth_method == "api_key":
        return identity.api_key_id
    return None


def _consume_api_units_or_raise(
    user_id: str,
    api_key_id: str | None,
    event_type: str,
    units: int,
    video_id: str | None = None,
    request_id: str | None = None,
) -> None:
    result = db_consume_api_units(
        user_id=user_id,
        api_key_id=api_key_id,
        event_type=event_type,
        units=units,
        video_id=video_id,
        request_id=request_id,
    )
    if not result.allowed:
        raise HTTPException(
            status_code=402,
            detail=INSUFFICIENT_API_UNITS_DETAIL,
        )


_MeteredResult = TypeVar("_MeteredResult")


def _bill_metered_call(
    *,
    user_id: str,
    api_key_id: str | None,
    event_type: str,
    units: int,
    video_id: str,
    work: Callable[[], _MeteredResult],
) -> _MeteredResult:
    """Consume API units, run ``work``, and refund the units if it raises.

    Mirrors the bill-then-compensate pattern already used for text search
    (see ``v1_search_video``): a request-scoped ``request_id`` ties the
    consume and compensate calls together, and any exception from ``work``
    (DB reads, storage/presign failures, or an unexpected wholesale ffmpeg
    failure) triggers a best-effort refund before re-raising, so the
    caller's error response is unaffected. Per-item errors that ``work``
    already handles internally (e.g. one bad frame timestamp among several)
    must not raise here, so they are never compensated -- only a failure of
    the call as a whole is.

    Shared by both the REST handlers (``v1_get_video_transcript``,
    ``v1_get_video_frames``) and the MCP ``get_frames`` tool, which runs its
    own retrieval flow rather than delegating to the REST handler.
    """
    request_id = f"{event_type}:{uuid4()}"
    _consume_api_units_or_raise(
        user_id=user_id,
        api_key_id=api_key_id,
        event_type=event_type,
        units=units,
        video_id=video_id,
        request_id=request_id,
    )
    try:
        return work()
    except Exception:
        try:
            db_compensate_api_units(
                user_id=user_id,
                units=units,
                video_id=video_id,
                request_id=request_id,
                metadata={"event_type": f"{event_type}_failed"},
            )
        except Exception as compensation_exc:
            logger.exception(
                "Failed to compensate billed %s for video_id=%s: %s",
                event_type,
                video_id,
                compensation_exc,
            )
        raise


def _mcp_oauth_provider_or_raise():
    try:
        get_mcp_oauth_settings()
        return get_mcp_oauth_provider()
    except McpOAuthConfigError as exc:
        raise HTTPException(status_code=503, detail="MCP OAuth is not configured") from exc


def _frontend_url_for_path(path: str) -> str:
    try:
        frontend_base = get_mcp_oauth_settings().frontend_base_url
    except McpOAuthConfigError:
        frontend_base = os.environ.get("FRONTEND_BASE_URL", "").strip().rstrip("/")
        if not frontend_base:
            raise HTTPException(status_code=503, detail="MCP OAuth is not configured")
    return f"{frontend_base}{path}"


def _source_url_ttl_s() -> int:
    return get_env_int("VIDEO_SOURCE_URL_TTL_S", 3600)


def _upload_url_ttl_s() -> int:
    return get_env_int("VIDEO_UPLOAD_URL_TTL_S", 900)


def _rate_limit_window_s() -> int:
    return get_env_int("RATE_LIMIT_WINDOW_S", DEFAULT_RATE_LIMIT_WINDOW_S)


def _user_write_rate_limit() -> int:
    return get_env_int(
        "RATE_LIMIT_USER_WRITE_REQUESTS_PER_WINDOW",
        DEFAULT_RATE_LIMIT_USER_WRITE_REQUESTS_PER_WINDOW,
    )


def _search_rate_limit() -> int:
    return get_env_int(
        "RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW",
        DEFAULT_RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW,
    )


def _webhook_rate_limit() -> int:
    return get_env_int(
        "RATE_LIMIT_WEBHOOK_REQUESTS_PER_WINDOW",
        DEFAULT_RATE_LIMIT_WEBHOOK_REQUESTS_PER_WINDOW,
    )


def _oauth_rate_limit() -> int:
    return get_env_int(
        "RATE_LIMIT_OAUTH_REQUESTS_PER_WINDOW",
        DEFAULT_RATE_LIMIT_OAUTH_REQUESTS_PER_WINDOW,
    )


def _raise_rate_limit_exceeded(retry_after_s: float) -> None:
    retry_after_seconds = max(1, ceil(retry_after_s))
    raise HTTPException(
        status_code=429,
        detail="Rate limit exceeded. Please retry later.",
        headers={"Retry-After": str(retry_after_seconds)},
    )


def _enforce_rate_limit(
    *,
    limiter: SlidingWindowRateLimiter,
    key: str,
    limit: int,
) -> None:
    result = limiter.check(
        key=key,
        limit=limit,
        window_s=_rate_limit_window_s(),
    )
    if not result.allowed:
        _raise_rate_limit_exceeded(result.retry_after_s)


def _enforce_user_write_rate_limit(user_id: str) -> None:
    _enforce_rate_limit(
        limiter=USER_WRITE_RATE_LIMITER,
        key=user_id,
        limit=_user_write_rate_limit(),
    )


def _enforce_search_rate_limit(user_id: str) -> None:
    _enforce_rate_limit(
        limiter=SEARCH_RATE_LIMITER,
        key=user_id,
        limit=_search_rate_limit(),
    )


def _request_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").strip()
    if forwarded_for:
        first = forwarded_for.split(",")[0].strip()
        if first:
            return first
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _enforce_webhook_rate_limit(request: Request) -> None:
    ip = _request_ip(request)
    _enforce_rate_limit(
        limiter=WEBHOOK_RATE_LIMITER,
        key=ip,
        limit=_webhook_rate_limit(),
    )


def _enforce_oauth_rate_limit(request: Request) -> None:
    ip = _request_ip(request)
    _enforce_rate_limit(
        limiter=OAUTH_RATE_LIMITER,
        key=ip,
        limit=_oauth_rate_limit(),
    )


def _billing_grant_event_names() -> set[str]:
    raw = os.environ.get("BILLING_GRANT_EVENT_NAMES", "").strip()
    if not raw:
        return DEFAULT_BILLING_GRANT_EVENTS
    names = {name.strip() for name in raw.split(",") if name.strip()}
    return names or DEFAULT_BILLING_GRANT_EVENTS


def _lemonsqueezy_webhook_secret() -> str:
    secret = os.environ.get("LEMON_SQUEEZY_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="Billing webhook is not configured")
    return secret


def _verify_lemonsqueezy_signature(raw_body: bytes, signature: str) -> bool:
    digest = hmac.new(
        _lemonsqueezy_webhook_secret().encode("utf-8"),
        raw_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(digest, signature.strip())


def _lemonsqueezy_event_name(payload: LemonSqueezyPayload) -> str | None:
    if payload.meta and payload.meta.event_name and payload.meta.event_name.strip():
        return payload.meta.event_name.strip()
    if payload.event_name and payload.event_name.strip():
        return payload.event_name.strip()
    return None


def _lemonsqueezy_event_id(
    payload: LemonSqueezyPayload, raw_body: bytes, event_name: str
) -> str:
    if payload.data and payload.data.id is not None:
        value = payload.data.id
        if isinstance(value, str) and value.strip():
            return f"{event_name}:{value.strip()}"
        if isinstance(value, int):
            return f"{event_name}:{value}"
    return f"{event_name}:sha256:{hashlib.sha256(raw_body).hexdigest()}"


def _extract_credit_grant(payload: LemonSqueezyPayload) -> tuple[str, int, str] | None:
    """Extract (user_id, credits, grant_target) from webhook payload."""
    if not payload.meta or not payload.meta.custom_data:
        return None
    cd = payload.meta.custom_data
    if not isinstance(cd.user_id, str) or not cd.user_id.strip():
        return None
    try:
        credits = int(str(cd.credits))
    except (TypeError, ValueError):
        return None
    if credits <= 0:
        return None
    grant_target = "web"
    if isinstance(cd.grant_target, str) and cd.grant_target.strip() in ("web", "api"):
        grant_target = cd.grant_target.strip()
    return cd.user_id.strip(), credits, grant_target


def _webhook_variant_id(payload: LemonSqueezyPayload) -> str | None:
    custom_data = payload.meta.custom_data if payload.meta else None
    variant_id = custom_data.variant_id if custom_data else None
    if not isinstance(variant_id, str):
        return None
    cleaned = variant_id.strip()
    return cleaned or None


def _parse_lemonsqueezy_payload(payload_dict: dict) -> LemonSqueezyPayload:
    """Parse a raw webhook dict into a typed payload with per-field salvage."""
    try:
        return LemonSqueezyPayload.model_validate(payload_dict)
    except ValidationError:
        pass

    meta: LemonSqueezyMeta | None = None
    data: LemonSqueezyData | None = None
    raw_meta = payload_dict.get("meta")
    raw_data = payload_dict.get("data")

    if isinstance(raw_meta, dict):
        try:
            meta = LemonSqueezyMeta.model_validate(raw_meta)
        except ValidationError:
            raw_en = raw_meta.get("event_name")
            raw_cd = raw_meta.get("custom_data")
            cd = None
            if isinstance(raw_cd, dict):
                try:
                    cd = LemonSqueezyCustomData.model_validate(raw_cd)
                except ValidationError:
                    pass
            meta = LemonSqueezyMeta(
                event_name=raw_en if isinstance(raw_en, str) else None,
                custom_data=cd,
            )

    if isinstance(raw_data, dict):
        try:
            data = LemonSqueezyData.model_validate(raw_data)
        except ValidationError:
            pass

    raw_event = payload_dict.get("event_name")
    return LemonSqueezyPayload(
        meta=meta,
        data=data,
        event_name=raw_event if isinstance(raw_event, str) else None,
    )


def _billing_plan_variant_id(plan: BillingPlanType) -> str:
    env_name = BILLING_PLAN_VARIANT_ENV[plan]
    variant_id = os.environ.get(env_name, "").strip()
    if not variant_id:
        raise LemonSqueezyConfigError(f"{env_name} environment variable is required")
    return variant_id


def _source_url_for_record(record: VideoRecord) -> str | None:
    if record.source_type != "upload":
        return None
    if record.status != "ready":
        return None
    if not record.source_r2_key:
        return None

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        logger.warning(
            "R2 config missing; cannot build source URL for video_id=%s: %s",
            record.id,
            exc,
        )
        return None

    store = R2Store(r2_config)
    try:
        if not store.source_exists(record.source_r2_key):
            return None
    except R2StorageError as exc:
        logger.warning(
            "Failed to check source existence for video_id=%s: %s",
            record.id,
            exc,
        )
        return None

    try:
        return store.generate_presigned_url(
            record.source_r2_key,
            expires_in=_source_url_ttl_s(),
        )
    except R2StorageError as exc:
        logger.warning(
            "Failed to generate source URL for video_id=%s: %s",
            record.id,
            exc,
        )
        return None


class VideoCreateRequest(BaseModel):
    youtube_url: str = Field(min_length=1, max_length=500)

    @field_validator("youtube_url")
    @classmethod
    def normalize_youtube_url(cls, value: str) -> str:
        normalized = normalize_youtube_url(value)
        if normalized is None:
            raise ValueError("youtube_url must be a valid YouTube video URL")
        return normalized


class VideoResponse(BaseModel):
    id: str
    youtube_url: HttpUrl | None
    status: StatusType
    source_type: Literal["youtube", "upload"]
    source_filename: str | None = None
    source_url: HttpUrl | None = None
    created_at: datetime
    error_message: str | None = None


class UploadInitRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    content_type: str | None = Field(default=None, max_length=200)


class UploadInitResponse(BaseModel):
    video_id: str
    key: str
    upload_url: HttpUrl
    expires_in: int


class UploadCompleteRequest(BaseModel):
    video_id: str = Field(min_length=1, max_length=200)
    filename: str = Field(min_length=1, max_length=200)

    @field_validator("video_id")
    @classmethod
    def validate_video_id(cls, value: str) -> str:
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError("video_id must be a valid UUID") from exc
        return value


class VideoSearchRequest(BaseModel):
    query_text: str | None = Field(default=None, max_length=500)
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description=(
            "Per-source result cap. Text search can return up to this many "
            "visual matches plus this many spoken matches."
        ),
    )

    @field_validator("query_text")
    @classmethod
    def normalize_query_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class SearchResult(BaseModel):
    timestamp_s: float
    thumbnail_url: HttpUrl | None = None
    score: float
    source: Literal["visual", "transcript"] = Field(
        description="Retrieval source for this result."
    )
    transcript_text: str | None = None


class VideoSearchResponse(BaseModel):
    video_id: str
    youtube_url: HttpUrl | None
    source_url: HttpUrl | None = None
    status: StatusType
    results: list[SearchResult]


class TranscriptSegmentResponse(BaseModel):
    segment_index: int
    start_s: float
    end_s: float
    text: str


class VideoTranscriptResponse(BaseModel):
    video_id: str
    has_transcript: bool
    language_code: str | None = None
    segment_count: int
    segments: list[TranscriptSegmentResponse]


class VideoFramesRequest(BaseModel):
    timestamps: list[float]
    resolution: Literal["thumb", "high"] = "thumb"


class VideoFrameResult(BaseModel):
    requested_timestamp_s: float
    actual_timestamp_s: float | None = None
    resolution: Literal["thumb", "high"]
    url: HttpUrl | None = None
    image_base64: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None


class VideoFramesResponse(BaseModel):
    frames: list[VideoFrameResult]


class BillingWebhookResponse(BaseModel):
    received: bool
    processed: bool
    granted: bool
    reason: str | None = None


class BillingCheckoutRequest(BaseModel):
    plan: BillingPlanType


class BillingCheckoutResponse(BaseModel):
    provider: Literal["lemonsqueezy"]
    plan: BillingPlanType
    credits: int
    checkout_url: HttpUrl
    test_mode: bool


class AnalyticsEventRequest(BaseModel):
    event_name: Literal["signup_complete"]
    metadata: dict | None = None


class BillingSummaryResponse(BaseModel):
    credits_balance: int
    free_videos_limit: int
    free_videos_used: int
    free_videos_remaining: int
    has_unlimited_access: bool


class ApiBillingSummaryResponse(BaseModel):
    api_units_balance: int
    unit_cost_index_video: int
    unit_cost_text_query: int
    approx_videos: int
    approx_queries: int


class ApiUsageEventResponse(BaseModel):
    id: str
    api_key_id: str | None
    event_type: str
    units: int
    video_id: str | None
    created_at: str | None


class ApiCheckoutRequest(BaseModel):
    plan: Literal["developer"]
    return_path: str | None = None

    @field_validator("return_path")
    @classmethod
    def validate_return_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if not cleaned.startswith("/"):
            raise ValueError("return_path must start with /")
        if cleaned.startswith("//"):
            raise ValueError("return_path must be a relative site path")
        return cleaned


class McpConnectorToolSummary(BaseModel):
    name: str
    title: str
    description: str
    cost: str


class McpConnectorRequestResponse(BaseModel):
    request_id: str
    client_id: str
    resource: str
    scope: str
    scopes: list[str]
    status: Literal["pending", "approved", "denied", "expired"]
    expires_at: str | None = None
    tools: list[McpConnectorToolSummary]


class McpConnectorDecisionResponse(BaseModel):
    redirect_url: str


class LemonSqueezyCustomData(BaseModel):
    model_config = ConfigDict(strict=True)

    user_id: str | None = None
    credits: Any = None
    grant_target: str | None = None
    variant_id: str | None = None


class LemonSqueezyMeta(BaseModel):
    model_config = ConfigDict(strict=True)

    event_name: str | None = None
    custom_data: LemonSqueezyCustomData | None = None


class LemonSqueezyData(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str | int | None = None


class LemonSqueezyPayload(BaseModel):
    model_config = ConfigDict(strict=True)

    meta: LemonSqueezyMeta | None = None
    data: LemonSqueezyData | None = None
    event_name: str | None = None


def _video_record_to_response(record: VideoRecord) -> VideoResponse:
    """Convert VideoRecord to VideoResponse."""
    parsed = parse_iso_datetime(record.created_at)
    if parsed is None and record.created_at:
        # parse_iso_datetime already logged a warning for the bad value
        parsed = datetime.now(timezone.utc)
    created_at = parsed or datetime.now(timezone.utc)

    return VideoResponse(
        id=record.id,
        youtube_url=record.youtube_url,
        status=record.status,
        source_type=record.source_type,
        source_filename=record.source_filename,
        source_url=_source_url_for_record(record),
        created_at=created_at,
        error_message=record.error_message,
    )


def _sanitize_filename(value: str | None) -> str:
    if not value:
        return "upload.mp4"
    name = os.path.basename(value)
    if not name.strip():
        return "upload.mp4"
    return name


def _get_idempotent_upload_record(
    *,
    video_id: str,
    user_id: str,
    source_r2_key: str,
    source_filename: str,
) -> VideoRecord | None:
    """Return an existing upload record only when request metadata matches exactly."""
    existing = db_get_video(video_id, user_id=user_id)
    if existing is None:
        return None
    return _require_matching_upload_record(
        existing,
        source_r2_key=source_r2_key,
        source_filename=source_filename,
    )


def _require_matching_upload_record(
    record: VideoRecord,
    *,
    source_r2_key: str,
    source_filename: str,
) -> VideoRecord:
    """Require an upload record to match the caller's source metadata exactly."""
    if record.source_type != "upload":
        raise HTTPException(
            status_code=409,
            detail="video_id already exists with a different source type",
        )
    if record.source_r2_key != source_r2_key or record.source_filename != source_filename:
        raise HTTPException(
            status_code=409,
            detail="video_id already exists with different upload metadata",
        )
    return record


def _ensure_enqueued(record: VideoRecord) -> None:
    """Re-enqueue a stranded video that was billed but never got a job row.

    Called on the retry path when an existing dedupe record is found.
    If the video is still ``queued`` and has no ``video_jobs`` entry, an
    enqueue failure on the original request left it stranded — try again.

    Raises 503 if the re-enqueue fails so the client knows the video is
    not yet processing and can retry.
    """
    if record.status != "queued":
        return
    if db_get_video_job(record.id) is not None:
        return
    try:
        enqueue_video_job(record.id)
        logger.info("Re-enqueued stranded video_id=%s on retry", record.id)
    except Exception as exc:
        logger.exception("Failed to re-enqueue stranded video_id=%s", record.id)
        raise HTTPException(
            status_code=503,
            detail="Video was created but processing could not be started. Please retry.",
        ) from exc


def _should_retry_failed_upload(record: VideoRecord) -> bool:
    if record.status != "failed":
        return False
    if record.error_message not in RECOVERABLE_UPLOAD_RETRY_ERRORS:
        return False
    return db_get_video_job(record.id) is None


def _reset_upload_retry_record(record: VideoRecord) -> VideoRecord:
    if record.status == "queued":
        return record

    updated = update_video_status(record.id, "queued")
    if updated is not None:
        return updated

    # DB update returned None — the row may have been deleted concurrently.
    # Patch in-memory so the response is consistent, but log since DB state
    # may now disagree.
    logger.warning(
        "update_video_status returned None for video_id=%s during retry reset",
        record.id,
    )
    record.status = "queued"
    record.error_message = None
    return record


def _enqueue_or_raise(video_id: str) -> None:
    """Enqueue a video processing job, raising 500 on failure.

    Does NOT mark the video as failed — callers that need the row to
    survive as a dedupe anchor (v1 idempotent paths) use this directly.
    """
    try:
        enqueue_video_job(video_id)
    except Exception as exc:
        logger.exception("Failed to enqueue video_id=%s: %s", video_id, exc)
        raise HTTPException(status_code=500, detail="Failed to enqueue processing job") from exc


def _enqueue_video_or_fail(video_id: str) -> None:
    """Enqueue a video processing job, marking the video as failed on error."""
    try:
        _enqueue_or_raise(video_id)
    except HTTPException:
        update_video_status(
            video_id,
            "failed",
            error_message=FAILED_ERROR_ENQUEUE,
        )
        raise


def _get_ready_video_for_search(video_id: str, user_id: str) -> VideoRecord:
    """Return a video record only when it exists, belongs to the user, and is searchable."""
    record = db_get_video(video_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    if record.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready for search (status: {record.status})",
        )
    return record


def _build_video_search_response(
    record: VideoRecord,
    results: list[Any],
) -> "VideoSearchResponse":
    return VideoSearchResponse(
        video_id=record.id,
        youtube_url=record.youtube_url,
        source_url=_source_url_for_record(record),
        status=record.status,
        results=[
            SearchResult(
                timestamp_s=r.timestamp_s,
                thumbnail_url=r.thumbnail_url,
                score=r.score,
                source=r.source,
                transcript_text=r.transcript_text,
            )
            for r in results
        ],
    )


def _read_query_image_bytes(upload: UploadFile) -> bytes:
    """Read one uploaded query image with a hard size cap."""
    image_bytes = upload.file.read(MAX_QUERY_IMAGE_BYTES + 1)
    if len(image_bytes) > MAX_QUERY_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Uploaded image exceeds 10 MB limit")
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    return image_bytes


def _raise_search_backend_unavailable(video_id: str, exc: Exception) -> None:
    logger.exception("Search backend failure for video_id=%s: %s", video_id, exc)
    raise HTTPException(
        status_code=503,
        detail="Search is temporarily unavailable. Please try again.",
    ) from exc


@asynccontextmanager
async def app_lifespan(_app: FastAPI):
    await startup_mcp_session_manager()
    try:
        yield
    finally:
        await shutdown_mcp_session_manager()


app = FastAPI(
    title="Video Moment Finder Public API",
    version="0.2.0",
    description=(
        "Agent-ready REST API for uploading a video, polling processing status, "
        "and searching by text. Internal web-only routes are excluded from this schema."
    ),
    lifespan=app_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_origin_regex=_allowed_cors_origin_regex(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def report_unhandled_exceptions(request: Request, call_next):
    try:
        return await call_next(request)
    except ClientDisconnect:
        raise
    except Exception as exc:
        await asyncio.to_thread(
            capture_exception,
            exc,
            context={
                "path": request.url.path,
                "method": request.method,
            },
        )
        raise


@app.post("/analytics/event", status_code=204, include_in_schema=False)
def analytics_event(
    request: AnalyticsEventRequest,
    user_id: str | None = Depends(get_optional_user_id),
) -> None:
    """Record a frontend-originating analytics event."""
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required for signup_complete")
    track(request.event_name, user_id=user_id, metadata=request.metadata)


def _mcp_connector_request_status(
    expires_at: str | None,
    status: Literal["pending", "approved", "denied"],
) -> Literal["pending", "approved", "denied", "expired"]:
    if status != "pending":
        return status
    parsed = parse_iso_datetime(expires_at)
    if parsed is not None and parsed <= datetime.now(timezone.utc):
        return "expired"
    return "pending"


@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_authorization_server_metadata(request: Request):
    try:
        return await MetadataHandler(mcp_oauth_authorization_metadata()).handle(request)
    except McpOAuthConfigError as exc:
        raise HTTPException(status_code=503, detail="MCP OAuth is not configured") from exc


@app.api_route("/authorize", methods=["GET", "POST"], include_in_schema=False)
async def oauth_authorize(request: Request):
    _enforce_oauth_rate_limit(request)
    provider = _mcp_oauth_provider_or_raise()
    return await AuthorizationHandler(provider).handle(request)


@app.post("/register", include_in_schema=False)
async def oauth_register(request: Request):
    _enforce_oauth_rate_limit(request)
    provider = _mcp_oauth_provider_or_raise()
    return await RegistrationHandler(
        provider,
        options=mcp_oauth_client_registration_options(),
    ).handle(request)


@app.post("/token", include_in_schema=False)
async def oauth_token(request: Request):
    _enforce_oauth_rate_limit(request)
    provider = _mcp_oauth_provider_or_raise()
    return await TokenHandler(provider, FlexibleClientAuthenticator(provider)).handle(request)


@app.post("/revoke", include_in_schema=False)
async def oauth_revoke(request: Request):
    _enforce_oauth_rate_limit(request)
    provider = _mcp_oauth_provider_or_raise()
    return await RevocationHandler(provider, FlexibleClientAuthenticator(provider)).handle(request)


@app.get("/.well-known/oauth-protected-resource/mcp", include_in_schema=False)
async def oauth_mcp_resource_metadata(request: Request):
    try:
        return await ProtectedResourceMetadataHandler(
            mcp_oauth_protected_resource_metadata()
        ).handle(request)
    except McpOAuthConfigError as exc:
        raise HTTPException(status_code=503, detail="MCP OAuth is not configured") from exc


@app.get(
    "/oauth/mcp/requests/{request_id}",
    response_model=McpConnectorRequestResponse,
    include_in_schema=False,
)
def get_mcp_connector_request(request_id: str) -> McpConnectorRequestResponse:
    provider = _mcp_oauth_provider_or_raise()
    record = provider.get_authorization_request(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Connector request not found")

    payload = mcp_oauth_request_public_payload(record)
    payload["status"] = _mcp_connector_request_status(record.expires_at, record.status)
    payload["tools"] = [McpConnectorToolSummary(**tool) for tool in mcp_tool_approval_items()]
    return McpConnectorRequestResponse(**payload)


@app.post(
    "/oauth/mcp/requests/{request_id}/approve",
    response_model=McpConnectorDecisionResponse,
    include_in_schema=False,
)
def approve_mcp_connector_request(
    request_id: str,
    user_id: str = Depends(get_current_user_id),
) -> McpConnectorDecisionResponse:
    api_credits = db_get_api_credits(user_id)
    if api_credits is None or api_credits.balance <= 0:
        raise HTTPException(status_code=402, detail=INSUFFICIENT_API_UNITS_DETAIL)

    provider = _mcp_oauth_provider_or_raise()
    try:
        redirect_url = provider.approve_authorization_request(request_id, user_id=user_id)
    except McpOAuthFlowError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return McpConnectorDecisionResponse(redirect_url=redirect_url)


@app.post(
    "/oauth/mcp/requests/{request_id}/deny",
    response_model=McpConnectorDecisionResponse,
    include_in_schema=False,
)
def deny_mcp_connector_request(
    request_id: str,
    user_id: str = Depends(get_current_user_id),
) -> McpConnectorDecisionResponse:
    _ = user_id
    provider = _mcp_oauth_provider_or_raise()
    try:
        redirect_url = provider.deny_authorization_request(request_id)
    except McpOAuthFlowError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    return McpConnectorDecisionResponse(redirect_url=redirect_url)


def _try_cleanup_r2(store: R2Store, key: str, user_id: str) -> None:
    """Best-effort delete of an R2 object. Logs on failure, never raises."""
    try:
        store.delete_source_object(key)
    except R2StorageError as exc:
        logger.warning("Failed to cleanup R2 object for user_id=%s key=%s: %s", user_id, key, exc)


def _validate_and_upload_file(
    file: UploadFile, user_id: str, video_id: str,
) -> tuple[R2Store, Any, str, bool, float]:
    """Validate an uploaded file, store it in R2, and return context for billing.

    Returns (store, upload_result, filename, requires_credit, duration_s).
    """
    if not file.filename and not file.content_type:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")
    requires_credit = _precheck_video_processing_admission(user_id)
    try:
        duration_s = _validate_upload_file_duration_or_raise(file)
    except UploadDurationLimitExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadSizeLimitExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadDurationProbeUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Failed to verify upload") from exc

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503, detail="Upload storage is not configured",
        ) from exc

    filename = _sanitize_filename(file.filename)
    store = R2Store(r2_config)
    try:
        upload_result = store.upload_source_video(
            video_id=video_id, filename=filename,
            file_obj=file.file, content_type=file.content_type,
        )
    except R2StorageError as exc:
        logger.exception("Failed to upload source video: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to store uploaded video") from exc

    return store, upload_result, filename, requires_credit, duration_s


def upload_video(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Upload a video file and enqueue durable processing job."""
    _enforce_user_write_rate_limit(user_id)
    video_id = str(uuid4())
    store, upload_result, filename, requires_credit, duration_s = _validate_and_upload_file(
        file, user_id, video_id,
    )

    try:
        if requires_credit:
            _consume_processing_credit_or_raise(user_id)
    except HTTPException as exc:
        if exc.status_code == 402:
            _try_cleanup_r2(store, upload_result.key, user_id)
        raise

    record = db_create_uploaded_video(
        video_id=video_id,
        source_r2_key=upload_result.key,
        source_filename=filename,
        user_id=user_id,
        status="queued",
        duration_s=duration_s,
    )
    _enqueue_video_or_fail(record.id)
    track("video_submitted", user_id=user_id, metadata={"source_type": "upload"})
    return _video_record_to_response(record)


def init_upload(
    request: UploadInitRequest,
    user_id: str = Depends(get_current_user_id),
) -> UploadInitResponse:
    """Generate a presigned upload URL for direct-to-R2 uploads."""
    _enforce_user_write_rate_limit(user_id)
    if request.content_type and not request.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")
    _precheck_video_processing_admission(user_id)

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload storage is not configured",
        ) from exc

    video_id = str(uuid4())
    filename = _sanitize_filename(request.filename)
    key = source_key(video_id, filename)
    expires_in = _upload_url_ttl_s()
    store = R2Store(r2_config)

    try:
        upload_url = store.generate_presigned_upload_url(
            key,
            content_type=request.content_type,
            expires_in=expires_in,
        )
    except R2StorageError as exc:
        logger.exception("Failed to generate upload URL: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to prepare upload",
        ) from exc

    return UploadInitResponse(
        video_id=video_id,
        key=key,
        upload_url=upload_url,
        expires_in=expires_in,
    )


def _complete_upload_core(
    video_id: str,
    filename: str,
    user_id: str,
    bill: Callable[[VideoRecord], None],
    admit: Callable[[], None] | None = None,
) -> VideoResponse:
    """Shared upload-complete flow for both JWT and API-key paths.

    ``admit`` runs after the idempotent check but before R2 validation and
    DB insert — the correct point for free-tier / credit pre-checks.
    ``bill`` runs after the DB insert and should raise on failure.
    """
    key = source_key(video_id, filename)

    retry_record: VideoRecord | None = None
    existing = _get_idempotent_upload_record(
        video_id=video_id,
        user_id=user_id,
        source_r2_key=key,
        source_filename=filename,
    )
    if existing is not None:
        if _should_retry_failed_upload(existing):
            retry_record = existing
        else:
            _ensure_enqueued(existing)
            return _video_record_to_response(existing)

    if admit is not None:
        admit()

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload storage is not configured",
        ) from exc

    store = R2Store(r2_config)

    try:
        if not store.source_exists(key):
            raise HTTPException(
                status_code=400,
                detail="Uploaded source not found",
            )
    except R2StorageError as exc:
        logger.exception("Failed to check uploaded source: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to verify upload",
        ) from exc

    # Check size via a cheap HEAD request before ffprobe, which downloads the
    # full object — this avoids paying that download cost for an oversized
    # upload we are about to reject anyway.
    try:
        _validate_uploaded_source_size_with_cleanup(store, key, user_id)
    except UploadSizeLimitExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadSizeProbeUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Failed to verify upload") from exc

    try:
        duration_s = _validate_uploaded_source_duration_with_cleanup(store, key, user_id)
    except UploadDurationLimitExceededError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UploadDurationProbeUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Failed to verify upload") from exc

    if retry_record is None:
        record, created = db_insert_uploaded_video_idempotent(
            video_id, user_id, key, filename, duration_s=duration_s,
        )
        if not created:
            record = _require_matching_upload_record(
                record, source_r2_key=key, source_filename=filename,
            )
            if _should_retry_failed_upload(record):
                retry_record = record
            else:
                _ensure_enqueued(record)
                return _video_record_to_response(record)
    else:
        record = retry_record

    bill(record)

    if retry_record is not None:
        record = _reset_upload_retry_record(record)
    _enqueue_video_or_fail(record.id)
    track("video_submitted", user_id=user_id, metadata={"source_type": "upload"})
    return _video_record_to_response(record)


def complete_upload(
    request: UploadCompleteRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Finalize a presigned upload and enqueue processing."""
    _enforce_user_write_rate_limit(user_id)
    filename = _sanitize_filename(request.filename)
    requires_credit = False

    def _admit() -> None:
        nonlocal requires_credit
        requires_credit = _precheck_video_processing_admission(user_id)

    def _bill_web_credits(record: VideoRecord) -> None:
        if not requires_credit:
            return
        try:
            _consume_processing_credit_or_raise(user_id)
        except HTTPException as exc:
            if exc.status_code == 402:
                update_video_status(record.id, "failed", error_message=FAILED_ERROR_INSUFFICIENT_CREDITS)
            raise

    return _complete_upload_core(
        request.video_id, filename, user_id, _bill_web_credits, admit=_admit,
    )


def get_video(
    video_id: str,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Get video status and details."""
    record = db_get_video(video_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")

    return _video_record_to_response(record)


def list_my_videos(
    user_id: str = Depends(get_current_user_id),
) -> list[VideoResponse]:
    """List videos for the authenticated user."""
    records = db_list_videos(user_id=user_id)
    return [_video_record_to_response(record) for record in records]


def get_billing_summary(
    user_id: str = Depends(get_current_user_id),
) -> BillingSummaryResponse:
    """Return billing-relevant usage and credit balance for the authenticated user."""
    max_free_videos = _max_free_videos()
    used_videos = db_count_videos_for_user(user_id)
    credit_record = db_get_credits(user_id)
    has_unlimited_access = db_has_unlimited_video_access(user_id)
    free_videos_remaining = max(max_free_videos - used_videos, 0)
    raw_credits_balance = credit_record.balance if credit_record else 0
    credits_balance = max(raw_credits_balance, 0)

    return BillingSummaryResponse(
        credits_balance=credits_balance,
        free_videos_limit=max_free_videos,
        free_videos_used=used_videos,
        free_videos_remaining=free_videos_remaining,
        has_unlimited_access=has_unlimited_access,
    )


def search_video(
    video_id: str,
    request: VideoSearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoSearchResponse:
    """Search for moments in a processed video."""
    _enforce_search_rate_limit(user_id)
    if not request.query_text:
        raise HTTPException(status_code=400, detail="Provide query_text")

    record = _get_ready_video_for_search(video_id, user_id)
    track("search_run", user_id=user_id, metadata={"video_id": video_id, "mode": "text"})

    try:
        results = search_video_by_text_service(
            video_id=video_id,
            query_text=request.query_text,
            limit=request.limit,
        )
    except (QdrantStorageError, StorageConfigError, RuntimeError) as exc:
        _raise_search_backend_unavailable(video_id, exc)

    track("search_success", user_id=user_id, metadata={"video_id": video_id, "mode": "text", "result_count": len(results)})
    return _build_video_search_response(record, results)


def search_video_by_image(
    video_id: str,
    query_image: UploadFile = File(...),
    limit: int = Form(default=5, ge=1, le=20),
    user_id: str = Depends(get_current_user_id),
) -> VideoSearchResponse:
    """Search for moments in a processed video using an uploaded image."""
    _enforce_search_rate_limit(user_id)
    record = _get_ready_video_for_search(video_id, user_id)

    if not query_image.content_type or not query_image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image uploads are supported")

    image_bytes = _read_query_image_bytes(query_image)
    track("search_run", user_id=user_id, metadata={"video_id": video_id, "mode": "image"})

    try:
        results = search_video_by_image_service(
            video_id=video_id,
            query_image_bytes=image_bytes,
            limit=limit,
        )
    except QueryImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (QdrantStorageError, StorageConfigError, RuntimeError) as exc:
        _raise_search_backend_unavailable(video_id, exc)

    track("search_success", user_id=user_id, metadata={"video_id": video_id, "mode": "image", "result_count": len(results)})
    return _build_video_search_response(record, results)


def create_billing_checkout(
    request: BillingCheckoutRequest,
    user_id: str = Depends(get_current_user_id),
) -> BillingCheckoutResponse:
    """Create a Lemon Squeezy checkout session for a paid credit pack."""
    _enforce_user_write_rate_limit(user_id)
    credits = DEFAULT_BILLING_PLAN_CREDITS[request.plan]
    try:
        session = create_checkout_session(
            user_id=user_id,
            plan=request.plan,
            credits=credits,
            variant_id=_billing_plan_variant_id(request.plan),
        )
    except LemonSqueezyConfigError as exc:
        logger.error("Lemon Squeezy checkout config error: %s", exc)
        raise HTTPException(status_code=503, detail="Billing checkout is not configured") from exc
    except LemonSqueezyProviderError as exc:
        logger.exception(
            "Lemon Squeezy checkout failed for user_id=%s plan=%s: %s",
            user_id,
            request.plan,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Billing checkout is temporarily unavailable",
        ) from exc

    track("checkout_started", user_id=user_id, metadata={"plan": request.plan})
    return BillingCheckoutResponse(
        provider="lemonsqueezy",
        plan=request.plan,
        credits=credits,
        checkout_url=session.url,
        test_mode=session.test_mode,
    )


@app.post(
    "/webhooks/lemonsqueezy",
    response_model=BillingWebhookResponse,
    include_in_schema=False,
)
async def lemonsqueezy_webhook(request: Request) -> BillingWebhookResponse:
    """Handle Lemon Squeezy webhook events and apply idempotent credit grants."""
    _enforce_webhook_rate_limit(request)
    signature = request.headers.get("x-signature", "").strip()
    if not signature:
        raise HTTPException(status_code=401, detail="Missing webhook signature")

    raw_body = await request.body()
    if not _verify_lemonsqueezy_signature(raw_body, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    try:
        payload_dict = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid webhook payload") from exc

    if not isinstance(payload_dict, dict):
        raise HTTPException(status_code=400, detail="Invalid webhook payload")

    payload = _parse_lemonsqueezy_payload(payload_dict)

    event_name = _lemonsqueezy_event_name(payload)
    if not event_name:
        raise HTTPException(status_code=400, detail="Missing event name")

    if event_name not in _billing_grant_event_names():
        return BillingWebhookResponse(
            received=True,
            processed=False,
            granted=False,
            reason=f"Event {event_name} is ignored",
        )

    grant = _extract_credit_grant(payload)
    if grant is None:
        return BillingWebhookResponse(
            received=True,
            processed=False,
            granted=False,
            reason="No credit grant metadata found in meta.custom_data",
        )

    user_id, credits, grant_target = grant
    event_id = _lemonsqueezy_event_id(payload, raw_body, event_name)

    if grant_target == "api":
        try:
            expected_variant = _billing_plan_variant_id("developer")
        except LemonSqueezyConfigError as exc:
            raise HTTPException(status_code=503, detail="Billing checkout is not configured") from exc
        actual_variant = _webhook_variant_id(payload)
        if actual_variant != expected_variant:
            logger.warning(
                "API grant variant mismatch: expected=%s actual=%s event_id=%s",
                expected_variant,
                actual_variant,
                event_id,
            )
            return BillingWebhookResponse(
                received=True,
                processed=False,
                granted=False,
                reason="API grant variant mismatch",
            )
        credits = DEFAULT_BILLING_PLAN_CREDITS["developer"]
        applied = db_apply_api_billing_credit_grant(
            provider="lemonsqueezy",
            event_id=event_id,
            event_type=event_name,
            user_id=user_id,
            credits=credits,
            payload=payload_dict,
        ).applied
    else:
        applied = db_apply_billing_credit_grant(
            provider="lemonsqueezy",
            event_id=event_id,
            event_type=event_name,
            user_id=user_id,
            credits=credits,
            payload=payload_dict,
        ).applied

    if applied:
        track("checkout_success", user_id=user_id, metadata={"credits": credits, "grant_target": grant_target})
    logger.info(
        "Lemon webhook processed event=%s event_id=%s user_id=%s credits=%s applied=%s",
        event_name,
        event_id,
        user_id,
        credits,
        applied,
    )
    return BillingWebhookResponse(
        received=True,
        processed=True,
        granted=applied,
        reason=None if applied else "Event already applied",
    )


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


class CreateApiKeyRequest(BaseModel):
    name: str = ""


class ApiKeyResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    key_prefix: str
    created_at: str | None = None
    last_used_at: str | None = None


class ApiKeyCreatedResponse(ApiKeyResponse):
    key: str


def create_api_key(
    body: CreateApiKeyRequest,
    identity: AuthIdentity = Depends(get_current_user),
) -> ApiKeyCreatedResponse:
    """Create a new API key. The raw key is returned once and never stored."""
    raw_key = "vmf_" + secrets.token_hex(16)
    key_hash = hash_api_key(raw_key)
    key_prefix = "vmf_" + raw_key[4:8]

    try:
        record = db_create_api_key(identity.user_id, body.name, key_hash, key_prefix)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ApiKeyCreatedResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        created_at=record.created_at,
        last_used_at=record.last_used_at,
        key=raw_key,
    )


def list_api_keys(
    identity: AuthIdentity = Depends(get_current_user),
) -> list[ApiKeyResponse]:
    """List the caller's active API keys."""
    records = db_list_api_keys(identity.user_id)
    return [ApiKeyResponse.model_validate(r) for r in records]


def revoke_api_key(
    key_id: str,
    identity: AuthIdentity = Depends(get_current_user),
) -> None:
    """Revoke an API key (soft delete)."""
    revoked = db_revoke_api_key(key_id, identity.user_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="API key not found")


# ---------------------------------------------------------------------------
# Versioned API (v1)
# ---------------------------------------------------------------------------

# Public developer routes accept JWT or API keys when appropriate.
# Internal web-only v1 routes remain JWT-only and are excluded from the
# curated public schema.


def v1_create_video(
    request: VideoCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    _enforce_user_write_rate_limit(user_id)
    _validate_video_duration(request.youtube_url)

    # Atomic insert: the unique partial index on (user_id, youtube_url) serializes
    # concurrent retries at the DB level.  The record exists BEFORE billing so a
    # racing retry sees it and short-circuits — no double charge.
    record, created = db_insert_youtube_video_idempotent(
        request.youtube_url, user_id
    )
    if not created:
        _ensure_enqueued(record)
        return _video_record_to_response(record)

    try:
        _consume_and_admit_video_processing(user_id)
    except HTTPException:
        update_video_status(record.id, "failed", error_message=FAILED_ERROR_INSUFFICIENT_CREDITS)
        raise

    _enqueue_or_raise(record.id)
    track("video_submitted", user_id=user_id, metadata={"source_type": "youtube"})
    return _video_record_to_response(record)


def v1_upload_video(
    file: UploadFile = File(...),
    identity: AuthIdentity = Depends(get_current_user),
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> VideoResponse:
    if idempotency_key is None:
        if _uses_api_unit_billing(identity):
            # API-key upload: same flow as upload_video but with API unit billing
            # instead of web credit admission.
            user_id = identity.user_id
            _enforce_user_write_rate_limit(user_id)
            if not file.filename and not file.content_type:
                raise HTTPException(status_code=400, detail="No file uploaded")
            if file.content_type and not file.content_type.startswith("video/"):
                raise HTTPException(status_code=400, detail="Only video uploads are supported")
            try:
                duration_s = _validate_upload_file_duration_or_raise(file)
            except UploadDurationLimitExceededError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except UploadSizeLimitExceededError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except UploadDurationProbeUnavailableError as exc:
                raise HTTPException(status_code=503, detail="Failed to verify upload") from exc

            video_id = str(uuid4())
            try:
                r2_config = R2Config.from_env()
            except StorageConfigError as exc:
                raise HTTPException(status_code=503, detail="Upload storage is not configured") from exc

            filename = _sanitize_filename(file.filename)
            store = R2Store(r2_config)
            try:
                upload_result = store.upload_source_video(
                    video_id=video_id, filename=filename,
                    file_obj=file.file, content_type=file.content_type,
                )
            except R2StorageError as exc:
                logger.exception("Failed to upload source video: %s", exc)
                raise HTTPException(status_code=503, detail="Failed to store uploaded video") from exc

            try:
                _consume_api_units_or_raise(
                    user_id=user_id, api_key_id=_api_usage_key_id(identity),
                    event_type="index_video", units=API_UNIT_COST_INDEX_VIDEO,
                )
            except HTTPException as exc:
                if exc.status_code == 402:
                    _try_cleanup_r2(store, upload_result.key, user_id)
                raise

            record = db_create_uploaded_video(
                video_id=video_id, source_r2_key=upload_result.key,
                source_filename=filename, user_id=user_id, status="queued",
                duration_s=duration_s,
            )
            _enqueue_video_or_fail(record.id)
            track("video_submitted", user_id=user_id, metadata={"source_type": "upload"})
            return _video_record_to_response(record)
        return upload_video(file, user_id=identity.user_id)

    # Idempotent upload: deterministic video_id from the key so concurrent
    # retries collide on PK.  DB insert happens BEFORE billing.
    user_id = identity.user_id
    video_id = str(uuid5(NAMESPACE_URL, f"{user_id}:{idempotency_key}"))
    filename = _sanitize_filename(file.filename)
    expected_key = source_key(video_id, filename)

    # Fast path: if the record already exists, skip file validation, R2
    # upload, and billing entirely.  Re-enqueue if the original request
    # failed after billing but before creating the job row.
    existing = db_get_video(video_id, user_id=user_id)
    if existing is not None:
        existing = _require_matching_upload_record(
            existing,
            source_r2_key=expected_key,
            source_filename=filename,
        )
        _ensure_enqueued(existing)
        return _video_record_to_response(existing)

    _enforce_user_write_rate_limit(user_id)

    if _uses_api_unit_billing(identity):
        # API-key idempotent upload: inline file validation to skip web
        # admission (_precheck_video_processing_admission).
        if not file.filename and not file.content_type:
            raise HTTPException(status_code=400, detail="No file uploaded")
        if file.content_type and not file.content_type.startswith("video/"):
            raise HTTPException(status_code=400, detail="Only video uploads are supported")
        try:
            duration_s = _validate_upload_file_duration_or_raise(file)
        except UploadDurationLimitExceededError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UploadSizeLimitExceededError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UploadDurationProbeUnavailableError as exc:
            raise HTTPException(status_code=503, detail="Failed to verify upload") from exc

        try:
            r2_config = R2Config.from_env()
        except StorageConfigError as exc:
            raise HTTPException(status_code=503, detail="Upload storage is not configured") from exc

        store = R2Store(r2_config)
        upload_filename = _sanitize_filename(file.filename)
        try:
            upload_result = store.upload_source_video(
                video_id=video_id, filename=upload_filename,
                file_obj=file.file, content_type=file.content_type,
            )
        except R2StorageError as exc:
            logger.exception("Failed to upload source video: %s", exc)
            raise HTTPException(status_code=503, detail="Failed to store uploaded video") from exc

        record, created = db_insert_uploaded_video_idempotent(
            video_id, user_id, upload_result.key, upload_filename, duration_s=duration_s,
        )
        if not created:
            record = _require_matching_upload_record(
                record, source_r2_key=upload_result.key, source_filename=upload_filename,
            )
            if upload_result.key != record.source_r2_key:
                _try_cleanup_r2(store, upload_result.key, user_id)
            _ensure_enqueued(record)
            return _video_record_to_response(record)

        try:
            _consume_api_units_or_raise(
                user_id=user_id, api_key_id=_api_usage_key_id(identity),
                event_type="index_video", units=API_UNIT_COST_INDEX_VIDEO,
                video_id=video_id,
            )
        except HTTPException as exc:
            if exc.status_code == 402:
                _try_cleanup_r2(store, upload_result.key, user_id)
                update_video_status(record.id, "failed", error_message=FAILED_ERROR_INSUFFICIENT_CREDITS)
            raise

        _enqueue_or_raise(record.id)
        track("video_submitted", user_id=user_id, metadata={"source_type": "upload"})
        return _video_record_to_response(record)

    # JWT idempotent upload: uses web-credit admission via _validate_and_upload_file
    store, upload_result, filename, requires_credit, duration_s = _validate_and_upload_file(
        file, user_id, video_id,
    )

    # Atomic insert: PK constraint serializes concurrent retries.
    record, created = db_insert_uploaded_video_idempotent(
        video_id, user_id, upload_result.key, filename, duration_s=duration_s,
    )
    if not created:
        record = _require_matching_upload_record(
            record,
            source_r2_key=upload_result.key,
            source_filename=filename,
        )
        # Only delete the loser-side object when it uses a different key.
        # Same-key retries share the winner's object and must not delete it.
        if upload_result.key != record.source_r2_key:
            _try_cleanup_r2(store, upload_result.key, user_id)
        _ensure_enqueued(record)
        return _video_record_to_response(record)

    if requires_credit:
        try:
            _consume_processing_credit_or_raise(user_id)
        except HTTPException as exc:
            if exc.status_code == 402:
                _try_cleanup_r2(store, upload_result.key, user_id)
                update_video_status(record.id, "failed", error_message=FAILED_ERROR_INSUFFICIENT_CREDITS)
            raise

    _enqueue_or_raise(record.id)
    track("video_submitted", user_id=user_id, metadata={"source_type": "upload"})
    return _video_record_to_response(record)


def v1_init_upload(
    request: UploadInitRequest,
    identity: AuthIdentity = Depends(get_current_user),
) -> UploadInitResponse:
    if not _uses_api_unit_billing(identity):
        return init_upload(request, user_id=identity.user_id)

    # API-key path: soft-check API balance, skip web admission
    user_id = identity.user_id
    _enforce_user_write_rate_limit(user_id)
    if request.content_type and not request.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")

    api_credits = db_get_api_credits(user_id)
    if api_credits is None or api_credits.balance < API_UNIT_COST_INDEX_VIDEO:
        raise HTTPException(status_code=402, detail=INSUFFICIENT_API_UNITS_DETAIL)

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(status_code=503, detail="Upload storage is not configured") from exc

    video_id = str(uuid4())
    filename = _sanitize_filename(request.filename)
    key = source_key(video_id, filename)
    expires_in = _upload_url_ttl_s()
    store = R2Store(r2_config)

    try:
        upload_url = store.generate_presigned_upload_url(
            key, content_type=request.content_type, expires_in=expires_in,
        )
    except R2StorageError as exc:
        logger.exception("Failed to generate upload URL: %s", exc)
        raise HTTPException(status_code=503, detail="Failed to prepare upload") from exc

    return UploadInitResponse(video_id=video_id, key=key, upload_url=upload_url, expires_in=expires_in)


def v1_complete_upload(
    request: UploadCompleteRequest,
    identity: AuthIdentity = Depends(get_current_user),
) -> VideoResponse:
    if not _uses_api_unit_billing(identity):
        return complete_upload(request, user_id=identity.user_id)

    user_id = identity.user_id
    _enforce_user_write_rate_limit(user_id)
    filename = _sanitize_filename(request.filename)

    def _bill_api_units(record: VideoRecord) -> None:
        try:
            _consume_api_units_or_raise(
                user_id=user_id, api_key_id=_api_usage_key_id(identity),
                event_type="index_video", units=API_UNIT_COST_INDEX_VIDEO,
                video_id=request.video_id,
            )
        except HTTPException as exc:
            if exc.status_code == 402:
                update_video_status(record.id, "failed", error_message=FAILED_ERROR_INSUFFICIENT_CREDITS)
            raise

    return _complete_upload_core(request.video_id, filename, user_id, _bill_api_units)


def v1_get_video(
    video_id: str,
    identity: AuthIdentity = Depends(get_current_user),
) -> VideoResponse:
    return get_video(video_id, user_id=identity.user_id)


def v1_list_my_videos(
    identity: AuthIdentity = Depends(get_current_user),
) -> list[VideoResponse]:
    return list_my_videos(user_id=identity.user_id)


def v1_search_video(
    video_id: str,
    request: VideoSearchRequest,
    identity: AuthIdentity = Depends(get_current_user),
) -> VideoSearchResponse:
    if not _uses_api_unit_billing(identity):
        return search_video(video_id, request, user_id=identity.user_id)

    # API-key path: validate, bill, then search (rate limiter runs once)
    user_id = identity.user_id
    _enforce_search_rate_limit(user_id)
    if not request.query_text:
        raise HTTPException(status_code=400, detail="Provide query_text")
    record = _get_ready_video_for_search(video_id, user_id)
    request_id = f"text_query:{uuid4()}"
    _consume_api_units_or_raise(
        user_id=user_id,
        api_key_id=_api_usage_key_id(identity),
        event_type="text_query",
        units=API_UNIT_COST_TEXT_QUERY,
        video_id=video_id,
        request_id=request_id,
    )
    track("search_run", user_id=user_id, metadata={"video_id": video_id, "mode": "text"})
    try:
        results = search_video_by_text_service(
            video_id=video_id,
            query_text=request.query_text,
            limit=request.limit,
        )
    except (QdrantStorageError, StorageConfigError, RuntimeError) as exc:
        try:
            db_compensate_api_units(
                user_id=user_id,
                units=API_UNIT_COST_TEXT_QUERY,
                video_id=video_id,
                request_id=request_id,
                metadata={"event_type": "text_query_failed"},
            )
        except Exception as compensation_exc:
            logger.exception(
                "Failed to compensate billed text search for video_id=%s: %s",
                video_id,
                compensation_exc,
            )
        _raise_search_backend_unavailable(video_id, exc)

    track("search_success", user_id=user_id, metadata={"video_id": video_id, "mode": "text", "result_count": len(results)})
    return _build_video_search_response(record, results)


def v1_search_video_by_image(
    video_id: str,
    query_image: UploadFile = File(...),
    limit: int = Form(default=5, ge=1, le=20),
    user_id: str = Depends(get_current_user_id),
) -> VideoSearchResponse:
    return search_video_by_image(
        video_id=video_id,
        query_image=query_image,
        limit=limit,
        user_id=user_id,
    )


def _require_owned_video_or_404(video_id: str, user_id: str) -> VideoRecord:
    """Return a video record only when it exists and belongs to the user."""
    record = db_get_video(video_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    return record


def _require_ready_owned_video_or_404(video_id: str, user_id: str) -> VideoRecord:
    """Return a video record only when it exists, belongs to the user, and is ready.

    Mirrors the ready-status gate ``_get_ready_video_for_search`` applies
    before search (same 400 status code and error shape), so transcript and
    frame retrieval on a queued/processing/failed video fail cleanly instead
    of consuming API units for an empty or unusable result.
    """
    record = _require_owned_video_or_404(video_id, user_id)
    if record.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready (status: {record.status})",
        )
    return record


def _raise_transcript_fetch_unavailable(video_id: str, exc: Exception) -> None:
    logger.exception("Transcript fetch failed for video_id=%s: %s", video_id, exc)
    raise HTTPException(
        status_code=503,
        detail="Failed to fetch transcript. Please try again.",
    ) from exc


def v1_get_video_transcript(
    video_id: str,
    start_s: float | None = None,
    end_s: float | None = None,
    identity: AuthIdentity = Depends(get_current_user),
) -> VideoTranscriptResponse:
    """Return transcript segments for a video, optionally filtered to a time range."""
    user_id = identity.user_id
    _enforce_search_rate_limit(user_id)
    _require_ready_owned_video_or_404(video_id, user_id)

    def _fetch_segments() -> list[Any]:
        try:
            return db_get_video_transcript_segments(video_id, start_s=start_s, end_s=end_s)
        except Exception as exc:
            _raise_transcript_fetch_unavailable(video_id, exc)
            raise  # pragma: no cover - _raise_transcript_fetch_unavailable always raises

    if _uses_api_unit_billing(identity):
        segments = _bill_metered_call(
            user_id=user_id,
            api_key_id=_api_usage_key_id(identity),
            event_type="transcript_fetch",
            units=API_UNIT_COST_TRANSCRIPT_FETCH,
            video_id=video_id,
            work=_fetch_segments,
        )
    else:
        segments = _fetch_segments()

    language_code = segments[0].language_code if segments else None

    return VideoTranscriptResponse(
        video_id=video_id,
        has_transcript=bool(segments),
        language_code=language_code,
        segment_count=len(segments),
        segments=[
            TranscriptSegmentResponse(
                segment_index=segment.segment_index,
                start_s=segment.start_s,
                end_s=segment.end_s,
                text=segment.text,
            )
            for segment in segments
        ],
    )


def _thumb_frames_response(
    video_id: str, timestamps: list[float], *, duration_s: float | None,
) -> VideoFramesResponse:
    plan = build_thumb_frame_plan(timestamps, duration_s=duration_s)

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503, detail="Frame storage is not configured",
        ) from exc
    store = R2Store(r2_config)

    urls_by_key: dict[int, str] = {}
    errors_by_key: dict[int, str] = {}
    for frame_index in unique_dedupe_keys(plan):
        try:
            urls_by_key[frame_index] = store.generate_presigned_url(
                thumbnail_key(video_id, frame_index), expires_in=3600,
            )
        except R2StorageError as exc:
            logger.warning(
                "Failed to presign thumbnail URL for video_id=%s frame_index=%d: %s",
                video_id,
                frame_index,
                exc,
            )
            errors_by_key[frame_index] = "Failed to generate thumbnail URL"

    return VideoFramesResponse(
        frames=[
            VideoFrameResult(
                requested_timestamp_s=item.requested_timestamp_s,
                actual_timestamp_s=item.actual_timestamp_s,
                resolution="thumb",
                url=urls_by_key.get(item.dedupe_key),
                error=errors_by_key.get(item.dedupe_key),
            )
            for item in plan
        ]
    )


def _require_retained_source_url(video_id: str, record: VideoRecord) -> str:
    """Return a presigned source URL, or raise 409/503 for missing retention."""
    if record.source_type != "upload" or not record.source_r2_key:
        raise HTTPException(status_code=409, detail=SOURCE_NOT_RETAINED_DETAIL)

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503, detail="Video storage is not configured",
        ) from exc
    store = R2Store(r2_config)

    try:
        exists = store.source_exists(record.source_r2_key)
    except R2StorageError as exc:
        logger.exception(
            "Failed to check retained source for video_id=%s: %s", video_id, exc,
        )
        raise HTTPException(
            status_code=503, detail="Failed to verify retained source",
        ) from exc

    if not exists:
        raise HTTPException(status_code=409, detail=SOURCE_NOT_RETAINED_DETAIL)

    try:
        return store.generate_presigned_url(record.source_r2_key, expires_in=3600)
    except R2StorageError as exc:
        logger.exception(
            "Failed to presign source URL for video_id=%s: %s", video_id, exc,
        )
        raise HTTPException(
            status_code=503, detail="Failed to prepare source video access",
        ) from exc


def _high_res_frames_response(source_url: str, timestamps: list[float]) -> VideoFramesResponse:
    plan = build_high_res_frame_plan(timestamps)
    extracted_by_key = extract_high_res_frames(source_url, unique_dedupe_keys(plan))

    frames: list[VideoFrameResult] = []
    for item in plan:
        extracted = extracted_by_key.get(item.dedupe_key)
        frames.append(
            VideoFrameResult(
                requested_timestamp_s=item.requested_timestamp_s,
                actual_timestamp_s=item.actual_timestamp_s,
                resolution="high",
                image_base64=extracted.image_base64 if extracted else None,
                width=extracted.width if extracted else None,
                height=extracted.height if extracted else None,
                error=(extracted.error if extracted else "Frame extraction failed"),
            )
        )
    return VideoFramesResponse(frames=frames)


def v1_get_video_frames(
    video_id: str,
    request: VideoFramesRequest,
    identity: AuthIdentity = Depends(get_current_user),
) -> VideoFramesResponse:
    """Return frames for a video at the requested timestamps.

    ``resolution="thumb"`` returns presigned URLs to already-stored 1-fps
    thumbnails. ``resolution="high"`` extracts frames on demand from the
    retained source video and returns them as base64 JPEG bytes.
    """
    user_id = identity.user_id
    try:
        validate_frame_request(request.timestamps, request.resolution)
    except FrameRequestValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _enforce_search_rate_limit(user_id)
    record = _require_ready_owned_video_or_404(video_id, user_id)

    if request.resolution == "high":
        source_url = _require_retained_source_url(video_id, record)

        def _extract_high_res() -> VideoFramesResponse:
            return _high_res_frames_response(source_url, request.timestamps)

        if _uses_api_unit_billing(identity):
            return _bill_metered_call(
                user_id=user_id,
                api_key_id=_api_usage_key_id(identity),
                event_type="frames_high",
                units=API_UNIT_COST_FRAMES_HIGH,
                video_id=video_id,
                work=_extract_high_res,
            )
        return _extract_high_res()

    def _resolve_thumbs() -> VideoFramesResponse:
        return _thumb_frames_response(
            video_id, request.timestamps, duration_s=record.duration_s,
        )

    if _uses_api_unit_billing(identity):
        return _bill_metered_call(
            user_id=user_id,
            api_key_id=_api_usage_key_id(identity),
            event_type="frames_thumb",
            units=API_UNIT_COST_FRAMES_THUMB,
            video_id=video_id,
            work=_resolve_thumbs,
        )
    return _resolve_thumbs()


def v1_billing_credits_summary(
    user_id: str = Depends(get_current_user_id),
) -> BillingSummaryResponse:
    return get_billing_summary(user_id=user_id)


def v1_billing_credits_checkout(
    request: BillingCheckoutRequest,
    user_id: str = Depends(get_current_user_id),
) -> BillingCheckoutResponse:
    return create_billing_checkout(request, user_id=user_id)


# ---------------------------------------------------------------------------
# API billing endpoints (v1)
# ---------------------------------------------------------------------------


def v1_api_billing_checkout(
    body: ApiCheckoutRequest,
    user_id: str = Depends(get_current_user_id),
) -> BillingCheckoutResponse:
    """Create checkout session for Developer Pack (JWT-only)."""
    _enforce_user_write_rate_limit(user_id)
    credits = DEFAULT_BILLING_PLAN_CREDITS["developer"]
    redirect_url = _frontend_url_for_path(body.return_path) if body.return_path else None
    try:
        session = create_checkout_session(
            user_id=user_id,
            plan="developer",
            credits=credits,
            variant_id=_billing_plan_variant_id("developer"),
            redirect_url=redirect_url,
            grant_target="api",
        )
    except LemonSqueezyConfigError as exc:
        logger.error("Lemon Squeezy checkout config error: %s", exc)
        raise HTTPException(status_code=503, detail="Billing checkout is not configured") from exc
    except LemonSqueezyProviderError as exc:
        logger.exception(
            "Lemon Squeezy checkout failed for user_id=%s plan=developer: %s",
            user_id,
            exc,
        )
        raise HTTPException(
            status_code=502,
            detail="Billing checkout is temporarily unavailable",
        ) from exc

    track("checkout_started", user_id=user_id, metadata={"plan": "developer"})
    return BillingCheckoutResponse(
        provider="lemonsqueezy",
        plan="developer",
        credits=credits,
        checkout_url=session.url,
        test_mode=session.test_mode,
    )


def v1_api_billing_summary(
    identity: AuthIdentity = Depends(get_current_user),
) -> ApiBillingSummaryResponse:
    """Return API unit balance and approximate equivalents."""
    record = db_get_api_credits(identity.user_id)
    balance = record.balance if record else 0
    cost_video = API_UNIT_COST_INDEX_VIDEO
    cost_query = API_UNIT_COST_TEXT_QUERY
    return ApiBillingSummaryResponse(
        api_units_balance=balance,
        unit_cost_index_video=cost_video,
        unit_cost_text_query=cost_query,
        approx_videos=balance // cost_video if cost_video > 0 else 0,
        approx_queries=balance // cost_query if cost_query > 0 else 0,
    )


def v1_api_billing_usage(
    identity: AuthIdentity = Depends(get_current_user),
    api_key_id: str | None = None,
    limit: int = 50,
) -> list[ApiUsageEventResponse]:
    """Return recent API usage events."""
    events = db_list_api_usage_events(
        user_id=identity.user_id,
        api_key_id=api_key_id,
        limit=min(limit, 200),
    )
    return [
        ApiUsageEventResponse(
            id=e.id,
            api_key_id=e.api_key_id,
            event_type=e.event_type,
            units=e.units,
            video_id=e.video_id,
            created_at=e.created_at,
        )
        for e in events
    ]


public_v1_router = APIRouter(prefix="/api/v1", tags=["v1"])
internal_v1_router = APIRouter(prefix="/api/v1", tags=["v1-internal"])

# Public developer billing.
public_v1_router.add_api_route(
    "/billing/units/checkout",
    v1_api_billing_checkout,
    methods=["POST"],
    response_model=BillingCheckoutResponse,
)
public_v1_router.add_api_route(
    "/billing/units/summary",
    v1_api_billing_summary,
    methods=["GET"],
    response_model=ApiBillingSummaryResponse,
)
public_v1_router.add_api_route(
    "/billing/units/usage",
    v1_api_billing_usage,
    methods=["GET"],
    response_model=list[ApiUsageEventResponse],
)

# Internal web billing.
internal_v1_router.add_api_route(
    "/billing/credits/summary",
    v1_billing_credits_summary,
    methods=["GET"],
    response_model=BillingSummaryResponse,
    include_in_schema=False,
)
internal_v1_router.add_api_route(
    "/billing/credits/checkout",
    v1_billing_credits_checkout,
    methods=["POST"],
    response_model=BillingCheckoutResponse,
    include_in_schema=False,
)

# Key management — static paths registered before parameterized /videos/{id}.
public_v1_router.add_api_route(
    "/keys",
    create_api_key,
    methods=["POST"],
    response_model=ApiKeyCreatedResponse,
    status_code=201,
)
public_v1_router.add_api_route(
    "/keys",
    list_api_keys,
    methods=["GET"],
    response_model=list[ApiKeyResponse],
)
public_v1_router.add_api_route(
    "/keys/{key_id}",
    revoke_api_key,
    methods=["DELETE"],
    status_code=204,
)

# Static paths first (Starlette matches by registration order).
public_v1_router.add_api_route(
    "/videos/upload",
    v1_upload_video,
    methods=["POST"],
    response_model=VideoResponse,
)
public_v1_router.add_api_route(
    "/videos/upload/init",
    v1_init_upload,
    methods=["POST"],
    response_model=UploadInitResponse,
)
public_v1_router.add_api_route(
    "/videos/upload/complete",
    v1_complete_upload,
    methods=["POST"],
    response_model=VideoResponse,
)

# Collection + parameterized paths.
public_v1_router.add_api_route(
    "/videos",
    v1_list_my_videos,
    methods=["GET"],
    response_model=list[VideoResponse],
)
internal_v1_router.add_api_route(
    "/videos",
    v1_create_video,
    methods=["POST"],
    response_model=VideoResponse,
    include_in_schema=False,
)
public_v1_router.add_api_route(
    "/videos/{video_id}",
    v1_get_video,
    methods=["GET"],
    response_model=VideoResponse,
)
public_v1_router.add_api_route(
    "/videos/{video_id}/search",
    v1_search_video,
    methods=["POST"],
    response_model=VideoSearchResponse,
)
internal_v1_router.add_api_route(
    "/videos/{video_id}/search/image",
    v1_search_video_by_image,
    methods=["POST"],
    response_model=VideoSearchResponse,
    include_in_schema=False,
)
public_v1_router.add_api_route(
    "/videos/{video_id}/transcript",
    v1_get_video_transcript,
    methods=["GET"],
    response_model=VideoTranscriptResponse,
)
public_v1_router.add_api_route(
    "/videos/{video_id}/frames",
    v1_get_video_frames,
    methods=["POST"],
    response_model=VideoFramesResponse,
)

app.include_router(public_v1_router)
app.include_router(internal_v1_router)
app.add_route("/mcp", build_mcp_asgi_app(), methods=["GET", "POST", "DELETE"], include_in_schema=False)
