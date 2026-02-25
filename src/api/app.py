"""FastAPI app with real Supabase, Modal, and Qdrant integrations."""
from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Literal
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.api.auth import get_current_user_id
from src.api.search import search_video as search_video_service
from src.config.env import load_env
from src.db.supabase import (
    VideoRecord,
    count_videos_for_user as db_count_videos_for_user,
    create_video as db_create_video,
    create_uploaded_video as db_create_uploaded_video,
    enqueue_video_job,
    get_video as db_get_video,
    list_videos as db_list_videos,
    update_video_status,
)
from src.storage.config import R2Config, StorageConfigError
from src.storage.r2 import R2Store, R2StorageError, source_key
from src.storage.qdrant import QdrantStorageError
from src.utils.logging import get_logger
from src.video.download import VideoMetadataError, fetch_video_metadata
from src.video.youtube import normalize_youtube_url
from uuid import UUID, uuid4

load_env()
logger = get_logger(__name__)

StatusType = Literal["queued", "processing", "ready", "failed"]
DEFAULT_MAX_VIDEO_DURATION_S = 30 * 60
DEFAULT_MAX_FREE_VIDEOS = 1
DEFAULT_CORS_ORIGINS = ["http://localhost:3000"]


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


def _max_video_duration_s() -> int:
    raw = os.environ.get("VIDEO_MAX_DURATION_S", "").strip()
    if not raw:
        return DEFAULT_MAX_VIDEO_DURATION_S
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid VIDEO_MAX_DURATION_S=%r; using default", raw)
        return DEFAULT_MAX_VIDEO_DURATION_S
    if value <= 0:
        logger.warning("VIDEO_MAX_DURATION_S must be positive; using default")
        return DEFAULT_MAX_VIDEO_DURATION_S
    return value


def _max_free_videos() -> int:
    raw = os.environ.get("VIDEO_MAX_FREE_VIDEOS", "").strip()
    if not raw:
        return DEFAULT_MAX_FREE_VIDEOS
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid VIDEO_MAX_FREE_VIDEOS=%r; using default", raw)
        return DEFAULT_MAX_FREE_VIDEOS
    if value <= 0:
        logger.warning("VIDEO_MAX_FREE_VIDEOS must be positive; using default")
        return DEFAULT_MAX_FREE_VIDEOS
    return value


def _validate_video_duration(youtube_url: str) -> None:
    try:
        metadata = fetch_video_metadata(youtube_url)
    except VideoMetadataError as exc:
        logger.warning("Failed to fetch YouTube metadata for %s: %s", youtube_url, exc)
        raise HTTPException(
            status_code=400,
            detail="Unable to fetch YouTube metadata for this URL",
        ) from exc

    if metadata.is_live:
        raise HTTPException(status_code=400, detail="Live streams are not supported")
    if metadata.duration_s is None:
        raise HTTPException(status_code=400, detail="Unable to determine video duration")

    max_duration_s = _max_video_duration_s()
    if metadata.duration_s > max_duration_s:
        max_minutes = max_duration_s // 60
        raise HTTPException(
            status_code=400,
            detail=f"Video exceeds {max_minutes}-minute limit",
        )


def _enforce_free_video_limit(user_id: str) -> None:
    max_videos = _max_free_videos()
    current_count = db_count_videos_for_user(user_id)
    if current_count >= max_videos:
        raise HTTPException(
            status_code=403,
            detail="Free video limit reached",
        )


def _source_url_ttl_s() -> int:
    raw = os.environ.get("VIDEO_SOURCE_URL_TTL_S", "").strip()
    if not raw:
        return 3600
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid VIDEO_SOURCE_URL_TTL_S=%r; using default", raw)
        return 3600
    if value <= 0:
        logger.warning("VIDEO_SOURCE_URL_TTL_S must be positive; using default")
        return 3600
    return value


def _upload_url_ttl_s() -> int:
    raw = os.environ.get("VIDEO_UPLOAD_URL_TTL_S", "").strip()
    if not raw:
        return 900
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid VIDEO_UPLOAD_URL_TTL_S=%r; using default", raw)
        return 900
    if value <= 0:
        logger.warning("VIDEO_UPLOAD_URL_TTL_S must be positive; using default")
        return 900
    return value


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
    query_image_url: HttpUrl | None = None
    limit: int = Field(default=5, ge=1, le=20)

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


class VideoSearchResponse(BaseModel):
    video_id: str
    youtube_url: HttpUrl | None
    source_url: HttpUrl | None = None
    status: StatusType
    results: list[SearchResult]


def _parse_iso_datetime(iso_string: str | None) -> datetime:
    """Parse ISO 8601 datetime string from Supabase."""
    if not iso_string:
        return datetime.now()
    return datetime.fromisoformat(iso_string.replace("Z", "+00:00"))


def _video_record_to_response(record: VideoRecord) -> VideoResponse:
    """Convert VideoRecord to VideoResponse."""
    return VideoResponse(
        id=record.id,
        youtube_url=record.youtube_url,
        status=record.status,
        source_type=record.source_type,
        source_filename=record.source_filename,
        source_url=_source_url_for_record(record),
        created_at=_parse_iso_datetime(record.created_at),
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
    if existing.source_type != "upload":
        raise HTTPException(
            status_code=409,
            detail="video_id already exists with a different source type",
        )
    if existing.source_r2_key != source_r2_key or existing.source_filename != source_filename:
        raise HTTPException(
            status_code=409,
            detail="video_id already exists with different upload metadata",
        )
    return existing


app = FastAPI(
    title="Video Moment Finder API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_origin_regex=_allowed_cors_origin_regex(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/videos", response_model=VideoResponse)
def create_video(
    request: VideoCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Create a new video and enqueue durable processing job."""
    _enforce_free_video_limit(user_id)
    _validate_video_duration(request.youtube_url)
    record = db_create_video(request.youtube_url, user_id=user_id, status="queued")
    try:
        enqueue_video_job(record.id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue processing job for video_id=%s: %s",
            record.id,
            exc,
        )
        update_video_status(
            record.id,
            "failed",
            error_message=f"Failed to enqueue processing job: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue processing job",
        ) from exc

    return _video_record_to_response(record)


@app.post("/videos/upload", response_model=VideoResponse)
def upload_video(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Upload a video file and enqueue durable processing job."""
    if not file.filename and not file.content_type:
        raise HTTPException(status_code=400, detail="No file uploaded")
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")
    _enforce_free_video_limit(user_id)

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload storage is not configured",
        ) from exc

    video_id = str(uuid4())
    filename = _sanitize_filename(file.filename)
    store = R2Store(r2_config)

    try:
        upload_result = store.upload_source_video(
            video_id=video_id,
            filename=filename,
            file_obj=file.file,
            content_type=file.content_type,
        )
    except R2StorageError as exc:
        logger.exception("Failed to upload source video: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to store uploaded video",
        ) from exc

    record = db_create_uploaded_video(
        video_id=video_id,
        source_r2_key=upload_result.key,
        source_filename=filename,
        user_id=user_id,
        status="queued",
    )
    try:
        enqueue_video_job(record.id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue processing job for video_id=%s: %s",
            record.id,
            exc,
        )
        update_video_status(
            record.id,
            "failed",
            error_message=f"Failed to enqueue processing job: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue processing job",
        ) from exc

    return _video_record_to_response(record)


@app.post("/videos/upload/init", response_model=UploadInitResponse)
def init_upload(
    request: UploadInitRequest,
    user_id: str = Depends(get_current_user_id),
) -> UploadInitResponse:
    """Generate a presigned upload URL for direct-to-R2 uploads."""
    if request.content_type and not request.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Only video uploads are supported")
    _enforce_free_video_limit(user_id)

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


@app.post("/videos/upload/complete", response_model=VideoResponse)
def complete_upload(
    request: UploadCompleteRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Finalize a presigned upload and enqueue processing."""
    filename = _sanitize_filename(request.filename)
    key = source_key(request.video_id, filename)

    existing = _get_idempotent_upload_record(
        video_id=request.video_id,
        user_id=user_id,
        source_r2_key=key,
        source_filename=filename,
    )
    if existing is not None:
        return _video_record_to_response(existing)
    _enforce_free_video_limit(user_id)

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

    try:
        record = db_create_uploaded_video(
            video_id=request.video_id,
            source_r2_key=key,
            source_filename=filename,
            user_id=user_id,
            status="queued",
        )
    except Exception:
        # Handle the create race where another request inserted this upload first.
        existing = _get_idempotent_upload_record(
            video_id=request.video_id,
            user_id=user_id,
            source_r2_key=key,
            source_filename=filename,
        )
        if existing is not None:
            return _video_record_to_response(existing)
        raise

    try:
        enqueue_video_job(record.id)
    except Exception as exc:
        logger.exception(
            "Failed to enqueue processing job for video_id=%s: %s",
            record.id,
            exc,
        )
        update_video_status(
            record.id,
            "failed",
            error_message=f"Failed to enqueue processing job: {exc}",
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to enqueue processing job",
        ) from exc

    return _video_record_to_response(record)


@app.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(
    video_id: str,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Get video status and details."""
    record = db_get_video(video_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")

    return _video_record_to_response(record)


@app.get("/users/me/videos", response_model=list[VideoResponse])
def list_my_videos(
    user_id: str = Depends(get_current_user_id),
) -> list[VideoResponse]:
    """List videos for the authenticated user."""
    records = db_list_videos(user_id=user_id)
    return [_video_record_to_response(record) for record in records]


@app.post("/videos/{video_id}/search", response_model=VideoSearchResponse)
def search_video(
    video_id: str,
    request: VideoSearchRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoSearchResponse:
    """Search for moments in a processed video."""
    record = db_get_video(video_id, user_id=user_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")

    if request.query_image_url:
        raise HTTPException(status_code=501, detail="Image queries not yet supported")

    if not request.query_text:
        raise HTTPException(status_code=400, detail="Provide query_text or query_image_url")

    if record.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Video not ready for search (status: {record.status})",
        )

    try:
        results = search_video_service(
            video_id=video_id,
            query_text=request.query_text,
            limit=request.limit,
        )
    except (QdrantStorageError, StorageConfigError, RuntimeError) as exc:
        logger.exception("Search backend failure for video_id=%s: %s", video_id, exc)
        raise HTTPException(
            status_code=503,
            detail="Search is temporarily unavailable. Please try again.",
        ) from exc

    return VideoSearchResponse(
        video_id=video_id,
        youtube_url=record.youtube_url,
        source_url=_source_url_for_record(record),
        status=record.status,
        results=[
            SearchResult(
                timestamp_s=r.timestamp_s,
                thumbnail_url=r.thumbnail_url,
                score=r.score,
            )
            for r in results
        ],
    )
