"""FastAPI app with real Supabase, Modal, and Qdrant integrations."""
from __future__ import annotations

from datetime import datetime
import os
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.api.auth import get_current_user_id
from src.api.search import search_video as search_video_service
from src.config.env import load_env
from src.db.supabase import (
    VideoRecord,
    create_video as db_create_video,
    create_uploaded_video as db_create_uploaded_video,
    enqueue_video_job,
    get_video as db_get_video,
    update_video_status,
)
from src.storage.config import R2Config, StorageConfigError
from src.storage.r2 import R2Store, R2StorageError
from src.storage.qdrant import QdrantStorageError
from src.utils.logging import get_logger
from src.video.download import VideoMetadataError, fetch_video_metadata
from src.video.youtube import normalize_youtube_url
from uuid import uuid4

load_env()
logger = get_logger(__name__)

StatusType = Literal["queued", "processing", "ready", "failed"]
DEFAULT_MAX_VIDEO_DURATION_S = 30 * 60


def _allowed_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:3000"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]


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


app = FastAPI(
    title="Video Moment Finder API",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/videos", response_model=VideoResponse)
def create_video(
    request: VideoCreateRequest,
    user_id: str = Depends(get_current_user_id),
) -> VideoResponse:
    """Create a new video and enqueue durable processing job."""
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
