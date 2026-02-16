"""FastAPI app with real Supabase, Modal, and Qdrant integrations."""
from __future__ import annotations

from datetime import datetime
import os
import re
from typing import Literal
from urllib.parse import parse_qs, urlparse

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, field_validator

from src.api.auth import get_current_user_id
from src.api.search import search_video as search_video_service
from src.config.env import load_env
from src.db.supabase import (
    VideoRecord,
    create_video as db_create_video,
    enqueue_video_job,
    get_video as db_get_video,
    update_video_status,
)
from src.storage.config import StorageConfigError
from src.storage.qdrant import QdrantStorageError
from src.utils.logging import get_logger

load_env()
logger = get_logger(__name__)

StatusType = Literal["queued", "processing", "ready", "failed"]
YOUTUBE_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _allowed_cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ALLOWED_ORIGINS", "").strip()
    if not raw:
        return ["http://localhost:3000"]
    origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
    return origins or ["http://localhost:3000"]


def _extract_youtube_video_id(youtube_url: str) -> str | None:
    parsed = urlparse(youtube_url.strip())
    host = parsed.netloc.lower()
    path_parts = [part for part in parsed.path.split("/") if part]
    video_id: str | None = None

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            query = parse_qs(parsed.query)
            video_id = query.get("v", [None])[0]
        elif path_parts and path_parts[0] in {"shorts", "live"}:
            video_id = path_parts[1] if len(path_parts) > 1 else None
    elif host == "youtu.be":
        video_id = path_parts[0] if path_parts else None

    if video_id is None:
        return None
    video_id = video_id.strip()
    if not YOUTUBE_VIDEO_ID_RE.fullmatch(video_id):
        return None
    return video_id


class VideoCreateRequest(BaseModel):
    youtube_url: str = Field(min_length=1, max_length=500)

    @field_validator("youtube_url")
    @classmethod
    def normalize_youtube_url(cls, value: str) -> str:
        video_id = _extract_youtube_video_id(value)
        if video_id is None:
            raise ValueError("youtube_url must be a valid YouTube video URL")
        return f"https://www.youtube.com/watch?v={video_id}"


class VideoResponse(BaseModel):
    id: str
    youtube_url: HttpUrl
    status: StatusType
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
    youtube_url: HttpUrl
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
        created_at=_parse_iso_datetime(record.created_at),
        error_message=record.error_message,
    )


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
