"""FastAPI app with real Supabase, Modal, and Qdrant integrations."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.api.processing import process_video_task
from src.api.search import search_video as search_video_service
from src.config.env import load_env
from src.db.supabase import (
    VideoRecord,
    create_video as db_create_video,
    get_video as db_get_video,
)

load_env()

StatusType = Literal["processing", "ready", "failed"]


class VideoCreateRequest(BaseModel):
    youtube_url: HttpUrl


class VideoResponse(BaseModel):
    id: str
    youtube_url: HttpUrl
    status: StatusType
    created_at: datetime
    error_message: str | None = None


class VideoSearchRequest(BaseModel):
    query_text: str | None = None
    query_image_url: HttpUrl | None = None
    limit: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    timestamp_s: float
    thumbnail_url: HttpUrl
    score: float


class VideoSearchResponse(BaseModel):
    video_id: str
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
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/videos", response_model=VideoResponse)
def create_video(
    request: VideoCreateRequest, background_tasks: BackgroundTasks
) -> VideoResponse:
    """Create a new video and start background processing."""
    record = db_create_video(str(request.youtube_url))

    background_tasks.add_task(
        process_video_task,
        video_id=record.id,
        youtube_url=str(request.youtube_url),
    )

    return _video_record_to_response(record)


@app.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str) -> VideoResponse:
    """Get video status and details."""
    record = db_get_video(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")

    return _video_record_to_response(record)


@app.post("/videos/{video_id}/search", response_model=VideoSearchResponse)
def search_video(video_id: str, request: VideoSearchRequest) -> VideoSearchResponse:
    """Search for moments in a processed video."""
    record = db_get_video(video_id)
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

    results = search_video_service(
        video_id=video_id,
        query_text=request.query_text,
        limit=request.limit,
    )

    return VideoSearchResponse(
        video_id=video_id,
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
