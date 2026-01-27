"""FastAPI app with mock endpoints for Phase 2.2."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from src.config.env import load_env

load_env()

StatusType = Literal["processing", "ready", "failed"]
MOCK_PROCESSING_SECONDS = 3


@dataclass
class VideoRecord:
    video_id: str
    youtube_url: str
    status: StatusType
    created_at: datetime


_VIDEO_STORE: dict[str, VideoRecord] = {}


class VideoCreateRequest(BaseModel):
    youtube_url: HttpUrl


class VideoResponse(BaseModel):
    id: str
    youtube_url: HttpUrl
    status: StatusType
    created_at: datetime


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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_mock_status(record: VideoRecord) -> None:
    elapsed = (_now() - record.created_at).total_seconds()
    if record.status == "processing" and elapsed >= MOCK_PROCESSING_SECONDS:
        record.status = "ready"


def _to_video_response(record: VideoRecord) -> VideoResponse:
    return VideoResponse(
        id=record.video_id,
        youtube_url=record.youtube_url,
        status=record.status,
        created_at=record.created_at,
    )


def _mock_results(video_id: str, limit: int) -> list[SearchResult]:
    base = abs(hash(video_id)) % 100
    results: list[SearchResult] = []
    for idx in range(limit):
        timestamp_s = float(base + idx * 7)
        results.append(
            SearchResult(
                timestamp_s=timestamp_s,
                thumbnail_url=f"https://example.com/thumbs/{video_id}/{idx}.jpg",
                score=max(0.1, 0.95 - idx * 0.05),
            )
        )
    return results


app = FastAPI(
    title="Video Moment Finder API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/videos", response_model=VideoResponse)
def create_video(request: VideoCreateRequest) -> VideoResponse:
    video_id = uuid4().hex
    record = VideoRecord(
        video_id=video_id,
        youtube_url=str(request.youtube_url),
        status="processing",
        created_at=_now(),
    )
    _VIDEO_STORE[video_id] = record
    return _to_video_response(record)


@app.get("/videos/{video_id}", response_model=VideoResponse)
def get_video(video_id: str) -> VideoResponse:
    record = _VIDEO_STORE.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    _ensure_mock_status(record)
    return _to_video_response(record)


@app.post("/videos/{video_id}/search", response_model=VideoSearchResponse)
def search_video(video_id: str, request: VideoSearchRequest) -> VideoSearchResponse:
    record = _VIDEO_STORE.get(video_id)
    if not record:
        raise HTTPException(status_code=404, detail="Video not found")
    if not request.query_text and not request.query_image_url:
        raise HTTPException(
            status_code=400, detail="Provide query_text or query_image_url"
        )
    _ensure_mock_status(record)
    results = _mock_results(video_id, request.limit)
    return VideoSearchResponse(video_id=video_id, status=record.status, results=results)
