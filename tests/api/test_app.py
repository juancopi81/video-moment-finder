from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.app import app
from src.db.supabase import VideoRecord
from src.storage.qdrant import SearchResult


def _video_record(video_id: str, *, status: str = "queued") -> VideoRecord:
    return VideoRecord(
        id=video_id,
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status=status,  # type: ignore[arg-type]
        user_id=None,
        error_message=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


def test_create_video_enqueues_job(monkeypatch) -> None:
    client = TestClient(app)
    enqueue_calls: list[str] = []

    def fake_create_video(youtube_url: str, user_id=None, status="queued") -> VideoRecord:
        assert status == "queued"
        return _video_record("video_123", status=status)

    def fake_enqueue(video_id: str) -> object:
        enqueue_calls.append(video_id)
        return object()

    monkeypatch.setattr("src.api.app.db_create_video", fake_create_video)
    monkeypatch.setattr("src.api.app.enqueue_video_job", fake_enqueue)

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "video_123"
    assert payload["status"] == "queued"
    assert enqueue_calls == ["video_123"]


def test_create_video_returns_500_when_enqueue_fails(monkeypatch) -> None:
    client = TestClient(app)
    failure_updates: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        "src.api.app.db_create_video",
        lambda youtube_url, user_id=None, status="queued": _video_record(
            "video_500", status=status
        ),
    )

    def fake_update(video_id: str, status: str, error_message: str | None = None):
        failure_updates.append((video_id, status, error_message))
        return _video_record(video_id, status="failed")

    monkeypatch.setattr(
        "src.api.app.enqueue_video_job",
        lambda video_id: (_ for _ in ()).throw(RuntimeError("queue down")),
    )
    monkeypatch.setattr("src.api.app.update_video_status", fake_update)

    response = client.post(
        "/videos",
        json={"youtube_url": "https://www.youtube.com/watch?v=abc123xyz45"},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to enqueue processing job"
    assert failure_updates
    assert failure_updates[0][0] == "video_500"
    assert failure_updates[0][1] == "failed"


def test_get_video_returns_status(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id: _video_record(video_id, status="processing"),
    )

    response = client.get("/videos/video_status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"] == "video_status"
    assert payload["status"] == "processing"


def test_search_video_accepts_nullable_thumbnail_url(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_service",
        lambda video_id, query_text, limit=5: [
            SearchResult(
                video_id=video_id,
                frame_index=0,
                timestamp_s=12.5,
                thumbnail_url=None,
                score=0.92,
            )
        ],
    )

    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "robot in blue hoodie"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["results"][0]["thumbnail_url"] is None
    assert payload["results"][0]["timestamp_s"] == 12.5


def test_search_video_returns_503_when_backend_fails(monkeypatch) -> None:
    client = TestClient(app)
    monkeypatch.setattr(
        "src.api.app.db_get_video",
        lambda video_id: _video_record(video_id, status="ready"),
    )
    monkeypatch.setattr(
        "src.api.app.search_video_service",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )

    response = client.post(
        "/videos/video_ready/search",
        json={"query_text": "an elevator"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Search is temporarily unavailable. Please try again."
