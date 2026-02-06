from __future__ import annotations

from datetime import datetime, timezone

from src.db.supabase import VideoJobRecord, VideoRecord
from src.worker.runner import run_once


def _job(video_id: str, *, status: str = "processing") -> VideoJobRecord:
    now = datetime.now(timezone.utc).isoformat()
    return VideoJobRecord(
        id="job_123",
        video_id=video_id,
        status=status,  # type: ignore[arg-type]
        worker_id="worker:test",
        attempt_count=1,
        error_message=None,
        locked_at=now,
        started_at=now,
        completed_at=None,
        created_at=now,
        updated_at=now,
    )


def _video(video_id: str, *, status: str = "queued") -> VideoRecord:
    now = datetime.now(timezone.utc).isoformat()
    return VideoRecord(
        id=video_id,
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status=status,  # type: ignore[arg-type]
        user_id=None,
        error_message=None,
        created_at=now,
        updated_at=now,
    )


def test_run_once_returns_false_when_queue_empty(monkeypatch) -> None:
    monkeypatch.setattr("src.worker.runner.claim_next_video_job", lambda worker_id: None)

    assert run_once(worker_id="worker:test") is False


def test_run_once_processes_successful_job(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123"),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))
    monkeypatch.setattr(
        "src.worker.runner.process_video",
        lambda video_id, youtube_url: object(),
    )
    monkeypatch.setattr(
        "src.worker.runner.update_video_status",
        lambda video_id, status, error_message=None: status_updates.append(
            (video_id, status, error_message)
        )
        or _video(video_id, status=status),
    )
    monkeypatch.setattr(
        "src.worker.runner.complete_video_job",
        lambda job_id, status, error_message=None: completions.append(
            (job_id, status, error_message)
        )
        or _job("video_123", status="completed"),
    )

    assert run_once(worker_id="worker:test") is True
    assert status_updates == [
        ("video_123", "processing", None),
        ("video_123", "ready", None),
    ]
    assert completions == [("job_123", "completed", None)]


def test_run_once_marks_failed_job(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123"),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))

    def _raise(video_id: str, youtube_url: str) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.worker.runner.process_video", _raise)
    monkeypatch.setattr(
        "src.worker.runner.update_video_status",
        lambda video_id, status, error_message=None: status_updates.append(
            (video_id, status, error_message)
        )
        or _video(video_id, status=status),
    )
    monkeypatch.setattr(
        "src.worker.runner.complete_video_job",
        lambda job_id, status, error_message=None: completions.append(
            (job_id, status, error_message)
        )
        or _job("video_123", status="failed"),
    )

    assert run_once(worker_id="worker:test") is True
    assert status_updates[0] == ("video_123", "processing", None)
    assert status_updates[1][0] == "video_123"
    assert status_updates[1][1] == "failed"
    assert status_updates[1][2] == "boom"
    assert completions == [("job_123", "failed", "boom")]
