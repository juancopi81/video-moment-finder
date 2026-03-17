from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from src.api.processing import NonRetriableVideoProcessingError, VideoValidationError
from src.db.supabase import VideoJobRecord, VideoRecord
from src.worker.runner import run_forever, run_once


def _job(
    video_id: str,
    *,
    job_id: str = "job_123",
    status: str = "processing",
    attempt_count: int = 1,
    worker_id: str | None = "worker:test",
) -> VideoJobRecord:
    now = datetime.now(timezone.utc).isoformat()
    return VideoJobRecord(
        id=job_id,
        video_id=video_id,
        status=status,  # type: ignore[arg-type]
        worker_id=worker_id,
        attempt_count=attempt_count,
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
        source_type="youtube",
        source_r2_key=None,
        source_filename=None,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture(autouse=True)
def _default_no_stale_jobs(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.worker.runner.list_stale_processing_video_jobs",
        lambda stale_before_iso, limit=25: [],
    )
    monkeypatch.setattr("src.worker.runner.track", lambda *args, **kwargs: None)


def test_run_once_returns_false_when_queue_empty(monkeypatch) -> None:
    metrics: list[dict[str, object]] = []
    monkeypatch.setattr("src.worker.runner.claim_next_video_job", lambda worker_id: None)
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test") is False
    assert metrics == []


def test_run_once_processes_successful_job(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123", attempt_count=1),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))
    monkeypatch.setattr("src.worker.runner.process_video", lambda video: object())
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
        or _job("video_123", job_id=job_id, status="completed", attempt_count=1),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test") is True
    assert status_updates == [
        ("video_123", "processing", None),
        ("video_123", "ready", None),
    ]
    assert completions == [("job_123", "completed", None)]
    assert [metric["event"] for metric in metrics] == ["job_attempt_started", "job_completed"]


def test_run_once_requeues_failed_job_below_retry_cap(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123", attempt_count=1),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))

    def _raise(video: VideoRecord) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.worker.runner.process_video", _raise)
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message))
        or _job("video_123", job_id=job_id, status="queued", attempt_count=1, worker_id=None),
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
        ),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3) is True
    assert status_updates == [
        ("video_123", "processing", None),
        ("video_123", "queued", None),
    ]
    assert requeues == [("job_123", "boom")]
    assert completions == []
    assert [metric["event"] for metric in metrics] == ["job_attempt_started", "job_requeued"]
    assert metrics[1]["reason"] == "attempt_failure"


def test_run_once_marks_terminal_failure_when_retry_cap_reached(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123", attempt_count=3),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))

    def _raise(video: VideoRecord) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("src.worker.runner.process_video", _raise)
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message)),
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
        or _job("video_123", job_id=job_id, status="failed", attempt_count=3),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3) is True
    assert status_updates[0] == ("video_123", "processing", None)
    assert status_updates[1] == ("video_123", "failed", "boom")
    assert requeues == []
    assert completions == [("job_123", "failed", "boom")]
    assert [metric["event"] for metric in metrics] == [
        "job_attempt_started",
        "job_terminal_failure",
    ]
    assert metrics[1]["reason"] == "max_attempts"


def test_run_once_does_not_retry_validation_failures(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123", attempt_count=1),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))

    def _raise(video: VideoRecord) -> None:
        raise VideoValidationError("Uploaded video exceeds 30-minute limit")

    monkeypatch.setattr("src.worker.runner.process_video", _raise)
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message)),
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
        or _job("video_123", job_id=job_id, status="failed", attempt_count=1),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3) is True
    assert status_updates == [
        ("video_123", "processing", None),
        ("video_123", "failed", "Uploaded video exceeds 30-minute limit"),
    ]
    assert completions == [("job_123", "failed", "Uploaded video exceeds 30-minute limit")]
    assert requeues == []
    assert [metric["event"] for metric in metrics] == [
        "job_attempt_started",
        "job_terminal_failure",
    ]
    assert metrics[1]["reason"] == "validation_failure"


def test_run_once_marks_non_retriable_processing_failure_terminal(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    capture_calls: list[tuple[BaseException, dict[str, object] | None]] = []
    metrics: list[dict[str, object]] = []

    now = datetime.now(timezone.utc).isoformat()
    video = VideoRecord(
        id="video_123",
        youtube_url="https://www.youtube.com/watch?v=abc123xyz45",
        status="queued",
        user_id="user_123",
        error_message=None,
        source_type="youtube",
        source_r2_key=None,
        source_filename=None,
        created_at=now,
        updated_at=now,
    )

    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_123", attempt_count=1),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: video)

    def _raise(video: VideoRecord) -> None:
        raise NonRetriableVideoProcessingError("Failed to upsert frames: larger than allowed")

    monkeypatch.setattr("src.worker.runner.process_video", _raise)
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message)),
    )
    monkeypatch.setattr(
        "src.worker.runner.capture_exception",
        lambda exc, *, context=None: capture_calls.append((exc, context)),
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
        or _job("video_123", job_id=job_id, status="failed", attempt_count=1),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3) is True
    assert status_updates == [
        ("video_123", "processing", None),
        ("video_123", "failed", "Failed to upsert frames: larger than allowed"),
    ]
    assert completions == [
        ("job_123", "failed", "Failed to upsert frames: larger than allowed")
    ]
    assert requeues == []
    assert len(capture_calls) == 1
    captured_exc, context = capture_calls[0]
    assert str(captured_exc) == "Failed to upsert frames: larger than allowed"
    assert context == {
        "job_id": "job_123",
        "video_id": "video_123",
        "user_id": "user_123",
        "attempt_count": 1,
        "source_type": "youtube",
    }
    assert [metric["event"] for metric in metrics] == [
        "job_attempt_started",
        "job_terminal_failure",
    ]
    assert metrics[1]["reason"] == "non_retriable_failure"


def test_run_once_recovers_stale_lock_and_processes_next_job(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.list_stale_processing_video_jobs",
        lambda stale_before_iso, limit=25: [
            _job("video_stale", job_id="job_stale", attempt_count=1)
        ],
    )
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message))
        or _job(
            "video_stale",
            job_id=job_id,
            status="queued",
            attempt_count=1,
            worker_id=None,
        ),
    )
    monkeypatch.setattr(
        "src.worker.runner.claim_next_video_job",
        lambda worker_id: _job("video_fresh", job_id="job_fresh", attempt_count=2),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))
    monkeypatch.setattr("src.worker.runner.process_video", lambda video: object())
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
        or _job("video_fresh", job_id=job_id, status="completed", attempt_count=2),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3, stale_lock_timeout_s=600.0) is True
    assert ("video_stale", "queued", None) in status_updates
    assert ("video_fresh", "processing", None) in status_updates
    assert ("video_fresh", "ready", None) in status_updates
    assert requeues == [("job_stale", "Recovered stale processing lock")]
    assert completions == [("job_fresh", "completed", None)]
    assert any(
        metric["event"] == "job_requeued" and metric.get("reason") == "stale_lock"
        for metric in metrics
    )


def test_run_once_marks_stale_job_terminal_after_retry_cap(monkeypatch) -> None:
    status_updates: list[tuple[str, str, str | None]] = []
    completions: list[tuple[str, str, str | None]] = []
    requeues: list[tuple[str, str | None]] = []
    metrics: list[dict[str, object]] = []

    monkeypatch.setattr(
        "src.worker.runner.list_stale_processing_video_jobs",
        lambda stale_before_iso, limit=25: [
            _job("video_stale", job_id="job_stale", attempt_count=3)
        ],
    )
    monkeypatch.setattr("src.worker.runner.claim_next_video_job", lambda worker_id: None)
    monkeypatch.setattr(
        "src.worker.runner.requeue_video_job",
        lambda job_id, error_message=None: requeues.append((job_id, error_message)),
    )
    monkeypatch.setattr("src.worker.runner.get_video", lambda video_id: _video(video_id))
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
        or _job("video_stale", job_id=job_id, status="failed", attempt_count=3),
    )
    monkeypatch.setattr(
        "src.worker.runner._log_worker_metric",
        lambda event, **fields: metrics.append({"event": event, **fields}),
    )

    assert run_once(worker_id="worker:test", max_attempts=3, stale_lock_timeout_s=600.0) is False
    assert requeues == []
    assert len(status_updates) == 1
    assert status_updates[0][0] == "video_stale"
    assert status_updates[0][1] == "failed"
    assert status_updates[0][2] is not None
    assert "max attempts" in status_updates[0][2]
    assert len(completions) == 1
    assert completions[0][0] == "job_stale"
    assert completions[0][1] == "failed"
    assert completions[0][2] is not None
    assert "max attempts" in completions[0][2]
    assert [metric["event"] for metric in metrics] == ["job_terminal_failure"]
    assert metrics[0]["reason"] == "stale_lock_max_attempts"


class _StopLoop(Exception):
    """Test-only exception to break infinite loop worker."""


def test_run_forever_retries_on_transient_transport_error(monkeypatch) -> None:
    run_once_calls: list[str] = []
    sleep_calls: list[float] = []
    reset_calls: list[str] = []

    responses: list[object] = [httpx.RemoteProtocolError("goaway"), False]

    def _run_once(**kwargs):
        run_once_calls.append("called")
        next_response = responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response

    def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 2:
            raise _StopLoop

    monkeypatch.setattr("src.worker.runner.run_once", _run_once)
    monkeypatch.setattr("src.worker.runner.reset_client", lambda: reset_calls.append("reset"))
    monkeypatch.setattr("src.worker.runner.time.sleep", _sleep)

    with pytest.raises(_StopLoop):
        run_forever(
            worker_id="worker:test",
            poll_interval_s=2.0,
            idle_backoff_max_s=8.0,
            db_retry_base_delay_s=1.0,
            db_retry_max_delay_s=8.0,
        )

    assert len(run_once_calls) == 2
    assert reset_calls == ["reset"]
    assert sleep_calls == [1.0, 2.0]


def test_run_forever_applies_idle_backoff_until_cap(monkeypatch) -> None:
    sleep_calls: list[float] = []

    monkeypatch.setattr("src.worker.runner.run_once", lambda **kwargs: False)

    def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 4:
            raise _StopLoop

    monkeypatch.setattr("src.worker.runner.time.sleep", _sleep)

    with pytest.raises(_StopLoop):
        run_forever(
            worker_id="worker:test",
            poll_interval_s=1.0,
            idle_backoff_max_s=3.0,
        )

    assert sleep_calls == [1.0, 2.0, 3.0, 3.0]


def test_run_forever_resets_idle_backoff_after_processed_job(monkeypatch) -> None:
    sleep_calls: list[float] = []
    responses: list[bool] = [False, False, True, False]

    def _run_once(**kwargs) -> bool:
        return responses.pop(0)

    def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)
        if len(sleep_calls) >= 3:
            raise _StopLoop

    monkeypatch.setattr("src.worker.runner.run_once", _run_once)
    monkeypatch.setattr("src.worker.runner.time.sleep", _sleep)

    with pytest.raises(_StopLoop):
        run_forever(
            worker_id="worker:test",
            poll_interval_s=1.0,
            idle_backoff_max_s=8.0,
        )

    assert sleep_calls == [1.0, 2.0, 1.0]


def test_run_forever_reraises_non_transient_exception(monkeypatch) -> None:
    reset_calls: list[str] = []

    def _raise(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("src.worker.runner.run_once", _raise)
    monkeypatch.setattr("src.worker.runner.reset_client", lambda: reset_calls.append("reset"))

    with pytest.raises(RuntimeError, match="boom"):
        run_forever(worker_id="worker:test", poll_interval_s=1.0)

    assert reset_calls == []
