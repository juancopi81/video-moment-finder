"""Queue lock refresh and recovery compare-and-set regressions."""
from types import SimpleNamespace

import pytest

from src.db.supabase import complete_video_job, refresh_video_job_lock, requeue_video_job


def _job_row(status, *, attempt_count):
    return {
        "id": "job_123", "video_id": "video_123", "status": status,
        "worker_id": None, "attempt_count": attempt_count, "error_message": None,
        "locked_at": None, "started_at": None, "completed_at": None,
        "created_at": "2026-02-06T00:00:00Z", "updated_at": "2026-02-06T00:00:00Z",
    }


class _ConditionalUpdate:
    def __init__(self, row):
        self.row = row
        self.matches = True

    def table(self, name):
        assert name == "video_jobs"
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, name, value):
        self.matches = self.matches and self.row.get(name) == value
        return self

    def execute(self):
        if self.matches:
            self.row.update(self.payload)
        return SimpleNamespace(data=[self.row] if self.matches else [])


@pytest.mark.parametrize("changed", [None, "worker_id", "attempt_count", "status"])
def test_refresh_lock_only_updates_the_owned_attempt(monkeypatch, changed):
    row = _job_row("processing", attempt_count=2)
    row.update(worker_id="worker:test", locked_at="old")
    if changed:
        row[changed] = {"worker_id": "other", "attempt_count": 3, "status": "queued"}[changed]
    monkeypatch.setattr("src.db.supabase.get_client", lambda: _ConditionalUpdate(row))
    assert refresh_video_job_lock("job_123", "worker:test", 2) is (changed is None)
    assert (row["locked_at"] != "old") is (changed is None)


@pytest.mark.parametrize("operation", [requeue_video_job, complete_video_job])
def test_stale_transition_requires_unchanged_lock(monkeypatch, operation):
    row = _job_row("processing", attempt_count=2)
    row["locked_at"] = "refreshed"
    monkeypatch.setattr("src.db.supabase.get_client", lambda: _ConditionalUpdate(row))
    args = ("job_123", "failed") if operation is complete_video_job else ("job_123",)
    assert operation(*args, expected_locked_at="old") is None
    assert row["status"] == "processing"
    assert operation(*args, expected_locked_at="refreshed") is not None
