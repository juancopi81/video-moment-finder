"""Tests for GET /api/v1/videos/{id}/transcript and POST /api/v1/videos/{id}/frames."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.frames import FRAME_SAMPLE_FPS, ExtractedFrame
from src.db.supabase import ApiUnitConsumeResult, TranscriptSegmentRecord
from src.storage.config import StorageConfigError
from src.video.metadata import max_video_duration_s

MAX_THUMB_FRAME_INDEX = int(max_video_duration_s() * FRAME_SAMPLE_FPS) - 1

from tests.api.conftest import (
    _authenticate,
    _setup_api_key_auth,
    _upload_video_record,
    _video_record,
)

client = TestClient(app)


def _segment(video_id: str, idx: int, start: float, end: float, text: str) -> TranscriptSegmentRecord:
    return TranscriptSegmentRecord(
        video_id=video_id,
        segment_index=idx,
        start_s=start,
        end_s=end,
        text=text,
        language_code="en",
    )


class FakeThumbR2Store:
    """Fake R2Store for thumb-mode presigned URL generation."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return f"https://signed.example.com/{key}"


class FakeHighResR2Store:
    """Fake R2Store for high-res retained-source checks."""

    def __init__(self, *, exists: bool = True) -> None:
        self._exists = exists

    def __call__(self, *_args, **_kwargs):
        return self

    def source_exists(self, key: str) -> bool:
        return self._exists

    def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        return "https://source.example.com/presigned"


# ---------------------------------------------------------------------------
# GET /api/v1/videos/{video_id}/transcript
# ---------------------------------------------------------------------------


class TestVideoTranscript:
    def test_happy_path_returns_segments(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_get_video_transcript_segments",
            lambda vid, start_s=None, end_s=None: [
                _segment(vid, 0, 0.0, 2.0, "Hello"),
                _segment(vid, 1, 2.0, 4.5, "World"),
            ],
        )

        response = client.get("/api/v1/videos/vid_1/transcript")

        assert response.status_code == 200
        payload = response.json()
        assert payload["video_id"] == "vid_1"
        assert payload["has_transcript"] is True
        assert payload["language_code"] == "en"
        assert payload["segment_count"] == 2
        assert payload["segments"][0] == {
            "segment_index": 0,
            "start_s": 0.0,
            "end_s": 2.0,
            "text": "Hello",
        }

    def test_no_transcript_returns_200_not_error(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_get_video_transcript_segments",
            lambda vid, start_s=None, end_s=None: [],
        )

        response = client.get("/api/v1/videos/vid_1/transcript")

        assert response.status_code == 200
        payload = response.json()
        assert payload["has_transcript"] is False
        assert payload["segment_count"] == 0
        assert payload["segments"] == []
        assert payload["language_code"] is None

    def test_range_filtering_passes_start_and_end_through(self, monkeypatch) -> None:
        _authenticate("user_123")
        captured: dict = {}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        def fake_segments(vid, start_s=None, end_s=None):
            captured["start_s"] = start_s
            captured["end_s"] = end_s
            return []

        monkeypatch.setattr("src.api.app.db_get_video_transcript_segments", fake_segments)

        response = client.get(
            "/api/v1/videos/vid_1/transcript",
            params={"start_s": 10, "end_s": 20},
        )

        assert response.status_code == 200
        assert captured == {"start_s": 10.0, "end_s": 20.0}

    def test_ownership_404_for_missing_video(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr("src.api.app.db_get_video", lambda vid, user_id=None: None)

        response = client.get("/api/v1/videos/missing/transcript")

        assert response.status_code == 404

    def test_jwt_caller_is_not_billed(self, monkeypatch) -> None:
        _authenticate("user_123")
        consumed = {"called": False}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_get_video_transcript_segments",
            lambda vid, start_s=None, end_s=None: [],
        )

        def mock_consume(**kwargs):
            consumed["called"] = True
            return ApiUnitConsumeResult(allowed=True, remaining_balance=0)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.get("/api/v1/videos/vid_1/transcript")

        assert response.status_code == 200
        assert consumed["called"] is False

    def test_api_key_caller_is_billed_transcript_fetch(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed: dict = {}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_get_video_transcript_segments",
            lambda vid, start_s=None, end_s=None: [],
        )

        def mock_consume(**kwargs):
            consumed.update(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.get(
            "/api/v1/videos/vid_1/transcript",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert consumed["event_type"] == "transcript_fetch"
        assert consumed["units"] == 1

    def test_insufficient_units_returns_402_without_fetching_segments(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        fetch_called = {"called": False}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        def fake_segments(*_a, **_kw):
            fetch_called["called"] = True
            return []

        monkeypatch.setattr("src.api.app.db_get_video_transcript_segments", fake_segments)
        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
        )

        response = client.get(
            "/api/v1/videos/vid_1/transcript",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 402
        assert fetch_called["called"] is False

    def test_not_ready_video_returns_400_without_billing(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed = {"called": False}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="processing"),
        )

        def mock_consume(**kwargs):
            consumed["called"] = True
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.get(
            "/api/v1/videos/vid_1/transcript",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 400
        assert "not ready" in response.json()["detail"].lower()
        assert consumed["called"] is False

    def test_failed_video_returns_400_without_billing(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="failed"),
        )

        response = client.get("/api/v1/videos/vid_1/transcript")

        assert response.status_code == 400

    def test_db_read_failure_after_billing_compensates_units(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed: list[dict] = []
        compensated: list[dict] = []
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        def mock_consume(**kwargs):
            consumed.append(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        def mock_compensate(**kwargs):
            compensated.append(kwargs)

        def fake_segments(*_a, **_kw):
            raise RuntimeError("db unavailable")

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr("src.api.app.db_compensate_api_units", mock_compensate)
        monkeypatch.setattr("src.api.app.db_get_video_transcript_segments", fake_segments)

        response = client.get(
            "/api/v1/videos/vid_1/transcript",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 503
        assert len(consumed) == 1
        assert len(compensated) == 1
        assert consumed[0]["request_id"] == compensated[0]["request_id"]
        assert compensated[0]["units"] == 1
        assert compensated[0]["video_id"] == "vid_1"
        assert compensated[0]["metadata"]["event_type"] == "transcript_fetch_failed"

    def test_rate_limit_enforced_same_as_search(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
        monkeypatch.setenv("RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW", "1")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_get_video_transcript_segments",
            lambda vid, start_s=None, end_s=None: [],
        )

        first = client.get("/api/v1/videos/vid_1/transcript")
        second = client.get("/api/v1/videos/vid_1/transcript")

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"] == "Rate limit exceeded. Please retry later."


# ---------------------------------------------------------------------------
# POST /api/v1/videos/{video_id}/frames -- validation
# ---------------------------------------------------------------------------


class TestVideoFramesValidation:
    def test_empty_timestamps_400(self) -> None:
        _authenticate("user_123")

        response = client.post("/api/v1/videos/vid_1/frames", json={"timestamps": []})

        assert response.status_code == 400

    def test_negative_timestamp_400(self) -> None:
        _authenticate("user_123")

        response = client.post("/api/v1/videos/vid_1/frames", json={"timestamps": [-1.0]})

        assert response.status_code == 400

    def test_thumb_cap_exceeded_400(self) -> None:
        _authenticate("user_123")

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [float(i) for i in range(26)], "resolution": "thumb"},
        )

        assert response.status_code == 400

    def test_high_cap_exceeded_400(self) -> None:
        _authenticate("user_123")

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [float(i) for i in range(9)], "resolution": "high"},
        )

        assert response.status_code == 400

    def test_ownership_404_for_missing_video(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr("src.api.app.db_get_video", lambda vid, user_id=None: None)

        response = client.post(
            "/api/v1/videos/missing/frames", json={"timestamps": [1.0]},
        )

        assert response.status_code == 404

    def test_infinite_timestamp_400(self) -> None:
        _authenticate("user_123")

        # `1e1000` is valid JSON syntax but overflows to `inf` once parsed.
        # httpx's own JSON encoder rejects out-of-range floats client-side
        # (matching strict JSON), so send the raw wire bytes directly to
        # exercise the server's (more permissive) JSON parsing of this exact
        # payload shape.
        response = client.post(
            "/api/v1/videos/vid_1/frames",
            content=b'{"timestamps": [1e1000]}',
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 400

    def test_not_ready_video_returns_400_without_billing(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed = {"called": False}
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="queued"),
        )

        def mock_consume(**kwargs):
            consumed["called"] = True
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 400
        assert consumed["called"] is False

    def test_rate_limit_enforced_same_as_search(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setenv("RATE_LIMIT_WINDOW_S", "60")
        monkeypatch.setenv("RATE_LIMIT_SEARCH_REQUESTS_PER_WINDOW", "1")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeThumbR2Store)

        first = client.post("/api/v1/videos/vid_1/frames", json={"timestamps": [1.0]})
        second = client.post("/api/v1/videos/vid_1/frames", json={"timestamps": [1.0]})

        assert first.status_code == 200
        assert second.status_code == 429
        assert second.json()["detail"] == "Rate limit exceeded. Please retry later."


# ---------------------------------------------------------------------------
# POST /api/v1/videos/{video_id}/frames -- resolution=thumb
# ---------------------------------------------------------------------------


class TestVideoFramesThumb:
    def test_happy_path_returns_presigned_urls_in_request_order(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeThumbR2Store)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.4, 9.6], "resolution": "thumb"},
        )

        assert response.status_code == 200
        frames = response.json()["frames"]
        assert len(frames) == 2
        assert frames[0]["requested_timestamp_s"] == 1.4
        assert frames[0]["actual_timestamp_s"] == 1.0
        assert frames[0]["resolution"] == "thumb"
        assert frames[0]["url"] == "https://signed.example.com/thumb/vid_1/thumb_00001.jpg"
        assert frames[1]["requested_timestamp_s"] == 9.6
        assert frames[1]["actual_timestamp_s"] == 10.0
        assert frames[1]["url"] == "https://signed.example.com/thumb/vid_1/thumb_00010.jpg"

    def test_dedupe_reuses_url_but_preserves_request_order(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())

        calls: list[str] = []

        class CountingFakeR2Store(FakeThumbR2Store):
            def generate_presigned_url(self, key: str, expires_in: int = 3600) -> str:
                calls.append(key)
                return super().generate_presigned_url(key, expires_in=expires_in)

        monkeypatch.setattr("src.api.app.R2Store", CountingFakeR2Store)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [2.4, 2.3, 50.0], "resolution": "thumb"},
        )

        assert response.status_code == 200
        frames = response.json()["frames"]
        assert len(frames) == 3
        # 2.4 and 2.3 both round to frame_index 2 -> same URL, generated once.
        assert frames[0]["url"] == frames[1]["url"]
        assert calls.count("thumb/vid_1/thumb_00002.jpg") == 1
        assert frames[2]["actual_timestamp_s"] == 50.0

    def test_frame_index_clamped_to_max_frames(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeThumbR2Store)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [999999.0], "resolution": "thumb"},
        )

        assert response.status_code == 200
        frame = response.json()["frames"][0]
        assert frame["actual_timestamp_s"] == float(MAX_THUMB_FRAME_INDEX)
        assert frame["url"] == (
            f"https://signed.example.com/thumb/vid_1/thumb_{MAX_THUMB_FRAME_INDEX:05d}.jpg"
        )

    def test_insufficient_units_402_without_generating_url(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=False, remaining_balance=0),
        )

        presign_called = {"called": False}

        class FailingR2Store:
            def __init__(self, *_a, **_kw) -> None:
                pass

            def generate_presigned_url(self, *_a, **_kw):
                presign_called["called"] = True
                raise AssertionError("should not be called when units are insufficient")

        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FailingR2Store)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "thumb"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 402
        assert presign_called["called"] is False

    def test_api_key_bills_frames_thumb_units(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeThumbR2Store)

        consumed: dict = {}

        def mock_consume(**kwargs):
            consumed.update(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0]},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert consumed["event_type"] == "frames_thumb"
        assert consumed["units"] == 1

    def test_clamps_to_video_duration_s_when_present(self, monkeypatch) -> None:
        # A 10s video's own duration should clamp tighter than the global
        # VIDEO_MAX_DURATION_S-derived ceiling used when duration_s is null.
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready", duration_s=10.0),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeThumbR2Store)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [999999.0], "resolution": "thumb"},
        )

        assert response.status_code == 200
        frame = response.json()["frames"][0]
        assert frame["actual_timestamp_s"] == 9.0
        assert frame["url"] == "https://signed.example.com/thumb/vid_1/thumb_00009.jpg"

    def test_storage_config_failure_after_billing_compensates_units(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed: list[dict] = []
        compensated: list[dict] = []
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        def mock_consume(**kwargs):
            consumed.append(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        def mock_compensate(**kwargs):
            compensated.append(kwargs)

        def fail_r2_config():
            raise StorageConfigError("R2 is not configured")

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr("src.api.app.db_compensate_api_units", mock_compensate)
        monkeypatch.setattr("src.api.app.R2Config.from_env", fail_r2_config)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "thumb"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 503
        assert len(consumed) == 1
        assert len(compensated) == 1
        assert consumed[0]["request_id"] == compensated[0]["request_id"]
        assert compensated[0]["units"] == 1
        assert compensated[0]["metadata"]["event_type"] == "frames_thumb_failed"


# ---------------------------------------------------------------------------
# POST /api/v1/videos/{video_id}/frames -- resolution=high
# ---------------------------------------------------------------------------


class TestVideoFramesHigh:
    def test_source_not_retained_for_youtube_video_409(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "high"},
        )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "source_not_retained"

    def test_source_not_retained_when_r2_object_missing_409(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=False))

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "high"},
        )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "source_not_retained"

    def test_happy_path_returns_base64_frames(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=True))

        captured: dict = {}

        def fake_extract(source_url, dedupe_keys):
            captured["source_url"] = source_url
            captured["keys"] = dedupe_keys
            return {
                key: ExtractedFrame(image_base64="ZmFrZQ==", width=640, height=360)
                for key in dedupe_keys
            }

        monkeypatch.setattr("src.api.app.extract_high_res_frames", fake_extract)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [3.2], "resolution": "high"},
        )

        assert response.status_code == 200
        frame = response.json()["frames"][0]
        assert frame["resolution"] == "high"
        assert frame["image_base64"] == "ZmFrZQ=="
        assert frame["width"] == 640
        assert frame["height"] == 360
        assert frame["actual_timestamp_s"] == 3.0
        assert captured["source_url"] == "https://source.example.com/presigned"
        assert captured["keys"] == [3]

    def test_per_frame_error_does_not_fail_whole_call(self, monkeypatch) -> None:
        _authenticate("user_123")
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=True))

        def fake_extract(source_url, dedupe_keys):
            result = {}
            for key in dedupe_keys:
                if key == 500:
                    result[key] = ExtractedFrame(
                        image_base64=None, width=None, height=None,
                        error="No frame was produced (timestamp may be past the end of the video)",
                    )
                else:
                    result[key] = ExtractedFrame(image_base64="AA==", width=100, height=50)
            return result

        monkeypatch.setattr("src.api.app.extract_high_res_frames", fake_extract)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [5.0, 500.0], "resolution": "high"},
        )

        assert response.status_code == 200
        frames = response.json()["frames"]
        assert frames[0]["image_base64"] == "AA=="
        assert frames[0]["error"] is None
        assert frames[1]["image_base64"] is None
        assert "past the end" in frames[1]["error"]

    def test_api_key_bills_frames_high_units(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=True))
        monkeypatch.setattr(
            "src.api.app.extract_high_res_frames",
            lambda source_url, keys: {
                key: ExtractedFrame(image_base64="AA==", width=1, height=1) for key in keys
            },
        )

        consumed: dict = {}

        def mock_consume(**kwargs):
            consumed.update(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "high"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert consumed["event_type"] == "frames_high"
        assert consumed["units"] == 5

    def test_source_not_retained_is_not_billed(self, monkeypatch) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _video_record(vid, status="ready"),
        )

        consumed = {"called": False}

        def mock_consume(**kwargs):
            consumed["called"] = True
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [1.0], "resolution": "high"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 409
        assert consumed["called"] is False

    def test_per_frame_error_does_not_compensate_units(self, monkeypatch) -> None:
        """A partial (per-frame) extraction error is not a wholesale failure."""
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=True))

        def fake_extract(source_url, dedupe_keys):
            return {
                key: (
                    ExtractedFrame(
                        image_base64=None, width=None, height=None,
                        error="No frame was produced",
                    )
                    if key == 500
                    else ExtractedFrame(image_base64="AA==", width=100, height=50)
                )
                for key in dedupe_keys
            }

        monkeypatch.setattr("src.api.app.extract_high_res_frames", fake_extract)

        compensated = {"called": False}

        def mock_compensate(**kwargs):
            compensated["called"] = True

        monkeypatch.setattr(
            "src.api.app.db_consume_api_units",
            lambda **kwargs: ApiUnitConsumeResult(allowed=True, remaining_balance=99),
        )
        monkeypatch.setattr("src.api.app.db_compensate_api_units", mock_compensate)

        response = client.post(
            "/api/v1/videos/vid_1/frames",
            json={"timestamps": [5.0, 500.0], "resolution": "high"},
            headers={"Authorization": f"Bearer {raw_key}"},
        )

        assert response.status_code == 200
        assert compensated["called"] is False

    def test_wholesale_extraction_failure_after_billing_compensates_units(
        self, monkeypatch,
    ) -> None:
        raw_key, _ = _setup_api_key_auth(monkeypatch)
        consumed: list[dict] = []
        compensated: list[dict] = []
        monkeypatch.setattr(
            "src.api.app.db_get_video",
            lambda vid, user_id=None: _upload_video_record(vid, status="ready"),
        )
        monkeypatch.setattr("src.api.app.R2Config.from_env", lambda: object())
        monkeypatch.setattr("src.api.app.R2Store", FakeHighResR2Store(exists=True))

        def mock_consume(**kwargs):
            consumed.append(kwargs)
            return ApiUnitConsumeResult(allowed=True, remaining_balance=99)

        def mock_compensate(**kwargs):
            compensated.append(kwargs)

        def fake_extract(*_args, **_kwargs):
            # An unexpected, wholesale extraction failure (not a per-frame
            # error the service already handles internally).
            raise RuntimeError("extraction subsystem crashed")

        monkeypatch.setattr("src.api.app.db_consume_api_units", mock_consume)
        monkeypatch.setattr("src.api.app.db_compensate_api_units", mock_compensate)
        monkeypatch.setattr("src.api.app.extract_high_res_frames", fake_extract)

        with pytest.raises(RuntimeError, match="extraction subsystem crashed"):
            client.post(
                "/api/v1/videos/vid_1/frames",
                json={"timestamps": [1.0], "resolution": "high"},
                headers={"Authorization": f"Bearer {raw_key}"},
            )

        assert len(consumed) == 1
        assert len(compensated) == 1
        assert consumed[0]["request_id"] == compensated[0]["request_id"]
        assert compensated[0]["units"] == 5
        assert compensated[0]["metadata"]["event_type"] == "frames_high_failed"
