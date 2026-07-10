"""Unit tests for src.api.frames: request validation, plan/dedupe, ffmpeg extraction."""
from __future__ import annotations

import subprocess
import threading
import time

import pytest

from src.api import frames as frames_module
from src.api.frames import (
    DEFAULT_FRAMES_FFMPEG_MAX_CONCURRENCY,
    FRAME_SAMPLE_FPS,
    FrameRequestValidationError,
    MAX_TIMESTAMPS_HIGH,
    MAX_TIMESTAMPS_THUMB,
    build_high_res_frame_plan,
    build_thumb_frame_plan,
    extract_high_res_frame,
    extract_high_res_frames,
    ffmpeg_max_concurrency,
    unique_dedupe_keys,
    validate_frame_request,
)
from src.video.metadata import max_video_duration_s

MAX_THUMB_FRAME_INDEX = int(max_video_duration_s() * FRAME_SAMPLE_FPS) - 1


# ---------------------------------------------------------------------------
# validate_frame_request
# ---------------------------------------------------------------------------


class TestValidateFrameRequest:
    def test_rejects_empty_timestamps(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="non-empty"):
            validate_frame_request([], "thumb")

    def test_rejects_negative_timestamps(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="non-negative"):
            validate_frame_request([1.0, -0.5], "thumb")

    def test_accepts_max_thumb_count(self) -> None:
        validate_frame_request([float(i) for i in range(MAX_TIMESTAMPS_THUMB)], "thumb")

    def test_rejects_over_thumb_cap(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="At most 25"):
            validate_frame_request([float(i) for i in range(MAX_TIMESTAMPS_THUMB + 1)], "thumb")

    def test_accepts_max_high_count(self) -> None:
        validate_frame_request([float(i) for i in range(MAX_TIMESTAMPS_HIGH)], "high")

    def test_rejects_over_high_cap(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="At most 8"):
            validate_frame_request([float(i) for i in range(MAX_TIMESTAMPS_HIGH + 1)], "high")

    def test_rejects_infinite_timestamp(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="finite"):
            validate_frame_request([1.0, float("inf")], "thumb")

    def test_rejects_negative_infinite_timestamp(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="finite"):
            validate_frame_request([float("-inf")], "thumb")

    def test_rejects_nan_timestamp(self) -> None:
        with pytest.raises(FrameRequestValidationError, match="finite"):
            validate_frame_request([float("nan")], "high")

    def test_rejects_json_overflow_timestamp(self) -> None:
        # JSON `1e1000` is valid syntax but overflows to `inf` when parsed to
        # a Python float -- exactly the payload shape this guards against.
        with pytest.raises(FrameRequestValidationError, match="finite"):
            validate_frame_request([1e1000], "thumb")


# ---------------------------------------------------------------------------
# Frame plan building + dedupe
# ---------------------------------------------------------------------------


class TestFramePlan:
    def test_thumb_plan_rounds_to_frame_index(self) -> None:
        plan = build_thumb_frame_plan([1.4, 9.6])

        assert [item.dedupe_key for item in plan] == [1, 10]
        assert [item.actual_timestamp_s for item in plan] == [1.0, 10.0]
        assert [item.requested_timestamp_s for item in plan] == [1.4, 9.6]

    def test_thumb_plan_clamps_to_max_frame_index(self) -> None:
        plan = build_thumb_frame_plan([999999.0])

        assert plan[0].dedupe_key == MAX_THUMB_FRAME_INDEX
        assert plan[0].actual_timestamp_s == float(MAX_THUMB_FRAME_INDEX)

    def test_high_res_plan_does_not_clamp(self) -> None:
        plan = build_high_res_frame_plan([999999.0])

        assert plan[0].dedupe_key == 999999
        assert plan[0].actual_timestamp_s == 999999.0

    def test_unique_dedupe_keys_preserves_first_seen_order(self) -> None:
        plan = build_thumb_frame_plan([2.4, 2.3, 50.0, 2.6])

        # 2.4 and 2.3 both round to frame_index 2; 2.6 rounds to 3.
        assert [item.dedupe_key for item in plan] == [2, 2, 50, 3]
        assert unique_dedupe_keys(plan) == [2, 50, 3]

    def test_thumb_plan_clamps_to_video_duration_when_provided(self) -> None:
        # A 10s video's last full sampled frame is index 9, well below the
        # global VIDEO_MAX_DURATION_S-derived ceiling.
        plan = build_thumb_frame_plan([999999.0], duration_s=10.0)

        assert plan[0].dedupe_key == 9
        assert plan[0].actual_timestamp_s == 9.0

    def test_thumb_plan_falls_back_to_global_cap_when_duration_none(self) -> None:
        plan = build_thumb_frame_plan([999999.0], duration_s=None)

        assert plan[0].dedupe_key == MAX_THUMB_FRAME_INDEX

    def test_thumb_plan_clamp_never_goes_negative_for_sub_second_video(self) -> None:
        plan = build_thumb_frame_plan([5.0], duration_s=0.2)

        assert plan[0].dedupe_key == 0
        assert plan[0].actual_timestamp_s == 0.0

    def test_thumb_plan_ignores_non_positive_duration(self) -> None:
        # Defensive: a non-positive duration_s (shouldn't happen in practice,
        # since ffprobe rejects <= 0) falls back to the global cap instead of
        # producing a negative clamp.
        plan = build_thumb_frame_plan([999999.0], duration_s=0.0)

        assert plan[0].dedupe_key == MAX_THUMB_FRAME_INDEX


# ---------------------------------------------------------------------------
# extract_high_res_frame (ffmpeg mocked, never invoked for real)
# ---------------------------------------------------------------------------


class TestExtractHighResFrame:
    def test_returns_error_when_ffmpeg_binary_missing(self, monkeypatch) -> None:
        def fake_run(*_args, **_kwargs):
            raise FileNotFoundError("no ffmpeg")

        monkeypatch.setattr(frames_module.subprocess, "run", fake_run)

        result = extract_high_res_frame("https://example.com/source.mp4", 5.0)

        assert result.image_base64 is None
        assert result.error is not None
        assert "ffmpeg" in result.error.lower()

    def test_returns_error_on_ffmpeg_timeout(self, monkeypatch) -> None:
        def fake_run(*_args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=kwargs.get("timeout", 60))

        monkeypatch.setattr(frames_module.subprocess, "run", fake_run)

        result = extract_high_res_frame("https://example.com/source.mp4", 5.0)

        assert result.image_base64 is None
        assert "timed out" in result.error.lower()

    def test_returns_error_on_ffmpeg_failure(self, monkeypatch) -> None:
        def fake_run(*_args, **_kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd="ffmpeg", output="", stderr="Invalid seek target",
            )

        monkeypatch.setattr(frames_module.subprocess, "run", fake_run)

        result = extract_high_res_frame("https://example.com/source.mp4", 5.0)

        assert result.image_base64 is None
        assert "Invalid seek target" in result.error

    def test_returns_error_when_no_output_file_produced(self, monkeypatch) -> None:
        # ffmpeg "succeeds" (e.g. exit 0 with no output) but never writes a file --
        # this is how a timestamp past the end of the video should surface.
        monkeypatch.setattr(frames_module.subprocess, "run", lambda *a, **kw: None)

        result = extract_high_res_frame("https://example.com/source.mp4", 99999.0)

        assert result.image_base64 is None
        assert "past the end" in result.error.lower()

    def test_returns_base64_image_on_success(self, monkeypatch, tmp_path) -> None:
        from PIL import Image

        def fake_run(cmd, **_kwargs):
            output_path = cmd[-1]
            Image.new("RGB", (64, 32), color="red").save(output_path, format="JPEG")

        monkeypatch.setattr(frames_module.subprocess, "run", fake_run)

        result = extract_high_res_frame("https://example.com/source.mp4", 5.0)

        assert result.error is None
        assert result.width == 64
        assert result.height == 32
        assert result.image_base64 is not None

    def test_bounds_process_wide_concurrent_extractions(self, monkeypatch) -> None:
        """N+2 concurrent extractions must never exceed the semaphore's bound.

        Replaces the module-level semaphore with a small bound and a fake
        ``subprocess.run`` that blocks until released, then asserts the
        observed in-flight count never exceeds that bound -- proving the
        semaphore, not just the per-request ThreadPoolExecutor, is gating
        concurrency process-wide.
        """
        bound = 2
        monkeypatch.setattr(
            frames_module, "_FFMPEG_CONCURRENCY_SEMAPHORE", threading.BoundedSemaphore(bound),
        )

        lock = threading.Lock()
        in_flight = 0
        max_observed = 0
        release_event = threading.Event()

        def fake_run(*_args, **_kwargs):
            nonlocal in_flight, max_observed
            with lock:
                in_flight += 1
                max_observed = max(max_observed, in_flight)
            release_event.wait(timeout=5)
            with lock:
                in_flight -= 1
            raise subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", output="", stderr="boom")

        monkeypatch.setattr(frames_module.subprocess, "run", fake_run)

        threads = [
            threading.Thread(
                target=extract_high_res_frame,
                args=("https://example.com/source.mp4", float(i)),
            )
            for i in range(bound + 2)
        ]
        for thread in threads:
            thread.start()

        # Give every thread a chance to either enter fake_run or block on the
        # semaphore's acquire.
        time.sleep(0.2)
        assert max_observed == bound

        release_event.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()

        assert max_observed == bound


class TestFfmpegMaxConcurrency:
    def test_default_is_four(self, monkeypatch) -> None:
        monkeypatch.delenv("FRAMES_FFMPEG_MAX_CONCURRENCY", raising=False)

        assert ffmpeg_max_concurrency() == DEFAULT_FRAMES_FFMPEG_MAX_CONCURRENCY == 4

    def test_reads_env_override(self, monkeypatch) -> None:
        monkeypatch.setenv("FRAMES_FFMPEG_MAX_CONCURRENCY", "9")

        assert ffmpeg_max_concurrency() == 9


class TestExtractHighResFrames:
    def test_extracts_each_unique_key_concurrently(self, monkeypatch) -> None:
        calls: list[float] = []

        def fake_extract_one(source_url, timestamp_s):
            calls.append(timestamp_s)
            return frames_module.ExtractedFrame(
                image_base64="AA==", width=1, height=1,
            )

        monkeypatch.setattr(frames_module, "extract_high_res_frame", fake_extract_one)

        results = extract_high_res_frames("https://example.com/source.mp4", [1, 2, 3])

        assert set(calls) == {1.0, 2.0, 3.0}
        assert set(results.keys()) == {1, 2, 3}
        assert all(r.image_base64 == "AA==" for r in results.values())

    def test_empty_keys_returns_empty_dict(self) -> None:
        assert extract_high_res_frames("https://example.com/source.mp4", []) == {}
