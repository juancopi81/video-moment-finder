"""Frame retrieval service: timestamp-to-frame mapping and high-res ffmpeg extraction.

Two resolutions are supported by the ``/api/v1/videos/{video_id}/frames`` endpoint:

- ``thumb``: maps requested timestamps to the 1-fps frame index sampled during
  processing (see ``src/video/frames.py``) and returns presigned R2 URLs for the
  already-stored thumbnail JPEGs. No image bytes pass through this service.
- ``high``: reads directly from the retained source video in R2 via a presigned
  URL and extracts frames on demand with ffmpeg, returning JPEG bytes as base64.

Both modes round every requested timestamp to the nearest whole second before
deduplicating and processing, matching the 1-fps granularity frames were
extracted at. ``thumb`` additionally clamps to the known frame range so a
timestamp past the end of a (long-truncated) video still resolves to a URL.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import base64
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image, UnidentifiedImageError

from src.utils.logging import Timer, get_logger
from src.utils.subprocess import format_subprocess_error
from src.video.metadata import max_video_duration_s

logger = get_logger(__name__)

FRAME_SAMPLE_FPS = 1.0
MAX_TIMESTAMPS_THUMB = 25
MAX_TIMESTAMPS_HIGH = 8
HIGH_RES_MAX_WIDTH = 1280
HIGH_RES_JPEG_QUALITY = "3"
HIGH_RES_MAX_WORKERS = 4
HIGH_RES_FFMPEG_TIMEOUT_S = 60

Resolution = Literal["thumb", "high"]


class FrameRequestValidationError(ValueError):
    """Raised when a frame retrieval request fails validation."""


@dataclass(frozen=True)
class FramePlanItem:
    """One request-order entry mapped to a dedupe key for frame retrieval."""

    requested_timestamp_s: float
    actual_timestamp_s: float
    dedupe_key: int


@dataclass(frozen=True)
class ExtractedFrame:
    """Result of a single high-resolution ffmpeg frame extraction."""

    image_base64: str | None
    width: int | None
    height: int | None
    error: str | None = None


def validate_frame_request(timestamps: list[float], resolution: Resolution) -> None:
    """Validate a frame retrieval request, raising with a caller-facing detail."""
    if not timestamps:
        raise FrameRequestValidationError("timestamps must be a non-empty list")

    max_allowed = MAX_TIMESTAMPS_HIGH if resolution == "high" else MAX_TIMESTAMPS_THUMB
    if len(timestamps) > max_allowed:
        raise FrameRequestValidationError(
            f"At most {max_allowed} timestamps are allowed for resolution="
            f"{resolution!r} (got {len(timestamps)})"
        )

    if any(t < 0 for t in timestamps):
        raise FrameRequestValidationError("timestamps must be non-negative")


def _round_to_frame_index(timestamp_s: float, *, fps: float = FRAME_SAMPLE_FPS) -> int:
    return round(timestamp_s * fps)


def _clamp(frame_index: int, *, max_frame_index: int) -> int:
    return max(0, min(frame_index, max_frame_index))


def build_frame_plan(
    timestamps: list[float],
    *,
    max_frame_index: int | None,
) -> list[FramePlanItem]:
    """Map each requested timestamp to a dedupe key, preserving request order.

    ``max_frame_index`` clamps the mapped index to ``[0, max_frame_index]``
    (used for ``thumb``, where the stored frame range is known). Pass ``None``
    for ``high``, where a timestamp past the end of the source video should
    surface as a per-frame ffmpeg extraction error instead of being clamped.
    """
    plan: list[FramePlanItem] = []
    for timestamp_s in timestamps:
        frame_index = _round_to_frame_index(timestamp_s)
        if max_frame_index is not None:
            frame_index = _clamp(frame_index, max_frame_index=max_frame_index)
        plan.append(
            FramePlanItem(
                requested_timestamp_s=timestamp_s,
                actual_timestamp_s=float(frame_index),
                dedupe_key=frame_index,
            )
        )
    return plan


def build_thumb_frame_plan(timestamps: list[float]) -> list[FramePlanItem]:
    """Build a frame plan clamped to the known 1-fps thumbnail frame range.

    The upper bound is derived from the configured max video duration so it
    stays in lockstep with VIDEO_MAX_DURATION_S rather than duplicating a
    frame-count literal.
    """
    max_frame_index = int(max_video_duration_s() * FRAME_SAMPLE_FPS) - 1
    return build_frame_plan(timestamps, max_frame_index=max_frame_index)


def build_high_res_frame_plan(timestamps: list[float]) -> list[FramePlanItem]:
    """Build a frame plan for on-demand high-res extraction (no upper clamp)."""
    return build_frame_plan(timestamps, max_frame_index=None)


def unique_dedupe_keys(plan: list[FramePlanItem]) -> list[int]:
    """Return the unique dedupe keys from a plan, in first-seen order."""
    seen: set[int] = set()
    ordered: list[int] = []
    for item in plan:
        if item.dedupe_key not in seen:
            seen.add(item.dedupe_key)
            ordered.append(item.dedupe_key)
    return ordered


def extract_high_res_frame(source_url: str, timestamp_s: float) -> ExtractedFrame:
    """Extract a single frame from a remote source video via ffmpeg over HTTP.

    ``-ss`` is placed before ``-i`` so ffmpeg seeks using HTTP range requests
    instead of downloading the whole file first. Never raises: extraction
    failures (including a timestamp past the end of the video) are returned
    as an ``ExtractedFrame`` with ``error`` set, so one bad timestamp doesn't
    fail the whole request.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "frame.jpg"
        cmd = [
            "ffmpeg",
            "-ss",
            str(timestamp_s),
            "-i",
            source_url,
            "-frames:v",
            "1",
            "-vf",
            f"scale='min({HIGH_RES_MAX_WIDTH},iw)':-2",
            "-q:v",
            HIGH_RES_JPEG_QUALITY,
            "-y",
            str(output_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=HIGH_RES_FFMPEG_TIMEOUT_S,
            )
        except FileNotFoundError as exc:
            logger.error("ffmpeg binary not found for high-res frame extraction: %s", exc)
            return ExtractedFrame(
                image_base64=None, width=None, height=None,
                error="ffmpeg is not available in the runtime environment",
            )
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg timed out extracting frame at t=%.2fs", timestamp_s)
            return ExtractedFrame(
                image_base64=None, width=None, height=None,
                error="Frame extraction timed out",
            )
        except subprocess.CalledProcessError as exc:
            message = format_subprocess_error(exc, "ffmpeg failed with no output")
            logger.warning(
                "ffmpeg failed extracting frame at t=%.2fs: %s", timestamp_s, message,
            )
            return ExtractedFrame(image_base64=None, width=None, height=None, error=message)

        if not output_path.exists():
            return ExtractedFrame(
                image_base64=None,
                width=None,
                height=None,
                error="No frame was produced (timestamp may be past the end of the video)",
            )

        image_bytes = output_path.read_bytes()
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
        except (SyntaxError, UnidentifiedImageError, OSError) as exc:
            return ExtractedFrame(
                image_base64=None, width=None, height=None,
                error=f"Invalid extracted frame: {exc}",
            )

        return ExtractedFrame(
            image_base64=base64.b64encode(image_bytes).decode("ascii"),
            width=width,
            height=height,
        )


def extract_high_res_frames(
    source_url: str,
    dedupe_keys: list[int],
    *,
    max_workers: int = HIGH_RES_MAX_WORKERS,
) -> dict[int, ExtractedFrame]:
    """Extract multiple frames concurrently, keyed by (deduped) frame second."""
    if not dedupe_keys:
        return {}

    workers = max(1, min(max_workers, len(dedupe_keys)))
    results: dict[int, ExtractedFrame] = {}
    with Timer("High-res frame extraction", logger, level="debug"):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_key = {
                executor.submit(extract_high_res_frame, source_url, float(key)): key
                for key in dedupe_keys
            }
            for future, key in future_to_key.items():
                results[key] = future.result()

    return results
