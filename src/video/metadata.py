from __future__ import annotations

from pathlib import Path
import subprocess

from src.utils.env import get_env_int
from src.utils.subprocess import format_subprocess_error

DEFAULT_MAX_VIDEO_DURATION_S = 30 * 60


class VideoMetadataProbeError(RuntimeError):
    """Raised when local video metadata probing fails."""


def max_video_duration_s() -> int:
    """Return configured max allowed video duration in seconds."""
    return get_env_int("VIDEO_MAX_DURATION_S", DEFAULT_MAX_VIDEO_DURATION_S)


def probe_video_duration_s(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    if not video_path.exists():
        raise VideoMetadataProbeError(f"Video not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise VideoMetadataProbeError(
            "ffprobe binary not found in runtime environment. Install ffmpeg in the service image."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "ffprobe failed with no output")
        raise VideoMetadataProbeError(message) from exc

    raw_duration = result.stdout.strip()
    if not raw_duration:
        raise VideoMetadataProbeError("ffprobe returned empty duration output")

    try:
        duration_s = float(raw_duration)
    except ValueError as exc:
        raise VideoMetadataProbeError("ffprobe returned invalid duration output") from exc

    if duration_s <= 0:
        raise VideoMetadataProbeError("Video duration must be greater than zero")

    return duration_s
