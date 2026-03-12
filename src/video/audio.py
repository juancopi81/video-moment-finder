from __future__ import annotations

from pathlib import Path
import subprocess

from src.utils.subprocess import format_subprocess_error


class AudioExtractionError(RuntimeError):
    """Raised when ffmpeg audio extraction fails."""


def extract_audio_track(
    video_path: Path,
    output_path: Path,
    *,
    sample_rate_hz: int = 16000,
) -> Path:
    """Extract a mono WAV audio track from a video file."""
    if not video_path.exists():
        raise AudioExtractionError(f"Video not found: {video_path}")
    if sample_rate_hz <= 0:
        raise AudioExtractionError("sample_rate_hz must be greater than zero")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate_hz),
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise AudioExtractionError(
            "ffmpeg binary not found in runtime environment. Install ffmpeg in the service image."
        ) from exc
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "ffmpeg audio extraction failed")
        raise AudioExtractionError(message) from exc

    if not output_path.exists():
        raise AudioExtractionError(f"ffmpeg did not create audio output: {output_path}")

    return output_path
