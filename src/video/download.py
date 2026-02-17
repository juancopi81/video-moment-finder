from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess

from src.utils.subprocess import format_subprocess_error


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails to download a video."""


class VideoMetadataError(RuntimeError):
    """Raised when yt-dlp fails to fetch video metadata."""


@dataclass(frozen=True)
class VideoMetadata:
    duration_s: float | None
    is_live: bool


def fetch_video_metadata(url: str) -> VideoMetadata:
    """Fetch video metadata with yt-dlp without downloading the file."""
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--dump-json",
        url,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "yt-dlp metadata fetch failed with no output")
        raise VideoMetadataError(message) from exc

    payload = result.stdout.strip()
    if not payload:
        raise VideoMetadataError("yt-dlp returned empty metadata output")

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise VideoMetadataError("yt-dlp returned invalid JSON metadata") from exc

    duration = data.get("duration")
    duration_s = float(duration) if duration is not None else None
    is_live = bool(data.get("is_live"))
    live_status = str(data.get("live_status") or "").lower()
    if live_status in {"is_live", "is_upcoming"}:
        is_live = True

    return VideoMetadata(duration_s=duration_s, is_live=is_live)


def download_video(
    url: str,
    output_dir: Path,
    *,
    quality: str = "best[height<=720]",
    output_basename: str = "video",
) -> Path:
    """
    Download a video via yt-dlp.

    Note: YouTube may block downloads from cloud IPs (bot detection).
    Fail-fast behavior:
    - Raises DownloadError on any yt-dlp failure.
    - Raises DownloadError if output file cannot be uniquely determined.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / f"{output_basename}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        quality,
        "-o",
        output_template,
        "--merge-output-format",
        "mp4",
        url,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "yt-dlp failed with no output")
        raise DownloadError(message) from exc

    matches = sorted(output_dir.glob(f"{output_basename}.*"))
    if len(matches) != 1:
        raise DownloadError(
            f"Expected exactly one output file, found {len(matches)}: {matches}"
        )

    return matches[0]
