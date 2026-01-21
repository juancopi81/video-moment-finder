from __future__ import annotations

from pathlib import Path
import subprocess


class DownloadError(RuntimeError):
    """Raised when yt-dlp fails to download a video."""


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
        stderr = exc.stderr.strip()
        stdout = exc.stdout.strip()
        message = stderr or stdout or "yt-dlp failed with no output"
        raise DownloadError(message) from exc

    matches = sorted(output_dir.glob(f"{output_basename}.*"))
    if len(matches) != 1:
        raise DownloadError(
            f"Expected exactly one output file, found {len(matches)}: {matches}"
        )

    return matches[0]
