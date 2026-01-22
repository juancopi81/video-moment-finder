from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from src.utils.subprocess import format_subprocess_error


class FrameExtractionError(RuntimeError):
    """Raised when frame extraction fails."""


@dataclass(frozen=True)
class FrameInfo:
    index: int
    timestamp_s: float
    path: Path
    thumbnail_path: Path | None = None


def extract_frames(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float = 1.0,
    quality: int = 2,
    thumbnail_dir: Path | None = None,
    thumbnail_max_width: int = 320,
    thumbnail_quality: int = 4,
) -> list[FrameInfo]:
    """
    Extract frames at a fixed FPS and optionally generate thumbnails.

    Fail-fast behavior:
    - Raises FrameExtractionError on any ffmpeg failure.
    - Raises FrameExtractionError if no frames are extracted.
    - Raises FrameExtractionError if thumbnails are requested but count mismatches.
    """
    if not video_path.exists():
        raise FrameExtractionError(f"Video not found: {video_path}")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if thumbnail_dir is not None and thumbnail_max_width <= 0:
        raise ValueError("thumbnail_max_width must be > 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(output_dir / "frame_%05d.jpg")

    _run_ffmpeg(
        [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            str(quality),
            output_pattern,
        ]
    )

    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise FrameExtractionError("No frames were extracted.")

    thumbnail_paths: list[Path] | None = None
    if thumbnail_dir is not None:
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        thumbnail_pattern = str(thumbnail_dir / "thumb_%05d.jpg")
        thumbnail_filter = f"fps={fps},scale={thumbnail_max_width}:-1"

        _run_ffmpeg(
            [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vf",
                thumbnail_filter,
                "-q:v",
                str(thumbnail_quality),
                thumbnail_pattern,
            ]
        )

        thumbnail_paths = sorted(thumbnail_dir.glob("thumb_*.jpg"))
        if len(thumbnail_paths) != len(frame_paths):
            raise FrameExtractionError(
                "Thumbnail count does not match frame count: "
                f"{len(thumbnail_paths)} != {len(frame_paths)}"
            )

    frames: list[FrameInfo] = []
    for idx, frame_path in enumerate(frame_paths):
        timestamp_s = idx / fps
        thumbnail_path = thumbnail_paths[idx] if thumbnail_paths is not None else None
        frames.append(
            FrameInfo(
                index=idx,
                timestamp_s=timestamp_s,
                path=frame_path,
                thumbnail_path=thumbnail_path,
            )
        )

    return frames


def _run_ffmpeg(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "ffmpeg failed with no output")
        raise FrameExtractionError(message) from exc
