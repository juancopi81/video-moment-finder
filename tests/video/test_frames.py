from __future__ import annotations

from pathlib import Path

import pytest

from src.video.frames import FrameExtractionError, extract_frames


def test_extract_frames_raises_clear_error_when_ffmpeg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video-bytes")

    def _missing_ffmpeg(*args, **kwargs):  # pragma: no cover - deterministic exception path
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr("src.video.frames.subprocess.run", _missing_ffmpeg)

    with pytest.raises(FrameExtractionError, match="ffmpeg binary not found"):
        extract_frames(video_path, tmp_path / "frames")
