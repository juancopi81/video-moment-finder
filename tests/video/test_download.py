from __future__ import annotations

from pathlib import Path

import pytest

from src.video.download import DownloadError, download_video


def test_download_video_uses_local_cache(tmp_path: Path, monkeypatch) -> None:
    local_dir = tmp_path / "cache"
    local_dir.mkdir()
    local_video = local_dir / "abc123xyz45.mp4"
    local_video.write_bytes(b"video-bytes")

    def _fail(*args, **kwargs):  # pragma: no cover - should not run
        raise AssertionError("yt-dlp should not be invoked when local cache exists")

    monkeypatch.setattr("src.video.download.subprocess.run", _fail)

    resolved = download_video(
        "https://www.youtube.com/watch?v=abc123xyz45",
        tmp_path / "downloads",
        local_video_dir=local_dir,
    )

    assert resolved == local_video


def test_download_video_rejects_multiple_local_matches(tmp_path: Path) -> None:
    local_dir = tmp_path / "cache"
    local_dir.mkdir()
    (local_dir / "abc123xyz45.mp4").write_bytes(b"one")
    (local_dir / "abc123xyz45.webm").write_bytes(b"two")

    with pytest.raises(DownloadError, match="Multiple local video files found"):
        download_video(
            "https://www.youtube.com/watch?v=abc123xyz45",
            tmp_path / "downloads",
            local_video_dir=local_dir,
        )
