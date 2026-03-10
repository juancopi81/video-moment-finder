from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from src.video.download import DownloadError, VideoMetadata, download_video, fetch_video_metadata


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


def test_fetch_video_metadata_uses_hardened_yt_dlp_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=0,
            stdout='{"duration": 42, "is_live": false}\n',
        )

    monkeypatch.setattr("src.video.download.subprocess.run", _fake_run)

    metadata = fetch_video_metadata("https://www.youtube.com/watch?v=abc123xyz45")

    assert metadata == VideoMetadata(duration_s=42.0, is_live=False)
    assert calls == [
        [
            "yt-dlp",
            "--no-update",
            "--js-runtimes",
            "node",
            "--skip-download",
            "--no-playlist",
            "--dump-json",
            "https://www.youtube.com/watch?v=abc123xyz45",
        ]
    ]


def test_download_video_uses_hardened_yt_dlp_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    output_dir = tmp_path / "downloads"

    def _fake_run(cmd: list[str], **kwargs):
        calls.append(cmd)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "video.mp4").write_bytes(b"video-bytes")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")

    monkeypatch.setattr("src.video.download.subprocess.run", _fake_run)

    resolved = download_video(
        "https://www.youtube.com/watch?v=abc123xyz45",
        output_dir,
    )

    assert resolved == output_dir / "video.mp4"
    assert calls == [
        [
            "yt-dlp",
            "--no-update",
            "--js-runtimes",
            "node",
            "-f",
            "best[height<=720]",
            "-o",
            str(output_dir / "video.%(ext)s"),
            "--merge-output-format",
            "mp4",
            "https://www.youtube.com/watch?v=abc123xyz45",
        ]
    ]
