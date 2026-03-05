from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.video.metadata import VideoMetadataProbeError, probe_video_duration_s


def test_probe_video_duration_s_parses_ffprobe_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video-bytes")

    def _fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args=["ffprobe"], returncode=0, stdout="1799.2\n")

    monkeypatch.setattr("src.video.metadata.subprocess.run", _fake_run)

    assert probe_video_duration_s(video_path) == pytest.approx(1799.2)


def test_probe_video_duration_s_raises_clear_error_when_ffprobe_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video-bytes")

    def _missing_ffprobe(*args, **kwargs):  # pragma: no cover - deterministic exception path
        raise FileNotFoundError(2, "No such file or directory", "ffprobe")

    monkeypatch.setattr("src.video.metadata.subprocess.run", _missing_ffprobe)

    with pytest.raises(VideoMetadataProbeError, match="ffprobe binary not found"):
        probe_video_duration_s(video_path)
