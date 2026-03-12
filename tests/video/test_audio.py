from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.video.audio import AudioExtractionError, extract_audio_track


def test_extract_audio_track_writes_expected_ffmpeg_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "audio" / "input.wav"
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"wav")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="")

    monkeypatch.setattr("src.video.audio.subprocess.run", _fake_run)

    result = extract_audio_track(video_path, output_path)

    assert result == output_path
    assert captured["cmd"] == [
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
        "16000",
        str(output_path),
    ]


def test_extract_audio_track_raises_clear_error_when_ffmpeg_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"video")

    def _missing_ffmpeg(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr("src.video.audio.subprocess.run", _missing_ffmpeg)

    with pytest.raises(AudioExtractionError, match="ffmpeg binary not found"):
        extract_audio_track(video_path, tmp_path / "audio.wav")
