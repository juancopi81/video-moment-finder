"""Portability and fail-fast checks for local original-video generation."""
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from scripts.learning.make_fixture_video import audio_duration, choose_font, choose_speech_engine


def test_auto_uses_macos_speech_when_ffmpeg_has_no_flite():
    with patch("scripts.learning.make_fixture_video.shutil.which", return_value="available"), patch(
        "scripts.learning.make_fixture_video.run", return_value=CompletedProcess([], 0, " .. anull A->A\n")
    ):
        assert choose_speech_engine("auto") == "say"


def test_explicit_flite_does_not_silently_change_engine():
    with patch("scripts.learning.make_fixture_video.shutil.which", return_value="available"), patch(
        "scripts.learning.make_fixture_video.run", return_value=CompletedProcess([], 0, "")
    ), pytest.raises(ValueError, match="no flite filter"):
        choose_speech_engine("flite")


@pytest.mark.parametrize("value", ["N/A", "0", "-1", "nan", "inf"])
def test_empty_or_invalid_audio_stops_before_video_generation(value):
    with patch("scripts.learning.make_fixture_video.run", return_value=CompletedProcess([], 0, value)), pytest.raises(
        ValueError, match="no usable audio"
    ):
        audio_duration(Path("empty.wav"))


def test_valid_audio_duration_is_preserved():
    with patch("scripts.learning.make_fixture_video.run", return_value=CompletedProcess([], 0, "13.057875\n")):
        assert audio_duration(Path("speech.wav")) == 13.057875


def test_explicit_missing_font_does_not_silently_substitute(tmp_path):
    with pytest.raises(ValueError, match="Supply --font"):
        choose_font(tmp_path / "missing.ttf")
