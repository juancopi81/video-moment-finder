from __future__ import annotations

import json
from pathlib import Path
import subprocess

from src.video.transcripts import extract_transcript_segments


def test_extract_transcript_segments_returns_empty_when_no_tracks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"subtitles": {}, "automatic_captions": {}}),
            stderr="",
        )

    monkeypatch.setattr("src.video.transcripts.subprocess.run", _fake_run)

    segments = extract_transcript_segments(
        "https://www.youtube.com/watch?v=abc123xyz45",
        tmp_path,
    )

    assert segments == []
    assert len(calls) == 1
    assert "--dump-json" in calls[0]


def test_extract_transcript_segments_prefers_manual_english_and_merges_growth_cues(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        calls.append(cmd)
        if "--dump-json" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {
                        "subtitles": {"en": [{"ext": "vtt"}], "es": [{"ext": "vtt"}]},
                        "automatic_captions": {"en": [{"ext": "vtt"}]},
                    }
                ),
                stderr="",
            )

        output_template = Path(cmd[cmd.index("-o") + 1])
        transcript_path = Path(str(output_template).replace("%(ext)s", "en.vtt"))
        transcript_path.write_text(
            "\n".join(
                [
                    "WEBVTT",
                    "",
                    "00:00:01.000 --> 00:00:02.000",
                    "<c>hello</c>",
                    "",
                    "00:00:02.000 --> 00:00:04.000",
                    "hello world",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("src.video.transcripts.subprocess.run", _fake_run)

    segments = extract_transcript_segments(
        "https://www.youtube.com/watch?v=abc123xyz45",
        tmp_path,
    )

    assert len(calls) == 2
    assert "--write-subs" in calls[1]
    assert "--write-auto-subs" not in calls[1]
    assert segments[0].segment_index == 0
    assert segments[0].start_s == 1.0
    assert segments[0].end_s == 4.0
    assert segments[0].text == "hello world"
    assert segments[0].language_code == "en"


def test_extract_transcript_segments_uses_automatic_captions_when_needed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []

    def _fake_run(cmd: list[str], check: bool, capture_output: bool, text: bool):
        calls.append(cmd)
        if "--dump-json" in cmd:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout=json.dumps(
                    {"subtitles": {}, "automatic_captions": {"en-US": [{}]}}
                ),
                stderr="",
            )

        output_template = Path(cmd[cmd.index("-o") + 1])
        transcript_path = Path(str(output_template).replace("%(ext)s", "en-US.vtt"))
        transcript_path.write_text(
            "\n".join(
                [
                    "WEBVTT",
                    "",
                    "00:00:05.000 --> 00:00:07.000",
                    "automatic captions here",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("src.video.transcripts.subprocess.run", _fake_run)

    segments = extract_transcript_segments(
        "https://www.youtube.com/watch?v=abc123xyz45",
        tmp_path,
    )

    assert len(calls) == 2
    assert "--write-auto-subs" in calls[1]
    assert segments[0].text == "automatic captions here"
    assert segments[0].language_code == "en-US"
