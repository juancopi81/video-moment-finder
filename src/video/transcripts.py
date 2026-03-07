from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from src.utils.subprocess import format_subprocess_error


class TranscriptError(RuntimeError):
    """Raised when transcript extraction fails."""


class TranscriptDownloadError(TranscriptError):
    """Raised when yt-dlp transcript download fails."""


class TranscriptParseError(TranscriptError):
    """Raised when transcript parsing fails."""


@dataclass(frozen=True)
class TranscriptSegment:
    """One caption segment aligned to a start/end time."""

    segment_index: int
    start_s: float
    end_s: float
    text: str
    language_code: str | None = None


_TAG_RE = re.compile(r"<[^>]+>")


def extract_transcript_segments(url: str, output_dir: Path) -> list[TranscriptSegment]:
    """Download the best available YouTube subtitle track and parse it."""
    metadata = _fetch_video_metadata(url)
    selection = _select_subtitle_track(metadata)
    if selection is None:
        return []

    language_code, auto_generated = selection
    subtitle_path = _download_subtitle_track(
        url=url,
        output_dir=output_dir,
        language_code=language_code,
        auto_generated=auto_generated,
    )
    return _parse_vtt_segments(subtitle_path, language_code=language_code)


def _fetch_video_metadata(url: str) -> dict[str, Any]:
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
        message = format_subprocess_error(
            exc,
            "yt-dlp transcript metadata fetch failed",
        )
        raise TranscriptDownloadError(message) from exc

    payload = result.stdout.strip()
    if not payload:
        raise TranscriptDownloadError(
            "yt-dlp returned empty transcript metadata output"
        )

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise TranscriptDownloadError(
            "yt-dlp returned invalid transcript metadata JSON"
        ) from exc

    if not isinstance(data, dict):
        raise TranscriptDownloadError(
            "yt-dlp transcript metadata response was not an object"
        )
    return data


def _select_subtitle_track(metadata: dict[str, Any]) -> tuple[str, bool] | None:
    manual_languages = _sorted_languages(metadata.get("subtitles"))
    if manual_languages:
        return manual_languages[0], False

    automatic_languages = _sorted_languages(metadata.get("automatic_captions"))
    if automatic_languages:
        return automatic_languages[0], True

    return None


def _sorted_languages(raw: object) -> list[str]:
    if not isinstance(raw, dict):
        return []

    languages = [str(key).strip() for key in raw if str(key).strip()]
    return sorted(
        languages,
        key=lambda language: _language_priority(language.lower()),
    )


def _language_priority(language: str) -> tuple[int, str]:
    if language == "en":
        return (0, language)
    if language.startswith("en-") or language.startswith("en_"):
        return (1, language)
    return (2, language)


def _download_subtitle_track(
    *,
    url: str,
    output_dir: Path,
    language_code: str,
    auto_generated: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "transcript.%(ext)s")

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--no-playlist",
        "--sub-langs",
        language_code,
        "--sub-format",
        "vtt",
        "-o",
        output_template,
    ]
    cmd.append("--write-auto-subs" if auto_generated else "--write-subs")
    cmd.append(url)

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = format_subprocess_error(exc, "yt-dlp transcript download failed")
        raise TranscriptDownloadError(message) from exc

    matches = sorted(output_dir.glob("transcript*.vtt"))
    if not matches:
        raise TranscriptDownloadError("yt-dlp did not produce a VTT transcript file")

    preferred_matches = [
        path
        for path in matches
        if f".{language_code}." in path.name or f".{language_code}-" in path.name
    ]
    return preferred_matches[0] if preferred_matches else matches[0]


def _parse_vtt_segments(
    path: Path,
    *,
    language_code: str | None,
) -> list[TranscriptSegment]:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TranscriptParseError(f"Failed to read transcript file {path}") from exc

    blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n"))
    merged_segments: list[tuple[float, float, str]] = []

    for block in blocks:
        raw_lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not raw_lines:
            continue

        first_line = raw_lines[0].lstrip("\ufeff")
        if first_line.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue

        time_index = next(
            (index for index, line in enumerate(raw_lines) if "-->" in line),
            None,
        )
        if time_index is None or time_index == len(raw_lines) - 1:
            continue

        start_s, end_s = _parse_time_range(raw_lines[time_index])
        text = _normalize_caption_text(" ".join(raw_lines[time_index + 1 :]))
        if not text:
            continue

        if merged_segments:
            previous_start_s, previous_end_s, previous_text = merged_segments[-1]
            if text == previous_text and start_s <= previous_end_s + 0.25:
                merged_segments[-1] = (
                    previous_start_s,
                    max(previous_end_s, end_s),
                    previous_text,
                )
                continue
            if text.startswith(previous_text) and start_s <= previous_end_s + 0.25:
                merged_segments[-1] = (
                    previous_start_s,
                    max(previous_end_s, end_s),
                    text,
                )
                continue

        merged_segments.append((start_s, end_s, text))

    return [
        TranscriptSegment(
            segment_index=index,
            start_s=start_s,
            end_s=end_s,
            text=text,
            language_code=language_code,
        )
        for index, (start_s, end_s, text) in enumerate(merged_segments)
    ]


def _parse_time_range(line: str) -> tuple[float, float]:
    parts = [part.strip() for part in line.split("-->", maxsplit=1)]
    if len(parts) != 2:
        raise TranscriptParseError(f"Invalid VTT time range: {line!r}")

    start_s = _parse_timestamp(parts[0])
    end_s = _parse_timestamp(parts[1].split()[0])
    return start_s, end_s


def _parse_timestamp(value: str) -> float:
    cleaned = value.strip().replace(",", ".")
    parts = cleaned.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
    elif len(parts) == 2:
        hours = 0
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        raise TranscriptParseError(f"Invalid timestamp: {value!r}")

    return (hours * 3600) + (minutes * 60) + seconds


def _normalize_caption_text(value: str) -> str:
    without_tags = _TAG_RE.sub(" ", value)
    unescaped = html.unescape(without_tags).replace("\xa0", " ")
    return re.sub(r"\s+", " ", unescaped).strip()
