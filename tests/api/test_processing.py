from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.api import processing as processing_module
from src.db.supabase import TranscriptSegmentRecord, VideoRecord
from src.video.transcripts import TranscriptDownloadError, TranscriptSegment


def _video(
    video_id: str,
    *,
    source_type: str = "youtube",
    youtube_url: str | None = "https://www.youtube.com/watch?v=abc123xyz45",
) -> VideoRecord:
    now = datetime.now(timezone.utc).isoformat()
    return VideoRecord(
        id=video_id,
        youtube_url=youtube_url,
        status="queued",
        user_id="user_123",
        error_message=None,
        source_type=source_type,  # type: ignore[arg-type]
        source_r2_key=None,
        source_filename=None,
        created_at=now,
        updated_at=now,
    )


def test_sync_video_transcript_segments_replaces_segments(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        processing_module,
        "extract_transcript_segments",
        lambda youtube_url, output_dir: [
            TranscriptSegment(
                segment_index=0,
                start_s=12.0,
                end_s=14.0,
                text="Let me explain the experiment setup.",
                language_code="en",
            ),
            TranscriptSegment(
                segment_index=1,
                start_s=20.0,
                end_s=22.0,
                text="Now we compare the two versions.",
                language_code="en",
            ),
        ],
    )
    monkeypatch.setattr(
        processing_module,
        "replace_video_transcript_segments",
        lambda video_id, segments: captured.update(
            {"video_id": video_id, "segments": segments}
        )
        or len(segments),
    )

    processing_module._sync_video_transcript_segments(_video("video_123"), tmp_path)

    assert captured["video_id"] == "video_123"
    assert captured["segments"] == [
        TranscriptSegmentRecord(
            video_id="video_123",
            segment_index=0,
            start_s=12.0,
            end_s=14.0,
            text="Let me explain the experiment setup.",
            language_code="en",
            score=None,
        ),
        TranscriptSegmentRecord(
            video_id="video_123",
            segment_index=1,
            start_s=20.0,
            end_s=22.0,
            text="Now we compare the two versions.",
            language_code="en",
            score=None,
        ),
    ]


def test_sync_video_transcript_segments_skips_upload_sources(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        processing_module,
        "extract_transcript_segments",
        lambda youtube_url, output_dir: pytest.fail(
            "upload videos should not request transcripts"
        ),
    )

    processing_module._sync_video_transcript_segments(
        _video("video_upload", source_type="upload", youtube_url=None),
        tmp_path,
    )


def test_sync_video_transcript_segments_swallows_transcript_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []

    def _raise(youtube_url: str, output_dir: Path) -> list[TranscriptSegment]:
        raise TranscriptDownloadError("subtitles unavailable")

    monkeypatch.setattr(processing_module, "extract_transcript_segments", _raise)
    monkeypatch.setattr(
        processing_module,
        "replace_video_transcript_segments",
        lambda video_id, segments: pytest.fail(
            "should not store transcript rows on extraction failure"
        ),
    )
    monkeypatch.setattr(
        processing_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    processing_module._sync_video_transcript_segments(_video("video_123"), tmp_path)

    assert any("Transcript extraction skipped" in message for message, _ in warnings)
