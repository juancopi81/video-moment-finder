from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.api import processing as processing_module
from src.db.supabase import TranscriptSegmentRecord, VideoRecord
from src.pipeline.orchestrator import ProcessingResult
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


class _ImmediateFuture:
    def __init__(self, fn, *args, **kwargs) -> None:
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def result(self):
        return self._fn(*self._args, **self._kwargs)


class _RecordingExecutor:
    def __init__(self, *, events: list[str], max_workers: int) -> None:
        self._events = events
        self._max_workers = max_workers

    def __enter__(self) -> _RecordingExecutor:
        self._events.append(f"enter:{self._max_workers}")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._events.append("exit")
        return False

    def submit(self, fn, *args, **kwargs) -> _ImmediateFuture:
        self._events.append(f"submit:{fn.__name__}")
        return _ImmediateFuture(fn, *args, **kwargs)


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


def test_index_transcript_segments_embeds_and_stores(monkeypatch) -> None:
    captured: dict[str, object] = {}
    segments = [
        TranscriptSegment(
            segment_index=0,
            start_s=1.0,
            end_s=2.0,
            text="hello world",
            language_code="en",
        ),
        TranscriptSegment(
            segment_index=1,
            start_s=3.0,
            end_s=4.0,
            text="second segment",
            language_code="en",
        ),
    ]

    class FakeEmbedFunction:
        def remote(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
            captured["texts"] = texts
            captured["batch_size"] = batch_size
            return [[0.1, 0.2], [0.3, 0.4]]

    class FakePipeline:
        def store_transcripts(self, video_id: str, passed_segments, embeddings):
            captured["video_id"] = video_id
            captured["segments"] = passed_segments
            captured["embeddings"] = embeddings
            return len(passed_segments)

    monkeypatch.setattr(
        processing_module,
        "get_embedding_modal_function",
        lambda function_name: FakeEmbedFunction(),
    )

    processing_module._index_transcript_segments(
        video_id="video_123",
        transcript_segments=segments,
        pipeline=FakePipeline(),  # type: ignore[arg-type]
    )

    assert captured == {
        "texts": ["hello world", "second segment"],
        "batch_size": 32,
        "video_id": "video_123",
        "segments": segments,
        "embeddings": [[0.1, 0.2], [0.3, 0.4]],
    }


def test_index_transcript_segments_skips_storage_errors(monkeypatch) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []
    segments = [
        TranscriptSegment(
            segment_index=0,
            start_s=1.0,
            end_s=2.0,
            text="hello world",
            language_code="en",
        )
    ]

    class FakeEmbedFunction:
        def remote(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
            return [[0.1, 0.2]]

    class FakePipeline:
        def store_transcripts(self, video_id: str, passed_segments, embeddings):
            raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(
        processing_module,
        "get_embedding_modal_function",
        lambda function_name: FakeEmbedFunction(),
    )
    monkeypatch.setattr(
        processing_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    processing_module._index_transcript_segments(
        video_id="video_123",
        transcript_segments=segments,
        pipeline=FakePipeline(),  # type: ignore[arg-type]
    )

    assert any("Transcript vector storage skipped" in message for message, _ in warnings)


def test_process_video_youtube_runs_branches_via_executor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    def _resolve_source(video: VideoRecord, video_dir: Path) -> tuple[Path, None]:
        events.append("resolve_source")
        return video_path, None

    def _validate_duration(video: VideoRecord, passed_video_path: Path) -> None:
        assert passed_video_path == video_path
        events.append("validate_duration")

    def _resolve_storage(*, video_id: str, r2_config):
        assert video_id == "video_123"
        assert r2_config is None
        events.append("resolve_storage")
        return object(), None

    def _ensure_ready(*, video_id: str, qdrant_config) -> None:
        assert video_id == "video_123"
        assert qdrant_config is not None
        events.append("ensure_ready")

    def _visual_branch(**kwargs) -> ProcessingResult:
        events.append("visual_branch")
        return ProcessingResult(
            video_id="video_123",
            frames_processed=3,
            embeddings_stored=3,
            thumbnails_uploaded=2,
        )

    def _transcript_branch(**kwargs) -> None:
        events.append("transcript_branch")

    monkeypatch.setattr(processing_module, "_resolve_video_source", _resolve_source)
    monkeypatch.setattr(processing_module, "_validate_video_duration", _validate_duration)
    monkeypatch.setattr(processing_module, "_resolve_processing_storage", _resolve_storage)
    monkeypatch.setattr(processing_module, "_ensure_storage_ready", _ensure_ready)
    monkeypatch.setattr(processing_module, "_process_visual_branch", _visual_branch)
    monkeypatch.setattr(processing_module, "_process_transcript_branch", _transcript_branch)
    monkeypatch.setattr(
        processing_module,
        "ThreadPoolExecutor",
        lambda max_workers: _RecordingExecutor(events=events, max_workers=max_workers),
    )

    result = processing_module.process_video(_video("video_123"))

    assert result == ProcessingResult(
        video_id="video_123",
        frames_processed=3,
        embeddings_stored=3,
        thumbnails_uploaded=2,
    )
    assert events.index("ensure_ready") < events.index("submit:_visual_branch")
    assert events.index("ensure_ready") < events.index("submit:_transcript_branch")
    assert "enter:2" in events
    assert "visual_branch" in events
    assert "transcript_branch" in events


def test_process_video_upload_skips_executor_and_transcript_branch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "upload.mp4"
    video_path.write_bytes(b"video")
    called: dict[str, int] = {"visual": 0}

    monkeypatch.setattr(
        processing_module,
        "_resolve_video_source",
        lambda video, video_dir: (video_path, object()),
    )
    monkeypatch.setattr(
        processing_module,
        "_validate_video_duration",
        lambda video, passed_video_path: None,
    )
    monkeypatch.setattr(
        processing_module,
        "_resolve_processing_storage",
        lambda *, video_id, r2_config: (object(), r2_config),
    )
    monkeypatch.setattr(
        processing_module,
        "_ensure_storage_ready",
        lambda *, video_id, qdrant_config: None,
    )
    monkeypatch.setattr(
        processing_module,
        "_process_visual_branch",
        lambda **kwargs: called.__setitem__("visual", called["visual"] + 1)
        or ProcessingResult(
            video_id="video_upload",
            frames_processed=1,
            embeddings_stored=1,
            thumbnails_uploaded=1,
        ),
    )
    monkeypatch.setattr(
        processing_module,
        "_process_transcript_branch",
        lambda **kwargs: pytest.fail("upload videos should not run transcript branch"),
    )
    monkeypatch.setattr(
        processing_module,
        "ThreadPoolExecutor",
        lambda max_workers: pytest.fail("upload videos should not use the executor"),
    )

    result = processing_module.process_video(
        _video("video_upload", source_type="upload", youtube_url=None)
    )

    assert called["visual"] == 1
    assert result == ProcessingResult(
        video_id="video_upload",
        frames_processed=1,
        embeddings_stored=1,
        thumbnails_uploaded=1,
    )


def test_process_video_transcript_branch_failure_is_non_fatal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    warnings: list[tuple[str, tuple[object, ...]]] = []
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        processing_module,
        "_resolve_video_source",
        lambda video, video_dir: (video_path, None),
    )
    monkeypatch.setattr(
        processing_module,
        "_validate_video_duration",
        lambda video, passed_video_path: None,
    )
    monkeypatch.setattr(
        processing_module,
        "_resolve_processing_storage",
        lambda *, video_id, r2_config: (object(), None),
    )
    monkeypatch.setattr(
        processing_module,
        "_ensure_storage_ready",
        lambda *, video_id, qdrant_config: None,
    )
    monkeypatch.setattr(
        processing_module,
        "_process_visual_branch",
        lambda **kwargs: ProcessingResult(
            video_id="video_123",
            frames_processed=2,
            embeddings_stored=2,
            thumbnails_uploaded=0,
        ),
    )

    def _raise_transcript(**kwargs) -> None:
        raise RuntimeError("transcript boom")

    monkeypatch.setattr(
        processing_module,
        "_process_transcript_branch",
        _raise_transcript,
    )
    monkeypatch.setattr(
        processing_module,
        "ThreadPoolExecutor",
        lambda max_workers: _RecordingExecutor(events=[], max_workers=max_workers),
    )
    monkeypatch.setattr(
        processing_module.logger,
        "warning",
        lambda message, *args: warnings.append((message, args)),
    )

    result = processing_module.process_video(_video("video_123"))

    assert result.video_id == "video_123"
    assert any("Transcript branch failed" in message for message, _ in warnings)


def test_process_video_visual_branch_failure_remains_fatal(
    monkeypatch,
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    monkeypatch.setattr(
        processing_module,
        "_resolve_video_source",
        lambda video, video_dir: (video_path, None),
    )
    monkeypatch.setattr(
        processing_module,
        "_validate_video_duration",
        lambda video, passed_video_path: None,
    )
    monkeypatch.setattr(
        processing_module,
        "_resolve_processing_storage",
        lambda *, video_id, r2_config: (object(), None),
    )
    monkeypatch.setattr(
        processing_module,
        "_ensure_storage_ready",
        lambda *, video_id, qdrant_config: None,
    )

    def _raise_visual(**kwargs):
        raise RuntimeError("visual boom")

    monkeypatch.setattr(processing_module, "_process_visual_branch", _raise_visual)
    monkeypatch.setattr(
        processing_module,
        "_process_transcript_branch",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        processing_module,
        "ThreadPoolExecutor",
        lambda max_workers: _RecordingExecutor(events=[], max_workers=max_workers),
    )

    with pytest.raises(processing_module.VideoProcessingError, match="visual boom"):
        processing_module.process_video(_video("video_123"))
