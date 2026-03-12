"""Video processing pipeline implementation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
from pathlib import Path
from typing import TypedDict

from src.config.modal import (
    EMBED_IMAGES_FUNCTION_NAME,
    EMBED_TEXTS_FUNCTION_NAME,
    TRANSCRIBE_AUDIO_FUNCTION_NAME,
    get_embedding_modal_function,
    raise_modal_auth_error,
)
from src.db.supabase import (
    TranscriptSegmentRecord,
    VideoRecord,
    replace_video_transcript_segments,
)
from src.pipeline.orchestrator import ProcessingResult, StoragePipeline
from src.storage.config import QdrantConfig, R2Config, StorageConfigError
from src.storage.r2 import R2Store, R2StorageError
from src.utils.logging import get_logger
from src.video.audio import AudioExtractionError, extract_audio_track
from src.video.download import download_video
from src.video.frames import extract_frames
from src.video.metadata import (
    VideoMetadataProbeError,
    max_video_duration_s,
    probe_video_duration_s,
)
from src.video.transcripts import TranscriptError, extract_transcript_segments
from src.video.transcripts import TranscriptSegment

logger = get_logger(__name__)

MAX_FRAMES = 1800  # 30 min at 1fps


class VideoProcessingError(RuntimeError):
    """Raised when the processing pipeline fails."""


class VideoValidationError(VideoProcessingError):
    """Raised when input validation fails and retries should not continue."""


class TranscriptionSegmentPayload(TypedDict):
    """Normalized ASR segment payload returned by the Modal transcription function."""

    start_s: float
    end_s: float
    text: str
    language_code: str | None


def process_video(video: VideoRecord) -> ProcessingResult:
    """
    Processing pipeline: download -> extract -> embed -> store.

    Returns:
        ProcessingResult with processed counts.

    Raises:
        VideoProcessingError: If any stage fails.
    """
    logger.info(
        "Starting processing for video_id=%s source_type=%s",
        video.id,
        video.source_type,
    )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            video_dir = temp_path / "video"
            frames_dir = temp_path / "frames"
            thumbnails_dir = temp_path / "thumbnails"

            video_path, r2_config = _resolve_video_source(video, video_dir)
            _validate_video_duration(video, video_path)
            qdrant_config, r2_config = _resolve_processing_storage(
                video_id=video.id,
                r2_config=r2_config,
            )
            _ensure_storage_ready(video_id=video.id, qdrant_config=qdrant_config)

            transcript_dir = temp_path / "transcript"
            with ThreadPoolExecutor(max_workers=2) as executor:
                visual_future = executor.submit(
                    _process_visual_branch,
                    video_id=video.id,
                    video_path=video_path,
                    frames_dir=frames_dir,
                    thumbnails_dir=thumbnails_dir,
                    qdrant_config=qdrant_config,
                    r2_config=r2_config,
                )
                transcript_future = executor.submit(
                    _process_transcript_branch,
                    video=video,
                    video_path=video_path,
                    output_dir=transcript_dir,
                    qdrant_config=qdrant_config,
                )
                try:
                    result = visual_future.result()
                finally:
                    try:
                        transcript_future.result()
                    except Exception as exc:
                        logger.warning(
                            "Spoken branch failed for video_id=%s: %s",
                            video.id,
                            exc,
                        )
            logger.info(
                "Stored %d embeddings, %d thumbnails for video_id=%s",
                result.embeddings_stored,
                result.thumbnails_uploaded,
                video.id,
            )
            logger.info("Video processing complete for video_id=%s", video.id)
            return result

    except VideoValidationError:
        raise
    except Exception as exc:
        logger.exception("Failed to process video_id=%s: %s", video.id, exc)
        raise VideoProcessingError(str(exc)) from exc


def _resolve_processing_storage(
    *,
    video_id: str,
    r2_config: R2Config | None,
) -> tuple[QdrantConfig, R2Config | None]:
    qdrant_config = QdrantConfig.from_env()
    if r2_config is not None:
        return qdrant_config, r2_config

    try:
        r2_config = R2Config.from_env()
    except StorageConfigError:
        logger.warning(
            "R2 not configured, thumbnails will not be uploaded for video_id=%s",
            video_id,
        )
        r2_config = None

    return qdrant_config, r2_config


def _ensure_storage_ready(*, video_id: str, qdrant_config: QdrantConfig) -> None:
    logger.info("Ensuring Qdrant collection is ready for video_id=%s", video_id)
    StoragePipeline(qdrant_config).ensure_ready()


def _process_visual_branch(
    *,
    video_id: str,
    video_path: Path,
    frames_dir: Path,
    thumbnails_dir: Path,
    qdrant_config: QdrantConfig,
    r2_config: R2Config | None,
) -> ProcessingResult:
    logger.info("Starting visual branch for video_id=%s", video_id)
    logger.info("Extracting frames for video_id=%s", video_id)
    frames = extract_frames(
        video_path,
        frames_dir,
        fps=1.0,
        thumbnail_dir=thumbnails_dir,
    )
    logger.info("Extracted %d frames", len(frames))

    if len(frames) > MAX_FRAMES:
        logger.warning(
            "Truncating frames from %d to %d (30-min limit)",
            len(frames),
            MAX_FRAMES,
        )
        frames = frames[:MAX_FRAMES]

    logger.info("Embedding %d frames via Modal", len(frames))
    frame_bytes = [frame.path.read_bytes() for frame in frames]
    embed_fn = get_embedding_modal_function(EMBED_IMAGES_FUNCTION_NAME)
    try:
        embeddings = embed_fn.remote(frame_bytes, batch_size=8)
    except Exception as exc:
        raise_modal_auth_error(exc, context="embedding video frames")
        raise
    logger.info("Got %d embeddings", len(embeddings))

    logger.info("Storing embeddings and thumbnails for video_id=%s", video_id)
    pipeline = StoragePipeline(qdrant_config, r2_config)
    return pipeline.process_video(video_id, frames, embeddings)


def _process_transcript_branch(
    *,
    video: VideoRecord,
    video_path: Path,
    output_dir: Path,
    qdrant_config: QdrantConfig,
) -> None:
    logger.info("Starting spoken branch for video_id=%s", video.id)
    transcript_segments = _sync_video_transcript_segments(
        video,
        output_dir,
        video_path=video_path,
    )
    if not transcript_segments:
        return

    pipeline = StoragePipeline(qdrant_config)
    _index_transcript_segments(
        video_id=video.id,
        transcript_segments=transcript_segments,
        pipeline=pipeline,
    )


def _validate_video_duration(video: VideoRecord, video_path: Path) -> None:
    if video.source_type != "upload":
        return

    # Defense in depth: upload endpoints validate duration before enqueue, but
    # worker-side validation prevents retries/processing for bypassed invalid sources.
    try:
        duration_s = probe_video_duration_s(video_path)
    except VideoMetadataProbeError as exc:
        raise VideoProcessingError(
            f"Unable to determine uploaded video duration: {exc}"
        ) from exc

    max_duration_s = max_video_duration_s()
    if duration_s > max_duration_s:
        max_minutes = max_duration_s // 60
        raise VideoValidationError(f"Uploaded video exceeds {max_minutes}-minute limit")


def _resolve_video_source(
    video: VideoRecord,
    video_dir: Path,
) -> tuple[Path, R2Config | None]:
    if video.source_type == "upload":
        if not video.source_r2_key:
            raise VideoProcessingError("Missing source_r2_key for uploaded video")
        try:
            r2_config = R2Config.from_env()
        except StorageConfigError as exc:
            raise VideoProcessingError(
                "R2 config is required for uploaded videos"
            ) from exc

        filename = video.source_filename or "upload.mp4"
        destination = video_dir / filename
        logger.info(
            "Downloading uploaded source for video_id=%s to %s",
            video.id,
            destination,
        )
        store = R2Store(r2_config)
        try:
            store.download_source_video(video.source_r2_key, destination)
        except R2StorageError as exc:
            raise VideoProcessingError(str(exc)) from exc

        return destination, r2_config

    if video.source_type != "youtube":
        raise VideoProcessingError(f"Unsupported source_type {video.source_type!r}")

    if not video.youtube_url:
        raise VideoProcessingError("Missing youtube_url for video source")

    local_video_dir = _local_video_dir()
    if local_video_dir is not None:
        logger.info(
            "Checking local video cache for video_id=%s in %s",
            video.id,
            local_video_dir,
        )
    logger.info("Downloading video for video_id=%s", video.id)
    video_path = download_video(
        video.youtube_url,
        video_dir,
        local_video_dir=local_video_dir,
    )
    if local_video_dir is not None and _is_within_dir(video_path, local_video_dir):
        logger.info("Using local video file %s", video_path)
    logger.info("Downloaded video to %s", video_path)
    return video_path, None


def _sync_video_transcript_segments(
    video: VideoRecord,
    output_dir: Path,
    *,
    video_path: Path | None = None,
) -> list[TranscriptSegment]:
    segments = _fetch_video_transcript_segments(
        video,
        output_dir,
        video_path=video_path,
    )
    if not segments:
        return []

    try:
        stored = replace_video_transcript_segments(
            video.id,
            _build_transcript_segment_records(video.id, segments),
        )
    except Exception as exc:
        logger.warning(
            "Transcript storage skipped for video_id=%s: %s",
            video.id,
            exc,
        )
        return []

    logger.info(
        "Stored %d transcript segments for video_id=%s",
        stored,
        video.id,
    )
    return segments


def _fetch_video_transcript_segments(
    video: VideoRecord,
    output_dir: Path,
    *,
    video_path: Path | None = None,
) -> list[TranscriptSegment]:
    if video.source_type == "youtube":
        if not video.youtube_url:
            return []
        try:
            return extract_transcript_segments(video.youtube_url, output_dir)
        except TranscriptError as exc:
            logger.warning(
                "Transcript extraction skipped for video_id=%s: %s",
                video.id,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning(
                "Unexpected transcript extraction failure for video_id=%s: %s",
                video.id,
                exc,
            )
            return []

    if video.source_type == "upload":
        if video_path is None:
            raise VideoProcessingError(
                f"Missing video_path for upload transcript sync: {video.id}"
            )
        try:
            return _transcribe_uploaded_video_segments(
                video_id=video.id,
                video_path=video_path,
                output_dir=output_dir,
            )
        except AudioExtractionError as exc:
            logger.warning(
                "Upload audio extraction skipped for video_id=%s: %s",
                video.id,
                exc,
            )
            return []
        except Exception as exc:
            logger.warning(
                "Upload ASR skipped for video_id=%s: %s",
                video.id,
                exc,
            )
            return []

    return []


def _transcribe_uploaded_video_segments(
    *,
    video_id: str,
    video_path: Path,
    output_dir: Path,
) -> list[TranscriptSegment]:
    logger.info("Extracting audio track for video_id=%s", video_id)
    audio_path = extract_audio_track(video_path, output_dir / "input.wav")
    logger.info("Transcribing uploaded audio via Modal for video_id=%s", video_id)
    transcribe_fn = get_embedding_modal_function(TRANSCRIBE_AUDIO_FUNCTION_NAME)
    segment_payloads: list[TranscriptionSegmentPayload] = transcribe_fn.remote(
        audio_path.read_bytes()
    )
    return _transcript_segments_from_payloads(segment_payloads)


def _transcript_segments_from_payloads(
    segment_payloads: list[TranscriptionSegmentPayload],
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for payload in segment_payloads:
        text = payload["text"].strip()
        if not text:
            continue

        start_s = float(payload["start_s"])
        end_s = float(payload["end_s"])
        language_code = payload["language_code"]
        if language_code is not None:
            language_code = language_code.strip() or None

        segments.append(
            TranscriptSegment(
                segment_index=len(segments),
                start_s=start_s,
                end_s=end_s,
                text=text,
                language_code=language_code,
            )
        )

    return segments


def _build_transcript_segment_records(
    video_id: str,
    segments: list[TranscriptSegment],
) -> list[TranscriptSegmentRecord]:
    return [
        TranscriptSegmentRecord(
            video_id=video_id,
            segment_index=segment.segment_index,
            start_s=segment.start_s,
            end_s=segment.end_s,
            text=segment.text,
            language_code=segment.language_code,
        )
        for segment in segments
    ]


def _index_transcript_segments(
    *,
    video_id: str,
    transcript_segments: list[TranscriptSegment],
    pipeline: StoragePipeline,
) -> None:
    logger.info(
        "Embedding %d transcript segments via Modal for video_id=%s",
        len(transcript_segments),
        video_id,
    )
    texts = [segment.text for segment in transcript_segments]
    embed_fn = get_embedding_modal_function(EMBED_TEXTS_FUNCTION_NAME)
    try:
        embeddings = embed_fn.remote(texts, batch_size=32)
    except Exception as exc:
        raise_modal_auth_error(exc, context="embedding transcript segments")
        logger.warning(
            "Transcript vector indexing skipped for video_id=%s: %s",
            video_id,
            exc,
        )
        return

    try:
        stored = pipeline.store_transcripts(video_id, transcript_segments, embeddings)
    except Exception as exc:
        logger.warning(
            "Transcript vector storage skipped for video_id=%s: %s",
            video_id,
            exc,
        )
        return

    logger.info(
        "Stored %d transcript vectors for video_id=%s",
        stored,
        video_id,
    )


def _local_video_dir() -> Path | None:
    raw = os.environ.get("VIDEO_LOCAL_VIDEO_DIR", "").strip()
    if not raw:
        return None
    return Path(raw)


def _is_within_dir(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True
