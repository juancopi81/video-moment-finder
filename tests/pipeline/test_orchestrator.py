from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.orchestrator import PipelineError, StoragePipeline
from src.storage.config import QdrantConfig, R2Config
from src.storage.qdrant import EMBEDDING_DIM
from src.storage.r2 import UploadResult
from src.video.frames import FrameInfo


class FakeR2Store:
    def __init__(self, config: R2Config) -> None:
        self._config = config
        self.uploaded: list[tuple[int, Path]] = []

    def upload_thumbnails(self, video_id: str, thumbnails: list[tuple[int, Path]]):
        self.uploaded = thumbnails
        results = []
        for frame_index, _ in thumbnails:
            key = f"{video_id}/thumb_{frame_index:05d}.jpg"
            url = f"https://cdn.example.com/{key}"
            results.append(UploadResult(key=key, url=url))
        return results

    def delete_video_thumbnails(self, video_id: str) -> int:
        return len(self.uploaded)


def _vector(value: float) -> list[float]:
    return [value] * EMBEDDING_DIM


def _make_frames(tmp_path: Path, count: int) -> list[FrameInfo]:
    frames = []
    for idx in range(count):
        thumb_path = tmp_path / f"thumb_{idx:05d}.jpg"
        thumb_path.write_bytes(b"\xff\xd8\xff")
        frames.append(
            FrameInfo(
                index=idx,
                timestamp_s=float(idx),
                path=tmp_path / f"frame_{idx:05d}.jpg",
                thumbnail_path=thumb_path,
            )
        )
    return frames


def test_process_video_in_memory_no_r2(tmp_path: Path) -> None:
    config = QdrantConfig.in_memory(collection_name="orchestrator_test")
    pipeline = StoragePipeline(config)
    pipeline.ensure_ready()

    frames = _make_frames(tmp_path, 3)
    embeddings = [_vector(0.1), _vector(0.2), _vector(0.3)]

    result = pipeline.process_video("video_a", frames, embeddings)

    assert result.frames_processed == 3
    assert result.embeddings_stored == 3
    assert result.thumbnails_uploaded == 0


def test_process_video_with_r2_uploads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("src.pipeline.orchestrator.R2Store", FakeR2Store)

    qdrant_config = QdrantConfig.in_memory(collection_name="orchestrator_test_r2")
    r2_config = R2Config(
        endpoint_url="https://r2.example.com",
        access_key_id="access",
        secret_access_key="secret",
        bucket_name="video-thumbnails",
    )
    pipeline = StoragePipeline(qdrant_config, r2_config)
    pipeline.ensure_ready()

    frames = _make_frames(tmp_path, 2)
    embeddings = [_vector(0.1), _vector(0.2)]

    result = pipeline.process_video("video_b", frames, embeddings)

    assert result.frames_processed == 2
    assert result.thumbnails_uploaded == 2

    results = pipeline._qdrant.search(_vector(0.1), video_id="video_b", limit=1)
    assert results
    assert results[0].thumbnail_url.startswith("https://cdn.example.com/")


def test_process_video_cleanup_temp_dirs(tmp_path: Path) -> None:
    config = QdrantConfig.in_memory(collection_name="orchestrator_cleanup")
    pipeline = StoragePipeline(config)
    pipeline.ensure_ready()

    frames = _make_frames(tmp_path, 1)
    embeddings = [_vector(0.1)]

    cleanup_dir = tmp_path / "cleanup"
    cleanup_dir.mkdir()
    (cleanup_dir / "temp.txt").write_text("data")

    pipeline.process_video(
        "video_cleanup",
        frames,
        embeddings,
        cleanup_temp_dirs=[cleanup_dir],
    )

    assert not cleanup_dir.exists()


def test_process_video_mismatched_lengths_raises(tmp_path: Path) -> None:
    config = QdrantConfig.in_memory(collection_name="orchestrator_mismatch")
    pipeline = StoragePipeline(config)

    frames = _make_frames(tmp_path, 2)
    embeddings = [_vector(0.1)]

    with pytest.raises(PipelineError):
        pipeline.process_video("video_mismatch", frames, embeddings)
