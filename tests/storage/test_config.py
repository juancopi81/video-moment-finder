from __future__ import annotations

import pytest

from src.storage.config import QdrantConfig, StorageConfigError


def test_qdrant_from_env_uses_remote_mode_by_default(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.delenv("QDRANT_COLLECTION_NAME", raising=False)

    config = QdrantConfig.from_env()

    assert config.use_in_memory is False
    assert config.url == "https://example.qdrant.io"
    assert config.api_key == "test-key"
    assert config.collection_name == "video_frames"


def test_qdrant_from_env_selects_isolated_staging_collection(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", " video_frames_staging ")

    assert QdrantConfig.from_env().collection_name == "video_frames_staging"


def test_qdrant_explicit_collection_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "video_frames_staging")

    assert QdrantConfig.from_env(collection_name="explicit_frames").collection_name == "explicit_frames"


@pytest.mark.parametrize("collection_name", ["", "   "])
def test_qdrant_blank_collection_does_not_fall_back_to_production(monkeypatch, collection_name) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", collection_name)

    with pytest.raises(StorageConfigError, match="must not be blank"):
        QdrantConfig.from_env()


def test_qdrant_from_env_normalizes_blank_api_key_to_none(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "   ")

    config = QdrantConfig.from_env()

    assert config.api_key is None


def test_qdrant_in_memory_factory_sets_local_mode() -> None:
    config = QdrantConfig.in_memory(collection_name="unit_test")

    assert config.use_in_memory is True
    assert config.url is None
