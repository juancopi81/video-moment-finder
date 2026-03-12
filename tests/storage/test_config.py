from __future__ import annotations

import pytest

from src.storage.config import QdrantConfig, StorageConfigError


def test_qdrant_from_env_uses_remote_mode_by_default(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")

    config = QdrantConfig.from_env()

    assert config.use_in_memory is False
    assert config.url == "https://example.qdrant.io"


def test_qdrant_in_memory_factory_sets_local_mode() -> None:
    config = QdrantConfig.in_memory(collection_name="unit_test")

    assert config.use_in_memory is True
    assert config.url is None


def test_qdrant_from_env_returns_in_memory_when_env_var_set(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_USE_IN_MEMORY", "true")
    monkeypatch.delenv("QDRANT_URL", raising=False)

    config = QdrantConfig.from_env()

    assert config.use_in_memory is True
    assert config.url is None


def test_qdrant_from_env_ignores_in_memory_when_falsy(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_USE_IN_MEMORY", "false")
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")

    config = QdrantConfig.from_env()

    assert config.use_in_memory is False
    assert config.url == "https://example.qdrant.io"


def test_qdrant_from_env_raises_without_url_when_not_in_memory(monkeypatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_USE_IN_MEMORY", raising=False)

    with pytest.raises(StorageConfigError):
        QdrantConfig.from_env()
