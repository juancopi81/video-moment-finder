from __future__ import annotations

from src.storage.config import QdrantConfig


def test_qdrant_from_env_uses_remote_mode_by_default(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "https://example.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")

    config = QdrantConfig.from_env()

    assert config.use_in_memory is False
    assert config.url == "https://example.qdrant.io"
    assert config.api_key == "test-key"


def test_qdrant_from_env_normalizes_blank_api_key_to_none(monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_API_KEY", "   ")

    config = QdrantConfig.from_env()

    assert config.api_key is None


def test_qdrant_in_memory_factory_sets_local_mode() -> None:
    config = QdrantConfig.in_memory(collection_name="unit_test")

    assert config.use_in_memory is True
    assert config.url is None
