from __future__ import annotations

from src.storage.config import QdrantConfig


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
