from __future__ import annotations

import io

import pytest
from PIL import Image

from src.api import search as search_module
from src.config.modal import ModalAuthError
from src.storage.qdrant import SearchResult


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_search_video_by_text_logs_timing_and_returns_results(monkeypatch) -> None:
    expected_results = [
        SearchResult(
            video_id="video_123",
            frame_index=10,
            timestamp_s=10.0,
            thumbnail_url="https://cdn.example.com/thumb.jpg",
            score=0.95,
        )
    ]

    info_calls: list[tuple[str, tuple[object, ...]]] = []
    captured_search_args: dict[str, object] = {}

    class FakeEmbedTextMethod:
        def remote(self, query_text: str) -> list[float]:
            assert query_text == "find opening scene"
            return [0.1, 0.2, 0.3]

    class FakeEmbedderInstance:
        embed_text = FakeEmbedTextMethod()

    class FakeEmbedderClass:
        def __call__(self) -> FakeEmbedderInstance:
            return FakeEmbedderInstance()

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
        ) -> list[SearchResult]:
            captured_search_args["query_vector"] = query_vector
            captured_search_args["video_id"] = video_id
            captured_search_args["limit"] = limit
            return expected_results

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(),
    )
    monkeypatch.setattr(
        search_module.QdrantConfig,
        "from_env",
        lambda: object(),
    )
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)
    monkeypatch.setattr(
        search_module.logger,
        "info",
        lambda message, *args: info_calls.append((message, args)),
    )

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="find opening scene",
        limit=7,
    )

    assert results == expected_results
    assert captured_search_args == {
        "query_vector": [0.1, 0.2, 0.3],
        "video_id": "video_123",
        "limit": 7,
    }
    assert any("Search timing" in message for message, _ in info_calls)


def test_search_video_by_image_logs_timing_and_returns_results(monkeypatch) -> None:
    expected_results = [
        SearchResult(
            video_id="video_123",
            frame_index=5,
            timestamp_s=42.0,
            thumbnail_url=None,
            score=0.91,
        )
    ]
    png_bytes = _png_bytes()
    captured_search_args: dict[str, object] = {}

    class FakeEmbedImageMethod:
        def remote(self, image_bytes: bytes) -> list[float]:
            assert image_bytes == png_bytes
            return [0.9, 0.8, 0.7]

    class FakeEmbedderInstance:
        embed_image = FakeEmbedImageMethod()

    class FakeEmbedderClass:
        def __call__(self) -> FakeEmbedderInstance:
            return FakeEmbedderInstance()

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
        ) -> list[SearchResult]:
            captured_search_args["query_vector"] = query_vector
            captured_search_args["video_id"] = video_id
            captured_search_args["limit"] = limit
            return expected_results

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(),
    )
    monkeypatch.setattr(
        search_module.QdrantConfig,
        "from_env",
        lambda: object(),
    )
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_image(
        video_id="video_123",
        query_image_bytes=png_bytes,
        limit=3,
    )

    assert results == expected_results
    assert captured_search_args == {
        "query_vector": [0.9, 0.8, 0.7],
        "video_id": "video_123",
        "limit": 3,
    }


def test_search_video_by_image_rejects_invalid_image_without_modal_call(monkeypatch) -> None:
    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: pytest.fail("should not call Modal for invalid image input"),
    )

    with pytest.raises(
        search_module.QueryImageValidationError,
        match="not a valid image",
    ):
        search_module.search_video_by_image(
            video_id="video_123",
            query_image_bytes=b"not-an-image",
            limit=5,
        )


def test_search_video_by_text_raises_modal_auth_error_when_token_missing(monkeypatch) -> None:
    class FakeEmbedTextMethod:
        def remote(self, query_text: str) -> list[float]:
            assert query_text == "find opening scene"
            raise RuntimeError("Token missing. Could not authenticate client.")

    class FakeEmbedderInstance:
        embed_text = FakeEmbedTextMethod()

    class FakeEmbedderClass:
        def __call__(self) -> FakeEmbedderInstance:
            return FakeEmbedderInstance()

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(),
    )

    with pytest.raises(ModalAuthError, match="MODAL_TOKEN_ID"):
        search_module.search_video_by_text(
            video_id="video_123",
            query_text="find opening scene",
            limit=7,
        )


def test_search_video_by_image_raises_modal_auth_error_when_token_missing(monkeypatch) -> None:
    png_bytes = _png_bytes()

    class FakeEmbedImageMethod:
        def remote(self, image_bytes: bytes) -> list[float]:
            assert image_bytes == png_bytes
            raise RuntimeError("Token missing. Could not authenticate client.")

    class FakeEmbedderInstance:
        embed_image = FakeEmbedImageMethod()

    class FakeEmbedderClass:
        def __call__(self) -> FakeEmbedderInstance:
            return FakeEmbedderInstance()

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(),
    )

    with pytest.raises(ModalAuthError, match="MODAL_TOKEN_ID"):
        search_module.search_video_by_image(
            video_id="video_123",
            query_image_bytes=png_bytes,
            limit=7,
        )
