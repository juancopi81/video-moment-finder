from __future__ import annotations

import io

import pytest
from PIL import Image

from src.api import search as search_module
from src.config.modal import ModalAuthError
from src.db.supabase import TranscriptSegmentRecord
from src.storage.qdrant import SearchResult


def _png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (1, 1), color="white").save(buffer, format="PNG")
    return buffer.getvalue()


class FakeTextEmbedMethod:
    def __init__(self, vector: list[float] | Exception) -> None:
        self._vector = vector

    def remote(self, query_text: str) -> list[float]:
        if isinstance(self._vector, Exception):
            raise self._vector
        return self._vector


class FakeImageEmbedMethod:
    def __init__(self, vector: list[float] | Exception, expected_image: bytes) -> None:
        self._vector = vector
        self._expected_image = expected_image

    def remote(self, image_bytes: bytes) -> list[float]:
        assert image_bytes == self._expected_image
        if isinstance(self._vector, Exception):
            raise self._vector
        return self._vector


class FakeEmbedderInstance:
    def __init__(self, *, text_vector: list[float] | Exception | None = None, image_vector=None):
        if text_vector is not None:
            self.embed_text = FakeTextEmbedMethod(text_vector)
        if image_vector is not None:
            vector, expected_image = image_vector
            self.embed_image = FakeImageEmbedMethod(vector, expected_image)


class FakeEmbedderClass:
    def __init__(self, instance: FakeEmbedderInstance) -> None:
        self._instance = instance

    def __call__(self) -> FakeEmbedderInstance:
        return self._instance


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

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            captured_search_args[source or "all"] = {
                "query_vector": query_vector,
                "video_id": video_id,
                "limit": limit,
            }
            if source == "transcript":
                return []
            return expected_results

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
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
        "transcript": {
            "query_vector": [0.1, 0.2, 0.3],
            "video_id": "video_123",
            "limit": 7,
        },
        "visual": {
            "query_vector": [0.1, 0.2, 0.3],
            "video_id": "video_123",
            "limit": 7,
        },
    }
    assert any("Search timing" in message for message, _ in info_calls)


def test_search_video_by_text_limit_applies_per_result_source(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            captured[f"{source}_limit"] = limit
            if source == "transcript":
                return [
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=2.0,
                        thumbnail_url=None,
                        score=0.8,
                        source="transcript",
                        transcript_text="one transcript hit",
                    ),
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=6.0,
                        thumbnail_url=None,
                        score=0.7,
                        source="transcript",
                        transcript_text="two transcript hit",
                    ),
                ]
            return [
                SearchResult(
                    video_id=video_id,
                    frame_index=0,
                    timestamp_s=10.0,
                    thumbnail_url="https://cdn.example.com/one.jpg",
                    score=0.9,
                ),
                SearchResult(
                    video_id=video_id,
                    frame_index=1,
                    timestamp_s=11.0,
                    thumbnail_url="https://cdn.example.com/two.jpg",
                    score=0.85,
                ),
            ]

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="where is the cat",
        limit=2,
    )

    assert captured == {"transcript_limit": 2, "visual_limit": 2}
    assert len(results) == 4
    assert [result.source for result in results] == [
        "visual",
        "visual",
        "transcript",
        "transcript",
    ]


def test_search_video_by_text_orders_spoken_queries_with_transcript_first(
    monkeypatch,
) -> None:
    captured_search_args: dict[str, object] = {}

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            captured_search_args[source or "all"] = query_vector
            if source == "transcript":
                return [
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=42.5,
                        thumbnail_url=None,
                        score=0.84,
                        source="transcript",
                        transcript_text="This is where the host explains sponsorship.",
                    )
                ]
            return [
                SearchResult(
                    video_id=video_id,
                    frame_index=12,
                    timestamp_s=90.0,
                    thumbnail_url="https://cdn.example.com/frame.jpg",
                    score=0.77,
                )
            ]

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="where did they mention sponsorship",
        limit=5,
    )

    assert results == [
        SearchResult(
            video_id="video_123",
            frame_index=-1,
            timestamp_s=42.5,
            thumbnail_url=None,
            score=0.84,
            source="transcript",
            transcript_text="This is where the host explains sponsorship.",
        ),
        SearchResult(
            video_id="video_123",
            frame_index=12,
            timestamp_s=90.0,
            thumbnail_url="https://cdn.example.com/frame.jpg",
            score=0.77,
        ),
    ]
    assert captured_search_args == {
        "transcript": [0.1, 0.2, 0.3],
        "visual": [0.1, 0.2, 0.3],
    }


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

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            captured_search_args["query_vector"] = query_vector
            captured_search_args["video_id"] = video_id
            captured_search_args["limit"] = limit
            captured_search_args["source"] = source
            return expected_results

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(
            FakeEmbedderInstance(image_vector=([0.9, 0.8, 0.7], png_bytes))
        ),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
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
        "source": "visual",
    }


def test_search_video_by_image_rejects_invalid_image_without_modal_call(
    monkeypatch,
) -> None:
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


def test_search_video_by_text_orders_visual_queries_with_visual_first(
    monkeypatch,
) -> None:
    captured_search_args: dict[str, object] = {}

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            captured_search_args[source or "all"] = {
                "query_vector": query_vector,
                "video_id": video_id,
                "limit": limit,
            }
            if source == "transcript":
                return [
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=5.0,
                        thumbnail_url=None,
                        score=0.51,
                        source="transcript",
                        transcript_text="The narrator says the word cat once.",
                    )
                ]
            return [
                SearchResult(
                    video_id=video_id,
                    frame_index=10,
                    timestamp_s=10.0,
                    thumbnail_url="https://cdn.example.com/thumb.jpg",
                    score=0.95,
                )
            ]

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="where is the cat",
        limit=7,
    )

    assert results == [
        SearchResult(
            video_id="video_123",
            frame_index=10,
            timestamp_s=10.0,
            thumbnail_url="https://cdn.example.com/thumb.jpg",
            score=0.95,
        ),
        SearchResult(
            video_id="video_123",
            frame_index=-1,
            timestamp_s=5.0,
            thumbnail_url=None,
            score=0.51,
            source="transcript",
            transcript_text="The narrator says the word cat once.",
        ),
    ]
    assert captured_search_args == {
        "transcript": {
            "query_vector": [0.1, 0.2, 0.3],
            "video_id": "video_123",
            "limit": 7,
        },
        "visual": {
            "query_vector": [0.1, 0.2, 0.3],
            "video_id": "video_123",
            "limit": 7,
        },
    }


def test_search_video_by_text_falls_back_to_supabase_transcript_search(
    monkeypatch,
) -> None:
    captured_fallback: dict[str, object] = {}

    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            if source == "transcript":
                return []
            raise RuntimeError("visual qdrant down")

    monkeypatch.setattr(
        search_module,
        "search_video_transcript_segments",
        lambda video_id, query_text, limit=5: captured_fallback.update(
            {"video_id": video_id, "query_text": query_text, "limit": limit}
        )
        or [
            TranscriptSegmentRecord(
                video_id=video_id,
                segment_index=1,
                start_s=12.0,
                end_s=15.0,
                text="Here the host explains the benchmark.",
                language_code="en",
                score=0.8,
            )
        ],
    )
    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="where does he explain the benchmark",
        limit=5,
    )

    assert results == [
        SearchResult(
            video_id="video_123",
            frame_index=-1,
            timestamp_s=12.0,
            thumbnail_url=None,
            score=0.8,
            source="transcript",
            transcript_text="Here the host explains the benchmark.",
        )
    ]
    assert captured_fallback == {
        "video_id": "video_123",
        "query_text": "explain benchmark",
        "limit": 5,
    }


def test_search_video_by_text_returns_transcript_results_when_visual_search_fails(
    monkeypatch,
) -> None:
    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            if source == "transcript":
                return [
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=12.0,
                        thumbnail_url=None,
                        score=0.8,
                        source="transcript",
                        transcript_text="Here the host explains the benchmark.",
                    )
                ]
            raise RuntimeError("qdrant down")

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="where does he explain the benchmark",
        limit=5,
    )

    assert results == [
        SearchResult(
            video_id="video_123",
            frame_index=-1,
            timestamp_s=12.0,
            thumbnail_url=None,
            score=0.8,
            source="transcript",
            transcript_text="Here the host explains the benchmark.",
        )
    ]


def test_search_video_by_text_intent_ordering_uses_token_matches(monkeypatch) -> None:
    class FakeQdrantStore:
        def __init__(self, config: object) -> None:
            self._config = config

        def search(
            self,
            *,
            query_vector: list[float],
            video_id: str,
            limit: int,
            source: str | None = None,
        ) -> list[SearchResult]:
            if source == "transcript":
                return [
                    SearchResult(
                        video_id=video_id,
                        frame_index=-1,
                        timestamp_s=3.0,
                        thumbnail_url=None,
                        score=0.8,
                        source="transcript",
                        transcript_text="spoken hit",
                    )
                ]
            return [
                SearchResult(
                    video_id=video_id,
                    frame_index=0,
                    timestamp_s=10.0,
                    thumbnail_url="https://cdn.example.com/one.jpg",
                    score=0.9,
                )
            ]

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(FakeEmbedderInstance(text_vector=[0.1, 0.2, 0.3])),
    )
    monkeypatch.setattr(search_module.QdrantConfig, "from_env", lambda: object())
    monkeypatch.setattr(search_module, "QdrantStore", FakeQdrantStore)

    results = search_module.search_video_by_text(
        video_id="video_123",
        query_text="intelligence overview",
        limit=2,
    )

    assert [result.source for result in results] == ["visual", "transcript"]


def test_search_video_by_text_raises_modal_auth_error_when_token_missing(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        search_module,
        "search_video_transcript_segments",
        lambda video_id, query_text, limit=5: [],
    )
    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(
            FakeEmbedderInstance(
                text_vector=RuntimeError("Token missing. Could not authenticate client.")
            )
        ),
    )

    with pytest.raises(ModalAuthError, match="MODAL_TOKEN_ID"):
        search_module.search_video_by_text(
            video_id="video_123",
            query_text="find opening scene",
            limit=7,
        )


def test_search_video_by_image_raises_modal_auth_error_when_token_missing(
    monkeypatch,
) -> None:
    png_bytes = _png_bytes()

    monkeypatch.setattr(
        search_module,
        "get_query_embedder_class",
        lambda: FakeEmbedderClass(
            FakeEmbedderInstance(
                image_vector=(
                    RuntimeError("Token missing. Could not authenticate client."),
                    png_bytes,
                )
            )
        ),
    )

    with pytest.raises(ModalAuthError, match="MODAL_TOKEN_ID"):
        search_module.search_video_by_image(
            video_id="video_123",
            query_image_bytes=png_bytes,
            limit=7,
        )
