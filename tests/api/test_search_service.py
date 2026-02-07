from __future__ import annotations

from src.api import search as search_module
from src.storage.qdrant import SearchResult


def test_search_video_logs_timing_and_returns_results(monkeypatch) -> None:
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

    class FakeEmbedMethod:
        def remote(self, query_text: str) -> list[float]:
            assert query_text == "find opening scene"
            return [0.1, 0.2, 0.3]

    class FakeEmbedderInstance:
        embed = FakeEmbedMethod()

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
        "get_text_embedder_class",
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

    results = search_module.search_video(
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

    timing_messages = [message for message, _ in info_calls if "Search timing" in message]
    assert timing_messages
