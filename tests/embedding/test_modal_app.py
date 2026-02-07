from __future__ import annotations

import pytest

from src.embedding import modal_app


def test_optional_non_negative_int_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_TEXT_EMBED_MIN_CONTAINERS", raising=False)
    assert modal_app._optional_non_negative_int_env("MODAL_TEXT_EMBED_MIN_CONTAINERS") is None


def test_optional_non_negative_int_env_valid(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TEXT_EMBED_MIN_CONTAINERS", "2")
    assert modal_app._optional_non_negative_int_env("MODAL_TEXT_EMBED_MIN_CONTAINERS") == 2


def test_optional_non_negative_int_env_invalid(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TEXT_EMBED_MIN_CONTAINERS", "abc")
    with pytest.raises(ValueError, match="must be an integer"):
        modal_app._optional_non_negative_int_env("MODAL_TEXT_EMBED_MIN_CONTAINERS")


def test_optional_non_negative_int_env_negative(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TEXT_EMBED_MIN_CONTAINERS", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        modal_app._optional_non_negative_int_env("MODAL_TEXT_EMBED_MIN_CONTAINERS")


def test_resolve_text_embed_min_containers_prefers_new_name(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TEXT_EMBED_MIN_CONTAINERS", "3")
    monkeypatch.setenv("MODAL_TEXT_EMBED_KEEP_WARM", "1")

    assert modal_app._resolve_text_embed_min_containers() == 3


def test_get_text_embedder_caches_instance(monkeypatch) -> None:
    calls = {"count": 0}

    class FakeEmbedder:
        pass

    def fake_create_embedder() -> FakeEmbedder:
        calls["count"] += 1
        return FakeEmbedder()

    monkeypatch.setattr(modal_app, "_TEXT_EMBED_MODEL", None)
    monkeypatch.setattr(modal_app, "_create_qwen_embedder", fake_create_embedder)

    first = modal_app._get_text_embedder()
    second = modal_app._get_text_embedder()

    assert first is second
    assert calls["count"] == 1
