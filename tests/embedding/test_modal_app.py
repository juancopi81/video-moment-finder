from __future__ import annotations

import pytest

from src.embedding import modal_app


def test_modal_uv_sync_command_includes_modal_group() -> None:
    assert modal_app.MODAL_UV_SYNC_COMMAND == (
        "uv sync --frozen --group modal --compile-bytecode --python-preference=only-system"
    )


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


def test_resolve_text_embed_max_containers_default(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_TEXT_EMBED_MAX_CONTAINERS", raising=False)
    assert modal_app._resolve_text_embed_max_containers() == 1


def test_resolve_text_embed_max_containers_rejects_zero(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_TEXT_EMBED_MAX_CONTAINERS", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        modal_app._resolve_text_embed_max_containers()


def test_validate_text_embed_container_bounds_accepts_valid() -> None:
    modal_app._validate_text_embed_container_bounds(1, 2)
    modal_app._validate_text_embed_container_bounds(None, 1)


def test_validate_text_embed_container_bounds_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        modal_app._validate_text_embed_container_bounds(3, 2)
