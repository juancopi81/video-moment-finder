from __future__ import annotations

import pytest

from src.config import modal as modal_config


def test_get_text_embedder_class_requires_complete_modal_token_pair(monkeypatch) -> None:
    modal_config.get_text_embedder_class.cache_clear()
    monkeypatch.setenv("MODAL_TOKEN_ID", "tok_123")
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    with pytest.raises(modal_config.ModalAuthError, match="Incomplete Modal credentials"):
        modal_config.get_text_embedder_class()

    modal_config.get_text_embedder_class.cache_clear()


def test_raise_modal_auth_error_wraps_known_modal_auth_failure() -> None:
    with pytest.raises(modal_config.ModalAuthError, match="MODAL_TOKEN_ID"):
        modal_config.raise_modal_auth_error(
            RuntimeError("Token missing. Could not authenticate client."),
            context="embedding video frames",
        )
