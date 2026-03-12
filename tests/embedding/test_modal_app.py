from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.embedding import modal_app


def test_modal_uv_sync_command_includes_modal_group() -> None:
    assert modal_app.MODAL_UV_SYNC_COMMAND == (
        "uv sync --frozen --group modal --compile-bytecode --python-preference=only-system"
    )


def test_optional_non_negative_int_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", raising=False)
    assert modal_app._optional_non_negative_int_env("MODAL_QUERY_EMBED_MIN_CONTAINERS") is None


def test_optional_non_negative_int_env_valid(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", "2")
    assert modal_app._optional_non_negative_int_env("MODAL_QUERY_EMBED_MIN_CONTAINERS") == 2


def test_optional_non_negative_int_env_invalid(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", "abc")
    with pytest.raises(ValueError, match="must be an integer"):
        modal_app._optional_non_negative_int_env("MODAL_QUERY_EMBED_MIN_CONTAINERS")


def test_optional_non_negative_int_env_negative(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", "-1")
    with pytest.raises(ValueError, match="must be >= 0"):
        modal_app._optional_non_negative_int_env("MODAL_QUERY_EMBED_MIN_CONTAINERS")


def test_resolve_query_embed_min_containers_prefers_new_name(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", "3")

    assert modal_app._resolve_query_embed_min_containers() == 3


def test_resolve_query_embed_min_containers_default(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_QUERY_EMBED_MIN_CONTAINERS", raising=False)

    assert modal_app._resolve_query_embed_min_containers() is None


def test_resolve_query_embed_max_containers_default(monkeypatch) -> None:
    monkeypatch.delenv("MODAL_QUERY_EMBED_MAX_CONTAINERS", raising=False)
    assert modal_app._resolve_query_embed_max_containers() == 1


def test_resolve_query_embed_max_containers_rejects_zero(monkeypatch) -> None:
    monkeypatch.setenv("MODAL_QUERY_EMBED_MAX_CONTAINERS", "0")
    with pytest.raises(ValueError, match="must be >= 1"):
        modal_app._resolve_query_embed_max_containers()


def test_validate_query_embed_container_bounds_accepts_valid() -> None:
    modal_app._validate_query_embed_container_bounds(1, 2)
    modal_app._validate_query_embed_container_bounds(None, 1)


def test_validate_query_embed_container_bounds_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        modal_app._validate_query_embed_container_bounds(3, 2)


def test_normalize_transcription_segments_filters_blank_text_and_clamps_end() -> None:
    segments = [
        SimpleNamespace(start=1.25, end=2.5, text="  first segment  "),
        SimpleNamespace(start=3.0, end=2.0, text="second\nsegment"),
        SimpleNamespace(start=4.0, end=5.0, text="   "),
    ]

    assert modal_app._normalize_transcription_segments(
        segments,
        language_code="en",
    ) == [
        {
            "start_s": 1.25,
            "end_s": 2.5,
            "text": "first segment",
            "language_code": "en",
        },
        {
            "start_s": 3.0,
            "end_s": 3.0,
            "text": "second segment",
            "language_code": "en",
        },
    ]


def test_transcribe_audio_bytes_uses_whisper_pipeline(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class FakePipeline:
        def transcribe(self, audio_path: str, **kwargs):
            path = tmp_path / "copied.wav"
            path.write_bytes(Path(audio_path).read_bytes())
            captured["audio_path"] = audio_path
            captured["kwargs"] = kwargs
            return (
                iter(
                    [
                        SimpleNamespace(start=0.0, end=1.2, text=" Hello world "),
                        SimpleNamespace(start=2.0, end=3.4, text=""),
                    ]
                ),
                SimpleNamespace(language="en"),
            )

    monkeypatch.setattr(modal_app, "_create_whisper_pipeline", lambda: FakePipeline())

    payload = modal_app.transcribe_audio_bytes.local(b"audio-bytes", batch_size=4)

    assert Path(captured["audio_path"]).suffix == ".wav"
    assert captured["kwargs"] == {
        "batch_size": 4,
        "task": "transcribe",
        "vad_filter": True,
        "word_timestamps": False,
    }
    assert payload == [
        {
            "start_s": 0.0,
            "end_s": 1.2,
            "text": "Hello world",
            "language_code": "en",
        }
    ]


def test_transcribe_audio_bytes_rejects_empty_payload() -> None:
    with pytest.raises(ValueError, match="audio_bytes must be non-empty"):
        modal_app.transcribe_audio_bytes.local(b"")
