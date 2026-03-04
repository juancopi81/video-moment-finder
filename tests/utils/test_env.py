from __future__ import annotations

import logging

import pytest

from src.utils.env import get_env_float, get_env_int

ENV_LOGGER = "src.utils.env"


@pytest.fixture(autouse=True)
def _enable_propagation():
    """Enable propagation so caplog can capture log output."""
    log = logging.getLogger(ENV_LOGGER)
    log.propagate = True
    yield
    log.propagate = False


class TestGetEnvInt:
    def test_missing_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_INT", raising=False)
        assert get_env_int("TEST_INT", 42) == 42

    def test_empty_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "  ")
        assert get_env_int("TEST_INT", 42) == 42

    def test_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "10")
        assert get_env_int("TEST_INT", 42) == 10

    def test_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "  7  ")
        assert get_env_int("TEST_INT", 42) == 7

    def test_invalid_string_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_INT", "abc")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_int("TEST_INT", 42) == 42
        assert "Invalid TEST_INT" in caplog.text

    def test_non_positive_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_INT", "-5")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_int("TEST_INT", 42) == 42
        assert "must be positive" in caplog.text

    def test_zero_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_INT", "0")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_int("TEST_INT", 42) == 42

    def test_strict_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "abc")
        with pytest.raises(ValueError, match="not a valid integer"):
            get_env_int("TEST_INT", 42, strict=True)

    def test_strict_non_positive_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_INT", "-1")
        with pytest.raises(ValueError, match="must be > 0"):
            get_env_int("TEST_INT", 42, strict=True)


class TestGetEnvFloat:
    def test_missing_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("TEST_FLOAT", raising=False)
        assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_empty_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "")
        assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_valid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "3.14")
        assert get_env_float("TEST_FLOAT", 1.5) == pytest.approx(3.14)

    def test_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "  2.5  ")
        assert get_env_float("TEST_FLOAT", 1.5) == pytest.approx(2.5)

    def test_invalid_string_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_FLOAT", "xyz")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_float("TEST_FLOAT", 1.5) == 1.5
        assert "Invalid TEST_FLOAT" in caplog.text

    def test_nan_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_FLOAT", "nan")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_float("TEST_FLOAT", 1.5) == 1.5
        assert "must be positive and finite" in caplog.text

    def test_inf_returns_default(self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
        monkeypatch.setenv("TEST_FLOAT", "inf")
        with caplog.at_level(logging.WARNING, logger=ENV_LOGGER):
            assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_negative_inf_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "-inf")
        assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_non_positive_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "-2.0")
        assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_zero_returns_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "0")
        assert get_env_float("TEST_FLOAT", 1.5) == 1.5

    def test_strict_nan_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "nan")
        with pytest.raises(ValueError, match="positive and finite"):
            get_env_float("TEST_FLOAT", 1.5, strict=True)

    def test_strict_inf_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "inf")
        with pytest.raises(ValueError, match="positive and finite"):
            get_env_float("TEST_FLOAT", 1.5, strict=True)

    def test_strict_invalid_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_FLOAT", "abc")
        with pytest.raises(ValueError, match="not a valid float"):
            get_env_float("TEST_FLOAT", 1.5, strict=True)
