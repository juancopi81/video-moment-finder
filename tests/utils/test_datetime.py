from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta

import pytest

from src.utils.datetime import parse_iso_datetime

DT_LOGGER = "src.utils.datetime"


@pytest.fixture(autouse=True)
def _enable_propagation():
    """Enable propagation so caplog can capture log output."""
    log = logging.getLogger(DT_LOGGER)
    log.propagate = True
    yield
    log.propagate = False


def test_none_returns_none() -> None:
    assert parse_iso_datetime(None) is None


def test_empty_string_returns_none() -> None:
    assert parse_iso_datetime("") is None


def test_z_suffix_parsed_as_utc() -> None:
    result = parse_iso_datetime("2024-01-15T12:30:00Z")
    assert result is not None
    assert result.tzinfo is not None
    assert result == datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)


def test_explicit_offset() -> None:
    result = parse_iso_datetime("2024-01-15T12:30:00+05:00")
    assert result is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2024, 1, 15, 7, 30, 0, tzinfo=timezone.utc)


def test_naive_datetime_treated_as_utc() -> None:
    result = parse_iso_datetime("2024-01-15T12:30:00")
    assert result is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)


def test_invalid_string_returns_none_and_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="src.utils.datetime"):
        result = parse_iso_datetime("not-a-date")
    assert result is None
    assert "Unparseable ISO datetime" in caplog.text


def test_non_utc_offset_converted() -> None:
    offset = timezone(timedelta(hours=-3))
    result = parse_iso_datetime("2024-06-01T09:00:00-03:00")
    assert result is not None
    assert result.tzinfo == timezone.utc
    assert result == datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_microseconds_preserved() -> None:
    result = parse_iso_datetime("2024-01-15T12:30:00.123456Z")
    assert result is not None
    assert result.microsecond == 123456
