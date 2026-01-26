"""Shared logging utilities for the project."""
from __future__ import annotations

import logging
import os
import sys
import time
from typing import Callable


_HANDLER: logging.Handler | None = None


def _resolve_level(level_name: str) -> int:
    level = getattr(logging, level_name.upper(), None)
    if isinstance(level, int):
        return level
    return logging.INFO


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger with a consistent format."""
    global _HANDLER

    level_name = os.environ.get("LOG_LEVEL", "INFO")
    level = _resolve_level(level_name)

    if _HANDLER is None:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.setLevel(level)
        _HANDLER = handler
    else:
        _HANDLER.setLevel(level)

    logger = logging.getLogger(name or "app")
    logger.setLevel(level)
    if _HANDLER not in logger.handlers:
        logger.addHandler(_HANDLER)
    logger.propagate = False
    return logger


class Timer:
    """Context manager for timing code blocks."""

    def __init__(
        self,
        label: str,
        logger: logging.Logger,
        *,
        level: str = "info",
        log_start: bool = False,
    ) -> None:
        self._label = label
        self._logger = logger
        self._level = level
        self._log_start = log_start
        self._start: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> Timer:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _log(self, message: str) -> None:
        log_fn: Callable[[str], None] = getattr(self._logger, self._level, self._logger.info)
        log_fn(message)

    def start(self) -> Timer:
        self._start = time.perf_counter()
        if self._log_start:
            self._log(f"Started: {self._label}")
        return self

    def stop(self) -> float:
        if self._start is None:
            self.elapsed = 0.0
            return self.elapsed
        self.elapsed = time.perf_counter() - self._start
        self._log(f"Completed: {self._label} in {self.elapsed:.2f}s")
        return self.elapsed
