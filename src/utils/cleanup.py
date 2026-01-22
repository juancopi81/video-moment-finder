"""Cleanup utilities for temporary files and directories."""
from __future__ import annotations

import shutil
from pathlib import Path


class CleanupError(RuntimeError):
    """Raised when cleanup operations fail."""


def cleanup_directory(path: Path, *, ignore_errors: bool = False) -> bool:
    """
    Remove a directory and all its contents.

    Returns True if directory was removed, False if it didn't exist.
    Raises CleanupError on failure unless ignore_errors is True.
    """
    if not path.exists():
        return False

    try:
        shutil.rmtree(path)
        return True
    except Exception as exc:
        if ignore_errors:
            return False
        raise CleanupError(f"Failed to remove directory {path}: {exc}") from exc


def cleanup_file(path: Path, *, ignore_errors: bool = False) -> bool:
    """
    Remove a single file.

    Returns True if file was removed, False if it didn't exist.
    Raises CleanupError on failure unless ignore_errors is True.
    """
    if not path.exists():
        return False

    try:
        path.unlink()
        return True
    except Exception as exc:
        if ignore_errors:
            return False
        raise CleanupError(f"Failed to remove file {path}: {exc}") from exc


def cleanup_paths(paths: list[Path], *, ignore_errors: bool = False) -> int:
    """
    Remove multiple paths (files or directories).

    Returns count of successfully removed paths.
    Raises CleanupError on first failure unless ignore_errors is True.
    """
    removed = 0
    for path in paths:
        try:
            if path.is_dir():
                if cleanup_directory(path, ignore_errors=ignore_errors):
                    removed += 1
            elif path.is_file():
                if cleanup_file(path, ignore_errors=ignore_errors):
                    removed += 1
        except CleanupError:
            if not ignore_errors:
                raise
    return removed
