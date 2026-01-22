from __future__ import annotations

from pathlib import Path

from src.utils.cleanup import cleanup_directory, cleanup_file, cleanup_paths


def test_cleanup_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("data")

    removed = cleanup_file(file_path)

    assert removed is True
    assert not file_path.exists()


def test_cleanup_directory(tmp_path: Path) -> None:
    dir_path = tmp_path / "nested"
    dir_path.mkdir()
    (dir_path / "sample.txt").write_text("data")

    removed = cleanup_directory(dir_path)

    assert removed is True
    assert not dir_path.exists()


def test_cleanup_paths_mixed(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("data")

    dir_path = tmp_path / "nested"
    dir_path.mkdir()
    (dir_path / "sample.txt").write_text("data")

    removed_count = cleanup_paths([file_path, dir_path])

    assert removed_count == 2
    assert not file_path.exists()
    assert not dir_path.exists()


def test_cleanup_missing_paths_ignore(tmp_path: Path) -> None:
    file_path = tmp_path / "missing.txt"
    dir_path = tmp_path / "missing_dir"

    removed_count = cleanup_paths([file_path, dir_path], ignore_errors=True)

    assert removed_count == 0
