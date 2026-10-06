#!/usr/bin/env python3
"""Build a reproducible VMF portable archive, then inspect that final ZIP."""

from __future__ import annotations

import argparse
import json
import re
import stat
import sys
import zipfile
from pathlib import Path

from validate_package import PLUGIN_NAME, SEMVER, validate_archive


def build_archive(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"Missing plugin source: {source}")
    if output == source or source in output.parents:
        raise ValueError("Archive must be outside the plugin source directory")
    members = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if any(part == "__pycache__" for part in relative.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink is not portable: {relative}")
        if path.is_file():
            members.append((relative.as_posix(), path.read_bytes()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for relative, payload in members:
            info = zipfile.ZipInfo(f"{PLUGIN_NAME}/{relative}", date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "plugins" / PLUGIN_NAME)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.source / "plugin.json").read_text(encoding="utf-8"))
    version = manifest.get("version")
    if not isinstance(version, str) or not re.fullmatch(SEMVER, version):
        sys.stderr.write("Package build failed: expected a safe semantic version\n")
        return 1
    output = args.output or root / "dist" / f"{PLUGIN_NAME}-{version}.zip"
    try:
        build_archive(args.source, output)
        result = validate_archive(output)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"Package build failed: {exc}\n")
        return 1
    encoded = json.dumps(result, indent=2) + "\n"
    report = args.report or output.with_suffix(".validation.json")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
