from __future__ import annotations

import subprocess


def format_subprocess_error(exc: subprocess.CalledProcessError, fallback: str) -> str:
    """Extract error message from a CalledProcessError, preferring stderr."""
    stderr = exc.stderr.strip() if exc.stderr else ""
    stdout = exc.stdout.strip() if exc.stdout else ""
    return stderr or stdout or fallback
