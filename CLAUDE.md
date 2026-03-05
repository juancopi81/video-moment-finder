# CLAUDE.md

This file is a thin Claude-specific shim for this repository.

## Source of Truth

Use `AGENTS.md` as the canonical repository workflow and policy document.
Do not duplicate full workflow content here.

## Claude Defaults

- Prefer repository scripts and commands from `AGENTS.md`.
- Use `uv`-based execution for Python commands.
- Keep changes small, testable, and reversible.
- Preserve doc ownership boundaries (`README`, `PROJECT_SPEC`, `ROADMAP`, `STATUS`).
- Treat `docs/archive/` as historical and non-authoritative.
