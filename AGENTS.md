# AGENTS.md (Repository-Local)

This file defines the preferred workflow for this repository.
It complements global guidance and takes precedence for project-specific execution.

## Canonical Harness Contract

- `AGENTS.md` is the canonical source for repository agent workflow.
- `CLAUDE.md` must remain a thin shim that points here plus small Claude-only defaults.
- Do not duplicate full workflow policy in `CLAUDE.md`.

## Operating Defaults

- Keep changes small, testable, and reversible.
- Prefer implementation over long planning once scope is clear.
- Run the narrowest checks first, then full checks before merge.
- Update docs when behavior or workflow changes.
- Before finalizing, verify documentation freshness for impacted areas and remove stale or duplicated statements.

## Development Baseline

- Package manager baseline: use `uv`, not `pip`, for environment-aware execution.
- One-command setup (migrations + deps):
  - `set -a && source .env && set +a`
  - `./scripts/setup_local.sh`
- Existing DB without migration history:
  - `./scripts/setup_local.sh --baseline-existing-db`
- Run API:
  - `uv run uvicorn src.api.app:app --reload --port 8000`
- Run worker:
  - `uv run python -m src.worker.runner`
- Full validation:
  - `./scripts/workflow/check_all.sh`

## Standard Branch and PR Workflow

1. Sync and branch from `main`:
   - `git checkout main`
   - `git pull --ff-only`
   - `git checkout -b codex/<short-topic>`
   - Worktree-safe alternative when `main` is already checked out elsewhere:
     - `git fetch origin main`
     - `git checkout -b codex/<short-topic> origin/main`
2. Implement focused changes for a single concern per PR.
3. Validate locally:
   - `./scripts/workflow/check_all.sh`
4. If performance-related, run benchmark protocol and capture before/after numbers.
5. Commit with clear message:
   - `git add -A`
   - `git commit -m "<type>(<area>): <summary>"`
6. Push and open draft PR:
   - `git push -u origin codex/<short-topic>`
   - `gh pr create --draft --base main --head codex/<short-topic> ...`
7. Move PR to ready when checks pass and benchmark evidence is recorded.
8. Merge with squash:
   - `gh pr merge <number> --squash --delete-branch`
9. Local cleanup:
   - `git checkout main && git pull --ff-only`
   - `git branch -d codex/<short-topic>`

## Automation Hybrid Publish Mode (Required)

Use this mode for repository automations: implement locally, attempt publication automatically, and fall back to local handoff only when publication fails.

Run this preflight before implementation work:

1. Local git readiness:
   - `git status -sb`
   - `git fetch origin main`
2. Branch base selection:
   - `git checkout -b codex/<short-topic> origin/main`
   - If the branch already exists, continue on the existing branch instead of creating a duplicate.
3. Validation readiness:
   - Ensure required local tooling for planned checks is installed.
4. Skip GitHub API preflight checks for automation:
   - Do not require `gh auth status`, `gh api ...`, label checks, or `gh pr list` before implementation.
   - These checks are optional diagnostics and must not block implementation.

Delivery rule:

- First attempt Outcome A publication:
  - `git push -u origin codex/<short-topic>`
  - `gh pr create --draft --base main --head codex/<short-topic> --label codex --label codex-automation ...`
- If publication succeeds, return Outcome A with PR link and writeup.
- If publication fails due auth, network, or permissions, return Outcome B:
  - Leave a ready-to-review branch plus commit(s).
  - Provide a complete PR-ready writeup.
  - Provide exact manual publication commands.

Graded preflight rule:

- If a failure blocks implementation (for example cannot branch, cannot commit, cannot run required validation), stop and report Outcome B with exact unblock commands.
- If a failure only blocks PR publication steps (for example GitHub auth/API/permissions issues), continue implementation as far as possible and return manual publication commands in Outcome B.
- Use Outcome C only when a defensible next PR cannot be chosen from repo evidence.

## Benchmark Protocol (Latency Changes)

- Use one `video_id` from DB (`videos.id` UUID), not a YouTube ID.
- Always redeploy Modal app for each mode before benchmarking:
  - `uv run modal deploy src/embedding/modal_app.py`
- For warm-container experiments, set env at deploy time, then deploy again:
  - `MODAL_TEXT_EMBED_MIN_CONTAINERS=1 uv run modal deploy src/embedding/modal_app.py`
- Use `scripts/phase3/search_latency_benchmark.py` with JSON output to compare runs.
- Base merge decisions on hot-path metrics (`run_index > 1`) and p95, not only means.

## Modal Integration Rules

- For Modal class methods, use `modal.Cls.from_name(...)` (not `Function.from_name`).
- Env changes that affect Modal scaling or containers require redeploy to take effect.
- Keep warm/min containers opt-in unless explicitly approved for production cost.

## Logging and Timing Rules

- Use `src.utils.logging.get_logger()` for logs in app code and scripts.
- Use `src.utils.logging.Timer` for duration measurements.
- Avoid ad-hoc `print()` and manual timing for operational measurements.

## Queue Reliability Defaults

- `VIDEO_JOB_MAX_ATTEMPTS=3` controls terminal failure threshold.
- `VIDEO_JOB_STALE_LOCK_TIMEOUT_S=600` controls stale `processing` lock recovery.
- `VIDEO_JOB_IDLE_BACKOFF_MAX_S=15` caps empty-queue exponential polling backoff.
- `VIDEO_JOB_DB_RETRY_BASE_DELAY_S=1` sets initial retry delay after transient DB transport errors.
- `VIDEO_JOB_DB_RETRY_MAX_DELAY_S=30` caps transient DB transport retry delay.

## Documentation Ownership

- `README.md`: public-facing overview and quick-start.
- `docs/DEPLOYMENT.md`: deployment env ownership and operational contract reference.
- `PROJECT_SPEC.md`: stable product charter (vision, user, scope, constraints, success metrics, risks, high-level architecture).
- `ROADMAP.md`: planned future work only (phases, tasks, gates).
- `STATUS.md`: execution history only (progress log, blockers, decisions, metrics).
- `docs/archive/`: historical/non-authoritative snapshots and research.

Guidelines:

- Update only the owning doc for each change type.
- Keep `README.md` user-focused; keep agent/developer process in `AGENTS.md`.
- Temporary runbooks/checklists are allowed during active testing but remove them once done.

## Path Privacy Rule

- In repository content (commits, PR descriptions, markdown docs, comments), use project-relative paths (for example `src/api/app.py`) instead of absolute local filesystem paths.
- Never include machine-specific prefixes like `/Users/...` in committed artifacts.

## Lessons Memory

- If a new reusable lesson is discovered, append a short entry to:
  - `/Users/juanpineros/.codex/CODEX_SCRATCHPAD.md`
