# AGENTS.md (Repository-Local)

This file defines the preferred workflow for this repository.
It complements global guidance and takes precedence for project-specific execution.

## Operating Defaults

- Keep changes small, testable, and reversible.
- Prefer implementation over long planning once scope is clear.
- Run the narrowest checks first, then run full project checks before merge.
- Update docs when behavior or workflow changes.

## Standard Branch and PR Workflow

1. Sync and branch from `main`:
   - `git checkout main`
   - `git pull --ff-only`
   - `git checkout -b codex/<short-topic>`
   - Worktree-safe alternative when `main` is already checked out in another worktree:
     - `git fetch origin main`
     - `git checkout -b codex/<short-topic> origin/main`
2. Implement focused changes for a single concern per PR.
3. Validate locally:
   - `./scripts/workflow/check_all.sh`
4. If performance-related, run benchmark protocol (see below) and capture before/after numbers.
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

## Automation PR Preflight (Required)

Run this preflight before implementation work for automation-driven PRs:

1. Authentication and API reachability:
   - `gh auth status`
   - `gh api user --jq '.login'`
2. Repository permissions:
   - `gh api repos/juancopi81/video-moment-finder --jq '.full_name + " push=" + (.permissions.push|tostring)'`
3. Required labels exist:
   - `gh label list --search "codex" --limit 20`
   - Must include `codex` and `codex-automation`.
4. Existing PR collision check:
   - `gh pr list --head codex/<short-topic> --state all --json number,state,url,title,isDraft`
   - If a PR already exists for the same head branch, update that PR instead of creating a duplicate.

Graded preflight rule:
- If a failure blocks implementation (for example cannot branch, cannot commit, cannot run required validation), stop and report Outcome B with exact unblock commands.
- If a failure only blocks PR publication steps (for example GitHub auth/API/label/PR-create issues), continue implementation as far as possible, then report Outcome B with ready branch/commit state, PR writeup, and exact unblock commands.
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
- Env changes that affect Modal scaling/containers require redeploy to take effect.
- Keep warm/min containers opt-in unless explicitly approved for production cost.

## Logging and Timing Rules

- Use `src.utils.logging.get_logger()` for logs in app code and scripts.
- Use `src.utils.logging.Timer` for duration measurements.
- Avoid ad-hoc `print()` and manual timing for operational measurements.

## Documentation and Temporary Notes

- Temporary runbooks/checklists are allowed during active testing but remove them once done.
- Keep `README.md` user-focused; keep agent/developer process in `AGENTS.md`/`CLAUDE.md`.

## Lessons Memory

- If a new reusable lesson is discovered, append a short entry to:
  - `/Users/juanpineros/.codex/CODEX_SCRATCHPAD.md`
