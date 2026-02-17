# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Moment Finder is a SaaS product for semantic video frame search. Users paste a YouTube URL and search for specific moments using text descriptions or reference images, powered by Qwen3-VL-Embedding-2B multimodal embeddings.

## Tech Stack

- **Frontend**: Next.js 16 + Clerk (auth) + Stripe (payments)
- **Backend**: FastAPI
- **Queue Worker**: Supabase-backed durable queue + Python worker
- **GPU Processing**: Modal (serverless)
- **AI Model**: Qwen3-VL-Embedding-2B for frame/query embeddings
- **Vector DB**: Qdrant Cloud
- **Database**: Supabase (PostgreSQL)
- **Storage**: Cloudflare R2 (thumbnails)
- **Video Tools**: yt-dlp + ffmpeg

## Architecture

```
Next.js → FastAPI → Supabase job queue → Worker → Modal (GPU)
              ↓                            ↓        ↓
          Supabase                      Qdrant     Cloudflare R2
```

**Processing Pipeline** (runs on Modal GPU):
1. Download video via yt-dlp
2. Extract frames at 1 fps with ffmpeg
3. Embed frames with Qwen3-VL-Embedding-2B
4. Store embeddings in Qdrant with timestamp metadata
5. Upload thumbnails to R2

**Search Flow**:
1. Embed text query with Qwen3-VL-Embedding-2B
2. Vector search in Qdrant filtered by video_id
3. Return top 5 results with timestamps and thumbnails

## Development Commands

```bash
# Package manager: uv (not pip)
uv add <package>          # Add dependencies
uv run <command>          # Run commands in the virtual environment

# One-command local setup (migrations + deps)
set -a && source .env && set +a
./scripts/setup_local.sh

# Run API
uv run uvicorn src.api.app:app --reload --port 8000

# Run worker (required for queued processing)
uv run python -m src.worker.runner
# Optional reliability tuning
uv run python -m src.worker.runner --max-attempts 3 --stale-lock-timeout 600

# Python version requirement
python --version  # Must be 3.11+

# Modal CLI (after adding modal)
uv run modal setup        # Authenticate with Modal
uv run modal run --help   # Test Modal CLI
```

## Key Constraints

- YouTube videos only (MVP)
- Max 30-minute videos
- Credit-based pricing model

## Project Planning

- **PROJECT_SPEC.md**: Stable product charter only (vision, user, scope, constraints, success metrics, risks, high-level architecture)
- **ROADMAP.md**: Planned future work only (phases, tasks, gates)
- **STATUS.md**: Execution history only (progress log, blockers, decisions, metrics)
- **RESEARCH_*.md**: Time-bounded research snapshots
- Keep these boundaries strict to avoid duplicate or stale sources of truth

## Workflow

After completing significant work, update STATUS.md with:
- Progress log entry (date, phase, task, status, notes)
- Any new blockers or decisions
- Metrics if measured during the task
- Do not add progress logs or milestone checklists to `PROJECT_SPEC.md`
- Update `ROADMAP.md` only when planned future work or gates change

## Logging and Timing Conventions

- Use `src.utils.logging.get_logger()` for runtime logs in app code and scripts.
- Use `src.utils.logging.Timer` for duration measurements (stage timing, latency timing).
- Prefer structured logger output over `print()` for operational/debug information.
- Avoid ad-hoc `time.perf_counter()` timing in app code unless `Timer` cannot cover the use case.

## Queue Reliability Defaults

- `VIDEO_JOB_MAX_ATTEMPTS=3` controls terminal failure threshold.
- `VIDEO_JOB_STALE_LOCK_TIMEOUT_S=600` controls stale `processing` lock recovery.
