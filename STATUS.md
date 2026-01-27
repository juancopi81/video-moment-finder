# Video Moment Finder - Status

## Current Phase
**Phase 2: End-to-End Skeleton** - In Progress

## Progress Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2026-01-19 | Setup | Create ROADMAP.md | Done | Phased plan with gates |
| 2026-01-19 | Setup | Create STATUS.md | Done | Progress tracking |
| 2026-01-19 | Setup | Update README.md & CLAUDE.md | Done | Added doc links |
| 2026-01-20 | Phase 0 | 0.1 Modal + Qwen3 Setup | Done | A10G smoke test OK. Qwen3-VL-Embedding-2B loads and embeds text+image. |
| 2026-01-20 | Phase 0 | 0.2 Video Processing Test | Done | **GATE PASSED.** Initial: 0.766s/frame (batch=1). Later optimized to 0.146s/frame (batch=8). |
| 2026-01-21 | Phase 0 | 0.3 Vector Search Validation | Done | **GATE PASSED.** Recall@5 = 90% (9/10 queries). In-memory Qdrant validated. |
| 2026-01-21 | Phase 1 | 1.1 Video Download Module | Done | Added fail-fast yt-dlp wrapper in src/video. |
| 2026-01-21 | Phase 1 | 1.2 Frame Extraction Module | Done | Added fail-fast ffmpeg wrapper with timestamps + thumbnails; refactored tests to use it. |
| 2026-01-21 | Phase 1 | 1.3 Batch Embedding Pipeline | Done | Modal-only validation passed with batch=8. |
| 2026-01-22 | Phase 1 | Code cleanup | Done | Removed dead code, DRY refactor: -36 lines. Created src/utils/subprocess.py utility. |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration | Done | Added Qdrant + R2 storage modules, pipeline orchestrator, cleanup utils. Local test passed. |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration Tests | Done | Added unit tests for Qdrant/R2/cleanup/orchestrator. pytest: 16 passed. |
| 2026-01-22 | Phase 1 | 1.4 Storage Integration (Cloud) | Done | Full Qdrant+R2 integration test passed via scripts/phase1/storage_integration_test.py |
| 2026-01-23 | Phase 1 | Phase 1 Gate Check (Cost) | Done | Extrapolated 30-min cost $0.1263 (1871s video, 0.1317s/frame). **GATE PASSED** |
| 2026-01-23 | Phase 1 | Phase 1 Gate Check (End-to-End) | Done | 1800 frames processed end-to-end (Qdrant+R2). Embed 357.36s, process 889.62s, total 1254.02s. |
| 2026-01-26 | Phase 1 | R2 Parallel Uploads | Done | 1800 thumbnails uploaded in 68.25s (26.37 thumbs/s) with 16 workers. Total pipeline 429.74s. |
| 2026-01-27 | Phase 2 | 2.2 Backend API (FastAPI) | In Progress | Added mock FastAPI app with /videos and /search endpoints (Phase 2.2 skeleton). |

## Blockers
- YouTube bot detection blocks yt-dlp from Modal IPs. Workaround: download videos locally first, then upload to Modal.

## Decisions Made
- **Batch=8 is optimal for A10G** - Tested batch sizes 1, 4, 8, 16, 32. All fit in 24GB VRAM. Batch=8 fastest (0.146s/frame), larger batches plateau (~0.15s/frame) due to GPU saturation.
- **Qwen3-VL-Embedding-2B is suitable for semantic search** - 90% Recall@5 validates the model for finding video moments via text queries.

## Metrics / Measurements

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Search quality (Recall@5) | >70% | **90%** | **GATE PASSED** - 9/10 queries found correct frame in top 5 |
| Cost per 30-min video | <$1 | **~$0.13** | **GATE PASSED** - Extrapolated from 1871s video, 0.1317s/frame, 25.7s load |
| Processing time (30-min) | <20 min | **~20.9 min** | Full pipeline: 1800 frames end-to-end (embed+upload+store) |
| Single embed latency | - | **0.1317s** | Per-frame with batch=8 (1871s video sample) |
| Model load time | - | ~25.7s | Cold load in container |
| Frame extraction | - | 10.46s/1871 frames | ffmpeg at 1 fps |
| GPU device | - | NVIDIA A10 | Modal A10G (24GB VRAM) |
| Embedding dim | - | 2048 | Qwen3-VL-Embedding-2B |
| GPU cost rate | - | $0.000463/s | A10G pricing |
| Qdrant store time | - | 0.05s | 40 vectors (in-memory) |
| Query embedding time | - | ~0.2s | Per text query |
