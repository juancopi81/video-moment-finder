# Video Moment Finder - Status

## Current Phase
**Phase 0: Core Validation** - In Progress

## Progress Log

| Date | Phase | Task | Status | Notes |
|------|-------|------|--------|-------|
| 2026-01-19 | Setup | Create ROADMAP.md | Done | Phased plan with gates |
| 2026-01-19 | Setup | Create STATUS.md | Done | Progress tracking |
| 2026-01-19 | Setup | Update README.md & CLAUDE.md | Done | Added doc links |
| 2026-01-20 | Phase 0 | 0.1 Modal + Qwen3 Setup | Done | A10G smoke test OK. Qwen3-VL-Embedding-2B loads and embeds text+image. |
| 2026-01-20 | Phase 0 | 0.2 Video Processing Test | Done | **GATE PASSED.** 100 frames embedded in 76.6s (0.766s/frame). 30-min extrapolated cost: $0.66. |

## Blockers
- YouTube bot detection blocks yt-dlp from Modal IPs. Workaround: download videos locally first, then upload to Modal.

## Decisions Made
- **Sequential embedding is sufficient** - No batching optimization needed to meet <$1 cost target. 0.766s/frame is 3.5x faster than initial estimates.

## Metrics / Measurements

| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Search quality (top 3) | >70% | - | Pending 0.3 |
| Cost per 30-min video | <$1 | **$0.66** | **GATE PASSED** - Extrapolated from 100-frame test |
| Processing time (30-min) | <20 min | ~25 min | Extrapolated: 1800 frames × 0.766s + 23s model load |
| Single embed latency | - | **0.766s** | Per-frame embedding (sequential) |
| Model load time | - | ~23s | Cold load in container |
| Frame extraction | - | 1.53s/193 frames | ffmpeg at 1 fps |
| GPU device | - | NVIDIA A10 | Modal A10G (24GB VRAM) |
| Embedding dim | - | 2048 | Qwen3-VL-Embedding-2B |
| GPU cost rate | - | $0.000463/s | A10G pricing |
