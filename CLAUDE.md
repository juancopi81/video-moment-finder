# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Moment Finder is a SaaS product for semantic video frame search. Users paste a YouTube URL and search for specific moments using text descriptions or reference images, powered by Qwen3-VL-Embedding-2B multimodal embeddings.

## Tech Stack

- **Frontend**: Next.js 14 + Clerk (auth) + Stripe (payments)
- **Backend**: FastAPI
- **GPU Processing**: Modal (serverless)
- **AI Model**: Qwen3-VL-Embedding-2B for frame/query embeddings
- **Vector DB**: Qdrant Cloud
- **Database**: Supabase (PostgreSQL)
- **Storage**: Cloudflare R2 (thumbnails)
- **Video Tools**: yt-dlp + ffmpeg

## Architecture

```
Next.js → FastAPI → Modal (GPU)
              ↓          ↓
          Supabase    Qdrant
              ↓
        Cloudflare R2
```

**Processing Pipeline** (runs on Modal GPU):
1. Download video via yt-dlp
2. Extract frames at 1 fps with ffmpeg
3. Embed frames with Qwen3-VL-Embedding-2B
4. Store embeddings in Qdrant with timestamp metadata
5. Upload thumbnails to R2

**Search Flow**:
1. Embed query (text or image) with Qwen3-VL-Embedding-2B
2. Vector search in Qdrant filtered by video_id
3. Return top 5 results with timestamps and thumbnails

## Development Commands

```bash
# Run the main entry point
python main.py

# Python version requirement
python --version  # Must be 3.11+
```

## Key Constraints

- YouTube videos only (MVP)
- Max 30-minute videos
- Credit-based pricing model

## Project Planning

- **ROADMAP.md**: Phased development plan with gates
- **STATUS.md**: Progress tracking and metrics

## Workflow

After completing significant work, update STATUS.md with:
- Progress log entry (date, phase, task, status, notes)
- Any new blockers or decisions
- Metrics if measured during the task
