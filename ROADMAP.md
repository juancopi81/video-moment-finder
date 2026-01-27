# Video Moment Finder - Roadmap

## How to Use

- Each phase has a **GATE** (exit criteria) - don't proceed until it passes
- Tasks marked with `||` can be done in parallel
- Track progress in STATUS.md

---

## Phase 0: Core Validation

**GATE**: Text query finds relevant frames with >70% accuracy

### 0.1 Modal + Qwen3 Setup

- Set up Modal account
- Deploy Qwen3-VL-Embedding-2B on A10G GPU
- Test embedding a single image
- Measure latency and cost per embedding

### 0.2 Video Processing Test

- Download test video with yt-dlp (pick a 5-min video)
- Extract frames at 1 fps with ffmpeg
- Embed 100 frames on Modal
- Calculate cost extrapolation for 30-min video

### 0.3 Search Quality Validation

- Test with in-memory Qdrant (no external service needed for validation)
- Embed all frames from test video
- Test 10 ground truth queries - measure Recall@5
- **Decision point**: Is search quality good enough? (Gate: >70%)

---

## Phase 1: Processing Pipeline

**GATE**: Can process a 30-min video end-to-end, cost < $1 (Phase 0 validated ~$0.14)

### 1.1 Video Download Module

- yt-dlp wrapper with error handling
- Support for different quality levels
- Temp file management

### 1.2 Frame Extraction Module

- ffmpeg wrapper for 1 fps extraction
- Thumbnail generation (smaller resolution for storage)
- Frame timestamp mapping

### 1.3 Batch Embedding Pipeline

- Modal function for batch processing (batch=8 optimal, validated in Phase 0.3)
- Progress tracking / status updates
- Error recovery for failed frames

### 1.4 Storage Integration

- Qdrant: batch upsert with metadata (video_id, timestamp)
- R2: thumbnail upload with consistent naming
- Cleanup temp files after processing

---

## Phase 2: End-to-End Skeleton

**GATE**: Can paste URL → process → search → see results (with mocks where needed)

### 2.1 Database Schema (Supabase)

- users table
- videos table (id, user_id, youtube_url, status, created_at)
- credits table (user_id, balance)

### 2.2 Backend API (FastAPI) `|| parallel`

- POST /videos - submit video for processing
- GET /videos/{id} - get video status
- POST /videos/{id}/search - search within video
- Mock responses initially

### 2.3 Frontend Shell (Next.js) `|| parallel`

- Landing page with URL input
- Processing status page
- Search results page
- Use mock data from backend

### 2.4 Connect the Pieces

- Frontend → Backend → (Mock) Processing
- Verify full flow works

---

## Phase 3: Real Implementation

**GATE**: Real data flows through entire system

### 3.1 Backend Track `|| parallel`

- Replace mock endpoints with real logic
- Connect to Supabase
- Trigger Modal processing
- Query Qdrant for search

### 3.2 Frontend Track `|| parallel`

- Real API integration
- Loading states and error handling
- Search UI refinements

### 3.3 Infra Track `|| parallel`

- Modal webhook for job completion
- Background job queue (or polling)
- R2 thumbnail serving

---

## Phase 4: Product Launch

**GATE**: Ready for paying users

### 4.1 Authentication

- Clerk integration (Next.js)
- Protected routes
- User session in API calls

### 4.2 Payments

- Stripe integration
- Credit purchase flow
- Credit deduction on video process

### 4.3 Production Hardening

- Error monitoring (Sentry or similar)
- Rate limiting
- Input validation (URL format, video length check)

### 4.4 Launch Checklist

- Environment variables secured
- Database backups configured
- Monitoring dashboards
- Landing page copy finalized
