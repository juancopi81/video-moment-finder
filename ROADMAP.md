# Video Moment Finder - Roadmap

## How to Use

- Each phase has a **GATE** (exit criteria) - don't proceed until it passes
- Tasks marked with `||` can be done in parallel
- This file is plan-only: keep future phases/tasks/gates here
- Track completion/progress updates in [STATUS.md](./STATUS.md), not in `PROJECT_SPEC.md` or this file

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
- Frontend upload UX for authenticated users (file picker, progress, and limits)
- Signed or multipart upload flow for large files
- Storage lifecycle cleanup for uploaded source videos

---

## Phase 4: Product Launch

**GATE**: Ready for paying users

### 4.1 Authentication

- Clerk integration (Next.js)
- Protected routes
- User session in API calls

### 4.2 Payments

- **Preflight gate (before implementation)**: select the provider for Colombia-based launch and confirm activation prerequisites are met.
- **Decision (2026-02-17)**: Lemon Squeezy is the first provider path (Merchant of Record), with Paddle as fallback if onboarding or activation fails.
- **Activation prerequisite**: publish a live, non-placeholder product URL before attempting live activation (must describe the product/service and include support contact and legal pages).
- Reference: [`RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md`](./RESEARCH_PAYMENTS_COLOMBIA_GLOBAL.md)
- Lemon Squeezy integration (checkout + webhook)
- Credit purchase flow
- Credit deduction on video process
- ~~Webhook signature verification and idempotent credit grants~~ — **done** (2026-02-25)
- ~~Checkout session creation endpoint~~ — **done** (2026-03-02)
- ~~Pricing CTA wired to live checkout~~ — **done** (2026-03-02)
- ~~Enforce paid credits after free-trial cap~~ — **done** (2026-03-04)
- **Next milestone**: billing status UX (post-checkout feedback + refreshed credit balance).

### 4.3 Production Hardening

- Error monitoring (Sentry or similar)
- Rate limiting
- Input validation (URL format, video length check)
- Enable Supabase Row Level Security (RLS) on `public.videos`, `public.credits`, and `public.video_jobs`, with per-user access policies
- Resolve Supabase "Function Search Path Mutable" warnings by setting explicit `search_path` on custom DB functions (including update timestamp trigger functions)

### 4.5 Authenticated User Experience

- ~~Pricing page CTAs link to actual Lemon Squeezy checkout URLs~~ — **done** (2026-03-02)

### 4.4 Launch Checklist

- ~~Environment variables secured~~ — **done** (managed in Railway + Vercel dashboards)
- ~~Payment provider decision documented~~ — **done** (Lemon Squeezy first, Paddle fallback)
- ~~Public payment-onboarding URL is live~~ — **done** (videomomentfinder.com)
- Supabase Security Advisor errors and warnings resolved (including RLS disabled in `public` schema tables and mutable function `search_path` warnings)
- Database backups configured
- Monitoring dashboards
- Landing page copy finalized
