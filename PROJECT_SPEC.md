# Video Moment Finder

> One-line pitch: Paste a YouTube URL, search for any moment using text or images

## Goals

- [x] **Revenue** — build a paid product from day 1
- [x] **Learning** — explore multimodal embeddings, RAG, and video processing

**Specific goals**:

- Ship a working product and see if people pay
- Learn Qwen3-VL-Embedding for video frame search
- Build end-to-end: GPU processing, vector search, payments

## Pain / Opportunity

Content creators and researchers struggle to find specific moments in videos:

- Scrubbing through hours of footage is tedious
- YouTube search only finds videos, not moments within them
- Transcript search misses visual content ("when did I show the diagram?")
- No good tool for "find frames that look like this image"

**Opportunity**: Qwen3-VL-Embedding can embed video frames into semantic vectors. Combined with vector search, users can find moments by describing what they SEE, not just what was SAID.

## User

- **Primary**: YouTubers and video editors searching their own footage
- **Secondary**: Researchers analyzing video content
- **Tertiary**: Anyone who wants to find a specific moment in a long video

## Inputs

- **Required**: YouTube video URL
- **Query options**:
  - Text: "person holding a phone", "code editor on screen"
  - Image: upload a reference image to find similar frames

## Outputs

- **Timestamped results**: Top 5 moments with thumbnails and timestamps
- **Relevance scores**: Confidence indicator for each match
- **Click to view**: Jump directly to that moment in the video

## MVP Scope (v0)

### In Scope

- YouTube videos only (via yt-dlp)
- Max 30-minute videos
- Text search and image search
- Top 5 results with thumbnails
- Credit-based payments (Stripe)
- User accounts (Clerk)

### Out of Scope (for now)

- Other video sources (Vimeo, direct upload)
- Videos longer than 30 minutes
- Video clip export
- Team/collaboration features
- API access

## Tech Stack / Learning Goals

- **Stack**:
  - Next.js 14 + Clerk + Stripe (frontend)
  - FastAPI (backend API)
  - Modal (serverless GPU for processing)
  - Qwen3-VL-Embedding-2B (frame embeddings)
  - Qdrant Cloud (vector database)
  - Supabase (Postgres for users/videos)
  - Cloudflare R2 (thumbnail storage)
  - yt-dlp + ffmpeg (video download/processing)

- **Skills to learn**:
  - Multimodal embeddings for video frames
  - Serverless GPU with Modal
  - Vector similarity search at scale
  - Building a paid SaaS product

## Architecture

```
User Flow:
1. Paste YouTube URL → 2. Wait for processing → 3. Search with text/image → 4. Get timestamped results

Technical Flow:
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Next.js   │────▶│   FastAPI   │────▶│    Modal    │
│  (Frontend) │     │  (Backend)  │     │    (GPU)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                   │
                           ▼                   ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  Supabase   │     │   Qdrant    │
                    │ (Users/DB)  │     │  (Vectors)  │
                    └─────────────┘     └─────────────┘
```

### Processing Pipeline (Modal GPU)

```python
def process_video(youtube_url, video_id):
    # 1. Download video with yt-dlp
    # 2. Extract frames at 1 fps with ffmpeg
    # 3. Embed each frame with Qwen3-VL-Embedding-2B
    # 4. Upload embeddings to Qdrant with timestamp metadata
    # 5. Upload thumbnails to R2
    # 6. Update video status in Supabase
```

### Search Flow

```python
def search_video(video_id, query_text=None, query_image=None):
    # 1. Embed the query (text or image) with Qwen3-VL-Embedding-2B
    # 2. Search Qdrant for similar frames (filtered by video_id)
    # 3. Return top 5 results with timestamps and thumbnails
```

## Monetization

- **Pricing model**: Credit-based
- **Free tier**: 1 video (trial)
- **Starter**: $5 for 5 videos (~$1/video)
- **Pro**: $15 for 20 videos (~$0.75/video)

**Unit economics**:
- Cost per video: ~$0.50-1.00 (Modal GPU time)
- Break-even: Charge $1+ per video

## Success Metrics

| Signal              | Target       | Notes                              |
| ------------------- | ------------ | ---------------------------------- |
| Search quality      | >70%         | Relevant result in top 3          |
| Processing time     | <20 min      | For 30-min video                  |
| Conversion          | >5%          | Free trial → paid                 |
| Cost per video      | <$1          | Modal GPU + storage               |

## Risks & Unknowns

- **YouTube ToS**: Downloading videos may violate terms. Mitigation: Users process their own videos; consider direct upload later.
- **Processing time**: 30-min video = ~1800 frames = ~15-30 min processing. Mitigation: Show progress, email when ready.
- **Embedding quality**: Will frame embeddings match text queries well? Need to test with real videos.
- **GPU costs**: Modal A10G is ~$1/hour. A 30-min video might take 15-30 min = $0.25-0.50 GPU cost.

## Progress Log

| Date | Event | Result |
| ---- | ----- | ------ |
| -    | -     | -      |

## Next Milestone

**Goal**: Build and test the processing pipeline

**Tasks**:

- [ ] Set up Modal account and test Qwen3-VL-Embedding-2B on GPU
- [ ] Implement video download (yt-dlp) and frame extraction (ffmpeg)
- [ ] Test embedding 100 frames and storing in Qdrant
- [ ] Test search with text queries - does it find relevant frames?
- [ ] Calculate actual cost per video
- [ ] If search quality is good and costs are acceptable → build full product
