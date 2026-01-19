# Research: Video Moment Finder

## Concept

A web application that allows users to search within YouTube videos using text descriptions or reference images. Instead of scrubbing through footage manually, users can describe what they're looking for ("person writing on whiteboard") or upload a similar image, and the system returns timestamped moments that match.

## Why This is Now Possible

**Qwen3-VL-Embedding** (released January 2025) enables this:

1. **Multimodal embeddings**: Same model embeds both images AND text into a shared vector space
2. **Video frame support**: Can process up to 64 video frames natively
3. **Open source**: Run locally or on serverless GPU (Modal) - no expensive API calls
4. **SOTA quality**: Ranks #1 on MMEB-V2 benchmark for multimodal retrieval

Previously, this would require:
- Expensive OpenAI/Google APIs for each frame
- Separate models for text and image embedding
- Lower quality results

## Technical Approach

### Two-Phase Architecture

**Phase 1: Ingestion (async, GPU)**
```
YouTube URL → Download → Extract Frames → Embed Each Frame → Store in Vector DB
```

**Phase 2: Search (real-time)**
```
User Query (text/image) → Embed Query → Vector Search → Return Top Matches
```

### Video Processing Pipeline

```python
# On Modal (serverless GPU)

import modal
from yt_dlp import YoutubeDL
import cv2
from qwen3_vl_embedding import Qwen3VLEmbedder

app = modal.App("video-moment-finder")

@app.function(gpu="A10G", timeout=1800)
def process_video(youtube_url: str, video_id: str):
    # 1. Download video
    ydl_opts = {
        'format': 'best[height<=720]',
        'outtmpl': '/tmp/video.mp4'
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])

    # 2. Extract frames at 1 fps
    cap = cv2.VideoCapture('/tmp/video.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Sample at 1 fps
        if frame_idx % int(fps) == 0:
            frames.append(frame)
        frame_idx += 1

    # 3. Embed frames with Qwen3-VL-Embedding
    model = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")

    embeddings = []
    for i, frame in enumerate(frames):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        embedding = model.process([{
            "image": frame_rgb,
            "instruction": "Represent the visual content of this video frame"
        }])

        embeddings.append({
            "id": f"{video_id}_{i}",
            "vector": embedding[0].tolist(),
            "payload": {
                "video_id": video_id,
                "timestamp_seconds": i,
                "frame_index": i
            }
        })

    # 4. Upload to Qdrant
    qdrant_client.upsert(
        collection_name="video_frames",
        points=embeddings
    )

    return {"frame_count": len(frames)}
```

### Search Implementation

```python
from qdrant_client import QdrantClient
from qwen3_vl_embedding import Qwen3VLEmbedder

model = Qwen3VLEmbedder("Qwen/Qwen3-VL-Embedding-2B")
qdrant = QdrantClient(url="...", api_key="...")

def search_video(video_id: str, query_text: str = None, query_image = None):
    # Embed the query
    if query_text:
        query_embedding = model.process([{
            "text": query_text,
            "instruction": "Find video frames matching this description"
        }])
    else:
        query_embedding = model.process([{
            "image": query_image,
            "instruction": "Find video frames similar to this image"
        }])

    # Search Qdrant
    results = qdrant.search(
        collection_name="video_frames",
        query_vector=query_embedding[0].tolist(),
        query_filter={
            "must": [{"key": "video_id", "match": {"value": video_id}}]
        },
        limit=5
    )

    return [{
        "timestamp_seconds": r.payload["timestamp_seconds"],
        "score": r.score,
        "thumbnail_url": f"https://r2.../thumbnails/{r.id}.jpg"
    } for r in results]
```

## Model Specifications

### Qwen3-VL-Embedding-2B

| Spec | Value |
|------|-------|
| Parameters | 2B |
| Embedding dimension | 2048 |
| Max image resolution | 1280x1440 |
| Max video frames | 64 (native), unlimited with batching |
| Context length | 32k tokens |
| Languages | 30+ |

### Hardware Requirements

| Setup | RAM | GPU VRAM | Processing Speed |
|-------|-----|----------|------------------|
| CPU only | 16GB | - | ~10s per frame |
| A10G (Modal) | - | 24GB | ~0.5s per frame |
| A100 | - | 40GB | ~0.2s per frame |

**Cost estimate** (Modal A10G at ~$1.10/hour):
- 30-min video = 1800 frames
- At 0.5s/frame = 900 seconds = 15 minutes
- GPU cost: ~$0.27 per video

## Frame Sampling Strategies

| Strategy | Frames for 30min | Pros | Cons |
|----------|------------------|------|------|
| 1 fps | 1800 | Comprehensive | Expensive |
| 0.5 fps | 900 | Good balance | May miss quick moments |
| Keyframes | ~200-500 | Efficient | Misses static scenes |
| Scene detection | ~100-300 | Very efficient | Complex to implement |

**Recommendation**: Start with 1 fps for quality, optimize later if costs are too high.

## Vector Database: Qdrant

### Collection Schema

```python
from qdrant_client.models import VectorParams, Distance

client.create_collection(
    collection_name="video_frames",
    vectors_config=VectorParams(
        size=2048,  # Qwen3-VL-Embedding-2B output dimension
        distance=Distance.COSINE
    )
)
```

### Payload Structure

```json
{
  "video_id": "uuid",
  "user_id": "clerk_user_id",
  "timestamp_seconds": 45,
  "frame_index": 45,
  "thumbnail_url": "https://r2.example.com/thumb_45.jpg"
}
```

### Filtering by Video

```python
# Search only within a specific video
results = client.search(
    collection_name="video_frames",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="video_id", match=MatchValue(value="abc123"))]
    ),
    limit=5
)
```

## Existing Solutions

| Solution | How it works | Limitations |
|----------|--------------|-------------|
| YouTube chapters | Manual timestamps | Creator must add them |
| YouTube transcript search | Text search in captions | Misses visual content |
| Rewind.ai | Screen recording search | Desktop only, privacy concerns |
| Frame.io | Professional video review | Expensive, no semantic search |
| Twelve Labs | Video understanding API | API-only, expensive |

**Gap**: No consumer tool for semantic visual search within videos.

## Competitive Advantages

1. **Open source model**: No per-query API costs after GPU processing
2. **Visual search**: Find by what you SEE, not just what was SAID
3. **Image-to-video search**: Upload reference image to find similar moments
4. **Simple UX**: Paste URL → wait → search (no upload required for YouTube)

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| YouTube blocks downloads | High | Consider direct upload option |
| Low search quality | High | Test extensively, add reranker |
| High GPU costs | Medium | Optimize frame sampling |
| Slow processing | Medium | Show progress, email notification |
| Vector DB costs at scale | Medium | Start with free tier, optimize |

## Success Criteria for MVP

1. **Technical**: Successfully embed and search 10 test videos
2. **Quality**: >70% of searches return relevant result in top 3
3. **Performance**: <20 min processing for 30-min video
4. **Cost**: <$1 per video (GPU + storage)

## References

- [Qwen3-VL-Embedding GitHub](https://github.com/QwenLM/Qwen3-VL-Embedding)
- [Qwen3-VL-Embedding-2B on HuggingFace](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B)
- [Technical Paper (arXiv)](https://arxiv.org/abs/2601.04720)
- [Modal Documentation](https://modal.com/docs)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)

## Qualitative Assessment

| Factor | Rating | Notes |
|--------|--------|-------|
| Competition | Low | No direct consumer tool exists |
| MVP Complexity | Medium-High | Multiple services, GPU processing |
| Potential Value | High | Solves real pain for video creators |
| Learning Value | Very High | Multimodal ML, serverless GPU, SaaS |
| Revenue Potential | Medium | Niche but willing-to-pay audience |

**Overall**: Strong learning project with genuine market potential. The tech is cutting-edge (Qwen3-VL released Jan 2025) and the problem is real. Main risks are GPU costs and search quality.
