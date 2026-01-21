"""
Phase 0.3: Vector Search Validation

End-to-end semantic search test: extract frames → embed → store in Qdrant → search.

Due to YouTube bot detection on cloud IPs, download the video locally first:
    uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "https://www.youtube.com/watch?v=RmXOoJXBYLk"

Then run:
    uv run modal run scripts/phase0/modal_search_test.py

Gate: >70% of queries should return the correct frame in top 5 results.
"""

import json
from pathlib import Path

import modal

from src.video.frames import extract_frames

APP_NAME = "video-moment-finder-search-test"
APP_PATH = Path("/root/app")
VIDEO_PATH = Path("/root/video/input.mp4")
GROUND_TRUTH_PATH = Path("/root/app/ground_truth.json")

LOCAL_VIDEO_PATH = Path("test_video.mp4")
LOCAL_GROUND_TRUTH_PATH = Path("test_data/search_validation/reachy_unboxing.json")

app = modal.App(APP_NAME)

# Build image with all dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("uv")
    .workdir(APP_PATH)
    .add_local_file("pyproject.toml", str(APP_PATH / "pyproject.toml"), copy=True)
    .add_local_file("uv.lock", str(APP_PATH / "uv.lock"), copy=True)
    .add_local_dir("src", str(APP_PATH / "src"), copy=True)
    .add_local_file(
        str(LOCAL_GROUND_TRUTH_PATH.resolve()),
        remote_path=str(GROUND_TRUTH_PATH),
        copy=True,
    )
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands(
        "uv sync --frozen --compile-bytecode --python-preference=only-system",
        "git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding.git /root/qwen3-vl-embedding",
    )
    .env({"PYTHONPATH": "/root/qwen3-vl-embedding/src"})
)

# Add video file if it exists
if LOCAL_VIDEO_PATH.exists():
    image = base_image.add_local_file(
        str(LOCAL_VIDEO_PATH.resolve()), remote_path=str(VIDEO_PATH), copy=True
    )
else:
    image = base_image


@app.function(image=image, gpu="A10G", timeout=1800)
def search_validation_test() -> dict:
    """
    Run semantic search validation:
    1. Extract frames from video
    2. Embed all frames
    3. Store in in-memory Qdrant
    4. Run ground truth queries
    5. Calculate Recall@5
    """
    import time

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    total_start = time.perf_counter()

    # Load ground truth
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    print(f"Loaded {len(ground_truth['queries'])} ground truth queries")
    print(f"Video: {ground_truth['description']}")

    # Check video exists
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found at {VIDEO_PATH}. "
            "Download first with: uv run yt-dlp -f 'best[height<=720]' -o 'test_video.mp4' 'URL'"
        )

    # Step 1: Extract frames
    print("\n=== Step 1: Extracting frames ===")
    frames_dir = Path("/tmp/frames")
    extraction_start = time.perf_counter()
    frames = extract_frames(VIDEO_PATH, frames_dir, fps=1)
    extraction_time = time.perf_counter() - extraction_start
    frame_paths = [f.path for f in frames]
    print(f"Extracted {len(frame_paths)} frames in {extraction_time:.2f}s")

    # Step 2: Load embedding model
    print("\n=== Step 2: Loading model ===")
    model_load_start = time.perf_counter()
    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Step 3: Embed all frames (with batching)
    print("\n=== Step 3: Embedding frames ===")

    def embed_with_batch_size(batch_size: int) -> tuple[list, float]:
        """Embed all frames with given batch size. Returns (embeddings, time)."""
        torch.cuda.empty_cache()
        embeddings = []
        t0 = time.perf_counter()

        for i in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[i : i + batch_size]
            imgs = [Image.open(p) for p in batch_paths]
            batch_input = [{"image": img} for img in imgs]

            batch_embeddings = model.process(batch_input)
            if isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = batch_embeddings.cpu()
            else:
                batch_embeddings = torch.tensor(batch_embeddings)

            # Normalize
            batch_embeddings = F.normalize(batch_embeddings.float(), dim=1)

            for emb in batch_embeddings:
                embeddings.append(emb.tolist())

            print(
                f"  Embedded {min(i + batch_size, len(frame_paths))}/{len(frame_paths)} frames (batch={batch_size})..."
            )

        return embeddings, time.perf_counter() - t0

    # Batch=8 is optimal for A10G (24GB VRAM) with Qwen3-VL-Embedding-2B.
    # Tested batch sizes: 1, 4, 8, 16, 32 - all fit in memory.
    # Results: batch=8 fastest (0.146s/frame), larger batches plateau (~0.15s/frame).
    # GPU saturates at batch=8; more batching adds overhead without benefit.
    frame_embeddings = None
    batch_size_used = None

    for batch_size in [8, 4]:
        try:
            print(f"Trying batch_size={batch_size}...")
            frame_embeddings, embed_time = embed_with_batch_size(batch_size)
            batch_size_used = batch_size
            print(
                f"Embedded all frames in {embed_time:.2f}s with batch_size={batch_size}"
            )
            break
        except torch.cuda.OutOfMemoryError:
            print(f"OOM with batch_size={batch_size}, trying smaller batch...")
            torch.cuda.empty_cache()

    if frame_embeddings is None:
        # Final fallback to batch=1
        print("Falling back to batch_size=1...")
        frame_embeddings, embed_time = embed_with_batch_size(1)
        batch_size_used = 1
        print(f"Embedded all frames in {embed_time:.2f}s with batch_size=1")

    # Get embedding dimension
    embedding_dim = len(frame_embeddings[0])
    print(f"Embedding dimension: {embedding_dim}")

    # Step 4: Create in-memory Qdrant and store embeddings
    print("\n=== Step 4: Storing in Qdrant ===")
    qdrant_start = time.perf_counter()

    client = QdrantClient(":memory:")
    collection_name = "frames"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
    )

    # Store with timestamp metadata
    points = []
    for i, embedding in enumerate(frame_embeddings):
        # Frame index i corresponds to timestamp i seconds (1 fps)
        timestamp_s = i
        points.append(
            PointStruct(
                id=i,
                vector=embedding,
                payload={"timestamp_s": timestamp_s, "frame_index": i},
            )
        )

    client.upsert(collection_name=collection_name, points=points)
    qdrant_time = time.perf_counter() - qdrant_start
    print(f"Stored {len(points)} embeddings in Qdrant in {qdrant_time:.2f}s")

    # Step 5: Run queries and check results
    print("\n=== Step 5: Running queries ===")
    query_results = []
    hits = 0

    for query_data in ground_truth["queries"]:
        query_id = query_data["id"]
        query_text = query_data["query"]
        expected_ts = query_data["timestamp_s"]
        tolerance = query_data["tolerance_s"]

        # Embed query text
        query_embedding = model.process([{"text": query_text}])
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu()
        else:
            query_embedding = torch.tensor(query_embedding)
        query_embedding = F.normalize(query_embedding.float(), dim=1)
        query_vector = query_embedding.squeeze(0).tolist()

        # Search Qdrant
        search_response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=5,
        )

        # Check if any result is within tolerance
        result_timestamps = [r.payload["timestamp_s"] for r in search_response.points]
        result_scores = [r.score for r in search_response.points]

        is_hit = any(abs(ts - expected_ts) <= tolerance for ts in result_timestamps)
        if is_hit:
            hits += 1

        result = {
            "id": query_id,
            "query": query_text,
            "expected_ts": expected_ts,
            "tolerance": tolerance,
            "result_timestamps": result_timestamps,
            "result_scores": result_scores,
            "is_hit": is_hit,
        }
        query_results.append(result)

        status = "HIT" if is_hit else "MISS"
        print(
            f"  Q{query_id}: {status} - Expected: {expected_ts}s, Got: {result_timestamps}"
        )

    # Calculate Recall@5
    recall_at_5 = hits / len(ground_truth["queries"])
    print("=== Results ===")
    print(f"Hits: {hits}/{len(ground_truth['queries'])}")
    print(f"Recall@5: {recall_at_5:.1%}")

    total_time = time.perf_counter() - total_start

    gate_passed = recall_at_5 >= 0.70

    results = {
        "video_id": ground_truth["video_id"],
        "total_frames": len(frame_paths),
        "embedding_dim": embedding_dim,
        "batch_size": batch_size_used,
        "extraction_time_s": extraction_time,
        "model_load_time_s": model_load_time,
        "embed_time_s": embed_time,
        "embed_time_per_frame_s": embed_time / len(frame_paths),
        "qdrant_store_time_s": qdrant_time,
        "total_time_s": total_time,
        "total_queries": len(ground_truth["queries"]),
        "hits": hits,
        "recall_at_5": recall_at_5,
        "gate_passed": gate_passed,
        "query_results": query_results,
    }

    return results


@app.local_entrypoint()
def main():
    """
    Run vector search validation test.

    Expects test_video.mp4 to exist in the current directory.
    Download first with:
        uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "https://www.youtube.com/watch?v=RmXOoJXBYLk"
    """
    if not LOCAL_VIDEO_PATH.exists():
        print("ERROR: No test video found!")
        print("Download the video first:")
        print(
            '  uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "https://www.youtube.com/watch?v=RmXOoJXBYLk"'
        )
        return

    if not LOCAL_GROUND_TRUTH_PATH.exists():
        print(f"ERROR: Ground truth not found at {LOCAL_GROUND_TRUTH_PATH}")
        return

    print("=" * 60)
    print("Phase 0.3: Vector Search Validation")
    print("=" * 60)
    print(f"Video: {LOCAL_VIDEO_PATH}")
    print(f"Ground truth: {LOCAL_GROUND_TRUTH_PATH}")
    print()

    results = search_validation_test.remote()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total frames: {results['total_frames']}")
    print(f"Embedding dimension: {results['embedding_dim']}")
    print(f"Batch size: {results['batch_size']}")
    print(
        f"Embed time: {results['embed_time_s']:.2f}s ({results['embed_time_per_frame_s']:.3f}s/frame)"
    )
    print(f"Total time: {results['total_time_s']:.2f}s")
    print()
    print(f"Queries: {results['total_queries']}")
    print(f"Hits: {results['hits']}")
    print(f"Recall@5: {results['recall_at_5']:.1%}")

    print("\n" + "=" * 60)
    print("Query Details:")
    print("=" * 60)
    for qr in results["query_results"]:
        status = "HIT" if qr["is_hit"] else "MISS"
        print(f"\nQ{qr['id']}: [{status}]")
        print(f"  Query: {qr['query']}")
        print(f"  Expected: {qr['expected_ts']}s (tolerance: +/-{qr['tolerance']}s)")
        print(
            f"  Top 5 results: {qr['result_timestamps']} (scores: {[f'{s:.3f}' for s in qr['result_scores']]})"
        )

    print("\n" + "=" * 60)
    if results["gate_passed"]:
        print(f"GATE CHECK: PASSED - Recall@5 = {results['recall_at_5']:.1%} >= 70%")
    else:
        print(f"GATE CHECK: FAILED - Recall@5 = {results['recall_at_5']:.1%} < 70%")
        print("Consider:")
        print("  - Adjusting query phrasings")
        print("  - Changing frame extraction rate")
        print("  - Trying a different embedding model")
    print("=" * 60)
