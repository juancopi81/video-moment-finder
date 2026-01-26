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
from src.utils.logging import Timer, get_logger

APP_NAME = "video-moment-finder-search-test"
APP_PATH = Path("/root/app")
VIDEO_PATH = Path("/root/video/input.mp4")
GROUND_TRUTH_PATH = Path("/root/app/ground_truth.json")

LOCAL_VIDEO_PATH = Path("test_video.mp4")
LOCAL_GROUND_TRUTH_PATH = Path("test_data/search_validation/reachy_unboxing.json")

app = modal.App(APP_NAME)
logger = get_logger(__name__)

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
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    total_timer = Timer("Total search validation", logger, level="debug").start()

    # Load ground truth
    with open(GROUND_TRUTH_PATH) as f:
        ground_truth = json.load(f)

    logger.info("Loaded %d ground truth queries", len(ground_truth["queries"]))
    logger.info("Video: %s", ground_truth["description"])

    # Check video exists
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found at {VIDEO_PATH}. "
            "Download first with: uv run yt-dlp -f 'best[height<=720]' -o 'test_video.mp4' 'URL'"
        )

    # Step 1: Extract frames
    logger.info("=== Step 1: Extracting frames ===")
    frames_dir = Path("/tmp/frames")
    extraction_timer = Timer("Frame extraction", logger).start()
    frames = extract_frames(VIDEO_PATH, frames_dir, fps=1)
    extraction_time = extraction_timer.stop()
    frame_paths = [f.path for f in frames]
    logger.info("Extracted %d frames in %.2fs", len(frame_paths), extraction_time)

    # Step 2: Load embedding model
    logger.info("=== Step 2: Loading model ===")
    model_timer = Timer("Model load", logger).start()
    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
    model_load_time = model_timer.stop()
    logger.info("Model loaded in %.2fs", model_load_time)

    # Step 3: Embed all frames (with batching)
    logger.info("=== Step 3: Embedding frames ===")

    def embed_with_batch_size(batch_size: int) -> tuple[list, float]:
        """Embed all frames with given batch size. Returns (embeddings, time)."""
        torch.cuda.empty_cache()
        embeddings = []
        embed_timer = Timer(
            f"Embedding batch_size={batch_size}", logger, level="debug"
        ).start()

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

            logger.info(
                "Embedded %d/%d frames (batch=%d)...",
                min(i + batch_size, len(frame_paths)),
                len(frame_paths),
                batch_size,
            )

        return embeddings, embed_timer.stop()

    # Batch=8 is optimal for A10G (24GB VRAM) with Qwen3-VL-Embedding-2B.
    # Tested batch sizes: 1, 4, 8, 16, 32 - all fit in memory.
    # Results: batch=8 fastest (0.146s/frame), larger batches plateau (~0.15s/frame).
    # GPU saturates at batch=8; more batching adds overhead without benefit.
    frame_embeddings = None
    batch_size_used = None

    for batch_size in [8, 4]:
        try:
            logger.info("Trying batch_size=%d...", batch_size)
            frame_embeddings, embed_time = embed_with_batch_size(batch_size)
            batch_size_used = batch_size
            logger.info(
                "Embedded all frames in %.2fs with batch_size=%d",
                embed_time,
                batch_size,
            )
            break
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM with batch_size=%d, trying smaller batch...", batch_size)
            torch.cuda.empty_cache()

    if frame_embeddings is None:
        # Final fallback to batch=1
        logger.info("Falling back to batch_size=1...")
        frame_embeddings, embed_time = embed_with_batch_size(1)
        batch_size_used = 1
        logger.info("Embedded all frames in %.2fs with batch_size=1", embed_time)

    # Get embedding dimension
    embedding_dim = len(frame_embeddings[0])
    logger.info("Embedding dimension: %d", embedding_dim)

    # Step 4: Create in-memory Qdrant and store embeddings
    logger.info("=== Step 4: Storing in Qdrant ===")
    qdrant_timer = Timer("Qdrant upsert", logger).start()

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
    qdrant_time = qdrant_timer.stop()
    logger.info(
        "Stored %d embeddings in Qdrant in %.2fs", len(points), qdrant_time
    )

    # Step 5: Run queries and check results
    logger.info("=== Step 5: Running queries ===")
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
        logger.info(
            "Q%d: %s - Expected: %ss, Got: %s",
            query_id,
            status,
            expected_ts,
            result_timestamps,
        )

    # Calculate Recall@5
    recall_at_5 = hits / len(ground_truth["queries"])
    logger.info("=== Results ===")
    logger.info("Hits: %d/%d", hits, len(ground_truth["queries"]))
    logger.info("Recall@5: %.1f%%", recall_at_5 * 100)

    total_time = total_timer.stop()

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
        logger.error("No test video found!")
        logger.error("Download the video first:")
        logger.error(
            '  uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "https://www.youtube.com/watch?v=RmXOoJXBYLk"'
        )
        return

    if not LOCAL_GROUND_TRUTH_PATH.exists():
        logger.error("Ground truth not found at %s", LOCAL_GROUND_TRUTH_PATH)
        return

    logger.info("=" * 60)
    logger.info("Phase 0.3: Vector Search Validation")
    logger.info("=" * 60)
    logger.info("Video: %s", LOCAL_VIDEO_PATH)
    logger.info("Ground truth: %s", LOCAL_GROUND_TRUTH_PATH)

    results = search_validation_test.remote()

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info("Total frames: %s", results["total_frames"])
    logger.info("Embedding dimension: %s", results["embedding_dim"])
    logger.info("Batch size: %s", results["batch_size"])
    logger.info(
        "Embed time: %.2fs (%.3fs/frame)",
        results["embed_time_s"],
        results["embed_time_per_frame_s"],
    )
    logger.info("Total time: %.2fs", results["total_time_s"])
    logger.info("Queries: %s", results["total_queries"])
    logger.info("Hits: %s", results["hits"])
    logger.info("Recall@5: %.1f%%", results["recall_at_5"] * 100)

    logger.info("=" * 60)
    logger.info("Query Details:")
    logger.info("=" * 60)
    for qr in results["query_results"]:
        status = "HIT" if qr["is_hit"] else "MISS"
        logger.info("Q%s: [%s]", qr["id"], status)
        logger.info("  Query: %s", qr["query"])
        logger.info(
            "  Expected: %ss (tolerance: +/-%ss)",
            qr["expected_ts"],
            qr["tolerance"],
        )
        logger.info(
            "  Top 5 results: %s (scores: %s)",
            qr["result_timestamps"],
            [f"{s:.3f}" for s in qr["result_scores"]],
        )

    logger.info("=" * 60)
    if results["gate_passed"]:
        logger.info(
            "GATE CHECK: PASSED - Recall@5 = %.1f%% >= 70%",
            results["recall_at_5"] * 100,
        )
    else:
        logger.error(
            "GATE CHECK: FAILED - Recall@5 = %.1f%% < 70%",
            results["recall_at_5"] * 100,
        )
        logger.error("Consider:")
        logger.error("  - Adjusting query phrasings")
        logger.error("  - Changing frame extraction rate")
        logger.error("  - Trying a different embedding model")
    logger.info("=" * 60)
