"""
Phase 0.2: Video Processing Test

End-to-end pipeline test: extract frames → embed on GPU → measure cost.

Due to YouTube bot detection on cloud IPs, download the video locally first:
    uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "YOUTUBE_URL"

Then run:
    uv run modal run scripts/phase0/modal_video_test.py --max-frames 100
"""

from pathlib import Path

import modal

from src.video.frames import extract_frames

APP_NAME = "video-moment-finder-video-test"
APP_PATH = Path("/root/app")
VIDEO_PATH = Path("/root/video/input.mp4")

# Check for test video at module level
LOCAL_VIDEO_PATH = Path("test_video.f136.mp4")  # Default video-only file from yt-dlp
if not LOCAL_VIDEO_PATH.exists():
    LOCAL_VIDEO_PATH = Path("test_video.mp4")  # Merged version

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

A10G_COST_PER_SEC = 0.000463  # Modal A10G pricing (approximate)


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    import json
    import subprocess

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


@app.function(image=image, gpu="A10G", timeout=1800)
def process_video_test(max_frames: int = 100) -> dict:
    """
    Process the video file mounted at VIDEO_PATH.

    Args:
        max_frames: Maximum number of frames to embed (default: 100)

    Returns:
        Dictionary with timing metrics and cost estimates
    """
    import time

    import torch
    from PIL import Image
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    total_start = time.perf_counter()

    video_path = VIDEO_PATH
    print(f"Using video: {video_path}")

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found at {video_path}. "
            "Download a video first with: uv run yt-dlp -f 'best[height<=720]' -o 'test_video.mp4' 'URL'"
        )

    # Get video duration for extrapolation
    video_duration = get_video_duration(video_path)
    print(f"Video duration: {video_duration:.1f}s")

    # Step 1: Extract frames at 1 fps
    print("Extracting frames at 1 fps...")
    frames_dir = Path("/tmp/frames")
    extraction_start = time.perf_counter()
    frames = extract_frames(video_path, frames_dir, fps=1)
    extraction_time = time.perf_counter() - extraction_start
    frame_paths = [f.path for f in frames]
    total_frames = len(frames)
    print(f"Extracted {total_frames} frames in {extraction_time:.2f}s")

    # Limit frames for embedding test
    frames_to_embed = frame_paths[:max_frames]
    print(f"Will embed {len(frames_to_embed)} frames (max_frames={max_frames})")

    # Step 2: Load model
    print("Loading Qwen3-VL-Embedding model...")
    model_load_start = time.perf_counter()
    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")
    model_load_time = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_time:.2f}s")

    # Step 3: Embed frames one at a time (to avoid OOM)
    print(f"Embedding {len(frames_to_embed)} frames (one at a time)...")
    embed_start = time.perf_counter()

    all_embeddings = []
    for i, frame_path in enumerate(frames_to_embed):
        img = Image.open(frame_path)
        embedding = model.process([{"image": img}])
        all_embeddings.append(embedding)
        if (i + 1) % 10 == 0:
            print(f"  Embedded {i + 1}/{len(frames_to_embed)} frames...")

    # Stack all embeddings
    embeddings = torch.cat(
        [
            torch.tensor(e) if not isinstance(e, torch.Tensor) else e
            for e in all_embeddings
        ],
        dim=0,
    )

    embed_time_total = time.perf_counter() - embed_start
    embed_time_per_frame = embed_time_total / len(frames_to_embed)
    print(f"Embedded in {embed_time_total:.2f}s ({embed_time_per_frame:.3f}s/frame)")

    # Convert to tensor for shape info
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)

    total_gpu_time = time.perf_counter() - total_start

    # Cost calculations
    estimated_cost = total_gpu_time * A10G_COST_PER_SEC

    # Extrapolate to 30-min video (1800 frames at 1fps)
    frames_30min = 1800
    # Time = extraction + model_load + (embed_per_frame * 1800)
    extrapolated_embed_time = embed_time_per_frame * frames_30min
    # Assume extraction scales linearly with duration
    scale_factor = (30 * 60) / video_duration  # 30 min / actual duration
    extrapolated_extraction = extraction_time * scale_factor
    extrapolated_total = (
        extrapolated_extraction + model_load_time + extrapolated_embed_time
    )
    extrapolated_30min_cost = extrapolated_total * A10G_COST_PER_SEC

    results = {
        "video_duration_s": video_duration,
        "frame_extraction_time_s": extraction_time,
        "frames_extracted": total_frames,
        "frames_embedded": len(frames_to_embed),
        "model_load_time_s": model_load_time,
        "embed_time_total_s": embed_time_total,
        "embed_time_per_frame_s": embed_time_per_frame,
        "embeddings_shape": list(embeddings.shape),
        "total_gpu_time_s": total_gpu_time,
        "estimated_cost_usd": estimated_cost,
        "extrapolated_30min_cost_usd": extrapolated_30min_cost,
        "gate_passed": extrapolated_30min_cost < 1.0,
    }

    print("\n=== Results ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    return results


@app.local_entrypoint()
def main(max_frames: int = 100):
    """
    Run the video processing test.

    Expects test_video.f136.mp4 or test_video.mp4 to exist in the current directory.
    Download first with: uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "URL"

    Usage:
        uv run modal run scripts/phase0/modal_video_test.py --max-frames 100
    """
    if not LOCAL_VIDEO_PATH.exists():
        print("ERROR: No test video found!")
        print("Download a video first:")
        print(
            '  uv run yt-dlp -f "best[height<=720]" -o "test_video.mp4" "YOUTUBE_URL"'
        )
        return

    print("Starting video processing test...")
    print(f"Using video: {LOCAL_VIDEO_PATH}")
    print(f"Max frames: {max_frames}")
    print()

    results = process_video_test.remote(max_frames)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n" + "=" * 60)
    if results["gate_passed"]:
        print("GATE CHECK: PASSED - Extrapolated 30-min cost < $1")
    else:
        print("GATE CHECK: FAILED - Extrapolated 30-min cost >= $1")
        print("Batching optimization may be required.")
    print("=" * 60)
