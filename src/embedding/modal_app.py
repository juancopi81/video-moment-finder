from __future__ import annotations

from pathlib import Path
import io

import modal

APP_NAME = "video-moment-finder-embed"
APP_PATH = Path("/root/app")

app = modal.App(APP_NAME)

image = (
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


@app.function(image=image, gpu="A10G", timeout=1800)
def embed_image_batch(images: list[bytes]) -> list[list[float]]:
    """
    Embed a batch of images on GPU and return normalized vectors.

    Fail-fast behavior:
    - Raises ValueError if images list is empty.
    - Raises RuntimeError if embedding output size mismatches input.
    """
    if not images:
        raise ValueError("images must be a non-empty list")

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

    pil_images = []
    for img_bytes in images:
        if not img_bytes:
            raise ValueError("image bytes must be non-empty")
        pil_images.append(Image.open(io.BytesIO(img_bytes)))

    batch_input = [{"image": img} for img in pil_images]
    batch_embeddings = model.process(batch_input)

    if isinstance(batch_embeddings, torch.Tensor):
        batch_embeddings = batch_embeddings.cpu()
    else:
        batch_embeddings = torch.tensor(batch_embeddings)

    batch_embeddings = F.normalize(batch_embeddings.float(), dim=1)

    if batch_embeddings.shape[0] != len(images):
        raise RuntimeError(
            f"Embedding count mismatch: {batch_embeddings.shape[0]} != {len(images)}"
        )

    return [emb.tolist() for emb in batch_embeddings]


@app.function(image=image, timeout=1800)
def extract_frame_bytes(
    video_bytes: bytes, *, fps: float = 1.0, max_frames: int = 64
) -> list[bytes]:
    """
    Extract frames with ffmpeg inside Modal and return them as JPEG bytes.

    Fail-fast behavior:
    - Raises ValueError if video_bytes is empty.
    - Raises ValueError if fps or max_frames are invalid.
    - Raises RuntimeError if no frames are extracted.
    """
    if not video_bytes:
        raise ValueError("video_bytes must be non-empty")
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if max_frames <= 0:
        raise ValueError("max_frames must be > 0")

    from pathlib import Path

    from src.video.frames import extract_frames

    video_path = Path("/tmp/input.mp4")
    video_path.write_bytes(video_bytes)

    frames_dir = Path("/tmp/frames")
    frames = extract_frames(video_path, frames_dir, fps=fps)
    frames = frames[:max_frames]
    if not frames:
        raise RuntimeError("No frames extracted")

    return [frame.path.read_bytes() for frame in frames]


@app.function(image=image, gpu="A10G", timeout=1800)
def embed_images_in_batches(
    images: list[bytes], *, batch_size: int = 8
) -> list[list[float]]:
    """
    Embed images in fixed-size batches and return normalized vectors.

    Fail-fast behavior:
    - Raises ValueError if images list is empty.
    - Raises ValueError if batch_size is not 8.
    - Raises RuntimeError if embedding output size mismatches input.
    """
    if not images:
        raise ValueError("images must be a non-empty list")
    if batch_size != 8:
        raise ValueError("batch_size must be 8 (validated on A10G)")

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

    embeddings: list[list[float]] = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        pil_images = []
        for img_bytes in batch:
            if not img_bytes:
                raise ValueError("image bytes must be non-empty")
            pil_images.append(Image.open(io.BytesIO(img_bytes)))

        batch_input = [{"image": img} for img in pil_images]
        batch_embeddings = model.process(batch_input)

        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = batch_embeddings.cpu()
        else:
            batch_embeddings = torch.tensor(batch_embeddings)

        batch_embeddings = F.normalize(batch_embeddings.float(), dim=1)

        if batch_embeddings.shape[0] != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: {batch_embeddings.shape[0]} != {len(batch)}"
            )

        embeddings.extend([emb.tolist() for emb in batch_embeddings])

    if len(embeddings) != len(images):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(images)}")

    return embeddings
