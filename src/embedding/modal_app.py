from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

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


def _optional_non_negative_int_env(name: str) -> int | None:
    """
    Parse an optional non-negative integer environment variable.

    Returns None when unset/blank so Modal uses default scaling behavior.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")

    return value


_TEXT_EMBED_MODEL: Any | None = None


def _resolve_text_embed_min_containers() -> int | None:
    """
    Resolve optional warm-container config for text embedding.

    `MODAL_TEXT_EMBED_MIN_CONTAINERS` is preferred.
    `MODAL_TEXT_EMBED_KEEP_WARM` remains as backward-compatible alias.
    """
    min_containers = _optional_non_negative_int_env("MODAL_TEXT_EMBED_MIN_CONTAINERS")
    if min_containers is not None:
        return min_containers
    return _optional_non_negative_int_env("MODAL_TEXT_EMBED_KEEP_WARM")


TEXT_EMBED_MIN_CONTAINERS = _resolve_text_embed_min_containers()


def _create_qwen_embedder():
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    return Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")


def _get_text_embedder():
    """Cache the text embedder per Modal container to avoid per-request reloads."""
    global _TEXT_EMBED_MODEL
    if _TEXT_EMBED_MODEL is None:
        _TEXT_EMBED_MODEL = _create_qwen_embedder()
    return _TEXT_EMBED_MODEL


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


def _normalize_embedding(embedding):
    """Convert embedding to normalized tensor and return as CPU tensor."""
    import torch
    import torch.nn.functional as F

    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu()
    else:
        embedding = torch.tensor(embedding)
    return F.normalize(embedding.float(), dim=1)


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
        batch_embeddings = _normalize_embedding(model.process(batch_input))

        if batch_embeddings.shape[0] != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: {batch_embeddings.shape[0]} != {len(batch)}"
            )

        embeddings.extend([emb.tolist() for emb in batch_embeddings])

    if len(embeddings) != len(images):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(images)}")

    return embeddings


@app.function(
    image=image,
    gpu="A10G",
    timeout=300,
    min_containers=TEXT_EMBED_MIN_CONTAINERS,
)
def embed_text(text: str) -> list[float]:
    """
    Embed a text query and return normalized 2048-dim vector.

    Fail-fast behavior:
    - Raises ValueError if text is empty.
    """
    if not text or not text.strip():
        raise ValueError("text must be non-empty")

    model = _get_text_embedder()
    embedding = _normalize_embedding(model.process([{"text": text.strip()}]))
    return embedding[0].tolist()
