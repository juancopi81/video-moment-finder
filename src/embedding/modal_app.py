from __future__ import annotations

import io
import os
from pathlib import Path
import tempfile
from typing import Any

import modal

APP_NAME = "video-moment-finder-embed"
APP_PATH = Path("/root/app")
MODAL_UV_SYNC_COMMAND = "uv sync --frozen --group modal --compile-bytecode --python-preference=only-system"

app = modal.App(APP_NAME)
WHISPER_MODEL_NAME = "turbo"
WHISPER_COMPUTE_TYPE = "int8_float16"
WHISPER_BATCH_SIZE = 16

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
        MODAL_UV_SYNC_COMMAND,
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


def _resolve_query_embed_min_containers() -> int | None:
    """
    Resolve optional warm-container config for query embedding.

    Returns None when unset so Modal uses default scaling behavior.
    """
    return _optional_non_negative_int_env("MODAL_QUERY_EMBED_MIN_CONTAINERS")


QUERY_EMBED_MIN_CONTAINERS = _resolve_query_embed_min_containers()


def _resolve_query_embed_max_containers() -> int:
    """
    Resolve max query embed containers.

    Defaults to 1 so repeated sequential searches hit a reused container/model.
    """
    max_containers = _optional_non_negative_int_env("MODAL_QUERY_EMBED_MAX_CONTAINERS")
    if max_containers is None:
        return 1
    if max_containers == 0:
        raise ValueError("MODAL_QUERY_EMBED_MAX_CONTAINERS must be >= 1")
    return max_containers


QUERY_EMBED_MAX_CONTAINERS = _resolve_query_embed_max_containers()


def _validate_query_embed_container_bounds(
    min_containers: int | None,
    max_containers: int,
) -> None:
    if min_containers is not None and min_containers > max_containers:
        raise ValueError(
            "MODAL_QUERY_EMBED_MIN_CONTAINERS cannot exceed MODAL_QUERY_EMBED_MAX_CONTAINERS"
        )


_validate_query_embed_container_bounds(
    QUERY_EMBED_MIN_CONTAINERS,
    QUERY_EMBED_MAX_CONTAINERS,
)


def _create_qwen_embedder():
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    return Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")


def _create_whisper_pipeline():
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore

    model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cuda",
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    return BatchedInferencePipeline(model=model)


def _normalize_transcription_segments(
    segments: Any,
    *,
    language_code: str | None,
) -> list[dict[str, Any]]:
    normalized_segments: list[dict[str, Any]] = []

    for segment in segments:
        text = " ".join(segment.text.split()).strip()
        if not text:
            continue

        start_s = float(segment.start)
        end_s = float(segment.end)
        normalized_segments.append(
            {
                "start_s": start_s,
                "end_s": max(end_s, start_s),
                "text": text,
                "language_code": language_code,
            }
        )

    return normalized_segments


@app.cls(
    image=image,
    gpu="A10G",
    timeout=300,
    min_containers=QUERY_EMBED_MIN_CONTAINERS,
    max_containers=QUERY_EMBED_MAX_CONTAINERS,
)
class QueryEmbedder:
    """Container-scoped query embedder with startup model preload."""

    model: Any | None = None

    @modal.enter()
    def load_model(self) -> None:
        self.model = _create_qwen_embedder()

    @modal.method()
    def embed_text(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        # Fallback guard if a container enters without initialization.
        if self.model is None:
            self.model = _create_qwen_embedder()

        embedding = _normalize_embedding(self.model.process([{"text": text.strip()}]))
        return embedding[0].tolist()

    @modal.method()
    def embed_image(self, image_bytes: bytes) -> list[float]:
        if not image_bytes:
            raise ValueError("image_bytes must be non-empty")

        if self.model is None:
            self.model = _create_qwen_embedder()

        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            pil_image = img.convert("RGB")

        embedding = _normalize_embedding(self.model.process([{"image": pil_image}]))
        return embedding[0].tolist()


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
def embed_texts_in_batches(
    texts: list[str], *, batch_size: int = 32
) -> list[list[float]]:
    """Embed texts in fixed-size batches and return normalized vectors."""
    if not texts:
        raise ValueError("texts must be a non-empty list")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    model = Qwen3VLEmbedder(model_name_or_path="Qwen/Qwen3-VL-Embedding-2B")

    embeddings: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        cleaned_batch = []
        for text in batch:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError("text items must be non-empty")
            cleaned_batch.append({"text": cleaned})

        batch_embeddings = _normalize_embedding(model.process(cleaned_batch))
        if batch_embeddings.shape[0] != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch: {batch_embeddings.shape[0]} != {len(batch)}"
            )

        embeddings.extend([emb.tolist() for emb in batch_embeddings])

    if len(embeddings) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: {len(embeddings)} != {len(texts)}")

    return embeddings


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


@app.function(image=image, gpu="T4", timeout=1800)
def transcribe_audio_bytes(
    audio_bytes: bytes,
    *,
    batch_size: int = WHISPER_BATCH_SIZE,
) -> list[dict[str, Any]]:
    """Transcribe audio bytes into normalized timestamped segments."""
    if not audio_bytes:
        raise ValueError("audio_bytes must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    pipeline = _create_whisper_pipeline()
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = Path(temp_dir) / "input.wav"
        audio_path.write_bytes(audio_bytes)
        segments, info = pipeline.transcribe(
            str(audio_path),
            batch_size=batch_size,
            task="transcribe",
            vad_filter=True,
            word_timestamps=False,
        )
        language_code = getattr(info, "language", None)
        return _normalize_transcription_segments(
            segments,
            language_code=language_code,
        )
