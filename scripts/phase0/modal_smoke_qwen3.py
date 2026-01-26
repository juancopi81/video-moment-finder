from pathlib import Path

import modal

from src.utils.logging import Timer, get_logger

APP_NAME = "video-moment-finder-smoke-test"
APP_PATH = Path("/root/app")

app = modal.App(APP_NAME)
logger = get_logger(__name__)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("uv")
    .workdir(APP_PATH)
    # Copy dependency manifest
    .add_local_file("pyproject.toml", str(APP_PATH / "pyproject.toml"), copy=True)
    .add_local_file("uv.lock", str(APP_PATH / "uv.lock"), copy=True)
    # Make uv install into system site-packages (no .venv)
    .env({"UV_PROJECT_ENVIRONMENT": "/usr/local"})
    .run_commands(
        "uv sync --frozen --compile-bytecode --python-preference=only-system",
        # Bring the Qwen embedder script into PYTHONPATH (simple + reliable)
        "git clone --depth 1 https://github.com/QwenLM/Qwen3-VL-Embedding.git /root/qwen3-vl-embedding",
    )
    .env({"PYTHONPATH": "/root/qwen3-vl-embedding/src"})
)


@app.function(image=image, gpu="A10G", timeout=1800)
def smoke_test():
    import torch
    import torch.nn.functional as F
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    with Timer("Total smoke test", logger, level="debug") as total_timer:
        model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
        with Timer("Model load", logger) as load_timer:
            model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
        load_s = load_timer.elapsed or 0.0

        inputs = [
            {"text": "a person holding a phone"},
            {
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
            },
        ]

        with Timer("Embedding", logger) as embed_timer:
            embeddings = model.process(inputs)
        embed_s = embed_timer.elapsed or 0.0

        # embeddings can come as torch.Tensor in GPU
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)

        embeddings = embeddings.float()
        embeddings = F.normalize(embeddings, dim=1)
        cosine_similarity = float((embeddings[0] * embeddings[1]).sum().item())

    return {
        "cuda": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "embeddings_shape": embeddings.shape,
        "cosine_similarity(text, image)": cosine_similarity,
        "load_time": load_s,
        "embed_time": embed_s,
        "total_time": total_timer.elapsed or 0.0,
    }


@app.local_entrypoint()
def main():
    out = smoke_test.remote()
    logger.info("Smoke test output: %s", out)
