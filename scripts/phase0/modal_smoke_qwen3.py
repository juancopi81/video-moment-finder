from pathlib import Path

import modal

APP_NAME = "video-moment-finder-smoke-test"
APP_PATH = Path("/root/app")

app = modal.App(APP_NAME)

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
    import time

    import torch
    import torch.nn.functional as F
    from models.qwen3_vl_embedding import Qwen3VLEmbedder  # type: ignore

    t0 = time.perf_counter()
    model_name_or_path = "Qwen/Qwen3-VL-Embedding-2B"
    model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path)
    load_s = time.perf_counter() - t0

    inputs = [
        {"text": "a person holding a phone"},
        {
            "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        },
    ]

    t1 = time.perf_counter()
    embeddings = model.process(inputs)
    embed_s = time.perf_counter() - t1

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
        "total_time": time.perf_counter() - t0,
    }


@app.local_entrypoint()
def main():
    out = smoke_test.remote()
    print(out)
