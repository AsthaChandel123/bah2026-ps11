# xsretrieval — serving image for the FastAPI retrieval API (apps.api:app).
#
# Default base is CPU (python:3.11-slim): CPU PyTorch + faiss-cpu + the serving
# deps. CPU is normally sufficient — vector search is sub-millisecond and the
# backbone runs one forward pass per query.
#
# Build (from the repo root, ideally with a trained artifacts/index/ present so it
# gets baked in):
#     docker build -t xsretrieval-api:latest .
#
# Run:
#     docker run --rm -p 8000:8000 \
#       -e XSRETRIEVAL_INDEX=/app/artifacts/index xsretrieval-api:latest
#     curl http://localhost:8000/health
#
# GPU ALTERNATIVE (only if backbone inference is the latency bottleneck): swap the
# base for a CUDA runtime, install Python + GPU torch wheels, and keep the rest:
#     FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
#     RUN apt-get update && apt-get install -y python3.11 python3-pip && ln -s ...
#     RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
#     RUN pip install faiss-gpu-cu12
#   then the same `pip install -e .` + serving deps + CMD below. Deploy on Cloud
#   Run with `--gpu 1 --gpu-type nvidia-l4`.

FROM python:3.11-slim

# System libs needed by rasterio/pillow (GeoTIFF + image IO) and HTTPS for HF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libexpat1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    XSRETRIEVAL_INDEX=/app/artifacts/index \
    HF_HOME=/app/.hf_cache

WORKDIR /app

# 1) Install CPU PyTorch + faiss-cpu first (heaviest layer; cached across rebuilds).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<2.9" \
    && pip install "faiss-cpu>=1.7.4"

# 2) Serving + compute + model + IO dependencies (mirrors requirements-cpu.txt,
#    minus the torch/faiss already installed above).
RUN pip install \
        "numpy>=1.24,<3" "scipy>=1.10" "scikit-learn>=1.3" \
        "transformers>=4.38" "timm>=0.9.16" "open_clip_torch>=2.24" "huggingface_hub>=0.21" \
        "pyyaml>=6.0" "pillow>=10.0" "rasterio>=1.3" "tqdm>=4.65" \
        "fastapi>=0.110" "uvicorn>=0.27" "python-multipart>=0.0.9"

# 3) Copy the repo (the .dockerignore trims data/, caches, .git, etc.). This also
#    bakes in artifacts/index/ if it is present (option (i) in docs/DEPLOYMENT.md).
COPY . /app

# 4) Install the package itself (no deps — they are installed above).
RUN pip install -e . --no-deps

# Cloud Run sends traffic to $PORT (default 8000). uvicorn serves apps.api:app.
EXPOSE 8000

# Use the shell form so $PORT is expanded (Cloud Run injects it).
CMD uvicorn apps.api:app --host 0.0.0.0 --port ${PORT}
