# Diffusers Hathora container
FROM python:3.10-slim-bullseye

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY setup.py README.md .
COPY src/ ./src
RUN pip install -e .[torch]

COPY src/diffusers/kernels/requirements_kernels.txt .hathora_build/app/requirements.txt /app/
RUN pip install -r /app/requirements_kernels.txt && \
    pip install -r /app/requirements.txt

COPY .hathora_build/app/ .


EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

ENV USE_TRITON_OPS=0
ENV USE_TORCH_COMPILE=1

ENV TORCH_COMPILE_MODE=max-autotune-no-cudagraphs
ENV MAX_AUTOTUNE_GEMM_SEARCH_SPACE=2
ENV MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE=2

ENV COMPILE_VAE=0
ENV VERBOSE_COMPILE=1
ENV WARMUP_SHAPE=1024x1024

ENV TORCH_COMPILE_CACHE_DIR=/app/.cache/torch_compile
ENV TRITON_CACHE_DIR=/app/.cache/triton

RUN mkdir -p /app/.cache/torch_compile /app/.cache/triton

ENTRYPOINT ["/app/entrypoint.sh"]
