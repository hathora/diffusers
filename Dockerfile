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
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

COPY setup.py README.md ./
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

ENV MAX_AUTOTUNE_GEMM_SEARCH_SPACE=2
ENV MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE=2

ENV COMPILE_VAE=0
ENV VERBOSE_COMPILE=1
# Warmup shape: used for image models, auto-capped at 832x480 for Wan video
ENV WARMUP_SHAPE=512x512,768x768,1024x1024,1280x720,1920x1080

# Wan video optimizations
ENV VAE_AUTOCAST_BF16=1
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN mkdir -p /opt/inductor_cache /opt/triton_cache
COPY flux_cache/ /opt/
RUN chmod -R a+rX /opt/inductor_cache /opt/triton_cache

ENTRYPOINT ["/app/entrypoint.sh"]
