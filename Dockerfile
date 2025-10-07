# Diffusers Hathora container
FROM python:3.10-slim-bullseye

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

COPY setup.py README.md .
COPY src/ ./src
RUN pip install --no-cache-dir -e .[torch]

COPY .hathora_build/app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY .hathora_build/app/ .


EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
