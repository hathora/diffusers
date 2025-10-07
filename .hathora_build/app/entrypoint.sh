#!/usr/bin/env bash
set -euo pipefail

export LOG_LEVEL="${LOG_LEVEL:-info}"
export UVICORN_WORKERS="${UVICORN_WORKERS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export HF_TOKEN="${HF_TOKEN:-}"
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

export MODEL_ID="${MODEL_ID:-}"
export MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" && -n "${MODEL_ID}" ]]; then
  export MODEL_PATH="${MODEL_ID}"
fi

export VACE_AUTOCast_BF16="${VACE_AUTOCast_BF16:-true}"

# CUDA envs
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# Diagnostics
echo "[entrypoint] Starting Diffusers service on port 8000 (host ${HOST})"
which nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[entrypoint] nvidia-smi not found; continuing"
python - <<'PY'
try:
    import torch, os
    print(f"[entrypoint] torch.cuda.is_available: {torch.cuda.is_available()}")
    print(f"[entrypoint] torch.cuda.device_count: {torch.cuda.device_count()}")
    print(f"[entrypoint] HF token set: {'HUGGINGFACE_HUB_TOKEN' in os.environ}")
except Exception as e:
    print(f"[entrypoint] torch import failed: {e}")
PY

# Start the FastAPI server
exec uvicorn hathora_serve:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers "$UVICORN_WORKERS" \
  --log-level "$LOG_LEVEL" \
  --timeout-keep-alive 5
