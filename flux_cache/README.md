# Precompiled FLUX Cache (H100 sm_90)

This directory holds precompiled torch.compile and Triton caches for FLUX on H100 GPUs.

## Initial Build (No Caches)
The first build will have empty cache directories. The container will compile on first startup (~2-3 min).

## Generating Caches
After the first successful run:

```bash
# 1. Start container and wait for compilation to complete
docker run --rm --name warmup --gpus=all \
  -e MODEL_ID=black-forest-labs/FLUX.1-dev \
  -e WARMUP_SHAPES=1024x1024,768x768 \
  andrehathora/hathora-diffusers:v1

# 2. Wait for "Pipeline ready" in logs

# 3. Extract caches
docker cp warmup:/opt/inductor_cache/. flux_cache/inductor_cache/
docker cp warmup:/opt/triton_cache/. flux_cache/triton_cache/

# 4. Stop warmup container
docker stop warmup

# 5. Rebuild image with baked caches
docker build -t andrehathora/hathora-diffusers:v1 .
```

## Result
All subsequent containers from the rebuilt image will start instantly without recompilation.

**Note:** Caches are GPU-specific (H100 only). Different architectures need separate cache builds.
