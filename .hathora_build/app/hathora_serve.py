from __future__ import annotations

import os
import logging
import pathlib
import signal
import sys
from typing import Optional
import asyncio

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, WanPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
import torch
import imageio
import numpy as np

# Local compile manager (supports both package/standalone runs)
try:
    from .compile_manager import CompileManager, shape_key
except Exception:
    from compile_manager import CompileManager, shape_key

def setup_compile_optimizations():
    """
    Set up torch.compile optimizations to reduce first-time compilation time.
    
    Strategy for fast first-launch:
    - Limit autotuning search space (faster compile, still good perf)
    - Use pre-tuned configs for common operations
    - Skip excessive kernel benchmarking
    """
    # FLUX uses /opt/*_cache (baked in), other models DON'T use any cache
    model_id = os.getenv("MODEL_ID", "").lower()
    
    if "flux" in model_id:
        # FLUX uses the baked-in cache
        cache_dir = "/opt/inductor_cache"
        triton_cache_dir = "/opt/triton_cache"
        
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(triton_cache_dir, exist_ok=True)
        
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
        os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    else:
        # Wan and other models: disable torch.compile cache entirely
        # This prevents loading FLUX cache files into memory
        if "TORCHINDUCTOR_CACHE_DIR" in os.environ:
            del os.environ["TORCHINDUCTOR_CACHE_DIR"]
        if "TRITON_CACHE_DIR" in os.environ:
            del os.environ["TRITON_CACHE_DIR"]
        if "TORCHINDUCTOR_FX_GRAPH_CACHE" in os.environ:
            del os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"]
        
        print("INFO: Torch.compile cache disabled for non-FLUX models")
        return  # Skip the rest of compile optimization setup
    
    max_autotune_gemm = os.getenv("MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "4")
    os.environ["MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = max_autotune_gemm
    
    max_autotune_pointwise = os.getenv("MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE", "8")
    os.environ["MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE"] = max_autotune_pointwise
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SKIP_SIMILAR"] = "1"
    
    # Balance compile speed vs runtime performance
    os.environ["TORCHINDUCTOR_BENCHMARK_KERNEL_ITERATIONS"] = "3"
    
    # Enable freezing for better performance
    os.environ["TORCHINDUCTOR_FREEZING"] = "1"
    
    logging.info(f"Torch compile cache: {cache_dir}")
    logging.info(f"Triton cache: {triton_cache_dir}")
    logging.info(f"Max autotune GEMM search space: {max_autotune_gemm} (limited for faster compile)")
    logging.info(f"Max autotune pointwise search space: {max_autotune_pointwise}")

setup_compile_optimizations()

def setup_dynamic_shapes():
    """
    Configure for dynamic shapes (works with any height/width).
    
    Strategy: Compile once with dynamic shapes, works for all sizes without recompilation.
    Uses max-autotune-no-cudagraphs for best autotuning without CUDA graphs.
    
    Benefits:
    - Compile once at startup (~2-3 min)
    - Any height/width works instantly (no recompilation)
    - Excellent autotuned performance
    - No CUDA graphs (not compatible with dynamic shapes)
    """
    import torch._dynamo.config
    import torch._inductor.config
    
    # Enable automatic dynamic shapes
    torch._dynamo.config.automatic_dynamic_shapes = True
    
    # CRITICAL: Don't assume static by default (required for dynamic to work!)
    torch._dynamo.config.assume_static_by_default = False
    
    # Disable CUDA graphs (incompatible with dynamic shapes)
    torch._inductor.config.triton.cudagraphs = False
    
    # Suppress warnings about graph breaks
    torch._dynamo.config.suppress_errors = True
    
    # Capture scalar outputs for better dynamic shape handling
    torch._dynamo.config.capture_scalar_outputs = True
    
    # INFERENCE-ONLY OPTIMIZATIONS: Skip all gradient/training-related compilation
    # Disable gradient-related optimizations (inference only, no backprop needed)
    torch._inductor.config.fallback_random = True  # Faster for inference
    
    # Skip coordinate descent tuning (training optimization, not needed for inference)
    # This also speeds up compilation significantly
    os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "0"
    
    try:
        from torch._inductor.fx_passes import post_grad
        original_remove_noop_ops = post_grad.remove_noop_ops
        
        def safe_remove_noop_ops(graph):
            """Wrapper that catches SymFloat errors and skips the optimization"""
            try:
                return original_remove_noop_ops(graph)
            except AttributeError as e:
                if "'SymFloat' object has no attribute 'size'" in str(e):
                    logging.debug("Skipped remove_noop_ops due to SymFloat bug")
                    return  # Skip this optimization pass
                raise
        
        post_grad.remove_noop_ops = safe_remove_noop_ops
        logging.info("Applied SymFloat bug workaround (patched remove_noop_ops)")
    except Exception as e:
        logging.warning(f"Could not apply SymFloat workaround: {e}")
    
    # Enable verbose logging for compilation progress
    verbose_compile = os.getenv("VERBOSE_COMPILE", "1")
    if verbose_compile == "2":
        torch._dynamo.config.verbose = True
        torch._inductor.config.debug = True
    
    logging.info("=" * 60)
    logging.info("COMPILATION SETTINGS")
    logging.info("=" * 60)
    logging.info("Mode: Dynamic shapes with max-autotune-no-cudagraphs")
    logging.info("Strategy: Compile with dynamic=True, works for all shapes")
    logging.info("CUDA graphs: Disabled (incompatible with dynamic shapes)")
    logging.info("Assume static by default: False (required!)")
    logging.info("Shape flexibility: Any height/width without recompilation")
    logging.info("Inference-only: Gradient/training optimizations disabled")
    logging.info("Coordinate descent tuning: Disabled (inference only)")
    logging.info("Verbose compile logs: Level %s", verbose_compile)
    logging.info("=" * 60)

model_id_env = os.getenv("MODEL_ID", "").lower()
if "flux" in model_id_env or os.getenv("USE_TORCH_COMPILE", "1") == "1":
    setup_dynamic_shapes()
else:
    logging.info("Skipping torch.compile setup for non-FLUX models")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diffusers_svc")

class T2VRequest(BaseModel):
    prompt: str
    num_frames: Optional[int] = 49
    fps: Optional[int] = 8
    height: Optional[int] = 480
    width: Optional[int] = 832
    seed: Optional[int] = None
    repo_id: Optional[str] = None
    steps: Optional[int] = None


class T2IRequest(BaseModel):
    prompt: str
    height: Optional[int] = 512
    width: Optional[int] = 512
    steps: Optional[int] = None
    seed: Optional[int] = None

app = FastAPI(default_response_class=JSONResponse)

_PIPELINE: dict = {"pipe": None, "model_id": None, "eager_transformer": None, "compiled_transformer": None}

# Compile manager state (for FLUX only)
_COMPILE_STATE: dict = {
    "compiled_shapes": set(),  # e.g., {"1024x1024"}
    "pending": set(),
    "compiling": False,
    "swap_lock": None,  # asyncio.Lock, initialized on startup
}

def _shape_key(h: int, w: int) -> str:
    return f"{int(h)}x{int(w)}"

# FLUX compile orchestrator
_FLUX_COMPILE = CompileManager()

# Authentication
auth = HTTPBearer(auto_error=False)
API_TOKEN = os.getenv("API_KEY") or os.getenv("HATHORA_APP_SECRET")

def require_token(creds: HTTPAuthorizationCredentials = Depends(auth)):
    if not API_TOKEN:
        return  # Skip auth if not configured
    if not creds or creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def cleanup_gpu_memory():
    """Clear GPU memory on shutdown"""
    try:
        logger.info("=" * 70)
        logger.info("SHUTTING DOWN - CLEARING GPU MEMORY")
        logger.info("=" * 70)
        
        # Clear pipeline references
        if _PIPELINE["pipe"] is not None:
            del _PIPELINE["pipe"]
            _PIPELINE["pipe"] = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get memory stats
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory after cleanup:")
            logger.info(f"  Allocated: {allocated:.2f} GB")
            logger.info(f"  Reserved: {reserved:.2f} GB")
        
        logger.info("Cleanup complete")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(signum, frame):
    """Handle SIGTERM/SIGINT gracefully"""
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_gpu_memory()
    sys.exit(0)

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Register FastAPI shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    cleanup_gpu_memory()
def custom_openapi():
    if getattr(app, "openapi_schema", None):
        return app.openapi_schema
    try:
        schema = get_openapi(
            title="Diffusers Service",
            version="1.0.0",
            routes=app.routes,
        )
    except Exception as e:
        logger.exception("OpenAPI generation failed: %s", e)
        schema = {"openapi": "3.1.0", "info": {"title": "Diffusers Service", "version": "1.0.0"}, "paths": {}}
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi


def _ensure_fhwc_uint8(frames_any):
    arr = np.array(frames_any)
    # Expect (F,H,W,C). Convert common alternatives.
    if arr.ndim == 5:
        # (B,F,H,W,C) or (B,C,F,H,W)
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[0]
        elif arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr[0], (1, 2, 3, 0))  # C,F,H,W -> F,H,W,C
    elif arr.ndim == 4:
        # (F,H,W,C) or (C,F,H,W)
        if arr.shape[-1] in (1, 3, 4):
            pass
        elif arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 3, 0))  # C,F,H,W -> F,H,W,C
    else:
        raise ValueError("Unexpected frames shape; expected 4D or 5D array")
    # to uint8
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)
    return arr


@app.get("/", include_in_schema=False, dependencies=[Depends(require_token)])
async def root():
    return RedirectResponse(url="/docs")


@app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(require_token)])
async def openapi_json():
    try:
        schema = app.openapi()
        return JSONResponse(schema)
    except Exception as e:
        logger.exception("OpenAPI JSON error: %s", e)
        fallback = {"openapi": "3.1.0", "info": {"title": "Diffusers Service", "version": "1.0.0"}, "paths": {}}
        return JSONResponse(fallback)


@app.post(
    "/image_gen/t2i",
    response_class=FileResponse,
    responses={
        200: {"content": {"image/png": {}}, "description": "Generated PNG image"},
        400: {"content": {"application/json": {}}, "description": "Bad Request"},
        500: {"content": {"application/json": {}}, "description": "Error"},
    },
    dependencies=[Depends(require_token)],
)
async def t2i(req: T2IRequest):
    try:
        if _PIPELINE["pipe"] is None:
            raise HTTPException(status_code=503, detail="Pipeline is not ready")
        pipe: DiffusionPipeline = _PIPELINE["pipe"]
        # Wan pipeline is video-only; reject
        if isinstance(pipe, WanPipeline):
            raise HTTPException(status_code=400, detail="Current pipeline is video-only. Set DIFFUSERS_MODEL_ID to an image pipeline (e.g., FLUX/SDXL).")

        # Seed
        if req.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(req.seed)
        else:
            generator = None

        num_steps = req.steps or int(os.getenv("IMG_STEPS", "30"))
        
        # Decide eager vs compiled per-shape (FLUX only). Unknown shapes served eager and compiled in background.
        height, width = int(req.height), int(req.width)
        skey = shape_key(height, width)
        eager_t = _PIPELINE.get("eager_transformer")
        compiled_t = _PIPELINE.get("compiled_transformer")
        prefer_compiled = compiled_t is not None and skey in _FLUX_COMPILE.compiled_shapes

        def _run_with_transformer(target_transformer):
            original = getattr(pipe, "transformer", None)
            try:
                if target_transformer is not None and original is not target_transformer:
                    pipe.transformer = target_transformer
                # Use bf16 autocast for VAE if enabled (memory + speed optimization)
                use_vae_autocast = os.getenv("VAE_AUTOCAST_BF16", "0") in ("1", "true", "yes")
                if use_vae_autocast and torch.cuda.is_available():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        return pipe(prompt=req.prompt, height=height, width=width, num_inference_steps=num_steps, generator=generator)
                return pipe(prompt=req.prompt, height=height, width=width, num_inference_steps=num_steps, generator=generator)
            finally:
                # Restore eager as default to avoid accidental compiled-trigger on unknown shapes
                if eager_t is not None:
                    pipe.transformer = eager_t

        if prefer_compiled:
            result = _run_with_transformer(compiled_t)
        else:
            # Serve eager immediately and queue background compile if available
            result = _run_with_transformer(eager_t or getattr(pipe, "transformer", None))
            # Queue background compile for this shape if we have a compiled transformer prepared
            if compiled_t is not None:
                try:
                    if skey not in _FLUX_COMPILE.compiled_shapes and skey not in _COMPILE_STATE["pending"]:
                        _COMPILE_STATE["pending"].add(skey)
                        async def _compile_shape_bg(h, w, key):
                            try:
                                async with _COMPILE_STATE["swap_lock"]:
                                    _COMPILE_STATE["compiling"] = True
                                    # one-step warmup compile in background
                                    await asyncio.to_thread(
                                        _FLUX_COMPILE.warmup_flux,
                                        pipe,
                                        compiled_t,
                                        [(h, w)],
                                        1,
                                        logger,
                                    )
                            except Exception as be:
                                logger.warning(f"Background compile failed for {h}x{w}: {be}")
                            finally:
                                _COMPILE_STATE["compiling"] = False
                                _COMPILE_STATE["pending"].discard(key)
                        asyncio.create_task(_compile_shape_bg(height, width, skey))
                except Exception:
                    pass

        images = getattr(result, "images", None) or result
        if not images:
            raise HTTPException(status_code=500, detail="No image returned by pipeline")

        out_path = "/tmp/out_image.png"
        img0 = images[0]
        # Prefer PIL save when available
        if hasattr(img0, "save"):
            img0.save(out_path)
        else:
            arr = np.array(img0)
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0.0, 1.0)
                arr = (arr * 255.0).astype(np.uint8)
            imageio.imwrite(out_path, arr)
        return FileResponse(out_path, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("T2I error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



def _resolve_model_id(model_id: str) -> str:
    if model_id and "/" in model_id and not os.path.isdir(model_id):
        base_dir = os.getenv("WAN_LOCAL_DIR", "/models")
        safe = model_id.replace("/", "_")
        local_dir = os.path.join(base_dir, safe)
        pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False,
                          token=os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN"))
        return local_dir
    return model_id


def _normalize_to_diffusers_repo(model_id: str) -> str:
    # Map base Wan repo to diffusers-converted repo
    if "Wan2.1-T2V" in str(model_id) and "-Diffusers" not in str(model_id):
        # Handle common ids like Wan-AI/Wan2.1-T2V-14B
        if "Wan-AI/Wan2.1-T2V-14B" in model_id:
            return "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        if "Wan-AI/Wan2.1-T2V-1.3B" in model_id:
            return "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        # default append if pattern unknown
        return f"{model_id}-Diffusers"
    return model_id


@app.on_event("startup")
async def startup():
    model_id = os.getenv("MODEL_PATH") or "Wan-AI/Wan2.1-T2V-14B-Diffusers"
    model_id = _normalize_to_diffusers_repo(model_id)
    model_id = _resolve_model_id(model_id)
    logger.info(f"Loading pipeline: {model_id}")
    
    offload_model = os.getenv("WAN_OFFLOAD_MODEL", "0").lower() in ("1", "true", "yes")
    convert_model_dtype = os.getenv("WAN_CONVERT_DTYPE", "0").lower() in ("1", "true", "yes")
    t5_cpu = os.getenv("WAN_T5_CPU", "0").lower() in ("1", "true", "yes")
    
    torch_dtype = torch.bfloat16 if os.getenv("VAE_AUTOCAST_BF16", "1").lower() in ("1","true","yes") else torch.float16
    device = 0 if torch.cuda.is_available() else -1
    try:
        if "Wan2.1-T2V" in model_id or "Wan2.2-T2V" in model_id or "Wan2.2-TI2V" in model_id:
            try:
                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                dynamo.reset()
                torch.jit._state.disable()
                logger.info("Disabled torch.compile and JIT for Wan model")
            except Exception as e:
                logger.warning(f"Could not disable torch.compile/JIT: {e}")
            # Apply OOM mitigation: keep T5 on CPU if requested
            text_encoder_device = "cpu" if t5_cpu else None
            text_encoder_dtype = torch_dtype if convert_model_dtype else (torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float32)
            
            transformer_dtype = torch_dtype if convert_model_dtype else None
            
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=True)
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=text_encoder_dtype,
                low_cpu_mem_usage=True,
            )
            if text_encoder_device == "cpu":
                text_encoder = text_encoder.to("cpu")
                logger.info("T5 text encoder kept on CPU to reduce VRAM usage")
            
            vae = AutoencoderKLWan.from_pretrained(
                model_id, 
                subfolder="vae", 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            
            pipe = WanPipeline.from_pretrained(
                model_id,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                torch_dtype=transformer_dtype,
                low_cpu_mem_usage=True,
            )
            # Wan2.2 uses different flow_shift defaults
            if "Wan2.2" in model_id:
                flow_shift = float(os.getenv("WAN_FLOW_SHIFT", "7.0"))  # Higher for Wan2.2
            else:
                flow_shift = float(os.getenv("WAN_FLOW_SHIFT", "3.0"))  # Wan2.1 default
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        else:
            try:
                # Improve performance consistency (use TF32 tensor cores for fp32 matmuls)
                try:
                    torch.set_float32_matmul_precision("high")
                    logger.info("Enabled TF32 matmul precision: high")
                except Exception:
                    pass
                from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
                # Avoid passing unsupported kwargs to underlying transformers models
                pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
        if device >= 0:
            if offload_model:
                # Use model CPU offloading to reduce VRAM
                logger.info("Enabling model CPU offloading to reduce VRAM usage")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to("cuda")
        
        # Initialize compile manager lock
        if _COMPILE_STATE.get("swap_lock") is None:
            _COMPILE_STATE["swap_lock"] = asyncio.Lock()
        
        # Log Wan-specific optimizations
        is_wan = isinstance(pipe, WanPipeline)
        if is_wan:
            logger.info("")
            logger.info("=" * 70)
            logger.info("WAN VIDEO PIPELINE OPTIMIZATIONS")
            logger.info("=" * 70)
            vae_autocast = os.getenv("VAE_AUTOCAST_BF16", "0") in ("1", "true", "yes")
            cuda_alloc = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "")
            logger.info(f"VAE BF16 Autocast: {'Enabled' if vae_autocast else 'Disabled'}")
            logger.info(f"CUDA Allocator: {cuda_alloc if cuda_alloc else 'Default'}")
            logger.info(f"Model CPU Offload: {'Enabled' if offload_model else 'Disabled'}")
            logger.info(f"Convert Model Dtype: {'Enabled' if convert_model_dtype else 'Disabled'}")
            logger.info(f"T5 on CPU: {'Enabled' if t5_cpu else 'Disabled'}")
            if vae_autocast:
                logger.info("  VAE will use bfloat16 for faster decoding & lower memory")
            if "expandable_segments" in cuda_alloc:
                logger.info("  Memory allocator uses expandable segments (reduces fragmentation)")
            if offload_model:
                logger.info("  Models will be offloaded to CPU when not in use")
            if t5_cpu:
                logger.info("  T5 text encoder kept on CPU to save VRAM")
            logger.info("=" * 70)
        
        # Compile the model with torch.compile for faster inference
        use_compile = os.getenv("USE_TORCH_COMPILE", "1").lower() in ("1", "true", "yes")
        if use_compile and device >= 0:
            is_wan = isinstance(pipe, WanPipeline)
            
            env_mode = os.getenv("TORCH_COMPILE_MODE", "")
            if env_mode:
                compile_mode = env_mode
                logger.info(f"Using custom TORCH_COMPILE_MODE: {compile_mode}")
            elif is_wan:
                # Wan needs dynamic shapes without CUDA graphs
                # max-autotune-no-cudagraphs is designed for this (~3 min compile)
                compile_mode = "max-autotune-no-cudagraphs"
                logger.info("Detected Wan pipeline: Using max-autotune-no-cudagraphs mode")
            else:
                # FLUX/SDXL also uses max-autotune-no-cudagraphs for best quality
                compile_mode = "max-autotune-no-cudagraphs"
                logger.info("Detected image pipeline: Using max-autotune-no-cudagraphs mode")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("TORCH.COMPILE: PREPARING MODEL FOR COMPILATION")
            logger.info("=" * 70)
            logger.info(f"Pipeline: {'Video (Wan)' if is_wan else 'Image (FLUX/SDXL)'}")
            logger.info(f"Mode: {compile_mode}")
            logger.info(f"Device: cuda:{device}")
            logger.info("Note: First compilation is slow but results are cached")
            logger.info("=" * 70)
            
            try:
                # Skip torch.compile entirely for WAN pipeline (per design)
                if isinstance(pipe, WanPipeline):
                    raise RuntimeError("SKIP_COMPILE_WAN")
                # Compile the main denoising component (transformer/unet)
                # Use dynamic=True for height/width/frames flexibility
                
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    logger.info("")
                    logger.info(">>> Step 1/3: Wrapping transformer with torch.compile...")
                    
                    if is_wan:
                        # Wan video pipeline - compile transformers
                        logger.info("    Detected Wan pipeline - applying video optimizations")
                        logger.info("    Note: channels_last not used (requires 4D, video is 5D)")
                        
                        # Compile primary transformer with explicit dynamic shapes
                        logger.info("    Enabling dynamic shapes for height/width/frames flexibility")
                        logger.info("    Using fullgraph=False for dynamic shape compatibility")
                        logger.info("    CUDA graphs: DISABLED (incompatible with dynamic shapes)")
                        
                        # Configure backend to handle dynamic shapes properly (same as FLUX)
                        import torch._dynamo as _torch_dynamo
                        import torch._inductor.config as _inductor_config
                        _torch_dynamo.config.capture_dynamic_output_shape_ops = True
                        
                        # CRITICAL: Disable CUDA graphs for dynamic shapes
                        _inductor_config.triton.cudagraphs = False
                        
                        pipe.transformer = torch.compile(
                            pipe.transformer,
                            mode=compile_mode,
                            fullgraph=False,  # False for dynamic shapes (same as FLUX)
                            dynamic=True,
                        )
                        logger.info("    Primary transformer wrapped successfully")
                        
                        # Compile secondary transformer by default for Wan to avoid runtime recompiles
                        compile_transformer_2 = os.getenv("COMPILE_TRANSFORMER_2", "1").lower() in ("1", "true", "yes")
                        if compile_transformer_2 and hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
                            logger.info("")
                            logger.info(">>> Step 2/3: Wrapping transformer_2 (low-noise stage)...")
                            logger.info("    This adds ~2 minutes to compilation time")
                            pipe.transformer_2 = torch.compile(
                                pipe.transformer_2,
                                mode=compile_mode,
                                fullgraph=False,  # False for dynamic shapes
                                dynamic=True,
                            )
                            logger.info("    Secondary transformer wrapped successfully")
                        elif hasattr(pipe, "transformer_2") and pipe.transformer_2 is not None:
                            logger.info("")
                            logger.info(">>> Skipping transformer_2 compilation by request")
                            logger.info("    Set COMPILE_TRANSFORMER_2=1 to enable at startup")
                    else:
                        # FLUX/SDXL image pipeline: keep eager default and prepare compiled wrapper
                        logger.info("    Enabling dynamic shapes for height/width flexibility (FLUX)")
                        eager_t, compiled_t = _FLUX_COMPILE.prepare_flux_transformer(pipe, compile_mode, logger)
                        _PIPELINE["eager_transformer"] = eager_t
                        _PIPELINE["compiled_transformer"] = compiled_t
                        pipe.transformer = eager_t  # keep eager by default; we'll swap per-shape
                        logger.info("    Transformer compile wrapper created (compiled available per-shape)")
                        
                elif hasattr(pipe, "unet") and pipe.unet is not None:
                    logger.info("")
                    logger.info(">>> Step 1/3: Wrapping unet with torch.compile...")
                    logger.info("    Enabling dynamic shapes for height/width flexibility")
                    pipe.unet = torch.compile(
                        pipe.unet,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=True,
                    )
                    logger.info("    Unet wrapped successfully")
                
                # Optionally compile VAE decoder
                compile_vae = os.getenv("COMPILE_VAE", "0").lower() in ("1", "true", "yes")
                
                if compile_vae and hasattr(pipe, "vae") and pipe.vae is not None:
                    logger.info("")
                    logger.info(">>> Wrapping VAE decoder with torch.compile...")
                    if is_wan:
                        logger.info("    Wan VAE: Compiling video decoder")
                    pipe.vae.decoder = torch.compile(
                        pipe.vae.decoder,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=True,
                    )
                    logger.info("    VAE decoder wrapped successfully")
                elif hasattr(pipe, "vae") and pipe.vae is not None:
                    logger.info("")
                    logger.info(">>> Skipping VAE compilation")
                    logger.info("    Set COMPILE_VAE=1 to enable (~30-60s compilation, minor speedup)")
                
                logger.info("")
                logger.info("=" * 70)
                logger.info("WARMUP (FLUX): PRECOMPILE COMMON SHAPES")
                logger.info("=" * 70)
                warmup_shape_str = os.getenv("WARMUP_SHAPES", "1024x1024,768x768")
                warmup_shapes = []
                for _s in warmup_shape_str.split(","):
                    try:
                        ww, hh = _s.strip().split("x")
                        warmup_shapes.append((int(hh), int(ww)))
                    except Exception:
                        logger.warning(f"Invalid warmup shape: {_s}, skipping")
                logger.info(f"Warmup shapes: {', '.join([f'{h}x{w}' for h, w in warmup_shapes])}")
                logger.info("After warmup: warmed shapes use compiled path; others served eager + background compile")
                logger.info("=" * 70)
                
                try:
                    import time
                    warmup_prompt = "test warmup"
                    warmup_steps = 1  # Single step is enough to trigger compilation
                    
                    logger.info("")
                    logger.info("Starting compilation (this triggers autotuning)...")
                    compile_start = time.time()
                    
                    if isinstance(pipe, WanPipeline):
                        logger.info("Wan pipeline - skipping warmup since torch.compile is disabled")
                    else:
                        # Image pipeline warmup (FLUX/SDXL) into compiled path
                        logger.info("Pipeline type: Image (FLUX/SDXL)")
                        if _PIPELINE.get("compiled_transformer") is not None and warmup_shapes:
                            _FLUX_COMPILE.warmup_flux(
                                pipe,
                                _PIPELINE["compiled_transformer"],
                                warmup_shapes,
                                warmup_steps,
                                logger,
                            )
                    
                    compile_elapsed = time.time() - compile_start
                    logger.info("")
                    logger.info("=" * 70)
                    logger.info("WARMUP COMPLETE!")
                    logger.info("=" * 70)
                    logger.info(f"Compilation time: {compile_elapsed:.1f}s")
                    try:
                        compiled_list = ", ".join([f"{h}x{w}" for h, w in warmup_shapes]) if warmup_shapes else "none"
                    except Exception:
                        compiled_list = "unknown"
                    logger.info(f"Compiled shapes (FLUX): {compiled_list}")
                    logger.info("Unknown shapes: served eager; background compile queued")
                    logger.info("=" * 70)
                except Exception as warmup_error:
                    logger.error("=" * 70)
                    logger.error("WARMUP FAILED")
                    logger.error("=" * 70)
                    logger.error(f"Error: {warmup_error}")
                    logger.error("Model may still work but without compilation benefits")
                    logger.error("=" * 70)
                    
            except Exception as compile_error:
                if str(compile_error).startswith("SKIP_COMPILE_WAN"):
                    logger.info("Wan pipeline detected: skipping torch.compile and startup warmup")
                else:
                    logger.warning(f"torch.compile failed, continuing without compilation: {compile_error}")
        
        _PIPELINE["pipe"] = pipe
        _PIPELINE["model_id"] = model_id
        logger.info("Pipeline ready.")
    except Exception as e:
        logger.exception(f"Pipeline load failed: {e}")


@app.get("/health", response_class=JSONResponse, dependencies=[Depends(require_token)])
async def health():
    ok = _PIPELINE["pipe"] is not None
    return {"status": "ok" if ok else "loading"}


@app.post(
    "/video_gen/t2v",
    response_class=FileResponse,
    responses={
        200: {"content": {"video/mp4": {}}, "description": "Generated MP4 video"},
        500: {"content": {"application/json": {}}, "description": "Error"},
    },
    dependencies=[Depends(require_token)],
)
async def t2v(req: T2VRequest):
    try:
        if _PIPELINE["pipe"] is None:
            raise HTTPException(status_code=503, detail="Pipeline is not ready")
        pipe: DiffusionPipeline = _PIPELINE["pipe"]
        # Set seed
        if req.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(req.seed)
        else:
            generator = None
        # Generate frames (use image-to-video/video pipelines depending on model)
        # Here we treat as text-to-video-like if supported; otherwise raise
        if not hasattr(pipe, "__call__"):
            raise HTTPException(status_code=400, detail="Pipeline is not invokable")
        # Use bf16 autocast for VAE if enabled (memory + speed optimization)
        use_vae_autocast = os.getenv("VAE_AUTOCAST_BF16", "0") in ("1", "true", "yes")
        
        if isinstance(_PIPELINE["pipe"], WanPipeline):
            if use_vae_autocast and torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = _PIPELINE["pipe"](
                        prompt=req.prompt,
                        height=req.height,
                        width=req.width,
                        num_frames=req.num_frames,
                        num_inference_steps=(req.steps or int(os.getenv("WAN_STEPS", "30"))),
                        guidance_scale=5.0,
                        generator=generator,
                    )
            else:
                result = _PIPELINE["pipe"](
                    prompt=req.prompt,
                    height=req.height,
                    width=req.width,
                    num_frames=req.num_frames,
                    num_inference_steps=(req.steps or int(os.getenv("WAN_STEPS", "30"))),
                    guidance_scale=5.0,
                    generator=generator,
                )
            frames = result.frames
        else:
            if use_vae_autocast and torch.cuda.is_available():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = _PIPELINE["pipe"](prompt=req.prompt, num_frames=req.num_frames, height=req.height, width=req.width, generator=generator)
            else:
                result = _PIPELINE["pipe"](prompt=req.prompt, num_frames=req.num_frames, height=req.height, width=req.width, generator=generator)
            frames = result.frames if hasattr(result, "frames") else result.images
        if frames is None:
            raise HTTPException(status_code=500, detail="No frames returned by pipeline")
        frames_np = _ensure_fhwc_uint8(frames)
        out_path = "/tmp/out_video.mp4"
        imageio.mimwrite(out_path, list(frames_np), fps=req.fps, quality=8)
        return FileResponse(out_path, media_type="video/mp4")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("T2V error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
