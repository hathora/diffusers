from __future__ import annotations

import os
import logging
import pathlib
from typing import Optional

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

def setup_compile_optimizations():
    """
    Set up torch.compile optimizations to reduce first-time compilation time.
    
    Strategy for fast first-launch:
    - Limit autotuning search space (faster compile, still good perf)
    - Use pre-tuned configs for common operations
    - Skip excessive kernel benchmarking
    """
    cache_dir = os.getenv("TORCH_COMPILE_CACHE_DIR", "/app/.cache/torch_compile")
    triton_cache_dir = os.getenv("TRITON_CACHE_DIR", "/app/.cache/triton")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(triton_cache_dir, exist_ok=True)
    
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
    os.environ["TRITON_CACHE_DIR"] = triton_cache_dir
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    
    max_autotune_gemm = os.getenv("MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "4")
    os.environ["MAX_AUTOTUNE_GEMM_SEARCH_SPACE"] = max_autotune_gemm
    
    max_autotune_pointwise = os.getenv("MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE", "8")
    os.environ["MAX_AUTOTUNE_POINTWISE_SEARCH_SPACE"] = max_autotune_pointwise
    os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SKIP_SIMILAR"] = "1"
    
    # Coordinate descent tuning for better performance with dynamic shapes
    os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
    
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
    logging.info("Verbose compile logs: Level %s", verbose_compile)
    logging.info("=" * 60)

setup_dynamic_shapes()

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

_PIPELINE: dict = {"pipe": None, "model_id": None}

# Authentication
auth = HTTPBearer(auto_error=False)
API_TOKEN = os.getenv("HATHORA_APP_SECRET")

def require_token(creds: HTTPAuthorizationCredentials = Depends(auth)):
    if not API_TOKEN:
        return  # Skip auth if not configured
    if not creds or creds.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
        result = pipe(prompt=req.prompt, height=req.height, width=req.width, num_inference_steps=num_steps, generator=generator)

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
    torch_dtype = torch.bfloat16 if os.getenv("VACE_AUTOCast_BF16", "true").lower() in ("1","true","yes") else torch.float16
    device = 0 if torch.cuda.is_available() else -1
    try:
        if "Wan2.1-T2V" in model_id:
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=True)
            text_encoder = UMT5EncoderModel.from_pretrained(
                model_id,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16 if torch_dtype == torch.bfloat16 else torch.float32,
                low_cpu_mem_usage=False,
            )
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
            pipe = WanPipeline.from_pretrained(
                model_id,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
            )
            flow_shift = float(os.getenv("WAN_FLOW_SHIFT", "3.0"))
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        else:
            try:
                from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
                # Avoid passing unsupported kwargs to underlying transformers models
                pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
            except Exception:
                pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False)
        if device >= 0:
            pipe.to("cuda")
        
        # Compile the model with torch.compile for faster inference
        use_compile = os.getenv("USE_TORCH_COMPILE", "1").lower() in ("1", "true", "yes")
        if use_compile and device >= 0:
            compile_mode = os.getenv("TORCH_COMPILE_MODE", "max-autotune")
            
            logger.info("")
            logger.info("=" * 70)
            logger.info("TORCH.COMPILE: PREPARING MODEL FOR COMPILATION")
            logger.info("=" * 70)
            logger.info(f"Mode: {compile_mode}")
            logger.info(f"Device: cuda:{device}")
            logger.info("Note: First compilation is slow but results are cached")
            logger.info("=" * 70)
            
            try:
                # Compile the transformer/unet component for maximum performance
                # Use dynamic=True to explicitly enable dynamic shapes
                if hasattr(pipe, "transformer") and pipe.transformer is not None:
                    logger.info("")
                    logger.info(">>> Step 1/2: Wrapping transformer with torch.compile...")
                    logger.info("    Enabling dynamic shapes for height/width flexibility")
                    pipe.transformer = torch.compile(
                        pipe.transformer,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=True,  # Explicitly enable dynamic shapes
                    )
                    logger.info("    Transformer wrapped successfully")
                elif hasattr(pipe, "unet") and pipe.unet is not None:
                    logger.info("")
                    logger.info(">>> Step 1/2: Wrapping unet with torch.compile...")
                    logger.info("    Enabling dynamic shapes for height/width flexibility")
                    pipe.unet = torch.compile(
                        pipe.unet,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=True,  # Explicitly enable dynamic shapes
                    )
                    logger.info("    Unet wrapped successfully")
                
                compile_vae = os.getenv("COMPILE_VAE", "0").lower() in ("1", "true", "yes")
                if compile_vae and hasattr(pipe, "vae") and pipe.vae is not None:
                    logger.info("")
                    logger.info(">>> Step 2/2: Wrapping VAE decoder with torch.compile...")
                    pipe.vae.decoder = torch.compile(
                        pipe.vae.decoder,
                        mode=compile_mode,
                        fullgraph=False,
                        dynamic=True,  # Explicitly enable dynamic shapes
                    )
                    logger.info("    VAE decoder wrapped successfully")
                elif hasattr(pipe, "vae") and pipe.vae is not None:
                    logger.info("")
                    logger.info(">>> Step 2/2: Skipping VAE compilation")
                    logger.info("    Set COMPILE_VAE=1 to enable (~30-60s compilation, minor speedup)")
                
                logger.info("")
                logger.info("=" * 70)
                logger.info("WARMUP: COMPILING MODEL WITH DYNAMIC SHAPES")
                logger.info("=" * 70)
                
                # With dynamic shapes, we only need to compile once
                # The compiled model will work for any height/width
                warmup_shape_str = os.getenv("WARMUP_SHAPE", "1024x1024")
                try:
                    w, h = warmup_shape_str.strip().split("x")
                    warmup_height, warmup_width = int(h), int(w)
                except:
                    logger.warning(f"⚠ Invalid warmup shape: {warmup_shape_str}, using default")
                    warmup_height, warmup_width = 1024, 1024
                
                logger.info(f"Warmup size: {warmup_height}x{warmup_width}")
                logger.info("After compilation: ALL sizes will work instantly")
                logger.info("Estimated time: ~2-3 minutes (compile once, use everywhere)")
                logger.info("=" * 70)
                
                try:
                    import time
                    warmup_prompt = "test warmup"
                    warmup_steps = 2  # Minimal steps for warmup
                    
                    logger.info("")
                    logger.info("Starting compilation (this triggers autotuning)...")
                    compile_start = time.time()
                    
                    if isinstance(pipe, WanPipeline):
                        # Video pipeline warmup
                        logger.info("Pipeline type: Video (WanPipeline)")
                        _ = pipe(
                            prompt=warmup_prompt,
                            height=warmup_height,
                            width=warmup_width,
                            num_frames=9,
                            num_inference_steps=warmup_steps,
                            guidance_scale=5.0,
                        )
                    else:
                        # Image pipeline warmup (FLUX/SDXL)
                        logger.info("Pipeline type: Image (FLUX/SDXL)")
                        _ = pipe(
                            prompt=warmup_prompt,
                            height=warmup_height,
                            width=warmup_width,
                            num_inference_steps=warmup_steps,
                        )
                    
                    compile_elapsed = time.time() - compile_start
                    logger.info("")
                    logger.info("=" * 70)
                    logger.info("WARMUP COMPLETE!")
                    logger.info("=" * 70)
                    logger.info(f"Compilation time: {compile_elapsed:.1f}s")
                    logger.info(f"Compiled with: {warmup_height}x{warmup_width}")
                    logger.info("✓ Dynamic shapes enabled: ANY size now works instantly")
                    logger.info("✓ No recompilation needed for different heights/widths")
                    logger.info("=" * 70)
                except Exception as warmup_error:
                    logger.error("=" * 70)
                    logger.error("WARMUP FAILED")
                    logger.error("=" * 70)
                    logger.error(f"Error: {warmup_error}")
                    logger.error("Model may still work but without compilation benefits")
                    logger.error("=" * 70)
                    
            except Exception as compile_error:
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
        if isinstance(_PIPELINE["pipe"], WanPipeline):
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
