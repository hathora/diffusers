from __future__ import annotations

import os
import logging
import pathlib
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi

from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline, WanPipeline, AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import AutoTokenizer, UMT5EncoderModel
import torch
import imageio
import numpy as np

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


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/openapi.json", include_in_schema=False)
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
    model_id = os.getenv("DIFFUSERS_MODEL_ID", "Wan-AI/Wan2.1-T2V-14B-Diffusers")
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
        _PIPELINE["pipe"] = pipe
        _PIPELINE["model_id"] = model_id
        logger.info("Pipeline ready.")
    except Exception as e:
        logger.exception(f"Pipeline load failed: {e}")


@app.get("/health", response_class=JSONResponse)
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
