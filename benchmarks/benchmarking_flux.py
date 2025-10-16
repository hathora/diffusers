"""
Benchmarking script for FLUX Transformer2DModel.

This script benchmarks various configurations of the FLUX model, including:

1. Triton Ops Comparison:
   - bf16-base-pytorch: PyTorch baseline (USE_TRITON_OPS=0)
   - bf16-triton-ops: Triton kernels for RMSNorm + RoPE (USE_TRITON_OPS=1, USE_GEGLU=0)
   - bf16-triton-geglu: Triton kernels + GeGLU MLP (USE_TRITON_OPS=1, USE_GEGLU=1)
   
   Note: GeGLU uses a different architecture (2 projections instead of 1), so it's
   benchmarked with random initialization rather than pretrained weights.

2. Standard benchmarks: bf16, quantization, layerwise upcasting, group offload, etc.

Usage:
    python benchmarking_flux.py

To manually control Triton ops:
    USE_TRITON_OPS=0 python benchmarking_flux.py  # Disable Triton
    USE_TRITON_OPS=1 python benchmarking_flux.py  # Enable Triton (default)
    USE_GEGLU=1 python benchmarking_flux.py       # Enable GeGLU (requires USE_TRITON_OPS=1)

Results are saved to flux.csv
"""
from functools import partial
import os

import torch
from benchmarking_utils import BenchmarkMixin, BenchmarkScenario, model_init_fn

from diffusers import BitsAndBytesConfig, FluxTransformer2DModel
from diffusers.utils.testing_utils import torch_device


CKPT_ID = "black-forest-labs/FLUX.1-dev"
RESULT_FILENAME = "flux.csv"


def model_init_fn_with_triton_flag(model_cls, use_triton_ops=True, use_geglu=False, **kwargs):
    """Wrapper to set USE_TRITON_OPS and USE_GEGLU env vars before model initialization"""
    os.environ["USE_TRITON_OPS"] = "1" if use_triton_ops else "0"
    os.environ["USE_GEGLU"] = "1" if use_geglu else "0"
    return model_init_fn(model_cls, **kwargs)


def model_init_fn_geglu(model_cls, **init_kwargs):
    """
    Initialize model with GeGLU enabled but without loading pretrained weights.
    GeGLU uses a different architecture (2 projections instead of 1) so it's
    incompatible with pretrained weights. We initialize from config only.
    """
    os.environ["USE_TRITON_OPS"] = "1"
    os.environ["USE_GEGLU"] = "1"
    
    # Extract the pretrained path and load just the config
    pretrained_path = init_kwargs["pretrained_model_name_or_path"]
    subfolder = init_kwargs.get("subfolder", None)
    torch_dtype = init_kwargs.get("torch_dtype", torch.float32)
    
    # Load config from pretrained
    config = model_cls.load_config(pretrained_path, subfolder=subfolder)
    
    # Initialize model from config (random weights)
    model = model_cls.from_config(config, torch_dtype=torch_dtype).eval()
    model.to(torch_device)
    
    return model


def get_input_dict(**device_dtype_kwargs):
    # resolution: 1024x1024
    # maximum sequence length 512
    hidden_states = torch.randn(1, 4096, 64, **device_dtype_kwargs)
    encoder_hidden_states = torch.randn(1, 512, 4096, **device_dtype_kwargs)
    pooled_prompt_embeds = torch.randn(1, 768, **device_dtype_kwargs)
    image_ids = torch.ones(512, 3, **device_dtype_kwargs)
    text_ids = torch.ones(4096, 3, **device_dtype_kwargs)
    timestep = torch.tensor([1.0], **device_dtype_kwargs)
    guidance = torch.tensor([1.0], **device_dtype_kwargs)

    return {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "img_ids": image_ids,
        "txt_ids": text_ids,
        "pooled_projections": pooled_prompt_embeds,
        "timestep": timestep,
        "guidance": guidance,
    }


if __name__ == "__main__":
    scenarios = [
        # Triton Ops Comparison: Base PyTorch vs Triton
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16-base-pytorch",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn_with_triton_flag, use_triton_ops=False),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16-triton-ops",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn_with_triton_flag, use_triton_ops=True, use_geglu=False),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16-triton-geglu",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn_geglu,
        ),
        # Original benchmarks
        BenchmarkScenario(
            name=f"{CKPT_ID}-bf16",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
            compile_kwargs={"fullgraph": True},
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-bnb-nf4",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
                ),
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=model_init_fn,
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-layerwise-upcasting",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(model_init_fn, layerwise_upcasting=True),
        ),
        BenchmarkScenario(
            name=f"{CKPT_ID}-group-offload-leaf",
            model_cls=FluxTransformer2DModel,
            model_init_kwargs={
                "pretrained_model_name_or_path": CKPT_ID,
                "torch_dtype": torch.bfloat16,
                "subfolder": "transformer",
            },
            get_model_input_dict=partial(get_input_dict, device=torch_device, dtype=torch.bfloat16),
            model_init_fn=partial(
                model_init_fn,
                group_offload_kwargs={
                    "onload_device": torch_device,
                    "offload_device": torch.device("cpu"),
                    "offload_type": "leaf_level",
                    "use_stream": True,
                    "non_blocking": True,
                },
            ),
        ),
    ]

    runner = BenchmarkMixin()
    runner.run_bencmarks_and_collate(scenarios, filename=RESULT_FILENAME)
