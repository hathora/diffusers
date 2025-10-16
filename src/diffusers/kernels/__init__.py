# Triton Kernels

try:
    from .triton_ops import (
        RopeFunction,
        RMSNormFunction,
        LayerNormFunction,
        GELUMulFunction,
    )
    from .layers import (
        TritonRMSNorm,
        TritonGEGLUMLP,
        triton_rotary_pos_emb,
    )
    TRITON_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Triton kernels: {e}")
    TRITON_AVAILABLE = False
    RopeFunction = None
    RMSNormFunction = None
    LayerNormFunction = None
    GELUMulFunction = None
    TritonRMSNorm = None
    TritonGEGLUMLP = None
    triton_rotary_pos_emb = None

__all__ = [
    "RopeFunction",
    "RMSNormFunction",
    "LayerNormFunction",
    "GELUMulFunction",
    "TritonRMSNorm",
    "TritonGEGLUMLP",
    "triton_rotary_pos_emb",
    "TRITON_AVAILABLE",
]

