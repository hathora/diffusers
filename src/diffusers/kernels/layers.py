"""
Triton-accelerated layer implementations for diffusion models.

These layers provide drop-in replacements for standard PyTorch layers with
Triton kernel acceleration, while maintaining compatibility with pretrained weights.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_ops import RMSNormFunction, GELUMulFunction, RopeFunction, TRITON_AVAILABLE

_USE_TRITON_OPS = os.environ.get("USE_TRITON_OPS", "1").lower() in ("1", "true", "yes")
_TRITON_ENABLED = TRITON_AVAILABLE and _USE_TRITON_OPS


class TritonRMSNorm(nn.Module):
    """
    RMSNorm using Triton kernels with automatic fallback to PyTorch.
    
    This is a drop-in replacement for standard RMSNorm that uses Triton
    kernels when available and running on CUDA. API matches Liger's LigerRMSNorm.
    
    Args:
        hidden_size (int): Size of the hidden dimension
        eps (float): A value added to the denominator for numerical stability. Default: 1e-6
        offset (float): Offset to add to weight (used in some models like Gemma). Default: 0.0
        init_fn (str): Initialization function for weight ('ones' or 'zeros'). Default: 'ones'
        in_place (bool): Whether to modify gradients in-place during backward. Default: True
        row_mode (bool): Row-wise computation mode for optimization. Default: None (auto-detect)
    """
    
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        offset=0.0,
        init_fn="ones",
        in_place=True,
        row_mode=None,
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon = eps
        self.offset = offset
        self.in_place = in_place
        self.row_mode = row_mode
    
    def forward(self, hidden_states):
        """
        Forward pass with automatic Triton/PyTorch selection.
        
        Uses Triton kernel when enabled and on CUDA,
        otherwise falls back to PyTorch implementation.
        """
        if _TRITON_ENABLED and hidden_states.is_cuda:
            return RMSNormFunction.apply(
                hidden_states,
                self.weight,
                self.variance_epsilon,
                self.offset,
                self.in_place,
                self.row_mode,
            )
        else:
            # PyTorch fallback: RMS normalize and scale
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return hidden_states * (self.offset + self.weight)
    
    def extra_repr(self):
        return (
            f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, "
            f"offset={self.offset}, in_place={self.in_place}, row_mode={self.row_mode}"
        )


class TritonGEGLUMLP(nn.Module):
    """
    GEGLU MLP using Triton kernels with automatic fallback to PyTorch.
    
    This implements a GeGLU-based MLP layer:
        output = down_proj(GELU(gate_proj(x)) * up_proj(x))
    
    Args:
        hidden_size: Size of input features
        intermediate_size: Size of intermediate features
        bias: Whether to use bias in linear layers. Default: False
    """
    
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Two projections for GeGLU
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x):
        """
        Forward pass with automatic Triton/PyTorch selection.
        
        Uses Triton GeGLU kernel when enabled and on CUDA,
        otherwise falls back to PyTorch implementation.
        """
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        if _TRITON_ENABLED and x.is_cuda:
            # Triton fused GELU + multiply
            intermediate = GELUMulFunction.apply(gate, up)
        else:
            # PyTorch fallback
            intermediate = F.gelu(gate, approximate="tanh") * up
        
        return self.down_proj(intermediate)
    
    def extra_repr(self):
        return f'hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}'


def triton_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.
    
    This is a drop-in replacement for apply_rotary_emb that uses Triton kernels
    when available. API matches Liger's liger_rotary_pos_emb.
    
    Args:
        q (torch.Tensor): The query tensor of shape (bsz, n_q_head, seq_len, head_dim).
        k (torch.Tensor): The key tensor of shape (bsz, n_kv_head, seq_len, head_dim).
        cos (torch.Tensor): The cosine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        sin (torch.Tensor): The sine tensor of shape (1, seq_len, head_dim) or (bsz, seq_len, head_dim).
        position_ids (torch.Tensor, optional): The position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): The dimension to unsqueeze. Defaults to 1.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The query and key tensors after applying the RoPE operation.
    """
    if _TRITON_ENABLED and q.is_cuda:
        return RopeFunction.apply(q, k, cos, sin, position_ids, unsqueeze_dim)
    else:
        # PyTorch fallback - import here to avoid circular dependency
        from ..models.embeddings import apply_rotary_emb
        # apply_rotary_emb expects (B, H, S, D) and sequence_dim parameter
        q_out = apply_rotary_emb(q, (cos, sin), sequence_dim=2)
        k_out = apply_rotary_emb(k, (cos, sin), sequence_dim=2)
        return q_out, k_out


__all__ = [
    "TritonRMSNorm",
    "TritonGEGLUMLP",
    "triton_rotary_pos_emb",
]

