"""
Comprehensive tests for Triton kernels:
- Forward and backward pass validation
- Multiple dtype support (fp32, fp16, bf16)
- Various shape configurations
- Numerical accuracy checks against PyTorch reference implementations
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from src.diffusers.kernels.triton_ops import (
    RopeFunction,
    RMSNormFunction,
    LayerNormFunction,
    GELUMulFunction,
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def assert_verbose_allclose(tensor1, tensor2, rtol=1e-5, atol=1e-5, name=""):
    try:
        assert_close(tensor1, tensor2, rtol=rtol, atol=atol)
    except AssertionError as e:
        max_diff = (tensor1 - tensor2).abs().max().item()
        mean_diff = (tensor1 - tensor2).abs().mean().item()
        print(f"\n{name} Assertion Failed:")
        print(f"  Max difference: {max_diff}")
        print(f"  Mean difference: {mean_diff}")
        print(f"  Tensor1 stats: min={tensor1.min():.6f}, max={tensor1.max():.6f}, mean={tensor1.mean():.6f}")
        print(f"  Tensor2 stats: min={tensor2.min():.6f}, max={tensor2.max():.6f}, mean={tensor2.mean():.6f}")
        raise e

def torch_rope_reference(q, k, cos, sin):
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()
    batch_size, num_heads, seq_len, head_dim = q.shape
    if cos.shape[0] == 1:
        cos = cos.expand(batch_size, -1, -1)
        sin = sin.expand(batch_size, -1, -1)
    q_half_1 = q[..., : head_dim // 2]
    q_half_2 = q[..., head_dim // 2 :]
    k_half_1 = k[..., : head_dim // 2]
    k_half_2 = k[..., head_dim // 2 :]
    cos_expanded = cos.unsqueeze(1)
    sin_expanded = sin.unsqueeze(1)
    q_rotated_half_1 = q_half_1 * cos_expanded - q_half_2 * sin_expanded
    q_rotated_half_2 = q_half_1 * sin_expanded + q_half_2 * cos_expanded
    q_rotated = torch.cat([q_rotated_half_1, q_rotated_half_2], dim=-1)
    k_rotated_half_1 = k_half_1 * cos_expanded - k_half_2 * sin_expanded
    k_rotated_half_2 = k_half_1 * sin_expanded + k_half_2 * cos_expanded
    k_rotated = torch.cat([k_rotated_half_1, k_rotated_half_2], dim=-1)
    return q_rotated, k_rotated

@pytest.mark.parametrize(
    "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim",
    [
        (1, 8, 8, 128, 64),
        (2, 16, 16, 256, 128),
        (4, 24, 24, 512, 128),
        (2, 32, 8, 1024, 64),
        (1, 8, 1, 64, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_forward(batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    cos = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=dtype)
    q_triton, k_triton = RopeFunction.apply(q.clone(), k.clone(), cos, sin)
    q_torch, k_torch = torch_rope_reference(q.clone(), k.clone(), cos, sin)
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-5
    assert_verbose_allclose(q_triton, q_torch, rtol=rtol, atol=atol, name="RoPE Q")
    assert_verbose_allclose(k_triton, k_torch, rtol=rtol, atol=atol, name="RoPE K")

@pytest.mark.parametrize(
    "batch_size, num_q_heads, seq_len, head_dim",
    [
        (2, 8, 128, 64),
        (1, 16, 256, 128),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_backward(batch_size, num_q_heads, seq_len, head_dim, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    q = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, num_q_heads, seq_len, head_dim, device=device, dtype=dtype, requires_grad=True)
    cos = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=dtype)
    sin = torch.randn(1, seq_len, head_dim // 2, device=device, dtype=dtype)
    q_out, k_out = RopeFunction.apply(q, k, cos, sin)
    loss = (q_out.sum() + k_out.sum())
    loss.backward()
    assert q.grad is not None
    assert k.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()

def torch_rms_norm_reference(x, weight, eps=1e-6, offset=0.0):
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    return x_normalized * (weight + offset)

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (1, 128, 512),
        (2, 256, 1024),
        (4, 512, 2048),
        (8, 1024, 4096),
        (2, 64, 3072),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_rms_norm_forward(batch_size, seq_len, hidden_size, dtype, offset):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    eps = 1e-6
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    y_triton = RMSNormFunction.apply(x, weight, eps, offset)
    y_torch = torch_rms_norm_reference(x, weight, eps, offset)
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
    else:
        rtol, atol = 1e-5, 1e-5
    assert_verbose_allclose(y_triton, y_torch, rtol=rtol, atol=atol, name="RMSNorm forward")

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 128, 512),
        (1, 256, 1024),
        (4, 64, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rms_norm_backward(batch_size, seq_len, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    eps = 1e-6
    x_triton = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight_triton = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    y_triton = RMSNormFunction.apply(x_triton, weight_triton, eps, 0.0)
    loss_triton = y_triton.sum()
    loss_triton.backward()
    x_torch = x_triton.detach().clone().requires_grad_(True)
    weight_torch = weight_triton.detach().clone().requires_grad_(True)
    y_torch = torch_rms_norm_reference(x_torch, weight_torch, eps, 0.0)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    if dtype == torch.bfloat16:
        rtol, atol = 5e-2, 5e-2
        rtol_w, atol_w = 2e-1, 2e-1
    elif dtype == torch.float16:
        rtol, atol = 1e-2, 1e-2
        rtol_w, atol_w = 5e-2, 5e-2
    else:
        rtol, atol = 1e-4, 1e-4
        rtol_w, atol_w = 1e-4, 1e-4
    assert_verbose_allclose(x_triton.grad, x_torch.grad, rtol=rtol, atol=atol, name="RMSNorm dx")
    assert_verbose_allclose(weight_triton.grad, weight_torch.grad, rtol=rtol_w, atol=atol_w, name="RMSNorm dw")

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (1, 128, 512),
        (2, 256, 1024),
        (4, 512, 2048),
        (8, 1024, 4096),
        (2, 64, 768),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layer_norm_forward(batch_size, seq_len, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    eps = 1e-6
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    weight = torch.randn(hidden_size, device=device, dtype=dtype)
    bias = torch.randn(hidden_size, device=device, dtype=dtype)
    y_triton = LayerNormFunction.apply(x, weight, bias, eps)
    y_torch = F.layer_norm(x, (hidden_size,), weight, bias, eps)
    rtol, atol = (1e-2, 1e-2) if dtype in [torch.float16, torch.bfloat16] else (1e-5, 1e-5)
    assert_verbose_allclose(y_triton, y_torch, rtol=rtol, atol=atol, name="LayerNorm forward")

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 128, 512),
        (1, 256, 1024),
        (4, 64, 768),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_layer_norm_backward(batch_size, seq_len, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    eps = 1e-6
    x_triton = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)
    weight_triton = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    bias_triton = torch.randn(hidden_size, device=device, dtype=dtype, requires_grad=True)
    y_triton = LayerNormFunction.apply(x_triton, weight_triton, bias_triton, eps)
    loss_triton = y_triton.sum()
    loss_triton.backward()
    x_torch = x_triton.detach().clone().requires_grad_(True)
    weight_torch = weight_triton.detach().clone().requires_grad_(True)
    bias_torch = bias_triton.detach().clone().requires_grad_(True)
    y_torch = F.layer_norm(x_torch, (hidden_size,), weight_torch, bias_torch, eps)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    if dtype == torch.bfloat16:
        rtol_input, atol_input = 5e-2, 5e-2
        rtol_param, atol_param = 1e-1, 1e-1
    elif dtype == torch.float16:
        rtol_input, atol_input = 1e-2, 1e-2
        rtol_param, atol_param = 1e-1, 1e-1
    else:
        rtol_input, atol_input = 1e-4, 1e-4
        rtol_param, atol_param = 1e-3, 1e-3
    assert_verbose_allclose(x_triton.grad, x_torch.grad, rtol=rtol_input, atol=atol_input, name="LayerNorm dx")
    assert_verbose_allclose(weight_triton.grad, weight_torch.grad, rtol=rtol_param, atol=atol_param, name="LayerNorm dw")
    assert_verbose_allclose(bias_triton.grad, bias_torch.grad, rtol=rtol_param, atol=atol_param, name="LayerNorm db")

def torch_geglu_reference(a, b):
    return F.gelu(a, approximate='tanh') * b

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (1, 128, 512),
        (2, 256, 1024),
        (4, 512, 2048),
        (8, 64, 4096),
        (2, 256, 3072),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_geglu_forward(batch_size, seq_len, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    a = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    b = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    c_triton = GELUMulFunction.apply(a.clone(), b.clone())
    c_torch = torch_geglu_reference(a.clone(), b.clone())
    rtol, atol = (1e-2, 1e-2) if dtype in [torch.float16, torch.bfloat16] else (1e-5, 1e-5)
    assert_verbose_allclose(c_triton, c_torch, rtol=rtol, atol=atol, name="GeGLU forward")

@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 128, 512),
        (1, 256, 1024),
        (4, 64, 2048),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_geglu_backward(batch_size, seq_len, hidden_size, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    a_init = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    b_init = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    a_triton = a_init.clone().requires_grad_(True)
    b_triton = b_init.clone().requires_grad_(True)
    c_triton = GELUMulFunction.apply(a_triton, b_triton)
    loss_triton = c_triton.sum()
    loss_triton.backward()
    a_torch = a_init.clone().requires_grad_(True)
    b_torch = b_init.clone().requires_grad_(True)
    c_torch = torch_geglu_reference(a_torch, b_torch)
    loss_torch = c_torch.sum()
    loss_torch.backward()
    if dtype in [torch.float16, torch.bfloat16]:
        rtol, atol = 5e-2, 5e-2
    else:
        rtol, atol = 1e-3, 1e-3
    assert_verbose_allclose(a_triton.grad, a_torch.grad, rtol=rtol, atol=atol, name="GeGLU da")
    assert_verbose_allclose(b_triton.grad, b_torch.grad, rtol=rtol, atol=atol, name="GeGLU db")

def test_rms_norm_zero_input():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    x = torch.zeros(2, 128, 512, device=device)
    weight = torch.ones(512, device=device)
    eps = 1e-6
    y = RMSNormFunction.apply(x, weight, eps, 0.0)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

def test_layer_norm_zero_variance():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    x = torch.ones(2, 128, 512, device=device) * 5.0
    weight = torch.ones(512, device=device)
    bias = torch.zeros(512, device=device)
    eps = 1e-6
    y = LayerNormFunction.apply(x, weight, bias, eps)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()

@pytest.mark.parametrize("head_dim", [32, 64, 128, 256])
def test_rope_various_head_dims(head_dim):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    set_seed()
    device = "cuda"
    q = torch.randn(2, 8, 128, head_dim, device=device)
    k = torch.randn(2, 8, 128, head_dim, device=device)
    cos = torch.randn(1, 128, head_dim // 2, device=device)
    sin = torch.randn(1, 128, head_dim // 2, device=device)
    q_out, k_out = RopeFunction.apply(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.isnan(q_out).any()
    assert not torch.isnan(k_out).any()

@pytest.mark.benchmark
@pytest.mark.parametrize("hidden_size", [512, 1024, 2048, 4096])
def test_rms_norm_benchmark(benchmark, hidden_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    x = torch.randn(8, 256, hidden_size, device=device)
    weight = torch.randn(hidden_size, device=device)
    eps = 1e-6
    def run():
        y = RMSNormFunction.apply(x, weight, eps, 0.0)
        torch.cuda.synchronize()
        return y
    benchmark(run)

@pytest.mark.benchmark
@pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
def test_rope_benchmark(benchmark, seq_len):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = "cuda"
    q = torch.randn(2, 32, seq_len, 128, device=device)
    k = torch.randn(2, 32, seq_len, 128, device=device)
    cos = torch.randn(1, seq_len, 64, device=device)
    sin = torch.randn(1, seq_len, 64, device=device)
    def run():
        q_out, k_out = RopeFunction.apply(q, k, cos, sin)
        torch.cuda.synchronize()
        return q_out, k_out
    benchmark(run)

if __name__ == "__main__":
    print("Running quick sanity checks...")
    if torch.cuda.is_available():
        print("\nTesting RoPE...")
        test_rope_forward(2, 8, 8, 128, 64, torch.float32)
        print("RoPE forward passed")
        print("\nTesting RMSNorm...")
        test_rms_norm_forward(2, 128, 512, torch.float32, 0.0)
        print("RMSNorm forward passed")
        print("\nTesting LayerNorm...")
        test_layer_norm_forward(2, 128, 512, torch.float32)
        print("LayerNorm forward passed")
        print("\nTesting GeGLU...")
        test_geglu_forward(2, 128, 512, torch.float32)
        print("GeGLU forward passed")
        print("\nAll sanity checks passed!")
        print("\nRun full test suite with: pytest test_kernels.py -v")
        print("Run with coverage: pytest test_kernels.py -v --cov=triton_ops")
        print("Run benchmarks: pytest test_kernels.py -v -m benchmark")
    else:
        print("CUDA not available, skipping tests")
