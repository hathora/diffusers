# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import grid, split_scan_grid, grid_combo_kernels, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
import torch._inductor.kernel.mm_common

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()


# kernel path: /opt/inductor_cache/ve/cvezxwbsj6cs7i2u34gb53yyu7uxbc7vv2gzcgbudfok7icca7ef.py
# Topologically Sorted Source Nodes: [emb_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   emb_3 => cat
# Graph fragment:
#   %cat : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sin, %cos], -1), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = triton_helpers.div_floor_integer(ks0,  2)
    tmp4 = tmp0 < tmp3
    tmp7 = 1000.0
    tmp8 = tmp6 * tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp0.to(tl.float32)
    tmp11 = -9.210340371976184
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(((-1)*ks1) + (triton_helpers.div_floor_integer(ks0,  2)), [XBLOCK])
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 / tmp14
    tmp16 = tl_math.exp(tmp15)
    tmp17 = tmp9 * tmp16
    tmp18 = tl.broadcast_to(ks2, [XBLOCK])
    tmp19 = tmp18.to(tl.float32)
    tmp20 = tmp17 * tmp19
    tmp21 = tl_math.sin(tmp20)
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp4, tmp21, tmp22)
    tmp24 = tmp0 >= tmp3
    tmp25 = 2*(triton_helpers.div_floor_integer(ks0,  2))
    tmp26 = tmp0 < tmp25
    tmp27 = x0 + ((-1)*(triton_helpers.div_floor_integer(ks0,  2)))
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp11
    tmp30 = tmp29 / tmp14
    tmp31 = tl_math.exp(tmp30)
    tmp32 = tmp9 * tmp31
    tmp33 = tmp32 * tmp19
    tmp34 = tl_math.cos(tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp24, tmp34, tmp35)
    tmp37 = tl.where(tmp4, tmp23, tmp36)
    tl.store(out_ptr0 + (x0), tmp37, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/36/c36q4gtcegfhqdtofnfqzqrzlnfq5d2yf5jcnrdscrfqirs2b3bg.py
# Topologically Sorted Source Nodes: [emb_4, to_2], Original ATen: [aten.cat, aten._to_copy]
# Source node to ATen node mapping:
#   emb_4 => cat_1
#   to_2 => convert_element_type_6
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6], -1), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_cat_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*bf16', 2: 'i32', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_1', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = triton_helpers.div_floor_integer(ks0,  2)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((triton_helpers.div_floor_integer(ks0,  2)) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = 2*(triton_helpers.div_floor_integer(ks0,  2))
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + ((-1)*(triton_helpers.div_floor_integer(ks0,  2)))), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/n5/cn5g5zz72sicq65rwlpdm6irzpchj44jrgegrzdqdd5zmz2zqanw.py
# Topologically Sorted Source Nodes: [emb_4, to_2, sample_1], Original ATen: [aten.cat, aten._to_copy, aten.silu]
# Source node to ATen node mapping:
#   emb_4 => cat_1
#   sample_1 => convert_element_type_10, convert_element_type_11, mul_36, sigmoid
#   to_2 => convert_element_type_6
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_4, %slice_6], -1), kwargs = {})
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_1, torch.bfloat16), kwargs = {})
#   %mm_default_426 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_6, %permute_1), kwargs = {})
#   %add_tensor_426 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_426, %arg10_1), kwargs = {})
#   %convert_element_type_10 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_426, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_10,), kwargs = {})
#   %mul_36 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_10, %sigmoid), kwargs = {})
#   %convert_element_type_11 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_36, torch.bfloat16), kwargs = {})
triton_tem_fused__to_copy_cat_silu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused__to_copy_cat_silu_2', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, out_ptr1, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = False
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 3072
    K = 2*(triton_helpers.div_floor_integer(ks0,  2))
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 2*(triton_helpers.div_floor_integer(ks0,  2))
    stride_ak = 1
    stride_bk = 1
    stride_bn = 256

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(xindex, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp5, mask)
''', device_str='cuda')
meta0 = {'GROUP_M': 8, 'EVEN_K': False, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}


# kernel path: /opt/inductor_cache/ji/cjidaxru2mtihb4rhx7pv6kimo27y7dyzhdzq6hxcikfif7diwsj.py
# Topologically Sorted Source Nodes: [emb_8], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   emb_8 => cat_2
# Graph fragment:
#   %cat_2 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%sin_1, %cos_1], -1), kwargs = {})
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32', 4: 'i32', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp5 = tl.load(in_ptr0 + (0))
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = triton_helpers.div_floor_integer(ks0,  2)
    tmp4 = tmp0 < tmp3
    tmp7 = tmp6.to(tl.float32)
    tmp8 = 1000.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp0.to(tl.float32)
    tmp12 = -9.210340371976184
    tmp13 = tmp11 * tmp12
    tmp14 = tl.broadcast_to(((-1)*ks1) + (triton_helpers.div_floor_integer(ks0,  2)), [XBLOCK])
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tl_math.exp(tmp16)
    tmp18 = tmp10 * tmp17
    tmp19 = tl.broadcast_to(ks2, [XBLOCK])
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp18 * tmp20
    tmp22 = tl_math.sin(tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp4, tmp22, tmp23)
    tmp25 = tmp0 >= tmp3
    tmp26 = 2*(triton_helpers.div_floor_integer(ks0,  2))
    tmp27 = tmp0 < tmp26
    tmp28 = x0 + ((-1)*(triton_helpers.div_floor_integer(ks0,  2)))
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tmp29 * tmp12
    tmp31 = tmp30 / tmp15
    tmp32 = tl_math.exp(tmp31)
    tmp33 = tmp10 * tmp32
    tmp34 = tmp33 * tmp20
    tmp35 = tl_math.cos(tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp25, tmp35, tmp36)
    tmp38 = tl.where(tmp4, tmp24, tmp37)
    tl.store(out_ptr0 + (x0), tmp38, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/qh/cqhrdegqcasxrfsxsygw5z3cke47epb2sdz2g3a3dmx5kpxy57s3.py
# Topologically Sorted Source Nodes: [hidden_states_2], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   hidden_states_2 => convert_element_type_29, convert_element_type_30, mul_72, sigmoid_2
# Graph fragment:
#   %mm_default_422 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%arg8_1, %permute_5), kwargs = {})
#   %add_tensor_422 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_422, %arg18_1), kwargs = {})
#   %convert_element_type_29 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_tensor_422, torch.float32), kwargs = {})
#   %sigmoid_2 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_29,), kwargs = {})
#   %mul_72 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_29, %sigmoid_2), kwargs = {})
#   %convert_element_type_30 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_72, torch.bfloat16), kwargs = {})
triton_tem_fused_silu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=2,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_silu_4', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, in_ptr2, out_ptr1):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 3072
    K = 768
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 768
    stride_ak = 1
    stride_bk = 1
    stride_bn = 768

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(xindex, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp5, mask)
''', device_str='cuda')
meta1 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}


# kernel path: /opt/inductor_cache/lz/clzlidfkb7q6z5g4kmago5h7seyetlxr35p7katfd3nvexgi246o.py
# Topologically Sorted Source Nodes: [time_guidance_emb, conditioning, silu_3, silu_4, silu_5, silu_6, silu_7, silu_8, silu_9, silu_10, silu_11, silu_12, silu_13, silu_14, silu_15, silu_16, silu_17, silu_18, silu_19, silu_20, silu_21, silu_22, silu_23, silu_24, silu_25, silu_26, silu_27, silu_28, silu_29, silu_30, silu_31, silu_32, silu_33, silu_34, silu_35, silu_36, silu_37, silu_38, silu_39, silu_40, silu_41, silu_42, silu_43, silu_44, silu_45, silu_46, silu_47, silu_48, silu_49, silu_50, silu_51, silu_52, silu_53, silu_54, silu_55, silu_56, silu_57, silu_58, silu_59, silu_60, silu_61, silu_62, silu_63, silu_64, silu_65], Original ATen: [aten.add, aten.silu]
# Source node to ATen node mapping:
#   conditioning => add_73
#   silu_10 => convert_element_type_246, convert_element_type_247, mul_1267, sigmoid_10
#   silu_11 => convert_element_type_303, convert_element_type_304, mul_1614, sigmoid_11
#   silu_12 => convert_element_type_310, convert_element_type_311, mul_1617, sigmoid_12
#   silu_13 => convert_element_type_367, convert_element_type_368, mul_1964, sigmoid_13
#   silu_14 => convert_element_type_374, convert_element_type_375, mul_1967, sigmoid_14
#   silu_15 => convert_element_type_431, convert_element_type_432, mul_2314, sigmoid_15
#   silu_16 => convert_element_type_438, convert_element_type_439, mul_2317, sigmoid_16
#   silu_17 => convert_element_type_495, convert_element_type_496, mul_2664, sigmoid_17
#   silu_18 => convert_element_type_502, convert_element_type_503, mul_2667, sigmoid_18
#   silu_19 => convert_element_type_559, convert_element_type_560, mul_3014, sigmoid_19
#   silu_20 => convert_element_type_566, convert_element_type_567, mul_3017, sigmoid_20
#   silu_21 => convert_element_type_623, convert_element_type_624, mul_3364, sigmoid_21
#   silu_22 => convert_element_type_630, convert_element_type_631, mul_3367, sigmoid_22
#   silu_23 => convert_element_type_687, convert_element_type_688, mul_3714, sigmoid_23
#   silu_24 => convert_element_type_694, convert_element_type_695, mul_3717, sigmoid_24
#   silu_25 => convert_element_type_751, convert_element_type_752, mul_4064, sigmoid_25
#   silu_26 => convert_element_type_758, convert_element_type_759, mul_4067, sigmoid_26
#   silu_27 => convert_element_type_815, convert_element_type_816, mul_4414, sigmoid_27
#   silu_28 => convert_element_type_822, convert_element_type_823, mul_4417, sigmoid_28
#   silu_29 => convert_element_type_879, convert_element_type_880, mul_4764, sigmoid_29
#   silu_3 => convert_element_type_47, convert_element_type_48, mul_214, sigmoid_3
#   silu_30 => convert_element_type_886, convert_element_type_887, mul_4767, sigmoid_30
#   silu_31 => convert_element_type_943, convert_element_type_944, mul_5114, sigmoid_31
#   silu_32 => convert_element_type_950, convert_element_type_951, mul_5117, sigmoid_32
#   silu_33 => convert_element_type_1007, convert_element_type_1008, mul_5464, sigmoid_33
#   silu_34 => convert_element_type_1014, convert_element_type_1015, mul_5467, sigmoid_34
#   silu_35 => convert_element_type_1071, convert_element_type_1072, mul_5814, sigmoid_35
#   silu_36 => convert_element_type_1078, convert_element_type_1079, mul_5817, sigmoid_36
#   silu_37 => convert_element_type_1135, convert_element_type_1136, mul_6164, sigmoid_37
#   silu_38 => convert_element_type_1142, convert_element_type_1143, mul_6167, sigmoid_38
#   silu_39 => convert_element_type_1199, convert_element_type_1200, mul_6514, sigmoid_39
#   silu_4 => convert_element_type_54, convert_element_type_55, mul_217, sigmoid_4
#   silu_40 => convert_element_type_1206, convert_element_type_1207, mul_6517, sigmoid_40
#   silu_41 => convert_element_type_1263, convert_element_type_1264, mul_6867, sigmoid_41
#   silu_42 => convert_element_type_1293, convert_element_type_1294, mul_7166, sigmoid_42
#   silu_43 => convert_element_type_1323, convert_element_type_1324, mul_7465, sigmoid_43
#   silu_44 => convert_element_type_1353, convert_element_type_1354, mul_7764, sigmoid_44
#   silu_45 => convert_element_type_1383, convert_element_type_1384, mul_8063, sigmoid_45
#   silu_46 => convert_element_type_1413, convert_element_type_1414, mul_8362, sigmoid_46
#   silu_47 => convert_element_type_1443, convert_element_type_1444, mul_8661, sigmoid_47
#   silu_48 => convert_element_type_1473, convert_element_type_1474, mul_8960, sigmoid_48
#   silu_49 => convert_element_type_1503, convert_element_type_1504, mul_9259, sigmoid_49
#   silu_5 => convert_element_type_111, convert_element_type_112, mul_564, sigmoid_5
#   silu_50 => convert_element_type_1533, convert_element_type_1534, mul_9558, sigmoid_50
#   silu_51 => convert_element_type_1563, convert_element_type_1564, mul_9857, sigmoid_51
#   silu_52 => convert_element_type_1593, convert_element_type_1594, mul_10156, sigmoid_52
#   silu_53 => convert_element_type_1623, convert_element_type_1624, mul_10455, sigmoid_53
#   silu_54 => convert_element_type_1653, convert_element_type_1654, mul_10754, sigmoid_54
#   silu_55 => convert_element_type_1683, convert_element_type_1684, mul_11053, sigmoid_55
#   silu_56 => convert_element_type_1713, convert_element_type_1714, mul_11352, sigmoid_56
#   silu_57 => convert_element_type_1743, convert_element_type_1744, mul_11651, sigmoid_57
#   silu_58 => convert_element_type_1773, convert_element_type_1774, mul_11950, sigmoid_58
#   silu_59 => convert_element_type_1803, convert_element_type_1804, mul_12249, sigmoid_59
#   silu_6 => convert_element_type_118, convert_element_type_119, mul_567, sigmoid_6
#   silu_60 => convert_element_type_1833, convert_element_type_1834, mul_12548, sigmoid_60
#   silu_61 => convert_element_type_1863, convert_element_type_1864, mul_12847, sigmoid_61
#   silu_62 => convert_element_type_1893, convert_element_type_1894, mul_13146, sigmoid_62
#   silu_63 => convert_element_type_1923, convert_element_type_1924, mul_13445, sigmoid_63
#   silu_64 => convert_element_type_1953, convert_element_type_1954, mul_13744, sigmoid_64
#   silu_65 => convert_element_type_1983, convert_element_type_1984, mul_14043, sigmoid_65
#   silu_7 => convert_element_type_175, convert_element_type_176, mul_914, sigmoid_7
#   silu_8 => convert_element_type_182, convert_element_type_183, mul_917, sigmoid_8
#   silu_9 => convert_element_type_239, convert_element_type_240, mul_1264, sigmoid_9
#   time_guidance_emb => add_72
# Graph fragment:
#   %add_tensor_425 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_425, %arg12_1), kwargs = {})
#   %add_tensor_423 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_423, %arg16_1), kwargs = {})
#   %add_72 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_tensor_425, %add_tensor_423), kwargs = {})
#   %add_tensor_421 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_421, %arg20_1), kwargs = {})
#   %add_73 : [num_users=77] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_72, %add_tensor_421), kwargs = {})
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_47,), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %sigmoid_3), kwargs = {})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_214, torch.bfloat16), kwargs = {})
#   %convert_element_type_54 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_54,), kwargs = {})
#   %mul_217 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_54, %sigmoid_4), kwargs = {})
#   %convert_element_type_55 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_217, torch.bfloat16), kwargs = {})
#   %convert_element_type_111 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_5 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_111,), kwargs = {})
#   %mul_564 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_111, %sigmoid_5), kwargs = {})
#   %convert_element_type_112 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_564, torch.bfloat16), kwargs = {})
#   %convert_element_type_118 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_118,), kwargs = {})
#   %mul_567 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_118, %sigmoid_6), kwargs = {})
#   %convert_element_type_119 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_567, torch.bfloat16), kwargs = {})
#   %convert_element_type_175 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_7 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_175,), kwargs = {})
#   %mul_914 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_175, %sigmoid_7), kwargs = {})
#   %convert_element_type_176 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_914, torch.bfloat16), kwargs = {})
#   %convert_element_type_182 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_182,), kwargs = {})
#   %mul_917 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_182, %sigmoid_8), kwargs = {})
#   %convert_element_type_183 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_917, torch.bfloat16), kwargs = {})
#   %convert_element_type_239 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_239,), kwargs = {})
#   %mul_1264 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_239, %sigmoid_9), kwargs = {})
#   %convert_element_type_240 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1264, torch.bfloat16), kwargs = {})
#   %convert_element_type_246 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_10 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_246,), kwargs = {})
#   %mul_1267 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_246, %sigmoid_10), kwargs = {})
#   %convert_element_type_247 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1267, torch.bfloat16), kwargs = {})
#   %convert_element_type_303 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_11 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_303,), kwargs = {})
#   %mul_1614 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_303, %sigmoid_11), kwargs = {})
#   %convert_element_type_304 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1614, torch.bfloat16), kwargs = {})
#   %convert_element_type_310 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_310,), kwargs = {})
#   %mul_1617 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_310, %sigmoid_12), kwargs = {})
#   %convert_element_type_311 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1617, torch.bfloat16), kwargs = {})
#   %convert_element_type_367 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_367,), kwargs = {})
#   %mul_1964 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_367, %sigmoid_13), kwargs = {})
#   %convert_element_type_368 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1964, torch.bfloat16), kwargs = {})
#   %convert_element_type_374 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_374,), kwargs = {})
#   %mul_1967 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_374, %sigmoid_14), kwargs = {})
#   %convert_element_type_375 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1967, torch.bfloat16), kwargs = {})
#   %convert_element_type_431 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_15 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_431,), kwargs = {})
#   %mul_2314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_431, %sigmoid_15), kwargs = {})
#   %convert_element_type_432 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2314, torch.bfloat16), kwargs = {})
#   %convert_element_type_438 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_438,), kwargs = {})
#   %mul_2317 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_438, %sigmoid_16), kwargs = {})
#   %convert_element_type_439 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2317, torch.bfloat16), kwargs = {})
#   %convert_element_type_495 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_495,), kwargs = {})
#   %mul_2664 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_495, %sigmoid_17), kwargs = {})
#   %convert_element_type_496 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2664, torch.bfloat16), kwargs = {})
#   %convert_element_type_502 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_18 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_502,), kwargs = {})
#   %mul_2667 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_502, %sigmoid_18), kwargs = {})
#   %convert_element_type_503 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2667, torch.bfloat16), kwargs = {})
#   %convert_element_type_559 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_19 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_559,), kwargs = {})
#   %mul_3014 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_559, %sigmoid_19), kwargs = {})
#   %convert_element_type_560 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3014, torch.bfloat16), kwargs = {})
#   %convert_element_type_566 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_566,), kwargs = {})
#   %mul_3017 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_566, %sigmoid_20), kwargs = {})
#   %convert_element_type_567 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3017, torch.bfloat16), kwargs = {})
#   %convert_element_type_623 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_623,), kwargs = {})
#   %mul_3364 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_623, %sigmoid_21), kwargs = {})
#   %convert_element_type_624 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3364, torch.bfloat16), kwargs = {})
#   %convert_element_type_630 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_630,), kwargs = {})
#   %mul_3367 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_630, %sigmoid_22), kwargs = {})
#   %convert_element_type_631 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3367, torch.bfloat16), kwargs = {})
#   %convert_element_type_687 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_23 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_687,), kwargs = {})
#   %mul_3714 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_687, %sigmoid_23), kwargs = {})
#   %convert_element_type_688 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3714, torch.bfloat16), kwargs = {})
#   %convert_element_type_694 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_694,), kwargs = {})
#   %mul_3717 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_694, %sigmoid_24), kwargs = {})
#   %convert_element_type_695 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_3717, torch.bfloat16), kwargs = {})
#   %convert_element_type_751 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_25 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_751,), kwargs = {})
#   %mul_4064 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_751, %sigmoid_25), kwargs = {})
#   %convert_element_type_752 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4064, torch.bfloat16), kwargs = {})
#   %convert_element_type_758 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_26 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_758,), kwargs = {})
#   %mul_4067 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_758, %sigmoid_26), kwargs = {})
#   %convert_element_type_759 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4067, torch.bfloat16), kwargs = {})
#   %convert_element_type_815 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_27 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_815,), kwargs = {})
#   %mul_4414 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_815, %sigmoid_27), kwargs = {})
#   %convert_element_type_816 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4414, torch.bfloat16), kwargs = {})
#   %convert_element_type_822 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_822,), kwargs = {})
#   %mul_4417 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_822, %sigmoid_28), kwargs = {})
#   %convert_element_type_823 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4417, torch.bfloat16), kwargs = {})
#   %convert_element_type_879 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_29 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_879,), kwargs = {})
#   %mul_4764 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_879, %sigmoid_29), kwargs = {})
#   %convert_element_type_880 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4764, torch.bfloat16), kwargs = {})
#   %convert_element_type_886 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_30 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_886,), kwargs = {})
#   %mul_4767 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_886, %sigmoid_30), kwargs = {})
#   %convert_element_type_887 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4767, torch.bfloat16), kwargs = {})
#   %convert_element_type_943 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_31 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_943,), kwargs = {})
#   %mul_5114 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_943, %sigmoid_31), kwargs = {})
#   %convert_element_type_944 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5114, torch.bfloat16), kwargs = {})
#   %convert_element_type_950 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_32 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_950,), kwargs = {})
#   %mul_5117 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_950, %sigmoid_32), kwargs = {})
#   %convert_element_type_951 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5117, torch.bfloat16), kwargs = {})
#   %convert_element_type_1007 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_33 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1007,), kwargs = {})
#   %mul_5464 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1007, %sigmoid_33), kwargs = {})
#   %convert_element_type_1008 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5464, torch.bfloat16), kwargs = {})
#   %convert_element_type_1014 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_34 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1014,), kwargs = {})
#   %mul_5467 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1014, %sigmoid_34), kwargs = {})
#   %convert_element_type_1015 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5467, torch.bfloat16), kwargs = {})
#   %convert_element_type_1071 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_35 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1071,), kwargs = {})
#   %mul_5814 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1071, %sigmoid_35), kwargs = {})
#   %convert_element_type_1072 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5814, torch.bfloat16), kwargs = {})
#   %convert_element_type_1078 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_36 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1078,), kwargs = {})
#   %mul_5817 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1078, %sigmoid_36), kwargs = {})
#   %convert_element_type_1079 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_5817, torch.bfloat16), kwargs = {})
#   %convert_element_type_1135 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_37 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1135,), kwargs = {})
#   %mul_6164 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1135, %sigmoid_37), kwargs = {})
#   %convert_element_type_1136 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6164, torch.bfloat16), kwargs = {})
#   %convert_element_type_1142 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_38 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1142,), kwargs = {})
#   %mul_6167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1142, %sigmoid_38), kwargs = {})
#   %convert_element_type_1143 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6167, torch.bfloat16), kwargs = {})
#   %convert_element_type_1199 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_39 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1199,), kwargs = {})
#   %mul_6514 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1199, %sigmoid_39), kwargs = {})
#   %convert_element_type_1200 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6514, torch.bfloat16), kwargs = {})
#   %convert_element_type_1206 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_40 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1206,), kwargs = {})
#   %mul_6517 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1206, %sigmoid_40), kwargs = {})
#   %convert_element_type_1207 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6517, torch.bfloat16), kwargs = {})
#   %convert_element_type_1263 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1263,), kwargs = {})
#   %mul_6867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1263, %sigmoid_41), kwargs = {})
#   %convert_element_type_1264 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6867, torch.bfloat16), kwargs = {})
#   %convert_element_type_1293 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_42 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1293,), kwargs = {})
#   %mul_7166 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1293, %sigmoid_42), kwargs = {})
#   %convert_element_type_1294 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7166, torch.bfloat16), kwargs = {})
#   %convert_element_type_1323 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_43 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1323,), kwargs = {})
#   %mul_7465 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1323, %sigmoid_43), kwargs = {})
#   %convert_element_type_1324 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7465, torch.bfloat16), kwargs = {})
#   %convert_element_type_1353 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_44 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1353,), kwargs = {})
#   %mul_7764 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1353, %sigmoid_44), kwargs = {})
#   %convert_element_type_1354 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7764, torch.bfloat16), kwargs = {})
#   %convert_element_type_1383 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_45 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1383,), kwargs = {})
#   %mul_8063 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1383, %sigmoid_45), kwargs = {})
#   %convert_element_type_1384 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8063, torch.bfloat16), kwargs = {})
#   %convert_element_type_1413 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_46 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1413,), kwargs = {})
#   %mul_8362 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1413, %sigmoid_46), kwargs = {})
#   %convert_element_type_1414 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8362, torch.bfloat16), kwargs = {})
#   %convert_element_type_1443 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_47 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1443,), kwargs = {})
#   %mul_8661 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1443, %sigmoid_47), kwargs = {})
#   %convert_element_type_1444 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8661, torch.bfloat16), kwargs = {})
#   %convert_element_type_1473 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_48 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1473,), kwargs = {})
#   %mul_8960 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1473, %sigmoid_48), kwargs = {})
#   %convert_element_type_1474 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8960, torch.bfloat16), kwargs = {})
#   %convert_element_type_1503 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_49 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1503,), kwargs = {})
#   %mul_9259 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1503, %sigmoid_49), kwargs = {})
#   %convert_element_type_1504 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9259, torch.bfloat16), kwargs = {})
#   %convert_element_type_1533 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_50 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1533,), kwargs = {})
#   %mul_9558 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1533, %sigmoid_50), kwargs = {})
#   %convert_element_type_1534 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9558, torch.bfloat16), kwargs = {})
#   %convert_element_type_1563 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_51 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1563,), kwargs = {})
#   %mul_9857 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1563, %sigmoid_51), kwargs = {})
#   %convert_element_type_1564 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9857, torch.bfloat16), kwargs = {})
#   %convert_element_type_1593 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_52 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1593,), kwargs = {})
#   %mul_10156 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1593, %sigmoid_52), kwargs = {})
#   %convert_element_type_1594 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10156, torch.bfloat16), kwargs = {})
#   %convert_element_type_1623 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_53 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1623,), kwargs = {})
#   %mul_10455 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1623, %sigmoid_53), kwargs = {})
#   %convert_element_type_1624 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10455, torch.bfloat16), kwargs = {})
#   %convert_element_type_1653 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_54 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1653,), kwargs = {})
#   %mul_10754 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1653, %sigmoid_54), kwargs = {})
#   %convert_element_type_1654 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_10754, torch.bfloat16), kwargs = {})
#   %convert_element_type_1683 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_55 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1683,), kwargs = {})
#   %mul_11053 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1683, %sigmoid_55), kwargs = {})
#   %convert_element_type_1684 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11053, torch.bfloat16), kwargs = {})
#   %convert_element_type_1713 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_56 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1713,), kwargs = {})
#   %mul_11352 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1713, %sigmoid_56), kwargs = {})
#   %convert_element_type_1714 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11352, torch.bfloat16), kwargs = {})
#   %convert_element_type_1743 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_57 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1743,), kwargs = {})
#   %mul_11651 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1743, %sigmoid_57), kwargs = {})
#   %convert_element_type_1744 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11651, torch.bfloat16), kwargs = {})
#   %convert_element_type_1773 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_58 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1773,), kwargs = {})
#   %mul_11950 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1773, %sigmoid_58), kwargs = {})
#   %convert_element_type_1774 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_11950, torch.bfloat16), kwargs = {})
#   %convert_element_type_1803 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_59 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1803,), kwargs = {})
#   %mul_12249 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1803, %sigmoid_59), kwargs = {})
#   %convert_element_type_1804 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12249, torch.bfloat16), kwargs = {})
#   %convert_element_type_1833 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_60 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1833,), kwargs = {})
#   %mul_12548 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1833, %sigmoid_60), kwargs = {})
#   %convert_element_type_1834 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12548, torch.bfloat16), kwargs = {})
#   %convert_element_type_1863 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_61 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1863,), kwargs = {})
#   %mul_12847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1863, %sigmoid_61), kwargs = {})
#   %convert_element_type_1864 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_12847, torch.bfloat16), kwargs = {})
#   %convert_element_type_1893 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_62 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1893,), kwargs = {})
#   %mul_13146 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1893, %sigmoid_62), kwargs = {})
#   %convert_element_type_1894 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13146, torch.bfloat16), kwargs = {})
#   %convert_element_type_1923 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_63 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1923,), kwargs = {})
#   %mul_13445 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1923, %sigmoid_63), kwargs = {})
#   %convert_element_type_1924 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13445, torch.bfloat16), kwargs = {})
#   %convert_element_type_1953 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_64 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1953,), kwargs = {})
#   %mul_13744 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1953, %sigmoid_64), kwargs = {})
#   %convert_element_type_1954 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_13744, torch.bfloat16), kwargs = {})
#   %convert_element_type_1983 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_65 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1983,), kwargs = {})
#   %mul_14043 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1983, %sigmoid_65), kwargs = {})
#   %convert_element_type_1984 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14043, torch.bfloat16), kwargs = {})
triton_poi_fused_add_silu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: '*bf16', 15: '*bf16', 16: '*bf16', 17: '*bf16', 18: '*bf16', 19: '*bf16', 20: '*bf16', 21: '*bf16', 22: '*bf16', 23: '*bf16', 24: '*bf16', 25: '*bf16', 26: '*bf16', 27: '*bf16', 28: '*bf16', 29: '*bf16', 30: '*bf16', 31: '*bf16', 32: '*bf16', 33: '*bf16', 34: '*bf16', 35: '*bf16', 36: '*bf16', 37: '*bf16', 38: '*bf16', 39: '*bf16', 40: '*bf16', 41: '*bf16', 42: '*bf16', 43: '*bf16', 44: '*bf16', 45: '*bf16', 46: '*bf16', 47: '*bf16', 48: '*bf16', 49: '*bf16', 50: '*bf16', 51: '*bf16', 52: '*bf16', 53: '*bf16', 54: '*bf16', 55: '*bf16', 56: '*bf16', 57: '*bf16', 58: '*bf16', 59: '*bf16', 60: '*bf16', 61: '*bf16', 62: '*bf16', 63: '*bf16', 64: '*bf16', 65: '*bf16', 66: '*bf16', 67: '*bf16', 68: '*bf16', 69: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_silu_5', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr30, out_ptr31, out_ptr32, out_ptr33, out_ptr34, out_ptr35, out_ptr36, out_ptr37, out_ptr38, out_ptr39, out_ptr40, out_ptr41, out_ptr42, out_ptr43, out_ptr44, out_ptr45, out_ptr46, out_ptr47, out_ptr48, out_ptr49, out_ptr50, out_ptr51, out_ptr52, out_ptr53, out_ptr54, out_ptr55, out_ptr56, out_ptr57, out_ptr58, out_ptr59, out_ptr60, out_ptr61, out_ptr62, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x0), xmask).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr0 + (x0), tmp14, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    tl.store(out_ptr3 + (x0), tmp14, xmask)
    tl.store(out_ptr4 + (x0), tmp14, xmask)
    tl.store(out_ptr5 + (x0), tmp14, xmask)
    tl.store(out_ptr6 + (x0), tmp14, xmask)
    tl.store(out_ptr7 + (x0), tmp14, xmask)
    tl.store(out_ptr8 + (x0), tmp14, xmask)
    tl.store(out_ptr9 + (x0), tmp14, xmask)
    tl.store(out_ptr10 + (x0), tmp14, xmask)
    tl.store(out_ptr11 + (x0), tmp14, xmask)
    tl.store(out_ptr12 + (x0), tmp14, xmask)
    tl.store(out_ptr13 + (x0), tmp14, xmask)
    tl.store(out_ptr14 + (x0), tmp14, xmask)
    tl.store(out_ptr15 + (x0), tmp14, xmask)
    tl.store(out_ptr16 + (x0), tmp14, xmask)
    tl.store(out_ptr17 + (x0), tmp14, xmask)
    tl.store(out_ptr18 + (x0), tmp14, xmask)
    tl.store(out_ptr19 + (x0), tmp14, xmask)
    tl.store(out_ptr20 + (x0), tmp14, xmask)
    tl.store(out_ptr21 + (x0), tmp14, xmask)
    tl.store(out_ptr22 + (x0), tmp14, xmask)
    tl.store(out_ptr23 + (x0), tmp14, xmask)
    tl.store(out_ptr24 + (x0), tmp14, xmask)
    tl.store(out_ptr25 + (x0), tmp14, xmask)
    tl.store(out_ptr26 + (x0), tmp14, xmask)
    tl.store(out_ptr27 + (x0), tmp14, xmask)
    tl.store(out_ptr28 + (x0), tmp14, xmask)
    tl.store(out_ptr29 + (x0), tmp14, xmask)
    tl.store(out_ptr30 + (x0), tmp14, xmask)
    tl.store(out_ptr31 + (x0), tmp14, xmask)
    tl.store(out_ptr32 + (x0), tmp14, xmask)
    tl.store(out_ptr33 + (x0), tmp14, xmask)
    tl.store(out_ptr34 + (x0), tmp14, xmask)
    tl.store(out_ptr35 + (x0), tmp14, xmask)
    tl.store(out_ptr36 + (x0), tmp14, xmask)
    tl.store(out_ptr37 + (x0), tmp14, xmask)
    tl.store(out_ptr38 + (x0), tmp14, xmask)
    tl.store(out_ptr39 + (x0), tmp14, xmask)
    tl.store(out_ptr40 + (x0), tmp14, xmask)
    tl.store(out_ptr41 + (x0), tmp14, xmask)
    tl.store(out_ptr42 + (x0), tmp14, xmask)
    tl.store(out_ptr43 + (x0), tmp14, xmask)
    tl.store(out_ptr44 + (x0), tmp14, xmask)
    tl.store(out_ptr45 + (x0), tmp14, xmask)
    tl.store(out_ptr46 + (x0), tmp14, xmask)
    tl.store(out_ptr47 + (x0), tmp14, xmask)
    tl.store(out_ptr48 + (x0), tmp14, xmask)
    tl.store(out_ptr49 + (x0), tmp14, xmask)
    tl.store(out_ptr50 + (x0), tmp14, xmask)
    tl.store(out_ptr51 + (x0), tmp14, xmask)
    tl.store(out_ptr52 + (x0), tmp14, xmask)
    tl.store(out_ptr53 + (x0), tmp14, xmask)
    tl.store(out_ptr54 + (x0), tmp14, xmask)
    tl.store(out_ptr55 + (x0), tmp14, xmask)
    tl.store(out_ptr56 + (x0), tmp14, xmask)
    tl.store(out_ptr57 + (x0), tmp14, xmask)
    tl.store(out_ptr58 + (x0), tmp14, xmask)
    tl.store(out_ptr59 + (x0), tmp14, xmask)
    tl.store(out_ptr60 + (x0), tmp14, xmask)
    tl.store(out_ptr61 + (x0), tmp14, xmask)
    tl.store(out_ptr62 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/5o/c5o3a6fh556booe3n2wnednopsu7ocxx2o7n5ofhncnzt6ikl4o7.py
# Topologically Sorted Source Nodes: [silu_3], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_3 => convert_element_type_47, convert_element_type_48, mul_214, sigmoid_3
# Graph fragment:
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_3 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_47,), kwargs = {})
#   %mul_214 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_47, %sigmoid_3), kwargs = {})
#   %convert_element_type_48 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_214, torch.bfloat16), kwargs = {})
#   %mm_default_420 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_48, %permute_8), kwargs = {})
triton_tem_fused_silu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_silu_6', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 18432
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (18432*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), acc, mask)
''', device_str='cuda')
meta2 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}


# kernel path: /opt/inductor_cache/lp/clpfyotlswvoubdkobe3iywwsowaoduasdi5aqwwpoqarixoij6s.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_419 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view, %permute), kwargs = {})
triton_tem_fused_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_7', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 32
    A = arg_A
    B = arg_B

    M = 4096
    N = 3072
    K = 64
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 64
    stride_ak = 1
    stride_bk = 1
    stride_bn = 64

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')
meta3 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}


# kernel path: /opt/inductor_cache/nj/cnjuhvbe2yvymh3eduzcjh2ajekconyfxj4pvdepe5hlupfskpuk.py
# Topologically Sorted Source Nodes: [layer_norm, add_2, mul_11, x], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_2 => add_259
#   layer_norm => add_258, convert_element_type_52, convert_element_type_53, mul_215, rsqrt, sub_94, var_mean
#   mul_11 => mul_216
#   x => add_260
# Graph fragment:
#   %convert_element_type_52 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_52, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_94 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_1, %getitem_7), kwargs = {})
#   %add_258 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_258,), kwargs = {})
#   %mul_215 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_94, %rsqrt), kwargs = {})
#   %convert_element_type_53 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_215, torch.bfloat16), kwargs = {})
#   %add_259 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_10, 1), kwargs = {})
#   %mul_216 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_53, %add_259), kwargs = {})
#   %add_260 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_216, %unsqueeze_11), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_8', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(rmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp5
        tmp13 = 3072.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-06
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = 1.0
        tmp24 = tmp22 + tmp23
        tmp25 = tmp19 * tmp24
        tmp28 = tmp26 + tmp27
        tmp29 = tmp25 + tmp28
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp29, rmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/5r/c5r6u5tbqakovlyox4kaufp57lh4hlyozsm4lkrj2hqoemsl6npt.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_417 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_2, %permute_7), kwargs = {})
triton_tem_fused_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_9', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = ks0
    N = 3072
    K = 4096
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 4096
    stride_ak = 1
    stride_bk = 1
    stride_bn = 4096

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')
meta4 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}


# kernel path: /opt/inductor_cache/js/cjst2fupnj25vtu4nnevhnrjje3c36yzgaclppoatv45vyse2bp5.py
# Topologically Sorted Source Nodes: [layer_norm_1, add_4, mul_12, x_1], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_4 => add_271
#   layer_norm_1 => add_261, convert_element_type_59, convert_element_type_60, mul_218, rsqrt_1, sub_95, var_mean_1
#   mul_12 => mul_226
#   x_1 => add_275
# Graph fragment:
#   %convert_element_type_59 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_3, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_59, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_95 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%view_3, %getitem_15), kwargs = {})
#   %add_261 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_261,), kwargs = {})
#   %mul_218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_95, %rsqrt_1), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_218, torch.bfloat16), kwargs = {})
#   %add_271 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_12, 1), kwargs = {})
#   %mul_226 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_60, %add_271), kwargs = {})
#   %add_275 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_226, %unsqueeze_13), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_10', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(rmask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(rmask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(rmask & xmask, tmp5_weight_next, tmp5_weight)
    tmp5_tmp, tmp6_tmp, tmp7_tmp = triton_helpers.welford(
        tmp5_mean, tmp5_m2, tmp5_weight, 1
    )
    tmp5 = tmp5_tmp[:, None]
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr2 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr3 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp11 - tmp5
        tmp13 = 3072.0
        tmp14 = tmp6 / tmp13
        tmp15 = 1e-06
        tmp16 = tmp14 + tmp15
        tmp17 = libdevice.rsqrt(tmp16)
        tmp18 = tmp12 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = 1.0
        tmp24 = tmp22 + tmp23
        tmp25 = tmp19 * tmp24
        tmp28 = tmp26 + tmp27
        tmp29 = tmp25 + tmp28
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp29, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/az/cazfq2gp3twcfaicjeuqtlm6vsvscdlwtamqup3jbphphzc6wwux.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_416 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_19, %permute_13), kwargs = {})
triton_tem_fused_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_11', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = ks0
    N = 3072
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/pz/cpznml3oyu7tbn6gsmmr2sp2oh4c6lj3k3hxmjx7apwuatfv7asi.py
# Topologically Sorted Source Nodes: [pow_6, variance_2], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   pow_6 => pow_6
#   variance_2 => mean_2
# Graph fragment:
#   %pow_6 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_28, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_6, [-1], True), kwargs = {})
triton_per_fused_mean_pow_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_12', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), xmask, other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/6e/c6enlnrynjsiytpvfrwupchvhqieecw255lhus6xgldihqvlgtit.py
# Topologically Sorted Source Nodes: [pow_4, variance], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   pow_4 => pow_4
#   variance => mean
# Graph fragment:
#   %pow_4 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_25, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_4, [-1], True), kwargs = {})
triton_per_fused_mean_pow_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.persistent_reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_pow_13', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rindex = tl.arange(0, RBLOCK)[None, :]
    roffset = 0
    rmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    r2 = rindex
    x3 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_ptr0 + (r2 + (128*x3)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp7, None)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/hy/chyieuxy32gjmwnssevdufabejvnnfy52o5w53hl5p6hh26fvs4l.py
# Topologically Sorted Source Nodes: [query_3], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   query_3 => cat_7
# Graph fragment:
#   %cat_7 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%mul_285, %mul_257], 1), kwargs = {})
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*fp32', 7: '*bf16', 8: '*bf16', 9: 'i32', 10: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 10), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 3072)
    x3 = xindex % 3072
    x1 = (xindex // 128) % 24
    x0 = xindex % 128
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (3072*x2)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x3), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x1 + (24*x2)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 128.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1e-06
    tmp13 = tmp11 + tmp12
    tmp14 = libdevice.rsqrt(tmp13)
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = 0.0
    tmp18 = tmp16 + tmp17
    tmp19 = tmp15 * tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp4, tmp19, tmp20)
    tmp22 = tmp0 >= tmp3
    tmp23 = 4096 + ks0
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr4 + (x3 + (3072*(x2 + ((-1)*ks0)))), tmp22 & xmask, other=0.0).to(tl.float32)
    tmp26 = tl.load(in_ptr5 + (x3), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.load(in_ptr6 + (x1 + (24*(x2 + ((-1)*ks0)))), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp28 / tmp9
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30 + tmp12
    tmp32 = libdevice.rsqrt(tmp31)
    tmp33 = tmp27 * tmp32
    tmp34 = tl.load(in_ptr7 + (x0), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp34 + tmp17
    tmp36 = tmp33 * tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp22, tmp36, tmp37)
    tmp39 = tl.where(tmp4, tmp21, tmp38)
    tl.store(out_ptr0 + (x4), tmp39, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/7n/c7nstbowtgi55jtwomxgxgq6lxqdhex7xlu6bo3bor2b7w3fuzu2.py
# Topologically Sorted Source Nodes: [freqs_cos, freqs_sin], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   freqs_cos => convert_element_type_39
#   freqs_sin => convert_element_type_40
# Graph fragment:
#   %convert_element_type_39 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_5, torch.float32), kwargs = {})
#   %convert_element_type_40 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_6, torch.float32), kwargs = {})
triton_poi_fused__to_copy_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[131072], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_15', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16)
    x0 = xindex % 16
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (3*x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = 4096 + ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (3*(x1 + ((-1)*ks0))), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11.to(tl.float64)
    tmp13 = 1.00000000000000*(ks1.to(tl.float64))
    tmp14 = tmp13.to(tl.float64)
    tmp15 = 2*(x0 // 2)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = tl.full([1], 0.0625, tl.float64)
    tmp18 = tmp16 * tmp17
    tmp19 = libdevice.pow(tmp14, tmp18)
    tmp20 = tl.full([1], 1, tl.int32)
    tmp21 = tmp20 / tmp19
    tmp22 = tl.full([1], 1.0, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tmp12 * tmp24
    tmp26 = libdevice.cos(tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = libdevice.sin(tmp25)
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp27, xmask)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/kl/cklphsvc3zsl5m4m4ll5si7xwrq44dqanqyfzacvth72gnh7fvu7.py
# Topologically Sorted Source Nodes: [freqs_cos_1, freqs_sin_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   freqs_cos_1 => convert_element_type_42
#   freqs_sin_1 => convert_element_type_43
# Graph fragment:
#   %convert_element_type_42 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_8, torch.float32), kwargs = {})
#   %convert_element_type_43 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.float32), kwargs = {})
triton_poi_fused__to_copy_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_16', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56)
    x0 = xindex % 56
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1 + (3*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = 4096 + ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (1 + (3*(x1 + ((-1)*ks0)))), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11.to(tl.float64)
    tmp13 = 1.00000000000000*(ks1.to(tl.float64))
    tmp14 = tmp13.to(tl.float64)
    tmp15 = 2*(x0 // 2)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = tl.full([1], 0.017857142857142856, tl.float64)
    tmp18 = tmp16 * tmp17
    tmp19 = libdevice.pow(tmp14, tmp18)
    tmp20 = tl.full([1], 1, tl.int32)
    tmp21 = tmp20 / tmp19
    tmp22 = tl.full([1], 1.0, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tmp12 * tmp24
    tmp26 = libdevice.cos(tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = libdevice.sin(tmp25)
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp27, xmask)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/rt/crtwdaopdwkm5bssimj4iwdw3wbqisq442r4env5easwkdgblfkf.py
# Topologically Sorted Source Nodes: [freqs_cos_2, freqs_sin_2], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   freqs_cos_2 => convert_element_type_45
#   freqs_sin_2 => convert_element_type_46
# Graph fragment:
#   %convert_element_type_45 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_11, torch.float32), kwargs = {})
#   %convert_element_type_46 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.float32), kwargs = {})
triton_poi_fused__to_copy_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_17', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr1, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56)
    x0 = xindex % 56
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2 + (3*x1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = 4096 + ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (2 + (3*(x1 + ((-1)*ks0)))), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp11.to(tl.float64)
    tmp13 = 1.00000000000000*(ks1.to(tl.float64))
    tmp14 = tmp13.to(tl.float64)
    tmp15 = 2*(x0 // 2)
    tmp16 = tmp15.to(tl.float64)
    tmp17 = tl.full([1], 0.017857142857142856, tl.float64)
    tmp18 = tmp16 * tmp17
    tmp19 = libdevice.pow(tmp14, tmp18)
    tmp20 = tl.full([1], 1, tl.int32)
    tmp21 = tmp20 / tmp19
    tmp22 = tl.full([1], 1.0, tl.float64)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp23 * tmp22
    tmp25 = tmp12 * tmp24
    tmp26 = libdevice.cos(tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = libdevice.sin(tmp25)
    tmp29 = tmp28.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*x1)), tmp27, xmask)
    tl.store(out_ptr1 + (x0 + (128*x1)), tmp29, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/q2/cq27u2fkr2br6fstneitzn4iv5k6upmiqqbn2nqu2iex2jtcfoms.py
# Topologically Sorted Source Nodes: [encoder_value], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   encoder_value => addmm_15
# Graph fragment:
#   %addmm_15 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg43_1, %view_23, %permute_15), kwargs = {})
triton_tem_fused_addmm_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=8,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_18', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(in_ptr0, arg_A, arg_B, out_ptr0, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = ks0
    N = 3072
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (3072*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/k3/ck3lmz5tppgjvum6q7bphfoqurfnkuhzoumdc6fqndkt6asgtzdn.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   out_2 => _scaled_dot_product_flash_attention
# Graph fragment:
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_16, %permute_17, %permute_18), kwargs = {scale: 0.08838834764831843})
triton_poi_fused__scaled_dot_product_flash_attention_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*fp32', 2: '*fp32', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_19', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 3072)
    x4 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr3 + (x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = x3 % 2
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = tl.full([1], 1, tl.int64)
    tmp8 = tmp4 < tmp7
    tmp9 = tl.load(in_ptr0 + (1 + (2*(x0 // 2)) + (128*x4)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = -tmp9
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp8, tmp10, tmp11)
    tmp13 = tmp4 >= tmp7
    tmp14 = tl.full([1], 2, tl.int64)
    tmp15 = tmp4 < tmp14
    tmp16 = tl.load(in_ptr0 + ((2*(x0 // 2)) + (128*x4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tl.where(tmp8, tmp12, tmp16)
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = tmp3 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp2
    tmp26 = tl.load(in_ptr3 + (1 + (2*(x0 // 2)) + (128*x4)), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = -tmp26
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp8, tmp27, tmp28)
    tmp30 = tl.load(in_ptr3 + ((2*(x0 // 2)) + (128*x4)), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.where(tmp8, tmp29, tmp30)
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp19
    tmp34 = tmp25 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp22, xmask)
    tl.store(out_ptr1 + (x3), tmp35, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/d4/cd4syrxbsa5qrgmozcnqj5tzeb6x74x2hsjgwgx7exso6kclfpwo.py
# Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   out_2 => _scaled_dot_product_flash_attention
# Graph fragment:
#   %_scaled_dot_product_flash_attention : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_16, %permute_17, %permute_18), kwargs = {scale: 0.08838834764831843})
triton_poi_fused__scaled_dot_product_flash_attention_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__scaled_dot_product_flash_attention_20', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3072)
    x0 = xindex % 3072
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (3072*x1)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = 4096 + ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + (3072*(x1 + ((-1)*ks0)))), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/uo/cuoexpevfvf2xj63hx4fek4hdkosep3itmy7ymz2gtmxoljkqskn.py
# Topologically Sorted Source Nodes: [hidden_states_12, attn_output, hidden_states_13, norm_hidden_states, add_17, mul_26, norm_hidden_states_1], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_17 => add_582
#   attn_output => mul_495
#   hidden_states_12 => clone_6
#   hidden_states_13 => add_580
#   mul_26 => mul_497
#   norm_hidden_states => add_581, convert_element_type_91, convert_element_type_92, mul_496, rsqrt_6, sub_180, var_mean_2
#   norm_hidden_states_1 => add_583
# Graph fragment:
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_37,), kwargs = {})
#   %mul_495 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_26, %clone_6), kwargs = {})
#   %add_580 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_1, %mul_495), kwargs = {})
#   %convert_element_type_91 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_580, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_91, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_180 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_580, %getitem_32), kwargs = {})
#   %add_581 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_31, 1e-06), kwargs = {})
#   %rsqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_581,), kwargs = {})
#   %mul_496 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_180, %rsqrt_6), kwargs = {})
#   %convert_element_type_92 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_496, torch.bfloat16), kwargs = {})
#   %add_582 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_27, 1), kwargs = {})
#   %mul_497 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_92, %add_582), kwargs = {})
#   %add_583 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_497, %unsqueeze_28), kwargs = {})
triton_red_fused_add_clone_mul_native_layer_norm_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_mul_native_layer_norm_21', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(rmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask, tmp13_weight_next, tmp13_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp10, rmask)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr2 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr3 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp13
        tmp19 = 3072.0
        tmp20 = tmp14 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = 1.0
        tmp30 = tmp28 + tmp29
        tmp31 = tmp25 * tmp30
        tmp34 = tmp32 + tmp33
        tmp35 = tmp31 + tmp34
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp35, rmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/dy/cdy4pus3on5kquuix5zrzxmm6jauv2setwny6gqm6ncb47xs2ejn.py
# Topologically Sorted Source Nodes: [context_attn_output, encoder_hidden_states_3, norm_encoder_hidden_states, add_21, mul_29, norm_encoder_hidden_states_1], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_21 => add_604
#   context_attn_output => mul_505
#   encoder_hidden_states_3 => add_590
#   mul_29 => mul_520
#   norm_encoder_hidden_states => add_594, convert_element_type_101, convert_element_type_102, mul_512, rsqrt_7, sub_183, var_mean_3
#   norm_encoder_hidden_states_1 => add_608
# Graph fragment:
#   %mul_505 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_30, %view_39), kwargs = {})
#   %add_590 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_3, %mul_505), kwargs = {})
#   %convert_element_type_101 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_590, torch.float32), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_101, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_183 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_590, %getitem_34), kwargs = {})
#   %add_594 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_33, 1e-06), kwargs = {})
#   %rsqrt_7 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_594,), kwargs = {})
#   %mul_512 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_183, %rsqrt_7), kwargs = {})
#   %convert_element_type_102 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_512, torch.bfloat16), kwargs = {})
#   %add_604 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_31, 1), kwargs = {})
#   %mul_520 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_102, %add_604), kwargs = {})
#   %add_608 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_520, %unsqueeze_32), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: 'i32', 8: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_22', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 11, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp8 = tmp6 + tmp7
        tmp9 = tmp5 * tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp10, rmask & xmask)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr2 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr3 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr2 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr3 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp17 - tmp13
        tmp19 = 3072.0
        tmp20 = tmp14 / tmp19
        tmp21 = 1e-06
        tmp22 = tmp20 + tmp21
        tmp23 = libdevice.rsqrt(tmp22)
        tmp24 = tmp18 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = 1.0
        tmp30 = tmp28 + tmp29
        tmp31 = tmp25 * tmp30
        tmp34 = tmp32 + tmp33
        tmp35 = tmp31 + tmp34
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp35, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/av/cavtpsjyymdo4lcy67nrucxapowavsqs6tl2d3fkgiugegimhtfd.py
# Topologically Sorted Source Nodes: [hidden_states_15, hidden_states_16], Original ATen: [aten.gelu, aten.clone]
# Source node to ATen node mapping:
#   hidden_states_15 => add_584, add_585, convert_element_type_96, convert_element_type_97, mul_498, mul_499, mul_500, mul_501, mul_502, mul_503, tanh
#   hidden_states_16 => clone_7
# Graph fragment:
#   %convert_element_type_96 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_41, torch.float32), kwargs = {})
#   %mul_502 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_96, 0.5), kwargs = {})
#   %mul_498 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_96, %convert_element_type_96), kwargs = {})
#   %mul_499 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_498, %convert_element_type_96), kwargs = {})
#   %mul_500 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_499, 0.044715), kwargs = {})
#   %add_584 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_96, %mul_500), kwargs = {})
#   %mul_501 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_584, 0.7978845608028654), kwargs = {})
#   %tanh : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_501,), kwargs = {})
#   %add_585 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh, 1), kwargs = {})
#   %mul_503 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_502, %add_585), kwargs = {})
#   %convert_element_type_97 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_503, torch.bfloat16), kwargs = {})
#   %clone_7 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_97,), kwargs = {})
triton_poi_fused_clone_gelu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_gelu_23', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50331648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 12288
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = tmp3 * tmp3
    tmp7 = tmp6 * tmp3
    tmp8 = 0.044715
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 + tmp9
    tmp11 = 0.7978845608028654
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.tanh(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp5 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/et/cetyujylb5xqlvfpzlwqsaok5nqfh326swj3n342tbkbkzetd4sk.py
# Topologically Sorted Source Nodes: [ff_output, hidden_states_18, layer_norm_4, add_24, mul_31, x_2], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_24 => add_646
#   ff_output => mul_504
#   hidden_states_18 => add_586
#   layer_norm_4 => add_645, convert_element_type_116, convert_element_type_117, mul_565, rsqrt_8, sub_199, var_mean_4
#   mul_31 => mul_566
#   x_2 => add_647
# Graph fragment:
#   %mul_504 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_29, %view_43), kwargs = {})
#   %add_586 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_580, %mul_504), kwargs = {})
#   %convert_element_type_116 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_586, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_116, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_199 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_586, %getitem_42), kwargs = {})
#   %add_645 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_41, 1e-06), kwargs = {})
#   %rsqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_645,), kwargs = {})
#   %mul_565 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_199, %rsqrt_8), kwargs = {})
#   %convert_element_type_117 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_565, torch.bfloat16), kwargs = {})
#   %add_646 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_34, 1), kwargs = {})
#   %mul_566 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_117, %add_646), kwargs = {})
#   %add_647 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_566, %unsqueeze_35), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_24', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (15360 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (15360 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(rmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp8, rmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr4 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr5 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp11
        tmp17 = 3072.0
        tmp18 = tmp12 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = 1.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp23 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp29 + tmp32
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/6d/c6d5jf7ixlk7x5ddld3yfpp4filrwhc54q24a5eqyrjvgpgxqvum.py
# Topologically Sorted Source Nodes: [], Original ATen: []
# Source node to ATen node mapping:
# Graph fragment:
#   %mm_default_406 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%view_44, %permute_24), kwargs = {})
triton_tem_fused_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=3,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_25', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0, ks0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 128
    BLOCK_N : tl.constexpr = 128
    BLOCK_K : tl.constexpr = 64
    A = arg_A
    B = arg_B

    M = ks0
    N = 12288
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (12288*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/t7/ct76z3ppccvibpyaoi4gb2mbwluwmpiduh2svzjcw7z4n7qg7svf.py
# Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21], Original ATen: [aten.gelu, aten.clone]
# Source node to ATen node mapping:
#   hidden_states_20 => add_621, add_622, convert_element_type_106, convert_element_type_107, mul_536, mul_537, mul_538, mul_539, mul_540, mul_541, tanh_1
#   hidden_states_21 => clone_8
# Graph fragment:
#   %convert_element_type_106 : [num_users=4] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_45, torch.float32), kwargs = {})
#   %mul_540 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_106, 0.5), kwargs = {})
#   %mul_536 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_106, %convert_element_type_106), kwargs = {})
#   %mul_537 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_536, %convert_element_type_106), kwargs = {})
#   %mul_538 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_537, 0.044715), kwargs = {})
#   %add_621 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%convert_element_type_106, %mul_538), kwargs = {})
#   %mul_539 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%add_621, 0.7978845608028654), kwargs = {})
#   %tanh_1 : [num_users=1] = call_function[target=torch.ops.aten.tanh.default](args = (%mul_539,), kwargs = {})
#   %add_622 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%tanh_1, 1), kwargs = {})
#   %mul_541 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_540, %add_622), kwargs = {})
#   %convert_element_type_107 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_541, torch.bfloat16), kwargs = {})
#   %clone_8 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%convert_element_type_107,), kwargs = {})
triton_poi_fused_clone_gelu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[8388608], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_gelu_26', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = xindex % 12288
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.5
    tmp5 = tmp3 * tmp4
    tmp6 = tmp3 * tmp3
    tmp7 = tmp6 * tmp3
    tmp8 = 0.044715
    tmp9 = tmp7 * tmp8
    tmp10 = tmp3 + tmp9
    tmp11 = 0.7978845608028654
    tmp12 = tmp10 * tmp11
    tmp13 = libdevice.tanh(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp5 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(in_out_ptr0 + (x2), tmp17, None)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/gv/cgvyecyhsgkq5ptmwmdgm67dczxeicnf7mmjj7sjugelwv5q4kh7.py
# Topologically Sorted Source Nodes: [mul_30, encoder_hidden_states_4, layer_norm_5, add_26, mul_32, x_3], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_26 => add_658
#   encoder_hidden_states_4 => add_641
#   layer_norm_5 => add_648, convert_element_type_123, convert_element_type_124, mul_568, rsqrt_9, sub_200, var_mean_5
#   mul_30 => mul_557
#   mul_32 => mul_576
#   x_3 => add_662
# Graph fragment:
#   %mul_557 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_33, %view_47), kwargs = {})
#   %add_641 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_590, %mul_557), kwargs = {})
#   %convert_element_type_123 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_641, torch.float32), kwargs = {})
#   %var_mean_5 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_123, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_200 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_641, %getitem_50), kwargs = {})
#   %add_648 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_49, 1e-06), kwargs = {})
#   %rsqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_648,), kwargs = {})
#   %mul_568 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_200, %rsqrt_9), kwargs = {})
#   %convert_element_type_124 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_568, torch.bfloat16), kwargs = {})
#   %add_658 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_36, 1), kwargs = {})
#   %mul_576 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_124, %add_658), kwargs = {})
#   %add_662 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_576, %unsqueeze_37), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: 'i32', 9: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_27', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (15360 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (15360 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(rmask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask & xmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp8, rmask & xmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr4 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr5 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp11
        tmp17 = 3072.0
        tmp18 = tmp12 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = 1.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp23 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp29 + tmp32
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp33, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/5d/c5doacgkcjmadaew4nigdhhawppehsccl55ieseqw5r5objqfxpm.py
# Topologically Sorted Source Nodes: [hidden_states_31, attn_output_1, hidden_states_32, norm_hidden_states_2, add_39, mul_46, norm_hidden_states_3], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_39 => add_969
#   attn_output_1 => mul_845
#   hidden_states_31 => clone_9
#   hidden_states_32 => add_967
#   mul_46 => mul_847
#   norm_hidden_states_2 => add_968, convert_element_type_155, convert_element_type_156, mul_846, rsqrt_14, sub_285, var_mean_6
#   norm_hidden_states_3 => add_970
# Graph fragment:
#   %clone_9 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%view_72,), kwargs = {})
#   %mul_845 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_50, %clone_9), kwargs = {})
#   %add_967 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_586, %mul_845), kwargs = {})
#   %convert_element_type_155 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_967, torch.float32), kwargs = {})
#   %var_mean_6 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_155, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_285 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_967, %getitem_67), kwargs = {})
#   %add_968 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_66, 1e-06), kwargs = {})
#   %rsqrt_14 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_968,), kwargs = {})
#   %mul_846 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_285, %rsqrt_14), kwargs = {})
#   %convert_element_type_156 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_846, torch.bfloat16), kwargs = {})
#   %add_969 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_51, 1), kwargs = {})
#   %mul_847 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_156, %add_969), kwargs = {})
#   %add_970 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_847, %unsqueeze_52), kwargs = {})
triton_red_fused_add_clone_mul_native_layer_norm_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_clone_mul_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(rmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp8, rmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr1 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp11
        tmp17 = 3072.0
        tmp18 = tmp12 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = 1.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp23 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp29 + tmp32
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp33, rmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/5l/c5l7qkerilplucgnoawzaxj2xpzuj4nkmpjnw5fyyk4st7enhy3l.py
# Topologically Sorted Source Nodes: [context_attn_output_1, encoder_hidden_states_7, norm_encoder_hidden_states_2, add_43, mul_49, norm_encoder_hidden_states_3], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_43 => add_991
#   context_attn_output_1 => mul_855
#   encoder_hidden_states_7 => add_977
#   mul_49 => mul_870
#   norm_encoder_hidden_states_2 => add_981, convert_element_type_165, convert_element_type_166, mul_862, rsqrt_15, sub_288, var_mean_7
#   norm_encoder_hidden_states_3 => add_995
# Graph fragment:
#   %mul_855 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_54, %view_74), kwargs = {})
#   %add_977 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_641, %mul_855), kwargs = {})
#   %convert_element_type_165 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_977, torch.float32), kwargs = {})
#   %var_mean_7 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_165, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_288 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_977, %getitem_69), kwargs = {})
#   %add_981 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_68, 1e-06), kwargs = {})
#   %rsqrt_15 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_981,), kwargs = {})
#   %mul_862 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_288, %rsqrt_15), kwargs = {})
#   %convert_element_type_166 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_862, torch.bfloat16), kwargs = {})
#   %add_991 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_55, 1), kwargs = {})
#   %mul_870 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_166, %add_991), kwargs = {})
#   %add_995 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_870, %unsqueeze_56), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_29', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp11_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp11_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (6144 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp6 = tmp4 + tmp5
        tmp7 = tmp3 * tmp6
        tmp8 = tmp0 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp11_mean_next, tmp11_m2_next, tmp11_weight_next = triton_helpers.welford_reduce(
            tmp10, tmp11_mean, tmp11_m2, tmp11_weight, roffset == 0
        )
        tmp11_mean = tl.where(rmask & xmask, tmp11_mean_next, tmp11_mean)
        tmp11_m2 = tl.where(rmask & xmask, tmp11_m2_next, tmp11_m2)
        tmp11_weight = tl.where(rmask & xmask, tmp11_weight_next, tmp11_weight)
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp8, rmask & xmask)
    tmp11_tmp, tmp12_tmp, tmp13_tmp = triton_helpers.welford(
        tmp11_mean, tmp11_m2, tmp11_weight, 1
    )
    tmp11 = tmp11_tmp[:, None]
    tmp12 = tmp12_tmp[:, None]
    tmp13 = tmp13_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr1 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (12288 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr1 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (9216 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = tmp15 - tmp11
        tmp17 = 3072.0
        tmp18 = tmp12 / tmp17
        tmp19 = 1e-06
        tmp20 = tmp18 + tmp19
        tmp21 = libdevice.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp26 = tmp24 + tmp25
        tmp27 = 1.0
        tmp28 = tmp26 + tmp27
        tmp29 = tmp23 * tmp28
        tmp32 = tmp30 + tmp31
        tmp33 = tmp29 + tmp32
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp33, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/ww/cww67ptnusfsz5guhjv6qcgnxx6zeuzbxpggrn7tdgukvj3ekc4l.py
# Topologically Sorted Source Nodes: [silu_41], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_41 => convert_element_type_1263, convert_element_type_1264, mul_6867, sigmoid_41
# Graph fragment:
#   %convert_element_type_1263 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_41 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_1263,), kwargs = {})
#   %mul_6867 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1263, %sigmoid_41), kwargs = {})
#   %convert_element_type_1264 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6867, torch.bfloat16), kwargs = {})
#   %mm_default_194 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_1264, %permute_350), kwargs = {})
triton_tem_fused_silu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=2,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_silu_30', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 32
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 9216
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (9216*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/jw/cjwu6ilecswgqpxzxcwon43fntv5djbp6j4qy5j3mvcadvxvuqhn.py
# Topologically Sorted Source Nodes: [hidden_states_365, layer_norm_76, add_420, mul_391, x_38], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_420 => add_7624
#   hidden_states_365 => cat_102
#   layer_norm_76 => add_7614, convert_element_type_1268, convert_element_type_1269, mul_6868, rsqrt_152, sub_2090, var_mean_76
#   mul_391 => mul_6876
#   x_38 => add_7628
# Graph fragment:
#   %cat_102 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%add_7607, %add_7552], 1), kwargs = {})
#   %convert_element_type_1268 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_102, torch.float32), kwargs = {})
#   %var_mean_76 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1268, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2090 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_102, %getitem_669), kwargs = {})
#   %add_7614 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_668, 1e-06), kwargs = {})
#   %rsqrt_152 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7614,), kwargs = {})
#   %mul_6868 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2090, %rsqrt_152), kwargs = {})
#   %convert_element_type_1269 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6868, torch.bfloat16), kwargs = {})
#   %add_7624 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_466, 1), kwargs = {})
#   %mul_6876 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1269, %add_7624), kwargs = {})
#   %add_7628 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_6876, %unsqueeze_467), kwargs = {})
triton_red_fused_add_cat_mul_native_layer_norm_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*bf16', 8: '*bf16', 9: '*bf16', 10: '*bf16', 11: '*bf16', 12: '*bf16', 13: '*bf16', 14: 'i32', 15: 'i32', 16: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_mul_native_layer_norm_31', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 15, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr3, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp33_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp33_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks0
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(15360 + r1, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr2 + (tl.broadcast_to(15360 + r1, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tl.load(in_ptr3 + (r1 + (3072*x0)), rmask & tmp4 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp10 = tl.load(in_ptr4 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 * tmp11
        tmp13 = tmp5 + tmp12
        tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
        tmp15 = tl.where(tmp4, tmp13, tmp14)
        tmp16 = tmp0 >= tmp3
        tmp17 = 4096 + ks0
        tmp18 = tmp0 < tmp17
        tmp19 = tl.load(in_ptr5 + (r1 + (3072*(x0 + ((-1)*ks0)))), rmask & tmp16 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp20 = tl.load(in_ptr6 + (tl.broadcast_to(15360 + r1, [XBLOCK, RBLOCK])), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr7 + (tl.broadcast_to(15360 + r1, [XBLOCK, RBLOCK])), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tmp20 + tmp21
        tmp23 = tl.load(in_ptr8 + (r1 + (3072*(x0 + ((-1)*ks0)))), rmask & tmp16 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp24 = tl.load(in_ptr9 + (tl.broadcast_to(r1, [XBLOCK, RBLOCK])), rmask & tmp16 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp22 * tmp25
        tmp27 = tmp19 + tmp26
        tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
        tmp29 = tl.where(tmp16, tmp27, tmp28)
        tmp30 = tl.where(tmp4, tmp15, tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp33_mean_next, tmp33_m2_next, tmp33_weight_next = triton_helpers.welford_reduce(
            tmp32, tmp33_mean, tmp33_m2, tmp33_weight, roffset == 0
        )
        tmp33_mean = tl.where(rmask & xmask, tmp33_mean_next, tmp33_mean)
        tmp33_m2 = tl.where(rmask & xmask, tmp33_m2_next, tmp33_m2)
        tmp33_weight = tl.where(rmask & xmask, tmp33_weight_next, tmp33_weight)
        tl.store(out_ptr0 + (r1 + (3072*x0)), tmp30, rmask & xmask)
    tmp33_tmp, tmp34_tmp, tmp35_tmp = triton_helpers.welford(
        tmp33_mean, tmp33_m2, tmp33_weight, 1
    )
    tmp33 = tmp33_tmp[:, None]
    tmp34 = tmp34_tmp[:, None]
    tmp35 = tmp35_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(out_ptr0 + (r1 + (3072*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp46 = tl.load(in_ptr10 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp47 = tl.load(in_ptr11 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp53 = tl.load(in_ptr11 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp37 - tmp33
        tmp39 = 3072.0
        tmp40 = tmp34 / tmp39
        tmp41 = 1e-06
        tmp42 = tmp40 + tmp41
        tmp43 = libdevice.rsqrt(tmp42)
        tmp44 = tmp38 * tmp43
        tmp45 = tmp44.to(tl.float32)
        tmp48 = tmp46 + tmp47
        tmp49 = 1.0
        tmp50 = tmp48 + tmp49
        tmp51 = tmp45 * tmp50
        tmp54 = tmp52 + tmp53
        tmp55 = tmp51 + tmp54
        tl.store(out_ptr3 + (r1 + (3072*x0)), tmp55, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/j7/cj7ile7uzareqr5ehayxmzrhk2k7dljr62gyn7edkfjuybf342qj.py
# Topologically Sorted Source Nodes: [pow_80, variance_76, pow_81, variance_77, stack_38, stack_39, out_78], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
# Source node to ATen node mapping:
#   out_78 => _scaled_dot_product_flash_attention_19
#   pow_80 => pow_80
#   pow_81 => pow_81
#   stack_38 => cat_103
#   stack_39 => cat_104
#   variance_76 => mean_76
#   variance_77 => mean_77
# Graph fragment:
#   %pow_80 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_686, 2), kwargs = {})
#   %mean_76 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_80, [-1], True), kwargs = {})
#   %pow_81 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%view_687, 2), kwargs = {})
#   %mean_77 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_81, [-1], True), kwargs = {})
#   %cat_103 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_472, %unsqueeze_473], -1), kwargs = {})
#   %cat_104 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%unsqueeze_478, %unsqueeze_479], -1), kwargs = {})
#   %_scaled_dot_product_flash_attention_19 : [num_users=1] = call_function[target=torch.ops.aten._scaled_dot_product_flash_attention.default](args = (%permute_355, %permute_356, %permute_357), kwargs = {scale: 0.08838834764831843})
triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: '*bf16', 6: '*bf16', 7: '*fp32', 8: '*fp32', 9: '*bf16', 10: 'i32', 11: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'no_x_dim': False, 'num_load': 24, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x5 = xindex
    x0 = xindex % 24
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr2 + (r2 + (128*x5)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask & xmask, tmp7, _tmp6)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask & xmask, tmp15, _tmp14)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    x1 = (xindex // 24)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex % 2
        r4 = (rindex // 2)
        r2 = rindex
        tmp51 = tl.load(in_ptr0 + (r2 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp52 = tl.load(in_ptr1 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp55 = tl.load(in_ptr4 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp62 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp90 = tl.load(in_ptr2 + (r2 + (128*x5)), rmask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp91 = tl.load(in_ptr3 + (r2 + (128*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp94 = tl.load(in_ptr7 + (r2), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = r3
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 >= tmp17
        tmp19 = tl.full([1, 1], 1, tl.int64)
        tmp20 = tmp16 < tmp19
        tmp21 = tl.load(in_ptr0 + (1 + (2*r4) + (128*x5)), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (1 + (2*r4) + (128*x0)), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tmp21 + tmp22
        tmp24 = 128.0
        tmp25 = tmp6 / tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = 1e-06
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp23 * tmp29
        tmp31 = tl.load(in_ptr4 + (tl.broadcast_to(1 + (2*r4), [XBLOCK, RBLOCK])), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp32 = 0.0
        tmp33 = tmp31 + tmp32
        tmp34 = tmp30 * tmp33
        tmp35 = -tmp34
        tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
        tmp37 = tl.where(tmp20, tmp35, tmp36)
        tmp38 = tmp16 >= tmp19
        tmp39 = tl.full([1, 1], 2, tl.int64)
        tmp40 = tmp16 < tmp39
        tmp41 = tl.load(in_ptr0 + ((2*r4) + (128*x5)), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr1 + ((2*r4) + (128*x0)), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tmp41 + tmp42
        tmp44 = tmp43 * tmp29
        tmp45 = tl.load(in_ptr4 + (tl.broadcast_to(2*r4, [XBLOCK, RBLOCK])), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp46 = tmp45 + tmp32
        tmp47 = tmp44 * tmp46
        tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
        tmp49 = tl.where(tmp38, tmp47, tmp48)
        tmp50 = tl.where(tmp20, tmp37, tmp49)
        tmp53 = tmp51 + tmp52
        tmp54 = tmp53 * tmp29
        tmp56 = tmp55 + tmp32
        tmp57 = tmp54 * tmp56
        tmp58 = tmp57.to(tl.float32)
        tmp60 = tmp58 * tmp59
        tmp61 = tmp50.to(tl.float32)
        tmp63 = tmp61 * tmp62
        tmp64 = tmp60 + tmp63
        tmp65 = tmp64.to(tl.float32)
        tmp66 = tl.load(in_ptr2 + (1 + (2*r4) + (128*x5)), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr3 + (1 + (2*r4) + (128*x0)), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp68 = tmp66 + tmp67
        tmp69 = tmp14 / tmp24
        tmp70 = tmp69.to(tl.float32)
        tmp71 = tmp70 + tmp27
        tmp72 = libdevice.rsqrt(tmp71)
        tmp73 = tmp68 * tmp72
        tmp74 = tl.load(in_ptr7 + (tl.broadcast_to(1 + (2*r4), [XBLOCK, RBLOCK])), rmask & tmp20 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp75 = tmp74 + tmp32
        tmp76 = tmp73 * tmp75
        tmp77 = -tmp76
        tmp78 = tl.full(tmp77.shape, 0.0, tmp77.dtype)
        tmp79 = tl.where(tmp20, tmp77, tmp78)
        tmp80 = tl.load(in_ptr2 + ((2*r4) + (128*x5)), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp81 = tl.load(in_ptr3 + ((2*r4) + (128*x0)), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp82 = tmp80 + tmp81
        tmp83 = tmp82 * tmp72
        tmp84 = tl.load(in_ptr7 + (tl.broadcast_to(2*r4, [XBLOCK, RBLOCK])), rmask & tmp38 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp85 = tmp84 + tmp32
        tmp86 = tmp83 * tmp85
        tmp87 = tl.full(tmp86.shape, 0.0, tmp86.dtype)
        tmp88 = tl.where(tmp38, tmp86, tmp87)
        tmp89 = tl.where(tmp20, tmp79, tmp88)
        tmp92 = tmp90 + tmp91
        tmp93 = tmp92 * tmp72
        tmp95 = tmp94 + tmp32
        tmp96 = tmp93 * tmp95
        tmp97 = tmp96.to(tl.float32)
        tmp98 = tmp97 * tmp59
        tmp99 = tmp89.to(tl.float32)
        tmp100 = tmp99 * tmp62
        tmp101 = tmp98 + tmp100
        tmp102 = tmp101.to(tl.float32)
        tl.store(in_out_ptr0 + (r2 + (128*x5)), tmp65, rmask & xmask)
        tl.store(in_out_ptr1 + (r2 + (128*x5)), tmp102, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/g4/cg4jqz3pxxnbi34ji25dh6oryzkltgw74p6kqfao7ilfn44msqfa.py
# Topologically Sorted Source Nodes: [hidden_states_370], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   hidden_states_370 => cat_105
# Graph fragment:
#   %cat_105 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_693, %convert_element_type_1274], 2), kwargs = {})
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 15360
    x1 = (xindex // 15360)
    x2 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3072, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((3072*x1) + x0), tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 15360, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + ((12288*x1) + ((-3072) + x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.load(in_ptr2 + ((-3072) + x0), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = tmp12 * tmp12
    tmp16 = tmp15 * tmp12
    tmp17 = 0.044715
    tmp18 = tmp16 * tmp17
    tmp19 = tmp12 + tmp18
    tmp20 = 0.7978845608028654
    tmp21 = tmp19 * tmp20
    tmp22 = libdevice.tanh(tmp21)
    tmp23 = 1.0
    tmp24 = tmp22 + tmp23
    tmp25 = tmp14 * tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp6, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp5, tmp28)
    tl.store(out_ptr0 + (x2), tmp29, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/vm/cvmzgneyeh3bc2ytevm76iswkrgw6vqhg2qch2kt2wx3xshxwq3f.py
# Topologically Sorted Source Nodes: [hidden_states_371, hidden_states_372], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   hidden_states_371 => mul_7145
#   hidden_states_372 => add_7932
# Graph fragment:
#   %mul_7145 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_480, %view_695), kwargs = {})
#   %add_7932 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_102, %mul_7145), kwargs = {})
triton_poi_fused_add_mul_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_34', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (6144 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (6144 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tmp3 * tmp6
    tmp8 = tmp0 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/h7/ch72cqpz3upiv3kph67k34cfqtskdiyxolths4a5le7k2yg2u5n7.py
# Topologically Sorted Source Nodes: [hidden_states_374, layer_norm_77, add_429, mul_401, x_39], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_429 => add_7959
#   hidden_states_374 => cat_106
#   layer_norm_77 => add_7949, convert_element_type_1298, convert_element_type_1299, mul_7167, rsqrt_155, sub_2181, var_mean_77
#   mul_401 => mul_7175
#   x_39 => add_7963
# Graph fragment:
#   %cat_106 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_331, %slice_333], 1), kwargs = {})
#   %convert_element_type_1298 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%cat_106, torch.float32), kwargs = {})
#   %var_mean_77 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_1298, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_2181 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%cat_106, %getitem_687), kwargs = {})
#   %add_7949 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_686, 1e-06), kwargs = {})
#   %rsqrt_155 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7949,), kwargs = {})
#   %mul_7167 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_2181, %rsqrt_155), kwargs = {})
#   %convert_element_type_1299 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_7167, torch.bfloat16), kwargs = {})
#   %add_7959 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%unsqueeze_481, 1), kwargs = {})
#   %mul_7175 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1299, %add_7959), kwargs = {})
#   %add_7963 : [num_users=4] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7175, %unsqueeze_482), kwargs = {})
triton_red_fused_add_cat_mul_native_layer_norm_35 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[8192, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_mul_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = x0
        tmp1 = tl.full([1, 1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = ks0
        tmp4 = tmp0 < tmp3
        tmp5 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & tmp4 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp6 = tmp0 >= tmp3
        tmp7 = 4096 + ks0
        tmp8 = tmp0 < tmp7
        tmp9 = tl.load(in_ptr0 + (r1 + (3072*ks0) + (3072*(x0 + ((-1)*ks0)))), rmask & tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(rmask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(rmask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(rmask & xmask, tmp13_weight_next, tmp13_weight)
    tmp13_tmp, tmp14_tmp, tmp15_tmp = triton_helpers.welford(
        tmp13_mean, tmp13_m2, tmp13_weight, 1
    )
    tmp13 = tmp13_tmp[:, None]
    tmp14 = tmp14_tmp[:, None]
    tmp15 = tmp15_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp36 = tl.load(in_ptr1 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp37 = tl.load(in_ptr2 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp42 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp43 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = x0
        tmp17 = tl.full([1, 1], 0, tl.int64)
        tmp18 = tmp16 >= tmp17
        tmp19 = ks0
        tmp20 = tmp16 < tmp19
        tmp21 = tl.load(in_ptr0 + (r1 + (3072*x0)), rmask & tmp20 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp22 = tmp16 >= tmp19
        tmp23 = 4096 + ks0
        tmp24 = tmp16 < tmp23
        tmp25 = tl.load(in_ptr0 + (r1 + (3072*ks0) + (3072*(x0 + ((-1)*ks0)))), rmask & tmp22 & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp26 = tl.where(tmp20, tmp21, tmp25)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp27 - tmp13
        tmp29 = 3072.0
        tmp30 = tmp14 / tmp29
        tmp31 = 1e-06
        tmp32 = tmp30 + tmp31
        tmp33 = libdevice.rsqrt(tmp32)
        tmp34 = tmp28 * tmp33
        tmp35 = tmp34.to(tl.float32)
        tmp38 = tmp36 + tmp37
        tmp39 = 1.0
        tmp40 = tmp38 + tmp39
        tmp41 = tmp35 * tmp40
        tmp44 = tmp42 + tmp43
        tmp45 = tmp41 + tmp44
        tl.store(in_out_ptr0 + (r1 + (3072*x0)), tmp45, rmask & xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/4f/c4fcp2t5bzic3pefy3wdpvt4eyf5cpgopiw3oz4z2kjzhoestyb5.py
# Topologically Sorted Source Nodes: [hidden_states_374, hidden_states_380, hidden_states_381], Original ATen: [aten.cat, aten.mul, aten.add]
# Source node to ATen node mapping:
#   hidden_states_374 => cat_106
#   hidden_states_380 => mul_7444
#   hidden_states_381 => add_8267
# Graph fragment:
#   %cat_106 : [num_users=3] = call_function[target=torch.ops.aten.cat.default](args = ([%slice_331, %slice_333], 1), kwargs = {})
#   %mul_7444 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%unsqueeze_495, %view_713), kwargs = {})
#   %add_8267 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%cat_106, %mul_7444), kwargs = {})
triton_poi_fused_add_cat_mul_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[16777216], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_mul_36', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3072)
    x0 = xindex % 3072
    x2 = xindex
    tmp11 = tl.load(in_ptr1 + (6144 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr2 + (6144 + x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = ks0
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + (3072*x1)), tmp4 & xmask, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = 4096 + ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + (3072*ks0) + (3072*(x1 + ((-1)*ks0)))), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 * tmp16
    tmp18 = tmp10 + tmp17
    tl.store(in_out_ptr0 + (x2), tmp18, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/oe/coeiqddesvdvlck2piozrhp7x3bpv63ylie72lok2wb6ftdjr7i7.py
# Topologically Sorted Source Nodes: [silu_66, silu_67, silu_68, silu_69], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_66 => convert_element_type_2013, convert_element_type_2014, mul_14342, sigmoid_66
#   silu_67 => convert_element_type_2043, convert_element_type_2044, mul_14641, sigmoid_67
#   silu_68 => convert_element_type_2073, convert_element_type_2074, mul_14940, sigmoid_68
#   silu_69 => convert_element_type_2103, convert_element_type_2104, mul_15239, sigmoid_69
# Graph fragment:
#   %convert_element_type_2013 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_66 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2013,), kwargs = {})
#   %mul_14342 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2013, %sigmoid_66), kwargs = {})
#   %convert_element_type_2014 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14342, torch.bfloat16), kwargs = {})
#   %convert_element_type_2043 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_67 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2043,), kwargs = {})
#   %mul_14641 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2043, %sigmoid_67), kwargs = {})
#   %convert_element_type_2044 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14641, torch.bfloat16), kwargs = {})
#   %convert_element_type_2073 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_68 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2073,), kwargs = {})
#   %mul_14940 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2073, %sigmoid_68), kwargs = {})
#   %convert_element_type_2074 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14940, torch.bfloat16), kwargs = {})
#   %convert_element_type_2103 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_69 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2103,), kwargs = {})
#   %mul_15239 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2103, %sigmoid_69), kwargs = {})
#   %convert_element_type_2104 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15239, torch.bfloat16), kwargs = {})
triton_poi_fused_silu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: '*bf16', 5: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_37', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
    tl.store(out_ptr2 + (x0), tmp4, xmask)
    tl.store(out_ptr3 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/x7/cx7ptmw64s3yncbbl35uhpmzjabecc6qdtsheo7gluot3bigi7e3.py
# Topologically Sorted Source Nodes: [silu_78, silu_79], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_78 => convert_element_type_2373, convert_element_type_2374, mul_17930, sigmoid_78
#   silu_79 => convert_element_type_2403, convert_element_type_2404, mul_18220, sigmoid_79
# Graph fragment:
#   %convert_element_type_2373 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_78 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2373,), kwargs = {})
#   %mul_17930 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2373, %sigmoid_78), kwargs = {})
#   %convert_element_type_2374 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_17930, torch.bfloat16), kwargs = {})
#   %convert_element_type_2403 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2403,), kwargs = {})
#   %mul_18220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2403, %sigmoid_79), kwargs = {})
#   %convert_element_type_2404 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18220, torch.bfloat16), kwargs = {})
triton_poi_fused_silu_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.pointwise(
    size_hints=[4096], 
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_38', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp4, xmask)
    tl.store(out_ptr1 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/u6/cu6h3quq3m5pdwvt7u47edhfsjdh4xl6qwcyva4bdatrzz6vxrk6.py
# Topologically Sorted Source Nodes: [silu_79], Original ATen: [aten.silu]
# Source node to ATen node mapping:
#   silu_79 => convert_element_type_2403, convert_element_type_2404, mul_18220, sigmoid_79
# Graph fragment:
#   %convert_element_type_2403 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_73, torch.float32), kwargs = {})
#   %sigmoid_79 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_2403,), kwargs = {})
#   %mul_18220 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2403, %sigmoid_79), kwargs = {})
#   %convert_element_type_2404 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18220, torch.bfloat16), kwargs = {})
#   %mm_default_2 : [num_users=1] = call_function[target=torch.ops.aten.mm.default](args = (%convert_element_type_2404, %permute_730), kwargs = {})
triton_tem_fused_silu_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_silu_39', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 16
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 1
    N = 6144
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (6144*idx_m)
    tl.store(out_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), acc, mask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/ao/caotyuuwy55ux5t6rdwwfe3ooyfogdwqvwgzkecqj5hvnnl7x7lz.py
# Topologically Sorted Source Nodes: [layer_norm_114, mul_771, x_76], Original ATen: [aten.native_layer_norm, aten.mul, aten.add]
# Source node to ATen node mapping:
#   layer_norm_114 => add_20335, convert_element_type_2408, convert_element_type_2409, mul_18221, rsqrt_266, sub_5545, var_mean_114
#   mul_771 => mul_18222
#   x_76 => add_20337
# Graph fragment:
#   %convert_element_type_2408 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%slice_849, torch.float32), kwargs = {})
#   %var_mean_114 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_2408, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_5545 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%slice_849, %getitem_1352), kwargs = {})
#   %add_20335 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1351, 1e-06), kwargs = {})
#   %rsqrt_266 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_20335,), kwargs = {})
#   %mul_18221 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5545, %rsqrt_266), kwargs = {})
#   %convert_element_type_2409 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18221, torch.bfloat16), kwargs = {})
#   %mul_18222 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_2409, %slice_851), kwargs = {})
#   %add_20337 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_18222, %slice_853), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.reduction(
    size_hints=[4096, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32', 6: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 5, 6), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_40', 'mutated_arg_names': [], 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, ks0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, RBLOCK], True, tl.int1)
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (3072*ks0) + (3072*x0)), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(rmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(rmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(rmask, tmp3_weight_next, tmp3_weight)
    tmp3_tmp, tmp4_tmp, tmp5_tmp = triton_helpers.welford(
        tmp3_mean, tmp3_m2, tmp3_weight, 1
    )
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tmp5 = tmp5_tmp[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (3072*ks0) + (3072*x0)), rmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr2 + (3072 + r1), rmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 - tmp3
        tmp9 = 3072.0
        tmp10 = tmp4 / tmp9
        tmp11 = 1e-06
        tmp12 = tmp10 + tmp11
        tmp13 = libdevice.rsqrt(tmp12)
        tmp14 = tmp8 * tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = 1.0
        tmp20 = tmp18 + tmp19
        tmp21 = tmp15 * tmp20
        tmp24 = tmp22 + tmp23
        tmp25 = tmp21 + tmp24
        tl.store(out_ptr2 + (r1 + (3072*x0)), tmp25, rmask)
''', device_str='cuda')


# kernel path: /opt/inductor_cache/pf/cpfos6fpk35te6lilqybnnlvmjiarfzppssrru6huphujukdiu77.py
# Topologically Sorted Source Nodes: [output], Original ATen: [aten.addmm]
# Source node to ATen node mapping:
#   output => addmm_503
# Graph fragment:
#   %addmm_503 : [num_users=1] = call_function[target=torch.ops.aten.addmm.default](args = (%arg1171_1, %view_1362, %permute_731), kwargs = {})
triton_tem_fused_addmm_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, instance_descriptor, DeviceProperties

@triton_heuristics.template(
    num_stages=5,
    num_warps=4,
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]},
    inductor_meta={'kernel_name': 'triton_tem_fused_addmm_41', 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
)
@triton.jit
def triton_(in_ptr0, arg_A, arg_B, out_ptr0):
    GROUP_M : tl.constexpr = 8
    EVEN_K : tl.constexpr = True
    ALLOW_TF32 : tl.constexpr = True
    ACC_TYPE : tl.constexpr = tl.float32
    B_PROLOGUE_CAST_TYPE : tl.constexpr = None
    BLOCK_M : tl.constexpr = 64
    BLOCK_N : tl.constexpr = 64
    BLOCK_K : tl.constexpr = 128
    A = arg_A
    B = arg_B

    M = 4096
    N = 64
    K = 3072
    if M * N == 0:
        # early exit due to zero-size input(s)
        return
    stride_am = 3072
    stride_ak = 1
    stride_bk = 1
    stride_bn = 3072

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    if (stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1):
        ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    else:
        ram = rm % M
    if (stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1):
        rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    else:
        rbn = rn % N
    rk = tl.arange(0, BLOCK_K)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.)
            b = tl.load(B, mask=rk[:, None] < k, other=0.)
        if B_PROLOGUE_CAST_TYPE is not None:
            b = b.to(B_PROLOGUE_CAST_TYPE)
        acc += tl.dot(a, b, allow_tf32=ALLOW_TF32)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    # inductor generates a suffix
    xindex = idx_n + (64*idx_m)
    tmp0 = tl.load(in_ptr0 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = acc + tmp0
    tl.store(out_ptr0 + (tl.broadcast_to(xindex, acc.shape)), tmp1, mask)
''', device_str='cuda')
meta5 = {'GROUP_M': 8, 'EVEN_K': True, 'ALLOW_TF32': True, 'ACC_TYPE': 'tl.float32', 'B_PROLOGUE_CAST_TYPE': None, 'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1 = args
    args.clear()
    s2 = arg5_1
    s3 = arg6_1
    s4 = arg7_1
    s6 = arg23_1
    s8 = arg27_1
    assert_size_stride(arg0_1, (3072, 64), (64, 1))
    assert_size_stride(arg1_1, (3072, ), (1, ))
    assert_size_stride(arg2_1, (1, 4096, 64), (262144, 64, 1))
    assert_size_stride(arg3_1, (1, ), (1, ))
    assert_size_stride(arg4_1, (1, ), (1, ))
    assert_size_stride(arg8_1, (1, 768), (768, 1))
    assert_size_stride(arg9_1, (3072, 256), (256, 1))
    assert_size_stride(arg10_1, (3072, ), (1, ))
    assert_size_stride(arg11_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg12_1, (3072, ), (1, ))
    assert_size_stride(arg13_1, (3072, 256), (256, 1))
    assert_size_stride(arg14_1, (3072, ), (1, ))
    assert_size_stride(arg15_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg16_1, (3072, ), (1, ))
    assert_size_stride(arg17_1, (3072, 768), (768, 1))
    assert_size_stride(arg18_1, (3072, ), (1, ))
    assert_size_stride(arg19_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg20_1, (3072, ), (1, ))
    assert_size_stride(arg21_1, (3072, 4096), (4096, 1))
    assert_size_stride(arg22_1, (3072, ), (1, ))
    assert_size_stride(arg24_1, (1, s6, 4096), (4096*s6, 4096, 1))
    assert_size_stride(arg25_1, (s6, 3), (3, 1))
    assert_size_stride(arg26_1, (4096, 3), (3, 1))
    assert_size_stride(arg28_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg29_1, (18432, ), (1, ))
    assert_size_stride(arg30_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg31_1, (18432, ), (1, ))
    assert_size_stride(arg32_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg33_1, (3072, ), (1, ))
    assert_size_stride(arg34_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg35_1, (3072, ), (1, ))
    assert_size_stride(arg36_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg37_1, (3072, ), (1, ))
    assert_size_stride(arg38_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg39_1, (3072, ), (1, ))
    assert_size_stride(arg40_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg41_1, (3072, ), (1, ))
    assert_size_stride(arg42_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg43_1, (3072, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg49_1, (3072, ), (1, ))
    assert_size_stride(arg50_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg51_1, (3072, ), (1, ))
    assert_size_stride(arg52_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg53_1, (12288, ), (1, ))
    assert_size_stride(arg54_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg55_1, (3072, ), (1, ))
    assert_size_stride(arg56_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg57_1, (12288, ), (1, ))
    assert_size_stride(arg58_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg59_1, (3072, ), (1, ))
    assert_size_stride(arg60_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg61_1, (18432, ), (1, ))
    assert_size_stride(arg62_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg63_1, (18432, ), (1, ))
    assert_size_stride(arg64_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg65_1, (3072, ), (1, ))
    assert_size_stride(arg66_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg67_1, (3072, ), (1, ))
    assert_size_stride(arg68_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg69_1, (3072, ), (1, ))
    assert_size_stride(arg70_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg71_1, (3072, ), (1, ))
    assert_size_stride(arg72_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg73_1, (3072, ), (1, ))
    assert_size_stride(arg74_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg75_1, (3072, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (128, ), (1, ))
    assert_size_stride(arg78_1, (128, ), (1, ))
    assert_size_stride(arg79_1, (128, ), (1, ))
    assert_size_stride(arg80_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg81_1, (3072, ), (1, ))
    assert_size_stride(arg82_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg83_1, (3072, ), (1, ))
    assert_size_stride(arg84_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg85_1, (12288, ), (1, ))
    assert_size_stride(arg86_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg87_1, (3072, ), (1, ))
    assert_size_stride(arg88_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg89_1, (12288, ), (1, ))
    assert_size_stride(arg90_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg91_1, (3072, ), (1, ))
    assert_size_stride(arg92_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg93_1, (18432, ), (1, ))
    assert_size_stride(arg94_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg95_1, (18432, ), (1, ))
    assert_size_stride(arg96_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg97_1, (3072, ), (1, ))
    assert_size_stride(arg98_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg99_1, (3072, ), (1, ))
    assert_size_stride(arg100_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg101_1, (3072, ), (1, ))
    assert_size_stride(arg102_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg103_1, (3072, ), (1, ))
    assert_size_stride(arg104_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg105_1, (3072, ), (1, ))
    assert_size_stride(arg106_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg107_1, (3072, ), (1, ))
    assert_size_stride(arg108_1, (128, ), (1, ))
    assert_size_stride(arg109_1, (128, ), (1, ))
    assert_size_stride(arg110_1, (128, ), (1, ))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg113_1, (3072, ), (1, ))
    assert_size_stride(arg114_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg115_1, (3072, ), (1, ))
    assert_size_stride(arg116_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg117_1, (12288, ), (1, ))
    assert_size_stride(arg118_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg119_1, (3072, ), (1, ))
    assert_size_stride(arg120_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg121_1, (12288, ), (1, ))
    assert_size_stride(arg122_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg123_1, (3072, ), (1, ))
    assert_size_stride(arg124_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg125_1, (18432, ), (1, ))
    assert_size_stride(arg126_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg127_1, (18432, ), (1, ))
    assert_size_stride(arg128_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg129_1, (3072, ), (1, ))
    assert_size_stride(arg130_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg131_1, (3072, ), (1, ))
    assert_size_stride(arg132_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg133_1, (3072, ), (1, ))
    assert_size_stride(arg134_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg135_1, (3072, ), (1, ))
    assert_size_stride(arg136_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg137_1, (3072, ), (1, ))
    assert_size_stride(arg138_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg139_1, (3072, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (128, ), (1, ))
    assert_size_stride(arg142_1, (128, ), (1, ))
    assert_size_stride(arg143_1, (128, ), (1, ))
    assert_size_stride(arg144_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg145_1, (3072, ), (1, ))
    assert_size_stride(arg146_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg147_1, (3072, ), (1, ))
    assert_size_stride(arg148_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg149_1, (12288, ), (1, ))
    assert_size_stride(arg150_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg151_1, (3072, ), (1, ))
    assert_size_stride(arg152_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg153_1, (12288, ), (1, ))
    assert_size_stride(arg154_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg155_1, (3072, ), (1, ))
    assert_size_stride(arg156_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg157_1, (18432, ), (1, ))
    assert_size_stride(arg158_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg159_1, (18432, ), (1, ))
    assert_size_stride(arg160_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg161_1, (3072, ), (1, ))
    assert_size_stride(arg162_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg163_1, (3072, ), (1, ))
    assert_size_stride(arg164_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg165_1, (3072, ), (1, ))
    assert_size_stride(arg166_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg167_1, (3072, ), (1, ))
    assert_size_stride(arg168_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg169_1, (3072, ), (1, ))
    assert_size_stride(arg170_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg171_1, (3072, ), (1, ))
    assert_size_stride(arg172_1, (128, ), (1, ))
    assert_size_stride(arg173_1, (128, ), (1, ))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg177_1, (3072, ), (1, ))
    assert_size_stride(arg178_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg179_1, (3072, ), (1, ))
    assert_size_stride(arg180_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg181_1, (12288, ), (1, ))
    assert_size_stride(arg182_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg183_1, (3072, ), (1, ))
    assert_size_stride(arg184_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg185_1, (12288, ), (1, ))
    assert_size_stride(arg186_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg187_1, (3072, ), (1, ))
    assert_size_stride(arg188_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg189_1, (18432, ), (1, ))
    assert_size_stride(arg190_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg191_1, (18432, ), (1, ))
    assert_size_stride(arg192_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg193_1, (3072, ), (1, ))
    assert_size_stride(arg194_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg195_1, (3072, ), (1, ))
    assert_size_stride(arg196_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg197_1, (3072, ), (1, ))
    assert_size_stride(arg198_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg199_1, (3072, ), (1, ))
    assert_size_stride(arg200_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg201_1, (3072, ), (1, ))
    assert_size_stride(arg202_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg203_1, (3072, ), (1, ))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (128, ), (1, ))
    assert_size_stride(arg207_1, (128, ), (1, ))
    assert_size_stride(arg208_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg209_1, (3072, ), (1, ))
    assert_size_stride(arg210_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg211_1, (3072, ), (1, ))
    assert_size_stride(arg212_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg213_1, (12288, ), (1, ))
    assert_size_stride(arg214_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg215_1, (3072, ), (1, ))
    assert_size_stride(arg216_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg217_1, (12288, ), (1, ))
    assert_size_stride(arg218_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg219_1, (3072, ), (1, ))
    assert_size_stride(arg220_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg221_1, (18432, ), (1, ))
    assert_size_stride(arg222_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg223_1, (18432, ), (1, ))
    assert_size_stride(arg224_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg225_1, (3072, ), (1, ))
    assert_size_stride(arg226_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg227_1, (3072, ), (1, ))
    assert_size_stride(arg228_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg229_1, (3072, ), (1, ))
    assert_size_stride(arg230_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg231_1, (3072, ), (1, ))
    assert_size_stride(arg232_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg233_1, (3072, ), (1, ))
    assert_size_stride(arg234_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg235_1, (3072, ), (1, ))
    assert_size_stride(arg236_1, (128, ), (1, ))
    assert_size_stride(arg237_1, (128, ), (1, ))
    assert_size_stride(arg238_1, (128, ), (1, ))
    assert_size_stride(arg239_1, (128, ), (1, ))
    assert_size_stride(arg240_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg241_1, (3072, ), (1, ))
    assert_size_stride(arg242_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg243_1, (3072, ), (1, ))
    assert_size_stride(arg244_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg245_1, (12288, ), (1, ))
    assert_size_stride(arg246_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg247_1, (3072, ), (1, ))
    assert_size_stride(arg248_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg249_1, (12288, ), (1, ))
    assert_size_stride(arg250_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg251_1, (3072, ), (1, ))
    assert_size_stride(arg252_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg253_1, (18432, ), (1, ))
    assert_size_stride(arg254_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg255_1, (18432, ), (1, ))
    assert_size_stride(arg256_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg257_1, (3072, ), (1, ))
    assert_size_stride(arg258_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg259_1, (3072, ), (1, ))
    assert_size_stride(arg260_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg261_1, (3072, ), (1, ))
    assert_size_stride(arg262_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg263_1, (3072, ), (1, ))
    assert_size_stride(arg264_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg265_1, (3072, ), (1, ))
    assert_size_stride(arg266_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg267_1, (3072, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (128, ), (1, ))
    assert_size_stride(arg270_1, (128, ), (1, ))
    assert_size_stride(arg271_1, (128, ), (1, ))
    assert_size_stride(arg272_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg273_1, (3072, ), (1, ))
    assert_size_stride(arg274_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg275_1, (3072, ), (1, ))
    assert_size_stride(arg276_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg277_1, (12288, ), (1, ))
    assert_size_stride(arg278_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg279_1, (3072, ), (1, ))
    assert_size_stride(arg280_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg281_1, (12288, ), (1, ))
    assert_size_stride(arg282_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg283_1, (3072, ), (1, ))
    assert_size_stride(arg284_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg285_1, (18432, ), (1, ))
    assert_size_stride(arg286_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg287_1, (18432, ), (1, ))
    assert_size_stride(arg288_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg289_1, (3072, ), (1, ))
    assert_size_stride(arg290_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg291_1, (3072, ), (1, ))
    assert_size_stride(arg292_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg293_1, (3072, ), (1, ))
    assert_size_stride(arg294_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg295_1, (3072, ), (1, ))
    assert_size_stride(arg296_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg297_1, (3072, ), (1, ))
    assert_size_stride(arg298_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg299_1, (3072, ), (1, ))
    assert_size_stride(arg300_1, (128, ), (1, ))
    assert_size_stride(arg301_1, (128, ), (1, ))
    assert_size_stride(arg302_1, (128, ), (1, ))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg305_1, (3072, ), (1, ))
    assert_size_stride(arg306_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg307_1, (3072, ), (1, ))
    assert_size_stride(arg308_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg309_1, (12288, ), (1, ))
    assert_size_stride(arg310_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg311_1, (3072, ), (1, ))
    assert_size_stride(arg312_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg313_1, (12288, ), (1, ))
    assert_size_stride(arg314_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg315_1, (3072, ), (1, ))
    assert_size_stride(arg316_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg317_1, (18432, ), (1, ))
    assert_size_stride(arg318_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg319_1, (18432, ), (1, ))
    assert_size_stride(arg320_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg321_1, (3072, ), (1, ))
    assert_size_stride(arg322_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg323_1, (3072, ), (1, ))
    assert_size_stride(arg324_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg325_1, (3072, ), (1, ))
    assert_size_stride(arg326_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg327_1, (3072, ), (1, ))
    assert_size_stride(arg328_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg329_1, (3072, ), (1, ))
    assert_size_stride(arg330_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg331_1, (3072, ), (1, ))
    assert_size_stride(arg332_1, (128, ), (1, ))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg337_1, (3072, ), (1, ))
    assert_size_stride(arg338_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg339_1, (3072, ), (1, ))
    assert_size_stride(arg340_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg341_1, (12288, ), (1, ))
    assert_size_stride(arg342_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg343_1, (3072, ), (1, ))
    assert_size_stride(arg344_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg345_1, (12288, ), (1, ))
    assert_size_stride(arg346_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg347_1, (3072, ), (1, ))
    assert_size_stride(arg348_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg349_1, (18432, ), (1, ))
    assert_size_stride(arg350_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg351_1, (18432, ), (1, ))
    assert_size_stride(arg352_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg353_1, (3072, ), (1, ))
    assert_size_stride(arg354_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg355_1, (3072, ), (1, ))
    assert_size_stride(arg356_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg357_1, (3072, ), (1, ))
    assert_size_stride(arg358_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg359_1, (3072, ), (1, ))
    assert_size_stride(arg360_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg361_1, (3072, ), (1, ))
    assert_size_stride(arg362_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg363_1, (3072, ), (1, ))
    assert_size_stride(arg364_1, (128, ), (1, ))
    assert_size_stride(arg365_1, (128, ), (1, ))
    assert_size_stride(arg366_1, (128, ), (1, ))
    assert_size_stride(arg367_1, (128, ), (1, ))
    assert_size_stride(arg368_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg369_1, (3072, ), (1, ))
    assert_size_stride(arg370_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg371_1, (3072, ), (1, ))
    assert_size_stride(arg372_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg373_1, (12288, ), (1, ))
    assert_size_stride(arg374_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg375_1, (3072, ), (1, ))
    assert_size_stride(arg376_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg377_1, (12288, ), (1, ))
    assert_size_stride(arg378_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg379_1, (3072, ), (1, ))
    assert_size_stride(arg380_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg381_1, (18432, ), (1, ))
    assert_size_stride(arg382_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg383_1, (18432, ), (1, ))
    assert_size_stride(arg384_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg385_1, (3072, ), (1, ))
    assert_size_stride(arg386_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg387_1, (3072, ), (1, ))
    assert_size_stride(arg388_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg389_1, (3072, ), (1, ))
    assert_size_stride(arg390_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg391_1, (3072, ), (1, ))
    assert_size_stride(arg392_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg393_1, (3072, ), (1, ))
    assert_size_stride(arg394_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg395_1, (3072, ), (1, ))
    assert_size_stride(arg396_1, (128, ), (1, ))
    assert_size_stride(arg397_1, (128, ), (1, ))
    assert_size_stride(arg398_1, (128, ), (1, ))
    assert_size_stride(arg399_1, (128, ), (1, ))
    assert_size_stride(arg400_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg401_1, (3072, ), (1, ))
    assert_size_stride(arg402_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg403_1, (3072, ), (1, ))
    assert_size_stride(arg404_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg405_1, (12288, ), (1, ))
    assert_size_stride(arg406_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg407_1, (3072, ), (1, ))
    assert_size_stride(arg408_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg409_1, (12288, ), (1, ))
    assert_size_stride(arg410_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg411_1, (3072, ), (1, ))
    assert_size_stride(arg412_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg413_1, (18432, ), (1, ))
    assert_size_stride(arg414_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg415_1, (18432, ), (1, ))
    assert_size_stride(arg416_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg417_1, (3072, ), (1, ))
    assert_size_stride(arg418_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg419_1, (3072, ), (1, ))
    assert_size_stride(arg420_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg421_1, (3072, ), (1, ))
    assert_size_stride(arg422_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg423_1, (3072, ), (1, ))
    assert_size_stride(arg424_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg425_1, (3072, ), (1, ))
    assert_size_stride(arg426_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg427_1, (3072, ), (1, ))
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (128, ), (1, ))
    assert_size_stride(arg430_1, (128, ), (1, ))
    assert_size_stride(arg431_1, (128, ), (1, ))
    assert_size_stride(arg432_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg433_1, (3072, ), (1, ))
    assert_size_stride(arg434_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg435_1, (3072, ), (1, ))
    assert_size_stride(arg436_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg437_1, (12288, ), (1, ))
    assert_size_stride(arg438_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg439_1, (3072, ), (1, ))
    assert_size_stride(arg440_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg441_1, (12288, ), (1, ))
    assert_size_stride(arg442_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg443_1, (3072, ), (1, ))
    assert_size_stride(arg444_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg445_1, (18432, ), (1, ))
    assert_size_stride(arg446_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg447_1, (18432, ), (1, ))
    assert_size_stride(arg448_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg449_1, (3072, ), (1, ))
    assert_size_stride(arg450_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg451_1, (3072, ), (1, ))
    assert_size_stride(arg452_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg453_1, (3072, ), (1, ))
    assert_size_stride(arg454_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg455_1, (3072, ), (1, ))
    assert_size_stride(arg456_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg457_1, (3072, ), (1, ))
    assert_size_stride(arg458_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg459_1, (3072, ), (1, ))
    assert_size_stride(arg460_1, (128, ), (1, ))
    assert_size_stride(arg461_1, (128, ), (1, ))
    assert_size_stride(arg462_1, (128, ), (1, ))
    assert_size_stride(arg463_1, (128, ), (1, ))
    assert_size_stride(arg464_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg465_1, (3072, ), (1, ))
    assert_size_stride(arg466_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg467_1, (3072, ), (1, ))
    assert_size_stride(arg468_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg469_1, (12288, ), (1, ))
    assert_size_stride(arg470_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg471_1, (3072, ), (1, ))
    assert_size_stride(arg472_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg473_1, (12288, ), (1, ))
    assert_size_stride(arg474_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg475_1, (3072, ), (1, ))
    assert_size_stride(arg476_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg477_1, (18432, ), (1, ))
    assert_size_stride(arg478_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg479_1, (18432, ), (1, ))
    assert_size_stride(arg480_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg481_1, (3072, ), (1, ))
    assert_size_stride(arg482_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg483_1, (3072, ), (1, ))
    assert_size_stride(arg484_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg485_1, (3072, ), (1, ))
    assert_size_stride(arg486_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg487_1, (3072, ), (1, ))
    assert_size_stride(arg488_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg489_1, (3072, ), (1, ))
    assert_size_stride(arg490_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg491_1, (3072, ), (1, ))
    assert_size_stride(arg492_1, (128, ), (1, ))
    assert_size_stride(arg493_1, (128, ), (1, ))
    assert_size_stride(arg494_1, (128, ), (1, ))
    assert_size_stride(arg495_1, (128, ), (1, ))
    assert_size_stride(arg496_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg497_1, (3072, ), (1, ))
    assert_size_stride(arg498_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg499_1, (3072, ), (1, ))
    assert_size_stride(arg500_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg501_1, (12288, ), (1, ))
    assert_size_stride(arg502_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg503_1, (3072, ), (1, ))
    assert_size_stride(arg504_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg505_1, (12288, ), (1, ))
    assert_size_stride(arg506_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg507_1, (3072, ), (1, ))
    assert_size_stride(arg508_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg509_1, (18432, ), (1, ))
    assert_size_stride(arg510_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg511_1, (18432, ), (1, ))
    assert_size_stride(arg512_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg513_1, (3072, ), (1, ))
    assert_size_stride(arg514_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg515_1, (3072, ), (1, ))
    assert_size_stride(arg516_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg517_1, (3072, ), (1, ))
    assert_size_stride(arg518_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg519_1, (3072, ), (1, ))
    assert_size_stride(arg520_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg521_1, (3072, ), (1, ))
    assert_size_stride(arg522_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg523_1, (3072, ), (1, ))
    assert_size_stride(arg524_1, (128, ), (1, ))
    assert_size_stride(arg525_1, (128, ), (1, ))
    assert_size_stride(arg526_1, (128, ), (1, ))
    assert_size_stride(arg527_1, (128, ), (1, ))
    assert_size_stride(arg528_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg529_1, (3072, ), (1, ))
    assert_size_stride(arg530_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg531_1, (3072, ), (1, ))
    assert_size_stride(arg532_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg533_1, (12288, ), (1, ))
    assert_size_stride(arg534_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg535_1, (3072, ), (1, ))
    assert_size_stride(arg536_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg537_1, (12288, ), (1, ))
    assert_size_stride(arg538_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg539_1, (3072, ), (1, ))
    assert_size_stride(arg540_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg541_1, (18432, ), (1, ))
    assert_size_stride(arg542_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg543_1, (18432, ), (1, ))
    assert_size_stride(arg544_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg545_1, (3072, ), (1, ))
    assert_size_stride(arg546_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg547_1, (3072, ), (1, ))
    assert_size_stride(arg548_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg549_1, (3072, ), (1, ))
    assert_size_stride(arg550_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg551_1, (3072, ), (1, ))
    assert_size_stride(arg552_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg553_1, (3072, ), (1, ))
    assert_size_stride(arg554_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg555_1, (3072, ), (1, ))
    assert_size_stride(arg556_1, (128, ), (1, ))
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (128, ), (1, ))
    assert_size_stride(arg559_1, (128, ), (1, ))
    assert_size_stride(arg560_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg561_1, (3072, ), (1, ))
    assert_size_stride(arg562_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg563_1, (3072, ), (1, ))
    assert_size_stride(arg564_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg565_1, (12288, ), (1, ))
    assert_size_stride(arg566_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg567_1, (3072, ), (1, ))
    assert_size_stride(arg568_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg569_1, (12288, ), (1, ))
    assert_size_stride(arg570_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg571_1, (3072, ), (1, ))
    assert_size_stride(arg572_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg573_1, (18432, ), (1, ))
    assert_size_stride(arg574_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg575_1, (18432, ), (1, ))
    assert_size_stride(arg576_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg577_1, (3072, ), (1, ))
    assert_size_stride(arg578_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg579_1, (3072, ), (1, ))
    assert_size_stride(arg580_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg581_1, (3072, ), (1, ))
    assert_size_stride(arg582_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg583_1, (3072, ), (1, ))
    assert_size_stride(arg584_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg585_1, (3072, ), (1, ))
    assert_size_stride(arg586_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg587_1, (3072, ), (1, ))
    assert_size_stride(arg588_1, (128, ), (1, ))
    assert_size_stride(arg589_1, (128, ), (1, ))
    assert_size_stride(arg590_1, (128, ), (1, ))
    assert_size_stride(arg591_1, (128, ), (1, ))
    assert_size_stride(arg592_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg593_1, (3072, ), (1, ))
    assert_size_stride(arg594_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg595_1, (3072, ), (1, ))
    assert_size_stride(arg596_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg597_1, (12288, ), (1, ))
    assert_size_stride(arg598_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg599_1, (3072, ), (1, ))
    assert_size_stride(arg600_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg601_1, (12288, ), (1, ))
    assert_size_stride(arg602_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg603_1, (3072, ), (1, ))
    assert_size_stride(arg604_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg605_1, (18432, ), (1, ))
    assert_size_stride(arg606_1, (18432, 3072), (3072, 1))
    assert_size_stride(arg607_1, (18432, ), (1, ))
    assert_size_stride(arg608_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg609_1, (3072, ), (1, ))
    assert_size_stride(arg610_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg611_1, (3072, ), (1, ))
    assert_size_stride(arg612_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg613_1, (3072, ), (1, ))
    assert_size_stride(arg614_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg615_1, (3072, ), (1, ))
    assert_size_stride(arg616_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg617_1, (3072, ), (1, ))
    assert_size_stride(arg618_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg619_1, (3072, ), (1, ))
    assert_size_stride(arg620_1, (128, ), (1, ))
    assert_size_stride(arg621_1, (128, ), (1, ))
    assert_size_stride(arg622_1, (128, ), (1, ))
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg625_1, (3072, ), (1, ))
    assert_size_stride(arg626_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg627_1, (3072, ), (1, ))
    assert_size_stride(arg628_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg629_1, (12288, ), (1, ))
    assert_size_stride(arg630_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg631_1, (3072, ), (1, ))
    assert_size_stride(arg632_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg633_1, (12288, ), (1, ))
    assert_size_stride(arg634_1, (3072, 12288), (12288, 1))
    assert_size_stride(arg635_1, (3072, ), (1, ))
    assert_size_stride(arg636_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg637_1, (9216, ), (1, ))
    assert_size_stride(arg638_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg639_1, (12288, ), (1, ))
    assert_size_stride(arg640_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg641_1, (3072, ), (1, ))
    assert_size_stride(arg642_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg643_1, (3072, ), (1, ))
    assert_size_stride(arg644_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg645_1, (3072, ), (1, ))
    assert_size_stride(arg646_1, (128, ), (1, ))
    assert_size_stride(arg647_1, (128, ), (1, ))
    assert_size_stride(arg648_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg649_1, (3072, ), (1, ))
    assert_size_stride(arg650_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg651_1, (9216, ), (1, ))
    assert_size_stride(arg652_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg653_1, (12288, ), (1, ))
    assert_size_stride(arg654_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg655_1, (3072, ), (1, ))
    assert_size_stride(arg656_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg657_1, (3072, ), (1, ))
    assert_size_stride(arg658_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg659_1, (3072, ), (1, ))
    assert_size_stride(arg660_1, (128, ), (1, ))
    assert_size_stride(arg661_1, (128, ), (1, ))
    assert_size_stride(arg662_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg663_1, (3072, ), (1, ))
    assert_size_stride(arg664_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg665_1, (9216, ), (1, ))
    assert_size_stride(arg666_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg667_1, (12288, ), (1, ))
    assert_size_stride(arg668_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg669_1, (3072, ), (1, ))
    assert_size_stride(arg670_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg671_1, (3072, ), (1, ))
    assert_size_stride(arg672_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg673_1, (3072, ), (1, ))
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (128, ), (1, ))
    assert_size_stride(arg676_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg677_1, (3072, ), (1, ))
    assert_size_stride(arg678_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg679_1, (9216, ), (1, ))
    assert_size_stride(arg680_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg681_1, (12288, ), (1, ))
    assert_size_stride(arg682_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg683_1, (3072, ), (1, ))
    assert_size_stride(arg684_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg685_1, (3072, ), (1, ))
    assert_size_stride(arg686_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg687_1, (3072, ), (1, ))
    assert_size_stride(arg688_1, (128, ), (1, ))
    assert_size_stride(arg689_1, (128, ), (1, ))
    assert_size_stride(arg690_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg691_1, (3072, ), (1, ))
    assert_size_stride(arg692_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg693_1, (9216, ), (1, ))
    assert_size_stride(arg694_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg695_1, (12288, ), (1, ))
    assert_size_stride(arg696_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg697_1, (3072, ), (1, ))
    assert_size_stride(arg698_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg699_1, (3072, ), (1, ))
    assert_size_stride(arg700_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg701_1, (3072, ), (1, ))
    assert_size_stride(arg702_1, (128, ), (1, ))
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg705_1, (3072, ), (1, ))
    assert_size_stride(arg706_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg707_1, (9216, ), (1, ))
    assert_size_stride(arg708_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg709_1, (12288, ), (1, ))
    assert_size_stride(arg710_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg711_1, (3072, ), (1, ))
    assert_size_stride(arg712_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg713_1, (3072, ), (1, ))
    assert_size_stride(arg714_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg715_1, (3072, ), (1, ))
    assert_size_stride(arg716_1, (128, ), (1, ))
    assert_size_stride(arg717_1, (128, ), (1, ))
    assert_size_stride(arg718_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg719_1, (3072, ), (1, ))
    assert_size_stride(arg720_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg721_1, (9216, ), (1, ))
    assert_size_stride(arg722_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg723_1, (12288, ), (1, ))
    assert_size_stride(arg724_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg725_1, (3072, ), (1, ))
    assert_size_stride(arg726_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg727_1, (3072, ), (1, ))
    assert_size_stride(arg728_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg729_1, (3072, ), (1, ))
    assert_size_stride(arg730_1, (128, ), (1, ))
    assert_size_stride(arg731_1, (128, ), (1, ))
    assert_size_stride(arg732_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg733_1, (3072, ), (1, ))
    assert_size_stride(arg734_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg735_1, (9216, ), (1, ))
    assert_size_stride(arg736_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg737_1, (12288, ), (1, ))
    assert_size_stride(arg738_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg739_1, (3072, ), (1, ))
    assert_size_stride(arg740_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg741_1, (3072, ), (1, ))
    assert_size_stride(arg742_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg743_1, (3072, ), (1, ))
    assert_size_stride(arg744_1, (128, ), (1, ))
    assert_size_stride(arg745_1, (128, ), (1, ))
    assert_size_stride(arg746_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg747_1, (3072, ), (1, ))
    assert_size_stride(arg748_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg749_1, (9216, ), (1, ))
    assert_size_stride(arg750_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg751_1, (12288, ), (1, ))
    assert_size_stride(arg752_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg753_1, (3072, ), (1, ))
    assert_size_stride(arg754_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg755_1, (3072, ), (1, ))
    assert_size_stride(arg756_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg757_1, (3072, ), (1, ))
    assert_size_stride(arg758_1, (128, ), (1, ))
    assert_size_stride(arg759_1, (128, ), (1, ))
    assert_size_stride(arg760_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg761_1, (3072, ), (1, ))
    assert_size_stride(arg762_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg763_1, (9216, ), (1, ))
    assert_size_stride(arg764_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg765_1, (12288, ), (1, ))
    assert_size_stride(arg766_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg767_1, (3072, ), (1, ))
    assert_size_stride(arg768_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg769_1, (3072, ), (1, ))
    assert_size_stride(arg770_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg771_1, (3072, ), (1, ))
    assert_size_stride(arg772_1, (128, ), (1, ))
    assert_size_stride(arg773_1, (128, ), (1, ))
    assert_size_stride(arg774_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg775_1, (3072, ), (1, ))
    assert_size_stride(arg776_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg777_1, (9216, ), (1, ))
    assert_size_stride(arg778_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg779_1, (12288, ), (1, ))
    assert_size_stride(arg780_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg781_1, (3072, ), (1, ))
    assert_size_stride(arg782_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg783_1, (3072, ), (1, ))
    assert_size_stride(arg784_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg785_1, (3072, ), (1, ))
    assert_size_stride(arg786_1, (128, ), (1, ))
    assert_size_stride(arg787_1, (128, ), (1, ))
    assert_size_stride(arg788_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg789_1, (3072, ), (1, ))
    assert_size_stride(arg790_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg791_1, (9216, ), (1, ))
    assert_size_stride(arg792_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg793_1, (12288, ), (1, ))
    assert_size_stride(arg794_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg795_1, (3072, ), (1, ))
    assert_size_stride(arg796_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg797_1, (3072, ), (1, ))
    assert_size_stride(arg798_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg799_1, (3072, ), (1, ))
    assert_size_stride(arg800_1, (128, ), (1, ))
    assert_size_stride(arg801_1, (128, ), (1, ))
    assert_size_stride(arg802_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg803_1, (3072, ), (1, ))
    assert_size_stride(arg804_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg805_1, (9216, ), (1, ))
    assert_size_stride(arg806_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg807_1, (12288, ), (1, ))
    assert_size_stride(arg808_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg809_1, (3072, ), (1, ))
    assert_size_stride(arg810_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg811_1, (3072, ), (1, ))
    assert_size_stride(arg812_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg813_1, (3072, ), (1, ))
    assert_size_stride(arg814_1, (128, ), (1, ))
    assert_size_stride(arg815_1, (128, ), (1, ))
    assert_size_stride(arg816_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg817_1, (3072, ), (1, ))
    assert_size_stride(arg818_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg819_1, (9216, ), (1, ))
    assert_size_stride(arg820_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg821_1, (12288, ), (1, ))
    assert_size_stride(arg822_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg823_1, (3072, ), (1, ))
    assert_size_stride(arg824_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg825_1, (3072, ), (1, ))
    assert_size_stride(arg826_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg827_1, (3072, ), (1, ))
    assert_size_stride(arg828_1, (128, ), (1, ))
    assert_size_stride(arg829_1, (128, ), (1, ))
    assert_size_stride(arg830_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg831_1, (3072, ), (1, ))
    assert_size_stride(arg832_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg833_1, (9216, ), (1, ))
    assert_size_stride(arg834_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg835_1, (12288, ), (1, ))
    assert_size_stride(arg836_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg837_1, (3072, ), (1, ))
    assert_size_stride(arg838_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg839_1, (3072, ), (1, ))
    assert_size_stride(arg840_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg841_1, (3072, ), (1, ))
    assert_size_stride(arg842_1, (128, ), (1, ))
    assert_size_stride(arg843_1, (128, ), (1, ))
    assert_size_stride(arg844_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg845_1, (3072, ), (1, ))
    assert_size_stride(arg846_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg847_1, (9216, ), (1, ))
    assert_size_stride(arg848_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg849_1, (12288, ), (1, ))
    assert_size_stride(arg850_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg851_1, (3072, ), (1, ))
    assert_size_stride(arg852_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg853_1, (3072, ), (1, ))
    assert_size_stride(arg854_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg855_1, (3072, ), (1, ))
    assert_size_stride(arg856_1, (128, ), (1, ))
    assert_size_stride(arg857_1, (128, ), (1, ))
    assert_size_stride(arg858_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg859_1, (3072, ), (1, ))
    assert_size_stride(arg860_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg861_1, (9216, ), (1, ))
    assert_size_stride(arg862_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg863_1, (12288, ), (1, ))
    assert_size_stride(arg864_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg865_1, (3072, ), (1, ))
    assert_size_stride(arg866_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg867_1, (3072, ), (1, ))
    assert_size_stride(arg868_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg869_1, (3072, ), (1, ))
    assert_size_stride(arg870_1, (128, ), (1, ))
    assert_size_stride(arg871_1, (128, ), (1, ))
    assert_size_stride(arg872_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg873_1, (3072, ), (1, ))
    assert_size_stride(arg874_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg875_1, (9216, ), (1, ))
    assert_size_stride(arg876_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg877_1, (12288, ), (1, ))
    assert_size_stride(arg878_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg879_1, (3072, ), (1, ))
    assert_size_stride(arg880_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg881_1, (3072, ), (1, ))
    assert_size_stride(arg882_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg883_1, (3072, ), (1, ))
    assert_size_stride(arg884_1, (128, ), (1, ))
    assert_size_stride(arg885_1, (128, ), (1, ))
    assert_size_stride(arg886_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg887_1, (3072, ), (1, ))
    assert_size_stride(arg888_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg889_1, (9216, ), (1, ))
    assert_size_stride(arg890_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg891_1, (12288, ), (1, ))
    assert_size_stride(arg892_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg893_1, (3072, ), (1, ))
    assert_size_stride(arg894_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg895_1, (3072, ), (1, ))
    assert_size_stride(arg896_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg897_1, (3072, ), (1, ))
    assert_size_stride(arg898_1, (128, ), (1, ))
    assert_size_stride(arg899_1, (128, ), (1, ))
    assert_size_stride(arg900_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg901_1, (3072, ), (1, ))
    assert_size_stride(arg902_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg903_1, (9216, ), (1, ))
    assert_size_stride(arg904_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg905_1, (12288, ), (1, ))
    assert_size_stride(arg906_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg907_1, (3072, ), (1, ))
    assert_size_stride(arg908_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg909_1, (3072, ), (1, ))
    assert_size_stride(arg910_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg911_1, (3072, ), (1, ))
    assert_size_stride(arg912_1, (128, ), (1, ))
    assert_size_stride(arg913_1, (128, ), (1, ))
    assert_size_stride(arg914_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg915_1, (3072, ), (1, ))
    assert_size_stride(arg916_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg917_1, (9216, ), (1, ))
    assert_size_stride(arg918_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg919_1, (12288, ), (1, ))
    assert_size_stride(arg920_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg921_1, (3072, ), (1, ))
    assert_size_stride(arg922_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg923_1, (3072, ), (1, ))
    assert_size_stride(arg924_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg925_1, (3072, ), (1, ))
    assert_size_stride(arg926_1, (128, ), (1, ))
    assert_size_stride(arg927_1, (128, ), (1, ))
    assert_size_stride(arg928_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg929_1, (3072, ), (1, ))
    assert_size_stride(arg930_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg931_1, (9216, ), (1, ))
    assert_size_stride(arg932_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg933_1, (12288, ), (1, ))
    assert_size_stride(arg934_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg935_1, (3072, ), (1, ))
    assert_size_stride(arg936_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg937_1, (3072, ), (1, ))
    assert_size_stride(arg938_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg939_1, (3072, ), (1, ))
    assert_size_stride(arg940_1, (128, ), (1, ))
    assert_size_stride(arg941_1, (128, ), (1, ))
    assert_size_stride(arg942_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg943_1, (3072, ), (1, ))
    assert_size_stride(arg944_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg945_1, (9216, ), (1, ))
    assert_size_stride(arg946_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg947_1, (12288, ), (1, ))
    assert_size_stride(arg948_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg949_1, (3072, ), (1, ))
    assert_size_stride(arg950_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg951_1, (3072, ), (1, ))
    assert_size_stride(arg952_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg953_1, (3072, ), (1, ))
    assert_size_stride(arg954_1, (128, ), (1, ))
    assert_size_stride(arg955_1, (128, ), (1, ))
    assert_size_stride(arg956_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg957_1, (3072, ), (1, ))
    assert_size_stride(arg958_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg959_1, (9216, ), (1, ))
    assert_size_stride(arg960_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg961_1, (12288, ), (1, ))
    assert_size_stride(arg962_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg963_1, (3072, ), (1, ))
    assert_size_stride(arg964_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg965_1, (3072, ), (1, ))
    assert_size_stride(arg966_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg967_1, (3072, ), (1, ))
    assert_size_stride(arg968_1, (128, ), (1, ))
    assert_size_stride(arg969_1, (128, ), (1, ))
    assert_size_stride(arg970_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg971_1, (3072, ), (1, ))
    assert_size_stride(arg972_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg973_1, (9216, ), (1, ))
    assert_size_stride(arg974_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg975_1, (12288, ), (1, ))
    assert_size_stride(arg976_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg977_1, (3072, ), (1, ))
    assert_size_stride(arg978_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg979_1, (3072, ), (1, ))
    assert_size_stride(arg980_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg981_1, (3072, ), (1, ))
    assert_size_stride(arg982_1, (128, ), (1, ))
    assert_size_stride(arg983_1, (128, ), (1, ))
    assert_size_stride(arg984_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg985_1, (3072, ), (1, ))
    assert_size_stride(arg986_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg987_1, (9216, ), (1, ))
    assert_size_stride(arg988_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg989_1, (12288, ), (1, ))
    assert_size_stride(arg990_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg991_1, (3072, ), (1, ))
    assert_size_stride(arg992_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg993_1, (3072, ), (1, ))
    assert_size_stride(arg994_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg995_1, (3072, ), (1, ))
    assert_size_stride(arg996_1, (128, ), (1, ))
    assert_size_stride(arg997_1, (128, ), (1, ))
    assert_size_stride(arg998_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg999_1, (3072, ), (1, ))
    assert_size_stride(arg1000_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1001_1, (9216, ), (1, ))
    assert_size_stride(arg1002_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1003_1, (12288, ), (1, ))
    assert_size_stride(arg1004_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1005_1, (3072, ), (1, ))
    assert_size_stride(arg1006_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1007_1, (3072, ), (1, ))
    assert_size_stride(arg1008_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1009_1, (3072, ), (1, ))
    assert_size_stride(arg1010_1, (128, ), (1, ))
    assert_size_stride(arg1011_1, (128, ), (1, ))
    assert_size_stride(arg1012_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1013_1, (3072, ), (1, ))
    assert_size_stride(arg1014_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1015_1, (9216, ), (1, ))
    assert_size_stride(arg1016_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1017_1, (12288, ), (1, ))
    assert_size_stride(arg1018_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1019_1, (3072, ), (1, ))
    assert_size_stride(arg1020_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1021_1, (3072, ), (1, ))
    assert_size_stride(arg1022_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1023_1, (3072, ), (1, ))
    assert_size_stride(arg1024_1, (128, ), (1, ))
    assert_size_stride(arg1025_1, (128, ), (1, ))
    assert_size_stride(arg1026_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1027_1, (3072, ), (1, ))
    assert_size_stride(arg1028_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1029_1, (9216, ), (1, ))
    assert_size_stride(arg1030_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1031_1, (12288, ), (1, ))
    assert_size_stride(arg1032_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1033_1, (3072, ), (1, ))
    assert_size_stride(arg1034_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1035_1, (3072, ), (1, ))
    assert_size_stride(arg1036_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1037_1, (3072, ), (1, ))
    assert_size_stride(arg1038_1, (128, ), (1, ))
    assert_size_stride(arg1039_1, (128, ), (1, ))
    assert_size_stride(arg1040_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1041_1, (3072, ), (1, ))
    assert_size_stride(arg1042_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1043_1, (9216, ), (1, ))
    assert_size_stride(arg1044_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1045_1, (12288, ), (1, ))
    assert_size_stride(arg1046_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1047_1, (3072, ), (1, ))
    assert_size_stride(arg1048_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1049_1, (3072, ), (1, ))
    assert_size_stride(arg1050_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1051_1, (3072, ), (1, ))
    assert_size_stride(arg1052_1, (128, ), (1, ))
    assert_size_stride(arg1053_1, (128, ), (1, ))
    assert_size_stride(arg1054_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1055_1, (3072, ), (1, ))
    assert_size_stride(arg1056_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1057_1, (9216, ), (1, ))
    assert_size_stride(arg1058_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1059_1, (12288, ), (1, ))
    assert_size_stride(arg1060_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1061_1, (3072, ), (1, ))
    assert_size_stride(arg1062_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1063_1, (3072, ), (1, ))
    assert_size_stride(arg1064_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1065_1, (3072, ), (1, ))
    assert_size_stride(arg1066_1, (128, ), (1, ))
    assert_size_stride(arg1067_1, (128, ), (1, ))
    assert_size_stride(arg1068_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1069_1, (3072, ), (1, ))
    assert_size_stride(arg1070_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1071_1, (9216, ), (1, ))
    assert_size_stride(arg1072_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1073_1, (12288, ), (1, ))
    assert_size_stride(arg1074_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1075_1, (3072, ), (1, ))
    assert_size_stride(arg1076_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1077_1, (3072, ), (1, ))
    assert_size_stride(arg1078_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1079_1, (3072, ), (1, ))
    assert_size_stride(arg1080_1, (128, ), (1, ))
    assert_size_stride(arg1081_1, (128, ), (1, ))
    assert_size_stride(arg1082_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1083_1, (3072, ), (1, ))
    assert_size_stride(arg1084_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1085_1, (9216, ), (1, ))
    assert_size_stride(arg1086_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1087_1, (12288, ), (1, ))
    assert_size_stride(arg1088_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1089_1, (3072, ), (1, ))
    assert_size_stride(arg1090_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1091_1, (3072, ), (1, ))
    assert_size_stride(arg1092_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1093_1, (3072, ), (1, ))
    assert_size_stride(arg1094_1, (128, ), (1, ))
    assert_size_stride(arg1095_1, (128, ), (1, ))
    assert_size_stride(arg1096_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1097_1, (3072, ), (1, ))
    assert_size_stride(arg1098_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1099_1, (9216, ), (1, ))
    assert_size_stride(arg1100_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1101_1, (12288, ), (1, ))
    assert_size_stride(arg1102_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1103_1, (3072, ), (1, ))
    assert_size_stride(arg1104_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1105_1, (3072, ), (1, ))
    assert_size_stride(arg1106_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1107_1, (3072, ), (1, ))
    assert_size_stride(arg1108_1, (128, ), (1, ))
    assert_size_stride(arg1109_1, (128, ), (1, ))
    assert_size_stride(arg1110_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1111_1, (3072, ), (1, ))
    assert_size_stride(arg1112_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1113_1, (9216, ), (1, ))
    assert_size_stride(arg1114_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1115_1, (12288, ), (1, ))
    assert_size_stride(arg1116_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1117_1, (3072, ), (1, ))
    assert_size_stride(arg1118_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1119_1, (3072, ), (1, ))
    assert_size_stride(arg1120_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1121_1, (3072, ), (1, ))
    assert_size_stride(arg1122_1, (128, ), (1, ))
    assert_size_stride(arg1123_1, (128, ), (1, ))
    assert_size_stride(arg1124_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1125_1, (3072, ), (1, ))
    assert_size_stride(arg1126_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1127_1, (9216, ), (1, ))
    assert_size_stride(arg1128_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1129_1, (12288, ), (1, ))
    assert_size_stride(arg1130_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1131_1, (3072, ), (1, ))
    assert_size_stride(arg1132_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1133_1, (3072, ), (1, ))
    assert_size_stride(arg1134_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1135_1, (3072, ), (1, ))
    assert_size_stride(arg1136_1, (128, ), (1, ))
    assert_size_stride(arg1137_1, (128, ), (1, ))
    assert_size_stride(arg1138_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1139_1, (3072, ), (1, ))
    assert_size_stride(arg1140_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1141_1, (9216, ), (1, ))
    assert_size_stride(arg1142_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1143_1, (12288, ), (1, ))
    assert_size_stride(arg1144_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1145_1, (3072, ), (1, ))
    assert_size_stride(arg1146_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1147_1, (3072, ), (1, ))
    assert_size_stride(arg1148_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1149_1, (3072, ), (1, ))
    assert_size_stride(arg1150_1, (128, ), (1, ))
    assert_size_stride(arg1151_1, (128, ), (1, ))
    assert_size_stride(arg1152_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1153_1, (3072, ), (1, ))
    assert_size_stride(arg1154_1, (9216, 3072), (3072, 1))
    assert_size_stride(arg1155_1, (9216, ), (1, ))
    assert_size_stride(arg1156_1, (12288, 3072), (3072, 1))
    assert_size_stride(arg1157_1, (12288, ), (1, ))
    assert_size_stride(arg1158_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1159_1, (3072, ), (1, ))
    assert_size_stride(arg1160_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1161_1, (3072, ), (1, ))
    assert_size_stride(arg1162_1, (3072, 3072), (3072, 1))
    assert_size_stride(arg1163_1, (3072, ), (1, ))
    assert_size_stride(arg1164_1, (128, ), (1, ))
    assert_size_stride(arg1165_1, (128, ), (1, ))
    assert_size_stride(arg1166_1, (3072, 15360), (15360, 1))
    assert_size_stride(arg1167_1, (3072, ), (1, ))
    assert_size_stride(arg1168_1, (6144, 3072), (3072, 1))
    assert_size_stride(arg1169_1, (6144, ), (1, ))
    assert_size_stride(arg1170_1, (64, 3072), (3072, 1))
    assert_size_stride(arg1171_1, (64, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 2*(s2 // 2)), (2*(s2 // 2), 1), torch.float32)
        # Topologically Sorted Source Nodes: [emb_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_0_xnumel = 2*(s2 // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg3_1, buf0, s2, s3, s4, triton_poi_fused_cat_0_xnumel, grid=grid(triton_poi_fused_cat_0_xnumel), stream=stream0)
        del arg3_1
        buf1 = empty_strided_cuda((1, 2*(s2 // 2)), (2*(s2 // 2), 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb_4, to_2], Original ATen: [aten.cat, aten._to_copy]
        triton_poi_fused__to_copy_cat_1_xnumel = 2*(s2 // 2)
        triton_poi_fused__to_copy_cat_1.run(buf0, buf1, s2, triton_poi_fused__to_copy_cat_1_xnumel, grid=grid(triton_poi_fused__to_copy_cat_1_xnumel), stream=stream0)
        buf3 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [emb_4, to_2, sample_1], Original ATen: [aten.cat, aten._to_copy, aten.silu]
        triton_tem_fused__to_copy_cat_silu_2.run(buf1, arg9_1, arg10_1, buf3, s2, grid=torch._inductor.kernel.mm_common.mm_grid(1, 3072, meta0), stream=stream0)
        del arg10_1
        del arg9_1
        buf4 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf3, reinterpret_tensor(arg11_1, (3072, 3072), (1, 3072), 0), out=buf4)
        del arg11_1
        buf5 = buf0; del buf0  # reuse
        # Topologically Sorted Source Nodes: [emb_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_3_xnumel = 2*(s2 // 2)
        triton_poi_fused_cat_3.run(arg4_1, buf5, s2, s3, s4, triton_poi_fused_cat_3_xnumel, grid=grid(triton_poi_fused_cat_3_xnumel), stream=stream0)
        del arg4_1
        buf6 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [emb_9, to_3], Original ATen: [aten.cat, aten._to_copy]
        triton_poi_fused__to_copy_cat_1_xnumel = 2*(s2 // 2)
        triton_poi_fused__to_copy_cat_1.run(buf5, buf6, s2, triton_poi_fused__to_copy_cat_1_xnumel, grid=grid(triton_poi_fused__to_copy_cat_1_xnumel), stream=stream0)
        del buf5
        buf8 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [emb_9, to_3, sample_4], Original ATen: [aten.cat, aten._to_copy, aten.silu]
        triton_tem_fused__to_copy_cat_silu_2.run(buf6, arg13_1, arg14_1, buf8, s2, grid=torch._inductor.kernel.mm_common.mm_grid(1, 3072, meta0), stream=stream0)
        del arg13_1
        del arg14_1
        del buf6
        buf9 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf8, reinterpret_tensor(arg15_1, (3072, 3072), (1, 3072), 0), out=buf9)
        del arg15_1
        buf11 = buf8; del buf8  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_2], Original ATen: [aten.silu]
        triton_tem_fused_silu_4.run(arg8_1, arg17_1, arg18_1, buf11, grid=torch._inductor.kernel.mm_common.mm_grid(1, 3072, meta1), stream=stream0)
        del arg17_1
        del arg18_1
        del arg8_1
        buf12 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf11, reinterpret_tensor(arg19_1, (3072, 3072), (1, 3072), 0), out=buf12)
        del arg19_1
        buf13 = buf12; del buf12  # reuse
        buf14 = buf11; del buf11  # reuse
        buf20 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf67 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf77 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf120 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf130 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf173 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf183 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf226 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf236 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf279 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf289 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf332 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf342 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf385 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf395 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf438 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf448 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf491 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf501 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf544 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf554 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf597 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf607 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf650 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf660 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf703 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf713 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf756 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf766 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf809 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf819 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf862 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf872 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf915 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf925 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf968 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf978 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1021 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1051 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1077 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1103 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1129 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1155 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1181 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1207 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1233 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1259 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1285 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1311 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1337 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1363 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1389 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1415 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1441 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1467 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1493 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1519 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1545 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1571 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1597 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1623 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        buf1649 = empty_strided_cuda((1, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [time_guidance_emb, conditioning, silu_3, silu_4, silu_5, silu_6, silu_7, silu_8, silu_9, silu_10, silu_11, silu_12, silu_13, silu_14, silu_15, silu_16, silu_17, silu_18, silu_19, silu_20, silu_21, silu_22, silu_23, silu_24, silu_25, silu_26, silu_27, silu_28, silu_29, silu_30, silu_31, silu_32, silu_33, silu_34, silu_35, silu_36, silu_37, silu_38, silu_39, silu_40, silu_41, silu_42, silu_43, silu_44, silu_45, silu_46, silu_47, silu_48, silu_49, silu_50, silu_51, silu_52, silu_53, silu_54, silu_55, silu_56, silu_57, silu_58, silu_59, silu_60, silu_61, silu_62, silu_63, silu_64, silu_65], Original ATen: [aten.add, aten.silu]
        triton_poi_fused_add_silu_5.run(buf13, buf4, arg12_1, buf9, arg16_1, arg20_1, buf14, buf20, buf67, buf77, buf120, buf130, buf173, buf183, buf226, buf236, buf279, buf289, buf332, buf342, buf385, buf395, buf438, buf448, buf491, buf501, buf544, buf554, buf597, buf607, buf650, buf660, buf703, buf713, buf756, buf766, buf809, buf819, buf862, buf872, buf915, buf925, buf968, buf978, buf1021, buf1051, buf1077, buf1103, buf1129, buf1155, buf1181, buf1207, buf1233, buf1259, buf1285, buf1311, buf1337, buf1363, buf1389, buf1415, buf1441, buf1467, buf1493, buf1519, buf1545, buf1571, buf1597, buf1623, buf1649, 3072, grid=grid(3072), stream=stream0)
        del arg12_1
        del arg16_1
        del arg20_1
        del buf4
        del buf9
        buf15 = empty_strided_cuda((1, 18432), (18432, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_3], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf14, arg28_1, buf15, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg28_1
        del buf14
        buf16 = empty_strided_cuda((4096, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_7.run(arg2_1, arg0_1, buf16, grid=torch._inductor.kernel.mm_common.mm_grid(4096, 3072, meta3), stream=stream0)
        del arg0_1
        del arg2_1
        buf29 = empty_strided_cuda((1, 4096, 3072), (12582912, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm, add_2, mul_11, x], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_mul_native_layer_norm_8.run(buf16, arg1_1, buf15, arg29_1, buf29, 4096, 3072, grid=grid(4096), stream=stream0)
        buf21 = empty_strided_cuda((1, 18432), (18432, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_4], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf20, arg30_1, buf21, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg30_1
        del buf20
        buf22 = empty_strided_cuda((s6, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_9.run(arg24_1, arg21_1, buf22, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg21_1
        del arg24_1
        buf26 = empty_strided_cuda((1, s6, 3072), (3072*s6, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm_1, add_4, mul_12, x_1], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_mul_native_layer_norm_10.run(buf22, arg22_1, buf21, arg31_1, buf26, s6, 3072, grid=grid(s6), stream=stream0)
        buf27 = empty_strided_cuda((s6, 3072), (3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf26, arg38_1, buf27, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg38_1
        buf28 = empty_strided_cuda((1, s6, 24, 1), (24*s6, 24, 1, 24*s6), torch.float32)
        # Topologically Sorted Source Nodes: [pow_6, variance_2], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf27, arg39_1, buf28, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf30 = empty_strided_cuda((4096, 3072), (3072, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg32_1, (3072, 3072), (1, 3072), 0), out=buf30)
        del arg32_1
        buf31 = empty_strided_cuda((1, 4096, 24, 1), (98304, 24, 1, 98304), torch.float32)
        # Topologically Sorted Source Nodes: [pow_4, variance], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf30, arg33_1, buf31, 98304, 128, grid=grid(98304), stream=stream0)
        buf32 = empty_strided_cuda((1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [query_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf27, arg39_1, buf28, arg46_1, buf30, arg33_1, buf31, arg44_1, buf32, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg33_1
        del arg39_1
        del arg44_1
        del arg46_1
        buf33 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf26, arg40_1, buf33, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg40_1
        buf34 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [pow_7, variance_3], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf33, arg41_1, buf34, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf35 = buf30; del buf30  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf29, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg34_1, (3072, 3072), (1, 3072), 0), out=buf35)
        del arg34_1
        buf36 = buf31; del buf31  # reuse
        # Topologically Sorted Source Nodes: [pow_5, variance_1], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf35, arg35_1, buf36, 98304, 128, grid=grid(98304), stream=stream0)
        buf37 = empty_strided_cuda((1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [key_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf33, arg41_1, buf34, arg47_1, buf35, arg35_1, buf36, arg45_1, buf37, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg35_1
        del arg41_1
        del arg45_1
        del arg47_1
        buf41 = empty_strided_cuda((4096 + s6, 128), (128, 1), torch.float32)
        buf38 = reinterpret_tensor(buf41, (4096 + s6, 16), (128, 1), 0)  # alias
        buf45 = empty_strided_cuda((4096 + s6, 128), (128, 1), torch.float32)
        buf42 = reinterpret_tensor(buf45, (4096 + s6, 16), (128, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [freqs_cos, freqs_sin], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_15_xnumel = 65536 + (16*s6)
        triton_poi_fused__to_copy_15.run(arg25_1, arg26_1, buf38, buf42, s6, s8, triton_poi_fused__to_copy_15_xnumel, grid=grid(triton_poi_fused__to_copy_15_xnumel), stream=stream0)
        buf39 = reinterpret_tensor(buf41, (4096 + s6, 56), (128, 1), 16)  # alias
        buf43 = reinterpret_tensor(buf45, (4096 + s6, 56), (128, 1), 16)  # alias
        # Topologically Sorted Source Nodes: [freqs_cos_1, freqs_sin_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_16_xnumel = 229376 + (56*s6)
        triton_poi_fused__to_copy_16.run(arg25_1, arg26_1, buf39, buf43, s6, s8, triton_poi_fused__to_copy_16_xnumel, grid=grid(triton_poi_fused__to_copy_16_xnumel), stream=stream0)
        buf40 = reinterpret_tensor(buf41, (4096 + s6, 56), (128, 1), 72)  # alias
        buf44 = reinterpret_tensor(buf45, (4096 + s6, 56), (128, 1), 72)  # alias
        # Topologically Sorted Source Nodes: [freqs_cos_2, freqs_sin_2], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_17_xnumel = 229376 + (56*s6)
        triton_poi_fused__to_copy_17.run(arg25_1, arg26_1, buf40, buf44, s6, s8, triton_poi_fused__to_copy_17_xnumel, grid=grid(triton_poi_fused__to_copy_17_xnumel), stream=stream0)
        del arg25_1
        del arg26_1
        del buf38
        del buf39
        del buf40
        del buf42
        del buf43
        del buf44
        buf46 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [encoder_value], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg43_1, buf26, arg42_1, buf46, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg42_1
        del arg43_1
        buf47 = buf35; del buf35  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg37_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf29, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg36_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf47)
        del arg36_1
        del arg37_1
        buf48 = empty_strided_cuda((1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), torch.bfloat16)
        buf49 = empty_strided_cuda((1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf32, buf41, buf45, buf37, buf48, buf49, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf32
        buf50 = reinterpret_tensor(buf37, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf37  # reuse
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf46, buf47, buf50, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_2], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf51 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf48, buf49, buf50, scale=0.08838834764831843)
        buf52 = buf51[0]
        del buf51
        buf57 = buf47; del buf47  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf52, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg48_1, (3072, 3072), (1, 3072), 0), out=buf57)
        del arg48_1
        buf58 = reinterpret_tensor(buf57, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf57  # reuse
        buf69 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_12, attn_output, hidden_states_13, norm_hidden_states, add_17, mul_26, norm_hidden_states_1], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_21.run(buf58, buf16, arg1_1, buf15, arg29_1, arg49_1, buf69, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg1_1
        del arg49_1
        buf62 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf52, arg50_1, buf62, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg50_1
        buf63 = reinterpret_tensor(buf62, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf62  # reuse
        buf79 = buf26; del buf26  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output, encoder_hidden_states_3, norm_encoder_hidden_states, add_21, mul_29, norm_encoder_hidden_states_1], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_22.run(buf63, buf22, arg22_1, buf21, arg31_1, arg51_1, buf79, s6, 3072, grid=grid(s6), stream=stream0)
        del arg22_1
        del arg51_1
        buf68 = empty_strided_cuda((1, 18432), (18432, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_5], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf67, arg60_1, buf68, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg60_1
        del buf67
        buf70 = empty_strided_cuda((4096, 12288), (12288, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf69, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg52_1, (3072, 12288), (1, 3072), 0), out=buf70)
        del arg52_1
        buf71 = reinterpret_tensor(buf70, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf70  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_15, hidden_states_16], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf71, arg53_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg53_1
        buf72 = reinterpret_tensor(buf69, (4096, 3072), (3072, 1), 0); del buf69  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf71, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg54_1, (12288, 3072), (1, 12288), 0), out=buf72)
        del arg54_1
        buf73 = reinterpret_tensor(buf72, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf72  # reuse
        buf90 = reinterpret_tensor(buf16, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [ff_output, hidden_states_18, layer_norm_4, add_24, mul_31, x_2], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf73, buf58, buf15, arg29_1, arg55_1, buf68, arg61_1, buf90, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg29_1
        del arg55_1
        buf78 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [silu_6], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf77, arg62_1, buf78, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg62_1
        del buf77
        buf80 = empty_strided_cuda((s6, 12288), (12288, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf79, arg56_1, buf80, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg56_1
        buf81 = reinterpret_tensor(buf80, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_20, hidden_states_21], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf81, arg57_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg57_1
        buf82 = reinterpret_tensor(buf79, (s6, 3072), (3072, 1), 0); del buf79  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf81, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg58_1, (12288, 3072), (1, 12288), 0), out=buf82)
        del arg58_1
        buf83 = reinterpret_tensor(buf82, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf82  # reuse
        buf87 = reinterpret_tensor(buf22, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf22  # reuse
        # Topologically Sorted Source Nodes: [mul_30, encoder_hidden_states_4, layer_norm_5, add_26, mul_32, x_3], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf83, buf63, buf21, arg31_1, arg59_1, buf78, arg63_1, buf87, s6, 3072, grid=grid(s6), stream=stream0)
        del arg31_1
        del arg59_1
        buf88 = reinterpret_tensor(buf63, (s6, 3072), (3072, 1), 0); del buf63  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf87, arg70_1, buf88, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg70_1
        buf89 = buf34; del buf34  # reuse
        # Topologically Sorted Source Nodes: [pow_10, variance_6], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf88, arg71_1, buf89, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf91 = reinterpret_tensor(buf58, (4096, 3072), (3072, 1), 0); del buf58  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg64_1, (3072, 3072), (1, 3072), 0), out=buf91)
        del arg64_1
        buf92 = buf36; del buf36  # reuse
        # Topologically Sorted Source Nodes: [pow_8, variance_4], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf91, arg65_1, buf92, 98304, 128, grid=grid(98304), stream=stream0)
        buf93 = reinterpret_tensor(buf52, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [query_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf88, arg71_1, buf89, arg78_1, buf91, arg65_1, buf92, arg76_1, buf93, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg65_1
        del arg71_1
        del arg76_1
        del arg78_1
        buf94 = buf88; del buf88  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf87, arg72_1, buf94, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg72_1
        buf95 = buf89; del buf89  # reuse
        # Topologically Sorted Source Nodes: [pow_11, variance_7], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf94, arg73_1, buf95, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf96 = buf91; del buf91  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf90, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg66_1, (3072, 3072), (1, 3072), 0), out=buf96)
        del arg66_1
        buf97 = buf92; del buf92  # reuse
        # Topologically Sorted Source Nodes: [pow_9, variance_5], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf96, arg67_1, buf97, 98304, 128, grid=grid(98304), stream=stream0)
        buf98 = reinterpret_tensor(buf50, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf50  # reuse
        # Topologically Sorted Source Nodes: [key_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf94, arg73_1, buf95, arg79_1, buf96, arg67_1, buf97, arg77_1, buf98, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg67_1
        del arg73_1
        del arg77_1
        del arg79_1
        buf99 = buf94; del buf94  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_2], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg75_1, buf87, arg74_1, buf99, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg74_1
        del arg75_1
        buf100 = buf96; del buf96  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg69_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf90, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg68_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf100)
        del arg68_1
        del arg69_1
        buf101 = buf49; del buf49  # reuse
        buf102 = buf48; del buf48  # reuse
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf93, buf41, buf45, buf98, buf101, buf102, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf93
        buf103 = reinterpret_tensor(buf98, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf98  # reuse
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf99, buf100, buf103, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_6], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf104 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf101, buf102, buf103, scale=0.08838834764831843)
        buf105 = buf104[0]
        del buf104
        buf110 = buf100; del buf100  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf105, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg80_1, (3072, 3072), (1, 3072), 0), out=buf110)
        del arg80_1
        buf111 = reinterpret_tensor(buf110, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf110  # reuse
        buf122 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_31, attn_output_1, hidden_states_32, norm_hidden_states_2, add_39, mul_46, norm_hidden_states_3], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf111, buf73, buf68, arg61_1, arg81_1, buf122, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg81_1
        buf115 = buf99; del buf99  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf105, arg82_1, buf115, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg82_1
        buf116 = reinterpret_tensor(buf115, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf115  # reuse
        buf132 = buf87; del buf87  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_1, encoder_hidden_states_7, norm_encoder_hidden_states_2, add_43, mul_49, norm_encoder_hidden_states_3], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf116, buf83, buf78, arg63_1, arg83_1, buf132, s6, 3072, grid=grid(s6), stream=stream0)
        del arg83_1
        buf121 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [silu_7], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf120, arg92_1, buf121, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg92_1
        del buf120
        buf123 = reinterpret_tensor(buf71, (4096, 12288), (12288, 1), 0); del buf71  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf122, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg84_1, (3072, 12288), (1, 3072), 0), out=buf123)
        del arg84_1
        buf124 = reinterpret_tensor(buf123, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_34, hidden_states_35], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf124, arg85_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg85_1
        buf125 = reinterpret_tensor(buf122, (4096, 3072), (3072, 1), 0); del buf122  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf124, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg86_1, (12288, 3072), (1, 12288), 0), out=buf125)
        del arg86_1
        buf126 = reinterpret_tensor(buf125, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf125  # reuse
        buf143 = buf73; del buf73  # reuse
        # Topologically Sorted Source Nodes: [ff_output_1, hidden_states_37, layer_norm_8, add_46, mul_51, x_4], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf126, buf111, buf68, arg61_1, arg87_1, buf121, arg93_1, buf143, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg61_1
        del arg87_1
        buf131 = buf68; del buf68  # reuse
        # Topologically Sorted Source Nodes: [silu_8], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf130, arg94_1, buf131, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg94_1
        del buf130
        buf133 = reinterpret_tensor(buf81, (s6, 12288), (12288, 1), 0); del buf81  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf132, arg88_1, buf133, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg88_1
        buf134 = reinterpret_tensor(buf133, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_39, hidden_states_40], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf134, arg89_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg89_1
        buf135 = reinterpret_tensor(buf132, (s6, 3072), (3072, 1), 0); del buf132  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf134, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg90_1, (12288, 3072), (1, 12288), 0), out=buf135)
        del arg90_1
        buf136 = reinterpret_tensor(buf135, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf135  # reuse
        buf140 = buf83; del buf83  # reuse
        # Topologically Sorted Source Nodes: [mul_50, encoder_hidden_states_8, layer_norm_9, add_48, mul_52, x_5], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf136, buf116, buf78, arg63_1, arg91_1, buf131, arg95_1, buf140, s6, 3072, grid=grid(s6), stream=stream0)
        del arg63_1
        del arg91_1
        buf141 = reinterpret_tensor(buf116, (s6, 3072), (3072, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf140, arg102_1, buf141, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg102_1
        buf142 = buf95; del buf95  # reuse
        # Topologically Sorted Source Nodes: [pow_14, variance_10], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf141, arg103_1, buf142, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf144 = reinterpret_tensor(buf111, (4096, 3072), (3072, 1), 0); del buf111  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg96_1, (3072, 3072), (1, 3072), 0), out=buf144)
        del arg96_1
        buf145 = buf97; del buf97  # reuse
        # Topologically Sorted Source Nodes: [pow_12, variance_8], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf144, arg97_1, buf145, 98304, 128, grid=grid(98304), stream=stream0)
        buf146 = reinterpret_tensor(buf105, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf105  # reuse
        # Topologically Sorted Source Nodes: [query_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf141, arg103_1, buf142, arg110_1, buf144, arg97_1, buf145, arg108_1, buf146, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg103_1
        del arg108_1
        del arg110_1
        del arg97_1
        buf147 = buf141; del buf141  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf140, arg104_1, buf147, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg104_1
        buf148 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [pow_15, variance_11], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf147, arg105_1, buf148, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf149 = buf144; del buf144  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf143, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg98_1, (3072, 3072), (1, 3072), 0), out=buf149)
        del arg98_1
        buf150 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [pow_13, variance_9], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf149, arg99_1, buf150, 98304, 128, grid=grid(98304), stream=stream0)
        buf151 = reinterpret_tensor(buf103, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf103  # reuse
        # Topologically Sorted Source Nodes: [key_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf147, arg105_1, buf148, arg111_1, buf149, arg99_1, buf150, arg109_1, buf151, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg105_1
        del arg109_1
        del arg111_1
        del arg99_1
        buf152 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_4], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg107_1, buf140, arg106_1, buf152, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg106_1
        del arg107_1
        buf153 = buf149; del buf149  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg101_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf143, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg100_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf153)
        del arg100_1
        del arg101_1
        buf154 = buf102; del buf102  # reuse
        buf155 = buf101; del buf101  # reuse
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf146, buf41, buf45, buf151, buf154, buf155, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf146
        buf156 = reinterpret_tensor(buf151, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf151  # reuse
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf152, buf153, buf156, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_10], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf157 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf154, buf155, buf156, scale=0.08838834764831843)
        buf158 = buf157[0]
        del buf157
        buf163 = buf153; del buf153  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf158, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg112_1, (3072, 3072), (1, 3072), 0), out=buf163)
        del arg112_1
        buf164 = reinterpret_tensor(buf163, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf163  # reuse
        buf175 = buf143; del buf143  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_50, attn_output_2, hidden_states_51, norm_hidden_states_4, add_61, mul_66, norm_hidden_states_5], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf164, buf126, buf121, arg93_1, arg113_1, buf175, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg113_1
        buf168 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf158, arg114_1, buf168, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg114_1
        buf169 = reinterpret_tensor(buf168, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf168  # reuse
        buf185 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_2, encoder_hidden_states_11, norm_encoder_hidden_states_4, add_65, mul_69, norm_encoder_hidden_states_5], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf169, buf136, buf131, arg95_1, arg115_1, buf185, s6, 3072, grid=grid(s6), stream=stream0)
        del arg115_1
        buf174 = buf78; del buf78  # reuse
        # Topologically Sorted Source Nodes: [silu_9], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf173, arg124_1, buf174, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg124_1
        del buf173
        buf176 = reinterpret_tensor(buf124, (4096, 12288), (12288, 1), 0); del buf124  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf175, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg116_1, (3072, 12288), (1, 3072), 0), out=buf176)
        del arg116_1
        buf177 = reinterpret_tensor(buf176, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf176  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_53, hidden_states_54], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf177, arg117_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg117_1
        buf178 = reinterpret_tensor(buf175, (4096, 3072), (3072, 1), 0); del buf175  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf177, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg118_1, (12288, 3072), (1, 12288), 0), out=buf178)
        del arg118_1
        buf179 = reinterpret_tensor(buf178, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf178  # reuse
        buf196 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [ff_output_2, hidden_states_56, layer_norm_12, add_68, mul_71, x_6], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf179, buf164, buf121, arg93_1, arg119_1, buf174, arg125_1, buf196, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg119_1
        del arg93_1
        buf184 = buf121; del buf121  # reuse
        # Topologically Sorted Source Nodes: [silu_10], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf183, arg126_1, buf184, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg126_1
        del buf183
        buf186 = reinterpret_tensor(buf134, (s6, 12288), (12288, 1), 0); del buf134  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf185, arg120_1, buf186, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg120_1
        buf187 = reinterpret_tensor(buf186, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf186  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_58, hidden_states_59], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf187, arg121_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg121_1
        buf188 = reinterpret_tensor(buf185, (s6, 3072), (3072, 1), 0); del buf185  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf187, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg122_1, (12288, 3072), (1, 12288), 0), out=buf188)
        del arg122_1
        buf189 = reinterpret_tensor(buf188, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf188  # reuse
        buf193 = buf136; del buf136  # reuse
        # Topologically Sorted Source Nodes: [mul_70, encoder_hidden_states_12, layer_norm_13, add_70, mul_72, x_7], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf189, buf169, buf131, arg95_1, arg123_1, buf184, arg127_1, buf193, s6, 3072, grid=grid(s6), stream=stream0)
        del arg123_1
        del arg95_1
        buf194 = reinterpret_tensor(buf169, (s6, 3072), (3072, 1), 0); del buf169  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf193, arg134_1, buf194, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg134_1
        buf195 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [pow_18, variance_14], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf194, arg135_1, buf195, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf197 = reinterpret_tensor(buf164, (4096, 3072), (3072, 1), 0); del buf164  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg128_1, (3072, 3072), (1, 3072), 0), out=buf197)
        del arg128_1
        buf198 = buf150; del buf150  # reuse
        # Topologically Sorted Source Nodes: [pow_16, variance_12], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf197, arg129_1, buf198, 98304, 128, grid=grid(98304), stream=stream0)
        buf199 = reinterpret_tensor(buf158, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf158  # reuse
        # Topologically Sorted Source Nodes: [query_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf194, arg135_1, buf195, arg142_1, buf197, arg129_1, buf198, arg140_1, buf199, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg129_1
        del arg135_1
        del arg140_1
        del arg142_1
        buf200 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf193, arg136_1, buf200, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg136_1
        buf201 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [pow_19, variance_15], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf200, arg137_1, buf201, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf202 = buf197; del buf197  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf196, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg130_1, (3072, 3072), (1, 3072), 0), out=buf202)
        del arg130_1
        buf203 = buf198; del buf198  # reuse
        # Topologically Sorted Source Nodes: [pow_17, variance_13], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf202, arg131_1, buf203, 98304, 128, grid=grid(98304), stream=stream0)
        buf204 = reinterpret_tensor(buf156, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf156  # reuse
        # Topologically Sorted Source Nodes: [key_18], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf200, arg137_1, buf201, arg143_1, buf202, arg131_1, buf203, arg141_1, buf204, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg131_1
        del arg137_1
        del arg141_1
        del arg143_1
        buf205 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_6], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg139_1, buf193, arg138_1, buf205, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg138_1
        del arg139_1
        buf206 = buf202; del buf202  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg133_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf196, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg132_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf206)
        del arg132_1
        del arg133_1
        buf207 = buf155; del buf155  # reuse
        buf208 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf199, buf41, buf45, buf204, buf207, buf208, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf199
        buf209 = reinterpret_tensor(buf204, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf204  # reuse
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf205, buf206, buf209, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_14], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf210 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf207, buf208, buf209, scale=0.08838834764831843)
        buf211 = buf210[0]
        del buf210
        buf216 = buf206; del buf206  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf211, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg144_1, (3072, 3072), (1, 3072), 0), out=buf216)
        del arg144_1
        buf217 = reinterpret_tensor(buf216, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf216  # reuse
        buf228 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_69, attn_output_3, hidden_states_70, norm_hidden_states_6, add_83, mul_86, norm_hidden_states_7], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf217, buf179, buf174, arg125_1, arg145_1, buf228, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg145_1
        buf221 = buf205; del buf205  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf211, arg146_1, buf221, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg146_1
        buf222 = reinterpret_tensor(buf221, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf221  # reuse
        buf238 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_3, encoder_hidden_states_15, norm_encoder_hidden_states_6, add_87, mul_89, norm_encoder_hidden_states_7], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf222, buf189, buf184, arg127_1, arg147_1, buf238, s6, 3072, grid=grid(s6), stream=stream0)
        del arg147_1
        buf227 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [silu_11], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf226, arg156_1, buf227, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg156_1
        del buf226
        buf229 = reinterpret_tensor(buf177, (4096, 12288), (12288, 1), 0); del buf177  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf228, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg148_1, (3072, 12288), (1, 3072), 0), out=buf229)
        del arg148_1
        buf230 = reinterpret_tensor(buf229, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf229  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_72, hidden_states_73], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf230, arg149_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg149_1
        buf231 = reinterpret_tensor(buf228, (4096, 3072), (3072, 1), 0); del buf228  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf230, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg150_1, (12288, 3072), (1, 12288), 0), out=buf231)
        del arg150_1
        buf232 = reinterpret_tensor(buf231, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf231  # reuse
        buf249 = buf179; del buf179  # reuse
        # Topologically Sorted Source Nodes: [ff_output_3, hidden_states_75, layer_norm_16, add_90, mul_91, x_8], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf232, buf217, buf174, arg125_1, arg151_1, buf227, arg157_1, buf249, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg125_1
        del arg151_1
        buf237 = buf174; del buf174  # reuse
        # Topologically Sorted Source Nodes: [silu_12], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf236, arg158_1, buf237, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg158_1
        del buf236
        buf239 = reinterpret_tensor(buf187, (s6, 12288), (12288, 1), 0); del buf187  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf238, arg152_1, buf239, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg152_1
        buf240 = reinterpret_tensor(buf239, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf239  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_77, hidden_states_78], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf240, arg153_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg153_1
        buf241 = reinterpret_tensor(buf238, (s6, 3072), (3072, 1), 0); del buf238  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf240, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg154_1, (12288, 3072), (1, 12288), 0), out=buf241)
        del arg154_1
        buf242 = reinterpret_tensor(buf241, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf241  # reuse
        buf246 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [mul_90, encoder_hidden_states_16, layer_norm_17, add_92, mul_92, x_9], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf242, buf222, buf184, arg127_1, arg155_1, buf237, arg159_1, buf246, s6, 3072, grid=grid(s6), stream=stream0)
        del arg127_1
        del arg155_1
        buf247 = reinterpret_tensor(buf222, (s6, 3072), (3072, 1), 0); del buf222  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf246, arg166_1, buf247, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg166_1
        buf248 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [pow_22, variance_18], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf247, arg167_1, buf248, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf250 = reinterpret_tensor(buf217, (4096, 3072), (3072, 1), 0); del buf217  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg160_1, (3072, 3072), (1, 3072), 0), out=buf250)
        del arg160_1
        buf251 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [pow_20, variance_16], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf250, arg161_1, buf251, 98304, 128, grid=grid(98304), stream=stream0)
        buf252 = reinterpret_tensor(buf211, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf211  # reuse
        # Topologically Sorted Source Nodes: [query_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf247, arg167_1, buf248, arg174_1, buf250, arg161_1, buf251, arg172_1, buf252, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg161_1
        del arg167_1
        del arg172_1
        del arg174_1
        buf253 = buf247; del buf247  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf246, arg168_1, buf253, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg168_1
        buf254 = buf248; del buf248  # reuse
        # Topologically Sorted Source Nodes: [pow_23, variance_19], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf253, arg169_1, buf254, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf255 = buf250; del buf250  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf249, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg162_1, (3072, 3072), (1, 3072), 0), out=buf255)
        del arg162_1
        buf256 = buf251; del buf251  # reuse
        # Topologically Sorted Source Nodes: [pow_21, variance_17], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf255, arg163_1, buf256, 98304, 128, grid=grid(98304), stream=stream0)
        buf257 = reinterpret_tensor(buf209, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf209  # reuse
        # Topologically Sorted Source Nodes: [key_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf253, arg169_1, buf254, arg175_1, buf255, arg163_1, buf256, arg173_1, buf257, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg163_1
        del arg169_1
        del arg173_1
        del arg175_1
        buf258 = buf253; del buf253  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_8], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg171_1, buf246, arg170_1, buf258, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg170_1
        del arg171_1
        buf259 = buf255; del buf255  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg165_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf249, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg164_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf259)
        del arg164_1
        del arg165_1
        buf260 = buf208; del buf208  # reuse
        buf261 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf252, buf41, buf45, buf257, buf260, buf261, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf252
        buf262 = reinterpret_tensor(buf257, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf257  # reuse
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf258, buf259, buf262, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_18], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf263 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf260, buf261, buf262, scale=0.08838834764831843)
        buf264 = buf263[0]
        del buf263
        buf269 = buf259; del buf259  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf264, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg176_1, (3072, 3072), (1, 3072), 0), out=buf269)
        del arg176_1
        buf270 = reinterpret_tensor(buf269, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf269  # reuse
        buf281 = buf249; del buf249  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_88, attn_output_4, hidden_states_89, norm_hidden_states_8, add_105, mul_106, norm_hidden_states_9], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf270, buf232, buf227, arg157_1, arg177_1, buf281, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg177_1
        buf274 = buf258; del buf258  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf264, arg178_1, buf274, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg178_1
        buf275 = reinterpret_tensor(buf274, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf274  # reuse
        buf291 = buf246; del buf246  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_4, encoder_hidden_states_19, norm_encoder_hidden_states_8, add_109, mul_109, norm_encoder_hidden_states_9], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf275, buf242, buf237, arg159_1, arg179_1, buf291, s6, 3072, grid=grid(s6), stream=stream0)
        del arg179_1
        buf280 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [silu_13], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf279, arg188_1, buf280, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg188_1
        del buf279
        buf282 = reinterpret_tensor(buf230, (4096, 12288), (12288, 1), 0); del buf230  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf281, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg180_1, (3072, 12288), (1, 3072), 0), out=buf282)
        del arg180_1
        buf283 = reinterpret_tensor(buf282, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf282  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_91, hidden_states_92], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf283, arg181_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg181_1
        buf284 = reinterpret_tensor(buf281, (4096, 3072), (3072, 1), 0); del buf281  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf283, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg182_1, (12288, 3072), (1, 12288), 0), out=buf284)
        del arg182_1
        buf285 = reinterpret_tensor(buf284, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf284  # reuse
        buf302 = buf232; del buf232  # reuse
        # Topologically Sorted Source Nodes: [ff_output_4, hidden_states_94, layer_norm_20, add_112, mul_111, x_10], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf285, buf270, buf227, arg157_1, arg183_1, buf280, arg189_1, buf302, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg157_1
        del arg183_1
        buf290 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [silu_14], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf289, arg190_1, buf290, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg190_1
        del buf289
        buf292 = reinterpret_tensor(buf240, (s6, 12288), (12288, 1), 0); del buf240  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf291, arg184_1, buf292, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg184_1
        buf293 = reinterpret_tensor(buf292, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf292  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_96, hidden_states_97], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf293, arg185_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg185_1
        buf294 = reinterpret_tensor(buf291, (s6, 3072), (3072, 1), 0); del buf291  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf293, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg186_1, (12288, 3072), (1, 12288), 0), out=buf294)
        del arg186_1
        buf295 = reinterpret_tensor(buf294, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf294  # reuse
        buf299 = buf242; del buf242  # reuse
        # Topologically Sorted Source Nodes: [mul_110, encoder_hidden_states_20, layer_norm_21, add_114, mul_112, x_11], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf295, buf275, buf237, arg159_1, arg187_1, buf290, arg191_1, buf299, s6, 3072, grid=grid(s6), stream=stream0)
        del arg159_1
        del arg187_1
        buf300 = reinterpret_tensor(buf275, (s6, 3072), (3072, 1), 0); del buf275  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf299, arg198_1, buf300, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg198_1
        buf301 = buf254; del buf254  # reuse
        # Topologically Sorted Source Nodes: [pow_26, variance_22], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf300, arg199_1, buf301, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf303 = reinterpret_tensor(buf270, (4096, 3072), (3072, 1), 0); del buf270  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg192_1, (3072, 3072), (1, 3072), 0), out=buf303)
        del arg192_1
        buf304 = buf256; del buf256  # reuse
        # Topologically Sorted Source Nodes: [pow_24, variance_20], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf303, arg193_1, buf304, 98304, 128, grid=grid(98304), stream=stream0)
        buf305 = reinterpret_tensor(buf264, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf264  # reuse
        # Topologically Sorted Source Nodes: [query_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf300, arg199_1, buf301, arg206_1, buf303, arg193_1, buf304, arg204_1, buf305, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg193_1
        del arg199_1
        del arg204_1
        del arg206_1
        buf306 = buf300; del buf300  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf299, arg200_1, buf306, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg200_1
        buf307 = buf301; del buf301  # reuse
        # Topologically Sorted Source Nodes: [pow_27, variance_23], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf306, arg201_1, buf307, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf308 = buf303; del buf303  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf302, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg194_1, (3072, 3072), (1, 3072), 0), out=buf308)
        del arg194_1
        buf309 = buf304; del buf304  # reuse
        # Topologically Sorted Source Nodes: [pow_25, variance_21], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf308, arg195_1, buf309, 98304, 128, grid=grid(98304), stream=stream0)
        buf310 = reinterpret_tensor(buf262, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf262  # reuse
        # Topologically Sorted Source Nodes: [key_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf306, arg201_1, buf307, arg207_1, buf308, arg195_1, buf309, arg205_1, buf310, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg195_1
        del arg201_1
        del arg205_1
        del arg207_1
        buf311 = buf306; del buf306  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_10], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg203_1, buf299, arg202_1, buf311, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg202_1
        del arg203_1
        buf312 = buf308; del buf308  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg197_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf302, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg196_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf312)
        del arg196_1
        del arg197_1
        buf313 = buf261; del buf261  # reuse
        buf314 = buf260; del buf260  # reuse
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf305, buf41, buf45, buf310, buf313, buf314, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf305
        buf315 = reinterpret_tensor(buf310, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf310  # reuse
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf311, buf312, buf315, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_22], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf316 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf313, buf314, buf315, scale=0.08838834764831843)
        buf317 = buf316[0]
        del buf316
        buf322 = buf312; del buf312  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf317, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg208_1, (3072, 3072), (1, 3072), 0), out=buf322)
        del arg208_1
        buf323 = reinterpret_tensor(buf322, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf322  # reuse
        buf334 = buf302; del buf302  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_107, attn_output_5, hidden_states_108, norm_hidden_states_10, add_127, mul_126, norm_hidden_states_11], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf323, buf285, buf280, arg189_1, arg209_1, buf334, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg209_1
        buf327 = buf311; del buf311  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf317, arg210_1, buf327, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg210_1
        buf328 = reinterpret_tensor(buf327, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf327  # reuse
        buf344 = buf299; del buf299  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_5, encoder_hidden_states_23, norm_encoder_hidden_states_10, add_131, mul_129, norm_encoder_hidden_states_11], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf328, buf295, buf290, arg191_1, arg211_1, buf344, s6, 3072, grid=grid(s6), stream=stream0)
        del arg211_1
        buf333 = buf237; del buf237  # reuse
        # Topologically Sorted Source Nodes: [silu_15], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf332, arg220_1, buf333, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg220_1
        del buf332
        buf335 = reinterpret_tensor(buf283, (4096, 12288), (12288, 1), 0); del buf283  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf334, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg212_1, (3072, 12288), (1, 3072), 0), out=buf335)
        del arg212_1
        buf336 = reinterpret_tensor(buf335, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf335  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_110, hidden_states_111], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf336, arg213_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg213_1
        buf337 = reinterpret_tensor(buf334, (4096, 3072), (3072, 1), 0); del buf334  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf336, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg214_1, (12288, 3072), (1, 12288), 0), out=buf337)
        del arg214_1
        buf338 = reinterpret_tensor(buf337, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf337  # reuse
        buf355 = buf285; del buf285  # reuse
        # Topologically Sorted Source Nodes: [ff_output_5, hidden_states_113, layer_norm_24, add_134, mul_131, x_12], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf338, buf323, buf280, arg189_1, arg215_1, buf333, arg221_1, buf355, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg189_1
        del arg215_1
        buf343 = buf280; del buf280  # reuse
        # Topologically Sorted Source Nodes: [silu_16], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf342, arg222_1, buf343, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg222_1
        del buf342
        buf345 = reinterpret_tensor(buf293, (s6, 12288), (12288, 1), 0); del buf293  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf344, arg216_1, buf345, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg216_1
        buf346 = reinterpret_tensor(buf345, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf345  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_115, hidden_states_116], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf346, arg217_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg217_1
        buf347 = reinterpret_tensor(buf344, (s6, 3072), (3072, 1), 0); del buf344  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf346, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg218_1, (12288, 3072), (1, 12288), 0), out=buf347)
        del arg218_1
        buf348 = reinterpret_tensor(buf347, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf347  # reuse
        buf352 = buf295; del buf295  # reuse
        # Topologically Sorted Source Nodes: [mul_130, encoder_hidden_states_24, layer_norm_25, add_136, mul_132, x_13], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf348, buf328, buf290, arg191_1, arg219_1, buf343, arg223_1, buf352, s6, 3072, grid=grid(s6), stream=stream0)
        del arg191_1
        del arg219_1
        buf353 = reinterpret_tensor(buf328, (s6, 3072), (3072, 1), 0); del buf328  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf352, arg230_1, buf353, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg230_1
        buf354 = buf307; del buf307  # reuse
        # Topologically Sorted Source Nodes: [pow_30, variance_26], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf353, arg231_1, buf354, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf356 = reinterpret_tensor(buf323, (4096, 3072), (3072, 1), 0); del buf323  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg224_1, (3072, 3072), (1, 3072), 0), out=buf356)
        del arg224_1
        buf357 = buf309; del buf309  # reuse
        # Topologically Sorted Source Nodes: [pow_28, variance_24], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf356, arg225_1, buf357, 98304, 128, grid=grid(98304), stream=stream0)
        buf358 = reinterpret_tensor(buf317, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf317  # reuse
        # Topologically Sorted Source Nodes: [query_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf353, arg231_1, buf354, arg238_1, buf356, arg225_1, buf357, arg236_1, buf358, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg225_1
        del arg231_1
        del arg236_1
        del arg238_1
        buf359 = buf353; del buf353  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf352, arg232_1, buf359, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg232_1
        buf360 = buf354; del buf354  # reuse
        # Topologically Sorted Source Nodes: [pow_31, variance_27], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf359, arg233_1, buf360, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf361 = buf356; del buf356  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf355, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg226_1, (3072, 3072), (1, 3072), 0), out=buf361)
        del arg226_1
        buf362 = buf357; del buf357  # reuse
        # Topologically Sorted Source Nodes: [pow_29, variance_25], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf361, arg227_1, buf362, 98304, 128, grid=grid(98304), stream=stream0)
        buf363 = reinterpret_tensor(buf315, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf315  # reuse
        # Topologically Sorted Source Nodes: [key_33], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf359, arg233_1, buf360, arg239_1, buf361, arg227_1, buf362, arg237_1, buf363, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg227_1
        del arg233_1
        del arg237_1
        del arg239_1
        buf364 = buf359; del buf359  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_12], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg235_1, buf352, arg234_1, buf364, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg234_1
        del arg235_1
        buf365 = buf361; del buf361  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg229_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf355, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg228_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf365)
        del arg228_1
        del arg229_1
        buf366 = buf314; del buf314  # reuse
        buf367 = buf313; del buf313  # reuse
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf358, buf41, buf45, buf363, buf366, buf367, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf358
        buf368 = reinterpret_tensor(buf363, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf363  # reuse
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf364, buf365, buf368, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_26], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf369 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf366, buf367, buf368, scale=0.08838834764831843)
        buf370 = buf369[0]
        del buf369
        buf375 = buf365; del buf365  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf370, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg240_1, (3072, 3072), (1, 3072), 0), out=buf375)
        del arg240_1
        buf376 = reinterpret_tensor(buf375, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf375  # reuse
        buf387 = buf355; del buf355  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_126, attn_output_6, hidden_states_127, norm_hidden_states_12, add_149, mul_146, norm_hidden_states_13], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf376, buf338, buf333, arg221_1, arg241_1, buf387, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg241_1
        buf380 = buf364; del buf364  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf370, arg242_1, buf380, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg242_1
        buf381 = reinterpret_tensor(buf380, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf380  # reuse
        buf397 = buf352; del buf352  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_6, encoder_hidden_states_27, norm_encoder_hidden_states_12, add_153, mul_149, norm_encoder_hidden_states_13], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf381, buf348, buf343, arg223_1, arg243_1, buf397, s6, 3072, grid=grid(s6), stream=stream0)
        del arg243_1
        buf386 = buf290; del buf290  # reuse
        # Topologically Sorted Source Nodes: [silu_17], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf385, arg252_1, buf386, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg252_1
        del buf385
        buf388 = reinterpret_tensor(buf336, (4096, 12288), (12288, 1), 0); del buf336  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf387, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg244_1, (3072, 12288), (1, 3072), 0), out=buf388)
        del arg244_1
        buf389 = reinterpret_tensor(buf388, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf388  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_129, hidden_states_130], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf389, arg245_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg245_1
        buf390 = reinterpret_tensor(buf387, (4096, 3072), (3072, 1), 0); del buf387  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf389, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg246_1, (12288, 3072), (1, 12288), 0), out=buf390)
        del arg246_1
        buf391 = reinterpret_tensor(buf390, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf390  # reuse
        buf408 = buf338; del buf338  # reuse
        # Topologically Sorted Source Nodes: [ff_output_6, hidden_states_132, layer_norm_28, add_156, mul_151, x_14], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf391, buf376, buf333, arg221_1, arg247_1, buf386, arg253_1, buf408, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg221_1
        del arg247_1
        buf396 = buf333; del buf333  # reuse
        # Topologically Sorted Source Nodes: [silu_18], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf395, arg254_1, buf396, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg254_1
        del buf395
        buf398 = reinterpret_tensor(buf346, (s6, 12288), (12288, 1), 0); del buf346  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf397, arg248_1, buf398, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg248_1
        buf399 = reinterpret_tensor(buf398, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf398  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_134, hidden_states_135], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf399, arg249_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg249_1
        buf400 = reinterpret_tensor(buf397, (s6, 3072), (3072, 1), 0); del buf397  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf399, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg250_1, (12288, 3072), (1, 12288), 0), out=buf400)
        del arg250_1
        buf401 = reinterpret_tensor(buf400, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf400  # reuse
        buf405 = buf348; del buf348  # reuse
        # Topologically Sorted Source Nodes: [mul_150, encoder_hidden_states_28, layer_norm_29, add_158, mul_152, x_15], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf401, buf381, buf343, arg223_1, arg251_1, buf396, arg255_1, buf405, s6, 3072, grid=grid(s6), stream=stream0)
        del arg223_1
        del arg251_1
        buf406 = reinterpret_tensor(buf381, (s6, 3072), (3072, 1), 0); del buf381  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf405, arg262_1, buf406, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg262_1
        buf407 = buf360; del buf360  # reuse
        # Topologically Sorted Source Nodes: [pow_34, variance_30], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf406, arg263_1, buf407, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf409 = reinterpret_tensor(buf376, (4096, 3072), (3072, 1), 0); del buf376  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg256_1, (3072, 3072), (1, 3072), 0), out=buf409)
        del arg256_1
        buf410 = buf362; del buf362  # reuse
        # Topologically Sorted Source Nodes: [pow_32, variance_28], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf409, arg257_1, buf410, 98304, 128, grid=grid(98304), stream=stream0)
        buf411 = reinterpret_tensor(buf370, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf370  # reuse
        # Topologically Sorted Source Nodes: [query_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf406, arg263_1, buf407, arg270_1, buf409, arg257_1, buf410, arg268_1, buf411, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg257_1
        del arg263_1
        del arg268_1
        del arg270_1
        buf412 = buf406; del buf406  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf405, arg264_1, buf412, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg264_1
        buf413 = buf407; del buf407  # reuse
        # Topologically Sorted Source Nodes: [pow_35, variance_31], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf412, arg265_1, buf413, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf414 = buf409; del buf409  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf408, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg258_1, (3072, 3072), (1, 3072), 0), out=buf414)
        del arg258_1
        buf415 = buf410; del buf410  # reuse
        # Topologically Sorted Source Nodes: [pow_33, variance_29], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf414, arg259_1, buf415, 98304, 128, grid=grid(98304), stream=stream0)
        buf416 = reinterpret_tensor(buf368, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf368  # reuse
        # Topologically Sorted Source Nodes: [key_38], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf412, arg265_1, buf413, arg271_1, buf414, arg259_1, buf415, arg269_1, buf416, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg259_1
        del arg265_1
        del arg269_1
        del arg271_1
        buf417 = buf412; del buf412  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_14], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg267_1, buf405, arg266_1, buf417, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg266_1
        del arg267_1
        buf418 = buf414; del buf414  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg261_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf408, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg260_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf418)
        del arg260_1
        del arg261_1
        buf419 = buf367; del buf367  # reuse
        buf420 = buf366; del buf366  # reuse
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf411, buf41, buf45, buf416, buf419, buf420, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf411
        buf421 = reinterpret_tensor(buf416, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf416  # reuse
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf417, buf418, buf421, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_30], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf422 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf419, buf420, buf421, scale=0.08838834764831843)
        buf423 = buf422[0]
        del buf422
        buf428 = buf418; del buf418  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg272_1, (3072, 3072), (1, 3072), 0), out=buf428)
        del arg272_1
        buf429 = reinterpret_tensor(buf428, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf428  # reuse
        buf440 = buf408; del buf408  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_145, attn_output_7, hidden_states_146, norm_hidden_states_14, add_171, mul_166, norm_hidden_states_15], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf429, buf391, buf386, arg253_1, arg273_1, buf440, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg273_1
        buf433 = buf417; del buf417  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf423, arg274_1, buf433, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg274_1
        buf434 = reinterpret_tensor(buf433, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf433  # reuse
        buf450 = buf405; del buf405  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_7, encoder_hidden_states_31, norm_encoder_hidden_states_14, add_175, mul_169, norm_encoder_hidden_states_15], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf434, buf401, buf396, arg255_1, arg275_1, buf450, s6, 3072, grid=grid(s6), stream=stream0)
        del arg275_1
        buf439 = buf343; del buf343  # reuse
        # Topologically Sorted Source Nodes: [silu_19], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf438, arg284_1, buf439, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg284_1
        del buf438
        buf441 = reinterpret_tensor(buf389, (4096, 12288), (12288, 1), 0); del buf389  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf440, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg276_1, (3072, 12288), (1, 3072), 0), out=buf441)
        del arg276_1
        buf442 = reinterpret_tensor(buf441, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf441  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_148, hidden_states_149], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf442, arg277_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg277_1
        buf443 = reinterpret_tensor(buf440, (4096, 3072), (3072, 1), 0); del buf440  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf442, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg278_1, (12288, 3072), (1, 12288), 0), out=buf443)
        del arg278_1
        buf444 = reinterpret_tensor(buf443, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf443  # reuse
        buf461 = buf391; del buf391  # reuse
        # Topologically Sorted Source Nodes: [ff_output_7, hidden_states_151, layer_norm_32, add_178, mul_171, x_16], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf444, buf429, buf386, arg253_1, arg279_1, buf439, arg285_1, buf461, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg253_1
        del arg279_1
        buf449 = buf386; del buf386  # reuse
        # Topologically Sorted Source Nodes: [silu_20], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf448, arg286_1, buf449, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg286_1
        del buf448
        buf451 = reinterpret_tensor(buf399, (s6, 12288), (12288, 1), 0); del buf399  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf450, arg280_1, buf451, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg280_1
        buf452 = reinterpret_tensor(buf451, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf451  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_153, hidden_states_154], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf452, arg281_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg281_1
        buf453 = reinterpret_tensor(buf450, (s6, 3072), (3072, 1), 0); del buf450  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf452, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg282_1, (12288, 3072), (1, 12288), 0), out=buf453)
        del arg282_1
        buf454 = reinterpret_tensor(buf453, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf453  # reuse
        buf458 = buf401; del buf401  # reuse
        # Topologically Sorted Source Nodes: [mul_170, encoder_hidden_states_32, layer_norm_33, add_180, mul_172, x_17], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf454, buf434, buf396, arg255_1, arg283_1, buf449, arg287_1, buf458, s6, 3072, grid=grid(s6), stream=stream0)
        del arg255_1
        del arg283_1
        buf459 = reinterpret_tensor(buf434, (s6, 3072), (3072, 1), 0); del buf434  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf458, arg294_1, buf459, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg294_1
        buf460 = buf413; del buf413  # reuse
        # Topologically Sorted Source Nodes: [pow_38, variance_34], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf459, arg295_1, buf460, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf462 = reinterpret_tensor(buf429, (4096, 3072), (3072, 1), 0); del buf429  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf461, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg288_1, (3072, 3072), (1, 3072), 0), out=buf462)
        del arg288_1
        buf463 = buf415; del buf415  # reuse
        # Topologically Sorted Source Nodes: [pow_36, variance_32], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf462, arg289_1, buf463, 98304, 128, grid=grid(98304), stream=stream0)
        buf464 = reinterpret_tensor(buf423, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf423  # reuse
        # Topologically Sorted Source Nodes: [query_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf459, arg295_1, buf460, arg302_1, buf462, arg289_1, buf463, arg300_1, buf464, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg289_1
        del arg295_1
        del arg300_1
        del arg302_1
        buf465 = buf459; del buf459  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf458, arg296_1, buf465, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg296_1
        buf466 = buf460; del buf460  # reuse
        # Topologically Sorted Source Nodes: [pow_39, variance_35], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf465, arg297_1, buf466, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf467 = buf462; del buf462  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf461, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg290_1, (3072, 3072), (1, 3072), 0), out=buf467)
        del arg290_1
        buf468 = buf463; del buf463  # reuse
        # Topologically Sorted Source Nodes: [pow_37, variance_33], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf467, arg291_1, buf468, 98304, 128, grid=grid(98304), stream=stream0)
        buf469 = reinterpret_tensor(buf421, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf421  # reuse
        # Topologically Sorted Source Nodes: [key_43], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf465, arg297_1, buf466, arg303_1, buf467, arg291_1, buf468, arg301_1, buf469, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg291_1
        del arg297_1
        del arg301_1
        del arg303_1
        buf470 = buf465; del buf465  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_16], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg299_1, buf458, arg298_1, buf470, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg298_1
        del arg299_1
        buf471 = buf467; del buf467  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg293_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf461, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg292_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf471)
        del arg292_1
        del arg293_1
        buf472 = buf420; del buf420  # reuse
        buf473 = buf419; del buf419  # reuse
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf464, buf41, buf45, buf469, buf472, buf473, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf464
        buf474 = reinterpret_tensor(buf469, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf469  # reuse
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf470, buf471, buf474, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_34], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf475 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf472, buf473, buf474, scale=0.08838834764831843)
        buf476 = buf475[0]
        del buf475
        buf481 = buf471; del buf471  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf476, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg304_1, (3072, 3072), (1, 3072), 0), out=buf481)
        del arg304_1
        buf482 = reinterpret_tensor(buf481, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf481  # reuse
        buf493 = buf461; del buf461  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_164, attn_output_8, hidden_states_165, norm_hidden_states_16, add_193, mul_186, norm_hidden_states_17], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf482, buf444, buf439, arg285_1, arg305_1, buf493, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg305_1
        buf486 = buf470; del buf470  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf476, arg306_1, buf486, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg306_1
        buf487 = reinterpret_tensor(buf486, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf486  # reuse
        buf503 = buf458; del buf458  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_8, encoder_hidden_states_35, norm_encoder_hidden_states_16, add_197, mul_189, norm_encoder_hidden_states_17], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf487, buf454, buf449, arg287_1, arg307_1, buf503, s6, 3072, grid=grid(s6), stream=stream0)
        del arg307_1
        buf492 = buf396; del buf396  # reuse
        # Topologically Sorted Source Nodes: [silu_21], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf491, arg316_1, buf492, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg316_1
        del buf491
        buf494 = reinterpret_tensor(buf442, (4096, 12288), (12288, 1), 0); del buf442  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf493, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg308_1, (3072, 12288), (1, 3072), 0), out=buf494)
        del arg308_1
        buf495 = reinterpret_tensor(buf494, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf494  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_167, hidden_states_168], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf495, arg309_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg309_1
        buf496 = reinterpret_tensor(buf493, (4096, 3072), (3072, 1), 0); del buf493  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf495, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg310_1, (12288, 3072), (1, 12288), 0), out=buf496)
        del arg310_1
        buf497 = reinterpret_tensor(buf496, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf496  # reuse
        buf514 = buf444; del buf444  # reuse
        # Topologically Sorted Source Nodes: [ff_output_8, hidden_states_170, layer_norm_36, add_200, mul_191, x_18], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf497, buf482, buf439, arg285_1, arg311_1, buf492, arg317_1, buf514, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg285_1
        del arg311_1
        buf502 = buf439; del buf439  # reuse
        # Topologically Sorted Source Nodes: [silu_22], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf501, arg318_1, buf502, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg318_1
        del buf501
        buf504 = reinterpret_tensor(buf452, (s6, 12288), (12288, 1), 0); del buf452  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf503, arg312_1, buf504, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg312_1
        buf505 = reinterpret_tensor(buf504, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf504  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_172, hidden_states_173], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf505, arg313_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg313_1
        buf506 = reinterpret_tensor(buf503, (s6, 3072), (3072, 1), 0); del buf503  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf505, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg314_1, (12288, 3072), (1, 12288), 0), out=buf506)
        del arg314_1
        buf507 = reinterpret_tensor(buf506, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf506  # reuse
        buf511 = buf454; del buf454  # reuse
        # Topologically Sorted Source Nodes: [mul_190, encoder_hidden_states_36, layer_norm_37, add_202, mul_192, x_19], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf507, buf487, buf449, arg287_1, arg315_1, buf502, arg319_1, buf511, s6, 3072, grid=grid(s6), stream=stream0)
        del arg287_1
        del arg315_1
        buf512 = reinterpret_tensor(buf487, (s6, 3072), (3072, 1), 0); del buf487  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf511, arg326_1, buf512, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg326_1
        buf513 = buf466; del buf466  # reuse
        # Topologically Sorted Source Nodes: [pow_42, variance_38], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf512, arg327_1, buf513, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf515 = reinterpret_tensor(buf482, (4096, 3072), (3072, 1), 0); del buf482  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg320_1, (3072, 3072), (1, 3072), 0), out=buf515)
        del arg320_1
        buf516 = buf468; del buf468  # reuse
        # Topologically Sorted Source Nodes: [pow_40, variance_36], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf515, arg321_1, buf516, 98304, 128, grid=grid(98304), stream=stream0)
        buf517 = reinterpret_tensor(buf476, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf476  # reuse
        # Topologically Sorted Source Nodes: [query_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf512, arg327_1, buf513, arg334_1, buf515, arg321_1, buf516, arg332_1, buf517, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg321_1
        del arg327_1
        del arg332_1
        del arg334_1
        buf518 = buf512; del buf512  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf511, arg328_1, buf518, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg328_1
        buf519 = buf513; del buf513  # reuse
        # Topologically Sorted Source Nodes: [pow_43, variance_39], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf518, arg329_1, buf519, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf520 = buf515; del buf515  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf514, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg322_1, (3072, 3072), (1, 3072), 0), out=buf520)
        del arg322_1
        buf521 = buf516; del buf516  # reuse
        # Topologically Sorted Source Nodes: [pow_41, variance_37], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf520, arg323_1, buf521, 98304, 128, grid=grid(98304), stream=stream0)
        buf522 = reinterpret_tensor(buf474, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf474  # reuse
        # Topologically Sorted Source Nodes: [key_48], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf518, arg329_1, buf519, arg335_1, buf520, arg323_1, buf521, arg333_1, buf522, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg323_1
        del arg329_1
        del arg333_1
        del arg335_1
        buf523 = buf518; del buf518  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_18], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg331_1, buf511, arg330_1, buf523, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg330_1
        del arg331_1
        buf524 = buf520; del buf520  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg325_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf514, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg324_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf524)
        del arg324_1
        del arg325_1
        buf525 = buf473; del buf473  # reuse
        buf526 = buf472; del buf472  # reuse
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf517, buf41, buf45, buf522, buf525, buf526, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf517
        buf527 = reinterpret_tensor(buf522, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf522  # reuse
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf523, buf524, buf527, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_38], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf528 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf525, buf526, buf527, scale=0.08838834764831843)
        buf529 = buf528[0]
        del buf528
        buf534 = buf524; del buf524  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf529, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg336_1, (3072, 3072), (1, 3072), 0), out=buf534)
        del arg336_1
        buf535 = reinterpret_tensor(buf534, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf534  # reuse
        buf546 = buf514; del buf514  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_183, attn_output_9, hidden_states_184, norm_hidden_states_18, add_215, mul_206, norm_hidden_states_19], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf535, buf497, buf492, arg317_1, arg337_1, buf546, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg337_1
        buf539 = buf523; del buf523  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf529, arg338_1, buf539, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg338_1
        buf540 = reinterpret_tensor(buf539, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf539  # reuse
        buf556 = buf511; del buf511  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_9, encoder_hidden_states_39, norm_encoder_hidden_states_18, add_219, mul_209, norm_encoder_hidden_states_19], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf540, buf507, buf502, arg319_1, arg339_1, buf556, s6, 3072, grid=grid(s6), stream=stream0)
        del arg339_1
        buf545 = buf449; del buf449  # reuse
        # Topologically Sorted Source Nodes: [silu_23], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf544, arg348_1, buf545, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg348_1
        del buf544
        buf547 = reinterpret_tensor(buf495, (4096, 12288), (12288, 1), 0); del buf495  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf546, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg340_1, (3072, 12288), (1, 3072), 0), out=buf547)
        del arg340_1
        buf548 = reinterpret_tensor(buf547, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf547  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_186, hidden_states_187], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf548, arg341_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg341_1
        buf549 = reinterpret_tensor(buf546, (4096, 3072), (3072, 1), 0); del buf546  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf548, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg342_1, (12288, 3072), (1, 12288), 0), out=buf549)
        del arg342_1
        buf550 = reinterpret_tensor(buf549, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf549  # reuse
        buf567 = buf497; del buf497  # reuse
        # Topologically Sorted Source Nodes: [ff_output_9, hidden_states_189, layer_norm_40, add_222, mul_211, x_20], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf550, buf535, buf492, arg317_1, arg343_1, buf545, arg349_1, buf567, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg317_1
        del arg343_1
        buf555 = buf492; del buf492  # reuse
        # Topologically Sorted Source Nodes: [silu_24], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf554, arg350_1, buf555, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg350_1
        del buf554
        buf557 = reinterpret_tensor(buf505, (s6, 12288), (12288, 1), 0); del buf505  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf556, arg344_1, buf557, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg344_1
        buf558 = reinterpret_tensor(buf557, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf557  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_191, hidden_states_192], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf558, arg345_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg345_1
        buf559 = reinterpret_tensor(buf556, (s6, 3072), (3072, 1), 0); del buf556  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf558, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg346_1, (12288, 3072), (1, 12288), 0), out=buf559)
        del arg346_1
        buf560 = reinterpret_tensor(buf559, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf559  # reuse
        buf564 = buf507; del buf507  # reuse
        # Topologically Sorted Source Nodes: [mul_210, encoder_hidden_states_40, layer_norm_41, add_224, mul_212, x_21], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf560, buf540, buf502, arg319_1, arg347_1, buf555, arg351_1, buf564, s6, 3072, grid=grid(s6), stream=stream0)
        del arg319_1
        del arg347_1
        buf565 = reinterpret_tensor(buf540, (s6, 3072), (3072, 1), 0); del buf540  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf564, arg358_1, buf565, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg358_1
        buf566 = buf519; del buf519  # reuse
        # Topologically Sorted Source Nodes: [pow_46, variance_42], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf565, arg359_1, buf566, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf568 = reinterpret_tensor(buf535, (4096, 3072), (3072, 1), 0); del buf535  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg352_1, (3072, 3072), (1, 3072), 0), out=buf568)
        del arg352_1
        buf569 = buf521; del buf521  # reuse
        # Topologically Sorted Source Nodes: [pow_44, variance_40], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf568, arg353_1, buf569, 98304, 128, grid=grid(98304), stream=stream0)
        buf570 = reinterpret_tensor(buf529, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf529  # reuse
        # Topologically Sorted Source Nodes: [query_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf565, arg359_1, buf566, arg366_1, buf568, arg353_1, buf569, arg364_1, buf570, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg353_1
        del arg359_1
        del arg364_1
        del arg366_1
        buf571 = buf565; del buf565  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf564, arg360_1, buf571, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg360_1
        buf572 = buf566; del buf566  # reuse
        # Topologically Sorted Source Nodes: [pow_47, variance_43], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf571, arg361_1, buf572, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf573 = buf568; del buf568  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf567, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg354_1, (3072, 3072), (1, 3072), 0), out=buf573)
        del arg354_1
        buf574 = buf569; del buf569  # reuse
        # Topologically Sorted Source Nodes: [pow_45, variance_41], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf573, arg355_1, buf574, 98304, 128, grid=grid(98304), stream=stream0)
        buf575 = reinterpret_tensor(buf527, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf527  # reuse
        # Topologically Sorted Source Nodes: [key_53], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf571, arg361_1, buf572, arg367_1, buf573, arg355_1, buf574, arg365_1, buf575, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg355_1
        del arg361_1
        del arg365_1
        del arg367_1
        buf576 = buf571; del buf571  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_20], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg363_1, buf564, arg362_1, buf576, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg362_1
        del arg363_1
        buf577 = buf573; del buf573  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg357_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf567, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg356_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf577)
        del arg356_1
        del arg357_1
        buf578 = buf526; del buf526  # reuse
        buf579 = buf525; del buf525  # reuse
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf570, buf41, buf45, buf575, buf578, buf579, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf570
        buf580 = reinterpret_tensor(buf575, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf575  # reuse
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf576, buf577, buf580, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_42], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf581 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf578, buf579, buf580, scale=0.08838834764831843)
        buf582 = buf581[0]
        del buf581
        buf587 = buf577; del buf577  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf582, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg368_1, (3072, 3072), (1, 3072), 0), out=buf587)
        del arg368_1
        buf588 = reinterpret_tensor(buf587, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf587  # reuse
        buf599 = buf567; del buf567  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_202, attn_output_10, hidden_states_203, norm_hidden_states_20, add_237, mul_226, norm_hidden_states_21], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf588, buf550, buf545, arg349_1, arg369_1, buf599, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg369_1
        buf592 = buf576; del buf576  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf582, arg370_1, buf592, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg370_1
        buf593 = reinterpret_tensor(buf592, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf592  # reuse
        buf609 = buf564; del buf564  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_10, encoder_hidden_states_43, norm_encoder_hidden_states_20, add_241, mul_229, norm_encoder_hidden_states_21], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf593, buf560, buf555, arg351_1, arg371_1, buf609, s6, 3072, grid=grid(s6), stream=stream0)
        del arg371_1
        buf598 = buf502; del buf502  # reuse
        # Topologically Sorted Source Nodes: [silu_25], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf597, arg380_1, buf598, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg380_1
        del buf597
        buf600 = reinterpret_tensor(buf548, (4096, 12288), (12288, 1), 0); del buf548  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf599, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg372_1, (3072, 12288), (1, 3072), 0), out=buf600)
        del arg372_1
        buf601 = reinterpret_tensor(buf600, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf600  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_205, hidden_states_206], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf601, arg373_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg373_1
        buf602 = reinterpret_tensor(buf599, (4096, 3072), (3072, 1), 0); del buf599  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf601, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg374_1, (12288, 3072), (1, 12288), 0), out=buf602)
        del arg374_1
        buf603 = reinterpret_tensor(buf602, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf602  # reuse
        buf620 = buf550; del buf550  # reuse
        # Topologically Sorted Source Nodes: [ff_output_10, hidden_states_208, layer_norm_44, add_244, mul_231, x_22], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf603, buf588, buf545, arg349_1, arg375_1, buf598, arg381_1, buf620, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg349_1
        del arg375_1
        buf608 = buf545; del buf545  # reuse
        # Topologically Sorted Source Nodes: [silu_26], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf607, arg382_1, buf608, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg382_1
        del buf607
        buf610 = reinterpret_tensor(buf558, (s6, 12288), (12288, 1), 0); del buf558  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf609, arg376_1, buf610, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg376_1
        buf611 = reinterpret_tensor(buf610, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf610  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_210, hidden_states_211], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf611, arg377_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg377_1
        buf612 = reinterpret_tensor(buf609, (s6, 3072), (3072, 1), 0); del buf609  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf611, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg378_1, (12288, 3072), (1, 12288), 0), out=buf612)
        del arg378_1
        buf613 = reinterpret_tensor(buf612, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf612  # reuse
        buf617 = buf560; del buf560  # reuse
        # Topologically Sorted Source Nodes: [mul_230, encoder_hidden_states_44, layer_norm_45, add_246, mul_232, x_23], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf613, buf593, buf555, arg351_1, arg379_1, buf608, arg383_1, buf617, s6, 3072, grid=grid(s6), stream=stream0)
        del arg351_1
        del arg379_1
        buf618 = reinterpret_tensor(buf593, (s6, 3072), (3072, 1), 0); del buf593  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf617, arg390_1, buf618, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg390_1
        buf619 = buf572; del buf572  # reuse
        # Topologically Sorted Source Nodes: [pow_50, variance_46], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf618, arg391_1, buf619, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf621 = reinterpret_tensor(buf588, (4096, 3072), (3072, 1), 0); del buf588  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf620, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg384_1, (3072, 3072), (1, 3072), 0), out=buf621)
        del arg384_1
        buf622 = buf574; del buf574  # reuse
        # Topologically Sorted Source Nodes: [pow_48, variance_44], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf621, arg385_1, buf622, 98304, 128, grid=grid(98304), stream=stream0)
        buf623 = reinterpret_tensor(buf582, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf582  # reuse
        # Topologically Sorted Source Nodes: [query_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf618, arg391_1, buf619, arg398_1, buf621, arg385_1, buf622, arg396_1, buf623, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg385_1
        del arg391_1
        del arg396_1
        del arg398_1
        buf624 = buf618; del buf618  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf617, arg392_1, buf624, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg392_1
        buf625 = buf619; del buf619  # reuse
        # Topologically Sorted Source Nodes: [pow_51, variance_47], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf624, arg393_1, buf625, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf626 = buf621; del buf621  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf620, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg386_1, (3072, 3072), (1, 3072), 0), out=buf626)
        del arg386_1
        buf627 = buf622; del buf622  # reuse
        # Topologically Sorted Source Nodes: [pow_49, variance_45], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf626, arg387_1, buf627, 98304, 128, grid=grid(98304), stream=stream0)
        buf628 = reinterpret_tensor(buf580, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf580  # reuse
        # Topologically Sorted Source Nodes: [key_58], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf624, arg393_1, buf625, arg399_1, buf626, arg387_1, buf627, arg397_1, buf628, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg387_1
        del arg393_1
        del arg397_1
        del arg399_1
        buf629 = buf624; del buf624  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_22], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg395_1, buf617, arg394_1, buf629, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg394_1
        del arg395_1
        buf630 = buf626; del buf626  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg389_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf620, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg388_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf630)
        del arg388_1
        del arg389_1
        buf631 = buf579; del buf579  # reuse
        buf632 = buf578; del buf578  # reuse
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf623, buf41, buf45, buf628, buf631, buf632, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf623
        buf633 = reinterpret_tensor(buf628, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf628  # reuse
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf629, buf630, buf633, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_46], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf634 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf631, buf632, buf633, scale=0.08838834764831843)
        buf635 = buf634[0]
        del buf634
        buf640 = buf630; del buf630  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf635, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg400_1, (3072, 3072), (1, 3072), 0), out=buf640)
        del arg400_1
        buf641 = reinterpret_tensor(buf640, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf640  # reuse
        buf652 = buf620; del buf620  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_221, attn_output_11, hidden_states_222, norm_hidden_states_22, add_259, mul_246, norm_hidden_states_23], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf641, buf603, buf598, arg381_1, arg401_1, buf652, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg401_1
        buf645 = buf629; del buf629  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf635, arg402_1, buf645, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg402_1
        buf646 = reinterpret_tensor(buf645, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf645  # reuse
        buf662 = buf617; del buf617  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_11, encoder_hidden_states_47, norm_encoder_hidden_states_22, add_263, mul_249, norm_encoder_hidden_states_23], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf646, buf613, buf608, arg383_1, arg403_1, buf662, s6, 3072, grid=grid(s6), stream=stream0)
        del arg403_1
        buf651 = buf555; del buf555  # reuse
        # Topologically Sorted Source Nodes: [silu_27], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf650, arg412_1, buf651, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg412_1
        del buf650
        buf653 = reinterpret_tensor(buf601, (4096, 12288), (12288, 1), 0); del buf601  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf652, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg404_1, (3072, 12288), (1, 3072), 0), out=buf653)
        del arg404_1
        buf654 = reinterpret_tensor(buf653, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf653  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_224, hidden_states_225], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf654, arg405_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg405_1
        buf655 = reinterpret_tensor(buf652, (4096, 3072), (3072, 1), 0); del buf652  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf654, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg406_1, (12288, 3072), (1, 12288), 0), out=buf655)
        del arg406_1
        buf656 = reinterpret_tensor(buf655, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf655  # reuse
        buf673 = buf603; del buf603  # reuse
        # Topologically Sorted Source Nodes: [ff_output_11, hidden_states_227, layer_norm_48, add_266, mul_251, x_24], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf656, buf641, buf598, arg381_1, arg407_1, buf651, arg413_1, buf673, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg381_1
        del arg407_1
        buf661 = buf598; del buf598  # reuse
        # Topologically Sorted Source Nodes: [silu_28], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf660, arg414_1, buf661, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg414_1
        del buf660
        buf663 = reinterpret_tensor(buf611, (s6, 12288), (12288, 1), 0); del buf611  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf662, arg408_1, buf663, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg408_1
        buf664 = reinterpret_tensor(buf663, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf663  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_229, hidden_states_230], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf664, arg409_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg409_1
        buf665 = reinterpret_tensor(buf662, (s6, 3072), (3072, 1), 0); del buf662  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf664, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg410_1, (12288, 3072), (1, 12288), 0), out=buf665)
        del arg410_1
        buf666 = reinterpret_tensor(buf665, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf665  # reuse
        buf670 = buf613; del buf613  # reuse
        # Topologically Sorted Source Nodes: [mul_250, encoder_hidden_states_48, layer_norm_49, add_268, mul_252, x_25], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf666, buf646, buf608, arg383_1, arg411_1, buf661, arg415_1, buf670, s6, 3072, grid=grid(s6), stream=stream0)
        del arg383_1
        del arg411_1
        buf671 = reinterpret_tensor(buf646, (s6, 3072), (3072, 1), 0); del buf646  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf670, arg422_1, buf671, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg422_1
        buf672 = buf625; del buf625  # reuse
        # Topologically Sorted Source Nodes: [pow_54, variance_50], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf671, arg423_1, buf672, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf674 = reinterpret_tensor(buf641, (4096, 3072), (3072, 1), 0); del buf641  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf673, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg416_1, (3072, 3072), (1, 3072), 0), out=buf674)
        del arg416_1
        buf675 = buf627; del buf627  # reuse
        # Topologically Sorted Source Nodes: [pow_52, variance_48], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf674, arg417_1, buf675, 98304, 128, grid=grid(98304), stream=stream0)
        buf676 = reinterpret_tensor(buf635, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf635  # reuse
        # Topologically Sorted Source Nodes: [query_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf671, arg423_1, buf672, arg430_1, buf674, arg417_1, buf675, arg428_1, buf676, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg417_1
        del arg423_1
        del arg428_1
        del arg430_1
        buf677 = buf671; del buf671  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf670, arg424_1, buf677, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg424_1
        buf678 = buf672; del buf672  # reuse
        # Topologically Sorted Source Nodes: [pow_55, variance_51], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf677, arg425_1, buf678, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf679 = buf674; del buf674  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf673, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg418_1, (3072, 3072), (1, 3072), 0), out=buf679)
        del arg418_1
        buf680 = buf675; del buf675  # reuse
        # Topologically Sorted Source Nodes: [pow_53, variance_49], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf679, arg419_1, buf680, 98304, 128, grid=grid(98304), stream=stream0)
        buf681 = reinterpret_tensor(buf633, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf633  # reuse
        # Topologically Sorted Source Nodes: [key_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf677, arg425_1, buf678, arg431_1, buf679, arg419_1, buf680, arg429_1, buf681, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg419_1
        del arg425_1
        del arg429_1
        del arg431_1
        buf682 = buf677; del buf677  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_24], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg427_1, buf670, arg426_1, buf682, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg426_1
        del arg427_1
        buf683 = buf679; del buf679  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg421_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf673, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg420_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf683)
        del arg420_1
        del arg421_1
        buf684 = buf632; del buf632  # reuse
        buf685 = buf631; del buf631  # reuse
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf676, buf41, buf45, buf681, buf684, buf685, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf676
        buf686 = reinterpret_tensor(buf681, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf681  # reuse
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf682, buf683, buf686, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_50], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf687 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf684, buf685, buf686, scale=0.08838834764831843)
        buf688 = buf687[0]
        del buf687
        buf693 = buf683; del buf683  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf688, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg432_1, (3072, 3072), (1, 3072), 0), out=buf693)
        del arg432_1
        buf694 = reinterpret_tensor(buf693, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf693  # reuse
        buf705 = buf673; del buf673  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_240, attn_output_12, hidden_states_241, norm_hidden_states_24, add_281, mul_266, norm_hidden_states_25], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf694, buf656, buf651, arg413_1, arg433_1, buf705, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg433_1
        buf698 = buf682; del buf682  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf688, arg434_1, buf698, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg434_1
        buf699 = reinterpret_tensor(buf698, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf698  # reuse
        buf715 = buf670; del buf670  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_12, encoder_hidden_states_51, norm_encoder_hidden_states_24, add_285, mul_269, norm_encoder_hidden_states_25], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf699, buf666, buf661, arg415_1, arg435_1, buf715, s6, 3072, grid=grid(s6), stream=stream0)
        del arg435_1
        buf704 = buf608; del buf608  # reuse
        # Topologically Sorted Source Nodes: [silu_29], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf703, arg444_1, buf704, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg444_1
        del buf703
        buf706 = reinterpret_tensor(buf654, (4096, 12288), (12288, 1), 0); del buf654  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf705, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg436_1, (3072, 12288), (1, 3072), 0), out=buf706)
        del arg436_1
        buf707 = reinterpret_tensor(buf706, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf706  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_243, hidden_states_244], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf707, arg437_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg437_1
        buf708 = reinterpret_tensor(buf705, (4096, 3072), (3072, 1), 0); del buf705  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf707, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg438_1, (12288, 3072), (1, 12288), 0), out=buf708)
        del arg438_1
        buf709 = reinterpret_tensor(buf708, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf708  # reuse
        buf726 = buf656; del buf656  # reuse
        # Topologically Sorted Source Nodes: [ff_output_12, hidden_states_246, layer_norm_52, add_288, mul_271, x_26], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf709, buf694, buf651, arg413_1, arg439_1, buf704, arg445_1, buf726, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg413_1
        del arg439_1
        buf714 = buf651; del buf651  # reuse
        # Topologically Sorted Source Nodes: [silu_30], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf713, arg446_1, buf714, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg446_1
        del buf713
        buf716 = reinterpret_tensor(buf664, (s6, 12288), (12288, 1), 0); del buf664  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf715, arg440_1, buf716, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg440_1
        buf717 = reinterpret_tensor(buf716, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf716  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_248, hidden_states_249], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf717, arg441_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg441_1
        buf718 = reinterpret_tensor(buf715, (s6, 3072), (3072, 1), 0); del buf715  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf717, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg442_1, (12288, 3072), (1, 12288), 0), out=buf718)
        del arg442_1
        buf719 = reinterpret_tensor(buf718, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf718  # reuse
        buf723 = buf666; del buf666  # reuse
        # Topologically Sorted Source Nodes: [mul_270, encoder_hidden_states_52, layer_norm_53, add_290, mul_272, x_27], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf719, buf699, buf661, arg415_1, arg443_1, buf714, arg447_1, buf723, s6, 3072, grid=grid(s6), stream=stream0)
        del arg415_1
        del arg443_1
        buf724 = reinterpret_tensor(buf699, (s6, 3072), (3072, 1), 0); del buf699  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf723, arg454_1, buf724, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg454_1
        buf725 = buf678; del buf678  # reuse
        # Topologically Sorted Source Nodes: [pow_58, variance_54], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf724, arg455_1, buf725, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf727 = reinterpret_tensor(buf694, (4096, 3072), (3072, 1), 0); del buf694  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf726, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg448_1, (3072, 3072), (1, 3072), 0), out=buf727)
        del arg448_1
        buf728 = buf680; del buf680  # reuse
        # Topologically Sorted Source Nodes: [pow_56, variance_52], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf727, arg449_1, buf728, 98304, 128, grid=grid(98304), stream=stream0)
        buf729 = reinterpret_tensor(buf688, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf688  # reuse
        # Topologically Sorted Source Nodes: [query_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf724, arg455_1, buf725, arg462_1, buf727, arg449_1, buf728, arg460_1, buf729, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg449_1
        del arg455_1
        del arg460_1
        del arg462_1
        buf730 = buf724; del buf724  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf723, arg456_1, buf730, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg456_1
        buf731 = buf725; del buf725  # reuse
        # Topologically Sorted Source Nodes: [pow_59, variance_55], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf730, arg457_1, buf731, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf732 = buf727; del buf727  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf726, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg450_1, (3072, 3072), (1, 3072), 0), out=buf732)
        del arg450_1
        buf733 = buf728; del buf728  # reuse
        # Topologically Sorted Source Nodes: [pow_57, variance_53], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf732, arg451_1, buf733, 98304, 128, grid=grid(98304), stream=stream0)
        buf734 = reinterpret_tensor(buf686, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf686  # reuse
        # Topologically Sorted Source Nodes: [key_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf730, arg457_1, buf731, arg463_1, buf732, arg451_1, buf733, arg461_1, buf734, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg451_1
        del arg457_1
        del arg461_1
        del arg463_1
        buf735 = buf730; del buf730  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_26], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg459_1, buf723, arg458_1, buf735, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg458_1
        del arg459_1
        buf736 = buf732; del buf732  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg453_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf726, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg452_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf736)
        del arg452_1
        del arg453_1
        buf737 = buf685; del buf685  # reuse
        buf738 = buf684; del buf684  # reuse
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf729, buf41, buf45, buf734, buf737, buf738, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf729
        buf739 = reinterpret_tensor(buf734, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf734  # reuse
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf735, buf736, buf739, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_54], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf740 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf737, buf738, buf739, scale=0.08838834764831843)
        buf741 = buf740[0]
        del buf740
        buf746 = buf736; del buf736  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf741, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg464_1, (3072, 3072), (1, 3072), 0), out=buf746)
        del arg464_1
        buf747 = reinterpret_tensor(buf746, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf746  # reuse
        buf758 = buf726; del buf726  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_259, attn_output_13, hidden_states_260, norm_hidden_states_26, add_303, mul_286, norm_hidden_states_27], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf747, buf709, buf704, arg445_1, arg465_1, buf758, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg465_1
        buf751 = buf735; del buf735  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf741, arg466_1, buf751, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg466_1
        buf752 = reinterpret_tensor(buf751, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf751  # reuse
        buf768 = buf723; del buf723  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_13, encoder_hidden_states_55, norm_encoder_hidden_states_26, add_307, mul_289, norm_encoder_hidden_states_27], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf752, buf719, buf714, arg447_1, arg467_1, buf768, s6, 3072, grid=grid(s6), stream=stream0)
        del arg467_1
        buf757 = buf661; del buf661  # reuse
        # Topologically Sorted Source Nodes: [silu_31], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf756, arg476_1, buf757, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg476_1
        del buf756
        buf759 = reinterpret_tensor(buf707, (4096, 12288), (12288, 1), 0); del buf707  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf758, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg468_1, (3072, 12288), (1, 3072), 0), out=buf759)
        del arg468_1
        buf760 = reinterpret_tensor(buf759, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf759  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_262, hidden_states_263], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf760, arg469_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg469_1
        buf761 = reinterpret_tensor(buf758, (4096, 3072), (3072, 1), 0); del buf758  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf760, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg470_1, (12288, 3072), (1, 12288), 0), out=buf761)
        del arg470_1
        buf762 = reinterpret_tensor(buf761, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf761  # reuse
        buf779 = buf709; del buf709  # reuse
        # Topologically Sorted Source Nodes: [ff_output_13, hidden_states_265, layer_norm_56, add_310, mul_291, x_28], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf762, buf747, buf704, arg445_1, arg471_1, buf757, arg477_1, buf779, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg445_1
        del arg471_1
        buf767 = buf704; del buf704  # reuse
        # Topologically Sorted Source Nodes: [silu_32], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf766, arg478_1, buf767, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg478_1
        del buf766
        buf769 = reinterpret_tensor(buf717, (s6, 12288), (12288, 1), 0); del buf717  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf768, arg472_1, buf769, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg472_1
        buf770 = reinterpret_tensor(buf769, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf769  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_267, hidden_states_268], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf770, arg473_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg473_1
        buf771 = reinterpret_tensor(buf768, (s6, 3072), (3072, 1), 0); del buf768  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf770, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg474_1, (12288, 3072), (1, 12288), 0), out=buf771)
        del arg474_1
        buf772 = reinterpret_tensor(buf771, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf771  # reuse
        buf776 = buf719; del buf719  # reuse
        # Topologically Sorted Source Nodes: [mul_290, encoder_hidden_states_56, layer_norm_57, add_312, mul_292, x_29], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf772, buf752, buf714, arg447_1, arg475_1, buf767, arg479_1, buf776, s6, 3072, grid=grid(s6), stream=stream0)
        del arg447_1
        del arg475_1
        buf777 = reinterpret_tensor(buf752, (s6, 3072), (3072, 1), 0); del buf752  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf776, arg486_1, buf777, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg486_1
        buf778 = buf731; del buf731  # reuse
        # Topologically Sorted Source Nodes: [pow_62, variance_58], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf777, arg487_1, buf778, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf780 = reinterpret_tensor(buf747, (4096, 3072), (3072, 1), 0); del buf747  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf779, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg480_1, (3072, 3072), (1, 3072), 0), out=buf780)
        del arg480_1
        buf781 = buf733; del buf733  # reuse
        # Topologically Sorted Source Nodes: [pow_60, variance_56], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf780, arg481_1, buf781, 98304, 128, grid=grid(98304), stream=stream0)
        buf782 = reinterpret_tensor(buf741, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf741  # reuse
        # Topologically Sorted Source Nodes: [query_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf777, arg487_1, buf778, arg494_1, buf780, arg481_1, buf781, arg492_1, buf782, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg481_1
        del arg487_1
        del arg492_1
        del arg494_1
        buf783 = buf777; del buf777  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf776, arg488_1, buf783, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg488_1
        buf784 = buf778; del buf778  # reuse
        # Topologically Sorted Source Nodes: [pow_63, variance_59], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf783, arg489_1, buf784, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf785 = buf780; del buf780  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf779, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg482_1, (3072, 3072), (1, 3072), 0), out=buf785)
        del arg482_1
        buf786 = buf781; del buf781  # reuse
        # Topologically Sorted Source Nodes: [pow_61, variance_57], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf785, arg483_1, buf786, 98304, 128, grid=grid(98304), stream=stream0)
        buf787 = reinterpret_tensor(buf739, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf739  # reuse
        # Topologically Sorted Source Nodes: [key_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf783, arg489_1, buf784, arg495_1, buf785, arg483_1, buf786, arg493_1, buf787, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg483_1
        del arg489_1
        del arg493_1
        del arg495_1
        buf788 = buf783; del buf783  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_28], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg491_1, buf776, arg490_1, buf788, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg490_1
        del arg491_1
        buf789 = buf785; del buf785  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg485_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf779, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg484_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf789)
        del arg484_1
        del arg485_1
        buf790 = buf738; del buf738  # reuse
        buf791 = buf737; del buf737  # reuse
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf782, buf41, buf45, buf787, buf790, buf791, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf782
        buf792 = reinterpret_tensor(buf787, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf787  # reuse
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf788, buf789, buf792, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_58], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf793 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf790, buf791, buf792, scale=0.08838834764831843)
        buf794 = buf793[0]
        del buf793
        buf799 = buf789; del buf789  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf794, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg496_1, (3072, 3072), (1, 3072), 0), out=buf799)
        del arg496_1
        buf800 = reinterpret_tensor(buf799, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf799  # reuse
        buf811 = buf779; del buf779  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_278, attn_output_14, hidden_states_279, norm_hidden_states_28, add_325, mul_306, norm_hidden_states_29], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf800, buf762, buf757, arg477_1, arg497_1, buf811, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg497_1
        buf804 = buf788; del buf788  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf794, arg498_1, buf804, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg498_1
        buf805 = reinterpret_tensor(buf804, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf804  # reuse
        buf821 = buf776; del buf776  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_14, encoder_hidden_states_59, norm_encoder_hidden_states_28, add_329, mul_309, norm_encoder_hidden_states_29], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf805, buf772, buf767, arg479_1, arg499_1, buf821, s6, 3072, grid=grid(s6), stream=stream0)
        del arg499_1
        buf810 = buf714; del buf714  # reuse
        # Topologically Sorted Source Nodes: [silu_33], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf809, arg508_1, buf810, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg508_1
        del buf809
        buf812 = reinterpret_tensor(buf760, (4096, 12288), (12288, 1), 0); del buf760  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf811, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg500_1, (3072, 12288), (1, 3072), 0), out=buf812)
        del arg500_1
        buf813 = reinterpret_tensor(buf812, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf812  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_281, hidden_states_282], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf813, arg501_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg501_1
        buf814 = reinterpret_tensor(buf811, (4096, 3072), (3072, 1), 0); del buf811  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf813, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg502_1, (12288, 3072), (1, 12288), 0), out=buf814)
        del arg502_1
        buf815 = reinterpret_tensor(buf814, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf814  # reuse
        buf832 = buf762; del buf762  # reuse
        # Topologically Sorted Source Nodes: [ff_output_14, hidden_states_284, layer_norm_60, add_332, mul_311, x_30], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf815, buf800, buf757, arg477_1, arg503_1, buf810, arg509_1, buf832, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg477_1
        del arg503_1
        buf820 = buf757; del buf757  # reuse
        # Topologically Sorted Source Nodes: [silu_34], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf819, arg510_1, buf820, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg510_1
        del buf819
        buf822 = reinterpret_tensor(buf770, (s6, 12288), (12288, 1), 0); del buf770  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf821, arg504_1, buf822, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg504_1
        buf823 = reinterpret_tensor(buf822, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf822  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_286, hidden_states_287], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf823, arg505_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg505_1
        buf824 = reinterpret_tensor(buf821, (s6, 3072), (3072, 1), 0); del buf821  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf823, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg506_1, (12288, 3072), (1, 12288), 0), out=buf824)
        del arg506_1
        buf825 = reinterpret_tensor(buf824, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf824  # reuse
        buf829 = buf772; del buf772  # reuse
        # Topologically Sorted Source Nodes: [mul_310, encoder_hidden_states_60, layer_norm_61, add_334, mul_312, x_31], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf825, buf805, buf767, arg479_1, arg507_1, buf820, arg511_1, buf829, s6, 3072, grid=grid(s6), stream=stream0)
        del arg479_1
        del arg507_1
        buf830 = reinterpret_tensor(buf805, (s6, 3072), (3072, 1), 0); del buf805  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf829, arg518_1, buf830, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg518_1
        buf831 = buf784; del buf784  # reuse
        # Topologically Sorted Source Nodes: [pow_66, variance_62], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf830, arg519_1, buf831, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf833 = reinterpret_tensor(buf800, (4096, 3072), (3072, 1), 0); del buf800  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf832, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg512_1, (3072, 3072), (1, 3072), 0), out=buf833)
        del arg512_1
        buf834 = buf786; del buf786  # reuse
        # Topologically Sorted Source Nodes: [pow_64, variance_60], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf833, arg513_1, buf834, 98304, 128, grid=grid(98304), stream=stream0)
        buf835 = reinterpret_tensor(buf794, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf794  # reuse
        # Topologically Sorted Source Nodes: [query_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf830, arg519_1, buf831, arg526_1, buf833, arg513_1, buf834, arg524_1, buf835, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg513_1
        del arg519_1
        del arg524_1
        del arg526_1
        buf836 = buf830; del buf830  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf829, arg520_1, buf836, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg520_1
        buf837 = buf831; del buf831  # reuse
        # Topologically Sorted Source Nodes: [pow_67, variance_63], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf836, arg521_1, buf837, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf838 = buf833; del buf833  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf832, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg514_1, (3072, 3072), (1, 3072), 0), out=buf838)
        del arg514_1
        buf839 = buf834; del buf834  # reuse
        # Topologically Sorted Source Nodes: [pow_65, variance_61], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf838, arg515_1, buf839, 98304, 128, grid=grid(98304), stream=stream0)
        buf840 = reinterpret_tensor(buf792, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf792  # reuse
        # Topologically Sorted Source Nodes: [key_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf836, arg521_1, buf837, arg527_1, buf838, arg515_1, buf839, arg525_1, buf840, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg515_1
        del arg521_1
        del arg525_1
        del arg527_1
        buf841 = buf836; del buf836  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_30], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg523_1, buf829, arg522_1, buf841, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg522_1
        del arg523_1
        buf842 = buf838; del buf838  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg517_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf832, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg516_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf842)
        del arg516_1
        del arg517_1
        buf843 = buf791; del buf791  # reuse
        buf844 = buf790; del buf790  # reuse
        # Topologically Sorted Source Nodes: [out_62], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf835, buf41, buf45, buf840, buf843, buf844, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf835
        buf845 = reinterpret_tensor(buf840, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf840  # reuse
        # Topologically Sorted Source Nodes: [out_62], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf841, buf842, buf845, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_62], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf846 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf843, buf844, buf845, scale=0.08838834764831843)
        buf847 = buf846[0]
        del buf846
        buf852 = buf842; del buf842  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf847, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg528_1, (3072, 3072), (1, 3072), 0), out=buf852)
        del arg528_1
        buf853 = reinterpret_tensor(buf852, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf852  # reuse
        buf864 = buf832; del buf832  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_297, attn_output_15, hidden_states_298, norm_hidden_states_30, add_347, mul_326, norm_hidden_states_31], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf853, buf815, buf810, arg509_1, arg529_1, buf864, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg529_1
        buf857 = buf841; del buf841  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf847, arg530_1, buf857, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg530_1
        buf858 = reinterpret_tensor(buf857, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf857  # reuse
        buf874 = buf829; del buf829  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_15, encoder_hidden_states_63, norm_encoder_hidden_states_30, add_351, mul_329, norm_encoder_hidden_states_31], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf858, buf825, buf820, arg511_1, arg531_1, buf874, s6, 3072, grid=grid(s6), stream=stream0)
        del arg531_1
        buf863 = buf767; del buf767  # reuse
        # Topologically Sorted Source Nodes: [silu_35], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf862, arg540_1, buf863, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg540_1
        del buf862
        buf865 = reinterpret_tensor(buf813, (4096, 12288), (12288, 1), 0); del buf813  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf864, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg532_1, (3072, 12288), (1, 3072), 0), out=buf865)
        del arg532_1
        buf866 = reinterpret_tensor(buf865, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf865  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_300, hidden_states_301], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf866, arg533_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg533_1
        buf867 = reinterpret_tensor(buf864, (4096, 3072), (3072, 1), 0); del buf864  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf866, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg534_1, (12288, 3072), (1, 12288), 0), out=buf867)
        del arg534_1
        buf868 = reinterpret_tensor(buf867, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf867  # reuse
        buf885 = buf815; del buf815  # reuse
        # Topologically Sorted Source Nodes: [ff_output_15, hidden_states_303, layer_norm_64, add_354, mul_331, x_32], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf868, buf853, buf810, arg509_1, arg535_1, buf863, arg541_1, buf885, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg509_1
        del arg535_1
        buf873 = buf810; del buf810  # reuse
        # Topologically Sorted Source Nodes: [silu_36], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf872, arg542_1, buf873, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg542_1
        del buf872
        buf875 = reinterpret_tensor(buf823, (s6, 12288), (12288, 1), 0); del buf823  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf874, arg536_1, buf875, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg536_1
        buf876 = reinterpret_tensor(buf875, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf875  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_305, hidden_states_306], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf876, arg537_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg537_1
        buf877 = reinterpret_tensor(buf874, (s6, 3072), (3072, 1), 0); del buf874  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf876, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg538_1, (12288, 3072), (1, 12288), 0), out=buf877)
        del arg538_1
        buf878 = reinterpret_tensor(buf877, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf877  # reuse
        buf882 = buf825; del buf825  # reuse
        # Topologically Sorted Source Nodes: [mul_330, encoder_hidden_states_64, layer_norm_65, add_356, mul_332, x_33], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf878, buf858, buf820, arg511_1, arg539_1, buf873, arg543_1, buf882, s6, 3072, grid=grid(s6), stream=stream0)
        del arg511_1
        del arg539_1
        buf883 = reinterpret_tensor(buf858, (s6, 3072), (3072, 1), 0); del buf858  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf882, arg550_1, buf883, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg550_1
        buf884 = buf837; del buf837  # reuse
        # Topologically Sorted Source Nodes: [pow_70, variance_66], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf883, arg551_1, buf884, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf886 = reinterpret_tensor(buf853, (4096, 3072), (3072, 1), 0); del buf853  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf885, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg544_1, (3072, 3072), (1, 3072), 0), out=buf886)
        del arg544_1
        buf887 = buf839; del buf839  # reuse
        # Topologically Sorted Source Nodes: [pow_68, variance_64], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf886, arg545_1, buf887, 98304, 128, grid=grid(98304), stream=stream0)
        buf888 = reinterpret_tensor(buf847, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf847  # reuse
        # Topologically Sorted Source Nodes: [query_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf883, arg551_1, buf884, arg558_1, buf886, arg545_1, buf887, arg556_1, buf888, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg545_1
        del arg551_1
        del arg556_1
        del arg558_1
        buf889 = buf883; del buf883  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf882, arg552_1, buf889, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg552_1
        buf890 = buf884; del buf884  # reuse
        # Topologically Sorted Source Nodes: [pow_71, variance_67], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf889, arg553_1, buf890, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf891 = buf886; del buf886  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf885, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg546_1, (3072, 3072), (1, 3072), 0), out=buf891)
        del arg546_1
        buf892 = buf887; del buf887  # reuse
        # Topologically Sorted Source Nodes: [pow_69, variance_65], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf891, arg547_1, buf892, 98304, 128, grid=grid(98304), stream=stream0)
        buf893 = reinterpret_tensor(buf845, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf845  # reuse
        # Topologically Sorted Source Nodes: [key_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf889, arg553_1, buf890, arg559_1, buf891, arg547_1, buf892, arg557_1, buf893, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg547_1
        del arg553_1
        del arg557_1
        del arg559_1
        buf894 = buf889; del buf889  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_32], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg555_1, buf882, arg554_1, buf894, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg554_1
        del arg555_1
        buf895 = buf891; del buf891  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg549_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf885, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg548_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf895)
        del arg548_1
        del arg549_1
        buf896 = buf844; del buf844  # reuse
        buf897 = buf843; del buf843  # reuse
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf888, buf41, buf45, buf893, buf896, buf897, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        del buf888
        buf898 = reinterpret_tensor(buf893, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf893  # reuse
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf894, buf895, buf898, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_66], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf899 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf896, buf897, buf898, scale=0.08838834764831843)
        buf900 = buf899[0]
        del buf899
        buf905 = buf895; del buf895  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf900, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg560_1, (3072, 3072), (1, 3072), 0), out=buf905)
        del arg560_1
        buf906 = reinterpret_tensor(buf905, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf905  # reuse
        buf917 = buf885; del buf885  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_316, attn_output_16, hidden_states_317, norm_hidden_states_32, add_369, mul_346, norm_hidden_states_33], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf906, buf868, buf863, arg541_1, arg561_1, buf917, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg561_1
        buf910 = buf894; del buf894  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf900, arg562_1, buf910, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg562_1
        buf911 = reinterpret_tensor(buf910, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf910  # reuse
        buf927 = buf882; del buf882  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_16, encoder_hidden_states_67, norm_encoder_hidden_states_32, add_373, mul_349, norm_encoder_hidden_states_33], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf911, buf878, buf873, arg543_1, arg563_1, buf927, s6, 3072, grid=grid(s6), stream=stream0)
        del arg563_1
        buf916 = buf820; del buf820  # reuse
        # Topologically Sorted Source Nodes: [silu_37], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf915, arg572_1, buf916, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg572_1
        del buf915
        buf918 = reinterpret_tensor(buf866, (4096, 12288), (12288, 1), 0); del buf866  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf917, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg564_1, (3072, 12288), (1, 3072), 0), out=buf918)
        del arg564_1
        buf919 = reinterpret_tensor(buf918, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf918  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_319, hidden_states_320], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf919, arg565_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg565_1
        buf920 = reinterpret_tensor(buf917, (4096, 3072), (3072, 1), 0); del buf917  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf919, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg566_1, (12288, 3072), (1, 12288), 0), out=buf920)
        del arg566_1
        buf921 = reinterpret_tensor(buf920, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf920  # reuse
        buf938 = buf868; del buf868  # reuse
        # Topologically Sorted Source Nodes: [ff_output_16, hidden_states_322, layer_norm_68, add_376, mul_351, x_34], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf921, buf906, buf863, arg541_1, arg567_1, buf916, arg573_1, buf938, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg541_1
        del arg567_1
        buf926 = buf863; del buf863  # reuse
        # Topologically Sorted Source Nodes: [silu_38], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf925, arg574_1, buf926, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg574_1
        del buf925
        buf928 = reinterpret_tensor(buf876, (s6, 12288), (12288, 1), 0); del buf876  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf927, arg568_1, buf928, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg568_1
        buf929 = reinterpret_tensor(buf928, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf928  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_324, hidden_states_325], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf929, arg569_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg569_1
        buf930 = reinterpret_tensor(buf927, (s6, 3072), (3072, 1), 0); del buf927  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf929, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg570_1, (12288, 3072), (1, 12288), 0), out=buf930)
        del arg570_1
        buf931 = reinterpret_tensor(buf930, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf930  # reuse
        buf935 = buf878; del buf878  # reuse
        # Topologically Sorted Source Nodes: [mul_350, encoder_hidden_states_68, layer_norm_69, add_378, mul_352, x_35], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf931, buf911, buf873, arg543_1, arg571_1, buf926, arg575_1, buf935, s6, 3072, grid=grid(s6), stream=stream0)
        del arg543_1
        del arg571_1
        buf936 = reinterpret_tensor(buf911, (s6, 3072), (3072, 1), 0); del buf911  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf935, arg582_1, buf936, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg582_1
        buf937 = buf890; del buf890  # reuse
        # Topologically Sorted Source Nodes: [pow_74, variance_70], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf936, arg583_1, buf937, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf939 = reinterpret_tensor(buf906, (4096, 3072), (3072, 1), 0); del buf906  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf938, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg576_1, (3072, 3072), (1, 3072), 0), out=buf939)
        del arg576_1
        buf940 = buf892; del buf892  # reuse
        # Topologically Sorted Source Nodes: [pow_72, variance_68], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf939, arg577_1, buf940, 98304, 128, grid=grid(98304), stream=stream0)
        buf941 = reinterpret_tensor(buf900, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf900  # reuse
        # Topologically Sorted Source Nodes: [query_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf936, arg583_1, buf937, arg590_1, buf939, arg577_1, buf940, arg588_1, buf941, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg577_1
        del arg583_1
        del arg588_1
        del arg590_1
        buf942 = buf936; del buf936  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf935, arg584_1, buf942, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg584_1
        buf943 = buf937; del buf937  # reuse
        # Topologically Sorted Source Nodes: [pow_75, variance_71], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf942, arg585_1, buf943, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf944 = buf939; del buf939  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf938, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg578_1, (3072, 3072), (1, 3072), 0), out=buf944)
        del arg578_1
        buf945 = buf940; del buf940  # reuse
        # Topologically Sorted Source Nodes: [pow_73, variance_69], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf944, arg579_1, buf945, 98304, 128, grid=grid(98304), stream=stream0)
        buf946 = reinterpret_tensor(buf898, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf898  # reuse
        # Topologically Sorted Source Nodes: [key_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf942, arg585_1, buf943, arg591_1, buf944, arg579_1, buf945, arg589_1, buf946, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg579_1
        del arg585_1
        del arg589_1
        del arg591_1
        buf947 = buf942; del buf942  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_34], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg587_1, buf935, arg586_1, buf947, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg586_1
        del arg587_1
        buf948 = buf944; del buf944  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg581_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf938, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg580_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf948)
        del arg580_1
        del arg581_1
        buf949 = buf897; del buf897  # reuse
        buf950 = buf896; del buf896  # reuse
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf941, buf41, buf45, buf946, buf949, buf950, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        buf951 = reinterpret_tensor(buf946, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf946  # reuse
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf947, buf948, buf951, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_70], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf952 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf949, buf950, buf951, scale=0.08838834764831843)
        buf953 = buf952[0]
        del buf952
        buf958 = buf948; del buf948  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf953, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg592_1, (3072, 3072), (1, 3072), 0), out=buf958)
        del arg592_1
        buf959 = reinterpret_tensor(buf958, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf958  # reuse
        buf970 = buf938; del buf938  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_335, attn_output_17, hidden_states_336, norm_hidden_states_34, add_391, mul_366, norm_hidden_states_35], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf959, buf921, buf916, arg573_1, arg593_1, buf970, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg593_1
        buf963 = buf947; del buf947  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf953, arg594_1, buf963, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg594_1
        buf964 = reinterpret_tensor(buf963, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf963  # reuse
        buf980 = buf935; del buf935  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_17, encoder_hidden_states_71, norm_encoder_hidden_states_34, add_395, mul_369, norm_encoder_hidden_states_35], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf964, buf931, buf926, arg575_1, arg595_1, buf980, s6, 3072, grid=grid(s6), stream=stream0)
        del arg595_1
        buf969 = buf873; del buf873  # reuse
        # Topologically Sorted Source Nodes: [silu_39], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf968, arg604_1, buf969, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg604_1
        del buf968
        buf971 = reinterpret_tensor(buf919, (4096, 12288), (12288, 1), 0); del buf919  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf970, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg596_1, (3072, 12288), (1, 3072), 0), out=buf971)
        del arg596_1
        buf972 = reinterpret_tensor(buf971, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf971  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_338, hidden_states_339], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf972, arg597_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg597_1
        buf973 = reinterpret_tensor(buf970, (4096, 3072), (3072, 1), 0); del buf970  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf972, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg598_1, (12288, 3072), (1, 12288), 0), out=buf973)
        del arg598_1
        buf974 = reinterpret_tensor(buf973, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf973  # reuse
        buf991 = buf921; del buf921  # reuse
        # Topologically Sorted Source Nodes: [ff_output_17, hidden_states_341, layer_norm_72, add_398, mul_371, x_36], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_24.run(buf974, buf959, buf916, arg573_1, arg599_1, buf969, arg605_1, buf991, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg573_1
        del arg599_1
        buf979 = buf916; del buf916  # reuse
        # Topologically Sorted Source Nodes: [silu_40], Original ATen: [aten.silu]
        triton_tem_fused_silu_6.run(buf978, arg606_1, buf979, grid=torch._inductor.kernel.mm_common.mm_grid(1, 18432, meta2), stream=stream0)
        del arg606_1
        del buf978
        buf981 = reinterpret_tensor(buf929, (s6, 12288), (12288, 1), 0); del buf929  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf980, arg600_1, buf981, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg600_1
        buf982 = reinterpret_tensor(buf981, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf981  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_343, hidden_states_344], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf982, arg601_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg601_1
        buf983 = reinterpret_tensor(buf980, (s6, 3072), (3072, 1), 0); del buf980  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf982, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg602_1, (12288, 3072), (1, 12288), 0), out=buf983)
        del arg602_1
        buf984 = reinterpret_tensor(buf983, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf983  # reuse
        buf988 = buf931; del buf931  # reuse
        # Topologically Sorted Source Nodes: [mul_370, encoder_hidden_states_72, layer_norm_73, add_400, mul_372, x_37], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_27.run(buf984, buf964, buf926, arg575_1, arg603_1, buf979, arg607_1, buf988, s6, 3072, grid=grid(s6), stream=stream0)
        del arg575_1
        del arg603_1
        del buf926
        buf989 = reinterpret_tensor(buf964, (s6, 3072), (3072, 1), 0); del buf964  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf988, arg614_1, buf989, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg614_1
        buf990 = buf943; del buf943  # reuse
        # Topologically Sorted Source Nodes: [pow_78, variance_74], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf989, arg615_1, buf990, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf992 = reinterpret_tensor(buf959, (4096, 3072), (3072, 1), 0); del buf959  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf991, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg608_1, (3072, 3072), (1, 3072), 0), out=buf992)
        del arg608_1
        buf993 = buf945; del buf945  # reuse
        # Topologically Sorted Source Nodes: [pow_76, variance_72], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf992, arg609_1, buf993, 98304, 128, grid=grid(98304), stream=stream0)
        buf994 = reinterpret_tensor(buf953, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf953  # reuse
        # Topologically Sorted Source Nodes: [query_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf989, arg615_1, buf990, arg622_1, buf992, arg609_1, buf993, arg620_1, buf994, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg609_1
        del arg615_1
        del arg620_1
        del arg622_1
        buf995 = buf989; del buf989  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf988, arg616_1, buf995, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg616_1
        buf996 = buf990; del buf990  # reuse
        # Topologically Sorted Source Nodes: [pow_79, variance_75], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_12_xnumel = 24*s6
        triton_per_fused_mean_pow_12.run(buf995, arg617_1, buf996, triton_per_fused_mean_pow_12_xnumel, 128, grid=grid(triton_per_fused_mean_pow_12_xnumel), stream=stream0)
        buf997 = buf992; del buf992  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf991, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg610_1, (3072, 3072), (1, 3072), 0), out=buf997)
        del arg610_1
        buf998 = buf993; del buf993  # reuse
        # Topologically Sorted Source Nodes: [pow_77, variance_73], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_mean_pow_13.run(buf997, arg611_1, buf998, 98304, 128, grid=grid(98304), stream=stream0)
        buf999 = reinterpret_tensor(buf951, (1, 4096 + s6, 24, 128), (12582912 + (3072*s6), 3072, 128, 1), 0); del buf951  # reuse
        # Topologically Sorted Source Nodes: [key_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_14_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_cat_14.run(buf995, arg617_1, buf996, arg623_1, buf997, arg611_1, buf998, arg621_1, buf999, s6, triton_poi_fused_cat_14_xnumel, grid=grid(triton_poi_fused_cat_14_xnumel), stream=stream0)
        del arg611_1
        del arg617_1
        del arg621_1
        del arg623_1
        del buf996
        del buf998
        buf1000 = buf995; del buf995  # reuse
        # Topologically Sorted Source Nodes: [encoder_value_36], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_18.run(arg619_1, buf988, arg618_1, buf1000, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg618_1
        del arg619_1
        buf1001 = buf997; del buf997  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg613_1, (4096, 3072), (0, 1), 0), reinterpret_tensor(buf991, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg612_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1001)
        del arg612_1
        del arg613_1
        buf1002 = buf950; del buf950  # reuse
        buf1003 = buf949; del buf949  # reuse
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_19.run(buf994, buf41, buf45, buf999, buf1002, buf1003, triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_19_xnumel), stream=stream0)
        buf1004 = reinterpret_tensor(buf999, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf999  # reuse
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._scaled_dot_product_flash_attention]
        triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel = 12582912 + (3072*s6)
        triton_poi_fused__scaled_dot_product_flash_attention_20.run(buf1000, buf1001, buf1004, s6, triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel, grid=grid(triton_poi_fused__scaled_dot_product_flash_attention_20_xnumel), stream=stream0)
        # Topologically Sorted Source Nodes: [out_74], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1005 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1002, buf1003, buf1004, scale=0.08838834764831843)
        buf1006 = buf1005[0]
        del buf1005
        buf1011 = buf1001; del buf1001  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1006, (4096, 3072), (3072, 1), 3072*s6), reinterpret_tensor(arg624_1, (3072, 3072), (1, 3072), 0), out=buf1011)
        del arg624_1
        buf1012 = reinterpret_tensor(buf1011, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf1011  # reuse
        buf1027 = buf991; del buf991  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_354, attn_output_18, hidden_states_355, norm_hidden_states_36, add_413, mul_386, norm_hidden_states_37], Original ATen: [aten.clone, aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_clone_mul_native_layer_norm_28.run(buf1012, buf974, buf969, arg605_1, arg625_1, buf1027, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg625_1
        del buf974
        buf1016 = buf1000; del buf1000  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_11.run(buf1006, arg626_1, buf1016, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 3072, meta4), stream=stream0)
        del arg626_1
        buf1017 = reinterpret_tensor(buf1016, (1, s6, 3072), (3072*s6, 3072, 1), 0); del buf1016  # reuse
        buf1023 = buf988; del buf988  # reuse
        # Topologically Sorted Source Nodes: [context_attn_output_18, encoder_hidden_states_75, norm_encoder_hidden_states_36, add_417, mul_389, norm_encoder_hidden_states_37], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        triton_red_fused_add_mul_native_layer_norm_29.run(buf1017, buf984, buf979, arg607_1, arg627_1, buf1023, s6, 3072, grid=grid(s6), stream=stream0)
        del arg627_1
        del buf984
        buf1022 = empty_strided_cuda((1, 9216), (9216, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_41], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1021, arg636_1, buf1022, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg636_1
        del buf1021
        buf1024 = reinterpret_tensor(buf982, (s6, 12288), (12288, 1), 0); del buf982  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: []
        triton_tem_fused_25.run(buf1023, arg632_1, buf1024, s6, grid=torch._inductor.kernel.mm_common.mm_grid(s6, 12288, meta4), stream=stream0)
        del arg632_1
        buf1025 = reinterpret_tensor(buf1024, (1, s6, 12288), (12288*s6, 12288, 1), 0); del buf1024  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_362, hidden_states_363], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_26_xnumel = 12288*s6
        triton_poi_fused_clone_gelu_26.run(buf1025, arg633_1, triton_poi_fused_clone_gelu_26_xnumel, grid=grid(triton_poi_fused_clone_gelu_26_xnumel), stream=stream0)
        del arg633_1
        buf1026 = reinterpret_tensor(buf1023, (s6, 3072), (3072, 1), 0); del buf1023  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1025, (s6, 12288), (12288, 1), 0), reinterpret_tensor(arg634_1, (12288, 3072), (1, 12288), 0), out=buf1026)
        del arg634_1
        del buf1025
        buf1028 = reinterpret_tensor(buf972, (4096, 12288), (12288, 1), 0); del buf972  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1027, (4096, 3072), (3072, 1), 0), reinterpret_tensor(arg628_1, (3072, 12288), (1, 3072), 0), out=buf1028)
        del arg628_1
        buf1029 = reinterpret_tensor(buf1028, (1, 4096, 12288), (50331648, 12288, 1), 0); del buf1028  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_357, hidden_states_358], Original ATen: [aten.gelu, aten.clone]
        triton_poi_fused_clone_gelu_23.run(buf1029, arg629_1, 50331648, grid=grid(50331648), stream=stream0)
        del arg629_1
        buf1030 = reinterpret_tensor(buf1027, (4096, 3072), (3072, 1), 0); del buf1027  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1029, (4096, 12288), (12288, 1), 0), reinterpret_tensor(arg630_1, (12288, 3072), (1, 12288), 0), out=buf1030)
        del arg630_1
        del buf1029
        buf1031 = reinterpret_tensor(buf1006, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1006  # reuse
        buf1035 = reinterpret_tensor(buf1004, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1004  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_365, layer_norm_76, add_420, mul_391, x_38], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_31_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_31.run(buf1017, buf979, arg607_1, buf1026, arg635_1, buf1012, buf969, arg605_1, buf1030, arg631_1, buf1022, arg637_1, buf1031, buf1035, s6, triton_red_fused_add_cat_mul_native_layer_norm_31_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_31_xnumel), stream=stream0)
        del arg605_1
        del arg607_1
        del arg631_1
        del arg635_1
        del buf1012
        del buf1017
        del buf1026
        del buf969
        del buf979
        buf1036 = reinterpret_tensor(buf1003, (4096 + s6, 3072), (3072, 1), 0); del buf1003  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1035, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg640_1, (3072, 3072), (1, 3072), 0), out=buf1036)
        del arg640_1
        buf1038 = reinterpret_tensor(buf1002, (4096 + s6, 3072), (3072, 1), 0); del buf1002  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1035, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg642_1, (3072, 3072), (1, 3072), 0), out=buf1038)
        del arg642_1
        buf1040 = reinterpret_tensor(buf994, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf994  # reuse
        buf1043 = reinterpret_tensor(buf1040, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1040  # reuse
        buf1041 = reinterpret_tensor(buf941, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf941  # reuse
        buf1044 = reinterpret_tensor(buf1041, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1041  # reuse
        # Topologically Sorted Source Nodes: [pow_80, variance_76, pow_81, variance_77, stack_38, stack_39, out_78], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1043, buf1044, buf1036, arg641_1, buf1038, arg643_1, arg646_1, buf41, buf45, arg647_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg641_1
        del arg643_1
        del arg646_1
        del arg647_1
        del buf1036
        buf1042 = buf1038; del buf1038  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg645_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1035, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg644_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1042)
        del arg644_1
        del arg645_1
        # Topologically Sorted Source Nodes: [out_78], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1045 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1043, buf1044, reinterpret_tensor(buf1042, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1046 = buf1045[0]
        del buf1045
        buf1052 = empty_strided_cuda((1, 9216), (9216, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_42], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1051, arg650_1, buf1052, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg650_1
        del buf1051
        buf1053 = empty_strided_cuda((4096 + s6, 12288), (12288, 1), torch.bfloat16)
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1035, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg638_1, (3072, 12288), (1, 3072), 0), out=buf1053)
        del arg638_1
        buf1054 = empty_strided_cuda((1, 4096 + s6, 15360), (62914560 + (15360*s6), 15360, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [hidden_states_370], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1046, buf1053, arg639_1, buf1054, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg639_1
        buf1055 = reinterpret_tensor(buf1046, (4096 + s6, 3072), (3072, 1), 0); del buf1046  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1054, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg648_1, (15360, 3072), (1, 15360), 0), out=buf1055)
        del arg648_1
        buf1056 = reinterpret_tensor(buf1055, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1055  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_371, hidden_states_372], Original ATen: [aten.mul, aten.add]
        triton_poi_fused_add_mul_34_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_mul_34.run(buf1056, buf1031, buf1022, arg637_1, arg649_1, triton_poi_fused_add_mul_34_xnumel, grid=grid(triton_poi_fused_add_mul_34_xnumel), stream=stream0)
        del arg637_1
        del arg649_1
        buf1060 = buf1031; del buf1031  # reuse
        buf1061 = buf1060; del buf1060  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_374, layer_norm_77, add_429, mul_401, x_39], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1061, buf1056, buf1052, arg651_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1062 = reinterpret_tensor(buf1035, (4096 + s6, 3072), (3072, 1), 0); del buf1035  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1061, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg654_1, (3072, 3072), (1, 3072), 0), out=buf1062)
        del arg654_1
        buf1064 = reinterpret_tensor(buf1044, (4096 + s6, 3072), (3072, 1), 0); del buf1044  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1061, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg656_1, (3072, 3072), (1, 3072), 0), out=buf1064)
        del arg656_1
        buf1066 = reinterpret_tensor(buf1043, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1043  # reuse
        buf1069 = reinterpret_tensor(buf1066, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1066  # reuse
        buf1067 = reinterpret_tensor(buf1042, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1042  # reuse
        buf1070 = reinterpret_tensor(buf1067, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1067  # reuse
        # Topologically Sorted Source Nodes: [pow_82, variance_78, pow_83, variance_79, stack_40, stack_41, out_82], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1069, buf1070, buf1062, arg655_1, buf1064, arg657_1, arg660_1, buf41, buf45, arg661_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg655_1
        del arg657_1
        del arg660_1
        del arg661_1
        del buf1062
        buf1068 = buf1064; del buf1064  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg659_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1061, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg658_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1068)
        del arg658_1
        del arg659_1
        # Topologically Sorted Source Nodes: [out_82], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1071 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1069, buf1070, reinterpret_tensor(buf1068, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1072 = buf1071[0]
        del buf1071
        buf1078 = buf1022; del buf1022  # reuse
        # Topologically Sorted Source Nodes: [silu_43], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1077, arg664_1, buf1078, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg664_1
        del buf1077
        buf1079 = buf1053; del buf1053  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1061, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg652_1, (3072, 12288), (1, 3072), 0), out=buf1079)
        del arg652_1
        buf1080 = buf1054; del buf1054  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_379], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1072, buf1079, arg653_1, buf1080, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg653_1
        buf1081 = reinterpret_tensor(buf1072, (4096 + s6, 3072), (3072, 1), 0); del buf1072  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1080, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg662_1, (15360, 3072), (1, 15360), 0), out=buf1081)
        del arg662_1
        buf1082 = reinterpret_tensor(buf1081, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1081  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_374, hidden_states_380, hidden_states_381], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1082, buf1056, buf1052, arg651_1, arg663_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg651_1
        del arg663_1
        buf1086 = buf1056; del buf1056  # reuse
        buf1087 = buf1086; del buf1086  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_383, layer_norm_78, add_438, mul_411, x_40], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1087, buf1082, buf1078, arg665_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1088 = reinterpret_tensor(buf1061, (4096 + s6, 3072), (3072, 1), 0); del buf1061  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1087, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg668_1, (3072, 3072), (1, 3072), 0), out=buf1088)
        del arg668_1
        buf1090 = reinterpret_tensor(buf1070, (4096 + s6, 3072), (3072, 1), 0); del buf1070  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1087, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg670_1, (3072, 3072), (1, 3072), 0), out=buf1090)
        del arg670_1
        buf1092 = reinterpret_tensor(buf1069, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1069  # reuse
        buf1095 = reinterpret_tensor(buf1092, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1092  # reuse
        buf1093 = reinterpret_tensor(buf1068, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1068  # reuse
        buf1096 = reinterpret_tensor(buf1093, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1093  # reuse
        # Topologically Sorted Source Nodes: [pow_84, variance_80, pow_85, variance_81, stack_42, stack_43, out_86], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1095, buf1096, buf1088, arg669_1, buf1090, arg671_1, arg674_1, buf41, buf45, arg675_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg669_1
        del arg671_1
        del arg674_1
        del arg675_1
        del buf1088
        buf1094 = buf1090; del buf1090  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg673_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1087, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg672_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1094)
        del arg672_1
        del arg673_1
        # Topologically Sorted Source Nodes: [out_86], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1097 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1095, buf1096, reinterpret_tensor(buf1094, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1098 = buf1097[0]
        del buf1097
        buf1104 = buf1052; del buf1052  # reuse
        # Topologically Sorted Source Nodes: [silu_44], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1103, arg678_1, buf1104, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg678_1
        del buf1103
        buf1105 = buf1079; del buf1079  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1087, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg666_1, (3072, 12288), (1, 3072), 0), out=buf1105)
        del arg666_1
        buf1106 = buf1080; del buf1080  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_388], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1098, buf1105, arg667_1, buf1106, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg667_1
        buf1107 = reinterpret_tensor(buf1098, (4096 + s6, 3072), (3072, 1), 0); del buf1098  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1106, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg676_1, (15360, 3072), (1, 15360), 0), out=buf1107)
        del arg676_1
        buf1108 = reinterpret_tensor(buf1107, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1107  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_383, hidden_states_389, hidden_states_390], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1108, buf1082, buf1078, arg665_1, arg677_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg665_1
        del arg677_1
        buf1112 = buf1082; del buf1082  # reuse
        buf1113 = buf1112; del buf1112  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_392, layer_norm_79, add_447, mul_421, x_41], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1113, buf1108, buf1104, arg679_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1114 = reinterpret_tensor(buf1087, (4096 + s6, 3072), (3072, 1), 0); del buf1087  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1113, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg682_1, (3072, 3072), (1, 3072), 0), out=buf1114)
        del arg682_1
        buf1116 = reinterpret_tensor(buf1096, (4096 + s6, 3072), (3072, 1), 0); del buf1096  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1113, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg684_1, (3072, 3072), (1, 3072), 0), out=buf1116)
        del arg684_1
        buf1118 = reinterpret_tensor(buf1095, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1095  # reuse
        buf1121 = reinterpret_tensor(buf1118, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1118  # reuse
        buf1119 = reinterpret_tensor(buf1094, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1094  # reuse
        buf1122 = reinterpret_tensor(buf1119, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1119  # reuse
        # Topologically Sorted Source Nodes: [pow_86, variance_82, pow_87, variance_83, stack_44, stack_45, out_90], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1121, buf1122, buf1114, arg683_1, buf1116, arg685_1, arg688_1, buf41, buf45, arg689_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg683_1
        del arg685_1
        del arg688_1
        del arg689_1
        del buf1114
        buf1120 = buf1116; del buf1116  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg687_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1113, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg686_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1120)
        del arg686_1
        del arg687_1
        # Topologically Sorted Source Nodes: [out_90], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1123 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1121, buf1122, reinterpret_tensor(buf1120, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1124 = buf1123[0]
        del buf1123
        buf1130 = buf1078; del buf1078  # reuse
        # Topologically Sorted Source Nodes: [silu_45], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1129, arg692_1, buf1130, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg692_1
        del buf1129
        buf1131 = buf1105; del buf1105  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1113, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg680_1, (3072, 12288), (1, 3072), 0), out=buf1131)
        del arg680_1
        buf1132 = buf1106; del buf1106  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_397], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1124, buf1131, arg681_1, buf1132, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg681_1
        buf1133 = reinterpret_tensor(buf1124, (4096 + s6, 3072), (3072, 1), 0); del buf1124  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1132, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg690_1, (15360, 3072), (1, 15360), 0), out=buf1133)
        del arg690_1
        buf1134 = reinterpret_tensor(buf1133, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1133  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_392, hidden_states_398, hidden_states_399], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1134, buf1108, buf1104, arg679_1, arg691_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg679_1
        del arg691_1
        buf1138 = buf1108; del buf1108  # reuse
        buf1139 = buf1138; del buf1138  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_401, layer_norm_80, add_456, mul_431, x_42], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1139, buf1134, buf1130, arg693_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1140 = reinterpret_tensor(buf1113, (4096 + s6, 3072), (3072, 1), 0); del buf1113  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1139, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg696_1, (3072, 3072), (1, 3072), 0), out=buf1140)
        del arg696_1
        buf1142 = reinterpret_tensor(buf1122, (4096 + s6, 3072), (3072, 1), 0); del buf1122  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1139, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg698_1, (3072, 3072), (1, 3072), 0), out=buf1142)
        del arg698_1
        buf1144 = reinterpret_tensor(buf1121, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1121  # reuse
        buf1147 = reinterpret_tensor(buf1144, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1144  # reuse
        buf1145 = reinterpret_tensor(buf1120, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1120  # reuse
        buf1148 = reinterpret_tensor(buf1145, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1145  # reuse
        # Topologically Sorted Source Nodes: [pow_88, variance_84, pow_89, variance_85, stack_46, stack_47, out_94], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1147, buf1148, buf1140, arg697_1, buf1142, arg699_1, arg702_1, buf41, buf45, arg703_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg697_1
        del arg699_1
        del arg702_1
        del arg703_1
        del buf1140
        buf1146 = buf1142; del buf1142  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg701_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1139, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg700_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1146)
        del arg700_1
        del arg701_1
        # Topologically Sorted Source Nodes: [out_94], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1149 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1147, buf1148, reinterpret_tensor(buf1146, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1150 = buf1149[0]
        del buf1149
        buf1156 = buf1104; del buf1104  # reuse
        # Topologically Sorted Source Nodes: [silu_46], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1155, arg706_1, buf1156, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg706_1
        del buf1155
        buf1157 = buf1131; del buf1131  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1139, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg694_1, (3072, 12288), (1, 3072), 0), out=buf1157)
        del arg694_1
        buf1158 = buf1132; del buf1132  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_406], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1150, buf1157, arg695_1, buf1158, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg695_1
        buf1159 = reinterpret_tensor(buf1150, (4096 + s6, 3072), (3072, 1), 0); del buf1150  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1158, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg704_1, (15360, 3072), (1, 15360), 0), out=buf1159)
        del arg704_1
        buf1160 = reinterpret_tensor(buf1159, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1159  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_401, hidden_states_407, hidden_states_408], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1160, buf1134, buf1130, arg693_1, arg705_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg693_1
        del arg705_1
        buf1164 = buf1134; del buf1134  # reuse
        buf1165 = buf1164; del buf1164  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_410, layer_norm_81, add_465, mul_441, x_43], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1165, buf1160, buf1156, arg707_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1166 = reinterpret_tensor(buf1139, (4096 + s6, 3072), (3072, 1), 0); del buf1139  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1165, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg710_1, (3072, 3072), (1, 3072), 0), out=buf1166)
        del arg710_1
        buf1168 = reinterpret_tensor(buf1148, (4096 + s6, 3072), (3072, 1), 0); del buf1148  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1165, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg712_1, (3072, 3072), (1, 3072), 0), out=buf1168)
        del arg712_1
        buf1170 = reinterpret_tensor(buf1147, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1147  # reuse
        buf1173 = reinterpret_tensor(buf1170, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1170  # reuse
        buf1171 = reinterpret_tensor(buf1146, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1146  # reuse
        buf1174 = reinterpret_tensor(buf1171, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1171  # reuse
        # Topologically Sorted Source Nodes: [pow_90, variance_86, pow_91, variance_87, stack_48, stack_49, out_98], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1173, buf1174, buf1166, arg711_1, buf1168, arg713_1, arg716_1, buf41, buf45, arg717_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg711_1
        del arg713_1
        del arg716_1
        del arg717_1
        del buf1166
        buf1172 = buf1168; del buf1168  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg715_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1165, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg714_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1172)
        del arg714_1
        del arg715_1
        # Topologically Sorted Source Nodes: [out_98], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1175 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1173, buf1174, reinterpret_tensor(buf1172, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1176 = buf1175[0]
        del buf1175
        buf1182 = buf1130; del buf1130  # reuse
        # Topologically Sorted Source Nodes: [silu_47], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1181, arg720_1, buf1182, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg720_1
        del buf1181
        buf1183 = buf1157; del buf1157  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1165, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg708_1, (3072, 12288), (1, 3072), 0), out=buf1183)
        del arg708_1
        buf1184 = buf1158; del buf1158  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_415], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1176, buf1183, arg709_1, buf1184, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg709_1
        buf1185 = reinterpret_tensor(buf1176, (4096 + s6, 3072), (3072, 1), 0); del buf1176  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1184, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg718_1, (15360, 3072), (1, 15360), 0), out=buf1185)
        del arg718_1
        buf1186 = reinterpret_tensor(buf1185, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1185  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_410, hidden_states_416, hidden_states_417], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1186, buf1160, buf1156, arg707_1, arg719_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg707_1
        del arg719_1
        buf1190 = buf1160; del buf1160  # reuse
        buf1191 = buf1190; del buf1190  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_419, layer_norm_82, add_474, mul_451, x_44], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1191, buf1186, buf1182, arg721_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1192 = reinterpret_tensor(buf1165, (4096 + s6, 3072), (3072, 1), 0); del buf1165  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1191, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg724_1, (3072, 3072), (1, 3072), 0), out=buf1192)
        del arg724_1
        buf1194 = reinterpret_tensor(buf1174, (4096 + s6, 3072), (3072, 1), 0); del buf1174  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1191, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg726_1, (3072, 3072), (1, 3072), 0), out=buf1194)
        del arg726_1
        buf1196 = reinterpret_tensor(buf1173, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1173  # reuse
        buf1199 = reinterpret_tensor(buf1196, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1196  # reuse
        buf1197 = reinterpret_tensor(buf1172, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1172  # reuse
        buf1200 = reinterpret_tensor(buf1197, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1197  # reuse
        # Topologically Sorted Source Nodes: [pow_92, variance_88, pow_93, variance_89, stack_50, stack_51, out_102], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1199, buf1200, buf1192, arg725_1, buf1194, arg727_1, arg730_1, buf41, buf45, arg731_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg725_1
        del arg727_1
        del arg730_1
        del arg731_1
        del buf1192
        buf1198 = buf1194; del buf1194  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg729_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1191, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg728_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1198)
        del arg728_1
        del arg729_1
        # Topologically Sorted Source Nodes: [out_102], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1201 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1199, buf1200, reinterpret_tensor(buf1198, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1202 = buf1201[0]
        del buf1201
        buf1208 = buf1156; del buf1156  # reuse
        # Topologically Sorted Source Nodes: [silu_48], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1207, arg734_1, buf1208, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg734_1
        del buf1207
        buf1209 = buf1183; del buf1183  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1191, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg722_1, (3072, 12288), (1, 3072), 0), out=buf1209)
        del arg722_1
        buf1210 = buf1184; del buf1184  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_424], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1202, buf1209, arg723_1, buf1210, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg723_1
        buf1211 = reinterpret_tensor(buf1202, (4096 + s6, 3072), (3072, 1), 0); del buf1202  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1210, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg732_1, (15360, 3072), (1, 15360), 0), out=buf1211)
        del arg732_1
        buf1212 = reinterpret_tensor(buf1211, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1211  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_419, hidden_states_425, hidden_states_426], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1212, buf1186, buf1182, arg721_1, arg733_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg721_1
        del arg733_1
        buf1216 = buf1186; del buf1186  # reuse
        buf1217 = buf1216; del buf1216  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_428, layer_norm_83, add_483, mul_461, x_45], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1217, buf1212, buf1208, arg735_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1218 = reinterpret_tensor(buf1191, (4096 + s6, 3072), (3072, 1), 0); del buf1191  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1217, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg738_1, (3072, 3072), (1, 3072), 0), out=buf1218)
        del arg738_1
        buf1220 = reinterpret_tensor(buf1200, (4096 + s6, 3072), (3072, 1), 0); del buf1200  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1217, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg740_1, (3072, 3072), (1, 3072), 0), out=buf1220)
        del arg740_1
        buf1222 = reinterpret_tensor(buf1199, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1199  # reuse
        buf1225 = reinterpret_tensor(buf1222, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1222  # reuse
        buf1223 = reinterpret_tensor(buf1198, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1198  # reuse
        buf1226 = reinterpret_tensor(buf1223, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1223  # reuse
        # Topologically Sorted Source Nodes: [pow_94, variance_90, pow_95, variance_91, stack_52, stack_53, out_106], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1225, buf1226, buf1218, arg739_1, buf1220, arg741_1, arg744_1, buf41, buf45, arg745_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg739_1
        del arg741_1
        del arg744_1
        del arg745_1
        del buf1218
        buf1224 = buf1220; del buf1220  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg743_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1217, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg742_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1224)
        del arg742_1
        del arg743_1
        # Topologically Sorted Source Nodes: [out_106], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1227 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1225, buf1226, reinterpret_tensor(buf1224, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1228 = buf1227[0]
        del buf1227
        buf1234 = buf1182; del buf1182  # reuse
        # Topologically Sorted Source Nodes: [silu_49], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1233, arg748_1, buf1234, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg748_1
        del buf1233
        buf1235 = buf1209; del buf1209  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1217, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg736_1, (3072, 12288), (1, 3072), 0), out=buf1235)
        del arg736_1
        buf1236 = buf1210; del buf1210  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_433], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1228, buf1235, arg737_1, buf1236, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg737_1
        buf1237 = reinterpret_tensor(buf1228, (4096 + s6, 3072), (3072, 1), 0); del buf1228  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1236, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg746_1, (15360, 3072), (1, 15360), 0), out=buf1237)
        del arg746_1
        buf1238 = reinterpret_tensor(buf1237, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1237  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_428, hidden_states_434, hidden_states_435], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1238, buf1212, buf1208, arg735_1, arg747_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg735_1
        del arg747_1
        buf1242 = buf1212; del buf1212  # reuse
        buf1243 = buf1242; del buf1242  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_437, layer_norm_84, add_492, mul_471, x_46], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1243, buf1238, buf1234, arg749_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1244 = reinterpret_tensor(buf1217, (4096 + s6, 3072), (3072, 1), 0); del buf1217  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1243, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg752_1, (3072, 3072), (1, 3072), 0), out=buf1244)
        del arg752_1
        buf1246 = reinterpret_tensor(buf1226, (4096 + s6, 3072), (3072, 1), 0); del buf1226  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1243, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg754_1, (3072, 3072), (1, 3072), 0), out=buf1246)
        del arg754_1
        buf1248 = reinterpret_tensor(buf1225, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1225  # reuse
        buf1251 = reinterpret_tensor(buf1248, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1248  # reuse
        buf1249 = reinterpret_tensor(buf1224, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1224  # reuse
        buf1252 = reinterpret_tensor(buf1249, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1249  # reuse
        # Topologically Sorted Source Nodes: [pow_96, variance_92, pow_97, variance_93, stack_54, stack_55, out_110], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1251, buf1252, buf1244, arg753_1, buf1246, arg755_1, arg758_1, buf41, buf45, arg759_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg753_1
        del arg755_1
        del arg758_1
        del arg759_1
        del buf1244
        buf1250 = buf1246; del buf1246  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg757_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1243, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg756_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1250)
        del arg756_1
        del arg757_1
        # Topologically Sorted Source Nodes: [out_110], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1253 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1251, buf1252, reinterpret_tensor(buf1250, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1254 = buf1253[0]
        del buf1253
        buf1260 = buf1208; del buf1208  # reuse
        # Topologically Sorted Source Nodes: [silu_50], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1259, arg762_1, buf1260, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg762_1
        del buf1259
        buf1261 = buf1235; del buf1235  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1243, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg750_1, (3072, 12288), (1, 3072), 0), out=buf1261)
        del arg750_1
        buf1262 = buf1236; del buf1236  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_442], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1254, buf1261, arg751_1, buf1262, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg751_1
        buf1263 = reinterpret_tensor(buf1254, (4096 + s6, 3072), (3072, 1), 0); del buf1254  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1262, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg760_1, (15360, 3072), (1, 15360), 0), out=buf1263)
        del arg760_1
        buf1264 = reinterpret_tensor(buf1263, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1263  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_437, hidden_states_443, hidden_states_444], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1264, buf1238, buf1234, arg749_1, arg761_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg749_1
        del arg761_1
        buf1268 = buf1238; del buf1238  # reuse
        buf1269 = buf1268; del buf1268  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_446, layer_norm_85, add_501, mul_481, x_47], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1269, buf1264, buf1260, arg763_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1270 = reinterpret_tensor(buf1243, (4096 + s6, 3072), (3072, 1), 0); del buf1243  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1269, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg766_1, (3072, 3072), (1, 3072), 0), out=buf1270)
        del arg766_1
        buf1272 = reinterpret_tensor(buf1252, (4096 + s6, 3072), (3072, 1), 0); del buf1252  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1269, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg768_1, (3072, 3072), (1, 3072), 0), out=buf1272)
        del arg768_1
        buf1274 = reinterpret_tensor(buf1251, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1251  # reuse
        buf1277 = reinterpret_tensor(buf1274, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1274  # reuse
        buf1275 = reinterpret_tensor(buf1250, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1250  # reuse
        buf1278 = reinterpret_tensor(buf1275, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1275  # reuse
        # Topologically Sorted Source Nodes: [pow_98, variance_94, pow_99, variance_95, stack_56, stack_57, out_114], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1277, buf1278, buf1270, arg767_1, buf1272, arg769_1, arg772_1, buf41, buf45, arg773_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg767_1
        del arg769_1
        del arg772_1
        del arg773_1
        del buf1270
        buf1276 = buf1272; del buf1272  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg771_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1269, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg770_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1276)
        del arg770_1
        del arg771_1
        # Topologically Sorted Source Nodes: [out_114], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1279 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1277, buf1278, reinterpret_tensor(buf1276, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1280 = buf1279[0]
        del buf1279
        buf1286 = buf1234; del buf1234  # reuse
        # Topologically Sorted Source Nodes: [silu_51], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1285, arg776_1, buf1286, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg776_1
        del buf1285
        buf1287 = buf1261; del buf1261  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1269, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg764_1, (3072, 12288), (1, 3072), 0), out=buf1287)
        del arg764_1
        buf1288 = buf1262; del buf1262  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_451], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1280, buf1287, arg765_1, buf1288, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg765_1
        buf1289 = reinterpret_tensor(buf1280, (4096 + s6, 3072), (3072, 1), 0); del buf1280  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1288, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg774_1, (15360, 3072), (1, 15360), 0), out=buf1289)
        del arg774_1
        buf1290 = reinterpret_tensor(buf1289, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1289  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_446, hidden_states_452, hidden_states_453], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1290, buf1264, buf1260, arg763_1, arg775_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg763_1
        del arg775_1
        buf1294 = buf1264; del buf1264  # reuse
        buf1295 = buf1294; del buf1294  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_455, layer_norm_86, add_510, mul_491, x_48], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1295, buf1290, buf1286, arg777_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1296 = reinterpret_tensor(buf1269, (4096 + s6, 3072), (3072, 1), 0); del buf1269  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1295, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg780_1, (3072, 3072), (1, 3072), 0), out=buf1296)
        del arg780_1
        buf1298 = reinterpret_tensor(buf1278, (4096 + s6, 3072), (3072, 1), 0); del buf1278  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1295, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg782_1, (3072, 3072), (1, 3072), 0), out=buf1298)
        del arg782_1
        buf1300 = reinterpret_tensor(buf1277, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1277  # reuse
        buf1303 = reinterpret_tensor(buf1300, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1300  # reuse
        buf1301 = reinterpret_tensor(buf1276, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1276  # reuse
        buf1304 = reinterpret_tensor(buf1301, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1301  # reuse
        # Topologically Sorted Source Nodes: [pow_100, variance_96, pow_101, variance_97, stack_58, stack_59, out_118], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1303, buf1304, buf1296, arg781_1, buf1298, arg783_1, arg786_1, buf41, buf45, arg787_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg781_1
        del arg783_1
        del arg786_1
        del arg787_1
        del buf1296
        buf1302 = buf1298; del buf1298  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg785_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1295, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg784_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1302)
        del arg784_1
        del arg785_1
        # Topologically Sorted Source Nodes: [out_118], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1305 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1303, buf1304, reinterpret_tensor(buf1302, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1306 = buf1305[0]
        del buf1305
        buf1312 = buf1260; del buf1260  # reuse
        # Topologically Sorted Source Nodes: [silu_52], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1311, arg790_1, buf1312, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg790_1
        del buf1311
        buf1313 = buf1287; del buf1287  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1295, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg778_1, (3072, 12288), (1, 3072), 0), out=buf1313)
        del arg778_1
        buf1314 = buf1288; del buf1288  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_460], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1306, buf1313, arg779_1, buf1314, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg779_1
        buf1315 = reinterpret_tensor(buf1306, (4096 + s6, 3072), (3072, 1), 0); del buf1306  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1314, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg788_1, (15360, 3072), (1, 15360), 0), out=buf1315)
        del arg788_1
        buf1316 = reinterpret_tensor(buf1315, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1315  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_455, hidden_states_461, hidden_states_462], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1316, buf1290, buf1286, arg777_1, arg789_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg777_1
        del arg789_1
        buf1320 = buf1290; del buf1290  # reuse
        buf1321 = buf1320; del buf1320  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_464, layer_norm_87, add_519, mul_501, x_49], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1321, buf1316, buf1312, arg791_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1322 = reinterpret_tensor(buf1295, (4096 + s6, 3072), (3072, 1), 0); del buf1295  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1321, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg794_1, (3072, 3072), (1, 3072), 0), out=buf1322)
        del arg794_1
        buf1324 = reinterpret_tensor(buf1304, (4096 + s6, 3072), (3072, 1), 0); del buf1304  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1321, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg796_1, (3072, 3072), (1, 3072), 0), out=buf1324)
        del arg796_1
        buf1326 = reinterpret_tensor(buf1303, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1303  # reuse
        buf1329 = reinterpret_tensor(buf1326, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1326  # reuse
        buf1327 = reinterpret_tensor(buf1302, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1302  # reuse
        buf1330 = reinterpret_tensor(buf1327, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1327  # reuse
        # Topologically Sorted Source Nodes: [pow_102, variance_98, pow_103, variance_99, stack_60, stack_61, out_122], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1329, buf1330, buf1322, arg795_1, buf1324, arg797_1, arg800_1, buf41, buf45, arg801_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg795_1
        del arg797_1
        del arg800_1
        del arg801_1
        del buf1322
        buf1328 = buf1324; del buf1324  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg799_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1321, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg798_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1328)
        del arg798_1
        del arg799_1
        # Topologically Sorted Source Nodes: [out_122], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1331 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1329, buf1330, reinterpret_tensor(buf1328, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1332 = buf1331[0]
        del buf1331
        buf1338 = buf1286; del buf1286  # reuse
        # Topologically Sorted Source Nodes: [silu_53], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1337, arg804_1, buf1338, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg804_1
        del buf1337
        buf1339 = buf1313; del buf1313  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1321, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg792_1, (3072, 12288), (1, 3072), 0), out=buf1339)
        del arg792_1
        buf1340 = buf1314; del buf1314  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_469], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1332, buf1339, arg793_1, buf1340, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg793_1
        buf1341 = reinterpret_tensor(buf1332, (4096 + s6, 3072), (3072, 1), 0); del buf1332  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1340, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg802_1, (15360, 3072), (1, 15360), 0), out=buf1341)
        del arg802_1
        buf1342 = reinterpret_tensor(buf1341, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1341  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_464, hidden_states_470, hidden_states_471], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1342, buf1316, buf1312, arg791_1, arg803_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg791_1
        del arg803_1
        buf1346 = buf1316; del buf1316  # reuse
        buf1347 = buf1346; del buf1346  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_473, layer_norm_88, add_528, mul_511, x_50], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1347, buf1342, buf1338, arg805_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1348 = reinterpret_tensor(buf1321, (4096 + s6, 3072), (3072, 1), 0); del buf1321  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1347, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg808_1, (3072, 3072), (1, 3072), 0), out=buf1348)
        del arg808_1
        buf1350 = reinterpret_tensor(buf1330, (4096 + s6, 3072), (3072, 1), 0); del buf1330  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1347, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg810_1, (3072, 3072), (1, 3072), 0), out=buf1350)
        del arg810_1
        buf1352 = reinterpret_tensor(buf1329, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1329  # reuse
        buf1355 = reinterpret_tensor(buf1352, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1352  # reuse
        buf1353 = reinterpret_tensor(buf1328, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1328  # reuse
        buf1356 = reinterpret_tensor(buf1353, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1353  # reuse
        # Topologically Sorted Source Nodes: [pow_104, variance_100, pow_105, variance_101, stack_62, stack_63, out_126], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1355, buf1356, buf1348, arg809_1, buf1350, arg811_1, arg814_1, buf41, buf45, arg815_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg809_1
        del arg811_1
        del arg814_1
        del arg815_1
        del buf1348
        buf1354 = buf1350; del buf1350  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg813_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1347, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg812_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1354)
        del arg812_1
        del arg813_1
        # Topologically Sorted Source Nodes: [out_126], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1357 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1355, buf1356, reinterpret_tensor(buf1354, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1358 = buf1357[0]
        del buf1357
        buf1364 = buf1312; del buf1312  # reuse
        # Topologically Sorted Source Nodes: [silu_54], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1363, arg818_1, buf1364, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg818_1
        del buf1363
        buf1365 = buf1339; del buf1339  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1347, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg806_1, (3072, 12288), (1, 3072), 0), out=buf1365)
        del arg806_1
        buf1366 = buf1340; del buf1340  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_478], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1358, buf1365, arg807_1, buf1366, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg807_1
        buf1367 = reinterpret_tensor(buf1358, (4096 + s6, 3072), (3072, 1), 0); del buf1358  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1366, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg816_1, (15360, 3072), (1, 15360), 0), out=buf1367)
        del arg816_1
        buf1368 = reinterpret_tensor(buf1367, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1367  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_473, hidden_states_479, hidden_states_480], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1368, buf1342, buf1338, arg805_1, arg817_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg805_1
        del arg817_1
        buf1372 = buf1342; del buf1342  # reuse
        buf1373 = buf1372; del buf1372  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_482, layer_norm_89, add_537, mul_521, x_51], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1373, buf1368, buf1364, arg819_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1374 = reinterpret_tensor(buf1347, (4096 + s6, 3072), (3072, 1), 0); del buf1347  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1373, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg822_1, (3072, 3072), (1, 3072), 0), out=buf1374)
        del arg822_1
        buf1376 = reinterpret_tensor(buf1356, (4096 + s6, 3072), (3072, 1), 0); del buf1356  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1373, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg824_1, (3072, 3072), (1, 3072), 0), out=buf1376)
        del arg824_1
        buf1378 = reinterpret_tensor(buf1355, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1355  # reuse
        buf1381 = reinterpret_tensor(buf1378, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1378  # reuse
        buf1379 = reinterpret_tensor(buf1354, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1354  # reuse
        buf1382 = reinterpret_tensor(buf1379, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1379  # reuse
        # Topologically Sorted Source Nodes: [pow_106, variance_102, pow_107, variance_103, stack_64, stack_65, out_130], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1381, buf1382, buf1374, arg823_1, buf1376, arg825_1, arg828_1, buf41, buf45, arg829_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg823_1
        del arg825_1
        del arg828_1
        del arg829_1
        del buf1374
        buf1380 = buf1376; del buf1376  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg827_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1373, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg826_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1380)
        del arg826_1
        del arg827_1
        # Topologically Sorted Source Nodes: [out_130], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1383 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1381, buf1382, reinterpret_tensor(buf1380, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1384 = buf1383[0]
        del buf1383
        buf1390 = buf1338; del buf1338  # reuse
        # Topologically Sorted Source Nodes: [silu_55], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1389, arg832_1, buf1390, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg832_1
        del buf1389
        buf1391 = buf1365; del buf1365  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1373, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg820_1, (3072, 12288), (1, 3072), 0), out=buf1391)
        del arg820_1
        buf1392 = buf1366; del buf1366  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_487], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1384, buf1391, arg821_1, buf1392, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg821_1
        buf1393 = reinterpret_tensor(buf1384, (4096 + s6, 3072), (3072, 1), 0); del buf1384  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1392, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg830_1, (15360, 3072), (1, 15360), 0), out=buf1393)
        del arg830_1
        buf1394 = reinterpret_tensor(buf1393, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1393  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_482, hidden_states_488, hidden_states_489], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1394, buf1368, buf1364, arg819_1, arg831_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg819_1
        del arg831_1
        buf1398 = buf1368; del buf1368  # reuse
        buf1399 = buf1398; del buf1398  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_491, layer_norm_90, add_546, mul_531, x_52], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1399, buf1394, buf1390, arg833_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1400 = reinterpret_tensor(buf1373, (4096 + s6, 3072), (3072, 1), 0); del buf1373  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1399, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg836_1, (3072, 3072), (1, 3072), 0), out=buf1400)
        del arg836_1
        buf1402 = reinterpret_tensor(buf1382, (4096 + s6, 3072), (3072, 1), 0); del buf1382  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1399, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg838_1, (3072, 3072), (1, 3072), 0), out=buf1402)
        del arg838_1
        buf1404 = reinterpret_tensor(buf1381, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1381  # reuse
        buf1407 = reinterpret_tensor(buf1404, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1404  # reuse
        buf1405 = reinterpret_tensor(buf1380, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1380  # reuse
        buf1408 = reinterpret_tensor(buf1405, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1405  # reuse
        # Topologically Sorted Source Nodes: [pow_108, variance_104, pow_109, variance_105, stack_66, stack_67, out_134], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1407, buf1408, buf1400, arg837_1, buf1402, arg839_1, arg842_1, buf41, buf45, arg843_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg837_1
        del arg839_1
        del arg842_1
        del arg843_1
        del buf1400
        buf1406 = buf1402; del buf1402  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg841_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1399, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg840_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1406)
        del arg840_1
        del arg841_1
        # Topologically Sorted Source Nodes: [out_134], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1409 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1407, buf1408, reinterpret_tensor(buf1406, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1410 = buf1409[0]
        del buf1409
        buf1416 = buf1364; del buf1364  # reuse
        # Topologically Sorted Source Nodes: [silu_56], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1415, arg846_1, buf1416, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg846_1
        del buf1415
        buf1417 = buf1391; del buf1391  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1399, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg834_1, (3072, 12288), (1, 3072), 0), out=buf1417)
        del arg834_1
        buf1418 = buf1392; del buf1392  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_496], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1410, buf1417, arg835_1, buf1418, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg835_1
        buf1419 = reinterpret_tensor(buf1410, (4096 + s6, 3072), (3072, 1), 0); del buf1410  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1418, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg844_1, (15360, 3072), (1, 15360), 0), out=buf1419)
        del arg844_1
        buf1420 = reinterpret_tensor(buf1419, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1419  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_491, hidden_states_497, hidden_states_498], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1420, buf1394, buf1390, arg833_1, arg845_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg833_1
        del arg845_1
        buf1424 = buf1394; del buf1394  # reuse
        buf1425 = buf1424; del buf1424  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_500, layer_norm_91, add_555, mul_541, x_53], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1425, buf1420, buf1416, arg847_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1426 = reinterpret_tensor(buf1399, (4096 + s6, 3072), (3072, 1), 0); del buf1399  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1425, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg850_1, (3072, 3072), (1, 3072), 0), out=buf1426)
        del arg850_1
        buf1428 = reinterpret_tensor(buf1408, (4096 + s6, 3072), (3072, 1), 0); del buf1408  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1425, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg852_1, (3072, 3072), (1, 3072), 0), out=buf1428)
        del arg852_1
        buf1430 = reinterpret_tensor(buf1407, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1407  # reuse
        buf1433 = reinterpret_tensor(buf1430, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1430  # reuse
        buf1431 = reinterpret_tensor(buf1406, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1406  # reuse
        buf1434 = reinterpret_tensor(buf1431, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1431  # reuse
        # Topologically Sorted Source Nodes: [pow_110, variance_106, pow_111, variance_107, stack_68, stack_69, out_138], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1433, buf1434, buf1426, arg851_1, buf1428, arg853_1, arg856_1, buf41, buf45, arg857_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg851_1
        del arg853_1
        del arg856_1
        del arg857_1
        del buf1426
        buf1432 = buf1428; del buf1428  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg855_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1425, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg854_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1432)
        del arg854_1
        del arg855_1
        # Topologically Sorted Source Nodes: [out_138], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1435 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1433, buf1434, reinterpret_tensor(buf1432, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1436 = buf1435[0]
        del buf1435
        buf1442 = buf1390; del buf1390  # reuse
        # Topologically Sorted Source Nodes: [silu_57], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1441, arg860_1, buf1442, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg860_1
        del buf1441
        buf1443 = buf1417; del buf1417  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1425, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg848_1, (3072, 12288), (1, 3072), 0), out=buf1443)
        del arg848_1
        buf1444 = buf1418; del buf1418  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_505], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1436, buf1443, arg849_1, buf1444, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg849_1
        buf1445 = reinterpret_tensor(buf1436, (4096 + s6, 3072), (3072, 1), 0); del buf1436  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1444, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg858_1, (15360, 3072), (1, 15360), 0), out=buf1445)
        del arg858_1
        buf1446 = reinterpret_tensor(buf1445, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1445  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_500, hidden_states_506, hidden_states_507], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1446, buf1420, buf1416, arg847_1, arg859_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg847_1
        del arg859_1
        buf1450 = buf1420; del buf1420  # reuse
        buf1451 = buf1450; del buf1450  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_509, layer_norm_92, add_564, mul_551, x_54], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1451, buf1446, buf1442, arg861_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1452 = reinterpret_tensor(buf1425, (4096 + s6, 3072), (3072, 1), 0); del buf1425  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1451, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg864_1, (3072, 3072), (1, 3072), 0), out=buf1452)
        del arg864_1
        buf1454 = reinterpret_tensor(buf1434, (4096 + s6, 3072), (3072, 1), 0); del buf1434  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1451, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg866_1, (3072, 3072), (1, 3072), 0), out=buf1454)
        del arg866_1
        buf1456 = reinterpret_tensor(buf1433, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1433  # reuse
        buf1459 = reinterpret_tensor(buf1456, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1456  # reuse
        buf1457 = reinterpret_tensor(buf1432, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1432  # reuse
        buf1460 = reinterpret_tensor(buf1457, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1457  # reuse
        # Topologically Sorted Source Nodes: [pow_112, variance_108, pow_113, variance_109, stack_70, stack_71, out_142], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1459, buf1460, buf1452, arg865_1, buf1454, arg867_1, arg870_1, buf41, buf45, arg871_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg865_1
        del arg867_1
        del arg870_1
        del arg871_1
        del buf1452
        buf1458 = buf1454; del buf1454  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg869_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1451, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg868_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1458)
        del arg868_1
        del arg869_1
        # Topologically Sorted Source Nodes: [out_142], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1461 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1459, buf1460, reinterpret_tensor(buf1458, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1462 = buf1461[0]
        del buf1461
        buf1468 = buf1416; del buf1416  # reuse
        # Topologically Sorted Source Nodes: [silu_58], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1467, arg874_1, buf1468, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg874_1
        del buf1467
        buf1469 = buf1443; del buf1443  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1451, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg862_1, (3072, 12288), (1, 3072), 0), out=buf1469)
        del arg862_1
        buf1470 = buf1444; del buf1444  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_514], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1462, buf1469, arg863_1, buf1470, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg863_1
        buf1471 = reinterpret_tensor(buf1462, (4096 + s6, 3072), (3072, 1), 0); del buf1462  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1470, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg872_1, (15360, 3072), (1, 15360), 0), out=buf1471)
        del arg872_1
        buf1472 = reinterpret_tensor(buf1471, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1471  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_509, hidden_states_515, hidden_states_516], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1472, buf1446, buf1442, arg861_1, arg873_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg861_1
        del arg873_1
        buf1476 = buf1446; del buf1446  # reuse
        buf1477 = buf1476; del buf1476  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_518, layer_norm_93, add_573, mul_561, x_55], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1477, buf1472, buf1468, arg875_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1478 = reinterpret_tensor(buf1451, (4096 + s6, 3072), (3072, 1), 0); del buf1451  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1477, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg878_1, (3072, 3072), (1, 3072), 0), out=buf1478)
        del arg878_1
        buf1480 = reinterpret_tensor(buf1460, (4096 + s6, 3072), (3072, 1), 0); del buf1460  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1477, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg880_1, (3072, 3072), (1, 3072), 0), out=buf1480)
        del arg880_1
        buf1482 = reinterpret_tensor(buf1459, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1459  # reuse
        buf1485 = reinterpret_tensor(buf1482, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1482  # reuse
        buf1483 = reinterpret_tensor(buf1458, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1458  # reuse
        buf1486 = reinterpret_tensor(buf1483, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1483  # reuse
        # Topologically Sorted Source Nodes: [pow_114, variance_110, pow_115, variance_111, stack_72, stack_73, out_146], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1485, buf1486, buf1478, arg879_1, buf1480, arg881_1, arg884_1, buf41, buf45, arg885_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg879_1
        del arg881_1
        del arg884_1
        del arg885_1
        del buf1478
        buf1484 = buf1480; del buf1480  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg883_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1477, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg882_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1484)
        del arg882_1
        del arg883_1
        # Topologically Sorted Source Nodes: [out_146], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1487 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1485, buf1486, reinterpret_tensor(buf1484, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1488 = buf1487[0]
        del buf1487
        buf1494 = buf1442; del buf1442  # reuse
        # Topologically Sorted Source Nodes: [silu_59], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1493, arg888_1, buf1494, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg888_1
        del buf1493
        buf1495 = buf1469; del buf1469  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1477, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg876_1, (3072, 12288), (1, 3072), 0), out=buf1495)
        del arg876_1
        buf1496 = buf1470; del buf1470  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_523], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1488, buf1495, arg877_1, buf1496, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg877_1
        buf1497 = reinterpret_tensor(buf1488, (4096 + s6, 3072), (3072, 1), 0); del buf1488  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1496, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg886_1, (15360, 3072), (1, 15360), 0), out=buf1497)
        del arg886_1
        buf1498 = reinterpret_tensor(buf1497, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1497  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_518, hidden_states_524, hidden_states_525], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1498, buf1472, buf1468, arg875_1, arg887_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg875_1
        del arg887_1
        buf1502 = buf1472; del buf1472  # reuse
        buf1503 = buf1502; del buf1502  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_527, layer_norm_94, add_582, mul_571, x_56], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1503, buf1498, buf1494, arg889_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1504 = reinterpret_tensor(buf1477, (4096 + s6, 3072), (3072, 1), 0); del buf1477  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1503, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg892_1, (3072, 3072), (1, 3072), 0), out=buf1504)
        del arg892_1
        buf1506 = reinterpret_tensor(buf1486, (4096 + s6, 3072), (3072, 1), 0); del buf1486  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1503, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg894_1, (3072, 3072), (1, 3072), 0), out=buf1506)
        del arg894_1
        buf1508 = reinterpret_tensor(buf1485, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1485  # reuse
        buf1511 = reinterpret_tensor(buf1508, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1508  # reuse
        buf1509 = reinterpret_tensor(buf1484, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1484  # reuse
        buf1512 = reinterpret_tensor(buf1509, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1509  # reuse
        # Topologically Sorted Source Nodes: [pow_116, variance_112, pow_117, variance_113, stack_74, stack_75, out_150], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1511, buf1512, buf1504, arg893_1, buf1506, arg895_1, arg898_1, buf41, buf45, arg899_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg893_1
        del arg895_1
        del arg898_1
        del arg899_1
        del buf1504
        buf1510 = buf1506; del buf1506  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg897_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1503, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg896_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1510)
        del arg896_1
        del arg897_1
        # Topologically Sorted Source Nodes: [out_150], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1513 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1511, buf1512, reinterpret_tensor(buf1510, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1514 = buf1513[0]
        del buf1513
        buf1520 = buf1468; del buf1468  # reuse
        # Topologically Sorted Source Nodes: [silu_60], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1519, arg902_1, buf1520, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg902_1
        del buf1519
        buf1521 = buf1495; del buf1495  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1503, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg890_1, (3072, 12288), (1, 3072), 0), out=buf1521)
        del arg890_1
        buf1522 = buf1496; del buf1496  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_532], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1514, buf1521, arg891_1, buf1522, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg891_1
        buf1523 = reinterpret_tensor(buf1514, (4096 + s6, 3072), (3072, 1), 0); del buf1514  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1522, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg900_1, (15360, 3072), (1, 15360), 0), out=buf1523)
        del arg900_1
        buf1524 = reinterpret_tensor(buf1523, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1523  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_527, hidden_states_533, hidden_states_534], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1524, buf1498, buf1494, arg889_1, arg901_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg889_1
        del arg901_1
        buf1528 = buf1498; del buf1498  # reuse
        buf1529 = buf1528; del buf1528  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_536, layer_norm_95, add_591, mul_581, x_57], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1529, buf1524, buf1520, arg903_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1530 = reinterpret_tensor(buf1503, (4096 + s6, 3072), (3072, 1), 0); del buf1503  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1529, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg906_1, (3072, 3072), (1, 3072), 0), out=buf1530)
        del arg906_1
        buf1532 = reinterpret_tensor(buf1512, (4096 + s6, 3072), (3072, 1), 0); del buf1512  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1529, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg908_1, (3072, 3072), (1, 3072), 0), out=buf1532)
        del arg908_1
        buf1534 = reinterpret_tensor(buf1511, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1511  # reuse
        buf1537 = reinterpret_tensor(buf1534, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1534  # reuse
        buf1535 = reinterpret_tensor(buf1510, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1510  # reuse
        buf1538 = reinterpret_tensor(buf1535, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1535  # reuse
        # Topologically Sorted Source Nodes: [pow_118, variance_114, pow_119, variance_115, stack_76, stack_77, out_154], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1537, buf1538, buf1530, arg907_1, buf1532, arg909_1, arg912_1, buf41, buf45, arg913_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg907_1
        del arg909_1
        del arg912_1
        del arg913_1
        del buf1530
        buf1536 = buf1532; del buf1532  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg911_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1529, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg910_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1536)
        del arg910_1
        del arg911_1
        # Topologically Sorted Source Nodes: [out_154], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1539 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1537, buf1538, reinterpret_tensor(buf1536, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1540 = buf1539[0]
        del buf1539
        buf1546 = buf1494; del buf1494  # reuse
        # Topologically Sorted Source Nodes: [silu_61], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1545, arg916_1, buf1546, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg916_1
        del buf1545
        buf1547 = buf1521; del buf1521  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1529, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg904_1, (3072, 12288), (1, 3072), 0), out=buf1547)
        del arg904_1
        buf1548 = buf1522; del buf1522  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_541], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1540, buf1547, arg905_1, buf1548, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg905_1
        buf1549 = reinterpret_tensor(buf1540, (4096 + s6, 3072), (3072, 1), 0); del buf1540  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1548, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg914_1, (15360, 3072), (1, 15360), 0), out=buf1549)
        del arg914_1
        buf1550 = reinterpret_tensor(buf1549, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1549  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_536, hidden_states_542, hidden_states_543], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1550, buf1524, buf1520, arg903_1, arg915_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg903_1
        del arg915_1
        buf1554 = buf1524; del buf1524  # reuse
        buf1555 = buf1554; del buf1554  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_545, layer_norm_96, add_600, mul_591, x_58], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1555, buf1550, buf1546, arg917_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1556 = reinterpret_tensor(buf1529, (4096 + s6, 3072), (3072, 1), 0); del buf1529  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1555, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg920_1, (3072, 3072), (1, 3072), 0), out=buf1556)
        del arg920_1
        buf1558 = reinterpret_tensor(buf1538, (4096 + s6, 3072), (3072, 1), 0); del buf1538  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1555, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg922_1, (3072, 3072), (1, 3072), 0), out=buf1558)
        del arg922_1
        buf1560 = reinterpret_tensor(buf1537, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1537  # reuse
        buf1563 = reinterpret_tensor(buf1560, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1560  # reuse
        buf1561 = reinterpret_tensor(buf1536, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1536  # reuse
        buf1564 = reinterpret_tensor(buf1561, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1561  # reuse
        # Topologically Sorted Source Nodes: [pow_120, variance_116, pow_121, variance_117, stack_78, stack_79, out_158], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1563, buf1564, buf1556, arg921_1, buf1558, arg923_1, arg926_1, buf41, buf45, arg927_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg921_1
        del arg923_1
        del arg926_1
        del arg927_1
        del buf1556
        buf1562 = buf1558; del buf1558  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg925_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1555, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg924_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1562)
        del arg924_1
        del arg925_1
        # Topologically Sorted Source Nodes: [out_158], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1565 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1563, buf1564, reinterpret_tensor(buf1562, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1566 = buf1565[0]
        del buf1565
        buf1572 = buf1520; del buf1520  # reuse
        # Topologically Sorted Source Nodes: [silu_62], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1571, arg930_1, buf1572, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg930_1
        buf1573 = buf1547; del buf1547  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1555, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg918_1, (3072, 12288), (1, 3072), 0), out=buf1573)
        del arg918_1
        buf1574 = buf1548; del buf1548  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_550], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1566, buf1573, arg919_1, buf1574, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg919_1
        buf1575 = reinterpret_tensor(buf1566, (4096 + s6, 3072), (3072, 1), 0); del buf1566  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1574, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg928_1, (15360, 3072), (1, 15360), 0), out=buf1575)
        del arg928_1
        buf1576 = reinterpret_tensor(buf1575, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1575  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_545, hidden_states_551, hidden_states_552], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1576, buf1550, buf1546, arg917_1, arg929_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg917_1
        del arg929_1
        buf1580 = buf1550; del buf1550  # reuse
        buf1581 = buf1580; del buf1580  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_554, layer_norm_97, add_609, mul_601, x_59], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1581, buf1576, buf1572, arg931_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1582 = reinterpret_tensor(buf1555, (4096 + s6, 3072), (3072, 1), 0); del buf1555  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1581, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg934_1, (3072, 3072), (1, 3072), 0), out=buf1582)
        del arg934_1
        buf1584 = reinterpret_tensor(buf1564, (4096 + s6, 3072), (3072, 1), 0); del buf1564  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1581, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg936_1, (3072, 3072), (1, 3072), 0), out=buf1584)
        del arg936_1
        buf1586 = reinterpret_tensor(buf1563, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1563  # reuse
        buf1589 = reinterpret_tensor(buf1586, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1586  # reuse
        buf1587 = reinterpret_tensor(buf1562, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1562  # reuse
        buf1590 = reinterpret_tensor(buf1587, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1587  # reuse
        # Topologically Sorted Source Nodes: [pow_122, variance_118, pow_123, variance_119, stack_80, stack_81, out_162], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1589, buf1590, buf1582, arg935_1, buf1584, arg937_1, arg940_1, buf41, buf45, arg941_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg935_1
        del arg937_1
        del arg940_1
        del arg941_1
        del buf1582
        buf1588 = buf1584; del buf1584  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg939_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1581, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg938_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1588)
        del arg938_1
        del arg939_1
        # Topologically Sorted Source Nodes: [out_162], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1591 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1589, buf1590, reinterpret_tensor(buf1588, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1592 = buf1591[0]
        del buf1591
        buf1598 = buf1546; del buf1546  # reuse
        # Topologically Sorted Source Nodes: [silu_63], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1597, arg944_1, buf1598, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg944_1
        buf1599 = buf1573; del buf1573  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1581, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg932_1, (3072, 12288), (1, 3072), 0), out=buf1599)
        del arg932_1
        buf1600 = buf1574; del buf1574  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_559], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1592, buf1599, arg933_1, buf1600, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg933_1
        buf1601 = reinterpret_tensor(buf1592, (4096 + s6, 3072), (3072, 1), 0); del buf1592  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1600, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg942_1, (15360, 3072), (1, 15360), 0), out=buf1601)
        del arg942_1
        buf1602 = reinterpret_tensor(buf1601, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1601  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_554, hidden_states_560, hidden_states_561], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1602, buf1576, buf1572, arg931_1, arg943_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg931_1
        del arg943_1
        buf1606 = buf1576; del buf1576  # reuse
        buf1607 = buf1606; del buf1606  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_563, layer_norm_98, add_618, mul_611, x_60], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1607, buf1602, buf1598, arg945_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1608 = reinterpret_tensor(buf1581, (4096 + s6, 3072), (3072, 1), 0); del buf1581  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1607, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg948_1, (3072, 3072), (1, 3072), 0), out=buf1608)
        del arg948_1
        buf1610 = reinterpret_tensor(buf1590, (4096 + s6, 3072), (3072, 1), 0); del buf1590  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1607, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg950_1, (3072, 3072), (1, 3072), 0), out=buf1610)
        del arg950_1
        buf1612 = reinterpret_tensor(buf1589, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1589  # reuse
        buf1615 = reinterpret_tensor(buf1612, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1612  # reuse
        buf1613 = reinterpret_tensor(buf1588, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1588  # reuse
        buf1616 = reinterpret_tensor(buf1613, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1613  # reuse
        # Topologically Sorted Source Nodes: [pow_124, variance_120, pow_125, variance_121, stack_82, stack_83, out_166], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1615, buf1616, buf1608, arg949_1, buf1610, arg951_1, arg954_1, buf41, buf45, arg955_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg949_1
        del arg951_1
        del arg954_1
        del arg955_1
        del buf1608
        buf1614 = buf1610; del buf1610  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg953_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1607, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg952_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1614)
        del arg952_1
        del arg953_1
        # Topologically Sorted Source Nodes: [out_166], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1617 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1615, buf1616, reinterpret_tensor(buf1614, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1618 = buf1617[0]
        del buf1617
        buf1624 = buf1572; del buf1572  # reuse
        # Topologically Sorted Source Nodes: [silu_64], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1623, arg958_1, buf1624, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg958_1
        buf1625 = buf1599; del buf1599  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1607, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg946_1, (3072, 12288), (1, 3072), 0), out=buf1625)
        del arg946_1
        buf1626 = buf1600; del buf1600  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_568], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1618, buf1625, arg947_1, buf1626, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg947_1
        buf1627 = reinterpret_tensor(buf1618, (4096 + s6, 3072), (3072, 1), 0); del buf1618  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1626, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg956_1, (15360, 3072), (1, 15360), 0), out=buf1627)
        del arg956_1
        buf1628 = reinterpret_tensor(buf1627, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1627  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_563, hidden_states_569, hidden_states_570], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1628, buf1602, buf1598, arg945_1, arg957_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg945_1
        del arg957_1
        buf1632 = buf1602; del buf1602  # reuse
        buf1633 = buf1632; del buf1632  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_572, layer_norm_99, add_627, mul_621, x_61], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1633, buf1628, buf1624, arg959_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1634 = reinterpret_tensor(buf1607, (4096 + s6, 3072), (3072, 1), 0); del buf1607  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1633, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg962_1, (3072, 3072), (1, 3072), 0), out=buf1634)
        del arg962_1
        buf1636 = reinterpret_tensor(buf1616, (4096 + s6, 3072), (3072, 1), 0); del buf1616  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1633, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg964_1, (3072, 3072), (1, 3072), 0), out=buf1636)
        del arg964_1
        buf1638 = reinterpret_tensor(buf1615, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1615  # reuse
        buf1641 = reinterpret_tensor(buf1638, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1638  # reuse
        buf1639 = reinterpret_tensor(buf1614, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1614  # reuse
        buf1642 = reinterpret_tensor(buf1639, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1639  # reuse
        # Topologically Sorted Source Nodes: [pow_126, variance_122, pow_127, variance_123, stack_84, stack_85, out_170], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1641, buf1642, buf1634, arg963_1, buf1636, arg965_1, arg968_1, buf41, buf45, arg969_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg963_1
        del arg965_1
        del arg968_1
        del arg969_1
        del buf1634
        buf1640 = buf1636; del buf1636  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg967_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1633, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg966_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1640)
        del arg966_1
        del arg967_1
        # Topologically Sorted Source Nodes: [out_170], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1643 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1641, buf1642, reinterpret_tensor(buf1640, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1644 = buf1643[0]
        del buf1643
        buf1650 = buf1598; del buf1598  # reuse
        # Topologically Sorted Source Nodes: [silu_65], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1649, arg972_1, buf1650, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg972_1
        buf1651 = buf1625; del buf1625  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1633, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg960_1, (3072, 12288), (1, 3072), 0), out=buf1651)
        del arg960_1
        buf1652 = buf1626; del buf1626  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_577], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1644, buf1651, arg961_1, buf1652, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg961_1
        buf1653 = reinterpret_tensor(buf1644, (4096 + s6, 3072), (3072, 1), 0); del buf1644  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1652, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg970_1, (15360, 3072), (1, 15360), 0), out=buf1653)
        del arg970_1
        buf1654 = reinterpret_tensor(buf1653, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1653  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_572, hidden_states_578, hidden_states_579], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1654, buf1628, buf1624, arg959_1, arg971_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg959_1
        del arg971_1
        buf1658 = buf1628; del buf1628  # reuse
        buf1659 = buf1658; del buf1658  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_581, layer_norm_100, add_636, mul_631, x_62], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1659, buf1654, buf1650, arg973_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1660 = reinterpret_tensor(buf1633, (4096 + s6, 3072), (3072, 1), 0); del buf1633  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1659, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg976_1, (3072, 3072), (1, 3072), 0), out=buf1660)
        del arg976_1
        buf1662 = reinterpret_tensor(buf1642, (4096 + s6, 3072), (3072, 1), 0); del buf1642  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1659, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg978_1, (3072, 3072), (1, 3072), 0), out=buf1662)
        del arg978_1
        buf1664 = reinterpret_tensor(buf1641, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1641  # reuse
        buf1667 = reinterpret_tensor(buf1664, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1664  # reuse
        buf1665 = reinterpret_tensor(buf1640, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1640  # reuse
        buf1668 = reinterpret_tensor(buf1665, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1665  # reuse
        # Topologically Sorted Source Nodes: [pow_128, variance_124, pow_129, variance_125, stack_86, stack_87, out_174], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1667, buf1668, buf1660, arg977_1, buf1662, arg979_1, arg982_1, buf41, buf45, arg983_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg977_1
        del arg979_1
        del arg982_1
        del arg983_1
        del buf1660
        buf1666 = buf1662; del buf1662  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg981_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1659, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg980_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1666)
        del arg980_1
        del arg981_1
        # Topologically Sorted Source Nodes: [out_174], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1669 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1667, buf1668, reinterpret_tensor(buf1666, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1670 = buf1669[0]
        del buf1669
        buf1675 = buf1649; del buf1649  # reuse
        buf1701 = buf1623; del buf1623  # reuse
        buf1727 = buf1597; del buf1597  # reuse
        buf1753 = buf1571; del buf1571  # reuse
        # Topologically Sorted Source Nodes: [silu_66, silu_67, silu_68, silu_69], Original ATen: [aten.silu]
        triton_poi_fused_silu_37.run(buf13, buf1675, buf1701, buf1727, buf1753, 3072, grid=grid(3072), stream=stream0)
        buf1676 = buf1624; del buf1624  # reuse
        # Topologically Sorted Source Nodes: [silu_66], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1675, arg986_1, buf1676, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg986_1
        buf1677 = buf1651; del buf1651  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1659, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg974_1, (3072, 12288), (1, 3072), 0), out=buf1677)
        del arg974_1
        buf1678 = buf1652; del buf1652  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_586], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1670, buf1677, arg975_1, buf1678, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg975_1
        buf1679 = reinterpret_tensor(buf1670, (4096 + s6, 3072), (3072, 1), 0); del buf1670  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1678, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg984_1, (15360, 3072), (1, 15360), 0), out=buf1679)
        del arg984_1
        buf1680 = reinterpret_tensor(buf1679, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1679  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_581, hidden_states_587, hidden_states_588], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1680, buf1654, buf1650, arg973_1, arg985_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg973_1
        del arg985_1
        buf1684 = buf1654; del buf1654  # reuse
        buf1685 = buf1684; del buf1684  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_590, layer_norm_101, add_645, mul_641, x_63], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1685, buf1680, buf1676, arg987_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1686 = reinterpret_tensor(buf1659, (4096 + s6, 3072), (3072, 1), 0); del buf1659  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1685, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg990_1, (3072, 3072), (1, 3072), 0), out=buf1686)
        del arg990_1
        buf1688 = reinterpret_tensor(buf1668, (4096 + s6, 3072), (3072, 1), 0); del buf1668  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1685, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg992_1, (3072, 3072), (1, 3072), 0), out=buf1688)
        del arg992_1
        buf1690 = reinterpret_tensor(buf1667, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1667  # reuse
        buf1693 = reinterpret_tensor(buf1690, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1690  # reuse
        buf1691 = reinterpret_tensor(buf1666, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1666  # reuse
        buf1694 = reinterpret_tensor(buf1691, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1691  # reuse
        # Topologically Sorted Source Nodes: [pow_130, variance_126, pow_131, variance_127, stack_88, stack_89, out_178], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1693, buf1694, buf1686, arg991_1, buf1688, arg993_1, arg996_1, buf41, buf45, arg997_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg991_1
        del arg993_1
        del arg996_1
        del arg997_1
        del buf1686
        buf1692 = buf1688; del buf1688  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg995_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1685, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg994_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1692)
        del arg994_1
        del arg995_1
        # Topologically Sorted Source Nodes: [out_178], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1695 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1693, buf1694, reinterpret_tensor(buf1692, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1696 = buf1695[0]
        del buf1695
        buf1702 = buf1650; del buf1650  # reuse
        # Topologically Sorted Source Nodes: [silu_67], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1701, arg1000_1, buf1702, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1000_1
        buf1703 = buf1677; del buf1677  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1685, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg988_1, (3072, 12288), (1, 3072), 0), out=buf1703)
        del arg988_1
        buf1704 = buf1678; del buf1678  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_595], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1696, buf1703, arg989_1, buf1704, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg989_1
        buf1705 = reinterpret_tensor(buf1696, (4096 + s6, 3072), (3072, 1), 0); del buf1696  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1704, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg998_1, (15360, 3072), (1, 15360), 0), out=buf1705)
        del arg998_1
        buf1706 = reinterpret_tensor(buf1705, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1705  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_590, hidden_states_596, hidden_states_597], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1706, buf1680, buf1676, arg987_1, arg999_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg987_1
        del arg999_1
        buf1710 = buf1680; del buf1680  # reuse
        buf1711 = buf1710; del buf1710  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_599, layer_norm_102, add_654, mul_651, x_64], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1711, buf1706, buf1702, arg1001_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1712 = reinterpret_tensor(buf1685, (4096 + s6, 3072), (3072, 1), 0); del buf1685  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1711, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1004_1, (3072, 3072), (1, 3072), 0), out=buf1712)
        del arg1004_1
        buf1714 = reinterpret_tensor(buf1694, (4096 + s6, 3072), (3072, 1), 0); del buf1694  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1711, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1006_1, (3072, 3072), (1, 3072), 0), out=buf1714)
        del arg1006_1
        buf1716 = reinterpret_tensor(buf1693, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1693  # reuse
        buf1719 = reinterpret_tensor(buf1716, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1716  # reuse
        buf1717 = reinterpret_tensor(buf1692, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1692  # reuse
        buf1720 = reinterpret_tensor(buf1717, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1717  # reuse
        # Topologically Sorted Source Nodes: [pow_132, variance_128, pow_133, variance_129, stack_90, stack_91, out_182], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1719, buf1720, buf1712, arg1005_1, buf1714, arg1007_1, arg1010_1, buf41, buf45, arg1011_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1005_1
        del arg1007_1
        del arg1010_1
        del arg1011_1
        del buf1712
        buf1718 = buf1714; del buf1714  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1009_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1711, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1008_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1718)
        del arg1008_1
        del arg1009_1
        # Topologically Sorted Source Nodes: [out_182], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1721 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1719, buf1720, reinterpret_tensor(buf1718, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1722 = buf1721[0]
        del buf1721
        buf1728 = buf1676; del buf1676  # reuse
        # Topologically Sorted Source Nodes: [silu_68], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1727, arg1014_1, buf1728, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1014_1
        buf1729 = buf1703; del buf1703  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1711, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1002_1, (3072, 12288), (1, 3072), 0), out=buf1729)
        del arg1002_1
        buf1730 = buf1704; del buf1704  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_604], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1722, buf1729, arg1003_1, buf1730, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1003_1
        buf1731 = reinterpret_tensor(buf1722, (4096 + s6, 3072), (3072, 1), 0); del buf1722  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1730, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1012_1, (15360, 3072), (1, 15360), 0), out=buf1731)
        del arg1012_1
        buf1732 = reinterpret_tensor(buf1731, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1731  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_599, hidden_states_605, hidden_states_606], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1732, buf1706, buf1702, arg1001_1, arg1013_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1001_1
        del arg1013_1
        buf1736 = buf1706; del buf1706  # reuse
        buf1737 = buf1736; del buf1736  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_608, layer_norm_103, add_663, mul_661, x_65], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1737, buf1732, buf1728, arg1015_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1738 = reinterpret_tensor(buf1711, (4096 + s6, 3072), (3072, 1), 0); del buf1711  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1737, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1018_1, (3072, 3072), (1, 3072), 0), out=buf1738)
        del arg1018_1
        buf1740 = reinterpret_tensor(buf1720, (4096 + s6, 3072), (3072, 1), 0); del buf1720  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1737, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1020_1, (3072, 3072), (1, 3072), 0), out=buf1740)
        del arg1020_1
        buf1742 = reinterpret_tensor(buf1719, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1719  # reuse
        buf1745 = reinterpret_tensor(buf1742, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1742  # reuse
        buf1743 = reinterpret_tensor(buf1718, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1718  # reuse
        buf1746 = reinterpret_tensor(buf1743, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1743  # reuse
        # Topologically Sorted Source Nodes: [pow_134, variance_130, pow_135, variance_131, stack_92, stack_93, out_186], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1745, buf1746, buf1738, arg1019_1, buf1740, arg1021_1, arg1024_1, buf41, buf45, arg1025_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1019_1
        del arg1021_1
        del arg1024_1
        del arg1025_1
        del buf1738
        buf1744 = buf1740; del buf1740  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1023_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1737, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1022_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1744)
        del arg1022_1
        del arg1023_1
        # Topologically Sorted Source Nodes: [out_186], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1747 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1745, buf1746, reinterpret_tensor(buf1744, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1748 = buf1747[0]
        del buf1747
        buf1754 = buf1702; del buf1702  # reuse
        # Topologically Sorted Source Nodes: [silu_69], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1753, arg1028_1, buf1754, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1028_1
        buf1755 = buf1729; del buf1729  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1737, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1016_1, (3072, 12288), (1, 3072), 0), out=buf1755)
        del arg1016_1
        buf1756 = buf1730; del buf1730  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_613], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1748, buf1755, arg1017_1, buf1756, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1017_1
        buf1757 = reinterpret_tensor(buf1748, (4096 + s6, 3072), (3072, 1), 0); del buf1748  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1756, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1026_1, (15360, 3072), (1, 15360), 0), out=buf1757)
        del arg1026_1
        buf1758 = reinterpret_tensor(buf1757, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1757  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_608, hidden_states_614, hidden_states_615], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1758, buf1732, buf1728, arg1015_1, arg1027_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1015_1
        del arg1027_1
        buf1762 = buf1732; del buf1732  # reuse
        buf1763 = buf1762; del buf1762  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_617, layer_norm_104, add_672, mul_671, x_66], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1763, buf1758, buf1754, arg1029_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1764 = reinterpret_tensor(buf1737, (4096 + s6, 3072), (3072, 1), 0); del buf1737  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1763, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1032_1, (3072, 3072), (1, 3072), 0), out=buf1764)
        del arg1032_1
        buf1766 = reinterpret_tensor(buf1746, (4096 + s6, 3072), (3072, 1), 0); del buf1746  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1763, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1034_1, (3072, 3072), (1, 3072), 0), out=buf1766)
        del arg1034_1
        buf1768 = reinterpret_tensor(buf1745, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1745  # reuse
        buf1771 = reinterpret_tensor(buf1768, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1768  # reuse
        buf1769 = reinterpret_tensor(buf1744, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1744  # reuse
        buf1772 = reinterpret_tensor(buf1769, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1769  # reuse
        # Topologically Sorted Source Nodes: [pow_136, variance_132, pow_137, variance_133, stack_94, stack_95, out_190], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1771, buf1772, buf1764, arg1033_1, buf1766, arg1035_1, arg1038_1, buf41, buf45, arg1039_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1033_1
        del arg1035_1
        del arg1038_1
        del arg1039_1
        del buf1764
        buf1770 = buf1766; del buf1766  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1037_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1763, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1036_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1770)
        del arg1036_1
        del arg1037_1
        # Topologically Sorted Source Nodes: [out_190], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1773 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1771, buf1772, reinterpret_tensor(buf1770, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1774 = buf1773[0]
        del buf1773
        buf1779 = buf1753; del buf1753  # reuse
        buf1805 = buf1727; del buf1727  # reuse
        buf1831 = buf1701; del buf1701  # reuse
        buf1857 = buf1675; del buf1675  # reuse
        # Topologically Sorted Source Nodes: [silu_70, silu_71, silu_72, silu_73], Original ATen: [aten.silu]
        triton_poi_fused_silu_37.run(buf13, buf1779, buf1805, buf1831, buf1857, 3072, grid=grid(3072), stream=stream0)
        buf1780 = buf1728; del buf1728  # reuse
        # Topologically Sorted Source Nodes: [silu_70], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1779, arg1042_1, buf1780, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1042_1
        buf1781 = buf1755; del buf1755  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1763, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1030_1, (3072, 12288), (1, 3072), 0), out=buf1781)
        del arg1030_1
        buf1782 = buf1756; del buf1756  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_622], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1774, buf1781, arg1031_1, buf1782, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1031_1
        buf1783 = reinterpret_tensor(buf1774, (4096 + s6, 3072), (3072, 1), 0); del buf1774  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1782, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1040_1, (15360, 3072), (1, 15360), 0), out=buf1783)
        del arg1040_1
        buf1784 = reinterpret_tensor(buf1783, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1783  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_617, hidden_states_623, hidden_states_624], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1784, buf1758, buf1754, arg1029_1, arg1041_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1029_1
        del arg1041_1
        buf1788 = buf1758; del buf1758  # reuse
        buf1789 = buf1788; del buf1788  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_626, layer_norm_105, add_681, mul_681, x_67], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1789, buf1784, buf1780, arg1043_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1790 = reinterpret_tensor(buf1763, (4096 + s6, 3072), (3072, 1), 0); del buf1763  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1789, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1046_1, (3072, 3072), (1, 3072), 0), out=buf1790)
        del arg1046_1
        buf1792 = reinterpret_tensor(buf1772, (4096 + s6, 3072), (3072, 1), 0); del buf1772  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1789, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1048_1, (3072, 3072), (1, 3072), 0), out=buf1792)
        del arg1048_1
        buf1794 = reinterpret_tensor(buf1771, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1771  # reuse
        buf1797 = reinterpret_tensor(buf1794, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1794  # reuse
        buf1795 = reinterpret_tensor(buf1770, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1770  # reuse
        buf1798 = reinterpret_tensor(buf1795, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1795  # reuse
        # Topologically Sorted Source Nodes: [pow_138, variance_134, pow_139, variance_135, stack_96, stack_97, out_194], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1797, buf1798, buf1790, arg1047_1, buf1792, arg1049_1, arg1052_1, buf41, buf45, arg1053_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1047_1
        del arg1049_1
        del arg1052_1
        del arg1053_1
        del buf1790
        buf1796 = buf1792; del buf1792  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1051_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1789, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1050_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1796)
        del arg1050_1
        del arg1051_1
        # Topologically Sorted Source Nodes: [out_194], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1799 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1797, buf1798, reinterpret_tensor(buf1796, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1800 = buf1799[0]
        del buf1799
        buf1806 = buf1754; del buf1754  # reuse
        # Topologically Sorted Source Nodes: [silu_71], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1805, arg1056_1, buf1806, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1056_1
        buf1807 = buf1781; del buf1781  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1789, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1044_1, (3072, 12288), (1, 3072), 0), out=buf1807)
        del arg1044_1
        buf1808 = buf1782; del buf1782  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_631], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1800, buf1807, arg1045_1, buf1808, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1045_1
        buf1809 = reinterpret_tensor(buf1800, (4096 + s6, 3072), (3072, 1), 0); del buf1800  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1808, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1054_1, (15360, 3072), (1, 15360), 0), out=buf1809)
        del arg1054_1
        buf1810 = reinterpret_tensor(buf1809, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1809  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_626, hidden_states_632, hidden_states_633], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1810, buf1784, buf1780, arg1043_1, arg1055_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1043_1
        del arg1055_1
        buf1814 = buf1784; del buf1784  # reuse
        buf1815 = buf1814; del buf1814  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_635, layer_norm_106, add_690, mul_691, x_68], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1815, buf1810, buf1806, arg1057_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1816 = reinterpret_tensor(buf1789, (4096 + s6, 3072), (3072, 1), 0); del buf1789  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1815, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1060_1, (3072, 3072), (1, 3072), 0), out=buf1816)
        del arg1060_1
        buf1818 = reinterpret_tensor(buf1798, (4096 + s6, 3072), (3072, 1), 0); del buf1798  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1815, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1062_1, (3072, 3072), (1, 3072), 0), out=buf1818)
        del arg1062_1
        buf1820 = reinterpret_tensor(buf1797, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1797  # reuse
        buf1823 = reinterpret_tensor(buf1820, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1820  # reuse
        buf1821 = reinterpret_tensor(buf1796, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1796  # reuse
        buf1824 = reinterpret_tensor(buf1821, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1821  # reuse
        # Topologically Sorted Source Nodes: [pow_140, variance_136, pow_141, variance_137, stack_98, stack_99, out_198], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1823, buf1824, buf1816, arg1061_1, buf1818, arg1063_1, arg1066_1, buf41, buf45, arg1067_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1061_1
        del arg1063_1
        del arg1066_1
        del arg1067_1
        del buf1816
        buf1822 = buf1818; del buf1818  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1065_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1815, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1064_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1822)
        del arg1064_1
        del arg1065_1
        # Topologically Sorted Source Nodes: [out_198], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1825 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1823, buf1824, reinterpret_tensor(buf1822, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1826 = buf1825[0]
        del buf1825
        buf1832 = buf1780; del buf1780  # reuse
        # Topologically Sorted Source Nodes: [silu_72], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1831, arg1070_1, buf1832, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1070_1
        buf1833 = buf1807; del buf1807  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1815, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1058_1, (3072, 12288), (1, 3072), 0), out=buf1833)
        del arg1058_1
        buf1834 = buf1808; del buf1808  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_640], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1826, buf1833, arg1059_1, buf1834, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1059_1
        buf1835 = reinterpret_tensor(buf1826, (4096 + s6, 3072), (3072, 1), 0); del buf1826  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1834, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1068_1, (15360, 3072), (1, 15360), 0), out=buf1835)
        del arg1068_1
        buf1836 = reinterpret_tensor(buf1835, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1835  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_635, hidden_states_641, hidden_states_642], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1836, buf1810, buf1806, arg1057_1, arg1069_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1057_1
        del arg1069_1
        buf1840 = buf1810; del buf1810  # reuse
        buf1841 = buf1840; del buf1840  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_644, layer_norm_107, add_699, mul_701, x_69], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1841, buf1836, buf1832, arg1071_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1842 = reinterpret_tensor(buf1815, (4096 + s6, 3072), (3072, 1), 0); del buf1815  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1841, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1074_1, (3072, 3072), (1, 3072), 0), out=buf1842)
        del arg1074_1
        buf1844 = reinterpret_tensor(buf1824, (4096 + s6, 3072), (3072, 1), 0); del buf1824  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1841, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1076_1, (3072, 3072), (1, 3072), 0), out=buf1844)
        del arg1076_1
        buf1846 = reinterpret_tensor(buf1823, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1823  # reuse
        buf1849 = reinterpret_tensor(buf1846, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1846  # reuse
        buf1847 = reinterpret_tensor(buf1822, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1822  # reuse
        buf1850 = reinterpret_tensor(buf1847, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1847  # reuse
        # Topologically Sorted Source Nodes: [pow_142, variance_138, pow_143, variance_139, stack_100, stack_101, out_202], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1849, buf1850, buf1842, arg1075_1, buf1844, arg1077_1, arg1080_1, buf41, buf45, arg1081_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1075_1
        del arg1077_1
        del arg1080_1
        del arg1081_1
        del buf1842
        buf1848 = buf1844; del buf1844  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1079_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1841, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1078_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1848)
        del arg1078_1
        del arg1079_1
        # Topologically Sorted Source Nodes: [out_202], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1851 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1849, buf1850, reinterpret_tensor(buf1848, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1852 = buf1851[0]
        del buf1851
        buf1858 = buf1806; del buf1806  # reuse
        # Topologically Sorted Source Nodes: [silu_73], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1857, arg1084_1, buf1858, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1084_1
        buf1859 = buf1833; del buf1833  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1841, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1072_1, (3072, 12288), (1, 3072), 0), out=buf1859)
        del arg1072_1
        buf1860 = buf1834; del buf1834  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_649], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1852, buf1859, arg1073_1, buf1860, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1073_1
        buf1861 = reinterpret_tensor(buf1852, (4096 + s6, 3072), (3072, 1), 0); del buf1852  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1860, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1082_1, (15360, 3072), (1, 15360), 0), out=buf1861)
        del arg1082_1
        buf1862 = reinterpret_tensor(buf1861, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1861  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_644, hidden_states_650, hidden_states_651], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1862, buf1836, buf1832, arg1071_1, arg1083_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1071_1
        del arg1083_1
        buf1866 = buf1836; del buf1836  # reuse
        buf1867 = buf1866; del buf1866  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_653, layer_norm_108, add_708, mul_711, x_70], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1867, buf1862, buf1858, arg1085_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1868 = reinterpret_tensor(buf1841, (4096 + s6, 3072), (3072, 1), 0); del buf1841  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1867, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1088_1, (3072, 3072), (1, 3072), 0), out=buf1868)
        del arg1088_1
        buf1870 = reinterpret_tensor(buf1850, (4096 + s6, 3072), (3072, 1), 0); del buf1850  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1867, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1090_1, (3072, 3072), (1, 3072), 0), out=buf1870)
        del arg1090_1
        buf1872 = reinterpret_tensor(buf1849, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1849  # reuse
        buf1875 = reinterpret_tensor(buf1872, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1872  # reuse
        buf1873 = reinterpret_tensor(buf1848, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1848  # reuse
        buf1876 = reinterpret_tensor(buf1873, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1873  # reuse
        # Topologically Sorted Source Nodes: [pow_144, variance_140, pow_145, variance_141, stack_102, stack_103, out_206], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1875, buf1876, buf1868, arg1089_1, buf1870, arg1091_1, arg1094_1, buf41, buf45, arg1095_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1089_1
        del arg1091_1
        del arg1094_1
        del arg1095_1
        del buf1868
        buf1874 = buf1870; del buf1870  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1093_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1867, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1092_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1874)
        del arg1092_1
        del arg1093_1
        # Topologically Sorted Source Nodes: [out_206], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1877 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1875, buf1876, reinterpret_tensor(buf1874, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1878 = buf1877[0]
        del buf1877
        buf1883 = buf1857; del buf1857  # reuse
        buf1909 = buf1831; del buf1831  # reuse
        buf1935 = buf1805; del buf1805  # reuse
        buf1961 = buf1779; del buf1779  # reuse
        # Topologically Sorted Source Nodes: [silu_74, silu_75, silu_76, silu_77], Original ATen: [aten.silu]
        triton_poi_fused_silu_37.run(buf13, buf1883, buf1909, buf1935, buf1961, 3072, grid=grid(3072), stream=stream0)
        buf1884 = buf1832; del buf1832  # reuse
        # Topologically Sorted Source Nodes: [silu_74], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1883, arg1098_1, buf1884, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1098_1
        del buf1883
        buf1885 = buf1859; del buf1859  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1867, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1086_1, (3072, 12288), (1, 3072), 0), out=buf1885)
        del arg1086_1
        buf1886 = buf1860; del buf1860  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_658], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1878, buf1885, arg1087_1, buf1886, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1087_1
        buf1887 = reinterpret_tensor(buf1878, (4096 + s6, 3072), (3072, 1), 0); del buf1878  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1886, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1096_1, (15360, 3072), (1, 15360), 0), out=buf1887)
        del arg1096_1
        buf1888 = reinterpret_tensor(buf1887, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1887  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_653, hidden_states_659, hidden_states_660], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1888, buf1862, buf1858, arg1085_1, arg1097_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1085_1
        del arg1097_1
        buf1892 = buf1862; del buf1862  # reuse
        buf1893 = buf1892; del buf1892  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_662, layer_norm_109, add_717, mul_721, x_71], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1893, buf1888, buf1884, arg1099_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1894 = reinterpret_tensor(buf1867, (4096 + s6, 3072), (3072, 1), 0); del buf1867  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1893, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1102_1, (3072, 3072), (1, 3072), 0), out=buf1894)
        del arg1102_1
        buf1896 = reinterpret_tensor(buf1876, (4096 + s6, 3072), (3072, 1), 0); del buf1876  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1893, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1104_1, (3072, 3072), (1, 3072), 0), out=buf1896)
        del arg1104_1
        buf1898 = reinterpret_tensor(buf1875, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1875  # reuse
        buf1901 = reinterpret_tensor(buf1898, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1898  # reuse
        buf1899 = reinterpret_tensor(buf1874, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1874  # reuse
        buf1902 = reinterpret_tensor(buf1899, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1899  # reuse
        # Topologically Sorted Source Nodes: [pow_146, variance_142, pow_147, variance_143, stack_104, stack_105, out_210], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1901, buf1902, buf1894, arg1103_1, buf1896, arg1105_1, arg1108_1, buf41, buf45, arg1109_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1103_1
        del arg1105_1
        del arg1108_1
        del arg1109_1
        del buf1894
        buf1900 = buf1896; del buf1896  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1107_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1893, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1106_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1900)
        del arg1106_1
        del arg1107_1
        # Topologically Sorted Source Nodes: [out_210], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1903 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1901, buf1902, reinterpret_tensor(buf1900, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1904 = buf1903[0]
        del buf1903
        buf1910 = buf1858; del buf1858  # reuse
        # Topologically Sorted Source Nodes: [silu_75], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1909, arg1112_1, buf1910, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1112_1
        del buf1909
        buf1911 = buf1885; del buf1885  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1893, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1100_1, (3072, 12288), (1, 3072), 0), out=buf1911)
        del arg1100_1
        buf1912 = buf1886; del buf1886  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_667], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1904, buf1911, arg1101_1, buf1912, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1101_1
        buf1913 = reinterpret_tensor(buf1904, (4096 + s6, 3072), (3072, 1), 0); del buf1904  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1912, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1110_1, (15360, 3072), (1, 15360), 0), out=buf1913)
        del arg1110_1
        buf1914 = reinterpret_tensor(buf1913, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1913  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_662, hidden_states_668, hidden_states_669], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1914, buf1888, buf1884, arg1099_1, arg1111_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1099_1
        del arg1111_1
        buf1918 = buf1888; del buf1888  # reuse
        buf1919 = buf1918; del buf1918  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_671, layer_norm_110, add_726, mul_731, x_72], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1919, buf1914, buf1910, arg1113_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1920 = reinterpret_tensor(buf1893, (4096 + s6, 3072), (3072, 1), 0); del buf1893  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1919, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1116_1, (3072, 3072), (1, 3072), 0), out=buf1920)
        del arg1116_1
        buf1922 = reinterpret_tensor(buf1902, (4096 + s6, 3072), (3072, 1), 0); del buf1902  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1919, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1118_1, (3072, 3072), (1, 3072), 0), out=buf1922)
        del arg1118_1
        buf1924 = reinterpret_tensor(buf1901, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1901  # reuse
        buf1927 = reinterpret_tensor(buf1924, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1924  # reuse
        buf1925 = reinterpret_tensor(buf1900, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1900  # reuse
        buf1928 = reinterpret_tensor(buf1925, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1925  # reuse
        # Topologically Sorted Source Nodes: [pow_148, variance_144, pow_149, variance_145, stack_106, stack_107, out_214], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1927, buf1928, buf1920, arg1117_1, buf1922, arg1119_1, arg1122_1, buf41, buf45, arg1123_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1117_1
        del arg1119_1
        del arg1122_1
        del arg1123_1
        del buf1920
        buf1926 = buf1922; del buf1922  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1121_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1919, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1120_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1926)
        del arg1120_1
        del arg1121_1
        # Topologically Sorted Source Nodes: [out_214], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1929 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1927, buf1928, reinterpret_tensor(buf1926, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1930 = buf1929[0]
        del buf1929
        buf1936 = buf1884; del buf1884  # reuse
        # Topologically Sorted Source Nodes: [silu_76], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1935, arg1126_1, buf1936, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1126_1
        buf1937 = buf1911; del buf1911  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1919, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1114_1, (3072, 12288), (1, 3072), 0), out=buf1937)
        del arg1114_1
        buf1938 = buf1912; del buf1912  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_676], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1930, buf1937, arg1115_1, buf1938, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1115_1
        buf1939 = reinterpret_tensor(buf1930, (4096 + s6, 3072), (3072, 1), 0); del buf1930  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1938, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1124_1, (15360, 3072), (1, 15360), 0), out=buf1939)
        del arg1124_1
        buf1940 = reinterpret_tensor(buf1939, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1939  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_671, hidden_states_677, hidden_states_678], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1940, buf1914, buf1910, arg1113_1, arg1125_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1113_1
        del arg1125_1
        buf1944 = buf1914; del buf1914  # reuse
        buf1945 = buf1944; del buf1944  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_680, layer_norm_111, add_735, mul_741, x_73], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1945, buf1940, buf1936, arg1127_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1946 = reinterpret_tensor(buf1919, (4096 + s6, 3072), (3072, 1), 0); del buf1919  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1945, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1130_1, (3072, 3072), (1, 3072), 0), out=buf1946)
        del arg1130_1
        buf1948 = reinterpret_tensor(buf1928, (4096 + s6, 3072), (3072, 1), 0); del buf1928  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1945, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1132_1, (3072, 3072), (1, 3072), 0), out=buf1948)
        del arg1132_1
        buf1950 = reinterpret_tensor(buf1927, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1927  # reuse
        buf1953 = reinterpret_tensor(buf1950, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1950  # reuse
        buf1951 = reinterpret_tensor(buf1926, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1926  # reuse
        buf1954 = reinterpret_tensor(buf1951, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1951  # reuse
        # Topologically Sorted Source Nodes: [pow_150, variance_146, pow_151, variance_147, stack_108, stack_109, out_218], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1953, buf1954, buf1946, arg1131_1, buf1948, arg1133_1, arg1136_1, buf41, buf45, arg1137_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1131_1
        del arg1133_1
        del arg1136_1
        del arg1137_1
        del buf1946
        buf1952 = buf1948; del buf1948  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1135_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1945, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1134_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1952)
        del arg1134_1
        del arg1135_1
        # Topologically Sorted Source Nodes: [out_218], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1955 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1953, buf1954, reinterpret_tensor(buf1952, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1956 = buf1955[0]
        del buf1955
        buf1962 = buf1910; del buf1910  # reuse
        # Topologically Sorted Source Nodes: [silu_77], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1961, arg1140_1, buf1962, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1140_1
        buf1963 = buf1937; del buf1937  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1945, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1128_1, (3072, 12288), (1, 3072), 0), out=buf1963)
        del arg1128_1
        buf1964 = buf1938; del buf1938  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_685], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1956, buf1963, arg1129_1, buf1964, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1129_1
        buf1965 = reinterpret_tensor(buf1956, (4096 + s6, 3072), (3072, 1), 0); del buf1956  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1964, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1138_1, (15360, 3072), (1, 15360), 0), out=buf1965)
        del arg1138_1
        buf1966 = reinterpret_tensor(buf1965, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1965  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_680, hidden_states_686, hidden_states_687], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1966, buf1940, buf1936, arg1127_1, arg1139_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1127_1
        del arg1139_1
        buf1970 = buf1940; del buf1940  # reuse
        buf1971 = buf1970; del buf1970  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_689, layer_norm_112, add_744, mul_751, x_74], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1971, buf1966, buf1962, arg1141_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1972 = reinterpret_tensor(buf1945, (4096 + s6, 3072), (3072, 1), 0); del buf1945  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1971, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1144_1, (3072, 3072), (1, 3072), 0), out=buf1972)
        del arg1144_1
        buf1974 = reinterpret_tensor(buf1954, (4096 + s6, 3072), (3072, 1), 0); del buf1954  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1971, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1146_1, (3072, 3072), (1, 3072), 0), out=buf1974)
        del arg1146_1
        buf1976 = reinterpret_tensor(buf1953, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1953  # reuse
        buf1979 = reinterpret_tensor(buf1976, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1976  # reuse
        buf1977 = reinterpret_tensor(buf1952, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1952  # reuse
        buf1980 = reinterpret_tensor(buf1977, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf1977  # reuse
        # Topologically Sorted Source Nodes: [pow_152, variance_148, pow_153, variance_149, stack_110, stack_111, out_222], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf1979, buf1980, buf1972, arg1145_1, buf1974, arg1147_1, arg1150_1, buf41, buf45, arg1151_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1145_1
        del arg1147_1
        del arg1150_1
        del arg1151_1
        del buf1972
        buf1978 = buf1974; del buf1974  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1149_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1971, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1148_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf1978)
        del arg1148_1
        del arg1149_1
        # Topologically Sorted Source Nodes: [out_222], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf1981 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf1979, buf1980, reinterpret_tensor(buf1978, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        buf1982 = buf1981[0]
        del buf1981
        buf1987 = buf1961; del buf1961  # reuse
        buf2013 = buf1935; del buf1935  # reuse
        # Topologically Sorted Source Nodes: [silu_78, silu_79], Original ATen: [aten.silu]
        triton_poi_fused_silu_38.run(buf13, buf1987, buf2013, 3072, grid=grid(3072), stream=stream0)
        del buf13
        buf1988 = buf1936; del buf1936  # reuse
        # Topologically Sorted Source Nodes: [silu_78], Original ATen: [aten.silu]
        triton_tem_fused_silu_30.run(buf1987, arg1154_1, buf1988, grid=torch._inductor.kernel.mm_common.mm_grid(1, 9216, meta1), stream=stream0)
        del arg1154_1
        del buf1987
        buf1989 = buf1963; del buf1963  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1971, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1142_1, (3072, 12288), (1, 3072), 0), out=buf1989)
        del arg1142_1
        buf1990 = buf1964; del buf1964  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_694], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf1982, buf1989, arg1143_1, buf1990, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1143_1
        buf1991 = reinterpret_tensor(buf1982, (4096 + s6, 3072), (3072, 1), 0); del buf1982  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1990, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1152_1, (15360, 3072), (1, 15360), 0), out=buf1991)
        del arg1152_1
        buf1992 = reinterpret_tensor(buf1991, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf1991  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_689, hidden_states_695, hidden_states_696], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf1992, buf1966, buf1962, arg1141_1, arg1153_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1141_1
        del arg1153_1
        del buf1962
        buf1996 = buf1966; del buf1966  # reuse
        buf1997 = buf1996; del buf1996  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_698, layer_norm_113, add_753, mul_761, x_75], Original ATen: [aten.cat, aten.native_layer_norm, aten.add, aten.mul]
        triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel = 4096 + s6
        triton_red_fused_add_cat_mul_native_layer_norm_35.run(buf1997, buf1992, buf1988, arg1155_1, s6, triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel, 3072, grid=grid(triton_red_fused_add_cat_mul_native_layer_norm_35_xnumel), stream=stream0)
        buf1998 = reinterpret_tensor(buf1971, (4096 + s6, 3072), (3072, 1), 0); del buf1971  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1997, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1158_1, (3072, 3072), (1, 3072), 0), out=buf1998)
        del arg1158_1
        buf2000 = reinterpret_tensor(buf1980, (4096 + s6, 3072), (3072, 1), 0); del buf1980  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1997, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1160_1, (3072, 3072), (1, 3072), 0), out=buf2000)
        del arg1160_1
        buf2002 = reinterpret_tensor(buf1979, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1979  # reuse
        buf2005 = reinterpret_tensor(buf2002, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf2002  # reuse
        buf2003 = reinterpret_tensor(buf1978, (1, 4096 + s6, 24, 64, 2), (12582912 + (3072*s6), 3072, 128, 2, 1), 0); del buf1978  # reuse
        buf2006 = reinterpret_tensor(buf2003, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0); del buf2003  # reuse
        # Topologically Sorted Source Nodes: [pow_154, variance_150, pow_155, variance_151, stack_112, stack_113, out_226], Original ATen: [aten.pow, aten.mean, aten.stack, aten._scaled_dot_product_flash_attention]
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel = 98304 + (24*s6)
        triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32.run(buf2005, buf2006, buf1998, arg1159_1, buf2000, arg1161_1, arg1164_1, buf41, buf45, arg1165_1, triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel, 128, grid=grid(triton_red_fused__scaled_dot_product_flash_attention_mean_pow_stack_32_xnumel), stream=stream0)
        del arg1159_1
        del arg1161_1
        del arg1164_1
        del arg1165_1
        del buf1998
        del buf41
        del buf45
        buf2004 = buf2000; del buf2000  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.bias_addmm(reinterpret_tensor(arg1163_1, (4096 + s6, 3072), (0, 1), 0), reinterpret_tensor(buf1997, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1162_1, (3072, 3072), (1, 3072), 0), alpha=1, beta=1, out=buf2004)
        del arg1162_1
        del arg1163_1
        # Topologically Sorted Source Nodes: [out_226], Original ATen: [aten._scaled_dot_product_flash_attention]
        buf2007 = torch.ops.aten._scaled_dot_product_flash_attention.default(buf2005, buf2006, reinterpret_tensor(buf2004, (1, 24, 4096 + s6, 128), (12582912 + (3072*s6), 128, 3072, 1), 0), scale=0.08838834764831843)
        del buf2004
        del buf2005
        del buf2006
        buf2008 = buf2007[0]
        del buf2007
        buf2014 = empty_strided_cuda((1, 6144), (6144, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [silu_79], Original ATen: [aten.silu]
        triton_tem_fused_silu_39.run(buf2013, arg1168_1, buf2014, grid=torch._inductor.kernel.mm_common.mm_grid(1, 6144, meta2), stream=stream0)
        del arg1168_1
        del buf2013
        buf2015 = buf1989; del buf1989  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf1997, (4096 + s6, 3072), (3072, 1), 0), reinterpret_tensor(arg1156_1, (3072, 12288), (1, 3072), 0), out=buf2015)
        del arg1156_1
        del buf1997
        buf2016 = buf1990; del buf1990  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_703], Original ATen: [aten.cat]
        triton_poi_fused_cat_33_xnumel = 62914560 + (15360*s6)
        triton_poi_fused_cat_33.run(buf2008, buf2015, arg1157_1, buf2016, triton_poi_fused_cat_33_xnumel, grid=grid(triton_poi_fused_cat_33_xnumel), stream=stream0)
        del arg1157_1
        del buf2015
        buf2017 = reinterpret_tensor(buf2008, (4096 + s6, 3072), (3072, 1), 0); del buf2008  # reuse
        # Unsorted Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf2016, (4096 + s6, 15360), (15360, 1), 0), reinterpret_tensor(arg1166_1, (15360, 3072), (1, 15360), 0), out=buf2017)
        del arg1166_1
        del buf2016
        buf2018 = reinterpret_tensor(buf2017, (1, 4096 + s6, 3072), (12582912 + (3072*s6), 3072, 1), 0); del buf2017  # reuse
        # Topologically Sorted Source Nodes: [hidden_states_698, hidden_states_704, hidden_states_705], Original ATen: [aten.cat, aten.mul, aten.add]
        triton_poi_fused_add_cat_mul_36_xnumel = 12582912 + (3072*s6)
        triton_poi_fused_add_cat_mul_36.run(buf2018, buf1992, buf1988, arg1155_1, arg1167_1, s6, triton_poi_fused_add_cat_mul_36_xnumel, grid=grid(triton_poi_fused_add_cat_mul_36_xnumel), stream=stream0)
        del arg1155_1
        del arg1167_1
        del buf1988
        del buf1992
        buf2022 = reinterpret_tensor(buf1030, (1, 4096, 3072), (12582912, 3072, 1), 0); del buf1030  # reuse
        # Topologically Sorted Source Nodes: [layer_norm_114, mul_771, x_76], Original ATen: [aten.native_layer_norm, aten.mul, aten.add]
        triton_red_fused_add_mul_native_layer_norm_40.run(buf2018, buf2014, arg1169_1, buf2022, s6, 4096, 3072, grid=grid(4096), stream=stream0)
        del arg1169_1
        del buf2014
        del buf2018
        buf2023 = empty_strided_cuda((4096, 64), (64, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [output], Original ATen: [aten.addmm]
        triton_tem_fused_addmm_41.run(arg1171_1, buf2022, arg1170_1, buf2023, grid=torch._inductor.kernel.mm_common.mm_grid(4096, 64, meta5), stream=stream0)
        del arg1170_1
        del arg1171_1
        del buf2022
    return (reinterpret_tensor(buf2023, (1, 4096, 64), (262144, 64, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3072, 64), (64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 4096, 64), (262144, 64, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = 256
    arg6_1 = 0
    arg7_1 = 1
    arg8_1 = rand_strided((1, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((3072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((3072, 256), (256, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((3072, 768), (768, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((3072, 4096), (4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = 512
    arg24_1 = rand_strided((1, 512, 4096), (2097152, 4096, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((512, 3), (3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((4096, 3), (3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = 10000
    arg28_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg161_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg162_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg163_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg164_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg165_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg166_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg167_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg168_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg169_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg170_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg171_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg172_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg173_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg176_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg177_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg178_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg179_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg180_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg181_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg182_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg183_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg184_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg185_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg186_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg187_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg188_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg189_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg190_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg191_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg192_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg193_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg194_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg195_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg196_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg197_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg198_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg199_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg200_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg201_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg202_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg203_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg204_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg206_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg207_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg208_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg209_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg210_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg211_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg212_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg213_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg214_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg215_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg216_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg217_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg218_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg219_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg220_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg221_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg222_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg223_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg224_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg225_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg226_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg227_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg228_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg229_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg230_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg231_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg232_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg233_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg234_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg235_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg236_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg237_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg238_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg239_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg240_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg241_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg242_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg243_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg244_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg245_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg246_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg247_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg248_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg249_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg250_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg251_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg252_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg253_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg254_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg255_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg256_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg257_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg258_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg259_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg260_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg261_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg262_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg263_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg264_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg265_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg266_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg267_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg269_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg270_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg271_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg272_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg273_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg274_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg275_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg276_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg277_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg278_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg279_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg280_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg281_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg282_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg283_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg284_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg285_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg286_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg287_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg288_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg289_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg290_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg291_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg292_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg293_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg294_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg295_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg296_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg297_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg298_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg299_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg300_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg301_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg302_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg303_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg304_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg305_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg306_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg307_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg308_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg309_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg310_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg311_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg312_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg313_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg314_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg315_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg316_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg317_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg318_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg319_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg320_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg321_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg322_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg323_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg324_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg325_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg326_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg327_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg328_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg329_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg330_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg331_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg332_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg336_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg337_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg338_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg339_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg340_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg341_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg342_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg343_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg344_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg345_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg346_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg347_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg348_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg349_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg350_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg351_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg352_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg353_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg354_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg355_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg356_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg357_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg358_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg359_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg360_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg361_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg362_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg363_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg364_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg365_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg366_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg367_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg368_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg369_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg370_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg371_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg372_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg373_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg374_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg375_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg376_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg377_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg378_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg379_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg380_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg381_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg382_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg383_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg384_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg385_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg386_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg387_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg388_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg389_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg390_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg391_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg392_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg393_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg394_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg395_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg396_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg397_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg398_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg399_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg400_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg401_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg402_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg403_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg404_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg405_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg406_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg407_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg408_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg409_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg410_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg411_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg412_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg413_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg414_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg415_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg416_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg417_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg418_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg419_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg420_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg421_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg422_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg423_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg424_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg425_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg426_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg427_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg428_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg429_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg430_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg431_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg432_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg433_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg434_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg435_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg436_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg437_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg438_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg439_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg440_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg441_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg442_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg443_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg444_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg445_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg446_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg447_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg448_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg449_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg450_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg451_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg452_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg453_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg454_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg455_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg456_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg457_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg458_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg459_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg460_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg461_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg462_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg463_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg464_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg465_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg466_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg467_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg468_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg469_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg470_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg471_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg472_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg473_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg474_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg475_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg476_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg477_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg478_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg479_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg480_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg481_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg482_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg483_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg484_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg485_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg486_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg487_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg488_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg489_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg490_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg491_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg492_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg493_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg494_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg495_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg496_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg497_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg498_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg499_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg500_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg501_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg502_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg503_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg504_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg505_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg506_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg507_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg508_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg509_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg510_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg511_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg512_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg513_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg514_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg515_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg516_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg517_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg518_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg519_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg520_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg521_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg522_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg523_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg524_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg525_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg526_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg527_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg528_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg529_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg530_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg531_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg532_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg533_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg534_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg535_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg536_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg537_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg538_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg539_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg540_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg541_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg542_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg543_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg544_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg545_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg546_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg547_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg548_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg549_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg550_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg551_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg552_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg553_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg554_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg555_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg556_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg557_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg558_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg559_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg560_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg561_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg562_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg563_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg564_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg565_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg566_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg567_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg568_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg569_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg570_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg571_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg572_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg573_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg574_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg575_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg576_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg577_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg578_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg579_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg580_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg581_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg582_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg583_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg584_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg585_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg586_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg587_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg588_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg589_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg590_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg591_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg592_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg593_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg594_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg595_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg596_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg597_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg598_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg599_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg600_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg601_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg602_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg603_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg604_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg605_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg606_1 = rand_strided((18432, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg607_1 = rand_strided((18432, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg608_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg609_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg610_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg611_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg612_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg613_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg614_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg615_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg616_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg617_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg618_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg619_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg620_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg621_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg622_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg623_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg624_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg625_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg626_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg627_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg628_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg629_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg630_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg631_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg632_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg633_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg634_1 = rand_strided((3072, 12288), (12288, 1), device='cuda:0', dtype=torch.bfloat16)
    arg635_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg636_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg637_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg638_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg639_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg640_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg641_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg642_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg643_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg644_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg645_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg646_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg647_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg648_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg649_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg650_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg651_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg652_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg653_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg654_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg655_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg656_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg657_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg658_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg659_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg660_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg661_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg662_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg663_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg664_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg665_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg666_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg667_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg668_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg669_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg670_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg671_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg672_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg673_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg674_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg675_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg676_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg677_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg678_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg679_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg680_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg681_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg682_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg683_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg684_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg685_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg686_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg687_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg688_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg689_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg690_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg691_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg692_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg693_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg694_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg695_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg696_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg697_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg698_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg699_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg700_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg701_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg702_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg703_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg704_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg705_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg706_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg707_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg708_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg709_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg710_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg711_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg712_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg713_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg714_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg715_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg716_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg717_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg718_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg719_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg720_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg721_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg722_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg723_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg724_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg725_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg726_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg727_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg728_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg729_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg730_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg731_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg732_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg733_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg734_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg735_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg736_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg737_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg738_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg739_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg740_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg741_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg742_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg743_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg744_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg745_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg746_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg747_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg748_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg749_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg750_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg751_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg752_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg753_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg754_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg755_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg756_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg757_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg758_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg759_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg760_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg761_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg762_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg763_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg764_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg765_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg766_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg767_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg768_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg769_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg770_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg771_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg772_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg773_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg774_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg775_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg776_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg777_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg778_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg779_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg780_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg781_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg782_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg783_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg784_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg785_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg786_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg787_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg788_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg789_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg790_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg791_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg792_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg793_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg794_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg795_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg796_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg797_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg798_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg799_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg800_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg801_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg802_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg803_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg804_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg805_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg806_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg807_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg808_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg809_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg810_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg811_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg812_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg813_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg814_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg815_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg816_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg817_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg818_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg819_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg820_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg821_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg822_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg823_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg824_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg825_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg826_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg827_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg828_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg829_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg830_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg831_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg832_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg833_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg834_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg835_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg836_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg837_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg838_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg839_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg840_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg841_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg842_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg843_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg844_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg845_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg846_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg847_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg848_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg849_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg850_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg851_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg852_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg853_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg854_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg855_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg856_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg857_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg858_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg859_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg860_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg861_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg862_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg863_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg864_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg865_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg866_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg867_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg868_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg869_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg870_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg871_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg872_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg873_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg874_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg875_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg876_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg877_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg878_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg879_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg880_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg881_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg882_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg883_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg884_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg885_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg886_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg887_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg888_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg889_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg890_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg891_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg892_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg893_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg894_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg895_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg896_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg897_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg898_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg899_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg900_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg901_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg902_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg903_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg904_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg905_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg906_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg907_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg908_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg909_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg910_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg911_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg912_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg913_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg914_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg915_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg916_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg917_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg918_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg919_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg920_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg921_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg922_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg923_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg924_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg925_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg926_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg927_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg928_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg929_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg930_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg931_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg932_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg933_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg934_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg935_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg936_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg937_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg938_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg939_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg940_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg941_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg942_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg943_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg944_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg945_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg946_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg947_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg948_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg949_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg950_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg951_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg952_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg953_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg954_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg955_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg956_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg957_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg958_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg959_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg960_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg961_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg962_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg963_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg964_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg965_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg966_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg967_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg968_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg969_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg970_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg971_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg972_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg973_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg974_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg975_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg976_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg977_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg978_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg979_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg980_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg981_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg982_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg983_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg984_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg985_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg986_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg987_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg988_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg989_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg990_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg991_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg992_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg993_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg994_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg995_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg996_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg997_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg998_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg999_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1000_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1001_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1002_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1003_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1004_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1005_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1006_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1007_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1008_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1009_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1010_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1011_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1012_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1013_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1014_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1015_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1016_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1017_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1018_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1019_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1020_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1021_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1022_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1023_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1024_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1025_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1026_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1027_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1028_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1029_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1030_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1031_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1032_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1033_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1034_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1035_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1036_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1037_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1038_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1039_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1040_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1041_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1042_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1043_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1044_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1045_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1046_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1047_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1048_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1049_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1050_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1051_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1052_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1053_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1054_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1055_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1056_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1057_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1058_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1059_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1060_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1061_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1062_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1063_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1064_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1065_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1066_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1067_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1068_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1069_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1070_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1071_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1072_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1073_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1074_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1075_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1076_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1077_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1078_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1079_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1080_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1081_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1082_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1083_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1084_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1085_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1086_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1087_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1088_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1089_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1090_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1091_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1092_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1093_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1094_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1095_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1096_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1097_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1098_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1099_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1100_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1101_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1102_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1103_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1104_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1105_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1106_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1107_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1108_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1109_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1110_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1111_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1112_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1113_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1114_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1115_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1116_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1117_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1118_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1119_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1120_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1121_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1124_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1125_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1126_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1127_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1128_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1129_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1130_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1131_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1132_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1133_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1134_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1135_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1138_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1139_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1140_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1141_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1142_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1143_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1144_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1145_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1146_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1147_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1148_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1149_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1152_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1153_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1154_1 = rand_strided((9216, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1155_1 = rand_strided((9216, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1156_1 = rand_strided((12288, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1157_1 = rand_strided((12288, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1158_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1159_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1160_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1161_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1162_1 = rand_strided((3072, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1163_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1164_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1165_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1166_1 = rand_strided((3072, 15360), (15360, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1167_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1168_1 = rand_strided((6144, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1169_1 = rand_strided((6144, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg1170_1 = rand_strided((64, 3072), (3072, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1171_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
