
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
