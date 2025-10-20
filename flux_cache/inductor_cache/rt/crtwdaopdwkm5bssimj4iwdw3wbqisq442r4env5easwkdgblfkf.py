
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
