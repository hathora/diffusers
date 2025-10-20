
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
