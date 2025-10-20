
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
    triton_meta={'signature': {0: '*bf16', 1: '*bf16', 2: '*bf16', 3: '*bf16', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32'}, 'device': DeviceProperties(type='cuda', index=0, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, multi_processor_count=114), 'constants': {}, 'configs': [AttrsDescriptor(divisible_by_16=(0, 1, 2, 3, 7), equal_to_1=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_mul_native_layer_norm_35', 'mutated_arg_names': ['in_out_ptr0'], 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '4CCCA65D160E209BCA90DF8D7D98B2EF1474A5946F9D1F5951D0EE51BFFF8A87', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, ks0, ks1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp7 = ks0 + ks1
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
        tmp23 = ks0 + ks1
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
