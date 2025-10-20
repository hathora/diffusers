
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
