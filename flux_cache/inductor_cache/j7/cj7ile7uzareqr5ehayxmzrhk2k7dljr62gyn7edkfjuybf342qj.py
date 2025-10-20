
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
