
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
