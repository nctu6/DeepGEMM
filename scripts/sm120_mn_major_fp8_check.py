"""Correctness of MN-major FP8 operands on SM120 (in-kernel `ldmatrix.trans.b8`, and the host transpose route).

The MN-major kernels run the same MMAs on the same operand bytes in the same K order as the K-major ones, so their
results match the K-major kernel's on the same data (bitwise, unless the heuristic picks another tile or a stream-K split
for the MN-major layout, which reorders the FP32 sums: then a few elements differ by one rounding step); they are also compared with a
dequantized FP32 reference. Covers the FP32-scale 1D2D kernel and the UE8M0 block-scaled 1D1D kernel (32/128-element scales), BF16 and
FP32 output, accumulation (1D1D), and both routes (`DG_SM120_MN_MAJOR_FP8` 2 = in-kernel, 1 = host transpose).
Run on the test machine with GPU 7:
  sm120_remote.sh dg scripts/sm120_mn_major_fp8_check.py [--quick]
"""
import itertools
import os
import sys

import torch

import deep_gemm
from deep_gemm.utils.math import cast_back_from_fp8, per_block_cast_to_fp8, per_token_cast_to_fp8


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item() if denom > 0 else 0.0


def mn_major(x):
    """Same values, MN-major storage (`[MN, K]` with unit MN stride; `[G, MN, K]` likewise)"""
    return x.mT.contiguous().mT


def make_operands(kind, m, n, k):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    if kind == '1d2d':
        qa, sfa = per_token_cast_to_fp8(a, use_ue8m0=False)
        qb, sfb = per_block_cast_to_fp8(b, use_ue8m0=False)
        ref_a = cast_back_from_fp8(qa, sfa, gran_k=128)
        ref_b = qb.float() * sfb.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
        return qa, sfa, qb, sfb, ref_a @ ref_b.t(), {}
    gran_k = 32 if kind == '1d1d_g32' else 128
    qa, sfa = per_token_cast_to_fp8(a, use_ue8m0=True, gran_k=gran_k)
    qb, sfb = per_token_cast_to_fp8(b, use_ue8m0=True, gran_k=gran_k)
    ref = cast_back_from_fp8(qa, sfa, gran_k=gran_k) @ cast_back_from_fp8(qb, sfb, gran_k=gran_k).t()
    if gran_k == 128:
        sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
        sfb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfb)
    return qa, sfa, qb, sfb, ref, dict(recipe=(1, 1, gran_k))


def main():
    quick = '--quick' in sys.argv
    shapes = [(64, 256, 256), (128, 512, 384), (16, 2112, 7168), (192, 576, 1024), (1024, 4096, 2048), (4096, 7168, 2048),
              (48, 1520, 1152), (256, 128, 9984)]
    if quick:
        shapes = shapes[:4]
    majors = [('K', 'MN'), ('MN', 'K'), ('MN', 'MN')]
    num = 0
    for mode, kind, (m, n, k), (ma, mb) in itertools.product(('2', '1'), ('1d2d', '1d1d_g128', '1d1d_g32'), shapes, majors):
        os.environ['DG_SM120_MN_MAJOR_FP8'] = mode
        for out_dtype, accumulate in ([(torch.bfloat16, False)] if kind == '1d2d' else
                                      [(torch.bfloat16, False), (torch.float, False), (torch.bfloat16, True), (torch.float, True)]):
            torch.manual_seed(num)
            qa, sfa, qb, sfb, ref, extra = make_operands(kind, m, n, k)
            init = torch.randn((m, n), device='cuda', dtype=out_dtype) * 4

            def gemm(qa_, qb_):
                d = init.clone() if accumulate else torch.empty((m, n), device='cuda', dtype=out_dtype)
                deep_gemm.fp8_fp4_gemm_nt((qa_, sfa), (qb_, sfb), d, c=d if accumulate else None,
                                          disable_ue8m0_cast=(kind == '1d2d'), **extra)
                return d

            d_k = gemm(qa, qb)
            d_mn = gemm(qa if ma == 'K' else mn_major(qa), qb if mb == 'K' else mn_major(qb))
            expected = (ref + init.float()) if accumulate else ref
            diff = calc_diff(d_mn.float(), expected)
            bitwise = torch.equal(d_k, d_mn)
            # A few BF16 elements may round to the neighboring value when the heuristic picks another tile / split
            assert bitwise or calc_diff(d_k.float(), d_mn.float()) < 1e-7, (mode, kind, m, n, k, ma, mb, out_dtype, accumulate, 'differs from the K-major kernel')
            assert diff < 1e-3, (mode, kind, m, n, k, ma, mb, out_dtype, accumulate, diff)
            num += 1
            print(f'mode={mode} {kind:9} m={m:5} n={n:5} k={k:5} {ma:2}/{mb:2} {str(out_dtype)[6:]:8} acc={int(accumulate)} '
                  f'bitwise={bitwise} diff={diff:.1e}', flush=True)
    print(f'OK ({num} cases)')


if __name__ == '__main__':
    main()
