"""Correctness of the SM120 block-scaled GEMM (packed UE8M0 scales) against a dequantized reference.

Modes (all BF16 output, K-major NT):
  fp4      FP4 (E2M1) A/B, 32-element UE8M0 scales          -> mxf4 `m16n8k64`
  fp8g32   FP8 (E4M3) A/B, 32-element UE8M0 scales          -> mxf8f6f4 `m16n8k32`
  fp8g128  FP8 (E4M3) A/B, 128-element UE8M0 scales (packed) -> mxf8f6f4 `m16n8k32`

Run on the test machine with GPU 7 (see `sm120_remote.sh`):
  sm120_remote.sh dg scripts/sm120_blockscale_check.py [--mode fp4|fp8g32|fp8g128|all] [--grouped]
"""
import argparse
import sys

import torch

import deep_gemm
from deep_gemm.utils.math import (cast_back_from_fp4, cast_back_from_fp8, per_token_cast_to_fp4,
                                  per_token_cast_to_fp8)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item() if denom > 0 else 0.0


def quantize(x, mode):
    """Returns (data, fp32 power-of-two scales, dequantized reference)"""
    if mode == 'fp4':
        q, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
        return q, sf, cast_back_from_fp4(q, sf, gran_k=32)
    gran_k = 32 if mode == 'fp8g32' else 128
    q, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
    return q, sf, cast_back_from_fp8(q, sf, gran_k=gran_k)


def to_gemm_sf(sf, mode):
    """FP32 (1, 128) scales stay on the FP32 kernel, so 128-element UE8M0 scales are handed over pre-packed"""
    return deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf) if mode == 'fp8g128' else sf


def run_normal(mode, m, n, k, seed=0):
    torch.manual_seed(seed)
    gran_k = 128 if mode == 'fp8g128' else 32
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, ref_a = quantize(a, mode)
    qb, sfb, ref_b = quantize(b, mode)
    ref = (ref_a @ ref_b.t()).to(torch.bfloat16)
    d = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt((qa, to_gemm_sf(sfa, mode)), (qb, to_gemm_sf(sfb, mode)), d, recipe=(1, 1, gran_k))
    torch.cuda.synchronize()
    return calc_diff(d, ref)


def run_contiguous(mode, num_groups, m_per_group, n, k, seed=0):
    torch.manual_seed(seed)
    gran_k = 128 if mode == 'fp8g128' else 32
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    m_total = num_groups * ((m_per_group + align - 1) // align * align)
    a = torch.randn(m_total, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, ref_a = quantize(a, mode)
    bs, qbs, sfbs, ref_bs = [], [], [], []
    for _ in range(num_groups):
        b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
        q, sf, ref_b = quantize(b, mode)
        qbs.append(q); sfbs.append(sf); ref_bs.append(ref_b)
    qb, ref_b = torch.stack(qbs), torch.stack(ref_bs)
    sfb = to_gemm_sf(torch.stack(sfbs), mode)
    sfa = to_gemm_sf(sfa, mode)
    m_indices = torch.arange(num_groups, device='cuda', dtype=torch.int32).repeat_interleave(m_total // num_groups)
    ref = torch.empty(m_total, n, device='cuda', dtype=torch.bfloat16)
    for g in range(num_groups):
        sl = slice(g * (m_total // num_groups), (g + 1) * (m_total // num_groups))
        ref[sl] = (ref_a[sl] @ ref_b[g].t()).to(torch.bfloat16)
    d = torch.empty(m_total, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((qa, sfa), (qb, sfb), d, m_indices, recipe=(1, 1, gran_k))
    torch.cuda.synchronize()
    return calc_diff(d, ref)


def run_masked(mode, num_groups, max_m, n, k, seed=0):
    torch.manual_seed(seed)
    gran_k = 128 if mode == 'fp8g128' else 32
    masked_m = torch.randint(1, max_m + 1, (num_groups,), device='cuda', dtype=torch.int32)
    a = torch.randn(num_groups, max_m, k, device='cuda', dtype=torch.bfloat16)
    qas, sfas, ref_as = zip(*[quantize(a[g], mode) for g in range(num_groups)])
    qbs, sfbs, ref_bs = zip(*[quantize(torch.randn(n, k, device='cuda', dtype=torch.bfloat16), mode) for _ in range(num_groups)])
    qa, qb = torch.stack(qas), torch.stack(qbs)
    sfa, sfb = to_gemm_sf(torch.stack(sfas), mode), to_gemm_sf(torch.stack(sfbs), mode)
    d = torch.zeros(num_groups, max_m, n, device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((qa, sfa), (qb, sfb), d, masked_m, max_m // 2 + 1, recipe=(1, 1, gran_k))
    torch.cuda.synchronize()
    worst = 0.0
    for g in range(num_groups):
        mm = int(masked_m[g])
        ref = (ref_as[g][:mm] @ ref_bs[g].t()).to(torch.bfloat16)
        worst = max(worst, calc_diff(d[g, :mm], ref))
    return worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='all')
    parser.add_argument('--grouped', action='store_true')
    args = parser.parse_args()
    modes = ['fp4', 'fp8g32', 'fp8g128'] if args.mode == 'all' else [args.mode]
    # Operands and scales are exact in the reference, so only FP32 summation order and bf16 output rounding differ
    # (<= ~1e-8). A single wrong scale or flipped byte shows up at >= 1e-6 (see the negative control in the report)
    tol = 1e-7
    failed = False
    shapes = [(1, 4096, 7168), (8, 2112, 7168), (64, 7168, 2048), (128, 4096, 4096), (333, 576, 1536),
              (1000, 3072, 1536), (4096, 4096, 7168)]
    for mode in modes:
        for (m, n, k) in shapes:
            diff = run_normal(mode, m, n, k)
            ok = diff < tol
            failed |= not ok
            print(f'[{mode:7s}] normal      m={m:5d} n={n:5d} k={k:5d}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
        if args.grouped:
            for (g, mpg, n, k) in [(4, 128, 2048, 2048), (8, 64, 1536, 7168)]:
                diff = run_contiguous(mode, g, mpg, n, k)
                ok = diff < tol
                failed |= not ok
                print(f'[{mode:7s}] contiguous  g={g} m/g={mpg} n={n} k={k}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
            for (g, mm, n, k) in [(4, 128, 2048, 2048), (8, 64, 1536, 7168)]:
                diff = run_masked(mode, g, mm, n, k)
                ok = diff < tol
                failed |= not ok
                print(f'[{mode:7s}] masked      g={g} max_m={mm} n={n} k={k}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
