"""Correctness of the SM120 block-scaled GEMM (packed UE8M0 scales) against a dequantized reference.

Modes (all BF16 output, K-major NT):
  fp4      FP4 (E2M1) A/B, 32-element UE8M0 scales          -> mxf4 `m16n8k64`
  nvfp4    FP4 (E2M1) A/B, 16-element UE4M3 scales          -> mxf4nvf4 `scale_vec::4X` `m16n8k64`
  fp8g32   FP8 (E4M3) A/B, 32-element UE8M0 scales          -> mxf8f6f4 `m16n8k32`
  fp8g128  FP8 (E4M3) A/B, 128-element UE8M0 scales (packed) -> mxf8f6f4 `m16n8k32`
  bf16     BF16 A/B, no scales (same kernel)                 -> `m16n8k16` (cuBLASLt is bypassed via deterministic mode)

Run on the test machine with GPU 7 (see `sm120_remote.sh`):
  sm120_remote.sh dg scripts/sm120_blockscale_check.py [--mode fp4|fp8g32|fp8g128|all] [--grouped]
"""
import argparse
import sys

import torch

import deep_gemm
from deep_gemm.utils.math import (cast_back_from_fp4, cast_back_from_fp8, per_token_cast_to_fp4,
                                  per_token_cast_to_fp8, per_token_cast_to_nvfp4)


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denom = (x * x + y * y).sum()
    return (1 - 2 * (x * y).sum() / denom).item() if denom > 0 else 0.0


def quantize(x, mode):
    """Returns (data, fp32 power-of-two scales, dequantized reference)"""
    if mode == 'bf16':
        return x, None, x.float()
    if mode == 'nvfp4':
        q, sf = per_token_cast_to_nvfp4(x)
        return q, sf, cast_back_from_fp4(q, sf, gran_k=16)
    if mode == 'fp4':
        q, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
        return q, sf, cast_back_from_fp4(q, sf, gran_k=32)
    gran_k = 32 if mode == 'fp8g32' else 128
    q, sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=gran_k)
    return q, sf, cast_back_from_fp8(q, sf, gran_k=gran_k)


def to_gemm_sf(sf, mode):
    """FP32 (1, 128) scales stay on the FP32 kernel, so 128-element UE8M0 scales are handed over pre-packed. NVFP4 scales
    are pre-packed here too, to test that path (the PSUM test hands SFA over as FP32 values)"""
    if mode == 'nvfp4':
        return deep_gemm.get_mn_major_tma_aligned_packed_ue4m3_tensor(sf)
    return deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf) if mode == 'fp8g128' else sf


def gran_k_of(mode):
    return {'fp8g128': 128, 'nvfp4': 16}.get(mode, 32)


def gemm_nt(mode, qa, sfa, qb, sfb, d):
    if mode == 'bf16':
        deep_gemm.bf16_gemm_nt(qa, qb, d)
    else:
        deep_gemm.fp8_fp4_gemm_nt((qa, sfa), (qb, sfb), d, recipe=(1, 1, gran_k_of(mode)))


def run_epilogue(mode, m, n, k, out_dtype, alpha, accumulate, in_place, seed=0):
    """D = alpha * A @ B^T (+ C): `alpha` scales the product, C is added unscaled (as on SM100)"""
    torch.manual_seed(seed)
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, ref_a = quantize(a, mode)
    qb, sfb, ref_b = quantize(b, mode)
    sfa, sfb = to_gemm_sf(sfa, mode), to_gemm_sf(sfb, mode)
    c = torch.randn(m, n, device='cuda', dtype=out_dtype) if accumulate else None
    ref = alpha * (ref_a @ ref_b.t()) + (c.float() if accumulate else 0)
    d = c.clone() if in_place else torch.empty(m, n, device='cuda', dtype=out_dtype)
    c_arg = d if in_place else c
    if mode == 'bf16':
        deep_gemm.bf16_gemm_nt(qa, qb, d, c=c_arg, alpha=alpha)
    else:
        deep_gemm.fp8_fp4_gemm_nt((qa, sfa), (qb, sfb), d, c=c_arg, recipe=(1, 1, gran_k_of(mode)), alpha=alpha)
    torch.cuda.synchronize()
    return calc_diff(d, ref.to(out_dtype))


def gemm_contiguous(mode, qa, sfa, qb, sfb, d, layout, use_psum_layout=False):
    if mode == 'bf16':
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(qa, qb, d, layout, use_psum_layout=use_psum_layout)
    else:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((qa, sfa), (qb, sfb), d, layout, recipe=(1, 1, gran_k_of(mode)),
                                                       use_psum_layout=use_psum_layout)


def gemm_masked(mode, qa, sfa, qb, sfb, d, masked_m, expected_m):
    if mode == 'bf16':
        deep_gemm.m_grouped_bf16_gemm_nt_masked(qa, qb, d, masked_m, expected_m)
    else:
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked((qa, sfa), (qb, sfb), d, masked_m, expected_m,
                                                   recipe=(1, 1, gran_k_of(mode)))


def run_repeat(mode, m, n, k, repeats, seed=0):
    """Bitwise determinism over many launches: a shared-memory race (a stage overwritten while still being read) shows up
    as an occasional wrong tile, so the result must be identical on every run"""
    torch.manual_seed(seed)
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, _ = quantize(a, mode)
    qb, sfb, _ = quantize(b, mode)
    sfa, sfb = to_gemm_sf(sfa, mode), to_gemm_sf(sfb, mode)
    first = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    gemm_nt(mode, qa, sfa, qb, sfb, first)
    d = torch.empty_like(first)
    mismatches = 0
    for _ in range(repeats):
        d.fill_(float('nan'))
        gemm_nt(mode, qa, sfa, qb, sfb, d)
        mismatches += int(not torch.equal(d, first))
    torch.cuda.synchronize()
    return mismatches


def run_normal(mode, m, n, k, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
    b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, ref_a = quantize(a, mode)
    qb, sfb, ref_b = quantize(b, mode)
    ref = (ref_a @ ref_b.t()).to(torch.bfloat16)
    d = torch.empty(m, n, device='cuda', dtype=torch.bfloat16)
    gemm_nt(mode, qa, to_gemm_sf(sfa, mode), qb, to_gemm_sf(sfb, mode), d)
    torch.cuda.synchronize()
    return calc_diff(d, ref)


def run_contiguous(mode, num_groups, m_per_group, n, k, seed=0):
    torch.manual_seed(seed)
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
    sfb = None if mode == 'bf16' else to_gemm_sf(torch.stack(sfbs), mode)
    sfa = to_gemm_sf(sfa, mode)
    m_indices = torch.arange(num_groups, device='cuda', dtype=torch.int32).repeat_interleave(m_total // num_groups)
    ref = torch.empty(m_total, n, device='cuda', dtype=torch.bfloat16)
    for g in range(num_groups):
        sl = slice(g * (m_total // num_groups), (g + 1) * (m_total // num_groups))
        ref[sl] = (ref_a[sl] @ ref_b[g].t()).to(torch.bfloat16)
    d = torch.empty(m_total, n, device='cuda', dtype=torch.bfloat16)
    gemm_contiguous(mode, qa, sfa, qb, sfb, d, m_indices)
    torch.cuda.synchronize()
    return calc_diff(d, ref)


def run_contiguous_psum(mode, num_groups, m_per_group, n, k, seed=0):
    """PSUM layout: `grouped_layout[g]` is the cumulative logical end of group g, and each group starts at the previous
    end aligned up to the contiguous-layout alignment; rows in the gaps are not compared"""
    torch.manual_seed(seed)
    align = deep_gemm.get_mk_alignment_for_contiguous_layout()
    actual_ms = [max(1, int(m_per_group * f)) for f in torch.linspace(0.6, 1.4, num_groups).tolist()]
    starts, ends, start = [], [], 0
    for am in actual_ms:
        starts.append(start); ends.append(start + am)
        start = (start + am + align - 1) // align * align
    m_total = start
    a = torch.zeros(m_total, k, device='cuda', dtype=torch.bfloat16)
    for st, en in zip(starts, ends):
        a[st:en] = torch.randn(en - st, k, device='cuda', dtype=torch.bfloat16)
    qa, sfa, ref_a = quantize(a, mode)
    qbs, sfbs, ref_bs = zip(*[quantize(torch.randn(n, k, device='cuda', dtype=torch.bfloat16), mode) for _ in range(num_groups)])
    qb = torch.stack(qbs)
    sfb = None if mode == 'bf16' else to_gemm_sf(torch.stack(sfbs), mode)
    psum = torch.tensor(ends, device='cuda', dtype=torch.int32)
    if mode == 'fp8g128':
        sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa, psum_layout=psum)
    d = torch.zeros(m_total, n, device='cuda', dtype=torch.bfloat16)
    gemm_contiguous(mode, qa, sfa, qb, sfb, d, psum, use_psum_layout=True)
    torch.cuda.synchronize()
    worst = 0.0
    for g, (st, en) in enumerate(zip(starts, ends)):
        ref = (ref_a[st:en] @ ref_bs[g].t()).to(torch.bfloat16)
        worst = max(worst, calc_diff(d[st:en], ref))
    return worst


def run_masked(mode, num_groups, max_m, n, k, seed=0):
    torch.manual_seed(seed)
    masked_m = torch.randint(1, max_m + 1, (num_groups,), device='cuda', dtype=torch.int32)
    a = torch.randn(num_groups, max_m, k, device='cuda', dtype=torch.bfloat16)
    qas, sfas, ref_as = zip(*[quantize(a[g], mode) for g in range(num_groups)])
    qbs, sfbs, ref_bs = zip(*[quantize(torch.randn(n, k, device='cuda', dtype=torch.bfloat16), mode) for _ in range(num_groups)])
    qa, qb = torch.stack(qas), torch.stack(qbs)
    if mode == 'bf16':
        sfa = sfb = None
    else:
        sfa, sfb = to_gemm_sf(torch.stack(sfas), mode), to_gemm_sf(torch.stack(sfbs), mode)
    d = torch.zeros(num_groups, max_m, n, device='cuda', dtype=torch.bfloat16)
    gemm_masked(mode, qa, sfa, qb, sfb, d, masked_m, max_m // 2 + 1)
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
    parser.add_argument('--repeat', type=int, default=0, help='also check bitwise determinism over this many launches')
    args = parser.parse_args()
    modes = ['fp4', 'nvfp4', 'fp8g32', 'fp8g128', 'bf16'] if args.mode == 'all' else [args.mode]
    # BF16 GEMMs otherwise go to cuBLASLt
    deep_gemm.use_deterministic_algorithms(True)
    # Operands and scales are exact in the reference, so only FP32 summation order and bf16 output rounding differ
    # (<= ~1e-8). A single wrong scale or flipped byte shows up at >= 1e-6 (see the negative control in the report)
    tol = 1e-7
    failed = False
    shapes = [(1, 4096, 7168), (8, 2112, 7168), (64, 7168, 2048), (128, 4096, 4096), (333, 576, 1536),
              (1000, 3072, 1536), (4096, 4096, 7168),
              (64, 1024, 384), (17, 2048, 2176), (200, 1536, 640)]  # K tails: a half FP4 K block, a partial scale word
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
            for (g, mpg, n, k) in [(4, 128, 2048, 2048), (8, 100, 1536, 7168)]:
                diff = run_contiguous_psum(mode, g, mpg, n, k)
                ok = diff < tol
                failed |= not ok
                print(f'[{mode:7s}] psum        g={g} m/g~{mpg} n={n} k={k}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
            for (g, mm, n, k) in [(4, 128, 2048, 2048), (8, 64, 1536, 7168)]:
                diff = run_masked(mode, g, mm, n, k)
                ok = diff < tol
                failed |= not ok
                print(f'[{mode:7s}] masked      g={g} max_m={mm} n={n} k={k}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    # Epilogue: alpha, accumulation (distinct C, and C aliasing D), FP32 output. A non-power-of-two alpha changes the
    # rounding order vs. the reference, so the tolerance is looser
    for mode in modes:
        for (m, n, k) in [(1, 2112, 7168), (200, 1536, 640), (512, 4096, 2048)]:
            for out_dtype in (torch.bfloat16, torch.float):
                for alpha, accumulate, in_place in [(0.37, False, False), (1.0, True, False), (-1.5, True, True)]:
                    diff = run_epilogue(mode, m, n, k, out_dtype, alpha, accumulate, in_place)
                    ok = diff < 1e-6
                    failed |= not ok
                    print(f'[{mode:7s}] epilogue    m={m:5d} n={n:5d} k={k:5d} {str(out_dtype)[6:]:8s} alpha={alpha:5.2f} '
                          f'acc={int(accumulate)} in_place={int(in_place)}  diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    if args.repeat:
        for mode in modes:
            for (m, n, k) in [(128, 7168, 2048), (64, 4608, 7168), (1, 2112, 7168), (4096, 4096, 7168), (256, 3072, 1536)]:
                bad = run_repeat(mode, m, n, k, args.repeat)
                failed |= bad > 0
                print(f'[{mode:7s}] determinism m={m:5d} n={n:5d} k={k:5d}  {bad}/{args.repeat} runs differ  {"ok" if bad == 0 else "FAIL"}', flush=True)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
