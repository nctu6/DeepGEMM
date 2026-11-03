# Sweep SM120 FP8 GEMM tile layouts and K splits per shape (development helper for tuning heuristics)
import argparse
import os
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

from sm120_baseline import NK, MS, bench_cold

LAYOUTS = ['16x64', '16x128', '16x256', '32x64', '32x128', '32x256', '64x64', '64x128', '128x64', '128x128']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--nks', type=str, default=None)
    parser.add_argument('--splits', type=str, default='1')
    args = parser.parse_args()
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    nks = [tuple(int(y) for y in x.split('x')) for x in args.nks.split(',')] if args.nks else NK
    splits = [int(x) for x in args.splits.split(',')]

    for n, k in nks:
        for m in ms:
            a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
            a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=False)
            b_fp8 = per_block_cast_to_fp8(b, use_ue8m0=False)
            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            results = {}
            for key in ('DG_SM120_FORCE_LAYOUT', 'DG_SM120_FORCE_K_SPLITS'):
                os.environ.pop(key, None)
            results['auto'] = bench_cold(lambda: deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d))
            for layout in LAYOUTS:
                bm = int(layout.split('x')[0])
                if bm >= 2 * max(m, 16) and bm > 16:
                    continue
                for s in splits:
                    os.environ['DG_SM120_FORCE_LAYOUT'] = layout
                    os.environ['DG_SM120_FORCE_K_SPLITS'] = str(s)
                    try:
                        results[f'{layout}/{s}'] = bench_cold(lambda: deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d))
                    except RuntimeError:
                        pass
            for key in ('DG_SM120_FORCE_LAYOUT', 'DG_SM120_FORCE_K_SPLITS'):
                os.environ.pop(key, None)
            best = min((v, l) for l, v in results.items() if l != 'auto')
            flops = 2 * m * n * k
            top = sorted((v, l) for l, v in results.items() if l != 'auto')[:5]
            row = ' '.join(f'{l}:{v * 1e6:.1f}' for v, l in top)
            print(f'm={m:5} n={n:6} k={k:6} best={best[1]:>10} {best[0] * 1e6:8.1f}us {flops / best[0] / 1e12:6.1f}TF '
                  f'auto={results["auto"] * 1e6:.1f}us auto/best={results["auto"] / best[0]:.2f} | {row}', flush=True)


if __name__ == '__main__':
    main()
