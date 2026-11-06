"""K-major FP8 GEMM timings (CUDA graph, cold weights, min over rounds), to compare two DeepGEMM builds before / after a
change to the shared SM120 kernels (run it once per build with that build on PYTHONPATH, and alternate the runs).
  sm120_remote.sh ts /path/to/sm120_bench_kmajor.py [--kernel 1d2d|1d1d] [--repeats 5]
"""
import argparse
import math

import torch

import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from sm120_baseline import bench_graph

SHAPES = [(2112, 7168), (576, 7168), (3072, 1536), (7168, 2048), (4608, 7168), (4096, 4096), (8192, 8192)]
MS = [1, 16, 64, 256, 1024, 4096]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', type=str, default='1d2d', choices=['1d2d', '1d1d'])
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    for n, k in SHAPES:
        b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        if args.kernel == '1d2d':
            b, sfb = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)
            extra = {}
        else:
            b, sfb = per_token_cast_to_fp8(b_bf16, use_ue8m0=True, gran_k=128)
            sfb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfb)
            extra = dict(recipe=(1, 1, 128))
        copies = [(b.clone(), sfb.clone()) for _ in range(max(4, min(64, math.ceil(512e6 / (n * k)))))]
        for m in MS:
            a_bf16 = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            if args.kernel == '1d2d':
                a, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
            else:
                a, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=128)
                sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            t = min(bench_graph(lambda w: deep_gemm.fp8_fp4_gemm_nt((a, sfa), (w[0], w[1]), d,
                                                                    disable_ue8m0_cast=(args.kernel == '1d2d'), **extra), copies)
                    for _ in range(args.repeats))
            print(f'{args.kernel} m={m:5} n={n:5} k={k:5} {t * 1e6:8.2f}', flush=True)


if __name__ == '__main__':
    main()
