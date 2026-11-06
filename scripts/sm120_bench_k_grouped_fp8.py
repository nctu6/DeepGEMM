"""Benchmark `k_grouped_fp8_gemm_tn_contiguous` on SM120 (weight-gradient shapes) against
  - a host loop of the K-major FP8 kernel per group with an explicit transpose of each MN-major operand slice, and
  - BF16 references: DeepGEMM's k-grouped BF16 GEMM and a torch (cuBLAS) loop.
Each candidate is timed eagerly (host launch cost included, `--eager`) and replayed from a CUDA graph (GPU time only),
with operands cold in L2 (a 256 MB flush before every call).

  scripts/sm120_remote.sh dg scripts/sm120_bench_k_grouped_fp8.py [--gran-k 128] [--repeats 5]
"""
import argparse
import random

import torch

import deep_gemm
from deep_gemm.utils import align, per_channel_cast_to_fp8
from generators import k_grouped_cast_fp8_fp4_with_major, MajorTypeAB

# (num_groups, m, n, expected K per group): DeepSeek-V3 expert Wgrad, hidden 7168 / MoE intermediate 2048 (gate_up is
# 4096 = 2 x 2048 wide), then the shapes of `tests/test_fp8_fp4.py`
MOE_SHAPES = [(g, m, n, k) for g in (4, 8, 16, 32) for (m, n) in ((4096, 7168), (7168, 2048)) for k in (256, 1024, 4096)]
TEST_SHAPES = [(8, 768, 2048, 128), (4, 4096, 7168, 8192), (4, 7168, 2048, 8192), (8, 4096, 7168, 4096),
               (8, 7168, 2048, 4096), (16, 4096, 7168, 2048), (16, 7168, 2048, 2048)]


def flush():
    torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()


def time_eager(fn, iters=20):
    fn(); torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(iters):
        flush()
        start.record(); fn(); end.record(); end.synchronize()
        total += start.elapsed_time(end)
    return total / iters * 1e3


def time_graph(fn, iters=20):
    fn(); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    total = 0.0
    for _ in range(iters):
        flush()
        start.record(); graph.replay(); end.record(); end.synchronize()
        total += start.elapsed_time(end)
    return total / iters * 1e3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gran-k', type=int, default=128, choices=[32, 128])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--shapes', choices=['moe', 'test', 'all'], default='all')
    parser.add_argument('--accumulate', type=int, default=1)
    args = parser.parse_args()
    shapes = {'moe': MOE_SHAPES, 'test': TEST_SHAPES, 'all': TEST_SHAPES + MOE_SHAPES}[args.shapes]
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    random.seed(0)
    torch.manual_seed(0)
    gran_k = args.gran_k
    print(f'gran_k={gran_k} accumulate={args.accumulate}; times in us (eager | CUDA graph)')
    print(f'{"G":>3} {"m":>5} {"n":>5} {"sum_k":>6} | {"k-grouped FP8":>16} | {"host loop+transp":>16} | {"DG k-grouped BF16":>17} | {"torch bf16 loop":>16} | FP8 vs loop (eager/graph)')
    for g, m, n, ek in shapes:
        ks = [align(max(1, int(ek * random.uniform(0.7, 1.3))), 128) for _ in range(g)]
        sum_k = sum(ks)
        a_bf = torch.randn((sum_k, m), device='cuda', dtype=torch.bfloat16)
        b_bf = torch.randn((sum_k, n), device='cuda', dtype=torch.bfloat16)
        grouped_layout = torch.tensor(ks, device='cuda', dtype=torch.int32)
        a = k_grouped_cast_fp8_fp4_with_major(a_bf, ks, MajorTypeAB.MNMajor, True, gran_k, False)
        b = k_grouped_cast_fp8_fp4_with_major(b_bf, ks, MajorTypeAB.MNMajor, True, gran_k, False)
        d = torch.randn((g, m, n), device='cuda', dtype=torch.float)
        c = d if args.accumulate else None
        recipe = (1, 1, gran_k)

        def run_new():
            deep_gemm.k_grouped_fp8_gemm_tn_contiguous(a, b, d, ks, grouped_layout, c, recipe=recipe)

        # Baseline: per group, an explicit transpose of both operand slices and the K-major kernel on packed scales
        sfa = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(a[1], grouped_layout, ks, gran_k, 128, False)
        sfb = deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(b[1], grouped_layout, ks, gran_k, 128, False)
        offs, sf_offs, o, so = [], [], 0, 0
        for k in ks:
            offs.append(o); sf_offs.append(so); o += k; so += (k // gran_k + 3) // 4

        def run_loop():
            for i, k in enumerate(ks):
                a_k = a[0][offs[i]:offs[i] + k].t().contiguous()
                b_k = b[0][offs[i]:offs[i] + k].t().contiguous()
                sa = sfa[sf_offs[i]:sf_offs[i] + (k // gran_k + 3) // 4].t()
                sb = sfb[sf_offs[i]:sf_offs[i] + (k // gran_k + 3) // 4].t()
                deep_gemm.fp8_gemm_nt((a_k, sa), (b_k, sb), d[i], c=d[i] if args.accumulate else None, recipe=recipe)

        c_bf = d.clone() if args.accumulate else None

        def run_bf16():
            deep_gemm.k_grouped_bf16_gemm_tn_contiguous(a_bf, b_bf, d, ks, grouped_layout, c)

        def run_torch():
            for i, k in enumerate(ks):
                r = a_bf[offs[i]:offs[i] + k].t() @ b_bf[offs[i]:offs[i] + k]
                if args.accumulate:
                    d[i] += r
                else:
                    d[i] = r

        # Correctness of the FP8 path against the loop on the same scales
        d0 = d.clone()
        run_new(); out_new = d.clone(); d.copy_(d0)
        run_loop(); out_loop = d.clone(); d.copy_(d0)
        assert torch.equal(out_new, out_loop), 'k-grouped FP8 differs from the per-group loop'

        res = []
        for fn in (run_new, run_loop, run_bf16, run_torch):
            e = min(time_eager(fn) for _ in range(args.repeats))
            try:
                gr = min(time_graph(fn) for _ in range(args.repeats))
            except Exception:
                gr = float('nan')
            res.append((e, gr))
        fmt = lambda r: f'{r[0]:7.0f} | {r[1]:6.0f}'
        print(f'{g:>3} {m:>5} {n:>5} {sum_k:>6} | {fmt(res[0]):>16} | {fmt(res[1]):>16} | {fmt(res[2]):>17} | {fmt(res[3]):>16} | '
              f'{res[1][0] / res[0][0]:.2f}x / {res[1][1] / res[0][1]:.2f}x', flush=True)


if __name__ == '__main__':
    main()
