"""Benchmark of the deterministic BF16 `einsum` heads on SM120: one batched launch against a launch per head and cuBLASLt.
  bhr,hdr->bhd   K-major operands
  bhd,hdr->bhr   K-major A, MN-major B
Variants (us per call):
  batched   `deep_gemm.einsum` in deterministic mode (the SM120 kernel, all heads in one launch)
  perhead   the same with DG_SM120_BATCHED_LAUNCH=0 (a GEMM per head on the batched launch's layout, bit-identical)
  legacy    a `deep_gemm.bf16_gemm_nt` / `bf16_gemm_nn` call per head, each with its own layout and stream-K (what the
            einsum did before the batched launch)
  cublaslt  `deep_gemm.einsum` with deterministic mode off (cuBLASLt)
  ratios    batched / each of the others (below 1 is a win)
Timing: a CUDA graph over enough distinct operand copies to exceed L2 (cold weights), replayed; variants are interleaved
over `--repeats` rounds and the minimum is kept.
  DG_GPU=7 sm120_remote.sh dg scripts/sm120_bench_bf16_einsum.py --bs 4,32,128,4096,8192
"""
import argparse
import math
import os
import torch
import deep_gemm
from sm120_baseline import bench_graph
SHAPES = [(8, 4096, 1024), (128, 512, 128), (16, 576, 128), (64, 512, 192)]   # (h, r, d)
def make_case(expr, b, h, r, d):
    k, n = (r, d) if expr == 'bhr,hdr->bhd' else (d, r)
    x = torch.randn((b, h, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
    z = torch.empty((b, h, n), device='cuda', dtype=torch.bfloat16)
    return (x, y, z), (x.numel() + y.numel()) * 2
def run(expr, mode):
    if mode == 'legacy':
        def legacy(a):
            x, y, z = a
            for i in range(x.size(1)):
                if expr == 'bhr,hdr->bhd':
                    deep_gemm.bf16_gemm_nt(x[:, i], y[i], z[:, i])
                else:
                    deep_gemm.bf16_gemm_nn(x[:, i], y[i], z[:, i])
        return legacy
    def fn(a):
        # The kernel choice is made when the call is captured into the graph
        deep_gemm.use_deterministic_algorithms(mode != 'cublaslt')
        os.environ['DG_SM120_BATCHED_LAUNCH'] = '0' if mode == 'perhead' else '1'
        deep_gemm.einsum(expr, a[0], a[1], a[2])
    return fn
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exprs', type=str, default='bhr,hdr->bhd;bhd,hdr->bhr', help='semicolon-separated')
    parser.add_argument('--bs', type=str, default='1,4,32,128,512,4096,8192')
    parser.add_argument('--shapes', type=str, default=None, help='comma-separated hxrxd')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    shapes = [tuple(int(y) for y in s.split('x')) for s in args.shapes.split(',')] if args.shapes else SHAPES
    exprs = args.exprs.split(';')
    names = ['batched', 'perhead', 'legacy', 'cublaslt']
    print(f'us per call, cold operands, CUDA graph, min of {args.repeats} interleaved rounds')
    print(f'{"expr":>13} {"h":>4} {"r":>5} {"d":>5} {"b":>5} | ' + ' '.join(f'{n:>9}' for n in names) + ' | batched/legacy batched/cublaslt')
    ratios = {n: [] for n in names[1:]}
    for expr in exprs:
        for h, r, d in shapes:
            for b in (int(x) for x in args.bs.split(',')):
                torch.manual_seed(0)
                a0, nbytes = make_case(expr, b, h, r, d)
                copies = [a0] + [tuple(t.clone() for t in a0) for _ in range(max(2, min(32, math.ceil(400e6 / nbytes))) - 1)]
                times = {}
                for _ in range(args.repeats):
                    for name in names:
                        t = bench_graph(run(expr, name), copies) * 1e6
                        times[name] = min(times.get(name, t), t)
                for n in names[1:]:
                    ratios[n].append(times['batched'] / times[n])
                print(f'{expr:>13} {h:4} {r:5} {d:5} {b:5} | ' + ' '.join(f'{times[n]:9.1f}' for n in names) +
                      f' | {times["batched"] / times["legacy"]:14.2f} {times["batched"] / times["cublaslt"]:16.2f}', flush=True)
                del copies
    os.environ.pop('DG_SM120_BATCHED_LAUNCH', None)
    for n in names[1:]:
        r = ratios[n]
        print(f'batched / {n}: geomean {math.exp(sum(map(math.log, r)) / len(r)):.3f}, worst {max(r):.3f}, best {min(r):.3f}')
if __name__ == '__main__':
    main()
