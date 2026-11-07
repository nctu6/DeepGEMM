"""Benchmark of the MN-major `fp8_einsum` expressions on SM120 against their BF16 alternatives.
  bhd,hdr->bhr   FP8 (K-major A, MN-major B, 128x128 scales, BF16 D)
  bhd,bhr->hdr   FP8 (both MN-major, per-channel 1x128 scales, FP32 D accumulating)
  bhr,hdr->bhd   FP8 (K-major operands, 128x128 scales, BF16 D; through `fp8_bmm`)
Variants (us per call):
  fp8       `deep_gemm.fp8_einsum`
  cublaslt  `deep_gemm.einsum` on BF16 inputs with deterministic mode off (cuBLASLt)
  loop      a per-head `torch.matmul` on BF16 (for `bhd,bhr->hdr` plus an FP32 add per head)
  einsum    `torch.einsum` on BF16 (for `bhd,bhr->hdr` plus an FP32 add)
  best      the fastest of cublaslt / loop / einsum, and `fp8 / best` (the target is <= 1.03)
Timing: a CUDA graph over enough distinct operand copies to exceed L2 (cold weights), replayed; variants are interleaved
over `--repeats` rounds and the minimum is kept. With `--profile`, prints the GPU kernels one `fp8` call launches.
  DG_GPU=7 sm120_remote.sh dg scripts/sm120_bench_fp8_einsum.py --exprs hdr_bhr,bhr_hdr --bs 4,32,128,4096,8192
"""
import argparse
import math
import torch
import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp8
from sm120_baseline import bench_graph

SHAPES = [(8, 4096, 1024), (128, 512, 128)]   # (h, r, d)


def make_bhd_hdr_bhr(b, h, r, d):
    x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
    xq, xs = per_token_cast_to_fp8(x.view(-1, d), use_ue8m0=True)
    yq = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    ys = torch.empty((h, math.ceil(d / 128), math.ceil(r / 128)), device='cuda', dtype=torch.float)
    for i in range(h):
        yq[i], ys[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)
    z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
    x8, y8 = (xq.view(b, h, d), xs.view(b, h, -1)), (yq, ys)
    return dict(
        bytes=(y.numel() * 2 + x.numel() * 2),
        args=(x, y, x8, y8, z),
        fp8=lambda a: deep_gemm.fp8_einsum('bhd,hdr->bhr', a[2], a[3], a[4]),
        cublaslt=lambda a: deep_gemm.einsum('bhd,hdr->bhr', a[0], a[1], a[4]),
        loop=lambda a: [torch.matmul(a[0][:, i], a[1][i], out=a[4][:, i]) for i in range(h)],
        einsum=lambda a: torch.einsum('bhd,hdr->bhr', a[0], a[1]),
    )


def make_bhd_bhr_hdr(b, h, r, d):
    x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
    xq, xs = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=True)
    yq, ys = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=True)
    x8 = (xq.view(b, h, d), xs.view(math.ceil(b / 128), h, d))
    y8 = (yq.view(b, h, r), ys.view(math.ceil(b / 128), h, r))
    z = torch.zeros((h, d, r), device='cuda', dtype=torch.float32)
    return dict(
        bytes=(x.numel() + y.numel()) * 2,
        args=(x, y, x8, y8, z),
        fp8=lambda a: deep_gemm.fp8_einsum('bhd,bhr->hdr', a[2], a[3], a[4], c=a[4], recipe=(1, 1, 128)),
        cublaslt=lambda a: deep_gemm.einsum('bhd,bhr->hdr', a[0], a[1], a[4], c=a[4]),
        loop=lambda a: [a[4][i].add_(torch.matmul(a[0][:, i].T, a[1][:, i])) for i in range(h)],
        einsum=lambda a: a[4].add_(torch.einsum('bhd,bhr->hdr', a[0], a[1])),
    )


def make_bhr_hdr_bhd(b, h, r, d):
    x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
    xq, xs = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True)
    yq = torch.empty_like(y, dtype=torch.float8_e4m3fn)
    ys = torch.empty((h, math.ceil(d / 128), math.ceil(r / 128)), device='cuda', dtype=torch.float)
    for i in range(h):
        yq[i], ys[i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)
    z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
    x8, y8 = (xq.view(b, h, r), xs.view(b, h, -1)), (yq, ys)
    return dict(
        bytes=(y.numel() * 2 + x.numel() * 2),
        args=(x, y, x8, y8, z),
        fp8=lambda a: deep_gemm.fp8_einsum('bhr,hdr->bhd', a[2], a[3], a[4]),
        cublaslt=lambda a: deep_gemm.einsum('bhr,hdr->bhd', a[0], a[1], a[4]),
        loop=lambda a: [torch.matmul(a[0][:, i], a[1][i].T, out=a[4][:, i]) for i in range(h)],
        einsum=lambda a: torch.einsum('bhr,hdr->bhd', a[0], a[1]),
    )


EXPRS = {'hdr_bhr': make_bhd_hdr_bhr, 'bhr_hdr': make_bhd_bhr_hdr, 'bhr_bhd': make_bhr_hdr_bhd}


def clone_args(args):
    return tuple(tuple(t.clone() for t in a) if isinstance(a, tuple) else a.clone() for a in args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exprs', type=str, default='hdr_bhr,bhr_hdr,bhr_bhd', help='hdr_bhr: bhd,hdr->bhr; bhr_hdr: bhd,bhr->hdr; bhr_bhd: bhr,hdr->bhd')
    parser.add_argument('--bs', type=str, default='4,32,128,4096,8192')
    parser.add_argument('--shapes', type=str, default=None, help='comma-separated hxrxd; default: 8x4096x1024,128x512x128')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    shapes = [tuple(int(y) for y in s.split('x')) for s in args.shapes.split(',')] if args.shapes else SHAPES
    deep_gemm.use_deterministic_algorithms(False)
    names = ['fp8', 'cublaslt', 'loop', 'einsum']
    print(f'us per call, cold operands, CUDA graph, min of {args.repeats} interleaved rounds')
    print(f'{"expr":>8} {"h":>4} {"r":>5} {"d":>5} {"b":>5} | ' + ' '.join(f'{n:>9}' for n in names) + ' | best-of-bf16  fp8/best')
    worst = {}
    for expr in args.exprs.split(','):
        for h, r, d in shapes:
            for b in (int(x) for x in args.bs.split(',')):
                if expr == 'bhr_hdr' and b % 128 != 0:
                    continue   # per-channel 1x128 scales need whole 128-row blocks along the contracted `b`
                torch.manual_seed(0)
                case = EXPRS[expr](b, h, r, d)
                num_copies = max(2, min(32, math.ceil(400e6 / case['bytes'])))
                copies = [case['args']] + [clone_args(case['args']) for _ in range(num_copies - 1)]
                if args.profile:
                    case['fp8'](copies[0])
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                        case['fp8'](copies[0])
                        torch.cuda.synchronize()
                    evs = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                                 key=lambda e: e.time_range.start)
                    agg = {}
                    for e in evs:
                        k = e.name[:60]
                        n, t = agg.get(k, (0, 0.0))
                        agg[k] = (n + 1, t + e.time_range.elapsed_us())
                    span = evs[-1].time_range.end - evs[0].time_range.start
                    print(f'  [profile {expr} {h=} {r=} {d=} {b=}] span {span:.1f} us')
                    for k, (n, t) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
                        print(f'    {n:4d} x {t / n:8.2f} us  = {t:8.1f} us  {k}')
                times = {}
                for _ in range(args.repeats):
                    for name in names:
                        t = bench_graph(case[name], copies) * 1e6
                        times[name] = min(times.get(name, t), t)
                best = min(times[n] for n in names[1:])
                ratio = times['fp8'] / best
                worst[expr] = max(worst.get(expr, 0), ratio)
                print(f'{expr:>8} {h:4} {r:5} {d:5} {b:5} | ' + ' '.join(f'{times[n]:9.1f}' for n in names) +
                      f' | {best:12.1f}  {ratio:8.2f}', flush=True)
                del copies, case
    print('worst fp8/best per expr:', {k: round(v, 2) for k, v in worst.items()})


if __name__ == '__main__':
    main()
