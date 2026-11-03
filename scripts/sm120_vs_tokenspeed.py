# Phase 4.5: DeepGEMM vs. the TokenSpeed sm120 fallbacks for dense block-FP8 linears
#
# Run with the TokenSpeed venv and this DeepGEMM build on PYTHONPATH (from a cwd outside the repo), e.g.
#   PYTHONPATH=$DG/build312/lib.linux-x86_64-cpython-312:$DG/scripts \
#       /workspace/nctu6/infra/tokenspeed/.venv/bin/python -m sm120_vs_tokenspeed
#
# Every implementation is called exactly as TokenSpeed calls it: canonical FP32 scales
# (A: [M, K / 128] row-major from the activation quantizer, B: [N / 128, K / 128] from the checkpoint),
# and BF16 output, so DeepGEMM's per-call scale transforms are included in its time.
#
# Two timings per implementation:
#  - graph: a CUDA graph that runs the GEMM over enough distinct weight copies to exceed L2, replayed;
#    reported per GEMM. This is the decode serving condition (no launch overhead, cold weights).
#  - cold: median of single calls with a 512 MB L2 flush before each, summed GPU kernel time.
import argparse
import math

import torch

import deep_gemm
from tokenspeed_kernel.ops.quantization import quantize_fp8
from tokenspeed_kernel.thirdparty.deepgemm import block_fp8_mm

from sm120_baseline import bench_cold, bench_graph

# DeepSeek-V3/R1 dense block-FP8 linears (N, K) for TP1, with the TP split applied by `--tp`
#  - (n_split, k_split): which dimension a TP rank shards
NK_TP = [
    ('q_a/kv_a fused', 2112, 7168, False, False),
    ('q_b',            24576, 1536, True, False),
    ('kv_b',           32768, 512, True, False),
    ('o_proj',         7168, 16384, False, True),
    ('shared gate_up', 4096, 7168, True, False),
    ('shared down',    7168, 2048, False, True),
    ('dense gate_up',  36864, 7168, True, False),
    ('dense down',     7168, 18432, False, True),
]
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def make_impls():
    impls = {}

    def deepgemm(a, sa, b, sb, out):
        return block_fp8_mm(a, b, sa, sb, out)
    impls['deepgemm'] = deepgemm

    from tokenspeed_kernel.ops.gemm.triton import triton_mm_fp8_blockscale

    def triton(a, sa, b, sb, out):
        return triton_mm_fp8_blockscale(a, b, sa, sb, torch.bfloat16, block_size=[128, 128], out=out)
    impls['triton'] = triton

    try:
        from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale

        def flashinfer(a, sa, b, sb, out):
            return flashinfer_mm_fp8_blockscale(a, b, sa, sb, torch.bfloat16, block_size=[128, 128], out=out)
        impls['flashinfer'] = flashinfer
    except ImportError:
        pass
    return impls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--only', type=str, default=None, help='comma-separated linear names')
    parser.add_argument('--no-cold', action='store_true')
    parser.add_argument('--impls', type=str, default=None, help='comma-separated implementation names')
    args = parser.parse_args()
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    impls = make_impls()
    if args.impls:
        impls = {k: v for k, v in impls.items() if k in args.impls.split(',')}
    names = list(impls)
    reported = set()
    print(f'TP={args.tp}, implementations: {names}')

    header = f'{"linear":>15} {"m":>5} {"n":>6} {"k":>6} |'
    for name in names:
        header += f' {name + " graph":>16} {"cold":>7} |'
    header += ' best-other/DG (graph)'
    print(header)

    for linear, n, k, split_n, split_k in NK_TP:
        if args.only and linear not in args.only.split(','):
            continue
        n = n // args.tp if split_n else n
        k = k // args.tp if split_k else k
        b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        b, sb = quantize_fp8(b_bf16, granularity='block', block_size=(128, 128))
        num_copies = max(4, min(64, math.ceil(512e6 / (n * k))))
        weights = [(b.clone(), sb.clone()) for _ in range(num_copies)]
        for m in ms:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            a, sa = quantize_fp8(x, granularity='token_group', group_size=128, scale_encoding='float32')
            ref = x.float() @ b_bf16.float().t()
            out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            row = f'{linear:>15} {m:5} {n:6} {k:6} |'
            times = {}
            for name, impl in impls.items():
                # TokenSpeed only selects FlashInfer for N % 128 == 0 (`n_align`), and it silently
                # produces nothing under CUDA graphs otherwise
                if name == 'flashinfer' and n % 128 != 0:
                    row += f' {"n/a (n_align)":>16} {"":>7} |'
                    continue
                try:
                    impl(a, sa, b, sb, out)
                    diff = deep_gemm.testing.calc_diff(out, ref)
                    assert diff < 2e-3, f'{name}: diff={diff}'
                    t_graph = bench_graph(lambda w: impl(a, sa, w[0], w[1], out), weights)
                    t_cold = float('nan') if args.no_cold else bench_cold(lambda: impl(a, sa, b, sb, out))
                    times[name] = t_graph
                    row += f' {t_graph * 1e6:16.1f} {t_cold * 1e6:7.1f} |'
                except Exception as ex:  # noqa: BLE001 (report unsupported shapes and keep going)
                    row += f' {"n/a (" + type(ex).__name__ + ")":>16} {"":>7} |'
                    if (name, type(ex).__name__) not in reported:
                        reported.add((name, type(ex).__name__))
                        print(f'  [{name}] {type(ex).__name__}: {str(ex)[:300]}', flush=True)
            others = [t for name, t in times.items() if name != 'deepgemm']
            if 'deepgemm' in times and others:
                ratio = min(others) / times['deepgemm']
                row += f' {ratio:5.2f}{"  LOSE" if ratio < 1.0 else ""}'
            print(row, flush=True)


if __name__ == '__main__':
    main()
