# Block-scaled SM120 GEMMs (packed UE8M0: FP4, FP8 with 32/128-element scales) vs the FP32-scale DeepGEMM kernel
# and the TokenSpeed Triton / FlashInfer block-FP8 fallbacks, on DeepSeek-V3 TP-sharded dense linears.
#
# Run with the TokenSpeed venv and this DeepGEMM build (from a cwd outside the repo), e.g. via
#   sm120_remote.sh ts -m sm120_bench_blockscale --tp 8
#
# Timing is the decode serving condition: a CUDA graph over enough distinct weight copies to exceed L2, replayed,
# reported per GEMM (`bench_graph`). Activation scales are produced in the layout a fused quantizer would emit
# (packed once, outside the timed region), so the numbers compare kernels, not scale-transform launches.
import argparse
import math

import torch

import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp4, per_token_cast_to_fp8
from sm120_baseline import bench_graph

# DeepSeek-V3/R1 dense linears (name, N, K, shards N, shards K)
LINEARS = [
    ('q_a/kv_a',       2112, 7168, False, False),
    ('q_b',           24576, 1536, True, False),
    ('kv_b',          32768, 512, True, False),
    ('o_proj',         7168, 16384, False, True),
    ('shared gate_up', 4096, 7168, True, False),
    ('shared down',    7168, 2048, False, True),
    ('dense gate_up', 36864, 7168, True, False),
    ('dense down',     7168, 18432, False, True),
]
MS = [1, 8, 32, 64, 128, 256, 512, 2048, 4096]


def make_dg_format(mode, x_bf16):
    """Quantize `x` for the given DeepGEMM mode; scales are packed UE8M0 words (MN-major, TMA-aligned)"""
    if mode == 'fp4':
        q, sf = per_token_cast_to_fp4(x_bf16, use_ue8m0=True, gran_k=32)
    else:
        q, sf = per_token_cast_to_fp8(x_bf16, use_ue8m0=True, gran_k=32 if mode == 'ue8m0_g32' else 128)
    return q, deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--baselines', action='store_true', help='also time TokenSpeed Triton / FlashInfer block-FP8')
    args = parser.parse_args()
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS

    base = {}
    if args.baselines:
        from tokenspeed_kernel.ops.gemm.triton import triton_mm_fp8_blockscale
        from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale
        from tokenspeed_kernel.ops.quantization import quantize_fp8
        base['triton'] = lambda a, sa, b, sb, out: triton_mm_fp8_blockscale(a, b, sa, sb, torch.bfloat16, block_size=[128, 128], out=out)
        base['flashinfer'] = lambda a, sa, b, sb, out: flashinfer_mm_fp8_blockscale(a, b, sa, sb, torch.bfloat16, block_size=[128, 128], out=out)
        from tokenspeed_kernel.thirdparty.deepgemm import block_fp8_mm
        base['dg_fp32'] = lambda a, sa, b, sb, out: block_fp8_mm(a, b, sa, sb, out)

    modes = ['ue8m0_g128', 'ue8m0_g32', 'fp4']
    print(f'TP={args.tp}; times in us per GEMM (graph, cold weights); speedup = best baseline / mode')
    cols = (['triton', 'flashinfer', 'dg_fp32'] if args.baselines else []) + modes
    print(f'{"linear":>15} {"m":>5} {"n":>6} {"k":>6} | ' + ' '.join(f'{c:>10}' for c in cols) + ' | fp4 vs best FP8')
    for name, n, k, split_n, split_k in LINEARS:
        if args.only and name not in args.only.split(','):
            continue
        n = n // args.tp if split_n else n
        k = k // args.tp if split_k else k
        if k % 256:
            continue
        b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        copies = max(4, min(64, math.ceil(512e6 / (n * k))))
        weights = {}
        for mode in modes:
            q, sf = make_dg_format(mode, b_bf16)
            weights[mode] = [(q.clone(), sf.clone()) for _ in range(copies)]
        if args.baselines:
            bq, bsb = quantize_fp8(b_bf16, granularity='block', block_size=(128, 128))
            weights['base'] = [(bq.clone(), bsb.clone()) for _ in range(copies)]
        for m in ms:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            row, times = f'{name:>15} {m:5d} {n:6d} {k:6d} | ', {}
            if args.baselines:
                a, sa = quantize_fp8(x, granularity='token_group', group_size=128, scale_encoding='float32')
                for bname, fn in base.items():
                    if bname == 'flashinfer' and n % 128:
                        row += f'{"n/a":>10} '
                        continue
                    try:
                        fn(a, sa, bq, bsb, out)
                        times[bname] = bench_graph(lambda w: fn(a, sa, w[0], w[1], out), weights['base'])
                        row += f'{times[bname] * 1e6:10.1f} '
                    except Exception:  # noqa: BLE001
                        row += f'{"err":>10} '
            for mode in modes:
                a, sa = make_dg_format(mode, x)
                gran_k = 128 if mode == 'ue8m0_g128' else 32
                fn = lambda a, sa, b, sb, out, gk=gran_k: deep_gemm.fp8_fp4_gemm_nt((a, sa), (b, sb), out, recipe=(1, 1, gk))
                fn(a, sa, *weights[mode][0], out)
                times[mode] = bench_graph(lambda w: fn(a, sa, w[0], w[1], out), weights[mode])
                row += f'{times[mode] * 1e6:10.1f} '
            fp8_best = min(t for key, t in times.items() if key != 'fp4')
            row += f'| {fp8_best / times["fp4"]:5.2f}x'
            print(row, flush=True)


if __name__ == '__main__':
    main()
