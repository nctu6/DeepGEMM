"""Benchmark of MN-major FP8 operands on SM120: the in-kernel `ldmatrix.trans.b8` against its alternatives.

Variants, per operand configuration, all with the FP8 FP32-scale (1D2D) or UE8M0 block-scaled (1D1D) kernel:
  kernel    MN-major operands read in-kernel (`DG_SM120_MN_MAJOR_FP8=2`)
  explicit  (1) an explicit byte-transpose kernel, then the K-major kernel (`DG_SM120_MN_MAJOR_FP8=1`)
  kmajor    (2) the K-major kernel on pre-transposed data: the upper bound
  auto      the default routing (per operand: in-kernel or host transpose, see `sm120_fp8_mn_major_use_in_kernel`)
  triton    (3) TokenSpeed's Triton block-FP8 kernel, called on the strided MN-major operands (it is stride-generic; the
            wrapper's contiguity asserts are bypassed), 1D2D only
  fi        (3) FlashInfer's block-FP8 GEMM on a pre-transposed copy (needs K-major), 1D2D only; transpose time not counted
Timing: a CUDA graph over enough distinct operand copies to exceed L2 (cold weights, as in decode serving), replayed;
variants are interleaved over `--repeats` rounds and the minimum is kept (filters interference). Run with the TokenSpeed
venv for the Triton/FlashInfer baselines:
  DG_GPU=7 sm120_remote.sh ts scripts/sm120_bench_mn_major_fp8.py --configs b,a,ab --ms 1,16,64,256,1024,4096
"""
import argparse
import json
import math
import os

import torch

import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
from sm120_baseline import bench_graph

# DeepSeek-V3 dense linears at TP8 (N, K), then square shapes
DSV3 = [(2112, 7168), (576, 7168), (3072, 1536), (4096, 512), (7168, 2048), (4608, 7168), (7168, 2304), (512, 7168), (7168, 256)]
SQUARE = [(2048, 2048), (4096, 4096), (8192, 8192)]


def mn_major(x):
    return x.mT.contiguous().mT


def set_mode(mode):
    """`mode`: None (default routing), 2 = in-kernel, 1 = explicit transpose then the K-major kernel"""
    if mode is None:
        os.environ.pop('DG_SM120_MN_MAJOR_FP8', None)
    else:
        os.environ['DG_SM120_MN_MAJOR_FP8'] = str(mode)


def make_triton():
    try:
        from tokenspeed_kernel.ops.gemm._triton import fp8 as tfp8
    except ImportError:
        return None, None
    # The wrapper only asserts contiguity; the kernel takes strides, so MN-major operands work as they are
    def relaxed_prepare(A, B, As, Bs, block_size, output_dtype=torch.float16, out=None):
        N, K = B.shape
        return A.shape[0], N, K, out if out is not None else A.new_empty((A.shape[0], N), dtype=output_dtype)
    tfp8.prepare_block_fp8_matmul_inputs = relaxed_prepare

    def run(a, sa, b, sb, out):
        return tfp8.w8a8_block_fp8_matmul_triton(a, b, sa, sb, block_size=[128, 128], output_dtype=torch.bfloat16, out=out)

    fi = None
    try:
        from tokenspeed_kernel.ops.gemm.flashinfer import flashinfer_mm_fp8_blockscale

        def fi(a, sa, b, sb, out):
            return flashinfer_mm_fp8_blockscale(a, b, sa, sb, torch.bfloat16, block_size=[128, 128], out=out)
    except ImportError:
        pass
    return run, fi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', type=str, default='b,a,ab', help='MN-major operands: b (weights), a (activations), ab (both)')
    parser.add_argument('--kernel', type=str, default='1d2d', choices=['1d2d', '1d1d'])
    parser.add_argument('--ms', type=str, default='1,16,64,256,1024,4096')
    parser.add_argument('--nks', type=str, default=None, help='comma-separated NxK; default: DeepSeek-V3 TP8 + square')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()
    nks = [tuple(int(y) for y in x.split('x')) for x in args.nks.split(',')] if args.nks else DSV3 + SQUARE
    ms = [int(x) for x in args.ms.split(',')]
    triton_run, fi_run = make_triton() if args.kernel == '1d2d' else (None, None)
    records = []
    names = ['kernel', 'explicit', 'kmajor', 'auto'] +  (['triton'] if triton_run else []) + (['fi'] if fi_run else [])
    print(f'kernel={args.kernel} variants={names} (us per GEMM, cold operands, CUDA graph, min of {args.repeats} interleaved rounds)')
    print(f'{"cfg":>3} {"m":>5} {"n":>6} {"k":>6} | ' + ' '.join(f'{n:>9}' for n in names) + ' | explicit/kernel  kmajor/kernel  auto/best(kernel,explicit)  best-of-triton/fi / auto')
    for cfg in args.configs.split(','):
        for n, k in nks:
            for m in ms:
                if cfg in ('a', 'ab') and m % 16 != 0:
                    continue   # TMA needs 16-byte MN rows; the host takes the explicit route anyway
                torch.manual_seed(0)
                a_bf16 = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
                if args.kernel == '1d2d':
                    qa, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
                    qb, sfb = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)
                    extra = {}
                else:
                    qa, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=128)
                    qb, sfb = per_token_cast_to_fp8(b_bf16, use_ue8m0=True, gran_k=128)
                    sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
                    sfb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfb)
                    extra = dict(recipe=(1, 1, 128))
                # Distinct copies of the operands that are "cold" in this configuration
                varying_bytes = (n * k if 'b' in cfg else 0) + (m * k if 'a' in cfg else 0)
                num_copies = max(4, min(64, math.ceil(512e6 / max(varying_bytes, 1))))
                d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
                a_mn = mn_major(qa) if 'a' in cfg else qa
                b_mn = mn_major(qb) if 'b' in cfg else qb
                copies_mn = [(a_mn.clone() if 'a' in cfg else a_mn, b_mn.clone() if 'b' in cfg else b_mn) for _ in range(num_copies)]
                copies_k = [(qa.clone() if 'a' in cfg else qa, qb.clone() if 'b' in cfg else qb) for _ in range(num_copies)]
                # Keep the clones' MN-major strides (clone() preserves them for non-overlapping dense strides)
                assert all((x.stride(-2) == 1) == ('a' in cfg) and (y.stride(-2) == 1) == ('b' in cfg) for x, y in copies_mn)

                def gemm(pair):
                    deep_gemm.fp8_fp4_gemm_nt((pair[0], sfa), (pair[1], sfb), d, disable_ue8m0_cast=(args.kernel == '1d2d'), **extra)

                def tr(pair):
                    triton_run(pair[0], sfa, pair[1], sfb, d)

                def fi(pair):
                    fi_run(pair[0], sfa, pair[1], sfb, d)

                variants = {'kernel': (2, gemm, copies_mn), 'explicit': (1, gemm, copies_mn), 'kmajor': (None, gemm, copies_k), 'auto': (None, gemm, copies_mn)}
                if triton_run:
                    variants['triton'] = (None, tr, copies_mn)
                if fi_run and n % 128 == 0:
                    variants['fi'] = (None, fi, copies_k)
                times = {}
                ok = {}
                for _ in range(args.repeats):
                    for name, (mode, fn, copies) in variants.items():
                        if ok.get(name) is False:
                            continue
                        set_mode(mode)
                        try:
                            t = bench_graph(fn, copies)
                        except Exception as ex:   # noqa: BLE001 (report unsupported variants, keep going)
                            ok[name] = False
                            print(f'  [{name}] {type(ex).__name__}: {str(ex)[:200]}')
                            continue
                        times[name] = min(times.get(name, t), t)
                set_mode(None)
                row = f'{cfg:>3} {m:5} {n:6} {k:6} | ' + ' '.join(f'{times[x] * 1e6:9.1f}' if x in times else f'{"n/a":>9}' for x in names)
                others = [times[x] for x in ('triton', 'fi') if x in times]
                row += f' | {times["explicit"] / times["kernel"]:6.2f}x {times["kmajor"] / times["kernel"]:12.2f}' + \
                       f' {min(times["kernel"], times["explicit"]) / times["auto"]:20.2f}' + \
                       (f' {min(others) / times["auto"]:14.2f}' if others else '')
                print(row, flush=True)
                records.append({'cfg': cfg, 'm': m, 'n': n, 'k': k, 'times': times})
                if args.json:
                    with open(args.json, 'w') as f:
                        json.dump(records, f)


if __name__ == '__main__':
    main()
