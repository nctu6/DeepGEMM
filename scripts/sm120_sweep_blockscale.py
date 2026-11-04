# Sweep every legal tile layout (and forced stream-K on few-tile shapes) of the SM120 block-scaled FP4 GEMM on
# DeepSeek-V3 TP8 dense linears, under the serving condition: CUDA graph replay over enough distinct weight copies to
# exceed L2, with PDL. Writes a JSON list of {m, n, k, times: {"BMxBNxBK/skS": seconds}} for `sm120_fit_blockscale.py`.
#
# Run with the TokenSpeed venv and this DeepGEMM build (cwd outside the repo), e.g.
#   sm120_remote.sh ts -m sm120_sweep_blockscale --out /tmp/sweep_fp4.json
# GPU 7 is shared: the sweep waits for it to be idle before every shape (`--idle-secs`).
import argparse
import json
import math
import os
import subprocess
import time

import torch

import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp4
from sm120_baseline import bench_graph

LINEARS = [('q_a/kv_a', 2112, 7168), ('q_b', 3072, 1536), ('kv_b', 4096, 512), ('o_proj', 7168, 2048),
           ('shared gate_up', 512, 7168), ('dense gate_up', 4608, 7168), ('dense down', 7168, 2304)]
MS = [1, 8, 32, 64, 128, 256, 512, 2048, 4096]
NUM_SMS = 188


def gpu_idle_for(seconds, budget):
    quiet, waited = 0, 0
    while quiet < seconds and waited < budget:
        util = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits', '-i', '7'],
                              capture_output=True, text=True).stdout.strip()
        quiet = quiet + 1 if util == '0' else 0
        time.sleep(1)
        waited += 1
    return quiet >= seconds


def layouts_for(m):
    out = []
    for bm in (16, 32, 64, 128):
        if not (bm == 16 or bm < 2 * m):
            continue
        for bn in (32, 64, 128, 256):
            for bk in (128, 64):
                out.append((bm, bn, bk))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--idle-secs', type=int, default=10)
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--ms', type=str, default=None)
    args = parser.parse_args()
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    deep_gemm.set_pdl(True)
    os.environ['DG_SM120_AUTOTUNE'] = '0'
    records = []
    for name, n, k in LINEARS:
        if args.only and name not in args.only.split(','):
            continue
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        copies = max(4, min(64, math.ceil(256e6 / (n * k))))
        q, sf = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
        weights = [(q.clone(), deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)) for _ in range(copies)]
        for m in ms:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            a, asf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
            asf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(asf)
            out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            fn = lambda w_: deep_gemm.fp8_fp4_gemm_nt((a, asf), (w_[0], w_[1]), out, recipe=(1, 1, 32))
            idle = gpu_idle_for(args.idle_secs, 900)
            times = {}
            variants = []
            for bm, bn, bk in layouts_for(m):
                tiles = math.ceil(m / bm) * math.ceil(n / bn)
                variants.append((f'{bm}x{bn}x{bk}', '0'))
                if tiles * 4 <= NUM_SMS:
                    variants.append((f'{bm}x{bn}x{bk}', '2'))
            # Interleaved rounds, best per variant (the GPU is shared)
            for _ in range(args.repeats):
                for layout, sk in variants:
                    key = f'{layout}/sk{sk}'
                    os.environ['DG_SM120_FORCE_LAYOUT'] = layout
                    os.environ['DG_SM120_FORCE_STREAM_K'] = sk
                    try:
                        fn(weights[0])
                        t = bench_graph(fn, weights)
                        times[key] = min(times.get(key, float('inf')), t)
                    except Exception:  # noqa: BLE001 (illegal layout for this shape)
                        times.setdefault(key, None)
            times = {key: t for key, t in times.items() if t is not None}
            os.environ.pop('DG_SM120_FORCE_LAYOUT', None)
            os.environ.pop('DG_SM120_FORCE_STREAM_K', None)
            fn(weights[0])
            times['auto'] = min(bench_graph(fn, weights) for _ in range(args.repeats))
            best = min((key for key in times if key != 'auto'), key=times.get)
            print(f'{name:>15} m={m:5d} n={n:5d} k={k:5d} idle={int(idle)} best={best:>16} {times[best] * 1e6:7.2f} us '
                  f'auto {times["auto"] * 1e6:7.2f} us ({times["auto"] / times[best]:.3f}x)', flush=True)
            records.append({'name': name, 'm': m, 'n': n, 'k': k, 'times': times})
            json.dump(records, open(args.out, 'w'))


if __name__ == '__main__':
    main()
