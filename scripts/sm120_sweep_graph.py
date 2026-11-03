# Sweep SM120 FP8 GEMM layouts (and optionally stream-K) under CUDA graphs with cold weights (tuning helper)
#
# Prints a summary line per shape and, with `--json`, dumps every measurement for offline heuristic fitting
# (`scripts/sm120_fit_heuristic.py`). Each point is the minimum over `--repeats` graph replays, which filters
# interference from other processes sharing the GPU.
import argparse
import json
import math
import os
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

from sm120_baseline import bench_graph

LAYOUTS = ['16x64', '16x128', '16x256', '32x64', '32x128', '32x256', '64x64', '64x128', '128x64', '128x128']
KEYS = ('DG_SM120_FORCE_LAYOUT', 'DG_SM120_FORCE_STREAM_K')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=str, required=True)
    parser.add_argument('--nks', type=str, required=True)
    parser.add_argument('--sk', type=str, default='0')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--json', type=str, default=None)
    args = parser.parse_args()
    records = []
    for nk in args.nks.split(','):
        n, k = (int(x) for x in nk.split('x'))
        b = per_block_cast_to_fp8(torch.randn((n, k), device='cuda', dtype=torch.bfloat16), use_ue8m0=False)
        copies = [(b[0].clone(), b[1].clone()) for _ in range(max(4, min(64, math.ceil(512e6 / (n * k)))))]
        for m in (int(x) for x in args.ms.split(',')):
            a = per_token_cast_to_fp8(torch.randn((m, k), device='cuda', dtype=torch.bfloat16), use_ue8m0=False)
            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)

            # Candidates, measured round-robin so that interference from other GPU tenants hits all of them alike
            candidates = [('auto', None, None)]
            for layout in LAYOUTS:
                bm = int(layout.split('x')[0])
                if bm >= 2 * max(m, 16) and bm > 16:
                    continue
                for sk in args.sk.split(','):
                    candidates.append((f'{layout}/sk{sk}', layout, sk))
            results = {}
            for _ in range(args.repeats):
                for name, layout, sk in candidates:
                    for key in KEYS:
                        os.environ.pop(key, None)
                    if layout is not None:
                        os.environ['DG_SM120_FORCE_LAYOUT'] = layout
                        os.environ['DG_SM120_FORCE_STREAM_K'] = sk
                    try:
                        t = bench_graph(lambda w: deep_gemm.fp8_gemm_nt(a, w, d), copies)
                        results[name] = min(results.get(name, t), t)
                    except RuntimeError:
                        pass
            for key in KEYS:
                os.environ.pop(key, None)
            records.append({'m': m, 'n': n, 'k': k, 'times': results})
            top = sorted((v, l) for l, v in results.items() if l != 'auto')
            row = ' '.join(f'{l}:{v * 1e6:.1f}' for v, l in top[:6])
            print(f'm={m:5} n={n:6} k={k:6} best={top[0][1]:>12} {top[0][0] * 1e6:7.1f}us '
                  f'auto={results["auto"] * 1e6:7.1f}us ({results["auto"] / top[0][0]:.2f}x) | {row}', flush=True)
            if args.json:
                with open(args.json, 'w') as f:
                    json.dump(records, f)


if __name__ == '__main__':
    main()
