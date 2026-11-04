# SM120 MXFP4 (E2M1 + UE8M0 per 32) GEMM: DeepGEMM vs the fastest FlashInfer `mm_fp4` backend that runs MXFP4 on
# SM120 (each backend is tried per point; those that reject MXFP4 or SM120 are skipped). Both sides use PDL, as
# served (FlashInfer's `enable_pdl` defaults on; DeepGEMM's is `deep_gemm.set_pdl`). Same decode serving condition as `sm120_bench_blockscale.py`: CUDA graph over
# enough distinct weight copies to exceed L2, reported per GEMM; the activation is pre-quantized in each library's own
# scale layout, so only the GEMM is timed.
#
# Run with the TokenSpeed venv and this DeepGEMM build (from a cwd outside the repo):
#   sm120_remote.sh ts -m sm120_fp4_vs_flashinfer
import argparse
import math

import torch

import deep_gemm
from deep_gemm.utils.math import per_token_cast_to_fp4
from flashinfer import mxfp4_quantize
from flashinfer.gemm import mm_fp4
from sm120_baseline import bench_graph

# DeepSeek-V3 dense linears at TP8 (name, N, K)
LINEARS = [('q_a/kv_a', 2112, 7168), ('q_b', 3072, 1536), ('kv_b', 4096, 512), ('o_proj', 7168, 2048),
           ('shared gate_up', 512, 7168), ('dense gate_up', 4608, 7168), ('dense down', 7168, 2304)]
MS = [1, 8, 32, 64, 128, 256, 512, 2048, 4096]
# GPU 7 is shared with other jobs: take the best of a few interleaved runs per side
NUM_REPEATS = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--no-pdl', action='store_true')
    parser.add_argument('--only', type=str, default=None)
    args = parser.parse_args()
    deep_gemm.set_pdl(not args.no_pdl)
    backends = ['cudnn', 'b12x', 'cute-dsl', 'cutedsl_low_latency', 'cutlass', 'trtllm']
    working = {}
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    print(f'{"linear":>15} {"m":>5} {"n":>6} {"k":>6} | {"flashinfer":>10} {"backend":>10} {"deepgemm":>10} | speedup')
    for name, n, k in LINEARS:
        if args.only and name not in args.only.split(','):
            continue
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        copies = max(4, min(64, math.ceil(256e6 / (n * k))))
        dg_q, dg_sf = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
        dg_w = [(dg_q.clone(), deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(dg_sf)) for _ in range(copies)]
        fi_q, fi_sf = mxfp4_quantize(w)
        fi_w = [(fi_q.clone(), fi_sf.clone()) for _ in range(copies)]
        for m in ms:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            dg_a, dg_asf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
            dg_asf = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(dg_asf)
            fi_a, fi_asf = mxfp4_quantize(x)
            row = f'{name:>15} {m:5d} {n:6d} {k:6d} | '
            # Each side's candidates, as graph-replay closures
            runs = {}
            for backend in backends:
                if working.get(backend) is False:
                    continue
                try:
                    run_fi = lambda wq, wsf, be=backend: mm_fp4(fi_a, wq.T, fi_asf, wsf.T, None, torch.bfloat16, out,
                                                                 block_size=32, use_nvfp4=False, backend=be,
                                                                 enable_pdl=not args.no_pdl)
                    run_fi(*fi_w[0])
                    working[backend] = True
                    runs[backend] = (run_fi, fi_w)
                except Exception:  # noqa: BLE001 (backend does not support MXFP4 on SM120)
                    if working.get(backend) is None:
                        working[backend] = False
            run_dg = lambda wq, wsf: deep_gemm.fp8_fp4_gemm_nt((dg_a, dg_asf), (wq, wsf), out, recipe=(1, 1, 32))
            run_dg(*dg_w[0])
            runs['deepgemm'] = (run_dg, dg_w)
            # Interleaved rounds, best per candidate: the GPU is shared, and contention hitting a stretch of back-to-back
            # runs of one side would otherwise bias the comparison
            best = {impl: float('inf') for impl in runs}
            for _ in range(NUM_REPEATS):
                for impl, (fn, ws) in runs.items():
                    best[impl] = min(best[impl], bench_graph(lambda w_, fn=fn: fn(w_[0], w_[1]), ws))
            t_dg = best.pop('deepgemm')
            t_fi, fi_backend = (min(best.values()), min(best, key=best.get)) if best else (None, None)
            if t_fi is None:
                row += f'{"n/a":>10} {"":>10} {t_dg * 1e6:10.1f} |'
            else:
                row += f'{t_fi * 1e6:10.1f} {fi_backend:>10} {t_dg * 1e6:10.1f} | {t_fi / t_dg:5.2f}x{"  LOSE" if t_fi < t_dg else ""}'
            print(row, flush=True)


if __name__ == '__main__':
    main()

