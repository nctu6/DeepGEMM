# SM120 MXFP4 (E2M1 + UE8M0 per 32) or, with `--nvfp4`, NVFP4 (E2M1 + UE4M3 per 16) GEMM: DeepGEMM vs the fastest
# FlashInfer `mm_fp4` backend that runs the format on SM120 (each backend is tried per point; those that reject it are
# skipped), plus cuBLASLt for NVFP4. Both sides use PDL, as
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
from deep_gemm.utils.math import ceil_div, per_token_cast_to_fp4, per_token_cast_to_nvfp4
from flashinfer import SfLayout, mxfp4_quantize, nvfp4_quantize
from flashinfer.gemm import mm_fp4
from sm120_baseline import bench_graph

# DeepSeek-V3 dense linears at TP8 (name, N, K)
LINEARS = [('q_a/kv_a', 2112, 7168), ('q_b', 3072, 1536), ('kv_b', 4096, 512), ('o_proj', 7168, 2048),
           ('shared gate_up', 512, 7168), ('dense gate_up', 4608, 7168), ('dense down', 7168, 2304)]
MS = [1, 8, 32, 64, 128, 256, 512, 2048, 4096]
# GPU 7 is shared with other jobs: take the best of a few interleaved runs per side
NUM_REPEATS = 5


def to_cublaslt_nvfp4_sf(sf: torch.Tensor) -> torch.Tensor:
    """FP32 E4M3 scale values `[MN, K / 16]` -> cuBLASLt VEC16 UE4M3 tiled layout `[ceil(MN / 128), ceil(K / 64), 32, 4, 4]`"""
    ue4m3 = sf.to(torch.float8_e4m3fn).view(torch.uint8)
    mn, sf_k = ue4m3.shape
    num_mn_tiles, num_k_tiles = ceil_div(mn, 128), ceil_div(sf_k, 4)
    padded = torch.zeros((num_mn_tiles * 128, num_k_tiles * 4), device=sf.device, dtype=torch.uint8)
    padded[:mn, :sf_k] = ue4m3
    return padded.view(num_mn_tiles, 4, 32, num_k_tiles, 4).permute(0, 3, 2, 1, 4).contiguous()


def quantize_dg(x: torch.Tensor, nvfp4: bool):
    if nvfp4:
        q, sf = per_token_cast_to_nvfp4(x)
        return q, deep_gemm.get_mn_major_tma_aligned_packed_ue4m3_tensor(sf), sf
    q, sf = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
    return q, deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sf), sf


def quantize_fi(x: torch.Tensor, nvfp4: bool):
    """FlashInfer's quantizer: NVFP4 takes a per-tensor global scale (returned, the GEMM's alpha is its inverse)"""
    if nvfp4:
        global_sf = (448 * 6) / x.float().abs().nan_to_num().max()
        q, sf = nvfp4_quantize(x, global_sf, sfLayout=SfLayout.layout_128x4, do_shuffle=False)
        return q, sf, global_sf
    q, sf = mxfp4_quantize(x)
    return q, sf, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--no-pdl', action='store_true')
    parser.add_argument('--only', type=str, default=None)
    parser.add_argument('--nvfp4', action='store_true', help='NVFP4 (UE4M3 per 16) instead of MXFP4')
    args = parser.parse_args()
    gran_k = 16 if args.nvfp4 else 32
    deep_gemm.set_pdl(not args.no_pdl)
    backends = ['cudnn', 'b12x', 'cute-dsl', 'cutedsl_low_latency', 'cutlass', 'trtllm']
    working = {}
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    print(f'{"linear":>15} {"m":>5} {"n":>6} {"k":>6} | {"flashinfer":>10} {"backend":>10} {"deepgemm":>10} | speedup')
    for name, n, k in LINEARS:
        if args.only and name not in args.only.split(','):
            continue
        w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
        # 512 MB of FP4 weights (4x the 128 MB L2). At ~1x L2 (an earlier version) part of the set stayed L2-resident,
        # which made decode look up to 30% faster and favored the format with fewer scale bytes
        copies = max(4, min(64, math.ceil(512e6 / (n * k / 2))))
        dg_q, dg_sf, dg_sf_values = quantize_dg(w, args.nvfp4)
        dg_w = [(dg_q.clone(), dg_sf.clone()) for _ in range(copies)]
        fi_q, fi_sf, fi_w_gsf = quantize_fi(w, args.nvfp4)
        fi_w = [(fi_q.clone(), fi_sf.clone()) for _ in range(copies)]
        lt_w = [(dg_q.clone(), to_cublaslt_nvfp4_sf(dg_sf_values)) for _ in range(copies)] if args.nvfp4 else None
        for m in ms:
            x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            dg_a, dg_asf, dg_asf_values = quantize_dg(x, args.nvfp4)
            fi_a, fi_asf, fi_a_gsf = quantize_fi(x, args.nvfp4)
            fi_alpha = (1.0 / (fi_a_gsf * fi_w_gsf)).float().reshape(1) if args.nvfp4 else None
            row = f'{name:>15} {m:5d} {n:6d} {k:6d} | '
            # Each side's candidates, as graph-replay closures
            runs = {}
            for backend in backends:
                if working.get(backend) is False:
                    continue
                try:
                    run_fi = lambda wq, wsf, be=backend: mm_fp4(fi_a, wq.T, fi_asf, wsf.T, fi_alpha, torch.bfloat16, out,
                                                                 block_size=gran_k, use_nvfp4=args.nvfp4, backend=be,
                                                                 enable_pdl=not args.no_pdl)
                    run_fi(*fi_w[0])
                    working[backend] = True
                    runs[backend] = (run_fi, fi_w)
                except Exception:  # noqa: BLE001 (backend does not support the format on SM120)
                    if working.get(backend) is None:
                        working[backend] = False
            if args.nvfp4:
                lt_a = (dg_a, to_cublaslt_nvfp4_sf(dg_asf_values))
                runs['cublaslt'] = (lambda wq, wsf: deep_gemm.cublaslt_nvfp4_gemm_nt(lt_a, (wq, wsf), out), lt_w)
                runs['cublaslt'][0](*lt_w[0])
            # DeepGEMM pays for alpha too (FlashInfer's NVFP4 GEMM applies its global-scale alpha)
            dg_alpha = 1.0 if args.nvfp4 else None
            run_dg = lambda wq, wsf: deep_gemm.fp8_fp4_gemm_nt((dg_a, dg_asf), (wq, wsf), out, recipe=(1, 1, gran_k),
                                                               alpha=dg_alpha)
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

