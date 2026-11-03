# Quick SM120 FP8 GEMM correctness/perf probe (development helper)
import sys
import torch
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8


def run(m, n, k, bench=False):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    ref = a.float() @ b.float().t()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=False)
    b_fp8 = per_block_cast_to_fp8(b, use_ue8m0=False)
    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d)
    diff = calc_diff(d, ref)
    msg = f'm={m:5} n={n:6} k={k:6} diff={diff:.6f}'
    if bench:
        t = bench_kineto(lambda: deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d), 'fp8_gemm', suppress_kineto_output=True)
        msg += f' {t * 1e6:8.1f} us {2 * m * n * k / t / 1e12:7.1f} TFLOPS'
    print(msg, flush=True)
    return diff


if __name__ == '__main__':
    bench = '--bench' in sys.argv
    shapes = [(128, 256, 256), (1, 2112, 7168), (100, 576, 7168), (4096, 4096, 7168),
              (128, 24576, 1536), (4096, 7168, 2048), (17, 32768, 512), (4096, 2112, 7296)]
    worst = max(run(*s, bench=bench) for s in shapes)
    assert worst < 0.001, worst
    print('OK')
