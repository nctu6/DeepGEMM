# SM120 FP8 GEMM benchmark vs. cuBLASLt (development helper)
#
# Both kernels are timed the same way: L2 (128 MB on RTX PRO 6000) is flushed before every launch,
# and only the launch itself is timed with CUDA events.
import argparse
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp8, per_block_cast_to_fp8

NK = [(2112, 7168), (576, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168), (7168, 2048)]
MS = [1, 16, 64, 128, 512, 1024, 4096, 8192]


def bench_cold(fn, num_tests: int = 20) -> float:
    """Median GPU kernel time of `fn` (all kernels it launches), with L2 flushed before every call."""
    flush = torch.empty(int(512e6 // 4), dtype=torch.int, device='cuda')
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(num_tests):
            flush.zero_()
            torch.cuda._sleep(100000)
            fn()
        torch.cuda.synchronize()
    # Sum kernel durations between consecutive flushes
    events = sorted((e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA),
                    key=lambda e: e.time_range.start)
    times, current = [], None
    for e in events:
        name = e.name.lower()
        if 'spin_kernel' in name:
            # A sleep kernel starts every test iteration
            if current is not None:
                times.append(current)
            current = 0.0
        elif 'fill' in name or 'elementwise' in name:
            continue
        elif current is not None:
            current += e.time_range.elapsed_us()
    if current is not None:
        times.append(current)
    times = sorted(t for t in times if t > 0)
    return times[len(times) // 2] / 1e6


def bench_graph(fn, weights, num_replays: int = 10) -> float:
    """Per-call time of `fn(w)` over all `weights` (cold, distinct copies) inside one CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for w in weights[:2]:
            fn(w)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for w in weights:
            fn(w)
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(num_replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1e3 / num_replays / len(weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ms', type=str, default=None)
    parser.add_argument('--nks', type=str, default=None)
    args = parser.parse_args()
    ms = [int(x) for x in args.ms.split(',')] if args.ms else MS
    nks = [tuple(int(y) for y in x.split('x')) for x in args.nks.split(',')] if args.nks else NK

    print(f'{"m":>5} {"n":>6} {"k":>6} | {"DG us":>8} {"TF":>6} {"GB/s":>6} | {"cuBLAS us":>9} {"TF":>6} | ratio')
    for n, k in nks:
        for m in ms:
            a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
            b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
            a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=False)
            b_fp8 = per_block_cast_to_fp8(b, use_ue8m0=False)
            d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
            t_dg = bench_cold(lambda: deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d))

            # Per-tensor FP8 cuBLASLt (no block scaling): an upper-bound reference
            a8, b8 = a_fp8[0], b_fp8[0]
            one = torch.ones((), device='cuda')
            t_cb = bench_cold(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))
            num_bytes = m * k + n * k + m * n * 2
            print(f'{m:5} {n:6} {k:6} | {t_dg * 1e6:8.1f} {2 * m * n * k / t_dg / 1e12:6.1f} {num_bytes / t_dg / 1e9:6.0f} | '
                  f'{t_cb * 1e6:9.1f} {2 * m * n * k / t_cb / 1e12:6.1f} | {t_cb / t_dg:5.2f}', flush=True)


if __name__ == '__main__':
    main()
