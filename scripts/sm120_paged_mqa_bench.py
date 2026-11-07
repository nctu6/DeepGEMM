# Benchmark of the SM120 paged MQA logits kernel: time and effective KV bandwidth per (head dim, B, next_n, H, ctx)
import argparse
import torch
import deep_gemm
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8


def make_case(batch_size, next_n, num_heads, head_dim, ctx, block_kv=64):
    q_fp8 = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=torch.float)
    context_lens = torch.full((batch_size, next_n), ctx, device='cuda', dtype=torch.int)
    max_blocks = ceil_div(ctx, block_kv)
    num_blocks = batch_size * max_blocks
    block_table = torch.randperm(num_blocks, device='cuda', dtype=torch.int).view(batch_size, max_blocks).contiguous()
    kv = torch.randn(num_blocks * block_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0, ), False)
    fused = torch.empty(num_blocks, block_kv * (head_dim + 4), device='cuda', dtype=torch.uint8)
    fused[:, :block_kv * head_dim] = kv_fp8.view(num_blocks, block_kv * head_dim).view(torch.uint8)
    fused[:, block_kv * head_dim:] = kv_sf.view(num_blocks, block_kv).view(torch.uint8)
    fused = fused.view(num_blocks, block_kv, 1, head_dim + 4)
    meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())
    return (q_fp8, None), fused, weights, context_lens, block_table, meta, ctx


def bench(fn, iters=30):
    # Cold L2: a 256 MB memset before each timed call
    flush = torch.empty(256 * 1024 * 1024, device='cuda', dtype=torch.uint8)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float('inf')
    for _ in range(3):
        total = 0.0
        for _ in range(iters):
            flush.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            total += s.elapsed_time(e)
        best = min(best, total / iters)
    return best * 1e-3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dims', type=int, nargs='+', default=[128, 64, 32])
    args = parser.parse_args()
    torch.manual_seed(0)
    for d in args.dims:
        for b, nn, h, c in ((32, 1, 64, 8192), (32, 2, 64, 8192), (64, 1, 32, 8192), (128, 1, 64, 4096), (8, 1, 64, 32768), (64, 2, 32, 4096)):
            if b * ceil_div(c, 64) * 64 * (d + 4) > 12 * 1024**3:
                continue
            args_ = make_case(b, nn, h, d, c)
            t = bench(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(*args_))
            kv_bytes = b * c * (d + 4)
            print(f'D={d:3} B={b:3} next_n={nn} H={h:2} ctx={c:6}: {t * 1e6:8.1f} us, KV {kv_bytes / t / 1e9:7.1f} GB/s')
            del args_
