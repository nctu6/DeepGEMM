# Benchmark of the SM120 paged MQA logits kernel: time and effective KV bandwidth per (format, head dim, B, next_n, H, ctx)
#   --fmts fp8 mxfp4   compares the FP8 per-token-scale rows (132 bytes) with the MXFP4 rows of V4.1 (68 bytes) at the same shapes
import argparse
import torch
import deep_gemm
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8, per_token_cast_to_fp4, per_token_cast_to_fp8


def make_case(batch_size, next_n, num_heads, head_dim, ctx, block_kv=64, fmt='fp8', sequential=False):
    q = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=torch.float)
    context_lens = torch.full((batch_size, next_n), ctx, device='cuda', dtype=torch.int)
    max_blocks = ceil_div(ctx, block_kv)
    num_blocks = batch_size * max_blocks
    # Random pages model a long-running allocator; sequential ones one request per contiguous run
    order = torch.arange(num_blocks, device='cuda', dtype=torch.int) if sequential else torch.randperm(num_blocks, device='cuda', dtype=torch.int)
    block_table = order.view(batch_size, max_blocks).contiguous()
    kv = torch.randn(num_blocks * block_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    if fmt == 'fp8':
        q_in = (q.to(torch.float8_e4m3fn), None)
        kv_fp, kv_sf = per_custom_dims_cast_to_fp8(kv, (0, ), False)
    else:
        cast = per_token_cast_to_fp4 if fmt == 'mxfp4' else per_token_cast_to_fp8
        q_fp, q_sf = cast(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        q_in = (q_fp.view(batch_size, next_n, num_heads, -1), q_sf.view(batch_size, next_n, num_heads))
        kv_fp, kv_sf = cast(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    row = kv_fp.shape[-1]
    fused = torch.empty(num_blocks, block_kv * (row + 4), device='cuda', dtype=torch.uint8)
    fused[:, :block_kv * row] = kv_fp.view(num_blocks, block_kv * row).view(torch.uint8)
    fused[:, block_kv * row:] = kv_sf.view(num_blocks, block_kv).view(torch.uint8)
    fused = fused.view(num_blocks, block_kv, 1, row + 4)
    meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())
    return dict(q=q_in, kv_cache=fused, weights=weights, context_lens=context_lens, block_table=block_table,
                schedule_meta=meta, max_context_len=ctx,
                **({} if fmt == 'fp8' else {'logits_dtype': torch.float32})), row + 4


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
    parser.add_argument('--fmts', nargs='+', default=['fp8'])
    parser.add_argument('--sms', type=int, default=0, help='run the kernel on this many SMs')
    parser.add_argument('--sequential', action='store_true', help='contiguous pages per request instead of a random permutation')
    parser.add_argument('--v41', action='store_true', help='V4.1 decode shapes: 32 heads, 1-128 requests, 1K-32K rows')
    args = parser.parse_args()
    if args.sms:
        deep_gemm.set_num_sms(args.sms)
    torch.manual_seed(0)
    shapes = ((32, 1, 64, 8192), (32, 2, 64, 8192), (64, 1, 32, 8192), (128, 1, 64, 4096), (8, 1, 64, 32768), (64, 2, 32, 4096))
    if args.v41:
        shapes = tuple((b, 1, 32, c) for b in (1, 8, 32, 128) for c in (1024, 4096, 16384, 32768))
    for d in args.dims:
        for b, nn, h, c in shapes:
            times = []
            for fmt in args.fmts:
                if fmt != 'fp8' and d != 128:
                    continue
                kwargs, row_bytes = make_case(b, nn, h, d, c, fmt=fmt, sequential=args.sequential)
                t = bench(lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs))
                kv_bytes = b * c * row_bytes
                times.append(t)
                print(f'{fmt:5} D={d:3} B={b:3} next_n={nn} H={h:2} ctx={c:6}: {t * 1e6:8.1f} us, KV {kv_bytes / t / 1e9:7.1f} GB/s'
                      + (f', {times[0] / t:4.2f}x vs {args.fmts[0]}' if len(times) > 1 else ''), flush=True)
                del kwargs
