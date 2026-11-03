# Phase 4.5: DeepGEMM vs. Triton for the DeepSeek-V3.2 DSA indexer (logits + top-k), as TokenSpeed calls them
#
# Run in the TokenSpeed venv with this DeepGEMM build first on PYTHONPATH (see `sm120_remote.sh ts`).
#  - decode: `dsa_plan` + `dsa_decode_topk`, CUDA graph over several cache copies (exceeding L2), per call
#  - prefill: `dsa_prefill_topk`, eager (as served), CUDA-event wall time per call
import argparse
import math

import torch

from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_dg
from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

from sm120_baseline import bench_graph

PAGE = 64
HEADS = 64
DIM = 128
TOPK = 2048
ROW_BYTES = DIM + 4


def make_index_cache(num_pages: int) -> torch.Tensor:
    """Page-planar FP8 index-K cache (`[slots, 132]` bytes): 64 x 128 E4M3 keys, then 64 FP32 scales per page."""
    keys = torch.randn(num_pages, PAGE, DIM, device='cuda', dtype=torch.bfloat16)
    scales = keys.abs().amax(-1).float().clamp_min(1e-4) / 448.0
    fp8 = (keys / scales[..., None]).to(torch.float8_e4m3fn)
    cache = torch.empty(num_pages, PAGE * ROW_BYTES, device='cuda', dtype=torch.uint8)
    cache[:, :PAGE * DIM] = fp8.view(num_pages, -1).view(torch.uint8)
    cache[:, PAGE * DIM:] = scales.view(torch.uint8).view(num_pages, -1)
    return cache.view(num_pages * PAGE, ROW_BYTES)


def overlap(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean fraction of top-k entries shared row by row (ties may legitimately differ)."""
    total = 0.0
    for x, y in zip(a, b):
        x, y = x[x >= 0], y[y >= 0]
        if x.numel() == 0:
            total += 1.0
            continue
        total += torch.isin(x, y).float().mean().item()
    return total / a.shape[0]


def bench_decode(batch: int, ctx: int):
    seq_lens = torch.randint(ctx * 3 // 4, ctx + 1, (batch, ), device='cuda', dtype=torch.int32)
    max_pages = math.ceil(int(seq_lens.max()) / PAGE)
    num_pages = batch * max_pages
    copies = max(1, min(8, math.ceil(512e6 / (num_pages * PAGE * ROW_BYTES))))
    caches = [make_index_cache(num_pages) for _ in range(copies)]
    block_table = torch.randperm(num_pages, device='cuda', dtype=torch.int32).view(batch, max_pages).contiguous()
    q = torch.randn(batch, HEADS, DIM, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(batch, HEADS, device='cuda', dtype=torch.float32)
    seq_lens_2d = seq_lens.view(-1, 1).contiguous()
    kwargs = dict(page_size=PAGE, topk=TOPK, softmax_scale=DIM ** -0.5, seq_lens_2d=seq_lens_2d)

    impls = {
        'deepgemm': (dsa_dg.deep_gemm_dsa_plan, dsa_dg.deep_gemm_dsa_decode_topk),
        'triton': (dsa_triton.triton_dsa_plan, dsa_triton.triton_dsa_decode_topk_fp8),
    }
    times, outs = {}, {}
    for name, (plan_fn, topk_fn) in impls.items():
        plan = plan_fn(page_size=PAGE, seq_lens_2d=seq_lens_2d)
        out = torch.empty(batch, TOPK, device='cuda', dtype=torch.int32)
        lens_out = torch.empty(batch, device='cuda', dtype=torch.int32)

        def run(cache):
            plan_fn(page_size=PAGE, seq_lens_2d=seq_lens_2d, out=plan)
            topk_fn(q, weights, seq_lens, block_table, index_k_cache=cache, plan=plan, out=out, lens_out=lens_out, **kwargs)
        run(caches[0])
        torch.cuda.synchronize()
        outs[name] = out.clone()
        times[name] = min(bench_graph(run, caches) for _ in range(3))
    return times, overlap(outs['deepgemm'], outs['triton'])


def bench_prefill(num_tokens: int, prefix: int):
    num_keys = prefix + num_tokens
    num_pages = math.ceil(num_keys / PAGE)
    cache = make_index_cache(num_pages)
    slots = torch.arange(num_keys, device='cuda', dtype=torch.int64)
    row_starts = torch.zeros(num_tokens, device='cuda', dtype=torch.int32)
    row_ends = (prefix + 1 + torch.arange(num_tokens, device='cuda', dtype=torch.int32)).contiguous()
    q = torch.randn(num_tokens, HEADS, DIM, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(num_tokens, HEADS, device='cuda', dtype=torch.float32)
    kwargs = dict(topk=TOPK, softmax_scale=DIM ** -0.5, index_k_cache=cache, page_size=PAGE)
    candidate_lens_cpu = (row_ends - row_starts).cpu()

    # DeepGEMM prefill as TokenSpeed runs it (quantize Q, fold its scale into the weights, gather keys, MQA logits),
    # with Triton's radix top-k, as TokenSpeed's default top-k (`torch.ops.trtllm.indexer_topk_prefill`) and the
    # FlashInfer row-invariant fallback (`k=2048` needs 128 KB smem) have no sm120 support
    from tokenspeed_kernel.ops.quantization import quantize_fp8
    from tokenspeed_kernel.ops.attention.dsa._triton.topk import _topk_with_padding, combine_topk_weights
    import deep_gemm
    pages = cache.view(-1, PAGE * ROW_BYTES)
    k_fp8 = pages[:, :PAGE * DIM].contiguous().view(torch.float8_e4m3fn).view(-1, DIM)[:num_keys]
    k_scale = pages[:, PAGE * DIM:].contiguous().view(torch.float32).view(-1)[:num_keys]
    lens = (row_ends - row_starts).contiguous()

    def deepgemm_prefill():
        q_fp8, q_scale = quantize_fp8(q.view(-1, DIM), granularity='token_group', group_size=128, scale_encoding='float32')
        w = combine_topk_weights(weights, q_scale, DIM ** -0.5)
        logits = deep_gemm.fp8_mqa_logits(q_fp8.view_as(q), (k_fp8, k_scale), w, row_starts, row_ends,
                                          clean_logits=False, max_seqlen_k=num_keys)
        return (_topk_with_padding(logits, TOPK, seq_lens=lens, enable_pdl=False), )

    impls = {
        'deepgemm': deepgemm_prefill,
        'triton': lambda: dsa_triton.triton_dsa_prefill_topk_fp8(q, weights, slots, row_starts, row_ends, **kwargs),
    }
    times, outs = {}, {}
    for name, fn in impls.items():
        outs[name] = fn()[0].clone()
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1e3)
        times[name] = sorted(samples)[len(samples) // 2]
    return times, overlap(outs['deepgemm'], outs['triton'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decode', type=str, default='1x4096,8x4096,32x4096,128x4096,1x16384,8x16384,32x16384,128x16384,1x65536,8x65536,32x65536,1x131072,8x131072')
    parser.add_argument('--prefill', type=str, default='1024x0,4096x0,4096x4096,2048x14336,8192x0')
    args = parser.parse_args()
    print('decode (graph, per call)')
    for case in filter(None, args.decode.split(',')):
        batch, ctx = (int(x) for x in case.split('x'))
        times, ov = bench_decode(batch, ctx)
        ratio = times['triton'] / times['deepgemm']
        print(f'  B={batch:4} ctx={ctx:6}: deepgemm {times["deepgemm"] * 1e6:8.1f} us | triton {times["triton"] * 1e6:8.1f} us | '
              f'triton/DG {ratio:5.2f}{"  LOSE" if ratio < 1 else ""} | top-k overlap {ov:.3f}', flush=True)
    print('prefill (eager, per call)')
    for case in filter(None, args.prefill.split(',')):
        tokens, prefix = (int(x) for x in case.split('x'))
        times, ov = bench_prefill(tokens, prefix)
        ratio = times['triton'] / times['deepgemm']
        print(f'  T={tokens:5} prefix={prefix:6}: deepgemm {times["deepgemm"] * 1e6:9.1f} us | triton {times["triton"] * 1e6:9.1f} us | '
              f'triton/DG {ratio:5.2f}{"  LOSE" if ratio < 1 else ""} | top-k overlap {ov:.3f}', flush=True)


if __name__ == '__main__':
    main()
