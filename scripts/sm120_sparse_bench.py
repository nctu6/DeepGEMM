"""Benchmark the SM120 sparse MQA logits kernel: latency and relu-reduction throughput per shape.

  sm120_remote.sh dg scripts/sm120_sparse_bench.py
"""
import sys
import torch

import deep_gemm
from deep_gemm.testing.bench import bench_kineto
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8

sys.path.insert(0, 'tests')
from test_attention import make_sparse_kv_block_indices, to_mqa_weights  # noqa: E402


def bench_case(num_heads, fmt, num_q_tokens, kv_len, sparse_block_kv, num_max_sparse_blocks, seed=0):
    head_dim = 128
    is_mxfp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8
    elem_dim = head_dim // 2 if is_mxfp4 else head_dim
    torch.manual_seed(seed)

    context_lens = [kv_len] * num_q_tokens
    context_starts = [0] * num_q_tokens
    num_kv_tokens = kv_len

    q_fp, q_sf = cast(torch.randn((num_q_tokens, num_heads, head_dim), device='cuda', dtype=torch.bfloat16).view(-1, head_dim),
                      use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q = q_fp.view(num_q_tokens, num_heads, elem_dim), q_sf.view(num_q_tokens, num_heads)
    weights = to_mqa_weights(torch.randn((num_q_tokens, num_heads), device='cuda', dtype=torch.bfloat16))
    sparse_indices, num_sparse_blocks = make_sparse_kv_block_indices(
        context_lens, None, sparse_block_kv, num_max_sparse_blocks, seed, context_starts)
    kv_fp, kv_sf = cast(torch.randn((num_kv_tokens, head_dim), device='cuda', dtype=torch.bfloat16),
                        use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    kv = kv_fp, kv_sf.view(num_kv_tokens)
    starts = torch.tensor(context_starts, device='cuda', dtype=torch.int32)
    ends = torch.tensor(context_lens, device='cuda', dtype=torch.int32)
    metadata = deep_gemm.get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start=starts, cu_seq_len_k_end=ends, num_kv_tokens=num_kv_tokens,
        sparse_kv_block_indices=sparse_indices, qk_dtype=q[0].dtype, sparse_block_kv=sparse_block_kv)

    run = lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(
        q=q, kv=kv, weights=weights, metadata=metadata,
        num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)
    run()
    torch.cuda.synchronize()
    t = bench_kineto(run, 'sm120_sparse_mqa_logits')
    # Work: each selected (block, token) reduces num_heads relu-mul-adds over head_dim
    total_blocks = int(sum(num_sparse_blocks)) if hasattr(num_sparse_blocks, '__iter__') else num_sparse_blocks
    reduce_relus = total_blocks * sparse_block_kv * num_heads
    tflops = reduce_relus * head_dim * 2 / (t * 1e12)
    print(f'[{fmt:6}] H={num_heads:2} SQ={num_q_tokens:5} KV={kv_len:6} bkv={sparse_block_kv} '
          f'nmsb={num_max_sparse_blocks}: {t*1e6:8.1f} us  {tflops:7.2f} TFLOP/s', flush=True)


def main():
    deep_gemm.set_pdl(False)
    for fmt in ('mxfp8', 'mxfp4'):
        for num_heads in (32,):
            for (sq, kv, bkv, nmsb) in [(2048, 8192, 8, 2048), (4096, 16384, 8, 2048), (8192, 65536, 8, 2048)]:
                bench_case(num_heads, fmt, sq, kv, bkv, nmsb)


if __name__ == '__main__':
    main()
