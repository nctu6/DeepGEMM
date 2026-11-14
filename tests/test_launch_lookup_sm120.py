"""Launchers that find their kernel by its template integers (SM120): paged MQA logits, the paged metadata,
the 1D2D / 1D1D GEMMs (FP8, MXFP4, BF16) and the hyper-connection pre-norm GEMM.

Each of them keeps a per-call-site cache from the integers that fill the kernel's template to the loaded kernel, instead
of hashing the generated source on every call (see `kernel_lookup.hpp`). A key that misses an integer would hand one
configuration another's kernel. This runs configurations that differ in those integers back to back, several times each,
so every one is looked up after another's miss and hit, and checks each result against a PyTorch reference. The sparse
MX logits and their metadata launchers are covered by `scripts/sm120_sparse_mqa_check.py` and `sm120_sparse_paged_check.py`
(metadata buffers hold uninitialized padding, so they are compared through the logits).
"""
import os
import sys
import torch
import deep_gemm
from deep_gemm.testing import assert_bitwise_equal, calc_diff, get_arch_major
from deep_gemm.utils import (cast_back_from_fp4, cast_back_from_fp8, ceil_div, per_token_cast_to_fp4,
                             per_token_cast_to_fp8)
from deep_gemm.utils.math import per_block_cast_to_fp8
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_attention import (kv_cache_cast_to_fp8, kv_cache_cast_to_mxfp4, mqa_logits_tolerances, quantize_mqa_q,  # noqa: E402
                            ref_paged_mqa_logits, to_mqa_weights)
def run_paged(fmt, batch_size, next_n, num_heads, head_dim, context_len):
    dtype = torch.float if fmt == 'fp8' else torch.bfloat16
    block_kv = 64
    q = torch.randn((batch_size, next_n, num_heads, head_dim), device='cuda', dtype=torch.bfloat16)
    weights = to_mqa_weights(torch.randn((batch_size * next_n, num_heads), device='cuda', dtype=dtype))
    context_lens = torch.randint(context_len // 2, context_len, (batch_size, next_n), device='cuda', dtype=torch.int)
    context_lens[:, -1] = context_len
    num_blocks_per_query = ceil_div(context_len, block_kv)
    max_model_len = num_blocks_per_query * block_kv
    num_total_blocks = batch_size * num_blocks_per_query
    block_table = torch.randperm(num_total_blocks, device='cuda', dtype=torch.int).view(batch_size, num_blocks_per_query).contiguous()
    kv_cache = torch.randn((num_total_blocks, block_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
    q_in, q_simulated = quantize_mqa_q(q, fmt)
    kv_in, kv_simulated = (kv_cache_cast_to_fp8 if fmt == 'fp8' else kv_cache_cast_to_mxfp4)(kv_cache)
    schedule_meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())
    kwargs = dict(q=q_in, kv_cache=kv_in, weights=weights, context_lens=context_lens, block_table=block_table,
                  schedule_meta=schedule_meta, max_context_len=max_model_len)
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
    assert logits.dtype == dtype
    ref = ref_paged_mqa_logits(q_simulated, kv_simulated, weights.float(), context_lens, block_table, max_model_len)
    valid = torch.arange(max_model_len, device='cuda')[None, :] < context_lens.view(-1, 1)
    diff = calc_diff(logits.float().masked_fill(~valid, 0), ref.masked_fill(~valid, 0))
    assert diff < mqa_logits_tolerances(fmt)[1] * 10, (fmt, batch_size, next_n, num_heads, head_dim, context_len, diff)
    # The metadata kernel is looked up by the batch size too: a second call gives the same schedule
    assert_bitwise_equal(deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms()), schedule_meta)
    assert_bitwise_equal(deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs).float().masked_fill(~valid, 0),
                         logits.float().masked_fill(~valid, 0), 'paged logits repeat')
def test_paged_logits_interleaved():
    torch.manual_seed(0)
    cases = [
        # (fmt, batch, next_n, heads, head_dim, context)
        ('fp8', 3, 1, 32, 128, 1000), ('fp8', 40, 2, 64, 128, 300), ('fp8', 2, 1, 64, 64, 700), ('fp8', 5, 2, 32, 32, 500),
        ('mxfp4', 3, 1, 32, 128, 1000), ('mxfp4', 40, 2, 32, 128, 300), ('mxfp4', 33, 1, 64, 128, 130),
    ]
    for _ in range(3):
        for case in cases:
            run_paged(*case)
    print('Interleaved paged MQA logits and metadata OK')
def run_fp8_1d2d(m, n, k):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    a_q, a_sf = per_token_cast_to_fp8(a, use_ue8m0=False, gran_k=128)
    b_q, b_sf = per_block_cast_to_fp8(b, use_ue8m0=False)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((a_q, a_sf), (b_q, b_sf), out)
    b_dq = (b_q.float().view(ceil_div(n, 128), 128, ceil_div(k, 128), 128) * b_sf[:, None, :, None]).view(n, k)
    ref = cast_back_from_fp8(a_q, a_sf, gran_k=128) @ b_dq.T
    assert calc_diff(out.float(), ref) < 1e-3, ('fp8 1d2d', m, n, k)
def run_fp4_1d1d(m, n, k):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    a_q, a_sf = per_token_cast_to_fp4(a, use_ue8m0=True, gran_k=32)
    b_q, b_sf = per_token_cast_to_fp4(b, use_ue8m0=True, gran_k=32)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.fp8_fp4_gemm_nt((a_q, deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf)),
                              (b_q, deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(b_sf)), out, recipe=(1, 1, 32))
    ref = cast_back_from_fp4(a_q, a_sf, gran_k=32) @ cast_back_from_fp4(b_q, b_sf, gran_k=32).T
    assert calc_diff(out.float(), ref) < 1e-3, ('fp4 1d1d', m, n, k)
def run_bf16(m, n, k):
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.bf16_gemm_nt(a, b, out)
    assert calc_diff(out.float(), a.float() @ b.float().T) < 1e-4, ('bf16 1d1d', m, n, k)
def run_hc_prenorm(m, n, k, num_splits):
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    d = torch.empty((num_splits, m, n), dtype=torch.float, device='cuda')
    s = torch.empty((num_splits, m), dtype=torch.float, device='cuda')
    deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)
    assert calc_diff(d.sum(0), a.float() @ b.T) < 1e-6, ('hc prenorm', m, n, k, num_splits)
    assert calc_diff(s.sum(0), a.float().square().sum(-1)) < 1e-6, ('hc prenorm sqr_sum', m, n, k, num_splits)
def test_gemms_interleaved():
    torch.manual_seed(0)
    # Shapes differ in the integers of the template: block sizes (m), stages (k), N and K as compiled dimensions
    shapes = [(1, 2048, 1536), (128, 4096, 7168), (7, 1024, 512), (512, 512, 2048), (128, 4096, 7168)]
    # BF16 goes to cuBLASLt unless deterministic mode asks for the DeepGEMM kernel
    for _ in range(3):
        for m, n, k in shapes:
            run_fp8_1d2d(m, n, k)
            run_fp4_1d1d(m, n, k)
            deep_gemm.use_deterministic_algorithms(True)
            try:
                run_bf16(m, n, k)
            finally:
                deep_gemm.use_deterministic_algorithms(False)
        for m, n, k, splits in ((13, 24, 7168, 16), (137, 24, 28672, 4), (128, 32, 7680, 8), (13, 24, 7168, 16)):
            run_hc_prenorm(m, n, k, splits)
    print('Interleaved GEMM and hyper-connection launches OK')
if __name__ == '__main__':
    if get_arch_major() != 12:
        print('Skipped: SM120 only')
        sys.exit(0)
    test_paged_logits_interleaved()
    test_gemms_interleaved()
