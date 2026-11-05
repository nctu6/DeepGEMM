"""SM120 paged sparse MQA logits check (MXFP8 + MXFP4), dequantized-reference oracle.

Mirrors the paged branch of tests/test_attention.py::test_sparse_mqa_logits with a single request, so the
block table and sparse gather are easy to reason about. Validates the kIsPaged path of the SM120 kernel.

  sm120_remote.sh dg scripts/sm120_sparse_paged_check.py
"""
import sys
import torch

import deep_gemm
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8

sys.path.insert(0, 'tests')
from test_attention import (make_sparse_kv_block_indices, to_mqa_weights,  # noqa: E402
                            kv_cache_cast_to_mxfp4, kv_cache_cast_to_mxfp8)


def check(num_heads, fmt, num_q_tokens, context_len, sparse_block_kv, num_max_sparse_blocks, page_kv=64, seed=0):
    head_dim = 128
    is_mxfp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8
    kv_cache_cast = kv_cache_cast_to_mxfp4 if is_mxfp4 else kv_cache_cast_to_mxfp8
    elem_dim = head_dim // 2 if is_mxfp4 else head_dim
    torch.manual_seed(seed)

    # Single request: every q-token shares one context length. Paged q layout is [num_q_tokens, 1, num_heads, d].
    context_lens = [context_len] * num_q_tokens
    request_indices = [0] * num_q_tokens
    context_starts = [0] * num_q_tokens

    q_fp, q_sf = cast(torch.randn((num_q_tokens, 1, num_heads, head_dim), device='cuda', dtype=torch.bfloat16).view(-1, head_dim),
                      use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q = q_fp.view(num_q_tokens, 1, num_heads, elem_dim), q_sf.view(num_q_tokens, 1, num_heads)
    weights = to_mqa_weights(torch.randn((num_q_tokens, num_heads), device='cuda', dtype=torch.bfloat16))
    sparse_indices, num_sparse_blocks = make_sparse_kv_block_indices(
        context_lens, request_indices, sparse_block_kv, num_max_sparse_blocks, seed, context_starts)

    # Paged KV cache: one request, ceil(context_len / page_kv) pages
    num_pages = ceil_div(context_len, page_kv)
    page_bytes = page_kv * (elem_dim + 4)
    page_stride_bytes = ceil_div(page_bytes, 128) * 128
    kv_storage = torch.empty((num_pages, page_stride_bytes), device='cuda', dtype=torch.uint8)
    kv_cache = kv_storage.as_strided((num_pages, page_kv, 1, elem_dim + 4),
                                     (page_stride_bytes, elem_dim + 4, elem_dim + 4, 1))
    kv_ref = torch.randn((num_pages, page_kv, 1, head_dim), device='cuda', dtype=torch.bfloat16)
    kv_pages, kv_back = kv_cache_cast(kv_ref)
    kv_cache.copy_(kv_pages)

    indices = torch.tensor(request_indices, device='cuda', dtype=torch.int32)
    context_lens_tensor = torch.tensor(context_lens, device='cuda', dtype=torch.int32)
    # Identity page pool for a single request: logical page i -> physical page i
    block_table = torch.arange(num_pages, device='cuda', dtype=torch.int32)[None, :].expand(num_q_tokens, -1).contiguous()

    metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
        context_lens_tensor, block_table, indices, page_kv, sparse_indices, q[0].dtype, sparse_block_kv)
    sparse_logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        q=q, kv_cache=kv_cache, weights=weights, metadata=metadata,
        num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv)

    # Dequantized reference: KV tokens are the cast-back cache, laid out [num_pages*page_kv, head_dim]
    kv_deq = kv_back.view(num_pages * page_kv, head_dim).float()[:context_len]
    q_deq = (cast_back_from_fp4 if is_mxfp4 else cast_back_from_fp8)(
        q_fp, q_sf, gran_k=32, use_packed_ue8m0=True).view(num_q_tokens, num_heads, head_dim).float()
    scores = torch.einsum('qhd,td->qht', q_deq, kv_deq).relu_()
    ref_full = (scores * weights[:, :, None].float()).sum(1)   # [q, context_len]

    kv_offsets = torch.arange(sparse_block_kv, device='cuda', dtype=torch.int32)
    token_indices = (sparse_indices.unsqueeze(-1) * sparse_block_kv + kv_offsets).flatten(1).long()
    nsb = torch.tensor(num_sparse_blocks, device='cuda')
    valid = torch.arange(token_indices.size(1), device='cuda')[None, :] < (nsb * sparse_block_kv)[:, None]
    valid &= token_indices < context_lens_tensor[:, None]
    sp = sparse_logits[valid].float()
    ref = ref_full.gather(1, token_indices.clamp(0, context_len - 1))[valid]
    x, y = sp.double(), ref.double()
    diff = (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item() if x.numel() else 0.0
    ok = diff < 1e-3
    print(f'[{fmt:6}] H={num_heads:2} SQ={num_q_tokens:4} ctx={context_len:6} bkv={sparse_block_kv} '
          f'nmsb={num_max_sparse_blocks} page={page_kv}: n={sp.numel()} diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    return ok


def main():
    deep_gemm.set_pdl(False)
    failed = False
    for fmt in ('mxfp8', 'mxfp4'):
        for num_heads in (8, 32):
            for (sq, ctx, bkv, nmsb) in [(9, 2048, 8, 128), (64, 4096, 8, 512), (9, 1024, 16, 128)]:
                failed |= not check(num_heads, fmt, sq, ctx, bkv, nmsb)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
