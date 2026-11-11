"""Focused SM120 sparse MQA logits correctness check (contiguous, aligned ks), MXFP8 and MXFP4.

Oracle: the sparse logits must equal the dense `fp8_fp4_mqa_logits` gathered at the sparse-selected KV tokens
(the same check `tests/test_attention.py::test_sparse_mqa_logits` uses). Fast single-case iteration.

  sm120_remote.sh dg scripts/sm120_sparse_mqa_check.py
"""
import random
import sys

import torch

import deep_gemm
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8

sys.path.insert(0, 'tests')
from test_attention import make_sparse_kv_block_indices, to_mqa_weights  # noqa: E402


def check(num_heads, fmt, num_q_tokens, kv_len, sparse_block_kv, num_max_sparse_blocks, use_unaligned_ks=False, seed=0):
    head_dim = 128
    is_mxfp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_mxfp4 else per_token_cast_to_fp8
    elem_dim = head_dim // 2 if is_mxfp4 else head_dim
    torch.manual_seed(seed)

    # One request. Aligned ks starts at 0; unaligned ks starts at a non-multiple of sparse_block_kv.
    start = (sparse_block_kv + sparse_block_kv // 2) if use_unaligned_ks else 0
    assert (start % sparse_block_kv != 0) if use_unaligned_ks else (start % sparse_block_kv == 0)
    context_starts = [start] * num_q_tokens
    context_lens = [start + kv_len] * num_q_tokens
    num_kv_tokens = start + kv_len

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
        sparse_kv_block_indices=sparse_indices, qk_dtype=q[0].dtype, sparse_block_kv=sparse_block_kv,
        use_unaligned_ks=use_unaligned_ks)
    sparse_logits = deep_gemm.fp8_fp4_sparse_mqa_logits(
        q=q, kv=kv, weights=weights, metadata=metadata,
        num_max_sparse_blocks=num_max_sparse_blocks, sparse_block_kv=sparse_block_kv,
        use_unaligned_ks=use_unaligned_ks)

    # Dequantized reference: relu(Q_deq . KV_deq) * weight, summed over heads
    back = cast_back_from_fp4 if is_mxfp4 else cast_back_from_fp8
    q_deq = back(q_fp, q_sf, gran_k=32, use_packed_ue8m0=True).view(num_q_tokens, num_heads, head_dim).float()
    kv_deq = back(kv_fp, kv_sf, gran_k=32, use_packed_ue8m0=True).float()   # [num_kv_tokens, head_dim]
    scores = torch.einsum('qhd,td->qht', q_deq, kv_deq).relu_()
    ref_full = (scores * weights[:, :, None].float()).sum(1)               # [num_q_tokens, num_kv_tokens]

    kv_offsets = torch.arange(sparse_block_kv, device='cuda', dtype=torch.int32)
    block_offsets = (starts % sparse_block_kv)
    token_indices = (sparse_indices.unsqueeze(-1) * sparse_block_kv + block_offsets[:, None, None] + kv_offsets).flatten(1).long()
    nsb = torch.tensor(num_sparse_blocks, device='cuda')
    valid = torch.arange(token_indices.size(1), device='cuda')[None, :] < (nsb * sparse_block_kv)[:, None]
    valid &= token_indices >= starts[:, None]
    valid &= token_indices < ends[:, None]
    sp = sparse_logits[valid].float()
    ref = ref_full.gather(1, token_indices.clamp(0, num_kv_tokens - 1))[valid]
    x, y = sp.double(), ref.double()
    diff = (1 - 2 * (x * y).sum() / (x * x + y * y).sum()).item() if x.numel() else 0.0
    ok = diff < 1e-3
    print(f'[{fmt:6}] H={num_heads:2} SQ={num_q_tokens:4} KV={kv_len:6} bkv={sparse_block_kv} '
          f'nmsb={num_max_sparse_blocks} uaks={int(use_unaligned_ks)}: n={sp.numel()} diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    return ok


def main():
    deep_gemm.set_pdl(False)
    failed = False
    for fmt in ('mxfp8', 'mxfp4'):
        for num_heads in (8, 32):
            # The last two end inside a sparse block, so the gather clips the tensor's tail
            for (sq, kv, bkv, nmsb) in [(9, 2048, 8, 128), (64, 4096, 8, 512), (9, 1024, 16, 128), (9, 2051, 8, 128), (9, 1021, 16, 128)]:
                for uaks in (False, True):
                    failed |= not check(num_heads, fmt, sq, kv, bkv, nmsb, use_unaligned_ks=uaks)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
