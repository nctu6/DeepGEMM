"""Pre-2.8.2 FP8 MQA logits entry points, kept for existing callers (e.g. TokenSpeed's DSA indexer).

Both forward to the `fp8_fp4_*` APIs with E4M3 queries (no query scale factors).
"""
from typing import Optional, Tuple

import torch

from ._C import fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits


def fp8_mqa_logits(q: torch.Tensor,
                   kv: Tuple[torch.Tensor, torch.Tensor],
                   weights: torch.Tensor,
                   cu_seq_len_k_start: torch.Tensor,
                   cu_seq_len_k_end: torch.Tensor,
                   clean_logits: bool = True,
                   max_seqlen_k: int = 0) -> torch.Tensor:
    """Contiguous MQA logits with compressed rows.

    Args:
        q: `[seq_len, num_heads, head_dim]` E4M3 queries.
        kv: `([seq_len_kv, head_dim]` E4M3 keys, `[seq_len_kv]` FP32 per-key scales`)`.
        weights: `[seq_len, num_heads]` per-head weights.
        cu_seq_len_k_start / cu_seq_len_k_end: `[seq_len]` int32 key range `[start, end)` of every query.
        clean_logits: must be False, as compressed rows are never cleaned.
        max_seqlen_k: maximum `end - start`, must be positive (the uncompressed layout was removed).

    Returns:
        `[seq_len, max_seqlen_k]` logits; row `i` holds keys `start[i] + j`, entries past its range are undefined.
    """
    if max_seqlen_k <= 0:
        raise ValueError('`max_seqlen_k` must be positive: only compressed logits are supported')
    if clean_logits:
        raise ValueError('Compressed logits cannot be cleaned, pass `clean_logits=False`')
    return fp8_fp4_mqa_logits((q, None), kv, weights, cu_seq_len_k_start, cu_seq_len_k_end, max_seqlen_k)


def fp8_paged_mqa_logits(q: torch.Tensor,
                         fused_kv_cache: torch.Tensor,
                         weights: torch.Tensor,
                         context_lens: torch.Tensor,
                         block_table: torch.Tensor,
                         schedule_meta: torch.Tensor,
                         max_context_len: int,
                         clean_logits: bool = False,
                         indices: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Paged MQA logits.

    Args:
        q: `[batch_size, next_n, num_heads, head_dim]` E4M3 queries.
        fused_kv_cache: `[num_blocks, block_kv, 1, head_dim + 4]` bytes, keys followed by FP32 scales per block.
        weights: `[batch_size * next_n, num_heads]` per-head weights.
        context_lens: `[batch_size, next_n]` int32 context length of every query token.
        block_table: `[batch_size, max_blocks]` int32 block indices.
        schedule_meta: output of `get_paged_mqa_logits_metadata`.
        max_context_len: logits width.
        clean_logits: fill entries at or beyond each row's context length with `-inf`.
        indices: varlen request indices (SM100 only).

    Returns:
        `[batch_size * next_n, max_context_len]` logits.
    """
    logits = fp8_fp4_paged_mqa_logits((q, None), fused_kv_cache, weights, context_lens, block_table,
                                      schedule_meta, max_context_len, indices)
    if clean_logits:
        positions = torch.arange(logits.size(-1), device=logits.device)
        logits.masked_fill_(positions[None, :] >= context_lens.reshape(-1, 1), float('-inf'))
    return logits
