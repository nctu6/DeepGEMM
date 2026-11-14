"""Order-independent form of the paged sparse MQA metadata, for bitwise comparisons between kernel versions.

The metadata kernel hands each Q block its range of KV splits with an atomic counter, so the order of the Q blocks
inside the split array (and the split indices the schedule entries point to) depends on which CTA finished first.
Everything else is deterministic: the content of every split, and, for each Q block, which of its splits every
schedule entry covers and where in the wave table the entry sits. `canonical_paged_metadata` renumbers the splits
by (q_token_base, original order) and returns the arrays that must be equal between two builds.
"""
import numpy as np
import torch

HEADER_WORDS = 4
SPLIT_HEADER_WORDS = 4
ENTRY_WORDS = 4


def canonical_paged_metadata(metadata: torch.Tensor, split_kv: int, sparse_block_kv: int):
    """Returns (header, splits, entries): `header` = (num_splits, num_waves, use_unaligned_ks, num_sms),
    `splits` (num_splits, words) sorted by Q block, `entries` (num_waves * num_sms, 4) with the split ranges
    renumbered to the sorted order."""
    raw = metadata.cpu().numpy()
    header = raw[:HEADER_WORDS * 4].view(np.uint32).copy()
    num_splits, num_waves, _, num_sms = (int(x) for x in header)
    blocks_per_split = split_kv // sparse_block_kv
    split_words = SPLIT_HEADER_WORDS + 2 * blocks_per_split
    splits_end = HEADER_WORDS * 4 + num_splits * split_words * 4
    splits = raw[HEADER_WORDS * 4:splits_end].view(np.uint32).reshape(num_splits, split_words)
    entries = raw[splits_end:splits_end + num_waves * num_sms * ENTRY_WORDS * 4].view(np.uint32)
    entries = entries.reshape(num_waves * num_sms, ENTRY_WORDS)
    order = np.argsort(splits[:, 0], kind='stable')
    rank = np.empty(num_splits, dtype=np.int64)
    rank[order] = np.arange(num_splits)
    canonical = entries.astype(np.int64)
    live = canonical[:, 1] > canonical[:, 0]
    canonical[live, 0] = rank[canonical[live, 0]]
    canonical[live, 1] = rank[entries[live, 1].astype(np.int64) - 1] + 1
    return header, splits[order], canonical


def same_paged_metadata(a: torch.Tensor, b: torch.Tensor, split_kv: int, sparse_block_kv: int) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(canonical_paged_metadata(a, split_kv, sparse_block_kv),
                                                    canonical_paged_metadata(b, split_kv, sparse_block_kv)))
