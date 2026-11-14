"""Paged sparse MQA metadata (`get_paged_sparse_mqa_logits_metadata`, the SM100 metadata kernel shared with SM120)
against a NumPy model of its output.

The model merges the two queries of each Q block, cuts the merged blocks into splits and checks every word of every
split (header, physical block, slot offsets, padding). The schedule is checked by what it must cover: each Q block's
splits in entries of at most 8 splits, as even as possible, once. Q blocks pair queries from the first query of a run of
equal request indices, so the cases include runs longer than one search round of the kernel (1024 queries), runs that
repeat a request id after another request, and decode (every query its own request).

The order of the Q blocks in the split array depends on which CTA finishes first, so both sides are put in the order of
`tests/sparse_metadata_canon.py` before they are compared.
"""
import os
import sys
import numpy as np
import torch
import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sparse_metadata_canon import canonical_paged_metadata  # noqa: E402
from test_attention import make_sparse_kv_block_indices  # noqa: E402

PAGE, BLOCK_KV, SPLIT_KV, MAX_SPLITS_PER_ENTRY = 64, 8, 640, 8
BLOCKS_PER_SPLIT = SPLIT_KV // BLOCK_KV
INVALID_SLOT = 0xFFFF


def expected_metadata(sparse, context_lens, table, request_indices):
    """Canonical splits (sorted by Q block) and the Q blocks `(q_token_base, num_q_tokens, first_split, num_splits)`."""
    num_q, nmsb = sparse.shape
    splits, q_blocks = [], []
    q = 0
    while q < num_q:
        start = q
        while start > 0 and request_indices[start - 1] == request_indices[q]:
            start -= 1
        if (q - start) % 2 != 0:
            q += 1
            continue
        pair = q + 1 < num_q and request_indices[q + 1] == request_indices[q]
        rows = [q, q + 1] if pair else [q]
        counts = [min(nmsb, ceil_div(int(context_lens[r]), BLOCK_KV)) for r in rows]
        lists = [sparse[r, :n].tolist() for r, n in zip(rows, counts)]
        lists += [[]] * (2 - len(lists))
        merged, p = [], [0, 0]
        while p[0] < len(lists[0]) or p[1] < len(lists[1]):
            v0 = lists[0][p[0]] if p[0] < len(lists[0]) else 1 << 40
            v1 = lists[1][p[1]] if p[1] < len(lists[1]) else 1 << 40
            merged.append((min(v0, v1), p[0], v0 <= v1, p[1], v1 <= v0))
            p[0] += v0 <= v1
            p[1] += v1 <= v0
        first_split = len(splits)
        for begin in range(0, len(merged), BLOCKS_PER_SPLIT):
            chunk = merged[begin:begin + BLOCKS_PER_SPLIT]
            base0, base1 = chunk[0][1], chunk[0][3]
            words = [q, len(chunk), base0, base1 if pair else 0xFFFFFFFF]
            for logical, p0, in0, p1, in1 in chunk:
                physical = int(table[q, logical // (PAGE // BLOCK_KV)]) * (PAGE // BLOCK_KV) + logical % (PAGE // BLOCK_KV)
                words += [physical, ((p0 - base0) if in0 else INVALID_SLOT) | (((p1 - base1) if in1 else INVALID_SLOT) << 16)]
            words += [0, 0xFFFFFFFF] * (BLOCKS_PER_SPLIT - len(chunk))
            splits.append(words)
        q_blocks.append((q, len(rows), first_split, len(splits) - first_split))
        q += 1
    return np.array(splits, dtype=np.uint32).reshape(len(splits), -1), q_blocks


def expected_entries(q_blocks):
    entries = []
    for q, num_q_tokens, first, count in q_blocks:
        num_entries = ceil_div(count, MAX_SPLITS_PER_ENTRY)
        begin = first
        for i in range(num_entries):
            end = begin + count // num_entries + (i < count % num_entries)
            entries.append((begin, end, q, num_q_tokens))
            begin = end
    return sorted(entries)


def run_case(name, context_lens, request_indices, nmsb, seed):
    num_q = len(context_lens)
    torch.manual_seed(seed)
    max_pages = ceil_div(max(context_lens), PAGE)
    sparse, _ = make_sparse_kv_block_indices(context_lens, request_indices, BLOCK_KV, nmsb, seed)
    table = torch.randperm(num_q * max_pages, device='cuda', dtype=torch.int32).view(num_q, max_pages).contiguous()
    lens = torch.tensor(context_lens, device='cuda', dtype=torch.int32)
    indices = torch.tensor(request_indices, device='cuda', dtype=torch.int32)
    qk_dtype = per_token_cast_to_fp4(torch.randn(1, 128, device='cuda', dtype=torch.bfloat16),
                                     use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)[0].dtype
    for repeat in range(3):
        metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(lens, table, indices, PAGE, sparse, qk_dtype, BLOCK_KV)
        header, splits, entries = canonical_paged_metadata(metadata, SPLIT_KV, BLOCK_KV)
        want_splits, q_blocks = expected_metadata(sparse.cpu().numpy(), context_lens, table.cpu().numpy(), request_indices)
        assert splits.shape == want_splits.shape and (splits == want_splits).all(), (name, repeat, 'splits')
        want_entries = expected_entries(q_blocks)
        live = entries[entries[:, 1] > entries[:, 0]]
        # Entries are in the wave table in balance order; here only what they cover
        assert sorted(map(tuple, live.tolist())) == want_entries, (name, repeat, 'schedule entries')
        num_sms = int(header[3])
        assert int(header[0]) == len(want_splits) and int(header[1]) == max(1, ceil_div(len(want_entries), num_sms)), (name, repeat, 'header')
        assert (entries[entries[:, 1] <= entries[:, 0]] == 0).all(), (name, repeat, 'empty entries')
    print(f'  {name}: {num_q} queries, {len(want_splits)} splits, {len(want_entries)} entries, {int(header[1])} waves OK')


def test_paged_sparse_metadata():
    rng = np.random.default_rng(0)
    # Decode: every query is its own request; some contexts hold fewer blocks than the pool
    contexts = rng.integers(200, 4096, size=40).tolist()
    run_case('decode', contexts, list(range(40)), 256, 1)
    # One request, 2600 queries: the run start is 1024 + 1024 + 552 tokens back for the last queries
    contexts = (3000 + np.arange(2600) % 1000).tolist()
    run_case('prefill chunk, 3 search rounds', contexts, [0] * 2600, 64, 2)
    # Runs of different length, a request id that comes back, and a run of one at the end
    runs = [(0, 1), (1, 2), (2, 3), (0, 5), (3, 1100), (1, 7), (4, 1)]
    request_indices = [r for r, n in runs for _ in range(n)]
    contexts = rng.integers(300, 4000, size=len(request_indices)).tolist()
    run_case('mixed runs', contexts, request_indices, 128, 3)
    # Pool as large as the V4.1 one (2048 blocks, 4096 merged blocks per Q block)
    contexts = [32768] * 70
    run_case('V4.1 pool', contexts, [0] * 35 + [1] * 35, 2048, 4)


if __name__ == '__main__':
    if get_arch_major() not in (10, 12):
        print('Skipped: SM100 / SM120 only')
        sys.exit(0)
    test_paged_sparse_metadata()
