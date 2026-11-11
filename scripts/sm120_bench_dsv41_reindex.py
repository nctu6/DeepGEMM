"""Reindex pass of the V4.1 SM120 leaf `deep_gemm_sm120_dsv41_index_topk`: the pool-only sparse scorer against the
whole-history scorer and the Triton leaf (CUDA graph with the L2 flushed before every replay, us per call, median of 3).

  sm120_remote.sh ts scripts/sm120_bench_dsv41_reindex.py decode 32768 65536     # 8 / 32 / 128 requests, query tile 256
  ROWS="128 512 1024" sm120_remote.sh ts scripts/sm120_bench_dsv41_reindex.py prefill 32768   # one request's chunk, tile 1024

Needs the TokenSpeed venv with `tokenspeed_kernel` importable (the runtime tile sizes are the ones deepseek_v41.py passes).
`rule picks` is what `_sparse_reindex_wins` chooses for the shape. PDL=0 turns programmatic dependent launch off.
"""
import os, sys, statistics
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41 import deep_gemm as leaf

from tokenspeed_kernel.platform import pdl_enabled
pdl_enabled(bool(int(os.environ.get("PDL", "1"))))
mode = sys.argv[1]
ctxs = [int(x) for x in sys.argv[2:]] or [32768, 65536]
rows_list = [int(x) for x in os.environ.get("ROWS", "8 32 128" if mode == "decode" else "128 512 1024").split()]
shared = mode == "prefill"


def case(rows, history):
    torch.manual_seed(7)
    pages = -(-history // 64) + 1
    copies = 1 if shared else rows
    keys = torch.randn(pages * 64 * copies, 128, dtype=torch.bfloat16, device="cuda")
    cache = torch.zeros(pages * copies, 64, 68, dtype=torch.uint8, device="cuda")
    dsv41.cache_scatter(keys, cache, torch.arange(len(keys), device="cuda"), "index")
    del keys
    q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.rand(rows, 32, dtype=torch.bfloat16, device="cuda")
    if shared:
        table = torch.randperm(pages, device="cuda").int()[None, :].expand(rows, -1)
        visible = history - rows + 1 + torch.arange(rows, device="cuda").int()
    else:
        table = torch.randperm(pages * rows, device="cuda").int().view(rows, pages)
        visible = history - torch.randint(0, 100, (rows,), device="cuda").int()
    return q, weights, cache, table, visible


def run(q, weights, cache, table, visible, solution, candidates, candidate_topk):
    return dsv41.index_topk(q, weights, cache, table, visible, candidates, 512, candidate_topk, 8, 1024 if shared else 256, 256, None, None, solution)


def graph_us(fn, replays=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        fn()
    flush = torch.empty(300 << 20, dtype=torch.uint8, device="cuda")
    ts = []
    for _ in range(3):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        total = 0.0
        for _ in range(replays):
            flush.zero_()
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            total += a.elapsed_time(b)
        ts.append(total / replays * 1e3)
    return statistics.median(ts)


orig = leaf._sparse_reindex_wins
for ctx in ctxs:
    for rows in rows_list:
        q, weights, cache, table, visible = case(rows, ctx)
        _, _, blocks, _ = run(q, weights, cache, table, visible, None, None, 2048)
        out = {}
        for name, rule, solution in (("sparse", True, None), ("dense", False, None), ("triton", None, "triton")):
            leaf._sparse_reindex_wins = (lambda *a, _r=rule: _r) if rule is not None else orig
            out[name] = graph_us(lambda: run(q, weights, cache, table, visible, solution, blocks, 0))
        leaf._sparse_reindex_wins = orig
        rule = orig(table.shape[1] * 64, blocks.shape[1] * 8)
        print(f"{mode:7} rows={rows:5} ctx={ctx:6}: sparse {out['sparse']:8.1f} dense {out['dense']:8.1f} triton {out['triton']:9.1f} us  dense/sparse {out['dense'] / out['sparse']:5.2f}  rule picks {'sparse' if rule else 'dense'}", flush=True)
        del q, weights, cache, table, visible, blocks
        torch.cuda.empty_cache()
