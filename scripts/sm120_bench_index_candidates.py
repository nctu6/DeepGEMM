"""`dsa_index_candidates` of the sharded (DCP) V3.2 indexer on SM120: the DeepSelect candidate top-k against the framework
top-k it replaced and against the Triton leaf.
  new     `deep_gemm_dsa_index_candidates` (DeepGEMM paged logits + DeepSelect over each row's owned candidates)
  topk    the same leaf with `torch.topk` (what it did before)
  triton  `triton_dsa_index_candidates`
A rank owns `1/ranks` of the request's pages (interleaved). Rows are the query tile: a decode batch or a prefill tile
(`select_dsa_topk` bounds the tile so its logits fit `max_logits_bytes`). CUDA graph over cache copies larger than L2, per call.
  sm120_remote.sh ts scripts/sm120_bench_index_candidates.py
"""
import argparse
import math
import torch
from sm120_baseline import bench_graph
from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_dg
from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

PAGE, HEADS, DIM, TOPK = 64, 64, 128, 2048
ROW_BYTES = DIM + 4


def make_cache(num_pages):
    keys = torch.randn(num_pages, PAGE, DIM, device='cuda', dtype=torch.bfloat16)
    scales = keys.abs().amax(-1).float().clamp_min(1e-4) / 448.0
    cache = torch.empty(num_pages, PAGE * ROW_BYTES, device='cuda', dtype=torch.uint8)
    cache[:, :PAGE * DIM] = (keys / scales[..., None]).to(torch.float8_e4m3fn).view(num_pages, -1).view(torch.uint8)
    cache[:, PAGE * DIM:] = scales.view(torch.uint8).view(num_pages, -1)
    return cache.view(num_pages * PAGE, ROW_BYTES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=str, default='1,8,32,128,512,2048')
    parser.add_argument('--pages', type=str, default='256,1024,2048', help='pages of the global page table (context / 64)')
    parser.add_argument('--ranks', type=str, default='2,4,8')
    args = parser.parse_args()
    print('us per call (graph, cold cache); new vs topk / triton')
    print(f'{"rows":>5} {"pages":>6} {"ranks":>5} | {"new":>9} {"topk":>9} {"triton":>9} | topk/new triton/new')
    ratios_topk, ratios_triton = [], []
    for rows in (int(x) for x in args.rows.split(',')):
        for pages in (int(x) for x in args.pages.split(',')):
            for ranks in (int(x) for x in args.ranks.split(',')):
                torch.manual_seed(0)
                table_row = torch.arange(1, pages + 1, device='cuda', dtype=torch.int32)
                owned = (table_row - 1) % ranks == 1
                local_row = torch.where(owned, (table_row - 1) // ranks + 1, -1).to(torch.int32)
                local_pages = int(owned.sum()) + 1
                copies = [make_cache(local_pages) for _ in range(max(1, min(6, math.ceil(300e6 / (local_pages * PAGE * ROW_BYTES)))))]
                table = local_row[None, :].contiguous()
                causal = torch.randint(pages * PAGE * 3 // 4, pages * PAGE + 1, (rows, ), device='cuda', dtype=torch.int32)
                requests = torch.zeros(rows, device='cuda', dtype=torch.int32)
                q = torch.randn(rows, HEADS, DIM, device='cuda', dtype=torch.bfloat16)
                weights = torch.randn(rows, HEADS, device='cuda')
                kwargs = dict(page_size=PAGE, topk=TOPK, softmax_scale=DIM ** -0.5, initial_tokens=4, local_tokens=8)

                def new(cache):
                    return dsa_dg.deep_gemm_dsa_index_candidates(q, weights, cache, table, requests, causal, **kwargs)

                def topk_path(cache):
                    supports = dsa_dg.deepselect_supports
                    dsa_dg.deepselect_supports = lambda *_: False
                    try:
                        return dsa_dg.deep_gemm_dsa_index_candidates(q, weights, cache, table, requests, causal, **kwargs)
                    finally:
                        dsa_dg.deepselect_supports = supports

                def triton_path(cache):
                    return dsa_triton.triton_dsa_index_candidates(q, weights, cache, table, requests, causal, **kwargs)

                times = {}
                for name, fn in (('new', new), ('topk', topk_path), ('triton', triton_path)):
                    fn(copies[0])
                    torch.cuda.synchronize()
                    times[name] = min(bench_graph(fn, copies) for _ in range(3)) * 1e6
                ratios_topk.append(times['topk'] / times['new'])
                ratios_triton.append(times['triton'] / times['new'])
                print(f'{rows:5} {pages:6} {ranks:5} | {times["new"]:9.1f} {times["topk"]:9.1f} {times["triton"]:9.1f} | '
                      f'{ratios_topk[-1]:7.2f} {ratios_triton[-1]:9.2f}', flush=True)
                del copies
    for name, r in (('topk', ratios_topk), ('triton', ratios_triton)):
        print(f'{name}/new: geomean {math.exp(sum(map(math.log, r)) / len(r)):.2f}, min {min(r):.2f}, max {max(r):.2f}')


if __name__ == '__main__':
    main()
