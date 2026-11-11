"""Benchmark of the SM120 paged sparse MQA logits kernel (MXFP4 V4.1 rows) against the dense paged / contiguous kernels.

The sparse kernel scores only a candidate pool (`nmsb` blocks of 8 rows per query); the dense kernels score the whole
history, which the V4.1 Reindex pass then gathers the pool from. Each line is the kernel time (cold L2, kineto) in us.

  per-token  every query is its own request (decode): B requests of `ctx` rows
  broadcast  every query reads one request (a prefill chunk): the table has a zero row stride

  sm120_remote.sh dg scripts/sm120_sparse_paged_bench.py --mode per-token --queries 8 32 128 --ctx 16384 32768
  --save FILE / --cmp FILE   write the sparse logits / compare them bitwise with an earlier run (same shapes)
"""
import argparse
import torch
import deep_gemm
from deep_gemm.testing.bench import bench_kineto
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4
import sys
sys.path.insert(0, 'tests')
from test_attention import make_sparse_kv_block_indices, to_mqa_weights  # noqa: E402

PAGE, HEADS, DIM, BKV = 64, 32, 128, 8


def make_case(mode, num_q, ctx, nmsb, seed=0):
    torch.manual_seed(seed)
    requests = 1 if mode == 'broadcast' else num_q
    pages_per_request = ceil_div(ctx, PAGE)
    num_pages = requests * pages_per_request
    q_fp, q_sf = per_token_cast_to_fp4(torch.randn(num_q * HEADS, DIM, device='cuda', dtype=torch.bfloat16),
                                       use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q = q_fp.view(num_q, 1, HEADS, DIM // 2), q_sf.view(num_q, 1, HEADS)
    weights = to_mqa_weights(torch.randn(num_q, HEADS, device='cuda', dtype=torch.bfloat16))
    row = DIM // 2 + 4
    page_stride = ceil_div(PAGE * row, 128) * 128
    storage = torch.randint(0, 256, (num_pages, page_stride), device='cuda', dtype=torch.uint8)
    # UE8M0 scale words near 1.0 keep the logits finite
    sf = storage[:, PAGE * (DIM // 2):PAGE * row].view(num_pages, PAGE, 4)
    sf.copy_(torch.full_like(sf, 127))
    kv_cache = storage.as_strided((num_pages, PAGE, 1, row), (page_stride, row, row, 1))
    perm = torch.randperm(num_pages, device='cuda', dtype=torch.int32)
    if mode == 'broadcast':
        # The metadata kernel takes the row stride as the page bound, so the broadcast table is materialized
        table = perm.view(1, -1).expand(num_q, -1).contiguous()
        request_indices = [0] * num_q
    else:
        table = perm.view(requests, pages_per_request).contiguous()
        request_indices = list(range(num_q))
    sparse, _ = make_sparse_kv_block_indices([ctx] * num_q, request_indices, BKV, nmsb, seed)
    context_lens = torch.full((num_q,), ctx, device='cuda', dtype=torch.int32)
    indices = torch.tensor(request_indices, device='cuda', dtype=torch.int32)
    return dict(q=q, weights=weights, kv_cache=kv_cache, table=table, sparse=sparse, context_lens=context_lens,
                indices=indices, ctx=ctx, nmsb=nmsb, num_q=num_q, mode=mode)


def sparse_run(case):
    metadata = deep_gemm.get_paged_sparse_mqa_logits_metadata(
        case['context_lens'], case['table'], case['indices'], PAGE, case['sparse'], case['q'][0].dtype, BKV)
    return lambda: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
        q=case['q'], kv_cache=case['kv_cache'], weights=case['weights'], metadata=metadata,
        num_max_sparse_blocks=case['nmsb'], sparse_block_kv=BKV)


def dense_run(case):
    ctx, num_q = case['ctx'], case['num_q']
    weights = case['weights'].float().contiguous()
    if case['mode'] == 'broadcast':
        # One request's chunk: the contiguous kernel over the gathered history
        pages = case['table'][0].long()
        flat = case['kv_cache'][pages].view(len(pages), -1)
        values = flat[:, :PAGE * 64].reshape(-1, 64).view(torch.int8)
        scales = flat[:, PAGE * 64:PAGE * 68].reshape(-1).view(torch.int32)
        starts = torch.zeros(num_q, device='cuda', dtype=torch.int32)
        ends = torch.full((num_q,), ctx, device='cuda', dtype=torch.int32)
        q = (case['q'][0].view(num_q, HEADS, 64), case['q'][1].view(num_q, HEADS))
        return lambda: deep_gemm.fp8_fp4_mqa_logits(
            q=q, kv=(values, scales), weights=weights, cu_seq_len_k_start=starts, cu_seq_len_k_end=ends,
            clean_logits=False, max_seqlen_k=ctx, logits_dtype=torch.float32)
    context = case['context_lens'][:, None].contiguous()
    meta = deep_gemm.get_paged_mqa_logits_metadata(context, PAGE, deep_gemm.get_num_sms())
    return lambda: deep_gemm.fp8_fp4_paged_mqa_logits(
        q=case['q'], kv_cache=case['kv_cache'], weights=weights, context_lens=context, block_table=case['table'],
        schedule_meta=meta, max_context_len=ctx, clean_logits=False, logits_dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='per-token', choices=['per-token', 'broadcast'])
    parser.add_argument('--queries', type=int, nargs='+', default=[8, 32, 128])
    parser.add_argument('--ctx', type=int, nargs='+', default=[16384, 32768])
    parser.add_argument('--nmsb', type=int, nargs='+', default=[2048])
    parser.add_argument('--no-dense', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--cmp')
    args = parser.parse_args()
    deep_gemm.set_pdl(False)
    saved = {}
    reference = torch.load(args.cmp) if args.cmp else {}
    for nmsb in args.nmsb:
        for ctx in args.ctx:
            for num_q in args.queries:
                case = make_case(args.mode, num_q, ctx, nmsb)
                run = sparse_run(case)
                out = run()
                torch.cuda.synchronize()
                t_sparse = bench_kineto(run, 'sm120_sparse_mqa_logits') * 1e6
                line = f'{args.mode:9} Q={num_q:5} ctx={ctx:6} pool={nmsb * BKV:6}: sparse {t_sparse:8.1f} us'
                if not args.no_dense:
                    d = dense_run(case)
                    d()
                    torch.cuda.synchronize()
                    name = 'sm120_fp8_fp4_mqa_logits' if args.mode == 'broadcast' else 'sm120_fp8_paged_mqa_logits'
                    t_dense = bench_kineto(d, name) * 1e6
                    line += f', dense {t_dense:8.1f} us, dense/sparse {t_dense / t_sparse:5.2f}'
                key = f'{args.mode}-{num_q}-{ctx}-{nmsb}'
                host = out.float().cpu()
                saved[key] = host
                if key in reference:
                    line += f', bitwise equal to saved: {torch.equal(host, reference[key])}'
                print(line, flush=True)
                del case, run
    if args.save:
        torch.save(saved, args.save)


if __name__ == '__main__':
    main()
