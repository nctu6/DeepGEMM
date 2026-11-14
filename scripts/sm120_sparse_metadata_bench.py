"""Time of the shared sparse MQA metadata kernel (`sm100_sparse_mqa_logits_metadata`, paged form) on SM120.

  per-token  every query is its own request (decode): B requests of `ctx` rows
  broadcast  every query reads one request (a prefill chunk)

  sm120_remote.sh dg scripts/sm120_sparse_metadata_bench.py --mode broadcast --queries 128 256 1024 4096
  --save FILE / --cmp FILE   write the canonical metadata / compare it with an earlier run (order-independent, see
                             tests/sparse_metadata_canon.py) and the sparse logits bitwise
"""
import argparse
import hashlib
import sys

import torch
import deep_gemm
from deep_gemm.testing.bench import bench_kineto

sys.path.insert(0, 'tests')
from sm120_sparse_paged_bench import BKV, PAGE, make_case  # noqa: E402
from sparse_metadata_canon import canonical_paged_metadata  # noqa: E402

SPLIT_KV = 640


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='broadcast', choices=['per-token', 'broadcast'])
    parser.add_argument('--queries', type=int, nargs='+', default=[128, 256, 1024, 4096])
    parser.add_argument('--ctx', type=int, nargs='+', default=[32768])
    parser.add_argument('--nmsb', type=int, nargs='+', default=[2048])
    parser.add_argument('--save')
    parser.add_argument('--cmp')
    args = parser.parse_args()
    deep_gemm.set_pdl(False)
    saved = {}
    reference = torch.load(args.cmp, weights_only=False) if args.cmp else {}
    for nmsb in args.nmsb:
        for ctx in args.ctx:
            for num_q in args.queries:
                case = make_case(args.mode, num_q, ctx, nmsb)

                def metadata():
                    return deep_gemm.get_paged_sparse_mqa_logits_metadata(
                        case['context_lens'], case['table'], case['indices'], PAGE, case['sparse'],
                        case['q'][0].dtype, BKV)

                meta = metadata()
                logits = deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                    q=case['q'], kv_cache=case['kv_cache'], weights=case['weights'], metadata=meta,
                    num_max_sparse_blocks=nmsb, sparse_block_kv=BKV)
                torch.cuda.synchronize()
                t_meta = bench_kineto(metadata, 'sm100_sparse_mqa_logits_metadata') * 1e6
                t_logits = bench_kineto(lambda: deep_gemm.fp8_fp4_paged_sparse_mqa_logits(
                    q=case['q'], kv_cache=case['kv_cache'], weights=case['weights'], metadata=meta,
                    num_max_sparse_blocks=nmsb, sparse_block_kv=BKV), 'sm120_sparse_mqa_logits') * 1e6
                header = [int(x) for x in canonical_paged_metadata(meta, SPLIT_KV, BKV)[0]]
                line = (f'{args.mode:9} Q={num_q:5} ctx={ctx:6} pool={nmsb * BKV:6}: metadata {t_meta:8.1f} us, '
                        f'logits {t_logits:8.1f} us, splits {header[0]} waves {header[1]}')
                key = f'{args.mode}-{num_q}-{ctx}-{nmsb}'
                canon = canonical_paged_metadata(meta, SPLIT_KV, BKV)
                digest = hashlib.sha256(logits.cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
                saved[key] = (canon, digest)
                if key in reference:
                    (ref_canon, ref_logits) = reference[key]
                    same_meta = all((x == y).all() for x, y in zip(canon, ref_canon))
                    line += f', metadata equal: {same_meta}, logits equal: {digest == ref_logits}'
                print(line, flush=True)
                del case, meta, logits
    if args.save:
        torch.save(saved, args.save)


if __name__ == '__main__':
    main()
