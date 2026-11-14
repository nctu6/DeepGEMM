# Host time per call of the SM120 launchers on the V3.2 DSA / V4.1 serving paths (task 61).
#
# Same method as `sm120_bench_host_dsa.py` (task 37): the CPU time of the call (`perf_counter_ns` around it, no
# synchronization inside the timed region, one sync every 25 calls so the launch queue never fills; the GPU work of the
# shapes here is shorter than the host path). Medians over many calls. Run in the TokenSpeed venv with the build under
# test first on PYTHONPATH (`sm120_remote.sh ts scripts/sm120_bench_host_launchers.py`); set `DG_JIT_CACHE_DIR` to a
# warm cache. `--only a,b` selects cases; the path of the `deep_gemm` under test is printed.
import argparse
import os
import statistics
import sys
import time
import torch
import deep_gemm
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import per_block_cast_to_fp8
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tests'))
def host_time(fn, iters, warm=100, sync_every=25):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
        if (i + 1) % sync_every == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return statistics.median(times) / 1e3
def case_paged(batch=8, next_n=1, heads=64, ctx=4096):
    from sm120_paged_mqa_bench import make_case
    kwargs, _ = make_case(batch, next_n, heads, 128, ctx, fmt='fp8')
    meta_args = (kwargs['context_lens'], 64, deep_gemm.get_num_sms())
    meta = lambda: deep_gemm.get_paged_mqa_logits_metadata(*meta_args)
    logits = lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
    return meta, logits
def case_paged_mx(batch=8, next_n=1, heads=32, ctx=4096):
    from sm120_paged_mqa_bench import make_case
    kwargs, _ = make_case(batch, next_n, heads, 128, ctx, fmt='mxfp4')
    return lambda: deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
def case_sparse(heads=32, tokens=128, kv_len=4096, block=8, max_blocks=256):
    from test_attention import make_sparse_kv_block_indices, to_mqa_weights
    head_dim = 128
    torch.manual_seed(0)
    q_fp, q_sf = per_token_cast_to_fp4(torch.randn((tokens, heads, head_dim), device='cuda', dtype=torch.bfloat16).view(-1, head_dim),
                                       use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q = q_fp.view(tokens, heads, head_dim // 2), q_sf.view(tokens, heads)
    weights = to_mqa_weights(torch.randn((tokens, heads), device='cuda', dtype=torch.bfloat16))
    indices, _ = make_sparse_kv_block_indices([kv_len] * tokens, None, block, max_blocks, 0, [0] * tokens)
    kv_fp, kv_sf = per_token_cast_to_fp4(torch.randn((kv_len, head_dim), device='cuda', dtype=torch.bfloat16),
                                         use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    kv = kv_fp, kv_sf.view(kv_len)
    starts = torch.zeros(tokens, device='cuda', dtype=torch.int32)
    ends = torch.full((tokens,), kv_len, device='cuda', dtype=torch.int32)
    meta_fn = lambda: deep_gemm.get_sparse_mqa_logits_metadata(
        cu_seq_len_k_start=starts, cu_seq_len_k_end=ends, num_kv_tokens=kv_len,
        sparse_kv_block_indices=indices, qk_dtype=q[0].dtype, sparse_block_kv=block)
    metadata = meta_fn()
    logits = lambda: deep_gemm.fp8_fp4_sparse_mqa_logits(
        q=q, kv=kv, weights=weights, metadata=metadata, num_max_sparse_blocks=max_blocks, sparse_block_kv=block)
    return meta_fn, logits
def case_gemms(m=128, n=4096, k=7168):
    out = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
    x = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
    a_q, a_sf = per_token_cast_to_fp8(x, use_ue8m0=False, gran_k=128)
    b_q, b_sf = per_block_cast_to_fp8(w, use_ue8m0=False)
    fp8_1d2d = lambda: deep_gemm.fp8_gemm_nt((a_q, a_sf), (b_q, b_sf), out)
    a4, sa4 = per_token_cast_to_fp4(x, use_ue8m0=True, gran_k=32)
    b4, sb4 = per_token_cast_to_fp4(w, use_ue8m0=True, gran_k=32)
    sa4 = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sa4)
    sb4 = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sb4)
    fp4_1d1d = lambda: deep_gemm.fp8_fp4_gemm_nt((a4, sa4), (b4, sb4), out, recipe=(1, 1, 32))
    bf16 = lambda: deep_gemm.bf16_gemm_nt(x, w, out)
    # A small BF16 GEMM, shorter on the GPU than the host path
    xs, ws, outs = x[:, :1024].contiguous(), w[:512, :1024].contiguous(), torch.empty((m, 512), device='cuda', dtype=torch.bfloat16)
    bf16_small = lambda: deep_gemm.bf16_gemm_nt(xs, ws, outs)
    return fp8_1d2d, fp4_1d1d, bf16, bf16_small
def case_hc_prenorm(m=128, n=24, k=28672):
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
    b = torch.randn((n, k), dtype=torch.float, device='cuda')
    d = torch.empty((16, m, n), dtype=torch.float, device='cuda')
    s = torch.empty((16, m), dtype=torch.float, device='cuda')
    return lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=16)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=3000)
    parser.add_argument('--only', type=str, default=None)
    args = parser.parse_args()
    deep_gemm.set_pdl(True)
    print(f'deep_gemm: {deep_gemm.__file__}')
    cases = {}
    cases['paged_metadata'], cases['paged_fp8_logits'] = case_paged()
    cases['paged_mxfp4_logits'] = case_paged_mx()
    cases['sparse_metadata'], cases['sparse_mxfp4_logits'] = case_sparse()
    cases['gemm_fp8_1d2d'], cases['gemm_fp4_1d1d'], cases['gemm_bf16'], cases['gemm_bf16_small'] = case_gemms()
    cases['hc_prenorm'] = case_hc_prenorm()
    for name, fn in cases.items():
        if args.only and name not in args.only.split(','):
            continue
        # BF16 GEMMs go to cuBLASLt unless deterministic mode asks for the DeepGEMM kernel
        deep_gemm.use_deterministic_algorithms(name.startswith('gemm_bf16'))
        fn()
        torch.cuda.synchronize()
        print(f'{name:22s} host median {host_time(fn, args.iters):7.2f} us', flush=True)
if __name__ == '__main__':
    main()
