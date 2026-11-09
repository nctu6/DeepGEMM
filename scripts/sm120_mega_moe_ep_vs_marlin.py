# SM120 `fp8_fp4_mega_moe` across two GPUs (CUDA IPC over PCIe) against what TokenSpeed runs for the DeepSeek V4 / V4.1
# MXFP4 experts at expert parallelism 2 on SM120: Marlin W4A16 on each rank's experts plus the collective that sums the
# ranks' partial outputs. Run on two GPUs, with TokenSpeed on PYTHONPATH:
#   CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node 2 scripts/sm120_mega_moe_ep_vs_marlin.py
# Variants (T tokens per rank, CUDA graph, every call after a 512 MB read whose own time is subtracted; the time is the
# slower rank's, the median of 3 replays of 10 calls):
#   mega     stage the inputs and call fp8_fp4_mega_moe: each rank owns half of the experts and its own T tokens
#   marlin_ar  attention TP (replicated tokens): Marlin on the local half of the experts for the same T tokens on both
#              ranks, then an NCCL all-reduce of the [T, H] BF16 partial outputs (the `all_reduce` of `moe_output_all_reduced`)
#   marlin_ar2 the same attention TP serving the same 2T tokens in total: Marlin and the all-reduce over 2T replicated tokens
#   marlin_rsag  attention DP: all-gather the [T, H] inputs to 2T tokens, Marlin on the local experts for 2T tokens, and
#              reduce-scatter the partial outputs back to T (the token_all_gather / token_reduce_scatter of the runtime)
# Routing, the router GEMM and the shared expert are the same in all variants and left out. Env: SHAPES, TOKENS, PDL
import os, statistics, types
import torch, torch.distributed as dist
import deep_gemm
from deep_gemm.testing import calc_diff
from tokenspeed_kernel.ops.moe.marlin import marlin_mxfp4_moe_weights, marlin_mxfp4_precomputed_moe_apply
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import stage_mxfp4_mega_moe_inputs

rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
deep_gemm.set_pdl(bool(int(os.environ.get('PDL', '0'))))
SHAPES = {k: v for k, v in {'V4': (4096, 2048, 256, 6), 'V4.1': (5120, 2304, 384, 6)}.items() if k in os.environ.get('SHAPES', 'V4 V4.1').split()}
TOKENS = [int(t) for t in os.environ.get('TOKENS', '1 4 16 64 256 1024 4096').split()]
REPEATS = 3
FLUSH = torch.ones(128 << 20, dtype=torch.float, device='cuda')   # 512 MB, 4x the L2
FLUSH_OUT = torch.empty((), dtype=torch.float, device='cuda')
def flush():
    torch.sum(FLUSH, dim=0, out=FLUSH_OUT)
def slowest(values):
    t = torch.tensor(values, device='cuda', dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.tolist()
def graph_time_raw(fn, iters):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    dist.barrier()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            flush(); fn()
    dist.barrier()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(REPEATS):
        dist.barrier()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / iters)
    return slowest(ts)
FLUSH_US = None
def graph_time(fn, iters=10):
    global FLUSH_US
    if FLUSH_US is None:
        FLUSH_US = statistics.median(graph_time_raw(lambda: None, 20))
    ts = [t - FLUSH_US for t in graph_time_raw(fn, iters)]
    return statistics.median(ts), ts
class W(torch.nn.Module):
    pass

for name, (H, I, E, K) in SHAPES.items():
    El = E // world
    lo, hi = rank * El, (rank + 1) * El
    gen = torch.Generator(device='cuda').manual_seed(0)
    # Every rank draws the same global weights and keeps its experts
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device='cuda', generator=gen)[lo:hi].contiguous()
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device='cuda', generator=gen)[lo:hi].contiguous()
    s13 = torch.randint(118, 121, (E, 2 * I, H // 32), dtype=torch.uint8, device='cuda', generator=gen)[lo:hi].contiguous()
    s2 = torch.randint(118, 121, (E, H, I // 32), dtype=torch.uint8, device='cuda', generator=gen)[lo:hi].contiguous()
    to_f = lambda s: (s.int() << 23).view(torch.float)
    l1 = (w13.view(torch.int8), deep_gemm.transform_sf_into_required_layout(to_f(s13), 2 * I, H, (1, 32), El))
    l2 = (w2.view(torch.int8), deep_gemm.transform_sf_into_required_layout(to_f(s2), H, I, (1, 32), El))
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    m = W()
    m.w13_weight, m.w13_weight_scale = torch.nn.Parameter(w13.clone(), requires_grad=False), torch.nn.Parameter(s13.clone(), requires_grad=False)
    m.w2_weight, m.w2_weight_scale = torch.nn.Parameter(w2.clone(), requires_grad=False), torch.nn.Parameter(s2.clone(), requires_grad=False)
    m.num_local_experts, m.ep_size, m.ep_rank = El, world, rank
    m.activation = 'swiglu'; m.swiglu_arg = types.SimpleNamespace(limit=10.0, alpha=None); m.swiglu_beta = None
    m.top_k = K; m.hidden_size = H; m.intermediate_size = I
    plan = {'activation': 'swiglu', 'internal_activation_dtype': 'input'}
    marlin_mxfp4_moe_weights(plan, m)
    del w13, w2, s13, s2
    for T in TOKENS:
        # Per-rank tokens and routing (each rank has its own), and the all-ranks view for the gathered variant
        xs, ids, wts = [], [], []
        for r in range(world):
            g = torch.Generator(device='cuda').manual_seed(100 + r)
            xs.append((torch.randn(T, H, device='cuda', generator=g) * 0.5).bfloat16())
            tw, ti = torch.topk(torch.randn(T, E, device='cuda', generator=g), K, dim=-1)
            ids.append(ti.to(torch.int32)); wts.append(tw.softmax(-1))
        x, topk_ids, topk_weights = xs[rank], ids[rank], wts[rank]
        x_all, ids_all, wts_all = torch.cat(xs), torch.cat(ids), torch.cat(wts)
        buf = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, E, T, K, H, I)
        ids64 = topk_ids.to(torch.int64)
        y = torch.empty(T, H, dtype=torch.bfloat16, device='cuda')
        def mega():
            stage_mxfp4_mega_moe_inputs(x, topk_weights, ids64, buf.x[:T], buf.x_sf[:T], buf.topk_idx[:T], buf.topk_weights[:T])
            deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buf, activation_clamp=10.0, fast_math=True)
        # Attention TP: the same tokens on every rank, partial outputs all-reduced
        x_rep, ids_rep, wts_rep = xs[0], ids[0], wts[0]
        out_ar = torch.empty(T, H, dtype=torch.bfloat16, device='cuda')
        def marlin_ar():
            out_ar.copy_(marlin_mxfp4_precomputed_moe_apply(plan, x_rep, m, None, topk_weights=wts_rep, topk_ids=ids_rep))
            dist.all_reduce(out_ar)
        out_ar2 = torch.empty(world * T, H, dtype=torch.bfloat16, device='cuda')
        def marlin_ar2():
            out_ar2.copy_(marlin_mxfp4_precomputed_moe_apply(plan, x_all, m, None, topk_weights=wts_all, topk_ids=ids_all))
            dist.all_reduce(out_ar2)
        # Attention DP: gather the inputs, run the local experts for every rank's tokens, reduce-scatter the outputs
        gathered = torch.empty(world * T, H, dtype=torch.bfloat16, device='cuda')
        out_rs = torch.empty(T, H, dtype=torch.bfloat16, device='cuda')
        def marlin_rsag():
            dist.all_gather_into_tensor(gathered, x)
            partial = marlin_mxfp4_precomputed_moe_apply(plan, gathered, m, None, topk_weights=wts_all, topk_ids=ids_all)
            dist.reduce_scatter_tensor(out_rs, partial)
        mega(); marlin_rsag()
        dm = calc_diff(y.float(), out_rs.float())
        res = {'mega': graph_time(mega), 'marlin_ar': graph_time(marlin_ar), 'marlin_ar2': graph_time(marlin_ar2), 'marlin_rsag': graph_time(marlin_rsag)}
        if rank == 0:
            print(f'{name} T={T:5}/rank: mega {res["mega"][0]:8.1f} us, marlin_ar {res["marlin_ar"][0]:8.1f} us '
                  f'({res["marlin_ar"][0] / res["mega"][0]:.2f}x), marlin_ar2 {res["marlin_ar2"][0]:8.1f} us '
                  f'({res["marlin_ar2"][0] / res["mega"][0]:.2f}x), marlin_rsag {res["marlin_rsag"][0]:8.1f} us '
                  f'({res["marlin_rsag"][0] / res["mega"][0]:.2f}x) | diff mega vs rsag {dm:.1e} | '
                  f'repeats mega {[round(v, 1) for v in res["mega"][1]]}', flush=True)
        buf.destroy()
        dist.barrier()
    del m, t_l1, t_l2, l1, l2
    torch.cuda.empty_cache()
dist.destroy_process_group()
