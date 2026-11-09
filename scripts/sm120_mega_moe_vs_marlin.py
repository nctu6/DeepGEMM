# SM120 `fp8_fp4_mega_moe` (one rank) against the TokenSpeed MXFP4 MoE leaves on SM120 (Marlin W4A16, Triton MXFP4) at the
# DeepSeek V4 / V4.1 expert shapes. Needs TokenSpeed on PYTHONPATH (`sm120_remote.sh ts`). CUDA graph, every call after
# a 512 MB read (cold weights) whose own time is subtracted, 3 repeats. Env: SHAPES, TOKENS, PDL, NO_TRITON
import os, sys, random, types, statistics
import torch, torch.distributed as dist
import deep_gemm
from deep_gemm.testing import calc_diff
from tokenspeed_kernel.ops.moe.marlin import marlin_mxfp4_moe_weights, marlin_mxfp4_precomputed_moe_apply
from tokenspeed_kernel.ops.moe.triton import triton_mxfp4_moe_weights, triton_mxfp4_precomputed_moe_apply
from tokenspeed_kernel.ops.moe._deep_gemm.mega_moe_stage import stage_mxfp4_mega_moe_inputs

os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', str(random.randint(20000, 40000)))
dist.init_process_group('gloo', rank=0, world_size=1)
deep_gemm.set_pdl(bool(int(os.environ.get('PDL', '0'))))
print('pdl', deep_gemm.get_pdl())
SHAPES = {k: v for k, v in {'V4': (4096, 2048, 256, 6), 'V4.1': (5120, 2304, 384, 6)}.items() if k in os.environ.get('SHAPES', 'V4 V4.1').split()}
TOKENS = [int(t) for t in os.environ.get('TOKENS', '1 4 16 64 256 1024 4096').split()]
REPEATS = 3

FLUSH = torch.ones(128 << 20, dtype=torch.float, device='cuda')   # 512 MB, 4x the L2
FLUSH_OUT = torch.empty((), dtype=torch.float, device='cuda')
def flush():
    torch.sum(FLUSH, dim=0, out=FLUSH_OUT)

def graph_time_raw(fn, iters):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            flush(); fn()
    g.replay(); torch.cuda.synchronize()
    ts = []
    for _ in range(REPEATS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / iters)
    return ts

FLUSH_US = None
def graph_time(fn, iters=10):
    # Cold weights: every call follows a 512 MB read, whose own time is subtracted
    global FLUSH_US
    if FLUSH_US is None:
        FLUSH_US = statistics.median(graph_time_raw(lambda: None, 20))
    ts = [t - FLUSH_US for t in graph_time_raw(fn, iters)]
    return statistics.median(ts), ts

class W(torch.nn.Module):
    pass

for name, (H, I, E, K) in SHAPES.items():
    gen = torch.Generator(device='cuda').manual_seed(0)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device='cuda', generator=gen)
    w2 = torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8, device='cuda', generator=gen)
    s13 = torch.randint(118, 121, (E, 2 * I, H // 32), dtype=torch.uint8, device='cuda', generator=gen)
    s2 = torch.randint(118, 121, (E, H, I // 32), dtype=torch.uint8, device='cuda', generator=gen)
    # Mega: same weights, gate/up interleaved, packed UE8M0 scales
    to_f = lambda s: (s.int() << 23).view(torch.float)
    l1 = (w13.view(torch.int8), deep_gemm.transform_sf_into_required_layout(to_f(s13), 2 * I, H, (1, 32), E))
    l2 = (w2.view(torch.int8), deep_gemm.transform_sf_into_required_layout(to_f(s2), H, I, (1, 32), E))
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(l1, l2)
    def make_module(clone):
        m = W()
        m.w13_weight = torch.nn.Parameter(w13.clone() if clone else w13, requires_grad=False)
        m.w13_weight_scale = torch.nn.Parameter(s13.clone() if clone else s13, requires_grad=False)
        m.w2_weight = torch.nn.Parameter(w2.clone() if clone else w2, requires_grad=False)
        m.w2_weight_scale = torch.nn.Parameter(s2.clone() if clone else s2, requires_grad=False)
        m.num_local_experts = E; m.ep_size = 1; m.ep_rank = 0
        m.activation = 'swiglu'; m.swiglu_arg = types.SimpleNamespace(limit=10.0, alpha=None); m.swiglu_beta = None
        m.top_k = K; m.hidden_size = H; m.intermediate_size = I
        return m
    plan = {'activation': 'swiglu', 'internal_activation_dtype': 'input'}
    marlin_w = make_module(True); marlin_mxfp4_moe_weights(plan, marlin_w)
    triton_w = make_module(False); triton_mxfp4_moe_weights(plan, triton_w)
    for T in TOKENS:
        x = (torch.randn(T, H, device='cuda') * 0.5).bfloat16()
        topk_weights, topk_ids = torch.topk(torch.randn(T, E, device='cuda'), K, dim=-1)
        topk_weights = topk_weights.softmax(-1)
        buf = deep_gemm.get_symm_buffer_for_mega_moe(dist.group.WORLD, E, T, K, H, I)
        ids64 = topk_ids.to(torch.int64)
        y = torch.empty(T, H, dtype=torch.bfloat16, device='cuda')
        def mega():
            stage_mxfp4_mega_moe_inputs(x, topk_weights, ids64, buf.x[:T], buf.x_sf[:T], buf.topk_idx[:T], buf.topk_weights[:T])
            deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buf, activation_clamp=10.0, fast_math=True)
        marlin = lambda: marlin_mxfp4_precomputed_moe_apply(plan, x, marlin_w, None, topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32))
        triton_fn = lambda: triton_mxfp4_precomputed_moe_apply(plan, x, triton_w, None, topk_weights=topk_weights, topk_ids=topk_ids.to(torch.int32))
        mega(); ref_m = marlin(); 
        res = {'mega': graph_time(mega), 'marlin': graph_time(marlin)}
        try:
            if os.environ.get('NO_TRITON'):
                raise RuntimeError('skipped')
            ref_t = triton_fn(); res['triton'] = graph_time(triton_fn)
            dt = calc_diff(y.float(), ref_t.float())
        except Exception as e:
            ref_t = None; dt = float('nan'); print('triton failed', str(e)[:200])
        dm = calc_diff(y.float(), ref_m.float())
        best = min(v[0] for k, v in res.items() if k != 'mega')
        print(f'{name} T={T:5}: mega {res["mega"][0]:8.1f} us, marlin {res["marlin"][0]:8.1f} us'
              + (f', triton {res["triton"][0]:8.1f} us' if 'triton' in res else '')
              + f' | best/mega {best / res["mega"][0]:.2f}x | diff vs marlin {dm:.1e}, vs triton {dt:.1e}'
              + f' | repeats mega {[round(v, 1) for v in res["mega"][1]]}', flush=True)
        buf.destroy()
