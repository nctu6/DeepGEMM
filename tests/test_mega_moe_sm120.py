import argparse
import os
import random
import torch
import torch.distributed as dist
import deep_gemm
from deep_gemm.testing import bench_kineto, calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
# SM120 `fp8_fp4_mega_moe` on one rank: the fused SM100 kernel needs TMEM, 2-CTA UMMA and NVLink symmetric memory, so
# SM120 runs dispatch (with the gather), the two m-grouped FP8 x FP4 GEMMs (SwiGLU in the first one's epilogue) and the
# combine as separate kernels. The reference below
# replays the same arithmetic in PyTorch (dequantized FP32 GEMMs, BF16 rounding of the L1 output, BF16 clamp, SwiGLU,
# top-k weight, per-32 UE8M0 FP8 cast, FP32 combine of BF16 rows), so only the GEMM accumulation order differs


def _dequant_fp8(x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    return x.float() * sf.repeat_interleave(32, dim=-1)[..., :x.shape[-1]]


def _dequant_fp4(x: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor((0, 0.5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float, device=x.device)
    packed = x.view(torch.uint8).int()
    codes = torch.stack((packed & 0xf, packed >> 4), dim=-1).flatten(-2)
    values = lut[codes & 7] * torch.where(codes & 8 != 0, -1.0, 1.0)
    return values * sf.repeat_interleave(32, dim=-1)


def reference(x_fp8, x_sf, topk_idx, topk_weights, l1, l2, clamp):
    num_tokens, num_topk = topk_idx.shape
    num_experts, n1, _ = l1[0].shape
    intermediate = n1 // 2
    hidden = l2[0].shape[1]
    x = _dequant_fp8(x_fp8, x_sf)
    slot_rows = torch.zeros((num_tokens, num_topk, hidden), dtype=torch.bfloat16, device='cuda')
    for e in range(num_experts):
        tokens, slots = (topk_idx == e).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        l1_out = (x[tokens] @ _dequant_fp4(*[w[e] for w in l1]).T).bfloat16()
        gate, up = l1_out[:, :intermediate], l1_out[:, intermediate:]
        if clamp is not None:
            gate = gate.clamp(max=clamp)
            up = up.clamp(min=-clamp, max=clamp)
        gate, up = gate.float(), up.float()
        act = gate / (1 + torch.exp(-gate)) * up * topk_weights[tokens, slots].unsqueeze(1)
        act_fp8, act_sf = per_token_cast_to_fp8(act, use_ue8m0=True, gran_k=32)
        slot_rows[tokens, slots] = (_dequant_fp8(act_fp8, act_sf) @ _dequant_fp4(*[w[e] for w in l2]).T).bfloat16()
    # Combine: FP32 sum of the BF16 slot rows in top-k order (masked slots stay zero, which adds nothing)
    y = torch.zeros((num_tokens, hidden), dtype=torch.float, device='cuda')
    for k in range(num_topk):
        y += slot_rows[:, k].float()
    return y.bfloat16()


def test(args):
    torch.manual_seed(0)
    random.seed(0)
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', str(random.randint(20000, 40000)))
        dist.init_process_group('gloo', rank=0, world_size=1)
    group = dist.group.WORLD
    hidden, intermediate, num_experts, num_topk = args.hidden, args.intermediate_hidden, args.num_experts, args.num_topk
    for num_tokens in args.num_tokens:
        buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, max(num_tokens, 1), num_topk,
                                                        hidden, intermediate, mma_type='fp8xfp4')
        x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        def cast_weights(w):
            g, n, k = w.shape
            data = torch.empty((g, n, k // 2), dtype=torch.int8, device='cuda')
            sf = torch.empty((g, n, k // 32), dtype=torch.float, device='cuda')
            for i in range(g):
                data[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
            return data, sf
        l1 = cast_weights(torch.randn((num_experts, intermediate * 2, hidden), dtype=torch.bfloat16, device='cuda'))
        l2 = cast_weights(torch.randn((num_experts, hidden, intermediate), dtype=torch.bfloat16, device='cuda'))
        to_layout = lambda w: (w[0], deep_gemm.transform_sf_into_required_layout(w[1], w[0].shape[1], w[0].shape[2] * 2, (1, 32), num_experts))
        t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(to_layout(l1), to_layout(l2))
        scores = torch.randn((num_tokens, num_experts), dtype=torch.float, device='cuda')
        topk_weights, topk_idx = torch.topk(scores, num_topk, dim=-1, sorted=False)
        if args.masked_ratio > 0:
            topk_idx.masked_fill_(torch.rand_like(topk_weights) < args.masked_ratio, -1)
            topk_weights.masked_fill_(topk_idx < 0, 0)
        x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        _, x_sf_float = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
        stats = torch.randint(0, 100, (num_experts, ), dtype=torch.int, device='cuda')
        stats_ref = stats + torch.bincount(topk_idx[topk_idx >= 0].flatten(), minlength=num_experts).int()

        def run():
            buffer.x[:num_tokens].copy_(x_fp8)
            buffer.x_sf[:num_tokens].copy_(x_sf)
            buffer.topk_idx[:num_tokens].copy_(topk_idx)
            buffer.topk_weights[:num_tokens].copy_(topk_weights)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buffer, cumulative_local_expert_recv_stats=stats,
                                       activation_clamp=args.activation_clamp, fast_math=False)
            return y
        y = run()
        assert torch.equal(stats, stats_ref), 'expert stats mismatch'
        # Bitwise deterministic
        for _ in range(3):
            assert torch.equal(run(), y)
        # The intermediate rows are a persistent per-device workspace: once a call has made it, a repeated call takes
        # nothing from the framework's allocator (it used to be about 2.6 GB per call at V4.1 sizes, outside the KV planner)
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved()
        run()
        if num_tokens >= 64:
            assert torch.cuda.memory_reserved() - reserved < (1 << 20), 'mega MoE scratch is allocated on every call'
        diff = 0.0
        if num_tokens <= args.max_reference_tokens:
            y_ref = reference(x_fp8, x_sf_float, topk_idx, topk_weights, l1, l2, args.activation_clamp)
            diff = calc_diff(y, y_ref)
            assert diff < 1e-4, f'{num_tokens=}: {diff=}'
        names = ('mega_moe_dispatch', 'gemm_1d1d', 'mega_moe_combine')
        t = bench_kineto(run, names, suppress_kineto_output=True, with_multiple_kernels=True)
        print(f' > tokens={num_tokens:5}, H={hidden}, I={intermediate}, experts={num_topk}/{num_experts}: '
              f'diff={diff:.1e}, {sum(t) * 1e6:7.1f} us ({", ".join(f"{n.split("_")[-1]} {x * 1e6:.1f}" for n, x in zip(names, t))})', flush=True)
        buffer.destroy()


def test_scratch_is_sized_once(args):
    # With one rank the rows follow the call's tokens. A serving warmup counts up, and each size used to grow the
    # persistent scratch and keep every predecessor alive. The first call now takes the capacity of the worst routing of
    # `num_max_tokens`, and no later call (any size, any tile height) takes more
    hidden, intermediate, num_experts, num_topk = args.hidden, args.intermediate_hidden, args.num_experts, args.num_topk
    max_tokens = 2048
    group = dist.group.WORLD
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, max_tokens, num_topk, hidden, intermediate,
                                                    mma_type='fp8xfp4')
    weights = []
    for n, k in ((intermediate * 2, hidden), (hidden, intermediate)):
        w = torch.randn((num_experts, n, k), dtype=torch.bfloat16, device='cuda')
        data = torch.empty((num_experts, n, k // 2), dtype=torch.int8, device='cuda')
        sf = torch.empty((num_experts, n, k // 32), dtype=torch.float, device='cuda')
        for i in range(num_experts):
            data[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        weights.append((data, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), num_experts)))
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(*weights)

    def run(num_tokens):
        x_fp8, x_sf = per_token_cast_to_fp8(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda'),
                                            use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        topk_idx = torch.stack([torch.randperm(num_experts, device='cuda')[:num_topk] for _ in range(num_tokens)])
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].fill_(1.0 / num_topk)
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buffer, activation_clamp=args.activation_clamp, fast_math=False)
        torch.cuda.synchronize()

    run(1)
    torch.cuda.empty_cache()
    allocated = torch.cuda.memory_allocated()
    for num_tokens in (7, 64, 300, 1000, max_tokens):
        run(num_tokens)
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() - allocated < (1 << 20), f'{num_tokens=} grew the mega MoE scratch'
    buffer.destroy()
    print(' > the mega MoE scratch is sized by the first call', flush=True)


def test_tile_height_and_graph_replay(args):
    # The dispatch lays every expert out in runs of `BLOCK_M` rows and the fused SwiGLU epilogue of L1 works on 64-row
    # slices of a tile, so the tile height changes which CTA, warp and slice handles a row, but no value: every tile
    # height gives the same bits. A CUDA graph captured once and replayed with new inputs (routing included) gives the
    # bits of an eager call
    hidden, intermediate, num_experts, num_topk = args.hidden, args.intermediate_hidden, args.num_experts, args.num_topk
    max_tokens = 512
    group = dist.group.WORLD
    buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, max_tokens, num_topk, hidden, intermediate,
                                                    mma_type='fp8xfp4')
    weights = []
    for n, k in ((intermediate * 2, hidden), (hidden, intermediate)):
        w = torch.randn((num_experts, n, k), dtype=torch.bfloat16, device='cuda')
        data = torch.empty((num_experts, n, k // 2), dtype=torch.int8, device='cuda')
        sf = torch.empty((num_experts, n, k // 32), dtype=torch.float, device='cuda')
        for i in range(num_experts):
            data[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
        weights.append((data, deep_gemm.transform_sf_into_required_layout(sf, n, k, (1, 32), num_experts)))
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(*weights)

    def make_inputs(num_tokens):
        x_fp8, x_sf = per_token_cast_to_fp8(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda'),
                                            use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
        topk_weights, topk_idx = torch.topk(torch.randn((num_tokens, num_experts), device='cuda'), num_topk, dim=-1)
        topk_idx.masked_fill_(torch.rand_like(topk_weights) < 0.2, -1)
        # A token with every slot masked has a zero row
        topk_idx[0] = -1
        return x_fp8, x_sf, topk_idx, topk_weights.softmax(-1)

    def stage(inputs):
        x_fp8, x_sf, topk_idx, topk_weights = inputs
        num_tokens = x_fp8.shape[0]
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_sf)
        buffer.topk_idx[:num_tokens].copy_(topk_idx)
        buffer.topk_weights[:num_tokens].copy_(topk_weights)

    def call(y):
        deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, buffer, activation_clamp=args.activation_clamp, fast_math=True)

    for num_tokens in (1, 5, 300):
        inputs = make_inputs(num_tokens)
        stage(inputs)
        results = []
        for block_m in (16, 32, 64, 128):
            os.environ['DG_SM120_MEGA_BLOCK_M'] = str(block_m)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            call(y)
            results.append(y)
        del os.environ['DG_SM120_MEGA_BLOCK_M']
        assert not results[0][0].any(), 'a token without routed slots must be zero'
        for block_m, y in zip((32, 64, 128), results[1:]):
            assert torch.equal(y, results[0]), f'{num_tokens=}: BLOCK_M {block_m} differs from BLOCK_M 16'

        # Graph: capture with one routing, replay with another
        y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
        stage(make_inputs(num_tokens))
        call(y)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call(y)
        replay_inputs = make_inputs(num_tokens)
        stage(replay_inputs)
        graph.replay()
        y_eager = torch.empty_like(y)
        call(y_eager)
        torch.cuda.synchronize()
        assert torch.equal(y, y_eager), f'{num_tokens=}: graph replay differs from an eager call'
    buffer.destroy()
    print(' > tile heights 16-128 give the same bits; graph replay equals eager', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM120 mega MoE (one rank)')
    parser.add_argument('--num-tokens', type=int, nargs='+', default=[1, 7, 64, 300])
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=32)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.1)
    parser.add_argument('--max-reference-tokens', type=int, default=512)
    args = parser.parse_args()
    test(args)
    test_scratch_is_sized_once(args)
    test_tile_height_and_graph_replay(args)
