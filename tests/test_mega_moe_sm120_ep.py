import argparse
import os
import sys
import torch
import torch.distributed as dist
import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from test_mega_moe_sm120 import reference
# SM120 `fp8_fp4_mega_moe` across ranks (CUDA IPC / PCIe P2P, no NVLink). Run with two processes on two GPUs:
#   CUDA_VISIBLE_DEVICES=6,7 timeout 900 torchrun --nproc_per_node 2 tests/test_mega_moe_sm120_ep.py
# Each rank owns half of the experts and has its own tokens. Per rank the result must be bitwise equal to the one-rank
# `fp8_fp4_mega_moe` over all experts for the same tokens, and match the PyTorch replay of the SM100 arithmetic. It also
# checks the expert stats (counts of every rank's pairs), masked slots, determinism, ranks with no tokens, a different
# token count per rank, and replay in a CUDA graph. A rank that never arrives traps in the barrier instead of hanging


def cast_weights(w: torch.Tensor):
    g, n, k = w.shape
    data = torch.empty((g, n, k // 2), dtype=torch.int8, device='cuda')
    sf = torch.empty((g, n, k // 32), dtype=torch.float, device='cuda')
    for i in range(g):
        data[i], sf[i] = per_token_cast_to_fp4(w[i], use_ue8m0=True, gran_k=32)
    return data, sf


def to_layout(w, num_groups):
    return w[0], deep_gemm.transform_sf_into_required_layout(w[1], w[0].shape[1], w[0].shape[2] * 2, (1, 32), num_groups)


def make_inputs(num_tokens, seed, args):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn((num_tokens, args.hidden), dtype=torch.bfloat16, device='cuda', generator=generator)
    scores = torch.randn((num_tokens, args.num_experts), dtype=torch.float, device='cuda', generator=generator)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, sorted=False)
    if args.masked_ratio > 0:
        masked = torch.rand(topk_weights.shape, device='cuda', generator=generator) < args.masked_ratio
        topk_idx.masked_fill_(masked, -1)
        topk_weights.masked_fill_(masked, 0)
    x_fp8, x_sf = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    _, x_sf_float = per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32)
    return x_fp8, x_sf, x_sf_float, topk_idx, topk_weights


def stage(buffer, inputs):
    x_fp8, x_sf, _, topk_idx, topk_weights = inputs
    n = x_fp8.shape[0]
    buffer.x[:n].copy_(x_fp8)
    buffer.x_sf[:n].copy_(x_sf)
    buffer.topk_idx[:n].copy_(topk_idx)
    buffer.topk_weights[:n].copy_(topk_weights)


def test(args):
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
    group = dist.group.WORLD
    # One single-rank group per rank, to run the one-rank kernel next to the EP one
    solo_groups = [dist.new_group([r]) for r in range(world)]
    solo = solo_groups[rank]
    num_experts, hidden, intermediate = args.num_experts, args.hidden, args.intermediate_hidden
    num_local = num_experts // world
    lo, hi = rank * num_local, (rank + 1) * num_local

    deep_gemm.set_pdl(bool(args.pdl))
    torch.manual_seed(0)
    l1 = cast_weights(torch.randn((num_experts, intermediate * 2, hidden), dtype=torch.bfloat16, device='cuda'))
    l2 = cast_weights(torch.randn((num_experts, hidden, intermediate), dtype=torch.bfloat16, device='cuda'))
    t_l1_all, t_l2_all = deep_gemm.transform_weights_for_mega_moe(to_layout(l1, num_experts), to_layout(l2, num_experts))
    local = lambda w: (w[0][lo:hi].contiguous(), w[1][lo:hi].contiguous())
    t_l1, t_l2 = deep_gemm.transform_weights_for_mega_moe(to_layout(local(l1), num_local), to_layout(local(l2), num_local))

    max_tokens = max(max(counts) for counts in args.num_tokens)
    ep_buffer = deep_gemm.get_symm_buffer_for_mega_moe(group, num_experts, max(max_tokens, 1), args.num_topk,
                                                       hidden, intermediate, mma_type='fp8xfp4')
    one_buffer = deep_gemm.get_symm_buffer_for_mega_moe(solo, num_experts, max(max_tokens, 1), args.num_topk,
                                                        hidden, intermediate, mma_type='fp8xfp4')
    for case, counts in enumerate(args.num_tokens):
        counts = list(counts) + [counts[-1]] * (world - len(counts))
        num_tokens = counts[rank]
        # Every rank builds every rank's inputs (same seeds), so each can check the global stats
        all_inputs = [make_inputs(counts[r], 100 * case + r + 1, args) for r in range(world)]
        inputs = all_inputs[rank]
        stats_init = torch.randint(0, 100, (num_local, ), dtype=torch.int, device='cuda')
        global_idx = torch.cat([i[3].flatten() for i in all_inputs])
        stats_ref = stats_init + torch.bincount(global_idx[global_idx >= 0], minlength=num_experts).int()[lo:hi]

        def run_ep(stats=None):
            stage(ep_buffer, inputs)
            y = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device='cuda')
            deep_gemm.fp8_fp4_mega_moe(y, t_l1, t_l2, ep_buffer, cumulative_local_expert_recv_stats=stats,
                                       activation_clamp=args.activation_clamp, fast_math=False)
            return y

        stats = stats_init.clone()
        y = run_ep(stats)
        torch.cuda.synchronize()
        assert torch.equal(stats, stats_ref), f'{counts=}: expert stats mismatch on rank {rank}'
        # Bitwise deterministic
        for _ in range(3):
            assert torch.equal(run_ep(), y), f'{counts=}: not deterministic on rank {rank}'

        # The one-rank kernel over all experts for the same tokens
        stage(one_buffer, inputs)
        y_one = torch.empty_like(y)
        deep_gemm.fp8_fp4_mega_moe(y_one, t_l1_all, t_l2_all, one_buffer, activation_clamp=args.activation_clamp, fast_math=False)
        assert torch.equal(y, y_one), f'{counts=}: EP differs from the one-rank result on rank {rank}: {(y.float() - y_one.float()).abs().max()}'

        diff = 0.0
        if 0 < num_tokens <= args.max_reference_tokens:
            diff = calc_diff(y, reference(inputs[0], inputs[2], inputs[3], inputs[4], l1, l2, args.activation_clamp))
            assert diff < 1e-4, f'{counts=}: {diff=}'

        # CUDA graph: capture the staging copy and the call, then replay with new inputs
        if args.graph and max(counts) > 0:  # every rank takes part in every call
            graph_inputs = [t.clone() for t in inputs]
            static_y = torch.empty_like(y)
            def graph_body():
                stage(ep_buffer, graph_inputs)
                deep_gemm.fp8_fp4_mega_moe(static_y, t_l1, t_l2, ep_buffer, activation_clamp=args.activation_clamp, fast_math=False)
            # Calls that share a symmetric buffer must be ordered on one stream per rank: the barrier epoch lives in the
            # buffer. A rank without tokens has nothing above that synchronizes, so its last call may still be queued
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph_body()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                graph_body()
            for replay in range(3):
                new_inputs = make_inputs(num_tokens, 7000 + 10 * replay + rank, args)
                for dst, src in zip(graph_inputs, new_inputs):
                    dst.copy_(src)
                graph.replay()
                torch.cuda.synchronize()
                eager = torch.empty_like(y)
                stage(one_buffer, new_inputs)
                deep_gemm.fp8_fp4_mega_moe(eager, t_l1_all, t_l2_all, one_buffer, activation_clamp=args.activation_clamp, fast_math=False)
                assert torch.equal(static_y, eager), f'{counts=}: graph replay {replay} differs on rank {rank}'
        dist.barrier()
        if rank == 0:
            print(f' > tokens/rank={counts}, H={hidden}, I={intermediate}, experts={args.num_topk}/{num_experts} '
                  f'over {world} ranks: bitwise equal to one rank, diff vs PyTorch={diff:.1e}', flush=True)
    ep_buffer.destroy()
    one_buffer.destroy()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SM120 mega MoE across ranks')
    parser.add_argument('--num-tokens', type=lambda s: tuple(int(v) for v in s.split(',')), nargs='+',
                        default=[(1, 1), (7, 3), (64, 64), (300, 257), (0, 5), (5, 0), (1500, 900)],
                        help='tokens per rank of each case, as `rank0,rank1`')
    parser.add_argument('--hidden', type=int, default=1024)
    parser.add_argument('--intermediate-hidden', type=int, default=512)
    parser.add_argument('--num-experts', type=int, default=32)
    parser.add_argument('--num-topk', type=int, default=6)
    parser.add_argument('--activation-clamp', type=float, default=10.0)
    parser.add_argument('--masked-ratio', type=float, default=0.1)
    parser.add_argument('--max-reference-tokens', type=int, default=512)
    parser.add_argument('--graph', type=int, default=1)
    parser.add_argument('--pdl', type=int, default=0)
    test(parser.parse_args())
