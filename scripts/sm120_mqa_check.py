# Small-memory correctness check for SM120 MQA logits (contiguous and paged), against a dequantized reference
import argparse
import torch
import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import ceil_div, per_custom_dims_cast_to_fp8


def ref_logits(q_fp8, kv_fp8, kv_sf, weights):
    # q: [S, H, D] fp8, kv: [SK, D] fp8, kv_sf: [SK], weights: [S, H] -> [S, SK]
    kv = kv_fp8.float() * kv_sf[:, None]
    score = torch.einsum('shd,td->sht', q_fp8.float(), kv).relu_()
    return (score * weights[:, :, None]).sum(1)


def check_contiguous(seq_len, seq_len_kv, num_heads, head_dim=128):
    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = torch.randn(seq_len, num_heads, device='cuda', dtype=torch.float)
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv, (0, ), False)
    ks = torch.randint(0, seq_len_kv // 2, (seq_len, ), device='cuda', dtype=torch.int)
    ke = torch.minimum(ks + torch.randint(1, seq_len_kv, (seq_len, ), device='cuda', dtype=torch.int),
                       torch.full_like(ks, seq_len_kv))
    max_len = int((ke - ks).max())
    logits = deep_gemm.fp8_fp4_mqa_logits((q_fp8, None), (kv_fp8, kv_sf), weights, ks, ke, max_len)
    full = ref_logits(q_fp8, kv_fp8, kv_sf, weights)
    # Output is compressed: row i holds keys ks[i]..ke[i]
    idx = torch.arange(max_len, device='cuda')[None, :] + ks[:, None]
    mask = idx < ke[:, None]
    ref = full.gather(1, idx.clamp_max(seq_len_kv - 1))
    diff = calc_diff(logits[mask], ref[mask])
    print(f'contiguous S={seq_len} SK={seq_len_kv} H={num_heads} D={head_dim}: diff={diff:.2e}')
    assert diff < 1e-3


def check_paged(batch_size, next_n, num_heads, avg_ctx, block_kv=64):
    q = torch.randn(batch_size, next_n, num_heads, 128, device='cuda', dtype=torch.bfloat16)
    q_fp8 = q.to(torch.float8_e4m3fn)
    weights = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=torch.float)
    ctx = torch.randint(max(1, avg_ctx // 2), avg_ctx * 3 // 2, (batch_size, ), device='cuda', dtype=torch.int)
    # Token j of a request sees `ctx - next_n + 1 + j` keys
    context_lens = (ctx[:, None] - next_n + 1 + torch.arange(next_n, device='cuda', dtype=torch.int)[None, :]).clamp_min(1).int().contiguous()
    max_ctx = int(context_lens.max())
    max_blocks = ceil_div(max_ctx, block_kv)
    num_blocks = batch_size * max_blocks
    block_table = torch.randperm(num_blocks, device='cuda', dtype=torch.int).view(batch_size, max_blocks).contiguous()

    kv = torch.randn(num_blocks, block_kv, 128, device='cuda', dtype=torch.bfloat16)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(kv.view(-1, 128), (0, ), False)
    fused = torch.empty(num_blocks, block_kv * (128 + 4), device='cuda', dtype=torch.uint8)
    fused[:, :block_kv * 128] = kv_fp8.view(num_blocks, block_kv * 128).view(torch.uint8)
    fused[:, block_kv * 128:] = kv_sf.view(num_blocks, block_kv).view(torch.uint8)
    fused = fused.view(num_blocks, block_kv, 1, 128 + 4)

    meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())
    logits = deep_gemm.fp8_fp4_paged_mqa_logits((q_fp8, None), fused, weights, context_lens, block_table, meta, max_ctx)
    kv_deq = (kv_fp8.float() * kv_sf[:, None]).view(num_blocks, block_kv, 128)
    max_diff = 0.0
    for b in range(batch_size):
        keys = kv_deq[block_table[b].long()].view(-1, 128)
        for j in range(next_n):
            row = b * next_n + j
            n = int(context_lens[b, j])
            s = torch.einsum('hd,td->ht', q_fp8[b, j].float(), keys[:n]).relu_()
            ref = (s * weights[row][:, None]).sum(0)
            max_diff = max(max_diff, calc_diff(logits[row, :n], ref))
    print(f'paged B={batch_size} next_n={next_n} H={num_heads} ctx~{avg_ctx}: diff={max_diff:.2e}')
    assert max_diff < 1e-3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.parse_args()
    torch.manual_seed(0)
    for s, sk, h in ((128, 4096, 64), (300, 2048, 32), (1024, 8192, 64), (7, 300, 32)):
        check_contiguous(s, sk, h)
    # Head dim 64 (prefill)
    for s, sk, h in ((128, 4096, 64), (300, 2048, 32), (1024, 8192, 64)):
        check_contiguous(s, sk, h, head_dim=64)
    # NOTES: B=32 / next_n=2 cases wrap the 2-stage Q/KV pipelines (they caught a release bug and a 1-Q-stage deadlock)
    for b, nn, h, c in ((4, 1, 64, 1000), (33, 1, 32, 4096), (8, 2, 64, 3000), (64, 1, 64, 300), (3, 1, 64, 20000),
                        (32, 1, 64, 4096), (8, 2, 64, 8000), (64, 2, 64, 4096), (128, 1, 64, 16384)):
        check_paged(b, nn, h, c)
    print('OK')
