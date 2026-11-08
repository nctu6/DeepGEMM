"""SM120 block-scaled paged MQA logits check (MXFP4 and MXFP8), dequantized-reference oracle.

Validates the arch-12 MX path of `deep_gemm.fp8_fp4_paged_mqa_logits` (the 68-byte V4.1 index rows: packed FP4 values and
one packed UE8M0 word per token, page-planar) against relu(Q_deq . K_deq) * weight summed over heads, for BF16 and FP32
logits and weights, odd page strides, and the pipeline-wrapping shapes of `sm120_mqa_check.py`.

  sm120_remote.sh dg scripts/sm120_paged_mqa_blockscale_check.py
"""
import sys
import torch
import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import ceil_div, per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8


def make_cache(fmt, num_blocks, block_kv, head_dim, page_pad):
    """Random KV pages: returns the fused cache view and the dequantized keys [num_blocks, block_kv, head_dim]."""
    is_fp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8
    back = cast_back_from_fp4 if is_fp4 else cast_back_from_fp8
    elem = head_dim // 2 if is_fp4 else head_dim
    kv = torch.randn(num_blocks * block_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    kv_fp, kv_sf = cast(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    row = block_kv * (elem + 4)
    pages = torch.zeros(num_blocks, row + page_pad, device='cuda', dtype=torch.uint8)
    pages[:, :block_kv * elem] = kv_fp.view(num_blocks, block_kv * elem).view(torch.uint8)
    pages[:, block_kv * elem:row] = kv_sf.view(num_blocks, block_kv).view(torch.uint8)
    fused = pages.as_strided((num_blocks, block_kv, 1, elem + 4), (row + page_pad, elem + 4, elem + 4, 1))
    deq = back(kv_fp, kv_sf, gran_k=32, use_packed_ue8m0=True).float().view(num_blocks, block_kv, head_dim)
    return fused, deq


def check(fmt, batch_size, next_n, num_heads, avg_ctx, logits_dtype=None, weights_dtype=torch.bfloat16, page_pad=0,
          head_dim=128, block_kv=64, seed=0):
    is_fp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8
    back = cast_back_from_fp4 if is_fp4 else cast_back_from_fp8
    elem = head_dim // 2 if is_fp4 else head_dim
    torch.manual_seed(seed)
    q = torch.randn(batch_size, next_n, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    q_fp, q_sf = cast(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q_deq = back(q_fp, q_sf, gran_k=32, use_packed_ue8m0=True).float().view(batch_size, next_n, num_heads, head_dim)
    q_in = (q_fp.view(batch_size, next_n, num_heads, elem), q_sf.view(batch_size, next_n, num_heads))
    weights = torch.randn(batch_size * next_n, num_heads, device='cuda', dtype=weights_dtype)

    ctx = torch.randint(max(1, avg_ctx // 2), avg_ctx * 3 // 2, (batch_size,), device='cuda', dtype=torch.int)
    context_lens = (ctx[:, None] - next_n + 1 + torch.arange(next_n, device='cuda', dtype=torch.int)[None, :]).clamp_min(1).int().contiguous()
    max_ctx = int(context_lens.max())
    max_blocks = ceil_div(max_ctx, block_kv)
    num_blocks = batch_size * max_blocks
    block_table = torch.randperm(num_blocks, device='cuda', dtype=torch.int).view(batch_size, max_blocks).contiguous()
    fused, kv_deq = make_cache(fmt, num_blocks, block_kv, head_dim, page_pad)

    meta = deep_gemm.get_paged_mqa_logits_metadata(context_lens, block_kv, deep_gemm.get_num_sms())
    kwargs = dict(q=q_in, kv_cache=fused, weights=weights, context_lens=context_lens, block_table=block_table,
                  schedule_meta=meta, max_context_len=max_ctx)
    if logits_dtype is not None:
        kwargs['logits_dtype'] = logits_dtype
    logits = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
    want = logits_dtype or torch.bfloat16
    assert logits.dtype == want, f'logits dtype {logits.dtype} != {want}'
    # The kernel is deterministic: every launch gives the same bits
    again = deep_gemm.fp8_fp4_paged_mqa_logits(**kwargs)
    for b in range(batch_size):
        for j in range(next_n):
            n = int(context_lens[b, j])
            assert torch.equal(logits[b * next_n + j, :n], again[b * next_n + j, :n]), 'not deterministic'

    max_diff = 0.0
    max_rel = 0.0
    for b in range(batch_size):
        keys = kv_deq[block_table[b].long()].view(-1, head_dim)
        for j in range(next_n):
            row = b * next_n + j
            n = int(context_lens[b, j])
            score = torch.einsum('hd,td->ht', q_deq[b, j].double(), keys[:n].double()).relu_()
            ref = (score * weights[row].double()[:, None]).sum(0)
            got = logits[row, :n].double()
            max_diff = max(max_diff, calc_diff(got, ref))
            max_rel = max(max_rel, float(((got - ref).abs() / ref.abs().clamp_min(1e-3 * ref.abs().max())).max()))
    # BF16 logits round to 8 bits of mantissa; FP32 logits carry only the FP32 accumulation order
    tol_diff, tol_rel = (1e-4, 2 ** -7) if want == torch.bfloat16 else (1e-12, 1e-4)
    ok = max_diff < tol_diff and max_rel < tol_rel
    print(f'[{fmt:5}] B={batch_size:3} next_n={next_n} H={num_heads} ctx~{avg_ctx:5} logits={str(want)[6:]:8} '
          f'w={str(weights_dtype)[6:]:8} pad={page_pad:3}: diff={max_diff:.1e} rel={max_rel:.1e} {"ok" if ok else "FAIL"}',
          flush=True)
    return ok


def main():
    deep_gemm.set_pdl(False)
    failed = False
    # Pipeline-wrapping shapes of `sm120_mqa_check.py`: B >= 32, next_n 2, long and short contexts
    shapes = ((4, 1, 64, 1000), (33, 1, 32, 4096), (8, 2, 64, 3000), (64, 1, 64, 300), (3, 1, 64, 20000),
              (32, 1, 64, 4096), (8, 2, 64, 8000), (64, 2, 64, 4096), (128, 1, 64, 16384), (32, 2, 32, 4096),
              (64, 2, 32, 2000), (128, 2, 64, 3000), (32, 1, 32, 20000), (1, 1, 32, 70), (1, 1, 32, 130000),
              (128, 1, 32, 1500), (16, 1, 32, 32768))
    for fmt in ('mxfp4', 'mxfp8'):
        for b, nn, h, c in shapes:
            # MXFP8 is limited to 64 Q rows (next_n x heads) per request on SM120
            if fmt == 'mxfp8' and nn * h > 64:
                continue
            failed |= not check(fmt, b, nn, h, c)
    # FP32 logits and weights (what a selector wants), and rows with padding after each page
    for fmt in ('mxfp4', 'mxfp8'):
        for b, nn, h, c in ((4, 1, 32, 1000), (33, 1, 32, 4096), (8, 2, 64, 3000), (128, 1, 32, 2048)):
            if fmt == 'mxfp8' and nn * h > 64:
                continue
            failed |= not check(fmt, b, nn, h, c, logits_dtype=torch.float32, weights_dtype=torch.float32)
            failed |= not check(fmt, b, nn, h, c, logits_dtype=torch.float32, weights_dtype=torch.bfloat16)
            failed |= not check(fmt, b, nn, h, c, page_pad=80, logits_dtype=torch.float32, weights_dtype=torch.float32)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
