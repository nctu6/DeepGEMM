"""SM120 block-scaled dense MQA logits check (MXFP8 + MXFP4), dequantized-reference oracle.

Validates the new arch-12 MX path of deep_gemm.fp8_fp4_mqa_logits against relu(Q_deq . KV_deq) * weight summed over
heads, gathered into the compressed [S, max_len] output (row i holds keys ks[i]..ke[i]).

  sm120_remote.sh dg scripts/sm120_mqa_blockscale_check.py
"""
import sys
import torch

import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_token_cast_to_fp4, per_token_cast_to_fp8
from deep_gemm.utils.math import cast_back_from_fp4, cast_back_from_fp8

sys.path.insert(0, 'tests')
from test_attention import to_mqa_weights  # noqa: E402


def check(fmt, seq_len, seq_len_kv, num_heads, head_dim=128, seed=0):
    is_fp4 = fmt == 'mxfp4'
    cast = per_token_cast_to_fp4 if is_fp4 else per_token_cast_to_fp8
    back = cast_back_from_fp4 if is_fp4 else cast_back_from_fp8
    elem = head_dim // 2 if is_fp4 else head_dim
    torch.manual_seed(seed)

    q = torch.randn(seq_len, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
    kv = torch.randn(seq_len_kv, head_dim, device='cuda', dtype=torch.bfloat16)
    weights = to_mqa_weights(torch.randn(seq_len, num_heads, device='cuda', dtype=torch.bfloat16))

    q_fp, q_sf = cast(q.view(-1, head_dim), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    kv_fp, kv_sf = cast(kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
    q_in = (q_fp.view(seq_len, num_heads, elem), q_sf.view(seq_len, num_heads))
    kv_in = (kv_fp, kv_sf.view(seq_len_kv))

    ks = torch.randint(0, seq_len_kv // 2, (seq_len,), device='cuda', dtype=torch.int)
    ke = torch.minimum(ks + torch.randint(1, seq_len_kv, (seq_len,), device='cuda', dtype=torch.int),
                       torch.full_like(ks, seq_len_kv))
    max_len = int((ke - ks).max())
    logits = deep_gemm.fp8_fp4_mqa_logits(q_in, kv_in, weights, ks, ke, max_len).float()

    q_deq = back(q_fp, q_sf, gran_k=32, use_packed_ue8m0=True).view(seq_len, num_heads, head_dim).float()
    kv_deq = back(kv_fp, kv_sf, gran_k=32, use_packed_ue8m0=True).float()
    score = torch.einsum('shd,td->sht', q_deq, kv_deq).relu_()
    full = (score * weights[:, :, None].float()).sum(1)

    idx = torch.arange(max_len, device='cuda')[None, :] + ks[:, None]
    mask = idx < ke[:, None]
    ref = full.gather(1, idx.clamp_max(seq_len_kv - 1))
    diff = calc_diff(logits[mask], ref[mask])
    ok = diff < 1e-3
    print(f'[{fmt:6}] S={seq_len:5} SK={seq_len_kv:6} H={num_heads:2} D={head_dim}: diff={diff:.2e}  {"ok" if ok else "FAIL"}', flush=True)
    return ok


def main():
    deep_gemm.set_pdl(False)
    failed = False
    for fmt in ('mxfp8', 'mxfp4'):
        for num_heads in (32, 64):
            for (s, sk) in [(256, 2048), (1024, 4096), (777, 1031)]:
                failed |= not check(fmt, s, sk, num_heads)
    print('FAILED' if failed else 'ALL OK')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
