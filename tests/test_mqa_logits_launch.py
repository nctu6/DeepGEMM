"""Contiguous MQA logits launch path: the kernel found by its integers, the logits view, and interleaved configurations.

`fp8_fp4_mqa_logits` finds its kernel in a per-call-site cache keyed by the template integers (heads, head dim, block
sizes, SM count, dtypes) instead of hashing the generated source on every call. This runs the configurations that differ
in those integers back to back, several times each, so every one is looked up after another's miss and hit, and checks
each result against a dequantized reference, and the layout of the returned view.
"""
import torch
import deep_gemm
from deep_gemm.testing import calc_diff
from deep_gemm.utils import per_custom_dims_cast_to_fp8


def reference(q_fp8, kv_fp8, kv_sf, weights, starts, ends, max_len):
    kv = kv_fp8.float() * kv_sf[:, None]
    score = torch.einsum('shd,td->sht', q_fp8.float(), kv).relu_()
    full = (score * weights.float()[:, :, None]).sum(1)
    index = torch.arange(max_len, device='cuda')[None, :] + starts[:, None]
    valid = index < ends[:, None]
    return full.gather(1, index.clamp_max(kv.shape[0] - 1)), valid


def run(seq_len, seq_len_kv, num_heads, weights_dtype=torch.float32):
    q_fp8 = torch.randn(seq_len, num_heads, 128, device='cuda').to(torch.float8_e4m3fn)
    kv_fp8, kv_sf = per_custom_dims_cast_to_fp8(torch.randn(seq_len_kv, 128, device='cuda', dtype=torch.bfloat16), (0, ), False)
    weights = torch.randn(seq_len, num_heads, device='cuda').to(weights_dtype)
    starts = torch.randint(0, seq_len_kv // 2, (seq_len, ), device='cuda', dtype=torch.int)
    ends = torch.minimum(starts + torch.randint(1, seq_len_kv, (seq_len, ), device='cuda', dtype=torch.int),
                         torch.full_like(starts, seq_len_kv))
    max_len = int((ends - starts).max())
    logits = deep_gemm.fp8_fp4_mqa_logits((q_fp8, None), (kv_fp8, kv_sf), weights, starts, ends, max_len)

    # A view of the first `seq_len` rows and `max_len` columns of the allocation, rows 1024 bytes apart
    assert logits.shape == (seq_len, max_len) and logits.dtype == torch.float32
    assert logits.storage_offset() == 0 and logits.stride(1) == 1
    assert logits.stride(0) * 4 % 1024 == 0 and logits.stride(0) >= max_len

    ref, valid = reference(q_fp8, kv_fp8, kv_sf, weights, starts, ends, max_len)
    diff = calc_diff(logits[valid], ref[valid])
    assert diff < 1e-3, (seq_len, seq_len_kv, num_heads, diff)


def test_interleaved_configurations():
    if torch.cuda.get_device_capability()[0] != 12:
        print('Skipped: SM120 only')
        return
    torch.manual_seed(0)
    for _ in range(3):
        for num_heads in (32, 64, 32):
            for seq_len, seq_len_kv in ((1, 130), (128, 128), (200, 3000), (7, 4097)):
                run(seq_len, seq_len_kv, num_heads)
    print('Interleaved contiguous MQA configurations OK')


if __name__ == '__main__':
    test_interleaved_configurations()
