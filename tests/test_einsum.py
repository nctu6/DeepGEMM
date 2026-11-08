import os
import random
import torch

import deep_gemm
from deep_gemm.testing import (
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major, test_filter
)
from deep_gemm.utils.math import (
    align, cast_back_from_fp8, ceil_div,
    per_block_cast_to_fp8, per_channel_cast_to_fp8, per_token_cast_to_fp8
)

from generators import bench_cublaslt, get_bf16_gemm_kernel_name, is_sm120_loop
from utils import (
    assert_direct_output_matches_fp32_accumulation,
    assert_stochastic_bf16_matches_fp32_accumulation,
)


def get_bf16_einsum_tolerance(use_cublaslt: bool) -> float:
    # The reference is torch's BF16 einsum (cuBLAS). SM120's own kernel accumulates with `mma.sync` in a different order,
    # so a few elements (~0.02%) round to the neighboring BF16 value: up to ~1e-8 on small outputs. It matches the
    # correctly rounded FP32 result at least as often as cuBLAS does. Truncating instead of rounding would be ~1e-6
    return 1e-7 if get_arch_major() == 12 and not use_cublaslt else 1e-10


def get_fp8_einsum_shapes():
    # `(h, r, d)`. The many-head shape is what the SM120 batched launch is for (a launch per head costs ~1.5 us each there)
    return [(8, 4096, 1024)] + ([(128, 512, 128)] if get_arch_major() == 12 else [])
def test_bmk_bnk_mn() -> None:
    print('Testing "bmk, bnk -> mn":')
    for s in (129, 4096, 8192):
        for m, n, k in [(128, 384, 128), (256, 256, 256), (384, 128, 384)]:
            for dtype in (torch.float, torch.bfloat16):
                a = torch.randn((s, m, k), dtype=torch.bfloat16, device='cuda')
                b = torch.randn((s, n, k), dtype=torch.bfloat16, device='cuda')
                d = torch.randn((m, n), dtype=dtype, device='cuda')
                c = d if dtype == torch.float else None

                # Test correctness
                ref_d = (c if dtype == torch.float else 0) + torch.bmm(a.float(), b.float().mT).sum(0)
                deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)
                assert calc_diff(d, ref_d) < 1e-5

                t = bench_kineto(lambda: deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c),
                                 get_bf16_gemm_kernel_name() if is_sm120_loop() else 'bmn_bnk_mn_gemm_impl',
                                 suppress_kineto_output=True, per_call_total=is_sm120_loop())
                print(f' > Perf (b={s:4.0f}, {m=}, {n=}, {k=}, {"FP32" if dtype == torch.float else "BF16"}): ',
                    f'{t * 1e6:4.0f} us | '
                    f'{2 * s * m * n * k / t / 1e12:4.0f} TFLOPS | '
                    f'{(count_bytes(a, b) + (d.numel() * 4)) / 1e9 / t:4.0f} GB/s')
    print()


def test_bhr_hdr_bhd():
    print('Testing "bhr, hdr -> bhd":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)

            times = []
            for use_cublaslt in (False, True):
                deep_gemm.use_deterministic_algorithms(not use_cublaslt)
                z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
                deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
                assert calc_diff(z, ref_z) < get_bf16_einsum_tolerance(use_cublaslt)
                launch = lambda output, accumulator: deep_gemm.einsum('bhr,hdr->bhd', x, y, output, c=accumulator)
                assert_direct_output_matches_fp32_accumulation(z, launch, f'{b=}, {h=}, {r=}, {d=}')
                if not use_cublaslt and get_arch_major() == 10:
                    deep_gemm.einsum('bhr,hdr->bhd', x, y, z, epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                    assert_stochastic_bf16_matches_fp32_accumulation(z, launch, f'{b=}, {h=}, {r=}, {d=}')
                launch_einsum = lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z)
                times.append(bench_cublaslt(launch_einsum, 'nvjet', suppress_kineto_output=True) if use_cublaslt else
                             bench_kineto(launch_einsum, 'gemm', suppress_kineto_output=True, per_call_total=is_sm120_loop()))
            t, t_cublaslt = times
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_bhd_hdr_bhr():
    print('Testing "bhd, hdr -> bhr":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            fy = torch.randn((h, d, r + 128), device='cuda', dtype=torch.bfloat16)
            y = fy[:, :, :r]
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)

            times = []
            for use_cublaslt in (False, True):
                deep_gemm.use_deterministic_algorithms(not use_cublaslt)
                z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)
                deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
                assert calc_diff(z, ref_z) < get_bf16_einsum_tolerance(use_cublaslt)
                launch = lambda output, accumulator: deep_gemm.einsum('bhd,hdr->bhr', x, y, output, c=accumulator)
                assert_direct_output_matches_fp32_accumulation(z, launch, f'{b=}, {h=}, {r=}, {d=}')
                if not use_cublaslt and get_arch_major() == 10:
                    deep_gemm.einsum('bhd,hdr->bhr', x, y, z, epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                    assert_stochastic_bf16_matches_fp32_accumulation(z, launch, f'{b=}, {h=}, {r=}, {d=}')
                launch_einsum = lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z)
                times.append(bench_cublaslt(launch_einsum, 'nvjet', suppress_kineto_output=True) if use_cublaslt else
                             bench_kineto(launch_einsum, 'gemm', suppress_kineto_output=True, per_call_total=is_sm120_loop()))

            t, t_cublaslt = times
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


def test_bhd_bhr_hdr():
    print('Testing "bhd, bhr -> hdr":')
    for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            z_0 = torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10
            ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x.float(), y.float())

            z = z_0.clone()
            deep_gemm.einsum('bhd,bhr->hdr', x, y, z, z)
            assert calc_diff(z, ref_z) < 1e-10

            t = bench_cublaslt(lambda: deep_gemm.einsum('bhd,bhr->hdr', x, y, z, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x, y, z, z)) / t / 1e9:4.0f} GB/s')
    print()


def test_fp8_bhr_hdr_bhd(use_ue8m0: bool = True):
    print('Testing FP8 "bhr, hdr -> bhd":')
    deep_gemm.use_deterministic_algorithms(False)
    for h, r, d in get_fp8_einsum_shapes():
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhr,hdr->bhd', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, r), x_fp8[1].view(b, h, ceil_div(r, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, d), device='cuda', dtype=torch.bfloat16)
            output_dtypes = (torch.bfloat16, torch.float8_e4m3fn) if get_arch_major() == 10 else (torch.bfloat16,)
            for output_dtype in output_dtypes:
                if output_dtype == torch.float8_e4m3fn:
                    output = torch.empty_like(z, dtype=output_dtype)
                    sfd = torch.empty_strided(
                        (b, ceil_div(h * d, 32 * 4)), (1, align(b, 4)), dtype=torch.int32, device='cuda')
                    count_cuda = torch.empty(1, device='cuda', dtype=torch.uint32)
                    d_bf16 = torch.empty_like(z)
                    deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, d_bf16)

                    # Random dot products have variance r; 2.576 sigma clips about 1%.
                    for clamp_value in (None, 2.576 * r ** 0.5):
                        expected = d_bf16 if clamp_value is None else d_bf16.clamp(-clamp_value, clamp_value)
                        ref_fp8, ref_sfd = per_token_cast_to_fp8(
                            expected.view(b, h * d), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
                        ref_count = 0 if clamp_value is None else (d_bf16.abs() > clamp_value).sum().item()
                        # The scalar output must overwrite its previous contents.
                        for clamped_count in (None, count_cuda):
                            epilogue = deep_gemm.epilogue.FP8CastWithClamp(
                                sfd=sfd, clamp_value=clamp_value, clamped_count=clamped_count)
                            if clamped_count is not None:
                                clamped_count.fill_(0xffffffff)
                            if clamp_value is None and clamped_count is None:
                                deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, (output, sfd))
                            else:
                                deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, output, epilogue=epilogue)
                            if clamped_count is not None:
                                assert clamped_count.item() == ref_count, f'{b=}, {clamp_value=}: count'
                            assert torch.equal(output.view(b, h * d), ref_fp8), f'{b=}, {clamp_value=}: FP8'
                            assert torch.equal(sfd.contiguous(), ref_sfd), f'{b=}, {clamp_value=}: SF'
                        if clamp_value is None:
                            restored_output = cast_back_from_fp8(
                                output.view(b, h * d), sfd.contiguous(), gran_k=32, use_packed_ue8m0=True)
                            diff = calc_diff(restored_output, ref_z.reshape(b, h * d))
                else:
                    output = z
                    deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, output)
                    diff = calc_diff(output, ref_z)

                    if get_arch_major() == 10:
                        def launch_accumulated(out, acc):
                            assert out is acc
                            tmp = torch.zeros((h, b, d), device='cuda', dtype=torch.float).permute(1, 0, 2)
                            deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, tmp, c=tmp)
                            out.copy_(tmp)
                        z_sr = torch.empty_like(z)
                        deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z_sr,
                                             epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                        assert_stochastic_bf16_matches_fp32_accumulation(
                            z_sr, launch_accumulated, f'{b=}, {h=}, {r=}, {d=}')
                assert diff < (2e-3 if output_dtype == torch.float8_e4m3fn else 1e-3), \
                    f'{b=}, {h=}, {r=}, {d=}, {output_dtype=}, {diff=:.5f}'

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhr,hdr->bhd', x_fp8, y_fp8, z), 'gemm_', suppress_kineto_output=True, per_call_total=is_sm120_loop())
            t_cublaslt = bench_cublaslt(lambda: deep_gemm.einsum('bhr,hdr->bhd', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


# MN-major FP8 operands: SM100 in-kernel, SM120 one GEMM per head (`ldmatrix.trans.b8`). SM90 still rejects it
@test_filter(lambda: get_arch_major() in (10, 12))
def test_fp8_bhd_hdr_bhr(use_ue8m0: bool = True):
    deep_gemm.use_deterministic_algorithms(False)
    print('Testing FP8 "bhd, hdr -> bhr":')
    for h, r, d in get_fp8_einsum_shapes():
        for b in (4, 32, 128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((h, d, r), device='cuda', dtype=torch.bfloat16)
            ref_z = torch.einsum('bhd,hdr->bhr', x, y)

            x_fp8 = per_token_cast_to_fp8(x.view(-1, d), use_ue8m0=use_ue8m0)
            x_fp8 = x_fp8[0].view(b, h, d), x_fp8[1].view(b, h, ceil_div(d, 128))
            y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
                     torch.empty((h, ceil_div(d, 128), ceil_div(r, 128)), device='cuda', dtype=torch.float))
            for i in range(h):
                y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=use_ue8m0)
            z = torch.empty((b, h, r), device='cuda', dtype=torch.bfloat16)

            deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z)
            # SM120's own `mma.sync` accumulation order differs from the SM100 kernel, but the FP8 quantization error
            # (~1e-3) dominates it, so the `1e-3` tolerance is the same on both
            assert calc_diff(z, ref_z) < 1e-3

            # FP32 accumulation into `c` and the BF16 stochastic-rounding epilogue are SM100-only kernel features
            if get_arch_major() == 10:
                def launch_accumulated(out, acc):
                    assert out is acc
                    tmp = torch.zeros((h, b, r), device='cuda', dtype=torch.float).permute(1, 0, 2)
                    deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, tmp, c=tmp)
                    out.copy_(tmp)
                z_sr = torch.empty_like(z)
                deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z_sr,
                                     epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                assert_stochastic_bf16_matches_fp32_accumulation(
                    z_sr, launch_accumulated, f'{b=}, {h=}, {r=}, {d=}')

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,hdr->bhr', x_fp8, y_fp8, z), 'gemm_', suppress_kineto_output=True, per_call_total=is_sm120_loop())
            t_cublaslt = bench_cublaslt(lambda: deep_gemm.einsum('bhd,hdr->bhr', x, y, z), 'nvjet', suppress_kineto_output=True)
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z)) / t / 1e9:4.0f} GB/s | '
                  f'{t_cublaslt / t:4.2f} x')
    print()


# MN-major FP8 Wgrad-style einsum with per-channel (1x128) scales: SM100 in one batched launch, SM120 one block-scaled
# 1D1D GEMM per head (MN-major operands via `ldmatrix.trans.b8`, FP32 D accumulating in-kernel). SM90 still rejects it
@test_filter(lambda: get_arch_major() in (10, 12))
def test_fp8_bhd_bhr_hdr(use_ue8m0: bool = True):
    print('Testing FP8 "bhd, bhr -> hdr":')
    for h, r, d in get_fp8_einsum_shapes():
        for b in (128, 4096, 8192):
            x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
            y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
            z_0 = torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10
            ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x, y)

            x_fp8 = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=use_ue8m0)
            y_fp8 = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=use_ue8m0)
            x_fp8 = (x_fp8[0].view(b, h, d), x_fp8[1].view(ceil_div(b, 128), h, d))
            y_fp8 = (y_fp8[0].view(b, h, r), y_fp8[1].view(ceil_div(b, 128), h, r))
            z = z_0.clone()
            deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, c=z, recipe=(1, 1, 128))
            # Against the BF16 inputs the error is the FP8 quantization (~7e-4 here), the same on both architectures
            assert calc_diff(z, ref_z) < 1e-3
            if get_arch_major() == 12:
                # Against the dequantized FP8 operands only accumulation order differs (measured ~1e-13)
                xd = x_fp8[0].float() * x_fp8[1].repeat_interleave(128, 0)[:b]
                yd = y_fp8[0].float() * y_fp8[1].repeat_interleave(128, 0)[:b]
                assert calc_diff(z, z_0 + torch.einsum('bhd,bhr->hdr', xd, yd)) < 1e-10

            t = bench_kineto(lambda: deep_gemm.fp8_einsum('bhd,bhr->hdr', x_fp8, y_fp8, z, c=z, recipe=(1, 1, 128)), 'gemm_', suppress_kineto_output=True, per_call_total=is_sm120_loop())
            print(f' > Perf ({b=:4.0f}, {h=}, {r=}, {d=}): ',
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * b * h * r * d / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes((x_fp8, y_fp8, z, z)) / t / 1e9:4.0f} GB/s')
    print()


def quantize_fp8_einsum_operands(expr: str, b: int, h: int, r: int, d: int):
    # Quantized `(a, b, c)` of `expr`, in the layouts `fp8_einsum` takes
    if expr == 'bhd,bhr->hdr':
        x = torch.randn((b, h, d), device='cuda', dtype=torch.bfloat16)
        y = torch.randn((b, h, r), device='cuda', dtype=torch.bfloat16)
        x_fp8 = per_channel_cast_to_fp8(x.view(b, -1), use_ue8m0=True)
        y_fp8 = per_channel_cast_to_fp8(y.view(b, -1), use_ue8m0=True)
        return ((x_fp8[0].view(b, h, d), x_fp8[1].view(ceil_div(b, 128), h, d)),
                (y_fp8[0].view(b, h, r), y_fp8[1].view(ceil_div(b, 128), h, r)),
                torch.randn((h, d, r), device='cuda', dtype=torch.float32) * 10)
    k = d if expr == 'bhd,hdr->bhr' else r
    n = r if expr == 'bhd,hdr->bhr' else d
    x = torch.randn((b, h, k), device='cuda', dtype=torch.bfloat16)
    y = torch.randn((h, k, n) if expr == 'bhd,hdr->bhr' else (h, n, k), device='cuda', dtype=torch.bfloat16)
    x_fp8 = per_token_cast_to_fp8(x.view(-1, k), use_ue8m0=True)
    y_fp8 = (torch.empty_like(y, dtype=torch.float8_e4m3fn),
             torch.empty((h, ceil_div(y.size(1), 128), ceil_div(y.size(2), 128)), device='cuda', dtype=torch.float))
    for i in range(h):
        y_fp8[0][i], y_fp8[1][i] = per_block_cast_to_fp8(y[i], use_ue8m0=True)
    return (x_fp8[0].view(b, h, k), x_fp8[1].view(b, h, ceil_div(k, 128))), y_fp8, None
# SM120 runs the heads of an `fp8_einsum` as one batched launch (3D tensor maps over the strided head views), or with
# `DG_SM120_BATCHED_LAUNCH=0` as a GEMM per head. Both run the same MMAs in the same K order, so they must agree bit for bit
@test_filter(lambda: get_arch_major() == 12)
def test_fp8_einsum_batched_launch():
    print('Testing the SM120 batched launch of FP8 einsum against a launch per head:')
    deep_gemm.use_deterministic_algorithms(False)
    cases = [('bhd,hdr->bhr', 8, 4096, 1024, 4), ('bhd,hdr->bhr', 128, 512, 128, 33), ('bhd,hdr->bhr', 16, 384, 256, 200),
             ('bhr,hdr->bhd', 8, 4096, 1024, 5), ('bhr,hdr->bhd', 128, 512, 128, 130),
             ('bhd,bhr->hdr', 128, 512, 128, 128), ('bhd,bhr->hdr', 8, 4096, 1024, 256), ('bhd,bhr->hdr', 16, 384, 256, 384)]
    for expr, h, r, d, b in cases:
        a_fp8, b_fp8, c = quantize_fp8_einsum_operands(expr, b, h, r, d)
        outputs = []
        for batched in ('1', '0'):
            os.environ['DG_SM120_BATCHED_LAUNCH'] = batched
            if expr == 'bhd,bhr->hdr':
                out = c.clone()
                deep_gemm.fp8_einsum(expr, a_fp8, b_fp8, out, c=out, recipe=(1, 1, 128))
            else:
                out = torch.empty((b, h, r if expr == 'bhd,hdr->bhr' else d), device='cuda', dtype=torch.bfloat16)
                deep_gemm.fp8_einsum(expr, a_fp8, b_fp8, out)
            outputs.append(out)
        del os.environ['DG_SM120_BATCHED_LAUNCH']
        assert torch.equal(outputs[0], outputs[1]), f'{expr=}, {h=}, {r=}, {d=}, {b=}: batched != per-head'
        # Not just equal to each other: right (against the dequantized operands, which isolates the kernel from the FP8 error)
        if expr == 'bhd,bhr->hdr':
            xd = a_fp8[0].float() * a_fp8[1].repeat_interleave(128, 0)[:b]
            yd = b_fp8[0].float() * b_fp8[1].repeat_interleave(128, 0)[:b]
            ref = c + torch.einsum('bhd,bhr->hdr', xd, yd)
        else:
            xd = a_fp8[0].float() * a_fp8[1].repeat_interleave(128, -1)[..., :a_fp8[0].size(-1)]
            yd = b_fp8[0].float() * b_fp8[1].repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :b_fp8[0].size(1), :b_fp8[0].size(2)]
            ref = torch.einsum(expr, xd, yd)
        assert calc_diff(outputs[0], ref) < 1e-5, f'{expr=}, {h=}, {r=}, {d=}, {b=}: {calc_diff(outputs[0], ref)}'
    print(' > OK\n')
def test_bf16_einsum_batched_launch():
    # SM120 runs the heads of the BF16 einsums in one batched launch; a launch per head (DG_SM120_BATCHED_LAUNCH=0) must give
    # the same bits, with and without an accumulator, on views with padded strides
    if get_arch_major() != 12:
        return
    print('Testing the SM120 batched launch of BF16 einsum against a launch per head:')
    deep_gemm.use_deterministic_algorithms(True)
    cases = [('bhr,hdr->bhd', 8, 4096, 1024, 4), ('bhr,hdr->bhd', 128, 512, 128, 33), ('bhr,hdr->bhd', 16, 384, 256, 200),
             ('bhd,hdr->bhr', 8, 4096, 1024, 5), ('bhd,hdr->bhr', 128, 512, 128, 130), ('bhd,hdr->bhr', 16, 384, 256, 200)]
    for expr, h, r, d, b in cases:
        k = r if expr == 'bhr,hdr->bhd' else d
        n = d if expr == 'bhr,hdr->bhd' else r
        x = torch.randn((b, h, k), device='cuda', dtype=torch.bfloat16)
        fy = torch.randn((h, d, r + 64), device='cuda', dtype=torch.bfloat16)
        y = fy[:, :, :r]
        ref = torch.einsum(expr, x.float(), y.float())
        for accumulate in (False, True):
            init = torch.randn((b, h, n), device='cuda', dtype=torch.bfloat16)
            outputs = []
            for batched in ('1', '0'):
                os.environ['DG_SM120_BATCHED_LAUNCH'] = batched
                out = init.clone()
                deep_gemm.einsum(expr, x, y, out, c=out if accumulate else None)
                outputs.append(out)
            del os.environ['DG_SM120_BATCHED_LAUNCH']
            assert torch.equal(outputs[0], outputs[1]), f'{expr=}, {h=}, {r=}, {d=}, {b=}, {accumulate=}: batched != per-head'
            expected = ref + (init.float() if accumulate else 0)
            assert calc_diff(outputs[0], expected) < 1e-5, f'{expr=}, {h=}, {r=}, {d=}, {b=}: {calc_diff(outputs[0], expected)}'
    print(' > OK\n')
if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_bmk_bnk_mn()
    test_bhr_hdr_bhd()
    test_bhd_hdr_bhr()
    test_bhd_bhr_hdr()

    test_fp8_bhr_hdr_bhd()
    test_fp8_bhd_hdr_bhr()
    test_fp8_bhd_bhr_hdr()
    test_fp8_einsum_batched_launch()
    test_bf16_einsum_batched_launch()
