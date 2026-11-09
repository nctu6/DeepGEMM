import os
import random
import torch

import deep_gemm
from deep_gemm.utils import align, pack_ue8m0_to_int, cast_back_from_fp8, per_block_cast_to_fp8, per_token_cast_to_fp8
from deep_gemm.testing import (
    assert_bitwise_equal,
    bench_kineto,
    calc_diff, count_bytes,
    get_arch_major, test_filter
)
from utils import (
    add_sf_k_padding, assert_direct_output_matches_fp32_accumulation,
    assert_stochastic_bf16_matches_fp32_accumulation,
    assert_psum_zero_padding, convert_to_fp8, make_cublas_gemm,
)

from generators import (
    KernelType, QuantConfig, get_ue8m0_usage, is_sm120_loop,
    enumerate_normal, enumerate_m_grouped_contiguous, enumerate_m_grouped_masked, enumerate_k_grouped_contiguous,
    enumerate_k_grouped_contiguous_test_variants,
    generate_normal, generate_m_grouped_contiguous, generate_m_grouped_masked, generate_k_grouped_contiguous,
)


def test_gemm() -> None:
    print('Testing GEMM:')
    use_alpha_options = (False, True) if get_arch_major() == 10 else (False,)
    test_sf_k_padding = get_arch_major() == 10
    for kernel_type, quant_config, m, n, k, major_a, major_b, accumulate, out_dtype, scores in \
            enumerate_normal(torch.float8_e4m3fn, collect_cublas_scores=True):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        out_opt    = 'FP32' if out_dtype == torch.float else 'BF16'
        acc_opt    = f'acc={int(accumulate)}'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type, quant_config)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes(is_wgrad=(kernel_type.is_1d1d() and accumulate))

        for test_alias in (False, True):
            for use_alpha in use_alpha_options:
                alpha = random.uniform(-1.0, 1.0) if use_alpha else None
                a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype,
                                                     kernel_type, use_ue8m0=use_ue8m0,
                                                     quant_config=quant_config, alpha=alpha)
                func_name = f'fp8_fp4_gemm_{major_opt.lower() if test_alias else "nt"}'
                if test_alias:
                    a = a if major_a.is_k_major() else (a[0].T, a[1].T)
                    b = b if major_b.is_k_major() else (b[0].T, b[1].T)
                    assert a[0].is_contiguous() and b[0].is_contiguous()
                getattr(deep_gemm, func_name)(a, b, d, c=c, disable_ue8m0_cast=disable_ue8m0_cast,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b, alpha=alpha)
                diff = calc_diff(d, ref_d)
                assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}, '
                                                        f'{accumulate=}, {out_dtype=}, {use_alpha=}, {alpha=}, '
                                                        f'{diff:.5f}, alias={test_alias}')

        a, b, c, d, ref_d = generate_normal(m, n, k, major_a, major_b, accumulate, out_dtype, kernel_type, use_ue8m0=use_ue8m0, quant_config=quant_config)
        if get_arch_major() == 10 and not accumulate and out_dtype == torch.bfloat16:
            d_sr = torch.empty_like(d)
            deep_gemm.fp8_fp4_gemm_nt(a, b, d_sr, disable_ue8m0_cast=disable_ue8m0_cast,
                                      recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                                      epilogue=deep_gemm.epilogue.BF16StochasticRounding())
            assert_stochastic_bf16_matches_fp32_accumulation(
                d_sr,
                lambda output, accumulator: deep_gemm.fp8_fp4_gemm_nt(
                    a, b, output, c=accumulator, disable_ue8m0_cast=disable_ue8m0_cast,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b),
                f'FP8/FP4 GEMM stochastic rounding, {m=}, {n=}, {k=}, {kernel_opt}, {major_opt=}')
        initial_d = d.clone()
        def test_func(a_=a, b_=b, d_=d):
            deep_gemm.fp8_fp4_gemm_nt(a_, b_, d_, c=d_ if accumulate else None,
                                       disable_ue8m0_cast=disable_ue8m0_cast,
                                       recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
        equivalent_d = d.clone()
        test_func(d_=equivalent_d)
        # Bitwise deterministic test
        for _ in range(20):
            d.copy_(initial_d)
            test_func()
            assert torch.equal(d, equivalent_d), f'{m=}, {n=}, {k=}, {accumulate=}'
        if test_sf_k_padding and use_ue8m0 and not accumulate:
            packed_a = deep_gemm.transform_sf_into_required_layout(a[1], m, k, (1, quant_config.gran_k_a))
            gran_mn_b = 1 if quant_config.is_fp4_b else quant_config.gran_k_b
            packed_b = deep_gemm.transform_sf_into_required_layout(b[1], n, k, (gran_mn_b, quant_config.gran_k_b))
            strided_a = a[0], add_sf_k_padding(packed_a, padding=4)
            strided_b = b[0], add_sf_k_padding(packed_b, padding=4)
            strided_d = initial_d.clone()
            deep_gemm.fp8_fp4_gemm_nt(
                strided_a, strided_b, strided_d,
                recipe_a=(1, quant_config.gran_k_a), recipe_b=(1, quant_config.gran_k_b))
            assert_bitwise_equal(strided_d, equivalent_d, 'packed SF K-stride')
            test_sf_k_padding = False
        if quant_config.is_fp4_a or quant_config.is_fp4_b:
            equivalent_fp8_d = initial_d.clone()
            test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
            # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
            assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, (f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, '
                                                                      f'{accumulate=}')
        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        cublas_func = make_cublas_gemm(a, b, d, c)
        cublas_times = bench_kineto(cublas_func, ('nvjet', 'bstensorop', 'reduce'),
                                    suppress_kineto_output=True, with_multiple_kernels=True)
        cublas_t = sum(cublas_times)
        if cublas_t > 0:
            scores.append(cublas_t / t)
        print(f' > Perf (m={m:6}, n={n:6}, k={k:6}, {kernel_opt}, layout={major_opt}, {out_opt}, {acc_opt}): '
              f'{t * 1e6:6.1f} us | {2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{(count_bytes(a, b, d) + count_bytes(c) * int(accumulate)) / 1e9 / t:4.0f} GB/s | '
              f'{cublas_t / t:.2f}x cuBLAS speedup')


def test_m_grouped_gemm_contiguous() -> None:
    print('Testing m-grouped contiguous GEMM:')

    for kernel_type, quant_config, num_groups, expected_m_per_group, n, k, major_a, major_b, use_psum_layout, ensure_zero_padding in enumerate_m_grouped_contiguous(dtype=torch.float8_e4m3fn):
        major_opt  = 'N' if major_a.is_k_major() else 'T'
        major_opt += 'T' if major_b.is_k_major() else 'N'
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type, quant_config)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout()
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for test_alias in (False, True):
            m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                          quant_config=quant_config)
            func_name = f"m_grouped_fp8_fp4_gemm_{(major_opt.lower() if test_alias else 'nt')}_contiguous"
            if test_alias:
                assert major_a.is_k_major()
                b = b if major_b.is_k_major() else (b[0].mT, b[1].mT)
                assert a[0].is_contiguous() and b[0].is_contiguous()
            def test_func(a_=a, b_=b, d_=d):
                getattr(deep_gemm, func_name)(a_, b_, d_, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast,
                                              use_psum_layout=use_psum_layout, ensure_zero_padding=ensure_zero_padding,
                                              recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{m=}, {n=}, {k=}, alias={test_alias}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {m=}, {n=}, {k=}, alias={test_alias}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), (f'{m=}, {n=}, {k=}, {major_opt}, {kernel_opt}, '
                                                    f'{diff:.5f}, alias={test_alias}, {ensure_zero_padding=}')
            if use_psum_layout and ensure_zero_padding:
                assert_psum_zero_padding(a, d, grouped_layout, 'FP8/FP4')
        m, a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_contiguous(num_groups, expected_m_per_group, n, k, major_a, major_b,
                                                                          use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                          quant_config=quant_config)

        # noinspection PyShadowingNames
        def test_func():
            deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, disable_ue8m0_cast=disable_ue8m0_cast, use_psum_layout=use_psum_layout,
                                                           ensure_zero_padding=ensure_zero_padding,
                                                           recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)

        t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)
        print(f' > Perf ({num_groups=}, m={m:5}, n={n:6}, k={k:5}, {kernel_opt}, layout={major_opt}, '
              f'psum={use_psum_layout}, zero_pad={ensure_zero_padding}): '
              f'{t * 1e6:4.0f} us | '
              f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
              f'{count_bytes(a, b, d) / 1e9 / t:4.0f} GB/s')
    print()


def test_m_grouped_gemm_masked() -> None:
    print('Testing m-grouped masked GEMM:')

    # TODO: when the actual `m` is greater than `expected_m_per_group`, efficiency may significantly decrease.
    for kernel_type, quant_config, num_groups, max_m, expected_m_per_group, n, k, use_psum_layout in enumerate_m_grouped_masked(torch.float8_e4m3fn):
        kernel_opt = f'1D1D' if kernel_type.is_1d1d() else '1D2D'
        use_ue8m0 = get_ue8m0_usage(kernel_type, quant_config)
        disable_ue8m0_cast = not use_ue8m0
        recipe, recipe_a, recipe_b = quant_config.get_recipes()

        num_tests = 8
        sum_t, max_t = 0, 0
        sum_ops, sum_bytes = 0, 0
        expected_m = int(expected_m_per_group * 1.2)

        # Select best alignment
        alignment = deep_gemm.get_theoretical_mk_alignment_for_contiguous_layout(expected_m)
        deep_gemm.set_mk_alignment_for_contiguous_layout(alignment)

        for i in range(num_tests):
            a, b, grouped_layout, d, ref_d, valid_mask = generate_m_grouped_masked(num_groups, max_m, expected_m_per_group, n, k,
                                                                                   use_ue8m0=use_ue8m0, use_psum_layout=use_psum_layout,
                                                                                   quant_config=quant_config)
            def test_func(a_=a, b_=b, d_=d):
                common = dict(disable_ue8m0_cast=disable_ue8m0_cast, recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b)
                if use_psum_layout:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a_, b_, d_, grouped_layout,
                                                                   use_psum_layout=True, expected_m_for_psum_layout=expected_m, **common)
                else:
                    deep_gemm.m_grouped_fp8_fp4_gemm_nt_masked(a_, b_, d_, grouped_layout, expected_m, **common)

            equivalent_d = d.clone()
            test_func(d_=equivalent_d)
            # Bitwise deterministic test
            for _ in range(20):
                test_func()
                assert torch.equal(d[valid_mask], equivalent_d[valid_mask]), f'{max_m=}, {n=}, {k=}'
            if quant_config.is_fp4_a or quant_config.is_fp4_b:
                equivalent_fp8_d = d.clone()
                test_func(convert_to_fp8(a), convert_to_fp8(b), equivalent_fp8_d)
                # FP4 and converted FP8 have different UMMA_K, but BF16 outputs are usually bitwise identical.
                assert calc_diff(equivalent_d[valid_mask], equivalent_fp8_d[valid_mask]) < 1e-14, (
                    f'FP4/FP8 mismatch: {max_m=}, {n=}, {k=}, {num_groups=}')
            diff = calc_diff(d[valid_mask], ref_d[valid_mask])
            assert diff < quant_config.max_diff(), f'{max_m=}, {n=}, {k=}, {kernel_opt}, {num_groups=}, {diff:.5f}'

            # Test performance with fixed shapes
            valid_m = int(valid_mask.sum().item())
            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True)

            sum_t += t
            max_t = max(max_t, t)
            sum_ops += 2 * valid_m * n * k
            sum_bytes += count_bytes(a, d) * valid_m / (max_m * num_groups) + count_bytes(b)

        print(f' > Perf (num_groups={num_groups:2}, expected_m_per_group={expected_m_per_group:4}, n={n:4}, k={k:4}, '
              f'{kernel_opt}, psum={1 if use_psum_layout else 0}): '
              f'{sum_t / num_tests * 1e6:4.0f} us (max: {max_t * 1e6:3.0f} us) | '
              f'{sum_ops / sum_t / 1e12:4.0f} TFLOPS | '
              f'{sum_bytes / sum_t / 1e9:4.0f} GB/s')
    print()


def test_k_grouped_gemm_contiguous() -> None:
    print('Testing k-grouped GEMM:')

    arch_major = get_arch_major()
    test_sf_k_padding = arch_major == 10
    test_options = [(torch.float8_e4m3fn, QuantConfig(),
                     deep_gemm.k_grouped_fp8_gemm_nt_contiguous if arch_major == 9 else
                     deep_gemm.k_grouped_fp8_gemm_tn_contiguous)]
    if arch_major == 10:
        test_options.append((torch.float4_e2m1fn_x2, QuantConfig((32, 32, True, True)),
                             deep_gemm.k_grouped_fp4_gemm_nt_contiguous))
    # The 1x128 / 1x32 scales of both operands run on the block-scaled kernel as packed UE8M0 on SM100 and SM120
    use_ue8m0 = get_ue8m0_usage(KernelType.Kernel1D1D) or arch_major == 12
    for dtype, quant_config, gemm in test_options:
        is_fp4 = dtype == torch.float4_e2m1fn_x2
        dtype_opt = 'FP4' if is_fp4 else 'FP8'

        for num_groups, m, n, major_a, major_b, real_ks_cpu, _, _, gran_k, k_alignment, use_psum_layout, accumulate, out_dtype in \
                enumerate_k_grouped_contiguous(dtype):
            recipe = (1, 1, gran_k)

            for test_real_ks_cpu in enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
                total_k, a, b, c, d, ref_d, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                    num_groups, m, n, major_a, major_b, test_real_ks_cpu,
                    use_ue8m0=use_ue8m0, gran_k=gran_k,
                    quant_config=quant_config if is_fp4 else None,
                    use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                    accumulate=accumulate, out_dtype=out_dtype)

                initial_d = d.clone()
                if not accumulate:
                    initial_d.fill_(float('nan'))
                equivalent_d = initial_d.clone()
                gemm(a, b, equivalent_d, host_ks_cpu, grouped_layout, equivalent_d if accumulate else None,
                     recipe=recipe, use_psum_layout=use_psum_layout)
                if is_fp4 and accumulate:
                    fp8_a, fp8_b = convert_to_fp8(a), convert_to_fp8(b)
                    fp8_a = (fp8_a[0].T.contiguous(), fp8_a[1])
                    fp8_b = (fp8_b[0].T.contiguous(), fp8_b[1])
                    equivalent_fp8_d = initial_d.clone()
                    deep_gemm.k_grouped_fp8_gemm_tn_contiguous(
                        fp8_a, fp8_b, equivalent_fp8_d, host_ks_cpu, grouped_layout, equivalent_fp8_d if accumulate else None,
                        recipe=recipe, use_psum_layout=use_psum_layout)
                    mismatch_message = (f'FP4/FP8 mismatch: {m=}, {n=}, {total_k=}, '
                                        f'{test_real_ks_cpu=}, {use_psum_layout=}')
                    assert calc_diff(equivalent_d, equivalent_fp8_d) < 1e-14, mismatch_message

                # Bitwise deterministic test
                host_ks_options = (host_ks_cpu, None, []) if use_psum_layout else (host_ks_cpu, )
                for test_host_ks_cpu in host_ks_options:
                    for stress_idx in range(20):
                        d.copy_(initial_d)
                        gemm(a, b, d, test_host_ks_cpu, grouped_layout, c,
                             recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_bitwise_equal(
                            d, equivalent_d,
                            f'k-grouped self-consistency at {stress_idx=}, {dtype_opt}, {m=}, {n=}, {total_k=}, '
                            f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}'
                        )
                    if not accumulate:
                        case_label = (f'{dtype_opt} K-grouped direct output, {m=}, {n=}, {total_k=}, '
                                      f'{test_real_ks_cpu=}, {test_host_ks_cpu=}, {use_psum_layout=}, '
                                      f'{out_dtype=}')
                        launch = lambda output, accumulator: gemm(
                            a, b, output, test_host_ks_cpu, grouped_layout, accumulator,
                            recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_direct_output_matches_fp32_accumulation(d, launch, case_label)
                        # Stochastic rounding is an SM100 epilogue
                        if out_dtype == torch.bfloat16 and arch_major == 10:
                            gemm(a, b, d, test_host_ks_cpu, grouped_layout, None,
                                 recipe=recipe, use_psum_layout=use_psum_layout,
                                 epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                            assert_stochastic_bf16_matches_fp32_accumulation(d, launch, f'stochastic rounding, {case_label}')

                if accumulate:
                    diff = calc_diff(d, ref_d)
                    assert diff < quant_config.max_diff(), (
                        f'{dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                        f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, {diff:.5f}')

                # gran_k=128 requires FP32 SF input; only gran_k=32 accepts
                # the per-group packed INT32 UE8M0 layout.
                if gran_k == 32:
                    sf_ks = [k // gran_k for k in host_ks_cpu]
                    ref_packed_a = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(a[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    ref_packed_b = torch.cat([
                        pack_ue8m0_to_int(torch.nn.functional.pad(
                            group_sf.T, (0, align(sf_k, 4) - sf_k)).contiguous()).T
                        for group_sf, sf_k in zip(b[1].split(sf_ks), sf_ks) if sf_k > 0
                    ])
                    packed_a = (
                        a[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            a[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    packed_b = (
                        b[0], deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
                            b[1], grouped_layout, host_ks_cpu, gran_k, k_alignment, use_psum_layout))
                    assert torch.equal(packed_a[1], ref_packed_a)
                    assert torch.equal(packed_b[1], ref_packed_b)

                    packed_d = initial_d.clone()
                    gemm(packed_a, packed_b, packed_d, host_ks_cpu, grouped_layout, packed_d if accumulate else None,
                         recipe=recipe, use_psum_layout=use_psum_layout)
                    if test_sf_k_padding:
                        strided_a = packed_a[0], add_sf_k_padding(packed_a[1], padding=4, k_grouped=True)
                        strided_b = packed_b[0], add_sf_k_padding(packed_b[1], padding=4, k_grouped=True)
                        strided_d = initial_d.clone()
                        gemm(strided_a, strided_b, strided_d, host_ks_cpu, grouped_layout,
                             strided_d if accumulate else None, recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_bitwise_equal(strided_d, packed_d, 'K-grouped packed SF K-stride')
                        test_sf_k_padding = False
                    if accumulate:
                        packed_diff = calc_diff(packed_d, ref_d)
                        assert packed_diff < quant_config.max_diff(), (
                            f'pre-packed INT32 SF: {dtype_opt}, {m=}, {n=}, {total_k=}, {test_real_ks_cpu=}, '
                            f'{host_ks_cpu=}, {use_psum_layout=}, {accumulate=}, {out_dtype=}, '
                            f'{packed_diff:.5f}')
                    else:
                        case_label = (f'pre-packed INT32 SF direct output: {dtype_opt}, {m=}, {n=}, '
                                      f'{total_k=}, {test_real_ks_cpu=}, {host_ks_cpu=}, '
                                      f'{use_psum_layout=}, {out_dtype=}')
                        launch = lambda output, accumulator: gemm(
                            packed_a, packed_b, output, host_ks_cpu, grouped_layout, accumulator,
                            recipe=recipe, use_psum_layout=use_psum_layout)
                        assert_direct_output_matches_fp32_accumulation(packed_d, launch, case_label)
                        if out_dtype == torch.bfloat16 and arch_major == 10:
                            gemm(packed_a, packed_b, packed_d, host_ks_cpu, grouped_layout, None,
                                 recipe=recipe, use_psum_layout=use_psum_layout,
                                 epilogue=deep_gemm.epilogue.BF16StochasticRounding())
                            assert_stochastic_bf16_matches_fp32_accumulation(packed_d, launch, f'stochastic rounding, {case_label}')

            _, a, b, c, d, _, grouped_layout, host_ks_cpu = generate_k_grouped_contiguous(
                num_groups, m, n, major_a, major_b, real_ks_cpu,
                use_ue8m0=use_ue8m0, gran_k=gran_k,
                quant_config=quant_config if is_fp4 else None,
                use_psum_layout=use_psum_layout, k_alignment=k_alignment,
                accumulate=accumulate, out_dtype=out_dtype)

            # noinspection PyShadowingNames
            def test_func():
                gemm(a, b, d, host_ks_cpu, grouped_layout, c,
                     recipe=recipe, use_psum_layout=use_psum_layout)

            # SM120 launches one GEMM per group: sum them
            t = bench_kineto(test_func, 'gemm_', suppress_kineto_output=True, per_call_total=is_sm120_loop())
            logical_k = sum(real_ks_cpu)
            out_opt = 'FP32' if out_dtype == torch.float else 'BF16'
            print(f' > Perf ({dtype_opt}, {num_groups=:2}, m={m:5}, n={n:5}, k={logical_k:5}, gran_k={gran_k:3}, '
                  f'k_alignment={k_alignment:3}, psum={int(use_psum_layout)}, acc={int(accumulate)}, {out_opt}): '
                  f'{t * 1e6:4.0f} us | '
                  f'{2 * m * n * logical_k / t / 1e12:4.0f} TFLOPS | '
                  f'{count_bytes(a, b, c, d) / 1e9 / t:4.0f} GB/s')
    # `enumerate_k_grouped_contiguous` leaves the contiguous-layout MK alignment at its last value; restore the 128
    # default so later tests (e.g. the m-grouped mixed-scale pair) see a block_m their heuristics can tile
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    print()


@test_filter(lambda: get_arch_major() in (10, 12))
def test_mixed_ue8m0_and_fp32_scale_pair() -> None:
    # Packed UE8M0 activation scales with FP32 128x128 weight scales: the FP32 side is cast to packed UE8M0 (exact for
    # power-of-two scales), as on SM100. SM120 otherwise keeps 128-element FP32 scales in FP32 for its FP32-scale kernel
    print('Testing mixed UE8M0 / FP32 scale pairs:')
    block, num_groups, m_per_group, n, k = 128, 4, 256, 512, 1024
    x = torch.randn((num_groups * m_per_group, k), device='cuda')
    w = torch.randn((num_groups, n, k), device='cuda') * 0.1
    x_blocks = x.view(-1, k // block, block)
    x_sf = deep_gemm.ceil_to_ue8m0(x_blocks.abs().amax(-1, keepdim=True).clamp(min=1e-6) / 448)
    x_fp8 = (x_blocks / x_sf).to(torch.float8_e4m3fn).view(-1, k)
    x_sf = x_sf.squeeze(-1).contiguous()
    w_blocks = w.view(num_groups, n // block, block, k // block, block)
    w_sf = deep_gemm.ceil_to_ue8m0(w_blocks.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-6) / 448)
    w_fp8 = (w_blocks / w_sf).to(torch.float8_e4m3fn).view(num_groups, n, k)
    w_sf = w_sf.view(num_groups, n // block, k // block).contiguous()
    x_ref = x_fp8.float() * x_sf.repeat_interleave(block, -1)
    w_ref = w_fp8.float() * w_sf.repeat_interleave(block, -2).repeat_interleave(block, -1)
    x_sf_packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(x_sf)

    m_indices = torch.arange(num_groups, device='cuda', dtype=torch.int32).repeat_interleave(m_per_group)
    ref = torch.cat([x_ref[i * m_per_group:(i + 1) * m_per_group] @ w_ref[i].T for i in range(num_groups)])
    d = torch.empty((num_groups * m_per_group, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous((x_fp8, x_sf_packed), (w_fp8, w_sf), d, m_indices, recipe=(1, block, block))
    diff = calc_diff(d, ref)
    assert diff < 1e-5, f'm-grouped contiguous: {diff=}'

    d = torch.empty((num_groups * m_per_group, n), device='cuda', dtype=torch.bfloat16)
    deep_gemm.fp8_gemm_nt((x_fp8, x_sf_packed), (w_fp8[0], w_sf[0]), d, recipe=(1, block, block))
    diff = calc_diff(d, x_ref @ w_ref[0].T)
    assert diff < 1e-5, f'normal: {diff=}'

    if get_arch_major() == 12:
        try:
            deep_gemm.fp8_gemm_nt((x_fp8, x_sf_packed), (w_fp8[0], w_sf[0]), d, recipe=(1, block, block),
                                  disable_ue8m0_cast=True)
            assert False, 'mixing scale formats without the UE8M0 cast must be rejected'
        except RuntimeError as e:
            assert 'needs the UE8M0 cast' in str(e)
    print(' > Passed\n')


@test_filter(lambda: get_arch_major() == 12)
def test_mn_major_fp8_sm120() -> None:
    # SM120 reads MN-major FP8 operands in-kernel (the 8-bit transposing `ldmatrix`), or, where the host policy says so (a
    # small MN-major A, shapes the TMA descriptors cannot express), through an explicit byte transpose and the K-major
    # kernel. Both routes, on both FP8 kernels (FP32 block scales, and packed UE8M0 scales of 32/128 elements), must match
    # the dequantized reference as tightly as the K-major kernel does
    print('Testing MN-major FP8 operands on SM120:')
    mn_major = lambda x: x.mT.contiguous().mT
    shapes = [(16, 576, 1024), (48, 1536, 2048), (128, 2112, 7168), (192, 4096, 1536), (4096, 512, 512), (256, 24, 1152)]
    for route in ('0', '1', '2'):
        os.environ['DG_SM120_MN_MAJOR_FP8'] = route
        for kind in ('1d2d', 'g32', 'g128'):
            for m, n, k in shapes:
                a_bf16 = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
                b_bf16 = torch.randn((n, k), device='cuda', dtype=torch.bfloat16)
                if kind == '1d2d':
                    qa, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
                    qb, sfb = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)
                    ref = cast_back_from_fp8(qa, sfa, gran_k=128) @ (qb.float() * sfb.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]).T
                    kwargs = dict(disable_ue8m0_cast=True)
                else:
                    gran_k = 32 if kind == 'g32' else 128
                    qa, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=gran_k)
                    qb, sfb = per_token_cast_to_fp8(b_bf16, use_ue8m0=True, gran_k=gran_k)
                    ref = cast_back_from_fp8(qa, sfa, gran_k=gran_k) @ cast_back_from_fp8(qb, sfb, gran_k=gran_k).T
                    if gran_k == 128:
                        sfa = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfa)
                        sfb = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(sfb)
                    kwargs = dict(recipe=(1, 1, gran_k))
                for major_a, major_b in ((False, True), (True, False), (True, True)):
                    d = torch.empty((m, n), device='cuda', dtype=torch.bfloat16)
                    deep_gemm.fp8_fp4_gemm_nt((mn_major(qa) if major_a else qa, sfa), (mn_major(qb) if major_b else qb, sfb), d, **kwargs)
                    diff = calc_diff(d, ref)
                    assert diff < QuantConfig().max_diff(), f'{route=}, {kind=}, {m=}, {n=}, {k=}, {major_a=}, {major_b=}, {diff=}'

    # m-grouped contiguous with an MN-major B `[G, K, N]`: whole K blocks run in-kernel, a K tail takes the host transpose
    os.environ.pop('DG_SM120_MN_MAJOR_FP8', None)
    num_groups, m_per_group, n = 4, 128, 512
    for k in (1152, 1184):
        a_bf16 = torch.randn((num_groups * m_per_group, k), device='cuda', dtype=torch.bfloat16)
        b_bf16 = torch.randn((num_groups, n, k), device='cuda', dtype=torch.bfloat16)
        qa, sfa = per_token_cast_to_fp8(a_bf16, use_ue8m0=True, gran_k=32)
        qb = torch.empty_like(b_bf16, dtype=torch.float8_e4m3fn)
        sfb = torch.empty((num_groups, n, k // 32), device='cuda', dtype=torch.float)
        for i in range(num_groups):
            qb[i], sfb[i] = per_token_cast_to_fp8(b_bf16[i], use_ue8m0=True, gran_k=32)
        a_ref = cast_back_from_fp8(qa, sfa, gran_k=32)
        ref = torch.cat([a_ref[i * m_per_group:(i + 1) * m_per_group] @ cast_back_from_fp8(qb[i], sfb[i], gran_k=32).T for i in range(num_groups)])
        m_indices = torch.arange(num_groups, device='cuda', dtype=torch.int32).repeat_interleave(m_per_group)
        d = torch.empty((num_groups * m_per_group, n), device='cuda', dtype=torch.bfloat16)
        deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous((qa, sfa), (mn_major(qb), sfb), d, m_indices, recipe=(1, 1, 32))
        diff = calc_diff(d, ref)
        assert diff < QuantConfig().max_diff(), f'm-grouped MN-major B, {k=}, {diff=}'

    # FP4 stays K-major (no sub-byte transposing load): a clear error, not a wrong result
    fp4 = torch.zeros((64, 16), device='cuda', dtype=torch.int8).T   # packed FP4 `[16, 64]`, MN-major
    fp4_sf = torch.ones((16, 1), device='cuda', dtype=torch.int)
    try:
        deep_gemm.fp8_fp4_gemm_nt((fp4, fp4_sf), (fp4.contiguous(), fp4_sf), torch.empty((16, 16), device='cuda', dtype=torch.bfloat16),
                                  recipe=(1, 1, 32))
        assert False, 'MN-major FP4 must be rejected'
    except RuntimeError as e:
        assert 'K-major' in str(e), str(e)
    print(' > Passed\n')


if __name__ == '__main__':
    torch.manual_seed(0)
    random.seed(0)

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    test_gemm()
    test_m_grouped_gemm_contiguous()
    test_m_grouped_gemm_masked()
    test_k_grouped_gemm_contiguous()
    test_mixed_ue8m0_and_fp32_scale_pair()
    test_mn_major_fp8_sm120()
