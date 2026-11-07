#pragma once

#include <format>
#include <variant>

#include <pybind11/pybind11.h>
#include <torch/python.h>

#include "../utils/exception.hpp"
#include "../utils/layout.hpp"
#include "../utils/compatibility.hpp"
#include "gemm.hpp"

#include "../jit_kernels/impls/sm90_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm100_bmk_bnk_mn.hpp"
#include "../jit_kernels/impls/sm90_bf16_gemm.hpp"
#include "../jit_kernels/impls/sm100_bf16_gemm.hpp"
#include "../jit_kernels/impls/smxx_cublaslt.hpp"

namespace deep_gemm::einsum {

static void bmk_bnk_mn(const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& d,
                       const std::optional<torch::Tensor>& c) {
    // Currently FP32 only support the accumulated expression
    if (d.scalar_type() == torch::kFloat) {
        DG_HOST_ASSERT(c.has_value() and "FP32 `bmk,bnk->mn` accumulates: pass `c` aliasing `d`");
        DG_HOST_ASSERT(c->data_ptr() == d.data_ptr() and c->sizes() == d.sizes() and c->strides() == d.strides());
    } else {
        DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);
        DG_HOST_ASSERT(not c.has_value());

        const auto workspace = torch::empty_like(d, d.options().dtype(torch::kFloat32));
        DG_CUDA_RUNTIME_CHECK(cudaMemsetAsync(workspace.data_ptr(), 0, workspace.nbytes(),
                              c10::cuda::getCurrentCUDAStream()));
        bmk_bnk_mn(a, b, workspace, workspace);

        // This line has an implicit FP32-to-BF16 casting
        d.copy_(workspace);
        return;
    }

    DG_HOST_ASSERT(a.is_contiguous());
    DG_HOST_ASSERT(b.is_contiguous());
    DG_HOST_ASSERT(d.is_contiguous());

    const auto [s , m, k ] = get_shape<3>(a);
    const auto [s_, n, k_] = get_shape<3>(b);
    DG_HOST_ASSERT(s == s_ and k == k_);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 9) {
        sm90_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 10) {
        sm100_bmn_bnk_mn_gemm(a, b, d, s, m, n, k);
    } else if (arch_major == 12) {
        // `d = c + sum_s a[s] @ b[s].T`, with c == d: a BF16 GEMM per batch slice, every slice accumulating into the
        // FP32 D (which already holds c). a[s]/b[s] are [m,k]/[n,k], K-major and contiguous. The BF16-D case routes
        // through a zeroed FP32 workspace above, so this branch always sees FP32 D with c == d
        DG_HOST_ASSERT(c.has_value() and c->data_ptr() == d.data_ptr());
        for (int i = 0; i < s; ++ i)
            sm120_bf16_gemm_1d1d(a[i], b[i], std::make_optional(d), d, m, n, k,
                                 cute::UMMA::Major::K, cute::UMMA::Major::K, "mnk", 1.0f);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhr_hdr_bhd(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D,
                        const std::optional<torch::Tensor>& C,
                        const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto [b , h  , r ] = get_shape<3>(A);
    const auto [h_, d  , r_] = get_shape<3>(B);
    const auto [b_, h__, d_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.stride(2) == 1);

    // Early return for trivial cases
    if (h == 0 or gemm::early_return(b, d, r, D, C))
        return;

    const auto resolved_epilogue_class = resolve_epilogue_class(epilogue_class, C, D);
    const auto epilogue_alpha = resolved_epilogue_class->get_alpha();

    // Dispatch implementation; cuBLASLt only covers a plain store, optionally alpha-scaled
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
        and epilogue_alpha.has_value()) {
        cublaslt_bhr_hdr_bhd(A, B, D, b, h, r, d, C.has_value(), epilogue_alpha.value());
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_bf16_bhr_hdr_bhd(A, B, C, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhr_hdr_bhd(A, B, C, D, b, h, r, d, "nk", resolved_epilogue_class);
    } else if (arch_major == 12) {
        // One BF16 GEMM per head on strided views: `D[:, i] = A[:, i] @ B[i]^T` (+ C, which `early_return` has already
        // copied into D), all K-major
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        for (int i = 0; i < h; ++ i)
            sm120_bf16_gemm_1d1d(A.select(1, i), B[i], C.has_value() ? std::make_optional(D) : std::nullopt, D.select(1, i),
                                 b, d, r, cute::UMMA::Major::K, cute::UMMA::Major::K, "nk", 1.0f);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_hdr_bhr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D,
                        const std::optional<torch::Tensor>& C,
                        const std::shared_ptr<EpilogueClass>& epilogue_class) {
    const auto [b , h  , d ] = get_shape<3>(A);
    const auto [h_, d_ , r ] = get_shape<3>(B);
    const auto [b_, h__, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and r == r_ and d == d_ and h == h__);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.stride(2) == 1);

    // Early return for trivial cases
    if (h == 0 or gemm::early_return(b, r, d, D, C))
        return;

    const auto resolved_epilogue_class = resolve_epilogue_class(epilogue_class, C, D);
    const auto epilogue_alpha = resolved_epilogue_class->get_alpha();

    // Dispatch implementation; cuBLASLt only covers a plain store, optionally alpha-scaled
    const auto arch_major = jit->device.get_arch_major();
    if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
        and epilogue_alpha.has_value()) {
        cublaslt_bhd_hdr_bhr(A, B, D, b, h, r, d, C.has_value(), epilogue_alpha.value());
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm90_bf16_bhd_hdr_bhr(A, B, C, D, b, h, r, d);
    } else if (arch_major == 10) {
        sm100_bf16_bhd_hdr_bhr(A, B, C, D, b, h, r, d, "nk", resolved_epilogue_class);
    } else if (arch_major == 12) {
        // One BF16 GEMM per head: `D[:, i] = A[:, i] @ B[i]` (+ C, already in D). `B[i]` is `[d, r]` with `r` contiguous,
        // i.e. an MN-major `[r, d]` B operand
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        for (int i = 0; i < h; ++ i)
            sm120_bf16_gemm_1d1d(A.select(1, i), B[i].transpose(0, 1), C.has_value() ? std::make_optional(D) : std::nullopt,
                                 D.select(1, i), b, r, d, cute::UMMA::Major::K, cute::UMMA::Major::MN, "nk", 1.0f);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

static void bhd_bhr_hdr(const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& D,
                        const std::optional<torch::Tensor>& C) {
    const auto [b , h  , d ] = get_shape<3>(A);
    const auto [b_, h_ , r ] = get_shape<3>(B);
    const auto [h__, d_, r_] = get_shape<3>(D);
    DG_HOST_ASSERT(b == b_ and h == h_ and h == h__ and d == d_ and r == r_);

    DG_HOST_ASSERT(A.scalar_type() == torch::kBFloat16 and A.stride(2) == 1);
    DG_HOST_ASSERT(B.scalar_type() == torch::kBFloat16 and B.stride(2) == 1);
    DG_HOST_ASSERT(D.scalar_type() == torch::kFloat and D.stride(2) == 1);
    if (C.has_value()) {
        DG_HOST_ASSERT(C->scalar_type() == D.scalar_type() and C->sizes() == D.sizes() and C->stride(2) == 1);
    }

    // Early return for trivial cases
    if (h == 0 or gemm::early_return(d, r, b, D, C))
        return;

    // TODO: investigate cuBLAS determinism.
    cublaslt_bhd_bhr_hdr(A, B, D, b, h, r, d, C.has_value());
}

static void einsum(const std::string& expr,
                   const torch::Tensor& a,
                   const torch::Tensor& b,
                   const torch::Tensor& d,
                   const std::optional<torch::Tensor>& c,
                   const std::shared_ptr<EpilogueClass>& epilogue_class) {
    DG_HOST_ASSERT(a.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(b.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16 or d.scalar_type() == torch::kFloat);
    if (c.has_value()) {
        DG_HOST_ASSERT(d.scalar_type() == c->scalar_type());
    }

    // Some hardcoded Einstein sum kernels
    // TODO: support any expression
    // TODO: canonicalize expression
    if (expr == "bmk,bnk->mn") {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues are unsupported for this expression");
        bmk_bnk_mn(a, b, d, c);
    } else if (expr == "bhr,hdr->bhd") {
        bhr_hdr_bhd(a, b, d, c, epilogue_class);
    } else if (expr == "bhd,hdr->bhr") {
        bhd_hdr_bhr(a, b, d, c, epilogue_class);
    } else if (expr == "bhd,bhr->hdr") {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues are unsupported for this expression");
        bhd_bhr_hdr(a, b, d, c);
    } else {
        DG_HOST_UNREACHABLE(std::format("Unsupported einsum expression: {}", expr));
    }
}

// The D output is either a plain BF16/FP32 tensor, or an FP8 `(d, sfd)` pair
// quantized with dynamic per-32 packed UE8M0 SFs
static void fp8_bmm(const torch::Tensor& a, const torch::Tensor& sfa,
                    const torch::Tensor& b, const torch::Tensor& sfb,
                    const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
                    const std::optional<torch::Tensor>& c,
                    std::optional<std::tuple<int, int, int>> recipe,
                    const std::string& compiled_dims,
                    const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // The FP8 output pair is shorthand for `FP8CastWithClamp` without clamping
    const auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&d);
    const auto d_tensor = d_fp8 != nullptr ? d_fp8->first : std::get<torch::Tensor>(d);
    auto merged_epilogue_class = epilogue_class;
    std::optional<torch::Tensor> sfd;
    if (d_fp8 != nullptr) {
        DG_HOST_ASSERT(epilogue_class == nullptr and "The FP8 output pair supports no other epilogue");
        sfd = d_fp8->second;
        merged_epilogue_class = std::make_shared<FP8CastWithClampEpilogue>(d_fp8->second);
    } else {
        sfd = merged_epilogue_class != nullptr ? merged_epilogue_class->output_sfd() : std::nullopt;
    }

    // Shape must be `[B, M, K] @ [B, N, K].T`
    const auto major_a = a.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    const auto major_b = b.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
    DG_HOST_ASSERT(a.stride(-1) == 1 or a.stride(-2) == 1);
    DG_HOST_ASSERT(b.stride(-1) == 1 or b.stride(-2) == 1);
    DG_HOST_ASSERT(d_tensor.stride(-1) == 1);

    // Type and shape checks
    const auto [batch_size  , m , k ] = get_shape<3>(a);
    const auto [batch_size_ , n , k_] = get_shape<3>(b);
    const auto [batch_size__, m_, n_] = get_shape<3>(d_tensor);
    DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
    DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn);
    DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn);
    if (sfd.has_value()) {
        // The SF layout matches a per-token cast of D viewed as a 2D `(m, batch_size * n)`
        // matrix, where each row concatenates the `n` columns of all batches. This requires:
        //  - `d.stride(0) == n`: the batches are contiguous along the columns of the view
        //  - `n % 128 == 0`: each batch is quantized independently, so neither an SF group
        //    nor a packed 4-SF word may cross a batch boundary
        DG_HOST_ASSERT(jit->device.get_arch_major() == 10);
        DG_HOST_ASSERT(d_tensor.stride(0) == n and n % 128 == 0);
        check_sf_layout(sfd.value(), m, batch_size * n, 1, 32, std::nullopt, true, false, torch::kInt);
    }

    // Early return for trivial cases
    if (batch_size == 0 or gemm::early_return(m, n, k, d_tensor, c, sfd))
        return;

    // Transform scaling factors
    const auto [transformed_sfa, transformed_sfb, gran_k_a, gran_k_b] = layout::transform_sf_pair_into_required_layout(
        sfa, sfb, m, n, k, recipe, std::nullopt, std::nullopt, batch_size, batch_size, false);

    // Dispatch implementation
    const auto arch_major = jit->device.get_arch_major();
    if (arch_major == 10) {
        sm100_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d_tensor, batch_size, m, n, k,
                      gran_k_a, gran_k_b, major_a, major_b, compiled_dims,
                      resolve_epilogue_class(merged_epilogue_class, c, d_tensor));
    } else if (arch_major == 12) {
        // SM120: one launch for all batches through 3D tensor maps when they can describe the views (the heads of an
        // einsum are strided views, which they do), otherwise one FP32-scale GEMM per batch on the 2D view of each
        DG_HOST_ASSERT(merged_epilogue_class == nullptr and "Custom epilogues require SM100");
        DG_HOST_ASSERT(not c.has_value() and "SM120 FP8 bmm does not support accumulation");
        DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128 and transformed_sfa.scalar_type() == torch::kFloat);
        const auto major_sfb = get_major_type_ab(transformed_sfb[0]);
        if (sm120_fp8_bmm_1d2d_supported(a, transformed_sfa, b, transformed_sfb, d_tensor, major_a, major_b)) {
            sm120_fp8_bmm_1d2d(a, transformed_sfa, b, transformed_sfb, d_tensor, batch_size, m, n, k,
                               major_a, major_b, major_sfb, compiled_dims);
        } else {
            for (int i = 0; i < batch_size; ++ i) {
                const auto sfb_i = transformed_sfb[i];
                sm120_fp8_gemm_1d2d(a[i], transformed_sfa[i], b[i], sfb_i, std::nullopt, d_tensor[i], m, n, k,
                                    major_a, major_b, get_major_type_ab(sfb_i), compiled_dims);
            }
        }
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(merged_epilogue_class == nullptr and "Custom epilogues require SM100");
        const auto major_sfb = get_major_type_ab(sfb);
        DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);
        sm90_fp8_bmm(a, transformed_sfa, b, transformed_sfb, c, d_tensor, batch_size, m, n, k, major_a, major_b, major_sfb, compiled_dims);
    } else {
        DG_HOST_UNREACHABLE("Unsupported architecture");
    }
}

// SM120 `fp8_einsum bhd,bhr->hdr`: `D[i] = C[i] + A[:, i].T @ B[:, i]` per head, contracting over `b`. Both operands are
// MN-major FP8 (read in-kernel with the 8-bit transposing `ldmatrix`), D is FP32 and accumulates onto C in-kernel.
// The scales are per-channel (1 x 128, one per row and K block of each operand), which the FP32 1D2D kernel cannot take
// (it reads SFB as 128 x 128 blocks), so this runs on the block-scaled 1D1D kernel with packed UE8M0 scales, packed the
// way SM100 does (exact for power-of-two scales). All heads run in one batched launch (a GEMM per head only if the views
// are not TMA-describable, or with `DG_SM120_BATCHED_LAUNCH=0`)
static void sm120_fp8_bhd_bhr_hdr(const std::pair<torch::Tensor, torch::Tensor>& a,
                                  const std::pair<torch::Tensor, torch::Tensor>& b,
                                  const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
                                  const std::optional<torch::Tensor>& c,
                                  const std::tuple<int, int, int>& recipe) {
    DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
    const auto& D = std::get<torch::Tensor>(d);
    DG_HOST_ASSERT(D.scalar_type() == torch::kFloat and D.stride(-1) == 1);
    DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128) and "SM120 `bhd,bhr->hdr` takes per-channel 1x128 scales");

    // (batch_size, m, n, k): (h, d, r, b)
    const auto perm_a = a.first.permute({1, 2, 0});
    const auto perm_sfa = a.second.permute({1, 2, 0});
    const auto perm_b = b.first.permute({1, 2, 0});
    const auto perm_sfb = b.second.permute({1, 2, 0});

    const auto [h , m , k ] = get_shape<3>(perm_a);
    const auto [h_, n , k_] = get_shape<3>(perm_b);
    const auto [h__, m_, n_] = get_shape<3>(D);
    DG_HOST_ASSERT(h == h_ and h == h__ and m == m_ and n == n_ and k == k_);
    DG_HOST_ASSERT(perm_a.scalar_type() == torch::kFloat8_e4m3fn and perm_b.scalar_type() == torch::kFloat8_e4m3fn);

    // Early return for trivial cases; otherwise D holds C (the kernel adds to it) or is overwritten
    if (h == 0 or gemm::early_return(m, n, k, D, c))
        return;

    const auto sfa = get_mn_major_tma_aligned_packed_ue8m0_tensor(perm_sfa);
    const auto sfb = get_mn_major_tma_aligned_packed_ue8m0_tensor(perm_sfb);
    check_sf_layout(sfa, m, k, 1, 128, h, true, false, torch::kInt);
    check_sf_layout(sfb, n, k, 1, 128, h, true, false, torch::kInt);
    // One launch for all heads through 3D tensor maps when they can describe the views, otherwise a GEMM per head
    if (sm120_fp8_bmm_1d1d_supported(perm_a, sfa, perm_b, sfb, D, k, cute::UMMA::Major::MN, cute::UMMA::Major::MN)) {
        sm120_fp8_bmm_1d1d(perm_a, sfa, perm_b, sfb, c.has_value(), D, h, m, n, k, 128,
                           cute::UMMA::Major::MN, cute::UMMA::Major::MN, "mn");
    } else {
        for (int i = 0; i < h; ++ i)
            sm120_fp8_fp4_gemm_1d1d(perm_a[i], sfa[i], perm_b[i], sfb[i], c.has_value() ? std::make_optional(D[i]) : std::nullopt,
                                    D[i], m, n, k, 128, 128, cute::UMMA::Major::MN, cute::UMMA::Major::MN, "mn", 1.0f);
    }
}

static void fp8_einsum(const std::string& expr,
                       const std::pair<torch::Tensor, torch::Tensor>& a,
                       const std::pair<torch::Tensor, torch::Tensor>& b,
                       const std::variant<torch::Tensor, std::pair<torch::Tensor, torch::Tensor>>& d,
                       const std::optional<torch::Tensor>& c,
                       const std::tuple<int, int, int>& recipe,
                       const std::shared_ptr<EpilogueClass>& epilogue_class) {
    // Some hardcoded Einstein sum kernels
    // NOTES: only `bhr,hdr->bhd` accepts an FP8 `(d, sfd)` output pair; the other
    //        expressions take a plain BF16/FP32 D
    const auto arch_major = jit->device.get_arch_major();
    if (expr == "bhr,hdr->bhd") {
        // Permute dims to satisfy the order of (batch_size, m, n, k)
        // (batch_size, m, n, k): (h, b, d, r)
        // NOTES: the FP8 output SF columns are flattened over D's last two dims (matching a
        //        per-token cast of `d.flatten(1, 2)`), so the SF layout is invariant under
        //        the `(b, h)` permute of the data dims
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        auto perm_d = d;
        if (const auto d_fp8 = std::get_if<std::pair<torch::Tensor, torch::Tensor>>(&perm_d))
            d_fp8->first = d_fp8->first.permute({1, 0, 2});
        else
            std::get<torch::Tensor>(perm_d) = std::get<torch::Tensor>(perm_d).permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, b.first, b.second, perm_d, perm_c, recipe, "nk", epilogue_class);
    } else if (expr == "bhd,hdr->bhr" and (arch_major == 10 or arch_major == 12)) {
        // (batch_size, m, n, k): (h, b, r, d). K-major A and MN-major B (the 8-bit transposing `ldmatrix` reads it on SM120);
        // both architectures run all heads in one batched launch over strided views. Plain BF16 D, no accumulation
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        const auto perm_a = a.first.permute({1, 0, 2});
        const auto perm_sfa = a.second.permute({1, 0, 2});
        const auto perm_b = b.first.permute({0, 2, 1});
        const auto perm_sfb = b.second.permute({0, 2, 1});
        const auto perm_d = std::get<torch::Tensor>(d).permute({1, 0, 2});
        const auto perm_c = c.has_value() ? std::make_optional(c.value().permute({1, 0, 2})) : std::nullopt;
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, perm_d, perm_c, recipe, "nk", epilogue_class);
    } else if (expr == "bhd,bhr->hdr" and arch_major == 10) {
        // (batch_size, m, n, k): (h, d, r, b)
        DG_HOST_ASSERT(std::holds_alternative<torch::Tensor>(d));
        const auto perm_a = a.first.permute({1, 2, 0});
        const auto perm_sfa = a.second.permute({1, 2, 0});
        const auto perm_b = b.first.permute({1, 2, 0});
        const auto perm_sfb = b.second.permute({1, 2, 0});
        fp8_bmm(perm_a, perm_sfa, perm_b, perm_sfb, d, c, recipe, "mn", epilogue_class);
    } else if (expr == "bhd,bhr->hdr" and arch_major == 12) {
        DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
        sm120_fp8_bhd_bhr_hdr(a, b, d, c, recipe);
    } else if (expr == "bhd,hdr->bhr" or expr == "bhd,bhr->hdr") {
        DG_HOST_UNREACHABLE(std::format("fp8_einsum `{}` has no kernel on this architecture", expr));
    } else {
        DG_HOST_UNREACHABLE(std::format("Unsupported einsum expression: {}", expr));
    }
}
static void register_apis(pybind11::module_& m) {
    m.def("einsum", &einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("epilogue") = nullptr);
    m.def("fp8_einsum", &fp8_einsum,
          py::arg("expr"), py::arg("a"), py::arg("b"),
          py::arg("d"), py::arg("c") = std::nullopt,
          py::arg("recipe") = std::make_tuple(1, 128, 128),
          py::arg("epilogue") = nullptr);
}

} // namespace deep_gemm::einsum
