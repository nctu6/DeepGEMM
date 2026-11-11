#pragma once

#include <format>

#include "../../runtime/runtime.hpp"
#include "../heuristics/sm120.hpp"

#include "kernel_lookup.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

// SM120 MQA logits reuse the SM90 kernels' scheduling, TMA pipelines and epilogues
// (`sm120_fp8_mqa_logits.cuh` / `sm120_fp8_paged_mqa_logits.cuh`), with `mma.sync` instead of WGMMA,
// and pipeline depths reduced to fit ~99 KB of shared memory

// Contiguous: 8 math warps, each owning 16 of the `BLOCK_KV = 128` KV rows
static constexpr int kSM120MQALogitsBlockKV = 128;
// Paged: 4 math warp-groups, one 64-token KV block each, i.e. the same 256-token split as SM90
static constexpr int kSM120PagedMQALogitsSplitKV = 256;

static void sm120_fp8_mqa_logits(const torch::Tensor& q,
                                 const torch::Tensor& kv, const torch::Tensor& kv_scales,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seq_len_k_start,
                                 const torch::Tensor& cu_seq_len_k_end,
                                 const torch::Tensor& logits,
                                 const int& seq_len, const int& seq_len_kv,
                                 const int& stride_logits,
                                 const int& num_heads, const int& head_dim,
                                 const int& block_q, const int& block_kv) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 256;
    constexpr int num_q_stages = 2, num_kv_stages = 3;
    const int num_sms = runtime->get_num_sms();
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT((head_dim == 128 or head_dim == 64) and "SM120 MQA logits support head dim 128 or 64");
    DG_HOST_ASSERT(block_kv == kSM120MQALogitsBlockKV and block_kv == num_math_threads / 2);
    DG_HOST_ASSERT(128 % num_heads == 0);

    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, head_dim, head_dim);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, head_dim, head_dim);
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_scales,
                                                       get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_scales.element_size())),
                                                       1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, seq_len,
                                                     num_heads, block_q,
                                                     static_cast<int>(weights.stride(0)), 0);

    // Same layout as the kernel: Q stages, KV stages, weights, KV scales, then barriers
    int smem_size = 0;
    smem_size += num_q_stages * block_q * num_heads * head_dim * static_cast<int>(q.element_size());
    smem_size += num_kv_stages * block_kv * head_dim * static_cast<int>(kv.element_size());
    smem_size += num_q_stages * block_q * num_heads * static_cast<int>(weights.element_size());
    smem_size += num_kv_stages * block_kv * static_cast<int>(kv_scales.element_size());
    smem_size += (num_q_stages * 2 + num_kv_stages * 2) * 8;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::get_smem_capacity());

    const auto logits_dtype = logits.scalar_type();
    const auto kernel = get_cached_kernel("sm120_fp8_mqa_logits", std::array<int64_t, 6>{
        num_heads, head_dim, block_q, block_kv, num_sms, static_cast<int64_t>(logits_dtype)}, [&] {
        return std::format(R"(
#include <deep_gemm/impls/sm120_fp8_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}
    >);
}};
)",
        num_heads, head_dim,
        block_q, block_kv,
        num_q_stages, num_kv_stages,
        num_sms,
        num_specialized_threads, num_math_threads,
        to_string(logits_dtype));
    });

    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        seq_len, seq_len_kv, stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr(),
        tensor_map_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

// Block-scaled (MXFP8 / MXFP4) dense MQA logits for SM120. The tensor core applies the UE8M0 scales, so Q and KV each
// carry a per-row int32 SF (head dim 128 -> one 4-group word). Weights and logits are BF16. Head dim 128 only.
static void sm120_fp8_fp4_mqa_logits(const torch::Tensor& q, const torch::Tensor& q_sf,
                                     const torch::Tensor& kv, const torch::Tensor& kv_sf,
                                     const torch::Tensor& weights,
                                     const torch::Tensor& cu_seq_len_k_start,
                                     const torch::Tensor& cu_seq_len_k_end,
                                     const torch::Tensor& logits,
                                     const int& seq_len, const int& seq_len_kv,
                                     const int& stride_logits,
                                     const int& num_heads, const int& head_dim,
                                     const int& block_q) {
    constexpr int num_specialized_threads = 128;
    constexpr int num_math_threads = 256;
    const int block_kv = num_math_threads / 2;   // 128 KV rows per stage (8 math warps x 16)
    const bool is_fp4 = q.scalar_type() == kPackedFP4;
    const int num_q_stages = 2, num_kv_stages = is_fp4 ? 4 : 3;
    const int num_sms = runtime->get_num_sms();
    const int row_bytes = head_dim / (is_fp4 ? 2 : 1);
    const int swizzle_mode = row_bytes;          // 128B FP8 / 64B FP4
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT(head_dim == 128 and "SM120 block-scaled MQA logits only support head dim 128");
    DG_HOST_ASSERT(128 % num_heads == 0);
    const int num_weight_elems = align<int>(num_heads, 16 / static_cast<int>(weights.element_size()));
    DG_HOST_ASSERT(weights.scalar_type() == torch::kBFloat16 or weights.scalar_type() == torch::kFloat);

    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, seq_len * num_heads,
                                               head_dim, block_q * num_heads, row_bytes,
                                               swizzle_mode, 0, false, not is_fp4);
    const auto tensor_map_sf_q = make_tma_2d_desc(q_sf, num_heads, seq_len,
                                                  num_heads, block_q, num_heads, 0);
    const auto tensor_map_kv = make_tma_2d_desc(kv, head_dim, seq_len_kv,
                                                head_dim, block_kv, row_bytes,
                                                swizzle_mode, 0, false, not is_fp4);
    const auto tensor_map_sf_kv = make_tma_2d_desc(kv_sf,
                                                   get_tma_aligned_size(seq_len_kv, static_cast<int>(kv_sf.element_size())),
                                                   1, block_kv, 1, 0, 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_weight_elems, seq_len,
                                                     num_weight_elems, block_q, static_cast<int>(weights.stride(0)), 0);

    // Match the kernel's layout: Q, KV, SF-Q, SF-KV, weights, then barriers.
    int smem_size = 0;
    smem_size += num_q_stages * block_q * num_heads * row_bytes;
    smem_size += num_kv_stages * block_kv * row_bytes;
    smem_size += num_q_stages * align<int>(block_q * num_heads * 4, 128);
    smem_size += num_kv_stages * align<int>(block_kv * 4, 128);
    smem_size += num_q_stages * align<int>(block_q * num_weight_elems * static_cast<int>(weights.element_size()), 128);
    smem_size = align(smem_size, 16) + (num_q_stages * 2 + num_kv_stages * 2) * 16;
    DG_HOST_ASSERT(smem_size <= SM120ArchSpec::get_smem_capacity());

    const auto weights_dtype = weights.scalar_type(), logits_dtype = logits.scalar_type();
    const auto kernel = get_cached_kernel("sm120_fp8_fp4_mqa_logits", std::array<int64_t, 9>{
        num_heads, head_dim, block_q, block_kv, num_kv_stages, num_sms, is_fp4,
        static_cast<int64_t>(weights_dtype), static_cast<int64_t>(logits_dtype)}, [&] {
        return std::format(R"(
#include <deep_gemm/impls/sm120_fp8_fp4_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_fp4_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {}, {}
    >);
}};
)",
        num_heads, head_dim,
        block_q, block_kv,
        num_q_stages, num_kv_stages,
        num_sms,
        num_specialized_threads, num_math_threads,
        is_fp4 ? "true" : "false", to_string(weights_dtype), to_string(logits_dtype));
    });

    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        seq_len, seq_len_kv, stride_logits,
        cu_seq_len_k_start.data_ptr<int>(), cu_seq_len_k_end.data_ptr<int>(),
        logits.data_ptr(),
        tensor_map_q, tensor_map_sf_q, tensor_map_kv, tensor_map_sf_kv, tensor_map_weights
    );
}

// Paged MQA logits. FP8 queries and keys carry a float scale per KV token (`q_sf` empty); block-scaled MXFP8 / MXFP4 ones
// (`q_sf` given, head dim 128) carry a packed UE8M0 word per Q row and per KV token, applied by the tensor core. The same
// kernel serves both: the K dtype is a template parameter
static void sm120_fp8_paged_mqa_logits(const torch::Tensor& q,
                                       const std::optional<torch::Tensor>& q_sf,
                                       const torch::Tensor& kv_cache,
                                       const torch::Tensor& kv_cache_scales,
                                       const torch::Tensor& weights,
                                       const torch::Tensor& context_lens,
                                       const torch::Tensor& logits,
                                       const torch::Tensor& block_table,
                                       const torch::Tensor& schedule_meta,
                                       const int& batch_size, const int& next_n,
                                       const int& num_heads, const int& head_dim,
                                       const int& num_kv_blocks, const int& block_kv,
                                       const bool& is_context_lens_2d,
                                       const int& logits_stride,
                                       const int& block_table_stride,
                                       const int& num_sms,
                                       const int& split_kv) {
    constexpr int num_specialized_threads = 128;
    constexpr int mma_m = 64;
    const int num_math_warp_groups = split_kv / mma_m;
    const int num_math_threads = num_math_warp_groups * 128;
    const bool is_mx = q_sf.has_value();
    const bool is_fp4 = q.scalar_type() == kPackedFP4;
    // Bytes of one Q / KV row: FP4 packs two values per byte
    const int row_bytes = is_fp4 ? head_dim / 2 : head_dim;
    DG_HOST_ASSERT(jit->device.get_arch_major() == 12);
    DG_HOST_ASSERT((head_dim == 128 or head_dim == 64 or head_dim == 32) and "SM120 paged MQA logits support head dim 32, 64 or 128");
    DG_HOST_ASSERT((not is_mx or head_dim == 128) and "SM120 block-scaled paged MQA logits support head dim 128");
    DG_HOST_ASSERT(not is_fp4 or is_mx);
    // MXFP8 needs twice the K steps of MXFP4 per row; at 128 Q rows (next_n 2 x 64 heads) its fragments and scale words
    // overflow the 104 registers of a math thread (see `kNumMathRegisters`), so that one combination is rejected
    DG_HOST_ASSERT((not is_mx or is_fp4 or next_n * num_heads <= 64) and
                   "SM120 MXFP8 paged MQA logits support at most 64 Q rows (next_n x heads); use MXFP4 or FP8");
    DG_HOST_ASSERT(weights.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(block_kv == 64 and split_kv == kSM120PagedMQALogitsSplitKV);
    DG_HOST_ASSERT(split_kv % mma_m == 0 and logits_stride % split_kv == 0);
    DG_HOST_ASSERT(next_n == 1 or next_n == 2);
    DG_HOST_ASSERT(128 % num_heads == 0);

    const int next_n_atom = next_n >= 2 ? 2 : 1;
    // FP4 rows are 4-bit elements in the tensor maps, packed two per byte in shared memory
    const auto tensor_map_q = make_tma_2d_desc(q, head_dim, batch_size * next_n * num_heads,
                                               head_dim, next_n_atom * num_heads,
                                               static_cast<int>(q.stride(2)),
                                               row_bytes, 0, false, not is_fp4);
    const auto tensor_map_sf_q = is_mx ? make_tma_2d_desc(q_sf.value(), num_heads, batch_size * next_n,
                                                          num_heads, next_n_atom, num_heads, 0)
                                       : tensor_map_q;
    const auto tensor_map_kv = make_tma_3d_desc(kv_cache, head_dim, block_kv, num_kv_blocks,
                                                head_dim, block_kv, 1,
                                                static_cast<int>(kv_cache.stride(1)),
                                                static_cast<int>(kv_cache.stride(0)),
                                                row_bytes, 0, false, not is_fp4);
    const auto tensor_map_kv_scales = make_tma_2d_desc(kv_cache_scales, block_kv, num_kv_blocks,
                                                       block_kv, 1,
                                                       static_cast<int>(kv_cache_scales.stride(0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, num_heads, batch_size * next_n,
                                                     num_heads, next_n_atom,
                                                     static_cast<int>(weights.stride(0)), 0);

    // Same layout as the kernel: a Q pipeline, then one KV pipeline per math warp-group
    // NOTES: pick the deepest pipelines (at most 3 stages) that fit in shared memory; at least 2 Q stages are
    // required, as the TMA warp prefetches the next Q (blocking its KV loads) while the current one is in use
    // The kernel needs 1024-byte aligned tiles for every row width (see `kSwizzleAlignment` there)
    const int swizzle_alignment = std::max(row_bytes * 8, 1024);
    const int smem_q_size_per_stage = next_n * num_heads * row_bytes;
    const int aligned_smem_weight_size_per_stage = align(next_n * num_heads * static_cast<int>(weights.element_size()), swizzle_alignment);
    const int aligned_smem_q_sf_size_per_stage = is_mx ? align(next_n * num_heads * 4, swizzle_alignment) : 0;
    const int smem_kv_size_per_stage = block_kv * row_bytes;
    const int aligned_smem_kv_scale_size_per_stage = align(block_kv * static_cast<int>(kv_cache_scales.element_size()), swizzle_alignment);
    const auto get_smem_size = [&](const int& num_q_stages, const int& num_kv_stages) {
        const int smem_q_pipe_size = num_q_stages * (smem_q_size_per_stage + aligned_smem_weight_size_per_stage + aligned_smem_q_sf_size_per_stage) +
                                     align(num_q_stages * 8 * 2, swizzle_alignment);
        const int smem_kv_pipe_size = num_kv_stages * (smem_kv_size_per_stage + aligned_smem_kv_scale_size_per_stage) +
                                      align(num_kv_stages * 8 * 2, swizzle_alignment);
        return smem_q_pipe_size + num_math_warp_groups * smem_kv_pipe_size;
    };
    int num_q_stages = 0, num_kv_stages = 0;
    for (const auto& [q_stages, kv_stages]: {std::pair{3, 3}, {2, 3}, {2, 2}, {2, 1}}) {
        if (get_smem_size(q_stages, kv_stages) <= SM120ArchSpec::get_smem_capacity()) {
            num_q_stages = q_stages, num_kv_stages = kv_stages;
            break;
        }
    }
    DG_HOST_ASSERT(num_q_stages > 0 and "Not enough shared memory for SM120 paged MQA logits");
    const int smem_size = get_smem_size(num_q_stages, num_kv_stages);

    const auto kernel = jit->compile("sm120_fp8_paged_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_fp8_paged_mqa_logits<
        {}, {},
        {}, {},
        {}, {},
        {}, {},
        {},
        {}, {},
        {}, {},
        {}
    >);
}};
)",
    next_n, num_heads,
    head_dim, block_kv,
    is_context_lens_2d, "false",
    num_q_stages, num_kv_stages,
    split_kv,
    num_specialized_threads, num_math_threads,
    is_mx ? "true" : "false", is_fp4 ? "true" : "false",
    to_string(logits.scalar_type())));

    jit->launch(
        kernel, {
            .num_smem_bytes = smem_size,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(num_specialized_threads + num_math_threads, 1, 1),
        },
        batch_size,
        logits_stride, block_table_stride,
        context_lens.data_ptr<int>(), logits.data_ptr(),
        block_table.data_ptr<int>(), nullptr, schedule_meta.data_ptr<int>(),
        tensor_map_q, tensor_map_sf_q, tensor_map_kv,
        tensor_map_kv_scales, tensor_map_weights
    );
}

} // namespace deep_gemm
