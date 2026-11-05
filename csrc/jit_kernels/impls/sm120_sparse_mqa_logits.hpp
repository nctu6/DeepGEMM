#pragma once

#include <format>

#include <deep_gemm/layout/sparse_mqa_logits.cuh>

#include "../../runtime/runtime.hpp"
#include "runtime_utils.hpp"

namespace deep_gemm {

using namespace layout::sparse_mqa_logits;

// SM120 sparse MQA logits launcher (see sm120_sparse_mqa_logits.cuh). Q/SF-Q/weights come via TMA; sparse KV is
// gathered inside the kernel from a contiguous tensor (+ its per-token int32 SF) or a paged fused cache
static void launch_sm120_sparse_mqa_logits(const bool is_paged, const int sparse_block_kv,
                                           const torch::Tensor& q, const torch::Tensor& sf_q,
                                           const torch::Tensor& kv, const torch::Tensor& sf_kv,
                                           const torch::Tensor& fused_kv_cache,
                                           const torch::Tensor& weights, const torch::Tensor& metadata,
                                           const torch::Tensor& logits) {
    const bool is_fp4 = q.scalar_type() == kPackedFP4;
    const int split_kv = is_fp4 ? 640 : 512;
    DG_HOST_ASSERT(split_kv % 128 == 0);
    const int num_heads = static_cast<int>(weights.size(1));
    const int num_q_tokens = static_cast<int>(q.size(0));
    const int num_sms = runtime->get_num_sms();
    const int row_bytes = kHeadDim / (is_fp4 ? 2 : 1);
    const int swizzle_mode = row_bytes;   // 128B FP8 / 64B FP4
    DG_HOST_ASSERT(num_heads > 0 and static_cast<uint32_t>(num_heads) <= kNumMaxHeads and num_heads % 4 == 0);

    const auto tensor_map_q = make_tma_2d_desc(q, kHeadDim, num_q_tokens * num_heads, kHeadDim, kBlockQ * num_heads,
                                               static_cast<int>(q.stride(is_paged ? 2 : 1)), swizzle_mode,
                                               0, false, not is_fp4);
    const auto tensor_map_sf_q = make_tma_2d_desc(sf_q, num_heads, num_q_tokens, num_heads, kBlockQ,
                                                  static_cast<int>(sf_q.stride(is_paged ? 1 : 0)), 0);
    const auto tensor_map_weights = make_tma_2d_desc(weights, align<int>(num_heads, 8), num_q_tokens, align<int>(num_heads, 8), kBlockQ,
                                                     static_cast<int>(weights.stride(0)), 0);

    // Shared memory, matching the kernel's layout
    const int num_kv_blocks_per_split = split_kv / sparse_block_kv;
    int smem = kBlockQ * num_heads * row_bytes;                 // Q
    smem += split_kv * row_bytes;                               // KV
    smem += align<int>(kBlockQ * num_heads * 4, 128);           // SF-Q (128-aligned TMA dest)
    smem += align<int>(split_kv * 4, 128);                      // SF-KV
    smem += align<int>(static_cast<int>(kBlockQ) * align<int>(num_heads, 8) * 2, 16); // weights
    smem += num_kv_blocks_per_split * static_cast<int>(sizeof(KVBlockInfo));
    smem += align<int>(static_cast<int>(sizeof(KVSplitHeader)), 16);
    smem += 16;                                                 // one barrier (16B aligned)
    smem = align(smem, 1024);
    DG_HOST_ASSERT(smem <= jit->device.get_num_smem_bytes());

    const auto qk_dtype_name = is_fp4 ? "cutlass::float_e2m1_t" : "cutlass::float_e4m3_t";
    (void) qk_dtype_name;

    const auto kernel = jit->compile("sm120_sparse_mqa_logits", std::format(R"(
#include <deep_gemm/impls/sm120_sparse_mqa_logits.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&sm120_sparse_mqa_logits<
        {}, {}, {}, {}, {}, {}
    >);
}};
)", num_heads, sparse_block_kv, split_kv, num_sms, is_fp4 ? "true" : "false", is_paged ? "true" : "false"));

    const uint32_t page_stride_bytes = is_paged ? static_cast<uint32_t>(fused_kv_cache.stride(0)) : 0;
    const uint32_t page_kv = is_paged ? static_cast<uint32_t>(fused_kv_cache.size(1)) : 0;
    const uint32_t num_kv_tokens = is_paged ? 0 : static_cast<uint32_t>(kv.size(0));

    jit->launch(
        kernel, {
            .num_smem_bytes = smem,
            .grid_dim = dim3(num_sms, 1, 1),
            .block_dim = dim3(256, 1, 1),
        },
        static_cast<uint32_t>(logits.stride(0)), logits.data_ptr(),
        metadata.data_ptr<uint8_t>(),
        is_paged ? nullptr : reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        is_paged ? nullptr : reinterpret_cast<const uint32_t*>(sf_kv.data_ptr()),
        num_kv_tokens,
        is_paged ? reinterpret_cast<const uint8_t*>(fused_kv_cache.data_ptr()) : nullptr,
        page_stride_bytes, page_kv,
        tensor_map_q, tensor_map_sf_q, tensor_map_weights
    );
}

} // namespace deep_gemm
