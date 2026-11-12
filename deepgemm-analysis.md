# DeepGEMM on sm120: analysis, plan, and status log

Goal: DeepGEMM kernels running on NVIDIA sm120 (RTX PRO 6000 Blackwell Server
Edition) so TokenSpeed can serve DeepSeek models at the best performance on
that hardware.

This file is the single source of truth for the effort. Update the status log
(section 9) after every step. Re-read sections 2, 5 and 8 before starting any
new work so we don't drift into one of the known dead ends.

## Current state (2026-11-12, updated for task 47)

- **Supported on sm120:**
  - `fp8_gemm_nt`, m-grouped contiguous (incl. psum) and masked FP8 GEMMs, with FP32 1x128 / 128x128 scales.
  - `fp8_mqa_logits` and `fp8_paged_mqa_logits` (H 32/64, next_n 1/2; head dim 128/64 for the contiguous/prefill kernel, 128/64/32 for the paged one, section 9.13), plus `get_paged_mqa_logits_metadata`.
  - **Block-scaled paged MQA logits** (`fp8_fp4_paged_mqa_logits` with MXFP4 or MXFP8 rows, head dim 128, H 32/64, next_n 1/2; MXFP8 up to 64 Q rows): the same paged kernel with the K dtype as a template parameter, so the 68-byte V4.1 index rows are scored natively, 1.0-1.74x faster than the FP8 kernel at the same shapes. The block-scaled contiguous and paged kernels take FP32 weights and write FP32 logits on request (`logits_dtype`). Section 9.21.
  - Legacy aliases `fp8_mqa_logits` / `fp8_paged_mqa_logits` and the probe `get_supported_arch_majors()`.
  - **Block-scaled GEMMs** (`fp8_fp4_gemm_nt` and the m-grouped contiguous / masked variants): **MXFP4** (E2M1, 32-element UE8M0 scales), **NVFP4** (E2M1, 16-element UE4M3 scales, `scale_vec::4X`), and FP8 with 32- or 128-element UE8M0 scales. The tensor core applies the scales (`kind::mxf4nvf4` m16n8k64, `kind::mxf8f6f4` m16n8k32). BF16 or FP32 output, optional `alpha` and C accumulation. Includes the PSUM layout, BLOCK_N = 32 tiles, 64-byte FP4 K blocks, PDL weight pre-issue, and stream-K for few-tile decode. The cost model is fitted jointly on all three modes. See section 5.6 and the status log.
  - **W4A8** (task 21, section 9.26): FP8 activations against MXFP4 weights on the same kernel (`kMixedFP4B`, `kind::mxf8f6f4 ... e4m3.e2m1` with E2M1 in 8-bit containers), for `fp8_fp4_gemm_nt` and the m-grouped contiguous (incl. PSUM) / masked variants, 64-byte B rows, K-major. Bitwise equal to the FP8 path on converted weights.
  - **`fp8_fp4_mega_moe`, one rank or EP across GPUs** (task 21, section 9.26): dispatch, two W4A8 grouped GEMMs, SwiGLU and combine as a sequence of kernels with the SM100 arithmetic; same API and symmetric-buffer input slices. Several ranks (up to 8, run on 2) write into each other's torch symmetric-memory buffers with posted peer stores over PCIe (CUDA IPC) and order with a system-scope epoch barrier; results are bitwise equal to one rank over all experts, and the sequence replays in CUDA graphs. No fused shared experts. `get_mega_moe_max_ranks()` reports the rank count.
  - **BF16 GEMMs** on the same kernel: `bf16_gemm_nt` (cuBLASLt stays the default unless deterministic mode; `alpha` and C accumulation supported), `m_grouped_bf16_gemm_nt_contiguous` (incl. PSUM), `m_grouped_bf16_gemm_nt_masked`. **MN-major BF16 operands** (`bf16_gemm_nn/tn/tt`, a grouped MN-major B) and **`k_grouped_bf16_gemm_tn_contiguous`** also run, see section 9.9.
  - **`tf32_hc_prenorm_gemm`** (TF32 `m16n8k8`): `d = a @ b.T` and the row squared-norm, with K-split.
  - **`einsum`**: `bmk,bnk->mn` (BF16, a GEMM per batch slice accumulating into one FP32 D), and native BF16 `bhr,hdr->bhd` and `bhd,hdr->bhr` (deterministic mode only; since task 18 all heads in one batched launch, bit-identical to a launch per head, 0.21x of the per-head time (4.8x faster) and within 3% of cuBLASLt on 52 of 56 points; section 9.24). **`fp8_einsum`** `bhr,hdr->bhd` (K-major operands) and the MN-major `bhd,hdr->bhr` and `bhd,bhr->hdr` run all heads as one batched GEMM launch (3D tensor maps over the strided head views), which closed the per-head-launch gap of section 9.12; see section 9.14.
- **Sparse MX indexer kernel (task 35, section 9.39):** `fp8_fp4_sparse_mqa_logits` / `fp8_fp4_paged_sparse_mqa_logits` give each CTA an even share of the metadata's KV splits, gather with one `cp.async` per lane and block, overlap the gather of the next split with the MMAs, and keep a warp's Q fragments in registers (bitwise equal output; kernel time for MXFP4, 32 heads, 16384-row pool: 8 / 32 / 128 requests of 32K rows 82 / 94 / 252 -> 21 / 45 / 146 us, 128 / 1024 / 4096 queries of one request 140 / 710 / 2770 -> 43 / 243 / 912 us). TokenSpeed's `_sparse_reindex_wins` is one threshold now (history at least twice the pool). MXFP4 paged logits stay at 1.07 TB/s: FP8 with head dim 64 on the same 4352-byte pages reaches the same rate, so the gap to FP8's 1.23 TB/s is the page geometry, not the MX path.
- **compute-sanitizer (task 38, section 9.42):** the FP8 k-grouped, FP8 masked and `test_mqa_logits` runs that did not finish now complete under memcheck, synccheck, racecheck and initcheck with 0 kernel errors: full case lists (126, 64, 64), every kernel binary launched under each tool, DeepGEMM kernels only for the first three tools (kernel filter, positive control), unfiltered initcheck. Test-side fixes in `0d89c76`: `test_mqa_logits` no longer copies the unwritten padding columns, and `test_psum_gaps_do_not_reach_valid_rows_sm120` guards that the PSUM padding rows the SM120 kernels read at tile granularity never reach a valid row.
- **Host path of a 128-token V3.2 DSA chunk (task 37, section 9.41):** the contiguous MQA launchers find their kernel by its template integers (`kernel_lookup.hpp`, `get_cached_kernel`) instead of hashing the generated source through `jit->compile` on every call, return their view with one `as_strided` and read the debug env variables without a heap string: `fp8_mqa_logits` 8.6 -> 5.6 us of host time (DeepGEMM `f628449`). TokenSpeed's `deepselect_local_topk` and the DeepGEMM prefill leaf skip slices, copies and conversions of tensors that are already whole (TokenSpeed `436833133`): 17.2 -> 9.1 us; the leaf's host time 116 -> 87 us at 128 tokens, outputs bitwise equal, no change in a 4-layer V3.2 run (GPU-bound at ~11 ms per chunk).
- **Packed-row CuTe DSL V4 / V4.1 decode (task 36, section 9.40):** `cute_dsl_dsv4_mma_decode` and `cute_dsl_dsv41_mma_selected_attention` (TokenSpeed `2098a0c41`) read the FP8 / FP4 page-planar cache rows inside the CuTe core: a thread loads its packed bytes a tile ahead and dequantizes them exactly into the BF16 ring tile, so no BF16 workspace is written or read back; rows are bit-identical to the workspace route's, the batch-invariant fold is untouched. Over 105 points per model (8-128 heads, widths 128 / 640 / 1152, 1-2048 tokens): V4 1.75x over the workspace route and 1.71x over the Triton direct reader (worst 1.06x / 1.12x), V4.1 3.12x and 1.62x (1.92x with a compressed cache; window-only layers tie at 0.94-1.06x in places). Selected for every head count (V4 16 heads included), the Triton readers remain the fallback.
- **Done since this section was first written (2026-11-05/06):** sparse MQA logits (`fp8_fp4_sparse_mqa_logits` + paged + the two metadata entry points) and **block-scaled MXFP8/MXFP4 dense MQA logits** (`fp8_fp4_mqa_logits` MX path for arch 12). Both validated against a dequantized reference (`scripts/sm120_sparse_mqa_check.py`, `sm120_sparse_paged_check.py`, `sm120_mqa_blockscale_check.py`), diff ~1e-6.
- **Still not implemented on sm120** (rejected at the host): mega MoE with fused shared experts, mega gate and mega mHC. None needs missing hardware (section 9.26: gate and mHC are a GEMM plus elementwise work), and TokenSpeed calls none of them (it runs V4's shared expert separately). The plain `tf32_hc_prenorm_gemm` mHC kernel is implemented. MN-major FP8 GEMM operands (9.10), k-grouped FP8 GEMMs (9.11), the MN-major `fp8_einsum` expressions (9.12), paged MQA head dim 32/64 (9.13) and the batched `fp8_einsum` launch (9.14) are now done. See sections 8 and 9.9 to 9.14.
- **Implemented on sm120 but unused by TokenSpeed on this box:** the m-grouped contiguous (incl. psum) and masked FP8 GEMMs have real `arch_major == 12` branches (`gemm.hpp`) and pass `test_fp8_fp4.py`. TokenSpeed only reaches them through DeepEP MoE, which needs NVLink/RDMA (this box is PCIe-only), so nothing calls them here.
- **Tests (GPU 6 and 7, 2026-11-11, task 39, section 9.43, supersedes the task 33 run in 9.37):** the whole SM120 surface passes with `DG_JIT_CHECK_NO_SPILLS=1` and fresh caches on the DeepGEMM build of the local head: `test_fp8_fp4` (with `test_psum_gaps_do_not_reach_valid_rows_sm120`), `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention`, `test_mqa_logits_launch`, `test_mega_moe_sm120` exit 0; `sm120_mqa_check.py` (46 cases), `sm120_blockscale_check.py --mode all --grouped` (170), `sm120_paged_mqa_blockscale_check.py` (51), `sm120_sparse_paged_check.py` (28) and `sm120_sparse_mqa_check.py` (40) end in ALL OK / OK; the 2-rank `test_mega_moe_sm120_ep.py` is bitwise equal to one rank in 7 splits; `test_lazy_init --num-processes 2` 3 tests OK. TokenSpeed kernel suite 8011 passed, 2764 skipped, 0 failed (`PATH` with `/usr/local/cuda/bin`); runtime suite 9007 passed, 0 failed; `nvidia/ops/communication` with two GPUs visible 155 passed; the two-rank token-layout test passes; the four end-to-end models answer correctly; `pre-commit run --all-files` is clean on a clean tree. Earlier run (task 33, 2026-11-10, section 9.37): kernel suite 7952 passed, 2764 skipped; runtime suite 9007 passed. Earlier run (task 26, section 9.32): kernel suite 7898 passed, 2770 skipped; runtime suite 9003 passed. Earlier run (task 20, 2026-11-09, section 9.25): kernel suite 7770 passed, 2770 skipped; runtime suite 8997 passed; `sm120_mqa_check.py` 46 cases, `sm120_blockscale_check.py --mode all --grouped` 170 cases and `sm120_paged_mqa_blockscale_check.py` 51 cases all OK; `test_lazy_init` on GPUs 6 and 7 (3 tests) OK; the DeepGEMM source at the pinned commit builds like `install.sh` and passes `test_deepgemm_sm120_gating.py`.
- **Performance vs the TokenSpeed sm120 fallbacks:**
  - Dense decode is 1.2-4x faster at TP8 shapes; dense prefill is on par with FlashInfer.
  - The DSA indexer is 3-19x faster for decode and 8-52x faster for prefill.
  - End to end on a block-FP8 27B model: TPOT is 1.18-1.55x better.
  - **FP4 vs the best FP8 kernel** (DeepSeek-V3 TP8 dense linears, graph + cold weights): 1.4-2.3x (geomean 1.8x) at every shape except K = 256 `shared down` (0.94-1.33x), and 1.5-11.4x (geomean 3.1x) over the best Triton/FlashInfer block-FP8 kernel (smallest on `shared down` large M, largest on N = 512 `shared gate_up` decode); remeasured on 72 points with the shipped build at `2e77a46` (`sm120_bench_blockscale.py --tp 8 --baselines`, status log 2026-11-09). The older ranges (1.6-2.0x and 1.3-2.0x vs the best FP8 kernel, 1.1-10.2x and 1.07-9.1x vs Triton/FlashInfer) came from the 2026-11-04 runs before the refits and had no stored table; they are superseded. **FP4 vs FlashInfer MXFP4** (`cudnn` and `b12x`, the backends that run MXFP4 on sm120): DeepGEMM wins 62 of 63 points, 1.40x geomean, worst point 0.979x (early PDL trigger `76dc545`, section 9.15; it was 59 of 63 and 1.34x after the cost-model refit). The one tie left is dense gate_up M=256.
- **DSA / V4.1 indexer top-k (task 13.1, 2026-11-07, section 9.16):** the V3.2 DeepGEMM leaves were already offered on SM120 (the "blocked" finding of 9.7 applied to the V4.1 native indexer only). The in-tree CuTe DSL DeepSelect now runs in 93 KB (compact profile, 256 threads, two-stage ring, 3072-pair candidate buffer filled in two warp groups; same selected entries), clusters of 2 and 4 CTAs work on SM120. V3.2 prefill selects with it (1.3-1.9x over the Triton radix top-k, 8-81x over the Triton leaf in total); V3.2 decode uses it from 96 rows of 32K+ or 48 rows of 64K+ (up to 2.7x on the top-k). V4.1 scores FP8 `index_v4` rows with the SM120 MQA kernels and selects with it (2.3-7.2x decode, 1.4-5.1x chunked prefill over the Triton leaf), except Reindex passes over long histories. Since task 16 the 68-byte rows SM120 stores are native too (next bullet).
- **DSA sparse attention (task 13.2, 2026-11-07, section 9.17):** FlashMLA and trtllm-gen DSA are capped off SM120, and the selected core was `triton_dsa_*`, a CUDA-core kernel with one program per (token, head) that re-gathers every row for every head. New `triton_dsa_mma_decode` / `_prefill` (TokenSpeed `ops/attention/dsa/_triton/sparse_mma.py`) tile the heads of a token over one gather and use `mma.sync` (E4M3 `m16n8k32` for QK, FP16/BF16 for PV, FP32 online softmax, split columns merged by a second kernel for few tokens). 27-146x faster across 16-128 heads, decode batch 1-128 and prefill chunks 128-8192 (E4M3 128 heads: decode 13.6/46/136/538 us, prefill 0.48/1.8/7.4/29.5 ms), selected on SM120 for the dense cache. It reached 136-158 TFLOP/s against a combined roofline of about 540 TFLOP/s (E4M3 QK at 779, 16-bit PV at 404 TFLOP/s measured with cuBLAS), so a hand-written kernel could still gain up to 2-3x; not attempted. Does not use DeepGEMM. **Task 15 made it batch-invariant** (chunks by selection width, ordered fold, declares `batch_invariant`, `cuda_graph` and `slot_order`): 1.21x slower on geomean (1.86x worst, from a second accumulator that spills 34-52 registers), still 22-114x over `triton_dsa_*` (section 9.20).
- **DSA end to end (task 13.3, 2026-11-07, section 9.18):** a 4-layer DeepSeek-V3.2 with real dims and random weights runs through the real model class and `Engine` on SM120 with the new leaves selected. Prefill is 82-95x and TPOT 2.0-14.4x better than the forced `triton_dsa_*` path (TPOT 2.63 ms at batch 1 for any context up to 32K). No SM120 failure, no code change.
- **SM100-only leaves capped off SM120 (task 14, 2026-11-07, section 9.19):** each now has an SM120 tensor-core Triton leaf (TokenSpeed, no DeepGEMM code). `fast_hadamard_transform` (GLM-5.3-Flash indexer query): `triton_hadamard_transform_128_mma`, at the copy roofline from 16K rows, 55-117x over the portable leaf; the upstream CUDA source builds for sm_120 but is 1.6x slower, so the wheel stays capped. `tokenspeed_mla` (MLA decode): `triton_mla_mma_decode_with_kvcache`, 14-131x over the portable MLA decode, 0.9-1.5x of the memory roofline at 16 heads and 100-142 TFLOP/s at 128; K3 no longer defaults to the `tokenspeed_mla` backend on SM120. FlashMLA DSV4/V4.1: `triton_dsv4_mma_prefill`, 4-9x over `triton_dsv4_prefill`, batch-invariant. flashinfer trtllm-gen DSA: the zero-RoPE GLM-5.3-Flash layout was the one uncovered case; the 13.2 cores now take it (29-134x). The caps stay (wgmma/tcgen05, no sm_120 cubins).
- **SM120 attention headroom (task 17, 2026-11-08, sections 9.22-9.23):** all three SM120 attention kernels now have CuTe DSL versions in TokenSpeed, selected over the Triton ones: DSA sparse attention (`cute_dsl_dsa_mma_decode` / `_prefill`, `dsa/_cute_dsl/sparse_mma.py`: `cp.async` two-stage ring, `ldmatrix(.trans)`, `mma.sync`, one accumulator per warp, the bitwise batch-invariant chunk fold) at 1.91x (E4M3) / 1.16x (BF16) geomean over the Triton core on 64 buckets, 342 TFLOP/s E4M3 prefill (63% of the floor); V4 selected attention (`cute_dsl_dsv4_mma_prefill`, the DSA BF16 path with a 512-wide query and the sink as chunk 0's initial state) at 1.13x (0.89-1.63x, 117 points) for more than 16 heads, same batch invariance; MLA decode (`cute_dsl_mla_mma_decode_with_kvcache`, contiguous page gather, causal/window masks, `q_len` 1-8) at 1.46x (0.97-2.31x, 128 points) and 1.84x on 4/8-row blocks, BF16 213 and E4M3 380 TFLOP/s at 128 heads. The dots3-note DSA leaf no longer pins `triton`: it requires the new `strided_kv_cache` feature, so SM120 reaches the CuTe cores. Open: BF16 DSA/V4 above 42-48% of the floor, V4 at 16 heads (4-warp CTAs lose), decode of one DSA request (8 chunks).
- **Task 18 (2026-11-09, section 9.24):** the smaller follow-ups. TokenSpeed: V3.2 DSA prefill leaf is host-light (`dsa_prefill_prepare`, one launch; DeepSelect `index_base`), 118-124 us per call instead of 360 and ahead of the Triton leaf at every chunk size (1.35x at 128 tokens); sharded `dsa_index_candidates` selects with DeepSelect (1.25-4.7x over the `torch.topk` leaf); V4.1 decode reads the packed FP8/FP4 rows in the Triton dense kernel (`triton_dsv41_mma_selected_attention`, 1.66x geomean, bit-identical), `dsv4_decode` only for 16 heads from 32 tokens; the whisper graph-capture flake was the cyclic GC freeing a dead `CUDAGraph` inside a capture (`gc_paused_for_capture`). DeepGEMM: batched BF16 `einsum` (above).
- **V4.1 serving on the native indexer (task 16, 2026-11-08, section 9.21):** DeepGEMM's SM120 paged MQA kernel scores MXFP4/MXFP8 rows (`6b2df9e`), so TokenSpeed's `deep_gemm_sm120_dsv41_index_topk` scores the 68-byte `v41` rows the runtime stores on SM120, with no cache-format change: decode Full 5.6-16x, candidate-source Full 7.7-33x, Reindex 2.2-22x, chunked prefill 6.8-88x over `triton_dsv41_index_topk`; selected sets match an exact reference. The serving path (in-process `Engine`, V4.1 topology with dummy weights) now selects the native leaf; prefill 1.11-1.19x, TPOT 1.02-1.06x end to end, same greedy tokens.
- **Task 15 (2026-11-08, section 9.20):** DSA sparse attention is batch-invariant (`36211b436`); the intermittent `test_deepseek_v41_model[eager]` failure did not reproduce in 6 complete runs of the top-level directory, 150+ isolated runs, the pre-fix tree or the baseline (recorded, no known cause); `test_control_server.py` no longer uses fixed ports (`990e3e6cd`); final verification and a per-op table (`deepgemm-sm120-design.md` 7.8).
- **Commits:** everything from tasks 1-39 is pushed to `origin/nctu6` in both repos (tasks 1-26 in task 27, the task 28 lint commits, tasks 29-33 after task 33), DeepGEMM first, then TokenSpeed, whose `DeepGEMM` submodule pin points at a commit on the DeepGEMM remote. Tasks 34-39 were pushed after task 39 (DeepGEMM first, then TokenSpeed): DeepGEMM `aa1ecfe` (MegaMoE fusion), `b4eaebf` (sparse MX kernel), `f628449` (kernel lookup), `1ee8237`, `6df2878`, `211a7e9` (scripts), `0d89c76` (tests) and the docs commits; TokenSpeed `b5f761118`, `2098a0c41`, `436833133` and the pin bumps. The code head of DeepGEMM is `0d89c76` (docs-only commits follow it; the kernel head is `b4eaebf`); TokenSpeed's pin follows the DeepGEMM head. Use `git log origin/nctu6` for the exact heads. Key commits: DeepGEMM `5cf4862` (W4A8, one-rank mega MoE), `1a5c1cf` (EP across ranks), `fefd9de` (MegaMoE scratch), `26444a3` (deterministic = batch invariant), `343d194` (one-rank scratch); TokenSpeed `695c15770` (SM120 MegaMoE leaf), `8afd973bc` (Triton MXFP4 fix), `e4fc05813` (MLA RoPE fix), `6d6702bae` (V4.1 expert subset), `0b638da63` (MXFP8 leaf), `70a02f53d` (V4.1 prefill core route), `3de153200` (DSA launch planner). Sections 9.x written before the push say "local, not pushed"; that was the state when each was written. `WEEKLY_REPORT.md` stays local (git-excluded, never committed).
- **Install:** TokenSpeed carries DeepGEMM as a submodule (`DeepGEMM/`, branch `nctu6`). `scripts/install.sh` and `docker/Dockerfile.nvidia.unieai` build it from source and replace the published `tokenspeed-deepgemm` wheel. Both are tested on aws-raichu. See the "Submodule + install" rows in section 9.
- **Mega MoE in TokenSpeed (task 21):** `deep_gemm_sm120_mxfp4_mega_moe_apply` is registered on SM120 for EP 2 (opt-in, `--moe-backend mega_moe` with attention DP); at the same served tokens it ties Marlin W4A16 plus its collective from 1 to 1024 tokens per rank (0.95-1.05x) and wins 1.28-1.39x from 2048 (section 9.26). The Triton MXFP4 MoE illegal memory access at V4.1 shapes was a 32-bit expert offset overflow, fixed in TokenSpeed `8afd973bc`.
- **Task 23 (2026-11-10, section 9.28):** the dense gate_up M=256 MXFP4 tie (DeepGEMM 19.2 us vs b12x 19.5-19.6 us, 1.02x in an isolated run, 0.98x in the sweep) is not won by thread-block cluster tile sharing. B-slice sharing over DSMEM across 2 and 4 CTAs works and is bitwise equal, but is 2.3-7.3x slower (45-146 us vs 20 us): distributed shared memory on SM120 writes at 5 GB/s per CTA, reads at about 33 GB/s, and adds 95 ns per remote load and 220 ns per remote mbarrier arrive to a 3-stage pipeline. The ceiling with free sharing is 1.13x. No kernel change; probe and patch are in `scripts/`.
- **Task 24 (2026-11-10, section 9.29):** the real-V4.1 serving profile showed that both TokenSpeed paths spent most decode time in the dense 1x32 UE8M0 FP8 projections on a Triton kernel. TokenSpeed's new `deepgemm_sm120_mm_mxfp8` leaf runs them on the block-scaled 1D1D kernel (2.2-11.7x), batch invariant through deterministic mode, which now skips stream-K and the BLOCK_N=32 K-warp split on SM120 (`26444a3`, `get_deterministic_algorithms()`). MegaMoE's row scratch is one persistent per-device buffer the KV planner counts (`fefd9de`). With TokenSpeed's DP error-masking fix and the CuTe DSL V4 prefill core now reachable from V4.1 prefill: Marlin TP2 14.5-14.8K tok/s prefill and 10.4-10.7 ms TPOT (was 10.2-10.7K, 23-25 ms); MegaMoE DP2/EP2 18.1-18.6K tok/s (1.25x over Marlin) and 13.8-14.0 ms TPOT (was 11.8-13.3K, 48-50 ms; 1.31x behind Marlin: +2.6 ms per step is DP attention reading all dense weights per GPU, +2.3 ms is MegaMoE glue around GEMMs that tie Marlin). Util 0.93 works with 128 experts.
- **Task 25 (2026-11-10, section 9.31):** the intermittent `test_deepseek_v41_model[eager]` failure is closed as not reproducible after a bounded search: sanitizers clean (memcheck, racecheck, initcheck, synccheck, with and without the caching allocator), poisoned allocations and stream delays change no bit, about 1400 eager runs (idle, GEMM-loaded, beside the kernel suite) and 46 fresh whole-file runs pass. No code change. The original failure text is lost, so a recurrence needs its output kept.
- **Task 29 (2026-11-10, section 9.33):** MegaMoE now serves attention TP: with replicated tokens the SM120 leaf runs the one-rank kernel over a rank's shard of the experts (`moe_plan(ep_token_layout="replicated")`; an expert range or an MoE-TP slice) and the TP all-reduce sums the partials, with no exchange between the ranks. On real V4.1 (128 experts, GPUs 6+7) attention TP2 + MoE TP2 prefills 1.12x faster than Marlin TP2 (17.0K vs 15.1K tok/s) and ties it in TPOT (10.66 vs 10.69 ms; DP2/EP2 was 14.02 ms); an expert-range shard is 11.22 ms because the all-reduce waits for the rank with more routed experts. `auto` keeps Marlin. DeepGEMM `343d194` fixes the one-rank row scratch, which grew at every warmup size and kept its predecessors.
- **Task 34 (2026-11-11, section 9.38):** the SM120 MegaMoE call fuses SwiGLU + FP8 quantization into the L1 GEMM epilogue and the gather into dispatch (DeepGEMM `aa1ecfe`), bitwise equal to the unfused kernels (one rank, EP2): 3.7-13.4 us per layer at 1-16 tokens. On real V4.1 (128 experts, GPUs 6+7) attention TP2 + MoE TP2 prefills 1.13x faster than Marlin TP2 and has a TPOT of 10.49 ms against 10.57-10.60 ms (1.0%, the size of the run-to-run spread). `auto` keeps Marlin; opt-in `--moe-backend mega_moe` is faster in prefill and a hair better in TPOT.
- **Task 30 (2026-11-10, section 9.34):** the CuTe DSA core launches a stream-K schedule: whole tokens first, then the remaining tokens split 2/4/8 ways, so the short split CTAs fill the SMs the last wave leaves idle (TokenSpeed `e51c66452`). Over 140 decode shapes x both dtypes it is 1.032x geomean (up to 1.38x, worst 0.97x), over the 64 buckets 1.017x (up to 1.30x), rows bitwise equal; one-request decode 20.2 us BF16 / 13.3 us E4M3 (was 20.3 / 13.1). A persistent prefetching CTA is bounded at 3.4-4.9% and has no registers or shared memory for it; 16 chunks win 1.31x at 1-4 tokens and lose 0.91-0.95x at 13-64 even with an oracle plan; on-chip chunk states are not bit-exact. Closed.
- **Task 31 (2026-11-10, section 9.35):** V4 attention at 16 heads is closed on Triton again, with a ceiling and not a guess: at 16 heads a token's 640 rows are gathered for 16 heads only, the Triton leaf reads them at the L2 -> SM rate (1.34 GB in 340.7 us at 2048 tokens, 3.94 TB/s) and the CuTe core's gather alone takes 325 us on 8 warps; with perfect row overlap, two or four tokens per CTA would reach 0.92x / 0.99x of Triton at 640 rows and 0.83x / 0.89x at 128 rows (2048 tokens), and a token's tiles must stay its own 32 columns for bitwise batch invariance, so only gathers could be shared. The 8-warp CTA with the 4-warp bits (two warps per channel quarter split the rows of the score tile; PV splits the channels 8 ways; rows bitwise equal to the 4-warp CTA) is 1.13x of Triton at 1-16 tokens and 0.73-1.03x from 64 (mixed by token count), so the leaf stays by head count (`num_q_heads_min` 17). It ships for more than 16 heads together with a V4 wave-model planner (`plan_launch(tokens, heads, num_sms, width)`: head tile, split and a stream-K tail; TokenSpeed `d55bb24bd`): 1.095x geomean over 165 points of 17-128 heads x widths 128 / 640 / 1152 x 1-500 tokens, up to 1.66x, worst 0.89x; rows bitwise equal for every plan.
- **Task 47 (2026-11-12, section 9.44, report [`deepgemm_benchmark.md`](deepgemm_benchmark.md)):** two TokenSpeed images served on GPUs 6 and 7, `e27cee86` (before, no DeepGEMM SM120) and `ef057889` (after, DeepGEMM `976f212`), six models, same flags, RAG (500 prompts, 256 output tokens, C = 1 .. 512) and agentX (AIPerf full mode, 900 s, `submission_valid=true`, C = 1, 16, 64). Qwen3.8-27B-FP8: output tok/s 1.52x at C=1, 1.28x at 16, 1.11x at 64, 1.04x at 256-512 (TPOT 1.53x at C=1, TTFT 1.00x); agentX 1.14x / 1.10x / 1.01x. BF16 controls Qwen3.6-35B-A3B and gemma-4-31B-it 1.00-1.03x. Qwen3.8-27B-NVFP4, Qwen3.5-122B-A10B-NVFP4 (TP2) and the V4.1 128-expert subset (TP2) do not start on the before image and run on the after image; V4.1 Mega MoE is 1.01-1.17x over Marlin in RAG and ties in agentX. Findings: a scheduler abort with KV events on (both images, C >= 128; workaround `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` in all results), gemma-4 at TP1 cannot take 256K agentX (`max_req_input_len` 199,614), the default gateway parser fails gemma-4 chat, and decode batches above the 160-graph default show a step-time cliff. One run per point on a shared host.
- **Task 48 (2026-11-13, section 9.45, docs only):** the benchmark report is now [`deepgemm_benchmark.md`](deepgemm_benchmark.md) (renamed with `git mv`, one document): a summary table per model x dataset x C (output tok/s, TTFT p50, TPOT p50, before / after / ratio), testing conditions (hardware, driver 595.99.02 / CUDA 13.2, image IDs and the commits inside them checked by file hashes, the exact server and client commands, probes, warmup, run order by server launch), per-run tables with every AIPerf metric, the failures with error lines and log paths, the default-config crash runs, the excluded smoke runs and caveats. All 204 summary values were rechecked against the JSON. Correction: with KV events on, the before image also died at C=128 in its first session (342 of 500 requests failed; the run kept a `DONE` marker), not at C=256.
- **Real weights (task 22, section 9.27):** the 157 GB V4 checkpoint cannot load in this TokenSpeed tree (upstream removed `DeepseekV4ForCausalLM` in `0c1bc3ed3`, #2310), so the MegaMoE end-to-end check ran on the real 476 GB V4.1 checkpoint with all 40 layers, real Engram, embeddings, norm and head, and its first 128 of 384 routed experts per layer (TokenSpeed `6d6702bae`, `num_checkpoint_experts`; a layer subset is impossible because the FlatKV packing is pinned to 40 layers). DP2/EP2 serving selects `deep_gemm_sm120_mxfp4_mega_moe_apply`. Against a float32 reference built from the same weights with the official `inference/model.py`, layers 0 and 2 have the same single-layer error on all three paths (2.76-2.81% and 3.70-3.90% relative L2: MegaMoE DP2/EP2, Marlin TP2, forced portable Triton TP2). The paths differ from each other by 0.7% at layer 0; the gap grows through 40 layers to about 50% (logits KL 0.56-0.64, top-1 62-69%), equally for every pair, so the 1/3-expert model amplifies rounding and the greedy texts cannot be compared. Speed at 2 x 2K tokens: Marlin TP2 10.2-10.5K tok/s prefill, 23.5 ms TPOT; MegaMoE DP2/EP2 3.8-3.95K tok/s, 34.2-34.5 ms; portable 3.5K tok/s, 31-33 ms (not profiled; the leaf benchmark of 9.26 ties or wins, so the gap is outside the MoE leaf). MegaMoE allocates its worst-case row scratch (about 2.6 GB at 8192 tokens per rank) on every call, outside the KV planner's budget, so `--gpu-memory-utilization 0.93` ran out of memory where 0.86 works. Real MLA: Moonlight-16B-A3B (32 GB, downloaded to scratch) answers correctly on SM120 after the transformers-5 RoPE fix `e4fc05813`. Its greedy decode matches the forced portable MLA decode for 17-48 tokens, splitting only at near-ties, and TPOT is 2.3-3.2x better at 3K context x 8. No DSA, GLM-5.x, dots3-note or V2-Lite checkpoint is on the box.
- **Open:** (1) One FP4 tie with FlashInfer b12x: dense gate_up M=256 (0.98x in the sweep, 1.02x isolated), bound by L2 -> SM traffic (section 9.15). Cluster tile sharing over DSMEM is 2.3-7.3x slower (section 9.28) and the ceiling with free sharing is 1.13x; not shipped. (2) MegaMoE: `auto` keeps Marlin (sections 9.33, 9.38). Under attention TP2 + MoE TP2 on real V4.1 it prefills 1.13x faster and its TPOT is 10.49 ms against Marlin's 10.57-10.60 (1.0%, the run-to-run spread). Left: the combine and staging (about 0.3 ms per step; the combine in the L2 epilogue needs atomics, which break bitwise equality, or the next layer's staging + dispatch, 3-6 us per layer each) and the `auto` gates (numerics envelope, width 1 and 2 only, batch invariance, scratch in the KV plan). Attention DP stays 1.31x behind Marlin TP2 in TPOT (+2.6 ms of dense-weight reads per step, structural). The EP2 / DP2 paths use the fused kernels but were not timed on the real model after task 34. More than 2 ranks, fused shared experts, mega gate and mega mHC have no TokenSpeed caller. (3) V4.1 indexer: the multi-wave sparse MX kernel is done (section 9.39). Left: SM100's sparse metadata kernel (shared) is a quarter of the sparse leaf at 1024 queries; FP8 `index_v4` rows have no pool-only scorer (the runtime does not store them on SM120). MXFP4 paged logits at 1.07 vs 1.23 TB/s are closed as page geometry. (4) Attention: BF16 DSA / V4 stay at 42-48% of the tensor-pipe floor (section 9.30); V4 at 16 heads stays on Triton (section 9.35); one-request DSA decode stays at 20 us BF16 / 13 us E4M3 (sections 9.30, 9.34); persistent CTAs, 16 chunks and on-chip chunk states are closed. The packed-row CuTe DSL V4 / V4.1 decode is done (section 9.40); left: its 32-head CTA spills 120-168 B, V4.1 window-only layers at 32-128 heads tie the Triton direct reader at 1-4 and 256+ tokens (0.94-1.06x), and the overlap of real V4.1 selections beyond 2048 positions is unmeasured. (5) Host path of a 128-token V3.2 DSA chunk: done for `fp8_mqa_logits` and DeepSelect (section 9.41, leaf 116 -> 87 us). Left: the TVM-FFI call and `values` allocation of DeepSelect (5.5 of its 9 us), the `dsa_prefill_prepare` launcher (37 us), `nan_to_num_` (6 us), and the other launchers that call `jit->compile` per launch (about 3 us each). (6) Full-model real-weight runs: V3.2 TP8 (no checkpoint on the box, about 690 GB FP8, all 8 GPUs) and V4.1 with all 384 experts (4 GPUs). The V4 checkpoint cannot load (no TokenSpeed V4 model class); no V3.2, GLM-5.x, dots3-note or V2-Lite checkpoint is on the box (section 9.27). (7) Sanitizers and flakes: the three heavy sanitizer runs are done (section 9.42), but the tools see only the shapes the tests launch, the kernel filter leaves torch kernels unchecked in memcheck, synccheck and racecheck, racecheck models shared memory only (the 21 largest cases ran as one launch each) and the repeat loops were cut to 2. The V4.1 eager flake is closed as not reproducible (section 9.31); a recurrence needs its pytest output kept. (8) Verification limits: `test_projection_tp` tp4 / tp8 need 4 and 8 NVLink GPUs; scheduler clang-tidy and the C++ build are not run (no ninja on the box); TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90 / sm100 / sm103 / sm107, so upstreaming needs that scope to change; a wheel release is optional (the source build replaces the pin). (9) Serving benchmark (section 9.44): the scheduler KV-event abort at C >= 128 in both images (`scheduler.cpp:385/386`; the servicer enables KV events by default, `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` avoids it); gemma-4-31B-it at TP1 cannot serve the 256K agentX traces (`max_req_input_len` 199,614); the default gateway parser (`hf` tool-call parser) fails gemma-4 chat with HTTP 400 (`--tool-call-parser gemma4` avoids it); decode batches above the default 160 CUDA-graph size (V4.1: 49 ms a step at batch 116, 130-140 ms at 244-500; fallback path not verified, larger capture size not tried). (10) Push state: tasks 1-39 are pushed to `origin/nctu6` in both repos (DeepGEMM first, then TokenSpeed). **Closed since this list was first written:** the UE8M0 FP8 regression of the FP4-only cost-model refit (fixed by the joint refit `eff4cea`, status log 2026-11-05: UE8M0 FP8 is at parity with the FP32-scale kernel again); NVFP4 (done, `eb61610`, 2026-11-05); the NVIDIA MXFP4 `mm` leaf (TokenSpeed `deepgemm_sm120_mxfp4_linear`, with the PDL quantizer and weight scales packed at load: 1.23x geomean, 60 of 63 wins against FlashInfer quantize + `mm_fp4`; design 7.8 and 8).

## 0. Environments

| Role | Location | Notes |
|------|----------|-------|
| Code (edit here) | local `/Users/nctu6/Downloads/ascend/infra/DeepGEMM`, branch `nctu6` | Submodules are **not** checked out locally (empty `third-party/`) |
| Test (run here) | `ssh aws-raichu`, `/workspace/nctu6/infra/DeepGEMM`, branch `nctu6` | `/workspace` resolves to `/unieai`. **GPU 7 only**: always `CUDA_VISIBLE_DEVICES=7` |
| Runtime | `/workspace/nctu6/infra/DeepGEMM/.venv` | Python 3.15.0. Never install outside this venv |

**Scratch files.** All temporary files go in `DeepGEMM/tmp/` or
`tokenspeed/tmp/`, both locally and on aws-raichu (`/workspace/nctu6/infra/DeepGEMM/tmp/`,
`/workspace/nctu6/infra/tokenspeed/tmp/`), never in `/tmp`, so no write
permission prompts appear. The folders are git-excluded. Remote helpers:
`tokenspeed/tmp/ssh_cm/rssh` and `tx` (persistent SSH master plus tmux session
`sm120`). The full rule is in `tokenspeed/AGENTS.local.md`.

Both checkouts were at HEAD `5081f85` (DeepGEMM 2.8.2) when this was written.

Facts about the remote machine (verified):
- 8x RTX PRO 6000 Blackwell Server Edition, compute capability 12.0, 96 GB each,
  driver 595.99.02.
- CUDA 13.2 (`/usr/local/cuda` -> 13.2, nvcc V13.2.86). 13.0 is also installed.
- **No NVLink.** `nvidia-smi topo -m`: GPU pairs share a PCIe switch (PIX), pairs
  in the same NUMA node are NODE, and the two NUMA halves are SYS. GPU 7 pairs
  with GPU 6 (PIX).
- Submodules are now initialized on the remote: cutlass `f3fde58` (CUTLASS
  4.2.1) and deep_jit `2efdab4`.
- The venv has no torch yet. `torch-2.14.1+cu130` cp315 wheel exists on
  `download.pytorch.org/whl/cu130` (verified by download). `uv` is at
  `/snap/bin/uv`.

Not yet verified (check in Phase 0): SM count (spec says 188), max opt-in smem
per block (expected 101376 B = 99 KB, per SM 100 KB), max cluster size, and
FP8 `mma.sync` throughput with FP32 vs FP16 accumulate.

## 1. What TokenSpeed calls from DeepGEMM (the actual target surface)

All calls go through `tokenspeed-kernel`. The runtime `python/` never imports
`deep_gemm` directly. The pinned package is
`tokenspeed-deepgemm==2.8.0.post20260924`
(`tokenspeed-kernel/python/requirements/cuda-thirdparty.txt:2`).

| # | DeepGEMM API | TokenSpeed caller | Model use | Scale format | sm120 today |
|---|---|---|---|---|---|
| A | `fp8_gemm_nt` + `transform_sf_into_required_layout` | `thirdparty/deepgemm.py:26-51`, `ops/gemm/deepgemm.py` (`deepgemm_mm_fp8_blockscale`, min=max sm90) | V3/R1/V3.2 dense block-FP8 linears (attention projections, dense MLP, shared experts) | FP32, A 1x128, B 128x128, `disable_ue8m0_cast=True` | **Native.** `deepgemm_sm120_mm_fp8_blockscale` (12.0-12.9, SPECIALIZED) is selected over FlashInfer and Triton: decode 1.2-4x, prefill on par; TPOT 1.18-1.55x on a 27B block-FP8 model (status log 2026-11-03) |
| B | `m_grouped_fp8_gemm_nt_contiguous` / `_masked`, `get_mn_major_tma_aligned_tensor`, `transform_sf_into_required_layout` | `ops/moe/deep_ep.py` (`deep_gemm_deepep_fp8_moe_apply`, min 9.0, **no max**) | MoE experts with DeepEP (EP) | UE8M0 packed (1,1,128) when `deep_gemm_requires_ue8m0()` (SM100 family only since the Phase 5 change; sm120 uses FP32 scales, and a packed-UE8M0 SFA with an FP32 SFB also runs, `a765e73`) | **Kernels done** (real `arch_major == 12` branches, `test_fp8_fp4.py` passes), the leaf is offered when the build supports SM120. DeepEP itself needs NVLink or RDMA, so nothing calls it on this PCIe-only box |
| C | `get_paged_mqa_logits_metadata`, `fp8_paged_mqa_logits`, `fp8_mqa_logits` | `ops/attention/dsa/deep_gemm.py` (4 kernels, min 9.0, **no max**), `ops/attention/kpool/deep_gemm.py` | V3.2 DSA indexer (decode, prefill, index candidates), KPool prefill | q FP8 E4M3 with FP32 per-token-128 scale folded into weights; K cache FP8 with an FP32 per-row scale | **Native.** The SM120 FP8 MQA kernels (head dim 128/64/32, `fp8_mqa_logits` and `fp8_paged_mqa_logits`) back the V3.2 indexer leaves: decode 3-19x, prefill 8-52x over the Triton leaves, DeepSelect fits in 93 KB, section 9.16 |
| D | `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` | `ops/attention/dsv4/deep_gemm.py`, `dsv41/deep_gemm.py` | V4 / V4.1 indexer | int32 packed UE8M0 | **Native.** Block-scaled paged and contiguous MQA kernels score the 68-byte MXFP4 V4.1 rows (task 16, `6b2df9e`, `deep_gemm_sm120_dsv41_index_topk`); FP8 `index_v4` rows go through C's kernels (`deep_gemm_sm120_dsv41_*`), sections 9.16 and 9.21 |
| E | `tf32_hc_prenorm_gemm` | `ops/residual/deep_gemm.py` (max 10.3) | V4 / V4.1 mHC | FP32 | **Native.** `sm120_tf32_hc_prenorm_gemm` (TF32 `m16n8k8`), the `deep_gemm_mhc_pre` / `_mixes` leaves are enabled on SM120: 1.5-15x over Triton (status log 2026-11-05, design 8) |
| F | `fp8_fp4_mega_moe` | `ops/moe/deep_gemm.py` (sm100 only) | V4 MXFP4 experts | UE8M0 | DeepGEMM runs it on one SM120 rank (task 21, 9.26); the TokenSpeed leaf stays capped at 10.9: its MegaMoE path needs EP > 1 and V4/V4.1 experts do not fit one GPU (DeepGEMM `5cf4862`) |

Not used by TokenSpeed: `einsum` / `fp8_einsum` / `bmk_bnk_mn`, `bf16_gemm*`,
k-grouped GEMMs, sparse MQA logits, `mega_gate`, `mega_mhc`. The MLA absorb
bmm goes through the kernel `bmm` op (cuBLAS), not DeepGEMM.

TokenSpeed platform flags on sm120 (`platform.py:170-219`): `is_hopper=False`,
`is_blackwell=False` (major == 10 only), `is_hopper_plus=True`,
`is_blackwell_plus=True`. That is why C and B were imported and registered on
sm120 before DeepGEMM could serve them (written at the start of the port; the
table above shows the state at 2026-11-09).

Scope note: TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90/sm100/sm103/
sm107 only. sm120 is a fork-side effort (see `docker/Dockerfile.nvidia.unieai`,
which already adds `12.0a`). Upstreaming the TokenSpeed side needs that scope
to change.

## 2. Why DeepGEMM doesn't run on sm120 today

### 2.1 Host dispatch: equality checks on `arch_major`

Every dispatcher checks `arch_major == 9` or `== 10`. 12 falls through to
`DG_HOST_UNREACHABLE`, or, worse, silently into sm90 semantics. Paths below are
under `csrc/`.

- `utils/layout.hpp:86-97` `get_default_recipe`: `UNREACHABLE` for 12. This
  breaks every FP8 GEMM called without an explicit recipe.
- `utils/layout.hpp:43` `fp8_fp4_requires_k_major` (== 9 only) returns false
  on 12, so MN-major FP8 would be silently accepted.
- `utils/layout.hpp:69,78`: FP4 requires arch 10.
- `apis/gemm.hpp`:
  - `fp8_fp4_gemm_nt` 117/127 -> 131.
  - m-grouped contiguous 236/242 -> 248, masked 306/311 -> 316.
  - k-grouped 362/421/480.
  - bf16 526/530, which tries cuBLASLt first at 519-521.
  - m-grouped bf16 610-720.
- `apis/layout.hpp:38-57` `transform_sf_into_required_layout`: FP32
  passthrough only when `arch == 9 or disable_ue8m0_cast`. The UE8M0 cast is
  arch 10 only. Otherwise `UNREACHABLE`.
- `jit_kernels/impls/smxx_layout.hpp:224`: the packed-UE8M0 helper asserts
  arch == 10.
- `apis/attention.hpp`:
  - 32/43/80/111/192/233/258: `arch == 10` ternaries, so 12 silently gets
    sm90 dtype/block choices.
  - 120/124, 153/163, 201-202, 267/272 dispatch.
  - 313/344/380/414: sparse paths assert == 10.
- `jit_kernels/impls/sm90_fp8_mqa_logits.hpp:27,161` assert == 9.
- `apis/einsum.hpp`, `hyperconnection.hpp:48`, `mega_*.hpp` have the same
  pattern.
- `jit_kernels/heuristics/sm90.hpp:14` and `sm100.hpp:16` hard-code
  `smem_capacity = 232448`. sm120 is about 99 KB per block.

### 2.2 Device code: every kernel uses Hopper/datacenter-Blackwell-only hardware

| Feature | sm90 kernels | sm100 kernels | sm120 |
|---|---|---|---|
| `wgmma.mma_async` | yes (all) | - | **no** |
| `tcgen05` / TMEM / UMMA / 2-CTA | - | yes (all) | **no** |
| `setmaxnreg` | yes | yes | yes on `sm_120f`/`sm_120a` (verified) |
| TMA load/store, mbarrier, tensormap | yes | yes | yes |
| TMA multicast / clusters | 1x2 / 2x1 | yes | unverified; start with cluster 1 |
| `stmatrix` / `ldmatrix` | yes | yes | yes (verified, all targets) |
| `mma.sync kind::f8f6f4 m16n8k32` (E4M3) | - | - | yes on `sm_120f`/`sm_120a` (verified) |
| `mma.sync kind::mxf8f6f4.block_scale.scale_vec::1X ... .ue8m0` | - | - | yes on `sm_120f`/`sm_120a` (verified) |
| smem per block | 227 KB | 227 KB | ~99 KB |

Conclusion: no existing DeepGEMM kernel body can run on sm120. We need **new
sm120 kernels** built on TMA + mbarrier pipelines + `mma.sync`, plus host
dispatch/heuristics for arch 12.

### 2.3 Toolchain: already fine

- DeepJIT's `get_arch()` (`deep_jit/backend/cuda/device.hpp:68-79`) returns
  `"120f"` for major > 9 (or `"120a"` with `use_arch_family=false`). It
  compiles with `nvcc <src> --cubin --gpu-architecture=sm_120f`
  (`options.hpp:98`, `backend.hpp:95-101`). Verified: `--cubin` with
  `sm_120f`/`sm_120a` accepts `setmaxnreg`, `kind::f8f6f4` and the block-scale
  MMA.
- Gotcha: `nvcc -c --gpu-architecture=sm_120a` (object output) **fails** for
  these instructions, because it also embeds generic `compute_120` PTX. When
  writing standalone probes, use `--cubin` or
  `-gencode arch=compute_120a,code=sm_120a`.
- CUTLASS 4.2.1 has `cute/arch/mma_sm120.hpp`: `SM120_16x8x32_TN<e4m3,e4m3,float>`,
  block-scaled `mxf8f6f4 ... scale_vec::1X ... ue8m0` and
  `mxf4nvf4 ... scale_vec::2X/4X` atoms. We can include these atoms the same
  way DeepGEMM includes the sm90/sm100 arch headers, and we don't need CUTLASS
  collectives.
- No AOT CUDA build. `setup.py` only builds the host extension, so adding
  kernels needs no build-system change.

## 3. Reusable pieces

Reuse as-is or with small edits:
- `heuristics/common.hpp` `get_best_config<ArchSpec>`: plug in a new
  `SM120ArchSpec`.
- `jit_kernels/impls/runtime_utils.hpp`: TMA descriptor builders
  (`make_tma_a/b/cd/sf_desc`).
- `scheduler/gemm.cuh`: persistent tile scheduler, including the
  grouped-contiguous / masked logic. Use multicast = 1.
- `ptx/tma.cuh`, `common/tma_copy.cuh` (sm90 TMA variants only),
  `cutlass/arch/barrier.h` `ClusterTransactionBarrier`, `ptx/ld_st.cuh`
  (stmatrix/ldmatrix).
- `smxx_layout.cuh/.hpp`: SF transpose and UE8M0 pack kernels. Relax the
  `arch == 10` assert.
- `smxx_cublaslt.hpp`: cuBLASLt paths for bf16 and einsum (whether cuBLASLt
  picks sm120 kernels is up to cuBLAS).
- `deep_gemm/utils/math.py` cast helpers and `tests/generators.py` (add arch
  12 branches).

Don't reuse: `mma/sm90.cuh`, `ptx/wgmma.cuh`, `mma/sm100.cuh`,
`ptx/tcgen05.cuh`, `epilogue/sm100_*`, or anything under `impls/sm100_*`.

## 4. Scale-factor semantics decision

There are two families, and sm120 hardware supports both:

1. **FP32 scales, "1d2d" semantics (sm90 style)**: A 1x128 per-token, B 128x128
   blocks, arbitrary FP32 values. On sm120, run plain `kind::f8f6f4` MMAs into
   a per-K-block partial accumulator. Every 128 K, fold it into the main
   accumulator with `sa[row] * sb[nblk]` on CUDA cores. This gives exact
   checkpoint numerics for DeepSeek V3/R1 (FP32 `weight_scale_inv`) and is what
   TokenSpeed's dense path (A) and the DSA indexer (C) feed today.
2. **UE8M0 packed scales, "1d1d" semantics (sm100 style)**: use
   `mxf8f6f4.block_scale.scale_vec::1X` with UE8M0 scales. That's one scale per
   32 K, so gran_k 128 means the same byte repeats across 4 MMA k-steps. The
   hardware applies the scale for free. This matches DeepGEMM's sm100
   packed-int SF layout (`docs/scaling-factor-format.md` Path B) and
   TokenSpeed's `deep_gemm_requires_ue8m0()` (>= sm100, already true on
   sm120). It needs power-of-two scales, so FP32 checkpoints get requantized
   (small accuracy change; V3.2/V4 `scale_fmt=ue8m0` checkpoints are native).

Decision: implement **one sm120 FP8 GEMM kernel template with a compile-time
scale mode**, `kFP32Promote` vs `kUE8M0BlockScale`. Everything else (TMA
pipeline, scheduler, grouped layouts, epilogue) is shared, so this is a
parameter of one path, not two kernels. Dispatch keys on the SFA dtype exactly
like today (float goes to promote mode, int goes to block-scale mode).
`get_default_recipe` for arch 12 returns (1,128,128) for FP32 SFB and
(1,1,128) for int, which mirrors sm100.

## 5. Kernel design sketch: `sm120_fp8_gemm_1d1d.cuh` / `_1d2d` modes

- Warp specialization: 1 producer warp (TMA A/B/SFA[/SFB] into a smem ring,
  mbarrier full/empty) plus 2 consumer warpgroups (8 warps). Use `setmaxnreg`
  to shrink the producer and grow the consumers.
- Tiles: BLOCK_K = 128 (FP8, one 128B-swizzle row). Start with
  BLOCK_M x BLOCK_N = 128x128 and warp tile 64x32 (2x4 warps). One stage is
  then 16 KB A + 16 KB B + SF, so 2-3 stages fit in 99 KB. Also try 64xN for
  small M (decode) and 128x256 with 2 stages for large M.
- Operand fetch: `ldmatrix` (b16 view of FP8 pairs) from 128B-swizzled TMA
  tiles into the `m16n8k32` fragment layout. A and B are both K-major (NT),
  which is exactly `row.col`.
- Accumulators: 64x32 per warp is 64 FP32 per thread. Promote mode adds a
  second 64-register partial accumulator, so watch for spills
  (`--warn-on-spills` is wired in DeepJIT).
- Epilogue: BF16 via `stmatrix` into reused pipeline smem, then TMA store.
  Pipeline smem must be reused because there's no room for a dedicated
  epilogue buffer at 99 KB.
- Scheduler: reuse `scheduler/gemm.cuh` (normal, m-grouped contiguous, masked)
  with persistent CTAs = #SMs.
- No clusters or multicast in v1. Revisit only after profiling shows L2/DRAM
  bound on B loads.

MQA logits (C) on sm120: the same building blocks. FP8 Q x FP8 K^T with
`kind::f8f6f4`, then per-row K scale, ReLU, weighted head-sum on CUDA cores.
The paged variant reuses `scheduler/sm90_paged_mqa_logits.cuh` and the sm90
metadata kernel, which is pure scheduling, after checking it for
wgmma-specific constants.

## 5.5. Completing DeepGEMM on sm120: UE8M0 vs FP4 (decision, 2026-11-03)

The goal is DeepGEMM itself complete on sm120, not just the subset TokenSpeed
calls on this box. Of the unimplemented GEMM features, two are real work worth
doing: UE8M0 block-scale FP8, and FP4. Verified against the PTX ISA and the
SM100 path (context gather + ISA 9.7.16.3):

- **sm120 has the hardware block-scale MMA.**
  `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.f32.e4m3.e4m3.f32.ue8m0`
  assembles on `sm_120a`/`sm_120f`. Same A/B/C/D fragment layout as the plain
  `kind::f8f6f4` I already use, plus `scaleAData, {byte-id-a, thread-id-a},
  scaleBData, {byte-id-b, thread-id-b}`. scale_vec::1X is one UE8M0 byte per
  32-K block (`D = (A*2^(sfa-127)) * (B*2^(sfb-127)) + C`); the selectors pick
  which byte of a b32 reg and which lane in the quad supplies it. FP4 uses
  `kind::mxf4nvf4 ... m16n8k64` with scale_vec::2X/4X (ue8m0 or ue4m3).

- **UE8M0 FP8 on sm120 buys little over the FP32 path that already ships.**
  sm120 has no tcgen05, so the "scale is free + frees the promote registers +
  unlocks 128x256 tiles" benefit only comes from the *hardware* block-scale
  MMA, and even then TokenSpeed uses gran_k=128 (one scale per 4 MMA k-steps),
  so the register saving is marginal. A software-promote UE8M0 kernel (decode
  `exp2f(byte-127)`, same fold as FP32) would be nearly identical in cost to
  the existing FP32 kernel, differing only by reading a 1-byte SF instead of a
  4-byte one on an already-tiny SF stream. And TokenSpeed caps
  `deep_gemm_requires_ue8m0()` to sm100 and feeds sm120 FP32, so there is no
  engine caller for an sm120 int path unless that cap is also lifted. Net: low
  value, deferred.

- **FP4 is the feature the FP32 path cannot do, and it is where "beat the
  fallback" has the most room.** NVFP4/MXFP4 halves A/B bandwidth and roughly
  doubles tensor-core throughput over FP8. On the decode shapes that dominate
  serving (weight-bandwidth bound), that is a direct win no FP8 kernel —
  DeepGEMM's own FP32 path, Triton, or FlashInfer — can match. It needs the
  block-scale MMA machinery anyway (`mxf4nvf4`, scale_vec 2X/4X), so building
  FP4 first also lays the scale-register plumbing that a later UE8M0-FP8 mode
  would reuse.

**Decision:** do FP4 next (the real throughput win and the harder plumbing),
then fold in UE8M0-FP8 as a scale mode on the same block-scale kernel if a
caller materializes. Both share: load A/B FP4 (packed `e2m1`, 2 per byte) via
TMA, `ldmatrix` into the `m16n8k64` fragment, UE8M0/UE4M3 scale bytes staged to
registers with the `{byte-id, thread-id}` selector layout, block-scale MMA,
BF16 epilogue reusing the existing store. Dispatch keys on A/B dtype
(`kPackedFP4`) and SFA dtype, as sm100 already does.

## 5.6. SM120 block-scale MMA: verified instruction + SF lane mapping (2026-11-03)

`scripts/sm120_blockscale_probe.cu` (standalone, GPU 7) pins down the hardware
contract for
`mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0`:

- **Must compile arch-specific.** `nvcc -arch=sm_120a` still feeds ptxas a
  generic `compute_120` target, which rejects `.block_scale` / `.kind::mxf8f6f4`
  / `.scale_vec::1X` ("not supported on .target 'sm_120'"). Build with
  `-gencode arch=compute_120a,code=sm_120a` (DeepJIT already emits `sm_120a`).
- **Operands:** same A/B/C/D fragments as the plain `kind::f8f6f4` m16n8k32,
  plus `scaleAData, {byte-id-a, thread-id-a}, scaleBData, {byte-id-b,
  thread-id-b}`. The selector tuple components are `.u16` (`"h"`). CUTLASS
  (`mma_traits_sm120.hpp`) uses `SFARegisters = uint8_t[1]`, selectors all 0.
- **Decode:** UE8M0 byte `b` -> `2^(b-127)`. `D = (A*2^(sfa-127)) *
  (B*2^(sfb-127)) + C`. Confirmed exact (err 0) against a dequantized reference.
- **SF lane mapping with {bid=0, tid=0}** (lane `L = 4g + t`, `g = L/4`,
  `t = L%4`), the empirical result and what the kernel must load:
  - SF_A: `t==0` -> row `g`; `t==1` -> row `g+8`; `t∈{2,3}` unused.
  - SF_B: `t==0` -> col `g`; `t∈{1,2,3}` unused.
  This matches CUTLASS's `SFALayout = ((2,2,8),32):((8,0,1),16)` ("effectively
  16 threads, 2:0 broadcast") and `SFBLayout = ((4,8),32):((0,1),8)`
  ("effectively 8 threads, 4:0 broadcast"). It lines up with the existing
  SM120 kernel's lane layout (`group_row = lane/4`, within-warp `t`), so each
  math warp can load its own rows'/cols' UE8M0 bytes into one b32 reg (byte 0)
  and issue block-scale MMAs with no extra shuffles.

Consequence for the kernel: the scale is applied in the tensor core, so the
FP32 promote path's second accumulator and the per-K-block CUDA-core multiply
go away -- freeing the registers that currently cap tiles at 128x128. This is
what makes FP4 (and a later UE8M0-FP8 mode) able to run 128x256.

## 6. Phased plan

Each phase ends with tests passing on the remote, GPU 7, and a status-log
entry.

### Phase 0: environment and baselines (no kernel code)
1. Install torch into the venv:
   `uv pip install --python .venv/bin/python torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`.
   Then build/install DeepGEMM from source (`./develop.sh` or `./install.sh`).
   Confirm `import deep_gemm` works.
2. Record device facts: SM count, `cudaDevAttrMaxSharedMemoryPerBlockOptin`,
   cluster limits, L2 size.
3. Microbenchmark `mma.sync` E4M3 m16n8k32 throughput with f32 vs f16
   accumulate, and with vs without block_scale. If FP32 accumulate is half
   rate on this SKU, that changes the tile/accum strategy, so do this before
   writing the kernel.
4. Baselines for DeepSeek V3 shapes (N,K in {(24576,1536), (32768,512),
   (7168,16384), (4096,7168), (7168,2048), (2112,7168), (576,7168)}; M in
   {1..8192}):
   - cuBLASLt plain FP8 (roofline proxy);
   - FlashInfer `gemm_fp8_nt_groupwise` / TokenSpeed's
     `flashinfer_mm_fp8_blockscale`, if it runs on sm120;
   - TokenSpeed `triton_mm_fp8_blockscale`;
   - cuBLASLt block-scaled FP8 if supported on sm120.

   This defines "highest performance". If FlashInfer's CUTLASS sm120
   groupwise kernel is already near roofline, the dense GEMM priority drops.
5. Run the existing test suite to record which tests crash on arch 12. That's
   the baseline for "no regressions".

### Phase 1: host plumbing for arch 12 (no new kernels yet)
- `SM120ArchSpec` in `heuristics/sm120.hpp` (smem from the runtime attribute,
  not hard-coded 232448).
- `get_default_recipe`, `fp8_fp4_requires_k_major` (sm120 needs K-major),
  `transform_sf_into_required_layout` and the `smxx_layout` asserts.
- Explicit `arch_major == 12` branches in `gemm.hpp` / `attention.hpp`.
  Unsupported APIs must raise a clear "not supported on sm120" error, not
  `UNREACHABLE` and not a silent sm90 fallthrough.
- `tests/generators.py` and test filters: arch 12 branches. Fix the `>= 9` /
  `>= 10` filters that would wrongly run sm90/sm100-only tests.

### Phase 2: `fp8_gemm_nt` on sm120 (target A)
- Kernel template from section 5, FP32-promote mode first (matches
  TokenSpeed's dense path and V3/R1 checkpoints), then UE8M0 block-scale mode.
- Tests: `tests/test_fp8_fp4.py` normal GEMM with tight tolerance vs the
  existing reference, both recipes. Also `assert`-style checks for no
  register spills.
- Tune the heuristics on the Phase 0 shape table and compare to the baselines.

### Phase 3: m-grouped contiguous and masked (target B)
- Same kernel, `GemmType::MGroupedContiguous` / `MGroupedMasked` via the
  shared scheduler. Tests in `test_fp8_fp4.py`.
- Before investing in tuning, confirm the MoE serving path on this box. Without
  NVLink, DeepEP intranode is likely unavailable, so TP-MoE goes through
  Triton/FlashInfer fused MoE. A grouped DeepGEMM is then only useful if
  TokenSpeed gets a non-DeepEP DeepGEMM MoE path.

### Phase 4: MQA logits for the V3.2 DSA indexer (target C)
- `get_paged_mqa_logits_metadata` (likely reusable), `fp8_paged_mqa_logits`
  (decode, next_n = 1 as TokenSpeed flattens MTP), `fp8_mqa_logits` (prefill).
  Tests: `tests/test_attention.py` paths for arch 12.

### Phase 4.5: beat Triton/FlashInfer, or don't ship (gate for Phase 5)

DeepGEMM on sm120 is only worth using if it's faster than what TokenSpeed would
otherwise pick on sm120. **The baseline is the TokenSpeed fallback, not cuBLAS.**
Earlier cuBLAS comparisons (section 10) only show the roofline. They don't
justify anything.

Competitors, i.e. what the TokenSpeed registry selects on sm120 today:

| Op | DeepGEMM | Competitors (all that run on sm120) |
|---|---|---|
| Dense block-FP8 linear | `fp8_gemm_nt` | `flashinfer_mm_fp8_blockscale` (if it runs on sm120; it has priority today), `triton_mm_fp8_blockscale` |
| MoE grouped FP8 | `m_grouped_*_contiguous` / `_masked` | TokenSpeed Triton fused MoE (`ops/moe/triton.py`), FlashInfer/CUTLASS MoE paths registered for sm120 |
| DSA indexer decode | `get_paged_mqa_logits_metadata` + `fp8_paged_mqa_logits` | `triton_dsa_plan` + `triton_dsa_decode_topk_fp8` |
| DSA indexer prefill | `fp8_mqa_logits` | `triton_dsa_prefill_topk_fp8`, `triton_kpool_prefill_topk` |

Method:
- **Same call boundary as TokenSpeed.** Time the op as TokenSpeed calls it,
  including everything DeepGEMM adds per call: `transform_sf_into_required_layout`,
  the SFA transpose kernel (`transpose_fp32`, visible in the profiler), the
  split-K workspace allocation, and FP32 scale passthrough. Benchmark through
  the `tokenspeed_kernel` registry with a forced `solution=`, so each competitor
  runs exactly as served.
- Also time kernel-only (`scripts/sm120_baseline.py::bench_cold`, cold L2, sum
  of profiler kernel times), so overheads can be attributed.
- Also time hot L2 under CUDA graph. That's the decode serving condition: weights
  of one layer are cold, but small activations and scales are hot. Cold-only
  numbers penalize everyone equally, but launch-count overhead (DeepGEMM =
  transform + GEMM kernels) only shows under graphs.
- Shapes: DeepSeek V3/R1 and V3.2 per-GPU serving shapes for the TP size we'd
  actually use on this box (TP1/TP2/TP4/TP8, PCIe only). M buckets: decode
  {1, 2, 4, 8, 16, 32, 64, 128, 256} (batch x MTP tokens), prefill
  {512, 1024, 2048, 4096, 8192, 16384}. Indexer: H = 64, D = 128, page 64,
  context {4K, 16K, 64K, 128K}, batch {1, 8, 32, 128}.
- Correctness parity at each point: same reference, same `calc_diff` bound.

Win criteria, per op, evaluated before any TokenSpeed change:
1. Every decode-bucket shape: DeepGEMM <= 0.95x the best competitor's time. Decode
   dominates serving latency, so a single losing decode shape blocks that op.
2. Prefill buckets: geometric-mean speedup >= 1.0x, with no shape slower than 0.97x.
   (Revised from >= 1.10x: ncu shows both DeepGEMM and FlashInfer/CUTLASS at ~85% tensor-pipe
   utilization for large M, so a 10% margin is not physically available. See the status log.)
3. End-to-end: serve DeepSeek V3 (or V3.2) on this box with DeepGEMM vs the
   fallback, same config. TPOT and prefill throughput must both improve.

Loop: measure, then profile the losing shapes (ncu: tensor-pipe utilization,
DRAM throughput, stall reasons), then optimize, then re-measure. Log every round
in section 9 and every number in section 10. Optimization backlog, by expected
impact (decide by profile, not by guess):
- Remove per-call overhead: pre-transform weight SFB once at load (TokenSpeed
  weight preprocessor), and have TokenSpeed's quantizer emit the MN-major,
  TMA-aligned SFA directly so the transpose kernel disappears.
- UE8M0 block-scale MMA mode (`mxf8f6f4.block_scale`): the scale is applied in
  hardware, which removes the CUDA-core promote, halves accumulator registers
  and unlocks 128x256 tiles. Needs UE8M0 checkpoints or load-time requantization.
- A stmatrix + TMA-store epilogue reusing pipeline smem, for large M.
- Better decode schedule: stream-K or finer split-K, plus warp-specialized
  `ldmatrix` double buffering.
- Fuse SiLU-mul + FP8 quant into the grouped GEMM epilogue (MoE).

Exit rule: if after the backlog DeepGEMM still loses on an op, **don't enable it
for that op on sm120**. Keep the TokenSpeed fallback, cap the registration's
`max_arch_version` below 12.0, and record the numbers and the reason in section 8
(dead ends). Phase 5 only integrates ops that passed this gate.

### Phase 5: TokenSpeed integration (separate repo, separate change)
- Immediate safety fix, independent of DeepGEMM: cap `deep_gemm_dsa_*`,
  `deep_gemm_kpool_prefill_topk`, `deep_gemm_deepep_fp8_moe_apply` with a
  `max_arch_version` (10.3) until sm120 kernels ship. Today they're selected on
  sm120 and would fail.
- After DeepGEMM sm120 lands: widen the registrations to include 12.0, bump the
  `tokenspeed-deepgemm` pin, and make `deepgemm_mm_fp8_blockscale` accept sm120
  (it currently checks `is_hopper`). Benchmark end-to-end on DeepSeek V3/V3.2.

### Later / optional
- FP4 (`kind::mxf4nvf4`, `scale_vec::2X/4X`) for NVFP4/MXFP4 checkpoints, which
  matters given 96 GB/GPU.
- Clusters/multicast if profiling justifies them.
- `tf32_hc_prenorm_gemm` (V4) if the Triton fallback is a bottleneck.

## 7. Verification rules

- Remote only, `CUDA_VISIBLE_DEVICES=7`, venv python.
- Sync code local -> remote with git (commit on `nctu6`, push, pull on the
  remote) or rsync. Never edit on the remote.
- Correctness: tight atol/rtol versus the existing `deep_gemm.testing`
  reference (`calc_diff`), for both scale modes and all GEMM types.
- Performance: always against the Phase 0 baselines on the same shapes, with
  the same timing utility (`deep_gemm.testing.bench_kineto`).
- JIT: check compile flags and logs (`DG_JIT_DEBUG`, `DG_JIT_PTXAS_VERBOSE` /
  equivalent) for spills and local memory.

## 8. Dead ends (don't do these)

- Porting sm100 kernels (`tcgen05`, TMEM, UMMA, 2-CTA): that hardware doesn't
  exist on sm120.
- Relaxing arch checks so sm90 kernels run: `wgmma` doesn't exist on sm120, so
  ptxas rejects it.
- Reusing sm90/sm100 heuristics or stage counts: they assume 227 KB smem, and
  sm120 has about 99 KB.
- Starting with TMA multicast / clusters: unverified on this SKU and not needed
  for v1.
- Adopting CUTLASS collective GEMMs inside DeepGEMM: DeepGEMM only uses CuTe
  arch atoms. CUTLASS/FlashInfer sm120 kernels are **baselines**, not the
  implementation.
- Concluding "instruction unsupported" from `nvcc -c --gpu-architecture=sm_120a`
  probes: that form embeds `compute_120` PTX and fails. Use `--cubin` or an
  explicit `-gencode` (section 2.3).
- Spending effort on the DeepEP MoE path before confirming DeepEP works on this
  PCIe-only box.
- Using any GPU other than 7, or pip-installing outside `.venv`.
- Two CTAs per SM for small 1D1D tiles (64x64 / 32x64 / 16x32 with half the smem
  each): implemented and correct, 2 blocks/SM confirmed with the occupancy API,
  but never faster (o_proj M=128: 64x64x128 8.0 vs 8.2 us single, 64x128 7.1).
  The busiest SM still gets the same work; removed (2026-11-04).
- Stream-K on mid-M shapes (tiles > SMs / 8): every CTA pays two pipeline fills,
  e.g. 64x128 at M=128, N=4608: 23.9 vs 16.4 us. Tested again on o_proj M=128,
  q_b M=256, dense gate_up M=128/256: never in the top 6.
- Folding stream-K into the fitted 1D1D cost model: only a few measured shapes
  split, so the fit can't learn it and mispicks (N=512 decode lost 4.7 -> 5.9 us).
  The explicit rule (tiles <= SMs / 8 and a clear K-loop win) fits better.
- Random-restart fits of the 1D1D model: lower in-sample regret (1.017) with
  non-physical constants (64x256 cost 15592, box exponent 9.5). Keep the greedy
  fit with plausible constants.
- Timing-based autotuning on first call as the default: CUDA events resolve ~2 us
  here and eager launches can't see the PDL overlap; it mispicks as often as the
  model. Opt-in only (`DG_SM120_AUTOTUNE=1`).
- Raising the math register share of the 640-thread paged MQA kernel above 104 (e.g. 48/112): the CTA's pool is 96 x 640
  = 61440 registers, so `setmaxnreg.inc` blocks forever (a hang, section 9.21). Fit the MXFP8 128-row case some other way or
  leave it rejected; per-step B-scale reloads and a rolled K loop did not fit it.
- Routing the small or many-head `fp8_einsum` shapes to BF16 cuBLASLt inside `fp8_einsum`: its inputs are FP8, so the route has
  to dequantize first. Dequantize + cuBLASLt measured 53-211 us against 8-30 us for the batched FP8 launch (section 9.14), so
  the batched launch is the only way to meet cuBLASLt's launch count.

Deferred features (feasible but out of scope for inference on this box; the
host still rejects them, so there is no silent wrong path):
- **Sparse MQA logits** and **block-scaled MXFP8/MXFP4 dense MQA logits**: DONE
  (2026-11-06, section 9.6). Both ship for arch 12 and are validated against a
  dequantized reference. No longer deferred.
- **MN-major `fp8_einsum`** (`fp8_einsum` `bhd,hdr->bhr` / `bhd,bhr->hdr`): no longer here, done in section 9.12. The reason
  given here earlier, that MN-major FP8 needs a byte-granular transpose and `ldmatrix.b8` doesn't exist on sm_120, was
  wrong: `ldmatrix.m16n16.x{1,2}.trans.b8` exists on sm_120a/f (section 9.10). **k-grouped FP8 GEMMs / Wgrad** are no longer here:
  `k_grouped_fp8_gemm_tn_contiguous` / `_nt_contiguous` ship for arch 12 (section 9.11), running one MN-major block-scaled
  GEMM per group. FP4 dense MQA is no longer here either: it is done, section 9.6.
- **Mega MoE**: its point is NVLink symmetric-memory dispatch/combine across
  ranks; on a single PCIe GPU it degenerates to a local grouped-GEMM chain that
  `m_grouped_*` already covers, and a port is effectively a UMMA/TMEM -> mma.sync
  rewrite. **Mega gate / mHC**: single-GPU but TMEM/TF32-pipeline rewrites (mHC
  builds on the TF32 `m16n8k8` now in `mma/sm120.cuh`). Low priority.
  (Task 21, section 9.26: the single-rank mega MoE is now that grouped-GEMM chain with the SM100 numerics, on a new
  W4A8 MMA path. P2P works between GPUs 6 and 7, so EP > 1 is possible too; gate and mHC remain, with no TokenSpeed caller.)

## 9. Status log

| Date | Phase | Status | Notes |
|------|-------|--------|-------|
| 2026-11-03 | Analysis | done | Mapped DeepGEMM arch dispatch and TokenSpeed call sites. Initialized submodules on the remote. Verified that `kind::f8f6f4`, `mxf8f6f4.block_scale` and `setmaxnreg` assemble on `sm_120f`/`sm_120a` (`--cubin`), and that DeepJIT targets `sm_120f`. Verified CUTLASS 4.2.1 has `mma_sm120.hpp`. No NVLink on the box. Torch not yet installed. |
| 2026-11-03 | Phase 0 | done | torch 2.14.1+cu130 in `.venv` (py3.15; install with `pip install "torch==2.14.1+cu130" --index-url .../cu130`, `uv` refuses the cp315 tag). DeepGEMM builds with `bash develop.sh` (needs `CUDA_HOME=/usr/local/cuda`). Device: 188 SMs, 101376 B opt-in smem/block, 102400 B/SM, L2 128 MB, 64K regs/SM. Baseline: every FP8/BF16 API hits `UNREACHABLE` (`Unknown recipe` / `Unsupported architecture`). |
| 2026-11-03 | Phase 1+2 | done (v1) | New `sm120_fp8_gemm_1d2d` (FP32 SFA/SFB, normal + m-grouped contiguous/psum + masked), `heuristics/sm120.hpp`, arch-12 dispatch in `gemm.hpp`, `layout.hpp` (recipe, K-major, FP32 SF passthrough). Correct on first run (diff ~7e-4 on DeepSeek shapes). Added deterministic split-K for normal GEMMs (<= 4 splits). |
| 2026-11-03 | Phase 2 tuning | done (v1) | Heuristic fitted to sweeps: auto within 1.10x of best layout on most shapes, worst 1.25x (M=64). See section 10 for numbers. |
| 2026-11-03 | Phase 2+3 tests | done | `tests/test_fp8_fp4.py` passes on arch 12: normal (fwd + Dgrad), m-grouped contiguous (psum and non-psum), masked, plus bitwise determinism. Generators treat arch 12 as SM90-like for FP8 (`is_sm90_like_fp8`). Wgrad (1D1D, FP32 out, accumulation) and k-grouped are skipped explicitly, since they're training-only. Grouped contiguous 680-757 TFLOPS; masked small-M up to 1.37 TB/s. |
| 2026-11-03 | Split-K counters | done | Counters are now per-stream and assert they aren't first allocated during CUDA-graph capture, the same pattern as the sparse MQA workspace. Not yet rebuilt or tested on remote. |
| 2026-11-03 | Phase 4 | in progress | Added `mma::sm120::warp_mma_16xNx128_ss`, which keeps the SM90 WGMMA per-warp accumulator layout. Generated `sm120_fp8_mqa_logits.cuh` / `sm120_fp8_paged_mqa_logits.cuh` from the SM90 sources, so the scheduler, TMA and epilogue code is unchanged and only the MMA block is swapped. The metadata kernel (`scheduler/sm90_paged_mqa_logits.cuh`) is arch-agnostic and reused. **Remaining:** host launchers (smem must fit 99 KB, so the SM90 3+3 stages at 512 math threads need re-checking), arch-12 dispatch in `attention.hpp`, `test_attention.py` on arch 12, register/spill check. |
| 2026-11-03 | Phase 4.5 setup | done | Competitors come from the existing TokenSpeed venv (`/workspace/nctu6/infra/tokenspeed/.venv`, py3.12, has tokenspeed_kernel + flashinfer 0.7.2 + triton). DeepGEMM is built separately for it (`build312/`, used via PYTHONPATH, so the pinned `tokenspeed-deepgemm` is never touched). Helper: `scripts/sm120_remote.sh {build,dg,ts}`. FlashInfer JIT needs the venv `bin/` on PATH (`ninja`). |
| 2026-11-03 | Phase 4.5 round 1 (dense) | **losing in places** | `scripts/sm120_vs_tokenspeed.py`, CUDA-graph + cold weights, TP1. vs Triton: DG wins almost everywhere (1.1-3.8x) except q_b/kv_b at M=32-128 (0.81-0.99). vs FlashInfer (runs on sm120, picked by TokenSpeed except 17<=M<=32 and N%128!=0): DG wins decode M<=128 on most linears (1.1-2.3x), but loses q_b/kv_b/shared-down at M>=256 (0.84-0.94), and large M is parity (0.99-1.04). |
| 2026-11-03 | Split-K fix | done | Split-K counters were per-stream and asserted outside capture, so they broke under `torch.cuda.graph` (it captures on its own stream). Now they're plain `cudaMalloc` (relaxed capture mode) zeroed by a memset on the current stream, so first use inside a capture is safe. |
| 2026-11-03 | Stream-K | tried, not a win | Replaced split-K with stream-K (even iteration ranges; deterministic slot reduction; `DG_SM120_FORCE_STREAM_K`). Correct, but under graphs sk=0 is best on almost every shape; it only helps M=256 q_b by ~3%. **The real loss is the heuristic**: `bench_cold`-fitted choices are 1.1-1.9x slower than the best layout under graphs. Re-fitting on graph-mode sweeps (`scripts/sm120_sweep_graph.py`). |
| 2026-11-03 | MMA peak | measured | `mma.sync` E4M3 m16n8k32 FP32-accum peak ≈ 906 TFLOPS, with block_scale at the same rate. Large-M DG / FlashInfer reach ~730-750 (81-83%). |
| 2026-11-03 | Phase 4 | done (correctness) | Host launchers in `csrc/jit_kernels/impls/sm120_fp8_mqa_logits.hpp`, arch-12 dispatch in `attention.hpp`. Contiguous: 8 math warps, BLOCK_KV 128, 2 Q + 3 KV stages (fits 99 KB). Paged: 4 math WGs (SPLIT_KV 256, same as SM90 so the SM90 metadata kernel is reused unchanged), stage count chosen to fit smem (2+2 for next_n=1). `scripts/sm120_mqa_check.py` matches a dequantized reference to ~2e-14 (contiguous; paged next_n 1 and 2; H 32/64; ctx up to 20K). `tests/test_attention.py` asserts pass on the cases that fit memory; it OOMs on GPU 7 (see the next row) and its kineto perf print fails with CUPTI_ERROR_MULTIPLE_SUBSCRIBERS (run with `DG_USE_NVIDIA_TOOLS=1`). Test head dims are limited to 128 on arch 12. |
| 2026-11-03 | **Env caveat** | open | GPU 7 is **shared**: a root `VLLM::EngineCore` (it restarts; seen with 65-92 GB and up to 53% util) runs on it. Benchmarks taken while it's active are noisy: 22/210 sweep rows have the heuristic "beating" the best forced layout by >7%, which is impossible. Perf numbers must be compared within the same run, interleaved and repeated. |
| 2026-11-03 | API note | open | DeepGEMM 2.8.2 (this base) renamed the indexer APIs: `fp8_mqa_logits(q, kv, ..., clean_logits, max_seqlen_k)` became `fp8_fp4_mqa_logits((q, None), kv, w, ks, ke, max_seqlen_k)`, and `fp8_paged_mqa_logits` became `fp8_fp4_paged_mqa_logits((q, None), ...)`. TokenSpeed calls the old names (pinned 2.8.0.post20260924). Phase 5 must port the call sites or add thin compat wrappers. |
| 2026-11-03 | Heuristic refit | done | Round-robin graph sweep (`sm120_sweep_graph.py --repeats 5 --json`, 15 shapes x 13 M) plus offline fit (`sm120_fit_heuristic.py`). New model in `SM120ArchSpec::get_layout_info`: per wave, max(DRAM share, L2 share, CTA time + overhead). Geomean regret vs the best layout: 1.177 (old) -> 1.006 (new), worst 3.2x -> 1.3x. Stream-K is now opt-in only (`DG_SM120_FORCE_STREAM_K=1`). |
| 2026-11-03 | SFA without transpose | done | TokenSpeed's per-call `transform_sf_into_required_layout` launched a `transpose_fp32` kernel (~2.4 us + a launch) on every GEMM. The SM120 kernel now reads SFA with plain strided loads (K-major or MN-major, any strides), and SFA passes through untransformed on arch 12. SFA is no longer staged in smem, so stages are slightly deeper. |
| 2026-11-03 | **Race fix** | done | `test_fp8_fp4.py` determinism caught random wrong 8-col tiles with 8 math warps (BLOCK_M=128). Root cause: generic-proxy `ldmatrix` reads followed by async-proxy TMA writes to the same stage need `fence.proxy.async.shared::cta` before the empty-barrier arrive. Without it, about 1 run in 20 (128x128) or 7 in 20 (128x64) corrupted output. Moving the arrive or adding a `memory` clobber didn't help; disabling setmaxnreg didn't either. Fence added to the GEMM and both MQA kernels. All layouts are now 0/20 nondeterministic, and `test_fp8_fp4.py` passes. |
| 2026-11-03 | ncu available | note | `sudo -n` works on aws-raichu, so `sudo -n env ... ncu` gives counters. Large-M 128x128: tensor pipe 85%, L2 70%, so it's compute-bound near peak. FlashInfer (CUTLASS `MainloopSm120TmaWarpSpecializedBlockwiseScaling`, 384 non-persistent CTAs, 82 KB smem) is in the same regime. **A >= 1.10x prefill win over FlashInfer is not physically available for FP32-accumulate FP8 at these sizes**, so the prefill gate is revised to "no regression (>= 0.97x)". |
| 2026-11-03 | DSA decode | **DG wins big** | `sm120_dsa_vs_tokenspeed.py` (TokenSpeed's `deep_gemm_dsa_decode_topk` vs `triton_dsa_decode_topk_fp8`, graph): DG is 5.9-18.9x faster (B 1-128, ctx 16K-128K). Top-k overlap 0.97-0.98 (random data, near ties). Needed the `deep_gemm.compat` legacy wrappers (`fp8_mqa_logits`, `fp8_paged_mqa_logits`). |
| 2026-11-03 | DSA prefill | TokenSpeed blocker | TokenSpeed's `deep_gemm_dsa_prefill_topk` calls `torch.ops.trtllm.indexer_topk_prefill`, which has **no sm120 kernel image** in this build (`cudaErrorNoKernelImageForDevice`). Phase 5 must route sm120 to another top-k. The benchmark uses `batch_invariant=True` (row-invariant top-k) for now. |
| 2026-11-03 | MQA hang fixes | done | (1) My fence insertion into the brace-less `if (q_iter_idx > 0) arrive()` made the arrive unconditional, which hung paged MQA at B>=32. Now braced. (2) next_n=2 picked 1 Q stage to fit smem, which deadlocks: the TMA warp waits for an empty Q stage before issuing the KV loads the consumer needs. Paged configs now require >= 2 Q stages ((3,3), (2,3), (2,2), (2,1)). `sm120_mqa_check.py` gained B=32/64/128 and next_n=2 regression cases; all pass at ~4e-14. |
| 2026-11-03 | Ops note | note | Don't `pkill -f <script name>` over ssh: the pattern also matches the ssh command line and kills the session. |
| 2026-11-03 | Epilogue + SFA prefetch | done | ncu at kv_b M=512 (DG 44.5 us vs FlashInfer 38.7 us): DG issued 262K global-store requests at 8 half-empty sectors each, with lg_throttle 0.88 and long_scoreboard 2.5 (vs FI 0.63, which uses TMA stores). Changes: lanes `t`/`t^1` exchange halves with one `shfl.xor`, so each lane stores 8 B and each row writes a full 32 B sector per store (requests and sectors halved, lg_throttle 0.88 -> 0.03); SFA is prefetched one K block ahead (long_scoreboard 2.5 -> 0.93). kv_b M=512: 44.5 -> 39.2 us (FI 38.6). Requires `ld_d % 4 == 0`. |
| 2026-11-03 | Phase 4.5 dense | **passed (decode), parity (prefill)** | Round 4, CUDA graph + cold weights, vs the best of Triton/FlashInfer per shape. **TP8** (the config this 8x96 GB box needs for DeepSeek-V3): every decode shape M <= 128 wins 1.17-4.0x except shared-down M=128 (1.03); M=256 1.00-2.5x; prefill M >= 512 0.96-1.18x (FlashInfer parity within ~3% noise). **TP1**: decode 1.0-3.0x (K=512/1536 at M=64-128 are at Triton parity, 0.97-1.00); prefill 0.96-1.05x. The best layout is already chosen for the remaining prefill gaps (sweeps), so they're kernel-level (both kernels are tensor-pipe bound). See section 10. |
| 2026-11-03 | Phase 4.5 DSA | **passed** | Decode (`dsa_plan` + `dsa_decode_topk`, graph): DG 3.4-19.3x faster than Triton (B 1-128, ctx 4K-128K). Prefill: DG logits plus Triton's radix top-k (`_topk_with_padding`) is 7.8-52x faster than `triton_dsa_prefill_topk_fp8` (T 1K-8K, prefix up to 14K). Top-k overlap with Triton 0.97-1.00 (random data, near ties; DG matches a dequantized reference to 1e-14). |
| 2026-11-03 | Phase 4.5 MoE | n/a on this box | TokenSpeed uses DeepGEMM grouped GEMMs only through `deep_gemm_deepep_fp8_moe_apply` (DeepEP). DeepEP needs NVLink (intranode) or RDMA (low-latency), and this box is PCIe-only. The kernels are correct and fast (test_fp8_fp4: contiguous 680-757 TFLOPS, masked up to 1.37 TB/s), but there's no TokenSpeed consumer here. |
| 2026-11-03 | Phase 5 DeepGEMM side | done | `deep_gemm.get_supported_arch_majors()` returns (9, 10, 12), so callers can tell an SM120-capable build from the pinned 2.8.0. Legacy `fp8_mqa_logits` / `fp8_paged_mqa_logits` wrappers live in `deep_gemm/compat.py`. |
| 2026-11-03 | Phase 5 TokenSpeed side | done (local, uncommitted) | `thirdparty/deepgemm.py::deepgemm_supports_sm120()` probes the build. Dense: new `deepgemm_sm120_mm_fp8_blockscale` (12.0-12.9, SPECIALIZED, `block_scale_layout` `canonical_blackwell`, the label the `mm` facade uses on SM100+). `flashinfer_mm_fp8_blockscale` steps back to <= 11.9 when it's registered, since DG's shape contract covers FI's and an in-tree kernel can't go above FI's SPECIALIZED+3. DSA, KPool and DeepEP-MoE DeepGEMM leaves are capped at 10.3 unless the build supports SM120. DSA prefill on SM120 uses `triton_topk_from_logits` instead of `torch.ops.trtllm.indexer_topk_prefill` (no sm120 image). `deep_gemm_requires_ue8m0()` is now SM100-family only. server_args help and the gemm README are updated. Tests: `test/test_deepgemm_sm120_gating.py`, `test/nvidia/ops/gemm/test_sm120_deepgemm.py`, 17 passed on GPU 7. The related existing suites pass; `test_residual_mhc_arch_window::test_registry_picks_triton_mhc_on_sm120_not_deep_gemm` fails only when run after other files, also on the stock package (pre-existing order dependence). |
| 2026-11-03 | Phase 5 E2E | **DG wins** | Qwen3.8-27B-FP8 (block 128x128 FP32 scales, the same dense block-FP8 path as DeepSeek-V3), TokenSpeed TP1 on GPU 7, `scripts/sm120_e2e.sh base|dg`, two runs each. TPOT: C=1 31.9 -> **20.6 ms (1.55x)**, C=8 32.4 -> **22.4 ms (1.45x)**, C=32 33.5 -> **28.4 ms (1.18x)**. Output throughput +53% / +43% / +16%. Long-prompt (6000 words) TTFT 795 -> 781 ms. All 260 block-FP8 `mm` selections resolve to `deepgemm_sm120_mm_fp8_blockscale` (TOKENSPEED_KERNEL_VERBOSE). Greedy outputs match base (similarity 0.92-1.00, late divergence only). DeepSeek-V3/V3.2 itself needs 8 GPUs (TP8); only GPU 7 is allowed, so that run is still open. |
| 2026-11-03 | Commits | done | DeepGEMM `970b868` (no *.md), pushed to `origin/nctu6`. TokenSpeed `1c7a96a73` (integration; only my files, since `pre-commit --all-files` reformats ~100 unrelated files already on the branch). |
| 2026-11-03 | Submodule + install.sh | done (tested) | TokenSpeed: `DeepGEMM/` submodule (url `https://github.com/nctu6/DeepGEMM`, branch `nctu6`). `scripts/install.sh` inits it (recursive, only when unpopulated) and builds it after the kernel steps, because those re-pull the pinned wheel. It uninstalls `tokenspeed-deepgemm`, runs `pip install ./DeepGEMM --no-build-isolation --no-deps` with `DG_FORCE_BUILD=1` (never the upstream prebuilt wheel), and verifies through metadata. It reruns on `--force-deepgemm` / `REINSTALL_DEEPGEMM=1`, when `deep_gemm` is missing, or when `tokenspeed-deepgemm` reappears. Tested on aws-raichu with a scratch checkout plus a relocated copy of the TS venv, so the live venv is untouched: forced build OK (`2.8.2+970b868`), the rerun is a no-op, init from an empty checkout works, `fp8_gemm_nt` on GPU 7 gives calc_diff 1.4e-6 vs the dequantized reference, and the 17 TokenSpeed SM120 tests pass. |
| 2026-11-03 | Docker build deps | fixed | The DeepGEMM step failed in the image on two missing headers. (1) `elfutils/libdwfl.h` (deep_jit; `libdw.so.1` is only dlopen'd for stack traces): added `libdw-dev` to the Dockerfile apt layer, and to install.sh's system deps when DeepGEMM is built. (2) `cusparse.h` (torch's `CUDAContextLight.h`; the image's toolkit is trimmed to cudart/nvcc/cuBLAS): DeepGEMM `setup.py` now appends torch's pip `nvidia/*/include` dirs after `CUDA_HOME`'s (`3070f4c`). Tested with the exact Dockerfile step on top of `nctu6/tokenspeed:e27cee86-20261028`: build OK, the sanity check passes, and the GEMM inside the container on GPU 7 gives calc_diff 1.4e-6. `docker/unieai.nvidia.sh` with a stub `docker` repopulates an empty DeepGEMM recursively and keeps local edits. A full image build was not run (hours). |
| 2026-11-03 | Install auto-detect | done (tested) | TokenSpeed `scripts/install.sh` now auto-detects an sm120 GPU (compute-cap major 12 from `nvidia-smi`, torch fallback once the venv is resolved) and builds DeepGEMM from the submodule only there, with no flag. On any other GPU (or none visible) it keeps the published wheel and skips the DeepGEMM submodule init. `--force-deepgemm` / `REINSTALL_DEEPGEMM=1` / `FORCE_SM120=1` still force it anywhere. `docker/unieai.nvidia.sh`: the DeepGEMM submodule init is non-fatal now (keeps the wheel if it cannot populate), and the image still always builds DeepGEMM from source when present (it targets sm120 and the kernels JIT at runtime). Tested on aws-raichu: plain install on GPU 7 (cap 12.0) builds + verifies the source build; a stub `nvidia-smi` reporting 9.0 keeps the wheel and skips the init; `--force-deepgemm` forces the build with the stub still active. TokenSpeed commit `b4cf963fc`, local on `nctu6`. |
| 2026-11-03 | Limitations audit | verified | Re-read the host dispatch to confirm the "not supported on sm120" list against the code (not the report): every item hits `DG_HOST_UNREACHABLE` or an `arch_major == 10`/`== 9` assert with no arch-12 branch. The one nuance: m-grouped contiguous/masked FP8 GEMMs *are* implemented for arch 12, just unused by TokenSpeed on a PCIe box (DeepEP only). Current-state section updated to split "not implemented" from "implemented but unused". |
| 2026-11-04 | Block-scale probes | done | `scripts/sm120_blockscale_probe.cu` and `sm120_fp4_probe.cu` (standalone, GPU 7). Facts: (1) block-scale MMAs need arch-specific codegen (`-gencode arch=compute_120a,code=sm_120a` or `compute_120f`); `-arch=sm_120a` still hands ptxas a generic `compute_120` and fails. (2) UE8M0 decodes as `2^(b-127)`. (3) Scale lane map with all selectors 0 (`g = lane / 4`, `t = lane % 4`): SFA lane `t == 0` -> row `g`, `t == 1` -> row `g + 8`; SFB lane `t == 0` -> col `g`. FP4 `m16n8k64` `scale_vec::2X` takes a 16-bit register: byte 0 = K 0-31, byte 1 = K 32-63. (4) The packed-E2M1 A/B fragments load with the same `ldmatrix` pattern as FP8 (a 64-element FP4 K step is 32 bytes), low nibble = even k. |
| 2026-11-04 | Block-scaled GEMM (1D1D) | done (tested) | `sm120_fp8_fp4_gemm_1d1d`: FP4 (32-elem scales) and FP8 (32/128-elem scales), normal + m-grouped contiguous + masked, BF16 out, same host SF contract as SM100 (int32, 4 UE8M0 bytes per word, MN-major TMA-aligned; FP32 `gran_k = 32` scales are packed on the fly, `gran_k = 128` FP32 stays on the FP32 kernel). A 128-byte tile row is 256 FP4 or 128 FP8 elements, so the TMA pipeline, scheduler, warp tiling and epilogue are the FP32 kernel's; the scales ride the MMA, so there is no second accumulator. `scripts/sm120_blockscale_check.py` matches a dequantized reference (FP4 exactly, FP8 <= 5e-9) at tol 1e-7 on 7 + 3 shapes incl. K tails and the grouped kernels; negative control: one wrong scale -> 2.6e-6, one flipped byte -> 1.6e-5. `tests/test_fp8_fp4.py` and `sm120_mqa_check.py` still pass. |
| 2026-11-04 | FP4 scale latency | fixed | v1 read the packed words with `__ldg` one K block ahead. A 128-byte FP4 K step is shorter than a DRAM round trip, so decode FP4 ran at 15.7 us on q_a/kv_a (parity with FP8); with the loads stubbed out it ran 8.7 us. The scales now ride the stage: two more TMA copies on the same full barrier land them in per-stage smem (`make_tma_sf_desc`, box = rows x 1-2 words), read from smem by the math warps. FP4 q_a/kv_a M=1: 15.7 -> 9.5 us; FP8 g32 no longer pays for its extra scale traffic. |
| 2026-11-04 | Block-scaled perf | measured | Graph + cold weights, TP8 DeepSeek-V3 linears (`sm120_bench_blockscale.py --baselines`): FP8 UE8M0 (g128 and g32) is at parity with the FP32 kernel (+-5%), as expected (same bytes, same MMA rate; it is a coverage feature for UE8M0 checkpoints). FP4 is 1.6-2.0x faster than the best FP8 kernel at nearly every shape, e.g. dense gate_up: M=1 13.6 us (FP8 24.0, Triton 58.7), M=4096 205 us (FP8 403, FlashInfer 410). Weakest: shared down (K = 256, one K block): 1.06-1.3x. |
| 2026-11-04 | FP4 vs FlashInfer MXFP4 | **mixed** | FlashInfer `mm_fp4(use_nvfp4=False, backend="cudnn")` is the only FlashInfer backend that runs MXFP4 on sm120 (cute-dsl needs NVFP4; cutlass/trtllm don't support it). `sm120_fp4_vs_flashinfer.py`, 7 linears x 9 M: DeepGEMM wins 1.1-2.3x on most points (o_proj and dense down M=256-512: 2.0-2.3x; q_b M=512: 2.1x) but **loses** at q_a/kv_a M <= 64 (7.3 vs 9.4 us at M=1), shared gate_up M=32-256 (0.78-0.90x) and kv_b M=512 (0.89x). The losers are K = 7168 with few N tiles (2112/64 = 33 CTAs): per-CTA bytes in flight (~3 MB total) bound the DRAM rate, so they need more CTAs or deeper pipelines. |
| 2026-11-04 | Block-scaled stream-K | tried, off | Ported the deterministic stream-K to the 1D1D kernel (correct in all modes, `DG_SM120_FORCE_STREAM_K=1`, `=2` skips the cost gate for tests). It wins at N = 512 (FP4 M=1: 6.4 -> 5.3 us, M=8: 6.3 -> 5.4; beats cuDNN's 6.5) and loses at N = 2112 (9.6 -> 10.2 us at M=1, 10.6 -> 13.8 us at M=64), so the FP32-fitted cost model is wrong for this kernel. Unrolling the split-reduction loop (4x) did not change that. Off by default until the cost model is refitted on block-scaled sweeps. The model also counted K in elements; it now counts bytes (FP4 packs two per byte). |
| 2026-11-04 | FP4 tile sweep (q_a/kv_a, shared gate_up) | finding | `DG_SM120_FORCE_LAYOUT` sweep of the losing shapes: for q_a/kv_a (N = 2112) the heuristic's 16x64 is already the best candidate (M=1: 9.1 us; 16x128 25, 32x64 10.7, 64x64 25, 64x128 30.5, 128x64 32.2), so tile choice cannot close the gap to cuDNN (7.3 us). What is missing is a narrower tile (BLOCK_N = 32, 66 CTAs, more bytes in flight). It is not a drop-in: the kernel runs 128 math + 128 TMA threads so `setmaxnreg` (per 128-thread warpgroup) sees two clean groups; a 2-warp math group would share a warpgroup with TMA warps. It would need 4 math warps (e.g. 2 N-warps x 2 intra-CTA K-split warps with an smem reduction) or an 8-wide warp tile with unpaired ldmatrix/stores. GPU 7 is shared, so single-run noise is ~20% on small shapes: the shared gate_up M=32 "auto" read 13.9 us here vs 8.5 us in the full benchmark. |
| 2026-11-04 | 1D1D kernel: narrow tiles, 8 warps, stage staging | done | BLOCK_N = 32 tiles split each K block's MMA steps over 2 K-warps (4 math warps stay one warpgroup for `setmaxnreg`), reduced through smem. 64x128 / 64x256 tiles run 8 math warps as 2 x 4 (ncu: b12x runs 8 warps on the same tile; with 4, `stall_wait` was 2x). The whole stage (all K steps' fragments and scales) is copied into registers and released *before* the MMAs, so all stages refill while one computes (64x128 FP4: 0.93 -> 0.70 us per K block). 200x bitwise determinism. |
| 2026-11-04 | PDL in both GEMM kernels | done | The producer pre-issues B/SFB of the first `kNumStages` K blocks of its first tile before `cudaGridDependencySynchronize()` (transactions expected without arrival; A/SFA arrive after the wait). Trigger: by the producer if all K blocks fit the pipeline, else by the math warps at mainloop end. Rejected: trigger at start (slower everywhere), never, always after loads (dense gate_up 21.0 vs 19.5 us), extra L2 prefetch (neutral/worse; full-tile prefetch thrashed L2: K=16384 59.3 vs 46.8 us). Normal GEMMs launch `min(SMs, tiles)` CTAs. Stream-K workspace is per stream and persistent. Ported to the FP32 1D2D kernel: `test_fp8_fp4.py`, `sm120_mqa_check.py` pass; TP8 decode ratios vs Triton/FlashInfer equal or better than round 4 (q_a/kv_a M=1 3.35x, was 2.89x). |
| 2026-11-04 | FP4 64-byte K blocks, PSUM, BF16 | done | FP4 may use 64-byte K blocks (`BLOCK_K = 64`, swizzle 64B), twice the stages: wins at large M (128x128x64 at M=4096). Block-scaled m-grouped contiguous now supports the PSUM layout (`use_psum_layout`, `expected_m_for_psum_layout`). **BF16** runs on the same 1D1D kernel (`kIsBF16`, `KernelNoSF`, `m16n8k16`, no scale loads): `bf16_gemm_nt` (when cuBLASLt is not used: deterministic mode), `m_grouped_bf16_gemm_nt_contiguous` (incl. PSUM) and `m_grouped_bf16_gemm_nt_masked`. `sm120_blockscale_check.py --mode all --grouped --repeat N` covers fp4 / fp8g32 / fp8g128 / bf16, normal / contiguous / PSUM / masked, K tails, forced layouts (64x128x64, 16x32, 64x256), forced stream-K, PDL on: all OK. |
| 2026-11-04 | 1D1D cost model refit | done | Sweep of every layout x stream-K on 63 FP4 shapes (`sm120_sweep_blockscale.py`, graph + PDL + cold weights; mid-M re-measured with 5 repeats), fit with `sm120_fit_blockscale.py`. New model for data-parallel 1D1D layouts: max(DRAM share x small-box penalty, L2 share, per-CTA K loop x pipeline-depth penalty) per wave, plus the explicit stream-K rule (tiles <= SMs / 8 and a clear K-loop win). Regret vs the best layout: geomean 1.012, worst 1.105 (old model 1.084 / 1.358). The timing autotuner is now opt-in (`DG_SM120_AUTOTUNE=1`). |
| 2026-11-04 | FP4 vs FlashInfer MXFP4 (cudnn, b12x) | **60/63 wins** | `sm120_fp4_vs_flashinfer.py`, interleaved, PDL on both sides, GPU idle: DeepGEMM wins 60 of 63 points, 1.03-2.2x (decode: q_b/kv_b 1.8-2.2x, shared gate_up M=1 4.6 vs 6.4 us). Near-ties: q_b M=256 6.0 vs 5.4 us (b12x), o_proj M=128 7.3 vs 7.1, dense gate_up M=256 20.0 vs 19.9. On the first two, the sweep's best forced layout matches or beats b12x (5.32-5.49 and 6.63-6.97 us), but its margin over the model's pick is within run-to-run noise. b12x and our best layout run the same CTA count (96 / 112), so a real margin needs kernel work, not tile choice. |
| 2026-11-04 | UE8M0 FP8 after the refit | **regression** | `sm120_bench_blockscale.py --tp 8 --baselines`: FP4 is 1.3-2.0x over the best FP8 kernel (shared down 0.92-1.48x), but UE8M0 FP8 (g128/g32) fell behind the FP32-scale kernel at mid/large M (dense gate_up M=4096: 595 vs 402 us; q_a/kv_a M=4096: 312 vs 176). Cause: the 1D1D model was fitted on FP4 only; its depth penalty (up to 3x for 2 stages) never mattered there because FP4 always had a 5-stage 64-byte-K alternative (the 2-stage 128x128x128 FP4 tile measures only 1-14% slower). FP8 has no 64-byte K block, so the model picks 64x64 (647 us) over 128x128 (436 us). Committed in `b419c02` with this known issue. Fix: FP8/BF16 sweep + joint refit, or calibrate the depth penalty on the FP4 bk128/bk64 pairs. FP32-scale FP8 (what TokenSpeed uses) is unaffected. |
| | Open | - | (1) Release a `tokenspeed-deepgemm` wheel from this fork and bump the pin in `tokenspeed-kernel/python/requirements/cuda-thirdparty.txt` (until then, use PYTHONPATH like `sm120_e2e.sh`). (2) DeepSeek-V3.2 TP8 E2E, which needs GPUs 0-7. (3) Optional kernel work: UE8M0 block-scale MMA path and bigger tiles for prefill margin over FlashInfer. | Gate added: DeepGEMM must beat the TokenSpeed sm120 fallbacks (Triton/FlashInfer) per op before Phase 5 integrates it. cuBLAS parity alone doesn't justify integration. Needs triton + flashinfer + tokenspeed-kernel installed into `.venv` on the remote (none are there yet). |

| 2026-11-05 | UE8M0 FP8 regression | fixed | Reworked the sweep/fit to be mode-aware (`--mode fp4/fp8g32/nvfp4`, everything in bytes) and fit all three modes jointly (`sm120_fit_blockscale.py` reports per-mode regret, so an FP4 gain cannot hide an FP8 loss). New `k1D1D*` constants + cost table. Per-mode geomean regret 1.02 (worst layout 14%). UE8M0 FP8 at large M is back to ~1.0x of the FP32-scale kernel (dense gate_up M=4096: 406 vs 402 us, was 595); FP4 unchanged (71/71 vs the Triton/FlashInfer baselines; 59/63 vs FlashInfer MXFP4, the rest ties >= 0.97x). |
| 2026-11-05 | NVFP4 | done | FP4 with 16-element UE4M3 scales (`scale_vec::4X`), keyed on `gran_k == 16`. Host: `get_mn_major_tma_aligned_packed_ue4m3_tensor` packs FP32/E4M3 values to UE4M3 int words; `layout.hpp` accepts raw or pre-packed scales on arch 12. Kernel: a K step takes one whole scale word; the K-warp keeps only its own words in registers (fixed a local-memory spill on narrow tiles). `sm120_blockscale_check.py --mode nvfp4` passes (normal / contiguous / PSUM / masked / epilogue / determinism, forced layouts + stream-K). vs FlashInfer NVFP4 (b12x / cute-dsl / cudnn / cuBLASLt): wins ~40/63; losses are mid/large M within a few %. NVFP4 runs ~0-12% slower than MXFP4 here (more scale bytes), so MXFP4 stays the default where a model allows it. |
| 2026-11-05 | BF16/FP4 epilogue | done | `D = alpha * (A @ B^T) (+ C)`, BF16 or FP32 output, on the 1D1D kernel. `bf16_gemm_nt` takes `alpha` and `c`; the block-scaled entry points pass `alpha`. Accumulation reads D, adds in FP32, rounds once. `sm120_blockscale_check.py` epilogue cases (alpha, accumulate, C-aliases-D, FP32 out) pass for every mode. |
| 2026-11-05 | tf32_hc_prenorm_gemm | done | New `sm120_tf32_hc_prenorm_gemm.{cuh,hpp}` + `TF32MMA` (`m16n8k8`). Same TMA pipeline / K-split as SM90, `mma.sync` TF32 instead of WGMMA, D and sqr_sum written from registers. `test_hyperconnection.py` passes (< 1e-8 vs the TF32 reference); ~1.2 TB/s at large M. |
| 2026-11-05 | einsum / fp8_bmm | done | `bmk,bnk->mn` accumulates a BF16 GEMM per batch slice into FP32 D; `fp8_bmm` runs one FP32-scale GEMM per batch (so `fp8_einsum bhr,hdr->bhd` works). Both match the reference (bmk,bnk->mn to ~1e-6 BF16 / 1e-12 FP32; fp8 einsum ~7e-4). The arch-12 `fp8_bmm` else-branch previously fell through to the SM90 WGMMA kernel. The MN-major expressions (`bhd,hdr->bhr`, `bhd,bhr->hdr`) are still unported. |
| 2026-11-05 | MQA head dim 64 | done | `warp_mma_16xNx128_ss` -> `warp_mma_16xNxHd_ss<N, kHeadDim>` (128 or 64) with the matching TMA swizzle. The contiguous/prefill kernel + launcher accept head dim 64 (`sm120_mqa_check.py` D=64 cases pass, ~1e-14); the paged kernel stays at 128 (its KV gather has its own swizzle assumptions). FP4 MQA is still unported. |
| 2026-11-05 | Commits | done | DeepGEMM `nctu6`: `eff4cea` (joint refit), `eb61610` (NVFP4 + epilogue + spill fix, kernel), `4659181` (NVFP4/epilogue host), `bfb637c` (tf32 hc), `5dfd734` (einsum), `38ef2a2` (MQA head dim 64). Local, not pushed. |
| 2026-11-07 | FP4 ties vs b12x (task 12) | **3 of 4 won** | Early PDL trigger for FP4 grids on <= 60% of the SMs (`76dc545`). 63-point sweep, 3 runs per build, best of 3: geomean 1.336x -> 1.399x, wins 59 -> 62 of 63, worst 0.965x -> 0.979x. q_b M=256 1.059x, o_proj M=128 1.116x, dense down M=128 1.107x; dense gate_up M=256 stays 0.98x (L2 -> SM bound). Section 9.15. |
| 2026-11-07 | DSA indexer top-k on SM120 (task 13.1) | **done** | In-tree CuTe DSL DeepSelect fits 93 KB on SM120 (compact profile), identical selections to the full profile and to `torch.topk`. V3.2 prefill and large-batch decode select with it; V4.1 FP8 `index_v4` rows go native on SM120 (`deep_gemm_sm120_dsv41_*`). Measurements, ruled-out options and open items in section 9.16. TokenSpeed commits are local, not pushed. |
| 2026-11-07 | DSA sparse attention on SM120 (task 13.2) | **done (Triton `mma.sync`)** | New `triton_dsa_mma_decode` / `_prefill` for the dense BF16/E4M3 latent cache: 27-146x faster than `triton_dsa_*` on 64 buckets (16-128 heads, decode batch 1-128, prefill chunks 128-8192), same error vs a float32 reference. About 25-30% of the tensor-pipe roofline; a CuTe DSL rewrite is open. Section 9.17. TokenSpeed commits are local, not pushed. |
| 2026-11-07 | End-to-end V3.2 DSA check (task 13.3) | **done, no SM120 failure** | No V3.2/GLM-5/dots3-note checkpoint exists on aws-raichu, so a 4-layer DeepSeek-V3.2 config with real dims (128 heads, q 1536, kv 512+64, indexer 64x128, top-k 2048, YaRN) and random weights ran through the real `DeepseekV32ForCausalLM` and the in-process `Engine`. The registry picks `deep_gemm_dsa_*_topk` + `triton_dsa_mma_*` and CUDA-graph decode captures. Prefill is 82-95x faster (35-72K tok/s vs 393-753), TPOT 2.0-14.4x lower (2.63 ms vs 5.40 ms at batch 1, 3.85 vs 55.3 ms at batch 32). Prompt logprobs differ from `triton_dsa_*` by a mean 0.14-0.32 nats, less than the 0.33-0.80 between two chunkings of one path. No code change. Section 9.18. |
| 2026-11-08 | Final verification and follow-ups (task 15) | **done** | A: `triton_dsa_mma_*` made batch-invariant (`36211b436`): chunks by width, ordered fold, declared `batch_invariant` / `cuda_graph` / `slot_order`; cost 1.21x geomean, 1.86x worst, still 22-114x over `triton_dsa_*`. B: the intermittent v4.1 eager failure did not reproduce in 6 full runs, 150+ isolated runs, the pre-fix tree or the baseline; evidence in 9.20. Fixed fixed-port collisions in `test_control_server.py` (`990e3e6cd`). Kernel suite 7581 passed, runtime 8994 passed, DeepGEMM surface and four end-to-end models pass. Per-op table in `deepgemm-sm120-design.md` (7.8). Section 9.20. |
| 2026-11-07 | SM100-only leaves on SM120 (task 14) | **done (4 SM120 leaves)** | Hadamard `triton_hadamard_transform_128_mma` (copy roofline), MLA decode `triton_mla_mma_decode_with_kvcache` (14-131x), V4 selected attention `triton_dsv4_mma_prefill` (4-9x, batch-invariant), zero-RoPE DSA on the 13.2 cores (29-134x); K3 default backend fixed for SM120. Caps kept. Section 9.19. TokenSpeed `98d36c7b8`, `51019f24b`, `4bfa4febf`, `1334fcd70`, `5c24472f2`, `252f7686c`, `23c1ef371`; docs `caae09a85`, `30277183e`. |
| 2026-11-08 | Paged FP4 MQA logits, V4.1 serving native (task 16) | **done** | `sm120_fp8_paged_mqa_logits` takes the K dtype as a template parameter: FP8 rows with a float scale (unchanged) or MXFP4/MXFP8 rows with a packed UE8M0 word, applied by the block-scaled `mma.sync` (`6b2df9e`). Dequantized reference: FP32 logits ~2e-15, BF16 ~1.5e-6; `test_attention.py` 159 cases with the MX formats on SM120, no spills. MXFP4 is 1.0-1.74x faster than the FP8 kernel at the same shapes; MXFP8 at 128 Q rows spills and is rejected. TokenSpeed scores the 68-byte `v41` rows natively (`f01d594a5`, `76b2134a7`; docs `6d4188f43`): 2.2-88x over `triton_dsv41_index_topk` on every pass, so the cache format stays. Section 9.21. |
| 2026-11-08 | Tensor-core headroom of the SM120 attention kernels (task 17) | **done** | Triton: BF16 tiles prefetch only the next slots: DSA BF16 1.24x, V4 prefill 1.24-1.31x faster, E4M3 unchanged (`6e75d3e77`). CuTe DSL cores `cute_dsl_dsa_mma_*` (`bf5bbc37f`): 1.91x (E4M3) / 1.16x (BF16) geomean over the Triton core on 64 buckets, none slower, bitwise batch-invariant (section 9.22). dots3-note selects its DSA cores by the `strided_kv_cache` feature instead of pinning `triton` (`9bf202435`). CuTe DSL `cute_dsl_dsv4_mma_prefill` (`2e35979b7`), from 17 heads: 1.13x geomean (0.89-1.63x), bitwise batch-invariant; CuTe DSL `cute_dsl_mla_mma_decode_with_kvcache` (`58fe2ef22`), BF16/E4M3: 1.46x (0.97-2.31x), 1.84x on 4/8-row blocks (section 9.23). TokenSpeed commits local, not pushed. |
| 2026-11-09 | Smaller SM120 follow-ups (task 18) | **done (5 items)** | (1) V3.2 DSA prefill, tiny chunks: `dsa_prefill_prepare` (one Triton launch for quantization, weight folding, lengths and key gather, bit-identical to the framework ops) + DeepSelect `index_base`: host time 360 -> 118-124 us, 1.35x over the Triton leaf at 128 tokens (was 0.52x), 1.35-98x at 1-8192 tokens (`9850d5ddb`, `0375fbd7e`). (2) `dsa_index_candidates` selects with DeepSelect over each row's owned length: 1.25-4.7x over the `torch.topk` leaf, 3-64x over Triton (`c13e40aa7`). (3) Batched BF16 `einsum` (`bhr,hdr->bhd`, `bhd,hdr->bhr`), bit-identical to the launch per head: 0.21x of the per-head time, 52 of 56 points within 3% of cuBLASLt (`4ec333b`). (4) V4.1 decode `triton_dsv41_mma_selected_attention` reads the packed rows: 1.66x geomean (1.01-3.67x); `dsv4_decode` direct reader only for 16 heads from 32 tokens (`8bacf076b`). (5) `test_whisper_cross_kv_write`: a dead CUDAGraph freed by the cyclic GC inside a capture; `gc_paused_for_capture` (`10be2bcb8`). Section 9.24. |
| 2026-11-09 | Final verification of tasks 16-19 (task 20) | **done, 0 failures** | Doc numbers reconciled (FP4 vs the best FP8 kernel 1.4-2.3x and vs Triton/FlashInfer block-FP8 1.5-11.4x from a fresh 72-point run; BF16 einsum 0.21x = 4.8x; MLA decode 0.97-2.31x), Current state refreshed. TokenSpeed pin at DeepGEMM `fefe6a1`. Kernel suite 7770 passed, 2770 skipped, 0 failed; runtime suite 8997 passed, 0 failed (10 directories); DeepGEMM SM120 surface, fresh cache, `DG_JIT_CHECK_NO_SPILLS=1`: all pass; the pinned source builds like `install.sh` and passes the gating tests; four models answer correctly. Section 9.25. |
| 2026-11-09 | Mega MoE / gate / mHC on SM120 (task 21) | **done (MoE, one rank); gate, mHC, EP=2 open** | None of the three needs hardware SM120 lacks: TMEM, UMMA, 2-CTA MMA and symmetric memory are how SM100 fuses them. GPUs 6 and 7 do peer access (`PIX`, 54 GB/s, kernel peer loads OK). New W4A8 path in the SM120 1D1D kernel (`kind::mxf8f6f4 e4m3.e2m1`, E2M1 in bits [5:2] of a byte, probed): bitwise equal to the FP8 path on converted weights, 1.4e-6 vs FP32, 1.35-1.72x faster at decode/mid shapes. `fp8_fp4_mega_moe` runs on one SM120 rank as dispatch + two W4A8 grouped GEMMs + SwiGLU + combine with the SM100 arithmetic (0 / 1.8e-10 vs a PyTorch replay). PSUM scheduler rebuilt after the grid-dependency wait (24 B spill gone). Against Marlin W4A16 (cold, graph): 0.82x at 1 token, 0.97-0.99x at 4-64, 1.01-1.09x at 256-1024, 1.59-1.87x at 4096-8192. TokenSpeed keeps the leaf off SM120: its MegaMoE path needs EP > 1 and V4/V4.1 experts do not fit one GPU. Section 9.26. |
| 2026-11-09 | Mega MoE at EP=2 over CUDA IPC, Triton MXFP4 fix (task 21 continued) | **done (EP=2, TokenSpeed leaf, IMA fix); 1-token gap, gate, mHC open** | `fp8_fp4_mega_moe` serves several SM120 ranks (`1a5c1cf`): push (peer stores of top-k lists, FP8 rows and scales), epoch barrier (system-scope release/acquire, trap after `DG_SM120_COMM_TIMEOUT_S`), dispatch over every rank's pairs, local W4A8 GEMMs, return (peer stores of the L2 rows), barrier, combine in top-k order. Bitwise equal to one rank over all experts, 0-2e-11 vs PyTorch, graph replay, PDL, ranks without tokens; peer *loads* were 100-300x slower (0.5 ms gather, 1 ms combine) and are not used. V4/V4.1 at EP2 against Marlin W4A16 + its collective at the same served tokens: 0.95-1.05x at 1-1024 tokens per rank, 1.28-1.39x at 2048-4096. TokenSpeed registers `deep_gemm_sm120_mxfp4_mega_moe_apply` for EP 2 (`695c15770`). Triton MXFP4 IMA at V4.1: 32-bit `expert_id * rows * cols` wrapped past expert 182/364 (`8afd973bc`). Section 9.26. |
| 2026-11-09 | Real-weight checks (task 22) | **done (V4.1 real weights with 128 of 384 experts, Moonlight MLA); V4 and full models open** | V4 checkpoint not loadable (TokenSpeed has no `DeepseekV4ForCausalLM` since `0c1bc3ed3`). Real V4.1, 40 layers, first 128 experts (`num_checkpoint_experts`, TokenSpeed `6d6702bae`): DP2/EP2 picks `deep_gemm_sm120_mxfp4_mega_moe_apply`; the float32 reference gives the same single-layer error on every path (layer 0: 2.8%, layer 2: 3.7-3.9%); the paths differ by 0.7% at layer 0, amplified to about 50% by layer 39. Marlin TP2 is faster than MegaMoE DP2/EP2 end to end (prefill 10.3K vs 3.9K tok/s, TPOT 23.5 vs 34.4 ms; not profiled). MegaMoE scratch is outside the KV planner (OOM at util 0.93). Moonlight-16B MLA correct, TPOT 2.3-3.2x at 3K x 8; RoPE fix `e4fc05813`. Section 9.27. |
| 2026-11-10 | Cluster tile sharing for the dense gate_up M=256 FP4 tie (task 23) | **closed, not won; nothing shipped** | Tie reproduced (GPU 7 idle, 3 runs: DeepGEMM 19.2 us, b12x 19.5-19.6 us, 1.02x). Built B-slice sharing across 2- and 4-CTA clusters in `sm120_fp8_fp4_gemm_1d1d` (`kClusterSize` template parameter, bitwise equal output, no spills) in two forms, push (bulk copy) and pull (`ld.shared::cluster`): 63.7 / 146 us (push, cluster 2 / 4) and 45.5 / 92 us (pull) against 20.0 us unshared. A DSMEM probe (`scripts/sm120_cluster_probe.cu`) shows why: pushes cap at 5 GB/s per CTA (bulk copy, `st.async` and plain stores alike), pulls at about 33 GB/s per CTA, a remote load takes 95 ns and a remote mbarrier round trip 440 ns, and the 3-stage pipeline cannot cover those hops. With the sharing hops removed (wrong data, cluster launch only) the kernel reaches 17.5-17.6 us, so the best case was 1.13x. Raster order does not apply (one wave). The kernel is unchanged; the variant is kept as `scripts/sm120_cluster_tile_sharing.patch`. Section 9.28. |
| 2026-11-10 | V4.1 serving gap, MegaMoE scratch, DP error masking, CuTe V4 prefill selection (task 24) | **done for A-D (A: prefill won, TPOT structural + open MoE glue); E not attempted** | Profiled real V4.1 (40 layers, 128 experts, GPUs 6+7, torch profiler, graph-aware). Both paths spent most decode time in the dense 1x32 UE8M0 FP8 projections on the Triton block-FP8 kernel (Marlin TP2 21.5 ms of a 23.6 ms step; MegaMoE DP2 50.2 ms of kernel time, summed over two streams, in a 49.4 ms step). New TokenSpeed leaf `deepgemm_sm120_mm_mxfp8` on the block-scaled 1D1D kernel, weight scales packed once, activation quantized straight into the packed layout, batch invariant through deterministic mode (`26444a3`: no stream-K, no BLOCK_N=32 K-warp split): 2.2-11.7x over the Triton leaf on the six V4.1 projections. MegaMoE row scratch is one persistent per-device buffer grown by the warmup before the KV planner measures (`fefd9de`). End to end (2 x 2004 tokens, prefill graphs off): Marlin TP2 10.2-10.7K -> 14.5-14.8K tok/s, TPOT 23.0-25.1 -> 10.4-10.7 ms; MegaMoE DP2/EP2 11.8-13.3K -> 18.1-18.6K tok/s (1.25x over Marlin), TPOT 48.4-50.3 -> 13.8-14.0 ms (1.31x behind). TPOT gap per step: dense +2.6 ms (DP reads every dense weight on each GPU, structural), MoE glue +2.2 ms (two barriers, push, dispatch, gather, SwiGLU, return, combine; the two grouped GEMMs tie Marlin at 3.1 ms). Util 0.93 with 128 experts runs; 192 experts now fail at startup with the KV-planner message instead of an OOM mid-prefill. Section 9.29. |
| 2026-11-10 | Attention headroom and MegaMoE glue (task 24, item E) | **done: planner shipped, the rest measured and closed** | TokenSpeed `3de153200`: `cute_dsl_dsa_mma_*` `plan_launch` models the launch in SM waves (a CTA fills an SM, one CTA more than SMs costs a second wave), picks 1/2/4/8 CTAs per token with the fewest `waves x chunks per CTA`: 1.05x (BF16) / 1.04x (E4M3) geomean over 140 decode shapes of 1-192 tokens, up to 1.63x, worst 0.97x (noise), bitwise equal rows, 1.02x at 128-2048 tokens. Ablations show the BF16 core is not exchange-bound (removing the whole partial-score exchange: 1.19x). Lost or not selectable: bulk `cp.async.bulk` gathers (0.9x), 1-stage ring x 2 CTAs/SM (0.55-0.82x), 16 DSA chunks (1.3-1.46x at 1-8 tokens, 0.8-0.9x at 4-64), 8-warp V4 CTA at 16 heads (0.76-1.10x of Triton), a wave planner for V4 (0.97x). MegaMoE glue measured at 25 us per layer on the critical path, fusion bound 9 us (0.36 ms, 2.6% of TPOT), not built. Section 9.30. |
| 2026-11-10 | Intermittent `test_deepseek_v41_model[eager]` failure (task 25) | **closed: not reproducible, nothing changed** | Original log gone. compute-sanitizer memcheck / racecheck / initcheck / synccheck on one eager run, with and without the caching allocator: 0 kernel errors (the 40 memcheck lines are `cuGetProcAddress` version probes). Poison allocator (NaN, large, per-allocation byte) and `StreamFork` delays of 2e6-2e8 cycles on the aux and main streams: 0 failures in 450 iterations, outputs bit-identical. Stress: 400 eager iterations idle, 400 beside a GEMM loop, 150 beside the kernel suite, 46 fresh runs of the whole file (all pass). A one-off `test_unnormed_solutions_agree_byte_for_byte` failure in the kernel suite did not repeat either (200 + 70 replays, full suite 7905 passed). Section 9.31. |
| 2026-11-10 | Final verification of tasks 21-25 (task 26) | **done, no regression** | TokenSpeed pin bumped to `b6247c2` (`0d2f40c5b`; the `deepgemm_sm120_mm_mxfp8` leaf registers). Kernel suite 7898 passed, 2770 skipped, 0 failed; runtime suite 9003 passed, 0 failed; DeepGEMM SM120 surface (including `test_mega_moe_sm120.py` and the 2-rank `test_mega_moe_sm120_ep.py`) passes with `DG_JIT_CHECK_NO_SPILLS=1` and fresh caches; the four end-to-end models answer correctly. Section 9.32. |
| 2026-11-10 | Docs check, pre-commit, push (task 27) | **done** | Docs reconciled with tasks 21-26 (suite counts, V4.1 real-weight numbers, EP2 MegaMoE vs Marlin ratios, "Current state", open list, report verification section naming the task 26 run). `pre-commit run --all-files` run in both repos before pushing. Pushed `nctu6` to origin, DeepGEMM first, then TokenSpeed (its pin references a pushed DeepGEMM commit). `WEEKLY_REPORT.md` not tracked. |
| 2026-11-10 | MegaMoE under tensor-parallel attention (task 29) | **done: works and prefills 1.11-1.12x faster than Marlin TP2, TPOT ties; stays opt-in** | `moe_plan(ep_token_layout=)`: `scattered` (attention DP, the push / return protocol) or `replicated` (attention TP): the SM120 leaf runs the one-rank kernel over its shard of the experts (an expert range with the other ranks' routes masked in the staging pass, or a slice of every expert under MoE TP) and the TP all-reduce sums the partials; no new kernel, DeepGEMM API unchanged. DeepGEMM `343d194`: the one-rank scratch grew at every warmup size and kept every predecessor (OOM in the first real run); it is now sized by the worst routing, test `test_scratch_is_sized_once`. Layer tests on GPUs 6/7 (EP and TP shards vs the scattered protocol 0.0021 / 0.0037 relative, vs a float reference 0.047, graph replay, all-remote routes). Real V4.1, 128 experts, median of 9: Marlin TP2 15.1K tok/s, 10.69 ms; MegaMoE attention TP2 + MoE TP2 17.0K (1.12x), 10.66 ms; + EP2 16.8K (1.11x), 11.22 ms (the expert range leaves one rank with more of the 12 routes: all-reduce 0.8-1.1 vs 0.25 ms); DP2/EP2 19.1K, 14.02 ms. `auto` stays Marlin (no TPOT win). Section 9.33. |
| 2026-11-10 | Persistent / stream-K DSA CuTe core (task 30) | **done: stream-K schedule shipped (1.03x), persistent CTAs and 16 chunks closed** | TokenSpeed `e51c66452` (README `084396fc3`): `sparse_mma_attention(..., num_splits, full_tokens)` launches one grid row per whole token, then `num_splits` rows per split token, so the idle tail of the last wave runs short split CTAs; `plan_launch(tokens, heads, num_sms, fp8)` keeps the uniform plan as the base and takes a stream-K plan when a wave model is 8% ahead. HEAD vs new, interleaved, CUDA graph: 140 decode shapes x 2 dtypes 1.0315x geomean (BF16 1.034, E4M3 1.029), worst 0.970, best 1.378; 64 buckets 1.0173x, worst 0.986, best 1.300; rows and LSE bitwise equal for every plan; one-request decode 20.2 us BF16 / 13.3 us E4M3 (was 20.3 / 13.1). Found on the way: a 1D program-indexed first version cost the E4M3 kernel 3-6% (extra live value in the state store), fixed by scheduling by token. Closed: persistent CTA with next-item prefetch (an all-empty launch is 3.4-4.9% of the many-CTA launches, 26-35% of the 8-CTA ones where there is no second item; BF16 query is in 32 of 255 registers, E4M3 query tile 18-36 KB beside a 99 KiB ring); chunk states on chip across a run (left fold, not bit-exact; a hop per CTA > the 2 us combine); 16 chunks (oracle plan: 1.31x at 1-4 tokens, 0.91-0.95x at 13-64, worst 0.81x); combine prefetch (nothing). Section 9.34. |
| 2026-11-10 | V4 attention at 16 heads off Triton (task 31) | **closed for 16 heads; 8-warp CTA and V4 wave planner shipped for more than 16 heads (1.095x)** | TokenSpeed `d55bb24bd` (READMEs `75ba90021`). Ceiling first: the Triton leaf gathers at the L2 rate (3.94 TB/s at 2048 tokens) and the CuTe core with the MMAs removed takes 325 us (8 warps) / 348 us (4 warps) against Triton's 341 us total; issuing the `cp.async` for only every 2nd / 4th token (perfect overlap, no cost) gives 369 / 343 us (8 warps, 640 rows) and 111 / 104 us (128 rows) against 341 / 92 us, so two or more tokens per CTA cannot win and was not built (bits also force per-token tiles: only gathers could be shared). Real overlap not measured (window rows: 127 of 128 shared; compressed rows unmeasured, GPU 6 busy and the real V4.1 harness gone). New CTA shape: 8 warps for a 16-head CTA with the 4-warp bits (rows split between the two warps of a channel quarter, PV channels 8 ways); bitwise equal over layouts, splits, stream-K mixes, int32/int64 ids. At 16 heads vs Triton: 1.13x at 1-16 tokens (640 rows), 0.92 / 0.87 / 1.01 / 0.84x at 64 / 128 / 512 / 2048, 0.73-1.32x at 128 rows: stays on Triton. V4 planner with width, both head tiles and a stream-K tail (CTA time 5.0 + 1.44 us per tile at 16 heads, 7.8 + 1.9 at 32; 0.5 us per split CTA and combine): 165 points of 17-128 heads vs HEAD 1.095x geomean, 0.893-1.655x; brute-force uniform plans were only 1.027x ahead of the old rule, so the tail and the width carry the gain. Tests: 52 in `test_dsv4_mma_sm120.py` (bitwise over tile x layout x splits x `full_tokens`, plan validity, wave boundary), attention directory 920 passed. Section 9.35. |
| 2026-11-10 | The 7 passed-to-skipped kernel-suite tests, task 29-31 regressions (task 32) | **done: 7 = cuobjdump off PATH (fixed); 3 regressions fixed** | The 7 are the SASS checks of `test_allreduce_sandwich_rmsnorm.py`, skipped when `cuobjdump` is not on `PATH` (task 26's venv-only `PATH`); the tests now find it in `CUDA_HOME`. Not the DeepGEMM pin: the task 25 TokenSpeed tree on the new build gives task 26's 7898 / 2770, and the `62af89e` build has no passed-to-skipped test. `test_deepseek_v4_config.py` stand-ins lacked `ep_token_layout` (`689f9bd6e`); `ep_token_layout` is a required argument of `moe_plan` / `MoELayer` (`e6f9df2e4`); `test_gemm_bf16[triton_mm_bf16_skinny]` failed on cuBLAS's BF16 split-K reference (`3bec4e6f3`), `test_triton_adapter` on a stock-triton import (`a88b744a7`), both already failing at the baseline. Kernel suite 7938 passed, 2764 skipped, 0 failed. Section 9.36. |
| 2026-11-10 | Final verification of tasks 29-32 (task 33) | **done; one bug found and fixed (`test_projection_tp[tp2]`)** | DeepGEMM synced and rebuilt ("build OK"), mirror resynced. The task 32 side finding was a real bug: `flashinfer_fp8_blockscale_quantize_prepacked` called the TRT-LLM 1x128 quantizer for row counts divisible by four, an op with no image after SM100, so on SM120 the launch failed silently and a later `cudaGetLastError` (the A2A kernel's) reported "no kernel image"; the helper uses it only up to arch 10.9 now (TokenSpeed `05937a09b`), the test's reference computes the TRT-LLM arithmetic in PyTorch where the op is absent, and `test_fp8_quant_prepacked.py` guards it. Kernel suite 7952 passed, 2764 skipped, 0 failed; runtime suite 9007 passed, 0 failed; `nvidia/ops/communication` with 2 GPUs 155 passed; token-layout test (2 ranks) passes; DeepGEMM SM120 surface and `test_mega_moe_sm120_ep` / `test_lazy_init` pass with `DG_JIT_CHECK_NO_SPILLS=1` and fresh caches; four models answer correctly. Section 9.37. |
| 2026-11-11 | MegaMoE glue under attention TP (task 34) | **done: SwiGLU fused into the L1 epilogue and the gather into dispatch, bitwise equal; real V4.1 TPOT 10.49 vs Marlin 10.57-10.60 ms, prefill 1.13x; `auto` keeps Marlin** | DeepGEMM `aa1ecfe`. Leaf (graph, cold weights, PDL) 3.7 / 5.6 / 6.9 / 8.0 / 13.4 us per layer at 1 / 2 / 4 / 8 / 16 tokens; bitwise against the unfused build (one rank, `fast_math` on and off; EP2 equals one rank on 7 splits), no spills. Real V4.1: TPOT 10.49 (three runs) against 10.60 before the fusion and 10.57-10.60 for Marlin; a 1.0% win equal to the run-to-run spread, so `auto` stays Marlin (the numerics-envelope, width, batch-invariance and memory gates of 9.33 are also not built). Section 9.38. |
| 2026-11-11 | V4.1 indexer: multi-wave sparse MX kernel and MXFP4 paged bandwidth (task 35) | **sparse kernel shipped (2-6x, bitwise equal), `_sparse_reindex_wins` is one threshold; MXFP4 paged closed as page geometry, nothing shipped** | DeepGEMM `b4eaebf`, TokenSpeed `b5f761118`. The "85 us at any size" was the metadata's per-SM schedule (at most 8 splits per entry: 8 requests = 32 busy SMs at 12.5 us per split) and a gather that ran lane 0..3 per token before the MMAs. New kernel: even share of the KV splits per CTA (runtime-sized, no SM count in the JIT key), 16 warps = 8 row groups x 2 queries gathering their own blocks (one `cp.async` per lane and block) behind two 64-thread barriers per split, gather of split i + 1 between the MMAs of split i, Q fragments in registers. Output bitwise equal to the old kernel. Kernel (MXFP4, 32 heads, 16384-row pool): 8 / 32 / 128 requests 82 / 94 / 252 -> 21 / 45 / 146 us; 128 / 1024 / 4096 queries of one request 140 / 710 / 2770 -> 43 / 243 / 912 us (dense contiguous 160 / 344 / 1039 us for the 32K history). Leaf Reindex (CUDA graph), new default over old default from twice the pool: 0.95-2.9x (two points at twice the pool are 3-5% slower, the rest 1.17-2.9x); one threshold replaces the rows/layout rule. MXFP4 paged 1.07 vs 1.23 TB/s: FP8 head dim 64 on the same 4352-byte pages gets the same 1.07; stage depth, SM count, Q fragments, epilogue, TMA box shape and L2 promotion change nothing, so it is the page geometry. Section 9.39. |
| 2026-11-11 | CuTe DSL V4 kernel that reads the packed rows (task 36) | **shipped: bitwise equal to the workspace route, 1.75x (V4) / 3.12x (V4.1) over it and 1.71x / 1.62x over the Triton direct readers, selected for every head count** | TokenSpeed `2098a0c41`. Ceiling first: gather + dequantize are 16-65% of the V4 workspace route and 30-80% of the V4.1 one, so the bound is 1.3-4.8x. `_SparseMma(packed=...)`: a thread loads its packed bytes (FP8 g64 + BF16 RoPE tail, FP8 g32, FP4 g16 with E4M3 scales) into 20 registers one tile ahead with predicated loads and converts them exactly (E4M3 / E2M1 bits placed in a BF16 word, `mul.rn.bf16x2` by 2^120 or 2^126 and by the scale) into the BF16 ring tile; scores, softmax, PV, fold and combine unchanged, binaries keyed by formats and id dtypes only, `plan_paged` with packed tile times (within 1.007x of a grid of plans). 105 points per model (8-128 heads x widths 128 / 640 / 1152 x 1-2048 tokens): V4 1.75x over the workspace route (1.06-5.14x) and 1.71x over `triton_dsv4_mma_decode` (1.12-2.23x); V4.1 3.12x (1.43-7.93x) and 1.62x over `triton_dsv41_mma_selected_attention` (0.94-3.02x; 1.92x with a compressed cache, 1.16x window-only). Read-back tests cover every E4M3 byte x UE8M0 / E4M3 scale code. Limits: the 32-head packed CTA spills (120-168 B); the version that moves the conversion after PV is 1.1-1.2x slower. Section 9.40. |
| 2026-11-11 | Host path of a 128-token DSA chunk: `fp8_mqa_logits` and DeepSelect (task 37) | **shipped: `fp8_mqa_logits` 8.6 -> 5.6 us, `deepselect_local_topk` 17.2 -> 9.1 us per call; leaf host time 116 -> 87 us (1.34x eager back-to-back); bitwise equal; no end-to-end change on a 4-layer V3.2** | DeepGEMM `f628449`, `6df2878`, `211a7e9`; TokenSpeed `436833133`. `jit->compile` (copy options, format + hash flags and source) ran on every launch: kernels are now found by their template integers (`kernel_lookup.hpp`), the view is one `as_strided`, the env reads do not allocate. DeepSelect: whole tensors, one validation, cached `capacity_for` / `cluster_sizes_for`, SM count only for wide rows; the leaf skips slices and copies of whole tiles. Tensor maps are 0.13 us each (not cached). 44 cases bitwise equal; the 4-layer V3.2 chunk takes ~11 ms of GPU work, so 4 x 29 us is below the 1.4% run-to-run spread. Left: TVM-FFI call + `values` allocation (5.5 of 9 us), `dsa_prefill_prepare` launcher (37 us), `nan_to_num_` (6 us). Section 9.41. |
| 2026-11-11 | The three heavy compute-sanitizer runs (task 38) | **done: 0 kernel errors in memcheck, synccheck, racecheck and initcheck on FP8 masked (64 cases), FP8 k-grouped (126) and `test_mqa_logits` (64); two test-side fixes** | DeepGEMM `0d89c76` (tests only, no kernel change, no rebuild). The earlier runs did not finish because every kernel was instrumented, torch's reference GEMMs included; with `--kernel-name kns=sm120_ --kernel-name kns=pack_fp32_into_ue8m0` (positive control: an output view with half its storage is caught) a k-grouped case takes about 9 s under memcheck. Full lists, every kernel binary launched under each tool (91 masked, 23 k-grouped, 8 MQA); racecheck ran the 21 large cases as one launch each (a full test body took 41+ min). initcheck (unfiltered, it must see the torch writers): `test_mqa_logits` copied the never-written padding columns with `masked_fill` (fixed: the test compares `logits[self_mask]`, the kernel writes every defined element, probe-attributed); the PSUM masked path reads the uninitialized padding rows between groups in `transpose_and_pack_fp32_into_ue8m0` and `sm120_fp8_gemm_1d2d_impl` by design (rows are independent; 0 reports with the padding zero-filled, all 64 cases pass with NaN padding, new test `test_psum_gaps_do_not_reach_valid_rows_sm120`: valid rows bitwise equal for zero vs NaN padding). 18 masked cases were skipped once for GPU memory (another user's job) and rerun clean. Section 9.42. |

| 2026-11-11 | Final verification of tasks 34-38 (task 39) | **done, 0 failures, no code change** | Remote DeepGEMM tree synced from the tracked files of the local head and rebuilt ("build OK", py3.15 and py3.12), `/tmp/ts_mac` resynced; TokenSpeed pin already on the code head `1df3e7f` (the docs commit of this task moves it). Kernel suite (GPU 7, `PATH` with `/usr/local/cuda/bin`) 8011 passed, 2764 skipped, 0 failed; `test_numerics.py` + `test_triton_adapter.py` 24 passed, 2 skipped; two GPUs visible: `nvidia/ops/communication` 155 passed, 41 skipped, the two-rank token-layout test 2 passed (twice); runtime suite (GPU 6) 9007 passed, 0 failed in all ten directories; DeepGEMM surface (`DG_JIT_CHECK_NO_SPILLS=1`, fresh caches) all pass, including the new PSUM-gap test, `test_mqa_logits_launch`, both mega MoE tests, the sparse paged / sparse MQA checks and `test_lazy_init` on 2 processes; the four end-to-end models answer correctly; `pre-commit run --all-files` and the other Lint steps clean on a clean tree. Report sections 1, 3, 5 and 7 and the Open list reconciled. Section 9.43. |
| 2026-11-12 | Image benchmark and report (tasks 40-47) | **done: report in `deepgemm_benchmark.md`; no code change** | Before `nctu6/tokenspeed:e27cee86-20261028` vs after `nctu6/tokenspeed:ef057889-20261112`, GPUs 6 and 7, six models, identical server flags, RAG C = 1 .. 512 and agentX full mode C = 1, 16, 64, one run per point, AIPerf 0.13.0; primary set 70 RAG and 26 agentX runs done, 4 gemma agentX runs not runnable, 30 runs of three before-image sessions never started (harness-wide 112 done, 41 failed or not runnable). Qwen3.8-27B-FP8 output tok/s 1.52x (C=1), 1.28x (16), 1.11x (64), 1.04x (256-512); TPOT 1.53x at C=1, TTFT 1.00x; agentX 1.14x / 1.10x / 1.01x. BF16 controls (Qwen3.6-35B-A3B, gemma-4-31B-it) 1.00-1.03x. The before image fails to start for Qwen3.8-27B-NVFP4 (`fp4_quantize` kernel not found), Qwen3.5-122B-A10B-NVFP4 (flashinfer NVFP4 MoE `fc1_act_global` check) and the V4.1 128-expert subset (`Unexpected V4.1 expert tensor ...experts.128...`); the after image runs all three. V4.1 Mega MoE vs Marlin 1.01-1.17x RAG, 0.97-1.03x agentX. Findings: scheduler KV-event abort (both images, workaround env var), gemma-4 TP1 256K agentX not runnable, gemma-4 gateway parser, `<bos>` for completions, decode batch above the 160-graph default. Section 9.44. |
| 2026-11-13 | Detailed benchmark document (task 48) | **done: `deepgemm-image-benchmark.md` renamed to `deepgemm_benchmark.md` and rewritten from the result JSON; docs only** | Summary table per model x dataset with before / after / ratio of output tok/s, TTFT p50, TPOT p50 at every C (204 values rechecked against `metrics.json`); testing conditions (aws-raichu GPUs 6/7, driver 595.99.02, CUDA 13.2; image IDs; TokenSpeed `e27cee861` / `ef057889f`, DeepGEMM wheel 2.8.0.post20260924 / pin `976f212`, scheduler 0.1.41 in both, verified by MD5 of files inside the images); server and client commands; probes, warmup, run order; per-run tables (req/s, in/out tok/s, TTFT and TPOT mean/p50/p99, ITL p50/p99, ok/failed, effective concurrency, ISL, wall, agentX warmup and validity); failures with error lines and log paths; default-config and repro crash runs; smoke runs; caveats. Correction to 9.44: the before image's default-config session also crashed at C=128. Tables generated by `make_tables.py` + `make_detail_tables.py` (section 9.45) |
| 2026-11-13 | Scheduler KV-event crash (task 50) | **fixed: root cause in the C++ scheduler (`DeviceBoundaryResidency` ignored state checkpoints); default config (KV events on) now completes RAG C=128** | TokenSpeed `d61649596` (fix + coordinator test + scheduler scenario that aborts before the fix), docs `62f4ec1b5`. A hybrid model's state checkpoint can outlive the KV of its boundary (eviction cannot take a checkpoint a request pins); the boundary then read as empty, `DrainKvEvents` erased its descriptor, and the checkpoint's later eviction hit `markKvEventBoundaryForReconcile`'s check. A cached checkpoint now keeps the boundary partially resident. Rerun: after image + fixed scheduler module mounted, Qwen3.8-27B-FP8 RAG C=64 / 128, KV events on: 500 ok / 0 errors each, no fatal line (unfixed: abort 35 s into C=128). C++ 610 passed, scheduler Python 248, runtime cache/execution/KV-event files pass. No scheduler version bump. Section 9.46. |
## 9.5 Sparse MQA logits SM120 port plan (in progress, 2026-11-05)

Target: `fp8_fp4_sparse_mqa_logits` + `fp8_fp4_paged_sparse_mqa_logits` +
the two metadata entry points, on arch 12. Ground truth: `test_sparse_mqa_logits`
in `tests/test_attention.py` (MXFP4 + MXFP8, contiguous + paged + unaligned-ks,
num_heads 8/12/20/32, sparse_block_kv 8/16). No SM90 template exists (only
`sm100_sparse_mqa_logits.cuh`), so the compute core is a from-scratch SM120 kernel.

Math per (q_token, kv_token): `logits[q, slot] = sum_h relu(Q[q,h,:] . KV[kv,:]) * weight[q,h]`,
with Q and KV block-scaled (per-32-element UE8M0, int32 words); output BF16, written
to a compressed column via the metadata slot map.

Reuse unchanged:
- `scheduler/sm100_sparse_mqa_logits_metadata.cuh` and its launcher: pure CUDA
  scheduling, no TMEM/cluster/UMMA (grep confirmed 0 hits). Rename to a
  shared/`smxx` name or include as-is from an sm120 launcher.
- `layout/sparse_mqa_logits.cuh` structs (MetadataHeader, KVSplitHeader,
  KVBlockInfo, ScheduleEntry, QBlock). `SharedStorage` must be re-derived (drop
  the TMEM barriers / `tmem_ptr_in_smem`; keep Q/KV/SF/weights smem + barriers).
- The two KV accessors (`ContiguousSparseKVAccessor` TMA, `PagedSparseKVAccessor`
  cp.async gather from the fused cache) and `get_swizzled_kv_chunk_idx`.
- The ReLU x weight head-reduction epilogue and the slot-mapped compressed store
  (identical to the SM100 math-warpgroup tail and to the dense SM120 MQA epilogue).

Rewrite (the SM100 core uses UMMA m128nN with SF in TMEM; replace with mma.sync):
- Per KV split: one math warp owns 16 KV rows; run block-scaled `mma.sync`
  (FP8BlockScaledMMA m16n8k32 / FP4BlockScaledMMA m16n8k64) of
  `[16 KV, kHeadDim] x [BLOCK_Q*kNumHeads, kHeadDim]^T`, accumulating in registers
  (like `warp_mma_16xNxHd_ss`, but block-scaled). kHeadDim 128 -> 4 FP8 / 2 FP4
  K steps; stage the Q and KV MX SF words in smem and read them with the SFA/SFB
  lane map (g = lane/4, t = lane%4) already used by the 1D1D GEMM.
- Q is loaded once per q-block (TMA); KV is gathered per split (the accessors);
  SF words likewise. BLOCK_Q = 2 (two packed Q slots), kNumMaxHeads = 32.

Host: `sm120_sparse_mqa_logits.hpp` mirroring the SM100 launcher (TMA descs for
Q/SF-Q/weights, KV via accessor), smem sized from the new SharedStorage, grid =
num_sms. `get_sparse_mqa_split_kv` (640 fp4 / 512 fp8) and `get_num_metadata_bytes`
are reusable. Dispatch: `attention.hpp` arch-12 branches (all four entry points),
and the metadata launch is arch-agnostic.

Test: let `test_sparse_mqa_logits` run on arch 12 (the generators treat arch 12
as SM90-like for FP8; FP4 Q needs the `kPackedFP4` path enabled for arch 12 in
`generators.py`). Tolerance matches the dense MQA check.

Progress: `mma::sm120::warp_mma_16xNxHd_block_scaled_ss<N, kHeadDim, kIsFP4>`
landed in `mma/sm120.cuh` (the block-scaled compute primitive: `KV[16,hd] @
Q[N,hd]^T` with MX UE8M0 scales, SF lane map copied from the 1D1D GEMM). The
metadata-scheduler smem bound is now device-based. Remaining: the kernel body
(producer + KV-copy + math warps reusing the accessors), the launcher, and the
arch-12 dispatch.

Risk retired: the SF lane map the helper uses (SFA t&1 -> row g / g+8, SFB t==0
-> col g; FP4 16-bit byte0=K0-31/byte1=K32-63) is the one the 2026-11-04 probe
already confirmed on hardware, so the compute primitive rests on validated facts.

Kernel skeleton (for `sm120_sparse_mqa_logits.cuh`, synthesizing the dense SM120
MQA kernel's compute with the SM100 sparse kernel's metadata-driven schedule):
- Warp roles (collapse the SM100 5-role layout to 3; no UMMA/SF-transpose warps):
  (a) one Q+metadata producer warp: per ScheduleEntry, TMA Q / SF-Q / weights into
      a Q-stage ring (as the SM100 producer does), and cp.async the KVSplit header +
      KVBlockInfo array into a KV-stage ring; (b) KV-copy warps: the SM100 gather
      (contiguous TMA or paged cp.async via the two accessors) into smem KV + SF-KV;
      (c) math warps: per KV split, each warp owns 16 KV rows and calls
      `warp_mma_16xNxHd_block_scaled_ss<BLOCK_Q*kNumHeads, 128, kIsFP4>` with the
      staged KV/Q SF words, then the dense kernel's ReLU x weight head reduction.
- SharedStorage: reuse the dense kernel's Q/KV/weights rings, add smem SF-Q
  `[qstage][BLOCK_Q*kNumHeads words]` and SF-KV `[kvstage][SPLIT_KV words]` (int32,
  one per 128 elements = one per token here since hd=128), plus the metadata rings
  (`kv_split_headers`, `kv_block_infos`, `q_blocks`) from the SM100 layout. Drop the
  TMEM barriers / `tmem_ptr_in_smem`.
- SF staging: the SF word of KV token `r` goes to smem `sf_kv[r]`; the helper reads
  `a_sf[a_row + g (+8)]` with `a_sf_stride = SPLIT_KV` (words laid out `[word][row]`,
  one word for hd=128). Q SF likewise `b_sf[g]`, stride `BLOCK_Q*kNumHeads`. No
  transpose (unlike SM100's UTCCP-into-TMEM path).
- Store: the slot map from the SM100 math tail. `accum` layout is the dense
  kernel's (`accum[i*kNumAccumPerReduce ...]`, i over BLOCK_Q, reduced to `v_0`/`v_1`
  for KV rows `warp_offset + lane/4` and `+8`). Map each to a compressed column via
  `packed_slot_offsets` / `q{0,1}_slot_base` from `kv_block_infos` /
  `kv_split_headers`, writing `output_rows[q][ (q_slot_base + q_slot_offset) *
  SPARSE_BLOCK_KV + token_in_kv_block ]` when the slot != kInvalidSparseSlot. No
  FP32 per-KV `scale_kv` multiply (the MX scale is applied in the MMA).
- Launcher `sm120_sparse_mqa_logits.hpp`: mirror the SM100 one (TMA descs for Q /
  SF-Q / weights; KV via accessor), size smem from the new SharedStorage, grid =
  num_sms, `get_sparse_mqa_split_kv` 640 fp4 / 512 fp8. Dispatch in `attention.hpp`
  (four entry points), flip `test_sparse_mqa_logits` to `>= 10`.

Effort: the compute primitive is landed and validated; the remaining kernel body +
launcher + dispatch is the bulk and needs the structured editor for safe iteration
against `test_sparse_mqa_logits` (oracle: bitwise-equal to the dense MQA logits).

Trap found while scoping the gather: the SM100 producer writes KV into a
UMMA-descriptor smem layout via `get_swizzled_kv_chunk_idx`, which does NOT match
the TMA 128B/64B `ldmatrix` swizzle (`get_swizzle_offset`) that the SM120
block-scaled helper reads. So the gather producer must be rewritten to land each
gathered sparse KV block in the `ldmatrix` layout (as the dense SM120 kernel's TMA
KV tile already is), not copied from SM100. Plan: gather with `cp.async` (paged) or
TMA (contiguous) straight into a `[SPLIT_KV][kHeadDim bytes]` tile with the same
128B swizzle the dense kernel uses, and stage one int32 SF word per token into
`sf_kv[token]`. Build incrementally: contiguous MXFP8 aligned-ks first (simplest),
then FP4, unaligned-ks, and paged. Oracle each step against `test_sparse_mqa_logits`
restricted to that case.

## 10. Measurements (RTX PRO 6000, GPU 7)

Timing method matters (learned the hard way):
- `deep_gemm.testing.bench` flushes L2 only once, so repeated launches hit the 128 MB L2. That gives
  impossible numbers (>4 TB/s) for weight-bound decode shapes. **Don't use it for small M.**
- `scripts/sm120_baseline.py::bench_cold` flushes 512 MB before every call and sums profiler kernel
  times. Use it for every comparison. The cold-DRAM floor is real: a 4 MB GEMM takes ~11 us, and
  cuBLAS shows the same floor.
- Best observed DRAM throughput for decode GEMMs is ~1.0 TB/s for both DeepGEMM and cuBLAS (spec 1.79 TB/s).

v1 kernel, `sm120_fp8_gemm_1d2d` with the auto heuristic, vs cuBLASLt per-tensor FP8 `torch._scaled_mm`.
cuBLAS does *less* work here (no block scales), so ratio ~1.0 means parity with the roofline proxy.

| shape (m x n x k) | DG us | DG TFLOPS | cuBLAS TFLOPS |
|---|---|---|---|
| 4096 x 7168 x 16384 | 1309 | 734 | ~765 |
| 4096 x 4096 x 7168 | 378 | 634 | ~690 |
| 4096 x 24576 x 1536 | 517 | 597 | ~630 |
| 1 x 7168 x 16384 (decode, ~1 TB/s) | 114 | 2.1 | 1.9 |
| 1 x 2112 x 7168 | 23 | 1.3 | 1.1 |
| 64 x 2112 x 7168 | 33 | 60 | 58 |

Tile findings (`scripts/sm120_sweep.py`):
- Large M: 128x128 always wins (8 math warps, 64x32 warp tiles, 3 stages).
- Single-wave cases are bound by per-CTA load latency, so small tiles and/or split-K win.
- Split-K helps only while tiles x splits <= 1 wave. Above 4 splits it gets *slower*: one measurement,
  64x128 with 10 splits, ran 2x slower than with no splits. Capped at 4.
- 128x256 / 256x128 tiles are not available. A 64x64 warp tile needs 128 accumulators plus 128 promote
  registers, which spills. A bigger-tile design would need a different promotion scheme (future work).

Known gaps of v1:
- The epilogue stores BF16 straight from registers (16 B per row segment). A stmatrix + TMA-store
  epilogue reusing pipeline smem could help large-M shapes.
- No UE8M0 (packed int SF) path yet. Int SFs on sm120 hit `UNREACHABLE`.
- Split-K counters are allocated on first use. **Call the GEMM once before CUDA-graph capture**
  (TokenSpeed warmup does this).

### Phase 4.5 final numbers (round 4, GPU 7, graph + cold weights, us per GEMM)

TP8 per-GPU DeepSeek-V3 dense linears; ratio = best(Triton, FlashInfer) / DeepGEMM (> 1 means DeepGEMM is faster).
FlashInfer is excluded where TokenSpeed would not pick it (17 <= M <= 32, N % 128 != 0).

| linear (N x K) | M=1 | 8 | 32 | 64 | 128 | 256 | 512 | 2048 | 4096 |
|---|---|---|---|---|---|---|---|---|---|
| q_a/kv_a 2112x7168 | 2.89 | 3.01 | 2.75 | 2.61 | 2.18 | 1.65 | 1.18 | 1.22 | 1.34 |
| q_b 3072x1536 | 2.53 | 2.00 | 1.97 | 1.72 | 1.37 | 1.06 | 1.00 | 1.02 | 1.00 |
| kv_b 4096x512 | 2.55 | 1.77 | 1.75 | 1.64 | 1.37 | 1.03 | 1.06 | 1.02 | 0.98 |
| o_proj 7168x2048 | 1.88 | 1.41 | 1.39 | 1.30 | 1.17 | 1.00 | 1.03 | 1.01 | 1.04 |
| shared gate_up 512x7168 | 3.70 | 4.00 | 3.04 | 3.01 | 2.68 | 2.54 | 2.16 | 1.16 | 0.96 |
| shared down 7168x256 | 2.88 | 1.93 | 1.46 | 1.33 | 1.03 | 1.18 | 1.16 | 1.13 | 1.07 |
| dense gate_up 4608x7168 | 2.36 | 2.00 | 2.00 | 1.94 | 1.64 | 1.19 | 1.00 | 0.96 | 1.02 |
| dense down 7168x2304 | 1.89 | 1.43 | 1.36 | 1.33 | 1.19 | 1.01 | 1.00 | 0.97 | 1.05 |

DSA indexer (H=64, D=128, page 64, top-k 2048), Triton time / DeepGEMM time:

| decode B x ctx | 1x4K | 8x4K | 32x4K | 128x4K | 1x16K | 32x16K | 128x16K | 1x64K | 32x64K | 1x128K | 8x128K |
|---|---|---|---|---|---|---|---|---|---|---|---|
| speedup | 3.4 | 5.4 | 10.7 | 13.4 | 6.0 | 15.2 | 19.3 | 8.1 | 17.2 | 9.2 | 14.7 |

| prefill T x prefix | 1024x0 | 4096x0 | 4096x4096 | 2048x14336 | 8192x0 |
|---|---|---|---|---|---|
| speedup (DG logits + Triton radix top-k) | 7.8 | 32.5 | 35.3 | 30.1 | 52.3 |


## 9.6 Sparse MQA done + block-scaled dense MQA + perf findings (2026-11-06)

Sparse MQA logits SM120 port **complete and enabled by default** (gate removed):
contiguous (aligned and unaligned ks) and paged, MXFP8 and MXFP4, H {8,32},
sparse_block_kv {8,16}, KV up to 4096 all match a dequantized `relu(Q.K)*w`
reference at diff ~1e-6 (`scripts/sm120_sparse_mqa_check.py`,
`sm120_sparse_paged_check.py`). The kernel is a from-scratch SM120 compute core
(single warpgroup, `cp.async` gather, `warp_mma_16xNxHd_block_scaled_ss`) on the
SM100 metadata scheduler. Five correctness bugs fixed along the way: SF-Q TMA box
shape/expect_tx, Q-barrier phase, the block-scaled MMA's per-N-tile B scale, and
the FP4 Q TMA byte count (commits eeb4926, 678df95, c384000, 0cd1e67).

**Block-scaled dense MQA logits** (`sm120_fp8_fp4_mqa_logits.cuh`): the MX path
of `fp8_fp4_mqa_logits` for arch 12, matching SM100 (UE8M0 MXFP8/MXFP4, per-row
int32 SF for Q and KV, BF16 weights/output, head dim 128). Reuses the FP8 dense
kernel's pipeline + the sparse block-scaled MMA. The FP8 per-token-scale kernel
(the DSA indexer path) is unchanged; the host dispatches by `is_mx_sf`. Two bugs
fixed: the Q/KV/SF-Q TMA outer strides (must be the contiguous row stride, and
`row_bytes` for packed FP4), and a missing `num_total_kv_blocks` advance in the
consumer that deadlocked the multi-q-block path. Validated at diff ~1e-6.

**Sparse MQA performance**: ~190 TFLOP/s MXFP8 / ~230 MXFP4, ncu shows ~46%/46%
compute/memory at 16.7% occupancy (1 block/SM, exactly one wave of CTAs). The
structural limiter is too few CTAs to exceed 1 block/SM. Tried and reverted:
smaller SPLIT_KV, a 2-stage gather/compute pipeline, `launch_bounds(256,2)`, and
three dedicated-producer/consumer multi-stage rewrites (the dense-kernel pattern).
None beat the baseline: the single-wave pipeline wasn't faster, and the multi-wave
path had recurring subtle shared-memory sync bugs. The "more CTAs" route is blocked
by the shared metadata scheduler's `kNumSMs <= 256` assert. Left at the correct
baseline; `scripts/sm120_sparse_bench.py` / `sm120_sparse_prof.py` capture the
measurement. A future pass needs a ground-up pipeline, not an incremental patch. (Done in task 35, section 9.39: an even split of the KV splits over the CTAs, a row-group gather overlapped with the MMAs, 2-6x.)

Full regression on GPU 6: `tests/test_fp8_fp4.py` (k-grouped skipped), the four
MQA checks, all green.


## 9.7 Cost-model refit + V3.2 indexer SM120 finding (2026-11-06)

**Cost-model refit.** Re-swept every legal 1D1D layout of the DeepSeek-V3 TP8
dense linears for MXFP4 and UE8M0 FP8 (serving condition: CUDA graph, cold
weights, PDL) and jointly refit the model. Worst-mode geomean regret on the
sweep data: 1.027 -> 1.014. New constants (`depth_exp` 1.58 -> 0.5,
`cta_scale` 1.2 -> 1.5, recalibrated per-tile cost table: 64x128 968 -> 476,
128x64 900 -> 525) make mid-M shapes prefer the 64x128 / 128x64 tiles the sweep
measures fastest. End to end vs FlashInfer MXFP4 (63 points): geomean speedup
1.315x -> 1.33x, still 59/63 wins. Committed `1b065a2`.

The 4 remaining ties (q_b M=256, o_proj M=128, dense gate_up M=256, dense down
M=128; 0.96-0.99x vs b12x) were **not closable with tile choice, the model or the
autotuner** (three of them were closed later by the PDL trigger, section 9.15): the
analytic model cannot pick a per-M tile (its cost table is one value per tile
shape), and `DG_SM120_AUTOTUNE=1` is *worse* (geomean 1.318, q_b M=256 0.94)
because eager timing cannot see the graph-mode PDL overlap. Genuine kernel-level
parity at that M/N boundary; leave at the model's pick.

**V3.2 sparse indexer on SM120 is blocked upstream, not in DeepGEMM.** (Superseded by section 9.16: only the V4.1 native indexer was blocked; the V3.2 DeepGEMM leaves already ran on SM120, and the SM120 DeepSelect port described here was done.) The
DeepGEMM sparse MQA scoring kernel works on SM120 (section 9.6), but TokenSpeed's
native DSV4.1 indexer (`dsv41/deep_gemm.py`) gates on `platform.is_blackwell`,
which is arch-major==10 only (SM100); SM120 is arch 12 (`is_blackwell` False,
`is_blackwell_plus` True). More fundamentally, the top-k **DeepSelect** half needs
~207 KB of shared memory (sized for SM100's 227 KB); SM120 has 99 KB, so a launch
on sm_120a fails `CUDA_ERROR_INVALID_CONFIG` (probed directly with cluster_size=1).
Enabling the DeepGEMM indexer on SM120 therefore needs an SM120 smem-reduced
DeepSelect port (CuTe DSL, tokenspeed-kernel, core-team-owned), plus flipping the
`is_blackwell` gates to `is_blackwell_plus`. Hand-off noted; out of DeepGEMM scope.


## 9.8 FP8 1D2D model already optimal; FP8 competitive state (2026-11-06)

Benchmarked the FP8 FP32-scale path (the 1D2D kernel TokenSpeed's DSA/dense
linears use) at TP8 vs Triton/FlashInfer (`sm120_vs_tokenspeed.py --tp 8`):
**geomean 1.61x over 112 points**, 7 losses all 0.94-0.99x (q_b M=2048/4096,
kv_b M=256, o_proj M=1024, shared gate_up M=4096, shared down M=4096/8192).

Re-swept the 1D2D layouts (`sm120_sweep_graph.py`) and refit
(`sm120_fit_heuristic.py`): the current model is already near-optimal (geomean
regret 1.0081; a refit only reaches 1.0077, changing just
`kDRAMBytesPerUnit` 1.6e6 -> 1.2e6). The one mid-M mispick (kv_b M=256: auto
64x128 6.0us vs best 128x64 5.7us) is **not fixed by the refit** either (it
still picks 64x128). The large-M losses are `auto == best` in the sweep, i.e.
genuine kernel-level parity, not tile choice. Not committing the refit: no real
gain, target shape unfixed. The FP8 path stays at the proven model.


## 9.9 BF16 MN-major, native einsum, spill fixes, mixed scales (2026-11-06)

Commits on `nctu6` after `df1bb16`, not pushed.

- **`74e1f61` MN-major BF16 and native einsum.** The BF16 `sm120_fp8_fp4_gemm_1d1d`
  kernel takes `kMajorA`/`kMajorB`. An MN-major operand is TMA-loaded as
  128B-swizzled atoms (64 MN elements x 64 K rows) and read with `ldmatrix.x4.trans`
  into the same fragments as the K-major path. The heuristic only gives MN-major
  operands tiles whose MN dimension is a multiple of 64. Host limits (at this commit; FP8 followed in 9.10): MN-major is
  BF16 only, MN-major A has no groups, a grouped MN-major B needs `K % 64 == 0`
  (TMA bounds only the stacked `G*K` rows). With this, `bf16_gemm_nn/tn/tt` and
  MN-major grouped B work in deterministic mode (which skips cuBLASLt). `einsum`
  `bhr,hdr->bhd` and `bhd,hdr->bhr` run one BF16 GEMM per head on strided views.
  `k_grouped_bf16_gemm_tn_contiguous` runs one MN-major GEMM per group from host-side
  `ks` (no PSUM layout), as on SM90. All four layouts x BF16/FP32 out x accumulation,
  K tails 8/136/1000, match an FP32 reference (diff < 1e-5). Also fixes, on all
  arches, a segfault in `einsum bmk,bnk->mn` with FP32 D and no `c` (it
  dereferenced an empty optional; it now asserts that `c` aliases `d`).
- **`9704db0` tests.** `test_bf16.py` and `test_einsum.py` crashed on SM120 in the perf
  print because they time kernels by SM90/SM100 names. `bench_kineto` has a
  `per_call_total` mode (sum over the kernels a call launches), and the generators
  know SM120's BF16 kernel and cuBLASLt kernel names. The native einsum is compared
  to torch at 1e-7 (the `mma.sync` accumulation order rounds ~0.02% of elements to
  the neighboring BF16 value). Stochastic rounding and the MN-major `fp8_einsum` tests
  stay SM100-only.
- **`4a95a00` stream-K spill.** Both SM120 GEMM kernels computed the 64-bit stream-K
  iteration range before the warp-role split, so it was live across
  `setmaxnreg.dealloc` and ptxas spilled it (48 B on the BF16 16x64 stream-K tile,
  N=1024, K=4096, caught by `DG_JIT_CHECK_NO_SPILLS=1` in `tests/test_sanitizer.py`).
  Each role now computes the range after its reconfiguration: 0 B, same results.
- **`a765e73` mixed scale pair.** A packed-UE8M0 SFA (e.g. DeepEP activations) with an
  FP32 128x128 SFB used to fail the layout check on SM120. `transform_sf_pair_into_required_layout`
  now casts the FP32 side to packed per-row UE8M0 for such pairs (gran_k 128), as
  SM100 does. Pure-FP32 pairs keep the FP32-scale kernel, and
  `disable_ue8m0_cast=True` rejects the mix. Matches a dequantized reference (~1e-6)
  in normal and m-grouped contiguous modes.
- **`9f19e0a` 1D2D masked FP8 spill.** `sm120_fp8_gemm_1d2d` masked (N=6144, K=7168,
  128x128, 256 threads) spilled 20 B. The cause was not `setmaxnreg`: the eight
  clamped SFA row offsets (`sfa_row_offsets[kNumMTiles][2]`) stayed live across the
  K loop next to the 128 accumulators and the scale prefetch. Only the base row is
  kept now; the offsets are recomputed at each load. 0 B. Masked FP8 sweep: within
  +-0.4% on 20 of 26 configs, 0.8-2.5% faster on 6, none slower beyond noise.

Limits that remained on SM120 at that point: k-grouped FP8 and MN-major FP8 einsum. MN-major FP8 GEMM operands are in section 9.10, k-grouped FP8 in 9.11, the einsum in 9.12.

## 9.10 MN-major FP8 operands (2026-11-06)

`fp8_fp4_gemm_{nn,tn,tt}` and m-grouped contiguous with an MN-major B now run on both SM120 FP8 kernels
(`sm120_fp8_gemm_1d2d`, and `sm120_fp8_fp4_gemm_1d1d` in its FP8 UE8M0 g32/g128 modes) as `kMajorA` / `kMajorB`
template parameters of the existing kernels. FP4 operands stay K-major (there is no sub-byte transposing load).

**`ldmatrix.b8` exists on sm_120.** Section 8 said it didn't. `ldmatrix.sync.aligned.m16n16.x{1,2}.trans.shared.b8`
assembles for `compute_120a` and `sm_120f` (SASS `LDSM.8.MT1616`); there is no `m8n8` b8 form, and `.trans` is required.
`scripts/sm120_fp8_trans_probe.cu` verifies its layout bit-exact on hardware. For a 16 K-row x 16 MN-byte block (`.x1`;
lane `l < 16` supplies the address of K row `l`, 16 contiguous MN bytes), with `g = lane / 4`, `t = lane % 4`:

| register | contents |
|---|---|
| `r0` | K `4t..4t+3` of MN byte `g` |
| `r1` | K `4t..4t+3` of MN byte `g + 8` |
| `r2`, `r3` (`.x2`, lanes 16-31 point at K rows 16-31) | the same for K + 16 |

These are exactly the E4M3 `m16n8k32` fragment registers: an A fragment is `{r0, r1, r2, r3}`, and the B fragments of two
8-column tiles are `{r0, r2}` and `{r1, r3}`. No byte permutes and no MN reordering, so the epilogue, the scale lookups and
the K order are those of the K-major path, and an MN-major operand pairs with a K-major one. The TMA tile is the BF16
path's: swizzled atoms of 128 (64 for 64-wide tiles) MN bytes by 128 K rows.

**Design choice.** Four designs for the transpose-on-load were built and timed.

| design | what | probe (ns / 128x128x128 K block, 8 warps, 188 CTAs, smem resident) | in the real 1D2D kernel |
|---|---|---|---|
| bound | K-major `ldmatrix.x4` | 872 | - |
| (a) | b16 `ldmatrix.x4.trans` + 4 `prmt` per 16x32 block | 907 (1.04x) | bit-exact; MN order permuted inside 16, so the epilogue, scale rows and columns are remapped; at 4096^3 B-MN 1.00x, A-MN 1.05x, both 1.07x of the bound |
| (b) | helper warps transpose the next stage in shared memory | 2404 (2.76x) | not built (the probe already doubles smem traffic and needs a sync per stage) |
| (c) | 32-bit `ld.shared` + 4x4 byte transpose with `prmt` | 877 (1.01x) | spills 7 of 8 shapes under `DG_JIT_CHECK_NO_SPILLS=1`, and 8-18% slower than (a) |
| (d) | native `ldmatrix.trans.b8` | 881 (1.01x) | bit-exact, no remapping; at 4096^3 A-MN 0.99x, B-MN 1.06x, both 1.05x of the bound |

(d) is used. It is exact, needs no remapping, and its A side beats (a). Its B side costs about 6% on compute-bound
128x128 tiles (4096^3: 218 vs 205 us for the K-major bound): the B pairs come out of the instruction as `{r0, r2}` and
`{r1, r3}`, but an MMA reads its B operand as an adjacent register pair, so ptxas copies the pair into a scratch pair
before each MMA (two `MOV` per `QMMA`, and a write-after-read dependency on the scratch). Looping over N tiles first, and
packing the pairs into 64-bit registers, produced the same SASS. Using (a) for B only would recover that 6%, at the cost of
the MN permutation in the epilogue and the scale lookups on the B side; not done, since the explicit transpose ties there.

**Host routing.** An MN-major operand either runs in-kernel or is transposed first by a JIT byte-transpose kernel
(`sm120_fp8_transpose.cuh`, a 128x64 tile through shared memory with 4x4 `prmt` transposes and 16-byte stores) followed by the
K-major kernel; that transpose is also the route for what the kernels don't cover (`M % 16 != 0` or `N % 16 != 0`, which
TMA can't express for MN-major rows, a grouped MN-major B whose K is not a multiple of 128, an MN-major A in a grouped GEMM).
`DG_SM120_MN_MAJOR_FP8` = 0 auto, 1 always host transpose, 2 in-kernel when supported (developer override).
Auto picks per operand: **B in-kernel always; A in-kernel from M = 256.** An MN-major A tile is at least 64 rows (the TMA
atom), so below that the kernel computes padding.

**Numbers** (`scripts/sm120_bench_mn_major_fp8.py`; RTX PRO 6000, GPU 7; CUDA graph over enough operand copies to exceed L2, so
operands are cold; min of 3 interleaved rounds; FP32-scale 1D2D kernel, BF16 out; DeepSeek-V3 TP8 linears 2112x7168, 576x7168,
3072x1536, 4096x512, 7168x2048, 4608x7168, 7168x2304, 512x7168, 7168x256 and squares 2048, 4096, 8192, M in
{1, 16, 64, 256, 1024, 4096}; A-MN and both-MN on a 7-shape subset, M in {16, 64, 256, 1024, 4096}). "explicit" is the
byte-transpose kernel then the K-major kernel; "bound" the K-major kernel on pre-transposed data; Triton is TokenSpeed's
block-FP8 kernel called on the strided MN-major operands (its wrapper asserts contiguity, the kernel itself takes strides);
FlashInfer needs K-major, so its time is on pre-transposed data and excludes the transpose. Ratios are geomeans (range):

| MN-major | M | n | default route / explicit | default route / bound | Triton (strided) / default | FlashInfer (K-major) / default |
|---|---|---|---|---|---|---|
| B (weights) | <= 64 | 36 | 1.63x (1.36-1.96) | 1.00x | 2.92x (1.47+) | 2.44x (1.35+) |
| B | 256 | 12 | 1.47x (1.32-1.89) | 1.00x | 1.97x | 1.43x |
| B | >= 1024 | 24 | 1.09x (0.97-1.27) | 0.96x | 1.65x | 1.02x (0.92+) |
| A (activations) | <= 64 | 14 | 1.00x (host route) | 0.88x | 4.08x | 2.78x |
| A | 256 | 7 | 1.07x (0.86-1.16) | 0.97x | 2.47x | 1.32x |
| A | >= 1024 | 14 | 1.08x (0.95-1.39) | 0.99x | 2.16x | 0.99x |
| A and B (Wgrad) | <= 64 | 14 | 1.57x (1.35+) | 0.93x | 4.60x | 1.78x |
| A and B | 256 | 7 | 1.57x (1.09-1.89) | 0.94x | 3.76x | 1.25x |
| A and B | >= 1024 | 14 | 1.20x (1.02-1.47) | 0.96x | 3.07x | 0.96x |

The in-kernel path (forced) against the explicit transpose: B 1.00-1.96x everywhere (1.00 at the compute-bound M = 4096);
A 0.32-0.61x at M = 16, 0.62-1.13x at M = 64 (hence the M >= 256 rule), 0.86-1.39x above; both operands 0.79-1.49x at M = 16,
0.83-1.88x at M = 64, 1.07-1.89x from 256. The default route is within 6% of the better of the two in 137 of 142 points
(the rest: A-MN 0.86x at 576x7168 M = 256, 0.89x at 7168x2048 M = 64, 0.90x at 8192^2 M = 64; both-MN 0.93x at 8192^2
M = 4096; B-MN 0.94x at 2112x7168 M = 4096). Measurement noise at M = 4096 is about 5%: the same code measured later in a
round came out 4-6% slower.
The 1D1D UE8M0 g128 kernel gives the same picture (b/a/ab at M in {16, 256, 4096} on 3 shapes: B 1.04-1.77x over explicit, A from
M = 256 1.06-1.12x, both 1.09-1.74x from M = 256). Where FlashInfer's K-major kernel is ahead of the default route (B-MN and
both-MN at M >= 256, up to 13%), 0-4% of that is already its lead over DeepGEMM's own K-major kernel there, and the rest is
the B-side `MOV` cost above.

**K-major unchanged.** The shared kernels were edited, so the K-major benchmark was rerun against a build of the previous
commit (`scripts/sm120_bench_kmajor.py`, 7 shapes x M in {1, 16, 64, 256, 1024, 4096}, min of 3 alternating rounds x 3 repeats):
1D2D geomean new/old 0.9997 (worst +0.7%), 1D1D UE8M0 g128 1.000 (worst +0.5%).

**Tests.** `tests/generators.py` now yields the SM120 FP8 Dgrad (K-major A, MN-major B) and non-accumulating BF16 Wgrad (both
MN-major) cases and the m-grouped contiguous MN-major B cases; `test_mn_major_fp8_sm120` in `tests/test_fp8_fp4.py` covers both routes on
both kernels, the grouped host-transpose fallback and the FP4 rejection against the dequantized reference at the K-major
tolerance. `scripts/sm120_mn_major_fp8_check.py` checks the MN-major result against the K-major kernel's (bitwise, or FP32
rounding when the heuristic picks another tile or stream-K split) and a dequantized reference over 432 cases.

**Found while testing; fixed in 9.14.** The K-major 16x32 stream-K tile of the packed-UE8M0 FP8 kernel
(`sm120_fp8_fp4_gemm_1d1d`, N = 576, K = 1024, M = 16, g32 and g128, 15 stages) spilled registers (g128: 4 B of spill stores, 20 B
of loads) under `DG_JIT_CHECK_NO_SPILLS=1`. It did at the previous commit too, with no MN-major operand involved; the existing
tests didn't reach that tile, and `test_mn_major_fp8_sm120` skipped those small shapes for the UE8M0 kernels until the fix.

## 9.11 K-grouped FP8 GEMM (weight gradients) (2026-11-07)

`k_grouped_fp8_gemm_tn_contiguous` and `k_grouped_fp8_gemm_nt_contiguous` now run on SM120, closing one of the two
remaining FP8 limits from section 8 (the other, MN-major FP8 einsum, is done in section 9.12). This is the Wgrad shape of MoE
training: a batch of GEMMs that share M and N but have a different K per group, laid end to end along K, with the group
sizes known on the host.

**Design: one GEMM per group, like SM90 and the SM120 k-grouped BF16 path (74e1f61).** There is no k-grouped scheduler on
SM120. The host loops over `ks`, and for a nonzero group runs `sm120_fp8_fp4_gemm_1d1d` over that group's K slice:
`D[g] = A[K_g]^T @ B[K_g]` (+ C[g], which `early_return` has already copied into D, giving the accumulate-into-D semantics
the SM90/SM100 contract requires). The operand slices are MN-major (`[sum_k, M]` / `[sum_k, N]`, narrowed along K then
transposed to a `[MN, K_g]` view for the TN API; the NT API takes K-major `[MN, K_g]` blocks laid end to end), so they are
read in-kernel with the 8-bit transposing `ldmatrix` from 9.10 — no explicit transpose. A group with `k == 0` zeroes its D
slice (or is left for the accumulate copy). Per-group launch overhead is not a concern: even at `num_groups = 32` the eager
and CUDA-graph times are within 0.5% (section measurements below), because each launch is a full-size M×N GEMM, and the
MoE Wgrad groups are large-K, not the many-small-group case that would favour a fused scheduler. A single fused k-grouped
launch was therefore not built; it would only help a regime these shapes do not hit.

**Scales.** `transform_k_grouped_sf_into_required_layout` and `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`
(in `smxx_layout.hpp`) now accept arch 12. They pack the per-group FP32 (or already-UE8M0) scales into the contiguous
`[sum(ceil(K_g / (gran_k·4))), MN]` layout, where a group's packed rows, transposed, are exactly the `[MN, packed_k]`
MN-major packed scales a normal SM120 block-scaled GEMM reads. The FP32 side of a 128-element recipe is cast to packed
UE8M0 (exact only for power-of-two scales), as on SM100. gran_k 32 and 128 both work.

**Numbers** (`scripts/sm120_bench_k_grouped_fp8.py`; RTX PRO 6000, GPU 7; 256 MB L2 flush before every call, so operands
are cold; eager time includes host launch cost, CUDA-graph time is GPU only; min of 3 rounds; accumulate into FP32 D,
gran_k 128). Shapes are the test-file set plus MoE Wgrad: `num_groups` in {4, 8, 16, 32}, DeepSeek-V3 expert dims
(M/N 4096×7168 and 7168×2048, hidden 7168 / intermediate 2048, gate_up 4096 wide), K per group in {256, 1024, 4096}.
Baselines: a host loop that explicitly transposes each MN-major operand slice to K-major and runs the K-major FP8 kernel
per group (the route the kernel replaces), DeepGEMM's own k-grouped BF16 GEMM, and a torch (cuBLAS) BF16 loop. Correctness
is checked bitwise against the per-group loop on the same packed scales inside the script.

| MN-major FP8 vs. | eager | CUDA graph |
|---|---|---|
| host loop + explicit transpose, K-major kernel | 1.12-2.06x faster | 1.11-2.03x faster |
| DeepGEMM k-grouped BF16 | 1.01-1.44x faster | similar |
| torch BF16 loop | 0.76-1.00x (torch wins the small-K 4096×7168 groups) | — |

The win over the explicit-transpose loop grows with K per group (more transpose work avoided): ~1.1x at K ≈ 320/group up
to ~2.0x at K ≈ 4000/group on the 7168×2048 shapes. Eager and graph times match to within ~1%, confirming the per-group
launch overhead is negligible at these sizes.

**Tests.** `test_k_grouped_gemm_contiguous` in `tests/test_fp8_fp4.py` no longer early-returns on SM120. It runs with
packed UE8M0 scales on both operands (`use_ue8m0` is forced on arch 12), checks the accumulate result against the FP32
reference and bitwise self-consistency over 20 repeats and the empty/single-group `ks` variants, and times the per-group
kernels with `per_call_total` so the summed group time is reported. Stochastic rounding and the pre-packed INT32 strided-SF
checks stay SM100-only (they use the SM100 epilogue / SF-K-padding path). Because `enumerate_k_grouped_contiguous` mutates
the contiguous-layout MK alignment and leaves it at its last value (384), and the test now actually runs on SM120, the
k-grouped test restores the 128 default at the end so the later m-grouped `test_mixed_ue8m0_and_fp32_scale_pair` sees a
block_m its heuristics can tile — this ordering bug was latent while the test early-returned.

Verified with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh cache: `test_fp8_fp4.py`, `test_bf16.py` and `test_einsum.py` all pass
(the 16×32 stream-K UE8M0 spill from 9.10 was still open then; fixed in 9.14). K-major shapes do not regress: `scripts/sm120_bench_kmajor.py`
1D1D timings are unchanged (the shared K-major kernel code is untouched; only the host k-grouped dispatch and the arch
gating in the scale-layout helpers changed).

## 9.12 MN-major `fp8_einsum` (2026-11-07)
`fp8_einsum` `bhd,hdr->bhr` and `bhd,bhr->hdr` now run on SM120 (task 11.3), closing the last FP8 limit from section 8. Like the
native BF16 einsum (74e1f61), each ran one GEMM per head on strided views (the commit this section describes; section 9.14 batches
the heads into one launch); the MN-major operands are read in-kernel with the 8-bit transposing `ldmatrix` (9.10).
- **`bhd,hdr->bhr`** (`(batch, m, n, k) = (h, b, r, d)`): K-major A, MN-major B, 128x128 FP32 scales, BF16 D. It is just the
  existing per-batch SM120 branch of `fp8_bmm` (`sm120_fp8_gemm_1d2d`, which derives the majors from the strides); the only
  change is that the arch-10-only gate in `fp8_einsum` is gone.
- **`bhd,bhr->hdr`** (`(h, d, r, b)`): both operands MN-major, per-channel 1x128 scales (`recipe=(1,1,128)`), FP32 D that
  accumulates onto C. This does *not* fit `fp8_bmm`'s SM120 branch, for two reasons: the 1D2D kernel writes only BF16 with no
  accumulation, and it reads SFB as 128x128 blocks. So it runs on the block-scaled `sm120_fp8_fp4_gemm_1d1d` (FP32 D,
  in-kernel accumulation, `gran_k` 128) per head, with both scale tensors packed to UE8M0 the way SM100 packs them (exact for
  power-of-two scales, as `per_channel_cast_to_fp8(use_ue8m0=True)` produces).
- **A false start worth recording.** The first version sent `bhd,bhr->hdr` through the 1D2D kernel with a BF16 scratch and a
  host FP32 add. It built and ran, and the test failed at `calc_diff` 0.030. Comparing against a dequantized reference showed
  the *product* was wrong (0.030) while the quantization error alone was 7e-4: the per-channel SFB was read as block scales.
  `fp8_bmm`'s SM120 branch asserts only the SF dtype and `gran_k`, not the N granularity, so it accepts a per-channel SFB
  silently. That is a latent hole for any caller passing `(1, 1, 128)` through `fp8_bmm` on SM120; it is not reachable from
  `fp8_einsum` any more, but is not closed in `fp8_bmm` itself.
- **Tolerance.** Against the BF16 inputs the error is FP8 quantization, ~7e-4 for `bhd,bhr->hdr` (b = 4096 and 8192), the same
  as SM100, so the existing `calc_diff < 1e-3` is kept. That is only ~30% above pure quantization error, so on SM120 the test
  also compares to the dequantized FP8 operands at 1e-10 (measured 1e-13 for `bhd,bhr->hdr`, 1.4e-6 for `bhd,hdr->bhr`, which
  is BF16 output rounding). Unlike the BF16 einsum (1e-7 against cuBLAS), no loosening is needed: the FP8 quantization error
  swamps the `mma.sync` accumulation-order difference.
- **Still rejected, with clear errors.** Accumulation (`c`) and custom epilogues / the FP8 output pair for `bhd,hdr->bhr` on SM120
  ("Custom epilogues require SM100", "SM120 FP8 bmm does not support accumulation"); `bhd,bhr->hdr` accepts `c` and only
  `recipe=(1,1,128)`. SM90 gets "has no kernel on this architecture".

Verified with `DG_JIT_CHECK_NO_SPILLS=1` and fresh caches: `test_einsum.py`, `test_fp8_fp4.py` and `test_bf16.py` pass. Edge shapes
(unaligned `d`/`r` that take the host-transpose fallback, `b` not a multiple of 16, a single token, `c` aliased / separate /
absent) match the dequantized reference.

**Benchmark** (GPU 7, CUDA graph, best of 3 interleaved rounds, us per call, e4m3 + UE8M0 scales). "BF16 DG" is the
deterministic native per-head BF16 kernel (74e1f61); "cuBLASLt" is `deep_gemm.einsum` with deterministic mode off; "loop" is a
per-head `torch.matmul` on BF16 (for `bhd,bhr->hdr` plus an FP32 add per head).

`bhd,hdr->bhr`:

| h, r, d | b | FP8 | BF16 DG | cuBLASLt | torch loop | FP8 vs BF16 DG | vs loop |
|---|---|---|---|---|---|---|---|
| 8, 4096, 1024 | 4 | 33.9 | 41.8 | 18.4 | 59.3 | 1.24x | 1.75x |
| 8, 4096, 1024 | 32 | 36.7 | 44.9 | 18.8 | 72.1 | 1.22x | 1.97x |
| 8, 4096, 1024 | 128 | 48.8 | 62.1 | 33.4 | 58.8 | 1.27x | 1.21x |
| 8, 4096, 1024 | 4096 | 517 | 793 | 753 | 801 | 1.53x | 1.55x |
| 8, 4096, 1024 | 8192 | 969 | 1486 | 1500 | 1599 | 1.53x | 1.65x |
| 128, 512, 128 | 4 | 191 | 225 | 8.7 | 321 | 1.18x | 1.68x |
| 128, 512, 128 | 32 | 193 | 227 | 9.2 | 275 | 1.17x | 1.42x |
| 128, 512, 128 | 128 | 194 | 233 | 16.6 | 283 | 1.20x | 1.46x |
| 128, 512, 128 | 4096 | 613 | 757 | 483 | 818 | 1.24x | 1.33x |
| 128, 512, 128 | 8192 | 1049 | 1081 | 979 | 1372 | 1.03x | 1.31x |

`bhd,bhr->hdr` (FP32 D accumulate; BF16 baseline is the cuBLASLt path, which is the only BF16 path for this expression):

| h, r, d | b | FP8 | cuBLASLt | torch loop | FP8 vs cuBLASLt | vs loop |
|---|---|---|---|---|---|---|
| 8, 4096, 1024 | 128 | 337 | 180 | 235 | 0.54x | 0.70x |
| 8, 4096, 1024 | 4096 | 660 | 796 | 1006 | 1.21x | 1.53x |
| 8, 4096, 1024 | 8192 | 1164 | 1512 | 1689 | 1.30x | 1.45x |
| 128, 512, 128 | 128 | 435 | 23 | 588 | 0.05x | 1.35x |
| 128, 512, 128 | 4096 | 1345 | 489 | 1408 | 0.36x | 1.05x |
| 128, 512, 128 | 8192 | 2030 | 930 | 1874 | 0.46x | 0.92x |

Reading it: the FP8 einsum beats DeepGEMM's own per-head BF16 kernel everywhere (1.03-1.53x) and the per-head torch loop almost
everywhere, and beats cuBLASLt only at large `b` with few big heads (1.2-1.5x at h = 8, b >= 4096 for both expressions). It
loses badly to cuBLASLt when `b` is small or there are many small heads (h = 128: 8.7 us vs 191 us): cuBLASLt issues one
strided-batched launch, while this path launches `h` GEMMs (about 1.5 us each, which is what the 190 us at h = 128 is). The
`bhd,bhr->hdr` many-head, small-tile case is worst because its per-head GEMM is also launch- and tile-bound (M = d = 128).
This gap, and the advice to keep the BF16 cuBLASLt einsum for many small heads, are superseded by the batched launch of
section 9.14.

## 9.13 Paged MQA logits with head dim 32 and 64 on SM120 (task 11.4)

Source check: the SM90 paged kernel accepts head dim 32, 64 and 128 with H in {32, 64} and next_n in {1, 2}
(`sm90_fp8_mqa_logits.hpp`, `attention.hpp`). The SM100 paged kernel is varlen-only (next_n 1, `indices` required) and takes
H multiples of 4 up to 128 with head dim 32/64/128 (64/128 for MXFP4); that is a different API, which SM120 does not expose
(`DG_HOST_ASSERT(is_varlen == (arch_major == 10))`). So the SM120 target is the SM90 set, and it is now met: head dim 32, 64, 128
x H {32, 64} x next_n {1, 2}. The host rejection is replaced by an assert that names the supported dims.

Design (one kernel path, head dim a template parameter):
- `warp_mma_16xNxHd_ss<N, kHeadDim>` already looped over `kHeadDim / 32` K steps; the paged kernel was just passing 128. Head dim 32
  needs a third TMA swizzle mode in `get_swizzle_offset`: 32B rows use `Swizzle<1, 4, 3>` (address bit 7 into bit 4), i.e.
  `chunk ^= (row / 4) % 2`.
- Tile bases must be 1024B-aligned for the manual swizzle, but SM90 uses `kHeadDim * 8` as the alignment (256 B for head dim 32,
  512 B for 64). Kernel and host now use `max(kHeadDim * 8, 1024)`, so the smem layout is identical in both.
- Stage selection is unchanged ((3,3), (2,3), (2,2), (2,1) with at least 2 Q stages, the deadlock rule above). Smaller head dims need
  less smem, so they get deeper pipelines than head dim 128. Proxy-fence placement is untouched.

Validation (GPU 7, fresh JIT cache, `DG_JIT_CHECK_NO_SPILLS=1`, timeouts on every run): `scripts/sm120_mqa_check.py` runs 13 paged cases per
head dim (128, 64, 32): B 3-128, next_n 1/2, H 32/64, contexts up to 20K, including the B>=32 and next_n=2 pipeline-wrapping cases. All 39
match the dequantized reference with diff <= 4.1e-14 (the paged assert is now 1e-12). `tests/test_attention.py` passes on SM120 with
the new head dims enabled (paged only; the contiguous kernel stays at 64/128), no spills.

Performance (`scripts/sm120_paged_mqa_bench.py`, cold L2, best of 3 x 30 calls; "KV GB/s" = bytes of KV read / time):

| B | next_n | H | ctx | D=128 (c5a9656 -> new) | D=64 | D=32 |
|---|---|---|---|---|---|---|
| 32 | 1 | 64 | 8192 | 43.9 -> 43.8 us | 29.1 us | 23.0 us |
| 32 | 2 | 64 | 8192 | 45.8 -> 45.8 us | 29.7 us | 23.4 us |
| 64 | 1 | 32 | 8192 | 76.7 -> 76.9 us | 45.7 us | 33.5 us |
| 128 | 1 | 64 | 4096 | 77.8 -> 77.6 us | 46.0 us | 33.7 us |
| 8 | 1 | 64 | 32768 | 43.9 -> 44.2 us | 28.7 us | 23.1 us |
| 64 | 2 | 32 | 4096 | 44.5 -> 44.4 us | 29.5 us | 23.5 us |

Head dim 128 is unchanged within noise (two interleaved runs each, baseline built from c5a9656). The new dims run at 400-780 GB/s of KV
traffic against 750-900 GB/s for head dim 128; the short cases are bounded by the fixed launch and tail cost (about 20 us), not by
bandwidth, and the 64-wide cases with enough work (B=64 H=32, B=128) reach 775 GB/s. The TokenSpeed Triton op
(`triton_dsa_decode_topk_fp8`) only registers `head_dim=128`, so there is no Triton number to compare at 32 or 64.

## 9.14 Batched `fp8_einsum`, the 16x32 spill, and the full SM120 verification (task 11.5, 2026-11-07)

**The gap.** After 9.12, `fp8_einsum` `bhd,hdr->bhr` / `bhd,bhr->hdr` (and `bhr,hdr->bhd`, through `fp8_bmm`) ran one GEMM per
head. Against the best of cuBLASLt on BF16 inputs, the per-head torch loop and `torch.einsum`, with a new benchmark that times cold
operands in one CUDA graph (`scripts/sm120_bench_fp8_einsum.py`, min of 3 interleaved rounds), the worst case was 22x slower
(h = 128, small b) and 1.2-1.9x slower at h = 8 with b <= 128. Where the time went (profiled kernel lists of one call):
- `bhd,hdr->bhr`, h = 128: 128 GEMMs of 1.57 us each (201 us of kernel time; the graph does not overlap them).
- `bhd,bhr->hdr`, h = 128, b = 128: 128 GEMMs of 2.0 us **plus 128 `sm120_transpose_fp8` kernels** of 1.4 us. The auto host policy
  of 9.10 transposes an MN-major A below M = 256, and here M = d = 128. At b = 4096 the 128 GEMMs are 8.5 us each.
- `bhd,bhr->hdr`, h = 8, b = 128: 8 GEMMs of 36 us. Not launch-bound: each moves 33.5 MB of FP32 D (read and write) and runs two
  partial waves of 256 tiles on 188 SMs.

**Options.** (1) Route these shapes to cuBLASLt inside `fp8_einsum`: it needs BF16 operands, so the FP8 inputs must be
dequantized first. Measured (`bhd,hdr->bhr`, dequantize + cuBLASLt vs the batched launch below): 53 vs 8.2 us (h = 128, b = 4),
73 vs 13.7 (h = 128, b = 128), 189 vs 27.9 (h = 8, b = 4), 211 vs 29.7 (h = 8, b = 128). No. (2) One batched launch in the
SM120 FP8 kernels: chosen, since it also removes the host transposes and the partial waves.

**Design.** `GemmType::Batched` (the scheduler already has it, for SM90/SM100) in `sm120_fp8_gemm_1d2d` and
`sm120_fp8_fp4_gemm_1d1d`, so it stays one kernel path with a template parameter:
- A and B are loaded through 3D tensor maps (inner, outer, batch; `make_tma_batched_desc`), so each batch can be a strided view:
  the heads of `[b, h, d]` are batches `d` elements apart. K- or MN-major, always read in-kernel (the host transpose and the
  M >= 256 rule of 9.10 do not apply, which is what removes the 128 transposes).
- D is offset by `d_stride_g` per batch, SFA by `sfa_stride_g`, SFB (1D2D) by `sfb_stride_g`; the 1D1D scales are stacked along
  their outer dimension like a grouped GEMM's, selected by the `SF_K` index type of the scheduler (which only Batched honors).
- The grid is `min(SMs, tiles)` CTAs over all batches' tiles (m fastest, then n, then batch); no stream-K, no PDL weight
  pre-issue (both are Normal-only). The cost model counts A and D bytes per batch.
- The host checks `sm120_fp8_bmm_1d2d_supported` / `sm120_fp8_bmm_1d1d_supported` (16-byte-aligned strides, compact SFB, D alignment)
  and otherwise launches one GEMM per batch as before. `DG_SM120_BATCHED_LAUNCH=0` forces that route (developer override, also
  used to A/B).
- The 1D1D FP32 epilogue with accumulation, which `bhd,bhr->hdr` stresses (K = b = 128 is a single K block per tile, so the
  epilogue is most of the tile): the old D values were loaded one float2 at a time, interleaved with the stores, and the compiler
  does not hoist a load over a store to the same buffer. A thread now loads all its old values first, with
  `ld.global.L2::128B` (a 256B hint is the same), because 32 bytes of 8 rows per load instruction otherwise fetch DRAM sector by
  sector. Kernel time of `bhd,bhr->hdr` (`bench_kineto` over the `gemm_` kernels, D cold; the first column is the per-head loop
  of b01788a from the 9.12 table, the others are this commit's steps):

| h, r, d, b | store only (no C) | per-head loop | batched | + loads first | + `L2::128B` |
|---|---|---|---|---|---|
| 8, 4096, 1024, 128 | 104 us (1.29 TB/s of D) | 337 us | 326 us | 282 us (0.95 TB/s) | 197 us (1.37 TB/s) |
| 128, 512, 128, 128 | 33 us | 435 us | 99 us | 87 us | 59 us |

  Accumulating reads and writes D, so 197 us for 268 MB is 1.37 TB/s against 1.29 TB/s for the store-only kernel (peak 1.79).

  The same epilogue serves the BF16 `bmk,bnk->mn` einsum (FP32 D accumulating): `test_einsum`'s perf print goes from 362/402/497 to
  293/358/409 us for the three shapes at b = 129.

**Numbers** (GPU 7, e4m3 + UE8M0 scales, us per call including the scale packing kernels, cold operands, CUDA graph, min of 3
interleaved rounds; baseline b01788a vs the batched launch; cuBLASLt is `deep_gemm.einsum` with deterministic mode off, "torch loop"
a per-head `torch.matmul`, with an FP32 add per head for `bhd,bhr->hdr`; "best" is the fastest of the three BF16 alternatives).
Targets: after / best <= 1.03.

`bhd,hdr->bhr`:

| h, r, d | b | FP8 before | FP8 after | cuBLASLt | torch loop | torch einsum | before / best | after / best |
|---|---|---|---|---|---|---|---|---|
| 128, 512, 128 | 4 | 263.9 | 9.0 | 14.6 | 511.4 | 13.9 | 18.85 | 0.65 |
| 128, 512, 128 | 32 | 279.7 | 11.7 | 17.3 | 404.3 | 14.4 | 19.42 | 0.81 |
| 128, 512, 128 | 128 | 286.4 | 18.8 | 30.5 | 397.9 | 23.7 | 12.08 | 0.79 |
| 128, 512, 128 | 4096 | 678.5 | 457.3 | 485.4 | 867.2 | 484.0 | 1.40 | 0.94 |
| 128, 512, 128 | 8192 | 1123.8 | 926.3 | 972.2 | 1421.7 | 956.6 | 1.17 | 0.97 |
| 8, 4096, 1024 | 4 | 58.9 | 27.9 | 48.3 | 86.6 | 47.6 | 1.23 | 0.59 |
| 8, 4096, 1024 | 32 | 65.4 | 30.8 | 50.6 | 94.4 | 48.9 | 1.34 | 0.63 |
| 8, 4096, 1024 | 128 | 70.2 | 34.9 | 58.4 | 87.4 | 52.1 | 1.35 | 0.67 |
| 8, 4096, 1024 | 4096 | 531.1 | 467.0 | 748.5 | 800.4 | 751.8 | 0.71 | 0.62 |
| 8, 4096, 1024 | 8192 | 949.1 | 891.2 | 1454.4 | 1605.5 | 1457.8 | 0.65 | 0.61 |

`bhd,bhr->hdr`:

| h, r, d | b | FP8 before | FP8 after | cuBLASLt | torch loop | torch einsum | before / best | after / best |
|---|---|---|---|---|---|---|---|---|
| 128, 512, 128 | 128 | 639.0 | 55.1 | 61.1 | 783.7 | 69.7 | 10.44 | 0.90 |
| 128, 512, 128 | 4096 | 1376.6 | 312.9 | 490.3 | 1458.6 | 515.3 | 2.81 | 0.64 |
| 128, 512, 128 | 8192 | 2086.1 | 565.5 | 928.9 | 1914.5 | 953.3 | 2.24 | 0.61 |
| 8, 4096, 1024 | 128 | 355.4 | 193.1 | 191.7 | 237.4 | 263.5 | 1.85 | 1.01 |
| 8, 4096, 1024 | 4096 | 670.9 | 503.8 | 793.3 | 1010.5 | 910.4 | 0.82 | 0.64 |
| 8, 4096, 1024 | 8192 | 1152.5 | 882.3 | 1454.2 | 1661.8 | 1588.0 | 0.79 | 0.61 |

`bhr,hdr->bhd`:

| h, r, d | b | FP8 before | FP8 after | cuBLASLt | torch loop | torch einsum | before / best | after / best |
|---|---|---|---|---|---|---|---|---|
| 128, 512, 128 | 4 | 315.7 | 7.8 | 14.5 | 530.6 | 14.2 | 22.23 | 0.55 |
| 128, 512, 128 | 32 | 322.7 | 9.5 | 17.7 | 539.9 | 16.8 | 19.21 | 0.57 |
| 128, 512, 128 | 128 | 327.8 | 16.6 | 29.5 | 551.6 | 26.2 | 12.46 | 0.63 |
| 128, 512, 128 | 4096 | 713.5 | 288.4 | 469.2 | 859.5 | 467.7 | 1.53 | 0.62 |
| 128, 512, 128 | 8192 | 1112.6 | 570.6 | 920.9 | 1394.8 | 915.6 | 1.22 | 0.62 |
| 8, 4096, 1024 | 4 | 63.3 | 24.6 | 47.6 | 75.7 | 47.5 | 1.34 | 0.52 |
| 8, 4096, 1024 | 32 | 66.3 | 26.7 | 49.9 | 80.4 | 49.1 | 1.35 | 0.54 |
| 8, 4096, 1024 | 128 | 79.6 | 30.8 | 55.4 | 102.5 | 53.5 | 1.49 | 0.58 |
| 8, 4096, 1024 | 4096 | 526.6 | 390.0 | 721.5 | 861.1 | 729.9 | 0.73 | 0.54 |
| 8, 4096, 1024 | 8192 | 858.0 | 797.5 | 1329.4 | 1525.8 | 1361.6 | 0.64 | 0.60 |

Worst after / best: `bhd,hdr->bhr` 0.97, `bhd,bhr->hdr` 1.007, `bhr,hdr->bhd` 0.63. Over the 26 points the geomean of FP8 / best went
from 2.40 to 0.66 and the worst from 22.2 to 1.007. `bhd,bhr->hdr` at h = 8, b = 128 is the one point at parity (192.2 vs 191.5 us in
three more runs); it moves 268 MB of FP32 D for 4 MFLOP of work per head, and cuBLASLt is at the same DRAM bound. The b = 4096 / 8192
points at h = 128 (0.94-0.97, `bhd,hdr->bhr`) are compute-bound.
These cold-operand numbers differ from 9.12's table, which left the weights warm (cuBLASLt at h = 8, b = 4: 18 us there, 48 us
here); cold is the serving condition. BF16 weights are twice the bytes of FP8, so part of FP8's lead at h = 8 is bandwidth.

**Tests.** `test_fp8_bhd_hdr_bhr`, `test_fp8_bhd_bhr_hdr` and `test_fp8_bhr_hdr_bhd` also run (128, 512, 128) on SM120, with the same
tolerances as 9.12 (1e-3 against the BF16 inputs, 1e-10 against the dequantized operands for `bhd,bhr->hdr`), and
`bhd,bhr->hdr` now also at b = 128. `test_fp8_einsum_batched_launch` runs eight shapes (unaligned b, a d of 256 and an r of 384 that
are not whole 512-wide tiles, h of 8/16/128) with and without `DG_SM120_BATCHED_LAUNCH` and requires the outputs to be bit-identical
(same MMAs in the same K order), then compares with the dequantized operands (1e-5; the BF16 output rounding alone is ~1.4e-6).

**The 16x32 stream-K spill (open since 9.10), fixed.** Reproduced with M = 16, N = 576, K = 1024, g128 (`DG_JIT_CHECK_NO_SPILLS=1`):
8 B of stack, 4 B of spill stores, 20 B of loads. The same family as the stream-K range of `4a95a00`: the TMA warps get 40
registers after `setmaxnreg.dealloc`, and this tile has 15 stages and the extra scale-TMA state. A 128-thread math group leaves
register-file room (128 x 248 + 128 x 56 is well under 64K registers), so its TMA warps now keep 56; 256-thread configs keep 40.
0 B of spills, no timing change (K-major spot check below). `test_mn_major_fp8_sm120` no longer skips the small shapes, so it
guards the fix when `DG_JIT_CHECK_NO_SPILLS=1` is set (the sanitizer sets it).

**Verification** (GPU 7, `DG_JIT_CHECK_NO_SPILLS=1`, fresh JIT caches, at the tree of the two commits above):
- `test_fp8_fp4` (6 test functions), `test_bf16` (7), `test_einsum` (8, with the new batched-launch check), `test_layout` (3),
  `test_hyperconnection` (1) and `test_attention` (2) pass. `scripts/sm120_mqa_check.py`: 46 cases, max diff 4.1e-14.
  `scripts/sm120_blockscale_check.py --mode all --grouped`: 170 cases, ALL OK. `test_lazy_init --num-processes 2` on GPUs 6 and 7
  (the one run that used GPU 6): 3 tests OK.
- `test_sanitizer.py` (memcheck and synccheck, `DG_JIT_CHECK_NO_SPILLS=1`, venv python on PATH), run per function because the script
  stops at the first failure: 0 errors on 22 functions (`test_einsum` all 8, `test_layout` 3, `test_hyperconnection`,
  `test_mn_major_fp8_sm120`, `test_fp8_fp4.test_gemm`, `test_m_grouped_gemm_contiguous`, the mixed-scale test, `test_bf16`
  `test_gemm` / `test_m_grouped_gemm_contiguous` / `test_m_grouped_gemm_masked` / `test_k_grouped_gemm_contiguous`,
  `test_paged_mqa_logits`, `test_sparse_mqa_logits`). The FP8 `test_k_grouped_gemm_contiguous` (2400 s), FP8
  `test_m_grouped_gemm_masked` (900 s) and `test_mqa_logits` (900 s) did not finish under the sanitizer, with no error before the
  limit; a reduced k-grouped FP8 run (the test's own enumeration, first 8 distinct (groups, m, n, accumulate, psum) combinations x 2 K
  variants = 16 cases, one launch each) under memcheck and synccheck has 0 errors. The `test_legacy` entry fails as before: the venv has no triton. `test_mega_gate`, `test_mega_mhc` and the cuBLASLt
  BF16 tests were not rerun (nothing they use changed). `test_locality_domain` (MLOPart) and `test_mega_moe` (NVLink) are the known
  environment failures and were not run. The three tests that did not finish under the sanitizer were completed in task 38 (section 9.42).

**K-major shapes do not regress** (baseline: a build of b01788a; `sm120_bench_kmajor.py` 1D2D and 1D1D g128, 7 shapes x M in {1, 16, 64,
256, 1024, 4096}; `sm120_bench_blockscale.py` q_a/kv_a, o_proj and shared down at M in {1, 16, 64, 512, 4096} for g128, g32 and
FP4; deterministic `bf16_gemm_nt` on 3 shapes x M in {1, 64, 512, 4096}; CUDA graph, cold weights, min of 2 alternating rounds per
build). New / baseline: 1D2D 42 points geomean 0.994, worst 1.004; 1D1D 42 points 0.996, worst 1.001; block-scaled 30 points
0.991 (g128 0.992, g32 0.990, FP4 0.989), worst 1.001; BF16 12 points 0.995, worst 1.001. Nothing is slower than noise.

## 9.15 FP4 ties against FlashInfer's b12x (task 12, 2026-11-07)

Start: after the cost-model refit (`1b065a2`) DeepGEMM's MXFP4 GEMM won 59 of 63 DeepSeek-V3 TP8 dense-linear points against the
faster of FlashInfer's `cudnn` and `b12x` (1.33x geomean). The four ties: q_b M=256, o_proj M=128, dense gate_up M=256, dense down
M=128. Method: `scripts/sm120_fp4_vs_flashinfer.py` (CUDA graph over 512 MB of weight copies, PDL on both sides, implementations
interleaved, best of 5 rounds inside a run), GPU 7 idle, three runs per build, best of 3 per point. Reproduced on the unchanged tree
(`6aebaf0`), one point per process: q_b M=256 6.2 vs 5.5 us (0.89x), o_proj M=128 8.1 vs 7.8 (0.97x), dense gate_up M=256 19.3 vs 19.5
(1.01x), dense down M=128 8.7 vs 8.4 (0.97x); in the full sweeps (three runs) q_b 5.7-5.8 vs 5.5, o_proj 7.8-7.9 vs 7.7, dense down
8.4-8.5 vs 8.4, dense gate_up 19.0-20.1 vs 18.6-19.8 us. The faster FlashInfer backend was `b12x` on all four.

**Profile comparison** (ncu, one launch, flushed caches, no PDL; DeepGEMM 64x128 tile, 8 math warps + 4 TMA warps; b12x
`DenseGemmKernel`, 8 math warps + 1 load warp):

| shape | DG us | b12x us | grid | block DG / b12x | regs DG / b12x | smem KB DG / b12x | DRAM % DG / b12x | L2 % DG / b12x | issued instr DG / b12x |
|---|---|---|---|---|---|---|---|---|---|
| q_b M=256 | 8.03 | 7.17 | 96 | 384 / 288 | 168 / 117 | 91.7 / 97.3 | 21 / 24 | 29 / 31 | 1.04M / 1.00M |
| o_proj M=128 | 10.62 | 9.41 | 112 | 384 / 288 | 168 / 117 | 91.7 / 97.3 | 47 / 53 | 36 / 36 | 1.44M / 1.49M |
| dense gate_up M=256 | 21.76 | 22.62 | 144 | 384 / 288 | 168 / 117 | 78.6 / 97.3 | 54 / 51 | 65 / 63 | 4.51M / 5.65M |
| dense down M=128 | 11.78 | 10.37 | 112 | 384 / 288 | 168 / 117 | 91.7 / 97.3 | 48 / 54 | 36 / 37 | 1.60M / 1.61M |

Same grid and tile on every shape, occupancy 17-19% on both, DRAM and L2 well below saturation on the three short-K shapes, and an
instruction count within 5% (DeepGEMM's is 20% lower on gate_up). No tensor-pipe, stall or epilogue difference explains a 10-14%
standalone gap: both are latency chains at this size. ncu metrics are only meaningful for the standalone kernel, and the standalone
numbers say DeepGEMM is 12-14% *slower* than b12x on the three short-K shapes, yet the graph-mode gap is 0-3%, so the difference had
to be in what the graph exposes: the launch chain.

**Graph-mode trace** (q_b M=256, 24 launches in one graph, cold weights, PDL; each CTA wrote `clock64` / `globaltimer` stamps next to
D, read back after the replay; scratch instrumentation, not committed). Period 5.8-6.0 us. A CTA lives 5.9 us: it starts 0.8-0.9 us
*before* the previous GEMM ends, but its first K block lands only 2.1 us after the CTA started (1.3 us after the previous GEMM
ended), the 12 K blocks take 3.4 us (670 cycles each) and the epilogue 0.4 us. Ablations in the same setup (period): full 5.8 us; no
D stores 5.8; no MMA 5.0; no MMA and no `ldmatrix` 5.0; none of the three 4.6 us. So tensor cores and stores are not the limit; the
time is the launch, the first loads and the pipeline refills. Per-block supply without MMA (eager, warm L2) is about 450-460 cycles
(12.8 KB per K block = 28 B/cycle/SM), the same at 24, 48 and 96 CTAs, i.e. a per-SM TMA ingest limit at that size, not an L2 or
DRAM limit.

**What the data pointed at.** A scan over K (n=3072, m=256, graph, PDL): DeepGEMM beats b12x at K <= 512 (1.12-1.13x, where all loads
fit the pipeline and the old code triggered the next launch right after the last load), then drops to 0.91x at K=1024-1536 where the
trigger moved to the end of the mainloop, and recovers to 0.97-0.98x at K=3072-6144. A trigger distance sweep (math warp triggers when D
K blocks remain, D = 99 meaning "after the last load"): q_b M=256 (K=1536): D=0 6.07 us, 2: 6.54, 4: 6.06, 6: 5.71, 7: 5.68, 12 or
99: 5.28-5.29 (b12x 5.52); dense gate_up M=256 (K=7168): D=0 19.34, 1: 20.08, 2: 21.33, 3: 21.17, 4: 21.25, 6: 21.39, 8: 21.50.
Short grids want the earliest trigger, the 144-CTA grid wants the latest. The old comment in the kernel ("after the last load was
slower at K=7168: 21.0 vs 19.5 us") was right for the long-K, near-full-grid shapes and wrong for the rest.

**Change (`76dc545`):** the producer triggers right after its last load when the grid fills at most 60% of the SMs (FP4 normal GEMMs;
a template parameter chosen in `SM120ArchSpec::use_early_pdl_trigger`). All 63 points, a version that triggered early on
every FP4 grid vs the old trigger (one run each, DG us): 25 points 3-20% faster (q_a/kv_a M=1-64 8.4 -> 7.3, shared gate_up M=32-64
5.4 -> 4.3, dense gate_up M=1-32 13.6 -> 12.0, o_proj and dense down M=32-256, q_b M=128-512, plus q_a M=2048 and dense gate_up M=128
at 4-5%), 8 points 4-11% slower (q_a M=128-512, shared gate_up M=512 / 2048, dense gate_up M=256, o_proj / dense down M=512: grids of
68% or more of the SMs). The 60% rule keeps those on the old trigger (and gives up the two gains above it). The always-early version
had geomean 1.386 but a worst point of 0.934 (dense gate_up M=256); the rule fixes that.

| | before (`6aebaf0`) | after (`76dc545`) |
|---|---|---|
| geomean vs the faster of cudnn / b12x, run 1 / 2 / 3 | 1.334 / 1.338 / 1.337 | 1.393 / 1.396 / 1.397 |
| best of 3 per point: geomean, wins, >= 1.03x, worst | 1.336, 59 / 63, 58, 0.965 (q_b M=256) | 1.399, 62 / 63, 61, 0.979 (dense gate_up M=256) |
| q_b M=256 (DG / b12x us) | 5.7 / 5.5 = 0.965x | 5.1 / 5.4 = 1.059x |
| o_proj M=128 | 7.8 / 7.7 = 0.987x | 6.9 / 7.7 = 1.116x |
| dense down M=128 | 8.4 / 8.4 = 1.000x | 7.5 / 8.3 = 1.107x |
| dense gate_up M=256 | 19.0 / 18.6 = 0.979x | 19.0 / 18.6 = 0.979x |

No point's best DeepGEMM time got more than 3% slower; the DeepGEMM time geomean over the 63 points is 0.955x of before. NVFP4 (same
rule, `--nvfp4`, vs the best FlashInfer / cuBLASLt NVFP4 backend, one run each): geomean 1.171x -> 1.223x, wins 39 -> 47, worst 0.867x
-> 0.910x, none slower by more than 4%. FP8 and BF16 are not affected (the rule is FP4-only). Verification at the commit (fresh JIT
cache, `DG_JIT_CHECK_NO_SPILLS=1`, GPU 7): `scripts/sm120_blockscale_check.py --mode all --grouped --repeat 200` ALL OK (170+ cases, 0 of
200 runs differ on every determinism shape for fp4, nvfp4, fp8g32, fp8g128 and bf16; a later `--mode fp4 --repeat 200` run adds the dense
down M=128 shape, also 0 of 200), `tests/test_fp8_fp4.py` passes.

**Dense gate_up M=256 is not won (0.98x, a tie).** Evidence:
- Standalone it is already faster than b12x (21.8 vs 22.6 us in ncu, with 20% fewer instructions). In the graph the two are within run
  noise (DeepGEMM 19.0-19.4, b12x 18.6-19.7 us across the runs; a sweep ratio anywhere from 0.95x to 1.02x).
- 144 CTAs of 64x128 over 28 K blocks (3 stages of 128-byte K). Graph-mode trace with MMA and `ldmatrix` stubbed out: the loop takes
  36.5K cycles against 39.1K for the full kernel (7% less), and a block's loads take 880 cycles with 36 CTAs, 910 with 72 and 1300-1370
  with 144. Total tile traffic (99 MB L2 -> SM) at 144 CTAs is about 6 TB/s, so this shape is limited by L2 -> SM bandwidth, which
  b12x shares (same tile and grid).
- Every tile shape with at most 188 CTAs reads about the same bytes (A re-read by every N block, B by every M block): 64x128 and 128x64
  both give 99 MB, 64x112 or 64x96 would give 8-17% less but need 168-192 CTAs (96 would put 4 CTAs in a second wave, 112 needs 7
  8-column tiles per warp, which the paired `ldmatrix` doesn't do). A BLOCK_N = 96 tile (4 x 2 warps of 16x48) was built and bitwise
  identical to 64x128, but was worth only 1-3% on q_b / o_proj / dense down in a graph test (kernel support reverted), and the
  dense gate_up gain would be capped by the same L2 limit.
- An L2 prefetch of the B tiles ahead of the pipeline (`cp.async.bulk.prefetch.tensor`, distance 4 / 8 / 16 K blocks): 19.6 / 20.9 /
  22.5 us against 19.4: slower, more requests into a saturated L2.
- Triggering early (above) costs 1-2 us on this shape; never triggering is no better (19.4 us here, but q_a M=256 13.4 and o_proj
  M=512 20.1 vs 17.3).
- TMA multicast across a 2-CTA cluster would halve the B traffic, and clusters launch on SM120 (a bulk-copy multicast probe returned the
  right data in both CTAs), but the tensor-map multicast is not hardware: CUTLASS disables it for SM120 (`CUTE_ARCH_TMA_SM120_ENABLED`),
  `ptxas` warns that `.multicast::cluster` on `cp.async.bulk{.tensor}` is expected to be slower on this target, and a launch with the
  raw instruction trapped into `emulate_tma` (`INVALID_DESCRIPTOR_FORMAT` for our FP4 descriptor). It is software-emulated, so it cannot
  cut L2 traffic.
- Tried and dropped earlier (sections 9.7 and the status log): other tile choices, the cost-model refit, two CTAs per SM, stream-K at
  mid M, `DG_SM120_AUTOTUNE=1`.
The shape stays at the model's pick. Anyone retrying it should start from the L2 -> SM traffic bound above.

## 9.16 DSA indexer top-k on SM120: DeepSelect in 93 KB (task 13.1, 2026-11-07)
TokenSpeed `nctu6` (local, not pushed); DeepGEMM unchanged (HEAD `d941488`, build `build312`). RTX PRO 6000, GPU 7 for timings, graph or eager as stated, best of 3 (decode) or median of 7 with a 256 MB cache flush between calls (prefill).

**Baseline: what the registry selects on SM120** (DeepGEMM build with SM120 kernels):
- V3.2 `dsa_plan`, `dsa_decode_topk`, `dsa_prefill_topk`, `dsa_index_candidates` and `kpool_prefill_topk`: the DeepGEMM leaves (priority 8) are already offered and outrank the Triton ones. Section 9.7 and the former report called the V3.2 indexer blocked; that was wrong for these leaves. Their top-k: decode runs the TRT-LLM CuTe cluster runner (it works on SM120), prefill the Triton radix select (the TRT-LLM prefill top-k has no SM120 image). Before this task, DeepGEMM leaf vs Triton leaf (`sm120_dsa_vs_tokenspeed.py`, 16 decode shapes): 3.4-17.3x; prefill 7.5-50x.
- V4.1 `dsv41_index_topk`: only `triton_dsv41_index_topk`. The leaves `deep_gemm_dsv41_index_topk` (packed FP4 rows plus upstream DeepSelect, window 10.0-10.3) and `deepselect_dsv41_select_*` are SM100 only, and the FP8-row leaves were Hopper only.

**What fails when the gates are opened:**
- Scoring works. The V3.2 FP8 MQA kernels need nothing. DeepGEMM has no paged FP4 MQA kernel on SM120 (`fp8_fp4_paged_mqa_logits` asserts FP8 q for arch 12; FP4 exists only as dense/sparse MX), so flipping `is_blackwell` to `is_blackwell_plus` cannot make the packed 68-byte V4.1 path work. FP8 `index_v4` rows can.
- Top-k fails on shared memory, in both selectors. The in-tree CuTe kernel compiles for `sm_120a` and fails at launch with 207640 B (capacity 512) or 232216 B (2048) against 101376 B. The upstream `deep_select` package hits `smem_size * TARGET_OCCUPANCY <= shared_memory_size_per_sm`. The 207640 B figure of 9.7 is this kernel's capacity-512 size.
- Thread-block clusters with distributed shared memory (`mapa`, `st.shared::cluster`) work on SM120: sizes 2 and 4 pass the correctness checks. (CUTLASS SM120 kernels use 1x1x1 clusters, but nothing here needs more.)

**Design: one kernel, a compact profile chosen by platform** (`dsa/_cute_dsl/deep_select.py`):
- Full profile (Hopper): 512 threads, 8192-entry rounds, 3-stage ring (96 KB), survivors 2 x capacity pairs, candidate buffer 4032 + 8192 pairs (96 KB).
- Compact profile (SM120): 256 threads and 4096-entry rounds, a 2-stage ring (32 KB), a 3072-pair candidate buffer (24 KB) and 2 x 2048 survivor pairs (32 KB), 93 KB in total at capacity 2048. A round's hits are appended in two groups of four warps; a group that would overflow the buffer triggers a compaction first (the hits were found against an older threshold, which only admits extra candidates). The in-order tail becomes the row's last full round, so the init window still holds at least `capacity` real entries. Compaction also runs from 1024 candidates (was 4032).
- Results: identical to `torch.topk` (value multiset, distinct indices) on random, tied, ascending, descending and masked rows, widths 700-131849, capacities 512/1024/2048, every cluster size the capacity allows, ragged `ends`, NaN rows, CUDA-graph replay. The Hopper suite (331 passed, 162 skipped = Hopper-only parameter sets) runs unchanged on SM120.
- Clusters: the candidate buffer gathers `(cluster - 1) * capacity <= 3072` pairs, so capacity 2048 allows pairs and capacities 1024 and below allow quads (8 CTAs would need 3584 at capacity 512). A CTA fills an SM, so clusters tile the GPU almost fully; measured residency limits (cap 512, 128K entries, us): 4-CTA 21.3 at 32 rows, 22.2 at 40, 43.1 at 48 (second wave); 2-CTA 41.4 at 92 rows, 65.1 at 100. `choose_cluster_size` uses 95% (pairs) and 85% (quads) of the SMs on the compact profile. 2-CTA gains 10-25% from 64K entries; below 32768 entries a split loses (cap 2048, 16K: 16.0 us single vs 16.5 pair; 4K: 4.2 vs 10.1).
- Rows, width and top-k are runtime arguments: only the capacity and cluster size select a compiled binary (guarded by `test_new_shapes_reuse_the_compiled_selector`). The leaves must run once eagerly before graph capture, as the TRT-LLM runner already does.

**Selector timings** (cap 2048, graph, random FP32 rows, us; DeepSelect best cluster / TRT-LLM cluster runner / Triton radix):
| rows x width | 1 x 4K | 32 x 16K | 1 x 64K | 128 x 64K | 1 x 128K | 32 x 128K | 128 x 128K |
|---|---|---|---|---|---|---|---|
| DeepSelect | 4.2 | 17.7 | 31.3 | 40.4 | 43.0 | 45.2 | 64.3 |
| TRT-LLM | 6.2 | 12.1 | 16.7 | 86.8 | 33.5 | 41.5 | 171.2 |
| Triton radix | 41.2 | 128.0 | 124.6 | 283.3 | 198.2 | 269.9 | 512.1 |
Prefill-shaped (causal rows; 1024x0, 4096x0, 4096x4096, 2048x14336, 8192x0, 1024x65536, 4096x32768), DeepSelect vs Triton radix: 9.0 vs 131, 58.7 vs 605, 204 vs 1105, 197 vs 1131, 290 vs 1701, 253 vs 2021, 611 vs 4228 us.

**V3.2 leaves** (`deep_gemm_dsa_*`, eager prefill, graph decode; "old" is the same leaf with DeepSelect switched off):
- Prefill now selects with DeepSelect wherever the logits satisfy its alignment: new/old 1.28x (1024x0), 1.90x (4096x0), 1.67x (8192x0), 1.60-1.73x with a 16K prefix, 1.66-1.70x with 64K-128K prefixes (4096x131072: 31.7 -> 17.8 ms). Overlap with the old leaf's selection is 1.0000 on all 13 shapes. Versus the Triton leaf: 8.2x (1024x0), 55.9x (4096x0), 80.6x (8192x0), 13.5-58.5x with a 16K prefix, 22.9-52.1x at 64K. One loss, unchanged and not caused by this task: 128 tokens without a prefix, 277 us against 144 us (host-bound: about a dozen small launches around a tiny GEMM).
- Decode keeps the TRT-LLM runner except where it falls off its cluster cliff (rows x clusters over the SMs): DeepSelect is used from 96 rows at 32K+ entries and from 48 rows at 64K+ (`_use_deepselect_decode_topk`). Measured TRT-LLM / DeepSelect: 16K x 128 rows 12.4/18.2 (TRT wins), 32K x 64 rows 17.1/26.1 (TRT), 32K x 96 rows 31.1/25.7, 64K x 128 rows 87.4/41.1, 128K x 48 rows 88.6/47.0, 128K x 128 rows 173/65. Full leaf, graph: unchanged on 14 of 16 shapes (+-2%), 1.04x at 128 x 64K and 1.06x at 128 x 128K; versus the Triton leaf 3.1-20.1x on all 16 shapes (B 1/8/32/128, ctx 4K/16K/64K/128K), overlap with the old leaf's selection 1.0000.
- Ruled out for decode: DeepSelect everywhere (loses 1.4-2.0x at 16K entries and below 48 rows of 128K).

**V4.1** (`dsv41_index_topk`, 32 heads, top-k 512, 64 candidate blocks, graph, Triton leaf time / native time, rows overlap with the Triton leaf 0.97-1.00):
- New leaves `deep_gemm_sm120_dsv41_index_topk`, `_index_candidates`, `_merge_index_candidates` (window 12.0-12.9, only registered by builds that report SM120 kernels). They are the Hopper FP8-row leaves: DeepGEMM FP8 MQA logits plus the CuTe DeepSelect, with a dense fallback where Hopper's sparse CuTe scorer is not available.
- Full pass, decode (1/8/32/128 requests x 1K/4K/16K/32K rows): 5.4-6.2x, 4.7-6.1x, 3.0-6.8x, 2.3-7.2x. Chunked prefill (128/1024/4096 queries x 4K/16K/32K rows): 2.6/1.7/1.4x, 3.6/1.9/1.5x, 5.1/2.1/1.6x.
- Reindex passes score the whole history here, while the Triton scorer reads only the candidate pool. Native wins up to 16384 rows (1.1-4.7x), loses on long histories of many rows (decode 128 x 32K 0.79x, 32 x 32K 0.99x; prefill 128 x 32K 0.54x, 1024 x 32K 0.60x). `fp8_indexer_wins` keeps those passes on Triton: native when the history is at most 16384 rows or rows x history is at most 2^19 (every measured point then wins, 1.08-4.7x).
- Cache format: SM120 stores 68-byte MXFP4 index rows (`cache_format="v41"`), which no SM120 MQA kernel scores natively. The FP8 route needs 132-byte `index_v4` rows, whose `v4` format holds about 45% fewer tokens per byte (LCM packing 11/33 pages against 20/60). The runtime does not select `v4` on SM120, so serving does not reach the new leaves yet.

**Tests** (TokenSpeed, GPU 6 and 7; kernel suite 7452 passed at the first full run, whose 4 failures were the local Reindex test that assumed Hopper, fixed; the final 14-file DSA/DSV4.1 subset 2975 passed, 0 failed; runtime layers 370, models 564 (includes `test_dots3_note_model.py`), top 6776, execution 195 passed, 0 failed): `test_deep_select_cute_dsl.py` now runs on SM120 (331 passed, 162 skipped), `test_dsa_indexer_sm120.py` (exact top-k of the logits the leaves hand to the selector, prefill and decode, with and without the DeepSelect route; row-chunking beyond 65535 rows), `test_dsv41_indexer_sm120.py` (native against the portable leaf; the `fp8_indexer_wins` rule), registry tests in `test_deepgemm_sm120_gating.py` and `test_sm120_excludes_sm100_only_leaves.py`.

**Ruled out:**
- Keeping 512 threads and 8192-entry rounds: a 2-stage ring is 64 KB and the hit buffer needs 8192 pairs (64 KB) on top of the survivors. No fit.
- 256 threads with a whole-round hit buffer: 4096 pairs (32 KB) + ring 32 KB + survivors 32 KB = 101.6 KB at capacity 2048, over the 100352 B usable. It fits at capacities 512 and 1024 but not at 2048, hence the two-group fill.
- Cluster 8 at capacity 512 (3584 gathered pairs against 3072 slots); 4-CTA clusters past 40 rows and pairs past 92 rows (second wave, time doubles).
- Opening only the `is_blackwell` gates: the packed FP4 paged scorer does not exist on SM120.
- Defaulting SM120 to `v4` caches in this task: see the capacity note above; it is a serving decision, not a kernel one.

**Open:**
1. The runtime config keeps `v41` rows on SM120, so the V4.1 FP8 route (2.3-7.2x on the index step) is unreachable in serving until the capacity trade is decided. No V4.1 checkpoint was run end to end.
2. A Reindex scorer for FP8 rows on SM120 that reads only the candidate pool (DeepGEMM has `fp8_fp4_sparse_mqa_logits` for MX rows) would remove the `fp8_indexer_wins` rule. A paged FP4 MQA kernel would make the 68-byte rows native.
3. `dsa_index_candidates` (V3.2 sharded) still selects with the Triton candidate top-k.
4. The V3.2 prefill leaf loses to the Triton leaf for chunks of about 128 tokens with no prefix (host launch overhead); no shape rule exists for it because selection is static per backend.
## 9.17 DSA sparse attention on SM120: tensor-core Triton kernel (task 13.2, 2026-11-07)
TokenSpeed `nctu6` (local, not pushed); DeepGEMM unchanged (HEAD `de74245`, rebuilt as `build312`, "build OK", tree identical to the Mac). RTX PRO 6000, GPUs 6 and 7, CUDA graph timing, median of 3-5 repeats (ranges in the logs were within 1-3%).
**Baseline: what the registry selects on SM120.** `dsa_decode` / `dsa_prefill` offer FlashMLA, flashinfer trtllm-gen (both capped off SM120, `8a33e3341`), the AMD Gluon cores and the portable `triton_dsa_decode` / `triton_dsa_prefill`; the last is the only core on SM120. The runtime always passes the dense latent cache (`has_kv_cache=True`, `has_sparse_kv_cache=False`, E4M3 or BF16 rows of 512 + 64, `k_scale=1.0`), never the packed FP8 sparse cache. That core launches one program per (token, head, 64-wide value block) and reduces on CUDA cores in FP32, re-reading every selected row once per head and twice (scores, then values). Nothing in it touches a tensor core.
**Baseline numbers** (E4M3, 128 heads, top-k 2048, history 16K per request for decode and 32K for prefill): decode batch 1/8/32/128 = 531/3539/13179/69374 us; prefill chunks 128/512/2048/8192 = 52/209/843/3404 ms. BF16 is 1.0-1.3x slower. At 16 heads, decode batch 1 is 360 us and the 8192 chunk 427 ms.
**Profile.** `ncu` is not available on this box (`ERR_NVGPUCTRPERM`, no access to GPU performance counters), so the evidence is ablation plus roofline. Roofline inputs measured on this GPU: cuBLAS FP8 GEMM 779 TFLOP/s, BF16 404 TFLOP/s; spec DRAM bandwidth about 1.8 TB/s (the bandwidth probe in the scratch script printed a wrong unit and was not used).
- Baseline: arithmetic is on CUDA cores and every head re-gathers the same rows (128 heads re-read each 576-byte row 128 times, served by L2). At decode batch 128 it does 73 GFLOP in 69 ms (1 TFLOP/s).
- Gathered bytes are not the limit: decode batch 128 gathers 128 x 2048 x 576 B = 151 MB (84 us at 1.8 TB/s, no reuse across requests); prefill chunk 8192 gathers 9.7 GB if every token fetched its own rows, but each row is shared by the 4 head tiles through L2.
- Tensor-pipe time: decode batch 128 needs QK 38.6 GFLOP (50 us at 779) + PV 34.6 GFLOP (86 us at 404) = 136 us; prefill chunk 8192 needs 2.47 + 2.20 TFLOP = 3.2 + 5.5 = 8.7 ms.
**Decision: a better kernel is clearly achievable.** The baseline is 70-1000x off the roofline. FlashMLA's sparse kernels use wgmma/tcgen05 and cannot be ported; the SM120 route is `mma.sync`. A Triton kernel with `tl.dot` already compiles to `mma.sync` (E4M3 `m16n8k32`, BF16/FP16 `m16n8k16`) and wins by two orders of magnitude, so it ships; CuTe DSL is the follow-up for the remaining 3-4x, not a prerequisite.
**Design** (`ops/attention/dsa/_triton/sparse_mma.py`, README section "Tensor-core sparse attention on SM120"):
- A program owns `BLOCK_H` (16 or 32) heads of one token and walks the selected slots in tiles of 32 rows. A tile is gathered once (latent 512 + RoPE 64 channels) and used for `S = Q K^T` and `O += P V` (the 512 latent channels are both K and V). Online softmax in FP32 with `exp2`.
- QK is E4M3 `m16n8k32` when both query and cache are E4M3 (the runtime sends E4M3 queries when the cache is E4M3), BF16 `m16n8k16` otherwise. PV multiplies FP16 probabilities by an E4M3 cache (exact in FP16, 3 more mantissa bits than BF16) or BF16 probabilities by a BF16 cache.
- The next tile is gathered into registers before the current tile's math (manual double buffering). Shared memory is 53 KB with 246 registers and no spills at 8 warps (before section 9.20; the batch-invariant version uses 255 and spills 34-52).
- Few tokens do not fill 188 SMs, so `plan_launch(tokens, heads, num_sms)` splits each (token, head tile) over up to 16 programs and a small second kernel merges the partial (max, sum, accumulator) states; one split writes the normalized result directly. (Superseded by section 9.20: the split now follows the chunks of the selection width, so the result no longer depends on it.) The split count and the head-tile width (32 once there are enough programs, else 16) depend on the token count only, so CUDA-graph replay per batch size is fixed. Token count, head count, top-k width, strides and split count are runtime arguments (`do_not_specialize`), guarded by `assert_no_triton_compile`.
- One path with parameters, same numerics contract as `triton_dsa_*`: BF16 output for E4M3 queries, `k_scale` folded into the scale, natural-log FP32 LSE, `-1` slots and `topk_lens` masking, empty rows zero with `-inf` LSE, `out=` identity kept.
**Results** (E4M3, top-k 2048, speed-up over `triton_dsa_*`; 64 buckets over both cache dtypes, 27.2-146x in total, every bucket wins):
| heads | d1 | d8 | d32 | d128 | p128 | p512 | p2048 | p8192 |
|---|---|---|---|---|---|---|---|---|
| 128 (us) | 13.6 | 46.3 | 135.7 | 537.5 | 482 | 1818 | 7442 | 29508 |
| 128 (x) | 39 | 77 | 97 | 129 | 108 | 115 | 113 | 115 |
| 64 (x) | 31 | 67 | 88 | 130 | 107 | 111 | 114 | 115 |
| 32 (x) | 31 | 54 | 71 | 109 | 99 | 111 | 115 | 114 |
| 16 (x) | 29 | 39 | 53 | 72 | 66 | 75 | 82 | 75 |
BF16 caches: 27-146x (128 heads: decode 15.0/58/171/714 us, prefill 0.59/2.33/9.2/36.5 ms). Error against a float32 attention over the same stored values is 1e-3 to 2e-3 absolute (output max), identical to the baseline's error on the same inputs; LSE agrees to 2e-5.
**Roofline.** 128 heads, decode batch 128: 73 GFLOP in 537 us = 136 TFLOP/s against a 136 us tensor-pipe floor (25%). Prefill 8192: 4.67 TFLOP in 29.5 ms = 158 TFLOP/s against 8.7 ms (29%). Decode batch 1 (13.6 us) is latency bound: 1.2 MB of gathered rows would take 0.7 us, the rest is launch, the 16-way split and the combine kernel. Ablation (128 heads): dropping the QK product removes 220 us of 544 at decode batch 128 and 864 us of 1903 at prefill chunk 128, so QK is 40-45% and the remaining time is PV plus softmax and gathers.
**Tried and ruled out** (128 heads, decode batch 128, shipped 540 us; FP8; correctness unchanged in all):
- Compiler-pipelined `tl.range(num_stages=1..3)` loop instead of the manual prefetch: 639-739 us at 4 or 8 warps; at decode batch 8 with 16 splits it ties (19.1 vs 18.7 us).
- Register caps at 8 warps: 1547 us (168), 2045 us (128); the kernel needs 246 registers, so a cap spills. 4 warps: 2626-4525 us. 16 warps with a 128 cap: 812 us (64: 2752 us).
- `BLOCK_N` 16: 797 us; `BLOCK_N` 64 under caps of 128/200: 3711/2751 us.
- 16-head tiles once programs fill the GPU (half the reuse of each gathered row): the sweep preferred 32-head tiles there, which is why `plan_launch` widens the tile.
- Per-shape sweeps of tile width, `BLOCK_N`, warps, stages and splits (scratch `tune.py`, E4M3 and BF16) fixed the shipped tile and the split rule in `plan_launch` (about 2/3 of the SM count for up to 8 tokens, 2.75 programs per SM above); the full sweep logs were not kept.
**Tests** (GPU 7 kernel, GPU 6 runtime): `test_dsa_mma_sm120.py` (selection on SM120 and the other archs, plan rule, float32 reference at tight tolerance for E4M3 and BF16 caches, prefill and decode, empty/short/holed rows, head tail tile, width 2051, split-count invariance, `out=` buffer and flat cache layouts, no recompilation across 11 token counts); the focused kernel set (`test_sm120_excludes_sm100_only_leaves`, `test_dsv41_registration`, `test_kernel_api_selection`, `ops/test_attention_dsa`, `test_dsa_facade_traits`, `test_dsa_indexer_sm120`) 411 passed; the full kernel suite earlier in the task 7482 passed, 0 failed; runtime DSA set (dots3-note cache/config/model/nextn, DSA backend selection, numerics plan, query shard, FlashMLA DCP, cache groups, sparse KV offload, GLM-5.3-flash) 243 passed, 19 skipped, 0 failed. The registry tests needed no change: the new leaves declare window 12.0-12.9.
**Open:**
1. A CuTe DSL kernel (TMA or `cp.async` gathers into 99 KB of shared memory, `ldmatrix` fragments, warp-specialized softmax) could close part of the 3-4x gap to the tensor-pipe floor. Not written; the Triton kernel is already 27-146x ahead of the only alternative.
2. ~~The leaves do not declare `batch_invariant`~~ Done in task 15 (section 9.20): fixed chunks by width, the leaves declare `batch_invariant`, and bitwise-envelope plans select them.
3. The packed FP8 sparse cache and zero-width RoPE (GLM-5.3-flash) keep the portable kernels.
4. End-to-end DSA run: done in section 9.18 (task 13.3).

## 9.18 End-to-end V3.2 DSA check on SM120 (task 13.3, 2026-11-07)

TokenSpeed `nctu6` at `f9d0927c1` (no change made for this task), DeepGEMM `ed3b112` (`build312`). RTX PRO 6000, GPU 6 (new path) and GPU 7 (old path), in-process `Engine`, no HTTP server.

**Model.** The remote has no DSA checkpoint (`/home/ubuntu/models`, `/models` and the HF cache hold DeepSeek-V3.1, R1, V4/V4.1-Flash, Qwen, gemma, gpt-oss, Inkling and others; none is DeepSeek-V3.2, GLM-5.x or dots3-note). So the check uses a reduced `DeepseekV32ForCausalLM`: 4 layers (1 dense + 3 MoE with 16 experts, top 4), hidden 7168, 128 heads, q_lora 1536, kv_lora 512 + 64 RoPE, v 128, indexer 64 heads x 128, `index_topk` 2048, YaRN factor 40, vocab 32000, random BF16 weights (std 0.02-0.04, norms 1), 2.2 B parameters. The generator is a scratch script (`mk_v32.py`); prompts are token ids (a Qwen tokenizer file only satisfies the loader). Outputs are random tokens by construction, so correctness is judged against the old path, not by reading text.

**Selection (TOKENSPEED_KERNEL_VERBOSE).** Prefill: `dsa_prefill_topk` -> `deep_gemm_dsa_prefill_topk`, `dsa_prefill` -> `triton_dsa_mma_prefill`. Decode (CUDA graphs captured for batch 1-32): `dsa_plan` -> `deep_gemm_dsa_plan`, `dsa_decode_topk` -> `deep_gemm_dsa_decode_topk`, `dsa_decode` -> `triton_dsa_mma_decode`. The old path is forced with `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSA_DECODE=triton_dsa_decode` and `..._DSA_PREFILL=triton_dsa_prefill` (the registry's `TOKENSPEED_KERNEL_OVERRIDE_<FAMILY>_<MODE>` variables). The indexer stays on the DeepGEMM leaves in both runs; the Triton indexer was already 3-20x slower (9.16).

**Correctness.**
- Prefill is bit-reproducible on both paths (same prompt, two runs, two GPUs: 0.000 difference). Decode steps at more than 2048 tokens of context are not bit-reproducible on either path (old path vs old path: identical greedy prefix 2-4 of 24 tokens at 3000 and 6000 context; new vs new: 2-11 of 24). This is not caused by the new kernels, so greedy continuations cannot be compared token by token beyond a few steps.
- Teacher-forced prompt logprobs (deterministic, every position of 1500/3000/6000-token prompts), new vs old: mean |dlogprob| 0.14/0.19/0.32, p99 1.1-1.7, max 2.7-3.7, on logprobs averaging -18.9. At 1500 tokens the selected set is the whole context, so this is pure kernel numerics. Final logits are BF16 (ulp 0.125 at magnitude 16-32), and random layers amplify BF16 differences, so mean differences of about one ulp are expected.
- Control: the same path with chunked prefill 2048 instead of 8192 differs from itself by a mean 0.33/0.79 at 3000/6000 tokens (new) and 0.32/0.80 (old). That is larger than new vs old (0.19/0.32), because random indexer scores are near ties and the top-k selection flips with the chunking. Chunk 2048 at 1500 tokens is identical (one chunk).
- Last-prompt-position top-5 (max_new_tokens 1): top-1 token identical in 4 of 4 contexts (64, 1500, 3000, 6000); top-5 sets equal in 1 of 4, the others differ only in the 5th place (logprob -3.4 vs -3.5). First-token logprob differences 0.06-0.14.
- A second model with a 5x sharper lm_head and 10x larger o_proj/kv_b gain was chaotic (dlogprob up to 9 at 3000 tokens, also between chunkings of the old path) and carries no signal; discarded.
- Kernel-level accuracy against a float32 reference is in 9.17 and is unchanged.

**Performance** (greedy, `max_new_tokens` 1 for prefill, TPOT = (T(65 tokens) - T(1 token)) / 64 on the same prompts with prefix caching off; 4 repeats, first dropped, so n = 3; relative standard deviation under 1% everywhere).

| context x batch | prefill tok/s new | prefill tok/s old | speedup | TPOT new (ms) | TPOT old (ms) | speedup |
|---|---|---|---|---|---|---|
| 1024 x 32 | 71558 | 753 | 95x | 3.85 | 55.27 | 14.4x |
| 4096 x 1 | 41032 | 499 | 82x | 2.63 | 5.40 | 2.05x |
| 4096 x 8 | 44081 | 496 | 89x | 3.33 | 22.83 | 6.9x |
| 16384 x 1 | 37137 | 405 | 92x | 2.65 | 5.56 | 2.10x |
| 32768 x 1 | 35178 | 393 | 90x | 2.66 | 5.77 | 2.17x |

New path only: 4096 x 32 44333 tok/s and 4.23 ms; 16384 x 8 37505 tok/s and 3.39 ms; 1024 x 8 69046 tok/s and 3.28 ms. The old path was not run at 4096 x 32, 16384 x 8 and 32768 x 8 because one prefill pass alone takes 4-10 minutes at 400-500 tok/s. The model has 4 layers, so the totals are not V3.2 numbers; the ratios show how much attention costs per layer. At batch 1 the new TPOT is flat in context (2.63-2.66 ms from 4K to 32K), so the remaining time is MoE/dense GEMMs and launch gaps, not DSA.

**Issues found.** None on SM120: no crash, no NaN, graph capture succeeded, kernels selected as designed. Harness notes: `top_logprobs_num > 0` requires `max_new_tokens <= 1`; `log_level` must be upper case for `Engine`; `Engine` needs a loadable tokenizer even with `skip_tokenizer_init`.

**Not covered.** A real V3.2 checkpoint (none on the box, 671B parameters would not fit on two GPUs anyway); a text-quality check; batch-invariant plans (these selected `triton_dsa_*` then; since task 15 they select `triton_dsa_mma_*`, section 9.20); the TP8 run.

## 9.19 SM100-only leaves on SM120: Hadamard, MLA decode, V4 attention, zero-RoPE DSA (task 14, 2026-11-07)

TokenSpeed `nctu6` from `f9d0927c1` (local, not pushed); DeepGEMM unchanged (`932e8c6`, `build312`). RTX PRO 6000, kernel tests and benchmarks on GPU 7, runtime tests on GPU 6, CUDA-graph timing (best of 4-5 replays). Rooflines measured on this GPU: device copy 1461 GB/s, cuBLAS 8192^3 BF16 413 TFLOP/s, FP16 404, FP8 582.

`8a33e3341` capped four SM100-only solutions off SM120. None can be ported as is (wgmma/tcgen05/TMEM, or no sm_120 cubin), so for each one: who uses it, what SM120 selected instead (`get_for_operator` on the SM120 platform), how far that fallback is from the roofline, and the decision. Selection on SM120 before this task: `hadamard_transform` -> `triton_hadamard_transform_128`; `mla_decode_with_kvcache` -> `triton_mla_decode_with_kvcache`; `dsv4_decode` / `dsv4_prefill` -> `triton_dsv4_decode` / `triton_dsv4_prefill`; DSA -> `triton_dsa_mma_*` (13.2) except the cases below.

**`fast_hadamard_transform` (window 9.0-10.3).** User: the GLM-5.3-Flash indexer query (`[tokens, 32, 128]` BF16, `glm53_flash.py`). The wheel fails on SM120 with "no kernel image" (sm_90/100/103 cubins, no PTX). The upstream source (Dao-AILab, HEAD `e7706fa`) builds for `compute_120,code=sm_120` with nvcc 13.2 and is bitwise equal to the reference, so widening the window would only need a rebuilt wheel, but it is not the fastest option. The portable Triton leaf builds the sign matrix per output block and runs at 25 GB/s. New leaf `triton_hadamard_transform_128_mma`: one `[BLOCK_M, 128] x [128, 128]` matmul on `mma.sync` per row tile (signs exact in BF16/FP16, FP32 accumulation, bitwise equal to the CUDA butterfly). 32 heads x 128 BF16, tokens 1/8/64/512/4096/16384/65536: copy 1.0/0.9/1.0/2.0/10.8/183/734 us; portable 1.9/6.0/42/321/2507/10097/40380; upstream CUDA (sm_120 build) 1.0/1.1/1.9/9.7/73/292/1165; new 2.1/2.2/2.1/5.4/21.4/183.7/735. At the copy roofline (within 0.2%) from 16K rows, 1.6x faster than the CUDA build there; launch bound at 1-64 tokens (about 2 us, the CUDA kernel is 1 us at 1-8 tokens). Tile sweep (BLOCK_M 16-256, 2-8 warps, 1-3 stages): `[16,128]`/4 warps up to 2048 rows, `[64,128]`/8 warps above; 128-row tiles with more than one tile per program exceed 99 KB. Decision: ship the Triton leaf, keep the wheel capped.

**`tokenspeed_mla` decode (cap 10.7).** Users: MLA models on the generic `mla` backend (DeepSeek V3-style, Kimi-K3 full-attention layers) and MTP verify / DFLASH-DSPARK draft blocks (`q_len` 2-8). The portable decode runs one program per (row, head), re-reads the whole cache per head and multiplies on CUDA cores: BF16 16 heads 1x4K/1x32K/8x8K/32x8K = 705/5070/1328/3383 us against 3.2/25.9/51.7/207 us of memory roofline (16-218x off); 128 heads 689/5070/6125/22404 us. Decision: a new kernel is clearly worth it. `triton_mla_mma_decode_with_kvcache` (`mla/_triton/decode_mma.py`) tiles 16-32 `(query, head)` rows of a request over one pass of the paged cache, QK and PV on `mma.sync` (E4M3 `m16n8k32` when q and cache are E4M3), FP32 online softmax, split of the cached range (up to 128 programs) plus a merge kernel. It covers causal verify blocks and non-causal proposal blocks with and without a window, `q_len` 1-8 on the query axis, LSE; `logit_cap` stays portable. Results (BF16): 16 heads 20/38/45/236 us (0.88-1.47x of the memory roofline from 8 requests up), 128 heads 27/122/218/713 us (75-102 TFLOP/s); E4M3 128 heads 32x8K 513 us (142 TFLOP/s); 14-131x over the portable kernel. Query-axis blocks vs flattened rows at Q=4, B=32, 8K: 1.37x at 16 heads, tie at 128. Error: output within one output ulp of a float32 reference, LSE 2e-5. Tuning: key tile 16/32/64, 4/8 warps, 1-3 stages, row tile 16/32/auto (best 32/8/any/auto); a split limit of 32 left 1x32K at 16 heads at 83 us, 128 gives 38 us; skipping the stores of empty splits gained nothing (reverted). Also fixed: Kimi-K3 defaulted its history consumer to the `tokenspeed_mla` backend on every non-AMD GPU, which cannot run on SM120; it now resolves to the generic `mla` backend there (`252f7686c`). Roofline at 128 heads: 25-35% of cuBLAS; the same `mma.sync`-through-Triton ceiling as 13.2. MLA prefill (`triton_mla_prefill`, already `tl.dot`) runs 104-146 TFLOP/s and was not changed.

**FlashMLA DSV4/V4.1 (cap 10.9; the V4.1 API loads only on Hopper/SM100).** Users: DeepSeek V4/V4.1 attention. On SM120, `dsv41.selected_attention` (decode: bounded cache gather into a BF16 workspace; prefill: the backend's workspace) and `triton_dsv4_decode` all end in `dsv4_prefill` -> `triton_dsv4_prefill`, one CUDA-core program per (token, head). Gather share at 64 tokens: 25 of 141 us at 64 heads (the rest is attention plus mask construction). Decision: port the attention core, keep the gather. `triton_dsv4_mma_prefill` (`dsv4/_triton/sparse_mma.py`) uses the 13.2 tile (16-32 heads per program, one gather per tile of 32 rows, QK and PV on BF16 `mma.sync`, sink as an extra zero-valued key). First version: 5-11x over portable with a token-count split, but `test_deepseek_v41_model::test_cuda_40_layer_real_flatkv_and_moe[prefill]` failed (replayed chunk logits off by 0.3; passes with the portable kernel or with no split, so the cause is the batch-dependent split). Never splitting costs 4x at 1-16 tokens (42-60 vs 10 us). Shipped: at most 4 chunks per token selection, a count fixed by the width; each chunk's online softmax, chunk states folded in order with one explicit-`fma` helper, in registers or (split launch) by the combine kernel, so rows are bitwise equal across 1-600-token batches. Cost 5-15% (64 heads 2048 tokens 1618 vs 1427 us; one token 14.6 vs 9.9 us). Ruled out: one tile loop with merges at chunk ends (2352 us at 64 heads, 32-head tiles), 8 chunks (faster below 8 tokens, 4-25% slower above). Results: 4-9x over `triton_dsv4_prefill` (64 heads: 14.6 us at 1 token, 134 us at 128, 1618 us at 2048 = 106 TFLOP/s; 128 heads 2048 tokens 3080 us = 112 TFLOP/s). 25-30% of cuBLAS BF16. Open: a decode reader for the FP8 page-planar rows (drops the BF16 workspace).

**flashinfer trtllm-gen DSA (cap 10.7).** Check of every trait combination where SM100 picks trtllm-gen or FlashMLA (384 combinations of mode, dtype, q_len 1-6, nope/rope/topk, cache kind, LSE): SM120 picks a 13.2 mma core everywhere except (a) the packed FP8 sparse cache, which no runtime path sends (the DSA backend always passes `sparse_kv_cache=None`), and (b) the zero-RoPE KPool layout of GLM-5.3-Flash (nope 256, rope 0, top-k 2051), which went to `triton_dsa_*`. The 13.2 kernel now compiles the RoPE term out for rope 0 and declares it (`5c24472f2`): 29-134x over `triton_dsa_*` at 16 and 64 heads, 1-1024 tokens (decode 64 heads batch 128: 290 vs 36930 us; prefill 1024 tokens 2209 vs 295193 us; up to 125 TFLOP/s). Open then, done in task 15 (section 9.20): those cores split by token count, so (like the first V4 version) prefill rows depended on the batch; the fixed-chunk fold fixed it.

**Tests.** New: `test_hadamard_sm120.py` (float64 reference, recompile guard), `test_mla_decode_mma_sm120.py` (25: selection, float32 reference for 9 mask/shape cases x 2 dtypes, FP16, empty rows, strided out, graph replay, recompile guard), `test_dsv4_mma_sm120.py` (14: selection, float32 reference, strided q/out, batch invariance, recompile guard), zero-RoPE cases in `test_dsa_mma_sm120.py`, `test_sm120_runs_its_own_leaves` (which leaf replaces which), `test_hybrid_full_backend_resolution.py`, an MLA-mma case in `test_batch_shape_recompiles.py`. Kernel suite (GPU 7): 7551 passed, 2763 skipped, 0 failed. Runtime suite (GPU 6): layers 370, models 564, moe 72, cache 137, sampling 144, execution 195 passed, 0 failed. Top-level files: 6780 passed and 1 failed (`test_deepseek_v41_model ... [prefill]`, the batch-dependent split) before the V4 fix; after it 6781 passed, 0 failed (`rt_top3`), and `models` 564 passed. One intermediate full run also failed `test_cuda_40_layer_real_flatkv_and_moe[eager]` once; it did not reproduce in the next full run, 3 isolated runs or the `test_[a-d]*` subset (2394 passed), so it is recorded as an intermittent failure of unknown cause. Task 15 (section 9.20) tried to reproduce it on the baseline, the pre-fix tree and the current tree and could not.

**Not done.** CuTe DSL versions of the three attention kernels (the remaining 3x to the tensor-pipe floor); batch-invariant DSA prefill; an FP8 page-planar V4 decode reader; a rebuilt `fast_hadamard_transform` wheel (not needed).
## 9.20 Task 15: batch-invariant DSA attention, the intermittent v4.1 test, final verification (2026-11-08)
TokenSpeed `nctu6` at `990e3e6cd` (local, not pushed); DeepGEMM code unchanged since `76dc545` / `2f4e3e0` (`build312`). RTX PRO 6000, kernel tests and benchmarks on GPU 7, runtime tests on GPU 6, CUDA-graph timing (best of 5 replays).
**A. Batch-invariant DSA sparse attention** (`36211b436`, docs `ba8ad7c8b`). `triton_dsa_mma_{decode,prefill}` picked `num_splits` from the token count, so a token's row depended on its batch, the bug `1334fcd70` fixed for the V4 leaf. Same design now: a token's selection is cut into at most `MAX_CHUNKS = 8` chunks of whole 32-row tiles (`num_chunks_for(width)`, a function of the width alone); each chunk runs its own online softmax; the chunk states are folded in order by one `_merge` helper with explicit `fma`. One program folds them in registers. For small batches `plan_launch` still picks `num_splits` programs per (token, head tile), but each now runs a run of chunks and stores the state of every chunk, and the combine kernel folds all stored states with `_merge` in the same order (the V4 kernel gives each chunk its own program; a run per program keeps the old 16-way parallelism with 8 chunks and 2-4 programs at mid batch). So the bits do not depend on batch size, head-tile width (16 or 32) or split count. The two compile variants (`SPLIT` true or false) differ only in where the fold happens.
- Tests (`test_dsa_mma_sm120.py`, 67 passed): `test_rows_do_not_depend_on_the_batch` (decode and prefill, E4M3 and BF16, 16 / 40 / 128 heads, 600 tokens with full, short, one-token and empty rows and -1 holes: row slices of 1 to 600 tokens equal the full batch bitwise, output and LSE, and the test asserts that the plans of its batch sizes include both a split and an unsplit launch), `test_split_count_does_not_change_the_result` (split counts 1, 2, 3, 5, 8, 99 at both head tiles), `test_batch_size_does_not_recompile` (keyed by head tile and split-or-not), `test_bitwise_envelope_plans_the_mma_leaves` (the `KernelPlan` that `DSABackend` builds, both slot orders, both dtypes), `test_planned_call_matches_the_unplanned_call`, `test_any_selection_width_matches_the_float32_reference` (widths 1 to 4096).
- Declarations. The leaves now declare `batch_invariant` and `cuda_graph` (the graph benchmarks and 13.3 already ran them under capture) and the `slot_order` trait with both orders (they reduce the slots as they arrive, so they keep the top-k leaf's order; the facade passes the keyword). They no longer carry a `topk` trait: `DSABackend` plans with `topk` variable and `KernelPlan.select` drops every leaf that restricts a variable trait, so a declared feature alone would still have found nothing. Under `rl-bitwise` (`dsa_slot_order=sorted`) the plan now resolves to `triton_dsa_mma_*` on SM120 instead of failing; no in-tree model has been verified under that envelope (numerics.md), so this is plan-level support.
- Cost, 64 buckets (16 / 32 / 64 / 128 heads x decode batch 1 / 8 / 32 / 128, prefill chunks 128 / 512 / 2048 / 8192, E4M3 and BF16, top-k 2048, 16K history), same harness before and after: geomean 1.21x slower, worst 1.86x (E4M3 and BF16 decode of one request: 17-24 us instead of 10-14 us). E4M3 prefill +2-9% (up to +35% for 128-token chunks of 64-128 heads); BF16 prefill +5-9% at 16 heads and +28-37% at 32-128 heads; decode of 8-128 requests -5% to +51% (E4M3) and -5% to +36% (BF16). Cause: the second accumulator. The 32-head tile with 8 warps now uses 255 registers and spills 34 (E4M3) or 52 (BF16); it used 246 with no spills. Against the portable `triton_dsa_*` (same buckets; an 8192-token chunk at 128 heads takes 5.4 s there) it is still 22-114x faster on every bucket (30-110x E4M3, 22-114x BF16; geomean 72x, was 38-147x and 86x).
- Alternatives measured with the same harness (geomean / worst slowdown over the 64 buckets): 16 chunks with one program per chunk and the V4 rule "split when 2 x wanted programs >= chunks": 1.25x / 5.6x (128 heads, 8 requests stayed unsplit); 16 chunks with a run of chunks per program: 1.23x / 2.2x (1.4-2.2x slower for 8-128 requests of 64-128 heads, from the stored states: 16 x 128 heads x 2 KB per token); 8 chunks (shipped): 1.20-1.21x / 1.82-1.86x; 4 chunks (the V4 count): 1.32x / 3.6x (one request has 4 x 8 head tiles = 32 programs on 188 SMs). Warps and stages (E4M3 / BF16, 128 heads, 2048 tokens, unsplit, ms): 16 heads x 8 warps 11.7 / 14.0, 32 heads x 8 warps 7.8 / 12.0 (255 registers, 34 / 52 spills), 32 heads x 16 warps 11.3 / 14.5 (128 registers, 58 / 68 spills); 4 warps spill 108-466 values and are 2-5x slower; stages 1 and 2 are equal. The 32-head tile with 8 warps stays the best.
- Still open: a CuTe DSL kernel with a single accumulator (the chunk fold in shared memory or tensor-memory-free registers) would remove the spill cost together with the 3x gap to the tensor-pipe floor (section 9.17).
**B. `test_deepseek_v41_model::test_cuda_40_layer_real_flatkv_and_moe[eager]`: not reproduced.** It failed once in three full runs of `test/runtime/test_*.py` during task 14 (intermediate tree, failure text not kept). Trials, all on `nctu6` unless noted:
- the whole file 10 times (GPU 7): 150 passed, 4 skipped every time;
- the eager id alone, 150 iterations in one process: 0 failures; 73 iterations while a BF16 GEMM loop saturated the same GPU: 0 failures (stopped there); with `torch.use_deterministic_algorithms` plus `fill_uninitialized_memory` (every `torch.empty` NaN / max-int): passes, so no read of uninitialized memory;
- the eager forward 40 times in one process: the two outputs are bitwise identical every time (hashes) and their absolute maximum is 2.7 and 2.6, so neither nondeterministic kernels (atomics, split counts) nor an overflow near the BF16 range explain a non-finite result;
- five complete runs of the top-level directory (6781 passed, 286 skipped, 510 subtests each; one more run had 6780 passed and 1 failed in `test_control_server`, see below): eager passes in all six;
- the pre-fix V4 kernel (`4bfa4febf`, token-count split, the state the intermittent failure was seen in): eager 300 iterations and the file 10 times, only the known deterministic `[prefill]` failure (logits off by 0.31), eager passes; also with the poison fill;
- the baseline `892e00ce2` (`/tmp/ts_base`): `test_[a-d]*.py` three times (the subset that contains the file): 53 failed, 2345 passed each time, none of them in this test (known baseline failures that this round fixed: dots3, CLI config, device handle, compressed-tensors FP8 and other v4.1 tests); a complete top-level run cannot finish on the baseline (a baseline test ends the process at 46%, rc 138; `test_zero_output_prefill.py` has a repeated keyword and does not import).
It is recorded as not reproducible, with no known cause; the leading hypotheses, none supported by a run, are a race between concurrent suites that share the JIT caches and a transient out-of-memory with another process on the GPU. The fifth full run was made with the kernel suite running on GPU 7 at the same time on the same JIT cache, as the failing run was: eager passed (6781 passed, 286 skipped in 349 s) while the kernel suite passed beside it (7581 passed in 342 s).
One real flake was found and fixed on the way (`990e3e6cd`): `test_control_server.py` listened on fixed ports 28310-28340, so two runs of the suite on one host failed with "address already in use" (`test_reports_ready_after_bind` in the run that had another top-level run beside it). The tests now take free ports; two concurrent runs pass (20 passed each).
**C. Final verification** (GPU 6 and 7 only, JIT caches warm for TokenSpeed and fresh for DeepGEMM, `DG_JIT_CHECK_NO_SPILLS=1`):
- TokenSpeed kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, which contains every file added since task 8: `test_hadamard_sm120`, `test_mla_decode_mma_sm120`, `test_dsv4_mma_sm120`, `test_dsa_mma_sm120`, `test_dsa_indexer_sm120`, `test_dsv41_indexer_sm120`, `test_batch_shape_recompiles`): **7581 passed, 2763 skipped, 0 failed** (344 s); it was 7551 passed before the 30 new DSA tests.
- TokenSpeed runtime suite (every directory of `test/runtime`): cache 137 passed; diffusion 53; distributed 638 (106 skipped); entrypoints 40; execution 195 (7 skipped); layers 370 (51 skipped); models 564 (48 skipped); moe 72; sampling 144 (10 skipped); top-level files 6781 (286 skipped, 510 subtests). **8994 passed, 0 failed.**
- DeepGEMM (GPU 7, fresh cache, no-spill check): `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention` exit 0; `sm120_mqa_check.py` 46 cases OK; `sm120_blockscale_check.py --mode all --grouped` 170 cases ALL OK; `test_lazy_init --num-processes 2` on GPUs 6 and 7: 3 tests OK. The compute-sanitizer (memcheck and synccheck, one process per function, `DG_JIT_CHECK_NO_SPILLS=1`, fresh cache) reports 0 errors on the 22 functions of the task 11.5 list (all 8 of `test_einsum`, the 3 of `test_layout`, `test_hyperconnection`, `test_fp8_fp4` gemm / m-grouped contiguous / mixed-scale / MN-major, `test_bf16` gemm / m-grouped contiguous / masked / k-grouped, `test_paged_mqa_logits`, `test_sparse_mqa_logits`), now on the tree with the early PDL trigger; the three heavy ones (FP8 k-grouped and masked, `test_mqa_logits`) were not rerun, they did not finish under it before.
- End to end, in-process `Engine`, greedy, 48 new tokens (all correct):
  - Qwen3.6-35B-A3B (BF16 MoE, TP1, GPU 6): "The capital of France is" -> " Paris, a city renowned for its rich history, culture, and iconic landmarks. ..."; primes -> "2, 3, 5, 7, 11".
  - Qwen3.5-122B-A10B-NVFP4 (TP2, GPUs 6 and 7): " Paris.\n\nThe capital of France is Paris. ..." (greedy repetition); primes -> "2, 3, 5, 7, 11".
  - Qwen3.8-27B-NVFP4 (TP1, GPU 6): " Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome. ..."; primes -> "2, 3, 5, 7, 11".
  - gemma-4-31B-it with `<bos>` (TP1, GPU 6): " Paris. ..." (greedy repetition); primes -> "2, 3, 5, 7, 11".
- Environment-only, unchanged: `test_locality_domain` (MLOPart), `test_mega_moe` (NVLink) and the sanitizer's `test_legacy` entry (no triton in the venv).

## 9.21 Paged FP4 MQA logits on SM120: V4.1 serving goes native without a cache-format change (task 16, 2026-11-08)
DeepGEMM `nctu6` `6b2df9e` (kernel), on `0301a69`; TokenSpeed `nctu6` `76b2134a7` (Triton specialization fix), `f01d594a5` (leaf), `6d4188f43` (README). RTX PRO 6000, DeepGEMM checks on GPU 7 with a fresh JIT cache and `DG_JIT_CHECK_NO_SPILLS=1`, TokenSpeed kernel tests and benchmarks on GPU 7 and 6, runtime tests and the end-to-end run on GPU 6.

**The contract SM100 scores (step 1).** The 68-byte V4.1 index row (`cache_format="v41"`, TokenSpeed `ops/attention/dsv41`) is page-planar: 64 rows x 64 bytes of E2M1 pairs (low nibble first), then 64 x 4 bytes, each a packed UE8M0 word with the four 32-element scales of the row's 128 values. Queries are packed the same way (`pack_index_inputs`: uint8 `[T, H, 64]` and int32 `[T, H]`), 32 heads, head dim 128, page size 64. The sm100 leaf calls `fp8_fp4_paged_mqa_logits(q=(int8 [T, 1, H, 64], int32 [T, 1, H]), kv_cache=[pages, 64, 1, 68], weights FP32, context_lens [T, 1], ...)`, i.e. next_n 1 with every query its own request, FP32 weights and **FP32 logits** (the pre-2.8.2 `logits_dtype` keyword); Full passes in one request's chunk call the contiguous `fp8_fp4_mqa_logits` the same way; Reindex passes call `fp8_fp4_paged_sparse_mqa_logits` (BF16 weights and logits). On SM120 only the paged MX kernel was missing: `attention.hpp` asserted FP8 Q for arch 12.

**Design (one kernel path).** `sm120_fp8_paged_mqa_logits` gains `kIsMX, kIsFP4` template parameters:
- Rows are `kRowBytes` = 64 (FP4) or 128 bytes; the Q and KV tiles, the swizzle alignment and the TMA boxes follow it. FP4 tensor maps use 4-bit elements (`16U4_ALIGN8B`, packed smem), as the contiguous FP4 kernel does.
- The Q stage gains a scale tile (one int32 per head and token, TMA'd with Q and the weights, counted in the same `expect_tx`). The per-token KV scale slot (float for FP8) holds the UE8M0 word.
- The MMA is `warp_mma_16xNxHd_block_scaled_ss` (section 5.6 lane map; `kind::mxf4nvf4 ... scale_vec::2X m16n8k64` for FP4, `kind::mxf8f6f4 ... scale_vec::1X m16n8k32` for MXFP8), so the post-MMA float scale becomes 1. Everything else is the FP8 kernel: SM90 paged scheduler and metadata, SPLIT_KV 256 (4 math warp groups x 64 tokens), the stage search (3,3)/(2,3)/(2,2)/(2,1) with at least 2 Q stages, the proxy fences before every `empty` arrive, the 64/104 register split.
- Host: `sm120_fp8_paged_mqa_logits` takes `q_sf`; `attention.hpp` drops the FP8-only assert for arch 12, widens BF16 weights to FP32 (the kernel keeps FP32 weights in shared memory), and takes `logits_dtype` (BF16 default for MX, FP32 on request; only SM120 MX accepts a non-default dtype). The block-scaled contiguous kernel takes the weights and logits dtypes as template parameters too. `deep_gemm/compat.py` wraps both entry points so they accept `clean_logits` / `logits_dtype` again, which TokenSpeed passes.
- MXFP8 at next_n 2 x 64 heads (128 Q rows, 4 K steps) spills (108 B) in 104 registers and is rejected at the host; MXFP4 at 128 rows fits. No caller uses MXFP8 paged.

**Ruled out:**
- More math registers (48/112): 128 x 48 + 512 x 112 = 63488 exceeds the 61440 the 640-thread CTA is launched with (96 per thread), so `setmaxnreg.inc` never returns. The first check hung; with a timeout it showed as a deadlock, not a wrong result. The kernel now says so in a comment.
- For the MXFP8 spill: reloading the B scale words from shared memory at every K step (44 B left), and a rolled K loop on top of that (MXFP4 then spills 112 B). Both reverted; `mma/sm120.cuh` is unchanged.
- Switching SM120 to the 132-byte `v4` rows (about 45% fewer tokens per byte), which the task ruled out: not needed now.

**Correctness.** `scripts/sm120_paged_mqa_blockscale_check.py` (new, 51 cases: MXFP4 and MXFP8, the pipeline-wrapping shapes of `sm120_mqa_check.py` plus 1 x 70, 1 x 130000, 128 x 1500, 16 x 32768; BF16 and FP32 logits; FP32 and BF16 weights; pages padded by 80 bytes) against a float64 dequantized reference: FP32 logits diff 2e-16 to 3e-15, BF16 1.2-1.9e-6, every launch bitwise repeatable. `sm120_mqa_blockscale_check.py` gained FP32-weight/FP32-logit cases (diff 1.6-2.2e-15). `tests/test_attention.py` now runs `fp8`, `mxfp4` and `mxfp8` on SM120 (head dim 128 for MX): 159 cases (64 prefill, 95 paged; 36 MXFP4, 31 MXFP8) pass with no spills. `sm120_mqa_check.py` (46), the sparse checks and the contiguous MX check are unchanged and pass.

**Kernel benchmark** (`scripts/sm120_paged_mqa_bench.py --fmts fp8 mxfp4 mxfp8`, cold L2, best of 3 x 30 calls, us, FP8 / MXFP4; MXFP8 within 0.96-1.02x of FP8). V4.1 decode shapes, H 32, next_n 1:

| B \ ctx | 1K | 4K | 16K | 32K |
|---|---|---|---|---|
| 1 | 9.6 / 9.4 | 9.9 / 9.5 | 11.6 / 10.2 | 15.0 / 12.0 |
| 8 | 10.1 / 9.8 | 15.3 / 11.7 | 28.2 / 19.6 | 43.7 / 28.7 |
| 32 | 14.8 / 12.1 | 28.5 / 19.5 | 77.4 / 45.9 | 138.7 / 80.6 |
| 128 | 28.5 / 20.0 | 77.2 / 45.6 | 254.7 / 146.2 | 450.4 / 266.0 |

MXFP4 is 1.02-1.74x faster (half the bytes; 1.07 TB/s of KV at 128 x 32K against 1.23 TB/s for FP8, so it is not yet at the FP8 kernel's bandwidth). The section 9.13 shapes (H 64, next_n 2) give 1.51-1.72x, and the FP8 times there are unchanged (43.9 / 45.7 / 78.2 / 78.8 / 44.1 / 44.4 us).

**TokenSpeed (step 3).** `deep_gemm_sm120_dsv41_index_topk` (window 12.0-12.9, registered only when `deepgemm_supports_sm120`, traits `index_k_format` `{fp8_scaled, mxfp4}`) tells the row formats apart by width. Packed rows: Full passes on the paged kernel, one request's chunk (zero-stride table) on the contiguous kernel over the gathered pages, FP32 logits, then the cleaning, block maxima and CuTe DSL DeepSelect (compact profile) of the FP8 path; the upstream `deep_select` package stays sm100-only. The facade marks 68-byte rows native on SM120 builds without asking for the upstream package.

Leaf benchmark (CUDA graph, 20 replays x 3, median; 32 heads, top-k 512, 2048 candidate blocks, the shapes of 13.1; Triton / native, us). Decode, 1-128 requests x 1K-32K rows: Full 108.5/18.6 (1 x 1K) to 2354/264 (128 x 32K), 5.6-16x; Full that sources the pool 7.7-33x; Reindex 2.2-22x. Chunked prefill, 128-4096 queries x 4K-32K rows: Full 11-46x (4096 x 32K: 69.1 ms / 2.29 ms), source 22-88x, Reindex 6.8-22x. Every point wins, so packed rows need no rule against Triton. Row overlap with the Triton leaf 0.957-1.000 (it rounds through BF16).

**Reindex: the candidate-pool scorer.** DeepGEMM's SM120 sparse paged kernel reads MX rows and only the pool, as Hopper's CuTe scorer does, and the leaf can use it (page stride a multiple of 128 bytes, met by every 4352-byte page; `_sparse_candidate_scores` now takes the selector's row alignment instead of asking the upstream package). It is correct but slow: one wave of CTAs (section 9.6), about 85 us at any size (task 35 rewrote the kernel, section 9.39: the paragraph below is the state before it). Scoring the whole packed history and gathering the pool is cheaper almost everywhere (dense / sparse time):
- histories up to the pool (16384 rows): 0.31-0.70;
- paged, past the pool: 0.40-0.59 below 2^21 scores (8 x 131072, 32 x 49152), 1.05-3.49 from there (128 x 32768 1.05, 32 x 65536 1.59, 64 x 131072 3.49);
- one request's chunk: 0.46-1.04 at twice the pool, 0.74-0.90 at four times with 1024-4096 queries, 1.47 there with 128, 2.05-2.81 at eight times.
`_sparse_reindex_wins` encodes that split. **`fp8_indexer_wins` is not relaxed:** it governs the FP8 `index_v4` rows, for which SM120 still has no pool-only scorer (the sparse kernel takes MX rows only), and the runtime does not store those rows on SM120. For the rows it does store, no rule is needed.

**Tests (TokenSpeed).** `test_dsv41_indexer_sm120.py` (12): exact top-k against a float64 reference over the dequantized rows and queries for Full, candidate-source Full (with the pool checked against the reference block maxima) and Reindex, on 1 x 700, 8 x 4096, 33 x 20000 (ragged), 96 x 4096 (one request's chunk) and 64 x 33000 (the sparse Reindex path). Full picks are at or above the k-th reference score to 1e-9; Reindex picks to 2^-7, since the sparse kernel writes BF16 logits (worst measured gap 5.6e-3, 0.5-0.8% of picks below the k-th score). Also a small pool taken whole, the `_sparse_reindex_wins` rule, registry selection, and a recompile guard (8 new shapes after warming 9, each running all three passes: no Triton compile and no new DeepGEMM JIT kernel). The guard found `_finish_topk_kernel` (13.1) keying binaries on the selection width and row strides, which follow the histories; they are runtime now (`76b2134a7`). Registry: `test_sm120_excludes_sm100_only_leaves.py` (the packed leaf is SM120's own leaf for `mxfp4` rows) and `test_deepgemm_sm120_gating.py` (both formats select the SM120 leaf). Kernel subset (registration, selection, gating, the dsv41 / DSA / DeepSelect files under `nvidia` and `ops`): 1341 passed, 446 skipped. V4.1 runtime tests (`test/runtime/test_deepseek_v41_*.py`, GPU 6): 767 passed, 76 skipped.

**End to end.** The 476 GB checkpoint does not fit two GPUs, and its layer graph (KV sources 2/8/14/20, index sources every 4-6 layers, candidate source 20, engram, hyper-connections) does not cut down to a few layers consistently. So the in-process `Engine` ran the real config with all 40 layers and the real attention and index geometry, 16 routed experts instead of 384, a 10000-entry engram vocabulary, dummy weights, TP1 on GPU 6, prefix caching off. `TOKENSPEED_KERNEL_VERBOSE=1` shows `dsv41_index_topk -> deep_gemm_sm120_dsv41_index_topk` (with `cache_format="v41"`, the SM120 default). Against the same run with `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_DSV41_INDEX_TOPK=triton_dsv41_index_topk`: prefill 10.2-11.1K against 9.2-9.8K tok/s (1.11-1.19x, 4K-32K prompts, batch 1 and 4); TPOT 50.9 / 50.6 / 26.8 / 47.8 against 52.1 / 51.8 / 28.4 / 49.5 ms (1.02-1.06x, noisy: a difference of two timed calls). Greedy tokens are identical in all four cases. With these weights the index step is a small share of the step time.

**Open:**
1. The sparse MX scorer's one-wave design (section 9.6) is the reason Reindex rarely reads only the pool; a multi-wave version would make pool-only Reindex win from shorter histories. Done in task 35 (section 9.39): pool-only Reindex wins from twice the pool.
2. Packed paged MXFP4 reaches 1.07 TB/s against 1.23 TB/s for FP8 at 128 x 32K; the extra scale tile and the smaller rows leave some bandwidth unused.
3. MXFP8 paged at 128 Q rows (rejected), TP-sharded `dsv41_index_candidates` for packed rows (still FP8 rows only), and fusing RoPE and query packing into the SM120 leaf (the facade applies RoPE first, one extra launch).
4. A real V4.1 run with checkpoint weights (needs more than two GPUs).
## 9.22 Tensor-core headroom of the SM120 attention kernels: CuTe DSL DSA cores (task 17, 2026-11-08)
TokenSpeed `nctu6` from `6d4188f43` (local, not pushed): kernel `6e75d3e77` (Triton BF16 prefetch, DSA and V4), `bf5bbc37f` (CuTe DSL DSA cores), docs `ff76063b8`. DeepGEMM code unchanged (`a580d14`, `build312`). RTX PRO 6000, benchmarks and kernel tests on GPU 7, runtime tests on GPU 6, CUDA-graph timing (best of 5 replays). `ncu` is still blocked (`ERR_NVGPUCTRPERM`), so the evidence is ablation, register/spill counts from the cubins (`cuobjdump -res-usage`, SASS histograms) and roofline. Floors as before: E4M3 QK at 779 TFLOP/s, 16-bit PV at 404 TFLOP/s (cuBLAS); for 128 heads and an 8192-token chunk that is 8.6 ms (E4M3) and 11.6 ms (BF16).
**Baselines (same harness, 64 buckets: 16/32/64/128 heads x decode batch 1/8/32/128 with 16K history and prefill chunks 128/512/2048/8192 with 32K history, top-k 2048, E4M3 and BF16).** "pre" is the Triton core before batch invariance (`36211b436^`), "shipped" the batch-invariant one (`36211b436`): shipped/pre = 1.23x geomean, 1.65x worst (decode of one request 19-23 us vs 13-14 us), re-measured here (section 9.20 had 1.21x / 1.86x).
**Step 1: the batch-invariance cost in Triton.** Register counts per variant (32-head tile, 8 warps): pre 232 / 221 registers (E4M3 / BF16) and no spills; shipped 254 + 34 spills / 255 + 28 spills; the second accumulator is most of it, but the whole next tile held in registers (manual prefetch) is the rest. Measured (128 heads, E4M3 / BF16 decode 128 and prefill 2048 unless noted):
- Running accumulator in an L2-resident global scratch, folded in place at chunk ends (one register accumulator): spills 34 -> 0 (E4M3), but E4M3 603 vs 531 us and 8.97 vs 8.01 ms (13% slower; the fold reads and writes 64 KB per CTA per chunk, about 0.5 GB at decode 128); BF16 715 vs 839 us. Also needs a scratch of 2 GB at an 8192-token chunk. Dropped.
- One flat tile loop with the fold at chunk ends: 58 / 34 spills, E4M3 853 us (1.6x slower). Dropped.
- `tl.range(num_stages=2/3)` instead of the manual prefetch: 590-904 us. Dropped.
- One chunk (`MAX_CHUNKS = 1`, bitwise the same structure): BF16 still 852 us, so the cost is the fold's live accumulator, not the chunk count.
- **No row prefetch: BF16 533 us / 8.00 ms (pre 579 / 8.99, shipped 839 / 12.05), E4M3 561 / 8.63 (worse).** Prefetching only the next slots: BF16 551 / 7.68, E4M3 579 / 8.45. Shipped: E4M3 keeps the full-tile prefetch, BF16 prefetches the slots only (`PREFETCH_ROWS` keyed by the cache dtype). Over the 64 buckets: BF16 1.24x faster than shipped (1.04-1.64x), 0.96x of pre; E4M3 unchanged (0.98-1.03x); all 64: 0.90x of shipped, never slower than 1.02x. Bitwise results unchanged (all `test_dsa_mma_sm120.py` cases pass).
- 16 chunks instead of 8 (same kernel): one request 0.62-0.68x (back to pre-invariance speed), but 1.1-1.5x slower for 32-128 requests of 32-128 heads (stored chunk states; even unsplit, 16 folds cost 1-3%). The chunk count may depend on heads but not on tokens, so it cannot be chosen for small batches only. Kept 8.
- **V4 prefill has the same cost** (BF16, row prefetch): the same change gives 1.24-1.31x from 128 tokens (64 heads 2048 tokens 1266 vs 1660 us, 128 heads 2447 vs 3212 us; 640 rows) and equal below 32 tokens; shipped in `6e75d3e77`.
**Step 2: the CuTe DSL kernel** (`dsa/_cute_dsl/sparse_mma.py`, leaves `cute_dsl_dsa_mma_decode` / `_prefill`, solution `cute_dsl`, `PERFORMANT + 1`, same traits/features/signatures as the Triton cores, so they win selection on SM120 wherever the CuTe DSL imports; registered in `dsa/cute_dsl.py`). Written in the style of the in-tree DeepSelect: raw shared memory from `SmemAllocator`, inline PTX for `cp.async`, `ldmatrix`, `mma.sync`, `cvt`, `ex2`, compiled with `cute.compile(..., --enable-tvm-ffi)` and cached by `functools.cache` per (dtype, head tile, rank, RoPE width, device); the CUDA-graph warm-up runs compile before capture.
- CTA = 16 or 32 heads of one token, 4 warps per 16-head group (128 or 256 threads), one CTA per SM (255 registers, 87-99 KB of shared memory). The selected rows of a tile are gathered with `cp.async` (zero-fill for `-1` slots and masked columns) into a two-stage ring; the next tile's slot is loaded one iteration early so the issue never waits on it.
- Channel split for `O += P V`: warp (g, c) multiplies heads 16g.. by latent channels 128c.., so one 16 x 128 FP32 accumulator per warp (64 registers) plus the running fold (64). V fragments by `ldmatrix.trans` (E4M3: one PRMT and two `cvt.rn.f16x2.e4m3x2` per 4 MMA columns; BF16: as loaded).
- E4M3 scores (key split): 64-row tiles; the query tile is staged in shared memory and each warp scores 16 of the 64 rows over all 576 channels (`ldmatrix.x4` for Q and K, `m16n8k32`, two accumulators per tile to halve the dependency chain), then the four row slices are exchanged through 8 KB of shared memory with one 128-thread named barrier per head group. Rows are unpadded and XOR-swizzled in 16-byte units (latent chunks with `row & 7`, RoPE chunks with `row & 3`): every `ldmatrix`, `ldmatrix.trans` and V load is conflict-free and the CTA uses 100864 of 101376 bytes.
- BF16 scores (channel split): a 64-row BF16 tile plus the query tile does not fit, so 32-row tiles padded by 16 bytes; each warp keeps a quarter of the query channels in registers (36), scores all 32 rows over them (`ldmatrix` + `m16n8k16`) and the four partial tiles are added in a fixed order.
- FP32 online softmax in registers (each warp of a head group computes the same values; the row max is taken on the raw scores and the scale is applied in one FMA), probabilities packed straight into the PV A fragments; the rescale is skipped on a chunk's first tile (exactly zero accumulator).
- Fold: the Triton scheme (chunks of whole tiles by width alone, `num_chunks_for(width, rows)`, 8 chunks), with `_fold_scales` / `_fold` / `_finish` / `_lse` written once with explicit `mul.rn` / `fma.rn` / `div.rn` / `ex2.approx.ftz` and called by both the attention kernel and the combine kernel, so split and unsplit launches give the same bits. `plan_launch`: the Triton head-tile rule; splits only while the unsplit CTAs are fewer than half the SMs (32-head CTAs) or the SMs (16-head CTAs), aiming at 2/3 of the SMs up to 8 tokens and 2.75 CTAs per SM above.
- Mixed query/cache dtypes (never sent by the DSA backend) run the Triton core; the choice depends on the dtypes only.
**Path to the shipped version** (128 heads, E4M3 prefill 2048 / BF16 prefill 8192):
| version | E4M3 p2048 | BF16 p8192 | note |
|---|---|---|---|
| Triton (after step 1) | 8.01 ms | 30.5 ms | |
| v1: 32-row tiles, channel split for both dtypes, padded rows | 5.36 | 25.1 | first try already 1.5x / 1.2x |
| ablation of v1 (E4M3) | all MMAs removed: 4.68; gathers removed: 4.49; both: 3.51 | | not tensor-bound: per-tile softmax/fragment work of 2 warps per scheduler |
| v2: E4M3 64-row tiles, key split, swizzle | 4.32 | | |
| + per-step addresses in registers | 4.76 | | worse: 152 B of spill |
| + lane-constant swizzle offsets (4 per operand) | 3.64 | 24.6 | spill 104 B |
| + ldmatrix for BF16 K, FMA softmax, packed P | 3.48 | 24.3 | spill 48 / 56 B (outside the hot loop) |
`acc.fill(0.0)` instead of explicit zeroing loops raised the BF16 spill to 136 B and cost 1.4x (34.7 ms); the explicit loops stay (they print a DSL "static loop" note at compile time). 16-head CTAs with 2 CTAs per SM (E4M3, 46 KB) were no faster; 3- and 4-stage rings (E4M3 v1) were 2-4% slower than 2.
**Results** (64 buckets, final code, `bench_dsa_mma_sm120.py`):
| | E4M3 geomean (range) | BF16 geomean (range) |
|---|---|---|
| CuTe vs Triton after step 1 | 1.91x (1.49-2.34x) | 1.16x (1.00-1.42x) |
| CuTe vs shipped Triton (`36211b436`) | 1.92x (1.49-2.34x) | 1.44x (1.07-2.06x) |
| CuTe vs pre-invariance Triton | 1.62x (0.95-2.17x) | 1.12x (0.65-1.48x) |
128 heads (shipped Triton / Triton after step 1 / CuTe): E4M3 decode 1/8/32/128 = 19.9/19.4/13.0, 52.0/52.0/30.2, 158/158/80.6, 531/518/238 us; prefill 128/512/2048/8192 = 0.52/0.52/0.24, 1.95/1.96/0.87, 8.01/8.01/3.47, 31.7/31.7/13.7 ms (342 TFLOP/s, 63% of the floor). BF16 decode 22.7/21.7/20.4, 62.6/55.7/45.3, 178/161/140, 839/528/421 us; prefill 0.84/0.53/0.42, 3.20/1.95/1.55, 12.05/7.69/6.13, 47.2/30.5/24.3 ms (192 TFLOP/s, 48% of the 16-bit floor). No bucket is slower than the Triton core; the closest are BF16 16 heads (1.00-1.05x, 16-head CTAs of 4 warps, one per SM) and BF16 decode of one request (20 us, 1.5x the pre-invariance 13-14 us; E4M3 is back at 13 us).
**Tests.** `test_dsa_mma_sm120.py` (the reference, batch-invariance, width and zero-RoPE cases run for both solutions; new: CuTe split and head-tile invariance at RoPE 64 and 0, recompile guard over 11 token counts x 3 head/width pairs for both dtypes (`compiled_attention` / `compiled_combine` cache misses unchanged), `out=` and three cache layouts, the mixed-dtype route bitwise equal to the Triton core, the CuTe plan), `test_sm120_excludes_sm100_only_leaves.py` (SM120 now runs `cute_dsl_dsa_mma_*`). Focused kernel set (DSA mma, V4 mma, DSA indexer, DeepSelect, FlashMLA DSA DCP, flashinfer nope DSA, SM120 leaves, kernel API selection, kernel plan, V4.1 registration, DSA ops/facade, batch-shape recompiles): 929 passed, 435 skipped, 0 failed. Runtime DSA/MLA/V4 set (dots3-note, DSA, FlashMLA, GLM-5.3-Flash, DeepSeek V4/V4.1, MLA, K3 MLA files): 1043 passed, 136 skipped, 0 failed. dots3-note pins its DSA leaves to solution `triton` (`dots3_note.py`, for implementations that would materialize its strided cache), so the runtime tests keep the Triton core there; with the pin switched to `cute_dsl` in the mirror, `test_dots3_note_model.py` + `test_dots3_note_nextn.py` select `cute_dsl_dsa_mma_*` and pass (14). No other runtime test reaches a real DSA core on SM120 (the generic-backend tests use registered stubs).
**Ruled out (don't retry without new evidence):** an L2 scratch for the running accumulator in Triton; a flat tile loop with in-loop folds; `tl.range` pipelining; 16 chunks for every shape; in CuTe, per-step address registers, `acc.fill`, 2 CTAs per SM of 16 heads, 3-4 ring stages, splitting at 128 or more unsplit 32-head CTAs.
**Still open:**
1. Step 3 (done in 9.23): CuTe ports of `triton_mla_mma_decode_with_kvcache` and `triton_dsv4_mma_prefill`. The DSA BF16 path is the template for V4 (512-wide query, the sink as the first chunk's initial state, int64 row ids, 640-row selections); MLA needs a contiguous page-range gather, the causal/window mask and the `q_len` query axis. Expected gain about the BF16 DSA one (1.2-1.4x) for V4 and up to the E4M3 one for MLA with an E4M3 cache.
2. BF16 DSA at 48% of the floor: the channel split pays 48 FP32 adds and 16 shared loads per warp-tile for the partial scores; a 64-row BF16 tile needs the query out of shared memory (36 KB), which does not fit next to two 74 KB stages.
3. E4M3 at 63%: the remaining time is the softmax and V conversion of two warps per scheduler (both head groups synchronize on the stage barrier); a producer warp with mbarrier stages and ping-pong head groups would overlap them but needs registers the CTA does not have (255 per thread at 256 threads).
4. Decode of one request (BF16 20 us, E4M3 13 us) is bound by 8 chunks per selection; more chunks only help when the batch is tiny and the chunk count cannot follow the batch.
5. Lift the dots3-note `triton` pin on SM120: done in 9.23.

## 9.23 CuTe DSL V4 prefill and MLA decode, dots3-note pin (task 17 continued, 2026-11-08)

TokenSpeed `nctu6` from `ff76063b8` (local, not pushed): `9bf202435` (dots3-note feature selection), `2e35979b7` (V4 prefill), `58fe2ef22` (MLA decode), `143876e60` (SM120 leaf test), docs `03587ef78`. DeepGEMM code unchanged (`354889a`, `build312`). Same GPUs, harness and constraints as 9.22 (`ncu` blocked: ablation and roofline only).

**dots3-note pin.** `Dots3NoteDSABackend` and its router set `kernel_solution = "triton"` for every DSA leaf because the compact dots3-note latent cache is a strided view and a core that reshapes it into flat rows copies it in full each call. That reason is about how a core reads the cache, not about Triton, so it is now a feature: cores that read `kv_cache` through its own page and token strides declare `strided_kv_cache` (`STRIDED_KV_CACHE_FEATURE` in `ops/attention/dsa/__init__.py`; the Triton portable and mma cores and the CuTe cores declare it, FlashMLA / trtllm-gen / gluon do not), `dsa_decode` / `dsa_prefill` take `features=`, and `DSABackend.core_features` (empty by default, `{strided_kv_cache}` for dots3-note) joins the kernel plan's features. SM120 selects `cute_dsl_dsa_mma_*` for dots3-note; SM90 gets `triton_dsa_*` as before (selection test for both). `test_dots3_note_*.py` 41 passed; DSA/V4.1/dots3 runtime subset 352 passed.

**V4 prefill (`cute_dsl_dsv4_mma_prefill`).** No new kernel class: the DSA CuTe `_SparseMma` takes a `v4 = (sink, wide_ids)` compile-time variant (BF16 rows of a flat workspace, row id x row stride with no divide, int32/int64 ids checked against the row count in the kernel, query/output addressed by their own strides, sink as chunk 0's initial state `m = sink * log2 e`, `l = 1`; a split CTA past chunk 0 resets it; a CTA with no tiles stores the sink-only state). The DSA code path is unchanged (its 64 buckets re-measured: 1.49x geomean both dtypes, same per-bucket times within 1%). Host side `dsv4/_cute_dsl/sparse_mma.py`: chunks by width alone (`MAX_CHUNKS = 4`, 32-row tiles, the Triton leaf's count), `plan_launch` aiming at 2/3 of the SMs with one chunk per CTA when split. Registered `PERFORMANT + 1`, same signature, traits and features as the Triton leaf plus `num_q_heads_min = 17`.
- Sweeps (640 rows): chunk count 1/2/4/8 for the CuTe kernel: below 16 tokens 8 chunks gives 10.9 vs 13.7 us at one token, but from 16 tokens 8 is slower and 1-2 lose 2-3x below 16 tokens; the count may not follow the batch, so 4. Head tile x split: 16-head CTAs win only up to 32 CTAs (64 heads x 8 tokens: 32-head 14.4 us vs 16-head 19.1); the first plan (the DSA `plan_launch`, 16-head CTAs up to 8 tokens) lost 1.1-1.2x at 4-16 tokens and was replaced.
- 16 heads: the 4-warp CTA (one per SM) is 0.70-0.79x of the Triton leaf from 64 tokens (Triton's 8-warp program covers twice the channels per CTA). The head rule keeps it on Triton; by head count, so a model never mixes leaves and replayed rows stay bitwise.
- Results vs `triton_dsv4_mma_prefill` (sink, 1-4096 tokens, 117 points): 32/64/128 heads x width 128 1.10-1.13x (0.89-1.63x), width 640 1.13-1.15x (0.99-1.32x), width 1152 1.13-1.16x (1.02-1.39x); 1.13x overall. 128 heads x 4096 tokens: 4.38 vs 4.85 ms at 640 rows (157 TFLOP/s), 7.30 vs 8.15 ms at 1152 (170 TFLOP/s, 42% of the 404 TFLOP/s BF16 floor). Losses only at window-only selections (128 rows) of 1-2 tokens (7.2 vs 6.4 us, launch-bound); a token-count rule would switch kernels with the batch and break the replay invariance, so they are recorded and kept.
- Tests (`test_dsv4_mma_sm120.py`, 34): float32 reference (now also int64 ids with an odd width), strided q/out and bitwise batch invariance for both leaves; CuTe sink with LSE and rows with nothing live in split and unsplit launches (direct call; the facade does not combine sink and LSE); compile cache guard (`compiled_attention` / `compiled_combine` misses unchanged over token counts, widths 128-1408, row counts and id dtypes); selection by head count.

**MLA decode (`cute_dsl_mla_mma_decode_with_kvcache`).** `mla/_cute_dsl/decode_mma.py`, a subclass of `_SparseMma` that keeps the tile math (the shared per-tile softmax and `O += P V` step is now `_SparseMma._softmax_pv`, used by both kernels) and replaces the gather and mask with the Triton leaf's: a CTA owns 16 or 32 `(query, head)` rows of one request, walks the contiguous position range `[lo, hi)` its rows can see (`lo` aligned down to a tile), resolves each position through the page table (`pos >> log2(page_size)`, pages of 16-128), and masks each row to `first <= p < visible` (causal block, non-causal block, window). E4M3 caches use the 64-row key-split path with the query tile copied into swizzled shared memory row by `(query, head)`, BF16 the 32-row channel-split path; FP16 keeps the Triton leaf. Split and partials as the Triton leaf (`plan_launch` copied: shape by batch size only, up to 128 splits, not `batch_invariant`).
- First version: 1.38x geomean but 0.75x at one request of 16 heads x 1K (23.6 vs 17.8 us). Ablation: the merge kernel looped over 128 splits serially, each a 2 KB load; most splits are empty at short contexts. Fix: one thread per split loads its `(m, l)`, the maximum and the weighted `l` are reduced by shuffles and shared memory in a fixed order, and the accumulator loop loads 8 splits at a time and skips weight-zero splits (whose rows an empty CTA no longer writes). One request of 16 heads: 14.4 us (1.23x); no point below 0.97x.
- Results vs `triton_mla_mma_decode_with_kvcache` (1/2/4/8/16/32/64/128 requests x 1K/4K/16K/32K, `Q` = 1; 128 points): BF16 16 heads 1.09x (0.97-1.40x), BF16 128 heads 1.67x (1.23-1.95x, 213 TFLOP/s at 128 x 32K, 5.48 vs 10.17 ms), E4M3 16 heads 1.29x (1.01-1.96x), E4M3 128 heads 1.94x (1.23-2.31x, 355 TFLOP/s, 3.29 vs 7.60 ms); 1.46x overall. `Q` = 4/8 (16/128 heads, 1/16/128 requests, 4K/32K): 1.84x (1.13-2.34x), E4M3 128 heads 380 TFLOP/s (70% of the E4M3 QK + FP16 PV floor). The ties are BF16 16 heads from 16 x 16K up, where both leaves stream the cache at 1.44-1.46 TB/s (the DRAM roofline); smaller caches stay in the 128 MB L2 across replays. Selected for every BF16/E4M3 shape (no shape loses more than 3%).
- Tests (`test_mla_decode_mma_sm120.py`, 56): the float32 reference over causal / non-causal blocks, windows and `q_len` 1-8, strided output and graph replay for both leaves; empty and short requests in split and unsplit launches; compile cache guard over batch, lengths up to 70K, page sizes 16-128, masks and `q_len`; FP16 keeps Triton.

**Verification.** Kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, GPU 7): 7712 passed, 2 failed, both in `test_sm120_excludes_sm100_only_leaves.py` expecting the Triton MLA/V4 leaves, updated (`143876e60`; that file plus `test_kernel_api_selection.py` 274 passed). Runtime MLA/V4/V4.1/DSA/dots3/K3/FlashMLA files plus `layers/` and `models/` (GPU 6): 3030 passed, 1 failed: `models/test_whisper_cross_kv_write.py::test_graphed_encoder_prefill_equals_eager_and_never_syncs[1]` (stream-capture invalidated in a torch linear; the file passes alone, 19 passed, and does not reach the changed kernels).

**Ruled out (don't retry without new evidence):** V4: 8 or 1-2 chunks, 16-head CTAs past 32 CTAs, the DSA split rule, a token-count rule for window-only rows. MLA: the serial split merge.

**Still open:** BF16 DSA (48%) and V4 (42%) above the floor (the channel split's partial-sum exchange; see 9.22 item 2); V4 at 16 heads needs an 8-warp CTA (two channel groups per head group, registers permitting); decode of one DSA request (8 chunks); a V4 decode reader for the FP8 page-planar rows.
## 9.24 Task 18: smaller SM120 follow-ups (2026-11-09)
TokenSpeed `nctu6` (local, not pushed): `10be2bcb8` (whisper GC), `9850d5ddb` + `0375fbd7e` (DeepSelect zero-base fix), `5406a4c1b` (selection test) (host-light DSA prefill), `c13e40aa7` (DeepSelect candidates), `8bacf076b` (direct V4/V4.1 decode reader), docs `4da186931`. DeepGEMM `nctu6`: `4ec333b` (batched BF16 einsum), `721cc2c`, `5db97ac` (benchmark scripts). RTX PRO 6000, GPU 7 for timings (CUDA graph unless stated), GPU 6 for tests.

**1. V3.2 DSA prefill, tiny chunks.** The leaf lost to the Triton leaf at 128 tokens, 277 vs 144 us (9.16). Profile (`torch.profiler` + per-segment host timers, 128 tokens, no prefix): 22 GPU kernels summing 38 us, but 360 us of host time per call (back-to-back eager calls: the GPU is idle and the leaf is launch-bound). Segments, us: `quantize_fp8` 71 (a selector call, a signature, a Triton launch), `combine_topk_weights` 31, the key gather 63 (`//`, `%`, two advanced-index kernels, `as_strided`), candidate lengths 40 (clamp, full_like, minimum, copy), `out.fill_(-1)` and output checks 14, `zeros_like` 7, `.max().item()` on the CPU mirror 8, `fp8_mqa_logits` 26, `nan_to_num_` 9, DeepSelect 34, the `where`/add/copy epilogue 36.
- Fix: one Triton launch, `dsa_prefill_prepare` (`_triton/prefill_prepare.py`), with a token program per query (the canonical token-group quantizer, the scale folded into the weights in `combine_topk_weights`' FP32 order, 16-head indexers padded to 32 heads inside the kernel, candidate and `lens_out` counts) and a gather program per 32 key rows of the page-planar cache, in one grid. Bit-identical to the framework ops (tested for 16/32/64 heads, FP32 and BF16 weights with a strided row, zero/huge/tiny rows, random slots), so SM120 only uses it (other platforms' quantizers are not checked against it). Token and row counts are runtime arguments (`do_not_specialize`; one binary). DeepSelect takes an optional per-row `index_base` that it adds to every selected column (the `-1` padding stays), which removes the four-op epilogue; other callers pass none and read a zero buffer. `out.fill_(-1)` is gone where the selector writes every slot, and the TRT-LLM row starts are built only on that path.
- Result: host time per call 360 -> 118-124 us at 1-1024 tokens. Back-to-back eager calls, DeepGEMM leaf vs the Triton leaf (`scripts/sm120_dsa_vs_tokenspeed.py`, which now calls the shipped leaf): 1 token 122 vs 175 us (1.43x), 16: 1.41x, 64: 1.37x, 128: 124 vs 168 us (1.35x; was 0.52x), 256: 1.79x, 512: 6.0x, 1024: 21x, 128 tokens on a 2048 prefix 5.8x, 4096: 81x, 4096 on a 4096 prefix 65x, 2048 on a 14336 prefix 54x, 8192: 98x; top-k overlap with the Triton leaf 0.98-1.00. No shape rule.
- Tests: `test_dsa_prefill_prepare_sm120.py` (prepare bit-equal to the framework ops; the leaf's output equal to the framework-op path's for several requests at nonzero row starts; the leaf launches 4 kernels; no recompilation for new chunk sizes), `test_index_base_offsets_every_selected_column` (DeepSelect, shortcut and selecting rows).
- Not done: a persistent output workspace for the prepare kernel (five `torch.empty` calls, ~2-3 us each; reusing buffers across layers means stream-ordered aliasing), and a leaner DeepSelect host path (its argument checks and the CuTe call are ~34 us). `fp8_mqa_logits` (26 us) and DeepSelect are now the two biggest segments.

**2. `dsa_index_candidates` candidate top-k.** `deep_gemm_dsa_index_candidates` scored the owned pages with the paged MQA kernel and then ran `torch.topk(sorted=False)` over the whole compact row, padding included. Selector comparison on the logits a DCP rank produces (top-k 2048, graph, 1/4 of the pages owned; rows 1-2048, 4K-128K candidates per row): DeepSelect over each row's own `lengths` beats `torch.topk` everywhere (2-8x from 128 rows) and beats the TRT-LLM cluster runner (`cute_dsl_decode_topk`) from 128 rows or at 1/4 ownership. The runner is 1.1-1.7x faster only for full-length rows with up to 32 rows of 16K+ candidates, which a sharded rank never has, so there is no rule. The leaf now selects with DeepSelect when `deepselect_supports` (a short row yields all its candidates, then `-1`; scores masked to `-inf` inside the length may be picked, and the gather drops them as before) and keeps `torch.topk` otherwise. Whole leaf (`scripts/sm120_bench_index_candidates.py`, 1-2048 query rows x 16K-128K contexts x 2/4/8 ranks, 54 points): 1.25-4.7x faster than the `torch.topk` leaf (geomean about 2; 3.2x at 512 rows, 2048 pages, 8 ranks), 3-64x faster than the Triton leaf (geomean 18x). Test: `test_index_candidates_select_the_top_scores_of_a_shard` (same score multiset as `torch.topk` on the same logits for long, short, empty and missing-request rows; DeepSelect runs).

**3. Batched BF16 `einsum` (DeepGEMM).** `bhr,hdr->bhd` and `bhd,hdr->bhr` (deterministic mode; cuBLASLt otherwise) launched one BF16 GEMM per head. They now use the `GemmType::Batched` launch of 6a7d45e through the BF16 kernel (`sm120_bf16_bmm_1d1d`: A, B, D as the strided head views through 3D tensor maps, B MN-major for `bhd,hdr->bhr`; `sm120_bf16_bmm_supported` checks 16-byte-aligned strides and 8-byte D rows, anything else keeps the launch per head).
- Bitwise equality needed one more change. The first version differed from the launch per head in ~0.02% of the BF16 outputs (109 of 540672 at h=128, b=33; 83 of 32768 at h=8, b=4): a GEMM's bits depend on its layout (the warps split each K block) and on stream-K (partial tiles summed in another order), and the cost model picked a different layout and stream-K for a one-head GEMM than for the batch. Both routes now run the batched launch's data-parallel layout (`sm120_bf16_bmm_choice`, passed as a forced choice), so `DG_SM120_BATCHED_LAUNCH=0` (a GEMM per head) and the batched launch are bit-identical. The price: a few outputs move by one BF16 ulp against earlier builds, which chose per-head layouts.
- `bmk,bnk->mn` was left alone: its slices all accumulate into the one FP32 D in order, a K reduction rather than independent batches (a batched launch races on D, and one GEMM over the concatenated K sums in another order).
- Benchmark (`scripts/sm120_bench_bf16_einsum.py`: h x r x d = 8x4096x1024, 128x512x128, 16x576x128, 64x512x192; b = 1-8192; both expressions; 56 points; cold operands): batched / per-head GEMM calls (what it did before) geomean 0.21 (0.03-0.98: h=128 at b <= 32 is 14-17 us against 250-600 us); batched / cuBLASLt geomean 0.94 (0.72-1.24), 52 of 56 points within 3% of cuBLASLt or faster; the four others are three b=1 points (0.4-0.7 us apart) and h=128, b=8192 (1.07x).
- Tests: `test_bf16_einsum_batched_launch` (bit equality with the launch per head, with and without an accumulator, padded strides, against an FP32 reference). `test_einsum.py` passes with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh JIT cache (FP8 and BF16 batched-launch tests included).

**4. V4/V4.1 decode: read the packed rows.** There are two paths. `dsv4_decode` (V4 FP8 page-planar caches) has no runtime caller; the SM120 consumer is V4.1's portable `selected_attention` (rows `swa` FP8 g32 and `global` FP4 g16), which gathers each segment into BF16 with torch glue and calls the Triton dense kernel (`dsv4_prefill(solution="triton")`). New `dsv4/_triton/paged_mma.py`: `_dsv4_sparse_mma_kernel`'s loop with the tile built by dequantizing the packed rows in registers (FP8 UE8M0 g32 or g64 + the BF16 RoPE tail, FP4 E2M1 + E4M3 scale g16; page-planar, strided views via `_planar_offset`). An E4M3/E2M1 value times a power of two or an E4M3 scale has at most 4 significant bits, which BF16 holds exactly, so the result is bit-identical to the workspace route (same chunking by width, same merge), batch-invariant and graph-safe.
- `triton_dsv41_mma_selected_attention` (the V4.1 decode path on SM120): 128 window + 512 compressed rows, 16/64/128 heads x 1/4/16/32/64/128/256 tokens (`bench_dsv41_paged_sm120.py v41`): 1.66x geomean over 21 points, 1.01x worst (128 heads, 128 tokens: 351 vs 347 us), 3.67x best (16 heads, 256 tokens: 498 vs 136 us); 41 vs 24 us at one token (the glue and launches). It runs the portable route for anything it does not read (workspace prefill, a window off the 32-row tile, other layouts).
- `triton_dsv4_mma_decode` (`dsv4_decode`): above 16 heads the workspace route reaches the CuTe DSL dense kernel, which is 1.1-1.6x faster than the Triton one the direct reader extends, and wins (64 heads 0.69-1.11x, 128 heads 0.72-0.79x for the direct reader; 17 vs 22 us at one token). With 16 heads both routes run the same Triton kernel, so a rule by token count changes no bit: the direct reader wins from 32 tokens (1.23x at 32, 1.08x at 64, 1.79x at 128, 2.28x at 256) and loses at 1-4 tokens (0.75-0.79x). The leaf is registered for 16 heads from 32 tokens only.
- Tests (`test_dsv4_paged_mma_sm120.py`, 22; plus two entries in `test_sm120_excludes_sm100_only_leaves.py`): bit equality with the portable gather for 16/64/128 heads, 1-37 tokens, windows of 64-128 and 0-512 compressed rows with holes, out-of-cache slots and short lengths; the no-sink LSE; strided page views; batch invariance; no recompilation for new batches; V4 decode against the workspace route.
- Ruled out: the Triton direct reader for `dsv4_decode` above 16 heads or at 1-4 tokens (above). Open: a dequantizing tile in the CuTe DSL kernel (its tiles are BF16 `cp.async` copies), which would win above 16 heads too.

**5. `test_whisper_cross_kv_write` in the combined run.** Reproduced: the same file set as task 17 (`layers`, `models` and the MLA/V4/V4.1/DSA/dots3/K3/FlashMLA files) in the order "files first, then `layers models`" failed 1 of 2975 (`test_graphed_encoder_prefill_equals_eager_and_never_syncs[1]`; task 17 saw another test of the same file), and passes in the other order and alone. The traceback is `info.status != cudaStreamCaptureStatusInvalidated` in the caching allocator inside the encoder graph capture, preceded by seven `operation not permitted when stream is capturing (function reset)` warnings from `CUDAGraph.cpp`. `CUDAGraph.reset` runs in the destructor of a graph that died during the capture. The graphs of a model sit in a reference cycle (the model holds the runner holds the graphs holds the model), so earlier tests leave dead graphs for the cyclic GC, and the next capture dies when a collection starts at one of its allocations. It depends on the allocation count since the last collection, hence on the test order. Fix (`10be2bcb8`): `gc_paused_for_capture` collects the pending garbage and keeps automatic collection off while whisper's encoder graphs (and the shared multimodal encoder graphs, which have the same exposure) capture. `BreakableCapture` already did this by hand; `freeze_gc` in `forward_step` covers the decode captures. Tests: a dead graph's cycle is created during the capture with a collection due at every allocation (fails without the fix with `cudaErrorStreamCaptureInvalidated`, passes with it) and the context manager's behavior. The same dead-graph test fails on `/tmp/ts_base` (892e00ce2) with the same error, so the defect is older than this round and the baseline has it too (a plain combined run on the baseline hangs earlier in `models/test_gpt_oss.py`, which starts a server, so it cannot be used for the order test). After the fix the combined set passes in both orders, 2976 passed each ("files first" in 108 s, "dirs first" in 149 s).
**Verification (GPU 6 and 7).** TokenSpeed kernel suite (the task 8 command) 7769 passed, 2770 skipped, 0 failed in 30 minutes; after the two follow-ups the DeepSelect / DSA indexer / prefill-prepare / V4 reader / SM120-selection subset 487 passed. Runtime suite: cache 137, distributed 638, entrypoints 40, execution 195, layers 370, models 565, moe 72, sampling 144, diffusion 53, top 6782 passed with one failure, `test_deepseek_v41_dspark.py::test_window_attention_kernel_matches_reference`, which asserted the portable V4.1 leaf and now expects the SM120 direct reader (`5406a4c1b`, 27 passed); the combined set twice as above. DeepGEMM `test_einsum.py` with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh JIT cache: passes (every test, including the FP8 and BF16 batched-launch checks).

**Ruled out (don't retry without new evidence):** a shape rule for tiny DSA prefill chunks (the overhead is removable); the TRT-LLM runner for sharded candidates; a batched `bmk,bnk->mn`; letting the batched BF16 einsum pick its own layout while the per-head route picks another (0.02% of the outputs differ); the Triton direct reader for `dsv4_decode` above 16 heads or at 1-4 tokens.

**Still open:** a dequantizing tile in the CuTe DSL V4 kernel (the direct reader would then win above 16 heads); `fp8_mqa_logits` (26 us) and DeepSelect (34 us) host paths for 128-token DSA chunks (done in task 37, section 9.41: about 14.5 and 16.5 us); the sparse MX kernel and the other items of 9.21 and 9.22.

## 9.25 Task 20: number reconciliation and final verification of tasks 16-19 (2026-11-09)
DeepGEMM `fefe6a1` (docs on `2e77a46`; the kernels are those of `6b2df9e` and `4ec333b`), TokenSpeed `c6cbc9b4b` (pin at `fefe6a1`). GPUs 6 and 7 only, both at 0 MiB before and after.
**A. Numbers that disagreed after the task 19 split, and what settled them.**
- *FP4 vs the best FP8 kernel and vs the best Triton/FlashInfer block-FP8.* The old Current state said 1.6-2.0x and 1.1-10.2x, the condensed report 1.3-2.0x and 1.07-9.1x. The 2026-11-04 rows are pre-refit single runs (1.6-2.0x before the cost-model refit, 1.3-2.0x with `shared down` at 0.92-1.48x after it) and no table of the Triton/FlashInfer ratios was stored, so the range was remeasured with the shipped build: `sm120_bench_blockscale.py --tp 8 --baselines`, GPU 7, graph + cold weights, 8 linears x 9 M = 72 points. FP4 over the best FP8 kernel (faster of the FP32-scale and the two UE8M0 kernels): 1.36-2.27x outside `shared down` (geomean 1.82x), `shared down` 0.94-1.33x (a tie of 0.94x at M=32, 1.00x at M=8). FP4 over the best Triton/FlashInfer: 1.52-11.37x, geomean 3.1x (smallest `shared down` M=4096 / 128 / 2048 at 1.5-1.6x, largest `shared gate_up` M=32 / 64 at 11x). All files that quote the range now state these numbers (report, design 1 / 7.5 / 7.8, weekly report, Current state); the 2026-11-04 status-log rows stay as history.
- *MLA decode CuTe vs Triton.* 0.97-2.31x is the 128 points (`Q` = 1); 2.34x is the maximum of the separate `Q` = 4/8 block (1.13-2.34x, 1.84x geomean). The design per-op table had merged them; the TokenSpeed MLA README already had them apart.
- *Batched BF16 einsum.* The geomean time ratio against a launch per head is 0.21, i.e. 4.8x faster (1 / 0.21 = 4.76; the design table said 4.9x).
- *Current state.* Rows A-E of the section 1 table now state the SM120 status (native FP8 dense leaf, m-grouped kernels done but with no DeepEP consumer here, native V3.2 indexer, native V4.1 FP4/FP8 indexer, native mHC); the Open list drops the UE8M0 regression (fixed by the joint refit `eff4cea`), NVFP4 (done) and the MXFP4 `mm` leaf (exists); the Commits bullet says that tasks 1-15 are pushed.
**B. Verification** (TokenSpeed mirror synced from the Mac tree at `c6cbc9b4b`, DeepGEMM rebuilt from the synced tree with `sm120_remote.sh build`):
- TokenSpeed kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, GPU 7, cold DeepGEMM JIT cache): **7770 passed, 2770 skipped, 0 failed**, 37 subtests passed (11 min).
- TokenSpeed runtime suite, every directory of `test/runtime`, GPU 6, per-directory runner: cache 137 passed; diffusion 53; distributed 638 (106 skipped); entrypoints 40; execution 195 (7 skipped); layers 370 (51 skipped); models 565 (48 skipped); moe 72; sampling 144 (10 skipped); top-level files 6783 (286 skipped, 510 subtests). **8997 passed, 0 failed.**
- DeepGEMM at `fefe6a1`, GPU 7, fresh JIT cache, `DG_JIT_CHECK_NO_SPILLS=1`: `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout` (one skip: packed k-grouped scales, SM100 only), `test_hyperconnection`, `test_attention` exit 0; `sm120_mqa_check.py` 46 cases OK; `sm120_blockscale_check.py --mode all --grouped` 170 cases ALL OK; `sm120_paged_mqa_blockscale_check.py` 51 cases ALL OK; `test_lazy_init` on GPUs 6 and 7: 3 tests OK. The only error-like line in the logs is "Failed to load legacy DeepGEMM A100 Triton kernels: No module named 'triton'" (no triton in the DeepGEMM venv; environment only, as before).
- The pinned commit built the way `install.sh` builds it (`DG_FORCE_BUILD=1 pip wheel --no-build-isolation --no-deps` in the TokenSpeed venv, from a clone of `fefe6a1` with the same two submodule pins, nothing installed): wheel `deep_gemm-2.8.2+fefe6a1-cp312`; `test_deepgemm_sm120_gating.py` against the unpacked wheel: 11 passed.
- End to end, in-process `Engine`, greedy, 48 new tokens (every output correct):
  - Qwen3.6-35B-A3B (TP1, GPU 6): "The capital of France is" -> " Paris, a city renowned for its rich history, culture, and iconic landmarks. Situated in the north-central part of the country, along the Seine River, Paris has been the political, economic, and cultural hub of France since the Middle"; primes -> "\n\n<think>\n\n</think>\n\n2, 3, 5, 7, 11".
  - Qwen3.5-122B-A10B-NVFP4 (TP2, GPUs 6 and 7): " Paris.\nA. True\nB. False\n\n<think>\n\n</think>\n\nA. True\n\nParis is the capital and most populous city of France. ..."; primes -> "\n\n<think>\n\n</think>\n\n2, 3, 5, 7, 11". (The continuation differs from the task 15 run, which repeated " Paris.\n\nThe capital of France is Paris."; both start with the right answer and the primes are identical. The cause of the late-token difference was not investigated; the sampling is greedy, but the build, the cache and the attention kernels differ from the task 15 run.)
  - Qwen3.8-27B-NVFP4 (TP1, GPU 6): " Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome. ..."; primes -> "\n\n2, 3, 5, 7, 11".
  - gemma-4-31B-it with `<bos>` (TP1, GPU 6): " Paris.\n\nThe capital of France is Paris. ..." (greedy repetition); primes -> "\n2, 3, 5, 7, 11".
- Environment-only, unchanged: `test_locality_domain` (MLOPart), `test_mega_moe` (NVLink), the sanitizer's `test_legacy` entry (no triton in the venv). compute-sanitizer was not rerun (no kernel changed since task 15 other than those the task 16 and 18 checks above cover).
**C. Commits since DeepGEMM `0301a69` and TokenSpeed `8229f4181`** (all local on `nctu6`, not pushed).
- DeepGEMM: `6b2df9e` block-scaled paged MQA kernel; `a580d14` docs (task 16, 9.21); `354889a`, `abe48ee` docs (task 17, 9.22 and 9.23); `4ec333b` batched BF16 einsum; `721cc2c`, `5db97ac` benchmark scripts; `a0cfdfa` docs (task 18, 9.24); `85fc9cb` report condensed, long form in `deepgemm-sm120-design.md`; `2e77a46` weekly report; `fefe6a1` number reconciliation and Current state; then the task 20 docs commit.
- TokenSpeed: `76b2134a7`, `f01d594a5`, `6d4188f43` (V4.1 native indexer, task 16); `6e75d3e77`, `bf5bbc37f`, `ff76063b8`, `9bf202435`, `2e35979b7`, `58fe2ef22`, `143876e60`, `03587ef78` (task 17); `10be2bcb8`, `9850d5ddb`, `c13e40aa7`, `8bacf076b`, `4da186931`, `0375fbd7e`, `5406a4c1b` (task 18); `c6cbc9b4b` (pin at `fefe6a1`); then the pin at the task 20 docs commit.

## 9.26 Task 21: mega MoE, mega gate and mega mHC on SM120 (2026-11-09)

**What each SM100 kernel computes, and what it relies on.**

| Kernel | Computes | SM100 hardware it is built on | Truly needs missing hardware? |
|---|---|---|---|
| `fp8_fp4_mega_moe` (`sm100_fp8_fp4_mega_moe.cuh`, 1528 lines) | dispatch of (token, top-k slot) pairs to expert pools; L1 FP8 x MXFP4 grouped GEMM; SwiGLU + clamp + top-k weight + per-32 UE8M0 FP8 cast; L2 grouped GEMM; combine (FP32 sum of BF16 rows in top-k order). All in one persistent kernel | TMEM accumulators and `tcgen05.mma` (2-CTA `Allocator2Sm`, `UTCCP` scale copies), cluster pairs, NVLink symmetric memory (`sym_buffer.map`, `ld_acq_sys` / `st_rel_sys`, `red_add`) for cross-rank pulls, write-back and readiness flags, SM locality domains | No. Only the fusion into one kernel depends on TMEM, UMMA and symmetric memory. The algorithm is plain grouped GEMMs plus elementwise work. Cross-rank dispatch needs peer loads and stores, and GPUs 6 and 7 have them over PCIe (below) |
| `bf16_mega_gate` (`sm100_bf16_mega_gate.cuh` + `epilogue/sm100_mega_gate.cuh`) | BF16 gate GEMM `[T,H] x [E,H]^T` into FP32 scores, then sqrt-softplus, bias / image bias, top-k, normalization, routed scaling, shared-as-routed slots, physical-replica mapping, masks, fixed and random routing | UMMA into TMEM (1- or 2-CTA, `kNumMmaCtas`), `tmem_load` of the scores, cluster sync. The routing epilogue (`select_warp_topk`, `store_topk_token`, `store_random_topk_token`) is plain warp code on FP32 scores in global memory | No. On SM120: a BF16 `mma.sync` GEMM (the 1D1D kernel already does BF16 with FP32 output) plus a routing kernel that reuses the epilogue functions unchanged. Not done (below) |
| `mega_mhc` (`sm100_mega_mhc.cuh` + `epilogue/sm100_mega_mhc.cuh`) | shifted or normal mHC: TF32 pre-norm GEMM of the residual routes against `fn` (split-K), Sinkhorn mixes, the next post/comb mixes, RMSNorm with optional FP8 cast (GEMM, routed and Mega MoE shared scale layouts) | `SM100_MMA_TF32_TS` into TMEM, `tmem_load` / `tmem_store`, a persistent grid with split barriers | No. SM120 already has the TF32 `m16n8k8` `tf32_hc_prenorm_gemm` (section 9.9). The rest is elementwise and per-token work. Not done (below) |

None of the three needs hardware that SM120 lacks. TMEM, UMMA, 2-CTA MMA and cluster multicast are how SM100 implements them, not what they compute.

**P2P probe (GPUs 6 and 7, `scripts/sm120_p2p_probe.py`).** `torch.cuda.can_device_access_peer` is True in both directions. `nvidia-smi topo -m` shows `PIX` between GPU 6 and GPU 7 (one PCIe switch, no NVLink). A 256 MB device-to-device copy runs at 54.4 GB/s each way. A Triton kernel on GPU 6 that loads from a GPU 7 tensor returns the right values. So EP=2 over CUDA IPC with peer loads and stores is possible on this pair. It was not implemented (below).

**Mixed MMA probe (`scripts/sm120_mixed_mma_probe.cu`, `-gencode arch=compute_120a,code=sm_120a`).** `mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e2m1.f32.ue8m0` assembles and runs on sm_120a. It reads an E2M1 value from bits [5:2] of its 8-bit container. With 1.0 activations and a 1.0 scale, codes 1-7 shifted left by 2 give exactly 0.5, 1, 1.5, 2, 3, 4, 6. Shifted by 0, 1, 3 or 4 they come out scaled or wrapped. The generic `sm_120` target rejects the instruction (`.kind::mxf8f6f4` / `.block_scale` not supported), as for the other block-scaled MMAs.

**1. W4A8: FP8 x MXFP4 in the SM120 1D1D kernel (`kMixedFP4B`, one kernel path; DeepGEMM `5cf4862`).**
- B and its scales use the FP4 stage geometry (`BLOCK_K` = 64 B bytes = 128 elements, packed FP4 TMA). A uses the FP8 geometry for the same 128 elements (128 bytes, 128-byte swizzle). That is one 32-element UE8M0 word per row and stage for each operand.
- B fragments come from the FP4 `ldmatrix`: 8 nibbles per lane, K `8t .. 8t + 7` of a 64-element step. Each is spread into two container registers (`spread_e2m1_into_bytes`: one `prmt` and three logic ops, value in bits [5:2]).
- A is read to match. Lane `(g, t)` loads the 8 bytes at K `8t` and `32 + 8t` of rows `g` and `g + 8` with `ld.shared.v2`, which is two wavefronts, the minimum for 256 bytes. The two operands then share one K permutation inside each 32-element scale group, so each `m16n8k32` MMA covers exactly one scale group. MMA `h` of a step takes scale byte `h`.
- Only 64-byte B rows are offered (`block_k_candidates = {64}`), so the A fragments of both MMAs of a step fit. A 128-byte B row would need 128 A registers per stage.
- Host: `sm120_fp8_fp4_gemm_1d1d_launch` accepts FP8 A with FP4 B (K-major, 32-element scales). The heuristics have `is_mixed_fp4_b` / `get_block_k_a`, double the A stage, and use 128-byte A swizzling. `fp8_fp4_gemm_nt` and the m-grouped contiguous (incl. PSUM) and masked APIs all take W4A8 on SM120.
- Results are bitwise equal to the same GEMM with the FP4 weights converted to E4M3 (`convert_to_fp8`, every E2M1 value is exact in E4M3). Against an FP32 reference over the dequantized operands the error is 1.4e-6 (BF16 output rounding). Results are bitwise deterministic. `tests/test_fp8_fp4.py` now runs a `(32, 32, False, True)` config on SM120 (normal, m-grouped contiguous and masked, with the FP4/FP8 equivalence check), and FP4 B stays K-major (`requires_k_major_b`).
- Speed against the FP8 x FP8 path on converted weights (kineto, warm): 1.35-1.72x at decode and mid shapes (8 x 64 x 6144 x 7168: 233 vs 329 us; 16 x 32 x 4096 x 4096: 141 vs 242 us), 1.07-1.10x on small ones, and 0.84-0.96x on compute-bound 4 x 4096 x 4096 x 2048 (the 64-byte rows halve the K block).
- PSUM scheduler fix (all 1D1D PSUM kernels): the scheduler read `grouped_layout[0]` when it was built, before `cudaGridDependencySynchronize`, and its state was live across `setmaxnreg`. ptxas spilled 24 B on several W4A8 PSUM tiles (16x128, 32x64, 64x128 with runtime N/K). Each role now rebuilds a PSUM scheduler after its grid-dependency wait (`rebuild_psum_scheduler`), giving 0 B spill. This also removes a read-before-wait of a layout that the previous kernel may write under PDL.
- Tried and dropped: giving the producer 72-104 registers (it hid some spills but not the 256-thread ones, and the root cause was the pre-wait scheduler), and compiling N/K in (`"nk"`, which unrolls the producer loop and spills; the mega MoE GEMMs keep runtime N/K).

**2. `fp8_fp4_mega_moe` on SM120, one rank (`sm120_fp8_fp4_mega_moe.cuh` / `.hpp`).** It uses the same API and the same symmetric buffer slices (`x`, `x_sf`, `topk_idx`, `topk_weights`). `transform_weights_for_mega_moe` keeps the gate/up interleave and skips the UTCCP scale transpose on SM120, where TMA stages the scales. The steps, all with runtime token counts:
1. `sm120_mega_moe_dispatch` (one CTA): per-expert counts by shared-memory atomics, then a warp prefix sum of the counts aligned to `BLOCK_M`, which gives the PSUM layout. It writes each pair's row and adds the counts to `cumulative_local_expert_recv_stats`.
2. `sm120_mega_moe_gather` (a warp per pair): the FP8 row, its packed scales (MN-major) and the top-k weight.
3. L1 W4A8 grouped GEMM (PSUM), BF16 output.
4. `sm120_mega_moe_swiglu` (a warp per pair and 128 outputs): the SM100 epilogue operation by operation. The L1 accumulators are rounded to BF16, the BF16 clamp is applied (gate from above, up on both sides), then `__expf` + `rcp.approx` with fast math or `expf` + division without, round-to-nearest multiplies by `up` and the weight, the UE8M0 scale from the amax of 32 outputs (`get_ue8m0_sf_exp`), and the E4M3 cast.
5. L2 W4A8 grouped GEMM, BF16 output.
6. `sm120_mega_moe_combine`: the FP32 sum of the slot rows in top-k order, cast to BF16.

`BLOCK_M` is 16/32/64/128, picked from the expected pairs per routed expert, with 128-column tiles. It asserts one rank, no fused shared experts, MXFP4 weights and no localization.

**Tests.** `tests/test_mega_moe_sm120.py` replays the arithmetic in PyTorch (dequantized FP32 GEMMs, BF16 L1 rounding, BF16 clamp, SwiGLU, weight, `per_token_cast_to_fp8` with UE8M0 32, FP32 combine). It checks the expert stats, bitwise determinism over 4 runs, and masked slots (10%). At 1, 7, 64, 300 and 2000 tokens (H 1024, I 512, 6 of 32 experts) the difference is 0 at four sizes and 1.8e-10 at 300, i.e. a bitwise match except for one GEMM accumulation-order flip. `DG_JIT_CHECK_NO_SPILLS=1`, fresh cache. With a fresh cache after the final build, `test_fp8_fp4`, `test_bf16`, `test_layout`, `test_einsum`, `test_attention` and `sm120_blockscale_check.py --mode all --grouped` also pass. `tests/test_mega_moe.py` needs DeepEP and TileLang for its reference (not installed) and a multi-rank group, so it is not the SM120 check.

**Benchmark against what TokenSpeed runs for V4 / V4.1 MXFP4 experts on SM120.** `scripts/sm120_mega_moe_vs_marlin.py`: CUDA graph, 10 calls per replay, each call after a 512 MB read (4x the 128 MB L2) whose own time is subtracted, 3 repeats, median, GPU 7. Mega includes TokenSpeed's `stage_mxfp4_mega_moe_inputs` (BF16 to FP8 + staging). FlashInfer CUTLASS MXFP4 is SM90 only, so the SM120 leaves are Marlin W4A16 and Triton MXFP4. Results agree with Marlin to 1.1e-3 (FP8 activations vs BF16). Times in us:

| Tokens | V4 mega | V4 Marlin | V4 Triton | V4 Marlin / mega | V4.1 mega | V4.1 Marlin | V4.1 Marlin / mega |
|---|---|---|---|---|---|---|---|
| 1 | 93 | 77 | - | 0.82x | 119 | 98 | 0.82x |
| 4 | 231 | 229 | - | 0.99x | 340 | 329 | 0.97x |
| 16 | 708 | 699 | - | 0.99x | 1147 | 1129 | 0.98x |
| 64 | 1887 | 1842 | - | 0.98x | 3203 | 3145 | 0.98x |
| 256 | 2398 | 2534 | 22216 | 1.06x | 4967 | 5008 | 1.01x |
| 1024 | 2590 | 2811 | 22220 | 1.09x | 5183 | 5338 | 1.03x |
| 4096 | 3347 | 6104 | 43322 | 1.82x | 6151 | 9785 | 1.59x |
| 8192 | - | - | - | - | 10143 | 18934 | 1.87x |

- From 4 to 1024 tokens both kernels read every touched expert once and run at the DRAM roofline (V4 at 16 tokens: about 96 experts x 12.6 MB = 1.2 GB in 0.7 ms). That is a tie by construction.
- At 1 token the six extra launches cost about 9 us (dispatch 2.8, gather 2.1, SwiGLU 2.2, combine 2.1), plus staging. The GEMMs read 75 MB at about 1.4 TB/s.
- From 4096 tokens the FP8 tensor cores win over Marlin's BF16 MMA (1.6-1.9x). Triton MXFP4 is 9-13x slower than mega. At V4.1 it hit an illegal memory access in the first call (`triton_mxfp4_precomputed_moe_apply`; root cause and fix in part 4 below).
- Tile sweeps at 1-64 tokens (BLOCK_N 64/128/256, BLOCK_M 16, through the developer overrides `DG_SM120_MEGA_BLOCK_N` / `DG_SM120_MEGA_BLOCK_M`) and PDL on all stayed within 0.76-0.99x of Marlin. None closes the 1-token gap.

**TokenSpeed: not enabled (one-rank state; superseded by part 3 below).** `deep_gemm_mxfp4_mega_moe_apply` keeps `max_arch_version` 10.9, for three reasons:
1. The DeepGEMM SM120 version is single-rank.
2. TokenSpeed's MegaMoE path requires `ep_size > 1` and MoE TP 1 (`deepseek_v41_moe.py`).
3. The V4 / V4.1 routed experts (43 x 256 x 12.6 MB = 139 GB, 40 x 384 x 17.7 MB = 272 GB) do not fit one 96 GB GPU.

No SM120 serving configuration can reach a single-rank kernel. `test_sm120_excludes_sm100_only_leaves.py::test_mega_moe_is_not_offered_on_sm120` pins this (TokenSpeed `4ad42e528`, README `a2b701cc5`). The kernel would also lose at batch-1 decode. Kernel suite subset (SM120 leaves, API selection, SM120 gating, V4.1 registration, mHC arch window) 297 passed. `nvidia/ops/moe` + `ops/moe` 369 passed. V4 / V4.1 runtime (`test_deepseek_v4_config`, `test_deepseek_v4_mega_moe`, `test_deepseek_v41_model`, `test_deepseek_v41_eval`, GPU 6) 304 passed.

**Not done in this task, and why (no hardware blocker in any of them):**
- **EP=2 over CUDA IPC** (done in part 3 below, with peer stores rather than pulls). P2P works (above). It needs remote pulls of the routed rows in the gather, remote row stores in the L2 epilogue or a combine that reads peer buffers, and a system-scope readiness flag per launch, as the SM100 kernel does. At EP2 V4 would hold 70 GB of experts per GPU, which leaves little room next to the rest of the model. V4.1 would need 136 GB per GPU and does not fit.
- **Fused shared experts.** The host asserts `num_shared_experts == 0`. TokenSpeed runs V4's shared expert separately.
- **Mega gate.** A BF16 1D1D GEMM to FP32 scores plus a routing kernel reusing `epilogue/sm100_mega_gate.cuh`'s warp functions. TokenSpeed has no caller (V4 routes with its own router leaves), so there is nothing to enable.
- **Mega mHC.** `tf32_hc_prenorm_gemm` plus a fused Sinkhorn / RMSNorm / FP8 epilogue. TokenSpeed has no caller (it runs the `tf32_hc_prenorm_gemm` mHC leaves, 1.5-15x over Triton, section 9.9 and design 8).
- **The 1-token gap.** It needs fewer launches (dispatch and gather in one kernel, SwiGLU in the L1 epilogue) or a split-K grouped GEMM for decode.

**Closed (don't retry without new evidence):** the "mega kernels need NVLink symmetric memory or TMEM, SM120 can't run them" fact. The SM120 single-rank version runs with the SM100 numerics. Also closed: unpacked FP4 TMA (`16U4_ALIGN16B`) for the mixed MMA. It pads each 16 values to 16 bytes as 8 packed bytes plus 8 bytes of padding, not one value per byte, and the first attempt, which assumed one value per byte, hung until the timeout. Also closed: B-tile width or 16-row tiles as the fix for decode (0.76-0.84x at 1 token for every choice).

**3. EP across GPUs over CUDA IPC (task 21 continued; DeepGEMM `1a5c1cf`, script `28971f8`; TokenSpeed `695c15770`).**

*Buffer.* SM100 allocates the symmetric buffer with `torch.distributed._symmetric_memory` (`symm_mem.empty` + `rendezvous`, `SymmBuffer` in `deep_gemm/mega/__init__.py`) and passes every rank's base address (`buffer_ptrs`) to the kernel as `layout::SymBuffer`. That allocator works on PCIe peers: on GPUs 6 and 7 `rendezvous` maps both buffers and a kernel reads the peer's values (a two-process probe). So SM120 keeps the same allocation and API; only the layout behind `get_symm_buffer_size_for_mega_moe` differs on arch 12 (`SM120MegaMoESymLayout`): a 256-byte control block first (peer flags, the epoch, each peer's token count), the SM100 input views (`x`, `x_sf`, `topk_idx`, `topk_weights`), and with several ranks a receive area (every rank's top-k lists, and the FP8 rows and scales that reach this rank's experts, one slot per (source rank, token)) and a combine buffer (one BF16 row per (token, top-k slot)).

*Steps* (`sm120_fp8_fp4_mega_moe.cuh`; one rank keeps the old six steps):
1. `push`: a warp per (token, other rank) writes the token's top-k list to that rank, and the row and scales if one of its experts lives there.
2. `barrier`: lane 0 stores this rank's token count into every peer, `__threadfence_system`, then `st.release.sys` of the new epoch into each peer's `flags[rank]`; lanes `< R` spin on `ld.acquire.sys` of their own `flags[p]`. The epoch lives in the buffer, so graph replays keep counting. A rank that does not arrive makes the peers `trap` after `DG_SM120_COMM_TIMEOUT_S` (default 120 s) with a message naming the rank and epoch, instead of hanging.
3. `dispatch` (one CTA): counts the pairs of every rank that hit the local experts, lays out the BLOCK_M-aligned runs (PSUM layout), and writes `row_pair[row]` (source rank, pair) and `total_rows` to device memory.
4. `gather`, 5. L1 GEMM, 6. SwiGLU, 7. L2 GEMM: as for one rank, local. Gather, SwiGLU and return are grid-stride loops over `total_rows`, so their grids are sizing hints and routing changes speed, not results.
8. `return`: a warp per row writes the L2 row into the source rank's combine buffer at the pair's slot.
9. `barrier`, 10. `combine`: the FP32 sum of the slot rows in top-k order, as before, so a token's result does not depend on where its experts live.
Rows are sized for the worst routing of `num_max_tokens_per_rank` on every rank, so the intermediates are allocated (from the caching allocator) for the buffer's capacity, not the batch.

*Why stores, not loads.* The first version pulled rows with peer loads in the gather and read peer L2 rows in the combine. At V4 shapes and 16 tokens per rank the gather took 0.50-0.53 ms and the combine 0.96-0.97 ms (kineto), against 2.4 us for `push` + 3.3 us for the local gather and 13 us for `return` + 3.4 us for the combine with peer stores. A PCIe read is a round trip of microseconds and few are in flight per SM; posted writes stream. Closed: peer-load pulls for data.

*Ordering rule found on the way.* A hang showed up only on fresh JIT caches, in the test case where rank 0 has no tokens. Rank 0's eager checks are on empty tensors and never synchronize, so its last call's barriers were still queued on the default stream when the test started the graph warm-up on a new side stream that did not wait for it. Both barriers read the same epoch. Calls that share a buffer must be stream-ordered per rank (true of the SM100 buffer too). The test now orders the side stream (`wait_stream`). PDL is safe: `griddepcontrol.wait` waits for the previous grid to complete and flush, and the early trigger of the GEMM only starts the next kernel's preamble (the test passes with `--pdl 1`).

*Tests* (`tests/test_mega_moe_sm120_ep.py`, `torchrun --nproc_per_node 2` on GPUs 6 and 7, `DG_JIT_CHECK_NO_SPILLS=1`, run 3x on fresh caches, with and without PDL): per rank, the EP result is bitwise equal to the one-rank `fp8_fp4_mega_moe` over all experts for the same tokens, and matches the PyTorch replay to 0-2.2e-11 (H 1024 / I 512 / 32 experts) and 0-1.7e-9 (H 4096 / I 2048 / 64 experts); expert stats count every rank's pairs; masked slots; 4 calls bitwise identical; uneven tokens (7/3, 300/257, 1500/900) and ranks with none (0/5, 5/0); a captured graph replayed with new inputs equals eager. The one-rank `test_mega_moe_sm120` still passes with the same times (33.7 us at 300 tokens, 87.5 us at 2000).

*Benchmark* (`scripts/sm120_mega_moe_ep_vs_marlin.py`, GPUs 6 and 7 idle, CUDA graph, 10 calls per replay each after a 512 MB read whose own time is subtracted, slower rank, median of 3). What TokenSpeed runs today for V4/V4.1 MXFP4 experts at EP 2 on SM120 is Marlin W4A16 on each rank's experts plus a collective: with attention TP the tokens are replicated and the partial outputs are all-reduced (`moe_output_all_reduced` -> `all_reduce`); with attention DP the runtime all-gathers the tokens and reduce-scatters the outputs (`token_all_gather` / `token_reduce_scatter`). Mega needs attention DP and each rank brings its own T tokens, so the like-for-like baselines at the same served tokens (2T) are `marlin_ar2` (all-reduce over 2T replicated tokens) and `marlin_rsag` (gather 2T, reduce-scatter). Mega agrees with Marlin to 1.1e-3 (FP8 vs BF16 activations). Times in us; the ratio is baseline / mega:

| T / rank | V4 mega | V4 marlin_ar2 | V4 marlin_rsag | V4.1 mega | V4.1 marlin_ar2 | V4.1 marlin_rsag |
|---|---|---|---|---|---|---|
| 1 | 101 | 97 (0.95x) | 102 (1.00x) | 124 | 119 (0.96x) | 126 (1.02x) |
| 4 | 234 | 247 (1.05x) | 254 (1.09x) | 357 | 353 (0.99x) | 363 (1.02x) |
| 16 | 685 | 690 (1.01x) | 698 (1.02x) | 1068 | 1079 (1.01x) | 1087 (1.02x) |
| 64 | 1259 | 1275 (1.01x) | 1295 (1.03x) | 2304 | 2309 (1.00x) | 2335 (1.01x) |
| 256 | 1485 | 1506 (1.01x) | 1541 (1.04x) | 2851 | 2882 (1.01x) | 2922 (1.03x) |
| 1024 | 2381 | 2394 (1.01x) | 2484 (1.04x) | 3959 | 4047 (1.02x) | 4159 (1.05x) |
| 2048 | 3672 | 4777 (1.30x) | 4957 (1.35x) | 5473 | 7017 (1.28x) | 7207 (1.32x) |
| 4096 | 7239 | 9396 (1.30x) | 9646 (1.33x) | 9957 | 13465 (1.35x) | 13818 (1.39x) |

- Up to 1024 tokens per rank both read every touched local expert once and sit at the DRAM roofline, so it is a tie; the PCIe traffic of mega (rows out and back, about 2 x 6 x H bytes per token) hides under the weight reads. From 2048 the FP8 tensor cores win, as for one rank.
- At 1 token mega is 0.95-0.96x of the all-reduce baseline: 11 launches (push, two barriers, return on top of the six) against Marlin's few plus one small NCCL all-reduce.
- `marlin_ar` at the same T per rank (half the served tokens) is 0.55-0.93x of mega; it is not the same work and is listed in the script output only for reference.
- Not measured: attention. Mega requires attention DP (each rank holds all attention weights and runs its own tokens on all heads), while the all-reduce baseline is attention TP. Which is faster end to end depends on that part too.

*Memory.* At EP 2 a V4 rank holds 70 GB of routed experts plus the replicated rest (the checkpoint is 157 GB, 139 GB of it routed experts); V4.1 needs 136 GB of experts per rank and needs at least 4 ranks (the kernels take up to 8, run on 2 only).

*TokenSpeed.* `deep_gemm_sm120_mxfp4_mega_moe_apply` (`ops/moe/deep_gemm.py`) is registered when the platform is arch 12 and the DeepGEMM build reports `get_mega_moe_max_ranks(12) >= 2` (`deepgemm_sm120_mega_moe_max_ranks`, behind `deepgemm_supports_sm120`): arch window 12.0-12.9, trait `ep_size` {2}, `hidden_alignment` 512. It reuses the SM100 apply and warm-up, with the scale rows kept in order (no UTCCP transpose), and V4.1's 2304-wide experts are not padded to 2560 on SM120. MegaMoE stays opt-in (`--moe-backend mega_moe`); `auto` never selects a `mega_moe` leaf, and a per-batch switch would be a second execution path. The guarantee test became `test_sm100_mega_moe_leaf_is_not_offered_on_sm120` plus `test_sm120_mega_moe_leaf_follows_the_deepgemm_build` (SM120 only, EP 2 only, the widths the kernels take; V4.1 2304 selects). Kernel subset (SM120 gating, API selection, `nvidia/ops/moe`, `ops/moe`) 660 passed; V4/V4.1 runtime files and `test/runtime/moe` 304 passed.

*End to end: not run.* The plan was the real V4 checkpoint in-process on GPUs 6 and 7 (attention DP 2, EP 2), `mega_moe` against `marlin`, greedy. The first attempt failed at argument checking (`dist_init_addr` is required with DP), and by the second another user's vLLM held 92 GB on each of GPUs 6 and 7, so the 88 GB-per-rank model could not load. Still open: that run, to check that serving selects the leaf and the greedy tokens agree.

**4. The Triton MXFP4 illegal memory access at V4.1 shapes (TokenSpeed `8afd973bc`).** Reproduced alone at H 5120, I 2304, 384 experts with the first call. Under compute-sanitizer memcheck the run is clean when routing avoids the last experts and shows 2592 invalid 8-byte global reads at `_stage2_kernel` line 629 (the `w2` weight load) when a token routes to expert 383, each about 1.86 GB *before* the allocation. Root cause: the stage kernels address an expert as `w2_data + expert_id * hidden_size * packed_k` in 32-bit arithmetic. With `hidden_size * packed_k` = 5.9e6 bytes the product passes 2^31 from expert 364 and wraps negative. Stage 1 has the same product with `2 * intermediate * packed_k` = 11.8e6 bytes, which wraps from expert 182. It stays inside the allocation there, so it silently reads another expert's weights, a wrong result with no fault. V4 (256 experts, 8.4e6 bytes) stays below 2^31, so only V4.1 is hit. Fix: advance every expert base (weights, scales, biases, route ids) in int64, as the BF16 and FP8 Triton MoE kernels already do. The new `test_triton_mxfp4_large_expert_offsets.py` routes 1 and 20 tokens to experts 0, 181, 182, 363, 364 and 383 of a V4.1-sized layer and requires bitwise equality with the same call over a compact copy of those six experts. It failed with the illegal access before the fix and passes after; memcheck reports 0 invalid accesses. The leaf is registered for AMD too, so the fix applies there.

**Still open after part 3:** the 1-token decode gap at EP 2 (0.95x; a fused push + barrier + dispatch kernel, or SwiGLU in the L1 epilogue), the end-to-end run, fused shared experts, more than 2 ranks (V4.1 needs 4+ GPUs), and mega gate and mHC (no TokenSpeed caller: V4 routes with TokenSpeed's own router leaves and runs the `tf32_hc_prenorm_gemm` mHC leaves).


## 9.27 Task 22: real-weight checks (2026-11-09)

TokenSpeed `nctu6` from `ff9181366` (new: `e4fc05813` MLA RoPE fix, `6d6702bae` V4.1 expert subset, docs commit after them), DeepGEMM `ef3a8c1` (`build312`, no DeepGEMM change). RTX PRO 6000, GPUs 6 and 7, in-process `Engine`, greedy. Another user's vLLM held GPU 6 for most of the session, so the 2-GPU runs waited for gaps.

**Checkpoints on the box.** `/models` (`= /home/ubuntu/models`) holds DeepSeek-V4-Flash-Vision-Exp (`DeepseekV4ForCausalLM`, 157 GB, 43 layers, 256 experts) and DeepSeek-V4.1-Flash-1017 (476 GB, of which 203 GB are the two Engram tables and 274 GB routed experts). The HF cache holds only configs and tokenizers for DeepSeek-V3.1, R1, V4-Flash-NVFP4 and the others. There is no V3.2, GLM-5.x, dots3-note, Kimi or DeepSeek-V2-Lite checkpoint. For a real MLA check I downloaded Moonlight-16B-A3B-Instruct (`DeepseekV3ForCausalLM`, MLA with `kv_lora_rank` 512, no q LoRA, 64 experts, 32 GB BF16) to the remote `tokenspeed/tmp/models/` and deleted it after the run.

**1. V4 (157 GB) with MegaMoE: not loadable.** The harness fix was the missing `dist_init_addr` (DP attention needs it). The model then fails with `Model architectures ['DeepseekV4ForCausalLM'] are not supported`: upstream TokenSpeed deleted the V4 model and its attention backend in `0c1bc3ed3` (#2310) and kept only V4.1. Restoring a 3853-line model that was removed on purpose is out of scope, so the MegaMoE end-to-end check moved to V4.1.

**2. V4.1 with real weights on two GPUs.**
- *A layer subset does not work.* Loading the first N layers needs a loader change, and the FlatKV recipe also rejects it. It requires 40 layers, and its group packing and parent size are constants for the 40-layer layout (`deepseek_v41_geometry.py`: `group_packing`, `lcm_block_bytes`). With the first 22 layers (all four KV owners and the candidate source kept, recipe check relaxed as a probe) the SWA group pads 86% of its parent (limit 5%). Deriving the packing per geometry is a cache-subsystem change, so this direction is closed.
- *Expert subset instead* (TokenSpeed `6d6702bae`). `--hf-overrides '{"n_routed_experts": 128, "num_checkpoint_experts": 384}'` builds all 40 real layers with the checkpoint's first 128 routed experts per layer. Engram, attention, indexer, compressors, shared experts, embeddings, final norm and head are all real. The router keeps those experts' rows, and the other experts are not read: a shard-filter predicate drops them, and the iterator path skips them. Every other tensor keeps the strict coverage check. Tests: `test_expert_subset_loads_the_first_experts_of_a_wider_checkpoint` and `test_expert_subset_keeps_the_strict_checkpoint_check`. This is a different model than the checkpoint (top-6 of 128), but the same model on every path. 192 experts did not fit: 81 GB of weights per GPU, and the KV planner failed at util 0.86.
- *Paths.* (a) MegaMoE: `--moe-backend mega_moe`, attention DP2 + EP2. (b) Marlin W4A16: attention TP2 + MoE TP2. Marlin cannot run DP2/EP2 ("V4.1 requires attention TP == MoE TPxEP or fused EP"). (c) Forced portable at TP2: `--moe-backend triton` plus the `TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_{DSV41_INDEX_TOPK,DSV41_SELECTED_ATTENTION,DSV4_PREFILL,DSV4_DECODE}` overrides set to the `triton_*` leaves. Verbose selection: (a) `deep_gemm_sm120_mxfp4_mega_moe_apply`; (a) and (b) `deep_gemm_sm120_dsv41_index_topk`, `triton_dsv41_mma_selected_attention` and `triton_dsv4_mma_prefill`; (b) `marlin_mxfp4_precomputed_moe_apply`; (c) the Triton leaves and `triton_mxfp4_precomputed_moe_apply`. The CuTe DSL V4 prefill core was not picked at 32 or 64 heads in these runs. I did not investigate why.
- *Per-layer and logits comparison.* Scratch instrumentation wrapped `DeepseekV41Model._run_layer` and `decoder_forward` in the spawned schedulers and dumped each prefill's HC stream after every layer, plus the final normalized rows. Logits are the final rows times the checkpoint head in FP64. Mean relative L2 difference against the portable path at layers 0/4/8/12/16/20/24/28/32/36/39: Marlin 0.007/0.10/0.21/0.29/0.32/0.36/0.42/0.46/0.49/0.51/0.37; MegaMoE 0.008/0.11/0.21/0.30/0.33/0.37/0.43/0.47/0.49/0.52/0.35. Marlin against MegaMoE gives the same curve. Over the 48 decoder rows of 6 prompts, KL(portable ‖ path) is 0.56 for Marlin and 0.64 for MegaMoE, and top-1 agreement is 62.5% and 68.8%. Every pair diverges the same way. The greedy texts of all three paths degrade into repetition ("Paris. The capital of France is Paris.", "Au, which is the Latin word for gold", "William William"), and they agree on the first token in 2 to 3 of 6 prompts.
- *Float32 reference.* I ran the official `inference/model.py` `Block` with float stand-ins for its kernels: no activation quantization, dequantized FP8/FP4 weights, FP64 sparse attention and the same Sinkhorn. It runs on CPU for layers 0 and 2, with the same 128-expert router, and each path's own layer input. Single-layer relative L2 error at layer 0: portable 2.763e-2, Marlin 2.764e-2, MegaMoE 2.808e-2. At layer 2 (compressor and indexer): 3.887e-2, 3.896e-2, 3.701e-2. All paths sit at the same distance from the float model. That distance is the FP8 activation rounding the checkpoint is built for (the official kernels quantize activations as well). It is 4x the 0.7% gap between paths, so the late-layer divergence is amplification of rounding through 40 layers of a reduced-expert model, not a wrong kernel. In 13.3 a second, sharper random model was just as chaotic.
- *Speed.* Same request set: 2 prompts of 2004 tokens, 1 new token, 3 timed repeats; TPOT is (T(129) - T(1)) / 128 for 2 requests. Prefill graphs were off in all runs, because the dumps need eager prefill. Prefill: Marlin TP2 10166-10515 tok/s, MegaMoE DP2/EP2 3814-3953 (one repeat at 1601), portable TP2 3486-3545. TPOT: 23.5-23.6 ms, 34.2-34.5 ms and 31.3-33.5 ms. MegaMoE end to end is 2.7x slower in prefill and 1.46x in TPOT than Marlin. The leaf comparison of 9.26 (MegaMoE ties Marlin plus its collective at 1-1024 tokens per rank and wins at 2048+) says the time goes outside the MoE leaf: the DP attention path, the shared host Engram table read by both GPUs over PCIe, or scheduling. Not profiled.
- *MegaMoE memory.* With 192 experts at util 0.93, the first prefill ran out of memory inside `fp8_fp4_mega_moe`. The other rank then failed the Engram frontier assertion as a consequence. With several ranks, the SM120 host code sizes its row scratch for the worst routing of the largest batch: ranks x `num_max_tokens_per_rank` x top-k rows of L1 activations, L1 output, L2 activations and L2 output, about 2.6 GB at 8192 tokens per rank. It allocates this on every call through the caching allocator, outside the KV planner's budget. Util 0.86 leaves room for it. Fix: carve the scratch from the symmetric buffer, or let the leaf own it as a persistent workspace that the planner counts.

**3. Moonlight-16B-A3B (real MLA), one GPU.** The first start failed with `KeyError: 'factor'`. transformers 5 normalizes a config without `rope_scaling` into `{"rope_theta": ..., "rope_type": "default"}`, and the DeepSeek-V3 MLA layer turned every truthy scaling into deepseek-yarn. The fix (`e4fc05813`, `mla_rope_scaling`) maps the factor-less default dict to plain RoPE and leaves real scaling unchanged. Test: `test/runtime/models/test_deepseek_v3_rope_scaling.py`, which uses a real `DeepseekV3Config`. The draft model already had the same rule.
- *Selection.* SM120 decode `triton_mla_mma_decode_with_kvcache`; prefill `triton_mla_prefill` on both paths; MoE `flashinfer_cutlass_unquant_moe_apply` (BF16). The portable run overrides decode to `triton_mla_decode_with_kvcache`.
- *Outputs (SM120).* "Paris, which is located in the northern part of the country", "Au, which comes from the Latin word for gold, aurum ... atomic number 79", "2, 3, 5, 7, 11, 13, 17, 19", "A. William Shakespeare". The train prompt picks the wrong option (75; the right answer is 80) on both paths, which is the model, not the kernels.
- *Numerics.* Two SM120 runs give 48 of 48 identical greedy tokens on all 6 prompts. Against the portable decode, the paths agree for 19/34/48/24/17/40 tokens. At the split, both paths' chosen tokens have logprob -0.92 to -1.37 (near-ties), and the largest |dlogprob| on the shared prefix is 0.025-0.13 (one BF16 logit ulp).
- *Speed.* TPOT at short context: 5.0 vs 4.8 ms at batch 1 (0.96x), and a tie at batch 8. At about 3000 tokens of context x 8 requests: 6.6/9.0/8.8 vs 20.9/21.2/21.2 ms (2.3-3.2x). Prefill (4 x 2004 tokens) is 45.6-45.9K vs 46.8-47.8K tok/s with the same prefill kernel, a 3% engine-to-engine difference I did not investigate.

**Also seen.** When a scheduler fails at startup under attention DP, `DataParallelController` reports `AttributeError: ... no attribute 'multimodal_encoder_dtype'`, because the failed group thread never set the attribute, and the original error appears only in the scheduler log. This affects error reporting only. Not fixed.

**What full-model runs need (the remaining gap).**
- *V3.2 at TP8:* the checkpoint (DeepSeek-V3.2-Exp, about 690 GB FP8; 2.3 TB of disk is free), all 8 GPUs (GPUs 0-5 belong to other users' long-running servers), and room for KV: 690 GB of weights in 768 GB of HBM leaves about 9 GB per GPU before activations, so only short contexts fit. An NVFP4 build (about 400 GB) would leave about 45 GB. The all-reduce goes over PCIe (no NVLink).
- *V4.1 with all 384 experts:* 4 GPUs. That is 274 GB of experts plus about 8 GB of other weights per rank, with the 203 GB Engram tables on the host. Options are EP4 MegaMoE (runs up to 8 ranks, tested on 2) or TP4 Marlin. An older tree served the full model at TP4 on GPUs 0-3 on 2026-10-28 (`tokenspeed/tmp/dsv41-tp4-fix.log`, "ts serve ready").
- *V4:* a TokenSpeed V4 model class, which upstream removed.

**Closed directions.** Layer-subset V4.1 runs (FlatKV packing pinned to 40 layers); Marlin at DP2/EP2 (not a supported V4.1 layout); 192 experts on two GPUs (KV does not fit at util 0.86; MegaMoE scratch OOM at 0.93); greedy-text comparisons of the reduced-expert V4.1 model across paths (chaotic: compare single layers against the float reference instead).

**Still open.** MegaMoE scratch outside the KV planner; the MegaMoE DP2/EP2 serving gap against Marlin TP2 (profile it); the CuTe DSL V4 prefill core not being picked in these runs; the full-model runs above; the `DataParallelController` error masking.

## 9.28 Task 23: cluster tile sharing for the dense gate_up M=256 FP4 tie (2026-11-10)
**Goal.** Win dense gate_up M=256 (N=4608, K=7168, MXFP4) over FlashInfer's b12x by at least 1.03x. Section 9.15 left it a tie
bound by L2 -> SM tile traffic, with TMA multicast emulated (it traps), so this task tried the other way to read a tile once for
several CTAs: thread-block clusters with distributed shared memory (DSMEM), which launch and run on SM120 (13.1 used 2- and 4-CTA
clusters).
**Reproduction.** `scripts/sm120_fp4_vs_flashinfer.py --only "dense gate_up" --ms 256` (CUDA graph over 512 MB of weight copies, PDL
on both sides, interleaved best-of-5), GPU 7 idle, three processes: b12x 19.6 / 19.6 / 19.5 us, DeepGEMM 19.2 / 19.2 / 19.2 us (1.02x;
the 63-point sweep of 9.15 had it at 0.979x, both inside run noise). DeepGEMM layout: tile 64x128 (2 x 4 warps of 32x32, 8 math
warps), 4 x 36 = 144 CTAs (77% of 188 SMs, so the late PDL trigger), 28 K blocks of 128-byte rows (A 8 KB + B 16 KB + scales per
stage), 3 stages in 78.6 KB. 144 CTAs x 28 blocks x 24 KB = 99 MB of tile traffic, of which B is 66 MB (16.5 MB read by each of the 4
M blocks) and A 33 MB.
**(c) Raster / swizzle: does not apply.** The scheduler already maps `blockIdx` to `m = i % 4`, `n = i / 4` (groups of 8 or 16 along
the primary axis). With one tile per CTA all 144 tiles are resident at once, the unique data (17.4 MB) is L2-resident, and order
changes neither the L2 -> SM bytes nor the DRAM bytes. Raster only matters with more than one wave.
**DSMEM probe** (`scripts/sm120_cluster_probe.cu`, standalone, clocks warmed up, 2.37 GHz):
| measurement | result |
|---|---|
| push 4 KB, `cp.async.bulk.shared::cluster.shared::cta` + mbarrier | 5.0 GB/s per CTA; the issuing thread is blocked 1.5K cycles (650 ns) per 4 KB |
| push via `st.async` (v4) from a warp | 5.0 GB/s |
| push via plain `st.shared::cluster.v4` + one release arrive | 5.0-5.1 GB/s |
| 1, 2 and 3 warps pushing at once; 1 KB to 16 KB per push | the same 5 GB/s per CTA in total (the time grows with the bytes) |
| 144 CTAs (36 clusters of 4) pushing to a ring neighbour | 4.7 GB/s per CTA, 0.68 TB/s aggregate |
| pull: `ld.shared::cluster.v4` of a peer's 4 KB (8 loads in flight per lane) | 25-27 GB/s with one warp, 32.7 GB/s with 2 or 3 warps; 4.7 TB/s aggregate over 144 CTAs |
| dependent remote load | 225 cycles (95 ns) |
| remote mbarrier arrive + wait, ping-pong between two CTAs | 1050 cycles (440 ns) per round trip, about 220 ns one way |
| max active clusters (65.6 KB of smem) | 94 of 2, 46 of 4, 22 of 8 (so a 36-cluster grid is one wave) |
The kernel needs about 45 GB/s of B and A per CTA to hold its present speed (24.6 KB per 1300 cycles), so a push path at 5 GB/s
cannot carry even the half of B that cluster 2 would forward (8 KB per block, about 17 GB/s), and a pull path would run at about half
of its 33 GB/s cap with 16 KB of loads in flight, which the 40-register TMA warps cannot hold.
**Kernel variants** (`sm120_fp8_fp4_gemm_1d1d_impl`, `kClusterSize` in {1, 2, 4}, host rule and `DG_SM120_CLUSTER` override; the CTAs of
a cluster take consecutive M blocks of one N block, which the scheduler already makes consecutive `blockIdx`; each loads a
`BLOCK_N / kClusterSize`-row slice of B with its own TMA box and the peers supply the rest; the empty barriers become cluster-wide,
every math warp arrives on every CTA's barrier with `mbarrier.arrive.release.cluster`, and the producer waits with acquire
semantics at cluster scope; barrier init and exit are cluster-synchronized):
- *Push.* A forwarder warp waits for its slice to land and copies it into each peer's stage with a bulk copy that completes a
  transaction on the peer's full barrier.
- *Pull.* A notifier warp arrives on the peers' `peer_ready` barrier once its slice has landed; two puller warps then copy the peers'
  slices with `ld.shared::cluster` into the same offset of their own stage and arrive on the full barrier (count 1 + 2).
Both are bitwise equal to the unshared kernel and have no spills (`DG_JIT_CHECK_NO_SPILLS=1`).
**Timing** (`tmp/cl.py` style harness: m=256, graph over 512 MB of weights, PDL, interleaved best of 5, GPU 7; unshared = 20.0 us):
| shape (m=256) | unshared | push, cluster 2 / 4 | pull, cluster 2 / 4 |
|---|---|---|---|
| N=4608, K=7168 (the tie) | 19.97 | 63.7 / 146.4 | 45.5 / 92.4 |
| N=1152, K=7168 | 10.22 | 51.4 / 108.7 | |
| N=4608, K=1536 | 6.42 | 16.6 / 33.2 | |
| N=256, K=7168 | 4.61 | 20.2 / 29.1 | |
The loss is a fixed cost per K block (push 1.6 us at cluster 2 and 4.5 at cluster 4; pull 0.9 and 2.6 us), the same with 36 CTAs
(N=1152) as with 144, so it is latency in the pipeline, not bandwidth contention.
**Ceiling.** With the forwarding removed (each CTA loads only its slice and uses the stale rest: wrong results, but the same
cluster launch and traffic) the kernel takes 17.62 us at cluster 2 and 17.52 us at cluster 4 against 19.99 us: sharing B cannot give
more than 1.13-1.14x, and cluster 4 is no better than 2 (A traffic and latency then limit). Reaching 1.03x over b12x would need
nearly all of that ceiling, i.e. a sharing path that adds almost no latency to the stage chain.
**Why it cannot pay.** The mainloop is a latency-bandwidth product: 3 stages of 24.6 KB (73.8 KB) in flight over a 1.65 us chain per
stage is the 45 GB/s per CTA above. Sharing removes L2 traffic (the ceiling run shows the chain shrinking by 12%) but adds hops to
that chain: the slice's TMA must land, a notify crosses the cluster (220 ns), the peer pulls (95 ns latency per round plus the
copy) and re-signals, before the math warps may start. At 3 stages every extra 0.1 us of chain costs 0.03 us per K block; the
measured per-block cost implies a chain longer by about 2.7 us (pull) to 4.7 us (push) at cluster 2, which is 0.9-1.6 us per block. More stages are impossible (99 KB of smem), and
the copy is a 5 GB/s (push) or 33 GB/s (pull) pipe, not the SM-to-SM network of a data-center part.
**Not shipped.** The kernel, host code and heuristic are as in `c0d46c6` (no `kClusterSize`), so the 63-point sweep, the NVFP4 sweep,
the determinism and `tests/test_fp8_fp4.py` results of 9.15 and 9.25 stand. The pull variant is kept as
`scripts/sm120_cluster_tile_sharing.patch` (applies to `c0d46c6`; `DG_SM120_CLUSTER=2` or `4` selects it) together with the probe.
**Closed directions (task 23).** B-slice sharing across 2- or 4-CTA clusters over DSMEM, by push (bulk copy, `st.async`, plain
stores) or pull (`ld.shared::cluster`); a raster order for L2 reuse in the single-wave case. Don't retry without a change in the
hardware's DSMEM write rate (about 5 GB/s per CTA) and remote-hop latency (95-220 ns), or more than about 8 stages in flight. TMA
multicast (software-emulated, traps) stays closed from 9.15. **Still open:** nothing new for this shape; it remains a 1.00-1.02x tie.

## 9.29 Task 24: the V4.1 serving gap, MegaMoE scratch, DP error masking, CuTe V4 prefill selection (2026-11-10)

TokenSpeed `nctu6` from `fe98e8989`, DeepGEMM from `019a54a`. RTX PRO 6000, GPUs 6 and 7, in-process `Engine`, real V4.1
(`/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, all 40 layers, first 128 of 384 experts via
`--hf-overrides '{"n_routed_experts": 128, "num_checkpoint_experts": 384}'`), `--engram-host-table`, util 0.85, 2 prompts of
2004 random tokens, prefill graphs off, decode graphs on. Harness: `tokenspeed/tmp/v41_bench.py` (deleted after the run): prefill
tok/s from 1-token requests, TPOT = (T(129) - T(1)) / 128, then a torch-profiler capture (CPU + CUDA, graph replays traced) of
one prefill and one 33-token request. Per-step decode numbers below come from the last 40% of the decode trace, normalized by
the 80 `_mhc_pre_layer_norm_hc4` launches of a step.

**Baseline in this harness.** Marlin TP2: prefill 10.2-10.7K tok/s, TPOT 23.0-25.1 ms (task 22: 10.2-10.5K, 23.5 ms). MegaMoE
DP2/EP2: 11.8-13.3K tok/s, TPOT 48.4-50.3 ms. Task 22 measured 3.8-3.95K tok/s and 34.4 ms for MegaMoE; I did not reproduce
that prefill number (its harness is gone; this one uses random token prompts and the host Engram table, with GPU-sharded tables
the 45.8 GiB per-rank Engram allocation does not fit at TP2). The decode gap is the same sign and larger here.

**Where the time went (decode, per step).** Marlin TP2 23.6 ms: dense projections 21.5 ms, Marlin MoE 3.4, attention 1.7,
mHC/norms 1.5, all-reduce 0.26. MegaMoE DP2 49.4 ms: dense 50.2 ms (two streams), MoE 6.9, attention 2.0. The dense time is
`_w8a8_block_fp8_matmul`: the V4.1 projections (`wq_a`, `wq_b`, `wkv`, `wo_b`, the shared expert, Engram's projection) are
E4M3 with UE8M0 scales per 32 K on both sides, and the only `mm` leaf on SM120 for 1x32 UE8M0 was the Triton block-FP8
kernel (FlashInfer's MXFP8 GEMM is SM100 only). At DP2 each GPU holds the whole dense model, so the same kernel reads twice
the bytes of TP2 and dominates more. Host gaps were 0.1-0.2 ms per step on both paths, and the Engram host-row gathers 0.01-0.08
ms per step (overlapped, not a factor). Attention per step is the same on both paths (1.6-1.7 ms).

**Fix 1: `deepgemm_sm120_mm_mxfp8` (TokenSpeed `0b638da63`).** The SM120 block-scaled 1D1D kernel already runs FP8 with
32-element UE8M0 scales (`recipe=(1, 1, 32)`, packed MN-major int32 scales). New `gemm/mm` leaf for the `mxfp8` signature with
uint8 1x32 scales on both operands, `PERFORMANT` -> `SPECIALIZED`, N % 64, K % 128. Scales are packed by a one-launch Triton
kernel (`_triton/ue8m0_pack.py`, same values as `pack_ue8m0_scales`). `prepare_mxfp8_weight_scales` returns the packed int32
weight scales on SM120 (it returned None there before), and `mm(prepacked_scales=True)` with int32 scales quantizes 16-bit
activations straight into the packed layout (`quantize_fp8_group32_mn_major_scales`, same codes and bytes as
`triton_quantize_fp8_group32_ue8m0`): one quantizer launch and one GEMM per projection. Cold-weight CUDA-graph benchmark on GPU 7
(six V4.1 projections, M = 1-2048): 2.2-11.7x over the Triton leaf (`wq_b` M=1: 415 -> 35.6 us; `wo_b` M=2048: 1345 -> 260 us;
smallest `shared_w2` M=256: 2.2x). Error against the dequantized float reference within one BF16 rounding.

**Batch invariance (DeepGEMM `26444a3`).** The first version was not batch invariant: at `wq_a` (N=1280) row 3 changed by one
BF16 ulp between M=4 and M=1024. Two choices follow the batch: stream-K (few-tile shapes split K across CTAs) and the BLOCK_N=32
tile, whose K-warp split sums two partial accumulators. With `deep_gemm.use_deterministic_algorithms(True)` the SM120 heuristic
now skips both (new `get_deterministic_algorithms()` query); every output is then one CTA's in-order K loop whatever the tile.
The TokenSpeed leaf runs in that mode (saved and restored around the call), declares `batch_invariant`, and registers only with
builds that have the query. Cost: 8-20% at M <= 64 against the non-deterministic kernel (`wq_a` 8.8 -> 10.7 us; stream-K was not
faster under graphs with cold weights on these shapes anyway). Tests: DeepGEMM `test_deterministic_is_batch_invariant_sm120`
(`tests/test_fp8_fp4.py`, `DG_JIT_CHECK_NO_SPILLS=1`; the whole file passes); TokenSpeed `test_deepgemm_sm120_mxfp8.py`
(packer = torch packer, quantizer = quantize + pack, no recompiles across batch sizes, leaf = dequantized reference, rows equal
across M = 1-1024, prepacked = leaf bit for bit, unaligned weights keep canonical scales).

**Fix 2: MegaMoE scratch (DeepGEMM `fefd9de`).** With several ranks `sm120_fp8_fp4_mega_moe` sized its rows for the worst routing
of `num_max_tokens_per_rank` on every rank and allocated them with `torch::empty` on every call (about 2.6 GB at 8192 tokens per
rank, V4.1 shapes), through the caching allocator and after the KV planner had measured. The rows now live in one byte buffer per
device (`SM120MegaMoEScratch`): the first call allocates it, later calls carve the same views, a larger call grows it (the old
buffer stays alive for graphs captured earlier; growing during a capture is an assert). TokenSpeed's warmup runs the largest
token count before the KV planner profiles memory, so the planner counts it. Calls on one device are ordered on one stream (the
barrier epoch already requires this), so consecutive calls may share the rows. `tests/test_mega_moe_sm120.py` checks that a
repeated call reserves no new memory; `test_mega_moe_sm120_ep.py` (2 ranks, GPUs 6/7) stays bitwise equal to one rank. Serving:
util 0.93 with 128 experts runs (prefill 18.1-18.2K tok/s); 192 experts at util 0.93 now stop at startup with "the KV cache does
not fit ... the memory profile left -5.82 GiB" instead of running out of memory in the first prefill.

**Fix 3: DP startup error masking (TokenSpeed `0199b9c97`).** `DataParallelController.launch_dp_schedulers` ran each DP group in
a thread; an exception there ended the thread silently, the scheduler-info fields were never set, and the caller failed with
`AttributeError: ... multimodal_encoder_dtype`. The launcher now collects launch errors and re-raises the first after joining,
and the fields are declared in `__init__`. In the 192-expert run the controller reports the scheduler's real exit. Test:
`test/runtime/test_data_parallel_controller_startup.py` (a raising group surfaces its error; a clean launch returns the port
arguments).

**Fix 4: CuTe V4 prefill not selected (TokenSpeed `70a02f53d`).** Not a trait or layout problem: V4.1 prefill hands the
selected-attention leaf a gathered BF16 workspace (`prefill_kv`), and the SM120 leaf `triton_dsv41_mma_selected_attention`
delegated that case to the portable leaf, which calls `dsv4_prefill(..., solution="triton")`. The pin kept every SM120 prefill on
`triton_dsv4_mma_prefill`. The SM120 leaf now calls `dsv4_prefill` with an open solution (shared helper
`_prefill_workspace_attention`); the portable leaf keeps its pin. Test: `test_prefill_workspace_runs_the_registry_choice` (16 heads
-> Triton core, 32 and 64 heads -> CuTe DSL core, bitwise equal to a direct call, with and without LSE). End to end it adds about
7% prefill on MegaMoE (17.0 -> 18.2K tok/s at 64 heads) and is within noise on Marlin (32 heads).

**Result.** Same harness, all fixes:

| Path | Prefill tok/s | TPOT ms |
|---|---|---|
| Marlin TP2, before | 10.2-10.7K | 23.0-25.1 |
| Marlin TP2, after | 14.5-14.8K | 10.4-10.7 |
| MegaMoE DP2/EP2, before | 11.8-13.3K | 48.4-50.3 |
| MegaMoE DP2/EP2, after (util 0.85 / 0.93) | 18.1-18.6K | 13.8-14.0 (one outlier run 7.6-19.7) |

MegaMoE now beats Marlin in prefill by 1.25x. TPOT stays 1.31x behind. Per decode step (13.99 vs 10.69 ms): dense projections
8.99 vs 6.38 ms (DeepGEMM GEMMs 5.36 vs 4.08, the BF16 cutlass projection 1.95 vs 1.08, the LM head 0.95 vs 0.45): DP attention
reads every dense and attention weight on each GPU while TP2 reads half, and a 1-2 token decode is weight-bandwidth bound, so
this +2.6 ms is structural to DP attention at this batch. MoE 5.47 vs 3.33 ms: the two grouped W4A8 GEMMs take 3.14 ms against
Marlin's 3.16 ms, and the other 2.3 ms are the MegaMoE glue (2 barriers 0.8-1.05, return 0.35, SwiGLU 0.3, gather 0.22,
dispatch 0.2, push 0.18, combine and staging) against Marlin's SiLU 0.4 and fused all-reduce 0.24. That glue is the fixable part
(fuse push + barrier + dispatch, SwiGLU into the L1 epilogue, return into the L2 epilogue; it is the "1-token gap" of 9.26). Even
with zero glue, TPOT would be about 11.8 ms, still behind TP2, so the target "MegaMoE at least matches Marlin in TPOT" cannot be
met at 2 requests on 2 GPUs with attention DP. It flips with batch: the MoE GEMMs win from 2048 tokens per rank (9.26), and
attention DP halves per-GPU attention work at large decode batches.

**Item E (attention headroom) was not reached in A-D; it is section 9.30.** Still open as listed in 9.22/9.23: BF16
DSA/V4 CuTe cores at 42-48% of the floor (register-resident query with a key split, or a smaller ring), V4 at 16 heads (8-warp CTA
or 2 tokens per CTA), decode of one DSA request bound by its 8 chunks. Task 23's DSMEM numbers (5 GB/s push, 33 GB/s pull per CTA)
rule out 2-CTA sharing of gathered rows for attention as well: a 64-row BF16 tile is 64 KB, which would take about 2 us to pull.

**Verification.** DeepGEMM: build OK; `tests/test_fp8_fp4.py` (whole file, `DG_JIT_CHECK_NO_SPILLS=1`), `tests/test_mega_moe_sm120.py`,
`tests/test_mega_moe_sm120_ep.py` (2 ranks, fresh cache) pass. TokenSpeed: kernel subsets (gemm, quantization, MXFP8, V4/V4.1
attention, selection, recompile, SM120 leaf lists) pass; V4.1 runtime files + `runtime/moe` 603 passed; DP controller tests 48
passed; the full TokenSpeed kernel suite (selection files, `nvidia`, `ops`) 7904 passed, 2763 skipped, 0 failed (GPU 7).

**Closed directions (task 24).** Looking for the V4.1 serving gap in Engram host reads (0.01-0.08 ms per step, overlapped), host
gaps (0.1-0.2 ms per step) or DP-path CUDA graphs (decode graphs are captured on both paths); stream-K for the deterministic
MXFP8 leaf (it changes bits with the batch). **Still open:** MegaMoE glue at decode (2.3 ms per step at 2 requests), DP
attention's structural dense-weight cost at small batch (+2.6 ms per step), item E.

## 9.30 Task 24 item E: attention headroom and MegaMoE glue (2026-11-10)
TokenSpeed `nctu6` from `9a20d3f57`, DeepGEMM `348c0d6` (`build312`, unchanged). RTX PRO 6000, benchmarks and kernel tests on GPU 7,
runtime tests on GPU 6, CUDA-graph microseconds (best of 5). Scratch scripts were in `tokenspeed/tmp/e/` (deleted). One TokenSpeed
commit, `3de153200` (launch planner). `ncu` is still blocked: the evidence is ablation, as in 9.22.
**E1: what bounds the BF16 DSA/V4 CuTe core.** Task 17 blamed the channel-split partial-score exchange (store, barrier, 4-way sum) for
the 42-48% of the floor. Ablating pieces of `_SparseMma` (wrong results, timing only), 128 heads, 2048-token prefill, BF16, 6.12 ms:
| removed | time | note |
|---|---|---|
| the whole exchange (store, barrier, sum) | 5.11 ms | the ceiling of any better score layout is 1.19x |
| score MMAs + their `ldmatrix` | 4.86 | |
| PV MMAs | 5.28 | |
| every MMA | 4.88 | the kernel is not tensor-bound |
| the gathers (zero-fill, no L2 read) | 6.02 | no change: the gather is hidden |
| gathers + exchange + every MMA | 2.58 | what is left: softmax, barriers, fold, per-CTA prologue |
The same on decode 128 tokens and 64 heads give the same shape. With every MMA, exchange and MMA ablated the gather alone sets 4.81 ms
(`cp.async`, 19.3 GB of rows read through L2 at 4.0 TB/s; the real kernel is at 3.15 TB/s). So the BF16 core is limited by two things
that do not overlap well in a 99 KiB CTA with two warps per scheduler: the per-tile chain (wait, scores, exchange barrier, softmax,
PV) and the L2 -> SM gather. A key split for BF16 (query in registers per warp, no exchange) can win at most the 17% of the exchange
and needs the query (36 KiB for 32 heads) next to two 37 KiB stages: it does not fit. Deeper rings do not fit either (a third stage is
37 KiB more).
Tried and lost:
- Bulk gathers: one `cp.async.bulk` (+ mbarrier) per 1152-byte row issued by the row's owner thread, absent rows zero-filled by the owner
  (V of a skipped row has to be finite). Alone it gathers 1.27x faster (3.77 vs 4.81 ms with the MMAs ablated), but the real kernel is
  1.12x slower (6.85 vs 6.13 ms; 8-warp 16-head V4 0.70-0.94x of Triton against 0.76-1.10x with `cp.async`), with or without
  `fence.proxy.async`.
- One-stage ring with two 8-warp CTAs per SM (16 heads, `min_blocks_per_mp=2`): 0.55-0.82x of the Triton leaf.
**E2: V4 at 16 heads.** The core gained a `cwarps` parameter (4 or 8 warps per head group; 8 warps halve the accumulator to 32 registers
and put two warps on each scheduler). 8 warps are bitwise-consistent with themselves and correct (max error 2e-3 against the Triton
leaf, as the 4-warp CTA), but their 8-way score sum is not the 4-warp sum, so they could only serve 16-head models. Against
`triton_dsv4_mma_prefill` (16 heads, sink, 640 rows; 4-warp / 8-warp CTA): 1 token 1.00x / 1.08x, 4 tokens 0.97 / 1.09, 16 tokens
1.03 / 1.10, 64 tokens 0.83 / 0.93, 128 tokens 0.77 / 0.92, 512 tokens 0.90 / 1.02, 2048 tokens 0.77 / 0.86, window-only 2048 tokens
0.64 / 0.76. The 8-warp CTA is a clear step up from the 4-warp one and still loses where the time is (large batches), and the leaf can
only follow the head count, so 16 heads stay on Triton. Two tokens per CTA was not built: the ring would need two gathered tiles.
**E3: one-request DSA decode (20.3 us BF16, 13.1 E4M3, 8 chunks).** The attention kernel is 19.2 us and the combine 2.0 us (profiler);
with every MMA, exchange and gather ablated 10.6 us remain, so per-CTA latency (Q loads, 8 tile chains, epilogue) is the floor of a
chunk. Fusing the combine into the last CTA would save at most the 2 us. The lever is the chunk count, which batch invariance ties to
the width: 16 chunks (a CuTe-only constant, the Triton core keeps 8) with the same planner give one-request decode 1.30-1.46x faster
(BF16 20.3 -> 14.2 us at 16-32 heads, 20.8 -> 16.3 at 128; E4M3 13.1 -> 9.8 / 13.3 -> 10.7) and 1.2-1.4x at 2-8 tokens, 0.99x
unsplit (128-2048 tokens, the extra fold), but 0.8-0.9x at 4-64 tokens (every chunk state is stored: 16 states of 128 heads x 512 x 4 B
per token) and a split cannot pre-fold a chunk run in a register tree without changing the fold order. Because the count cannot follow
the batch, 16 chunks are not selected.
**E3 spin-off: the launch planner (shipped, TokenSpeed `3de153200`).** The sweep over (block_h, splits, chunks) showed the old rule
(target CTA counts of 2/3 and 2.75 SMs) often left a wave nearly empty: a CTA fills an SM, so 24-48 32-head programs with 8 splits are
192-384 CTAs, a second wave costs a whole chunk run. Measured, BF16 32-head programs x splits (time us): 32 programs 140 / 82 / 47 / 52
(1 / 2 / 4 / 8 splits, 32 / 64 / 128 / 256 CTAs), 24 programs 140 / 81 / 45 / 49, 20 programs 140 / 81 / 44 / 28.5, 48 programs 141 / 84
/ 90 / 75.6, 96 programs 143 / 169 / 138 / 125. The new `plan_launch` (`dsa/_cute_dsl/sparse_mma.py`) picks 1, 2, 4 or 8 CTAs per
token with the fewest `waves x chunks per CTA`, ties to fewer splits, accepts a split only against a 1.25x margin (16-head CTAs) or
1.5x (32-head CTAs) over unsplit (stored states are not in the model; 16-head BF16 at 512 tokens with 4 splits is 0.76x), and starts
32-head CTAs when they alone fill the SMs (the Triton rule's two-thirds point loses 1.07-1.14x at h32 t8, h64 t4, h128 t2). Over 140
decode shapes (1-192 tokens at 16/24/32/40/64/96/128 heads, both dtypes): geomean 1.05x BF16 / 1.04x E4M3, worst 0.97x (a different
split count with the same launch, noise), up to 1.63x (96-128 heads at 5-7 tokens, where 5-7 splits ran 2 chunks per CTA, and
1.2-1.4x at 96-192 CTAs, which the old rule left unsplit); 44 points of 128-2048 tokens: 1.02x, worst 0.96x / 0.99x. Rows are bitwise
equal across the old and new plans (0 differences in 368 points). The same wave model applied to the V4 core (4 chunks, 3 head counts x
14 token counts x 2 head tiles x 1/2/4 splits, widths 640 and 128) loses (0.99x at width 640, 0.85-0.98x at 128 depending on the margin):
the old V4 rule is within 1.01% of the best (tile, split) per point, so V4 is unchanged.
Tests (`test_dsa_mma_sm120.py`): the plan at the boundary cases, a sweep of 1-699 tokens x 5 head counts (the plan has no more chunk
times than unsplit), plus the existing split-count/head-tile bitwise tests and the recompile guards. Kernel set (DSA/V4/MLA files, kernel
selection, V4.1 registration): 578 passed, 275 skipped; `test_dsa_mma_sm120.py` + `test_dsv4_mma_sm120.py` + SM120 leaf list 193
passed; runtime DSA/V4/V4.1/dots3/GLM files (GPU 6): 1049 passed, 94 skipped.
**E4: MegaMoE decode glue.** A graph-replayed call at 2 tokens per rank (V4.1 shapes, 5120 / 2304 / 384 experts, EP 2, GPUs 6+7, cold
weights, PDL on) takes 195 us; the profiler timeline: input staging 2.2, push 2.1, barrier 3.0, dispatch 2.3, gather 4.0 (13.2 us before
the first GEMM), GEMM 1 108.6, SwiGLU (starts under GEMM 1's PDL tail, about 2 us of its own), GEMM 2 56.4, return 16.4 (4 after GEMM 2),
barrier 3.0, combine 3.1 (10.4 us after GEMM 2). The glue on the critical path is about 25 us per layer (13%); in the serving trace the
barriers cost 10-13 us each (0.8-1.05 ms per step), the difference being the skew between the two ranks, which a fusion does not remove.
Fusing push + barrier + dispatch + gather (one kernel after the stage) and return + barrier + combine would remove the launch gaps and the
repeated PDL waits: about 5 + 4 us per layer, 0.36 ms per step, 2.6% of the 13.9 ms TPOT (and SwiGLU in the L1 epilogue less than
2 us). That needs cross-CTA counters and a second barrier protocol per kernel with hang risks, for a gain below the TPOT run-to-run
noise (13.8-14.0 ms), so it was measured and not built; the batch-invariance and graph-safety constraints are unchanged. The microbench
(`scripts/sm120_mega_moe_ep_vs_marlin.py`, PDL on): mega 125 / 195 / 354 / 1067 us at 1 / 2 / 4 / 16 tokens per rank, Marlin all-gather +
reduce-scatter 0.99-1.04x of it. No serving rerun: nothing changed.
**Closed (don't retry without new evidence):** exchange-free BF16 scores as the fix for the 48% (ceiling 1.19x, no smem for it); bulk
gathers; 1-stage ring x 2 CTAs; 16 DSA chunks; 8-warp V4 at 16 heads; a wave planner for V4; fusing the MegaMoE glue for TPOT (bound
0.36 ms). **Open:** a CuTe-only chunk count that follows the width, if the store of chunk states could be made cheaper (bf16 states
change the fold bits); a persistent or stream-K DSA kernel that keeps a CTA's tile chain busy across tokens (the per-CTA prologue and the
2-warps-per-scheduler chain are the floor).

## 9.31 Task 25: the intermittent V4.1 eager failure, searched with sanitizers, poisoned memory, stream delays and stress (2026-11-10)
TokenSpeed `nctu6` at `4e6bf8277` (mirror `/tmp/ts_mac`, resynced), DeepGEMM `2716aae` (`build312` rebuilt from that tree). RTX PRO 6000, GPUs 6 and 7 only (GPUs 0-5 belong to other users). No code change: nothing reproduced, so there is nothing to fix and no test to add. Scratch scripts lived in `tokenspeed/tmp/t25/` (removed afterwards).
**What the failure can be.** The `eager` id builds the 40-layer model from the deterministic `_checkpoint` (no random or uninitialised weights), runs one 4-token prefill and one 1-token decode, and its only checks are the output shapes and `torch.isfinite(...).all()` on the prefill output (4 x 256) and the decode output (1 x 256). So a failure was a non-finite value (or an exception). The decode forward is the only part that overlaps work: `StreamFork` (`utils/cuda_stream.py`) puts the index/global-KV branch of every attention layer on `aux_stream` and the mHC mixes on `hc_stream` while the main stream runs the projections (the MoE fork is capture-only, so it is not active in eager).
**1. The original log.** Not recoverable. The per-directory logs of the failing run (`/tmp/rt2_top.txt`) were overwritten by later runs, `/tmp/rt_dirs2*.txt` keep only summaries, `/tmp/rt_s.txt` and `/tmp/rt_v.txt` (task 4, 2026-11-06) only list the id under "warnings", and section 9.20 recorded no failure text. Which of the two `isfinite` checks failed, and by how much, is unknown.
**2. compute-sanitizer** (one eager iteration per tool, warm JIT caches, `--print-limit 50`, each run twice: the default caching allocator, and `PYTORCH_NO_CUDA_MEMORY_CACHING=1` so every tensor is its own `cudaMalloc` and out-of-bounds and uninitialised reads are exact per tensor):

| tool | default allocator | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` |
|---|---|---|
| racecheck (`--racecheck-report all`) | 0 hazards | 0 hazards |
| initcheck | 0 errors | 0 errors |
| synccheck | 0 errors | 0 errors |
| memcheck | 40 "errors", all API probes | 40, the same |

The 40 memcheck lines are 20 `cuGetProcAddress_v2` calls (`cudaVersion 13030/13041 exceeds the driver version 13020`, reported twice each: a library asking the driver for a newer entry point than the 13.2 driver has, which the library handles). There is no kernel, memory or synchronisation error in any Triton, CuTe DSL, DeepGEMM, CUTLASS, cuBLAS or ATen kernel the test launches. racecheck only sees shared-memory hazards, so it cannot say anything about the cross-stream ordering below.
**3. Kernel audit.** The torch profiler lists about 100 distinct kernels in one eager run (prefill + decode): DeepGEMM `sm120_fp8_fp4_gemm_1d1d` (five N/K shapes), `sm120_tf32_hc_prenorm_gemm`, the Triton MXFP4 MoE (`_routing_kernel`, `_stage1/2_kernel`, `_silu_and_mul_kernel`) with `moeFinalizeKernel`, `_sqrt_softplus_topk_kernel`, `_dsv4_sparse_mma_kernel` + `_dsv4_sparse_mma_combine_kernel`, `_paged_mma_kernel`, `_swa_rope_insert`, `_rope_pad_query`, the mHC kernels, the compressor and Engram kernels, cuBLAS gemv / `gemmSN`, `torch.topk` `gatherTopK`, `_finish_topk_kernel`, and ATen elementwise kernels. Findings:
- Atomics: the only `tl.atomic_add` in the selected files are int32 histogram updates in the long-row `_block_keys_pass` / `_block_hist_pass` of `dsv41/triton.py` (commutative, so the result is order independent, and those kernels do not run in this test). There are no floating-point atomics.
- Split counts that follow the batch: the DSV4 sparse core folds fixed chunks by width alone (task 15 `1334fcd70`); the MoE routing is one program per expert; no kernel here picks a split from the token count.
- Uninitialised reads: every `torch.empty` the kernels use is fully written for the rows that are read (`_routing` writes `expert_route_ids` up to `expert_counts`, `route_output` is `torch.zeros`). The poison runs below confirm it without reading each kernel.
- DeepSelect: not launched (top-k is `torch.topk` + `_finish_topk_kernel` at this size), so its cluster and tie handling is not exercised.
- CUDA graph versus eager: not applicable to the failing id. The graph ids (`decode`, `prefill`, `split_prefill_graph`) pass in every file run below.
**4. Poisoned memory** (`poison_alloc.cu`, a `CUDAPluggableAllocator` that fills every torch allocation, Python and C++, with a pattern before use; unlike `torch.utils.deterministic.fill_uninitialized_memory` it also covers ATen-internal `empty`): NaN bytes (`0xFF`), a large finite pattern (`0x7B`) and a different byte per allocation, 20 eager iterations each: 0 failures, and the output hashes equal the unpoisoned ones (one hash for the prefill output and one for the decode output over every run below). Because the pluggable allocator frees with a device-wide synchronisation, it hides cross-stream reuse races, so the next item does not use it.
**5. Stream delays** (the real caching allocator; `torch.cuda._sleep` injected into `StreamFork` at the start and end of every aux branch and right after every fork on the main stream, which widens the window of any missing `record_stream` or event wait): aux 2e6 / 2e7 / 2e8 cycles, main 2e7 / 2e8, both 5e6, and two runs that also use the NaN and random poison: 30-60 iterations each, 390 in total, 0 failures and one output hash for each of the two outputs every time (`beab4c0ef31a` prefill, `68abc2c18604` decode). The delays do apply: 30 iterations took 18 s without a delay and 60 took 110 s with aux 2e7. So the fork, join and `record_stream` protocol of the decode layers holds under a 100 ms skew in both directions.
**6. Stress.** The eager id in one process and the whole file in fresh processes (152 passed, 4 skipped each):

| condition | eager iterations (1 process) | `test_deepseek_v41_model.py` runs |
|---|---|---|
| GPU 6 idle | 400, 0 failed | 20, 0 failed |
| GPU 7 with an 8192x8192 BF16 GEMM loop in another process | 400, 0 failed | 20, 0 failed |
| GPU 6 with the TokenSpeed kernel suite running beside it (the suite was between 0% and 60% done) | 150, 0 failed | 6, 0 failed |

Together with section 9.20 (150 eager iterations, 73 under load, 6 complete top-level runs, the pre-fix tree, the baseline) that is about 1400 eager executions in this task (plus the 500+ of 9.20) without a non-finite output; the outputs were hashed in the 390 delay and poison iterations and were bit-identical in all of them.
**7. A different flake met on the way.** In the first kernel-suite run of this session (started on GPU 6 while a stress process was starting) `nvidia/ops/attention/test_attention_prologue_nvidia.py::test_unnormed_solutions_agree_byte_for_byte[dtype0-True-neox-64]` failed once (the Triton prologue and the CUDA RoPE composite differed in bytes; the case is the first one of the file). It did not come back: 200 in-process replays of the case with a per-tensor diff report, 30 fresh pytest processes of the 96-case test beside the running suite, 40 fresh first-call replays started together with a V4.1 stress process on the same GPU, and the full suite (7905 passed, 2763 skipped, 0 failed). `nvidia-smi` reports 0 ECC errors on GPU 6 and the kernel log has no Xid after the earlier task 23 probe. That kernel is not on the V4.1 eager path (the V4.1 test runs `_rope_inplace_kernel`, `_swa_rope_insert` and `_rope_pad_query`), so it is recorded as a second unreproduced one-off, not as the cause of the first. Side note: after the topk JIT cache of FlashInfer (`cached_ops/topk`, arch `120f`) was invalidated, the first suite run spent about 18 minutes in `ninja`; it passes once built.
**Conclusion.** Not reproducible after a bounded search with every tool that can see this class of bug: racecheck, initcheck, synccheck and memcheck are clean (with and without the caching allocator), no read of unwritten memory changes any bit, no stream skew up to 100 ms changes any bit, and about 1400 eager runs under idle, GEMM-loaded and kernel-suite-loaded GPUs never produced a non-finite output. Not a defect that can be fixed from the evidence, so no code or test changed. If it comes back: keep the test output (`pytest -rfE --tb=long` for the top-level run; the assertion does not print values) and run `tmp/t25`-style `stress.py --hash --poison rand` against that tree; the two `isfinite` checks say whether the 4 x 256 prefill or the 1 x 256 decode output was non-finite. **Closed (don't retry without a failure log):** sanitizer sweeps of this id, poison allocators, aux/main stream skew, more stress loops on an idle or GEMM-loaded GPU.

## 9.32 Task 26: final verification of tasks 21-25 (2026-11-10)
TokenSpeed `nctu6` with the `DeepGEMM` pin bumped to `b6247c2` (`0d2f40c5b`, on top of `4e6bf8277`), DeepGEMM `b6247c2`. The remote DeepGEMM tree was synced from the local tree (tracked files, before the push of task 27) and `sm120_remote.sh build` printed "build OK" for both the py3.15 and py3.12 builds. The `/tmp/ts_mac` mirror was resynced from the tracked files of the local TokenSpeed tree. GPUs 6 and 7 only; all JIT caches were deleted first. No code change and no regression, so there is no new test.

| check | result |
|---|---|
| TokenSpeed kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, GPU 7) | 7898 passed, 2770 skipped, 37 subtests passed, 0 failed (two identical runs; `test_deepgemm_sm120_mxfp8.py` 117 passed, so the MXFP8 leaf registers against `b6247c2`; `test_unnormed_solutions_agree_byte_for_byte[dtype0-True-neox-64]` did not fail) |
| TokenSpeed runtime suite (GPU 6) | 9003 passed, 0 failed: cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 370, models 567, moe 72, sampling 144, top-level files 6787 |
| DeepGEMM (`DG_JIT_CHECK_NO_SPILLS=1`, fresh cache) | `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention`, `test_mega_moe_sm120` exit 0; `sm120_mqa_check.py`, `sm120_blockscale_check.py --mode all --grouped` and `sm120_paged_mqa_blockscale_check.py` end in ALL OK; `test_mega_moe_sm120_ep.py` (`torchrun --nproc_per_node 2`, GPUs 6+7): bitwise equal to one rank for every token split, up to 2.0e-11 against the PyTorch replay; `test_lazy_init --num-processes 2` 3 tests OK |
| End to end, greedy, in-process | Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2), Qwen3.8-27B-NVFP4 and gemma-4-31B-it (`<bos>` prefix) all answer "Paris" and "2, 3, 5, 7, 11" |

Notes. (1) The kernel suite has 7 more skips and 7 fewer passes than the task 25 run (7905 passed, 2763 skipped); the total is the same, the failure count is 0, and the 7 tests were not traced to a cause. (2) `test_mega_moe_sm120_ep.py` needs both GPUs, so it runs under `torchrun` with `CUDA_VISIBLE_DEVICES=6,7`, not through the one-GPU `sm120_remote.sh dg` helper. (3) Open items are unchanged: a real V3.2 TP8 run (checkpoint and 8 GPUs), full V4.1 with 384 experts (4 GPUs), the MegaMoE TPOT gap under attention DP (9.29), BF16 DSA / V4 at 42-48% of the tensor-pipe floor, V4 at 16 heads on Triton, one-request DSA decode (8 chunks) and the dense gate_up M=256 FP4 tie (9.28).

## 9.33 Task 29: MegaMoE under tensor-parallel attention (2026-11-10)
TokenSpeed `nctu6` from `040aac7dc` (commits `d4d972fe0` leaf + `moe_plan`, `c48b31c53` V4.1 MoE, `65574ea42` MoE-TP shard, `c48960960` tests, docs `cdd7b3631`, `b9de919e5`), DeepGEMM `343d194` (code) on `ac22b25`. RTX PRO 6000, GPUs 6 and 7 (idle, `PIX`), real V4.1
(`/models/deepseek-ai/DeepSeek-V4.1-Flash-1017`, all 40 layers, first 128 of 384 experts), in-process `Engine`, util 0.85, Engram host
table, prefill graphs off, decode graphs on, 2 requests of 2004 random tokens, prefix caching off, fresh prompts every repetition.
Prefill tok/s from 1-token requests; TPOT = (T(257) - T(1)) / 256; median of 9 repetitions after two warm runs. Harness:
`tokenspeed/tmp/t29/v41_bench.py` (scratch, deleted).
**Question.** Task 24 left MegaMoE DP2/EP2 1.31x behind Marlin TP2 in TPOT (+2.6 ms: every DP rank reads the whole dense model; +2.3 ms
glue). With attention TP the dense model is split again and both ranks hold the same tokens, so the push / return protocol is not needed.
**Design: one parameter of the existing path, no new kernel.** The kernels already serve one rank (`num_ranks == 1`, plain `torch`
buffer, `group.size() == 1`) and treat a top-k slot outside `[0, num_experts)` as masked (`dispatch` computes `is_local`, `combine`
skips the slot), so with replicated tokens each rank runs the one-rank kernel over its own shard of the experts and the caller sums the
partials with the all-reduce a TP layer already does. TokenSpeed adds `moe_plan(ep_token_layout=...)`: `scattered` (attention DP, the
existing protocol, EP 2) or `replicated` (attention TP). The SM120 leaf `deep_gemm_sm120_mxfp4_mega_moe_apply` declares both (trait
`ep_token_layout`; the SM100 leaf `scattered` only) and reads the layout from the plan. For `replicated`, the plan's process group is a
one-rank group (`MoELayer` takes the MoE TP or EP group, whichever is one rank; the initializer creates both), the kernel's expert count is
the rank's local count, and `stage_mxfp4_mega_moe_inputs(..., local_expert_begin, num_local_experts)` writes `id - begin` for the rank's
experts and -1 for the rest in the same pass that quantizes the rows (runtime ints, no recompile). `DeepseekV4MoE` picks the layout from
the mapping (`attention TP == MoE TP x EP`); `replicated` is then a normal TP / EP MoE layer (TP-EP sharded shared expert on the aux
stream, then the usual all-reduce), `scattered` is the old fused path. Two shards of the replicated layout work, both on the same kernels:
(a) an expert range (`--ep-size 2`, ids remapped as above) and (b) MoE TP (`--tensor-parallel-size 2`, no `--ep-size`): each rank holds a
1152-wide slice (9 x 128) of every expert, SwiGLU and the per-32 UE8M0 activation scales are elementwise inside a slice, so the activations
quantize exactly as unsharded and the second GEMM sums over the slice. Widths of one and two ranks are accepted, mixed TP x EP is refused.
**Bug found on the way (DeepGEMM `343d194`).** The first real run died in the warmup with an out-of-memory error (92.7 GiB allocated on
each GPU). Rows of the one-rank call follow the call's tokens, and the warmup counts up 1..2048 then in steps of 16, so the persistent
scratch (`SM120MegaMoEScratch`) grew at every size and pushed every predecessor to `retired` (kept for graphs captured earlier): a
quadratic leak, 9 minutes into the load. A call now asks for the capacity of its worst routing (`num_max_tokens_per_rank` tokens, every
pair local, tile height 128) and carves its own rows from it. `tests/test_mega_moe_sm120.py::test_scratch_is_sized_once` (new) calls with
1, 7, 64, 300, 1000, 2048 tokens and requires no growth after the first; `test_mega_moe_sm120_ep.py` stays bitwise equal to one rank
(7 token splits, fresh cache, `DG_JIT_CHECK_NO_SPILLS=1`).
**Layer-level tests** (`tokenspeed-kernel/test/nvidia/ops/moe/test_mega_moe_sm120_token_layouts.py`, `torchrun` on GPUs 6/7; real
`moe_plan` / `moe_process_weights` / `moe_warmup` / `moe_apply`; H 1024, I 256, 16 experts, top-4, 192 tokens with routes
spread over both ranks, all on rank 0 and all on rank 1): relative error (norm) of the summed partials against the scattered call
(in-kernel FP32 combine, then BF16): EP shard 0.0021, MoE TP shard 0.0037 (half an ulp of each BF16 partial; 83% / 47% of the
elements bitwise equal, largest difference 0.03 on values up to 6.9); against a float reference over the dequantized weights and the
unquantized activations 0.0467 for all three (FP8 activations), so the shards add nothing over the scattered protocol. A rank whose
routes are all remote returns exact zeros; calls are bitwise repeatable; a captured graph replayed with new inputs equals eager for
both shards. `test_mega_moe_stage.py` (masks, quantized rows independent of the range, no recompile over ranges), the selection tests
(`replicated` selects the SM120 leaf at TP width 2 / EP 2, SM100 does not) and the MoE-group test pass. Kernel subset (selection, SM120
gating, `ops/moe`, `nvidia/ops/moe`, `nvidia/thirdparty`) 973 passed, 826 skipped, 0 failed; V4.1 runtime files + `runtime/moe` 367
passed.
**Real V4.1** (median of 9; the same harness for all four):
| Path | Prefill tok/s | vs Marlin | TPOT ms | vs Marlin |
|---|---|---|---|---|
| Marlin TP2 (`auto`) | 15.1K (14.8-15.2K) | 1.00x | 10.69 (10.24-10.80) | 1.00x |
| MegaMoE, attention TP2 + MoE TP2 | 17.0K (16.8-17.1K) | 1.12x | 10.66 (9.52-10.76) | 1.00x |
| MegaMoE, attention TP2 + EP2 | 16.8K (16.7-17.0K) | 1.11x | 11.22 (10.64-11.36) | 1.05x slower |
| MegaMoE, attention DP2 + EP2 | 19.1K (18.8-19.3K) | 1.26x | 14.02 (13.80-14.08, one 18.6 outlier) | 1.31x slower |
TP attention removes the +2.6 ms DP term and the push / barrier / return glue: TPOT 14.02 -> 10.66 (MoE TP) or 11.22 (EP). Per decode step
(torch profiler, 20 steps, rank 0 and 1, kernel time): Marlin TP2: Marlin MoE GEMMs 3.3 ms, all-reduce 0.26-0.31, SiLU 0.43, align 0.10,
reduce 0.19, fill 0.09. MegaMoE EP2: GEMMs 3.34 ms, but the all-reduce takes 0.80 (rank 0) and 1.11 ms (rank 1): it waits for the rank that
owns more of the 12 routed (token, slot) pairs, because an expert range cannot balance 12 routes over two ranks. MegaMoE MoE TP2: GEMMs
3.17 ms, all-reduce 0.24 / 0.30 (Marlin's), glue 0.73 ms (staging 0.07, dispatch 0.10, gather 0.18, SwiGLU 0.18, combine 0.20) against
Marlin's 0.81; total kernel time 13.93 vs 14.52 ms per step, but the step is bound by the critical path, and TPOT ties. The expert-range
shard is what a larger EP group would do (and what task 24 measured as barrier skew: 0.8-1.05 ms of the DP2/EP2 glue was this imbalance, not
the barrier itself); MoE TP is the balanced shard at this width, with the same bytes per GPU.
**Greedy tokens.** First greedy token of 32 random-token prompts (200 tokens, 1 new token): MegaMoE TP2-EP agrees with DP2/EP2 on 22/32,
with Marlin on 23/32; DP2/EP2 with Marlin on 25/32. That is the path-to-path spread of the 128-expert model that task 22 measured (the
reduced model amplifies a 0.7% single-layer difference to a flat distribution by layer 39), not a defect of one path: the layer-level
agreement above is the correctness evidence. Random-token and two natural-text prompts (48-64 tokens) agree between paths up to
near-ties (" Paris. The capital" for Marlin and the TP shard, " the capital of France" for DP2 at token 0; "1, 2, 3, ..." identical for all);
the same prompts do not repeat bit-for-bit between two runs of one path when the scheduler batches them differently (not batch-invariant,
as before), so the token comparison is not a per-token test.
**Decision.** `auto` keeps Marlin: MegaMoE beats it in prefill (1.11-1.12x) but ties TPOT (10.66 vs 10.69 ms, inside the 0.1 ms spread of
either), and the rule is a win on both. Opt-in `--moe-backend mega_moe` serves attention TP with MoE TP2 (default for two GPUs) or EP2, and
attention DP with EP2. `auto` could not select it without more: the leaf is not batch-invariant (it would need the numerics envelope check
and a width gate), and it adds a persistent scratch (capacity of the worst routing: about 1.5-1.8 GB at 8192 tokens, V4.1 shapes) and a symmetric buffer to the memory plan.
**Open.** (1) TPOT parity needs the glue to shrink: fusing dispatch + gather (one kernel at one rank, an estimated 3-5 us per layer of launch gap,
0.1-0.2 ms per step) and SwiGLU into the L1 epilogue are the candidates; the bound is 0.2-0.3 ms of 10.7 (2-3%), below what a wrong
decision would cost, so not built. (2) EP shards at larger widths (EP4 needs V4.1's full 384 experts on 4 GPUs) will see the same
imbalance at small batch; MoE TP of the same width stays balanced. (3) Only widths 1 and 2 are tested. (4) Not tested: full 384-expert V4.1,
DSpark drafters, PD disaggregation with this layout.

## 9.34 Task 30: stream-K schedule for the CuTe DSA core; persistent CTAs and 16 chunks measured and closed (2026-11-10)
TokenSpeed `nctu6` from `735ad1c0b`, DeepGEMM `1571432` (`build312`, unchanged). RTX PRO 6000 (188 SMs), kernel benchmarks on GPU 7,
tests on GPU 6, CUDA-graph microseconds, best of 5, HEAD and the new core in one process with the replays interleaved (no clock drift).
`ncu` is still blocked: the evidence is ablation and timing. Scratch scripts were in `tokenspeed/tmp/t30/` (deleted). TokenSpeed
commits: `e51c66452` (kernel, planner, tests), `084396fc3` (README).
**Where the time of a many-CTA launch goes.** A CTA fills an SM (99 KiB), so a launch is waves of 188 CTAs. Unsplit 32-head BF16
programs: 1 program 138 us, 188 programs 149 us, 189 programs 287 us, 376 programs 296 us. The shipped planner (9.30) splits every
token alike (1/2/4/8 CTAs), so 256 programs (64 tokens x 128 heads) took 281 us (two waves) or 8 splits with stored states. A split costs
7-10 us (BF16 32 heads, 2 / 4 / 8 splits: 79 / 43 / 25 us against 138 / k; E4M3 51 / 29 / 17 against 81 / k; 16-head CTAs 7 / 7 / 6.5 us):
the chunk-state stores at the end of the CTA's chain plus the combine kernel (about 2 us).
**Shipped: a stream-K schedule as a parameter of the same core.** `sparse_mma_attention(..., num_splits, full_tokens)`: the grid is one
row of blocks per whole token followed by `num_splits` rows per split token, so the CTAs of the last, partly filled wave are short and fill
the SMs the whole tokens leave idle (the hardware hands CTAs out in grid order). `full_tokens` equal to the token count is the old unsplit
launch, 0 the old "split every token". `plan_launch(tokens, heads, num_sms, fp8)` returns `(block_h, num_splits, full_tokens)`: the old
rule (waves x chunks per CTA with the 1.25x / 1.5x margins) is the base plan, and a stream-K plan (a tail of 1/8..2 waves, or the remainder
of the last wave, split 2/4/8 ways) replaces it only if a wave model (unsplit CTA 115 / 138 us BF16, 62 / 81 us E4M3 at 16 / 32 heads,
+4 us per split CTA, +4 us combine, list scheduling of the CTAs) is 8% ahead. A brute-force oracle over (block_h, splits, whole tokens)
per shape gives 1.038x (BF16) / 1.039x (E4M3) geomean over the 140 decode shapes (up to 1.38x / 1.51x, never slower than the shipped plan);
the model-based planner reaches 1.033x / 1.032x of the shipped plan with no timing at run time. The fold order is untouched: rows and LSE are bitwise equal to the old
core for every plan (all 344 points of the final runs, the 280 of the plan comparison, 72 split / chunk combinations, and
`test_cute_split_count_and_head_tile_do_not_change_the_result` over `full_tokens` 0..4 x splits 1..99 x both head tiles).
Final numbers, HEAD core + planner against the new ones (interleaved):
| set | points | geomean | worst | best |
|---|---|---|---|---|
| 140 decode shapes (1-192 tokens x 16/24/32/40/64/96/128 heads), both dtypes | 280 | **1.0315x** (BF16 1.034, E4M3 1.029) | 0.970 (E4M3, 24 heads, 3 tokens, same plan) | 1.378 (E4M3, 40 heads, 192 tokens) |
| 64 DSA buckets (16/32/64/128 heads; decode 1/8/32/128, prefill 128/1024/2048/8192) | 64 | **1.0173x** (BF16 1.017, E4M3 1.017) | 0.986 | 1.300 (BF16 decode, 64 heads, 128 tokens: 293 -> 226 us) |
One-request decode (16 / 128 heads): BF16 20.3 / 20.5 -> 20.2 / 20.4 us, E4M3 13.1 / 12.9 -> 13.3 / 13.1 us. The gain sits where a batch
is a little past a whole number of waves: 192 tokens at 16 heads 176 -> 146 us, 48 tokens at 128 heads 225 -> 168 us, 192 at 128 heads
693 -> 584 us. Prefill chunks of 1024-8192 tokens are within 0.4-1.8% (tens of waves; the 128-token, 64-head chunk is 256 programs and gains
1.28-1.30x); 38 of the 64 buckets are within +-0.5%.
**A regression found and fixed on the way.** The first version indexed blocks one-dimensionally by program and stored the chunk states
as `[program][chunk][head of the tile]`: the E4M3 kernel (255 registers, 24-48 B stack) lost 3-6% (`16 heads, 128 tokens` 58.1 -> 60.4 us
of kernel time, 64.0 -> 67.7 us graph) and the combine 0.5-1.1 us. Bisecting with a 3D grid and the old store form brought it back to
58.25 us: the cost was the extra live value in `_store_state` plus the decode, not the divisions (a `mul.hi` replaces them, no change).
The schedule is therefore by token: grid x = head tile, grid y = row (whole tokens, then `num_splits` rows per split token, token-major),
the state store keeps its `[token][chunk][head]` form with the token offset by `full_tokens`, and the combine launches one block per
(split token, head). Cost left: E4M3 small decode +0.1-0.3 us (0.5-2% of 13 us), BF16 none; the 140-shape sweep is net positive.
CuTe resource usage matches HEAD (255 registers, 24 / 48 / 56 B stack; 4120 / 4000 / 3680 SASS instructions, BF16 16 heads 8 fewer).
**Persistent CTAs (prefetch the next work item's query and first tile): not built, bounded.**
| launch | full | all selections empty (launch + query + store + combine) | one tile |
|---|---|---|---|
| BF16 16 heads, 1 token (8 CTAs) | 20.1 us | 5.3 us (26%) | 7.1 us |
| BF16 128 heads, 8 tokens | 46.5 | 11.1 (24%) | 13.5 |
| BF16 128 heads, 32 / 128 / 192 tokens | 144 / 441 / 611 | 5.7 / 17.4 / 29.9 (3.9% / 4.0% / 4.9%) | 10.3 / 29.8 / 48.7 |
| E4M3 16 heads, 1 token | 13.4 | 4.6 (35%) | 6.8 |
| E4M3 128 heads, 8 / 32 / 128 / 192 tokens | 30.2 / 84.1 / 248 / 348 | 10.0 / 3.4 / 8.3 / 16.3 (33% / 4.0% / 3.4% / 4.7%) | 12.7 / 8.5 / 23.4 / 38.9 |
At one request every CTA has exactly one work item (8-16 CTAs on 188 SMs), so there is nothing to chain; the time is 5 us of launch, query
and combine plus the tile chain (about 1.9 us per 32-row BF16 tile, 2.2 us per 64-row E4M3 tile, 8 chunks of 8 / 4 tiles). Where there are
many items the whole empty launch, which includes the query read and the output write, is 3.4-4.9%, the ceiling of any prefetch.
And there is nowhere to prefetch to: the BF16 CTA keeps its query in 32 of 255 registers (56 B stack already), the E4M3 query tile is
18-36 KB next to a 99 KiB ring. The stream-K schedule takes the larger part of what a persistent kernel was meant to recover (the idle
tail, up to 1.38x), without any new per-CTA state.
**States on chip across a CTA's chunk run: not bit-exact, not built.** The fold is a left fold, `((s0 + s1) + s2) + ...`. A CTA that owns
chunks 4-7 would need the running state of chunk 3, which belongs to another CTA. A chain through flags in global memory pays one
dependent hop (store, flag, load; an estimate of 1-2 us, not measured) per CTA against the 2 us combine that folds all 8 or 16 in one pass; a chain over DSMEM
pays task 23's 5 GB/s writes (a 32 KB state per head tile is 1 us per hop at 33 GB/s reads). The stores themselves are not the cost: they
are asynchronous and the chunk-state traffic is only comparable to the gather traffic at 64+ tokens x 128 heads (64 tokens x 16 chunks
x 128 heads x 2 KB = 268 MB written and read again, against 604 MB of gathered rows).
**16 chunks, again, with the new planner as the oracle.** A brute-force plan per shape for 8 and 16 chunks (width-following, so the count
cannot depend on the batch), both dtypes, 140 decode shapes: 16 chunks against 8, BF16 1.057x geomean (1-4 tokens 1.31x, 5-12 tokens 1.06x,
13-64 tokens 0.95x, 65-192 tokens 0.99x; worst 0.85x), E4M3 1.007x (1.23x / 0.99x / 0.91x / 0.98x; worst 0.81x). One-request decode:
BF16 16 heads 20.6 -> 13.7 us, 128 heads 20.5 -> 16.6, E4M3 12.8 -> 9.6, 13.1 -> 10.3. The loss at 13-64 tokens (the common decode batch)
is the doubled state traffic and one more split CTA per token; 32 chunks lose everywhere past 4 tokens (0.92x geomean). Prefetching the
combine's loads (1, 2, 4 or 8 chunk states ahead, `ld.global.v2` for the m / l pair) gained nothing (graph times within 0.3 us), so the
combine is not latency-chained as the 9.30 estimate assumed, and the original loop stays.
Tests (`test_dsa_mma_sm120.py`, 144 passed; attention directory 960 passed, 213 skipped on GPU 6; DSA + V4 + MLA + selection files 513
passed): the plan at the wave boundaries (`test_cute_plan_fills_the_last_wave_with_split_programs`: 192 tokens x 16 heads is 188 whole + 4
split), plan invariants over 1-699 tokens x 5 head counts x both dtypes, `test_cute_rows_do_not_depend_on_a_stream_k_schedule` (192 tokens,
whole + split mix against slices of 1-191 tokens, bitwise rows and LSE, both dtypes), split invariance over every `full_tokens`, the
recompile guard extended with 192 / 193 / 389 tokens (stream-K counts), and the grid-row limit (65535 rows). The V4 prefill core shares
the kernel and combine (`full_tokens` 0 or all): `test_dsv4_mma_sm120.py` passes.
**Closed (don't retry without new evidence):** a persistent CTA with next-item prefetch (ceiling 3.4-4.9%, no register or shared memory
for the prefetch, nothing to chain at one request); on-chip chunk states across a run (not bit-exact; hop cost above the 2 us combine);
16 or 32 chunks for every batch (oracle 0.81-0.95x at 13-64 tokens); combine prefetch; stream-K at head-tile instead of token
granularity (costs the E4M3 core a register); empty-CTA z-planes for the split index (the empties run first). **Open:** the E4M3 core
spends 0.1-0.3 us more per small decode launch than before (block decode); BF16 prefill at 42-48% of the tensor-pipe floor and the V4
core's own planner (9.30: oracle 1.01x) are unchanged.

## 9.35 Task 31: V4 attention at 16 heads off Triton: closed; the 8-warp 16-head CTA and a V4 stream-K planner shipped for more than 16 heads (2026-11-10)
TokenSpeed `nctu6` from `d863d1cb5`, DeepGEMM `ab1aa37` (`build312`, unchanged). RTX PRO 6000 (188 SMs), kernel benchmarks on GPU 7
(GPU 6 was held by another user's job: 86 GB, 100%), CUDA-graph microseconds, best of 5. Scratch scripts were in `tokenspeed/tmp/t31/`
(deleted). TokenSpeed commits: `d55bb24bd` (kernel, planner, tests), `75ba90021` (READMEs). `ncu` is still blocked: the evidence is ablation and timing.
**Where the time is at 16 heads.** One token's 640 rows (1 KiB each) are gathered for 16 heads only, 32 flop per byte. At 2048 tokens that
is 1.34 GB: the Triton leaf (8 warps, loads straight to registers) takes 340.7 us, 3.94 TB/s, the L2 -> SM rate the `cp.async` core reaches
alone (4.0 TB/s in 9.30). The CuTe core with its MMAs removed (gather, mask, barrier per tile) takes 348.4 us on 4 warps and **325.2 us on
8 warps** (4.13 TB/s): that skeleton is the Triton total, so no compute chain, however short, can win more than 5% at 640 rows and 9%
at 128 rows (skeleton 84.7 us, Triton 92.4) at 2048 tokens. Only fewer bytes can, which means rows shared between tokens.
**Overlap of real selections (what is known).** The prefill selection of a token is `[window 128 | compressed top-k <= 512]`
(`deepseek_v41.py`): the window is the 128 workspace rows of positions p-127..p, consecutive, so token p+1 has token p's window shifted
by one column (127 of 128 shared: 100% of a width-128 layer's columns, 20% of a width-640 layer's). The compressed rows are the indexer's
top-k per token. For positions below 2048 at ratio 4 at most 512 compressed rows are visible, so top-512 takes them all and neighbours
select the same set; beyond that the overlap is data dependent and **was not measured**: the real V4.1 checkpoint needs both GPUs of the
earlier runs, GPU 6 was busy, and the task 22 / 24 harness was not kept. The ceiling below uses 100% overlap, so real data can only be
lower.
**Batch invariance limits what two tokens can share.** A token's tile k must stay its own columns 32k..32k+31 (the online softmax
steps and the BF16 rounding of the probabilities depend on the tile). A CTA with G tokens could therefore share only gathered rows
(each token addressing the shared rows with its own shift or row table), never the tile or the MMAs.
**Ceiling for sharing (ablation).** `cp.async` is issued only by the CTAs of every G-th token (stale shared memory, wrong results, all
other work unchanged), i.e. perfect overlap and no cost for finding it:
| 2048 tokens (us) | width 640 | width 128 |
|---|---:|---:|
| Triton | 340.7 | 92.4 |
| CuTe 4 warps, as shipped before / sharing G = 2 / G = 4 | 465.9 / 394.0 / 368.0 | 149.0 / 128.8 / 122.4 |
| CuTe 8 warps, as shipped now / sharing G = 2 / G = 4 | 410.3 / 368.9 / 343.1 | 118.7 / 110.9 / 104.0 |
| CuTe 8 warps, gather only | 325.2 | 84.7 |
At 512 tokens (Triton 116.3 / 28.4): 8 warps 115.6 / 34.5, G = 2 103.2 / 31.6, G = 4 96.2 / 30.0. Even with perfect overlap two or four
tokens per CTA reach 0.92x / 0.99x of Triton at 640 rows and 0.83x / 0.89x at 128 rows, and halving the gathered bytes of the 4-warp
kernel gains 18% (the gather is then hidden behind a chain that does not shrink). Removing the gather bytes altogether (zero-fill) leaves
426 us at 640 rows; at 128 rows it is slower (188 us, a zero-fill artifact). So the dedupe machinery (a row table per token, equality
tests of the shifted lists, an indirection in every `ldmatrix` address) cannot pay: **closed, not built**.
**8 warps with the 4-warp bits (built, shipped as a CTA shape).** 9.30's 8-warp CTA split the channels 8 ways, so its score sum differed
from the 4-warp sum and it could only serve 16 heads. New layout: the two warps of a channel quarter score half of the 32 rows each (the
same MMA sequence per row, so the same per-quarter partial; one 2 KB exchange slot per warp, the same 8 KB), every warp then adds the
four quarters of all 32 rows in the fixed order (so the softmax is the 4-warp softmax in every warp), and PV splits the output channels
8 ways (acc 16 x 64: 32 registers + 32 for the fold). Rows and LSE are bitwise equal to the 4-warp CTA (8 shapes with holes, short
lengths, int32/int64 ids, 1-12 live heads, 1-4 splits; `test_cute_rows_do_not_depend_on_the_launch_shape`,
`test_cute_eight_warp_cta_with_partial_head_tiles`). It is a parameter of the same core (`cwarps`, V4 variant, `block_h` 16), not a new
path, and serves every head count (the leaf may use either CTA freely).
**16 heads against Triton** (sink, CUDA graph, best of 5; CuTe 4-warp = HEAD, CuTe 8-warp = the shipped plan; ratios are Triton / CuTe):
| tokens | Triton 640 | 4 warps | 8 warps | Triton 128 | 4 warps | 8 warps |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.4 | 13.4 (1.00x) | 11.9 (1.13x) | 6.2 | 6.9 (0.89x) | 6.7 (0.92x) |
| 4 | 13.8 | 13.9 (0.99x) | 12.3 (1.12x) | 6.2 | 7.0 (0.88x) | 6.9 (0.89x) |
| 16 | 14.4 | 14.1 (1.02x) | 12.8 (1.13x) | 6.5 | 7.2 (0.92x) | 7.2 (0.90x) |
| 64 | 21.7 | 25.9 (0.84x) | 23.7 (0.92x) | 10.3 | 11.8 (0.91x) | 10.1 (1.03x) |
| 128 | 33.3 | 42.9 (0.78x) | 38.4 (0.87x) | 15.3 | 14.5 (1.05x) | 11.6 (1.32x) |
| 192 | 49.7 | | 48.5 (1.02x) | 20.0 | | 17.9 (1.12x) |
| 256 | 58.7 | | 62.3 (0.94x) | 25.1 | | 21.6 (1.16x) |
| 384 | 102.4 | | 86.6 (1.18x) | 26.6 | | 29.0 (0.92x) |
| 512 | 117.2 | 128.2 (0.91x) | 115.9 (1.01x) | 28.4 | 41.8 (0.68x) | 34.7 (0.82x) |
| 2048 | 340.6 | 459.0 (0.74x) | 403.5 (0.84x) | 89.1 | 146.7 (0.63x) | 122.5 (0.73x) |
The 8-warp CTA is a clear step over the 4-warp one (1.09-1.16x at width 640) and ties or beats Triton at 1-16 tokens and some mid counts
(wave edges), and loses at 64 / 128 / 2048 tokens of width 640 and at 1-16 / 512 / 2048 tokens of width 128. The two leaves differ bit
for bit, the leaf can only follow the head count, and choosing by token count would change a row with its batch, so **16 heads stay on
Triton** (`num_q_heads_min = 17`, unchanged).
**Splitting a token (two CTAs per token) and the stream-K tail at 16 heads.** 8-warp CTA, width 640, forced splits 1 / 2 / 4 (Triton):
4 tokens 33.3 / 20.1 / 12.1 (13.8), 16 tokens 33.6 / 20.3 / 12.5 (14.4), 64 tokens 34.2 / 23.4 / 25.5 (21.7), 128 tokens 37.9 / 44.5 /
39.1 (33.3), 512 tokens 115.6 / 132.5 / 136.3 (117). 128 programs of 4 chunk times on 188 SMs: the only split that finishes sooner is a
2 x 2 that needs 256 CTAs, a second wave. Splits already follow the planner; the best split per count is what the table above uses.
**The V4 wave planner, again, with stream-K tails (9.30 rejected it at 0.97x / 0.85-0.98x).** The DSA model had DSA's CTA times and no
tail; the V4 one (`_FIXED_US`, `_TILE_US` fitted to 1 / 2 / 4 splits of widths 128 / 640 / 1152, per head tile; 0.5 us per split CTA and
0.5 us combine) also knows the width. `plan_launch(tokens, heads, num_sms, width)` returns `(block_h, cwarps, num_splits, full_tokens)`;
the uniform rule stays the base and a plan replaces it when the model is 5% ahead. Brute force over (head tile, cwarps 8, splits) for 126
points (32 / 64 / 128 heads x widths 128 / 640 / 1152 x 14 token counts) gave the best uniform plan only 1.027x ahead of the old rule
(up to 1.34x); the model with tails does better. HEAD (4-warp CTAs, old rule) against the new core and plan, same GPU, graph, 165 points
(17 / 24 / 40 / 96 / 128 heads x widths 128 / 640 / 1152 x 1 / 3 / 7 / 12 / 24 / 40 / 100 / 150 / 200 / 300 / 500 tokens):
**1.095x geomean**, 0.893x worst (17 heads, width 128, 24 tokens: 8.34 -> 9.34 us; two more points below 0.97x, 24 heads / 24 tokens 0.935x
and 40 heads / 500 tokens 0.969x), 1.655x best (17 heads, width 1152, 40 tokens: 43.8 -> 26.5 us). 1-3 tokens of width 640 / 1152:
1.08-1.17x (the 8-warp CTA); 32 heads, 192 tokens, width 640: 97.6 -> 66.7 us (188 whole CTAs and 4 tokens split four ways in one wave);
width 128, 32 heads, 64 tokens: 16.0 -> 11.8 (16-head CTAs unsplit, no combine). On the 126-point grid (heads 32 / 64 / 128): 1.096x,
0.98-1.50x (the 0.98 is the same plan measured twice). A margin of 1.10 / 1.20 keeps 1.084x / 1.060x and does not remove the few
regressions (the model's split cost is too high for a 32-head CTA at 128 rows), so 1.05 stays. 128 heads and 512 tokens or more are the
same plan as before (whole CTAs).
**Tried and lost / bounded, 16 heads:** two or four tokens per CTA (above); a persistent CTA (the stream-K tail is the plan; at one
request the CTA count, not the chain, is the limit); more splits (above); 4 warps with a 2-stage ring (0.74x at 2048).
**Not tried:** 16-row tiles with 3-5 stages and two CTAs per SM, or software-pipelining scores of tile i+1 under PV of tile i (needs a third
stage, which 99 KiB does not hold at 32 rows): the 8-warp gather skeleton equals Triton's total, so their ceiling is 5-9% at 2048 tokens
before any extra per-tile cost, and they change the bits of the tile fold.
Tests (`test_dsv4_mma_sm120.py`, 52 passed; attention directory 920 passed, 272 skipped; V4 / V4.1 runtime files, GPU 7: 422 passed,
7 failed in `test_deepseek_v4_config.py`, all `AttributeError: 'SimpleNamespace' ... 'ep_token_layout'`, the stand-ins of the MoE
forward tests lack the attribute task 29 added, so they fail without this change too): bitwise equality over head tile x warp layout x
splits x `full_tokens` (int32 and int64 ids), the 8-warp CTA with 1 / 8 / 16 live heads (also against the float32 reference), plan
validity over 1-699 tokens x 4 head counts x 3 widths (cwarps follows the head tile, grid rows within 65535), the wave-boundary plan
(192 tokens x 32 heads is 188 whole + 4 split), the compile cache guard (head tile and warp layout are one binary each).
**Closed (don't retry without new evidence):** two or more tokens per CTA at 16 heads (perfect-overlap ceiling 0.83-0.99x of Triton at
2048 tokens); the 16-head V4 leaf on CuTe (mixed 0.63-1.32x by token count; the leaf follows the head count only); a persistent CTA for V4.
**Open:** the overlap of real V4.1 selections beyond 2048 positions (needs both GPUs and the real checkpoint); the three 128-row planner
regressions (a per-block split cost); 16 heads at width 640 for 64-128 tokens (0.87-0.92x), where 128 CTAs of 4 chunk times meet 188 SMs.

## 9.36 Task 32: the 7 kernel-suite tests that moved from passed to skipped, and three regressions of tasks 29-31 (2026-11-10)

**The 7 tests are not a DeepGEMM effect.** Task 25 had 7905 passed / 2763 skipped and task 26 7898 / 2770 over the same 10668 tests. They are the seven
SASS checks of `nvidia/ops/communication/test_allreduce_sandwich_rmsnorm.py`: `test_triton_rmsnorm_compiles_to_the_pinned_arithmetic` (6 cases,
`hidden` 1024 / 4096 / 7168 x residual) and `test_two_shot_kernels_fit_two_ctas_of_21_warps_per_sm`. They skip with "cuobjdump is not on PATH" /
"needs the built trtllm_comm.so and cuobjdump" when `shutil.which("cuobjdump")` finds nothing. The task 26 environment had `PATH` set to the venv
(`/workspace/nctu6/infra/tokenspeed/.venv/bin:$PATH`), which lacks `/usr/local/cuda/bin`; the task 25 shell had the toolkit on `PATH`.
Evidence: (1) the task 25 TokenSpeed tree (`4e6bf8277`) against the task 26 DeepGEMM build gives 7898 passed / 2770 skipped, the task 26 numbers, so the
TokenSpeed tree between the two runs is not the cause; (2) the same tree against a DeepGEMM `62af89e` build (built in `DeepGEMM/tmp/dg62`, fresh JIT
cache) has the same 10668 tests and no test that is passed there and skipped with `b6247c2` (the differences go the other way: 37 MXFP8-leaf tests fail and two
SM120-selection tests skip on the old pin and pass on the new one, e.g. `deepgemm_sm120_mm_mxfp8` is not registered), so the pin cannot move a
test from passed to skipped; (3) with `/usr/local/cuda/bin` on `PATH` the file passes 149 of 149 (7 skips otherwise). Other environment
differences were ruled out: a warm and a cold DeepGEMM JIT cache give the same skips; two visible GPUs (`CUDA_VISIBLE_DEVICES=6,7`) turn only 3
device-count skips (`test_trtllm_comm` x2 and `test_projection_tp[tp2]`) into runs (one of them fails, see below), which is a different set.
**Fix (TokenSpeed `32d0d0b95`):** the tests resolve `cuobjdump` from `PATH`, then from the toolkit PyTorch builds extensions with (`CUDA_HOME`), so a
venv-only `PATH` no longer hides them; `test_cuobjdump_is_found_when_the_toolkit_is_not_on_path` covers it. The file now runs 150 passed with a
venv-only `PATH`. This was an intended skip caused by the shell, now removed as a way to lose coverage silently.
**Final kernel suite** (the task 8 command, GPU 7, venv-only `PATH`, TokenSpeed `e6f9df2e4`): 7938 passed, 2764 skipped, 0 failed, 10702 tests (the 10668
of task 26 plus 33 from tasks 29-31 plus the `cuobjdump` test); the 2764 skips are the 2763 of the task 25 run plus the 2-rank
`test_mega_moe_sm120_token_layouts.py` case, which needs `torchrun`.

**Regression 1: `test/runtime/test_deepseek_v4_config.py`, 7 failures** (`AttributeError ... 'ep_token_layout'`). `DeepseekV4MoE.forward` chooses the MegaMoE protocol
from `ep_token_layout` since task 29 (`use_mega_moe` before). The `SimpleNamespace` stand-ins of the MoE forward tests only had `use_mega_moe`. They passed
on `/tmp/ts_base` (67 passed) and fail at the task 29 tree (7 failed, 62 passed). The stand-ins now carry `ep_token_layout` (None for the plain forward,
`"scattered"` for the MegaMoE one), TokenSpeed `689f9bd6e`; the file passes 69 of 69. (A real `DeepseekV4MoE` needs the whole MoE layer and its process
groups, so the forward tests keep their stand-ins, which now name what the code reads.)

**Regression 2: `ep_token_layout` had a default** (AGENTS.md: arguments that select an execution path have none). `moe_plan(ep_token_layout=)` and
`MoELayer(ep_token_layout=)` are now required (`MoELayer`: keyword-only). `None` is a stated value, "no kernel of the layer owns the token exchange", for
every backend except DeepGEMM MegaMoE, whose leaf still refuses a plan without `scattered` or `replicated`. It cannot be a hard error for all MegaMoE
plans: the SM100 CuTe `trtllm_nvfp4_mega_moe_apply` leaf (also solution `mega_moe`) has no layout trait and plans with None. All call sites name it: 11
model files, the benchmark generator, 41 test files (116 calls). TokenSpeed `e6f9df2e4`; runtime layers + moe + models + V4.1 model + the plan tests
1562 passed, 103 skipped; the NPU tests (`tokenspeed-kernel-npu/test`) already lacked `combine_order`, so they were not touched.

**Regression 3: `test_numerics.py::test_gemm_bf16[...triton_mm_bf16_skinny]` and `test_triton_adapter.py`** are pre-existing and outside the task 25/26 command
(neither file is in it): both fail identically on `/tmp/ts_base` (892e00ce2), at `4e6bf8277` and at HEAD, so tasks 29-31 did not cause them. Causes: (a) the
reference is wrong, not the kernel. cuBLAS splits K for M <= 16 and adds the partial sums in BF16 (`allow_bf16_reduced_precision_reduction`, on by
default): vs the correctly rounded result at K = 7168, N = 256 / 2112, M = 8 / 16, `torch.mm` has 123 / 304 / 2322 elements beyond the 1.5e-2 tolerance
(error up to 2.0), `triton_mm_bf16_skinny` has 0 (error 0.25-1.0, the output rounding) and an FP32-accumulated reference is within 0.03. `verify_kernel` now
runs the reference with the reduced-precision reductions off (TokenSpeed `3bec4e6f3`), which is what the GEMM tolerance assumes. (b) `ops/sampling/triton/single_step.py`
(added by `1d58e82be`) imported stock `triton`, against the adapter rule; it uses `tokenspeed_kernel._triton` now, and the vendored FA4
`fa4_sm90/cute_dsl_utils.py` (reads stock Triton's disassembler) is listed with the other interop helpers (`a88b744a7`). `test_numerics.py` +
`test_triton_adapter.py` 24 passed, 2 skipped (empty parameter sets). The whole `tokenspeed-kernel/test` directory on the mirror: 12 failures, all
file-not-found for sources the mirror rsync excludes (`tokenspeed-kernel-amd`, `.github/scripts`), none in code.
**Side finding:** `test_projection_tp[tp2]` fails when two GPUs are visible (a spawned rank raises); it skips on one GPU and was not part of any earlier
run. Not investigated here (NVLink assumption on a PCIe box).

## 9.37 Task 33: final verification of tasks 29-32 (2026-11-10)

Code heads: DeepGEMM `343d194` (one-rank MegaMoE scratch; docs-only commits after it), TokenSpeed `05937a09b` (the `prepacked` FP8 quantizer fix below, on top of the task 32 tree). The remote DeepGEMM tree was synced from the local tracked files (only the two `.md` files differed from the remote copy) and `sm120_remote.sh build` printed "build OK" (py3.15 and py3.12); the `/tmp/ts_mac` mirror was resynced. GPUs 6 and 7 only, all DeepGEMM JIT caches fresh, `DG_JIT_CHECK_NO_SPILLS=1`. The kernel suite ran with `PATH=/usr/local/cuda/bin:/workspace/nctu6/infra/tokenspeed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin` (task 32's fix also finds `cuobjdump` without it).

| check | result |
|---|---|
| TokenSpeed kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, GPU 7) | 7952 passed, 2764 skipped, 0 failed (7938 of task 32 plus the 14 cases of the new `test_fp8_quant_prepacked.py`); first run before the fix: 7938 / 2764 / 0 |
| `test_numerics.py` + `test_triton_adapter.py` | 24 passed, 2 skipped (empty parameter sets) |
| Two visible GPUs: `nvidia/ops/communication` (6+7) | 155 passed, 41 skipped; `test_projection_tp[tp2]` passes (tp4, tp8 skip: 4 and 8 NVLink GPUs required) |
| `test_mega_moe_sm120_token_layouts.py` (`torchrun --standalone --nproc-per-node=2 -m pytest`, GPUs 6+7) | 2 passed (one per rank), twice |
| TokenSpeed runtime suite (GPU 6, `/tmp/rt_dirs2.sh`, all ten directories) | 9007 passed, 0 failed: cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 374, models 567, moe 72, sampling 144, top-level files 6787 |
| DeepGEMM `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention`, `test_mega_moe_sm120` (including `test_scratch_is_sized_once`) | all exit 0, no spills |
| `sm120_mqa_check.py`, `sm120_blockscale_check.py --mode all --grouped` (170 cases), `sm120_paged_mqa_blockscale_check.py` (51 cases) | ALL OK |
| `test_mega_moe_sm120_ep.py` (`torchrun --nproc_per_node 2`, GPUs 6+7) | bitwise equal to one rank for all 7 token splits, up to 2.2e-11 against the PyTorch replay |
| `test_lazy_init.py --num-processes 2` (GPUs 6+7) | 3 tests OK |
| End to end, in-process, greedy | Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2, GPUs 6+7), Qwen3.8-27B-NVFP4 and gemma-4-31B-it (`<bos>` prefix) all answer "Paris" and "5, 7, ..." for the two prompts |
| Lint | `SKIP=no-commit-to-branch pre-commit run --all-files` (clang-format included) in a clean scratch clone of the local TokenSpeed HEAD: exit 0 twice, no file changed (no formatting commit needed); runtime ruff, scheduler ruff, the file-start / relative-import check clean, `test_weekly_release.py` 27 passed |

**`test_projection_tp[tp2]` (the task 32 side finding): a real bug, fixed.** It fails the same way at `/tmp/ts_base` (892e00ce2), so it predates this branch. The failing check was the fused A2A FP8 launch (`exchange_fp8_impl`, `TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess)`) at 128 rows. The error text was "no kernel image is available for execution on the device", but it came from an earlier call: a probe exported from the module (`cudaGetLastError` after each statement of the test) located it in the test's reference, `flashinfer_fp8_blockscale_quantize_prepacked(exchanged, 128)`, which for a row count divisible by four (TP2 x 128 rows = 256) calls the TRT-LLM `fp8_quantize_1x128` op. That op has no image after SM100 (the registered `trtllm_quantize_fp8_*` leaves stop at arch 10.9), so on SM120 its launch fails without raising in Python and leaves `torch.empty` output; the next `cudaGetLastError` in the process, here the A2A kernel's own check, reported it. The A2A kernel itself was fine (eager and graph runs of the same shapes pass). A direct call on an SM120 GPU raises "no kernel image" at the first synchronize. No production path reaches it on SM120 (its only runtime caller is the FlashInfer block-FP8 GEMM, SM100), but the helper returned garbage silently. Fix (TokenSpeed `05937a09b`): the TRT-LLM branch requires `platform.arch_version <= 10.9`; newer GPUs use the Triton kernel, which already handles `padded_rows == valid_rows`. The fused A2A kernel follows TRT-LLM's numerics for those row counts (division rounded to nearest, scale floor of the BF16 epsilon, unit scale only for padded shapes), so the test's reference for them is the same arithmetic in PyTorch where the op is absent (`tensor / tensor`, not `scalar / tensor`, which is a reciprocal times a multiply: the first attempt differed in 1128 of 2.1M elements by one code); the all-gather check uses it too instead of `quantize_fp8(solution="trtllm")`. `test_fp8_quant_prepacked.py` (new, any CUDA GPU) reconstructs the input for 1-256 rows and k = 128 / 1024: 8 of 14 cases fail without the fix on SM120, all pass with it. Not covered: tp4 / tp8 (4 and 8 NVLink GPUs); the other `trtllm` ops are untouched.

Lint: see the table; Lint workflow steps other than pre-commit (runtime ruff, scheduler ruff, file-start and relative-import check, `test/ci_system/test_weekly_release.py` 27 passed) ran on the same tree. Scheduler clang-tidy and the C++ build are not run (no ninja on the box).

Open items after this run: a real V3.2 TP8 run (checkpoint and 8 GPUs); full V4.1 with 384 experts (4 GPUs); the MegaMoE decode glue under attention DP (9.29) and the 0.2-0.3 ms that would make MegaMoE win TPOT under attention TP (9.33); BF16 DSA / V4 at 42-48% of the tensor-pipe floor; V4 at 16 heads on Triton (9.35); one-request DSA decode (8 chunks, 9.34); the dense gate_up M=256 FP4 tie (9.28); `test_projection_tp` tp4 / tp8 (needs 4 and 8 NVLink GPUs). Nothing is pushed: the user decides.

## 9.38 Task 34: MegaMoE glue under attention TP: SwiGLU in the L1 epilogue and the gather in dispatch; `auto` keeps Marlin (2026-11-11)
DeepGEMM `aa1ecfe` (code) on `956293a`; TokenSpeed `nctu6` unchanged (the DeepGEMM API did not change; only the pin moves). RTX PRO 6000,
GPUs 6 and 7 (idle when taken). Scratch in `DeepGEMM/tmp/` and `tokenspeed/tmp/t34/` (`v41_bench.py`, 9.33's harness).
**What fused.** The one-rank call (and EP, same kernels) loses three launches of the old sequence stage / dispatch / gather / L1 GEMM /
SwiGLU / L2 GEMM / combine: (1) the gather folds into the dispatch kernel (`sm120_mega_moe_dispatch` now counts the layout and copies the
FP8 rows, scales and weights in one pass; a few CTAs serve a decode batch, `kPairsPerDispatchBlock` = 8); (2) the L1 grouped GEMM carries
an epilogue (`SM120MegaMoEEpilogue::SwiGLU`, `sm120_mega_moe_epilogue.cuh`, a field of the same 1D1D kernel): the tile is rounded to BF16,
clamped, run through SwiGLU, scaled by the top-k weight and cast to FP8 with a UE8M0 scale per 32 outputs, and the L2 rows and packed
scale words are stored directly, so the BF16 L1 output (`l1_out`) and its scratch no longer exist; the epilogue's shared memory is
accounted in the stage count (`GemmDesc::epilogue_smem_bytes`). Rank count stays a parameter of the same kernels. The combine stays its own
kernel (a second grid dimension over the hidden width now), and the staging pass stays in TokenSpeed: moving it into DeepGEMM would change
the API for 1.9 us per layer.
**Bitwise.** Against the unfused HEAD build in one process pair (same inputs): one rank, V4.1 MoE-TP2 shapes
(H 5120, I 1152, 128 experts, top-6), `fast_math` on at 1/2/4/8/16 tokens and `fast_math` off at 1/7/64/300 tokens: every output equal
(`torch.equal`). `test_mega_moe_sm120_ep.py` (GPUs 6+7): EP2 bitwise equal to one rank on all 7 token splits. New in
`test_mega_moe_sm120.py`: tile heights 16-128 give the same bits, and a graph replay equals eager. `DG_JIT_CHECK_NO_SPILLS=1` with a fresh
cache: `test_mega_moe_sm120`, the V4.1-shape run, `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout`, `sm120_blockscale_check`
(ALL OK) all exit clean, no spills.
**Leaf** (graph replay, stage + call, cold weights (512 MB flushed before each call, subtracted), PDL on, best of 5):
| tokens | before us | after us | saved us | x40 layers |
|---|---|---|---|---|
| 1 | 68.1 | 64.4 | 3.7 | 0.15 ms |
| 2 | 107.4 | 101.8 | 5.6 | 0.22 ms |
| 4 | 179.0 | 172.1 | 6.9 | 0.28 ms |
| 8 | 269.7 | 261.7 | 8.0 | 0.32 ms |
| 16 | 500.4 | 487.0 | 13.4 | 0.54 ms |
Kernel times at 1 token: before staging 1.8, dispatch 1.9, gather 3.9, both GEMMs 55.2, SwiGLU 5.2, combine 4.3 (sum 72.6); after
staging 1.9, dispatch + gather 3.6, both GEMMs 60.7, combine 3.1 (sum 69.7). The L1 GEMM got 5.5 us slower (epilogue in the CTA), the three
removed kernels were 11.1 us: the fusion pays 3.7 us net at one token and more as the batch grows (SwiGLU grows with rows). What is left
of the glue per layer is staging 1.9, dispatch + gather 3.6 and combine 3.1-5.9 us (about 0.3 ms per step); the leaf gain cleared the
task's 0.2 ms bar for 2+ tokens per rank (0.22 ms and up) and not at one token (0.15 ms).
**Real V4.1** (128 experts, GPUs 6+7, 9.33's harness, median of 9, three runs per path; moe-backend `auto` = Marlin TP2, `mega_moe` = attention
TP2 + MoE TP2):
| Path | Prefill tok/s (median of each run) | TPOT ms (median of each run) |
|---|---|---|
| Marlin TP2 | 15.08K, 15.11K, 15.09K | 10.57, 10.60, 10.60 |
| MegaMoE before (HEAD DeepGEMM) | 17.03K | 10.60 |
| MegaMoE after (`aa1ecfe`) | 17.05K, 17.06K, 17.06K | 10.49, 10.49, 10.49 |
Prefill 1.13x of Marlin. TPOT is 0.08-0.11 ms (0.8-1.0%) under Marlin, against 0.11 ms gained by the fusion over the unfused kernel, so the
leaf gain carries to the model at about the size the leaf predicts for 2 requests (0.2 ms predicted, 0.11 measured: the gaps between kernels
shrink with PDL, and two ranks wait for the slower one). The per-repetition ranges of the two paths overlap (Marlin 10.08-10.92, MegaMoE
10.30-10.91); the medians of the three runs do not.
**Decision.** `auto` keeps Marlin. MegaMoE now wins prefill clearly (1.13x) and TPOT by 1.0%, which is the 0.1 ms run-to-run spread of the
repetitions, with ranges that overlap; the rule asked for a clear win on both. Selecting a different MoE for every V4 / V4.1 SM120 user
also needs what 9.33 listed and nothing here supplies: a numerics-envelope check (FP8 activations: 0.047 against a float reference, where
Marlin keeps BF16 activations), a width gate (widths 1 and 2 only are tested), batch invariance (the leaf is not), and the persistent
scratch (1.5-1.8 GB at 8192 tokens) planned into the KV budget. Opt-in `--moe-backend mega_moe` is the faster prefill and the same TPOT
or slightly better; no code selects it by default, so no selection test changed.
**Open.** (1) A TPOT win beyond the spread needs the other 0.3 ms: the combine into the L2 epilogue (it is a sum of at most 6 rows per
token at one rank, and the rows are written by different CTAs, so it needs either a second pass or atomics, which would break bitwise
equality) or into the next layer's staging + dispatch; each is 3-6 us per layer. (2) `auto` needs the four gates above before a TPOT win
would be enough. (3) EP2 / DP2 paths use the same fused kernels but were not timed on the real model here.

## 9.39 Task 35: V4.1 indexer: multi-wave sparse MX kernel and MXFP4 paged bandwidth (2026-11-11)

DeepGEMM `b4eaebf` (kernel, scripts) on `9d4cc8e`; TokenSpeed `nctu6` `b5f761118` (rule and tests). RTX PRO 6000, GPUs 6 and 7 (idle when taken, 0% utilization
checked). A second build of the previous commit gave the "before" numbers in the same session. New benchmarks: `scripts/sm120_sparse_paged_bench.py` (the sparse kernel against the
dense paged / contiguous scorers, `--save` / `--cmp` for bitwise comparison) and `scripts/sm120_bench_dsv41_reindex.py` (the leaf's Reindex pass: sparse, dense and
Triton); `sm120_paged_mqa_bench.py` gained `--sequential` and `--sms`.

**Where the 85 us came from.** It was not only "one wave". The kernel followed SM100's per-SM schedule entries, and the metadata cuts a Q block into entries of at
most 8 KV splits (a split is 640 rows of MXFP4): 8 requests x 26 splits made 32 entries, so 32 SMs worked at about 12.5 us per split and 156 idled; 128 requests
made 3 waves of the same 85 us. Per split the CTA copied one token at a time (lanes 0-3 issued the 16-byte `cp.async`, lane 0 loaded the scale word from global
memory first), synchronised the whole CTA, then 8 warps ran the MMAs with N = 64 and the epilogue, with nothing in flight meanwhile: 9.8 us per split at 4096 queries
of one request, where the MMAs alone take 2 us.

**What changed** (`sm120_sparse_mqa_logits.cuh`, one kernel path; the output is bitwise equal to the old kernel in every comparison below):
- *Schedule.* A CTA takes `total / grid` consecutive KV splits straight from the metadata (the splits of a Q block are adjacent), so the work is even to one split at
  any pool size, batch or SM count, and the SM count left the template arguments (no new JIT key per batch). The metadata's own schedule entries are not read, so the
  SM100 metadata kernel and the SM100 sparse kernel are untouched.
- *Gather.* A sparse block is 8 adjacent tokens, 512 bytes of FP4 rows: one `cp.async` per lane copies it, one 4-byte copy per token takes the scale words (zero-filled
  past the end of a contiguous tensor, where the old kernel had no guard), and the page of a block is resolved by shifts.
- *Pipeline.* Two KV buffers (96 KB of shared memory; the FP8 tile does not fit twice and runs one buffer): the gather of split i + 1 is issued between the MMAs of split i, one block
  per 16-row tile. 16 warps = 8 row groups x 2; the two warps of a group each score one of the two queries (N = 32, fewer accumulators, 117 registers) and gather the
  blocks of their group's own rows, so the only barriers are two named barriers of 64 threads per split. A single-token Q block (decode) skips the second query's MMAs.
  The Q tile, scales and weights are TMA'd into two buffers (warp 0, after every warp has read the buffer two blocks back) and each warp keeps its B fragments in
  registers for the whole Q block (`load_block_scaled_b` / `warp_mma_16xNxHd_block_scaled_rs` in `mma/sm120.cuh`, the same bits as the shared-memory version).
  Heads that are not a multiple of 16 (8) keep both queries in one warp of the first half.

**Steps** (4096 queries of one request, 32K history, pool 16384 rows, kernel time in us): old kernel 2767; even schedule + 2-stage gather + 256 threads 1503;
16 warps, one per query 1160; Q fragments in registers 1097; two-shuffle reduction and one store per row 1027; row groups with 64-thread barriers 985; gather
instructions interleaved with the MMAs and shift-based page resolve 913. Ablations of the last version (broadcast, 4096 queries): no epilogue 730, no gather 785, MMAs
only 636; the MMAs alone without any barrier take 442 us (the dense contiguous kernel does the same number of scores in 520 us per 16K scores, 1.06 PFLOP/s), so what
is left is the epilogue (183 us, about 70 instructions per tile and warp by estimate), the gather instructions (128) and the barriers, and they overlap only partly. Not a lever: the
L2 prefetch hint of the `cp.async` (64 vs 256 bytes: same time); more registers (the pair of accumulator sets for a software-pipelined tile would need 133).

**Kernel time** (us, cold L2, kineto; MXFP4, 32 heads, pool 16384 rows, old / new; dense = the whole-history scorer the leaf used instead):
| | history | old | new | dense |
|---|---|---|---|---|
| 8 requests | 32K | 81 | 21 | 26 |
| 32 requests | 32K | 94 | 45 | 78 |
| 128 requests | 16K / 32K / 64K | 259 / 252 / 251 | 145 / 146 / 150 | 144 / 263 / 474 |
| 128 queries of one request | 32K / 64K / 128K | 140 / 148 / 153 | 43 / 46 / 49 | 160 / 314 / 622 |
| 1024 queries of one request | 32K / 64K / 128K | 710 / 734 / 741 | 243 / 251 / 255 | 344 / 672 / 1323 |
| 4096 queries of one request | 32K / 64K / 128K | 2770 / 2864 / 2885 | 912 / 935 / 938 | 1039 / 1964 / 3996 |
Decode is bound by the cache read now (128 x 16K rows: 143 MB in 145 us, 0.98 TB/s, the paged kernel's own rate); one request's chunk is bound by the MMAs.

**Leaf** (`deep_gemm_sm120_dsv41_index_topk`, Reindex pass over a 2048-block pool, CUDA graph with the L2 flushed before each replay, `query_tile` 256 for decode and
1024 for a chunk as the runtime passes them; us; sparse old -> new, dense, Triton):
| pass | history | rows | sparse old -> new | dense | Triton |
|---|---|---|---|---|---|
| decode | 32K | 8 / 32 / 128 | 121 -> 64 / 134 -> 92 / 388 -> 240 | 61 / 114 / 338 | 231 / 447 / 1467 |
| decode | 64K | 8 / 32 / 128 | 122 -> 67 / 135 -> 93 / 388 -> 242 | 78 / 188 / 568 | 230 / 449 / 1478 |
| decode | 128K | 8 / 32 | 123 -> 67 / 136 -> 94 | 111 / 304 | 228 / 449 |
| chunk | 32K | 128 / 512 / 1024 | 205 -> 108 / 598 -> 295 / 1160 -> 560 | 210 / 286 / 862 | 1304 / 4962 / 9871 |
| chunk | 64K | 128 / 512 / 1024 | 262 -> 118 / 711 -> 323 / 1315 -> 608 | 375 / 945 / 1527 | 1309 / 4969 / 9903 |
| chunk | 128K | 128 / 512 / 1024 | 276 -> 123 / 738 -> 337 / 1379 -> 636 | 707 / 2352 / 4028 | 1308 / 4975 / 9927 |
(Decode at 128 rows x 128K was not run: the harness's key scatter overflows 32-bit indexing there, for both builds.) Between one and two pools (17K-24K history) the
dense scorers still win for 8-32 requests (0.77-1.06) and 512-1024 queries (0.74-1.00), and the sparse kernel for 128 requests (0.98-1.27) and 128 queries
(1.36-1.65). The leaf around the kernel is now the larger part at small batches: 8 requests = 2.5 us pack, 8.7 us candidate normalisation, 7.8 us metadata
(SM100's kernel, shared), 16 us sparse kernel, 1.4 us mask, 13.6 us DeepSelect, 3.5 us finish; 1024 queries = metadata 213 us and normalisation 81 us beside the
316 us kernel (the metadata kernel is the next thing to look at).

**`_sparse_reindex_wins`.** Not dropped: dense wins up to 1.3x between one and two pools. It is one threshold now, `capacity >= 2 * pool_rows` (rows and layout no
longer matter): default over old default at the 17 points above 0.95-2.9x (decode 8 x 32K 0.95 and chunk 512 x 32K 0.97 are the two slower ones; the others
1.17-2.9x: for example the 1024-query chunk at 64K 1527 -> 608 us and 512 at 64K 945 -> 322 us, which the old rule left dense). `test_packed_reindex_scores_only_the_pool_from_twice_its_size`
tests the threshold; the exact top-k test gained a chunk past twice the pool, and the recompile guard a shared-table sparse class.

**MXFP4 paged logits, 1.07 vs 1.23 TB/s: closed, nothing shipped.** The two numbers compare different page geometries. FP8 with head dim 64 (`kRowBytes` 64, scales
per token) lays its pages out as 64 x 64 bytes + 64 x 4 bytes, byte for byte the V4.1 page of 4352 bytes, and runs the same kernel at the same rate (128 x 32K:
258 us data-only, 266 us full, 1.07 TB/s of KV); FP8 head dim 128 has 8448-byte pages and reaches 1.27 TB/s data-only, 1.23 full. Nothing in the MX path holds it back.
What was tried at 128 requests x 32K rows (us, MXFP4; 266 is the unchanged kernel):
| change | us |
|---|---|
| 4 KV stages per group (was 3) | 265.7 |
| 1 / 2 KV stages (data-only run) | 263.4 / 261.3 |
| B fragments in registers (`warp_mma_..._rs`; the same trick as the sparse kernel) | 265.5 |
| 128-byte-row, 32-row TMA box with 128-byte swizzle (halves the row requests) | 263.2 |
| no MMA / no logits stores / neither (data path only) | 265.4 / 259.1 / 260.8 |
| neither + no scale TMA (-5.9% of the bytes) | 249.0 |
| 188 / 140 / 94 / 64 SMs (data-only) | 260.7 / 260.1 / 263.8 / 268.2 |
| contiguous pages per request instead of a random permutation | 263.4 |
| TMA L2 promotion none / 64 / 128 / 256 B (data-only) | 261.0 / 260.9 / 260.8 / 261.0 |
The rate does not move with the number of SMs, the pipeline depth, the TMA box, the compute or the page order, so it is a ceiling of the TMA gather at this transfer size,
not occupancy, load width or scale handling; the scale words' separate 256-byte TMA costs its share of the bytes (4.6% of the time for 5.9% of the bytes) and no more.
For scale, a plain `torch.sum` reads 1.52 TB/s and a copy moves 1.46 TB/s read + write, and a gather of random chunks with `index_select` gives (read + write) 1.41 TB/s at 512-2048
bytes, 1.31 at 4352 and 1.36 at 8448 (a throwaway script), so about 3.5 points of the 13% are the 4352-byte stride itself; the rest of the gap to a
copy was not found. Variants not kept: the stage list {3,4}, the register B fragments, the wide box (all within 1% of the unchanged kernel).

**Tests.** DeepGEMM with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh JIT cache (GPU 6): `sm120_sparse_paged_check.py` (28 cases: MXFP8 and MXFP4, 8 and 32 heads, single
requests, decode layout with single-token Q blocks, request pairs, 300 queries over 3 requests, a 128-row page; dequantized-reference diff 1.3e-6 to 1.5e-6),
`sm120_sparse_mqa_check.py` (40 cases, aligned and unaligned `ks`, tails ending inside a block), `sm120_paged_mqa_blockscale_check.py` (51), `sm120_mqa_check.py`,
`sm120_mqa_blockscale_check.py`, `tests/test_attention.py` (159 cases, no spills) all pass; compute-sanitizer memcheck 0 errors on both sparse scripts, racecheck 0
hazards and synccheck 0 errors on the paged one. The new kernel's logits are bitwise equal to the old kernel's in the bench comparisons (per-token and broadcast,
every shape above). TokenSpeed (GPU 7): `test_dsv41_indexer_sm120.py` 13 passed, the kernel subset (gating, registration, selection, the dsv41 files under `nvidia/ops`
and `ops`) 498 passed, 322 skipped; the V4.1 runtime files (`test/runtime/test_deepseek_v41_*.py`) 769 passed, 76 skipped.

## 9.40 Task 36: CuTe DSL V4 kernel that reads the packed rows (dequantizing tile) (2026-11-11)

TokenSpeed `nctu6` from `eea99471a`: `2098a0c41` (kernel, leaves, planner, tests, benchmark), READMEs and the spec log in the docs commit after it. DeepGEMM unchanged (`f8c32b6`,
`build312`). RTX PRO 6000 (188 SMs), kernel benchmarks on GPUs 6 and 7 (idle when taken, one process per GPU), CUDA graph, best of 5, cold caches; scratch scripts were in
`tokenspeed/tmp/t36/` (deleted). `ncu` is still blocked: the evidence is ablation, `cuobjdump` resource usage and timing.

**What the runtime stores and what the workspace route does.** V4.1 stores `swa` rows as FP8 g32 (528 bytes: 512 E4M3 + 16 UE8M0 scales) and `global` rows as FP4 g16 (288
bytes: 256 bytes of E2M1 nibbles + 32 E4M3 scales), page-planar `[pages, 64, row_bytes]` (a page is 64 data rows, then 64 scale rows); V4 decode (`dsv4_decode`) reads FP8
page-planar `[pages, page_size * 584]` (448 E4M3 + 64 BF16 RoPE values + 8 scale bytes, 7 UE8M0 used, per 64 channels). The V4 workspace route (`triton_dsv4_decode`) runs one
Triton program per row (E4M3 * 2^(code - 127) -> BF16 into `[tokens, width, 512]`) and then `dsv4_prefill`, which is the CuTe core above 16 heads; the V4.1 portable route
(`triton_dsv41_selected_attention`) gathers each segment with `cache_gather`, masks, concatenates, builds indices and pins `dsv4_prefill(solution="triton")`; the CuTe
workspace composition was built in the benchmark (gather glue + `cute_dsl_dsv4_mma_prefill`). Both the Triton direct reader (9.24) and the new kernel must equal this route
bit for bit.

**Ceiling first** (workspace route split into its gather + dequantize launches and the CuTe core on the finished workspace, 2048 tokens, 128 window + 512 compressed rows
unless noted; us):
| | full route | gather | core alone | Triton direct | bound on a win (best of the two baselines / core alone) |
|---|---:|---:|---:|---:|---|
| V4, 32 heads | 2241 | 1414 (63%) | 831 | 1199 | 1.44x |
| V4, 64 heads | 2641 | 1414 (54%) | 1248 | 2317 | 1.86x vs Triton, 2.1x vs workspace |
| V4, 128 heads | 3579 | 1414 (40%) | 2179 | 4456 | 1.64x vs workspace |
| V4.1, 32 heads | 3969 | 3142 (79%) | 834 | 1470 | 1.76x vs Triton, 4.8x vs workspace |
| V4.1, 64 heads | 4359 | 3139 (72%) | 1249 | 2887 | 2.3x vs Triton |
| V4.1, 128 heads | 5367 | 3141 (59%) | 2239 | 5734 | 2.4x vs workspace, 2.6x vs Triton |
| V4, 32 heads, 1 token | 15.3 | 3.8 (25%) | 12.1 | 22.3 | 1.26x vs workspace, 1.84x vs Triton |
The gather + dequantize share is 16-65% of the V4 route and 30-80% of the V4.1 one (the glue is part of it) over 32/64/128 heads x widths 128/640/1152 x 1-2048 tokens, and the core
is not gather bound (9.30), so the ceiling is large and the work was built; the gather reads half the bytes a BF16 workspace row costs, so a kernel can even beat "core alone".

**What was built** (`dsa/_cute_dsl/sparse_mma.py` `_SparseMma(packed=...)`, `dsv4/_cute_dsl/sparse_mma.py` `paged_attention` / `supports_paged` / `plan_paged`; leaves
`cute_dsl_dsv4_mma_decode`, `cute_dsl_dsv41_mma_selected_attention`, `PERFORMANT + 1`): one parameter of the same core, not a second path. The producer of the BF16 ring is
the only change. 8 threads per 32-row tile row, thread p owns the 16-byte chunks p + 8j; a thread loads its packed bytes and the row's scales into 20 registers one iteration
ahead (`ld.global.nc` predicated by validity, so nothing branches on the slot id it just read), and after the loop barrier converts them into the stage the previous tile freed
(`st.shared.v4`), then fetches the next row. The conversion is exact and float-free but for the BF16x2 multiply: an E4M3 byte is placed in a BF16 word whose exponent is too
small by 2^120 (so the subnormal codes come out right too), an E2M1 nibble by 2^126 (a `prmt` + shift/mask sequence), and `mul.rn.bf16x2` by that constant and by the group scale
finishes it (one rounding of an exact product = the FP32 product rounded to BF16). The selection is the first cache's columns then the second's, each with its own slots, lengths,
page size (a shift) and row count; a slot that is negative or past its cache and a column past its length are empty rows. Everything after the ring (scores, softmax, PV, chunk fold,
combine, stream-K schedule) is the existing code, so the result is bitwise the workspace route's, batch invariant, and the binary is keyed by the two formats, the id dtype, the
length dtypes, the sink and the head tile only. `plan_paged` is the V4 wave planner with packed tile times (16-head CTA 5.3 + 1.45 us per tile for V4, 6.6 + 1.9 for V4.1; 32-head 8.2 + 2.2
and 10.0 + 2.63; fitted to one token at 1/2/4 splits of widths 128-1152). Layouts the vector loads cannot address (misaligned or non-contiguous-within-page caches, a page size that is not
a power of two, id dtypes that differ) run the portable route (V4) or the Triton direct reader (V4.1); the choice follows the layout alone.

**Numbers** (`bench_dsv4_cute_paged_sm120.py v4|v41 8,16,32,64,128`; 105 points per model: 8/16/32/64/128 heads x widths 128/640/1152 x 1/4/16/64/256/1024/2048 tokens; workspace =
`triton_dsv4_decode` for V4 and gather glue + the CuTe core for V4.1; Triton = the direct readers of 9.24):
| model | heads | vs workspace route | vs Triton direct reader |
|---|---:|---|---|
| V4 | 8 / 16 | 2.31x (1.19-5.14) / 2.20x (1.15-4.98) | 1.81x (1.19-2.20) / 1.73x (1.12-2.23) |
| V4 | 32 / 64 / 128 | 1.83x (1.14-3.26) / 1.45x (1.09-2.14) / 1.23x (1.06-1.48) | 1.67x (1.14-2.16) / 1.62x (1.13-2.23) / 1.71x (1.17-2.14) |
| V4.1 | 8 / 16 | 4.37x (1.98-7.93) / 4.14x (1.99-7.78) | 1.71x (0.95-3.02) / 1.68x (0.95-2.95) |
| V4.1 | 32 / 64 / 128 | 3.39x (1.94-5.49) / 2.48x (1.74-3.46) / 1.95x (1.43-3.35) | 1.58x (0.98-2.22) / 1.52x (0.94-2.24) / 1.61x (0.97-2.18) |
V4 overall: 1.75x geomean over the workspace route (1.06-5.14x) and 1.71x over the Triton reader (1.12-2.23x), a win at every point. V4.1 overall: 3.12x (1.43-7.93x) and 1.62x
(0.94-3.02x). With a compressed cache (widths 640 / 1152) V4.1 is 1.92x over the Triton reader (1.36-3.02x); window-only layers (width 128) are 1.16x (0.94-1.77x: 1.0-1.7x at 16-64 tokens and at 8-16 heads, ties of
0.94-1.06x at 32-128 heads for 1-4 tokens and from 256 (64 and 128 heads) or 1024 tokens (32 heads); 0.4 us of 7 when launch bound). Examples: V4 32 heads, 640 rows, 2048 tokens 740 us against 2236 (workspace) and 1189 (Triton),
below the 831 of the core alone; V4.1 128 heads, 2048 tokens, 640 rows 2.81 ms against 5.40 and 5.83; one token, 640 rows, 32 heads 12.9 us (V4) / 15.1 us (V4.1) against 15.5 / 40.0 and
22.1 / 25.5. Against a grid of plans (112 points, 16-128 heads x 14 token counts of 1-500, width 640, both models) the planner is 1.007x behind on average, 6 points more than 1.05x, worst
1.12x (V4.1, 128 heads, 24 tokens). The version before the planner constants and the branch-free fetch was slower by 1.04-1.23x on the same points (the first one branched on the
slot id it had just loaded and around the row loads).

**Selection** (head count and format only, so a row never depends on the batch): `cute_dsl_dsv4_mma_decode` and `cute_dsl_dsv41_mma_selected_attention` are registered for every
head count and both caches; the Triton readers stay the fallback where the CuTe DSL is not built or a layout is unreadable. V4.1 window-only layers go through the CuTe kernel too: it
ties or wins (geomean 1.16x), and one rule for the model is simpler than a split at 0.94-1.06x (a rule "CuTe from 17 heads" or "from a compressed cache" would be a coin flip at 8 points of 35).
16 heads, which 9.35 closed on Triton for the workspace gather, are a win here for the same reason as everywhere else: the rows read are 0.57x the bytes (584 against 1024 per V4 row).

**Ruled out / limits** (don't retry without new evidence): (1) moving the dequantize + fetch after the PV MMAs so `ptxas` interleaves them with the chain: 1.1-1.2x slower (V4 128 heads,
2048 tokens 2884 vs 2512 us; V4.1 128 heads 3387 vs 2812). (2) Raw staging of the packed bytes in shared memory (`cp.async`): the 32-head ring (2 x 33 KB) + exchange (16 KB) leave
16 KB and a V4 tile is 18.7 KB; the 16-head CTA fits one stage only. (3) Fusing the two multiplies into one: only valid for scale codes up to 134 (the product 2^(code - 7) must fit
BF16), so a rare branch would be needed; not built. (4) The 32-head packed CTA spills (V4 168 B, V4.1 120 B of stack at 255 registers; the 16-head CTA uses 224-244 registers with no stack):
the 20 registers of the next tile cost 32-head tile times of 2.2-2.6 us against 1.9 us for the workspace core; the planner prefers 16-head CTAs where the model says so. Open: cut the
raw registers (stage the scales and RoPE tail through shared memory for the 16-head CTA only is not enough for 32 heads), or a key-split variant for the BF16 core (9.30). (5) Ablation
by skipping the conversion is not clean (garbage BF16 changes the softmax work); the ALU share was inferred from the fitted tile times instead (V4 +0.01 us per 16-head tile against the
workspace core, V4.1 FP4 +0.45 us).

**Tests.** `test_dsv4_cute_paged_sm120.py` (56): both leaves bit-equal to the workspace route (heads 8-128, 1-37 tokens, widths with holes, out-of-cache slots, short lengths, int32/int64 ids, a window
only layer, the no-sink LSE, strided page views, a misaligned cache and a page size of 48 falling back); three read-back tests that attend one live key of a zero query and compare the BF16
output with a PyTorch reference for every E4M3 byte x every UE8M0 scale code (subnormal 2^-127 included) in `fp8_g32` and `fp8_v4` and every nibble x every E4M3 scale in `fp4_g16`; rows and LSE
bitwise equal over every head tile, warp layout, split count and `full_tokens`; batch invariance through the leaf; no recompilation over a sweep of token counts, widths and splits. Selection
test and the V4.1 `select_kernel` expectation in the runtime test follow the leaf. Kernel suite (task 8 command, GPU 7): 8009 passed, 2764 skipped, 0 failed; runtime suite (GPU 6): 9007 passed,
0 failed (cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 374, models 567, moe 72, sampling 144, top 6787); `test_deepseek_v41_model.py` (the 40-layer real-FlatKV
decode among its cases) selects the `(fp8_g32, fp4_g16)` and `(fp8_g32, None)` binaries; pre-commit clean on the changed files. DeepGEMM was not changed, so its surface was not rerun.

**Still open:** the 32-head packed CTA's spills (20 live registers for the next tile), V4.1 window-only layers at 32-128 heads for 1024+ tokens (a tie with the Triton reader at 0.97-1.06x),
BF16 DSA / V4 at 42-48% of the floor (9.30).

## 9.41 Task 37: host overhead of a 128-token DSA chunk, `fp8_mqa_logits` and DeepSelect (2026-11-11)
TokenSpeed `nctu6` (local, not pushed): `436833133` (DeepSelect and leaf host path, test), README commit, pin commit after the DeepGEMM docs. DeepGEMM `nctu6`: `f628449` (kernel lookup, view, env reads, test), `6df2878` + `211a7e9` (benchmark script). RTX PRO 6000, GPU 7 for timings (eager, back-to-back calls, the GPU never waits for the host here), GPU 6 for tests. Scripts: `scripts/sm120_bench_host_dsa.py` (call level and leaf level, `--leaf`, `--profile` for cProfile).
**Method.** CPU time of each call with `perf_counter_ns` around it and no synchronization inside the timed region (a sync every 25 calls outside it), medians of 3000-5000 calls; the leaf run times the components by wrapping them (`dsa_prefill_prepare`, `_resolve_prefill_tile_max_seqlen_k`, `fp8_mqa_logits`, `mark_forced_initial_local_logits`, `deepselect_supports`, `deepselect_local_topk`), so the numbers include the leaf's cold-cache effects. 128 tokens, 64 index heads, no prefix, unless said. "Before" is TokenSpeed `HEAD` of task 36 (`ts_prev`) with the DeepGEMM build of `00dbf6e`; "after" is the tree with the two commits above. A temporary build with `steady_clock` marks (reverted) split the C++ call.
**Where the time was.**
- `fp8_mqa_logits` (8.5-9.0 us hot, 12.8 us inside the leaf). `jit->compile` runs on every launch: it copies the compiler options (two vectors of strings), formats and joins about 15 nvcc flags, hashes flags and source byte by byte, scans the source for includes (the include hashes are cached) and builds a hex digest, ~3 us, all to find a kernel that is already loaded. The view was `logits.index({Slice, Slice})` (two slices through `at::index`), and each of the four tensor-map builders read `DG_PRINT_CONFIGS` and `DG_TMA_L2_PROMOTION` through `deep_jit::get_env`, which builds a `std::string` per read (16 characters: a heap allocation) before `getenv`. After the change the C++ call is 5.6 us: checks 0.1, `torch::empty` + `as_strided` 1.3, the four `cuTensorMapEncodeTiled` 0.5 together (not worth caching: the address cannot be patched into an opaque `CUtensorMap`, and the encode is cheap), kernel lookup and shared-memory sum 0.2, launch 2.6 (`cuFuncSetAttribute`, `getCurrentCUDAStream`, `cuLaunchKernelEx` in `deep_jit`), pybind and the Python wrapper ~0.9.
- DeepSelect (17.2 us hot, 25.3 inside the leaf). cProfile over 2000 calls: the compiled TVM-FFI call 3.5 us (one DLPack conversion per tensor, five tensors), `torch.empty` for the `values` output 2.0, `deepselect_topk`'s own Python 3.5 (about 30 attribute reads for the checks, `cluster_sizes_for` built with a generator three times per call, `current_platform().sm_count`), `deepselect_local_topk`'s own Python 2.0 plus four slices of tensors that were already whole (1.1 us each) and a `lengths.to(torch.int32).contiguous()` (0.9 us).
- The leaf around them: the tile slices and `.contiguous()` of `q_fp8`, the weights, `row_starts`, `row_ends` (4.6 us hot), `candidate_lens[start:end]` twice, `out[start:end]`, `row_starts[start:end]`, `.to(...)` of the inputs, and `candidate_lens_cpu[start:end].max().item()` (5.6 us in the leaf). These are the rest of the 26 us (`.max().item()` 5.6 + call 12.8 + slicing and the PDL check ~7) and 34 us (`deepselect_supports` 2.3 + call 25.3 + slicing ~6) of 9.24.
**What was cut** (results bit-identical):
- DeepGEMM `kernel_lookup.hpp`: `get_cached_kernel(name, key, make_source)` keeps the loaded kernel in a per-call-site map keyed by the integers that fill the template (heads, head dim, block sizes, SM count, dtypes); the source is built and `jit->compile`d on the first call of a configuration only. Used by the contiguous FP8 launcher and the block-scaled one (`sm120_fp8_mqa_logits`, `sm120_fp8_fp4_mqa_logits`). The generated source text is unchanged, so JIT cache entries stay valid. `get_env_int_fast` reads the two variables with `getenv` on the literal and delegates to `deep_jit::get_env` only when set (same parsing and errors). The logits view is one `as_strided`.
- TokenSpeed: `deepselect_local_topk` passes whole tensors when one launch covers the rows, converts `lengths` only when its dtype or layout needs it, and asks for the SM count only for rows of 32768+ columns (`choose_cluster_size` returns 1 below). `deepselect_topk` reads `shape` and `stride()` once, validates with one helper for `indices` / `values`, and `capacity_for` / `cluster_sizes_for` are cached. The DeepGEMM leaf takes whole tensors for a tile that covers the chunk (`_rows`, `_tile_rows`), converts inputs only when they differ (`_dense`), and takes `max()` of the whole CPU mirror without a slice.
**Numbers** (before -> after, us):
| call | hot loop | inside the leaf |
|---|---:|---:|
| `deep_gemm.fp8_mqa_logits` (128 tokens, 64 heads) | 8.5 -> 5.6 | 12.8 -> 8.3 |
| same, 1 / 128 on a 2048 prefix / 1024 tokens / 32 heads | 9.0 -> 5.6 / 8.6 -> 5.7 / 8.2 -> 5.7 / 8.7 -> 5.7 | 13.5 -> 8.8 / 12.8 -> 8.5 / 12.8 -> 8.3 / 13.4 -> 8.3 |
| `deepselect_local_topk` with `index_base` | 17.1 -> 9.2 (17.3 -> 9.7 on a prefix, 17.2 -> 9.7 at 1024 tokens) | 25.3 -> 14.2 |
| `deepselect_supports` | 1.3 -> 1.3 | 2.3 -> 2.0 |
| `max_seqlen_k` (`.item()` of the CPU mirror) | | 5.6 -> 4.2 |
| mqa path (`.item()` + slices + PDL check + call), estimated | | ~26 -> ~14.5 |
| DeepSelect path (supports + slices + call), estimated | | ~34 -> ~16.5 |
| whole leaf, host median | | 115.9 -> 86.6 (1 token 114.9 -> 86.5; 1024 tokens 115.0 -> 87.5; 32 heads 120.1 -> 86.4; 16 heads 116.0 -> 88.1; 128 on a 2048 prefix 116.3 -> 89.1) |
| whole leaf, back-to-back eager call (CUDA events) | | 119.0 -> 88.8 (1.34x; the Triton leaf takes 168 us, 9.24) |
The leaf's other components did not move: `dsa_prefill_prepare` 38.1 -> 37.1 (a Triton launch), `mark_forced_initial_local_logits` 1.5. What is left of the 87 us: prepare 37, `nan_to_num_` ~6 (an aten launch), DeepSelect 14, `fp8_mqa_logits` 8, `.item()` 4, `_check_out` and the output allocations ~6, input checks and loop ~8.
**Bitwise equality.** `equal.py` (44 cases, old tree + old build against new tree + new build): the leaf's `out` and `lens_out` for 16 / 32 / 64 heads x tokens 1 / 17 / 128 / 129 / 513 / 1100 x prefixes 0 / 700 / 2048, each with a one-tile budget and a 3-key-row budget that forces several tiles; raw `fp8_mqa_logits` outputs (the valid entries as int32 bits, shape and stride) for 32 / 64 heads over four shapes. 0 cases differ.
**End to end** (4-layer `DeepseekV32ForCausalLM`, real dimensions (hidden 7168, 128 heads, indexer 64 x 128, top-k 2048), 16 experts top 4, `load_format="dummy"`, in-process `Engine`, `chunked_prefill_size` 128, batch 1, new random tokens per request so nothing hits the prefix cache, `max_new_tokens` 1, wall time of `engine.generate`, three runs before and two after, interleaved):
| prompt (chunks of 128) | before tok/s (median ms) | after tok/s (median ms) |
|---|---|---|
| 128 (1 chunk) | 8202 (15.61), 8090 (15.82), 8122 (15.76) | 8136 (15.73), 7947 (16.11) |
| 512 (4) | 10689 (47.90), 10656 (48.05), 10628 (48.18) | 10638 (48.13), 10608 (48.27) |
| 2048 (16) | 11569 (177.0), 11440 (179.0), 11463 (178.7) | 11463 (178.7), 11382 (179.9) |
No measurable difference (the runs of one tree differ from each other by up to 1.4%, and the single-chunk prompt by 3%). A 128-token chunk of this model takes ~11 ms (4 layers: the MoE and dense weights read per chunk, 128-head attention, scheduler and IPC), the leaf's host time is 4 x 87 us and was 4 x 116 us, and the forward is launched ahead of the GPU, so 0.12 ms would be 1% of a chunk even if fully exposed. The saving shows where the host is the bottleneck: the leaf alone, back-to-back (1.34x), and a 61-layer model would save 1.8 ms of host time per chunk (7.1 -> 5.3 ms), which matters only where the GPU is not the limit.
**Ruled out / limits** (don't retry without new evidence): (1) caching tensor maps (`cuTensorMapEncodeTiled` is 0.13 us each; an encoded map cannot have its address patched, and the pointers change per call); (2) a persistent `values` scratch for DeepSelect (2.0 us per call, but a captured graph would keep the pointer of a scratch that growth freed, a scratch big enough for 65535 rows x 2048 is 512 MB, and concurrent streams would share it); dropping the `values` store needs another compile variant of the kernel; (3) caching `candidate_lens_cpu.max()` by tensor identity and `_version` (a NumPy alias writes without bumping the version; the 3 us does not justify a stale width); (4) fusing `nan_to_num_` into DeepSelect (it maps NaN and +/-inf to -inf, DeepSelect drops NaN slots instead, so selections would differ). `cuFuncSetAttribute` per launch lives in `deep_jit` (a separate repository, ~0.3 us); not touched. Other launchers still call `jit->compile` per call (their ~3 us can be removed with one `get_cached_kernel` each; the contiguous MQA logits was the one on this path).
**Tests.** DeepGEMM `tests/test_mqa_logits_launch.py` (interleaved 32 / 64 heads x four shapes, 3 rounds, against a dequantized reference, view layout); `test_attention.py` (whole, `DG_JIT_CHECK_NO_SPILLS=1`, fresh JIT cache), `sm120_mqa_check.py` (46 cases), `sm120_mqa_blockscale_check.py`, `sm120_paged_mqa_blockscale_check.py` ALL OK. TokenSpeed `test_logits_budget_tiles_select_what_one_call_selects` (a budget that tiles the chunk selects what one call selects; 32 / 64 heads); `test_dsa_prefill_prepare_sm120.py`, `test_deep_select_cute_dsl.py`, `test_dsa_indexer_sm120.py`, `test_dsa_topk_batch_invariant.py`: 381 passed, 162 skipped. The shared tensor-map builders changed too (env reads), so the rest of the SM120 surface ran with `DG_JIT_CHECK_NO_SPILLS=1` and a fresh JIT cache after `sm120_remote.sh build` ("build OK", py3.15 and py3.12): `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection` exit 0. TokenSpeed also: `test_dsv41_indexer_sm120.py`, `test_dsv41_local_indexer.py`, `test_flashmla_dsa_dcp.py`, `test_kernel_api_selection.py`, `test_dsv41_registration.py` 278 passed (272 skipped); runtime `test_dots3_note_model.py`, `test_dots3_note_nextn.py`, `test_glm53_flash_*`, `test_dsa_*` 80 passed. pre-commit on the four changed Python files: black reformatted one line of `deep_gemm.py` (kept), the rest clean; the full kernel and runtime suites were not rerun (the change is host-only and the affected files above pass).
**Still open:** the TVM-FFI call and the `values` allocation (5.5 of DeepSelect's 9 us); `dsa_prefill_prepare` (37 us: the Triton launcher and five `torch.empty` calls around one kernel); `nan_to_num_` (6 us); the other `jit->compile` callers.

## 9.42 Task 38: the three heavy compute-sanitizer runs, finished (2026-11-11)

DeepGEMM `nctu6` (local, not pushed): `0d89c76` (two test changes), this docs commit. No kernel changed, so no rebuild was needed (the remote tree equals the local one, "build OK" at the start). compute-sanitizer 2026.1.1.0 (CUDA 13.2), RTX PRO 6000 on GPU 6, all runs inside the repo `tmp/`. The three tests that did not finish in 15-40 min in task 11.5 / report 7 (FP8 k-grouped, FP8 masked, `test_mqa_logits`) now run to the end under memcheck, synccheck, racecheck and initcheck. **No kernel error.** One test-harness read of unwritten memory (fixed in the test) and one by-design read of PSUM padding rows (classified, guarded by a new test).

**Why they did not finish.** `test_sanitizer.py` instruments every kernel, including the torch kernels that build the inputs and the FP32 reference GEMMs (`einsum` over 32 groups x 4096 x 6144 x 7168 for the masked test, an 8192 x 65536 x 64-head score for MQA), and those are hundreds of times slower under the tools. The DeepGEMM kernels themselves are cheap: with `--kernel-name kns=sm120_ --kernel-name kns=pack_fp32_into_ue8m0` (the GEMM / MQA kernels and the `pack` / `transpose_and_pack` layout kernels) memcheck takes about 9 s for the heaviest k-grouped case (1.9e12 FLOP).

**Method.**
- A scratch driver (`tmp/san_driver.py`, deleted) imports the three test modules and calls the unmodified test body once per case: the case list comes from the test's own enumerator (`enumerate_m_grouped_masked`, `enumerate_k_grouped_contiguous`, the list inside `test_mqa_logits`), every case has a fixed seed, `bench_kineto` returns a constant (no timing launches) and the `range(20)` / `range(8)` repeat loops are cut to 2. Everything else stays: the reference, the quantization, the self-consistency launches, the FP4-vs-FP8 comparison, the empty-group and single-group variants, the pre-packed scale launch. A CUDA out-of-memory (the GPU is shared) is logged and the case skipped.
- JIT pre-compiled by an unsanitized pass over the full lists (fresh `DG_JIT_CACHE_DIR`, `DG_JIT_DEBUG=1`); the "Loading CUBIN" lines give, for every run, the set of kernel binaries it launched, compared with that pass.
- Tools: `compute-sanitizer --tool {memcheck,synccheck,racecheck,initcheck} --target-processes all --destroy-on-device-error context --check-api-memory-access no --print-limit 100`, `--racecheck-report=all` (the `=` is needed), the environment of `test_sanitizer.py` (`CUDA_LAUNCH_BLOCKING=1`, `DG_JIT_CHECK_NO_SPILLS=1`, `DG_USE_NVIDIA_TOOLS=1`, `DG_USE_TEMP_CUBLASLT_WORKSPACE=1`, `PYTORCH_NO_CUDA_MEMORY_CACHING=1` so every tensor is its own `cudaMalloc`). memcheck / synccheck / racecheck use the kernel filter above. initcheck cannot: with the filter, the torch kernels that write the inputs are not instrumented, so their output looks uninitialized (100 false reports on the first `pack_fp32_into_ue8m0` of a trial); it runs unfiltered, one process per case. `--track-unused-memory` is a flag without argument and off by default.
- Positive control: an `fp8_fp4_gemm_nt` whose output view spans twice its allocation. With the filter on, memcheck reports `Invalid __global__ write of size 8` in `sm120_fp8_gemm_1d2d_impl`, 65,601 bytes after the allocation; with a filter that matches nothing the run dies with an illegal address.

**Configurations that ran** (all full lists; the lists are the tests' own):
- *FP8 masked* (`test_m_grouped_gemm_masked`), 64 cases: FP32-scale FP8 (128x128 B scales) and W4A8 (FP8 A, MXFP4 B, UE8M0 scales) x masked layout and PSUM contiguous layout x (groups, expected m per group) (32, 192), (6, 1024), (32, 20), (6, 20) x (n, k) (6144, 7168), (7168, 3072), (4096, 4096), (4096, 2048), max m 4096. The test also launches the FP4 operands converted to FP8 (the plain FP8 1D1D binary). 91 kernel binaries (masked and contiguous GEMMs, `pack` / `transpose_and_pack`).
- *FP8 k-grouped* (`test_k_grouped_gemm_contiguous`, `k_grouped_fp8_gemm_tn_contiguous`, MN-major A and B), 126 cases: 7 shapes (groups, m, n, average K per group) (8, 768, 2048, 128), (4, 4096, 7168, 8192), (4, 7168, 2048, 8192), (8, 4096, 7168, 4096), (8, 7168, 2048, 4096), (16, 4096, 7168, 2048), (16, 7168, 2048, 2048) x scale layouts (gran_k, k_alignment) (32, 128), (128, 128), (32, 256), (128, 256), (32, 384), (128, 384) x (accumulate FP32, direct FP32, direct BF16); each case with all groups, about a quarter of the groups empty, and one group; the pre-packed UE8M0 launch for gran_k 32. 23 binaries.
- *`test_mqa_logits`*, 64 cases: FP8 (H 32 / 64 x D 64 / 128 x sq 2048 / 8192 x sk 8192 / 65536 x CP on / off, 32 cases), MXFP4 and MXFP8 (H 32 / 64, D 128, 16 cases each). 8 binaries (`sm120_fp8_mqa_logits` x 4, `sm120_fp8_fp4_mqa_logits` x 4).

| tool | FP8 masked | FP8 k-grouped | `test_mqa_logits` |
|---|---|---|---|
| memcheck | 64 / 64 cases, 91 / 91 binaries, 0 kernel errors | 126 / 126, 23 / 23, 0 | 64 / 64, 8 / 8, 0 |
| synccheck | 64 / 64, 91 / 91, 0 | 126 / 126, 23 / 23, 0 | 64 / 64, 8 / 8, 0 |
| racecheck (`--racecheck-report=all`) | 49 full test bodies + 9 single launches of the large cases, 91 / 91, 0 hazards | 19 + 12 single launches, 23 / 23, 0 hazards | 22 (all 16 selected), 8 / 8, 0 hazards |
| initcheck | 55 cases (cheapest case of every binary and variant), 91 / 91, 0 once the PSUM padding is filled (below) | 19 cases, 23 / 23, 0 | 16 cases, 8 / 8, 0 (after the test fix below) |

Wall time (sanitizer time of the whole run, several processes sharing GPU 6 at times): memcheck 11 min masked (150 s, then 520 s for the 18 cases it had to skip), 14 min k-grouped, 1.5 h MQA; synccheck 9 min, 13 min, 57 min; racecheck about 2.7 h (masked full bodies), 1.9 h (k-grouped), 1.7 h (MQA) plus 4.2 h of single launches over 6 parallel workers; initcheck 1.7 h in total over one-process-per-case runs. About 7.5 h of wall clock including the plan pass and the diagnosis. The racecheck full-body runs of the large cases were stopped (the 9.5e11-FLOP k-grouped case 36 alone took 41 min; the 1.9e12 ones would take about twice that), and those cases ran as one launch of the kernel path each: `generate_*` of the test, one GEMM call (plus the FP8-converted call for W4A8), no reference, no repeats. A first version of that single-launch mode missed the FP4-converted FP8 binary of four W4A8 masked cases; the binary-set check caught it and those four were rerun with the converted launch. Each run's launched binary set equals the plan's set for every tool and test.

**Findings.**
1. **Masked: 18 cases out of memory (not a finding).** Another user's vLLM held 88.6 GB of GPU 6 for about 20 min, during the first memcheck run; 18 large cases failed in `cudaMalloc`, which memcheck counts as 36 API errors (`cudaMalloc` + `cudaGetLastError`). No kernel error was reported; the 18 cases were rerun after the job left: 0 errors. (GPU 7 was held by another job at 100% for the whole task.)
2. **`test_mqa_logits`, initcheck: the test reads unwritten padding (fixed in the test).** The kernel returns `[seq_len, max_seqlen_k]` and defines only the first `ke - ks` columns of a row. The test compared `logits.masked_fill(~self_mask, 0)`, which copies the whole tensor first, so an elementwise copy kernel (`direct_copy_kernel_cuda`) read the never-written padding: 269,003 reports in the first case. Attribution by a probe (`tmp/san_mqa_probe.py`, deleted): inputs + kernel only 0 errors; plus a boolean-index read of the valid columns only (8,387,584 values) 0 errors; plus `masked_fill` 375,800 errors, all in the copy kernel. The kernel writes every defined element. `tests/test_attention.py` now compares `logits[self_mask]` (a boolean index reads only the valid entries) for the self-consistency loop and the SM100 scheduled path; 16 cases of initcheck: 0 errors, and the whole `test_attention.py` passes (64 prefill + 95 paged cases, `DG_JIT_CHECK_NO_SPILLS=1`).
3. **Masked PSUM layout, initcheck: padding rows are read (by design, now guarded by a test).** `generate_m_grouped_masked` leaves the rows between PSUM groups (up to the MK alignment) uninitialized on purpose, "to verify the pack kernel skips them". Two kernels read them: `transpose_and_pack_fp32_into_ue8m0` (`smxx_layout.cuh:141`) loads whole rows of FP32 scales into shared memory, then writes 0 for rows outside every group (`is_valid_mn`), and `sm120_fp8_gemm_1d2d_impl` (`sm120_fp8_gemm_1d2d.cuh:333`) loads the A scales of every row of a warp tile (clamped to `shape_m - 1` only), so the accumulators of padding rows are garbage and those rows are never stored as results. Rows are independent, so valid rows cannot depend on them. Evidence: (a) with the padding zero-filled (`layout_masked_to_psum` patched in the driver) initcheck reports 0 on all 55 selected masked cases, PSUM ones included (before: 84,480 reports in one W4A8 case, and the FP32-scale case hit the 900 s timeout; with zeros it takes 13 s), so nothing else reads uninitialized memory; (b) with the padding filled with NaN (FP32 / FP8) or -1 bytes (FP4 data) all 64 masked cases pass the test's own checks (reference, 20-repeat determinism, FP4 vs FP8 equality); (c) new `test_psum_gaps_do_not_reach_valid_rows_sm120`: four configurations (FP32-scale and W4A8, 6 groups x 20 rows and 4 groups x 192 rows, n 4096, k 2048 / 4096), the padding of A and of its scales zero-filled vs NaN (0x7f for FP8): the valid rows are bitwise equal and within `max_diff` of the reference; 263 of 647 padding rows come out NaN, so the poison does flow through the kernel and stops at the row. The new test is clean under memcheck, synccheck, racecheck and initcheck (it fills the padding itself). A kernel change (clamping the scale row to the group's last valid row) would add integer work to the K loop of the hot kernel for rows nobody reads, so none was made; the generator keeps its uninitialized padding because that is the stronger input for the pack kernel.
4. **Noise.** The `cuGetProcAddress` version probes of 9.31 did not appear in these runs (`--check-api-memory-access no`). The `test_legacy` entry of `test_sanitizer.py` still fails without triton in the venv (unchanged, environment).

**Not done / limits.** (1) The sanitizers saw the kernels the tests launch: the 91 + 23 + 8 binaries above, not other shapes. The kernel filter leaves the torch kernels (and cuBLASLt) unchecked in memcheck / synccheck / racecheck by design; initcheck sees them all. (2) racecheck covers shared-memory hazards only, and the mbarrier / TMA-bulk synchronization the kernels use is outside what it models; the full-body racecheck runs of the large cases were replaced by one launch each (the kernel code path is the same, the repeats and variants are not). (3) The repeat loops of the tests were cut from 20 to 2 (and 8 inputs to 2 for the masked test) under the tools. (4) `--racecheck-report=all` printed no analysis report either. (5) The driver, the probe and the logs were scratch and are deleted; the commands above reproduce them: a driver of a few hundred lines that patches `bench_kineto`, the repeat `range` and the enumerator of each test module.

**Tests.** `tests/test_attention.py` (whole, 64 + 95 cases; the valid-entry comparison), `tests/test_fp8_fp4.py::test_psum_gaps_do_not_reach_valid_rows_sm120` (new, SM120 only, 4 configurations, also run under the four tools: 0 errors). Both are in the commit `0d89c76`.

## 9.43 Task 39: final verification of tasks 34-38 (2026-11-11)
Code heads: DeepGEMM `0d89c76` (tests; the kernels are `aa1ecfe` MegaMoE fusion, `b4eaebf` sparse MX kernel and `f628449` kernel lookup; docs-only commits after it), TokenSpeed `ad944dc2e` (pin on the DeepGEMM code head `1df3e7f`; the code is task 37's `436833133`). The remote DeepGEMM tree was synced from the tracked files of the local head (`git ls-files csrc deep_gemm tests scripts`; a dry-run `rsync --delete` listed only the old `csrc/jit_kernels/impls/sm120_sparse_mqa_logits.cuh` leftover, which nothing includes, and the `include/cutlass` / `include/cute` links) and `DG_GPU=7 bash scripts/sm120_remote.sh build` printed "build OK" (py3.15 and py3.12). The `/tmp/ts_mac` mirror was resynced from the local tree. GPUs 6 and 7 only, checked with `nvidia-smi` before each run (both idle, 0 MiB, at every start; no run was hit by another user's job), all DeepGEMM JIT caches fresh, `DG_JIT_CHECK_NO_SPILLS=1`. The kernel suite ran with `PATH=/usr/local/cuda/bin:/workspace/nctu6/infra/tokenspeed/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin`.
| check | result |
|---|---|
| TokenSpeed kernel suite (`test_dsv41_registration.py test_kernel_api_selection.py test_sm120_excludes_sm100_only_leaves.py test_residual_mhc_arch_window.py test_deepgemm_sm120_gating.py nvidia ops`, GPU 7) | 8011 passed, 2764 skipped, 0 failed (task 36: 8009 / 2764; the two extra cases are task 37's `test_logits_budget_tiles_select_what_one_call_selects` and its parameterization) |
| `test_numerics.py` + `test_triton_adapter.py` | 24 passed, 2 skipped (empty parameter sets) |
| Two visible GPUs: `nvidia/ops/communication` (6+7) | 155 passed, 41 skipped (tp4 / tp8 skip: 4 and 8 NVLink GPUs required) |
| `test_mega_moe_sm120_token_layouts.py` (`torchrun --standalone --nproc-per-node=2 -m pytest`, GPUs 6+7) | 2 passed (one per rank), twice |
| TokenSpeed runtime suite (GPU 6, all ten directories) | 9007 passed, 0 failed: cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 374, models 567, moe 72, sampling 144, top-level files 6787 |
| DeepGEMM `test_fp8_fp4` (including `test_psum_gaps_do_not_reach_valid_rows_sm120`), `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention`, `test_mqa_logits_launch`, `test_mega_moe_sm120` | all exit 0, no spills |
| `sm120_mqa_check.py` (46 cases), `sm120_blockscale_check.py --mode all --grouped` (170), `sm120_paged_mqa_blockscale_check.py` (51), `sm120_sparse_paged_check.py` (28), `sm120_sparse_mqa_check.py` (40) | all OK |
| `test_mega_moe_sm120_ep.py` (`torchrun --nproc_per_node 2`, GPUs 6+7) | bitwise equal to one rank for all 7 token splits, up to 2.2e-11 against the PyTorch replay |
| `test_lazy_init.py --num-processes 2` (GPUs 6+7) | 3 tests OK |
| End to end, in-process, greedy | Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2, GPUs 6+7), Qwen3.8-27B-NVFP4 and gemma-4-31B-it (`<bos>` prefix) all answer "Paris" and "5, 7, ..." for the two prompts |
| Lint | see below |
No regression, so no code change in either repo. The only new checks since task 33 are the tests of tasks 34-38 (MegaMoE fusion bitwise tests, the sparse MX checks, `test_mqa_logits_launch.py`, the PSUM-gap test, the V4.1 indexer and CuTe paged-decode kernel tests), and all of them are inside the runs above.
**Lint.** `SKIP=no-commit-to-branch pre-commit run --all-files` (clang-format included) in a clean scratch repo made from `git archive` of the local TokenSpeed HEAD, run twice: both exit 0 with no file changed, so there is no formatting commit. The other steps of `.github/workflows/lint.yml` on the same tree: `ruff check --select F401,F841,F821,F823,F811,F541,TID251,UP035,UP006,UP007,UP045 python/tokenspeed/runtime` and `ruff check tokenspeed-scheduler` (ruff 0.17.4 from the scratch venv of task 28; the TokenSpeed venv has no ruff), the file-start / relative-import check of `tokenspeed-scheduler`, and `test/ci_system/test_weekly_release.py` (27 passed). Not run: scheduler clang-tidy and the C++ build (no ninja on the box), and the CI's own ruff version.
**Open items after this run** (the same list as report 7 and the Open list in "Current state"): the dense gate_up M=256 FP4 tie (9.28); the MegaMoE combine + staging (0.3 ms per step) and the `auto` gates (9.38); the SM100 sparse metadata kernel in the V4.1 Reindex leaf (9.39); BF16 DSA / V4 at 42-48% of the tensor-pipe floor, V4 at 16 heads, one-request DSA decode, the 32-head packed CTA spill and the V4.1 window-only ties (9.30, 9.34, 9.35, 9.40); the DeepSelect / `dsa_prefill_prepare` / `nan_to_num_` host path (9.41); V3.2 TP8 and V4.1 with 384 experts; the limits of the sanitizer runs (9.42); `test_projection_tp` tp4 / tp8. Tasks 34-39 are local; nothing is pushed.

## 9.44 Task 47: image benchmark, before vs after the SM120 work (2026-11-12)

Full report with every table: [`deepgemm_benchmark.md`](deepgemm_benchmark.md). The harness (tasks 40-46) is in `tokenspeed/tmp/bench_images_local/` (git-ignored; live copy and raw results on aws-raichu in `/workspace/nctu6/infra/tokenspeed/tmp/bench_images/`).

Setup: `nctu6/tokenspeed:e27cee86-20261028` (before, TokenSpeed `e27cee861`) against `nctu6/tokenspeed:ef057889-20261112` (after, `ef057889f`, DeepGEMM pin `976f212`); RTX PRO 6000 on GPUs 6 and 7 of aws-raichu; one container per (image, model) with `--max-num-seqs 512 --gpu-memory-utilization 0.9 --max-model-len 262144`, TP1 or TP2, identical flags in both images; RAG (`neural-bridge/rag-dataset-12000`, 500 prompts, 256 output tokens) at C = 1, 16, 32, 64, 128, 256, 512 and AIPerf `inferencex-agentx-mvp` in official full mode (900 s, `submission_valid=true`) at C = 1, 16, 64. All primary results run with `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` in both images (below).

| Model | Result, after over before (RAG output tok/s, C = 1 / 16 / 64 / 256 / 512) |
|---|---|
| Qwen3.8-27B-FP8 (TP1) | 1.52x / 1.28x / 1.11x / 1.04x / 1.04x; TPOT p50 1.53x at C=1, TTFT p50 1.00x; agentX 1.14x / 1.10x / 1.01x |
| Qwen3.6-35B-A3B (TP1, BF16 control) | 1.01x / 1.01x / 1.00x / 1.00x / 1.00x; agentX 1.02x / 0.94x / 1.03x (TPOT 1.04-1.06x) |
| gemma-4-31B-it (TP1, BF16) | 1.03x / 1.03x / 1.00x / 1.02x / 1.03x; agentX C=1 1.00x, C=16 / 64 not runnable |
| Qwen3.8-27B-NVFP4 (TP1) | before fails to start; after 55 / 618 / 1321 / 1610 / 1612 tok/s |
| Qwen3.5-122B-A10B-NVFP4 (TP2) | before fails to start; after 133 / 803 / 1480 / 2282 / 2256 tok/s |
| V4.1-Flash, 128 of 384 experts (TP2) | before fails to load; after (Marlin) 105 / 613 / 1429 / 1650 / 2693 tok/s; `mega_moe` over Marlin 1.01x, 1.02x, 1.07x, 1.04x, 1.04x (1.17x at C=128), agentX 0.97-1.03x |

What it says: only the block-FP8 model changes speed. The gain is in decode (a 27B FP8 decode step at C=1 is 20.6 ms instead of 31.6 ms) and shrinks as the decode batch fills; prefill (TTFT) is at parity, as in the kernel tables. The BF16 control and gemma-4 show no change in RAG, so the FP8 gain is not a host or GPU effect (the FP8 pair also swapped GPUs between RAG and agentX). In agentX the control's TPOT is 4-6% better on the after image in all three runs (cause not investigated), so part of the FP8 agentX TPOT gain is not FP8. The attribution of the FP8 gain to the DeepGEMM dense leaf rests on the kernel benchmarks, not on a per-run kernel trace (the server log does not name the leaf).

Findings:
- **Before-image start failures** (30 runs never started): Qwen3.8-27B-NVFP4 `RuntimeError: Kernel implementation not found` from `fp4_quantize` (`layers/dense/nvfp4.py:196`, `tokenspeed_kernel/registry.py:531`); Qwen3.5-122B-A10B-NVFP4 `Check failed: (fc1_act_global.ndim() == 0 || fc1_act_global.size(0) == num_experts_on_rank)` in `flashinfer_cutlass_nvfp4_moe_apply`; V4.1 `ValueError: Unexpected V4.1 expert tensor: layers.0.ffn.experts.128.w1.scale` (`deepseek_v41.py:2444`, no `num_checkpoint_experts` support). All three run on the after image; the fixing commits were not bisected.
- **Scheduler abort with KV events on** (both images, Qwen3.8-27B-FP8, RAG): `Fatal check failed: cache mutation on a KV event boundary with no token descriptor` at `scheduler.cpp:385` (after) / `:386` (before) in `Scheduler::markKvEventBoundaryForReconcile`, after `[L2] writeback` lines. The `smg_grpc_servicer` publishes KV events when no `--kv-events-config` is given; `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` avoids it. Default configuration: the after image crashed at C=128 (also at C=128 in a repro on a fresh server, not with the variable set); the before image crashed at C=128 in its session too (that run has a `DONE` marker, but 342 of 500 requests failed with 503 and the fatal line is 33 s into it; its C=256 then failed on the dead server; corrected in task 48, section 9.45) and at C=128 in the repro. The gateway stays up after the engine dies, so the harness got a watchdog (client killed 3-4 s after the fatal line, server restarted). The fix is in the scheduler (open).
- **gemma-4-31B-it**: (1) completions prompts need a leading `<bos>` (the tokenizer adds none; without it the model repeats ` the the ...`), a harness fix, same in both images; (2) the default gateway parser selection (`hf` tool-call parser) fails every chat request with HTTP 400, `--tool-call-parser gemma4` fixes it (open item); (3) at TP1 the engine's `max_req_input_len` is 199,614 (KV capacity 293,760 tokens), below the 256K agentX traces, so agentX C=16 and 64 fail at warmup (gateway 400 at `WorkerSelection`) in both images; (4) RAG saturates from C=32 at about 27 running requests with the KV pool full (2291 of 2295 pages).
- **Decode batch above 160**: the server default `max_cudagraph_capture_size` is 160 while the benchmark ran `--max-num-seqs 512`. On V4.1 the server log shows about 49 ms a step at batch 116 (C=128), 130 ms at 244-256 (C=256) and 140 ms at 500 (C=512), which is why RAG C=256 is slower than C=128 (6.44 vs 7.43 req/s) and C=512 is 1.6x faster than 256, in both V4.1 backends. This fits a step that no longer replays a graph; the eager fallback was not read or verified, and a larger capture size was not tried. It applies to both images and to every model at the C values where the batch passes 160.
- **Smoke agentX** (8 entries, 240 s, `--unsafe-override`) is not submission-valid and is not compared; all 26 completed full-mode runs are valid.

Caveats: one run per point; shared host (concurrent TP1 sessions on GPUs 6 and 7, other users' jobs on GPUs 0-5); V4.1 is a 128-expert subset whose output is not meaningful; agentX only at C = 1, 16, 64 with effective concurrency far below C (1.8-8 at C=16); the GPU 6 "100% utilization at 0 MiB" reading was an operator observation and does not show in the records. Counts: primary set 70 RAG and 26 agentX full-mode runs done, 4 agentX runs failed (gemma), 30 runs of three before-image sessions never started; harness-wide 112 done and 41 failed or not runnable, including the default-configuration, repro, dry-run and smoke runs.


## 9.45 Task 48: detailed benchmark document (2026-11-13)

`deepgemm-image-benchmark.md` was renamed to [`deepgemm_benchmark.md`](deepgemm_benchmark.md) with `git mv` (an untracked, byte-identical copy named `deepgemm-image-benchmark-20261112.md` was in the working tree; it was moved out to `DeepGEMM/tmp/` so there is one document), and every link was updated (report, this file, local `WEEKLY_REPORT.md`, the spec). The document was rebuilt from the result JSON:

- Section 1: one row per model x dataset, each cell `before / after (ratio)` for output tok/s, TTFT p50 and TPOT p50 at every C (RAG 1-512, agentX 1, 16, 64). After-only rows are labelled, and the V4.1 row compares mega_moe with Marlin on the after image. Then a short headline list.
- Section 2, testing conditions. Read-only from aws-raichu: driver 595.99.02, CUDA 13.2, 2x Xeon Platinum 8559C, Ubuntu 24.04.4, GPUs 6/7 `PIX` on NUMA node 1. Image IDs and package versions from the images (`pip list`, `importlib.metadata`). The images have no `.git`, so the commits were checked by comparing MD5s of `python/pyproject.toml`, `models/deepseek_v41.py`, `scheduler.cpp` (and DeepGEMM's `deepgemm_report.md`) inside each image against `git show`: before `e27cee861` with the `tokenspeed-deepgemm 2.8.0.post20260924` wheel, after `ef057889f` with DeepGEMM `976f212` (`deep_gemm 2.8.2+local`). tokenspeed-scheduler is 0.1.41 in both, and the only scheduler change between them is `b39d7ccc9` (re-wrapping). The effective engine settings come from the `server_args` line (CUDA graphs up to 160, `Capturing batches: [160, 152, 144, 104, 40]` on every server, KV dtype `auto` = BF16, prefix cache and L2 on, chunked prefill 8192, `moe_backend` auto / mega_moe), along with the engine capacity per model, the exact docker / `tokenspeed serve` / AIPerf command lines, probes and warmup. The run order is now by server launch (`gpu_at_launch.txt` + run markers), because `session.json` is rewritten per launch: the FP8 pair had a RAG server and a later agentX server on swapped GPUs.
- Section 3: per-run tables with request throughput, input/output tok/s, TTFT mean/p50/p99, TPOT mean/p50/p99, ITL p50/p99, ok/failed, effective concurrency, ISL, wall time, and agentX warmup and `submission_valid`.
- Sections 4-7: failures with error lines and log paths (before-image startup failures; gemma agentX C=16/64 with the gateway `Stage WorkerSelection failed with status 400` and AIPerf `No profile results to export`), the default-config and repro crash runs, the excluded smoke runs, and the caveats.

Correction: in the default-config set, the before image's C=128 run had a `DONE` marker but 342 of 500 requests failed (`503 start_generation_failed`), and the fatal line (05:49:36) is 33 s into it. So that image also died at C=128 (C=256 then failed on the dead server), the same as the after image and both repros. Section 9.44 and the harness README said C=256; both are fixed.

Checks: `tmp/check_ratios.py` parsed the 204 summary values of section 1 and recomputed every ratio from the unrounded JSON: 0 mismatches. The tables are generated by `tokenspeed/tmp/bench_images_local/results/make_tables.py benchmark_template.md` (it expands `{{FRAG ...}}` lines from the new `make_detail_tables.py`). No benchmark was rerun and no GPU job was started.

## 9.46 Task 50: the scheduler KV-event abort (2026-11-13)
Symptom: with KV events on (the smg servicer's default), `Fatal check failed: cache mutation on a KV event boundary with no token descriptor` (`Scheduler::markKvEventBoundaryForReconcile`) aborts both images under RAG load at C=128 on Qwen3.8-27B (hybrid: one full-attention group plus state groups), 32-35 s into the run (section 5 of `deepgemm_benchmark.md`).
Reproduction in the C++ test binary (scratch build on aws-raichu, `tmp/sched50`): a randomized driver (full + one state group, prefix granularity 4, 8-token chunks, 2-4 requests, small pools, events on, random drains) aborted at seed 1 with L2 off, so neither L2 nor kvstore is needed. A backtrace put the abort in `CacheCoordinator::evictCachedBlock` of a **state** group during a decode admission, and a trace of the sink / register / drain calls showed the order: tail KV block removed (`removed g0`), `DRAIN ... residency 0` (descriptor erased), then `removed g1` of the same hash (the checkpoint) with no descriptor.
Root cause: `evictCachedBlock` / `evictTailEntries` evict a boundary's companions "alone" but leave one a request still pins (the checkpoint is the live slot 0 of a decoding request). `DeviceBoundaryResidency` skipped State groups, so with the KV gone it returned `kNone` and `DrainKvEvents` erased the descriptor ("drops descriptors of boundaries with no cached child"). The checkpoint was a cached child, so the invariant "descriptors are erased only when nothing is cached" was false for hybrid models, and the checkpoint's own eviction (after the request rotated it out) found no descriptor. This is the same class as the restore fix (`RestoreDone` re-registers prefix pages) but on a different path.
Fix (`cache_coordinator.cpp`): count a cached State checkpoint (page offset 0, the boundary's hash) as a child for `kNone`; it never counts toward `kComplete` (a checkpoint exists only at natural points), so the Stored/Removed events a consumer sees are unchanged: Removed is still sent when the KV leaves, and the boundary stays `kPartial` until the checkpoint is evicted. The check in `markKvEventBoundaryForReconcile` is untouched. Design docs (`scheduler.md`, `cache-concepts.md`) said State groups gate `BlockStored`; they never did, and now state what the code does.
Tests: `CacheCoordinatorAdmissionTest.ACheckpointAloneKeepsItsBoundaryResident` (kNone / kPartial / kComplete for checkpoint-only, +KV, KV-only; fails before the fix) and `StateKvEventSuite.ACheckpointOutlivingItsKvKeepsTheBoundaryDescriptor` (scripted scenario found by a pool-size search on the driver, five prompts of one page and a 1-token tail, drains every round; aborts before the fix, and after it the events a consumer folds end empty once the cache is cleared). The randomized sweep (300 seeds, L2 on and off) passed after the fix and was not kept.
Verification (aws-raichu, scratch copy, no scheduler version bump): C++ `tokenspeed_scheduler_tests` 610 passed; `tokenspeed-scheduler/python/tests` 248 passed; runtime `cache` 137, `execution` 195 (7 skipped), `test_kv_events` 10, `test_scheduler_cache_ops` 10, `test_lcm_scheduler_geometry` 2, `test_prefix_replay_scheduler_config` 25, `test_event_loop_scheduler_config` 4, `_stats` 2 passed, with the rebuilt module preloaded ahead of the venv's editable install. pre-commit (clang-format) on the four files: one blank line removed, then clean.
Rerun: the Python extension built from the same tree was mounted over `tokenspeed_scheduler_ext` in `nctu6/tokenspeed:ef057889-20261112` (`-v ...:ro`, new `mounts=` key in the harness queue). Qwen3.8-27B-FP8, TP1, GPU 6, default flags without `SMG_TOKENSPEED_SERVICER_KV_EVENTS`, the log shows `KV event publishing enabled` and `kv_events_config` with the ZMQ publisher; RAG 500 prompts: C=64 500 ok / 0 errors (124.7 s, 1060.7 output tok/s), C=128 500 ok / 0 errors (103.6 s, 1284.7 output tok/s), 0 `Fatal check` lines, where the unfixed container died 35 s into C=128. Results `tokenspeed/tmp/bench_images_local/results/fix50/`. The NVFP4 model was not rerun (same scheduler path). Not verified: the before image with the fix (the module is built for the current tree), long agentX runs with events on.
