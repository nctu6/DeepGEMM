# DeepGEMM on SM120 (RTX PRO 6000 Blackwell): report

Long form: [`deepgemm-sm120-design.md`](deepgemm-sm120-design.md) (designs, bugs, every table; the
section numbers below like "design 7.8" point there). Status log, dead ends and per-task
measurements: [`deepgemm-analysis.md`](deepgemm-analysis.md) ("analysis 9.x"). Serving benchmark of two
TokenSpeed images (before / after SM120 work, six models): [`deepgemm_benchmark.md`](deepgemm_benchmark.md).

## 1. Purpose and status

Upstream DeepGEMM only runs on Hopper and datacenter Blackwell (`wgmma`, `tcgen05`/TMEM). This work
adds SM120 (compute capability 12.0, consumer and workstation Blackwell) with new kernels built on
TMA, mbarrier pipelines and `mma.sync`, plus arch-12 host paths, so TokenSpeed can serve
DeepSeek-style models on RTX PRO 6000. Tasks 1-26 and 29-39 are done: the GEMMs, the indexer kernels and the
TokenSpeed attention leaves run on SM120 and beat the Triton / FlashInfer fallbacks everywhere
except a few ties and losses listed in section 3 and section 7. Tasks 1-39 are pushed to `origin/nctu6` in both
repos (DeepGEMM first, then TokenSpeed with its `DeepGEMM` pin on a pushed commit); use `git log origin/nctu6`
for the exact heads. Tasks 40-47 benchmark two TokenSpeed images on GPUs 6 and 7
([`deepgemm_benchmark.md`](deepgemm_benchmark.md), analysis 9.44): the block-FP8 27B model serves
1.52x faster at C=1 (1.04x at C=512), the BF16 control ties, and both NVFP4 models and the V4.1 subset start only on the
after image.

## 2. What is supported on SM120

| API / op | Status |
|---|---|
| `fp8_gemm_nt`, m-grouped contiguous (incl. PSUM) and masked, FP32 1x128 / 128x128 scales | supported (`sm120_fp8_gemm_1d2d`) |
| Block-scaled GEMMs: MXFP4, NVFP4, FP8 with UE8M0 32/128-element scales (same three GEMM shapes) | supported (`sm120_fp8_fp4_gemm_1d1d`, hardware block-scaled `mma.sync`), BF16 or FP32 output, `alpha`, C accumulation |
| W4A8: FP8 activations x MXFP4 weights (normal, m-grouped contiguous incl. PSUM, masked; task 21) | supported (same kernel, mixed `kind::mxf8f6f4 e4m3.e2m1`), K-major, 64-byte B rows |
| MN-major FP8 operands (`fp8_fp4_gemm_nn/tn/tt`), k-grouped FP8 | supported (native `ldmatrix.trans.b8`) |
| BF16 GEMMs (K- or MN-major, m-grouped, k-grouped) | supported; `bf16_gemm_nt` stays on cuBLASLt unless deterministic mode |
| `einsum`: `bmk,bnk->mn`, `bhr,hdr->bhd`, `bhd,hdr->bhr` (BF16); `fp8_einsum` (3 expressions) | supported; the head expressions run as one batched launch |
| `tf32_hc_prenorm_gemm` (mHC pre-norm) | supported |
| `fp8_fp4_mega_moe` (MXFP4 routed experts; task 21) | supported as a sequence of kernels with the SM100 arithmetic, on one rank or across up to 8 ranks (run on 2) with CUDA IPC peer stores over PCIe; bitwise equal to one rank, CUDA-graph safe. Fused shared experts are rejected. The one-rank form also serves tokens replicated over TP ranks (a rank's expert range or MoE-TP slice, partials summed by the caller; task 29) |
| `fp8_mqa_logits`, `fp8_paged_mqa_logits` (head dim 128/64, paged also 32), sparse MQA logits | supported |
| Block-scaled MQA logits, MXFP4 / MXFP8 rows (contiguous, paged, sparse) | supported; scores the 68-byte V4.1 index rows natively |
| Mixed packed-UE8M0 / FP32 scale pair, legacy `fp8_*` aliases, `get_supported_arch_majors()` | supported |
| TokenSpeed leaves: DSA/V4.1 indexer top-k (DeepSelect in 93 KB), DSA / MLA / V4 attention, Hadamard, MXFP4 linear and SwiGLU, mHC | supported (Triton or CuTe DSL for the attention kernels; no DeepGEMM code) |

## 3. Performance

Ratio = best Triton / FlashInfer (or the stated baseline) time over SM120-kernel time, so above 1 the
SM120 kernel is faster. CUDA graph, cold weights, interleaved runs, GPU 7 (design 7.1; the full per-op
table is design 7.8).

| Op | SM120 kernel | Baseline | Result | Worst shape |
|---|---|---|---|---|
| FP8 dense GEMM | DeepGEMM 1D2D | best Triton / FlashInfer block-FP8 | 1.61x geomean (112 TP8 points), 0.94-4.0x; decode 1.2-4.0x, prefill at parity | 0.94x; 7 points at 0.94-0.99x (e.g. q_b M=2048/4096) |
| MXFP4 GEMM | DeepGEMM 1D1D | FlashInfer MXFP4 (`cudnn` / `b12x`) | 1.40x geomean, 62 of 63 wins | 0.979x dense gate_up M=256 (L2-bound tie; cluster tile sharing tried, slower) |
| MXFP4 GEMM | same | best Triton / FlashInfer block-FP8 | 1.5-11.4x (geomean 3.1x); 1.4-2.3x (geomean 1.8x) over the best FP8 kernel | 1.52x (K = 256 `shared down`, M = 4096); 0.94x vs the best FP8 kernel (`shared down`, M = 32) |
| NVFP4 GEMM | same, 16-element scales | best FlashInfer NVFP4 | 1.223x geomean, 47 of 63 wins | 0.910x |
| W4A8 grouped GEMM (task 21) | DeepGEMM 1D1D, FP8 x MXFP4 | same kernel, FP8 x FP8 on converted weights (bitwise equal) | 1.35-1.72x at decode / mid shapes, 1.07-1.10x small | 0.84x (4 x 4096 x 4096 x 2048, compute-bound) |
| Mega MoE, one rank, V4 / V4.1 shapes (task 21) | `fp8_fp4_mega_moe` | Marlin W4A16 (TokenSpeed's SM120 MXFP4 MoE; Triton MXFP4 is 9-13x slower) | 0.97-0.99x at 4-64 tokens (DRAM roofline), 1.01-1.09x at 256-1024, 1.59-1.87x at 4096-8192 | 0.82x at 1 token (launch count); no TokenSpeed caller at one rank |
| Mega MoE, EP 2 on GPUs 6+7, V4 / V4.1 shapes (task 21) | `fp8_fp4_mega_moe` over CUDA IPC | Marlin W4A16 per rank + all-reduce (attention TP) or all-gather / reduce-scatter (attention DP), same served tokens | 0.99-1.05x at 4-1024 tokens per rank, 1.28-1.39x at 2048-4096 | 0.95-0.96x at 1 token (11 launches); TokenSpeed leaf `deep_gemm_sm120_mxfp4_mega_moe_apply`, opt-in `--moe-backend mega_moe` |
| Mega MoE under attention TP, real V4.1 (128 experts, GPUs 6+7, 2 x 2004 tokens; task 29) | one-rank `fp8_fp4_mega_moe` over a shard of the experts + the TP all-reduce | Marlin TP2 (`auto`): 15.1K tok/s prefill, 10.69 ms TPOT | prefill 1.12x (MoE TP2, 17.0K) / 1.11x (EP2, 16.8K); DP2/EP2 1.26x | TPOT ties with MoE TP2 (10.66 ms), 1.05x slower with EP2 (11.22 ms: the expert range leaves one rank more of the 12 routes), 1.31x slower with DP2/EP2 (14.02 ms); opt-in, `auto` keeps Marlin (analysis 9.33) |
| Same, after fusing SwiGLU into the L1 GEMM epilogue and the gather into dispatch (task 34, bitwise equal to the unfused kernels) | one-rank `fp8_fp4_mega_moe`, DeepGEMM `aa1ecfe` | Marlin TP2 (`auto`): 15.1K tok/s prefill, 10.57-10.60 ms TPOT | prefill 1.13x (17.05K) | leaf 3.7-13.4 us per layer faster at 1-16 tokens; TPOT 10.49 ms (three runs), 1.0% under Marlin and 0.11 ms under the unfused kernels, the size of the run-to-run spread; opt-in, `auto` keeps Marlin (analysis 9.38) |
| MXFP8 dense GEMM, 1x32 UE8M0 both sides (V4.1 projections, task 24) | `deepgemm_sm120_mm_mxfp8` (DeepGEMM 1D1D, deterministic, batch invariant) | `triton_mm_fp8_blockscale` (the only SM120 leaf before) | 2.2-11.7x (6 projections x M = 1-2048, graph, cold weights) | 2.2x `shared_w2` M=256 |
| MXFP4 linear leaf (with quantization) | `deepgemm_sm120_mxfp4_linear` | FlashInfer quantize + best `mm_fp4` | 1.23x geomean, 60 of 63 wins | 0.90x q_b M=256 |
| MN-major FP8 / k-grouped FP8 | DeepGEMM | explicit transpose + K-major kernel | 1.63x (B, M <= 64) / 1.11-2.06x | 0.97x (B operand) / 0.76x vs a torch loop at small K |
| `fp8_einsum` (3 expressions) | batched launch | best of cuBLASLt BF16, torch loop, `torch.einsum` | 1.52x geomean (26 shapes) | 0.99x `bhd,bhr->hdr` h=8, b=128 (tie) |
| BF16 `einsum` heads (task 18) | batched BF16 launch, bit-identical to per-head | per-head GEMM calls; cuBLASLt | 0.21x of the per-head time, 4.8x faster (56 points); 1.06x vs cuBLASLt, 52 of 56 within 3% or faster | 0.81x vs cuBLASLt, 16 heads, b = 1 |
| V3.2 indexer (MQA logits + top-k) | DeepGEMM logits | Triton DSA indexer | decode 3.4-19.3x, prefill 7.8-52x | 3.4x decode, 1 x 4K |
| V3.2 DSA prefill, tiny chunks (task 18) | `deep_gemm_dsa_prefill_topk`, host-light | `triton_dsa_prefill_topk_fp8` | 1.35-98x at 1-8192 tokens; 1.35x at 128 tokens (was 0.52x) | 1.35x, 128 tokens, no prefix |
| V3.2 DSA prefill, host path of a 128-token chunk (task 37) | `fp8_mqa_logits` (kernel found by its template integers, one `as_strided`) and `deepselect_local_topk` (whole tensors, one validation); bitwise equal | the same calls before (`jit->compile` per launch, four slices, double validation) | `fp8_mqa_logits` 8.6 -> 5.6 us, DeepSelect 17.2 -> 9.1 us of host time per call; leaf 116 -> 87 us, 1.34x eager back-to-back (119 -> 89 us); no change end to end on a 4-layer V3.2 (GPU-bound, ~11 ms per 128-token chunk) | 1.34x leaf; 5.5 of DeepSelect's 9 us are the TVM-FFI call and the `values` allocation |
| `dsa_index_candidates` (task 18) | DeepSelect over owned length | `torch.topk` leaf; Triton | 1.95x geomean (1.25-4.7x, 54 points); 18x vs Triton | 1.25x, 8 rows, 128K context, 2 ranks |
| V4.1 indexer, 68-byte rows (task 16) | `deep_gemm_sm120_dsv41_index_topk` | `triton_dsv41_index_topk` | 2.2-88x per pass: decode Full 5.6-16x, source 7.7-33x, Reindex 2.2-22x; prefill 11-46x, 22-88x, 6.8-22x | 2.2x Reindex, 1-8 requests |
| V4.1 indexer Reindex, sparse MX kernel (task 35) | `fp8_fp4_paged_sparse_mqa_logits` (even split of the KV splits, gather overlapped with the MMAs) | the previous sparse kernel; scoring the whole history (dense paged / contiguous) | kernel 2-6x (8 / 32 / 128 requests of 32K rows 82 / 94 / 252 -> 21 / 45 / 146 us; chunk of 128 / 1024 / 4096 queries 140 / 710 / 2770 -> 43 / 243 / 912 us), bitwise equal; leaf 0.95-2.9x over the old default from twice the 16384-row pool | 0.95x (8 requests, 32K history); dense wins up to 1.3x between one and two pools |
| DSA sparse attention (task 17) | `cute_dsl_dsa_mma_*` | `triton_dsa_mma_*` (64 buckets) | E4M3 1.91x, BF16 1.16x geomean, 1.00-2.34x | 1.00x BF16 16 heads |
| DSA CuTe launch planner in SM waves (task 24 E) | `cute_dsl_dsa_mma_*`, new `plan_launch` | the same cores, earlier planner (140 decode shapes of 1-192 tokens) | 1.05x BF16 / 1.04x E4M3 geomean, up to 1.63x; 1.02x at 128-2048 tokens; rows bitwise equal | 0.97x (noise) |
| DSA CuTe stream-K schedule (task 30) | `cute_dsl_dsa_mma_*`, whole tokens first, split tail, `plan_launch(..., fp8)` | the same cores, uniform-split planner (140 decode shapes x 2 dtypes; 64 buckets) | 1.032x geomean, up to 1.38x (decode); 1.017x on the 64 buckets, up to 1.30x; rows bitwise equal | 0.97x (E4M3, same plan, noise) |
| V4 prefill CuTe: 8-warp 16-head CTA + wave-model plan (task 31) | `cute_dsl_dsv4_mma_prefill`, `plan_launch(tokens, heads, num_sms, width)` | the same core with 4-warp CTAs and the uniform plan (165 points, 17-128 heads, widths 128 / 640 / 1152, 1-500 tokens) | 1.095x geomean, up to 1.66x; rows bitwise equal | 0.89x (17 heads, width 128, 24 tokens, 8.3 to 9.3 us) |
| DSA sparse attention | `triton_dsa_mma_*` | portable `triton_dsa_*` | 22-114x, 72x geomean (batch-invariant) | 22x BF16 decode, 16 heads, 1 request |
| MLA decode (task 17) | `cute_dsl_mla_mma_decode_with_kvcache` | `triton_mla_mma_decode_with_kvcache` | 1.46x geomean (128 points), 1.84x on 4/8-row blocks | 0.97x BF16 16 heads, 2 x 1K |
| V4 / V4.1 prefill attention (task 17) | `cute_dsl_dsv4_mma_prefill` (> 16 heads) | `triton_dsv4_mma_prefill` | 1.13x geomean (117 points), 0.89-1.63x | 0.89x window-only rows at 1-2 tokens |
| V4.1 decode, packed rows (task 18) | `triton_dsv41_mma_selected_attention` | gather into BF16 workspace + same kernel | 1.66x geomean (21 points), 1.01-3.67x | 1.01x, 128 heads, 128 tokens |
| V4 decode, FP8 page-planar rows read in the CuTe core (task 36) | `cute_dsl_dsv4_mma_decode` (dequantizing gather, no BF16 workspace; bitwise equal) | `triton_dsv4_decode` (workspace + CuTe core above 16 heads); `triton_dsv4_mma_decode` | 1.75x geomean over the workspace route (105 points, 8-128 heads, widths 128 / 640 / 1152, 1-2048 tokens), 1.06-5.14x; 1.71x over the Triton direct reader, 1.12-2.23x | 1.06x workspace (128 heads, 16 tokens, 1152 rows) / 1.12x Triton |
| V4.1 decode, packed rows read in the CuTe core (task 36) | `cute_dsl_dsv41_mma_selected_attention` (FP8 `swa` + FP4 `global`, bitwise equal) | gather glue + `cute_dsl_dsv4_mma_prefill`; `triton_dsv41_mma_selected_attention` | 3.12x geomean over the workspace route (1.43-7.93x), 1.62x over the Triton direct reader (0.94-3.02x; 1.92x with a compressed cache, 1.16x window-only) | 0.94x vs Triton, 64 heads, window-only, 1 token (0.4 us of 7) |
| MLA decode / V4 prefill / Hadamard (Triton, task 14) | `triton_*_mma_*` | portable Triton leaves | MLA 14-131x, V4 4-9x, Hadamard 55-117x from 512 tokens | Hadamard 0.9x at 1 token (launch-bound) |
| mHC | `tf32_hc_prenorm_gemm` leaves | Triton | `mhc_pre` 1.5-13.6x, `mhc_mixes` 1.6-15x | 1.5x |

End to end (in-process `Engine`):
- Qwen3.8-27B-FP8 (the block-FP8 path of DeepSeek-V3): TPOT 1.18-1.55x better (design 7.4).
- 4-layer V3.2, real dimensions, random weights, against forced `triton_dsa_*`: prefill 82-95x, TPOT 2.0-14.4x (analysis 9.18).
- Same model, 128-token chunks (task 37, analysis 9.41): 8.1K tok/s for one chunk, 10.6K for four, 11.4K for sixteen before and after the host-path cuts; the chunk is GPU-bound at ~11 ms, so the 4 x 29 us the leaf saves is below the run-to-run spread.
- V4.1 topology with dummy weights on the native indexer: prefill 1.11-1.19x, TPOT 1.02-1.06x, same greedy tokens (analysis 9.21).
- Real V4.1 checkpoint (all 40 layers, Engram, head; first 128 of 384 experts), task 22: every path has the same single-layer error against a float32 reference built from the same weights (2.8% layer 0, 3.7-3.9% layer 2). Task 24 (analysis 9.29) profiled it and moved the dense projections to DeepGEMM: Marlin TP2 14.5-14.8K tok/s prefill, 10.4-10.7 ms TPOT (was 10.2-10.7K, 23-25 ms); MegaMoE DP2/EP2 18.1-18.6K tok/s (1.25x over Marlin), 13.8-14.0 ms TPOT (1.31x behind: DP attention reads every dense weight per GPU, plus MegaMoE glue).
- Real MLA checkpoint (Moonlight-16B-A3B, one GPU): correct answers; TPOT 2.3-3.2x better than the portable MLA decode at 3K context x 8, a tie at short context (analysis 9.27).
- Served with `tokenspeed serve` (tasks 40-48, [`deepgemm_benchmark.md`](deepgemm_benchmark.md): testing conditions, every run, failures; analysis 9.44, 9.45), image `e27cee86` (before) against `ef057889` (after), same flags, RAG at C = 1 .. 512 and agentX at C = 1, 16, 64: Qwen3.8-27B-FP8 output tok/s 1.52x at C=1, 1.28x at 16, 1.11x at 64, 1.04x at 256-512 (TPOT 1.53x at C=1, TTFT 1.00x); Qwen3.6-35B-A3B (BF16 control) and gemma-4-31B-it 1.00-1.03x. Qwen3.8-27B-NVFP4, Qwen3.5-122B-A10B-NVFP4 (TP2) and the V4.1 128-expert subset (TP2) fail to start on the before image and run on the after image (V4.1: Mega MoE 1.01-1.17x over Marlin in RAG, a tie in agentX). One run per point on a shared host.

## 4. What is not supported, and why

| Not supported | Hardware reason |
|---|---|
| Mega MoE with fused shared experts; mega gate; mega mHC | not a hardware limit (analysis 9.26). The SM100 kernels fuse around TMEM / UMMA / NVLink, but the math maps to `mma.sync`. Not built because TokenSpeed has no caller (it runs V4's shared expert separately and calls no mega gate or mHC) |
| FlashMLA DSV4/V4.1 and DSA, trtllm-gen DSA, `tokenspeed_mla`, `fast_hadamard_transform` wheel | `wgmma` / `tcgen05` kernels with no sm_120 cubin; replaced by SM120 `mma.sync` leaves |
| MN-major FP4 operands | no sub-byte transposing `ldmatrix` (FP8 has `.b8`, BF16 the 16-bit form) |
| FP8-output (quantized D) and custom epilogues | SM100 epilogues built on TMEM |
| TMA multicast | emulated in software on SM120 (a multicast launch traps into `emulate_tma`) |
| Paged MQA with head dim other than 32/64/128 (128 only for block-scaled), MXFP8 at 128 Q rows | no kernel; MXFP8 at 128 Q rows spills in the 104 math registers |
| `fp8_einsum` with accumulation `c` on `bhd,hdr->bhr` / `bhr,hdr->bhd` | the SM120 FP8 batched GEMM has no accumulation epilogue |

SM120 has about 99 KB of shared memory per block (101376 B) against 227 KB on SM90/SM100, so no upstream
kernel fits without a redesign (design section 2). Details of each item: design 7.8.2 and 9.

## 5. Verification

Latest run 2026-11-11 (task 39, analysis 9.43), DeepGEMM code head `0d89c76` (docs-only commits after it; kernels `aa1ecfe`, `b4eaebf`, `f628449`) and TokenSpeed `ad944dc2e` (pin on the DeepGEMM head in the commit after it). Earlier runs: task 33 (analysis 9.37), task 26 (9.32).
- TokenSpeed kernel suite: 8011 passed, 2764 skipped, 0 failed (GPU 7, `PATH` with `/usr/local/cuda/bin`; `test_numerics.py` + `test_triton_adapter.py` 24 passed, 2 skipped). With two GPUs visible, `nvidia/ops/communication` 155 passed (tp4 / tp8 skip: 4 and 8 NVLink GPUs) and `test_mega_moe_sm120_token_layouts.py` (2 ranks) passes, twice.
- TokenSpeed runtime suite, 0 failed in every directory (9007 passed): cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 374, models 567, moe 72, sampling 144, top-level files 6787.
- DeepGEMM (`DG_JIT_CHECK_NO_SPILLS=1`, fresh JIT cache, GPUs 6 and 7): `test_fp8_fp4` (with the PSUM-gap test), `test_bf16`, `test_einsum`, `test_layout`, `test_hyperconnection`, `test_attention`, `test_mqa_logits_launch` and `test_mega_moe_sm120` exit 0; `test_mega_moe_sm120_ep.py` (2 ranks, GPUs 6+7) is bitwise equal to one rank; `sm120_mqa_check.py` (46 cases), `sm120_blockscale_check.py --mode all --grouped` (170), `sm120_paged_mqa_blockscale_check.py` (51), `sm120_sparse_paged_check.py` (28) and `sm120_sparse_mqa_check.py` (40) all OK; `test_lazy_init --num-processes 2` (3 tests) OK.
- End to end, greedy, in-process, all correct ("Paris", "5, 7, ..."): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2), Qwen3.8-27B-NVFP4, gemma-4-31B-it.
- Lint: `pre-commit run --all-files` (clang-format included) is clean twice on a clean tree of the TokenSpeed head, as are the other steps of the Lint workflow (runtime and scheduler ruff, the scheduler file-start check, `test_weekly_release.py`). Not run: scheduler clang-tidy and C++ build (no ninja on the box).
- compute-sanitizer: the three heavy tests are done (task 38, analysis 9.42): FP8 masked (64 cases), FP8 k-grouped (126) and `test_mqa_logits` (64) run to the end under memcheck, synccheck, racecheck and initcheck with 0 kernel errors, every kernel binary of the three tests launched under each tool (91 / 23 / 8; racecheck ran the 21 largest cases as one launch each). The tools instrument only DeepGEMM's kernels (initcheck all kernels). Two test-side fixes (`0d89c76`): `test_mqa_logits` no longer copies the never-written padding columns of the output, and `test_psum_gaps_do_not_reach_valid_rows_sm120` guards that the PSUM padding rows, which the SM120 kernels read at tile granularity, never reach a valid row. The 22 test functions of task 15 were not rerun.
- The DeepGEMM source at the pinned commit builds like `install.sh` (`pip wheel`, nothing installed) and `test_deepgemm_sm120_gating.py` passes against it (11 passed; task 26 run; the kernel suite above includes the gating test against the local build).
- Real weights (task 22, analysis 9.27): V4.1 checkpoint on GPUs 6+7 with its first 128 experts. MegaMoE DP2/EP2 (`deep_gemm_sm120_mxfp4_mega_moe_apply` selected), Marlin TP2 and forced portable TP2 all have the same single-layer error against a float32 reference (layers 0 and 2). Moonlight-16B MLA answers correctly; its greedy decode matches the portable path up to near-ties, and two SM120 runs are bit-identical. The V4 checkpoint does not load (TokenSpeed has no V4 model class since upstream `0c1bc3ed3`).
- Environment-only failures: `test_locality_domain` (MLOPart), `test_mega_moe` (NVLink), the sanitizer's `test_legacy` entry (no triton in the venv).

Test machine: aws-raichu, 8x RTX PRO 6000 Blackwell Server Edition (96 GB, 188 SMs), CUDA 13.2, driver 595.99.02, PCIe only. Tests and benchmarks ran on GPUs 6 and 7.

## 6. Build and use

TokenSpeed carries DeepGEMM as the `DeepGEMM/` submodule (`https://github.com/nctu6/DeepGEMM`, branch `nctu6`). `scripts/install.sh` detects SM120 and builds it from source, with no flag, and keeps the published wheel on other GPUs. The UnieAI Docker image always builds from source.

```bash
./scripts/install.sh                      # TokenSpeed root; SM120 builds DeepGEMM from source
./scripts/install.sh --force-deepgemm     # force the source build on any GPU (or FORCE_SM120=1)
```

In the DeepGEMM checkout (aws-raichu, `CUDA_VISIBLE_DEVICES=7`):

```bash
scripts/sm120_remote.sh build
python tests/test_fp8_fp4.py
python scripts/sm120_mqa_check.py
python scripts/sm120_blockscale_check.py --mode all --grouped --repeat 20
```

Benchmarks, refits and end-to-end scripts: design section 11. TokenSpeed selects DeepGEMM on SM120 only when
the installed build reports SM120 kernels (`deep_gemm.get_supported_arch_majors()` returns `(9, 10, 12)`).

## 7. Open items

This is the canonical open list; the Open list in `deepgemm-analysis.md` ("Current state") says the same.

- One FP4 tie with FlashInfer b12x: dense gate_up M=256 (0.98x in the sweep, 1.02x isolated), bound by L2 -> SM traffic (analysis 9.15). Cluster tile sharing over DSMEM is 2.3-7.3x slower (analysis 9.28) and the ceiling with free sharing is 1.13x; not shipped (probe and patch in `scripts/`).
- MegaMoE: `auto` keeps Marlin (analysis 9.33, 9.38). Under attention TP2 + MoE TP2 on real V4.1 it prefills 1.13x faster and its TPOT is 10.49 ms against Marlin's 10.57-10.60 (1.0%, the run-to-run spread). Left: the combine and staging (about 0.3 ms per step; the combine in the L2 epilogue needs atomics, which break bitwise equality, or the next layer's staging + dispatch, 3-6 us per layer each) and the `auto` gates (numerics envelope, width 1 and 2 only, batch invariance, scratch in the KV plan). Attention DP stays 1.31x behind Marlin TP2 in TPOT (+2.6 ms of dense-weight reads per step, structural). The EP2 / DP2 paths use the fused kernels but were not timed on the real model after task 34. More than 2 ranks, fused shared experts, mega gate and mega mHC have no TokenSpeed caller.
- V4.1 indexer: the multi-wave sparse MX kernel is done (analysis 9.39). Left: SM100's sparse metadata kernel (shared) is a quarter of the sparse leaf at 1024 queries; FP8 `index_v4` rows have no pool-only scorer (the runtime does not store them on SM120). MXFP4 paged logits at 1.07 vs 1.23 TB/s are closed as page geometry.
- Attention: BF16 DSA / V4 stay at 42-48% of the tensor-pipe floor (analysis 9.30); V4 at 16 heads stays on Triton (analysis 9.35); one-request DSA decode stays at 20 us BF16 / 13 us E4M3 (analysis 9.30, 9.34); persistent CTAs, 16 chunks and on-chip chunk states are closed. The packed-row CuTe DSL V4 / V4.1 decode is done (analysis 9.40); left: its 32-head CTA spills 120-168 B, V4.1 window-only layers at 32-128 heads tie the Triton direct reader at 1-4 and 256+ tokens (0.94-1.06x), and the overlap of real V4.1 selections beyond 2048 positions is unmeasured.
- Host path of a 128-token V3.2 DSA chunk: done for `fp8_mqa_logits` and DeepSelect (analysis 9.41, leaf 116 -> 87 us). Left: the TVM-FFI call and `values` allocation of DeepSelect (5.5 of its 9 us), the `dsa_prefill_prepare` launcher (37 us), `nan_to_num_` (6 us), and the other launchers that call `jit->compile` per launch (about 3 us each).
- Full-model real-weight runs: V3.2 TP8 (no checkpoint on the box, about 690 GB FP8, all 8 GPUs) and V4.1 with all 384 experts (4 GPUs). The V4 checkpoint cannot load (no TokenSpeed V4 model class); no V3.2, GLM-5.x, dots3-note or V2-Lite checkpoint is on the box (analysis 9.27).
- Sanitizers and flakes: the three heavy sanitizer runs are done (analysis 9.42), but the tools see only the shapes the tests launch, the kernel filter leaves torch kernels unchecked in memcheck, synccheck and racecheck, racecheck models shared memory only (the 21 largest cases ran as one launch each) and the repeat loops were cut to 2. The V4.1 eager flake is closed as not reproducible (analysis 9.31); a recurrence needs its pytest output kept.
- Verification limits: `test_projection_tp` tp4 / tp8 need 4 and 8 NVLink GPUs; scheduler clang-tidy is not run (the C++ tests build and run in a scratch tree since task 50); TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90 / sm100 / sm103 / sm107, so upstreaming needs that scope to change; a wheel release is optional (the source build replaces the pin).
- Serving benchmark (tasks 40-54, analysis 9.44 and 9.46-9.50): the four findings of the first run are closed in the TokenSpeed tree (pushed to `origin/nctu6` with tasks 50-55). (a) KV-event abort: fixed in the C++ scheduler (a cached state checkpoint keeps its boundary descriptor, task 50); default config (events on) completes RAG C=128. (b) gemma-4-31B-it at TP1 `max_req_input_len` 199,614 is a documented configuration, not a defect: five sliding groups reserve one 8192-token prefill chunk each (task 52); `--kv-cache-dtype fp8` or `--chunked-prefill-size 1024` give 262,143 and agentX C = 1/16/64 completes (`deepgemm_benchmark.md` 3.3). (c) gemma-4 default tool-call parser: `ts serve` now defaults to `--tool-call-parser gemma4` (task 51). (d) CUDA graph cap: the default now covers `--max-num-seqs` (task 53), V4.1 RAG C=256 +61% (Marlin) / +76% (mega_moe). The before-image NVFP4 and V4.1 start failures are fixed in the after image. Left: one run per point (no repeat spread), agentX on gemma-4 only with the two non-default argument sets, the thinking-channel output with the gemma4 tool parser, and the price of the larger graph ladder (-9.8% KV tokens at `--max-num-seqs 512` on V4.1), which no long-context run has weighed against the gain.
- TP2 end to end (task 56, head `9bc0d7029`, KV events on, graph ladder to 512): Qwen3.5-122B-A10B-NVFP4 and V4.1 128-of-384 (Marlin `auto` and `mega_moe`) serve through `tokenspeed serve` with RAG C=16 / C=128, 500 ok / 0 errors, no `Fatal check`, 0.97-1.00x of the task 53 rows; V4.1 `mega_moe` 2191 against Marlin 1912 out tok/s at C=128. The V4.1 subset does not produce "2, 3, 5, 7, 11" (list continuation " 7, 9, 11, ..." as in tasks 46 and 53, chat primes loops), which is the subset model and not a regression (`deepgemm_benchmark.md` 5.2).
- Graph ladder KV cost (task 58, analysis 9.51, `deepgemm_benchmark.md` 3.9): the default ladder to `--max-num-seqs` costs 1.9% (Qwen3.8-27B-FP8 TP1) to 9.8% (V4.1 TP2) KV tokens, but on KV-bound long-context runs (agentX C=16/64, V4.1 RAG C=192 at ~151K tokens) it matches cap 160 (0.98-1.01x out tok/s, 0 errors, no retraction); the default stays. Left: repeated runs for a spread, and a KV-bound workload with more than 160 running requests.
- Push state: tasks 1-55 are pushed to `origin/nctu6` in both repos (DeepGEMM first, then TokenSpeed); task 56 docs are local (not pushed).

## 8. Where to read more

| Topic | Where |
|---|---|
| Why upstream does not run; kernel designs, heuristics, block-scaled GEMM | design 2, 3, 3.4 |
| Indexer kernels and API compatibility | design 4, 5 |
| Bugs found and fixed (proxy fence race, spills, deadlocks) | design 6 |
| All performance tables, per-op table, CuTe headroom | design 7, 7.8 |
| TokenSpeed integration, MXFP4 leaves, install | design 8 |
| Limitations and open items in full | design 9, 10 |
| Reproducing | design 11 |
| Served throughput and latency of the before / after TokenSpeed images, six models | [`deepgemm_benchmark.md`](deepgemm_benchmark.md), analysis 9.44, 9.45 |
| Status log, dead ends, per-task measurements | analysis sections 8, 9, 9.5-9.26 |
