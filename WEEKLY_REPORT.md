# SM120 work: weekly report (tasks 1-18, through 2026-11-09)

## Goal and status

Goal: DeepGEMM fully supports SM120 (RTX PRO 6000 Blackwell), it beats the Triton / FlashInfer
fallbacks, and TokenSpeed has no SM120 issues. Status: tasks 1-18 are done. Every kernel suite
passes, four models give correct end-to-end output, and each op below runs faster than its
fallback except a few ties noted under open items. All commits are local on branch `nctu6` in
DeepGEMM and TokenSpeed; nothing is pushed.

## Highlights

| Area | Result |
|---|---|
| Dense FP8 GEMM | 1.61x geomean over Triton / FlashInfer (112 shapes); decode 1.2-4.0x; end to end TPOT 1.18-1.55x on a 27B model |
| FP4 GEMM | 1.40x geomean over FlashInfer MXFP4, 62 of 63 shapes; 1.3-2.0x over the best FP8 kernel |
| DeepSeek V4.1 indexer | Runs native with the existing 68-byte cache, 2.2-88x over Triton per pass |
| DSA sparse attention (CuTe DSL) | 1.91x (E4M3) / 1.16x (BF16) over the Triton tensor-core kernel; 22-114x over the original |
| V4 prefill / MLA decode (CuTe DSL) | V4 1.13x, MLA 1.46x over the Triton tensor-core kernels |
| DSA prefill, tiny chunks | Now 1.35x faster than Triton at 128 tokens (was 0.52x); `dsa_index_candidates` 1.95x geomean |
| BF16 einsum (batched heads) | 0.21x of the per-head time, bit-identical to the old route |
| V4.1 decode, direct reader | 1.66x geomean, bit-identical to the old route |
| Tests | Kernel suite 7769 passed, 0 failed; runtime suite 0 failed in all 10 directories |

## Key findings

- SM120 has no `wgmma` or `tcgen05`/TMEM and about 99 KB of shared memory per block, so upstream kernels cannot be ported. New kernels use TMA, mbarriers and `mma.sync`, including the hardware block-scaled form for FP4 and FP8 scales.
- The hardware has an 8-bit transposing `ldmatrix`, which made MN-major FP8, k-grouped FP8 and the MN-major `fp8_einsum` possible. TMA multicast is emulated in software, so it is not usable.
- A native paged MQA kernel for MXFP4 rows removed the need to change the V4.1 cache format.
- Small decode shapes lose time in the launch chain, not in the tensor cores. An early PDL trigger turned the FP4 comparison from 59 to 62 wins out of 63.
- The tiny-chunk DSA loss was host time (360 us of launches around 38 us of GPU work), fixed with one preparation launch.
- Mega MoE, mega gate and mega mHC need NVLink symmetric memory or TMEM, so SM120 cannot run them.

## Fixes

- Proxy-fence race that corrupted about 1 run in 20 on the GEMM and both MQA kernels; paged MQA hang at batch 32 and deadlock with next_n 2.
- Register spills after `setmaxnreg` in the GEMMs (three cases, now 0 B), and an `einsum bmk,bnk->mn` segfault.
- DSA sparse attention is batch-invariant (a token's output no longer depends on the batch), at 1.21x geomean cost.
- Whisper GC / graph-capture flake: the cyclic GC freed a dead CUDA graph inside a capture; fixed by pausing GC during capture. Also a fixed-port flake in `test_control_server.py`.
- TokenSpeed: SM100-only kernels no longer get picked on SM120 and crash; NVFP4 checkpoint loading and dense NVFP4 fixes.

## Verification

- TokenSpeed kernel suite 7769 passed, 2770 skipped, 0 failed; runtime suite 0 failed in every directory.
- DeepGEMM SM120 tests pass with the register-spill check on and a fresh JIT cache; compute-sanitizer reports 0 errors on 22 test functions.
- Correct greedy output end to end on Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2), Qwen3.8-27B-NVFP4 and gemma-4-31B-it.

## Open items

- One FP4 tie with FlashInfer (dense gate_up M=256, 0.98x), bound by L2 traffic.
- No real-checkpoint runs for V3.2 TP8 (needs 8 GPUs) or V4.1 (ran with dummy weights).
- Further attention headroom: BF16 DSA / V4 at 42-48% of the tensor-pipe floor, V4 at 16 heads, a multi-wave sparse indexer kernel.
- One test failure seen once and never reproduced (`test_deepseek_v41_model[eager]`); three heavy tests are too slow to finish under compute-sanitizer.
- Pushing the local commits and bumping the TokenSpeed `DeepGEMM` pin.

Details: [`deepgemm_report.md`](deepgemm_report.md), [`deepgemm-sm120-design.md`](deepgemm-sm120-design.md), [`deepgemm-analysis.md`](deepgemm-analysis.md).
