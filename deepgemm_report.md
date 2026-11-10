# DeepGEMM on SM120 (RTX PRO 6000 Blackwell): report

Long form: [`deepgemm-sm120-design.md`](deepgemm-sm120-design.md) (designs, bugs, every table; the
section numbers below like "design 7.8" point there). Status log, dead ends and per-task
measurements: [`deepgemm-analysis.md`](deepgemm-analysis.md) ("analysis 9.x").

## 1. Purpose and status

Upstream DeepGEMM only runs on Hopper and datacenter Blackwell (`wgmma`, `tcgen05`/TMEM). This work
adds SM120 (compute capability 12.0, consumer and workstation Blackwell) with new kernels built on
TMA, mbarrier pipelines and `mma.sync`, plus arch-12 host paths, so TokenSpeed can serve
DeepSeek-style models on RTX PRO 6000. Tasks 1-18 are done: the GEMMs, the indexer kernels and the
TokenSpeed attention leaves run on SM120 and beat the Triton / FlashInfer fallbacks everywhere
except a few ties and losses listed below. All commits are local on `nctu6` (DeepGEMM and
TokenSpeed); nothing is pushed.

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
| `fp8_fp4_mega_moe` (MXFP4 routed experts; task 21) | supported as a sequence of kernels with the SM100 arithmetic, on one rank or across up to 8 ranks (run on 2) with CUDA IPC peer stores over PCIe; bitwise equal to one rank, CUDA-graph safe. Fused shared experts are rejected |
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
| MXFP8 dense GEMM, 1x32 UE8M0 both sides (V4.1 projections, task 24) | `deepgemm_sm120_mm_mxfp8` (DeepGEMM 1D1D, deterministic, batch invariant) | `triton_mm_fp8_blockscale` (the only SM120 leaf before) | 2.2-11.7x (6 projections x M = 1-2048, graph, cold weights) | 2.2x `shared_w2` M=256 |
| MXFP4 linear leaf (with quantization) | `deepgemm_sm120_mxfp4_linear` | FlashInfer quantize + best `mm_fp4` | 1.23x geomean, 60 of 63 wins | 0.90x q_b M=256 |
| MN-major FP8 / k-grouped FP8 | DeepGEMM | explicit transpose + K-major kernel | 1.63x (B, M <= 64) / 1.11-2.06x | 0.97x (B operand) / 0.76x vs a torch loop at small K |
| `fp8_einsum` (3 expressions) | batched launch | best of cuBLASLt BF16, torch loop, `torch.einsum` | 1.52x geomean (26 shapes) | 0.99x `bhd,bhr->hdr` h=8, b=128 (tie) |
| BF16 `einsum` heads (task 18) | batched BF16 launch, bit-identical to per-head | per-head GEMM calls; cuBLASLt | 0.21x of the per-head time, 4.8x faster (56 points); 1.06x vs cuBLASLt, 52 of 56 within 3% or faster | 0.81x vs cuBLASLt, 16 heads, b = 1 |
| V3.2 indexer (MQA logits + top-k) | DeepGEMM logits | Triton DSA indexer | decode 3.4-19.3x, prefill 7.8-52x | 3.4x decode, 1 x 4K |
| V3.2 DSA prefill, tiny chunks (task 18) | `deep_gemm_dsa_prefill_topk`, host-light | `triton_dsa_prefill_topk_fp8` | 1.35-98x at 1-8192 tokens; 1.35x at 128 tokens (was 0.52x) | 1.35x, 128 tokens, no prefix |
| `dsa_index_candidates` (task 18) | DeepSelect over owned length | `torch.topk` leaf; Triton | 1.95x geomean (1.25-4.7x, 54 points); 18x vs Triton | 1.25x, 8 rows, 128K context, 2 ranks |
| V4.1 indexer, 68-byte rows (task 16) | `deep_gemm_sm120_dsv41_index_topk` | `triton_dsv41_index_topk` | 2.2-88x per pass: decode Full 5.6-16x, source 7.7-33x, Reindex 2.2-22x; prefill 11-46x, 22-88x, 6.8-22x | 2.2x Reindex, 1-8 requests |
| DSA sparse attention (task 17) | `cute_dsl_dsa_mma_*` | `triton_dsa_mma_*` (64 buckets) | E4M3 1.91x, BF16 1.16x geomean, 1.00-2.34x | 1.00x BF16 16 heads |
| DSA CuTe launch planner in SM waves (task 24 E) | `cute_dsl_dsa_mma_*`, new `plan_launch` | the same cores, earlier planner (140 decode shapes of 1-192 tokens) | 1.05x BF16 / 1.04x E4M3 geomean, up to 1.63x; 1.02x at 128-2048 tokens; rows bitwise equal | 0.97x (noise) |
| DSA sparse attention | `triton_dsa_mma_*` | portable `triton_dsa_*` | 22-114x, 72x geomean (batch-invariant) | 22x BF16 decode, 16 heads, 1 request |
| MLA decode (task 17) | `cute_dsl_mla_mma_decode_with_kvcache` | `triton_mla_mma_decode_with_kvcache` | 1.46x geomean (128 points), 1.84x on 4/8-row blocks | 0.97x BF16 16 heads, 2 x 1K |
| V4 / V4.1 prefill attention (task 17) | `cute_dsl_dsv4_mma_prefill` (> 16 heads) | `triton_dsv4_mma_prefill` | 1.13x geomean (117 points), 0.89-1.63x | 0.89x window-only rows at 1-2 tokens |
| V4.1 decode, packed rows (task 18) | `triton_dsv41_mma_selected_attention` | gather into BF16 workspace + same kernel | 1.66x geomean (21 points), 1.01-3.67x | 1.01x, 128 heads, 128 tokens |
| MLA decode / V4 prefill / Hadamard (Triton, task 14) | `triton_*_mma_*` | portable Triton leaves | MLA 14-131x, V4 4-9x, Hadamard 55-117x from 512 tokens | Hadamard 0.9x at 1 token (launch-bound) |
| mHC | `tf32_hc_prenorm_gemm` leaves | Triton | `mhc_pre` 1.5-13.6x, `mhc_mixes` 1.6-15x | 1.5x |

End to end (in-process `Engine`):
- Qwen3.8-27B-FP8 (the block-FP8 path of DeepSeek-V3): TPOT 1.18-1.55x better (design 7.4).
- 4-layer V3.2, real dimensions, random weights, against forced `triton_dsa_*`: prefill 82-95x, TPOT 2.0-14.4x (analysis 9.18).
- V4.1 topology with dummy weights on the native indexer: prefill 1.11-1.19x, TPOT 1.02-1.06x, same greedy tokens (analysis 9.21).
- Real V4.1 checkpoint (all 40 layers, Engram, head; first 128 of 384 experts), task 22: every path has the same single-layer error against a float32 reference built from the same weights (2.8% layer 0, 3.7-3.9% layer 2). Task 24 (analysis 9.29) profiled it and moved the dense projections to DeepGEMM: Marlin TP2 14.5-14.8K tok/s prefill, 10.4-10.7 ms TPOT (was 10.2-10.7K, 23-25 ms); MegaMoE DP2/EP2 18.1-18.6K tok/s (1.25x over Marlin), 13.8-14.0 ms TPOT (1.31x behind: DP attention reads every dense weight per GPU, plus MegaMoE glue).
- Real MLA checkpoint (Moonlight-16B-A3B, one GPU): correct answers; TPOT 2.3-3.2x better than the portable MLA decode at 3K context x 8, a tie at short context (analysis 9.27).

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

Latest run 2026-11-10 (task 26), DeepGEMM `b6247c2` and TokenSpeed `0d2f40c5b`.
- TokenSpeed kernel suite: 7898 passed, 2770 skipped, 0 failed (GPU 7).
- TokenSpeed runtime suite, 0 failed in every directory (9003 passed): cache 137, diffusion 53, distributed 638, entrypoints 40, execution 195, layers 370, models 567, moe 72, sampling 144, top-level files 6787.
- DeepGEMM (`DG_JIT_CHECK_NO_SPILLS=1`, fresh JIT cache, GPU 7): `test_fp8_fp4`, `test_bf16`, `test_einsum`, `test_layout` (one skip: packed k-grouped scales are SM100 only), `test_hyperconnection`, `test_attention` and `test_mega_moe_sm120` exit 0; `test_mega_moe_sm120_ep.py` (2 ranks, GPUs 6+7) is bitwise equal to one rank; `sm120_mqa_check.py` 46 cases, `sm120_blockscale_check.py --mode all --grouped` 170 cases and `sm120_paged_mqa_blockscale_check.py` 51 cases all OK; `test_lazy_init` on GPUs 6 and 7 (3 tests) OK.
- The DeepGEMM source at the pinned commit builds like `install.sh` (`pip wheel`, nothing installed) and `test_deepgemm_sm120_gating.py` passes against it (11 passed).
- compute-sanitizer (memcheck, synccheck): 0 errors on 22 test functions and a 16-case k-grouped FP8 run (task 15; not rerun in task 20).
- End to end, greedy, in-process, all correct ("Paris", "2, 3, 5, 7, 11"): Qwen3.6-35B-A3B, Qwen3.5-122B-A10B-NVFP4 (TP2), Qwen3.8-27B-NVFP4, gemma-4-31B-it.
- Real weights (task 22, analysis 9.27): V4.1 checkpoint on GPUs 6+7 with its first 128 experts. MegaMoE DP2/EP2 (`deep_gemm_sm120_mxfp4_mega_moe_apply` selected), Marlin TP2 and forced portable TP2 all have the same single-layer error against a float32 reference (layers 0 and 2). Moonlight-16B MLA answers correctly; its greedy decode matches the portable path up to near-ties, and two SM120 runs are bit-identical. New TokenSpeed tests: the V4.1 expert-subset loader and DeepSeek-V3 plain-RoPE configs (`e4fc05813`, `6d6702bae`). The V4 checkpoint does not load (TokenSpeed has no V4 model class since upstream `0c1bc3ed3`).
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

- One FP4 tie with FlashInfer b12x: dense gate_up M=256 (0.98x in the sweep, 1.02x isolated), bound by L2 -> SM traffic (analysis 9.15). Cluster tile sharing over DSMEM (task 23, analysis 9.28) is 2.3-7.3x slower, because DSMEM writes run at 5 GB/s per CTA and the hops lengthen the 3-stage pipeline; the ceiling with free sharing is 1.13x. Not shipped (probe and patch in `scripts/`).
- Mega MoE (analysis 9.26, 9.29): the decode glue (0.82x at one rank and 0.95x at EP 2 on the leaf; in serving 2.3 ms per step at 2 requests around GEMMs that tie Marlin; a graph timeline puts 25 us per layer on the critical path and fusing push + barrier + dispatch + gather and return + barrier + combine would save 9 us, 0.36 ms per step or 2.6% of TPOT, so it was not built: analysis 9.30), more than 2 ranks, fused shared experts, mega gate and mega mHC (no TokenSpeed caller). TPOT parity with Marlin TP2 at 2 requests is not reachable with attention DP (+2.6 ms per step of dense-weight reads, analysis 9.29). Done: EP 2 over CUDA IPC; the Triton MXFP4 MoE illegal memory access (TokenSpeed `8afd973bc`); the persistent row scratch counted by the KV planner and the DP2/EP2 prefill gap (task 24).
- V4.1 indexer: a multi-wave sparse MX kernel (the sparse kernel is one wave, about 85 us at any size); MXFP4 paged at 1.07 vs 1.23 TB/s for FP8; FP8 `index_v4` rows have no pool-only scorer (the runtime does not store them on SM120).
- Attention (task 24 item E, analysis 9.30): measured and closed: BF16 DSA / V4 stay at 42-48% of the tensor-pipe floor (bound by the per-tile chain and the L2 gather; the partial-score exchange is only 17%), V4 at 16 heads stays on Triton (8-warp CTA 0.76-1.10x), one-request DSA decode stays at 20 us BF16 / 13 us E4M3 (16 chunks would give 14 / 10 us and lose 0.8-0.9x at 4-64 tokens). Shipped: the DSA CuTe launch planner follows SM waves (1.05x over 1-192 tokens). Open: a dequantizing tile in the CuTe DSL V4 kernel; host paths of `fp8_mqa_logits` (26 us) and DeepSelect (34 us) in a 128-token DSA chunk.
- Full-model real-weight runs: V3.2 TP8 (no checkpoint on the box, about 690 GB FP8, all 8 GPUs) and V4.1 with all 384 experts (4 GPUs). Done in task 22: real V4.1 with 128 experts on two GPUs, and real MLA on Moonlight-16B. V4 cannot load (no TokenSpeed model class). No V3.2, GLM-5.x, dots3-note or V2-Lite checkpoint is on the box (analysis 9.27).
- Three heavy tests (FP8 k-grouped, FP8 masked, `test_mqa_logits`) do not finish under compute-sanitizer in 15-40 min; no error before the limit.
- `test_deepseek_v41_model[eager]` failed once in three full runs during task 14 and never reproduced. Task 25 (analysis 9.31) closed it as not reproducible: sanitizers (memcheck, racecheck, initcheck, synccheck) clean, poisoned allocations and stream delays change no bit, about 1400 eager runs and 46 whole-file runs pass; the original failure text is lost, so a recurrence needs its output kept.
- Nothing is pushed. TokenSpeed's `AGENTS.md` lists NVIDIA support as sm90/sm100/sm103/sm107, so upstreaming needs that scope to change.

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
| Status log, dead ends, per-task measurements | analysis sections 8, 9, 9.5-9.26 |
