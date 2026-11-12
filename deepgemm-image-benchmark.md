# Serving benchmark of two TokenSpeed images on RTX PRO 6000 (SM120)

Tasks 40-47 of the SM120 spec, run 2026-11-12 on aws-raichu, GPUs 6 and 7. Part of the DeepGEMM SM120
work: [`deepgemm_report.md`](deepgemm_report.md) (summary), [`deepgemm-analysis.md`](deepgemm-analysis.md) 9.44
(status log entry).

The question: what do the SM120 kernel and runtime changes (DeepGEMM SM120 pin, CuTe DSL / Triton
attention leaves, NVFP4 and V4.1 fixes) change for served throughput and latency? Two images, six models,
two client workloads, the same server flags in both.

## 1. Summary

| | before | after |
|---|---|---|
| Image | `nctu6/tokenspeed:e27cee86-20261028` | `nctu6/tokenspeed:ef057889-20261112` |
| TokenSpeed | `e27cee861`, no DeepGEMM SM120 | `ef057889f`, DeepGEMM pin `976f212` |

Which models run:

| Model | TP | before | after |
|---|---|---|---|
| Qwen3.8-27B-FP8 | 1 | runs | runs |
| Qwen3.6-35B-A3B (BF16, control: no FP8/FP4 GEMM) | 1 | runs | runs |
| gemma-4-31B-it (BF16) | 1 | RAG all C; agentX C=1 only | same |
| Qwen3.8-27B-NVFP4 | 1 | **fails to start** (`fp4_quantize`: "Kernel implementation not found") | runs |
| Qwen3.5-122B-A10B-NVFP4 | 2 | **fails to start** (flashinfer NVFP4 MoE check `fc1_act_global`) | runs |
| DeepSeek-V4.1-Flash, 128 of 384 experts | 2 | **fails to load** (`Unexpected V4.1 expert tensor ...experts.128...`) | runs (Marlin `auto` and `mega_moe`) |

Ratios are after over before for output tok/s, and before over after for TTFT p50 and TPOT p50, so a value
above 1 always means the after image is better. One run per point.

| Model | RAG out tok/s, C = 1 / 16 / 64 / 256 / 512 | RAG TPOT p50 ratio, C = 1 .. 512 | RAG TTFT p50 ratio | agentX out tok/s, C = 1 / 16 / 64 |
|---|---|---|---|---|
| Qwen3.8-27B-FP8 | **1.52x** / 1.28x / 1.11x / 1.04x / 1.04x | 1.53x at C=1, 1.31x at 16, 1.04-1.06x from 128 (agentX: 1.41x / 1.13x / 1.08x) | 0.98-1.01x | 1.14x / 1.10x / 1.01x |
| Qwen3.6-35B-A3B (BF16) | 1.01x / 1.01x / 1.00x / 1.00x / 1.00x | 1.00-1.01x (agentX: 1.04-1.06x) | 0.98-1.11x (noise) | 1.02x / 0.94x / 1.03x |
| gemma-4-31B-it (BF16) | 1.03x / 1.03x / 1.00x / 1.02x / 1.03x | 1.00-1.03x | 0.94-1.03x | 1.00x (C=1 only) |
| Qwen3.8-27B-NVFP4 | after only: 55 / 618 / 1321 / 1610 / 1612 tok/s | after only | after only | after only: 24 / 40 / 8.6 tok/s |
| Qwen3.5-122B-A10B-NVFP4 (TP2) | after only: 133 / 803 / 1480 / 2282 / 2256 tok/s | after only | after only | after only: 64 / 119 / 15 tok/s |
| DeepSeek-V4.1-Flash subset (TP2, Marlin) | after only: 105 / 613 / 1429 / 1650 / 2693 tok/s | after only | after only | after only: 90 / 151 / 524 tok/s |

Mega MoE against Marlin on V4.1 (both on the after image): RAG output tok/s 1.01x, 1.02x, 1.03x, 1.07x, **1.17x**,
1.04x, 1.04x at C = 1 .. 512; agentX 0.97x, 1.00x, 1.03x. Table in section 3.6.

Reading:
- The only model whose speed changes is the block-FP8 one. Its gain is in decode (TPOT 1.53x at C=1) and
  shrinks as the decode batch grows (1.04x at C=512), while TTFT p50 stays at 1.00x (prefill is at parity, as in
  the kernel tables of the report). The BF16 control (Qwen3.6-35B-A3B) and gemma-4 sit at 1.00-1.03x, so the
  1.52x is not a host, image or GPU effect (section 5). In agentX the control's TPOT p50 is 4-6% better on the
  after image in all three runs (cause not investigated), so part of the FP8 agentX TPOT ratios (1.08-1.13x at
  C=16 and 64) is not FP8.
- The after image is what makes three of six models run at all: both NVFP4 models and the V4.1 subset fail to
  start in the before image (section 4.1).
- Throughput saturates at high concurrency for reasons that are not the image (KV pool full, decode batch above
  the largest CUDA graph; sections 3 and 4.6). Read the ratios at C >= 128 as ratios of a saturated server.
- A scheduler crash (KV events) hits both images at C >= 128 in the default configuration. Every table below
  uses the workaround (section 2.4, 4.3).

## 2. Setup

### 2.1 Hardware, images, versions

- aws-raichu, 8x RTX PRO 6000 Blackwell Server Edition (96 GB), PCIe only; only GPUs 6 and 7 used (same PCIe
  switch). TP1 sessions on either GPU (recorded per run in `metrics.json` `gpus`), TP2 on both.
- Image IDs (from `session.json`): before `sha256:2b655629ac34c6723dc8ae6cf3c54a5ee67e354bd1f75fa485f2c5f179bce9d4`,
  after `sha256:349e1a2b79973c6446bb5ece04d7e6cf2b91cb110e0ba7fd0193e93d92c300bf`.
- Client: AIPerf 0.13.0 (own venv; from `metrics.json` `aiperf_version`), same venv for every run.
- GPU used per dataset (TP1):

| Model | RAG before / after | agentX before / after |
|---|---|---|
| Qwen3.8-27B-FP8 | GPU 7 / GPU 6 | GPU 6 / GPU 7 |
| Qwen3.6-35B-A3B | GPU 7 / GPU 7 | GPU 7 / GPU 7 |
| gemma-4-31B-it | GPU 6 / GPU 7 | GPU 6 / GPU 7 |
| Qwen3.8-27B-NVFP4 | - / GPU 6 | - / GPU 6 |

The FP8 model swapped GPUs between its RAG and agentX sessions, so the FP8 ratios do not come from one GPU
being faster. Qwen3.6 ran on one GPU in both images and shows 1.00x.

### 2.2 Server

One `docker run` per (image, model): `--gpus` limited to the listed GPUs, `--ipc=host`, `/models` and the HF
cache mounted read-only, `-e HF_HUB_OFFLINE=1 -e FLASHINFER_DISABLE_VERSION_CHECK=1`. Identical arguments in
both images:

```
tokenspeed serve <model path> --served-model-name <name> --host 0.0.0.0 --port 8000 --trust-remote-code \
  --tensor-parallel-size <1|2> --max-num-seqs 512 --gpu-memory-utilization 0.9 --max-model-len 262144
```

Extra, per model, also identical in both images: gemma-4 `--tool-call-parser gemma4`; V4.1
`--engram-host-table --engram-host-table-layout sharded --hf-overrides '{"n_routed_experts": 128,
"num_checkpoint_experts": 384}'`; the V4.1 mega MoE variant adds `--moe-backend mega_moe` (default `auto` selects
Marlin). CUDA graphs are on and prefix caching is on (defaults). The server picks `max_cudagraph_capture_size=160`
(its default; graphs for decode batches 1 .. 160) while `--max-num-seqs` is 512 (section 4.6).
Command lines: `<model>/<image>/server_cmd.txt` in the result tree.

Before every benchmark: health check and two greedy probes (France capital, first primes; both must pass; V4.1
records only, its 128-expert output is not meaningful). A container is removed after its session. Startup was
196-552 s (V4.1 about 540 s, TP2 122B 552 s, 27B FP8 336-432 s).

### 2.3 Workloads

- RAG: `neural-bridge/rag-dataset-12000`, prompt = context + question, 500 prompts (100 at C=1), seed 0,
  256 output tokens, `ignore_eos`, C = 1, 16, 32, 64, 128, 256, 512. Mean input about 820 tokens (Qwen).
  gemma-4 prompts start with a literal `<bos>` (section 4.4).
- agentX: AIPerf scenario `inferencex-agentx-mvp`, public dataset `semianalysis_cc_traces_weka_062126_256k`,
  official full mode (`--benchmark-duration 900`, full corpus, `--random-seed 1110`, no `--unsafe-override`),
  C = 1, 16, 64, `submission_valid` checked per run (all completed runs are valid). Mean input 88K tokens at
  C=16 on Qwen3.8-27B-FP8. Warmup of one request per trajectory lane takes 10 minutes or more at C=16, so a
  run is 16-65 minutes. "Requests done" below is `ok` in the 900 s profiling window.
- Metrics from AIPerf per run: request throughput, output tok/s, TTFT, TPOT (p50, p99), effective concurrency
  (time-average of in-flight requests, `effective_concurrency_avg`), ok and error counts. Token counts come from
  the server (`--use-server-token-count`).

### 2.4 Harness behavior and the KV-events workaround

- Harness: `run_bench.py` (a session: start, probe, run all datasets and concurrencies, tear down) and
  `bench_queue.py` (a resumable queue; one worker per GPU in tmux). A run writes `DONE` or `FAILED`,
  `metrics.json` and the AIPerf artifacts. Details: Reproduce (section 6).
- Watchdog: the gateway keeps the container "Up" and `/health` green after the engine aborts. During a run the
  harness polls the server log every 5 s for fatal lines and probes a 1-token completion when the log is quiet;
  on a dead engine it kills the client, marks the run `FAILED` (`engine crash`), restarts the server and
  continues with the next run (at most 3 restarts per session). No primary-set session needed a restart
  (`restarts: 0` in every `session.json`).
- KV events: the `smg_grpc_servicer` that wraps `tokenspeed serve` turns KV-event publishing on when no
  `--kv-events-config` is given, and with it on the scheduler aborts under RAG load at C >= 128 (finding in
  section 4.3). All primary results set `-e SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` in both images (the variable has
  the same name and meaning in both). Prefix caching and the L2 host cache stay on.

## 3. Results

Cell format `before / after` (after only where the before image does not start). Ratio columns as in
section 1: out tok/s after over before, TTFT p50 and TPOT p50 before over after. `eff. C` is the average number
of requests in flight; `ok/err` the request counts. TTFT p99 and TPOT p50 are ms.

### 3.1 Qwen3.8-27B-FP8 (TP1)

RAG:

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.12 / 0.19 | 31.3 / 47.7 | 103 / 103 | 167 / 172 | 31.6 / 20.6 | 1.0 / 1.0 | 100/0 / 100/0 | 1.52x | 1.00x | 1.53x |
| 16 | 1.57 / 2.01 | 402.9 / 515.3 | 970 / 979 | 2083 / 1811 | 34.8 / 26.6 | 15.7 / 15.7 | 500/0 / 500/0 | 1.28x | 0.99x | 1.31x |
| 32 | 2.77 / 3.11 | 709.3 / 795.1 | 1796 / 1803 | 2831 / 2854 | 37.4 / 32.6 | 31.3 / 31.4 | 500/0 / 500/0 | 1.12x | 1.00x | 1.15x |
| 64 | 3.76 / 4.16 | 962.2 / 1066.2 | 2822 / 2804 | 5522 / 5469 | 53.2 / 47.1 | 62.6 / 62.6 | 500/0 / 500/0 | 1.11x | 1.01x | 1.13x |
| 128 | 4.76 / 5.01 | 1219.4 / 1282.8 | 5798 / 5915 | 10831 / 10758 | 80.4 / 75.5 | 125.0 / 125.0 | 500/0 / 500/0 | 1.05x | 0.98x | 1.06x |
| 256 | 4.98 / 5.17 | 1275.4 / 1324.7 | 11421 / 11276 | 43413 / 42493 | 127.8 / 120.1 | 227.4 / 226.5 | 500/0 / 500/0 | 1.04x | 1.01x | 1.06x |
| 512 | 4.98 / 5.17 | 1275.8 / 1323.9 | 43074 / 42539 | 86664 / 84441 | 129.8 / 125.1 | 344.9 / 348.7 | 500/0 / 500/0 | 1.04x | 1.01x | 1.04x |

agentX (full mode, all runs `submission_valid=true`; GPU 6 before, GPU 7 after):

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | valid (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.015 / 0.018 | 21.2 / 24.2 | 3265 / 2703 | 28154 / 27062 | 39.2 / 27.8 | 0.9 / 0.8 | 14/0 / 17/0 | yes / yes | 1.14x | 1.21x | 1.41x |
| 16 | 0.056 / 0.057 | 29.0 / 31.8 | 1051 / 2519 | 214469 / 208692 | 94.1 / 83.0 | 6.3 / 6.2 | 53/0 / 54/0 | yes / yes | 1.10x | 0.42x | 1.13x |
| 64 | 0.015 / 0.016 | 10.2 / 10.3 | 398631 / 398626 | 641302 / 801530 | 243.9 / 225.8 | 8.0 / 8.5 | 14/0 / 15/0 | yes / yes | 1.01x | 1.00x | 1.08x |

Notes:
- The gain is decode: TPOT 31.6 -> 20.6 ms at C=1 (1.53x), TTFT unchanged (prefill is at kernel parity). The gain
  shrinks as the batch grows (TPOT 1.31x at C=16, 1.13x at 64, 1.04-1.06x from 128) because attention, the
  GatedDeltaNet layers and the scheduler take a larger part of the step.
- Throughput stops growing at C=256 (about 1275 / 1324 tok/s): the KV pool is full (567 of 569 pages active, 49
  requests queued, at most 207 running in the server log) and the decode batch is above the 160 graph limit
  (section 4.6).
- agentX: effective concurrency stays at 6-8 for C=16 and 64 (requests wait for 256K-token prefills); at C=64
  TTFT p50 is 399 s and only 14-15 requests finish in the 900 s window. With one run per point, the C=16 TTFT
  p50 difference (1051 / 2519 ms, "0.42x") is within the spread of a heavy-tailed distribution (p99 is 214 / 209 s);
  do not read it as a regression.
- Smoke-mode agentX of the first runs (`agentx_smoke/`, 8 entries, 240 s, `--unsafe-override`): c1 19.0 / 19.8
  out tok/s (before / after), c16 0.7 / 0.7, effective concurrency 0.7-0.9, `submission_valid=false`; c32 failed (AIPerf
  wrote no export), c64 was aborted. They are not submission-valid and cannot tell C=16 from C=512 (8 dataset
  entries), so they are not compared and not used in any ratio.

### 3.2 Qwen3.6-35B-A3B (TP1, BF16 control)

RAG:

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.66 / 0.66 | 168.2 / 170.1 | 64 / 64 | 86 / 82 | 5.7 / 5.6 | 1.0 / 1.0 | 100/0 / 100/0 | 1.01x | 1.00x | 1.01x |
| 16 | 2.78 / 2.82 | 712.5 / 721.0 | 296 / 292 | 798 / 788 | 21.0 / 20.8 | 15.8 / 15.8 | 500/0 / 500/0 | 1.01x | 1.01x | 1.01x |
| 32 | 3.75 / 3.76 | 960.2 / 962.1 | 517 / 515 | 825 / 825 | 31.0 / 30.8 | 31.4 / 31.4 | 500/0 / 500/0 | 1.00x | 1.00x | 1.00x |
| 64 | 5.30 / 5.31 | 1358.1 / 1360.2 | 862 / 824 | 1502 / 1506 | 43.0 / 42.9 | 62.6 / 62.6 | 500/0 / 500/0 | 1.00x | 1.05x | 1.00x |
| 128 | 7.57 / 7.57 | 1936.8 / 1938.5 | 1667 / 1506 | 2925 / 2902 | 58.4 / 58.4 | 125.0 / 125.1 | 500/0 / 500/0 | 1.00x | 1.11x | 1.00x |
| 256 | 7.73 / 7.73 | 1979.9 / 1977.7 | 3838 / 3900 | 20848 / 20939 | 67.2 / 67.1 | 208.6 / 208.3 | 500/0 / 500/0 | 1.00x | 0.98x | 1.00x |
| 512 | 7.69 / 7.71 | 1968.9 / 1972.7 | 20951 / 20823 | 57259 / 57314 | 67.5 / 66.9 | 305.0 / 304.3 | 500/0 / 500/0 | 1.00x | 1.01x | 1.01x |

agentX:

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | valid (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.100 / 0.102 | 71.7 / 73.0 | 709 / 733 | 12638 / 33205 | 12.4 / 11.9 | 1.0 / 1.0 | 93/0 / 94/0 | yes / yes | 1.02x | 0.97x | 1.04x |
| 16 | 0.120 / 0.120 | 108.1 / 101.1 | 848 / 871 | 59702 / 59098 | 22.7 / 21.6 | 3.9 / 3.8 | 112/0 / 113/0 | yes / yes | 0.94x | 0.97x | 1.05x |
| 64 | 0.053 / 0.054 | 43.7 / 44.9 | 293753 / 291473 | 549077 / 538605 | 57.0 / 53.9 | 19.1 / 19.3 | 50/0 / 51/0 | yes / yes | 1.03x | 1.01x | 1.06x |

Both images give the same numbers within the run-to-run spread (RAG out tok/s 1.00-1.01x at every C; TTFT p50 ratio
0.98-1.11x is noise on 300 ms-4 s values). No FP8/FP4 GEMM and no DeepGEMM path is used by this model, so this
is the control for host, driver and load effects. Plateau from C=256: the KV pool is full (403 of 405 pages)
with about 158 running requests.

### 3.3 gemma-4-31B-it (TP1, BF16)

RAG (prompts with a leading `<bos>`):

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.09 / 0.09 | 22.2 / 23.0 | 175 / 173 | 296 / 291 | 44.4 / 43.0 | 1.0 / 1.0 | 100/0 / 100/0 | 1.03x | 1.01x | 1.03x |
| 16 | 1.00 / 1.03 | 256.6 / 263.3 | 633 / 648 | 3019 / 2712 | 58.8 / 57.2 | 15.7 / 15.7 | 500/0 / 500/0 | 1.03x | 0.98x | 1.03x |
| 32 | 1.43 / 1.45 | 365.8 / 371.4 | 1428 / 1514 | 9625 / 7776 | 73.6 / 73.0 | 31.5 / 31.5 | 500/0 / 500/0 | 1.02x | 0.94x | 1.01x |
| 64 | 1.44 / 1.45 | 369.5 / 370.8 | 22836 / 22735 | 37947 / 38750 | 73.8 / 73.3 | 61.1 / 61.1 | 500/0 / 500/0 | 1.00x | 1.00x | 1.01x |
| 128 | 1.43 / 1.44 | 364.8 / 369.3 | 64409 / 64567 | 82344 / 81279 | 73.7 / 73.6 | 113.8 / 114.0 | 500/0 / 500/0 | 1.01x | 1.00x | 1.00x |
| 256 | 1.43 / 1.45 | 365.8 / 371.6 | 148361 / 146598 | 164578 / 164820 | 73.6 / 73.3 | 197.0 / 196.6 | 500/0 / 500/0 | 1.02x | 1.01x | 1.00x |
| 512 | 1.41 / 1.45 | 361.8 / 370.9 | 171128 / 166253 | 337140 / 329356 | 74.2 / 73.0 | 268.4 / 269.4 | 500/0 / 500/0 | 1.03x | 1.03x | 1.02x |

agentX (C=1 completes in both images; C=16 and 64 are not runnable at TP1, section 4.2):

| C | req/s (before / after) | out tok/s (before / after) | TTFT p50 ms (before / after) | TTFT p99 ms (before / after) | TPOT p50 ms (before / after) | eff. C (before / after) | ok/err (before / after) | valid (before / after) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.012 / 0.012 | 16.0 / 16.0 | 2254 / 2236 | 12284 / 12226 | 52.0 / 50.5 | 0.9 / 0.9 | 11/0 / 11/0 | yes / yes | 1.00x | 1.01x | 1.03x |
| 16 | FAILED/FAILED | - | - | - | - | - | - | - | - | - | - |
| 64 | FAILED/FAILED | - | - | - | - | - | - | - | - | - | - |

Effective concurrency caps: the server runs about 27 requests at a time at every C >= 32 (output tok/s times
TPOT; the log shows at most 34 running and 451 queued, with 2291 of 2295 KV pages active), so throughput stays at
about 365-371 tok/s and TPOT at 73-74 ms while TTFT p50 grows with the queue (before image: 1.4 s at C=32, 22.8 s at 64,
171 s at 512). The cap is the KV pool (token capacity 293,760 at TP1 and 0.9 memory use), the same in both
images. I did not trace why a ~1.1K-token request holds that many pages (the cache has a full-attention group and five
sliding-window groups).

### 3.4 Qwen3.8-27B-NVFP4 (TP1, after image only)

The before image does not start (section 4.1).

RAG:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) |
|---|---|---|---|---|---|---|---|
| 1 | 0.22 | 55.3 | 74 | 119 | 17.7 | 1.0 | 100/0 |
| 16 | 2.41 | 618.2 | 729 | 1246 | 22.4 | 15.7 | 500/0 |
| 32 | 3.77 | 966.1 | 1346 | 2099 | 27.5 | 31.4 | 500/0 |
| 64 | 5.16 | 1320.6 | 2217 | 3898 | 38.8 | 62.6 | 500/0 |
| 128 | 6.07 | 1554.5 | 3873 | 7890 | 64.6 | 125.1 | 500/0 |
| 256 | 6.29 | 1609.5 | 7733 | 36595 | 114.6 | 235.4 | 500/0 |
| 512 | 6.30 | 1611.6 | 36700 | 73077 | 117.2 | 355.7 | 500/0 |

agentX:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) | valid (after) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.018 | 24.2 | 2581 | 24432 | 25.0 | 0.7 | 17/0 | yes |
| 16 | 0.073 | 40.2 | 1009 | 183150 | 67.8 | 6.2 | 69/0 | yes |
| 64 | 0.020 | 8.6 | 404731 | 827134 | 147.1 | 11.6 | 19/0 | yes |

For orientation only (different checkpoints, different GPUs): on the after image the NVFP4 checkpoint gives 1.16x
the FP8 checkpoint's output tok/s at C=1 (55.3 vs 47.7), 1.22x at C=512 (1612 vs 1324).

### 3.5 Qwen3.5-122B-A10B-NVFP4 (TP2, after image only)

RAG:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) |
|---|---|---|---|---|---|---|---|
| 1 | 0.52 | 133.4 | 83 | 114 | 7.2 | 1.0 | 100/0 |
| 16 | 3.14 | 802.6 | 648 | 1243 | 17.0 | 15.8 | 500/0 |
| 32 | 4.29 | 1098.1 | 984 | 1573 | 24.8 | 31.4 | 500/0 |
| 64 | 5.78 | 1480.1 | 1527 | 2706 | 35.9 | 62.6 | 500/0 |
| 128 | 7.47 | 1912.2 | 2947 | 5361 | 53.6 | 125.0 | 500/0 |
| 256 | 8.91 | 2281.5 | 5606 | 10427 | 86.7 | 249.6 | 500/0 |
| 512 | 8.81 | 2255.7 | 10803 | 45482 | 115.1 | 390.5 | 500/0 |

agentX:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) | valid (after) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.098 | 63.7 | 919 | 5272 | 13.3 | 1.0 | 91/0 | yes |
| 16 | 0.137 | 118.6 | 720 | 78942 | 18.8 | 3.3 | 127/0 | yes |
| 64 | 0.057 | 14.9 | 123098 | 431467 | 583.7 | 22.4 | 54/0 | yes |

The 122B MoE keeps scaling to C=256 (2282 tok/s). agentX C=64 is saturated: TTFT p50 123 s, TPOT p50 584 ms,
effective concurrency 22 of 64, 54 requests in 900 s.

### 3.6 DeepSeek-V4.1-Flash, 128-expert subset (TP2, after image only)

Not a real model output: 128 of the 384 routed experts per layer (`n_routed_experts=128`,
`num_checkpoint_experts=384`), so the greedy probe is recorded and not gated. The speed is meaningful as a
128-expert V4.1 topology with real weights for everything else.

Default MoE backend (`auto` = Marlin):

RAG:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) |
|---|---|---|---|---|---|---|---|
| 1 | 0.41 | 104.5 | 98 | 1277 | 9.0 | 1.0 | 100/0 |
| 16 | 2.39 | 612.8 | 658 | 2236 | 22.8 | 15.8 | 500/0 |
| 32 | 3.81 | 974.9 | 815 | 891 | 29.4 | 31.3 | 500/0 |
| 64 | 5.58 | 1429.0 | 1199 | 1622 | 39.5 | 62.6 | 500/0 |
| 128 | 7.43 | 1901.0 | 2045 | 3817 | 57.9 | 125.0 | 500/0 |
| 256 | 6.44 | 1649.8 | 3185 | 5804 | 139.8 | 249.9 | 500/0 |
| 512 | 10.52 | 2692.5 | 6547 | 11420 | 160.2 | 499.6 | 500/0 |

agentX:

| C | req/s (after) | out tok/s (after) | TTFT p50 ms (after) | TTFT p99 ms (after) | TPOT p50 ms (after) | eff. C (after) | ok/err (after) | valid (after) |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.111 | 89.6 | 187 | 1483 | 11.3 | 0.9 | 100/0 | yes |
| 16 | 0.188 | 150.8 | 190 | 3992 | 11.5 | 1.8 | 175/0 | yes |
| 64 | 0.696 | 524.2 | 299 | 6042 | 34.3 | 19.4 | 654/0 | yes |

Mega MoE (`--moe-backend mega_moe`, attention TP2 + MoE TP2) against Marlin, same image. Cell format
`Marlin / mega_moe`; ratio = mega_moe over Marlin (TTFT, TPOT: Marlin over mega_moe):

RAG:

| C | req/s (Marlin / mega_moe) | out tok/s (Marlin / mega_moe) | TTFT p50 ms (Marlin / mega_moe) | TTFT p99 ms (Marlin / mega_moe) | TPOT p50 ms (Marlin / mega_moe) | eff. C (Marlin / mega_moe) | ok/err (Marlin / mega_moe) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.41 / 0.41 | 104.5 / 106.1 | 98 / 93 | 1277 / 1281 | 9.0 / 8.9 | 1.0 / 1.0 | 100/0 / 100/0 | 1.01x | 1.05x | 1.01x |
| 16 | 2.39 / 2.45 | 612.8 / 626.3 | 658 / 634 | 2236 / 2971 | 22.8 / 22.2 | 15.8 / 15.8 | 500/0 / 500/0 | 1.02x | 1.04x | 1.03x |
| 32 | 3.81 / 3.92 | 974.9 / 1003.7 | 815 / 770 | 891 / 851 | 29.4 / 28.6 | 31.3 / 31.3 | 500/0 / 500/0 | 1.03x | 1.06x | 1.03x |
| 64 | 5.58 / 5.95 | 1429.0 / 1522.3 | 1199 / 1087 | 1622 / 1508 | 39.5 / 37.0 | 62.6 / 62.6 | 500/0 / 500/0 | 1.07x | 1.10x | 1.07x |
| 128 | 7.43 / 8.71 | 1901.0 / 2228.9 | 2045 / 1756 | 3817 / 2786 | 57.9 / 49.7 | 125.0 / 125.0 | 500/0 / 500/0 | 1.17x | 1.16x | 1.17x |
| 256 | 6.44 / 6.71 | 1649.8 / 1717.8 | 3185 / 3166 | 5804 / 5230 | 139.8 / 134.0 | 249.9 / 249.9 | 500/0 / 500/0 | 1.04x | 1.01x | 1.04x |
| 512 | 10.52 / 10.89 | 2692.5 / 2788.3 | 6547 / 5738 | 11420 / 10257 | 160.2 / 157.0 | 499.6 / 499.6 | 500/0 / 500/0 | 1.04x | 1.14x | 1.02x |

agentX:

| C | req/s (Marlin / mega_moe) | out tok/s (Marlin / mega_moe) | TTFT p50 ms (Marlin / mega_moe) | TTFT p99 ms (Marlin / mega_moe) | TPOT p50 ms (Marlin / mega_moe) | eff. C (Marlin / mega_moe) | ok/err (Marlin / mega_moe) | valid (Marlin / mega_moe) | out tok/s ratio | TTFT p50 ratio | TPOT p50 ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.111 / 0.108 | 89.6 / 86.7 | 187 / 210 | 1483 / 1321 | 11.3 / 10.9 | 0.9 / 0.9 | 100/0 / 100/0 | yes / yes | 0.97x | 0.89x | 1.04x |
| 16 | 0.188 / 0.190 | 150.8 / 151.1 | 190 / 174 | 3992 / 4010 | 11.5 / 11.2 | 1.8 / 1.8 | 175/0 / 177/0 | yes / yes | 1.00x | 1.09x | 1.03x |
| 64 | 0.696 / 0.704 | 524.2 / 541.6 | 299 / 299 | 6042 / 6546 | 34.3 / 32.1 | 19.4 / 18.7 | 654/0 / 662/0 | yes / yes | 1.03x | 1.00x | 1.07x |

- Mega MoE is 1.01-1.07x faster in RAG up to C=64, 1.17x at C=128 (TPOT 57.9 -> 49.7 ms), 1.04x at 256 and 512;
  in agentX it ties (0.97-1.03x). This agrees with the in-process result of analysis 9.38 (prefill 1.13x, TPOT
  within 1%) and keeps `auto` on Marlin.
- RAG C=256 dips below C=128 (6.44 vs 7.43 req/s, TPOT 139.8 vs 57.9 ms) and C=512 is 1.6x higher than 256 (10.52
  req/s). It is not noise: the mega MoE variant has the same shape (8.71, 6.71, 10.89 req/s), effective concurrency is 250 and 500 as set. The server log
  shows the cause: the decode batch is 116-128 at C=128 (about 2390 gen tok/s, ~49 ms a step), 244-256 at C=256
  (~1900 tok/s, ~130 ms a step) and 500 at C=512 (~3550 tok/s, ~140 ms a step). The step time
  jumps 2.7x when the batch passes 160, the largest captured CUDA graph, and then barely grows with the batch. This
  is consistent with decode above the graph size running at a much higher per-step host cost (eager launch of
  the 40-layer step); I did not read the executor code or confirm an eager fallback, so treat that as the likely
  explanation and not as a verified one (section 4.6).

## 4. Failures and findings

### 4.1 Before-image startup failures (3 sessions, 30 runs not run)

All three exit during startup (`ts subprocess exited with rc=-10`) and are marked `SESSION_FAILED`. Logs on
aws-raichu under `/workspace/nctu6/infra/tokenspeed/tmp/bench_images/`.

| Model | Error | Log |
|---|---|---|
| Qwen3.8-27B-NVFP4, TP1 | `RuntimeError: Kernel implementation not found`, raised by `fp4_quantize(x, layer.input_scale_inv)` in `layers/dense/nvfp4.py:196` through `tokenspeed_kernel/registry.py:531`: no `fp4_quantize` kernel is registered for SM120 in this image. Session ended after 67 s | `qwen38-27b-nvfp4/e27cee86/server-20261112-115858.log` |
| Qwen3.5-122B-A10B-NVFP4, TP2 | `RuntimeError: Check failed: (fc1_act_global.ndim() == 0 \|\| fc1_act_global.size(0) == num_experts_on_rank) is false: fc1 act global must be scalar or (num_experts_on_rank,)`, from `flashinfer_cutlass_nvfp4_moe_apply` (`ops/moe/flashinfer.py:3172`) in the startup forward. Session ended after 362 s | `qwen35-122b-a10b-nvfp4/e27cee86/server-20261112-170130.log` |
| DeepSeek-V4.1-Flash subset, TP2 | `ValueError: Unexpected V4.1 expert tensor: layers.0.ffn.experts.128.w1.scale` in `models/deepseek_v41.py:2444` (`_load_text_weights`): the loader has no `num_checkpoint_experts` support, so it rejects the checkpoint's expert 128. Session ended after 46 s | `dsv41-flash-1017/e27cee86/server-20261112-185406.log` |

The after image runs all three, so there is no open item (the NVFP4 quantize kernel, the NVFP4 MoE activation
scale shape and the `num_checkpoint_experts` loader are in the after image). I did not bisect which commit fixed
each one.

### 4.2 gemma-4 agentX at C = 16 and 64: not runnable at TP1 (both images)

The engine reports `max_req_input_len=199614` (`max_total_num_tokens=293760`, one GPU at 0.9 memory use) while
the agentX traces reach 256K tokens. The gateway answers HTTP 400 at the `WorkerSelection` stage for the long
requests and AIPerf's warmup phase fails terminally ("No profile results to export"). The run is marked `FAILED`
(wall 571 s for before c16, 1063 s for after c64). C=1 completes in both images (11 requests, valid); why its
long traces did not fail the same way was not checked. Not an image difference. Needs TP2 or a smaller weight footprint for
the 256K agentX trace (open item).

### 4.3 KV-event scheduler crash (both images, default configuration)

With the default servicer configuration (KV events on, no env variable) the engine aborts under RAG load on
Qwen3.8-27B-FP8:

```
[critical] Fatal check failed: cache mutation on a KV event boundary with no token descriptor
  at tokenspeed-scheduler/csrc/scheduler/scheduler.cpp:385 (after) / :386 (before) in Scheduler::markKvEventBoundaryForReconcile
Fatal Python error: Aborted
```

`[L2] writeback queued/started` lines precede the abort. The gateway stays up, so an unmonitored client hangs
until its own timeout (this is why the harness got a watchdog). Trigger: the servicer enables KV-event
publishing when `--kv-events-config` is absent (`tokenspeed serve` alone leaves it off); the scheduler's
bookkeeping then meets a cache mutation for a block hash with no descriptor. Which path erased or skipped the
descriptor was not traced. With `SMG_TOKENSPEED_SERVICER_KV_EVENTS=0` it does not happen.

Repro (fresh server per row, 500 prompts, `repro/qwen38-27b-fp8/`):

| image | config | c64 | c128 |
|---|---|---|---|
| after `ef057889` | default | DONE (125 s) | engine crash 35 s into the run |
| after `ef057889` | `KV_EVENTS=0` | DONE (125 s) | DONE (104 s, 500 ok, 0 errors) |
| before `e27cee86` | default | DONE (137 s) | engine crash 35 s into the run |
| before `e27cee86` | `KV_EVENTS=0` | repro aborted by an operator error | covered by the primary session (c128 DONE, 500 ok) |

First default-config session (`default/qwen38-27b-fp8/`): before image c1 .. c128 done, **c256 crashed**; after
image c1 .. c64 done, **c128 crashed**. So the before image survived c128 once and crashed on it in the repro: the
trigger is load dependent, not a fixed concurrency. Restart test (after image, default config): c64 done, c128
crashed and was caught 4 s after the fatal line, the server restarted (336 s) and c16 completed on the new one.
This is a scheduler bug (open item), not specific to the model; only the FP8 27B model was tried with it on.

### 4.4 gemma-4: `<bos>` and the gateway tool-call parser (harness issue, fixed)

- The first gemma sessions failed the startup probe in both images (France ` the the the ...`, primes
  ` 2, 2, 2, ...`; records in `_probe_failed_no_bos/`). The gemma-4 tokenizer does not add BOS, so a raw
  `/v1/completions` prompt has none and the model degenerates; with a literal `<bos>` in front the same prompts
  answer " Paris." and " 7, 11, 13, ...". Only the chat template adds BOS. The harness now prefixes `<bos>` to
  probes and RAG prompts (queue key `prompt_prefix`), identically for both images; agentX uses the chat endpoint
  and is unchanged. Every gemma RAG prompt counts one server-side token more than the same text without it.
- With the default parser selection (`tokenizer_config.json` has a `response_template`, so the gateway gets
  `--parser symphony --reasoning-parser hf --tool-call-parser hf`) every `/v1/chat/completions` request on
  gemma-4 fails with HTTP 400 `invalid_parser_configuration`. An explicit `--tool-call-parser gemma4` avoids it
  (both images). It only changes how the gateway parses the text. Open item: the default is broken for gemma-4
  chat.

Neither is an image bug (same behavior in both), the first is a client-side requirement.

### 4.5 Other observations

- agentX smoke mode is not submission-valid (section 3.1). All 26 completed full-mode runs report
  `submission_valid=true` (no invalid reasons).
- Effective concurrency in agentX is far below C (FP8: 6.3 of 16 and 8 of 64; Qwen3.6: 3.9 of 16; 122B: 3.3 of 16;
  V4.1: 1.8 of 16 and 19 of 64), (requests wait on 256K-token prefills and between turns; I did not break this down further). V4.1 finishes 654 requests at C=64 against 14-54 for the other models, since its KV
  pool holds 24.5M tokens (19,356 pages of 256) against 0.3-3.6M for the others.
- V4.1 RAG C=1 TTFT p99 is 1277 ms against p50 98 ms: the first request of the run pays one-time work. Seen in the
  C=1 row only.
- Operator reading: GPU 6 showed 100% utilization at 0 MiB used at some point between sessions. The harness
  waits for less than 1 GB used and no compute process before it starts, and the recorded `gpu_at_launch.txt`
  of every session shows 0 MiB; no run was affected as far as these records show. I did not capture that reading
  myself (nvidia-smi now shows 0% on GPUs 6 and 7).

### 4.6 High-concurrency plateaus

At C >= 128 the numbers are those of a saturated server, for three reasons that are the same in both images:
the KV pool fills (gemma at C >= 32, Qwen3.6 and FP8 27B at C >= 256), the decode batch passes the largest
captured CUDA graph (160; the V4.1 step time measured above), and for gemma the pool allows only ~27 requests. The
ratios at those points compare saturated servers and are smaller than at low C (FP8: 1.04x). The graph size is
the server default (`max_cudagraph_capture_size=160`) and was not raised in this benchmark although
`--max-num-seqs` is 512. A run with a larger `--max-cudagraph-capture-size` would show whether the C=256-512
numbers improve (open item, not run).

## 5. Caveats

- Shared host. Concurrent TP1 sessions ran on GPUs 6 and 7 (the FP8 pair, the Qwen3.6 / NVFP4 pair and the gemma
  pair ran side by side) and other users' jobs ran on GPUs 0-5 and between sessions. They share CPU, PCIe, disk,
  page cache and the network; model loading, AIPerf tokenization and dataset generation compete. Before and
  after of one model usually ran under similar load, not identical.
- One run per point, no repeats. Differences of a few percent are inside the spread: the BF16 control differs by
  0-1% in RAG out tok/s and TPOT, but by up to 11% in TTFT p50 (RAG) and 4-6% in agentX TPOT p50. TTFT p99 and
  agentX TTFT are heavy-tailed.
- agentX only at C = 1, 16 and 64, 900 s windows; at C=64 only 14-54 requests finish for most models. The smoke
  runs are not submission-valid and are not compared.
- The V4.1 results are a 128-of-384-expert subset: the model output is not meaningful and the MoE cost is about a
  third of the full model's. Compare the two V4.1 backends with each other, not with a full V4.1 deployment.
- Effective concurrency stays below C in agentX and at C=512 in RAG (the gemma and FP8 servers queue); the
  caps are listed per table.
- The server log does not name the GEMM or attention leaf chosen per layer (`selected_kernels.txt` holds only
  the matching log lines), so I attribute the FP8 gain to the DeepGEMM SM120 dense leaf from the kernel
  benchmarks (decode 1.2-4.0x, prefill parity) and from the BF16 controls showing no change, not from a per-run
  kernel trace.
- Default `max_cudagraph_capture_size=160` with `--max-num-seqs 512` (section 4.6) applies to both images and
  is not "CUDA graph on" for batches above 160.

## 6. Reproduce

Harness: `tokenspeed/tmp/bench_images_local/` (git-ignored scratch in the TokenSpeed checkout; the live copy
and all raw results are on aws-raichu in `/workspace/nctu6/infra/tokenspeed/tmp/bench_images/`). Its
`README.md` documents the harness, the watchdog, agentX full mode, the KV-event crash and the gemma
details. The tables here were built from a copy of the result tree
(`metrics.json`, `DONE` / `FAILED` markers, `session.json`, `probe.json`, `server_cmd.txt`; no AIPerf artifacts)
by `results/make_tables.py` in that directory.

```bash
cd /workspace/nctu6/infra/tokenspeed/tmp/bench_images           # aws-raichu
# one worker per GPU; TP1 sessions run on whichever worker is free, TP2 on the GPU 6 worker
tmux new-session -d -s bench  'TMPDIR=$PWD/../bt/g6 python3 -u bench_queue.py --gpus 6 2>&1 | tee -a logs/worker-gpu6.out'
tmux new-session -d -s bench7 'TMPDIR=$PWD/../bt/g7 python3 -u bench_queue.py --gpus 7 2>&1 | tee -a logs/worker-gpu7.out'
cat STATUS                                   # progress
python3 bench_queue.py --gpus 6 --dry-parse  # done / failed counts from the markers
```

`queue.txt` has one session per line (`model= path= tp= gpus= image= env= args= prompt_prefix= hf_overrides=
variant=`); the primary lines set `env=SMG_TOKENSPEED_SERVICER_KV_EVENTS=0`. Runs with a marker are skipped on a
restart (`--retry-failed` reruns failed ones). The KV-event repro is `repro_queue.txt`. Results land in
`<model>/<image-short>[-variant]/{rag,agentx}/c<N>/metrics.json`; default-configuration runs in `default/`. The
agentX command is in the harness README ("agentX full mode").

Counts: the primary set has 70 RAG runs done, 26 agentX full-mode runs done and 4 agentX runs failed (the gemma
C=16 and 64, section 4.2), plus 30 runs of the three failed before-image sessions that never started. The
harness-wide totals (112 done, 41 failed or not runnable) also include the default-config, repro, dry-run and
smoke runs.
