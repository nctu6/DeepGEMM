# Host time of the two launch paths left in a small V3.2 DSA prefill chunk (task 37):
# `deep_gemm.fp8_mqa_logits` and DeepSelect (`deepselect_local_topk` with `index_base`).
#
# Run in the TokenSpeed venv with this DeepGEMM build first on PYTHONPATH (see `sm120_remote.sh ts`).
# The time is the CPU time of the call (`perf_counter_ns` around it, no synchronization inside the timed region; the GPU
# work of these shapes is shorter than the host path, so the launch queue never fills). Medians over many calls.
#   --prefix N   keys cached before the chunk (default 0)
#   --profile    also print a cProfile breakdown of the Python side of each call
import argparse
import cProfile
import io
import pstats
import statistics
import time

import torch
from tokenspeed_kernel.ops.attention.dsa import cute_dsl as dsa_cute

import deep_gemm

HEAD_DIM = 128
TOPK = 2048


def make_inputs(tokens: int, prefix: int, heads: int):
    keys = tokens + prefix
    k = torch.randn(keys, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    scales = (k.abs().amax(-1).float().clamp_min(1e-4) / 448.0).contiguous()
    k_fp8 = (k / scales[:, None]).to(torch.float8_e4m3fn).contiguous()
    q = torch.randn(tokens, heads, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_fp8 = (q / 4).to(torch.float8_e4m3fn).contiguous()
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    starts = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    ends = (torch.arange(tokens, device="cuda", dtype=torch.int32) + prefix + 1).contiguous()
    return q_fp8, (k_fp8, scales), weights, starts, ends


def host_time(fn, iters: int, warm: int = 50, sync_every: int = 25):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        times.append(time.perf_counter_ns() - t0)
        if (i + 1) % sync_every == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return statistics.median(times) / 1e3, min(times) / 1e3


def profile(fn, iters: int = 2000, top: int = 18):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    prof = cProfile.Profile()
    prof.enable()
    for i in range(iters):
        fn()
        if (i + 1) % 25 == 0:
            torch.cuda.synchronize()
    prof.disable()
    torch.cuda.synchronize()
    out = io.StringIO()
    pstats.Stats(prof, stream=out).sort_stats("tottime").print_stats(top)
    print(out.getvalue())


def bench_leaf(tokens: int, prefix: int, heads: int, iters: int, do_profile: bool):
    """The shipped prefill leaf, with a host timer around each of its launch paths (segments sum the CPU time of the
    call, including the argument slicing the leaf does around it)."""
    import math

    from sm120_dsa_vs_tokenspeed import DIM, PAGE, TOPK as LEAF_TOPK, make_index_cache
    from tokenspeed_kernel.ops.attention.dsa import deep_gemm as dsa_dg
    from tokenspeed_kernel.ops.attention.dsa._triton import prefill_prepare

    keys = prefix + tokens
    cache = make_index_cache(math.ceil(keys / PAGE))
    slots = torch.arange(keys, device="cuda", dtype=torch.int64)
    row_starts = torch.zeros(tokens, device="cuda", dtype=torch.int32)
    row_ends = (prefix + 1 + torch.arange(tokens, device="cuda", dtype=torch.int32)).contiguous()
    q = torch.randn(tokens, heads, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, heads, device="cuda", dtype=torch.float32)
    cpu = (row_ends - row_starts).cpu()

    def leaf():
        return dsa_dg.deep_gemm_dsa_prefill_topk(
            q, weights, slots, row_starts, row_ends, topk=LEAF_TOPK, softmax_scale=DIM**-0.5,
            index_k_cache=cache, page_size=PAGE, candidate_lens_cpu=cpu,
        )

    seg = {"prepare": 0, "mqa_logits_call": 0, "deepselect_call": 0, "nan_to_num+mark": 0}
    pc = time.perf_counter_ns
    real_mqa, real_sel, real_prep = dsa_dg.deep_gemm.fp8_mqa_logits, dsa_dg.deepselect_local_topk, prefill_prepare.dsa_prefill_prepare
    real_mark = dsa_dg.mark_forced_initial_local_logits

    def timed(name, fn):
        def run(*a, **k):
            t = pc()
            r = fn(*a, **k)
            seg[name] += pc() - t
            return r
        return run

    # The module object `deep_gemm` is shared, so restore it afterwards
    leaf()
    torch.cuda.synchronize()
    totals = []
    dsa_dg.deep_gemm.fp8_mqa_logits = timed("mqa_logits_call", real_mqa)
    dsa_dg.deepselect_local_topk = timed("deepselect_call", real_sel)
    prefill_prepare.dsa_prefill_prepare = timed("prepare", real_prep)
    dsa_dg.mark_forced_initial_local_logits = timed("nan_to_num+mark", real_mark)
    try:
        for _ in range(50):
            leaf()
        torch.cuda.synchronize()
        for k in seg:
            seg[k] = 0
        for i in range(iters):
            t = pc()
            leaf()
            totals.append(pc() - t)
            if (i + 1) % 25 == 0:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
    finally:
        dsa_dg.deep_gemm.fp8_mqa_logits = real_mqa
        dsa_dg.deepselect_local_topk = real_sel
        prefill_prepare.dsa_prefill_prepare = real_prep
        dsa_dg.mark_forced_initial_local_logits = real_mark
    print(f"leaf tokens {tokens} prefix {prefix} heads {heads}: host median per call {statistics.median(totals) / 1e3:.1f} us")
    for k, v in seg.items():
        print(f"  {k:20s} mean {v / iters / 1e3:7.2f} us")
    # Wall time of back-to-back calls (the GPU is not idle-bound at these sizes only if the host is slower)
    for _ in range(3):
        leaf()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    runs = []
    for _ in range(7):
        a.record()
        for _ in range(50):
            leaf()
        b.record()
        torch.cuda.synchronize()
        runs.append(a.elapsed_time(b) * 1e3 / 50)
    print(f"  back-to-back eager call (CUDA events): median {statistics.median(runs):.1f} us  min {min(runs):.1f} us")
    if do_profile:
        profile(leaf, iters=1000, top=30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", action="store_true")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--prefix", type=int, default=0)
    ap.add_argument("--heads", type=int, default=64)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()

    deep_gemm.set_pdl(True)
    print(f"tokenspeed: {dsa_cute.__file__}\ndeep_gemm : {deep_gemm.__file__}")
    if args.leaf:
        bench_leaf(args.tokens, args.prefix, args.heads, args.iters, args.profile)
        return
    q, kv, w, starts, ends = make_inputs(args.tokens, args.prefix, args.heads)
    max_seqlen_k = int(ends.max())

    def mqa():
        return deep_gemm.fp8_mqa_logits(
            q, kv, w, starts, ends, clean_logits=False, max_seqlen_k=max_seqlen_k
        )

    logits = mqa()
    torch.cuda.synchronize()
    out = torch.empty(args.tokens, TOPK, device="cuda", dtype=torch.int32)
    lens = (ends - starts).contiguous()
    supported = dsa_cute.deepselect_supports(logits, TOPK)

    def select():
        return dsa_cute.deepselect_local_topk(logits, lens, TOPK, out, index_base=starts)

    def supports():
        return dsa_cute.deepselect_supports(logits, TOPK)

    select()
    torch.cuda.synchronize()
    print(f"tokens {args.tokens} prefix {args.prefix} heads {args.heads} deepselect_supports {supported}")
    for name, fn in (("fp8_mqa_logits", mqa), ("deepselect_supports", supports), ("deepselect_local_topk", select)):
        med, best = host_time(fn, args.iters)
        print(f"{name:24s} host median {med:7.2f} us  min {best:7.2f} us")
    if args.profile:
        for name, fn in (("fp8_mqa_logits", mqa), ("deepselect_local_topk", select)):
            print(f"--- cProfile {name}")
            profile(fn)


if __name__ == "__main__":
    main()
