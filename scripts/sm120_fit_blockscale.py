# Fit the SM120 block-scaled (1D1D, FP4) layout model offline against `sm120_sweep_blockscale.py` measurements
#
# The model mirrors `SM120ArchSpec::get_layout_info` for `KernelType::Kernel1D1D` (keep both in sync):
#   per wave: max(dram_share, l2_share, cta)
#   cta = num_k_blocks * (block_k / 128) * cta_cost[layout] / 1000 * cta_scale * depth_penalty(stages) + wave_overhead
# where `depth_penalty` charges pipelines shallower than `full_depth` stages (latency-bound). The model ranks
# data-parallel layouts; the chosen layout is then split (stream-K) by the explicit `use_stream_k` policy. Reports the regret
# (chosen time / best time) per shape. Prints the fitted constants to paste into `sm120.hpp`.
import argparse
import json
import math

NUM_SMS = 188
SMEM_CAPACITY = 101376
MAX_STAGES = 16


def num_math_threads(bm, bn):
    if bm == 64 and bn >= 128:
        return 256
    return 128 if bm <= 64 else 256


def num_stages(bm, bn, bk):
    words = bk * 2 // 128
    align = lambda x: (x + 127) // 128 * 128
    per_stage = bm * bk + bn * bk + align(words * bm * 4) + align(words * bn * 4)
    reduce = 0
    if bn == 32:
        num_accum = (bm // 16) * (bn // 2 // 8) * 4
        reduce = align(num_accum * (num_math_threads(bm, bn) // 2) * 4)
    return min((SMEM_CAPACITY - (MAX_STAGES * 16 + 16) - reduce) // per_stage, MAX_STAGES)


def stream_k_ctas(m, n, k, bm, bn, bk, min_iters=4):
    """Mirrors `SM120ArchSpec::get_stream_k_split`: 0 means no split"""
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    num_kb = math.ceil(k // 2 / bk)
    if tiles > 1024:
        return 0
    ctas = min(NUM_SMS, tiles * num_kb // min_iters)
    return ctas if ctas > tiles else 0


def use_stream_k(m, n, k, bm, bn, bk):
    """Mirrors `SM120ArchSpec::get_stream_k_config` (1D1D, auto): split only where the data-parallel grid fills under
    1/8 of the GPU, and the split clearly shortens the per-CTA K loop"""
    ctas = stream_k_ctas(m, n, k, bm, bn, bk)
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    if not ctas or tiles * 8 > NUM_SMS:
        return False
    num_kb = math.ceil(k // 2 / bk)
    dp = math.ceil(tiles / NUM_SMS) * num_kb
    sk = math.ceil(tiles * num_kb / ctas) + 2
    return sk * 10 <= dp * 9


def predict(m, n, k, bm, bn, bk, p, cost):
    k_bytes = k // 2
    num_kb = math.ceil(k_bytes / bk)
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    full_waves, last = divmod(tiles, NUM_SMS)
    # Small weight boxes (`block_n` rows of `block_k` bytes per TMA) stream DRAM less efficiently
    box_penalty = max(1.0, p['box_ref'] / (bn * bk)) ** p['box_exp']
    dram = (n * k_bytes + m * k_bytes + 2 * m * n) / p['dram'] * box_penalty
    l2 = tiles * num_kb * (bm + bn) * bk / p['l2']
    stages = num_stages(bm, bn, bk)
    depth = max(1.0, p['full_depth'] / stages) ** p['depth_exp']
    per_block = bk / 128 * cost[(bm, bn)] / 1000 * p['cta_scale'] * depth
    cta = num_kb * per_block + p['overhead']
    t = 0.0
    for wave_tiles in [NUM_SMS] * full_waves + ([last] if last else []):
        share = wave_tiles / tiles
        t += max(dram * share, l2 * share, cta)
    return t


def parse(records):
    out = []
    for r in records:
        times = {}
        for key, v in r['times'].items():
            if key == 'auto':
                continue
            bm, bn, bk = (int(x) for x in key.split('/')[0].split('x'))
            sk = key.endswith('/sk2')
            # The sweep forces a split whenever there are spare CTAs; keep the ones the policy can choose
            if sk and not use_stream_k(r['m'], r['n'], r['k'], bm, bn, bk):
                continue
            times[(bm, bn, bk, sk)] = v
        if times:
            out.append((r['m'], r['n'], r['k'], times, r['times'].get('auto')))
    return out


def evaluate(data, p, cost, verbose=False):
    regrets = []
    for m, n, k, times, auto in data:
        best = min(times.values())
        # The model ranks data-parallel layouts; the stream-K policy then decides the split
        layout = min((l for l in times if not l[3]), key=lambda l: predict(m, n, k, *l[:3], p, cost))
        chosen = (*layout[:3], True) if (*layout[:3], True) in times else layout
        regrets.append(times[chosen] / best)
        if verbose:
            best_l = min(times, key=times.get)
            print(f'm={m:5} n={n:5} k={k:5} chosen={chosen} {times[chosen] * 1e6:7.2f} best={best_l} {best * 1e6:7.2f} '
                  f'regret={times[chosen] / best:.3f}' + (f' auto={auto / best:.3f}' if auto else ''))
    geo = math.exp(sum(map(math.log, regrets)) / len(regrets))
    return geo, max(regrets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json', nargs='+', help='sweep outputs; a later file replaces earlier records of the same shape')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--rounds', type=int, default=6)
    args = parser.parse_args()
    records = {}
    for path in args.json:
        for r in json.load(open(path)):
            records[(r['m'], r['n'], r['k'])] = r
    data = parse(list(records.values()))
    layouts = sorted({(bm, bn) for _, _, _, t, _ in data for (bm, bn, _, _) in t})

    p = {'dram': 1.6e6, 'l2': 6e6, 'cta_scale': 1.0, 'overhead': 2.0, 'full_depth': 3.0, 'depth_exp': 1.0,
         'box_ref': 8192.0, 'box_exp': 0.0}
    cost = {l: l[0] * l[1] * 1000 / (128 * 128) for l in layouts}
    grid = {
        'dram': [0.6e6, 0.8e6, 1.0e6, 1.2e6, 1.4e6, 1.6e6, 1.8e6, 2.0e6, 2.5e6, 3e6],
        'l2': [0.5e6, 1e6, 1.5e6, 2e6, 3e6, 4e6, 6e6, 8e6, 12e6, 1e9],
        'cta_scale': [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0],
        'overhead': [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        'full_depth': [2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
        'depth_exp': [0.0, 0.25, 0.5, 1.0, 1.5, 2.0],
        'box_ref': [4096.0, 8192.0, 16384.0, 32768.0],
        'box_exp': [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0],
    }
    score = lambda: evaluate(data, p, cost)
    best = score()
    # The objective is piecewise constant (an argmin), so after a coarse grid pass the constants and per-layout costs
    # are refined by multiplicative coordinate steps of shrinking size
    for _ in range(args.rounds):
        # Global constants
        for name, values in grid.items():
            for v in values:
                old = p[name]; p[name] = v
                s = score()
                if s < best:
                    best = s
                else:
                    p[name] = old
        # Per-layout CTA costs (multiplicative steps)
        for l in layouts:
            for f in (0.6, 0.8, 0.9, 1.1, 1.25, 1.6):
                old = cost[l]; cost[l] = old * f
                s = score()
                if s < best:
                    best = s
                else:
                    cost[l] = old
    for step in (1.5, 1.25, 1.1, 1.05):
        for _ in range(args.rounds):
            improved = False
            for name in p:
                for f in (1 / step, step):
                    old = p[name]; p[name] = old * f if old else (0.5 if f > 1 else 0.0)
                    s = score()
                    if s < best:
                        best, improved = s, True
                    else:
                        p[name] = old
            for l in layouts:
                for f in (1 / step, step):
                    old = cost[l]; cost[l] = old * f
                    s = score()
                    if s < best:
                        best, improved = s, True
                    else:
                        cost[l] = old
            if not improved:
                break
    print(f'fitted: geomean regret={best[0]:.4f}, worst={best[1]:.3f}')
    print('params:', {k: (f'{v:.3g}') for k, v in p.items()})
    print('cta costs (per 128-byte K block, 128x128 FP32 = 1000):')
    for l in layouts:
        print(f'    if (bm == {l[0]:3} and bn == {l[1]:3}) return {round(cost[l])};')
    autos = [a / min(t.values()) for _, _, _, t, a in data if a]
    if autos:
        print(f'current auto: geomean regret={math.exp(sum(map(math.log, autos)) / len(autos)):.4f}, worst={max(autos):.3f}')
    if args.verbose:
        evaluate(data, p, cost, verbose=True)


if __name__ == '__main__':
    main()
