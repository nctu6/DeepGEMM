# Fit the SM120 FP8 GEMM layout heuristic offline against `sm120_sweep_graph.py --json` measurements
#
# The model mirrors `SM120ArchSpec::get_layout_info` (keep both in sync):
#   time(layout) = sum over waves of max(dram_share(wave) / BW, num_k_blocks * cta_cost(layout)) + l2 term
# and reports the regret (chosen time / best time) per shape for a parameter grid
import argparse
import itertools
import json
import math

NUM_SMS = 188
LAYOUTS = [(16, 64), (16, 128), (16, 256), (32, 64), (32, 128), (32, 256), (64, 64), (64, 128), (128, 64), (128, 128)]

# Relative per-K-block cost of one CTA in a full wave (128x128 = 1000), see `get_relative_wave_cost`
CTA_COST = {(128, 128): 1000, (128, 64): 590, (64, 128): 570, (64, 64): 400, (32, 256): 840,
            (32, 128): 470, (32, 64): 310, (16, 256): 850, (16, 128): 450, (16, 64): 250}


def predict(m, n, k, bm, bn, p):
    num_kb = math.ceil(k / 128)
    tiles = math.ceil(m / bm) * math.ceil(n / bn)
    full_waves, last = divmod(tiles, NUM_SMS)
    # DRAM bytes (weights, activations, output), in units of one 128x128 K-block time (~1 us)
    dram = (n * k + m * k + 2 * m * n) / p['dram_bytes_per_unit']
    # L2 -> SM bytes (every tile reloads its A and B)
    l2 = tiles * num_kb * (bm + bn) * 128 / p['l2_bytes_per_unit']
    cta = num_kb * CTA_COST[(bm, bn)] / 1000 * p['cta_scale']
    t = 0.0
    for wave_tiles in [NUM_SMS] * full_waves + ([last] if last else []):
        share = wave_tiles / tiles
        t += max(dram * share, l2 * share, cta + p['wave_overhead'])
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    records = json.load(open(args.json))

    def evaluate(p, verbose=False):
        regrets = []
        for r in records:
            m, n, k = r['m'], r['n'], r['k']
            times = {tuple(int(x) for x in l.split('/')[0].split('x')): v
                     for l, v in r['times'].items() if l != 'auto' and l.endswith('sk0')}
            if not times:
                continue
            best = min(times.values())
            chosen = min(times, key=lambda l: predict(m, n, k, *l, p))
            regrets.append(times[chosen] / best)
            if verbose:
                best_l = min(times, key=times.get)
                print(f'm={m:5} n={n:6} k={k:6} chosen={chosen} {times[chosen] * 1e6:7.1f}us '
                      f'best={best_l} {best * 1e6:7.1f}us regret={times[chosen] / best:.3f} '
                      f'auto={r["times"]["auto"] / best:.3f}')
        geo = math.exp(sum(math.log(x) for x in regrets) / len(regrets))
        return geo, max(regrets)

    grid = {
        'dram_bytes_per_unit': [1.2e6, 1.4e6, 1.6e6, 1.8e6],
        'l2_bytes_per_unit': [4e6, 6e6, 8e6, 12e6, 1e9],
        'cta_scale': [0.6, 0.8, 1.0, 1.2],
        'wave_overhead': [0.0, 1.0, 2.0, 4.0, 8.0],
    }
    best = None
    for values in itertools.product(*grid.values()):
        p = dict(zip(grid.keys(), values))
        geo, worst = evaluate(p)
        score = (geo, worst)
        if best is None or score < best[0]:
            best = (score, p)
    print(f'best params: {best[1]}, geomean regret={best[0][0]:.4f}, worst regret={best[0][1]:.3f}')
    auto = [r['times']['auto'] / min(v for l, v in r['times'].items() if l != 'auto') for r in records]
    print(f'current auto: geomean regret={math.exp(sum(map(math.log, auto)) / len(auto)):.4f}, worst={max(auto):.3f}')
    if args.verbose:
        evaluate(best[1], verbose=True)


if __name__ == '__main__':
    main()
