# End-to-end serving benchmark client (OpenAI completions API, streaming), for SM120 DeepGEMM A/B runs
#
# Reports TTFT / TPOT / output throughput at several concurrencies, a long-prompt TTFT, and greedy outputs of a few
# fixed prompts (to compare numerics between runs).
import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp

WORDS = ('alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau '
         'upsilon phi chi psi omega river mountain cloud forest ocean desert valley city bridge tower').split()
GREEDY_PROMPTS = [
    'Question: What is 17 * 23? Answer step by step, then give the final number.\n',
    'Write a Python function that returns the n-th Fibonacci number iteratively.\n',
    'List the planets of the solar system in order from the Sun.\n',
    'Translate to French: "The weather is nice today and we will go to the park."\n',
]


def make_prompt(rng: random.Random, num_words: int) -> str:
    return ' '.join(rng.choice(WORDS) for _ in range(num_words))


async def one_request(session, url, model, prompt, max_tokens, temperature=0.0, ignore_eos=True):
    payload = {'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature,
               'stream': True, 'ignore_eos': ignore_eos}
    start = time.perf_counter()
    first, chunks, text = None, 0, []
    async with session.post(url, json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            obj = json.loads(data)
            piece = obj['choices'][0].get('text', '')
            if piece:
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                text.append(piece)
    end = time.perf_counter()
    return {'ttft': (first or end) - start, 'latency': end - start, 'chunks': chunks, 'text': ''.join(text)}


async def run_concurrency(url, model, concurrency, num_requests, prompt_words, max_tokens, seed):
    rng = random.Random(seed)
    prompts = [make_prompt(rng, prompt_words) for _ in range(num_requests)]
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        async def guarded(p):
            async with sem:
                return await one_request(session, url, model, p, max_tokens)
        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - start
    tpots = [(r['latency'] - r['ttft']) / max(1, r['chunks'] - 1) for r in results]
    return {
        'concurrency': concurrency,
        'ttft_ms': statistics.median(r['ttft'] for r in results) * 1e3,
        'tpot_ms': statistics.median(tpots) * 1e3,
        'out_tok_s': sum(r['chunks'] for r in results) / wall,
    }


async def main_async(args):
    url = f'http://127.0.0.1:{args.port}/v1/completions'
    report = {'runs': []}
    # Warmup (JIT compiles, CUDA graphs)
    await run_concurrency(url, args.model, 4, 8, args.prompt_words, 32, seed=1)
    for c in [int(x) for x in args.concurrency.split(',')]:
        r = await run_concurrency(url, args.model, c, max(c * 2, 8), args.prompt_words, args.max_tokens, seed=c)
        report['runs'].append(r)
        print(f"  concurrency={c:3}: TTFT {r['ttft_ms']:8.1f} ms | TPOT {r['tpot_ms']:7.2f} ms | {r['out_tok_s']:8.1f} tok/s",
              flush=True)
    # Long-prompt prefill
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        rng = random.Random(7)
        ttfts = []
        for _ in range(3):
            r = await one_request(session, url, args.model, make_prompt(rng, args.long_words), 1)
            ttfts.append(r['ttft'])
        report['long_prompt_ttft_ms'] = min(ttfts) * 1e3
        print(f"  long prompt ({args.long_words} words): TTFT {report['long_prompt_ttft_ms']:.1f} ms", flush=True)
        report['greedy'] = []
        for p in GREEDY_PROMPTS:
            r = await one_request(session, url, args.model, p, 96, ignore_eos=False)
            report['greedy'].append(r['text'])
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--concurrency', type=str, default='1,8,32')
    parser.add_argument('--prompt-words', type=int, default=256)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--long-words', type=int, default=6000)
    parser.add_argument('--out', type=str, default=None)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == '__main__':
    main()
