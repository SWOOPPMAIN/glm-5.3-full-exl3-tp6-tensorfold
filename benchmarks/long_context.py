#!/usr/bin/env python3
"""Verify actual token usage and three-position recall at a requested context."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import threading
import time
from qualify import chat, request


def prepare(base, target):
    identity = secrets.token_hex(8)
    expected = {key: secrets.token_hex(6) for key in ('first', 'middle', 'last')}
    filler = 'Every archive entry describes a routine equipment inspection without incident.\n'

    def prompt(n):
        return (f'Archive {identity}. Memorize the three special reference codes.\n'
                + filler * (n // 20)
                + f'REFERENCE first = {expected["first"]}\n'
                + filler * (n // 2)
                + f'REFERENCE middle = {expected["middle"]}\n'
                + filler * (n - n // 20 - n // 2)
                + f'REFERENCE last = {expected["last"]}\n'
                + 'Return only a JSON object with first, middle, and last reference codes.')

    def count(n):
        payload = chat(prompt(n))
        return request(base, '/tokenize', {key: payload[key] for key in
            ('model', 'messages', 'chat_template_kwargs')})['count']

    low, high = 0, max(1, target // 5)
    while count(high) < target:
        high *= 2
    while low + 1 < high:
        mid = (low + high) // 2
        if count(mid) <= target:
            low = mid
        else:
            high = mid
    actual = count(low)
    assert target - 32 <= actual <= target, (target, actual)
    return {'prompt': prompt(low), 'tokens': actual, 'expected': expected, 'id': identity}


def run(base, prepared, barrier):
    barrier.wait(timeout=30)
    start = time.monotonic()
    raw = request(base, '/v1/chat/completions', chat(prepared['prompt'], max_tokens=512), timeout=3600)
    content = raw['choices'][0]['message'].get('content') or ''
    try:
        answer = json.loads(content)
    except ValueError:
        answer = None
    usage = raw.get('usage', {})
    return {'id': prepared['id'], 'expected': prepared['expected'],
            'tokenized_prompt': prepared['tokens'], 'elapsed_seconds': time.monotonic()-start,
            'pass': answer == prepared['expected'] and usage.get('prompt_tokens') == prepared['tokens'],
            'response': raw}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8953')
    p.add_argument('--tokens', required=True, type=int)
    p.add_argument('--concurrency', type=int, default=1)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.tokens < 1024 or not 1 <= args.concurrency <= 4:
        p.error('Require at least 1024 tokens and concurrency 1..4')
    prepared = [prepare(args.base, args.tokens) for _ in range(args.concurrency)]
    print(json.dumps({'prepared_tokens': [x['tokens'] for x in prepared]}), flush=True)
    barrier = threading.Barrier(args.concurrency)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(lambda item: run(args.base, item, barrier), prepared))
    report = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'target_tokens': args.tokens, 'concurrency': args.concurrency, 'results': results,
              'limitation': 'Recall/usage check; does not prove full-window reasoning or simultaneous cache residency.'}
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k != 'response'}), flush=True)
    if not all(item['pass'] for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
