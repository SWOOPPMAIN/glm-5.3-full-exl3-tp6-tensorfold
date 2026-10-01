#!/usr/bin/env python3
"""Small native-API correctness gate and reproducible streaming speed probes."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import urllib.request


def headers():
    result = {'Content-Type': 'application/json'}
    key_file = os.environ.get('AMOS_API_KEY_FILE')
    if key_file:
        key = Path(key_file).read_text().strip()
        if not key or any(ord(c) < 33 or ord(c) > 126 for c in key):
            raise RuntimeError('Invalid API credential file')
        result['Authorization'] = 'Bearer ' + key
    return result


def request(base, path, payload=None, timeout=600):
    req = urllib.request.Request(base + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers=headers())
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def chat(prompt, **extra):
    return {'model': 'glm-5.3', 'messages': [{'role': 'user', 'content': prompt}],
            'temperature': 0, 'max_tokens': 512, 'reasoning_effort': 'low',
            'chat_template_kwargs': {'reasoning_effort': 'low'}, **extra}


def basic(base):
    cases = [
        ('arithmetic', 'What is 17 multiplied by 23? Reply with only the integer.', '391'),
        ('sorting', 'Sort 9, -3, 7, 0, 7 numerically. Return only a JSON array.', [-3, 0, 7, 7, 9]),
        ('code_trace', 'What does this Python expression return? Reply with only the integer.\n'
         'sum(x*x for x in range(6) if x % 2)', '35'),
        ('extraction', 'Read this record: name=Orion; count=14; color=teal. '
         'Return only a JSON object with the keys count and color.', {'count': 14, 'color': 'teal'}),
        ('negation', 'Four switches: A is off, B is on, C is off, D is on. '
         'Return only a JSON array containing the names of the switches that are off, in order.', ['A', 'C']),
        ('unicode', 'Translate the French word bonjour into English. Reply with one word.', 'hello'),
    ]
    results = []
    for label, prompt, expected in cases:
        started = time.monotonic()
        raw = request(base, '/v1/chat/completions', chat(prompt))
        choice = raw['choices'][0]
        content = (choice['message'].get('content') or '').strip()
        try:
            actual = json.loads(content) if isinstance(expected, (list, dict)) else content.lower().rstrip('.')
        except ValueError:
            actual = content
        results.append({'case': label, 'pass': actual == expected and choice['finish_reason'] == 'stop',
                        'elapsed': time.monotonic() - started, 'response': raw})
        print(json.dumps({'case': label, 'pass': results[-1]['pass'], 'content': content}), flush=True)

    raw = request(base, '/v1/completions', {'model': 'glm-5.3',
        'prompt': 'The capital of France is', 'temperature': 0, 'max_tokens': 16, 'logprobs': 1})
    values = raw['choices'][0]['logprobs']['token_logprobs']
    results.append({'case': 'finite_logprobs', 'pass': bool(values) and
                    all(x is not None and math.isfinite(x) for x in values), 'response': raw})

    tool = {'type': 'function', 'function': {'name': 'get_weather',
        'description': 'Look up the weather for a city.', 'parameters': {
            'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}
    raw = request(base, '/v1/chat/completions', chat('Call get_weather for Boston.',
        tools=[tool], tool_choice='auto', max_tokens=512))
    calls = raw['choices'][0]['message'].get('tool_calls') or []
    try:
        passed = len(calls) == 1 and calls[0]['function']['name'] == 'get_weather' and (
            json.loads(calls[0]['function']['arguments']) == {'city': 'Boston'})
    except (KeyError, ValueError):
        passed = False
    results.append({'case': 'tool_call', 'pass': passed, 'response': raw})
    return results


def benchmark(base, kind, tokens, *, record_event_timing=False):
    prompt = {
        'prose': 'Write a detailed practical explanation of how a community garden can collect '
                 'rainwater and use it throughout a dry summer. Discuss design, maintenance and tradeoffs.',
        'code': 'Write a complete Python implementation of an LRU cache with a dictionary and a '
                'doubly linked list. Include get, put, eviction, iteration and useful unit tests.',
    }[kind]
    payload = chat(prompt, max_tokens=tokens, stream=True, ignore_eos=True, seed=17,
                   stream_options={'include_usage': True})
    return {'case': kind, **stream_chat(base, payload, record_event_timing=record_event_timing)}


def stream_chat(base, payload, *, record_event_timing=False):
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                 headers=headers())
    started = time.monotonic()
    first = last = usage = None
    content = ''
    answer = ''
    done = False
    event_times = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            value = line[6:].strip()
            if value == b'[DONE]':
                done = True
                break
            event = json.loads(value)
            if 'error' in event:
                raise RuntimeError(event['error'])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                answer += delta.get('content') or ''
                text = delta.get('content') or delta.get('reasoning') or delta.get('reasoning_content')
                if text:
                    now = time.monotonic()
                    first = first or now
                    last = now
                    content += text
                    if record_event_timing:
                        event_times.append(now - started)
    if not (done and usage and first and last and last > first):
        raise RuntimeError('Incomplete stream or missing timing/usage')
    result = {'usage': usage, 'ttft_seconds': first - started,
            'decode_tok_s_estimate': (usage['completion_tokens'] - 1) / (last - first),
            'elapsed_seconds': time.monotonic() - started, 'content': content, 'answer': answer,
            'timing_note': 'Visible SSE timing estimate; MTP may emit multiple tokens per event.'}
    if record_event_timing:
        result['stream_event_elapsed_seconds'] = event_times
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['basic', 'bench'])
    p.add_argument('--base', default='http://127.0.0.1:8953')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, default=256)
    args = p.parse_args()
    catalog = request(args.base, '/v1/models')
    assert catalog['data'][0]['id'] == 'glm-5.3', catalog
    results = basic(args.base) if args.mode == 'basic' else [
        benchmark(args.base, kind, args.tokens) for kind in ('prose', 'code')]
    report = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'catalog': catalog, 'mode': args.mode, 'results': results}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for item in results:
        print(json.dumps({k: v for k, v in item.items() if k not in ('response', 'content', 'answer')}), flush=True)
    if any(item.get('pass') is False for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
