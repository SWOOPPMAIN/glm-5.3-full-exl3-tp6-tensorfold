#!/usr/bin/env python3
"""Generate the exact synthetic fixtures used in the P27 cache residency replay.

CPU chat rendering only; no generation or serving changes. Uses /render because
this pinned fork's /tokenize endpoint omits historical reasoning_content.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from qualify import request

TOOL = {'type': 'function', 'function': {'name': 'read_fixture_result',
    'description': 'Read the integer result for this synthetic code fixture.',
    'parameters': {'type': 'object', 'properties': {'module': {'type': 'string'}},
                   'required': ['module']}}}

QUESTION = 'Use the supplied tool result. Return only the JSON object {"result":42,"status":"ok"}.'

def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()

def payload(messages):
    return dict(model='glm-5.3', messages=messages, tools=[TOOL], temperature=0,
        seed=17, max_tokens=256, reasoning_effort='low', chat_template_kwargs={'reasoning_effort': 'low'},
        stream=True, stream_options={'include_usage': True})

def tokenize(api, body):
    # The pinned /tokenize path drops historical reasoning_content. The chat
    # renderer preserves it, so only /render gives the exact generation prefix.
    return api.json('/v1/chat/completions/render', body)['token_ids']

def tool_history(module):
    return [{'role': 'assistant', 'content': '', 'reasoning_content': 'Read the fixture result.',
             'tool_calls': [{'id': 'call_fixture', 'type': 'function', 'function': {
                 'name': 'read_fixture_result', 'arguments': json.dumps({'module': module})}}]},
            {'role': 'tool', 'tool_call_id': 'call_fixture', 'content': '{"result":42,"status":"ok"}'}]

def prepare(api, target):
    items = []
    for name in ('alpha', 'bravo', 'charlie'):
        def body(n):
            code = '\n'.join(f'def fixture_{i}(value):\n    # Pure fixture code; no external side effects.\n    return value + {i % 23}\n' for i in range(n))
            return payload([{'role': 'system', 'content': 'You review synthetic Python modules. Follow the final user instruction and retain the supplied tool result.'},
                {'role': 'user', 'content': f'Module {name}. Source follows.\n{code}\nRead the fixture result for this module.'},
                *tool_history(name), {'role': 'user', 'content': QUESTION}])
        lo, hi = 0, target // 12
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if len(tokenize(api, body(mid))) <= target:
                lo = mid
            else:
                hi = mid
        b = body(lo)
        ts = tokenize(api, b)
        assert target - 64 <= len(ts) <= target, len(ts)
        items.append(dict(name=name, payload=b, token_count=len(ts), token_sha256=digest(ts)))
    return dict(schema=1, target_tokens=target, conversations=items,
                synthetic=True, method='Same tools and system; independent long code histories; retained assistant reasoning.')

class RenderAPI:
    def __init__(self, base):
        self.base = base

    def json(self, path, payload):
        return request(self.base, path, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--key-file', required=True, type=Path)
    p.add_argument('--target', required=True, type=int, choices=[8192, 32768, 131072])
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        p.error('Refusing to replace frozen fixtures')
    os.environ['AMOS_API_KEY_FILE'] = str(args.key_file)
    result = prepare(RenderAPI(args.base), args.target)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'target_tokens': args.target,
                      'actual_tokens': [c['token_count'] for c in result['conversations']],
                      'fixture_sha256': digest(result)}))


if __name__ == '__main__':
    main()
