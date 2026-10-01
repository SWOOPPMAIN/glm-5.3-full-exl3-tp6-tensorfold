#!/usr/bin/env python3
"""Record matched full-GLM speed screens; credentials stay in a private file.

Run against a drained native endpoint. Prompt fixtures are frozen across
candidates; a unique cache salt prevents reuse during cold-prefill measurements.
This script never starts, stops, restores or changes a model configuration.
"""
import argparse
import hashlib
import json
import re
import os
from pathlib import Path
import re
import statistics
import time
import urllib.request
import uuid

from long_context import prepare
from qualify import benchmark, chat, headers, request, stream_chat


def metrics(base):
    req = urllib.request.Request(base + '/metrics', headers=headers())
    with urllib.request.urlopen(req, timeout=15) as response:
        return parse_metrics(response.read().decode())


def parse_metrics(text):
    names = ('num_requests_running', 'num_requests_waiting', 'num_preemptions_total',
             'request_prefill_time_seconds_sum', 'prefix_cache_hits_total',
             'prefix_cache_queries_total', 'spec_decode_num_accepted_tokens_total',
             'spec_decode_num_draft_tokens_total', 'spec_decode_num_drafts_total',
             'request_decode_time_seconds_sum','request_inference_time_seconds_sum')
    result = {}
    for line in text.splitlines():
        if line.startswith('#') or not line:
            continue
        label, value = line.rsplit(' ', 1)
        name = label.split('{', 1)[0]
        if name.removeprefix('vllm:') in names:
            result[name] = result.get(name, 0) + float(value)
        elif name == 'vllm:spec_decode_num_accepted_tokens_per_pos_total':
            match = re.search(r'(?:\{|,)position="(\d+)"(?:,|\})',label)
            if match:
                key = name+'{position="'+match[1]+'"}'
                result[key] = result.get(key,0)+float(value)
    return result


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def parse_recall(answer):
    """Separate recall correctness from the existing Markdown-fence behavior."""
    try:
        return json.loads(answer), True
    except ValueError:
        match = re.fullmatch(r'```(?:json)?\s*\n(.*)\n```', answer.strip(), flags=re.DOTALL)
        if match:
            try:
                return json.loads(match[1]), False
            except ValueError:
                pass
        return None, False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run'])
    p.add_argument('--base', default='http://127.0.0.1:8953')
    p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--fixtures', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.add_argument('--label')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--prefill-repetitions', type=int, default=1)
    p.add_argument('--contexts', type=int, nargs='+', default=[8192, 32768, 131072])
    p.add_argument('--skip-decode', action='store_true')
    p.add_argument('--skip-prefill', action='store_true')
    p.add_argument('--expected-draft-depth', type=int, choices=[1,2,3,4])
    args = p.parse_args()
    os.environ['AMOS_API_KEY_FILE'] = str(args.key_file)
    if min(args.repetitions, args.prefill_repetitions) < 1:
        p.error('Repetition counts must be positive')
    if args.mode == 'prepare':
        if args.fixtures.exists():
            p.error('Refusing to replace frozen fixtures')
        fixture = {'schema': 1, 'contexts': {}}
        for target in args.contexts:
            fixture['contexts'][str(target)] = prepare(args.base, target)
            print(json.dumps({'prepared': target,
                              'tokens': fixture['contexts'][str(target)]['tokens']}), flush=True)
        atomic_json(args.fixtures, fixture)
        return
    if not args.output or not args.label:
        p.error('run requires --output and --label')
    if args.output.exists():
        p.error('Refusing to replace an experiment result')
    fixture_bytes = args.fixtures.read_bytes()
    fixture = json.loads(fixture_bytes)
    before = metrics(args.base)
    if any(before.get('vllm:' + name, 0) for name in
           ('num_requests_running', 'num_requests_waiting')):
        raise RuntimeError('Drain the native endpoint before measurement')
    report = {'schema': 1, 'label': args.label, 'state': 'running',
              'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'fixtures_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
              'decode': [], 'prefill': [], 'metrics_before': before,
              'method': 'C1; seed 17; temperature 0; low reasoning; 512 decode output tokens; '
                        'cold prefill with a unique cache salt per request; '
                        'TTFT includes first generation step.'}
    atomic_json(args.output, report)
    try:
        if not args.skip_decode:
            # A fixed warmup excludes compilation/startup and warms the short prompts.
            for kind in ('prose', 'code'):
                benchmark(args.base, kind, 64)
            for repeat in range(args.repetitions):
                for kind in ('prose', 'code'):
                    start_metrics = metrics(args.base)
                    row = {'repeat': repeat, **benchmark(args.base, kind, 512)}
                    end_metrics = metrics(args.base)
                    row['metrics_delta'] = {k:v-start_metrics.get(k,0)
                                            for k,v in end_metrics.items()}
                    drafts = row['metrics_delta'].get('vllm:spec_decode_num_drafts_total',0)
                    attempted = row['metrics_delta'].get('vllm:spec_decode_num_draft_tokens_total',0)
                    row['mean_attempted_draft_depth'] = attempted/drafts if drafts else None
                    if row['usage']['completion_tokens'] != 512:
                        raise RuntimeError('Decode output-token count differs')
                    report['decode'].append(row)
                    atomic_json(args.output, report)
                    if args.expected_draft_depth is not None and (
                        not drafts or not args.expected_draft_depth - 0.1 <= attempted/drafts <= args.expected_draft_depth
                    ):
                        raise RuntimeError('Measured draft depth differs from requested calibration depth')
                    print(json.dumps({k: v for k, v in row.items()
                                      if k not in ('content', 'answer')}), flush=True)
        for target in ([] if args.skip_prefill else args.contexts):
            item = fixture['contexts'][str(target)]
            for repeat in range(args.prefill_repetitions):
                start_metrics = metrics(args.base)
                row = {'target_context': target, 'repeat': repeat,
                       **stream_chat(args.base, chat(item['prompt'], max_tokens=512, seed=17,
                           cache_salt=str(uuid.uuid4()), stream=True,
                           stream_options={'include_usage': True}))}
                end_metrics = metrics(args.base)
                row['metrics_delta'] = {k: v-start_metrics.get(k, 0)
                                        for k, v in end_metrics.items()}
                parsed, row['plain_json_pass'] = parse_recall(row['answer'])
                row['recall_pass'] = parsed == item['expected']
                row['usage_pass'] = row['usage']['prompt_tokens'] == item['tokens']
                row['prompt_tok_s_by_ttft'] = item['tokens'] / row['ttft_seconds']
                report['prefill'].append(row)
                atomic_json(args.output, report)
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ('content', 'answer')}), flush=True)
                if not row['recall_pass'] or not row['usage_pass']:
                    raise RuntimeError('Prefill recall/usage gate failed')
        report['decode_medians'] = {kind: statistics.median(
            row['decode_tok_s_estimate'] for row in report['decode'] if row['case'] == kind)
            for kind in ('prose', 'code')} if report['decode'] else {}
        report['state'] = 'complete'
    except Exception as error:
        report['state'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        report['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        atomic_json(args.output, report)
    print(json.dumps({'label': args.label, 'state': report['state'],
                      'decode_medians': report['decode_medians']}), flush=True)


if __name__ == '__main__':
    main()
