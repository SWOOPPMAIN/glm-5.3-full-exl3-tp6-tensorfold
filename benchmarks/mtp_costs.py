#!/usr/bin/env python3
"""Replay one fixed-depth calibration visit against a drained native API.

The operator applies the policy and verifies the image, exclusive GPU ownership
and fresh host guards externally. This client never changes serving settings.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from mtp_matrix import API, digest, idle, stream, tokenize, validate
from mtp_overlap_cost import measure_overlap
from performance import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--key-file', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--policy-file', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).with_name('mtp-fixtures.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--order-offset', type=int, choices=range(6), default=0)
    args = parser.parse_args()
    assert re.fullmatch(r'sha256:[a-f0-9]{64}', args.image)
    assert args.name.replace('-', '').isalnum()
    policy = validate(json.loads(args.policy_file.read_text()))
    assert policy['mode'] == 'fixed', 'Apply and acknowledge a fixed-depth control before calibration'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / (args.name + '-bench.json')
    assert not out.exists()
    api = API(args.base, args.key_file)
    idle(api)
    assert {r['id'] for r in api.json('/v1/models')['data']} == {'glm-5.3'}
    assert api.json('/v1/amos/capacity') == {'context_length':360000, 'max_running_requests':4}
    selected = [i for i in json.loads(args.fixtures.read_text())['items']
                if i['nominal_context'] == 8192 and i['kind'] in ('code', 'prose')]
    assert len(selected) == 2
    for item in selected:
        ids = tokenize(api, item['payload'])
        assert len(ids) == item['prompt_tokens'] and digest(ids) == item['token_sha256']
    report = dict(phase='running', scope='overlap_costs', repetitions=1, policy=policy,
                  image=args.image, image_and_policy_asserted_by_operator=True,
                  host_guard_checks='Required externally; this client does not inspect hosts.',
                  fixture_sha256=hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
                  warmup=[], rows=[])
    atomic_json(out, report)
    for item in selected:
        row = stream(api, dict(item['payload'], max_tokens=32))
        report['warmup'].append(dict(fixture=item['id'], usage=row['usage'], ttft_seconds=row['ttft_seconds']))
    atomic_json(out, report)
    cells = [(item, count) for count in (2, 3, 4) for item in selected]
    cells = cells[args.order_offset:] + cells[:args.order_offset]
    for item, count in cells:
        row = measure_overlap(api, item, count, policy['depth'])
        row.update(repeat=0, calibration_kind=item['kind'])
        report['rows'].append(row)
        atomic_json(out, report)
        if not row['eligible']:
            report.update(phase='failed_calibration_gate', passed=False)
            atomic_json(out, report)
            raise SystemExit('Calibration overlap gate failed; inspect the recorded cell')
        print(json.dumps(dict(kind=item['kind'], requests=count,
                              step_ms=row['steady_window']['step_ms'])), flush=True)
    report.update(phase='complete', passed=True)
    atomic_json(out, report)


if __name__ == '__main__':
    main()
