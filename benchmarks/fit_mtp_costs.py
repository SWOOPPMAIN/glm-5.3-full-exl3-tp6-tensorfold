#!/usr/bin/env python3
"""Fit a bounded MTP candidate from complete, repeated fixed-depth workloads.

This prepares controls; it never changes serving or claims a throughput win.
Concurrent costs use native counters inside verified simultaneous decoding.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from mtp_matrix import validate
from performance import atomic_json


def fit(paths):
    samples = defaultdict(list)
    sources, images, fixtures = {}, set(), set()
    expected_cells = {(n, 8192, k) for n in (1, 2, 3, 4) for k in ('code', 'prose')}
    for path in paths:
        raw = path.read_bytes()
        d = json.loads(raw)
        assert d['phase'] == 'complete' and d['passed']
        assert d['scope'] in ('matrix', 'overlap_costs') and d['policy']['mode'] == 'fixed'
        assert path.name not in sources
        sources[path.name] = hashlib.sha256(raw).hexdigest()
        images.add(d['image']); fixtures.add(d['fixture_sha256'])
        depth = d['policy']['depth']
        assert depth in (1, 2, 3, 4)
        seen = set()
        for row in d['rows']:
            count = row['requests']
            assert count == row['peak_running'] and len(row['rows']) == count
            ids = [r['fixture'] for r in row['rows']]
            context = int(ids[0].rsplit('-', 1)[1])
            assert all(int(i.rsplit('-', 1)[1]) == context for i in ids)
            kind = ids[0].rsplit('-', 1)[0]
            if d['scope'] == 'matrix' and (count != 1 or context != 8192 or kind not in ('code', 'prose')):
                continue
            assert (count, context, kind) in expected_cells
            if count > 1:
                assert d['scope'] == 'overlap_costs' and len(set(ids)) == 1 and row['eligible']
                assert row['steady_window']['draft_tokens'] == row['steady_window']['drafts']*depth
            assert row['metrics_delta']['vllm:num_preemptions_total'] == 0
            if count == 1:
                assert depth - .1 <= row['mean_attempted_depth'] <= depth
            cost = row['steady_window']['step_ms'] if count > 1 else row['server_step_ms_estimate']
            assert math.isfinite(cost) and 1 < cost < 10000
            visit = (row['repeat'], count, context, kind)
            assert visit not in seen, 'Duplicate workload within one receipt'
            seen.add(visit)
            samples[(depth, count, context, kind)].append(cost)
    assert len(images) == len(fixtures) == 1
    assert set(samples) == {(depth, *cell) for depth in (1, 2, 3, 4) for cell in expected_cells}
    assert all(len(values) >= 3 for values in samples.values()), 'Need at least three visits per cell'
    cells = [dict(depth=k[0], requests=k[1], context=k[2], kind=k[3],
                  visits=len(v), median_ms=statistics.median(v), range_ms=[min(v), max(v)])
             for k, v in sorted(samples.items())]
    costs = {str(n): [statistics.mean(c['median_ms'] for c in cells
                                    if c['requests'] == n and c['depth'] == depth)
                     for depth in (1, 2, 3, 4)] for n in (1, 2, 3, 4)}
    controls = [validate(dict(revision='mtp-fitted-window' + str(window), mode='costs',
                             depth=0, window=window, cost_ms=costs)) for window in (16, 8, 32)]
    return dict(schema=1, phase='candidate_unqualified', sources=sources,
                image=images.pop(), fixture_sha256=fixtures.pop(), cells=cells,
                cost_ms=costs, controls=controls,
                method='Equal-weight mean of code/prose8K per-cell medians. C2..4 use native counter slopes inside verified simultaneous decode; C1 uses request-decode milliseconds per draft round from matching matrix fixtures.',
                limitations=['Wall time includes target, draft and scheduling; API polling is present and this is not isolated GPU timing.',
                             'Identical prompts can produce different continuations. Cost calibration does not claim output equivalence or replace original numerical gates.',
                             'Calibration uses representative8K code/prose; full short/8K/32K code/prose/tool and mixed-concurrency matrices remain required for candidate comparison.',
                             'All fitted controls require repeated end-to-end comparisons and numerical/application qualification.',
                             'Native acceptance accounting, priors and target verification are unchanged.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = fit(args.input)
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('phase', 'cost_ms')}, indent=2))


if __name__ == '__main__':
    main()
