#!/usr/bin/env python3
"""Summarize matched MTP matrix visits without selecting a production policy."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(paths):
    groups=defaultdict(list);sources={};fixtures=set();images=set()
    for path in paths:
        raw=path.read_bytes();d=json.loads(raw)
        assert d['phase']=='complete' and d['passed'] and d['scope'] in ('matrix','candidate_matrix')
        assert path.name not in sources
        sources[path.name]=hashlib.sha256(raw).hexdigest()
        fixtures.add(d['fixture_sha256']);images.add(d['image'])
        policy=d['policy'];mode=policy['mode']
        name='fixed'+str(policy['depth']) if mode=='fixed' else mode
        if mode=='costs':
            name+='-'+hashlib.sha256(json.dumps({k:policy[k] for k in ['cost_ms','window']},sort_keys=True).encode()).hexdigest()[:12]
        for row in d['rows']:
            ids=[r['fixture'] for r in row['rows']]
            context=int(ids[0].rsplit('-',1)[1]);count=row['requests']
            assert row['peak_running']==count and all(int(i.rsplit('-',1)[1])==context for i in ids)
            key=(name,context,ids[0].rsplit('-',1)[0] if count==1 else 'mixed',count)
            groups[key].append(dict(source=path.name,repeat=row['repeat'],
                aggregate_output_tok_s=row['aggregate_output_tok_s'],
                mean_decode_tok_s_estimate=statistics.mean(r['decode_tok_s_estimate'] for r in row['rows']),
                mean_ttft_seconds=statistics.mean(r['ttft_seconds'] for r in row['rows']),
                mean_attempted_depth=row['mean_attempted_depth'],accepted_per_draft=row['accepted_per_draft'],
                server_step_ms_estimate=row['server_step_ms_estimate'],
                accepted_per_wall_second=row['accepted_per_wall_second'],
                accepted_per_server_request_decode_second=row['accepted_per_server_request_decode_second'],
                output_tokens=[r['usage']['completion_tokens'] for r in row['rows']],
                answer_hashes=[r['answer_sha256'] for r in row['rows']],
                reasoning_hashes=[r['reasoning_sha256'] for r in row['rows']],
                tool_streams=[r['tool_stream_present'] for r in row['rows']],
                preemptions=row['metrics_delta']['vllm:num_preemptions_total']))
    assert len(fixtures)==len(images)==1
    cells=[]
    for (policy,context,kind,count),samples in sorted(groups.items()):
        rates=[s['aggregate_output_tok_s'] for s in samples]
        cells.append(dict(policy=policy,context=context,kind=kind,requests=count,repetitions=len(samples),
            median_output_tok_s=statistics.median(rates),range_output_tok_s=[min(rates),max(rates)],
            median_step_ms_estimate=statistics.median(s['server_step_ms_estimate'] for s in samples),
            median_accepted_per_wall_second=statistics.median(s['accepted_per_wall_second'] for s in samples),
            samples=samples))
    baseline={(r['context'],r['kind'],r['requests']):r for r in cells if r['policy']=='original'}
    comparisons=[]
    for policy in sorted({r['policy'] for r in cells}-{'original'}):
        candidates=[r for r in cells if r['policy']==policy]
        assert {(r['context'],r['kind'],r['requests']) for r in candidates}==set(baseline)
        changes=[]
        for r in candidates:
            b=baseline[(r['context'],r['kind'],r['requests'])]
            ratio=r['median_output_tok_s']/b['median_output_tok_s']
            changes.append(dict(context=r['context'],kind=r['kind'],requests=r['requests'],ratio=ratio,
                candidate_repetitions=r['repetitions'],control_repetitions=b['repetitions']))
        comparisons.append(dict(policy=policy,equal_cell_geomean_ratio=math.exp(statistics.mean(math.log(x['ratio']) for x in changes)),
            three_visits_per_cell=all(x['candidate_repetitions']>=3 and x['control_repetitions']>=3 for x in changes),cells=changes))
    return dict(schema=1,phase='measurement_summary_only',image=images.pop(),fixtures_sha256=fixtures.pop(),
        sources=sources,cells=cells,comparisons=comparisons,production_policy_selected=False,
        limitations=['First-pass samples alone cannot establish a repeatable win. Require balanced repeated visits and final numerical/native/application gates.',
            'Output throughput includes cached prefill and client delivery; SSE decode estimates count all generated reasoning and content/tool tokens.',
            'Concurrent step costs are request-weighted full-run estimates with declining residency; not isolated fixed-batch GPU timings.',
            'Tool bodies can be truncated at256 tokens; separate functional tool tests establish basic API correctness, not comprehensive task quality.',
            'Same prompts/settings do not guarantee identical generated text across draft policies. Answer/reasoning hashes are retained; the first pass records tool presence but does not retain tool-argument hashes.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path,nargs='+')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();assert not a.output.exists()
    result=analyze(a.input);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(cells=len(result['cells']),comparisons=[{k:v for k,v in c.items() if k!='cells'} for c in result['comparisons']]),indent=2))
