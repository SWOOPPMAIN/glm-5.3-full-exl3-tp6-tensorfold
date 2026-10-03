#!/usr/bin/env python3
"""Recalculate the published E3 serving comparison from all exported samples."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def interval(values):
    return dict(median=statistics.median(values),minimum=min(values),maximum=max(values),count=len(values))


def analyze(data):
    visits=data['visits'];assert len(visits)==4
    assert [(v['rows'],v['repetitions']) for v in visits]==[(64,1),(32,1),(64,2),(32,2)]
    assert len({v['matrix_fixture_sha256'] for v in visits})==1
    assert len({v['performance_fixture_sha256'] for v in visits})==1
    arms={}
    for tile in (64,32):
        groups=defaultdict(list);prefill=defaultdict(list);decode=defaultdict(list)
        for visit in visits:
            if visit['rows']!=tile:continue
            for cell in visit['matrix']:
                assert cell['requests']==cell['peak_running']==len(cell['rows'])
                assert cell['metrics_delta']['vllm:num_preemptions_total']==0
                groups[tuple(r['fixture'] for r in cell['rows'])].append(cell)
            for row in visit['prefill']:
                assert row['recall_pass'] and row['usage_pass']
                assert row['metrics_delta'].get('vllm:num_preemptions_total',0)==0
                prefill[row['target_context']].append(row)
            for row in visit['decode']:
                assert row['usage']['completion_tokens']==512
                decode[row['case']].append(row)
        assert len(groups)==12 and sorted(prefill)==[8192,32768] and set(decode)=={'prose','code'}
        cells=[]
        for fixtures,rows in sorted(groups.items()):
            assert len(rows)==3
            lengths={tuple(r['usage']['completion_tokens'] for r in row['rows']) for row in rows}
            assert len(lengths)==1
            cells.append(dict(fixtures=list(fixtures),requests=len(fixtures),output_tokens=list(lengths.pop()),
                throughput=interval([r['aggregate_output_tok_s'] for r in rows]),
                accepted_per_wall_second=interval([r['accepted_per_wall_second'] for r in rows])))
        for rows in list(prefill.values())+list(decode.values()):assert len(rows)==3
        arms[str(tile)]=dict(cells=cells,
            prefill={str(n):dict(throughput=interval([r['prompt_tok_s_by_ttft'] for r in rows]),
                                ttft=interval([r['ttft_seconds'] for r in rows])) for n,rows in sorted(prefill.items())},
            decode={kind:interval([r['decode_tok_s_estimate'] for r in rows]) for kind,rows in sorted(decode.items())})
    old,new=arms['64'],arms['32'];ratios=[]
    for control,candidate in zip(old['cells'],new['cells'],strict=True):
        assert control['fixtures']==candidate['fixtures'] and control['output_tokens']==candidate['output_tokens']
        ratios.append(dict(fixtures=control['fixtures'],requests=control['requests'],
            throughput_ratio=candidate['throughput']['median']/control['throughput']['median']))
    return dict(phase='public_samples_recomputed',arms=arms,matrix_comparisons=ratios,
        matrix_geomean_ratio=math.exp(statistics.mean(math.log(r['throughput_ratio']) for r in ratios)),
        prefill_ratios={n:new['prefill'][n]['throughput']['median']/old['prefill'][n]['throughput']['median'] for n in old['prefill']},
        decode_ratios={k:new['decode'][k]['median']/old['decode'][k]['median'] for k in old['decode']},
        limitations=['Three samples per workload/policy across two visits; interval reports observed range, not a confidence interval.',
            'Recomputes timing/count checks only. Numerical qualification and application acceptance are separate receipts.',
            'Cached generation uses unchanged adaptive MTP; differences are not isolated E3 decode-kernel speedups.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',required=True,type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();result=analyze(json.loads(a.samples.read_text()))
    if a.output:
        assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('arms','matrix_comparisons')},indent=2))


if __name__=='__main__':main()
