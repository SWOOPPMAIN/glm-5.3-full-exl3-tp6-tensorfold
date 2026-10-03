#!/usr/bin/env python3
"""Recompute copy/MTP medians, ranges and ratios from the published timing samples."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def analyze(data):
    matrix=defaultdict(lambda:defaultdict(list));conventional=defaultdict(lambda:defaultdict(list))
    for visit in data['visits']:
        profile=visit['profile']
        assert len(visit['matrix'])==8*visit['repetitions']
        for row in visit['matrix']:
            assert row['requests']==row['peak_running']==len(row['rows'])
            assert all(r['usage']['completion_tokens']==256 for r in row['rows'])
            assert row['metrics_delta']['vllm:num_preemptions_total']==0
            total=sum(r['usage']['completion_tokens'] for r in row['rows'])
            rate=total/row['elapsed_seconds']
            assert math.isclose(rate,row['aggregate_output_tok_s'],rel_tol=1e-12)
            fixture=row['rows'][0]['fixture']
            cell=fixture if row['requests']==1 else 'mixed-c4-'+fixture.rsplit('-',1)[1]
            matrix[profile][cell].append(rate)
        for row in visit['decode']:
            conventional[profile]['decode-'+row['case']].append(row['decode_tok_s_estimate'])
        for row in visit['prefill']:
            assert row['recall_pass'] and row['usage_pass']
            conventional[profile]['cold-prefill-'+str(row['target_context'])].append(row['prompt_tok_s_by_ttft'])
            conventional[profile]['cold-ttft-'+str(row['target_context'])].append(row['ttft_seconds'])
    def compare(groups):
        assert set(groups)=={'mtp4','ngram-gpu4'} and groups['mtp4'].keys()==groups['ngram-gpu4'].keys()
        out={}
        for cell in sorted(groups['mtp4']):
            arms={}
            for profile in ('mtp4','ngram-gpu4'):
                values=groups[profile][cell];assert len(values)==3
                arms[profile]=dict(values=values,median=statistics.median(values),minimum=min(values),maximum=max(values))
            a,b=arms['mtp4'],arms['ngram-gpu4']
            out[cell]=dict(arms=arms,ratio=b['median']/a['median'],percent_change=100*(b['median']/a['median']-1),
                observed_ranges_separated=b['minimum']>a['maximum'] or a['minimum']>b['maximum'])
        return out
    cells=compare(matrix);ordinary=compare(conventional)
    return dict(matrix=cells,conventional=ordinary,
        copy_matrix_geomean_ratio=math.exp(statistics.mean(math.log(r['ratio']) for r in cells.values())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('samples',type=Path)
    p.add_argument('--summary',type=Path);a=p.parse_args();result=analyze(json.loads(a.samples.read_bytes()))
    if a.summary:
        expected=json.loads(a.summary.read_bytes())['comparison']
        assert math.isclose(result['copy_matrix_geomean_ratio'],expected['copy_matrix_geomean_ratio'],rel_tol=1e-12)
        for row in expected['copy_matrix']:
            actual=result['matrix'][row['cell']]
            assert actual['percent_change']==row['percent_change']
            assert actual['observed_ranges_separated']==row['observed_ranges_separated']
            for profile in ('mtp4','ngram-gpu4'):assert actual['arms'][profile]==row['arms'][profile]['output_tok_s']
        for cell,row in expected['conventional'].items():
            actual=result['conventional'][cell]
            assert actual['arms']['mtp4']==row['mtp'] and actual['arms']['ngram-gpu4']==row['copy']
            assert actual['percent_change']==row['percent_change']
    print(json.dumps(result,indent=2))
