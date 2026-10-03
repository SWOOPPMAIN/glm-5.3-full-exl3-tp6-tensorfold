#!/usr/bin/env python3
"""Summarize matched row32/64 E3 replay; never infer serving tok/s."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def analyze(directory):
    cells=[];checks=0;peak=0;sources={}
    for layer in (3,40,77):
        for rank in range(6):
            path=directory/f'e32-r{rank}-l{layer}-result.json';r=json.loads(path.read_text())
            assert r['phase']=='complete' and r['passed'] and r['quality_passed']
            assert (r['rank'],r['layer'])==(rank,layer)
            assert r['scratch_arenas']==1 and r['runtime_scratch_shared']
            assert len(r['cases'])==26 and all(c['exact_captured_output'] for c in r['cases'])
            assert sum(bool(c.get('graph')) for c in r['cases'])==6
            checks+=len(r['cases']);peak=max(peak,r['peak_cuda_allocated_gib'])
            sources[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            assert len(r['timings'])==10
            for t in r['timings']:
                assert len(t['samples'])==16
                for repeat in range(8):
                    group=[s for s in t['samples'] if s['repeat']==repeat]
                    assert [s['arm'] for s in group]==(['row64','row32'] if repeat%2==0 else ['row32','row64'])
                med={arm:statistics.median(s['gpu_ms'] for s in t['samples'] if s['arm']==arm) for arm in ('row64','row32')}
                assert med==t['median_ms']
                cells.append(dict(rank=rank,layer=layer,kind=t['kind'],rows=t['rows'],samples=t['samples'],
                    median_ms=med,speed_ratio=med['row64']/med['row32'],
                    latency_reduction_fraction=1-med['row32']/med['row64']))
    summaries=[]
    for rows in (513,768,1024,1536,3072):
        selected=[c for c in cells if c['rows']==rows];assert len(selected)==36
        ratio=math.exp(statistics.mean(math.log(c['speed_ratio']) for c in selected))
        summaries.append(dict(rows=rows,cells=len(selected),geomean_speed_ratio=ratio,
            equivalent_geomean_latency_reduction=1-1/ratio,min_cell_speed_ratio=min(c['speed_ratio'] for c in selected),
            max_cell_speed_ratio=max(c['speed_ratio'] for c in selected),
            faster_cells=sum(c['speed_ratio']>1 for c in selected)))
    return dict(phase='component_comparison_complete',all_exact=True,exact_comparisons=checks,
        layers=[3,40,77],ranks=6,prompts=['code','prose'],timed_cells=len(cells),samples_per_arm_cell=8,
        total_timed_calls=sum(len(c['samples']) for c in cells),peak_cuda_allocated_gib=peak,
        summaries=summaries,cells=cells,sources=sources,
        decision='Proceed to full-model row32 qualification and repeated matched serving comparison; no serving promotion from isolated timing.',
        limitations=['Three sampled layers and two synthetic prompts; results do not characterize all75layers or all prompts.',
            'Smaller-row cases are prefixes of actual3072-row captures, not independent full-model runs at those budgets.',
            'The fleet was stopped; each container held one layer and one shared scratch arena. Full-model memory/cache/scheduling behavior requires separate measurement.',
            'E3 timings include routing and epilogues. Ratios cannot be directly converted into full-model tok/s.'],
        tensorfold_work=False,serving_speed_claim=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();assert not a.output.exists();r=analyze(a.directory)
    a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('phase','exact_comparisons','timed_cells','total_timed_calls','peak_cuda_allocated_gib','summaries')},indent=2))


if __name__=='__main__':main()
