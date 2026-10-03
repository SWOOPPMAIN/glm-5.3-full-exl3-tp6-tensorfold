#!/usr/bin/env python3
"""Recompute native/E3 component medians from the public scalar samples."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROWS=(33,65,128,256,384,512,513,768,1024)
ARMS=('native','row64','row32')
ORDERS=(('native','row64','row32'),('row32','row64','native'),
        ('row64','native','row32'),('row32','native','row64'),
        ('native','row32','row64'),('row64','row32','native'),
        ('native','row64','row32'),('row32','row64','native'))


def analyze(data):
    assert data['schema']==1 and data['repetitions']==8
    assert data['execution_order_by_repeat']==[list(order) for order in ORDERS]
    cells=data['cells'];assert len(cells)==324
    expected={(rank,layer,kind,rows) for rank in range(6) for layer in (3,40,77)
              for kind in ('code','prose') for rows in ROWS}
    assert {(c['rank'],c['layer'],c['kind'],c['rows']) for c in cells}==expected
    computed=[]
    for c in cells:
        for clock in ('gpu_ms','wall_ms'):
            assert set(c[clock])==set(ARMS)
            for arm in ARMS:
                values=c[clock][arm]
                assert len(values)==8 and all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in values)
        med={arm:statistics.median(c['gpu_ms'][arm]) for arm in ARMS}
        computed.append(dict(rank=c['rank'],layer=c['layer'],kind=c['kind'],rows=c['rows'],median_ms=med,
            ratio=med['native']/med['row32'],nonoverlap=max(c['gpu_ms']['row32'])<min(c['gpu_ms']['native'])))
    summaries=[]
    for rows in ROWS:
        selected=[c for c in computed if c['rows']==rows];assert len(selected)==36
        ratios=[c['ratio'] for c in selected];critical=[]
        for kind in ('code','prose'):
            for layer in (3,40,77):
                group=[c for c in selected if c['kind']==kind and c['layer']==layer]
                maxima={arm:max(c['median_ms'][arm] for c in group) for arm in ARMS}
                critical.append(dict(kind=kind,layer=layer,max_rank_median_ms=maxima,row32_vs_native=maxima['native']/maxima['row32']))
        summaries.append(dict(rows=rows,cells=36,row32_vs_native_geomean=math.exp(statistics.mean(map(math.log,ratios))),
            min_cell_ratio=min(ratios),max_cell_ratio=max(ratios),row32_faster_cells=sum(x>1 for x in ratios),
            nonoverlapping_row32_faster_cells=sum(c['nonoverlap'] for c in selected),max_rank_comparisons=critical))
    assert len(data['quality_checks'])==18
    assert {(r['rank'],r['layer']) for r in data['quality_checks']}=={(rank,layer) for rank in range(6) for layer in (3,40,77)}
    exact=0
    for result in data['quality_checks']:
        checks=result['checks'];assert len(checks)==72
        eager=[tuple(c[:3]) for c in checks if not c[3]]
        graph=[tuple(c[:3]) for c in checks if c[3]]
        assert eager==[(kind,rows,arm) for kind in ('code','prose') for rows in ROWS for arm in ARMS]
        assert graph==[(kind,rows,arm) for rows in (256,513) for arm in ARMS for kind in ('prose','code','prose')]
        assert all(len(c)==6 and c[4] is True and c[5]==0 for c in checks)
        assert result['peak_cuda_allocated_gib']<=3 and result['shared_e3_scratch_arenas']==1
        exact+=len(checks)
    return dict(exact_comparisons=exact,timed_cells=len(cells),total_timed_calls=len(cells)*8*3,summaries=summaries)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--samples',required=True,type=Path)
    a=p.parse_args();r=analyze(json.loads(a.samples.read_text()))
    print(json.dumps({k:v for k,v in r.items() if k!='summaries'}))
    for row in r['summaries']:
        print(json.dumps({k:v for k,v in row.items() if k!='max_rank_comparisons'}))


if __name__=='__main__':main()
