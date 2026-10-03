#!/usr/bin/env python3
"""CPU analysis of real E3 route captures; geometry estimates are not timings."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def geometry(counts, tile):
    segments=sum((n+tile-1)//tile for n in counts)
    actual=sum(counts);padded=segments*tile
    return dict(segments=segments,actual_rows=actual,padded_rows=padded,
                unused_row_fraction=1-actual/padded if padded else 0)


def analyze(source):
    assert source['phase']=='complete' and source['passed']
    cases=[]
    for case in source['cases']:
        assert case['phase']=='verified' and case['control_removed']
        ranks=case['ranks'];assert [r['rank'] for r in ranks]==list(range(6))
        for rank in ranks:
            assert rank['verified'] and [m['layer'] for m in rank['rows']]==list(range(3,78))
        layers=[]
        for i,layer in enumerate(range(3,78)):
            rows=[r['rows'][i] for r in ranks]
            # All ranks should see the same routed input before local ownership.
            shared={key:len({r['tensor_sha256'][key] for r in rows})==1 for key in ('ids','weights')}
            assert all(shared.values()), (case['kind'],layer,shared)
            assert all(sum(r['global_route_counts'])==3072*8 for r in rows)
            observed=[]
            for r in rows:
                counts=r['local_route_counts'];mapping=r['global_to_combined']
                assert len(counts)==r['experts']==len(r['bits'])
                expect=[0]*len(counts)
                for expert,local in enumerate(mapping):
                    if local>=0:expect[local]+=r['global_route_counts'][expert]
                assert counts==expect
                for expert,local in enumerate(mapping):
                    assert (local>=0)==any((4*expert+s)%6==r['rank'] for s in range(4))
                g={str(tile):geometry(counts,tile) for tile in (16,32,64)}
                assert g['64']['segments']==r['tile64_count']
                assert g['64']['actual_rows']==r['actual_local_rows']
                observed.append(dict(rank=r['rank'],experts=r['experts'],active_experts=sum(n>0 for n in counts),
                    geometry=g,tail_rows_le32=sum(0<n%64<=32 for n in counts),
                    k3_rows=sum(n for n,b in zip(counts,r['bits']) if b==3),
                    k4_rows=sum(n for n,b in zip(counts,r['bits']) if b==4)))
            assert sum(r['actual_local_rows'] for r in rows)==4*3072*8
            active=[r['actual_local_rows'] for r in rows]
            padded=[r['tile64_rows'] for r in rows]
            layers.append(dict(layer=layer,ranks=observed,identical_ids_and_weights=True,
                active_max_to_mean=max(active)/statistics.mean(active),
                padded_max_to_mean=max(padded)/statistics.mean(padded)))
        aggregates=[]
        for rank in range(6):
            local=[l['ranks'][rank] for l in layers]
            geom={}
            for tile in ('16','32','64'):
                totals={k:sum(r['geometry'][tile][k] for r in local) for k in ('segments','actual_rows','padded_rows')}
                totals['unused_row_fraction']=1-totals['actual_rows']/totals['padded_rows']
                geom[tile]=totals
            aggregates.append(dict(rank=rank,geometry=geom,
                k3_rows=sum(r['k3_rows'] for r in local),k4_rows=sum(r['k4_rows'] for r in local)))
        totals={tile:{key:sum(r['geometry'][tile][key] for r in aggregates) for key in ('segments','actual_rows','padded_rows')} for tile in ('16','32','64')}
        for g in totals.values():g['unused_row_fraction']=1-g['actual_rows']/g['padded_rows']
        worst=sorted(layers,key=lambda l:l['padded_max_to_mean'],reverse=True)[:5]
        cases.append(dict(kind=case['kind'],layers=layers,rank_totals=aggregates,totals=totals,
            tile32_padded_row_reduction=1-totals['32']['padded_rows']/totals['64']['padded_rows'],
            tile32_segment_multiplier=totals['32']['segments']/totals['64']['segments'],
            mean_layer_padded_max_to_mean=statistics.mean(l['padded_max_to_mean'] for l in layers),
            worst_padded_imbalance=[{k:l[k] for k in ('layer','active_max_to_mean','padded_max_to_mean')} for l in worst]))
    return dict(phase='analysis_complete',image=source['image'],cases=cases,
        scope='Two synthetic cold prompts; first3072-token chunk,75 routed layers,6 ranks. Local shard outputs at3/40/77 saved for replay. Does not characterize all prompts or later chunks.',
        limitations=['Row padding and register pressure are hypotheses, not measured speedups.',
            'Smaller tiles reduce unused row arithmetic but increase segment count and packed-weight fetch/dequantization work.',
            'Current full64-row kernel and native512-row dispatch remain selected; no kernel change in this diagnostic.'],
        next_experiment='Compare row32 versus current row64 on exact captured inputs/routes with bitwise component checks, then repeated full-model gates and serving benchmarks if it wins.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    report=analyze(json.loads(a.input.read_text()));report['source_sha256']=hashlib.sha256(a.input.read_bytes()).hexdigest()
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:c[k] for k in ('kind','totals','tile32_padded_row_reduction','tile32_segment_multiplier','mean_layer_padded_max_to_mean','worst_padded_imbalance')} for c in report['cases']],indent=2))


if __name__=='__main__':main()
