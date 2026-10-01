"""Bounded real-weight projection screening; microtimings only rank candidates.

No persistent weight conversion. A64MiB write between timing samples reduces
hot-cache bias, but is not proof of a complete L2 flush. Full-model tests decide
whether the chosen candidate is useful. Numerical rejection is recorded, never
silently treated as qualification. CUDA execution errors are fatal.
"""
from dataclasses import asdict
import math
import statistics

DECODE=((16,64,2),(16,32,2),(16,128,2),(16,64,3),(16,128,3))
GENERATIONS={mode:'tfp20_projection_'+mode for mode in ('reference','candidate')}
PREFILL=((16,64,2),(32,64,2),(32,128,2),(64,64,2),(64,128,2))


def tile_key(tile):
    return 'x'.join(map(str,tile))


def choose(rows):
    """Use one common exact tile; rank by the slowest rank's weighted cost."""
    if len(rows)!=6 or sorted(r['rank'] for r in rows)!=list(range(6)):
        raise ValueError('Projection selection needs all six distinct ranks')
    answer={}
    for mode,tiles in (('decode',DECODE),('prefill',PREFILL)):
        eligible=[]
        for tile in tiles:
            key=tile_key(tile)
            values=[r[mode][key] for r in rows]
            if not all(v['exact'] for v in values):continue
            costs=[v['score_ms'] for v in values]
            if not all(math.isfinite(x) and x>0 for x in costs):
                raise ValueError('Invalid projection measurement')
            eligible.append((max(costs),tile))
        if not eligible:raise ValueError('No numerically qualified projection candidate')
        chosen=min(eligible)[1]
        answer[mode]=dict(zip(('m','n','stages'),chosen))
    return answer


def screen(weights,report,save):
    import torch
    from triton.runtime.errors import OutOfResources
    from tensorfold.families.glm_moe_dsa.dense import linear
    from tensorfold.families.glm_moe_dsa.projection_plan import LinearTile,REFERENCE_TILE
    a=weights.layers[0].attention.weights
    d=weights.layers[0].ffn.weights
    s=weights.layers[3].ffn.weights.shared.weights
    # Ranks4/5 have no shared shard. Their tests use dense-weight subsets with
    # shared geometry and zero timing weight; actual shared shards tested0..3.
    shared_up=s.gate_up if s.width else d.gate_up[:1024]
    shared_down=s.down if s.width else d.down[:,:512].contiguous()
    matrices=[
        ('qkv',a.qkv_a,78,3072),('qb',a.q_b,78,3072),('o',a.o,78,3072),
        ('dense_up',d.gate_up,3,1024),('dense_down',d.down,3,1024),
        ('shared_up',shared_up,75 if s.width else 0,1024),
        ('shared_down',shared_down,75 if s.width else 0,1024),
        ('index_q',a.indexer.weights.wq,21,3072),
        ('index_kw',a.indexer.weights.wk_weights,21,3072),
        ('head',weights.vocab.head,1,0),('eh',weights.mtp.eh,4,0),
    ]
    result=dict(rank=weights.rank,decode={},prefill={},details=[],reference_fp64_samples=[])
    for mode,tiles in (('decode',DECODE),('prefill',PREFILL)):
        result[mode]={tile_key(t):dict(exact=True,score_ms=0.,checks=0,rejections=[]) for t in tiles}
    report['projection_screen']=result
    sweep=torch.empty(64*2**20,dtype=torch.uint8,device='cuda')
    def timing(fn):
        fn();torch.cuda.synchronize()
        events=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(5)]
        for start,stop in events:
            sweep.zero_();start.record();fn();stop.record()
        torch.cuda.synchronize()
        return statistics.median(start.elapsed_time(stop) for start,stop in events)
    for name,weight,frequency,bulk_rows in matrices:
        n,k=weight.shape
        for mode,tiles in (('decode',DECODE),('prefill',PREFILL)):
            if mode=='prefill' and not bulk_rows:continue
            rows=(1,5,17,128,255) if mode=='decode' else (256,257,bulk_rows)
            if name=='head':rows=(1,5,17)
            if name=='eh':rows=(1,5,17,128)
            for m in dict.fromkeys(rows):
                x=torch.randn((m,k),device='cuda',dtype=torch.bfloat16)
                reference=torch.empty((m,n),device='cuda',dtype=torch.bfloat16)
                out=torch.empty_like(reference)
                linear(x,weight,reference,tile=REFERENCE_TILE)
                if mode=='decode' and m==1:
                    cols=torch.linspace(0,n-1,16,device='cuda').long()
                    exact=(x[0].cpu().double()[None,:]*weight[cols].cpu().double()).sum(-1)
                    observed=reference[0,cols].cpu().double()
                    result['reference_fp64_samples'].append(dict(matrix=name,samples=16,max_abs=float((exact-observed).abs().max()),
                        caveat='Characterizes baseline BF16 rounding; candidate qualification uses exact baseline equality.'))
                for tile in tiles:
                    key=tile_key(tile);item=result[mode][key]
                    if not item['exact']:continue
                    choice=LinearTile(*tile)
                    try:
                        linear(x,weight,out,tile=choice)
                        equal=bool(torch.equal(out,reference)) and bool(torch.isfinite(out).all())
                    except OutOfResources as exc:
                        # A static compiler resource rejection is safe to skip.
                        # Runtime CUDA errors intentionally escape and fail run.
                        item['exact']=False;item['rejections'].append(dict(matrix=name,rows=m,reason=type(exc).__name__+': '+str(exc)));continue
                    item['checks']+=1
                    if not equal:
                        item['exact']=False;item['rejections'].append(dict(matrix=name,rows=m,reason='BF16 differs from unchanged kernel',max_abs=float((out.float()-reference.float()).abs().max())));continue
                    measured=None
                    if (mode=='decode' and m in (1,5)) or (mode=='prefill' and m==bulk_rows):
                        measured=timing(lambda:linear(x,weight,out,tile=choice))
                        # Target verification + four draft steps + canonical
                        # MTP commit. Still only a ranking heuristic: real
                        # acceptance and canonical row counts vary by request.
                        if mode=='decode':
                            draft_frequency=0 if name.startswith('dense_') or (name.startswith('shared_') and not s.width) else 1
                            target_frequency=0 if name=='eh' else frequency
                            factor=4*draft_frequency if m==1 else target_frequency+draft_frequency
                            if name in ('head','eh'):factor=4 if m==1 else 1
                        else:factor=frequency*3072/bulk_rows
                        item['score_ms']+=measured*factor
                    result['details'].append(dict(mode=mode,tile=key,matrix=name,shape=[m,n,k],exact=True,median_ms=measured))
                # Changed input and FP32-output check exposes compiler changes
                # hidden by BF16 rounding, on every weight geometry and tile.
                if m==17 or (mode=='prefill' and m==257):
                    x.neg_();ref32=torch.empty((m,n),device='cuda',dtype=torch.float32);out32=torch.empty_like(ref32)
                    linear(x,weight,ref32,tile=REFERENCE_TILE)
                    for tile in tiles:
                        key=tile_key(tile);item=result[mode][key]
                        if not item['exact']:continue
                        try:
                            linear(x,weight,out32,tile=LinearTile(*tile))
                            equal=bool(torch.equal(out32,ref32)) and bool(torch.isfinite(out32).all())
                        except OutOfResources as exc:
                            item['exact']=False;item['rejections'].append(dict(matrix=name,rows=m,reason=type(exc).__name__+': '+str(exc)));continue
                        item['checks']+=1
                        if not equal:
                            item['exact']=False;item['rejections'].append(dict(matrix=name,rows=m,reason='FP32 changed-input output differs',max_abs=float((out32-ref32).abs().max())))
                    del ref32,out32
                del x,reference,out
            save('projection-screen-'+name+'-'+mode)
    del matrices,shared_up,shared_down,sweep
    torch.cuda.synchronize();torch.cuda.empty_cache()
    # Reference must be present and finite; choose also validates its timings.
    assert all(result[mode][tile_key((16,64,2))]['exact'] for mode in ('decode','prefill'))
    return result
