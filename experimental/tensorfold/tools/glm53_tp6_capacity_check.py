#!/usr/bin/env python3
"""All original weights, a resident804K cache pool and real3072-row target work.

Four explicit extents exercise batch isolation, including a360K frontier backed
by initialized synthetic old cache entries. This does not prove authentic360K
prefill quality, request scheduling, recursive speculation or serving throughput.
"""
import os
import argparse,faulthandler,hashlib,json,time,statistics
from datetime import timedelta
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    for name in ('model','manifest','output','admission'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert 0<=a.rank<6 and not a.output.exists()
    assert int(Path('/sys/fs/cgroup/memory.max').read_text())<=108*2**30
    def available():return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available()>=110*2**30
    manifest=json.loads(a.manifest.read_text())
    for path,digest in manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    deadline=time.monotonic()+90
    while not a.admission.exists():
        assert time.monotonic()<deadline,'Missing exact-CID guard admission';time.sleep(.1)
    import torch
    import torch.distributed as dist
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from tensorfold.families.glm_moe_dsa.compiled import load_experts
    from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
    from tensorfold.families.glm_moe_dsa.model import ModelWeights,ModelWorkspace,FullModel
    from tensorfold.families.glm_moe_dsa.memory import weight_plan,workspace_plan,tensor_storage_bytes
    from tensorfold.families.glm_moe_dsa.cache import LayerCache
    from tensorfold.families.glm_moe_dsa.attention import rope_table
    from tensorfold.families.glm_moe_dsa.indexer import select_tokens
    from tensorfold.families.glm_moe_dsa.indexer_plan import visible_token_bound
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(530612)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(104*2**30/torch.cuda.get_device_properties(0).total_memory)
    capacity=804000
    report=dict(passed=False,rank=a.rank,cases=[],started_at=time.time(),source_manifest=manifest,
        scope=__doc__,cache_capacity=capacity,model_rows=3072,timings={},minimum_available_gib=available()/2**30,
        real_long_prefill_quality=False,request_scheduler_qualified=False,serving_throughput=False)
    def save(phase):
        free=available()/2**30
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,available_gib=free,
            minimum_available_gib=min(report['minimum_available_gib'],free),
            cgroup_peak_gib=int(Path('/sys/fs/cgroup/memory.peak').read_text())/2**30)
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
        print(json.dumps({'rank':a.rank,'phase':phase,'cuda_gib':report['cuda_allocated_gib'],'available_gib':free}),flush=True)
    def check(got,expected,label):
        torch.cuda.synchronize();ok=torch.equal(got,expected) and bool(torch.isfinite(got).all())
        report['cases'].append(dict(case=label,passed=ok,exact=True));save(label);assert ok,label
    def same_ranks(x,label):
        other=torch.empty_like(x)
        for rank in range(6):
            other.copy_(x);dist.broadcast(other,rank);check(other,x,label+f'-rank{rank}')
    def timing(fn,repeats=3):
        samples=[]
        for _ in range(repeats):
            torch.cuda.synchronize();dist.barrier()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            begin.record();fn();end.record();end.synchronize();samples.append(begin.elapsed_time(end))
        return dict(milliseconds=samples,median_ms=statistics.median(samples),scope='CUDA stream interval including host launch gaps; no serving/tokenizer/scheduler claim')
    graphs=[];faulthandler.dump_traceback_later(600,repeat=True)
    try:
        reader=RankPieces(a.model,a.rank);plan=weight_plan(reader)
        wp=workspace_plan(reader.config,a.rank,3072,capacity,logit_rows=17)
        assert plan['total']+wp['total']+4*2**30<104*2**30
        report.update(weight_plan=plan,workspace_plan=wp)
        dist.init_process_group('nccl',init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',rank=a.rank,world_size=6,timeout=timedelta(seconds=240))
        reduction=TP6Reduction(dist.group.WORLD,a.rank).prepare(3072,'cuda');load_experts()
        def progress(layer):
            assert available()>=12*2**30
            if layer%4==0 or layer==78:save(f'loaded-layer-{layer}')
        weights=ModelWeights(reader,on_layer=progress)
        report['retained_weight_bytes']=tensor_storage_bytes(weights,device_type='cuda')
        assert report['retained_weight_bytes']==plan['total']
        model=FullModel(weights,reduction);arena=ModelWorkspace(weights,3072,capacity,logit_rows=17)
        assert tensor_storage_bytes(arena,device_type='cuda',skip=('weights','layer'))==wp['workspace']
        caches=[]
        for layer in range(79):
            c=LayerCache(layer,capacity,'cuda',indexer=reader.config.indexer_source(layer)==layer)
            # Touch the entire admitted pool, not just empty virtual allocations.
            c.latent.zero_();c.scales.fill_(1);c.rope.zero_()
            if c.index_keys is not None:c.index_keys.zero_();c.index_scales.fill_(1)
            caches.append(c)
            if layer%8==0:
                torch.cuda.synchronize();assert available()>=12*2**30;save(f'cache-resident-layer-{layer}')
        table=rope_table(capacity,'cuda');torch.cuda.synchronize()
        report['cache_bytes']=sum(c.nbytes() for c in caches);assert report['cache_bytes']==wp['cache']
        assert available()>=12*2**30
        save('resident804K-pool');dist.barrier()
        tensor=lambda x:torch.tensor(x,dtype=torch.int64,device='cuda')
        # Exact old/new selector comparison on identical cache and query data,
        # at topK, chunk and long-context boundaries, with nonzero extent bases.
        scratch=arena.decoder.attention.indexer;ic=caches[0]
        ic.index_keys[:32768].copy_(torch.randn((32768,128),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn))
        q8=torch.randn((4,32,128),device='cuda',dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        ww=torch.randn((4,32),device='cuda',dtype=torch.float32)
        for label,positions,bases in (
            ('short',[0,16,1023,2047],[0,0,100000,444000]),
            ('threshold',[2048,3071,4095,16384],[0,0,100000,444000]),
            ('long',[32767,131071,359999,511],[0,0,444000,100000])):
            po,ba=tensor(positions),tensor(bases);bound=visible_token_bound(positions,capacity)
            def selected(bound=None):return select_tokens(q8,ww,ic.index_keys,ic.index_scales,po,ba,scratch,visible_tokens=bound)
            old=tuple(t.clone() for t in selected());new=selected(bound)
            check(new[0],old[0],label+'-bounded-ids');check(new[1],old[1],label+'-bounded-counts')
            report['timings']['selector-'+label+'-full']=timing(selected)
            report['timings']['selector-'+label+'-bounded']=timing(lambda:selected(bound))
        ic.index_keys.zero_()
        def forward(ids,po,ba,slots,bound):
            return model.target_forward(ids,po,ba,slots,caches[:78],table,arena,scope=object(),visible_tokens=bound)
        # Execute the full3072 rows through all78 original target layers.
        ids=(torch.arange(3072,device='cuda',dtype=torch.int64)*137+1000)%154880
        po=torch.arange(3072,device='cuda');ba=torch.zeros_like(po)
        t=time.perf_counter();full=forward(ids,po,ba,po,3072).clone();torch.cuda.synchronize()
        report['first3072_seconds']=time.perf_counter()-t;save('full-target3072')
        same_ranks(full[-1:],'3072-last-hidden')
        for start in (0,1024,2048):
            end=start+1024
            check(forward(ids[start:end],po[start:end],ba[start:end],po[start:end],end),full[start:end],f'chunk1024-{start}')
        report['timings']['target3072-eager']=timing(lambda:forward(ids,po,ba,po,3072),repeats=2)
        logits=model.logits(full[-1:].contiguous(),arena).clone();same_ranks(logits,'3072-last-logits')
        del full
        # Four disjoint request extents: one360K and three148K, total804K.
        extent_bases=[0,360000,508000,656000];extent_sizes=[360000,148000,148000,148000]
        report['explicit_extents']=list(zip(extent_bases,extent_sizes))
        assert sum(extent_sizes)==capacity and all(extent_bases[i]+extent_sizes[i]==extent_bases[i+1] for i in range(3))
        positions=list(range(17))*4
        po=tensor(positions);ba=tensor([b for b in extent_bases for _ in range(17)])
        ids=tensor([(1000+j*53+r*10000)%154880 for r in range(4) for j in range(17)])
        slots=po+ba;multi=forward(ids,po,ba,slots,17).clone()
        for rank in range(4):
            sl=slice(rank*17,(rank+1)*17)
            check(forward(ids[sl],po[sl],ba[sl],slots[sl],17),multi[sl],f'four-extents-prefix-{rank}')
        # Long-frontier arithmetic over initialized synthetic old caches. Real
        #360K prefill and multi-request serving remain later gates.
        positions=[359999,147999,147999,147999];bound=visible_token_bound(positions,capacity)
        po=tensor(positions);ba=tensor(extent_bases);slots=po+ba;ids=tensor([401,509,607,701])
        frontier=forward(ids,po,ba,slots,bound).clone();same_ranks(frontier,'long-frontier')
        for row in range(4):
            sl=slice(row,row+1)
            check(forward(ids[sl],po[sl],ba[sl],slots[sl],positions[row]+1),frontier[sl],f'long-frontier-serial-{row}')
        draft=model.mtp_forward(ids,frontier,po,ba,slots,caches[78],table,arena,scope=object(),visible_tokens=bound).clone()
        same_ranks(draft,'long-mtp')
        check(forward(ids,po,ba,slots,bound),frontier,'target-after-long-mtp')
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):forward(ids,po,ba,slots,bound)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
        graph=torch.cuda.CUDAGraph();graphs.append(graph)
        with torch.cuda.graph(graph):out=forward(ids,po,ba,slots,bound)
        ids.add_(19);expected=forward(ids,po,ba,slots,bound).clone();graph.replay()
        check(out,expected,'long-frontier-changed-input-graph')
        report['timings']['long-frontier-graph']=timing(graph.replay)
        graph.reset();graphs.clear()
        assert available()>=12*2**30
        save('numerical_checks_complete');torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:
        for graph in graphs:graph.reset()
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
