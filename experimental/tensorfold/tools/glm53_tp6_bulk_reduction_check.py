#!/usr/bin/env python3
"""Qualify fixed-order row-sharded TP6 reductions and full original model.

Independent CPU rank-order oracle, adversarial cancellation, uneven tails and
changed-input collective graphs precede model loading. Then compare complete
target and MTP hidden states/logits,1024-row expert chunks,804K resident cache.
Synthetic forward benchmarks do not qualify serving speed or long-context quality.
"""
import os
import argparse,faulthandler,gc,hashlib,json,statistics,time
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
    from glm53_tp6_stage_profile import StageProfile
    from glm53_tp6_reduction_check import qualify
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(530614)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(104*2**30/torch.cuda.get_device_properties(0).total_memory)
    capacity=804000
    report=dict(passed=False,rank=a.rank,cases=[],started_at=time.time(),source_manifest=manifest,scope=__doc__,
        cache_capacity=capacity,model_rows=3072,expert_chunk_rows=1024,candidates={},minimum_available_gib=available()/2**30,
        full_quality_gate=False,serving_throughput=False)
    def save(phase):
        free=available()/2**30
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,cuda_reserved_gib=torch.cuda.memory_reserved()/2**30,
            available_gib=free,minimum_available_gib=min(free,report['minimum_available_gib']),
            cgroup_peak_gib=int(Path('/sys/fs/cgroup/memory.peak').read_text())/2**30)
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
        print(json.dumps({'rank':a.rank,'phase':phase,'available_gib':free}),flush=True)
    def check(got,expected,label):
        torch.cuda.synchronize();ok=torch.equal(got,expected) and bool(torch.isfinite(got).all())
        report['cases'].append(dict(case=label,passed=ok,exact=True));save(label);assert ok,label
    def timing(fn,repeats=5):
        samples=[];wall=[]
        for _ in range(repeats):
            torch.cuda.synchronize();dist.barrier()
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start=time.perf_counter();begin.record();fn();end.record();end.synchronize()
            wall.append((time.perf_counter()-start)*1000);samples.append(begin.elapsed_time(end))
        return dict(milliseconds=samples,median_ms=statistics.median(samples),wall_ms=wall,median_wall_ms=statistics.median(wall))
    graphs=[];faulthandler.dump_traceback_later(600,repeat=True)
    try:
        dist.init_process_group('nccl',init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',rank=a.rank,world_size=6,timeout=timedelta(seconds=240))
        qualify(a.rank,report,check,save,timing)
        reader=RankPieces(a.model,a.rank);plan=weight_plan(reader)
        wp=workspace_plan(reader.config,a.rank,3072,capacity,logit_rows=17,expert_chunk_rows=1024,bulk_min_rows=256)
        assert plan['total']+wp['total']+4*2**30<104*2**30
        report.update(weight_plan=plan,workspace_plan=wp)
        reduction=TP6Reduction(dist.group.WORLD,a.rank).prepare(3072,'cuda',bulk_min_rows=256)
        assert tensor_storage_bytes(reduction,device_type='cuda',skip=('group',))==wp['collective']
        load_experts()
        def progress(layer):
            assert available()>=12*2**30
            if layer%4==0 or layer==78:save(f'loaded-layer-{layer}')
        weights=ModelWeights(reader,on_layer=progress)
        report['retained_weight_bytes']=tensor_storage_bytes(weights,device_type='cuda');assert report['retained_weight_bytes']==plan['total']
        model=FullModel(weights,reduction)
        torch.cuda.synchronize();torch.cuda.empty_cache()
        caches=[]
        for layer in range(79):
            c=LayerCache(layer,capacity,'cuda',indexer=reader.config.indexer_source(layer)==layer)
            c.latent.zero_();c.scales.fill_(1);c.rope.zero_()
            if c.index_keys is not None:c.index_keys.zero_();c.index_scales.fill_(1)
            caches.append(c)
            if layer%8==0:
                torch.cuda.synchronize();assert available()>=10*2**30;save(f'cache-resident-layer-{layer}')
        table=rope_table(capacity,'cuda');torch.cuda.synchronize();torch.cuda.empty_cache()
        report['cache_bytes']=sum(c.nbytes() for c in caches);assert report['cache_bytes']==wp['cache']
        arena=ModelWorkspace(weights,3072,capacity,logit_rows=17,expert_chunk_rows=1024)
        assert tensor_storage_bytes(arena,device_type='cuda',skip=('weights','layer'))==wp['workspace']
        ids=(torch.arange(3072,device='cuda',dtype=torch.int64)*137+1000)%154880
        pos=torch.arange(3072,device='cuda');base=torch.zeros_like(pos);draft_ids=(ids+19)%154880
        references={}
        for mode,threshold in (('gather',None),('sharded256',256)):
            reduction.bulk_min_rows=threshold
            torch.cuda.synchronize();dist.barrier();assert available()>=10*2**30
            item={};report['candidates'][mode]=item;save(mode+'-begin')
            def compare(value,label):
                if mode=='gather':references[label]=value.clone()
                check(value,references[label],mode+'-'+label)
            def target():return model.target_forward(ids,pos,base,pos,caches[:78],table,arena,scope=object(),visible_tokens=3072)
            full=target().clone();compare(full,'target3072')
            compare(model.logits(full[-1:].contiguous(),arena),'target-logits')
            item['target3072']=timing(target)
            uneven=ids[:2053]+37;up=pos[:2053];ub=base[:2053]
            changed=model.target_forward(uneven,up,ub,up,caches[:78],table,arena,scope=object(),visible_tokens=2053).clone()
            compare(changed,'changed2053');target()
            def mtp():return model.mtp_forward(draft_ids,references['target3072'],pos,base,pos,caches[78],table,arena,scope=object(),visible_tokens=3072)
            draft=mtp().clone();compare(draft,'mtp3072')
            compare(model.logits(draft[-1:].contiguous(),arena),'mtp-logits');item['mtp3072']=timing(mtp)
            small=ids[:17].clone();sp=pos[:17];sb=base[:17]
            def short():return model.target_forward(small,sp,sb,sp,caches[:78],table,arena,scope=object(),visible_tokens=17)
            compare(short(),'short17')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):short()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
            graph=torch.cuda.CUDAGraph();graphs.append(graph)
            with torch.cuda.graph(graph):out=short()
            small.add_(31);expected=short().clone();graph.replay();check(out,expected,mode+'-changed-graph')
            item['short17_graph']=timing(graph.replay,repeats=11);graph.reset();graphs.clear()
            target();torch.cuda.synchronize();dist.barrier()
            with StageProfile() as profile:target()
            item['stage_profile']=profile.result()
            item['profile_scope']='Instrumented CUDA-event intervals overlap and include host gaps/stream waits; use separate uninstrumented target timings for the speed comparison.'
            del profile,full,changed,draft,small,sp,sb,expected,out,graph,stream,target,mtp,short,compare
            assert available()>=10*2**30;save(mode+'-complete')
        save('numerical_checks_complete');torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:
        for graph in graphs:graph.reset()
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
