#!/usr/bin/env python3
"""Compare original full-model expert chunks at resident804K capacity on TP6.

Complete target3072 and MTP3072 hidden states/logits must match128-row chunks.
Whole-pass timings are uninstrumented; CUDA-event stage profiles are separate.
This remains a forward benchmark, not a serving-quality or scheduler gate.
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
    torch.set_num_threads(1);torch.cuda.set_device(0);torch.manual_seed(530613)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(104*2**30/torch.cuda.get_device_properties(0).total_memory)
    capacity=804000
    report=dict(passed=False,rank=a.rank,cases=[],started_at=time.time(),source_manifest=manifest,scope=__doc__,
        cache_capacity=capacity,model_rows=3072,candidates={},minimum_available_gib=available()/2**30,
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
        return dict(milliseconds=samples,median_ms=statistics.median(samples))
    graphs=[];faulthandler.dump_traceback_later(600,repeat=True)
    try:
        reader=RankPieces(a.model,a.rank);plan=weight_plan(reader)
        wp=workspace_plan(reader.config,a.rank,3072,capacity,logit_rows=17,expert_chunk_rows=1024)
        assert plan['total']+wp['total']+4*2**30<104*2**30
        report.update(weight_plan=plan,largest_workspace_plan=wp)
        dist.init_process_group('nccl',init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',rank=a.rank,world_size=6,timeout=timedelta(seconds=240))
        reduction=TP6Reduction(dist.group.WORLD,a.rank).prepare(3072,'cuda');load_experts()
        def progress(layer):
            assert available()>=12*2**30
            if layer%4==0 or layer==78:save(f'loaded-layer-{layer}')
        weights=ModelWeights(reader,on_layer=progress)
        report['retained_weight_bytes']=tensor_storage_bytes(weights,device_type='cuda');assert report['retained_weight_bytes']==plan['total']
        model=FullModel(weights,reduction)
        # Release unused load-time allocator blocks before allocating the pool.
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
        ids=(torch.arange(3072,device='cuda',dtype=torch.int64)*137+1000)%154880
        pos=torch.arange(3072,device='cuda');base=torch.zeros_like(pos);draft_ids=(ids+19)%154880
        # Live reference copies have separate storage; no workspace is retained
        # while the next candidate is allocated. No model weights are reloaded.
        reference=reference_draft=reference_logits=reference_draft_logits=None
        for chunk in (128,256,512,1024):
            torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();dist.barrier()
            arena=ModelWorkspace(weights,3072,capacity,logit_rows=17,expert_chunk_rows=chunk)
            actual=tensor_storage_bytes(arena,device_type='cuda',skip=('weights','layer'))
            wanted=workspace_plan(reader.config,a.rank,3072,capacity,logit_rows=17,expert_chunk_rows=chunk)
            assert actual==wanted['workspace'] and arena.decoder.moe.routed.chunk_rows==chunk
            assert available()>=10*2**30
            item=dict(workspace_bytes=actual,grouping_dynamic_shared_bytes=chunk*8*4,guard_reserve_before_gib=available()/2**30)
            report['candidates'][str(chunk)]=item;save(f'chunk{chunk}-allocated')
            def target():return model.target_forward(ids,pos,base,pos,caches[:78],table,arena,scope=object(),visible_tokens=3072)
            full=target().clone()
            logits=model.logits(full[-1:].contiguous(),arena).clone()
            if reference is None:
                reference=full.clone();reference_logits=logits.clone();same_ranks(full[-1:],'reference-target')
            check(full,reference,f'chunk{chunk}-target3072');check(logits,reference_logits,f'chunk{chunk}-target-logits')
            item['target3072']=timing(target)
            # Changed routing inputs and an uneven last expert chunk.
            uneven=ids[:2053]+37;up=pos[:2053];ub=base[:2053]
            changed=model.target_forward(uneven,up,ub,up,caches[:78],table,arena,scope=object(),visible_tokens=2053).clone()
            if chunk==128:reference_changed=changed.clone()
            check(changed,reference_changed,f'chunk{chunk}-changed2053')
            target()  # Refill the original target prefix before the MTP comparison.
            def mtp():return model.mtp_forward(draft_ids,reference,pos,base,pos,caches[78],table,arena,scope=object(),visible_tokens=3072)
            draft=mtp().clone();draft_logits=model.logits(draft[-1:].contiguous(),arena).clone()
            if reference_draft is None:reference_draft=draft.clone();reference_draft_logits=draft_logits.clone()
            check(draft,reference_draft,f'chunk{chunk}-mtp3072');check(draft_logits,reference_draft_logits,f'chunk{chunk}-mtp-logits')
            item['mtp3072']=timing(mtp)
            # Short decode must remain exact even with a larger allocated arena.
            small=ids[:17].clone();sp=pos[:17];sb=base[:17]
            def short():return model.target_forward(small,sp,sb,sp,caches[:78],table,arena,scope=object(),visible_tokens=17)
            check(short(),reference[:17],f'chunk{chunk}-short17')
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):short()
            torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
            graph=torch.cuda.CUDAGraph();graphs.append(graph)
            with torch.cuda.graph(graph):out=short()
            small.add_(31);expected=short().clone();graph.replay();check(out,expected,f'chunk{chunk}-changed-graph')
            item['short17_graph']=timing(graph.replay)
            graph.reset();graphs.clear()
            # Stage profiles deliberately separate from uninstrumented timings.
            if chunk in (128,1024):
                target();torch.cuda.synchronize();dist.barrier()
                with StageProfile() as profile:target()
                item['stage_profile']=profile.result()
                item['profile_scope']='Eager CUDA-event intervals include stream waits/host launch gaps; attention/indexer/projection categories overlap. Performance comparison uses uninstrumented target timings.'
                del profile
            assert available()>=10*2**30
            item.update(available_after_gib=available()/2**30,cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,cuda_reserved_gib=torch.cuda.memory_reserved()/2**30)
            save(f'chunk{chunk}-complete')
            del full,logits,changed,draft,draft_logits,small,sp,sb,expected,out,arena,target,mtp,short
        save('numerical_checks_complete');torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:
        for graph in graphs:graph.reset()
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
