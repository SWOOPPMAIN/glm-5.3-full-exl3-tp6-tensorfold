#!/usr/bin/env python3
"""All original target/MTP weights and end-to-end forward on six physical ranks.

This first complete-model gate uses a4096-slot cache and3072-row workspace.
360K/C4 admission, request scheduling and speed promotion remain separate gates.
"""
import os
import argparse,faulthandler,hashlib,json,time
from datetime import timedelta
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    for name in ('model','manifest','output','admission'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert 0<=a.rank<6 and not a.output.exists()
    assert int(Path('/sys/fs/cgroup/memory.max').read_text())<=80*2**30
    def available():return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available()>=96*2**30
    manifest=json.loads(a.manifest.read_text())
    for path,digest in manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    # The controller releases all ranks only after every diagnostic guard binds
    # its exact container ID. There is no GPU work before that authorization.
    deadline=time.monotonic()+90
    while not a.admission.exists():
        assert time.monotonic()<deadline,'Missing full-model guard admission'
        time.sleep(.1)
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(64*2**30/torch.cuda.get_device_properties(0).total_memory)
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from tensorfold.families.glm_moe_dsa.compiled import load_experts
    from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
    from tensorfold.families.glm_moe_dsa.model import ModelWeights,ModelWorkspace,FullModel
    from tensorfold.families.glm_moe_dsa.memory import weight_plan,workspace_plan,tensor_storage_bytes
    from tensorfold.families.glm_moe_dsa.cache import LayerCache
    from tensorfold.families.glm_moe_dsa.attention import rope_table
    from transformers import AutoTokenizer
    report=dict(passed=False,rank=a.rank,cases=[],started_at=time.time(),source_manifest=manifest,
        scope='All79 original layers loaded and full target/MTP eager/graph forward;4096cache slots,3072model workspace. No360K/C4 runtime capacity, speculative scheduler, quality-suite or serving-speed qualification.')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            cuda_allocated_gib=torch.cuda.memory_allocated()/2**30,available_gib=available()/2**30)
        cg=Path('/sys/fs/cgroup');report['cgroup_peak_gib']=int((cg/'memory.peak').read_text())/2**30
        temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.output)
        print(json.dumps({'rank':a.rank,'phase':phase,'cuda_gib':report['cuda_allocated_gib']}),flush=True)
    def check(got,ref,label):
        torch.cuda.synchronize();same=torch.equal(got,ref)
        ok=same and bool(torch.isfinite(got).all())
        report['cases'].append(dict(case=label,passed=ok,exact=True));save(label);assert ok,label
    def same_ranks(x,label):
        other=torch.empty_like(x)
        for source in range(6):
            other.copy_(x);dist.broadcast(other,source);check(other,x,label+f'-rank{source}')
    graphs=[]
    faulthandler.dump_traceback_later(600,repeat=True)
    try:
        save('initializing');reader=RankPieces(a.model,a.rank)
        plan=weight_plan(reader);workspace=workspace_plan(reader.config,a.rank,3072,4096,logit_rows=17)
        report['weight_plan']=plan;report['workspace_plan']=workspace
        assert plan['total']+workspace['total']+4*2**30<64*2**30,'Full model exceeds bounded64GiB Torch plan'
        dist.init_process_group('nccl',init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',rank=a.rank,world_size=6,timeout=timedelta(seconds=180))
        reduction=TP6Reduction(dist.group.WORLD,a.rank).prepare(3072,'cuda')
        load_experts()
        def progress(layer):
            assert available()>=12*2**30,'Host reserve fell below diagnostic12GiB floor'
            if layer%4==0 or layer==78:save(f'loaded-layer-{layer}')
        weights=ModelWeights(reader,on_layer=progress)
        actual=tensor_storage_bytes(weights,device_type='cuda');report['retained_weight_bytes']=actual
        assert actual==plan['total'],('Weight storage differs from admission estimate',actual,plan['total'])
        model=FullModel(weights,reduction);arena=ModelWorkspace(weights,3072,4096,logit_rows=17)
        actual_workspace=tensor_storage_bytes(arena,device_type='cuda',skip=('weights','layer'))
        assert actual_workspace==workspace['workspace'],('Scratch storage differs',actual_workspace,workspace['workspace'])
        caches=[LayerCache(i,4096,'cuda',indexer=reader.config.indexer_source(i)==i) for i in range(79)]
        table=rope_table(4096,'cuda');report['cache_bytes']=sum(c.nbytes() for c in caches)
        assert report['cache_bytes']==workspace['cache']
        # Eager compilation can differ per rank. A generously bounded NCCL group
        # and explicit barriers keep each semantic test at the same phase.
        save('all-original-weights-loaded');dist.barrier()
        def forward(ids,pos,base,slots):
            return model.target_forward(ids,pos,base,slots,caches[:78],table,arena,scope=object())
        ids=(torch.arange(17,device='cuda',dtype=torch.int64)*137+1000)%154880
        pos=torch.arange(17,device='cuda');base=torch.zeros_like(pos);slots=pos.clone()
        target=forward(ids,pos,base,slots).clone();save('full-target17');same_ranks(target,'target17')
        logits=model.logits(target,arena).clone();same_ranks(logits[:1],'target-logits')
        # Serial target execution rewrites the same causal prefix one token at
        # a time; future stale slots must be ignored. Shared-indexer producers
        # and all residual branches run through the actual78-layer chain.
        for i in range(17):check(forward(ids[i:i+1],pos[i:i+1],base[i:i+1],slots[i:i+1]),target[i:i+1],f'target-serial-{i}')
        changed=(ids+19)%154880
        # Shared workspace after a whole target pass must support real layer78.
        draft=model.mtp_forward(changed,target,pos,base,slots,caches[78],table,arena,scope=object()).clone()
        same_ranks(draft,'mtp17')
        for i in range(17):check(model.mtp_forward(changed[i:i+1],target[i:i+1],pos[i:i+1],base[i:i+1],slots[i:i+1],caches[78],table,arena,scope=object()),draft[i:i+1],f'mtp-serial-{i}')
        check(forward(ids,pos,base,slots),target,'target-after-mtp-selection-isolation')

        # Capture the entire target: all78 decoder layers, both reductions per
        # layer, shared indexers, original vocabulary and final normalization.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):forward(ids,pos,base,slots)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
        graph=torch.cuda.CUDAGraph();graphs.append(graph)
        with torch.cuda.graph(graph):out=forward(ids,pos,base,slots)
        ids.copy_(changed);expected=forward(ids,pos,base,slots).clone();graph.replay()
        check(out,expected,'full-target-changed-input-graph');same_ranks(out,'target-graph')
        graph.reset();graphs.clear()

        # Human-readable greedy smoke using the original tokenizer. This is a
        # plumbing check, not a benchmark or a complete quality evaluation.
        tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False)
        messages=[{'role':'user','content':'Answer briefly: what is 2 + 2?'}]
        prompt=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if hasattr(prompt,'keys'):prompt=prompt['input_ids']
        prompt_ids=torch.tensor(prompt,device='cuda',dtype=torch.int64)
        assert len(prompt)<=3072
        pp=torch.arange(len(prompt),device='cuda');bb=torch.full_like(pp,512);ss=pp+bb
        x=forward(prompt_ids,pp,bb,ss);next_id=int(model.logits(x[-1:].contiguous(),arena).argmax(-1).item())
        generated=[]
        for step in range(32):
            generated.append(next_id)
            if next_id in reader.config.eos:break
            ti=torch.tensor([next_id],device='cuda');po=torch.tensor([len(prompt)+step],device='cuda');ba=torch.tensor([512],device='cuda')
            x=forward(ti,po,ba,po+ba);next_id=int(model.logits(x,arena).argmax(-1).item())
        report['greedy_smoke']=dict(prompt_tokens=len(prompt),token_ids=generated,text=tok.decode(generated),quality_gate=False)
        same_ranks(torch.tensor(generated,device='cuda'),'greedy-token-stream')
        save('numerical_checks_complete');torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
        report.update(passed=True,communicator_destroyed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:
        for graph in graphs:graph.reset()
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
