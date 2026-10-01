#!/usr/bin/env python3
"""Bounded six-node original vocabulary/EH checks; no full-model serving claim."""
import os
import argparse,hashlib,json,time
from datetime import timedelta
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    for name in ('model','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();assert 0<=a.rank<6 and not a.output.exists()
    assert int(Path('/sys/fs/cgroup/memory.max').read_text())<=4*2**30
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=64*2**30
    manifest=json.loads(a.manifest.read_text())
    for path,digest in manifest.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction
    from tensorfold.families.glm_moe_dsa.model import MTPWeights,MTPScratch,MTPMix
    from tensorfold.families.glm_moe_dsa.vocab import VocabWeights,VocabScratch,Vocabulary,WIDTH,VOCAB
    from glm53_tp6_reference import bf16_nearest
    report=dict(passed=False,rank=a.rank,cases=[],started_at=time.time(),source_manifest=manifest,
        scope='Original sharded embedding/head/EH on six physical ranks; bounded row loading, batch and graph invariance. No78-layer target/MTP engine or throughput proof.')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        cg=Path('/sys/fs/cgroup');stat=dict(s.split() for s in (cg/'memory.stat').read_text().splitlines())
        report['memory']=dict(current_gib=int((cg/'memory.current').read_text())/2**30,
            peak_gib=int((cg/'memory.peak').read_text())/2**30,anon_gib=int(stat['anon'])/2**30,shmem_gib=int(stat['shmem'])/2**30)
        temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.output)
        print(json.dumps({'rank':a.rank,'phase':phase}),flush=True)
    def compare(got,ref,label,*,exact=False,tolerance=.015):
        torch.cuda.synchronize();x,y=got.detach().cpu(),ref.detach().cpu()
        same=torch.equal(x,y);finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        if exact:rms=0. if same else float('inf')
        else:
            x,y=x.double(),y.double();rms=float((x-y).square().mean().sqrt())/(float(y.square().mean().sqrt()) or 1.)
        passed=finite and (same if exact else rms<=tolerance)
        report['cases'].append(dict(case=label,passed=passed,exact_required=exact,bit_equal=same,relative_rms=rms))
        save(label);assert passed,(label,rms)
    def same_ranks(value,label):
        other=torch.empty_like(value)
        for source in range(6):
            other.copy_(value);dist.broadcast(other,source)
            compare(other,value,label+f'-rank{source}',exact=True)
    def rms(x,w):
        x=x.double();return bf16_nearest(x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-5)*w.double())
    save('initializing')
    dist.init_process_group('nccl',init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{a.port}',rank=a.rank,world_size=6,timeout=timedelta(seconds=120))
    group=dist.group.WORLD
    reduction=TP6Reduction(group,a.rank).prepare(3072,'cuda')
    reader=RankPieces(a.model,a.rank);weights=VocabWeights(reader)
    vocab=Vocabulary(weights,reduction);vs=VocabScratch(3072,'cuda',logit_rows=17)
    mw=MTPWeights(reader);mix=MTPMix(mw,reduction);ms=MTPScratch(3072,'cuda')
    save('loaded-original-shards')
    tokens=(torch.arange(3072,device='cuda',dtype=torch.int64)*137+7)%VOCAB
    edge=[0,1,25855,25856,51711,51712,77567,77568,103423,103424,129279,129280,154878,154879]
    tokens[:len(edge)].copy_(torch.tensor(edge,device='cuda'))
    embedded=vocab.embed(tokens,vs).clone()
    for i,token in enumerate(edge):
        reference=reader.read_rows('model.embed_tokens.weight',token,token+1)
        compare(embedded[i:i+1],reference,f'original-embedding-boundary-{token}',exact=True)
    same_ranks(embedded[:17],'embedding-global')
    for i in (0,1,15,16,127,128,129,1023,1535,3071):
        compare(vocab.embed(tokens[i:i+1],vs),embedded[i:i+1],f'embedding-serial-{i}',exact=True)

    # Every rank computes its original vocabulary slice; padding must be masked
    # even when poisoned, and global outputs must retain exact token ordering.
    hidden=(torch.sin(torch.arange(17*6144,device='cuda').float()/19)*.05).reshape(17,6144).bfloat16()
    if a.rank==5:weights.head[weights.stop-weights.start:].fill_(float('nan'))
    logits=vocab.project(hidden,vs).clone()
    columns=torch.tensor([0,1,127,128,1023,4095,8191,16383,25599],device='cuda')
    reference=bf16_nearest(hidden[:3].cpu().double()@weights.head[columns].cpu().double().T).float()
    compare(logits[:3,weights.start+columns],reference,'original-head-fp64')
    same_ranks(logits[:1],'head-global')
    for i in (0,1,8,16):compare(vocab.project(hidden[i:i+1],vs),logits[i:i+1],f'head-serial-{i}',exact=True)
    assert logits.shape==(17,VOCAB) and bool(torch.isfinite(logits).all())
    if a.rank==5:assert bool(torch.isneginf(vs.local_logits[0,weights.stop-weights.start:]).all())
    report['padding_excluded']=True

    positions=torch.arange(3072,device='cuda',dtype=torch.int64)
    previous=(torch.cos(torch.arange(3072*6144,device='cuda').float()/71)*.1).reshape(3072,6144).bfloat16()
    mixed=mix.forward(embedded,previous,positions,ms).clone()
    # Independent CPU FP64 norm and EH projection, preserving the specified BF16
    # boundaries. The first row uses zero embedding before normalization.
    emb=embedded[:3].cpu().clone();emb[0].zero_()
    joined=torch.cat((rms(emb,mw.enorm.cpu()),rms(previous[:3].cpu(),mw.hnorm.cpu())),dim=-1)
    reference=bf16_nearest(joined.double()@mw.eh.cpu().double().T)
    compare(mixed[:3,a.rank*1024:(a.rank+1)*1024],reference,'original-mtp-mix-fp64')
    same_ranks(mixed[:17],'mtp-mix-global')
    compare(mix.forward(embedded[:17],previous[:17],positions[:17],ms),mixed[:17],'mtp-mix-17rows',exact=True)
    for i in (0,1,15,16,127,128,129,1023,1535,3071):
        compare(mix.forward(embedded[i:i+1],previous[i:i+1],positions[i:i+1],ms),mixed[i:i+1],f'mtp-mix-serial-{i}',exact=True)

    # Changed inputs must affect replay. Embedding, EH gather and head gather
    # are captured in one graph, with fixed caller-owned buffer addresses.
    ids=tokens[:17].clone();pos=positions[:17].clone();prev=previous[:17].clone()
    def run():
        x=vocab.embed(ids,vs);x=mix.forward(x,prev,pos,ms);return vocab.project(x,vs)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):run()
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=run()
    ids.copy_(tokens[31:48]);pos.copy_(positions[31:48]);prev.copy_(previous[31:48])
    expected=run().clone();graph.replay()
    compare(out,expected,'changed-input-vocabulary-mtp-graph',exact=True)
    same_ranks(out[:1],'graph-global')
    save('numerical_checks_passed')
    # NCCL finalization waits for callbacks retained by captured CUDA graphs.
    # Release that graph before destroying its communicator, even though its
    # last replay has synchronized. Otherwise every rank can hang in shutdown.
    graph.reset();torch.cuda.synchronize();dist.barrier();dist.destroy_process_group()
    report.update(passed=True,weights_unchanged=True,model_batch_rows=3072,logit_rows=17,mtp_chunk_rows=128,
                  communicator_destroyed=True,finished_at=time.time())
    save('complete')


if __name__=='__main__':main()
