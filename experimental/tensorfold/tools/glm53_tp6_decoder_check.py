#!/usr/bin/env python3
"""Six real ranks: complete original dense, routed and MTP decoder layers."""
import os
import argparse,faulthandler,hashlib,json,math,os,time
from datetime import timedelta
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',type=int,required=True);p.add_argument('--port',type=int,required=True)
    p.add_argument('--layer',type=int,choices=(0,3,6,40,77,78),default=0)
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
    from tensorfold.families.glm_moe_dsa.decoder import TP6Reduction,DecoderWeights,DecoderScratch,Decoder
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from tensorfold.families.glm_moe_dsa.mla import SelectionState,MLA,MLAWeights,MLAScratch
    from tensorfold.families.glm_moe_dsa.compiled import load_experts
    from tensorfold.families.glm_moe_dsa.norms import hidden_rms
    from tensorfold.families.glm_moe_dsa.cache import LayerCache
    from tensorfold.families.glm_moe_dsa.attention import rope_table
    from tensorfold.families.glm_moe_dsa.doorbell import RequestDoorbell
    from glm53_tp6_reference import bf16_nearest
    report=dict(passed=False,rank=a.rank,layer=a.layer,phase='initializing',cases=[],started_at=time.time(),source_manifest=manifest,
                scope='Actual six-node collective and one complete original decoder layer. Shared selections use the original source MLA with a common input fixture, not an end-to-end preceding decoder chain. No complete model, serving capacity or throughput qualification.')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        cg=Path('/sys/fs/cgroup');stat=dict(s.split() for s in (cg/'memory.stat').read_text().splitlines())
        report['memory']=dict(current_gib=int((cg/'memory.current').read_text())/2**30,
                              peak_gib=int((cg/'memory.peak').read_text())/2**30,
                              anon_gib=int(stat['anon'])/2**30,shmem_gib=int(stat['shmem'])/2**30)
        t=a.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.output)
        print(json.dumps({'rank':a.rank,'phase':phase}),flush=True)
    def compare(got,ref,label,*,exact=False,tolerance=.02):
        torch.cuda.synchronize()
        same=torch.equal(got,ref);finite=bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
        # Exact3072-row checks need no full-sized FP64 copies or differences.
        # Numerical comparisons below are bounded original-weight sample rows.
        if exact:rms=0. if same else float('inf')
        else:
            x,y=got.double(),ref.double()
            denominator=float(y.square().mean().sqrt());rms=float((x-y).square().mean().sqrt())/(denominator or 1.)
        passed=finite and (same if exact else rms<=tolerance)
        report['cases'].append(dict(case=label,passed=passed,exact_required=exact,bit_equal=same,relative_rms=rms,tolerance=None if exact else tolerance))
        save(label);assert passed,(label,rms)
    def sum_reference(local):
        # Separate oracle path: transport original BF16 values, then CPU Torch
        # adds each rank in FP32. It does not call the production sum kernel.
        result=torch.empty_like(local)
        for start in range(0,len(local),128):
            part=local[start:start+128];n=len(part)
            got=torch.empty((6*n,6144),dtype=torch.bfloat16,device='cuda')
            dist.all_gather_into_tensor(got,part)
            values=got.cpu().reshape(6,n,6144).float();out=values[0].clone()
            for i in range(1,6):out=out+values[i]
            result[start:start+n].copy_(out.bfloat16())
        return result
    def rms_reference(x,w):
        raw=x.double();return bf16_nearest(raw*torch.rsqrt(raw.square().mean(-1,keepdim=True)+1e-5)*w.double())
    def attention_reference(hidden,pos,cache,table,w,tokens,count):
        qkv=bf16_nearest(w.qkv_a.double()@hidden.double())
        ql=rms_reference(qkv[:2048],w.q_norm)
        q=bf16_nearest(w.q_b.double()@ql.double()).reshape(11,256)
        v=q[:,192:].double().reshape(11,32,2);cs=table[int(pos)].double();c,s=cs[:32],cs[32:]
        q[:,192:]=bf16_nearest(torch.stack((v[...,0]*c-v[...,1]*s,v[...,1]*c+v[...,0]*s),-1).reshape(11,64))
        real=w.real_heads
        qa=bf16_nearest(torch.einsum('hd,hdl->hl',q[:real,:192].double(),w.wk[:real].double())).double()
        ids=tokens[:int(count)].long();ids=ids[(ids>=0)&(ids<=pos)]
        lc=(cache.latent[ids].float().reshape(-1,4,128)*cache.scales[ids,:,None]).reshape(-1,512).double()
        scores=(qa@lc.T+q[:real,192:].double()@cache.rope[ids].double().T)/16.
        latent=bf16_nearest(scores.softmax(-1)@lc).double()
        values=bf16_nearest(torch.einsum('hl,hvl->hv',latent,w.wv[:real].double()))
        return bf16_nearest(w.o.double()@values.flatten().double())
    def dense_reference(hidden,w):
        if not w.width:return torch.zeros((6144,),dtype=torch.bfloat16,device='cuda')
        gu=bf16_nearest(w.gate_up.double()@hidden.double());gate,up=gu.chunk(2)
        act=bf16_nearest(bf16_nearest(torch.nn.functional.silu(gate.double())).double()*up.double())
        return bf16_nearest(w.down.double()@act.double())
    def ffn_reference(hidden,w):
        if a.layer<3:return dense_reference(hidden,w)
        import numpy as np
        from tensorfold.cuda.exl3 import format as reference
        # Router follows the independently nearest-BF16 projection, then native
        # FP32 sigmoid+bias scores. Selection and unscaled weights are global;
        # this rank evaluates only the original fragments that it owns.
        logits=bf16_nearest(w.gate.double()@hidden.double()).double().sigmoid().float()
        ids=torch.argsort(logits+w.bias,descending=True,stable=True)[:8]
        probs=logits[ids];total=torch.zeros((),device='cuda')
        for prob in probs:total=total+prob
        probs=probs/total
        x=hidden.double().cpu().numpy().reshape(1,-1)
        parts={part.expert:part for part in reader.fragments(a.layer)}
        accumulated=np.zeros((1,6144),dtype=np.float64)
        for expert,prob in zip(ids.cpu().tolist(),probs.cpu().tolist()):
            if expert not in parts:continue
            part=parts[expert]
            def projection(v,name):
                prefix=f'{part.prefix}.{name}.rank{part.original_rank}'
                t,suh,svh=[reader.read_tensor(prefix+'.'+field).numpy() for field in ('trellis','suh','svh')]
                decoded=reference.unpack(t,part.bits,'mcg').astype(np.float64)
                rotated=reference.rotate(v*suh.astype(np.float64),-1).astype(np.float16).astype(np.float64)
                return reference.rotate(rotated@decoded,-1)*svh.astype(np.float64)
            gate=projection(x,'gate_proj');up=projection(x,'up_proj')
            act=gate/(1.+np.exp(-gate))*up
            accumulated+=projection(act,'down_proj')*prob
        routed=bf16_nearest(torch.from_numpy(accumulated[0]).to('cuda'))
        scaled=bf16_nearest(routed.double()*2.5)
        return bf16_nearest(scaled.double()+dense_reference(hidden,w.shared.weights).double())
    faulthandler.dump_traceback_later(570,exit=True)
    try:
        # TCP control and NCCL data use one explicitly owned group on stopped GPUs.
        store=dist.TCPStore(os.environ['MASTER_ADDR'],a.port,6,a.rank==0,timeout=timedelta(seconds=90))
        dist.init_process_group('nccl',store=store,rank=a.rank,world_size=6,timeout=timedelta(seconds=90),device_id=torch.device('cuda:0'))
        group=dist.group.WORLD;reduction=TP6Reduction(group,a.rank).prepare(3072,'cuda:0')
        bell=RequestDoorbell(store,a.rank,generation='decoder_check_'+str(a.port)+'_530608')
        for step in range(3):
            if a.rank==0:assert bell.publish()==step+1
            else:assert bell.wait(timedelta(seconds=10))==step+1
            dist.barrier()
        report['cases'].append(dict(case='six_host_doorbell_three_sequences',passed=True));save('doorbell')
        torch.manual_seed(608+a.rank)
        x=torch.randn((3072,6144),dtype=torch.bfloat16,device='cuda')*.25;out=torch.empty_like(x)
        # Cancellation deliberately exercises the specified FP32 rank order.
        x[0,:6]=torch.tensor([2**24,1,-2**24,0,0,0][a.rank],dtype=torch.bfloat16,device='cuda')
        reduction.sum_into(x,out);reference=sum_reference(x)
        compare(out,reference,'collective3072_cpu_rank_order',exact=True)
        for row in (0,15,16,127,128,255,256,1535,1536,3071):
            one=torch.empty_like(x[row:row+1]);reduction.sum_into(x[row:row+1],one)
            compare(one,out[row:row+1],f'collective_serial{row}',exact=True)
        for n in (17,256):
            small=torch.empty_like(x[:n]);reduction.sum_into(x[:n],small)
            compare(small,out[:n],f'collective_rows{n}',exact=True)
        # Captured collectives must replay with new values on every rank.
        x=x[:17].clone();out=torch.empty_like(x)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):reduction.sum_into(x,out)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):reduction.sum_into(x,out)
        for step in range(3):
            x.copy_(torch.randn_like(x));expected=sum_reference(x);graph.replay()
            compare(out,expected,f'collective_graph{step}',exact=True)
        del graph,x,out,reference,small,one
        torch.cuda.empty_cache()
        save('load_original_decoder')
        if a.layer>=3:load_experts()
        reader=RankPieces(a.model,a.rank);w=DecoderWeights(reader,a.layer);layer=Decoder(w,reduction)
        source=reader.config.indexer_source(a.layer);report['indexer_source']=source
        table=rope_table(4096,'cuda');cache=LayerCache(a.layer,4096,'cuda',indexer=source==a.layer)
        def clear(c):
            c.latent.zero_();c.scales.fill_(1);c.rope.zero_()
            if c.index_keys is not None:c.index_keys.zero_();c.index_scales.fill_(1)
        clear(cache)
        if source!=a.layer:
            producer=MLA(MLAWeights(reader,source,a.rank));producer_cache=LayerCache(source,4096,'cuda',indexer=True)
            clear(producer_cache);producer_scratch=MLAScratch(128,4096,'cuda',real_heads=w.attention.weights.real_heads)
            producer_selection=SelectionState(128,'cuda')
            producer_norm=reader.read_tensor(f'model.layers.{source}.input_layernorm.weight').to('cuda')
            producer_hidden=torch.empty((128,6144),dtype=torch.bfloat16,device='cuda');producer_skip=torch.empty_like(producer_hidden)
        def forward(xx,rr,pp,bb,ww,arena,selected,*,scope):
            if source!=a.layer:
                for start in range(0,len(xx),128):
                    stop=min(start+128,len(xx));n=stop-start
                    hidden_rms(xx[start:stop],producer_norm,producer_hidden[:n],producer_skip[:n])
                    producer.forward(producer_hidden[:n],pp[start:stop],bb[start:stop],ww[start:stop],producer_cache,
                                     table,producer_scratch,producer_selection,scope=scope)
                    selected.tokens[start:stop].copy_(producer_selection.tokens[:n])
                    selected.counts[start:stop].copy_(producer_selection.counts[:n])
                selected.publish(source,scope,pp,bb,selected.tokens[:len(xx)],selected.counts[:len(xx)])
            return layer.forward(xx,rr,pp,bb,ww,cache,table,arena,selected,scope=scope)
        n=20;scratch=DecoderScratch(w,n,4096);selection=SelectionState(n,'cuda')
        torch.manual_seed(530608);x=torch.randn((n,6144),dtype=torch.bfloat16,device='cuda')*.25
        residual=torch.randn_like(x)*.25;pos=torch.arange(n,device='cuda');bases=torch.zeros_like(pos);slots=pos.clone();scope=object()
        for add in (False,True):
            rr=residual if add else None
            y,skip=forward(x,rr,pos,bases,slots,scratch,selection,scope=scope);y=y.clone();skip=skip.clone()
            # Every rank must produce the same replicated hidden/skip rows.
            ref=y.clone();dist.broadcast(ref,0);compare(y,ref,f'layer_rank_equal_add{add}',exact=True)
            ref=skip.clone();dist.broadcast(ref,0);compare(skip,ref,f'layer_skip_rank_equal_add{add}',exact=True)
            if a.layer>=3:
                ids=scratch.ffn.ids[:n].clone();dist.broadcast(ids,0)
                compare(scratch.ffn.ids[:n],ids,f'router_all_rank_ids_add{add}',exact=True)
                actual=scratch.normalized[:n]
                exact_logits=bf16_nearest(actual.double()@w.ffn.weights.gate.double().T)
                exact_ids=torch.argsort(exact_logits.double().sigmoid().float()+w.ffn.weights.bias,dim=-1,descending=True,stable=True)[:,:8]
                compare(scratch.ffn.ids[:n],exact_ids,f'router_original_fp64_ids_add{add}',exact=True)
                del ids,actual,exact_logits,exact_ids
            for row in ((0,7,19) if a.layer<3 else (0,19)):
                z=x[row] if rr is None else bf16_nearest(x[row].double()+rr[row].double())
                normalized=rms_reference(z,w.input_norm)
                local=attention_reference(normalized,pos[row],cache,table,w.attention.weights,selection.tokens[row],selection.counts[row]).reshape(1,6144)
                reduced=sum_reference(local)[0]
                post_skip=bf16_nearest(reduced.double()+z.double());post=rms_reference(post_skip,w.post_norm)
                local=ffn_reference(post,w.ffn.weights).reshape(1,6144)
                expected=sum_reference(local)[0]
                compare(skip[row],post_skip,f'layer_fp64_skip_add{add}_row{row}')
                compare(y[row],expected,f'layer_fp64_output_add{add}_row{row}')
            for row in (0,7,19):
                yy,ss=forward(x[row:row+1],None if rr is None else rr[row:row+1],pos[row:row+1],bases[row:row+1],slots[row:row+1],scratch,selection,scope=object())
                compare(yy,y[row:row+1],f'layer_serial_add{add}_row{row}',exact=True)
                compare(ss,skip[row:row+1],f'layer_serial_skip_add{add}_row{row}',exact=True)
            yy,ss=forward(x[:17],None if rr is None else rr[:17],pos[:17],bases[:17],slots[:17],scratch,selection,scope=object())
            compare(yy,y[:17],f'layer_verify17_add{add}',exact=True);compare(ss,skip[:17],f'layer_verify_skip17_add{add}',exact=True)
        # A complete layer including cache writes and both collectives in a graph.
        def run():return forward(x,residual,pos,bases,slots,scratch,selection,scope=scope)
        run();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):run()
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize();dist.barrier()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):gy,gs=run()
        for step in range(3):
            x.copy_(torch.randn_like(x)*(.25+step*.1));residual.copy_(torch.randn_like(residual)*.25)
            ey,es=run();ey=ey.clone();es=es.clone();graph.replay()
            compare(gy,ey,f'layer_graph{step}',exact=True);compare(gs,es,f'layer_graph_skip{step}',exact=True)
        del graph,gy,gs,scratch,selection,ey,es,y,skip,x,residual
        torch.cuda.empty_cache()
        # Actual causal3072-token prefill, with unique cache slots.
        torch.manual_seed(530609);n=3072
        x=torch.randn((n,6144),dtype=torch.bfloat16,device='cuda')*.25
        pos=torch.arange(n,device='cuda');bases=torch.zeros_like(pos);slots=pos.clone()
        scratch=DecoderScratch(w,n,4096);selection=SelectionState(n,'cuda')
        yy,ss=forward(x,None,pos,bases,slots,scratch,selection,scope=object());y=yy.clone();skip=ss.clone()
        for row in (0,15,16,127,128,255,256,1535,1536,3071):
            yy,ss=forward(x[row:row+1],None,pos[row:row+1],bases[row:row+1],slots[row:row+1],scratch,selection,scope=object())
            compare(yy,y[row:row+1],f'layer3072_serial{row}',exact=True)
            compare(ss,skip[row:row+1],f'layer3072_serial_skip{row}',exact=True)
        report.update(passed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
