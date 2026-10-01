#!/usr/bin/env python3
"""Isolated original-weight full local MLA and compact-cache GPU qualification."""
import argparse,faulthandler,hashlib,importlib.util,json,math,sys,time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','module','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,choices=(0,3,78),required=True)
    a=p.parse_args();assert not a.output.exists()
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit!='max' and int(limit)<=4*2**30
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=12*2**30
    manifest=json.loads(a.manifest.read_text())
    for name,digest in manifest.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    assert torch.cuda.get_device_capability()==(12,1)
    assert str(a.module) in manifest and str(Path(__file__)) in manifest
    def module(name):
        path=a.module.with_name(name+'.py')
        fullname='tensorfold.families.glm_moe_dsa.'+name
        spec=importlib.util.spec_from_file_location(fullname,path);mod=importlib.util.module_from_spec(spec)
        sys.modules[fullname]=mod;spec.loader.exec_module(mod);return mod
    config=module('config');dense=module('dense');norms=module('norms');cachemod=module('cache')
    attn=module('attention');indexer=module('indexer');mla=module('mla')
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    report=dict(passed=False,phase='initializing',layer=a.layer,cases=[],started_at=time.time(),source_manifest=manifest,
        scope='Complete TP6 local attention contribution with original BF16 weights and compact FP8 cache; no collective, residual/MLP, complete model, cache admission/commit or serving qualification')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        t=a.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.output)
        print(json.dumps({'phase':phase}),flush=True)
    def compare(got,expected,label,*,exact=False,tolerance=.008):
        torch.cuda.synchronize();x,y=got.float(),expected.float()
        same=torch.equal(x,y);finite=bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        denom=float(y.square().mean());rel=float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        passed=finite and (same if exact else rel<tolerance)
        report['cases'].append(dict(case=label,passed=passed,bit_equal=same,exact_required=exact,relative_rms=rel,tolerance=None if exact else tolerance))
        save(label);assert passed,(label,rel)
    def expect_reject(fn,label):
        try:fn()
        except ValueError:report['cases'].append(dict(case=label,passed=True,rejected=True));save(label);return
        raise AssertionError(label+' accepted invalid scope/cache')
    def seed_cache(c):
        for start in range(0,c.capacity,8192):
            end=min(c.capacity,start+8192)
            values=torch.randn((end-start,512),dtype=torch.bfloat16,device='cuda')*.5
            rope=torch.randn((end-start,64),dtype=torch.bfloat16,device='cuda')*.25
            cachemod.write_mla(values,rope,torch.arange(start,end,dtype=torch.int64,device='cuda'),c)
            if c.index_keys is not None:c.index_keys[start:end].copy_(torch.randn((end-start,128),dtype=torch.bfloat16,device='cuda'))
        if c.index_scales is not None:c.index_scales.fill_(2**-5)
    def rope_ref(x,pos,table,offset):
        result=x.clone();v=x[...,offset:offset+64].double().reshape(len(pos),-1,32,2)
        cs=table[pos].double();c,s=cs[:,None,:32],cs[:,None,32:]
        y=torch.stack((v[...,0]*c-v[...,1]*s,v[...,1]*c+v[...,0]*s),-1)
        result[...,offset:offset+64]=y.reshape(len(pos),*x.shape[1:-1],64).bfloat16()
        return result
    def reference_row(hidden,pos,base,c,table,w,tokens,count):
        # Independent original-weight projections, RMS normalization and RoPE.
        qkv=(w.qkv_a.double()@hidden.double()).bfloat16()
        ql=(qkv[:2048].double()*torch.rsqrt(qkv[:2048].double().square().mean()+1e-5)*w.q_norm.double()).bfloat16()
        q=(w.q_b.double()@ql.double()).reshape(1,11,256).bfloat16()
        q=rope_ref(q,pos.reshape(1),table,192)[0]
        real=w.real_heads
        qa=torch.einsum('hd,hdl->hl',q[:real,:192].double(),w.wk[:real].double()).bfloat16().double()
        ids=tokens[:int(count)].long();ids=ids[(ids>=0)&(ids<=pos)];slots=ids+base
        lc=(c.latent.view(torch.uint8)[slots].view(torch.float8_e4m3fn).float().reshape(-1,4,128)*c.scales[slots,:,None]).reshape(-1,512).double()
        kr=c.rope[slots].double()
        scores=(qa@lc.T+q[:real,192:].double()@kr.T)/16
        latent=(scores.softmax(-1)@lc).bfloat16().double()
        values=torch.einsum('hl,hvl->hv',latent,w.wv[:real].double()).bfloat16()
        return (w.o.double()@values.flatten().double()).bfloat16()
    faulthandler.dump_traceback_later(235,exit=True)
    try:
        torch.manual_seed(530604+a.layer);reader=RankPieces(a.model,0)
        table=attn.rope_table(1048576,'cuda')
        save('native_compact_cache')
        c=cachemod.LayerCache(a.layer,64,'cuda',indexer=False)
        c.latent.zero_();c.scales.zero_();c.rope.zero_()
        latent=torch.randn((19,512),dtype=torch.bfloat16,device='cuda')
        latent[0].zero_();latent[1].mul_(1e-10)
        # Per-block threshold/scale differences, including BF16 extrema near448.
        for block in range(4):latent[2,block*128:(block+1)*128].fill_(448*(2**(block-7)))
        kr=torch.randn((19,64),dtype=torch.bfloat16,device='cuda')
        slots=torch.tensor([0,1,2,63,62,17,18,19,20,21,22,23,24,25,26,27,28,29,-1],dtype=torch.int64,device='cuda')
        cachemod.write_mla(latent,kr,slots,c)
        from vllm import _custom_ops as native_ops
        native_bin=Path(importlib.util.find_spec('vllm._C_stable_libtorch').origin)
        report['native_binary']=dict(path=str(native_bin),sha256=hashlib.sha256(native_bin.read_bytes()).hexdigest())
        native=torch.zeros((1,64,656),dtype=torch.uint8,device='cuda')
        native_ops.concat_and_cache_mla(latent,kr,native,slots,'fp8_ds_mla',torch.ones(1,device='cuda'))
        records=native.reshape(64,656)
        report['cache_diagnostic']=dict(port_scales=c.scales[[0,1,2,63]].cpu().tolist(),native_scales=records[:,512:528].contiguous().view(torch.float32)[[0,1,2,63]].cpu().tolist(),native_codes=records[:,:512][63,:16].cpu().tolist(),port_codes=c.latent.view(torch.uint8)[63,:16].cpu().tolist())
        compare(c.latent,records[:,:512].contiguous().view(torch.float8_e4m3fn),'native_compact_latent',exact=True)
        compare(c.scales,records[:,512:528].contiguous().view(torch.float32),'native_compact_scales',exact=True)
        compare(c.rope,records[:,528:].contiguous().view(torch.bfloat16),'native_compact_rope',exact=True)
        assert c.nbytes()==64*656
        del c,native,latent,kr,slots,records
        # Original source2 indexer for the real shared layer3, with its own cache.
        source=None
        if a.layer==3:
            source=mla.MLAWeights(reader,2,0)
        for rank,real in ((0,11),(5,9)):
            save(f'prepare_rank{rank}')
            w=mla.MLAWeights(reader,a.layer,rank);layer=mla.MLA(w)
            n=20;cap=8192
            x=torch.randn((n,6144),dtype=torch.bfloat16,device='cuda')*.25
            pos=torch.tensor([0,31,511,512,2047,2048,3000,4000,4094,4095,0,32,510,512,2046,2048,3000,4000,4094,4095],dtype=torch.int64,device='cuda')
            base=torch.cat((torch.zeros(10),torch.full((10,),4096))).long().cuda();slots=pos+base
            c=cachemod.LayerCache(a.layer,cap,'cuda',indexer=source is None);seed_cache(c)
            s=mla.MLAScratch(n,4096,'cuda',real_heads=real);selection=mla.SelectionState(n,'cuda');scope=object()
            if source is not None:
                sc=cachemod.LayerCache(2,cap,'cuda',indexer=True);seed_cache(sc)
                ss=mla.MLAScratch(n,4096,'cuda',real_heads=11)
                sm=mla.MLA(source)
                sm.forward(x,pos,base,slots,sc,table,ss,selection,scope=scope)
                source_tokens=selection.tokens.clone();source_latent=sc.latent.float().clone()
            out=layer.forward(x,pos,base,slots,c,table,s,selection,scope=scope).clone()
            from vllm.amos_stable_attention_norm import _rms as native_rms
            compare(s.q_lora,native_rms(s.qkv[:,:2048],w.q_norm,1e-5),f'rank{rank}_native_q_rms',exact=True)
            compare(s.kv_lora,native_rms(s.qkv[:,2048:2560],w.kv_norm,1e-5),f'rank{rank}_native_kv_rms',exact=True)
            if source is not None:
                compare(selection.tokens,source_tokens,f'rank{rank}_shared_selection_unchanged',exact=True)
                compare(sc.latent,source_latent,f'rank{rank}_shared_own_cache',exact=True)
                expect_reject(lambda:layer.forward(x,pos,base,slots,sc,table,s,selection,scope=scope),f'rank{rank}_reject_other_layer_cache')
                expect_reject(lambda:layer.forward(x,pos,base,slots,c,table,s,selection,scope=object()),f'rank{rank}_reject_stale_scope')
                saved_source=selection.source;selection.source=78
                expect_reject(lambda:layer.forward(x,pos,base,slots,c,table,s,selection,scope=scope),f'rank{rank}_reject_mtp_selection')
                selection.source=saved_source
                expect_reject(lambda:layer.forward(x,pos.clone(),base,slots,c,table,s,selection,scope=scope),f'rank{rank}_reject_other_row_layout')
            ref_tokens,ref_counts=selection.tokens.clone(),selection.counts.clone()
            for row in (0,2,5,8,10,15,19):
                ref=reference_row(x[row],pos[row],base[row],c,table,w,ref_tokens[row],ref_counts[row])
                compare(out[row],ref,f'rank{rank}_full_fp64_row{row}',tolerance=.015)
            # The compact reader equals pre-dequantized BF16 reference cache exactly.
            dequant=(c.latent.float().reshape(cap,4,128)*c.scales[:,:,None]).reshape(cap,512).bfloat16()
            raw_out=torch.empty_like(s.attended)
            attn.attend(s.q_absorbed,s.q_rope,dequant,c.rope,selection.tokens,selection.counts,pos,base,raw_out,s.attention)
            # Fused FP8 dequantization and materialized BF16 paths may lower MMA differently.
            # Both must match an independent FP64 latent reference, with strict row invariance below.
            mismatch=(s.attended!=raw_out).nonzero()
            cr,br=s.attended.float(),raw_out.float()
            cross_rms=float((cr-br).square().mean().sqrt()/br.square().mean().sqrt())
            report.setdefault('reader_comparison',[]).append(dict(rank=rank,different_values=len(mismatch),total_values=s.attended.numel(),relative_rms=cross_rms,
                classification='Cross-format kernel characterization only; independent per-row FP64 gates below apply to both paths. Exact serial/batch/graph gates remain mandatory within the compact path.'))
            for row in range(n):
                ids=selection.tokens[row,:int(selection.counts[row])].long()+base[row]
                keys=dequant[ids].double();rope=c.rope[ids].double()
                score=(s.q_absorbed[row,:real].double()@keys.T+s.q_rope[row,:real].double()@rope.T)/16
                ref=score.softmax(-1)@keys
                compare(s.attended[row,:real],ref,f'rank{rank}_compact_latent_fp64_row{row}')
                compare(raw_out[row,:real],ref,f'rank{rank}_materialized_latent_fp64_row{row}')
            if real==9:
                assert not torch.count_nonzero(s.q_absorbed[:,9:]) and not torch.count_nonzero(s.attended[:,9:])
            # Serial and verification windows receive the exact same selection for shared layers.
            for row in (0,5,10,15,19):
                one=mla.SelectionState(1,'cuda');one_scope=object()
                pp,bb=pos[row:row+1],base[row:row+1]
                if source is not None:one.publish(2,one_scope,pp,bb,ref_tokens[row:row+1],ref_counts[row:row+1])
                y=layer.forward(x[row:row+1],pp,bb,slots[row:row+1],c,table,s,one,scope=one_scope)
                compare(y,out[row:row+1],f'rank{rank}_serial_row{row}',exact=True)
            short=mla.SelectionState(17,'cuda');short_scope=object();pp,bb=pos[:17],base[:17]
            if source is not None:short.publish(2,short_scope,pp,bb,ref_tokens[:17],ref_counts[:17])
            y=layer.forward(x[:17],pp,bb,slots[:17],c,table,s,short,scope=short_scope)
            compare(y,out[:17],f'rank{rank}_verify17',exact=True)
            # Graph includes source2 producer for shared layer3.
            def forward():
                if source is not None:sm.forward(x,pos,base,slots,sc,table,ss,selection,scope=scope)
                return layer.forward(x,pos,base,slots,c,table,s,selection,scope=scope)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):forward()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):forward()
            for change in range(3):
                x.copy_(x.roll(1,0)*.875)
                # Preserve per-request extent and unique slots as positions change.
                pos.copy_(pos.roll(1,0));base.copy_(base.roll(1,0));slots.copy_(pos+base)
                graph.replay();got=s.output.clone();got_cache=c.latent.float().clone();got_scales=c.scales.clone()
                expected=forward()
                compare(got,expected,f'rank{rank}_graph{change}',exact=True)
                compare(c.latent,got_cache,f'rank{rank}_graph{change}_cache',exact=True)
                compare(c.scales,got_scales,f'rank{rank}_graph{change}_scales',exact=True)
            del graph
            # Simultaneous requests: separate complete caches, scratch and selection scopes.
            second_c=cachemod.LayerCache(a.layer,cap,'cuda',indexer=source is None)
            def copy_cache(dst,src):
                for attr in ('latent','scales','rope','index_keys','index_scales'):
                    if getattr(src,attr) is not None:getattr(dst,attr).copy_(getattr(src,attr))
            copy_cache(second_c,c)
            second_s=mla.MLAScratch(n,4096,'cuda',real_heads=real)
            second_selection=mla.SelectionState(n,'cuda');second_scope=object();second_x=(-x*.75).contiguous()
            if source is not None:
                second_sc=cachemod.LayerCache(2,cap,'cuda',indexer=True);copy_cache(second_sc,sc)
                second_ss=mla.MLAScratch(n,4096,'cuda',real_heads=11)
            def second_forward():
                if source is not None:sm.forward(second_x,pos,base,slots,second_sc,table,second_ss,second_selection,scope=second_scope)
                return layer.forward(second_x,pos,base,slots,second_c,table,second_s,second_selection,scope=second_scope)
            expected_a=forward().clone();expected_b=second_forward().clone()
            expected_ca=c.latent.float().clone();expected_cb=second_c.latent.float().clone()
            stream_b=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream());stream_b.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):forward()
            with torch.cuda.stream(stream_b):second_forward()
            torch.cuda.synchronize()
            compare(s.output,expected_a,f'rank{rank}_full_stream_a',exact=True)
            compare(second_s.output,expected_b,f'rank{rank}_full_stream_b',exact=True)
            compare(c.latent,expected_ca,f'rank{rank}_full_stream_a_cache',exact=True)
            compare(second_c.latent,expected_cb,f'rank{rank}_full_stream_b_cache',exact=True)
            del second_c,second_s,second_selection,second_scope,second_x,expected_a,expected_b,expected_ca,expected_cb
            if source is not None:del second_sc,second_ss
            # Full model batch3072 remains supported; no writes to repeated slots.
            copies=(3072+n-1)//n
            bx=x.repeat(copies,1)[:3072].contiguous();bp=pos.repeat(copies)[:3072].contiguous();bb=base.repeat(copies)[:3072].contiguous()
            bslots=torch.full_like(bp,-1);bs=mla.MLAScratch(3072,4096,'cuda',real_heads=real)
            bselection=mla.SelectionState(3072,'cuda');bscope=object()
            if source is not None:
                bselection.publish(2,bscope,bp,bb,selection.tokens.repeat(copies,1)[:3072],selection.counts.repeat(copies)[:3072])
            by=layer.forward(bx,bp,bb,bslots,c,table,bs,bselection,scope=bscope)
            expected=s.output.clone()
            for row in (0,15,16,127,128,1535,1536,3071):compare(by[row:row+1],expected[row%n:row%n+1],f'rank{rank}_batch3072_row{row}',exact=True)
            del bs,bselection,bx,bp,bb,bslots,by
            # Long context on same original layer, with independent request extent.
            save(f'rank{rank}_long_context')
            lc=cachemod.LayerCache(a.layer,368212,'cuda',indexer=source is None);seed_cache(lc)
            lx=x[:3].contiguous();lp=torch.tensor([359999,360000,8191],dtype=torch.int64,device='cuda')
            lb=torch.tensor([0,0,360020],dtype=torch.int64,device='cuda');lslots=lp+lb
            ls=mla.MLAScratch(3,360001,'cuda',real_heads=real);lselection=mla.SelectionState(3,'cuda');lscope=object()
            if source is not None:
                # Shared layers consume canonical tokens from their owner; source2's full producer is covered above.
                ids=torch.stack([torch.linspace(0,int(p),2048,device='cuda').round().int() for p in lp])
                lselection.publish(2,lscope,lp,lb,ids,torch.full((3,),2048,dtype=torch.int32,device='cuda'))
            ly=layer.forward(lx,lp,lb,lslots,lc,table,ls,lselection,scope=lscope).clone()
            for row in range(3):
                ref=reference_row(lx[row],lp[row],lb[row],lc,table,w,lselection.tokens[row],lselection.counts[row])
                compare(ly[row],ref,f'rank{rank}_long_fp64_row{row}',tolerance=.015)
            assert lc.nbytes()==lc.capacity*(788 if source is None else 656)
            report.setdefault('cache_bytes_per_token',{})[str(a.layer)]=788 if source is None else 656
            del w,layer,s,c,selection,scope,ls,lc,lselection,ly,dequant,raw_out,out,ref_tokens,ref_counts,expected
            if source is not None:del sc,ss,sm,source_tokens,source_latent
        assert not torch.distributed.is_initialized()
        report.update(passed=True,finished_at=time.time(),distributed_initialized=False);save('complete')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}');save('failed');raise
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
