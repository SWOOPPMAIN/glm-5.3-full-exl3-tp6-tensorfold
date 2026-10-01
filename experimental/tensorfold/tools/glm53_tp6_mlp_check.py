#!/usr/bin/env python3
"""Original dense/shared/router/MoE qualification in a bounded isolated GPU process."""
import argparse,faulthandler,gc,hashlib,importlib.util,json,math,os,sys,time
from pathlib import Path
from glm53_tp6_reference import bf16_nearest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','module','manifest','output','binary','native-helper','native-source-pins'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,choices=(0,3,40,77,78),required=True)
    a=p.parse_args();assert not a.output.exists()
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit!='max' and int(limit)<=4*2**30
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    # Original-weight cold reads can trip the serving file-refault guard even
    # above its free-memory floor. This probe now requires a stopped fleet.
    assert available>=64*2**30
    manifest=json.loads(a.manifest.read_text())
    for name,digest in manifest.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert all(str(v) in manifest for v in (a.module,a.binary,a.native_helper,a.native_source_pins,Path(__file__)))
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    def module(name):
        fullname='tensorfold.families.glm_moe_dsa.'+name
        spec=importlib.util.spec_from_file_location(fullname,a.module.with_name(name+'.py'))
        mod=importlib.util.module_from_spec(spec);sys.modules[fullname]=mod;spec.loader.exec_module(mod);return mod
    config=module('config');dense=module('dense');compiled=module('compiled');experts=module('experts');mlp=module('mlp')
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from vllm import _custom_ops as native_ops
    native_bin=Path(importlib.util.find_spec('vllm._C_stable_libtorch').origin)
    assert hashlib.sha256(native_bin.read_bytes()).hexdigest()=='79d638a3c430b3ec51db39571b906b1e85a2875e65eee8dab549a77276f4eddd'
    report=dict(passed=False,phase='initializing',layer=a.layer,cases=[],started_at=time.time(),source_manifest=manifest,
        native_binary_sha256=hashlib.sha256(native_bin.read_bytes()).hexdigest(),
        scope='Original dense/shared sharding, router and local MoE; no TP collective, residual/attention integration, model quality or throughput claim')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        t=a.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.output)
        print(json.dumps({'phase':phase}),flush=True)
    def compare(got,expected,label,*,exact=False,tolerance=.01):
        torch.cuda.synchronize();x,y=got.double(),expected.double()
        same=torch.equal(x,y);finite=bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        denom=float(y.square().mean());rel=float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        passed=finite and (same if exact else rel<tolerance)
        report['cases'].append(dict(case=label,passed=passed,bit_equal=same,exact_required=exact,relative_rms=rel,tolerance=None if exact else tolerance))
        save(label);assert passed,(label,rel)
    def reject(fn,label):
        try:fn()
        except (ValueError,RuntimeError):report['cases'].append(dict(case=label,passed=True,rejected=True));save(label);return
        raise AssertionError(label+' did not reject')
    def dense_reference(x,w):
        if w.width==0:return torch.zeros_like(x)
        gu=(x.double()@w.gate_up.double().T).bfloat16()
        g,u=gu.chunk(2,-1)
        act=(torch.nn.functional.silu(g.double()).bfloat16().double()*u.double()).bfloat16()
        return (act.double()@w.down.double().T).bfloat16()
    def verify_execution(run,x,out,label):
        # One output must not depend on the request's batching arrangement.
        for row in (0,1,7,16,19):compare(run(x[row:row+1]),out[row:row+1],label+f'_serial{row}',exact=True)
        compare(run(x[:17]),out[:17],label+'_verify17',exact=True)
        run(x)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):run(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):graphout=run(x)
        for step in range(3):
            x.copy_(torch.randn_like(x)*(.1+step*.1))
            expected=run(x).clone();graph.replay()
            compare(graphout,expected,label+f'_graph{step}',exact=True)
    faulthandler.dump_traceback_later(235,exit=True)
    try:
        torch.manual_seed(530605+a.layer);reader=RankPieces(a.model,0)
        save('native_silu')
        for width in (512,2048):
            gu=torch.randn((19,2*width),dtype=torch.bfloat16,device='cuda')*4
            # Include zeros, saturation, signed small values and BF16 boundaries.
            gu[0]=torch.linspace(-64,64,2*width,device='cuda').bfloat16()
            got=torch.empty((19,width),dtype=torch.bfloat16,device='cuda');expected=torch.empty_like(got)
            mlp.silu_and_mul(gu,got);torch.ops._C.silu_and_mul(expected,gu)
            compare(got,expected,f'native_silu_width{width}',exact=True)
        del gu,got,expected
        # Scale/add rounding is part of the TP6 numerical contract.
        routed=torch.randn((19,6144),device='cuda')*4;shared=torch.randn_like(routed).bfloat16();out=torch.empty_like(shared)
        expected=(routed.bfloat16()*2.5+shared).bfloat16()
        compare(mlp.combine(routed,shared,out),expected,'native_bf16_scale_shared_order',exact=True)
        del routed,shared,out,expected
        # Exercise all six real dense/shared shard geometries from replicated original weights.
        for rank in range(6):
            save(f'dense_rank{rank}_load')
            w=mlp.DenseWeights(reader,a.layer,rank);layer=mlp.Dense(w);s=mlp.DenseScratch(20,w.width,'cuda')
            x=torch.randn((20,6144),dtype=torch.bfloat16,device='cuda')*.25
            out=layer.forward(x,s).clone()
            compare(out[:3],dense_reference(x[:3],w),f'dense_rank{rank}_fp64',tolerance=.015)
            if rank in (0,5):
                verify_execution(lambda xx:layer.forward(xx,s),x,out,f'dense_rank{rank}')
                # Distinct scratch/streams must not share intermediate storage.
                s2=mlp.DenseScratch(20,w.width,'cuda');x2=torch.randn_like(x)
                e1=layer.forward(x,s).clone();e2=layer.forward(x2,s2).clone()
                streams=[torch.cuda.Stream(),torch.cuda.Stream()]
                for st in streams:st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(streams[0]):o1=layer.forward(x,s)
                with torch.cuda.stream(streams[1]):o2=layer.forward(x2,s2)
                for st in streams:torch.cuda.current_stream().wait_stream(st)
                compare(o1,e1,f'dense_rank{rank}_stream1',exact=True);compare(o2,e2,f'dense_rank{rank}_stream2',exact=True)
                del s2,x2,e1,e2,o1,o2
                arena=mlp.DenseScratch(3072,w.width,'cuda');big=torch.randn((3072,6144),dtype=torch.bfloat16,device='cuda')*.25
                bo=layer.forward(big,arena).clone()
                for row in (0,15,16,127,128,1535,1536,3071):compare(layer.forward(big[row:row+1],s),bo[row:row+1],f'dense_rank{rank}_batch3072_{row}',exact=True)
                del arena,big,bo
            del w,layer,s,x,out
            gc.collect();torch.cuda.empty_cache()
        if a.layer==0:
            report.update(passed=True,finished_at=time.time());save('complete');return
        # Pure routing: native kernel and independent stable score ordering.
        bias=reader.read_tensor(f'model.layers.{a.layer}.mlp.gate.e_score_correction_bias').cuda()
        logits=torch.randn((35,256),device='cuda')*3
        ids=torch.empty((35,8),dtype=torch.int64,device='cuda');prob=torch.empty((35,8),device='cuda')
        ni=torch.empty_like(ids);nw=torch.empty_like(prob);idx=torch.empty_like(ids,dtype=torch.int32)
        for case in ('real_bias','tied','all_underflow','saturated','bias_changes_selection'):
            ll=logits.clone();bb=bias.clone()
            if case=='tied':ll.zero_();bb.zero_()
            if case=='all_underflow':ll.fill_(-1000);bb.zero_()
            if case=='saturated':ll.fill_(1000);bb.zero_()
            if case=='bias_changes_selection':bb.zero_();bb[200:].fill_(2.)
            mlp.top8(ll,bb,ids,prob)
            native_ops.topk_sigmoid(nw,ni,idx,ll,True,bb,1.)
            compare(ids,ni,'router_native_ids_'+case,exact=True)
            compare(prob,nw,'router_native_prob_'+case,tolerance=1e-6)
            scores=ll.double().sigmoid();refids=torch.argsort(scores+bb.double(),dim=-1,descending=True,stable=True)[:,:8]
            compare(ids,refids,'router_fp64_ids_'+case,exact=True)
            vals=torch.gather(scores,1,refids);denom=vals.sum(1,keepdim=True);refprob=vals/torch.where(denom>0,denom,1.)
            compare(prob,refprob,'router_fp64_prob_'+case,tolerance=1e-6)
        del logits,ids,prob,ni,nw,idx,ll,bb,scores,refids,vals,denom,refprob,bias
        reject(lambda:mlp.MoEWeights(reader,a.layer),'require_pinned_binary')
        reject(lambda:compiled.load_experts(a.module),'reject_wrong_binary')
        compiled.load_experts(a.binary)
        save('load_complete_moe')
        w=mlp.MoEWeights(reader,a.layer);layer=mlp.MoE(w);s=mlp.MoEScratch(w,20)
        x=torch.randn((20,6144),dtype=torch.bfloat16,device='cuda')*.25
        out=layer.forward(x,s).clone()
        from unittest.mock import patch
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
        with patch('vllm.model_executor.layers.linear.get_tensor_model_parallel_rank',return_value=0), \
             patch('vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size',return_value=6), \
             patch('vllm.model_executor.parameter.get_tensor_model_parallel_rank',return_value=0), \
             patch('vllm.model_executor.parameter.get_tensor_model_parallel_world_size',return_value=6):
            gate=GateLinear(6144,256,out_dtype=torch.float32,params_dtype=torch.bfloat16,
                            prefix=f'model.layers.{a.layer}.mlp.gate').cuda()
        with torch.no_grad():gate.weight.copy_(w.gate);native_logits,_=gate(x)
        report['native_gate']=dict(logits_are_bf16_rounded=torch.equal(native_logits,native_logits.bfloat16().float()),
            logits_dtype=str(native_logits.dtype),specialized=gate.allow_specialized_router_gemm,
            port_equal=torch.equal(s.logits,native_logits),
            relative_rms=float((s.logits-native_logits).square().mean().sqrt()/native_logits.square().mean().sqrt()))
        save('native_gate_characterization')
        exact_logits=x.double()@w.gate.double().T
        raw_logits=torch.empty_like(s.logits);mlp.gate_projection(x,w.gate,raw_logits)
        # Retain the exact failing input and original router bytes so a repair
        # can be isolated without repeatedly reading a full expert layer.
        from safetensors.torch import save_file
        trace=a.output.with_name('router-case.safetensors')
        save_file({name:value.detach().cpu().contiguous() for name,value in dict(
            hidden=x,gate=w.gate,bias=w.bias,raw=raw_logits,native=native_logits,
            ported=s.logits,fp64=exact_logits,ids=s.ids,probabilities=s.probabilities).items()},str(trace))
        trace.chmod(0o644)
        rounded_reference=bf16_nearest(exact_logits)
        full_projection=torch.empty_like(s.gate_output)
        full_raw=torch.empty_like(s.logits);full_bounds=torch.empty_like(s.logits)
        mlp.gate_projection(x,w.gate,full_projection,scratch=(full_raw,full_bounds))
        bad=(full_projection!=rounded_reference).nonzero().cpu().tolist()
        report['router_trace']=dict(path=str(trace),sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
            mismatches=[dict(row=r,expert=e,raw=float(raw_logits[r,e]),fp64=float(exact_logits[r,e]),
                             ported=float(s.logits[r,e]),native=float(native_logits[r,e])) for r,e in bad[:64]],
            mismatch_count=len(bad))
        save('router_trace_captured')
        compare(raw_logits,exact_logits,'original_router_unrounded_projection_fp64',tolerance=2e-6)
        # Cancellation near zero can differ by tiny BF16 ULPs across GEMM
        # reductions. Bound both aggregate and maximum error; expert membership
        # and order below remain exact requirements, as do batch/graph checks.
        for label,reference in (('original_router_rounded_projection_fp64',rounded_reference.float()),):
            compare(full_projection,reference,label,tolerance=1e-7)
            max_error=float((full_projection-reference).abs().max())
            scale=float(reference.abs().max())
            report['cases'].append(dict(case=label+'_max_error',passed=max_error<=1e-7*scale,
                                        max_absolute_error=max_error,reference_max=scale))
            save(label+'_max_error');assert max_error<=1e-7*scale
        compare(torch.gather(s.logits,1,s.ids),torch.gather(rounded_reference.float(),1,s.ids),
                'selected_logits_fp64',exact=True)
        assert bool(torch.gather(s.gate_candidates,1,s.ids).all())
        del full_projection,full_raw,full_bounds
        # cuBLAS may round on the other side of a BF16 midpoint. Unlike the
        # independent FP64 oracle above, its projection is a characterization
        # bounded to one BF16 step (plus the same FP32 cancellation floor).
        # Exact selected IDs/order and <=1e-6 probability error are still gates.
        compare(s.logits,native_logits,'native_gate_projection',tolerance=1e-4)
        nb=native_logits.bfloat16()
        upper=torch.nextafter(nb,torch.full_like(nb,float('inf'))).float()
        lower=torch.nextafter(nb,torch.full_like(nb,float('-inf'))).float()
        bound=torch.maximum((upper-native_logits).abs(),(native_logits-lower).abs())
        bound=torch.maximum(bound,torch.full_like(bound,1e-7*float(native_logits.abs().max())))
        passed=bool(((s.logits-native_logits).abs()<=bound).all())
        report['cases'].append(dict(case='native_gate_bf16_step_bound',passed=passed,
                                    max_absolute_error=float((s.logits-native_logits).abs().max())))
        save('native_gate_bf16_step_bound');assert passed
        del nb,upper,lower,bound
        ref_ids=torch.argsort(rounded_reference.double().sigmoid().float()+w.bias,dim=-1,descending=True,stable=True)[:,:8]
        compare(s.ids,ref_ids,'original_router_membership_order_fp64',exact=True)
        native_ids=torch.empty_like(s.ids);native_probs=torch.empty_like(s.probabilities)
        native_ops.topk_sigmoid(native_probs,native_ids,torch.empty_like(s.ids,dtype=torch.int32),native_logits,True,w.bias,1.)
        report['native_gate']['route_ids_equal']=torch.equal(native_ids,s.ids)
        report['native_gate']['different_route_slots']=int((native_ids!=s.ids).sum())
        compare(s.ids,native_ids,'native_gate_route_ids',exact=True)
        compare(s.probabilities,native_probs,'native_gate_route_probabilities',tolerance=1e-6)
        save('native_gate_routes')
        del gate,native_logits,exact_logits,ref_ids,native_ids,native_probs,raw_logits
        native_ops.topk_sigmoid(s.probabilities,s.ids,torch.empty_like(s.ids,dtype=torch.int32),s.logits,True,w.bias,1.)
        # Full original BF16 shared path versus independent projection/activation reference.
        shared_ref=dense_reference(x[:3],w.shared.weights)
        compare(s.shared.output[:3],shared_ref,'original_shared_fp64',tolerance=.015)
        direct=(s.routed.output[:20].bfloat16()*2.5+s.shared.output[:20]).bfloat16()
        compare(out,direct,'moe_composition_order',exact=True)
        verify_execution(lambda xx:layer.forward(xx,s),x,out,'moe')
        original=s.weights;s.weights=object()
        reject(lambda:layer.forward(x,s),'reject_other_layer_scratch');s.weights=original
        del original
        # Independent streams retain distinct router, shared and expert workspaces.
        s2=mlp.MoEScratch(w,20);x2=torch.randn_like(x)*.2
        e1=layer.forward(x,s).clone();e2=layer.forward(x2,s2).clone()
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for st in streams:st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):o1=layer.forward(x,s)
        with torch.cuda.stream(streams[1]):o2=layer.forward(x2,s2)
        for st in streams:torch.cuda.current_stream().wait_stream(st)
        compare(o1,e1,'moe_stream1',exact=True);compare(o2,e2,'moe_stream2',exact=True)
        # Keep small actual routed inputs for the deployed-native expert comparison.
        native_inputs=(x[:5].clone(),s.ids[:5].clone(),s.probabilities[:5].clone(),s.shared.output[:5].clone(),e1[:5].clone())
        del s2,x2,e1,e2,o1,o2,direct,out,shared_ref
        arena=mlp.MoEScratch(w,3072);big=torch.randn((3072,6144),dtype=torch.bfloat16,device='cuda')*.25
        bo=layer.forward(big,arena).clone()
        for row in (0,15,16,127,128,1535,1536,3071):
            compare(layer.forward(big[row:row+1],s),bo[row:row+1],f'moe_batch3072_{row}',exact=True)
        del arena,big,bo,s,w,layer,x
        gc.collect();torch.cuda.empty_cache()
        save('native_expert_contract')
        pins=json.loads(a.native_source_pins.read_text())
        for name,expected in pins.items():
            top,relative=name.split('/',1);package=Path(importlib.util.find_spec(top).origin).parent
            assert hashlib.sha256((package/relative).read_bytes()).hexdigest()==expected,name
        os.environ.update(AMOS_EXL3_TP6_PIECES='1',AMOS_TP6_E3_PREFILL='1',VLLM_EXL3_PREFILL_BLOCK_M='32')
        spec=importlib.util.spec_from_file_location('native_layer_helper',a.native_helper)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        method,native_layer=helper.load_layer(a.model,0,a.layer,batch_capacity=20)
        xx,ii,pp,ss,actual=native_inputs
        nr=method._apply_mixed_rank_sliced(native_layer,xx,pp,ii)
        compare(actual,(nr*2.5+ss).bfloat16(),'native_routed_scale_shared',tolerance=.01)
        assert not torch.distributed.is_initialized()
        report.update(passed=True,distributed_initialized=False,finished_at=time.time());save('complete')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}',finished_at=time.time());save('failed');raise
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
