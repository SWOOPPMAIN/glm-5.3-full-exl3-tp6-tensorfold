#!/usr/bin/env python3
"""Reproduce and qualify a router repair from exact saved BF16 operands."""
import argparse,faulthandler,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
from glm53_tp6_reference import bf16_nearest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','module','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,required=True)
    a=p.parse_args();assert not a.output.exists()
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit!='max' and int(limit)<=4*2**30
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=64*2**30
    manifest=json.loads(a.manifest.read_text())
    for name,digest in manifest.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert all(str(v) in manifest for v in (a.case,a.module,Path(__file__)))
    import torch
    from vllm import _custom_ops as native_ops
    from safetensors.torch import load_file
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    def module(name):
        fullname='tensorfold.families.glm_moe_dsa.'+name
        spec=importlib.util.spec_from_file_location(fullname,a.module.with_name(name+'.py'))
        mod=importlib.util.module_from_spec(spec);sys.modules[fullname]=mod;spec.loader.exec_module(mod);return mod
    for name in ('config','dense','compiled','experts'):module(name)
    mlp=module('mlp')
    report=dict(passed=False,phase='initializing',layer=a.layer,cases=[],started_at=time.time(),source_manifest=manifest,
        scope='Saved exact original router operands and BF16 projection rounding only; no whole model or performance claim')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output)
        print(json.dumps({'phase':phase}),flush=True)
    def compare(got,expected,label,*,exact=False,tolerance=1e-7,max_error=False):
        torch.cuda.synchronize();x,y=got.double(),expected.double()
        same=torch.equal(x,y);finite=bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        denom=float(y.square().mean());rel=float((x-y).square().mean().sqrt())/math.sqrt(denom) if denom else (0. if same else float('inf'))
        absmax=float((x-y).abs().max());scale=float(y.abs().max())
        passed=finite and (same if exact else rel<tolerance) and (not max_error or absmax<=tolerance*scale)
        report['cases'].append(dict(case=label,passed=passed,bit_equal=same,relative_rms=rel,max_absolute_error=absmax,
                                   exact_required=exact,tolerance=None if exact else tolerance))
        if not passed:
            indices=(got!=expected).nonzero().cpu().tolist()
            report['failure_values']=[dict(index=i,got=float(got[tuple(i)]),expected=float(expected[tuple(i)])) for i in indices[:32]]
        save(label);assert passed,(label,rel,absmax)
    def reference(xx,weight):return xx.double()@weight.double().T
    def run(xx,weight,arena,selective=False):
        raw,bound,out,candidates=arena;n=xx.shape[0]
        return mlp.gate_projection(xx,weight,out[:n],scratch=(raw[:n],bound[:n]),
            **dict(bias=case['bias'],candidates=candidates[:n]) if selective else {})
    def arena(rows):
        return (torch.empty((rows,256),device='cuda'),torch.empty((rows,256),device='cuda'),
                torch.empty((rows,256),dtype=torch.bfloat16,device='cuda'),
                torch.empty((rows,256),dtype=torch.bool,device='cuda'))
    def check_selection(got,reference,label,workspace):
        n=got.shape[0];ids=torch.empty((n,8),dtype=torch.int64,device='cuda');probs=torch.empty((n,8),device='cuda')
        mlp.top8(got.float(),case['bias'],ids,probs)
        # The router scores and bias addition are FP32. At saturation, distinct
        # FP64 sigmoid values become ties before native top-k selection.
        scores=reference.double().sigmoid().float()
        refids=torch.argsort(scores+case['bias'],dim=-1,descending=True,stable=True)[:,:8]
        refprobs=torch.gather(scores.double(),1,refids);refprobs=refprobs/refprobs.sum(1,keepdim=True)
        if not torch.equal(ids,refids):
            from vllm import _custom_ops as ops
            native_ids=torch.empty_like(ids);native_probs=torch.empty_like(probs)
            ops.topk_sigmoid(native_probs,native_ids,torch.empty_like(ids,dtype=torch.int32),reference.float(),True,case['bias'],1.)
            rounded_ids=torch.argsort(scores.float()+case['bias'],dim=-1,descending=True,stable=True)[:,:8]
            report['selection_diagnostic']=dict(label=label,native_matches_port=torch.equal(ids,native_ids),
                rounded_fp32_scores_match_port=torch.equal(ids,rounded_ids),
                mismatched_rows=(ids!=refids).any(1).nonzero().flatten().cpu().tolist(),
                native_ids=native_ids.cpu().tolist(),fp64_ids=refids.cpu().tolist(),ported_ids=ids.cpu().tolist())
            save('selection_diagnostic')
        compare(ids,refids,label+'_ids',exact=True)
        compare(torch.gather(got,1,ids),torch.gather(reference,1,refids),label+'_logits',exact=True)
        compare(probs,refprobs,label+'_probabilities',tolerance=1e-6)
        assert bool(torch.gather(workspace[3][:n],1,refids).all())
        native_ids=torch.empty_like(ids);native_probs=torch.empty_like(probs)
        native_ops.topk_sigmoid(native_probs,native_ids,torch.empty_like(ids,dtype=torch.int32),reference.float(),True,case['bias'],1.)
        compare(ids,native_ids,label+'_native_ids',exact=True)
        compare(probs,native_probs,label+'_native_probabilities',tolerance=1e-6)
    faulthandler.dump_traceback_later(235,exit=True)
    try:
        case={k:v.cuda() for k,v in load_file(str(a.case)).items()}
        x,w=case['hidden'],case['gate'];s=arena(3072)
        compare(reference(x,w),case['fp64'],'saved_exact_operands_fp64',exact=True)
        out=run(x,w,s).clone();ref=bf16_nearest(case['fp64'])
        compare(out,ref,'saved_draft_projection',max_error=True)
        bad=(case['ported']!=ref.float()).nonzero()
        report['original_mismatches']=bad.cpu().tolist()
        for row,col in bad.cpu().tolist():
            compare(out[row:row+1,col:col+1],ref[row:row+1,col:col+1],f'repair_{row}_{col}',exact=True)
        ids=torch.empty((len(x),8),dtype=torch.int64,device='cuda');prob=torch.empty((len(x),8),device='cuda')
        mlp.top8(out.float(),case['bias'],ids,prob)
        scores=bf16_nearest(case['fp64']).double().sigmoid()
        refids=torch.argsort(scores+case['bias'].double(),dim=-1,descending=True,stable=True)[:,:8]
        rp=torch.gather(scores,1,refids);rp=rp/rp.sum(1,keepdim=True)
        compare(ids,refids,'saved_draft_route_ids',exact=True)
        compare(prob,rp,'saved_draft_route_probabilities',tolerance=1e-6)
        filtered=run(x,w,s,True).clone();check_selection(filtered,ref,'saved_filtered',s)
        # Changing the batch arrangement cannot change any corrected logit.
        for row in range(len(x)):compare(run(x[row:row+1],w,s),out[row:row+1],f'serial_{row}',exact=True)
        compare(run(x[:17],w,s),out[:17],'verify17',exact=True)
        big=x.repeat((154,1))[:3072].contiguous()
        bo=run(big,w,s).clone()
        compare(bo,out.repeat((154,1))[:3072],'batch3072_all_rows',exact=True)
        compare(run(big,w,s,True),filtered.repeat((154,1))[:3072],'filtered_batch3072_all_rows',exact=True)
        del big,bo
        # Independent float64 references on new inputs: small and large scales,
        # sign reversal, exact cancellation and the original quantized weights.
        torch.manual_seed(536006)
        for scale in (1e-3,.25,2.,16.):
            xx=torch.randn((64,6144),dtype=torch.bfloat16,device='cuda')*scale
            expected=bf16_nearest(reference(xx,w));actual=run(xx,w,s).clone()
            compare(actual,expected,f'fresh64_scale{scale}',max_error=True)
            check_selection(run(xx,w,s,True),expected,f'fresh64_filtered_scale{scale}',s)
        # Manufactured BF16 midpoints approached from either direction. All
        # operands are exactly representable BF16; FP32 may round to the tie.
        ww=torch.zeros_like(w);xx=torch.zeros((16,6144),dtype=torch.bfloat16,device='cuda')
        ww[:,0]=1.;ww[:,1]=2**-8;ww[:,2]=2**-28
        xx[:,0]=1.;xx[:,1]=1.;xx[:,2]=torch.tensor([1.,-1.]*8,device='cuda')
        xx[8:,:3].mul_(-1.)
        expected=bf16_nearest(reference(xx,ww))
        compare(run(xx,ww,s),expected,'constructed_midpoints',exact=True)
        xx.zero_();xx[:,0]=1.;xx[:,1]=-256.;xx[:,2]=1.
        compare(run(xx,ww,s),bf16_nearest(reference(xx,ww)),'constructed_cancellation',max_error=True)
        # Capture and replay changing input rows with persistent caller scratch.
        gx=x.clone();gs=arena(len(x));run(gx,w,gs)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):run(gx,w,gs)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):go=run(gx,w,gs)
        for i in range(3):
            gx.copy_(torch.roll(x,i+1,0)*(-1 if i%2 else 1));expected=run(gx,w,gs).clone();graph.replay()
            compare(go,expected,f'graph_changed{i}',exact=True)
            compare(go,bf16_nearest(reference(gx,w)),f'graph_reference{i}',max_error=True)
        s1,s2=arena(len(x)),arena(len(x));xx=-x
        e1=run(x,w,s1).clone();e2=run(xx,w,s2).clone()
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for st in streams:st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):o1=run(x,w,s1)
        with torch.cuda.stream(streams[1]):o2=run(xx,w,s2)
        for st in streams:torch.cuda.current_stream().wait_stream(st)
        compare(o1,e1,'separate_stream1',exact=True);compare(o2,e2,'separate_stream2',exact=True)
        run(gx,w,gs,True)
        filtered_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(filtered_graph,stream=stream):fo=run(gx,w,gs,True)
        for i in range(3):
            gx.copy_(torch.roll(x,i+1,0)*(-1 if i%2 else 1))
            expected=run(gx,w,gs,True).clone();filtered_graph.replay()
            compare(fo,expected,f'filtered_graph_changed{i}',exact=True)
            check_selection(fo,bf16_nearest(reference(gx,w)),f'filtered_graph_reference{i}',gs)
        report['timing_and_uncertainty']={}
        for count,repetitions in ((1,20),(20,10),(3072,2)):
            xx=x.repeat((154,1))[:count].contiguous();ns=torch.empty((count,256),dtype=torch.bfloat16,device='cuda')
            run(xx,w,s,True);torch.mm(xx,w.T,out=ns);torch.cuda.synchronize()
            raw,bound,_,candidates=s
            fraction=float(((raw[:count]-bound[:count]).bfloat16()!=(raw[:count]+bound[:count]).bfloat16()).float().mean())
            filtered_fraction=float((((raw[:count]-bound[:count]).bfloat16()!=(raw[:count]+bound[:count]).bfloat16())&candidates[:count]).float().mean())
            timings={}
            for label,fn in (('port_full',lambda:run(xx,w,s)),('port_routing',lambda:run(xx,w,s,True)),('native_bf16_mm',lambda:torch.mm(xx,w.T,out=ns))):
                fn();capture_stream=torch.cuda.Stream();capture_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(capture_stream):fn()
                torch.cuda.current_stream().wait_stream(capture_stream)
                captured=torch.cuda.CUDAGraph()
                with torch.cuda.graph(captured,stream=capture_stream):fn()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(repetitions):captured.replay()
                end.record();end.synchronize();timings[label]=start.elapsed_time(end)/repetitions
            report['timing_and_uncertainty'][str(count)]=dict(graph_milliseconds=timings,fp64_recheck_fraction=fraction,
                                                            filtered_fp64_recheck_fraction=filtered_fraction)
            save(f'timing_rows{count}')
        assert not torch.distributed.is_initialized()
        report.update(passed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report.update(error=type(exc).__name__+': '+str(exc));save('failed');raise
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
