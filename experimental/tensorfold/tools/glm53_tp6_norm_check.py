#!/usr/bin/env python3
"""Original hidden norms: BF16 residual semantics and row/graph invariance."""
import argparse,faulthandler,hashlib,importlib.util,json,math,sys,time
from pathlib import Path
from glm53_tp6_reference import bf16_nearest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','module','manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--layer',type=int,choices=(0,),required=True)
    a=p.parse_args();assert not a.output.exists()
    limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    assert limit!='max' and int(limit)<=4*2**30
    available=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    assert available>=64*2**30
    manifest=json.loads(a.manifest.read_text())
    for name,digest in manifest.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
    assert all(str(v) in manifest for v in (a.module,Path(__file__)))
    import torch
    torch.set_num_threads(1);torch.cuda.set_device(0)
    assert torch.cuda.get_device_capability()==(12,1)
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    spec=importlib.util.spec_from_file_location('norm_under_test',a.module)
    norm=importlib.util.module_from_spec(spec);spec.loader.exec_module(norm)
    from tensorfold.families.glm_moe_dsa.checkpoint import RankPieces
    from vllm import _custom_ops as native
    native_bin=Path(importlib.util.find_spec('vllm._C_stable_libtorch').origin)
    digest=hashlib.sha256(native_bin.read_bytes()).hexdigest()
    assert digest=='79d638a3c430b3ec51db39571b906b1e85a2875e65eee8dab549a77276f4eddd'
    report=dict(passed=False,phase='initializing',cases=[],weights={},started_at=time.time(),source_manifest=manifest,
                native_binary_sha256=digest,scope='6144 hidden normalization and BF16 residual only; no distributed decoder or whole-model claim')
    def save(phase):
        report.update(phase=phase,updated_at=time.time(),peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        t=a.output.with_suffix('.tmp');t.write_text(json.dumps(report,indent=2)+'\n');t.replace(a.output)
        print(json.dumps({'phase':phase}),flush=True)
    def compare(got,ref,label,*,exact=False):
        torch.cuda.synchronize();x,y=got.double(),ref.double()
        same=torch.equal(x,y);finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all())
        denominator=float(y.square().mean().sqrt());rms=float((x-y).square().mean().sqrt())/(denominator or 1.)
        # At most one BF16 step, plus an explicit subnormal step at zero.
        # ldexp makes the BF16 step an exact power of two. Generic GPU pow
        # introduced a one-ulp error in the measuring scale itself.
        _,exponent=torch.frexp(y.abs())
        scale=torch.ldexp(torch.ones_like(y),torch.clamp(exponent-8,min=-133))
        steps=float(((x-y).abs()/scale).max())
        passed=finite and (same if exact else rms<=1e-4 and steps<=1.)
        report['cases'].append(dict(case=label,passed=passed,exact_required=exact,bit_equal=same,relative_rms=rms,max_bf16_steps=steps))
        save(label);assert passed,(label,rms,steps)
    def reject(fn,label):
        try:fn()
        except ValueError:report['cases'].append(dict(case=label,passed=True,rejected=True));save(label);return
        raise AssertionError(label+' did not reject')
    def reference(x,w,res):
        summed=x if res is None else bf16_nearest(x.double()+res.double())
        raw=summed.double();out=raw*torch.rsqrt(raw.square().mean(1,keepdim=True)+1e-5)*w.double()
        return bf16_nearest(out),summed
    def execute(x,w,res=None):
        out=torch.empty(x.shape,dtype=torch.bfloat16,device='cuda');summed=torch.empty_like(out)
        norm.hidden_rms(x,w,out,summed,residual=res)
        return out,summed
    faulthandler.dump_traceback_later(235,exit=True)
    try:
        torch.manual_seed(530607);reader=RankPieces(a.model,0)
        keys=[f'model.layers.{layer}.{name}.weight' for layer in (0,40,77,78)
              for name in ('input_layernorm','post_attention_layernorm')]+['model.norm.weight']
        for key in keys:
            meta=reader.tensor_meta(key);assert meta['dtype']=='BF16' and meta['shape']==[6144]
            raw=reader.read_bytes(key);report['weights'][key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            w=reader.read_tensor(key).cuda()
            x=torch.randn((20,6144),device='cuda').bfloat16()
            x[0].zero_();x[1].mul_(.001);x[2].mul_(16);x[3].mul_(10000)
            res=torch.randn_like(x);res[4]=-x[4];res[5].fill_(2**-8);x[5].fill_(1.)
            for add in (False,True):
                rr=res if add else None;out,summed=execute(x,w,rr);ref,sref=reference(x,w,rr)
                label=key+('_add' if add else '_plain')
                compare(summed,sref,label+'_residual',exact=True);compare(out,ref,label+'_fp64')
                if add:
                    nx=x.clone();nr=res.clone();native.fused_add_rms_norm(nx,nr,w,1e-5)
                    compare(summed,nr,label+'_native_residual',exact=True)
                else:nx=torch.empty_like(x);native.rms_norm(nx,x,w,1e-5)
                compare(out,nx,label+'_native')
        # Use original layer0 norm for shape boundaries, strided inputs and graphs.
        w=reader.read_tensor(keys[0]).cuda();n=3072
        backing=torch.randn((n,6160),dtype=torch.bfloat16,device='cuda')*.25
        rb=torch.randn_like(backing)*.25;x=backing[:,:6144];res=rb[:,:6144]
        for add in (False,True):
            rr=res if add else None;out,summed=execute(x,w,rr);label='batch3072'+('_add' if add else '_plain')
            for row in (0,15,16,127,128,255,256,1535,1536,3071):
                aout,asum=execute(x[row:row+1],w,None if rr is None else rr[row:row+1])
                compare(aout,out[row:row+1],label+f'_serial{row}',exact=True)
                compare(asum,summed[row:row+1],label+f'_skip{row}',exact=True)
            for size in (17,256):
                for start in range(0,n,size):
                    stop=min(start+size,n)
                    aa,ss=execute(x[start:stop],w,None if rr is None else rr[start:stop])
                    assert torch.equal(aa,out[start:stop]) and torch.equal(ss,summed[start:stop])
                report['cases'].append(dict(case=label+f'_all_rows_chunk{size}',passed=True,exact_required=True));save(label+f'_chunk{size}')
            sample=torch.tensor([0,128,255,256,1535,3071],device='cuda')
            ref,sref=reference(x[sample],w,None if rr is None else rr[sample]);compare(out[sample],ref,label+'_fp64')
            if add:
                nx=x.contiguous();nr=res.contiguous();native.fused_add_rms_norm(nx,nr,w,1e-5)
                compare(summed,nr,label+'_native_residual',exact=True)
            else:nx=torch.empty_like(out);native.rms_norm(nx,x,w,1e-5)
            compare(out,nx,label+'_native')
        del backing,rb,x,res,out,summed,nx
        x=torch.randn((17,6144),dtype=torch.bfloat16,device='cuda');res=torch.randn_like(x)
        y=torch.empty_like(x);skip=torch.empty_like(x)
        for add in (False,True):
            rr=res if add else None
            def run():return norm.hidden_rms(x,w,y,skip,residual=rr)
            run();st=torch.cuda.Stream();st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):run()
            torch.cuda.current_stream().wait_stream(st)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=st):run()
            for step in range(3):
                x.copy_(torch.randn_like(x)*(.25+step));res.copy_(torch.randn_like(res))
                ref,sref=execute(x,w,rr);graph.replay()
                compare(y,ref,f'graph_add{add}_{step}',exact=True);compare(skip,sref,f'graph_skip{add}_{step}',exact=True)
        x2=torch.randn_like(x);r2=torch.randn_like(res);y2=torch.empty_like(y);s2=torch.empty_like(skip)
        e1,k1=execute(x,w,res);e2,k2=execute(x2,w,r2)
        streams=[torch.cuda.Stream(),torch.cuda.Stream()]
        for st in streams:st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(streams[0]):norm.hidden_rms(x,w,y,skip,residual=res)
        with torch.cuda.stream(streams[1]):norm.hidden_rms(x2,w,y2,s2,residual=r2)
        for st in streams:torch.cuda.current_stream().wait_stream(st)
        compare(y,e1,'stream1_norm',exact=True);compare(skip,k1,'stream1_skip',exact=True)
        compare(y2,e2,'stream2_norm',exact=True);compare(s2,k2,'stream2_skip',exact=True)
        reject(lambda:norm.hidden_rms(x,w,x,skip),'reject_input_overwrite')
        reject(lambda:norm.hidden_rms(x,w,y,y),'reject_output_alias')
        reject(lambda:norm.hidden_rms(x,w,y,res,residual=res),'reject_residual_overwrite')
        reject(lambda:norm.hidden_rms(x,w,y,skip,residual=res[:1]),'reject_residual_shape')
        reject(lambda:norm.hidden_rms(x.float(),w,y,skip),'reject_dtype')
        reject(lambda:norm.hidden_rms(x[:0],w,y[:0],skip[:0]),'reject_empty')
        # Offset aliases must be rejected even when data pointers differ.
        alias=torch.empty((18,6144),dtype=torch.bfloat16,device='cuda')
        reject(lambda:norm.hidden_rms(alias[:17],w,alias[1:],skip),'reject_partial_alias')
        report.update(passed=True,finished_at=time.time());save('complete')
    except Exception as exc:
        report['error']=type(exc).__name__+': '+str(exc);save('failed');raise
    finally:faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
