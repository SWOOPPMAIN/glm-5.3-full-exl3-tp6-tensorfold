#!/usr/bin/env python3
"""Exclusive actual-route native/E3 comparison; no full-model speed claim.

Requires a stopped fleet and a fresh exact-container guard before /probe/go.
All kernels retain the production 3072-token capacity and original weights.
"""
import argparse
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

from grouped_prefill_kernel_check import load_layer, require

ROWS=(33,65,128,256,384,512,513,768,1024)
ARMS=('native','row64','row32')


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank',required=True,type=int,choices=range(6))
    p.add_argument('--layer',required=True,type=int,choices=(3,40,77))
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();require(not a.output.exists(),'Existing result')
    start=time.monotonic()
    while not Path('/probe/go').exists():
        require(time.monotonic()-start<40,'Exact-container guard was not released')
        time.sleep(.1)
    cap=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    require(cap!='max' and int(cap)<=4*2**30,'Requires 4 GiB cgroup')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available>=12*2**30,'Requires 12 GiB host headroom')
    faulthandler.dump_traceback_later(230,exit=True)
    os.environ.update(AMOS_EXL3_TP6_PIECES='1',AMOS_TP6_E3_PREFILL='0',VLLM_EXL3_PREFILL_BLOCK_M='32')
    manifest=json.loads(Path('/probe/manifest.json').read_text())
    for name,value in manifest['probe_files'].items():require(sha('/probe/'+name)==value,'Probe mismatch:'+name)
    for name,value in manifest['installed_files'].items():require(sha(name)==value,'Installed source mismatch:'+name)
    report=dict(phase='initializing',passed=False,rank=a.rank,layer=a.layer,capacity=3072,
        cases=[],timings=[],image=manifest['image'],source_manifest_sha256=sha('/probe/manifest.json'),
        container_limit_gib=4,torch_limit_gib=3,distributed_initialized=False,
        scope='Exact E31 actual-route operands and prefixes; isolated complete local MoE including routing and epilogues.')
    def save(phase):
        report.update(phase=phase,updated_at=time.time())
        temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(a.output)
        print(json.dumps(dict(rank=a.rank,layer=a.layer,phase=phase)),flush=True)
    save('loading_one_layer')
    import torch
    from safetensors.torch import load_file
    from vllm.amos_e3 import runtime as large, runtime_row32 as small
    torch.set_num_threads(1);torch.cuda.set_device(0)
    require(torch.cuda.get_device_capability()==(12,1),'Requires SM121')
    torch.cuda.set_per_process_memory_fraction(3*2**30/torch.cuda.get_device_properties(0).total_memory)
    small._SCRATCH=large._SCRATCH
    method,layer=load_layer(Path('/model'),a.rank,a.layer,3072)
    x0=torch.zeros((1,6144),device='cuda',dtype=torch.bfloat16)
    ids0=torch.zeros((1,8),device='cuda',dtype=torch.int64)
    native_plan=method._mixed_rank_sliced_runtime(layer,x0,ids0)
    require(native_plan['max_decode_m']==32 and native_plan['prefill_capacity']==3072,'Wrong native geometry')
    del x0,ids0
    large.bind(layer)
    data={}
    for kind in ('code','prose'):
        path=Path('/captures')/('e31-'+kind)/('rank'+str(a.rank))/f'layer{a.layer:02d}.safetensors'
        meta=json.loads(path.with_suffix('.json').read_text())
        expected=manifest['capture_files'][kind][str(a.rank)][str(a.layer)]
        require(meta['sha256']==expected and sha(path)==expected,'Capture changed')
        cpu=load_file(str(path),device='cpu')
        for name,t in cpu.items():
            require(hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()==meta['tensor_sha256'][name],'Tensor changed:'+name)
        require(torch.equal(cpu['mapping'],layer.exl3_mixed_trellis['global_to_combined'].cpu()),'Local mapping differs')
        require(torch.equal(cpu['bits'],layer.glm6_e3_binding['bits'].cpu()),'Local precision differs')
        data[kind]=cpu
    execute={'native':lambda v:method._apply_mixed_rank_sliced(layer,*v),
             'row64':lambda v:large.apply(layer,*v,stream_scratch=True),
             'row32':lambda v:small.apply(layer,*v,stream_scratch=True)}
    def values(kind,rows):return tuple(data[kind][n][:rows].contiguous().cuda() for n in ('input','weights','ids'))
    def check(out,kind,rows,arm,graph=False):
        actual=out.cpu();reference=data[kind]['output'][:rows]
        same=torch.equal(actual,reference)
        report['cases'].append(dict(kind=kind,rows=rows,arm=arm,graph=graph,exact_captured_output=same,
            max_abs_difference=float((actual.float()-reference.float()).abs().max())))
        save('checking_'+arm+'_'+str(rows))
        require(same,'Captured output differs:'+arm)
    save('exact_component_checks')
    for kind in ('code','prose'):
        for rows in ROWS:
            operands=values(kind,rows)
            for arm in ARMS:check(execute[arm](operands),kind,rows,arm)
            del operands
    torch.cuda.synchronize();large._SCRATCH.clear();torch.cuda.empty_cache()
    replay_stream=torch.cuda.Stream()
    with torch.cuda.stream(replay_stream):
        # Exercise both sides of the current boundary and an odd route count.
        for rows in (256,513):
            for arm in ARMS:
                operands=values('code',rows);execute[arm](operands);torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=replay_stream):out=execute[arm](operands)
                for kind in ('prose','code','prose'):
                    for target,name in zip(operands,('input','weights','ids')):target.copy_(data[kind][name][:rows])
                    graph.replay();check(out,kind,rows,arm,graph=True)
                del graph,out,operands
                torch.cuda.synchronize()
    large._SCRATCH.clear();del replay_stream;torch.cuda.empty_cache()
    report['quality_passed']=True;save('timing_qualified_arms')
    flush=torch.empty(64*1024**2,dtype=torch.uint8,device='cuda')
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    orders=(('native','row64','row32'),('row32','row64','native'),
            ('row64','native','row32'),('row32','native','row64'),
            ('native','row32','row64'),('row64','row32','native'),
            ('native','row64','row32'),('row32','row64','native'))
    for kind in ('code','prose'):
        for rows in ROWS:
            operands=values(kind,rows)
            for arm in ARMS:execute[arm](operands)
            samples=[]
            for repeat,order in enumerate(orders):
                for arm in order:
                    flush.zero_();torch.cuda.synchronize();wall=time.perf_counter();begin.record()
                    execute[arm](operands)
                    end.record();end.synchronize()
                    samples.append(dict(arm=arm,repeat=repeat,gpu_ms=begin.elapsed_time(end),wall_ms=1000*(time.perf_counter()-wall)))
            for target,name in zip(operands,('input','weights','ids')):
                require(torch.equal(target.cpu(),data[kind][name][:rows]),'Input mutated')
            report['timings'].append(dict(kind=kind,rows=rows,samples=samples,
                median_ms={arm:statistics.median(s['gpu_ms'] for s in samples if s['arm']==arm) for arm in ARMS}))
            del operands;save('timed_'+kind+'_'+str(rows))
    require(not torch.distributed.is_initialized(),'Unexpected distributed initialization')
    report.update(passed=True,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        scratch_arenas=len(large._SCRATCH),runtime_scratch_shared=small._SCRATCH is large._SCRATCH)
    require(report['peak_cuda_allocated_gib']<=3,'Exceeded Torch budget')
    save('complete');faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
