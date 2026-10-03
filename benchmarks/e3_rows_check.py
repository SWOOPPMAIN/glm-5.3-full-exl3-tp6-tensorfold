#!/usr/bin/env python3
"""Replay exact captured TP6 operands in one bounded, isolated GPU container.

The outer operator must stop every serving rank and arm this exact container's
memory guard before releasing /probe/go. No distributed initialization or NCCL.
Both E3 arms share one stream-local scratch arena and alternate timing order.
"""
import argparse
import faulthandler
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

from grouped_prefill_kernel_check import load_layer, require

ROWS = (513,768,1024,1536,3072)


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank',required=True,type=int,choices=range(6))
    parser.add_argument('--layer',required=True,type=int,choices=(3,40,77))
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();require(not args.output.exists(),'Existing result')
    start=time.monotonic()
    while not Path('/probe/go').exists():
        require(time.monotonic()-start<40,'Exact-container guard was not released')
        time.sleep(.1)
    envelope=Path('/sys/fs/cgroup/memory.max').read_text().strip()
    require(envelope!='max' and int(envelope)<=4*2**30,'Requires4GiB cgroup')
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    require(available>=12*2**30,'Requires12GiB host headroom')
    faulthandler.dump_traceback_later(230,exit=True)
    os.environ.update(AMOS_EXL3_TP6_PIECES='1',AMOS_TP6_E3_PREFILL='1',VLLM_EXL3_PREFILL_BLOCK_M='32')
    manifest=json.loads(Path('/probe/manifest.json').read_text())
    for name,expected in manifest['probe_files'].items():require(sha('/probe/'+name)==expected,'Probe source mismatch:'+name)
    for name,expected in manifest['installed_files'].items():require(sha(name)==expected,'Installed source mismatch:'+name)
    report=dict(phase='initializing',passed=False,rank=args.rank,layer=args.layer,capacity=3072,cases=[],timings=[],
                source_manifest_sha256=sha('/probe/manifest.json'),image=manifest['image'],
                scope='Exact3072-row live-capture operands and prefixes; isolated local shard, not full-model speed.',
                container_limit_gib=4,torch_limit_gib=2,distributed_initialized=False)
    def save(phase):
        report.update(phase=phase,updated_at=time.time())
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
        print(json.dumps(dict(rank=args.rank,layer=args.layer,phase=phase)),flush=True)
    save('loading_one_layer')
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(1);torch.cuda.set_device(0)
    require(torch.cuda.get_device_capability()==(12,1),'RequiresSM121')
    torch.cuda.set_per_process_memory_fraction(2*2**30/torch.cuda.get_device_properties(0).total_memory)
    from vllm.amos_e3 import runtime as base
    spec=importlib.util.spec_from_file_location('vllm.amos_e3.row32_runtime','/probe/row32/runtime.py')
    small=importlib.util.module_from_spec(spec);spec.loader.exec_module(small)
    require(small.SMEM==32768,'Wrong row32 shared memory')
    # Arrays have identical capacity/layout; both arms use them sequentially.
    small._SCRATCH=base._SCRATCH
    method,layer=load_layer(Path('/model'),args.rank,args.layer,3072)
    import vllm.model_executor.layers.quantization.exl3 as exl3
    x0=torch.zeros((1,6144),device='cuda',dtype=torch.bfloat16)
    ids0=torch.zeros((1,8),device='cuda',dtype=torch.int64)
    with patch.object(exl3,'_shared_mixed_buffers',return_value=SimpleNamespace()):
        planned=method._mixed_rank_sliced_runtime(layer,x0,ids0)
    require(vars(planned['decode']['buffers'])==vars(planned['prefill']['buffers'])=={},'Unexpected native arena')
    del x0,ids0
    base.bind(layer)
    data={}
    for kind in ('code','prose'):
        path=Path('/captures')/('e31-'+kind)/('rank'+str(args.rank))/f'layer{args.layer:02d}.safetensors'
        meta=json.loads(path.with_suffix('.json').read_text())
        expected=manifest['capture_files'][kind][str(args.rank)][str(args.layer)]
        require(meta['sha256']==expected and sha(path)==expected,'Capture changed')
        cpu=load_file(str(path),device='cpu')
        for name,t in cpu.items():require(hashlib.sha256(t.contiguous().view(torch.uint8).numpy()).hexdigest()==meta['tensor_sha256'][name],'Tensor mismatch:'+name)
        require(torch.equal(cpu['mapping'],layer.exl3_mixed_trellis['global_to_combined'].cpu()),'Local mapping differs')
        require(torch.equal(cpu['bits'],layer.glm6_e3_binding['bits'].cpu()),'Local precision differs')
        data[kind]=cpu
    execute={'row64':lambda v:base.apply(layer,*v,stream_scratch=True),
             'row32':lambda v:small.apply(layer,*v,stream_scratch=True)}
    def values(kind,rows):
        d=data[kind]
        return tuple(d[n][:rows].contiguous().cuda() for n in ('input','weights','ids'))
    save('exact_component_checks')
    for kind in ('code','prose'):
        for rows in ROWS:
            operands=values(kind,rows)
            for arm in ('row64','row32'):
                result=execute[arm](operands).cpu()
                reference=data[kind]['output'][:rows]
                same=torch.equal(result,reference)
                report['cases'].append(dict(kind=kind,rows=rows,arm=arm,exact_captured_output=same,
                    max_abs_difference=float((result.float()-reference.float()).abs().max())))
                save('checking_'+kind+'_'+str(rows)+'_'+arm)
                require(same,'Captured output differs:'+arm)
                del result
            del operands
    # Graph replay changes both inputs and routes; preserved scratch ownership
    # cannot accidentally make repeated fixed-input calls pass.
    save('graph_checks')
    torch.cuda.synchronize();base._SCRATCH.clear();torch.cuda.empty_cache()
    replay_stream=torch.cuda.Stream()
    with torch.cuda.stream(replay_stream):
        for arm in ('row64','row32'):
            operands=values('code',3072);execute[arm](operands);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=replay_stream):out=execute[arm](operands)
            for kind in ('prose','code','prose'):
                for target,name in zip(operands,('input','weights','ids')):target.copy_(data[kind][name])
                graph.replay();result=out.cpu()
                same=torch.equal(result,data[kind]['output'])
                report['cases'].append(dict(kind=kind,rows=3072,arm=arm,graph=True,exact_captured_output=same))
                require(same,'Graph replay differs:'+arm)
            del graph,out,operands,result
            torch.cuda.synchronize()
    base._SCRATCH.clear();del replay_stream;torch.cuda.empty_cache()
    report['quality_passed']=True;save('timing_qualified_arms')
    flush=torch.empty(64*1024**2,dtype=torch.uint8,device='cuda')
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for kind in ('code','prose'):
        for rows in ROWS:
            operands=values(kind,rows)
            for arm in execute:execute[arm](operands)
            samples=[]
            for repeat in range(8):
                order=('row64','row32') if repeat%2==0 else ('row32','row64')
                for arm in order:
                    flush.zero_();torch.cuda.synchronize();wall=time.perf_counter();begin.record()
                    execute[arm](operands)
                    end.record();end.synchronize()
                    samples.append(dict(arm=arm,repeat=repeat,gpu_ms=begin.elapsed_time(end),wall_ms=1000*(time.perf_counter()-wall)))
            for target,name in zip(operands,('input','weights','ids')):require(torch.equal(target.cpu(),data[kind][name][:rows]),'Input mutated')
            report['timings'].append(dict(kind=kind,rows=rows,samples=samples,
                median_ms={arm:statistics.median(s['gpu_ms'] for s in samples if s['arm']==arm) for arm in execute}))
            del operands
            save('timed_'+kind+'_'+str(rows))
    require(not torch.distributed.is_initialized(),'Unexpected distributed initialization')
    report.update(passed=True,peak_cuda_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  scratch_arenas=len(base._SCRATCH),runtime_scratch_shared=True)
    require(report['peak_cuda_allocated_gib']<=2,'Exceeded Torch budget')
    save('complete');faulthandler.cancel_dump_traceback_later()


if __name__=='__main__':main()
