#!/usr/bin/env python3
"""Execute pinned ngram tensor proposal logic on CPU, without vLLM or CUDA.

The compile decorator and unrelated classes/imports are omitted. This checks
tensor semantics, not compiled GPU execution, the scheduler, or full serving.
"""
import argparse
import ast
import contextlib
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import torch
from ngram_source_check import oracle

SHA='6b304fb5a2b7a2e4b6ac05c90c49d228975a7b1f7d566dc1445f048992163b72'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
    original=ast.parse(raw)
    klass=next(x for x in original.body if isinstance(x,ast.ClassDef) and x.name=='NgramGPUKernel')
    assert len(klass.decorator_list)==1;klass.decorator_list=[]
    proposer=next(x for x in original.body if isinstance(x,ast.ClassDef) and x.name=='NgramProposerGPU')
    method=next(x for x in proposer.body if isinstance(x,ast.FunctionDef) and x.name=='propose')
    module=ast.Module(body=[klass,method],type_ignores=[])
    ns={'torch':torch,'nn':torch.nn,'VllmConfig':object,
        'set_forward_context':lambda *a:contextlib.nullcontext(),
        'record_function_or_nullcontext':lambda *a:contextlib.nullcontext()}
    exec(compile(module,str(a.source),'exec'),ns)
    cfg=SimpleNamespace(speculative_config=SimpleNamespace(prompt_lookup_min=3,
        prompt_lookup_max=5,num_speculative_tokens=4),
        model_config=SimpleNamespace(max_model_len=128),
        scheduler_config=SimpleNamespace(max_num_seqs=4))
    kernel=ns['NgramGPUKernel'](cfg,device=torch.device('cpu'))
    rng=random.Random(27);checked=0
    for _ in range(100):
        histories=[[rng.randrange(5) for _ in range(rng.randrange(0,100))] for _ in range(4)]
        tokens=torch.zeros((4,128),dtype=torch.int32)
        for i,h in enumerate(histories):tokens[i,:len(h)]=torch.tensor(h,dtype=torch.int32)
        lengths=torch.tensor([len(h) for h in histories],dtype=torch.int32)
        masks=torch.tensor([rng.choice([True,True,False]) for _ in histories])
        got,counts=kernel(lengths,tokens,masks)
        for i,h in enumerate(histories):
            expected=oracle(h,3,5,4,128) if masks[i] else []
            assert counts[i].item()==len(expected)
            assert got[i].tolist()==expected+[-1]*(4-len(expected))
            checked+=1
    # Full proposer scatter must preserve the valid sampled suffix when some
    # padded positions share the final in-bounds index near capacity.
    owner=SimpleNamespace(k=4,device=torch.device('cpu'),min_n=3,kernel=kernel,vllm_config=None)
    boundary=[]
    for prior,new_count in [(20,1),(20,5),(125,1),(125,3),(126,1),(126,2),(127,1)]:
        history=torch.arange(128,dtype=torch.int32).remainder(11).unsqueeze(0)
        before=history.clone()
        samples=torch.tensor([[101,102,103,104,105]],dtype=torch.int32)
        counts=torch.tensor([new_count],dtype=torch.int32)
        ns['propose'](owner,4,torch.tensor([prior],dtype=torch.int32),history,samples,counts)
        expected=before.clone();expected[0,prior:prior+new_count]=samples[0,:new_count]
        boundary.append(dict(prior_tokens=prior,new_count=new_count,
            correct_history=torch.equal(history,expected),
            final_token=int(history[0,-1]),expected_final_token=int(expected[0,-1])))
    result=dict(phase='cpu_tensor_semantics_checked',source_sha256=SHA,gpu_used=False,
        target_model_loaded=False,compile_decorator_removed=True,proposal_oracle_checks=checked,
        boundary_scatter_cases=boundary,boundary_scatter_all_passed=all(x['correct_history'] for x in boundary),
        limitations=['Uncompiled PyTorch CPU semantics, not GPU scheduling/compilation qualification.',
            'Boundary scatter disagreements affect proposed copies; target verification remains required and this test makes no claim of incorrect final output.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
