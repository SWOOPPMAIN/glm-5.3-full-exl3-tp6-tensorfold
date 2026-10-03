#!/usr/bin/env python3
"""Prepare and CPU-check unique scatter destinations near history capacity.

This does not deploy or qualify GPU copy drafting. The unchanged target must
still verify every proposed token in a later full-model experiment.
"""
import argparse
import ast
import contextlib
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

SHA='6b304fb5a2b7a2e4b6ac05c90c49d228975a7b1f7d566dc1445f048992163b72'


def patch_source(raw):
    assert hashlib.sha256(raw).hexdigest()==SHA
    old=b'        write_positions.clamp_(max=max_seq_len - 1)\n'
    assert raw.count(old)==1
    new=b'''        # Padding must not alias the final valid sampled token. With at most
        # max_seq_len adjacent offsets, modulo destinations are unique; masked
        # out-of-bounds writes preserve their original values at the start.
        if max_new_tokens > max_seq_len:
            raise ValueError("Ngram sampled width exceeds history capacity")
        write_positions.remainder_(max_seq_len)
'''
    return raw.replace(old,new)


def method(raw):
    tree=ast.parse(raw)
    klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='NgramProposerGPU')
    f=next(x for x in klass.body if isinstance(x,ast.FunctionDef) and x.name=='propose')
    import torch
    namespace=dict(torch=torch,set_forward_context=lambda *a:contextlib.nullcontext(),
                   record_function_or_nullcontext=lambda *a:contextlib.nullcontext())
    exec(compile(ast.Module(body=[f],type_ignores=[]),'pinned_propose','exec'),namespace)
    return namespace['propose']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.candidate.exists() and not a.output.exists()
    raw=a.source.read_bytes();patched=patch_source(raw);compile(patched,'ngram_proposer_gpu.py','exec')
    original,candidate=method(raw),method(patched)
    import torch
    assert not torch.cuda.is_initialized();torch.set_num_threads(1)
    owner=SimpleNamespace(k=4,device=torch.device('cpu'),min_n=5,vllm_config=None,
        kernel=lambda lengths,tokens,mask:(lengths.clone(),tokens.clone()))
    rng=random.Random(36);failures=0;checks=0;boundary=[]
    for capacity in (8,32,128,360000):
        for case in range(128):
            priors=[max(0,capacity-1-(case%5)),rng.randrange(capacity),capacity,0]
            counts=[rng.randrange(min(5,capacity-prior)+1) for prior in priors]
            if case<5:counts[0]=capacity-priors[0]
            history=torch.arange(capacity,dtype=torch.int32).remainder(97).repeat(4,1)
            samples=torch.tensor([[1000+row*10+col for col in range(5)] for row in range(4)],dtype=torch.int32)
            expected=history.clone()
            for row,(prior,count) in enumerate(zip(priors,counts)):
                for col in range(count):expected[row,prior+col]=samples[row,col]
                for col in range(count,5):
                    if (case+col)%2:samples[row,col]=-1
            kwargs=(owner,4,torch.tensor(priors,dtype=torch.int32),None,samples,torch.tensor(counts,dtype=torch.int32))
            old_history=history.clone();original(*kwargs[:3],old_history,*kwargs[4:])
            new_history=history.clone();candidate(*kwargs[:3],new_history,*kwargs[4:])
            old_ok=torch.equal(old_history,expected);failures+=not old_ok
            assert torch.equal(new_history,expected),(capacity,case)
            assert not torch.cuda.is_initialized();checks+=4
            if case==0:boundary.append(dict(capacity=capacity,prior=priors[0],new_count=counts[0],
                original_correct=old_ok,candidate_correct=True))
    a.candidate.parent.mkdir(parents=True,exist_ok=True);a.candidate.write_bytes(patched)
    result=dict(phase='cpu_scatter_fix_checked_not_deployed',source_sha256=SHA,
        candidate_sha256=hashlib.sha256(patched).hexdigest(),history_rows_checked=checks,
        four_request_batches=checks//4,original_failing_batches=failures,candidate_all_passed=True,
        example_boundary_cases=boundary,capacities=[8,32,128,360000],gpu_used=False,
        target_model_loaded=False,proposal_matching_kernel_changed=False,
        limitations=['Only the original propose method scatter/history update was executed on PyTorch CPU; kernel return was a snapshot stub.',
            'Earlier source checks cover the unchanged matching kernel. GPU compilation/scheduling and full target verification remain unqualified.',
            'Original failures are history-state disagreements near terminal capacity, not proof of incorrect final target output.'])
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
