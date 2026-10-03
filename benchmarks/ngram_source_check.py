#!/usr/bin/env python3
"""CPU-only checks of the pinned vLLM copy proposer, without loading vLLM/GPU.

Only its config type import is replaced; the proposer and Numba functions run
unchanged. This is not serving compatibility, target fidelity, or Spark timing.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np

SHA='5051e1bca645fe4befec0a6340f0dfc169fbcbe070ac1da508b72f8ee9c91736'


def oracle(tokens, low, high, k, capacity):
    k=min(k,capacity-len(tokens))
    if k<=0:
        return []
    for n in range(min(high,len(tokens)),low-1,-1):
        suffix=tokens[-n:]
        for i in range(len(tokens)-n):
            if tokens[i:i+n]==suffix:
                return tokens[i+n:i+n+k]
    return []


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists()
    raw=a.source.read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
    tree=ast.parse(raw)
    removed=[x for x in tree.body if isinstance(x,ast.ImportFrom) and x.module=='vllm.config']
    assert len(removed)==1 and [x.name for x in removed[0].names]==['VllmConfig']
    tree.body.remove(removed[0])
    module={'VllmConfig':object,'__name__':'pinned_ngram_cpu_check'}
    exec(compile(tree,str(a.source),'exec'),module)
    config=SimpleNamespace(speculative_config=SimpleNamespace(prompt_lookup_min=3,
        prompt_lookup_max=5,num_speculative_tokens=4),
        model_config=SimpleNamespace(max_model_len=8192),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        parallel_config=SimpleNamespace(tensor_parallel_size=6))
    proposer=module['NgramProposer'](config)
    warm_signatures=len(module['batch_propose_numba'].signatures)
    proposer.max_model_len=360000
    histories=[[10,11,12,13,14,20,21,22,23,10,11,12,13,14],
               [10,11,12,13,14,20,21,22,23,10,11,12,99,14],
               list(range(15)),list(range(15))]
    counts=np.array([len(x) for x in histories],dtype=np.int32)
    matrix=np.zeros((4,128),dtype=np.int32)
    for i,history in enumerate(histories):matrix[i,:len(history)]=history
    started=time.monotonic()
    drafts=proposer.propose(4,[[14],[14],[14],[]],counts,matrix)
    first_seconds=time.monotonic()-started
    expected=[oracle(x,3,5,4,360000) for x in histories[:3]]+[[]]
    assert drafts==expected and drafts[0]==[20,21,22,23] and drafts[1:]==[[],[],[]]
    randomizer=random.Random(17);checked=0
    find=module['_find_longest_matched_ngram_and_propose_tokens']
    for _ in range(400):
        length=randomizer.randrange(0,100)
        history=[randomizer.randrange(5) for _ in range(length)]
        low=randomizer.randrange(1,6);high=randomizer.randrange(low,9)
        k=randomizer.randrange(1,5);capacity=length+randomizer.randrange(0,7)
        got=find(np.array(history,dtype=np.int32),low,high,capacity,k).tolist()
        assert got==oracle(history,low,high,k,capacity)
        checked+=1
    # Exercise long canonical histories and TP6's single CPU thread path. These
    # local coordinator timings are diagnostic only, not model performance.
    timings=[]
    for length in [8192,32768,131072,359995]:
        history=(np.arange(length,dtype=np.int32)%97)
        tokens=np.tile(history,(4,1));counts=np.full(4,length,dtype=np.int32)
        samples=[]
        for _ in range(3):
            start=time.monotonic()
            output=proposer.propose(4,[[int(history[-1])]]*4,counts,tokens)
            samples.append(time.monotonic()-start)
            assert len(output)==4 and all(len(row)==4 for row in output)
        timings.append(dict(context=length,requests=4,seconds=samples))
    report=dict(phase='cpu_source_checks_passed',source_sha256=SHA,
        gpu_used=False,vllm_runtime_loaded=False,target_model_loaded=False,
        config_type_import_replaced=True,proposer_implementation_changed=False,
        proposal_examples=drafts,randomized_oracle_checks=checked,
        batch_jit_signatures_after_constructor=warm_signatures,
        batch_jit_signatures_after_requests=len(module['batch_propose_numba'].signatures),
        first_actual_proposal_seconds=first_seconds,long_context_local_cpu_samples=timings,
        limitations=['Local coordinator CPU, not DGX Spark or serving throughput.',
            'Constructor smoke uses8192 maximum then proposal capacity360000; no full vLLM configuration or target verification is tested.',
            'Proposal correctness alone does not establish TP6 scheduler/graph compatibility or generated-output fidelity.'])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['phase','randomized_oracle_checks','batch_jit_signatures_after_constructor','batch_jit_signatures_after_requests','first_actual_proposal_seconds']}))


if __name__=='__main__':main()
