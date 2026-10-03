"""Explicit row64/row32 E3 comparison on unchanged TP6 weights.

Only an idle, drained operator may change the six control files. Each rank
latches its policy at the first routed target layer for a prefill chunk. Missing
or invalid controls fail visibly; there is no automatic kernel fallback.
"""
import hashlib
import json
from pathlib import Path
import re

CONTROL=Path('/root/.cache/amos-e3-rows.json')
_current=None
_digest=None
_acknowledged=None


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'revision','rows'}
            or not isinstance(value['revision'],str)
            or not re.fullmatch(r'e3[0-9]+-[a-z0-9-]{1,64}',value['revision'])
            or type(value['rows']) is not int or value['rows'] not in (32,64)):
        raise ValueError('E3 policy requires a revision and row32 or row64')
    return value


def latch(layer_name,path=CONTROL):
    global _current,_digest
    if _current is None or layer_name.endswith('layers.3.mlp.experts'):
        raw=path.read_bytes()
        if len(raw)>1024:raise ValueError('Oversized E3 row control')
        current=validate(json.loads(raw));_digest=hashlib.sha256(raw).hexdigest()
        _current=current
    return _current


def apply(layer,x,weights,ids,**kwargs):
    global _acknowledged
    policy=latch(layer.layer_name)
    from . import runtime
    if policy['rows']==64:
        selected=runtime
    else:
        from . import runtime_row32
        if runtime_row32._SCRATCH and runtime_row32._SCRATCH is not runtime._SCRATCH:
            raise RuntimeError('E3 policies cannot keep independent scratch arenas')
        runtime_row32._SCRATCH=runtime._SCRATCH
        selected=runtime_row32
    result=selected.apply(layer,x,weights,ids,**kwargs)
    if _acknowledged!=_digest:
        import torch
        if not torch.cuda.is_current_stream_capturing():
            from vllm.distributed import get_tensor_model_parallel_rank
            rank=get_tensor_model_parallel_rank()
            assert 0<=rank<6
            record=dict(policy,sha256=_digest,rank=rank,layer_name=layer.layer_name,
                        input_rows=int(x.shape[0]),shared_scratch=True)
            out=CONTROL.with_suffix('.applied.json');temp=out.with_suffix('.tmp')
            temp.write_text(json.dumps(record,sort_keys=True)+'\n');temp.replace(out)
            _acknowledged=_digest
    return result
