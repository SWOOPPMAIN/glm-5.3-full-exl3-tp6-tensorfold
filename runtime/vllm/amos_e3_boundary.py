"""Required, chunk-latched native/E3 boundary for the qualified row32 kernels.

Use the same dispatch in generation and teacher scoring. Operators may change
controls only with the six-rank service idle and drained. Native decode remains
outside this module; no file or kernel fallback is provided.
"""
import hashlib
import json
from pathlib import Path
import re

CONTROL=Path('/root/.cache/amos-e3-boundary.json')
_current=None
_digest=None
_acknowledged=None


def validate(value):
    if (not isinstance(value,dict) or set(value)!={'revision','native_max_rows'}
            or not isinstance(value['revision'],str)
            or not re.fullmatch(r'e3[0-9]+-[a-z0-9-]{1,64}',value['revision'])
            or type(value['native_max_rows']) is not int
            or value['native_max_rows'] not in (32,512)):
        raise ValueError('E3 boundary requires a revision and native_max_rows32 or512')
    return value


def latch(layer_name,path=CONTROL):
    global _current,_digest
    if _current is None or layer_name.endswith('layers.3.mlp.experts'):
        raw=path.read_bytes()
        if len(raw)>1024:raise ValueError('Oversized E3 boundary control')
        value=validate(json.loads(raw))
        from . import policy
        if policy.latch(layer_name)['rows']!=32:
            raise ValueError('Boundary comparison requires the qualified row32 policy')
        _current=value;_digest=hashlib.sha256(raw).hexdigest()
    return _current


def acknowledge(layer_name,rows,route):
    global _acknowledged
    if _acknowledged==_digest:return
    import torch
    if torch.cuda.is_current_stream_capturing():return
    from vllm.distributed import get_tensor_model_parallel_rank
    rank=get_tensor_model_parallel_rank()
    if not 0<=rank<6:raise ValueError('Boundary control requires TP6')
    record=dict(_current,sha256=_digest,rank=rank,layer_name=layer_name,
                input_rows=rows,route=route,e3_rows=32,scope='dispatch_policy_observed')
    out=CONTROL.with_suffix('.applied.json');tmp=out.with_suffix('.tmp')
    tmp.write_text(json.dumps(record,sort_keys=True)+'\n');tmp.replace(out)
    _acknowledged=_digest
