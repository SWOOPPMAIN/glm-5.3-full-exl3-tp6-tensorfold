#!/usr/bin/env python3
"""Add independent shared-expert padding to exact P17 sources."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HASHES={
    'config/virtual_tp.py':'abf46e9add946d41125b2ca76b3a0fbeb132afdda2b0dae4e2ba1dcef935f871',
    'model_executor/models/deepseek_v2.py':'6bc8a64950957cd8dd10cb094268fe32b01ea592c971dbfdcec9e7c9d3920fd0',
}


def patch(root):
    anchors={
        'config/virtual_tp.py':(
            '    vocab_axis = _make_virtual_vocab_axis(\n'
            '        _require_int_attr(text_config, "vocab_size"),\n'
            '        attention_tp_size,\n'
            '    )\n',
            '    from vllm.amos_shared_expert_padding import shared_axis\n'
            '    shared_expert_axis = shared_axis(\n'
            '        model_config, parallel_config, moe_original_size,\n'
            '        n_shared_experts, shared_expert_axis)\n'),
        'model_executor/models/deepseek_v2.py':(
            '            intermediate_size = config.moe_intermediate_size * config.n_shared_experts\n',
            '            from vllm.amos_shared_expert_padding import shared_width\n'
            '            intermediate_size = shared_width(config, self.tp_size, intermediate_size)\n'),
    }
    prepared={}
    for rel,digest in HASHES.items():
        p=root/rel;raw=p.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('Source differs from pinned P17: '+rel)
        source=raw.decode();before,addition=anchors[rel]
        if source.count(before)!=1:raise ValueError('Source anchor changed: '+rel)
        after=addition+before if rel.startswith('config/') else before+addition
        result=source.replace(before,after);compile(result,str(p),'exec');prepared[p]=result
    for p,value in prepared.items():p.write_text(value)
    shutil.copy2(Path(__file__).with_name('amos_shared_expert_padding.py'),root/'amos_shared_expert_padding.py')
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in prepared}


if __name__=='__main__':print(json.dumps(patch(Path(sys.argv[1]))))
