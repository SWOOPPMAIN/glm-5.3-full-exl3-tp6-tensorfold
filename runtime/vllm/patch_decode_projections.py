#!/usr/bin/env python3
"""Install narrow, source-pinned hooks in the existing opaque MXFP8 linear op."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RELATIVE='model_executor/kernels/linear/mxfp8/b12x.py'
SHA256='0e6d8e9ec6b428f26f3994685745ff46e93b6933277fe123126e615f854bdf32'
P12_SHA256='6be22b26e7a8df668c6e591b31ae3af730aa26bb1a396d8d7ffe372eb96ff10f'


def replace(text,before,after):
    if text.count(before)!=1:raise ValueError('Patch anchor changed: '+before[:100])
    return text.replace(before,after)


def transform(text):
    text=replace(text,'from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\n',
        'from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig\nfrom vllm import amos_decode_projections\n')
    text=replace(text,'    output = mxfp8.mm(\n        input_2d,\n',
        '    sharded = amos_decode_projections.apply_if_small(\n'
        '        layer, input_2d, bias, mxfp8, current_stream().cuda_stream)\n'
        '    if sharded is not None:\n'
        '        return sharded.view(*output_shape)\n'
        '    output = mxfp8.mm(\n        input_2d,\n')
    text=replace(text,'        _register_b12x_mxfp8_linear_layer(layer)\n',
        '        amos_decode_projections.prepare(layer, weight,\n'
        '            weight_scale[:out_features, :scale_k], mxfp8)\n'
        '        _register_b12x_mxfp8_linear_layer(layer)\n')
    text=replace(text,'        if warmed > 0 and last_device is not None and last_device.type == "cuda":\n',
        '        warmed += amos_decode_projections.warmup(model, mxfp8)\n'
        '        if warmed > 0 and last_device is not None and last_device.type == "cuda":\n')
    return text


def repair_warmup(text):
    # Worker runtime warmup runs outside set_current_vllm_config. The original
    # backend gate therefore sees 'auto' even though all layers chose B12X.
    return replace(text,'    if not _b12x_mxfp8_enabled():\n        return 0\n',
        '    if not _b12x_mxfp8_enabled() and not amos_decode_projections.enabled():\n        return 0\n')


def patch(root):
    path=root/RELATIVE;source=path.read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    if digest==SHA256:result=transform(source.decode())
    elif digest==P12_SHA256:result=source.decode()
    else:raise ValueError('Source differs from pinned runtime')
    result=repair_warmup(result);compile(result,str(path),'exec');path.write_text(result)
    shutil.copy2(Path(__file__).with_name('amos_decode_projections.py'),root/'amos_decode_projections.py')
    return {RELATIVE:hashlib.sha256(path.read_bytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('vllm_root',type=Path);a=p.parse_args()
    print(json.dumps(patch(a.vllm_root)))
