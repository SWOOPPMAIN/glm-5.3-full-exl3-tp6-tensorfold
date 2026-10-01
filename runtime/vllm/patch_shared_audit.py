#!/usr/bin/env python3
"""Record selected shared projections without changing their return values."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'model_executor/kernels/linear/mxfp8/b12x.py'
SHA256 = '8672c245865f8ae77f59b4637cbc723d2839b9531d17468f80a18cb79bb0d94c'


def patch(root):
    path = root / RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('MXFP8 source differs from pinned P18R2')
    source = raw.decode()
    replacements = {
        'from vllm import amos_decode_projections\n':
            'from vllm import amos_decode_projections, amos_shared_audit\n',
        '    return output.view(*output_shape)\n':
            '    amos_shared_audit.capture(layer, input_2d, packed_weight, output)\n'
            '    return output.view(*output_shape)\n',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError('MXFP8 recording anchor changed')
        source = source.replace(before, after)
    compile(source, str(path), 'exec')
    path.write_text(source)
    shutil.copy2(Path(__file__).with_name('amos_shared_audit.py'), root/'amos_shared_audit.py')
    return {RELATIVE: hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
