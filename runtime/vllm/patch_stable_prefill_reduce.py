#!/usr/bin/env python3
"""Install fixed large-prefill NCCL tiles on the pinned P23r3 communicator."""
import hashlib
import json
from pathlib import Path
import shutil

RELATIVE = 'distributed/device_communicators/cuda_communicator.py'
EXPECTED = 'f6ac0d17dd3933df0c6c60e57411b38cb09a086a3a53ec11ff298203d80d309f'


def transform(raw):
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('Stable prefill reduction requires the pinned P23r3 source')
    source = raw.decode()
    anchor = 'logger = init_logger(__name__)\n'
    assert source.count(anchor) == 1
    source = source.replace(anchor, 'from vllm import amos_stable_prefill_reduce\n\n'+anchor)
    # Scope replacements to their methods; other NCCL operations are untouched.
    for name, in_place in [('all_reduce', False), ('all_reduce_in_place', True)]:
        start = source.index('    def '+name+'(')
        stop = source.index('\n    def ', start+1)
        method = source[start:stop]
        anchor = '        pynccl_comm = self.pynccl_comm\n'
        assert method.count(anchor) == 1
        hook = ('        stable = amos_stable_prefill_reduce.apply(\n'
                '            self.pynccl_comm, input_, in_place='+str(in_place)+',\n'
                '            tail_reduce='+('None' if in_place else 'self.all_reduce')+')\n'
                '        if stable is not None:\n'
                '            return stable\n')
        source = source[:start]+method.replace(anchor, hook+anchor)+source[stop:]
    compile(source, RELATIVE, 'exec')
    return source


def main():
    roots = [Path('/usr/local/lib/python3.12/dist-packages/vllm'), Path('/opt/glm53-full/vllm/vllm')]
    prepared = [(root, transform((root/RELATIVE).read_bytes())) for root in roots]
    helper = Path(__file__).with_name('amos_stable_prefill_reduce.py')
    compile(helper.read_bytes(), str(helper), 'exec')
    for root, source in prepared:
        (root/RELATIVE).write_text(source)
        shutil.copy2(helper, root/helper.name)
    print(json.dumps({'before':EXPECTED, 'after':hashlib.sha256(prepared[0][1].encode()).hexdigest(),
                      'helper':hashlib.sha256(helper.read_bytes()).hexdigest(),
                      'compute_batch_changed':False, 'weights_changed':False}))


if __name__ == '__main__':
    main()
