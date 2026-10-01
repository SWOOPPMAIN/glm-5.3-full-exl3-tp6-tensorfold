#!/usr/bin/env python3
"""Prepare the missing indexer warmup specialization in the pinned TP6 source.

MiaAI-Lab identified that WarmupIntRange has an exclusive stop: (0, 2)
covers Triton's aligned and constant-one cases but misses the ordinary int
case. Including 2 warms that case without changing the GPU kernel or runtime
dispatch. This changes an offline image tree only; it does not deploy it.

Source: MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks at
674155dec2f2f62bb879801b5ce2cfc759a0bebf,
overlay/patch_indexer_warmup_range.py.
"""
import hashlib
import json
from pathlib import Path
import sys

RELATIVE = 'v1/attention/backends/mla/indexer.py'
EXPECTED = '3f57083b40e95f57b1dd687dc7f0c504f86784aa86dc4e669ca8d33da2ca3e52'
OLD = '                    query_slice_start=WarmupIntRange(0, 2),\n'
NEW = '                    query_slice_start=WarmupIntRange(0, 3),\n'


def patch(root):
    path = Path(root) / RELATIVE
    raw = path.read_bytes()
    source = raw.decode()
    before = hashlib.sha256(raw).hexdigest()
    # Accept only the original source or its exact one-line transformation.
    normalized = source.replace(NEW, OLD, 1).encode()
    if hashlib.sha256(normalized).hexdigest() != EXPECTED:
        raise ValueError('Indexer warmup requires the pinned TP6 source')
    if source.count(OLD) + source.count(NEW) != 1:
        raise ValueError('Indexer warmup anchor changed')
    changed = source.replace(OLD, NEW, 1)
    compile(changed, str(path), 'exec')
    after = hashlib.sha256(changed.encode()).hexdigest()
    if after != before:
        path.write_text(changed)
    return dict(before=before, after=after, changed=before != after,
                prepared=True, deployed=False, speed_gain_measured=False)


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
