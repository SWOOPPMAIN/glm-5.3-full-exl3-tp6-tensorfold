#!/usr/bin/env python3
"""Add the independent diagnostic oracle to the exact prepared canonical image."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    root = Path(importlib.util.find_spec('b12x').origin).parent/'attention/dsa_indexer'
    expected = {'canonical_index_topk.py': '26849f62be60644e8f169ba8c79bedb55e057c24d2d6b502354d627aaa22eefa',
                'paged.py': '08636b801a030634c35296745c76db34b90aa68365c1f37f5e55f301a719241e'}
    for name, digest in expected.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Oracle image requires exact P23r5 '+name)
    here = Path(__file__).parent
    files = ['canonical_index_topk.py', 'canonical_index_oracle.py']
    for name in files:
        compile((here/name).read_bytes(), name, 'exec')
    for name in files:
        shutil.copy2(here/name, root/name)
    print(json.dumps(dict(before=expected, after={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files},
                          model_output='Original fast outputs retained; oracle verifies every physical-slot prefill selection',
                          original_weights_unchanged=True)))


if __name__ == '__main__':
    main()
