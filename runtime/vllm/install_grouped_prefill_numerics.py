#!/usr/bin/env python3
"""Install the forward E3 numerical repair over the exact P21r1 runtime."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path('/opt/amos-tp6/grouped-prefill/source')
    manifest = json.loads((source/'manifest.json').read_text())
    for name, expected in manifest['sources'].items():
        if sha(source/name) != expected:
            raise ValueError('Changed candidate source: '+name)
    binary = source/'amos_e3/grouped_fragments.cubin'
    expected = 'a140d22bf44a661581a4931c8b0a7f04f576c1901b290d0ee979bba6d8fdb167'
    if sha(binary) != expected:
        raise ValueError('Candidate cubin differs from isolated tested kernel')
    for root in (Path('/usr/local/lib/python3.12/dist-packages/vllm'),
                 Path('/opt/glm53-full/vllm/vllm')):
        if sha(root/'model_executor/layers/quantization/exl3.py') != 'c5dd0c5d4025cd04832db5c2288a5d9aef1742358716ba53e60076b011ee22ed':
            raise ValueError('Requires P21r1 native runtime initialization repair')
        if sha(root/'amos_e3/grouped_fragments.cubin') != '804aee636a1888635cb2ad8295eae206cc3de7e92ee6f94d9e4eeb181ea52bb0':
            raise ValueError('Unexpected installed E3 kernel')
        shutil.copytree(source/'amos_e3', root/'amos_e3', dirs_exist_ok=True)
        if sha(root/'amos_e3/grouped_fragments.cubin') != expected:
            raise ValueError('Installed E3 hash mismatch')
    print(json.dumps(dict(cubin_sha256=expected, installed=True, full_model_qualified=False)))


if __name__ == '__main__':
    main()
