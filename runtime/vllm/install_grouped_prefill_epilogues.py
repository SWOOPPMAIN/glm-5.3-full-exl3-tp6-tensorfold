#!/usr/bin/env python3
"""Install the grouped GEMM/native epilogue repair over exact P21r2."""
import hashlib
import importlib.util
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
        if name.endswith('.py'):
            compile((source/name).read_text(), str(source/name), 'exec')
    binary = source/'amos_e3/grouped_fragments.cubin'
    expected = '799dd2587446158fad03aa6bf212072fff96a322bb8fa21f791fc53331dd847f'
    if sha(binary) != expected or manifest['cubin_sha256'] != expected:
        raise ValueError('Candidate cubin differs from the exact-parity component test')
    b12x = Path(importlib.util.find_spec('b12x').origin).parent
    if sha(b12x/'moe/_shared/kernels/w4a16/kernel.py') != '524af13b672674cd9ce1fd543163bc0b65fa5a6ac62057307da0cd6c3cda3fb8':
        raise ValueError('Native activation or ordered-sum source changed')
    roots = (Path('/usr/local/lib/python3.12/dist-packages/vllm'),
             Path('/opt/glm53-full/vllm/vllm'))
    for root in roots:
        if sha(root/'model_executor/layers/quantization/exl3.py') != 'c5dd0c5d4025cd04832db5c2288a5d9aef1742358716ba53e60076b011ee22ed':
            raise ValueError('Requires the native runtime initialization repair')
        if sha(root/'amos_e3/grouped_fragments.cubin') != 'a140d22bf44a661581a4931c8b0a7f04f576c1901b290d0ee979bba6d8fdb167':
            raise ValueError('Requires exact P21r2 parent kernel')
    for root in roots:
        shutil.copytree(source/'amos_e3', root/'amos_e3', dirs_exist_ok=True)
        if sha(root/'amos_e3/grouped_fragments.cubin') != expected:
            raise ValueError('Installed E3 hash mismatch')
    print(json.dumps(dict(cubin_sha256=expected, installed=True,
                         full_model_qualified=False)))


if __name__ == '__main__':
    main()
