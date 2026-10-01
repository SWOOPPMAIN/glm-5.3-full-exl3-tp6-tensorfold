#!/usr/bin/env python3
"""Patch only pinned chat sampling defaults; do not rewrite model output."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'entrypoints/openai/chat_completion/serving.py'
SHA256 = 'd016acf66372d449441b0ea75161002c0fd6aed0270c8082899f42f0266a3b74'


def patch(root):
    path = root / RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Chat serving source differs from pinned P14')
    source = raw.decode()
    before = ('                sampling_params = request.to_sampling_params(\n'
              '                    max_tokens,\n'
              '                    self.default_sampling_params,\n'
              '                )\n')
    after = ('                from vllm import amos_chat_boundaries\n'
             '                sampling_params = request.to_sampling_params(\n'
             '                    max_tokens,\n'
             '                    amos_chat_boundaries.defaults(\n'
             '                        self.default_sampling_params, request,\n'
             '                        self.model_config, tokenizer),\n'
             '                )\n')
    if source.count(before) != 1:
        raise ValueError('Chat sampling anchor changed')
    source = source.replace(before, after)
    compile(source, str(path), 'exec')
    path.write_text(source)
    shutil.copy2(Path(__file__).with_name('amos_chat_boundaries.py'), root/'amos_chat_boundaries.py')
    return {RELATIVE: hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
