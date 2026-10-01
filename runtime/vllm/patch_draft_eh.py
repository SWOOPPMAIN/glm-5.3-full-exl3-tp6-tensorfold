#!/usr/bin/env python3
"""Source-pinned draft-only EH hook; preserve loading, norms and rejection."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

RELATIVE = 'model_executor/models/deepseek_mtp.py'
SHA256 = '041c628057c14e70d9c6b24ffba023457f8de4e4f777011dc0afc58ae4c74e4a'


def transform(source):
    replacements = [
        ('from vllm.config import VllmConfig\n',
         'from vllm.config import VllmConfig\nfrom vllm import amos_draft_eh\n'),
        ('        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)\n',
         '        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)\n'
         '        self.amos_draft_eh_rank = amos_draft_eh.configure(vllm_config, prefix)\n'),
        ('        hidden_states = self.eh_proj(\n'
         '            torch.cat([inputs_embeds, previous_hidden_states], dim=-1)\n'
         '        )\n',
         '        hidden_states = amos_draft_eh.apply(\n'
         '            self.eh_proj,\n'
         '            torch.cat([inputs_embeds, previous_hidden_states], dim=-1),\n'
         '            self.amos_draft_eh_rank,\n'
         '        )\n'),
        ('        return loaded_params\n',
         '        amos_draft_eh.warmup(self)\n        return loaded_params\n'),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError('Draft EH patch anchor changed: ' + before[:100])
        source = source.replace(before, after)
    return source


def patch(root):
    p = root / RELATIVE
    source = p.read_bytes()
    if hashlib.sha256(source).hexdigest() != SHA256:
        raise ValueError('MTP source differs from the pinned runtime')
    result = transform(source.decode())
    compile(result, str(p), 'exec')
    p.write_text(result)
    shutil.copy2(Path(__file__).with_name('amos_draft_eh.py'), root / 'amos_draft_eh.py')
    return {RELATIVE: hashlib.sha256(p.read_bytes()).hexdigest()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('vllm_root', type=Path)
    a = p.parse_args()
    print(json.dumps(patch(a.vllm_root)))
