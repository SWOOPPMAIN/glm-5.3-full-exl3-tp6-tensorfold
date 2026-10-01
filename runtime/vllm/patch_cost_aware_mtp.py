#!/usr/bin/env python3
"""Extend the exact P15 scheduler with accepted-prefix cost control."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'v1/core/sched/scheduler.py'
SHA256 = '42b6e965c2cf113055866cd471a446087d962f1bf33a0fa94b4819cdd39d0b54'


def patch(root):
    path = root/RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Scheduler source differs from pinned P15')
    source = raw.decode()
    replacements = {
        '        # Create the KV cache manager.\n':
            '        from vllm import amos_cost_aware_mtp\n'
            '        amos_cost_aware_mtp.install(self, vllm_config)\n\n'
            '        # Create the KV cache manager.\n',
        '                    adaptive_num_accepted_tokens += num_accepted\n':
            '                    adaptive_num_accepted_tokens += num_accepted\n'
            '                    from vllm import amos_cost_aware_mtp\n'
            '                    amos_cost_aware_mtp.observe_request(\n'
            '                        acceptance_length_controller, num_draft_tokens, num_accepted)\n',
    }
    for before,after in replacements.items():
        if source.count(before)!=1:
            raise ValueError('Scheduler anchor changed')
        source=source.replace(before,after)
    compile(source,str(path),'exec')
    path.write_text(source)
    for name in ('amos_cost_aware_mtp.py','amos_mtp_costs.json'):
        shutil.copy2(Path(__file__).with_name(name),root/name)
    return {RELATIVE:hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__=='__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
