#!/usr/bin/env python3
"""Add request/phase policy hooks to the exact currently qualified P19 source."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

RELATIVE = 'v1/core/sched/scheduler.py'
SHA256 = '8ac65d2ef7bcc72a3906ed18619c02168fa7244937326743ce57905f28d2c8ef'


def patch(root):
    path = root/RELATIVE
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError('Scheduler source differs from pinned P19')
    source = raw.decode()
    old = '''                if acceptance_length_controller is not None:
                    adaptive_num_drafts += 1
                    adaptive_num_draft_tokens += num_draft_tokens
                    adaptive_num_accepted_tokens += num_accepted
                    from vllm import amos_cost_aware_mtp
                    amos_cost_aware_mtp.observe_request(
                        acceptance_length_controller, num_draft_tokens, num_accepted)
'''
    changes = {
        '        amos_cost_aware_mtp.install(self, vllm_config)\n':
            '        amos_cost_aware_mtp.install(self, vllm_config)\n'
            '        from vllm import amos_request_phase_mtp\n'
            '        amos_request_phase_mtp.install(self, vllm_config)\n',
        old: '',
        '            if observed_spec_decode:\n                spec_decoding_stats =':
            '            if observed_spec_decode:\n'
            '                if acceptance_length_controller is not None:\n'
            '                    adaptive_num_drafts += 1\n'
            '                    adaptive_num_draft_tokens += num_draft_tokens\n'
            '                    adaptive_num_accepted_tokens += num_accepted\n'
            '                    from vllm import amos_request_phase_mtp\n'
            '                    amos_request_phase_mtp.observe_request(\n'
            '                        acceptance_length_controller, request, num_draft_tokens,\n'
            '                        num_accepted, new_token_ids, output_is_stale)\n'
            '                spec_decoding_stats =',
        '        # Dynamic speculative decoding: compute optimal K\n':
            '        from vllm import amos_request_phase_mtp\n'
            '        amos_request_phase_mtp.choose(\n'
            '            self.acceptance_length_controller, num_scheduled_tokens, self.requests)\n'
            '        # Dynamic speculative decoding: compute optimal K\n',
    }
    for before, after in changes.items():
        if source.count(before) != 1:
            raise ValueError('Request-phase scheduler anchor changed')
        source = source.replace(before, after)
    compile(source, str(path), 'exec')
    path.write_text(source)
    shutil.copy2(Path(__file__).with_name('amos_request_phase_mtp.py'), root/'amos_request_phase_mtp.py')
    return {RELATIVE: hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(patch(Path(sys.argv[1]))))
